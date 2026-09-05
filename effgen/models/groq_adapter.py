"""
Groq Cloud SDK adapter for effGen.

Supports all Groq chat models with built-in rate-limit coordination,
real streaming via SSE, native function-calling on supported models, and
per-request cost tracking via CostTracker.

Groq uses an OpenAI-compatible API shape, so the implementation mirrors
the CerebrasAdapter closely with Groq-specific rate limits.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from effgen.models._adapter_utils import (
    annotate_reasoning_only,
    apply_tool_request,
    estimate_tokens,
    extract_reasoning_text,
    extract_reasoning_tokens,
    normalize_finish_reason,
    normalize_tools_call_args,
    not_loaded_error,
    provider_runtime_error,
    reasoning_delta_text,
    warn_reasoning_only_stream,
)
from effgen.models._cost import CostTracker
from effgen.models._multimodal import require_vision_support
from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models._usage import (
    accumulate_stream_tool_call_deltas,
    cost_label,
    stream_tool_call_entries,
    tool_call_entry,
    tool_calls_from_message,
)
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    TokenCount,
    accumulate_stream_cost,
    clear_stream_tool_calls,
    record_stream_tool_calls,
)
from effgen.models.errors import ModelAuthError, ModelNotFoundError, error_has_status
from effgen.models.groq_models import (
    GROQ_DEFAULT_MODEL,
    GROQ_MODELS,
    chat_models,
)
from effgen.models.latency_tracker import timed_call
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr
from effgen.utils.async_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)

_GROQ_MODEL_TYPE_VALUE = "groq"

#: How reasoning-family Groq models should report their reasoning chain.
#:
#: Groq's default is ``"raw"``, which embeds the chain in ``message.content``
#: between ``<think>`` tags. For the qwen3 family that means a one-word question
#: answers with several hundred tokens of reasoning and no answer in
#: ``result.text``, and a tool-calling turn can spend its whole output budget
#: thinking. ``"parsed"`` puts the chain on ``message.reasoning`` — the field
#: :func:`extract_reasoning_text` already reads — and leaves the answer alone.
#: A caller who wants the raw form back can pass ``reasoning_format="raw"``,
#: which overrides this (``kwargs`` are applied last).
_REASONING_FORMAT = "parsed"


def _redact_groq_org(message: str) -> str:
    """Remove the caller's organization id from a Groq error body before it is
    surfaced (it is an account identifier, not useful for debugging)."""
    return re.sub(r"organization `org_[^`]+`", "organization `***`", message)


def _is_request_too_large(message: str, message_lower: str) -> bool:
    """True when a Groq error is a 413 payload-too-large (a single oversized
    request), not a 429 rate limit. Groq returns 413 with a ``rate_limit_exceeded``
    code for a request over the per-minute token limit, so status + wording are
    checked rather than the misleading code."""
    return (
        "413" in message
        or "request too large" in message_lower
        or "reduce your message size" in message_lower
    )


def _parse_failed_generation_json_call(message: str) -> dict[str, Any] | None:
    """Read a ``{"name": ..., "arguments": {...}}`` call out of an error body.

    The gpt-oss families emit their call as a bare JSON object rather than in
    the ``<function=…>`` wrapper, which is the shape Groq quotes back in
    ``failed_generation`` when it refuses the completion. Without this the whole
    turn fails on a call the model did make and the text of it is right there in
    the error.
    """
    from effgen.core.structured_output import _extract_balanced

    search_from = 0
    while True:
        start = message.find("{", search_from)
        if start == -1:
            return None
        blob = _extract_balanced(message[start:])
        if blob is not None:
            try:
                data = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                data = None
            if isinstance(data, dict):
                name = data.get("name") or data.get("function")
                arguments = data.get("arguments")
                if arguments is None:
                    arguments = data.get("parameters")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"__raw_input__": arguments}
                if isinstance(name, str) and name and isinstance(arguments, dict):
                    return tool_call_entry(name, arguments)
        search_from = start + 1


def _parse_failed_generation_tool_call(message: str) -> dict[str, Any] | None:
    """Extract a tool call from Groq's ``tool_use_failed`` failed_generation text.

    The model's own function-call syntax is sometimes malformed enough that
    Groq's server-side parser rejects it outright — most commonly a missing
    ``</function>`` closing tag, or two calls run together with no separator
    (e.g. a model retrying a just-denied tool call, then trailing off into a
    second one). The closing tag, the start of a subsequent ``<function=``,
    or end of string all terminate the first call's arguments, so only the
    first call is ever recovered here — the same shape a well-formed single
    call would have produced.
    """
    match = re.search(
        r"<function=([A-Za-z_]\w*)\s*>?\s*(.*?)(?:</function>|(?=<function=)|$)",
        message,
        re.DOTALL,
    )
    if not match:
        return _parse_failed_generation_json_call(message)

    name = match.group(1)
    raw_args = match.group(2).strip()
    if raw_args.startswith("(") and raw_args.endswith(")"):
        raw_args = raw_args[1:-1].strip()

    try:
        arguments = json.loads(raw_args) if raw_args else {}
    except (json.JSONDecodeError, TypeError):
        # A missing closing tag can leave trailing junk after the JSON object
        # (stray escape sequences, a second concatenated call) that a plain
        # `json.loads` rejects even though the object itself is well-formed.
        # Fall back to the same bracket-balanced, string-aware extractor used
        # for structured-output parsing, which stops at the matching closing
        # brace regardless of what follows.
        from effgen.core.structured_output import _extract_balanced

        balanced = _extract_balanced(raw_args) if raw_args else None
        try:
            arguments = json.loads(balanced) if balanced is not None else {"__raw_input__": raw_args}
        except (json.JSONDecodeError, TypeError):
            arguments = {"__raw_input__": raw_args}

    if not isinstance(arguments, dict):
        arguments = {"__raw_input__": raw_args}

    return tool_call_entry(name, arguments)


class _GroqModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _GROQ_MODEL_TYPE_VALUE


class GroqAdapter(BaseModel):
    """
    Adapter for Groq Cloud inference API.

    Wraps the ``groq`` SDK with the standard effGen BaseModel interface.
    Groq mirrors the OpenAI API shape. Supports:

    - Synchronous and async generation
    - Real token-by-token streaming (``generate_stream``)
    - Native function-calling on supported models (``generate_with_tools``)
    - Per-request cost tracking via :class:`~effgen.models._cost.CostTracker`
    - Per-model rate-limit coordination (RPM, RPD, TPM, TPD)

    Args:
        model_name: Groq model ID. Must be a key in
            :data:`~effgen.models.groq_models.GROQ_MODELS`.
            Defaults to ``"llama-3.1-8b-instant"``.
        api_key: Groq API key. If omitted, reads ``GROQ_API_KEY``
            from the environment.
        max_retries: Total attempts this adapter makes for one call. The
            provider SDK's own retry is switched off, so this is the whole
            budget rather than a multiplier on it.
        timeout: Per-request timeout in seconds.
        enable_rate_limiting: Wire built-in
            :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
        enable_cost_tracking: Record token usage in the global
            :class:`~effgen.models._cost.CostTracker`.

    Example::

        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("llama-3.3-70b-versatile")
        adapter.load()

        result = adapter.generate("What is the capital of France?")
        print(result.text)

        for chunk in adapter.generate_stream("Count from 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "groq"

    def __init__(
        self,
        model_name: str = GROQ_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 6,
        timeout: int = 60,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        **kwargs: Any,
    ) -> None:
        if model_name not in GROQ_MODELS:
            from effgen.models._catalog import suggest_for_missing

            raise ModelNotFoundError(
                provider="groq",
                model_name=model_name,
                message=f"Unknown Groq model '{model_name}'."
                        + suggest_for_missing("groq", model_name),
            )

        info = GROQ_MODELS[model_name]
        if info.get("modality") not in ("chat", None):
            raise ValueError(
                f"Groq model '{model_name}' is a {info.get('modality')} model "
                f"and cannot be used via chat completions. "
                f"Chat models: {chat_models()}"
            )

        super().__init__(
            model_name=model_name,
            model_type=_GroqModelType(),  # type: ignore[arg-type]
            context_length=info.get("context", 131_072),
        )
        # Groq serves families that emit reasoning tokens before any visible
        # text. Flagging them here is what earns the larger default token budget
        # from default_max_output_tokens() — without it they can spend the whole
        # budget thinking and return an empty (but billed) result.
        self._is_reasoning_model = bool(info.get("reasoning", False))
        self._api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking

        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting:
            tpd = info.get("tpd") or 0
            rpd = info.get("rpd") or 0
            self._rate_limiter = RateLimitCoordinator(
                provider="groq",
                model=model_name,
                rpm=info.get("rpm", 30),
                rph=info.get("rpm", 30) * 60,
                rpd=rpd if rpd else 100_000,
                tpm=info.get("tpm", 6_000),
                tph=info.get("tpm", 6_000) * 60,
                tpd=tpd if tpd else 10_000_000,
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Groq SDK client.

        Raises:
            RuntimeError: If ``groq`` is not installed.
            ValueError: If no API key is available.
        """
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "groq SDK is not installed. "
                "Install with: pip install 'effgen[groq]'"
            ) from exc

        if not (self._api_key or os.getenv("GROQ_API_KEY")):
            raise ValueError(
                "Groq API key not found. Set the GROQ_API_KEY "
                "environment variable or pass api_key= to GroqAdapter."
            )

        self._client = Groq(
            api_key=self._api_key or os.getenv("GROQ_API_KEY"),
            timeout=self.timeout,
            # The SDK's own retry is switched off: the loop in ``_do_generate``
            # runs its own backoff and honours the delay Groq states, so a
            # second retry layer underneath does not share that budget — it
            # multiplies it, turning one client request into a dozen upstream
            # requests on a rate limit. Matches CerebrasAdapter and the Gemini
            # SDK's ``attempts=1``.
            max_retries=0,
        )
        self._is_loaded = True

        info = GROQ_MODELS.get(self.model_name, {})
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "groq",
            "family": info.get("family", ""),
            "modality": info.get("modality", "chat"),
            "supports_native_tools": info.get("supports_native_tools", False),
            "supports_streaming": info.get("supports_streaming", True),
            "rpm": info.get("rpm"),
            "rpd": info.get("rpd"),
            "tpm": info.get("tpm"),
            "tpd": info.get("tpd"),
            "notes": info.get("notes", ""),
        }
        logger.info("GroqAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("GroqAdapter unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via the Groq API.

        Args:
            prompt: User prompt text.
            config: Optional generation config.
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with text and token usage.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("groq", self.model_name, "generate")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="groq",
            model_name=self.model_name,
            supports_vision=GROQ_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use 'qwen/qwen3.6-27b' for Groq vision.",
        )

        try:
            prompt_text = prompt if isinstance(prompt, str) else str(getattr(prompt, "text", prompt))
            est_tokens = self.count_tokens(prompt_text).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(est_tokens))

        with timed_call("groq", self.model_name):
            result = self._do_generate(prompt, config, **kwargs)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    async def async_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Async version of :meth:`generate` — preferred inside async contexts.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Returns:
            The generated text with its usage metadata.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("groq", self.model_name, "async_generate")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="groq",
            model_name=self.model_name,
            supports_vision=GROQ_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use 'qwen/qwen3.6-27b' for Groq vision.",
        )

        try:
            prompt_text = prompt if isinstance(prompt, str) else str(getattr(prompt, "text", prompt))
            est_tokens = self.count_tokens(prompt_text).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(est_tokens)

        result = self._do_generate(prompt, config, **kwargs)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    @staticmethod
    def _message_to_groq(message: Any) -> dict[str, Any]:
        """Convert an effGen Message to a Groq/OpenAI-compatible dict."""
        import base64

        from effgen.core.messages import ImagePart, TextPart, VideoPart
        from effgen.multimodal.image_pre import prepare as _preprocess_image

        role = message.role.value
        content_parts: list[dict[str, Any]] = []

        for part in message.content:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                processed = _preprocess_image(part, "groq", "")
                b64 = base64.b64encode(processed.image).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{processed.mime};base64,{b64}"},
                })
            elif isinstance(part, VideoPart):
                for frame in part.frames:
                    b64 = base64.b64encode(frame).decode()
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.mime};base64,{b64}"},
                    })

        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            return {"role": role, "content": content_parts[0]["text"]}
        return {"role": role, "content": content_parts}

    def _estimate_prompt_tokens(self, request_params: dict[str, Any]) -> int:
        """Estimate prompt tokens from a request's messages and tool schemas.

        Used only when the API reports zero/absent usage on a response that
        clearly consumed input, so the run's token/cost accounting does not
        under-count by treating a billed call as free.
        """
        parts: list[str] = []
        for m in request_params.get("messages", []) or []:
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
        for t in request_params.get("tools", []) or []:
            try:
                parts.append(json.dumps(t))
            except (TypeError, ValueError):
                pass
        return self.count_tokens("\n".join(p for p in parts if p)).count

    def _estimate_failed_usage(
        self,
        request_params: dict[str, Any],
        exc: Exception,
        msg: str,
    ) -> tuple[int, int]:
        """Estimate (prompt_tokens, completion_tokens) for a billed call whose
        error response omitted usage (Groq's ``tool_use_failed`` recovery).

        Prompt tokens are counted from the request messages and any tool
        schemas; completion tokens from the model's ``failed_generation`` text
        when the error body exposes it, else from the visible error message.
        """
        prompt_tokens = self._estimate_prompt_tokens(request_params)

        failed_generation = ""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                failed_generation = err.get("failed_generation") or ""
        completion_tokens = self.count_tokens(failed_generation or msg).count
        return prompt_tokens, completion_tokens

    def _do_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Internal: make the SDK call and return a GenerationResult."""
        if messages is None:
            # Handle effGen Message objects
            try:
                from effgen.core.messages import Message
                if isinstance(prompt, Message):
                    messages = [self._message_to_groq(prompt)]
                elif isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                    messages = [self._message_to_groq(m) for m in prompt]
                else:
                    messages = [{"role": "user", "content": prompt}]
            except ImportError:
                messages = [{"role": "user", "content": prompt}]

        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        if config.temperature is not None and config.temperature != 0.7:
            request_params["temperature"] = config.temperature
        if config.top_p is not None and config.top_p != 0.9:
            request_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            request_params["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty
        if self._is_reasoning_model:
            request_params["reasoning_format"] = _REASONING_FORMAT

        info = GROQ_MODELS.get(self.model_name, {})
        apply_tool_request(request_params, tools, info)

        request_params.update(kwargs)

        # The whole retry budget for this call: the SDK's own retry is off, so
        # the caller's ``max_retries`` is the number of upstream requests one
        # client request can become, not a multiplier on it.
        _MAX_RETRIES = max(1, int(self.max_retries))
        _last_exc: Exception | None = None
        for _attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**request_params)
                break
            except Exception as exc:
                _last_exc = exc
                msg = str(exc)
                msg_lower = msg.lower()

                is_auth = (
                    error_has_status(exc, 401)
                    or error_has_status(exc, 403)
                    or "invalid_api_key" in msg_lower
                    or "invalid api key" in msg_lower
                    or "unauthorized" in msg_lower
                    or "authentication" in msg_lower
                )
                if is_auth:
                    raise ModelAuthError(
                        provider="groq",
                        model_name=self.model_name,
                        message=msg,
                    ) from exc

                # A 413 payload-too-large is a permanent property of this
                # request, not a transient rate limit — fail fast with a
                # fix-oriented hint instead of routing it through retry/failover.
                if _is_request_too_large(msg, msg_lower):
                    from effgen.models.errors import InvalidRequestError as _IRE
                    raise _IRE(
                        provider="groq",
                        model_name=self.model_name,
                        message=(
                            f"request too large for {self.model_name}: "
                            f"{_redact_groq_org(msg)} — reduce the request "
                            "(fewer/smaller tools or shorter input) or use a "
                            "larger-context model."
                        ),
                    ) from exc

                is_rate = error_has_status(exc, 429) or "rate_limit" in msg_lower or "rate limit" in msg_lower
                is_server = "500" in msg or "503" in msg or "internal" in msg_lower
                is_timeout = "timeout" in msg_lower

                if is_rate:
                    # Raise RateLimitExceeded so the router can failover to another provider
                    from effgen.models._rate_limit import RateLimitExceeded as _RLE
                    raise _RLE(
                        f"Groq rate limit hit for {self.model_name}: {msg}"
                    ) from exc

                if is_server or is_timeout:
                    if _attempt >= _MAX_RETRIES:
                        logger.error("Groq API error after %d retries: %s", _attempt, exc)
                        from effgen.models.errors import ProviderTransientError as _PTE
                        raise _PTE(
                            provider="groq",
                            model_name=self.model_name,
                            status_code=500 if is_server else 0,
                            message=f"Groq API failed after {_MAX_RETRIES} retries: {exc}",
                        ) from exc
                    delay = min(60.0, 2.0 * (2 ** (_attempt - 1)) + random.uniform(0, 0.5))
                    logger.warning(
                        "Groq transient error on attempt %d/%d — retrying in %.1fs: %s",
                        _attempt, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue

                failed_tool_call = _parse_failed_generation_tool_call(msg)
                # Not gated on ``tools``: the ReAct path describes its tools in
                # the prompt and sends no ``tools`` array, so Groq applies
                # ``tool_choice: "none"`` and rejects the whole completion when
                # a gpt-oss model calls a tool anyway. That is exactly the turn
                # this recovery exists for, and it was the one turn the guard
                # kept it from reaching. A parseable failed_generation is a
                # usable tool call whether or not the request advertised tools.
                if "tool_use_failed" in msg_lower and failed_tool_call is not None:
                    # Recovered, not actionable — the call still succeeds via
                    # failed_generation. INFO (not WARNING) so a successful
                    # turn emits no WARNING line by default; --verbose shows it.
                    logger.info(
                        "Groq returned tool_use_failed but included a parseable tool call; "
                        "using failed_generation as structured tool call."
                    )
                    # This turn was still billed, but the error body carries no
                    # usage object, so estimate token counts from the request
                    # and the model's failed generation rather than reporting
                    # zero (which would under-count a run's true cost/tokens).
                    prompt_tokens, completion_tokens = self._estimate_failed_usage(
                        request_params, exc, msg,
                    )
                    total_tokens = prompt_tokens + completion_tokens
                    cost: float | None = None
                    if self._enable_cost_tracking:
                        cost = CostTracker.get().record(
                            provider="groq",
                            model=self.model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                    return GenerationResult(
                        text="",
                        tokens_used=completion_tokens,
                        finish_reason="tool_calls",
                        model_name=self.model_name,
                        metadata={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "provider": "groq",
                            "cost_usd": cost,
                            "estimated_usage": True,
                            "tool_calls": [failed_tool_call],
                            "provider_error": "tool_use_failed",
                        },
                    )

                logger.error("Groq API call failed: %s", exc)
                if "tool_use_failed" in msg_lower:
                    # The recovery above could not read the call. Say what the
                    # model did and which mode answers it, rather than leaving
                    # a bare "invalid_request" that names neither.
                    raise provider_runtime_error(
                        "groq", self.model_name, "generate", exc,
                        message=(
                            f"Groq generation failed: {self.model_name} called a tool "
                            "in a request that advertised none, and the call it wrote "
                            "could not be read back. This model calls tools through the "
                            "API rather than from a prompt description, so run it with "
                            'tool_calling_mode="auto" (or "native") instead of "react"'
                        ),
                    ) from exc
                raise provider_runtime_error("groq", self.model_name, "generate", exc, message="Groq generation failed") from exc
        else:
            assert _last_exc is not None
            raise provider_runtime_error(
                "groq", self.model_name, "generate", _last_exc,
                message=f"Groq generation failed after {_MAX_RETRIES} retries",
            ) from _last_exc

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        finish_reason = normalize_finish_reason(choice.finish_reason)

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        tool_calls = tool_calls_from_message(message)

        # Groq can return an all-zero usage object on some tool-call responses
        # even though input was consumed and output produced. Reporting that
        # zero verbatim makes a billed call look free and under-counts a run's
        # token/cost totals, so estimate from the request and the produced
        # output instead and flag the numbers as estimated.
        estimated_usage = False
        if total_tokens == 0 and (text or tool_calls):
            prompt_tokens = self._estimate_prompt_tokens(request_params)
            completion_parts = [text] + [json.dumps(tc) for tc in tool_calls]
            completion_tokens = self.count_tokens(
                "\n".join(p for p in completion_parts if p)
            ).count
            total_tokens = prompt_tokens + completion_tokens
            estimated_usage = True

        cost = None
        if self._enable_cost_tracking:
            cost = CostTracker.get().record(
                provider="groq",
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        logger.info(
            "Groq generated %d tokens (prompt=%d, completion=%d, cost=%s)",
            total_tokens, prompt_tokens, completion_tokens, cost_label(cost),
        )
        # Emit span attributes on the current active span
        _set_span_attr(ModelAttrs.PROVIDER, "groq")
        _set_span_attr(ModelAttrs.NAME, self.model_name)
        _set_span_attr(ModelAttrs.INPUT_TOKENS, prompt_tokens)
        _set_span_attr(ModelAttrs.OUTPUT_TOKENS, completion_tokens)
        if cost is not None:
            _set_span_attr(ModelAttrs.COST_USD, float(cost))
        _set_span_attr(ModelAttrs.OUTCOME, "ok")

        metadata: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "provider": "groq",
            "cost_usd": cost,
            "estimated_usage": estimated_usage,
            "tool_calls": tool_calls,
        }
        annotate_reasoning_only(
            metadata,
            text=text,
            reasoning_text=extract_reasoning_text(message),
            reasoning_tokens=extract_reasoning_tokens(usage),
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=request_params.get("max_tokens"),
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            logger=logger,
        )

        return GenerationResult(
            text=text,
            tokens_used=completion_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata=metadata,
        )

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with native tool calling (OpenAI function-calling format).

        ``config`` is the third parameter on every adapter. It was ``messages``
        here until 1.0.0, so a positional conversation is still read correctly:
        the two are told apart by type, never by position.

        Args:
            prompt: User prompt text (ignored when *messages* is provided).
            tools: List of OpenAI-format tool dicts or effGen BaseTool objects.
            config: Optional generation config.
            messages: Optional full conversation history (overrides *prompt*).
            **kwargs: Extra parameters forwarded to the provider SDK.

        Returns:
            GenerationResult whose ``metadata["tool_calls"]`` contains parsed calls.
        """
        config, messages = normalize_tools_call_args(config, messages)
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("groq", self.model_name, "generate_with_tools")
        if config is None:
            config = GenerationConfig()
        return self._do_generate(prompt, config, tools=tools, messages=messages, **kwargs)

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a response token-by-token from the Groq API.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Yields:
            str: Successive text chunks from the model.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("groq", self.model_name, "generate_stream")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="groq",
            model_name=self.model_name,
            supports_vision=GROQ_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use 'qwen/qwen3.6-27b' for Groq vision.",
        )

        try:
            from effgen.core.messages import Message

            if isinstance(prompt, Message):
                messages = [self._message_to_groq(prompt)]
            elif isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                messages = [self._message_to_groq(m) for m in prompt]
            else:
                messages = [{"role": "user", "content": prompt}]
        except ImportError:
            messages = [{"role": "user", "content": prompt}]
        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }

        if config.temperature is not None and config.temperature != 0.7:
            request_params["temperature"] = config.temperature
        if config.top_p is not None and config.top_p != 0.9:
            request_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            request_params["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty
        if self._is_reasoning_model:
            request_params["reasoning_format"] = _REASONING_FORMAT

        # The same shaping the non-streaming path applies: the catalog gate and
        # ``tool_choice`` are one decision, so a streamed turn sends the same
        # request a non-streamed one would. A caller passing ``tools=`` as a raw
        # keyword used to reach the provider ungated and without ``tool_choice``.
        _stream_tools = kwargs.pop("tools", None)
        request_params.update(kwargs)
        apply_tool_request(
            request_params, _stream_tools, GROQ_MODELS.get(self.model_name, {})
        )

        clear_stream_tool_calls(self)
        self._last_stream_finish_reason: str | None = None

        try:
            with timed_call("groq", self.model_name) as _stream_timer:
                _first_token = True
                stream = self._client.chat.completions.create(**request_params)

                prompt_tokens = 0
                completion_tokens = 0
                tool_calls_buf: dict[int, dict[str, Any]] = {}
                reasoning_buf: list[str] = []
                stream_usage: Any = None

                for chunk in stream:
                    # The terminal usage chunk (some providers send it as a
                    # trailing chunk with an empty `choices` list) must be read
                    # before the no-choices skip below, or its token counts are
                    # silently dropped and cost/tokens never get recorded.
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        usage = chunk.usage
                        stream_usage = usage
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta and delta.content:
                        if _first_token:
                            _stream_timer.mark_first_token()
                            _first_token = False
                        yield delta.content

                    if delta and getattr(delta, "tool_calls", None):
                        accumulate_stream_tool_call_deltas(
                            tool_calls_buf, delta.tool_calls
                        )
                        # Recorded as it accumulates, so a consumer streaming
                        # this turn knows it is a tool call before it commits
                        # any text as the answer.
                        record_stream_tool_calls(
                            self, stream_tool_call_entries(tool_calls_buf)
                        )

                    if delta:
                        reasoning_buf.append(reasoning_delta_text(delta))

                    if choice.finish_reason:
                        self._last_stream_finish_reason = choice.finish_reason

                streamed_calls = stream_tool_call_entries(tool_calls_buf)
                record_stream_tool_calls(self, streamed_calls)

                warn_reasoning_only_stream(
                    model_name=self.model_name,
                    yielded_text=not _first_token,
                    reasoning_text="".join(reasoning_buf),
                    reasoning_tokens=extract_reasoning_tokens(stream_usage),
                    finish_reason=self._last_stream_finish_reason,
                    max_tokens=request_params.get("max_tokens"),
                    tool_calls=streamed_calls,
                    logger=logger,
                )

            if self._enable_cost_tracking and (prompt_tokens or completion_tokens):
                cost = CostTracker.get().record(
                    provider="groq",
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                accumulate_stream_cost(
                    self,
                    cost,
                    prompt_tokens + completion_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            if (
                error_has_status(exc, 401)
                or error_has_status(exc, 403)
                or "invalid_api_key" in msg_lower
                or "invalid api key" in msg_lower
                or "unauthorized" in msg_lower
                or "authentication" in msg_lower
            ):
                raise ModelAuthError(
                    provider="groq",
                    model_name=self.model_name,
                    message=msg,
                ) from exc
            if _is_request_too_large(msg, msg_lower):
                from effgen.models.errors import InvalidRequestError as _IRE
                raise _IRE(
                    provider="groq",
                    model_name=self.model_name,
                    message=(
                        f"request too large for {self.model_name}: "
                        f"{_redact_groq_org(msg)} — reduce the request "
                        "(fewer/smaller tools or shorter input) or use a "
                        "larger-context model."
                    ),
                ) from exc
            is_rate = error_has_status(exc, 429) or "rate_limit" in msg_lower or "rate limit" in msg_lower
            if is_rate:
                from effgen.models._rate_limit import RateLimitExceeded as _RLE
                raise _RLE(f"Groq rate limit hit for {self.model_name}: {msg}") from exc
            is_server = "500" in msg or "503" in msg or "internal" in msg_lower
            if is_server:
                from effgen.models.errors import ProviderTransientError as _PTE
                raise _PTE(provider="groq", model_name=self.model_name, status_code=500, message=msg) from exc
            logger.error("Groq streaming failed: %s", exc)
            raise provider_runtime_error("groq", self.model_name, "stream", exc, message="Groq streaming failed") from exc

    # ------------------------------------------------------------------
    # Token counting / context length
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken (cl100k_base, same as OpenAI)."""
        return TokenCount(count=estimate_tokens(text), model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return the context window size for the loaded model."""
        return GROQ_MODELS.get(self.model_name, {}).get("context", 131_072)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rate_limit_status(self) -> dict[str, Any]:
        """Return current rate-limit window status from the coordinator."""
        if self._rate_limiter is None:
            return {"enabled": False}
        return {"enabled": True, "status": str(self._rate_limiter)}

    @property
    def supports_native_tools(self) -> bool:
        """True if this model supports OpenAI-format function calling."""
        return GROQ_MODELS.get(self.model_name, {}).get("supports_native_tools", False)

    def supports_tool_calling(self) -> bool:
        """Return True if the loaded model supports native tool-calling."""
        return GROQ_MODELS.get(self.model_name, {}).get("supports_native_tools", False)

    def supports_forced_tool_call(self) -> bool:
        """True when tools are offered: ``tool_choice`` is honoured here.

        Groq's OpenAI-compatible endpoint enforces the
        choice, so a turn can be sent that requires a call. A model that is
        not offered tool definitions cannot be required to call one, so this
        follows :meth:`supports_tool_calling`.
        """
        return self.supports_tool_calling()

    def streams_tool_calls(self) -> bool:
        """True: a streamed turn's native tool calls are recorded."""
        return True

    def supports_function_calling(self) -> bool:
        """Alias for :meth:`supports_tool_calling`."""
        return self.supports_tool_calling()

    @property
    def supports_streaming(self) -> bool:
        """True if this model supports streaming responses."""
        return GROQ_MODELS.get(self.model_name, {}).get("supports_streaming", True)


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.groq_models import GROQ_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "groq",
            GroqAdapter,
            GROQ_MODELS,
            env_keys=["GROQ_API_KEY"],
            capabilities={Capability.chat, Capability.streaming, Capability.tools, Capability.json_schema, Capability.vision},
            # Free developer tier routes as zero out-of-pocket cost while quota remains.
            # Per-model paid list prices are retained in GROQ_MODELS for tie-break metadata.
            # llama-3.1-8b-instant: $0.05/$0.08; llama-3.3-70b: $0.59/$0.79 per 1M tokens.
            # Pricing verified: https://groq.com/pricing (2026-05-11)
            pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
