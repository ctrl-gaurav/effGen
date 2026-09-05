"""
Together AI SDK adapter for effGen.

Supports all Together AI chat models with built-in rate-limit coordination,
real streaming via SSE, native function-calling on supported models, and
per-request cost tracking via CostTracker.

Together AI uses an OpenAI-compatible API shape (chat.completions.create),
so the implementation mirrors GroqAdapter closely with Together-specific
rate limits and the serverless/dedicated-endpoint distinction.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from effgen.models._adapter_utils import (
    DIRECT_CALL_REASONING_MAX_TOKENS,
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
from effgen.models.errors import (
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    error_has_status,
)
from effgen.models.latency_tracker import timed_call
from effgen.models.together_models import (
    TOGETHER_DEFAULT_MODEL,
    TOGETHER_MODELS,
    available_models,
    serverless_models,
)
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr
from effgen.utils.async_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)

_TOGETHER_MODEL_TYPE_VALUE = "together"


class _TogetherModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _TOGETHER_MODEL_TYPE_VALUE


class TogetherAdapter(BaseModel):
    """
    Adapter for Together AI inference API.

    Wraps the ``together`` SDK with the standard effGen BaseModel interface.
    Together mirrors the OpenAI API shape. Supports:

    - Synchronous and async generation
    - Real token-by-token streaming (``generate_stream``)
    - Native function-calling on supported models (``generate_with_tools``)
    - Per-request cost tracking via :class:`~effgen.models._cost.CostTracker`
    - Per-model rate-limit coordination (RPM, TPM)
    - Automatic warning when non-serverless models are used

    Args:
        model_name: Together model ID. Must be a key in
            :data:`~effgen.models.together_models.TOGETHER_MODELS`.
            Defaults to ``"Qwen/Qwen3.5-9B"``.
        api_key: Together API key. If omitted, reads ``TOGETHER_API_KEY``
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

        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        adapter.load()

        result = adapter.generate("What is the capital of France?")
        print(result.text)

        for chunk in adapter.generate_stream("Count from 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "together"

    def __init__(
        self,
        model_name: str = TOGETHER_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        **kwargs: Any,
    ) -> None:
        if model_name not in TOGETHER_MODELS:
            raise ModelNotFoundError(
                provider="together",
                model_name=model_name,
                message=f"Unknown Together model '{model_name}'. "
                        f"Available: {available_models()}\n"
                        f"Serverless (no dedicated endpoint): {serverless_models()}",
            )

        info = TOGETHER_MODELS[model_name]

        # Warn if model requires a dedicated endpoint
        if not info.get("serverless", False):
            logger.warning(
                "Together model '%s' may require a dedicated endpoint. "
                "If you get a 400 error, start the endpoint at "
                "https://api.together.ai/models/%s",
                model_name, model_name,
            )

        super().__init__(
            model_name=model_name,
            model_type=_TogetherModelType(),  # type: ignore[arg-type]
            context_length=info.get("context", 131_072),
        )

        # Together serves several families that emit reasoning tokens before any
        # visible text. Flagging them here is what earns the larger default token
        # budget from default_max_output_tokens() — without it they can spend the
        # whole budget thinking and return an empty (but billed) result.
        self._is_reasoning_model = bool(info.get("reasoning", False))
        self._api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking

        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting:
            self._rate_limiter = RateLimitCoordinator(
                provider="together",
                model=model_name,
                rpm=info.get("rpm", 100),
                rph=info.get("rpm", 100) * 60,
                rpd=100_000,
                tpm=info.get("tpm", 100_000),
                tph=info.get("tpm", 100_000) * 60,
                tpd=10_000_000,
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Together SDK client.

        Raises:
            RuntimeError: If ``together`` is not installed.
            ValueError: If no API key is available.
        """
        try:
            from together import Together
        except ImportError as exc:
            raise RuntimeError(
                "together SDK is not installed. "
                "Install with: pip install 'effgen[together]'"
            ) from exc

        if not (self._api_key or os.getenv("TOGETHER_API_KEY")):
            raise ValueError(
                "Together API key not found. Set the TOGETHER_API_KEY "
                "environment variable or pass api_key= to TogetherAdapter."
            )

        self._client = Together(
            api_key=self._api_key or os.getenv("TOGETHER_API_KEY"),
            timeout=self.timeout,
            # Switched off for the same reason as Groq and Cerebras: the loop
            # in ``_do_generate`` already backs off and honours the stated
            # delay, and a second layer underneath multiplies the attempts
            # rather than sharing the budget.
            max_retries=0,
        )
        self._is_loaded = True

        info = TOGETHER_MODELS.get(self.model_name, {})
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "together",
            "family": info.get("family", ""),
            "organization": info.get("organization", ""),
            "modality": info.get("modality", "chat"),
            "supports_native_tools": info.get("supports_native_tools", False),
            "supports_streaming": info.get("supports_streaming", True),
            "serverless": info.get("serverless", False),
            "pricing_per_1m_input": info.get("pricing_per_1m_input", 0),
            "pricing_per_1m_output": info.get("pricing_per_1m_output", 0),
            "rpm": info.get("rpm"),
            "tpm": info.get("tpm"),
        }
        logger.info("TogetherAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("TogetherAdapter unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via the Together AI API.

        Args:
            prompt: User prompt text.
            config: Optional generation config.
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with text and token usage.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("together", self.model_name, "generate")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="together",
            model_name=self.model_name,
            supports_vision=TOGETHER_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use a Together model with supports_vision=True for image inputs.",
        )

        try:
            prompt_text = prompt if isinstance(prompt, str) else str(getattr(prompt, "text", prompt))
            est_tokens = self.count_tokens(prompt_text).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(est_tokens))

        with timed_call("together", self.model_name):
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
            raise not_loaded_error("together", self.model_name, "async_generate")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="together",
            model_name=self.model_name,
            supports_vision=TOGETHER_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use a Together model with supports_vision=True for image inputs.",
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
    def _message_to_together(message: Any) -> dict[str, Any]:
        """Convert an effGen Message to a Together/OpenAI-compatible dict."""
        import base64

        from effgen.core.messages import ImagePart, TextPart, VideoPart
        from effgen.multimodal.image_pre import prepare as _preprocess_image

        role = message.role.value
        content_parts: list[dict[str, Any]] = []

        for part in message.content:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                processed = _preprocess_image(part, "together", "")
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
            try:
                from effgen.core.messages import Message
                if isinstance(prompt, Message):
                    messages = [self._message_to_together(prompt)]
                elif isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                    messages = [self._message_to_together(m) for m in prompt]
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
        elif self._is_reasoning_model:
            # Together's own default is small enough that a reasoning model can
            # spend it all thinking and return empty text. Nothing retries a
            # direct call at a larger budget, so ask for the generous one; only
            # the tokens actually generated are billed.
            request_params["max_tokens"] = DIRECT_CALL_REASONING_MAX_TOKENS
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty

        info = TOGETHER_MODELS.get(self.model_name, {})
        apply_tool_request(request_params, tools, info)

        request_params.update(kwargs)

        _MAX_RETRIES = max(1, self.max_retries)
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
                        provider="together",
                        model_name=self.model_name,
                        message=msg,
                    ) from exc

                # Non-serverless model — dedicated endpoint not running
                is_endpoint_error = (
                    "model_not_available" in msg_lower
                    or "dedicated_endpoint_not_running" in msg_lower
                    or "unable to access non-serverless" in msg_lower
                    or "dedicated endpoint" in msg_lower
                )
                if is_endpoint_error:
                    # A permanent configuration condition (the endpoint isn't
                    # running), not a transient failure — non-retryable so a
                    # fallback_chain doesn't burn attempts retrying the same
                    # unstartable model.
                    raise InvalidRequestError(
                        provider="together",
                        model_name=self.model_name,
                        message=(
                            f"Together model '{self.model_name}' requires a dedicated "
                            f"endpoint that is not running. Start it at "
                            f"https://api.together.ai/models/{self.model_name} or use a "
                            f"serverless model: {serverless_models()}"
                        ),
                    ) from exc

                is_rate = error_has_status(exc, 429) or "rate_limit" in msg_lower or "rate limit" in msg_lower
                is_server = "500" in msg or "503" in msg or "internal" in msg_lower
                is_timeout = "timeout" in msg_lower

                if is_rate:
                    from effgen.models._rate_limit import RateLimitExceeded as _RLE

                    raise _RLE(
                        f"Together rate limit hit for {self.model_name}: {msg}"
                    ) from exc

                if is_server or is_timeout:
                    if _attempt >= _MAX_RETRIES:
                        logger.error("Together API error after %d retries: %s", _attempt, exc)
                        if is_timeout:
                            from effgen.models.errors import ModelTimeoutError as _MTE

                            raise _MTE(
                                provider="together",
                                model_name=self.model_name,
                                timeout_seconds=self.timeout,
                            ) from exc
                        from effgen.models.errors import ProviderTransientError as _PTE

                        raise _PTE(
                            provider="together",
                            model_name=self.model_name,
                            status_code=500,
                            message=f"Together API failed after {_MAX_RETRIES} retries: {exc}",
                        ) from exc
                    delay = min(60.0, 2.0 * (2 ** (_attempt - 1)) + random.uniform(0, 0.5))
                    logger.warning(
                        "Together transient error on attempt %d/%d — retrying in %.1fs: %s",
                        _attempt, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue

                logger.error("Together API call failed: %s", exc)
                raise provider_runtime_error("together", self.model_name, "generate", exc, message="Together generation failed") from exc
        else:
            assert _last_exc is not None
            raise provider_runtime_error(
                "together", self.model_name, "generate", _last_exc,
                message=f"Together generation failed after {_MAX_RETRIES} retries",
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

        cost: float | None = None
        if self._enable_cost_tracking:
            cost = CostTracker.get().record(
                provider="together",
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        logger.info(
            "Together generated %d tokens (prompt=%d, completion=%d, cost=%s)",
            total_tokens, prompt_tokens, completion_tokens, cost_label(cost),
        )
        _set_span_attr(ModelAttrs.PROVIDER, "together")
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
            "provider": "together",
            "cost_usd": cost,
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
            raise not_loaded_error("together", self.model_name, "generate_with_tools")
        if config is None:
            config = GenerationConfig()
        return self._do_generate(prompt, config, tools=tools, messages=messages, **kwargs)

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a response token-by-token from the Together AI API.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Yields:
            str: Successive text chunks from the model.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("together", self.model_name, "generate_stream")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="together",
            model_name=self.model_name,
            supports_vision=TOGETHER_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use a Together model with supports_vision=True for image inputs.",
        )

        try:
            from effgen.core.messages import Message

            if isinstance(prompt, Message):
                messages = [self._message_to_together(prompt)]
            elif isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                messages = [self._message_to_together(m) for m in prompt]
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
        elif self._is_reasoning_model:
            # Together's own default is small enough that a reasoning model can
            # spend it all thinking and return empty text. Nothing retries a
            # direct call at a larger budget, so ask for the generous one; only
            # the tokens actually generated are billed.
            request_params["max_tokens"] = DIRECT_CALL_REASONING_MAX_TOKENS
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty

        # The same shaping the non-streaming path applies: the catalog gate and
        # ``tool_choice`` are one decision, so a streamed turn sends the same
        # request a non-streamed one would. A caller passing ``tools=`` as a raw
        # keyword used to reach the provider ungated and without ``tool_choice``.
        _stream_tools = kwargs.pop("tools", None)
        request_params.update(kwargs)
        apply_tool_request(
            request_params, _stream_tools, TOGETHER_MODELS.get(self.model_name, {})
        )

        clear_stream_tool_calls(self)
        self._last_stream_finish_reason: str | None = None

        try:
            with timed_call("together", self.model_name) as _stream_timer:
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
                    provider="together",
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
                    provider="together",
                    model_name=self.model_name,
                    message=msg,
                ) from exc
            if (
                "model_not_available" in msg_lower
                or "dedicated_endpoint_not_running" in msg_lower
                or "unable to access non-serverless" in msg_lower
            ):
                # A permanent configuration condition, not a transient
                # failure — non-retryable so a fallback_chain doesn't burn
                # attempts retrying the same unstartable model.
                raise InvalidRequestError(
                    provider="together",
                    model_name=self.model_name,
                    message=(
                        f"Together model '{self.model_name}' requires a dedicated "
                        f"endpoint. Start it at "
                        f"https://api.together.ai/models/{self.model_name} or use a "
                        f"serverless model: {serverless_models()}"
                    ),
                ) from exc
            if error_has_status(exc, 429) or "rate_limit" in msg_lower or "rate limit" in msg_lower:
                from effgen.models._rate_limit import RateLimitExceeded as _RLE

                raise _RLE(f"Together rate limit hit for {self.model_name}: {msg}") from exc
            if "timeout" in msg_lower:
                from effgen.models.errors import ModelTimeoutError as _MTE

                raise _MTE(
                    provider="together",
                    model_name=self.model_name,
                    timeout_seconds=self.timeout,
                ) from exc
            if "500" in msg or "503" in msg or "internal" in msg_lower:
                from effgen.models.errors import ProviderTransientError as _PTE

                raise _PTE(
                    provider="together",
                    model_name=self.model_name,
                    status_code=500,
                    message=msg,
                ) from exc
            logger.error("Together streaming failed: %s", exc)
            raise provider_runtime_error("together", self.model_name, "stream", exc, message="Together streaming failed") from exc

    # ------------------------------------------------------------------
    # Token counting / context length
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken (cl100k_base)."""
        return TokenCount(count=estimate_tokens(text), model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return the context window size for the loaded model."""
        return TOGETHER_MODELS.get(self.model_name, {}).get("context", 131_072)

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
        return TOGETHER_MODELS.get(self.model_name, {}).get("supports_native_tools", False)

    def supports_tool_calling(self) -> bool:
        """Return True if the loaded model supports native tool-calling."""
        return TOGETHER_MODELS.get(self.model_name, {}).get("supports_native_tools", False)

    def supports_forced_tool_call(self) -> bool:
        """True when tools are offered: ``tool_choice`` is honoured here.

        Together's OpenAI-compatible endpoint enforces the
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
        return TOGETHER_MODELS.get(self.model_name, {}).get("supports_streaming", True)

    @property
    def is_serverless(self) -> bool:
        """True if this model is accessible without a dedicated endpoint."""
        return TOGETHER_MODELS.get(self.model_name, {}).get("serverless", False)

    def pricing(self) -> dict[str, float]:
        """Return per-request pricing info for the loaded model."""
        info = TOGETHER_MODELS.get(self.model_name, {})
        return {
            "input_per_1m_usd": info.get("pricing_per_1m_input", 0),
            "output_per_1m_usd": info.get("pricing_per_1m_output", 0),
        }


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.registry import ProviderRegistry
        from effgen.models.together_models import TOGETHER_MODELS
        ProviderRegistry.register(
            "together",
            TogetherAdapter,
            TOGETHER_MODELS,
            env_keys=["TOGETHER_API_KEY"],
            capabilities={Capability.chat, Capability.streaming, Capability.tools, Capability.json_schema, Capability.vision},
            # $1 free credits on signup; pay-per-token after. Provider default = cheapest LLM.
            # LFM2 24B A2B: $0.03/$0.12; gpt-oss-20B: $0.05/$0.20; Llama 3.3 70B: $0.88/$0.88 per 1M.
            # Pricing verified: https://together.ai/pricing (2026-05-11)
            pricing={"input_per_1m": 0.03, "output_per_1m": 0.12, "free_tier": False},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
