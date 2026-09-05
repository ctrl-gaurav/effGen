"""
Fireworks AI SDK adapter for effGen.

Supports all Fireworks chat models with built-in rate-limit coordination,
real streaming via SSE, native function-calling on supported models, and
per-request cost tracking via CostTracker.

Fireworks uses an OpenAI-compatible API shape (chat.completions.create) with
model IDs in the format ``accounts/fireworks/models/<id>``.
"""

from __future__ import annotations

import logging
import os
import random
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
from effgen.models.errors import ModelAuthError, ModelNotFoundError, error_has_status
from effgen.models.fireworks_models import (
    FIREWORKS_DEFAULT_MODEL,
    FIREWORKS_MODELS,
    REGISTRY_FETCH_DATE,
)
from effgen.models.latency_tracker import timed_call
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr
from effgen.utils.async_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)


def _not_found_message(model_name: str) -> str:
    """Explain a Fireworks "model not found" rejection.

    Fireworks answers a request for an undeployed model and a request the API
    key is not allowed to make with the same ``NOT_FOUND`` body, so the message
    names both causes instead of asserting one.
    """
    return (
        f"model '{model_name}' was not found. Fireworks returns this same "
        f"response for a model that is not deployed and for one this API key "
        f"cannot access, so check both. List deployed ids with: "
        f"from effgen.models.fireworks_models import available_models; "
        f"print(available_models())"
    )

_FIREWORKS_MODEL_TYPE_VALUE = "fireworks"
_FIREWORKS_PREFIX = "accounts/fireworks/models/"


class _FireworksModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _FIREWORKS_MODEL_TYPE_VALUE


class FireworksAdapter(BaseModel):
    """
    Adapter for Fireworks AI inference API.

    Wraps the ``fireworks-ai`` SDK (OpenAI-compatible) with the standard
    effGen BaseModel interface.  Supports:

    - Synchronous and async generation
    - Real token-by-token streaming (``generate_stream``)
    - Native function-calling on supported models (``generate_with_tools``)
    - Per-request cost tracking via :class:`~effgen.models._cost.CostTracker`
    - Per-model rate-limit coordination (RPM, TPM)
    - Dynamic model catalog refresh and drift detection (``refresh_models()``)

    Model IDs must use the full path format:
    ``accounts/fireworks/models/<id>``.
    Short IDs are accepted and automatically expanded.

    Args:
        model_name: Fireworks model ID.  Defaults to
            ``"accounts/fireworks/models/gpt-oss-120b"``.
        api_key: Fireworks API key.  If omitted, reads ``FIREWORKS_API_KEY``
            from the environment.
        max_retries: Maximum number of SDK retry attempts on transient errors.
        timeout: Per-request timeout in seconds.
        enable_rate_limiting: Wire built-in
            :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
        enable_cost_tracking: Record token usage in the global
            :class:`~effgen.models._cost.CostTracker`.
        warn_unknown_model: Emit a warning (not an error) when the model ID is
            not in the bundled registry.  Useful for newly-released models.

    Example::

        from effgen.models.fireworks_adapter import FireworksAdapter

        adapter = FireworksAdapter("accounts/fireworks/models/gpt-oss-120b")
        adapter.load()

        result = adapter.generate("What is the capital of France?")
        print(result.text)

        for chunk in adapter.generate_stream("Count from 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "fireworks"

    def __init__(
        self,
        model_name: str = FIREWORKS_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 6,
        timeout: int = 120,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        warn_unknown_model: bool = True,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        **kwargs: Any,
    ) -> None:
        # Normalise short IDs → full path
        if not model_name.startswith(_FIREWORKS_PREFIX):
            model_name = f"{_FIREWORKS_PREFIX}{model_name}"

        info = FIREWORKS_MODELS.get(model_name)
        if info is None and warn_unknown_model:
            logger.warning(
                "FireworksAdapter: model '%s' is not in the bundled registry "
                "(registry date: %s).  The model may be new — call "
                "fireworks_models.refresh_models() to check for drift.  "
                "Proceeding with default parameters.",
                model_name,
                REGISTRY_FETCH_DATE,
            )
            info = {}

        super().__init__(
            model_name=model_name,
            model_type=_FireworksModelType(),  # type: ignore[arg-type]
            context_length=(info or {}).get("context", 131_072),
        )
        # Every Fireworks chat/vision model emits a hidden reasoning chain before
        # any visible text. Flagging it here is what earns the larger default
        # token budget from default_max_output_tokens(), matching groq and
        # together; without it the model can spend a tight budget thinking and
        # return an empty, billed result.
        self._is_reasoning_model = bool((info or {}).get("reasoning", False))
        self._api_key = api_key
        # The retry loop below runs ``max_retries`` attempts, so a caller
        # asking for no retries at all must still get one request made.
        self.max_retries = max(1, int(max_retries))
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking

        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting and info:
            self._rate_limiter = RateLimitCoordinator(
                provider="fireworks",
                model=model_name,
                rpm=info.get("rpm", 10),
                rph=info.get("rpm", 10) * 60,
                rpd=10_000,
                tpm=info.get("tpm", 40_000),
                tph=info.get("tpm", 40_000) * 60,
                tpd=10_000_000,
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Fireworks SDK client.

        Raises:
            RuntimeError: If ``fireworks-ai`` is not installed.
            ValueError: If no API key is available.
        """
        try:
            from fireworks.client import Fireworks
        except ImportError as exc:
            raise RuntimeError(
                "fireworks-ai SDK is not installed. "
                "Install with: pip install 'effgen[fireworks]'"
            ) from exc

        if not (self._api_key or os.getenv("FIREWORKS_API_KEY")):
            raise ValueError(
                "Fireworks API key not found. Set the FIREWORKS_API_KEY "
                "environment variable or pass api_key= to FireworksAdapter."
            )

        self._client = Fireworks(
            api_key=self._api_key or os.getenv("FIREWORKS_API_KEY"),
            timeout=self.timeout,
        )
        self._is_loaded = True

        info = FIREWORKS_MODELS.get(self.model_name, {})
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "fireworks",
            "family": info.get("family", ""),
            "organization": info.get("organization", ""),
            "display_name": info.get("display_name", ""),
            "modality": info.get("modality", "chat"),
            "supports_native_tools": info.get("supports_native_tools", False),
            "supports_streaming": info.get("supports_streaming", True),
            "pricing_per_1m_input": info.get("pricing_per_1m_input", 0),
            "pricing_per_1m_output": info.get("pricing_per_1m_output", 0),
            "rpm": info.get("rpm"),
            "tpm": info.get("tpm"),
            "registry_fetch_date": REGISTRY_FETCH_DATE,
        }
        logger.info("FireworksAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("FireworksAdapter unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via the Fireworks AI API.

        Args:
            prompt: User prompt text.
            config: Optional generation config.
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with text and token usage.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("fireworks", self.model_name, "generate")

        if config is None:
            config = GenerationConfig()

        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(est_tokens))

        with timed_call("fireworks", self.model_name):
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
            raise not_loaded_error("fireworks", self.model_name, "async_generate")

        if config is None:
            config = GenerationConfig()

        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(est_tokens)

        result = self._do_generate(prompt, config, **kwargs)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

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

        info = FIREWORKS_MODELS.get(self.model_name, {})
        apply_tool_request(request_params, tools, info)

        request_params.update(kwargs)

        _last_exc: Exception | None = None
        for _attempt in range(1, self.max_retries + 1):
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
                    or "api_key" in msg_lower and "invalid" in msg_lower
                )
                if is_auth:
                    raise ModelAuthError(
                        provider="fireworks",
                        model_name=self.model_name,
                        message=msg,
                    ) from exc

                is_not_found = (
                    "NOT_FOUND" in msg
                    or "not found" in msg_lower
                    or "model not found" in msg_lower
                    or "inaccessible" in msg_lower
                    or "not deployed" in msg_lower
                )
                if is_not_found:
                    raise ModelNotFoundError(
                        provider="fireworks",
                        model_name=self.model_name,
                        message=_not_found_message(self.model_name),
                    ) from exc

                is_rate = (
                    error_has_status(exc, 429)
                    or "rate_limit" in msg_lower
                    or "rate limit" in msg_lower
                    or "exceeded your rate limit" in msg_lower
                )
                is_server = "500" in msg or "503" in msg or "internal" in msg_lower
                is_timeout = "timeout" in msg_lower

                if is_rate or is_server or is_timeout:
                    if _attempt >= self.max_retries:
                        logger.error(
                            "Fireworks API error after %d retries: %s", _attempt, exc
                        )
                        raise provider_runtime_error(
                            "fireworks", self.model_name, "generate", exc,
                            message=f"Fireworks API failed for {self.model_name} after "
                                    f"{self.max_retries} retries",
                        ) from exc
                    delay = min(60.0, 2.0 * (2 ** (_attempt - 1)) + random.uniform(0, 0.5))
                    logger.warning(
                        "Fireworks transient error on attempt %d/%d — retrying in %.1fs: %s",
                        _attempt, self.max_retries, delay, exc,
                    )
                    time.sleep(delay)
                    continue

                logger.error("Fireworks API call failed: %s", exc)
                raise provider_runtime_error("fireworks", self.model_name, "generate", exc, message="Fireworks generation failed") from exc
        else:
            if _last_exc is None:
                raise provider_runtime_error(
                    "fireworks", self.model_name, "generate",
                    RuntimeError("no request was made"),
                    message=(
                        f"Fireworks made no request for {self.model_name}: the "
                        f"retry budget was {self.max_retries} attempts"
                    ),
                )
            raise provider_runtime_error(
                "fireworks", self.model_name, "generate", _last_exc,
                message=f"Fireworks generation failed after {self.max_retries} retries",
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
                provider="fireworks",
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        logger.info(
            "Fireworks generated %d tokens (prompt=%d, completion=%d, cost=%s)",
            total_tokens, prompt_tokens, completion_tokens, cost_label(cost),
        )
        _set_span_attr(ModelAttrs.PROVIDER, "fireworks")
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
            "provider": "fireworks",
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
            raise not_loaded_error("fireworks", self.model_name, "generate_with_tools")
        if config is None:
            config = GenerationConfig()
        return self._do_generate(prompt, config, tools=tools, messages=messages, **kwargs)

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a response token-by-token from the Fireworks AI API.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Yields:
            str: Successive text chunks from the model.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("fireworks", self.model_name, "generate_stream")

        if config is None:
            config = GenerationConfig()

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

        # The same shaping the non-streaming path applies: the catalog gate and
        # ``tool_choice`` are one decision, so a streamed turn sends the same
        # request a non-streamed one would. A caller passing ``tools=`` as a raw
        # keyword used to reach the provider ungated and without ``tool_choice``.
        _stream_tools = kwargs.pop("tools", None)
        request_params.update(kwargs)
        apply_tool_request(
            request_params, _stream_tools, FIREWORKS_MODELS.get(self.model_name, {})
        )

        clear_stream_tool_calls(self)
        self._last_stream_finish_reason: str | None = None

        try:
            with timed_call("fireworks", self.model_name) as _stream_timer:
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
                    provider="fireworks",
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
                    provider="fireworks",
                    model_name=self.model_name,
                    message=msg,
                ) from exc
            if (
                "NOT_FOUND" in msg
                or "not found" in msg_lower
                or "not deployed" in msg_lower
            ):
                raise ModelNotFoundError(
                    provider="fireworks",
                    model_name=self.model_name,
                    message=_not_found_message(self.model_name),
                ) from exc
            logger.error("Fireworks streaming failed: %s", exc)
            raise provider_runtime_error("fireworks", self.model_name, "stream", exc, message="Fireworks streaming failed") from exc

    # ------------------------------------------------------------------
    # Token counting / context length
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken (cl100k_base)."""
        return TokenCount(count=estimate_tokens(text), model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return the context window size for the loaded model."""
        return int(FIREWORKS_MODELS.get(self.model_name, {}).get("context", 131_072))

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
        return bool(FIREWORKS_MODELS.get(self.model_name, {}).get("supports_native_tools", False))

    def supports_tool_calling(self) -> bool:
        """Return True if the loaded model supports native tool-calling."""
        return bool(FIREWORKS_MODELS.get(self.model_name, {}).get("supports_native_tools", False))

    def supports_forced_tool_call(self) -> bool:
        """True when tools are offered: ``tool_choice`` is honoured here.

        Fireworks' OpenAI-compatible endpoint enforces the
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
        return bool(FIREWORKS_MODELS.get(self.model_name, {}).get("supports_streaming", True))

    def pricing(self) -> dict[str, float]:
        """Return per-request pricing info for the loaded model."""
        info = FIREWORKS_MODELS.get(self.model_name, {})
        return {
            "input_per_1m_usd": float(info.get("pricing_per_1m_input", 0)),
            "output_per_1m_usd": float(info.get("pricing_per_1m_output", 0)),
        }


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.fireworks_models import FIREWORKS_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "fireworks",
            FireworksAdapter,
            FIREWORKS_MODELS,
            env_keys=["FIREWORKS_API_KEY"],
            capabilities={Capability.chat, Capability.streaming, Capability.tools, Capability.json_schema},
            # $1 free credits on signup; pay-per-token after. Provider default = cheapest chat model.
            # qwen3-1p7b: $0.10/$0.10; llama-v3p2-1b-instruct: $0.10/$0.10; llama-v3p3-70b: $0.90/$0.90 per 1M.
            # Pricing verified: https://fireworks.ai/pricing (2026-05-11)
            pricing={"input_per_1m": 0.10, "output_per_1m": 0.10, "free_tier": False},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
