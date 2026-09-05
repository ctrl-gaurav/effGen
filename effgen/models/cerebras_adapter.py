"""
Cerebras Cloud SDK adapter for effGen.

Supports all free-tier Cerebras models with built-in rate-limit coordination,
real streaming via SSE, native function-calling on supported models, and
per-request cost tracking via CostTracker.
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
    estimate_tokens,
    extract_reasoning_text,
    extract_reasoning_tokens,
    normalize_finish_reason,
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
from effgen.models.cerebras_models import (
    CEREBRAS_DEFAULT_MODEL,
    CEREBRAS_MODELS,
    available_models,
    free_tier_models,
    model_info,
)
from effgen.models.errors import ModelAuthError, ModelNotFoundError, error_has_status
from effgen.models.latency_tracker import timed_call
from effgen.observability import get_logger as _get_obs_logger
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr
from effgen.utils.async_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)
_obs_log = _get_obs_logger(__name__)

_CEREBRAS_MODEL_TYPE_VALUE = "cerebras"


class _CerebrasModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _CEREBRAS_MODEL_TYPE_VALUE


class CerebrasAdapter(BaseModel):
    """
    Adapter for Cerebras Cloud inference API.

    Wraps the ``cerebras-cloud-sdk`` with the standard effGen BaseModel
    interface. Supports:

    - Synchronous and async generation
    - Real token-by-token streaming (``generate_stream``)
    - Native function-calling on supported models (``generate_with_tools``)
    - Per-request cost tracking via :class:`~effgen.models._cost.CostTracker`
    - Per-model rate-limit coordination

    Args:
        model_name: Cerebras model ID. Must be a key in
            :data:`~effgen.models.cerebras_models.CEREBRAS_MODELS`.
            Defaults to ``"gpt-oss-120b"``.
        api_key: Cerebras API key. If omitted, reads ``CEREBRAS_API_KEY``
            from the environment.
        max_retries: Total attempts this adapter makes for one call. The
            provider SDK's own retry is switched off, so this is the whole
            budget rather than a multiplier on it.
        timeout: Per-request timeout in seconds.
        enable_rate_limiting: If ``True`` (default), acquire / record calls
            via the built-in :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
        enable_cost_tracking: If ``True`` (default), record token usage in
            the global :class:`~effgen.models._cost.CostTracker`.

    Example::

        from effgen.models.cerebras_adapter import CerebrasAdapter

        adapter = CerebrasAdapter("gpt-oss-120b")
        adapter.load()

        # Synchronous generation
        result = adapter.generate("What is the capital of France?")
        print(result.text)

        # Streaming
        for chunk in adapter.generate_stream("Count from 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "cerebras"

    def __init__(
        self,
        model_name: str = CEREBRAS_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        **kwargs: Any,
    ) -> None:
        if model_name not in CEREBRAS_MODELS:
            from effgen.models._catalog import suggest_for_missing

            raise ModelNotFoundError(
                provider="cerebras",
                model_name=model_name,
                message=f"Unknown Cerebras model '{model_name}'."
                        + suggest_for_missing("cerebras", model_name),
            )

        info = CEREBRAS_MODELS[model_name]
        super().__init__(
            model_name=model_name,
            model_type=_CerebrasModelType(),  # type: ignore[arg-type]
            context_length=info.get("context", 128_000),
        )
        # Both Cerebras models bill reasoning tokens before any visible text.
        # The flag earns them the larger default budget from
        # default_max_output_tokens(), so a first call is not spent thinking.
        self._is_reasoning_model = bool(info.get("reasoning", False))
        self._api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking

        # Rate-limit coordinator wired per-instance (in-memory)
        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting:
            self._rate_limiter = RateLimitCoordinator(
                provider="cerebras",
                model=model_name,
                rpm=info.get("rpm", 30),
                rph=info.get("rph", 900),
                rpd=info.get("rpd", 14_400),
                tpm=info.get("tpm", 60_000),
                tph=info.get("tph", 1_000_000),
                tpd=info.get("tpd", 1_000_000),
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Cerebras SDK client.

        Reads ``CEREBRAS_API_KEY`` from the environment if *api_key* was not
        passed to the constructor.

        Raises:
            RuntimeError: If ``cerebras-cloud-sdk`` is not installed.
            ValueError: If no API key is available.
        """
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError as exc:
            raise RuntimeError(
                "cerebras-cloud-sdk is not installed. "
                "Install with: pip install 'effgen[cerebras]'"
            ) from exc

        if not (self._api_key or os.getenv("CEREBRAS_API_KEY")):
            raise ValueError(
                "Cerebras API key not found. Set the CEREBRAS_API_KEY "
                "environment variable or pass api_key= to CerebrasAdapter."
            )

        api_key = self._api_key or os.getenv("CEREBRAS_API_KEY")
        try:
            self._client = Cerebras(
                api_key=api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        except TypeError:
            self._client = Cerebras(api_key=api_key)
        self._is_loaded = True

        info = CEREBRAS_MODELS.get(self.model_name, {})
        if not info.get("free_tier", False):
            logger.warning(
                "Cerebras model '%s' is not reliably callable on the free tier "
                "(high demand / restricted access). If you have a paid-tier key "
                "this will work; otherwise consider a free-tier model: %s",
                self.model_name, free_tier_models(),
            )
        if info.get("deprecated"):
            logger.warning(
                "Cerebras model '%s' is scheduled for deprecation on %s.",
                self.model_name, info["deprecated"],
            )
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "cerebras",
            "free_tier": CEREBRAS_MODELS[self.model_name].get("free_tier", False),
            "supports_native_tools": CEREBRAS_MODELS[self.model_name].get("supports_native_tools", False),
        }
        logger.info("CerebrasAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("CerebrasAdapter unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via the Cerebras API.

        Args:
            prompt: User prompt text.
            config: Optional generation config (temperature, max_tokens, …).
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with the generated text and token usage.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("cerebras", self.model_name, "generate")

        if config is None:
            config = GenerationConfig()

        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(est_tokens))

        with timed_call("cerebras", self.model_name):
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
            raise not_loaded_error("cerebras", self.model_name, "async_generate")

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

        if config.temperature != 0.7:
            request_params["temperature"] = config.temperature
        if config.top_p != 0.9:
            request_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            request_params["max_completion_tokens"] = config.max_tokens
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty

        # Attach tools if provided and model supports them
        model_info_dict = CEREBRAS_MODELS.get(self.model_name, {})
        if tools and model_info_dict.get("supports_native_tools", False):
            openai_tools = []
            for t in tools:
                if isinstance(t, dict):
                    # Already serialized (e.g. from agent's format_tools_for_prompt)
                    openai_tools.append(t if "type" in t else {"type": "function", "function": t})
                else:
                    # BaseTool object — convert to OpenAI format
                    openai_tools.append({"type": "function", "function": t.metadata.to_json_schema()})
            request_params["tools"] = openai_tools
            request_params["tool_choice"] = "auto"

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
                if error_has_status(exc, 401) or "wrong_api_key" in msg_lower or "wrong api key" in msg_lower:
                    raise ModelAuthError("cerebras", self.model_name, str(exc)) from exc
                if "404" in msg and "model_not_found" in msg:
                    from effgen.models._catalog import suggest_for_missing

                    hint = suggest_for_missing("cerebras", self.model_name)
                    logger.error("Cerebras API 404 for model '%s': %s", self.model_name, exc)
                    raise ModelNotFoundError("cerebras", self.model_name, str(exc) + hint) from exc
                is_rate = error_has_status(exc, 429) or "rate_limit" in msg_lower
                # Cerebras "queue_exceeded" / "high traffic" — transient backpressure that needs
                # longer waits than per-minute rate-limits. Treat as retryable but with a longer cap.
                is_queue = "queue_exceeded" in msg_lower or "high traffic" in msg_lower
                is_server = "500" in msg or "503" in msg or "internal server" in msg_lower
                if is_rate:
                    # Raise RateLimitExceeded so the router can failover to another provider
                    from effgen.models._rate_limit import RateLimitExceeded as _RLE
                    raise _RLE(
                        f"Cerebras rate limit hit for {self.model_name}: {exc}"
                    ) from exc
                if is_queue:
                    if _attempt >= _MAX_RETRIES:
                        logger.error("Cerebras queue exceeded after %d retries: %s", _attempt, exc)
                        from effgen.models._rate_limit import RateLimitExceeded as _RLE
                        raise _RLE(
                            f"Cerebras queue_exceeded for {self.model_name}: {exc}"
                        ) from exc
                    delay = min(60.0, 4.0 * (2 ** (_attempt - 1)) + random.uniform(0, 0.5))
                    logger.warning(
                        "Cerebras queue_exceeded on attempt %d/%d — retrying in %.1fs",
                        _attempt, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                if is_server:
                    if _attempt >= _MAX_RETRIES:
                        from effgen.models.errors import ProviderTransientError as _PTE
                        raise _PTE(
                            provider="cerebras", model_name=self.model_name, status_code=500,
                            message=f"Cerebras server error after {_MAX_RETRIES} retries: {exc}",
                        ) from exc
                    delay = min(60.0, 2.0 * (2 ** (_attempt - 1)) + random.uniform(0, 0.5))
                    logger.warning(
                        "Cerebras server error on attempt %d/%d — retrying in %.1fs: %s",
                        _attempt, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Cerebras API call failed: %s", exc)
                raise provider_runtime_error("cerebras", self.model_name, "generate", exc, message="Cerebras generation failed") from exc
        else:
            assert _last_exc is not None
            raise provider_runtime_error("cerebras", self.model_name, "generate", _last_exc, message=f"Cerebras generation failed after {_MAX_RETRIES} retries") from _last_exc

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        finish_reason = normalize_finish_reason(choice.finish_reason)

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        tool_calls = tool_calls_from_message(message)

        # Cost tracking
        cost: float | None = None
        if self._enable_cost_tracking:
            cost = CostTracker.get().record(
                provider="cerebras",
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        logger.info(
            "Cerebras generated %d tokens (prompt=%d, completion=%d, cost=%s)",
            total_tokens, prompt_tokens, completion_tokens, cost_label(cost),
        )
        _obs_log.model_event(
            "call.done",
            provider="cerebras",
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
        )
        # Emit span attributes on the current active span
        _set_span_attr(ModelAttrs.PROVIDER, "cerebras")
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
            "provider": "cerebras",
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
            max_tokens=request_params.get("max_completion_tokens"),
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

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a response token-by-token from the Cerebras API.

        Uses the SDK's ``stream=True`` mode.  Yields text deltas as they
        arrive.  Usage statistics (from the terminal chunk) are logged after
        the stream ends.

        Args:
            prompt: User prompt text.
            config: Optional generation config.
            **kwargs: Forwarded to ``chat.completions.create``.

        Yields:
            str: Successive text chunks from the model.

        Raises:
            RuntimeError: If ``load()`` has not been called or the stream fails.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("cerebras", self.model_name, "generate_stream")

        if config is None:
            config = GenerationConfig()

        messages = [{"role": "user", "content": prompt}]
        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }

        if config.temperature != 0.7:
            request_params["temperature"] = config.temperature
        if config.top_p != 0.9:
            request_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            request_params["max_completion_tokens"] = config.max_tokens
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty

        request_params.update(kwargs)

        # Optional per-call buffer for tool_calls that may appear
        # incrementally across chunks or only on the terminal chunk.
        clear_stream_tool_calls(self)
        self._last_stream_finish_reason: str | None = None

        try:
            with timed_call("cerebras", self.model_name) as _stream_timer:
                _first_token = True
                stream = self._client.chat.completions.create(**request_params)

                prompt_tokens = 0
                completion_tokens = 0
                tool_calls_buf: dict[int, dict[str, Any]] = {}
                reasoning_buf: list[str] = []
                stream_usage: Any = None

                for chunk in stream:
                    # Capture usage from the terminal chunk (some SDKs attach
                    # it here as a trailing chunk with an empty `choices`
                    # list) before the no-choices skip below, or its token
                    # counts are silently dropped and cost/tokens never
                    # get recorded.
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

                    # Accumulate tool-call fragments (OpenAI-style streaming of
                    # tool_calls: name once, arguments can be chunked).
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
                    max_tokens=request_params.get("max_completion_tokens"),
                    tool_calls=streamed_calls,
                    logger=logger,
                )

            # Cost tracking after stream completes
            if self._enable_cost_tracking and (prompt_tokens or completion_tokens):
                cost = CostTracker.get().record(
                    provider="cerebras",
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

            if self._rate_limiter is not None:
                self._rate_limiter.record(prompt_tokens + completion_tokens)

            logger.info(
                "Cerebras stream complete: prompt=%d completion=%d",
                prompt_tokens, completion_tokens,
            )

        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            if "404" in msg and "model_not_found" in msg:
                from effgen.models._catalog import suggest_for_missing

                raise ModelNotFoundError(
                    "cerebras",
                    self.model_name,
                    f"streaming failed (model not found): {exc}"
                    + suggest_for_missing("cerebras", self.model_name),
                ) from exc
            if error_has_status(exc, 429) or "rate_limit" in msg_lower:
                from effgen.models._rate_limit import RateLimitExceeded as _RLE
                raise _RLE(f"Cerebras rate limit hit for {self.model_name}: {msg}") from exc
            if "500" in msg or "503" in msg or "internal server" in msg_lower:
                from effgen.models.errors import ProviderTransientError as _PTE
                raise _PTE(provider="cerebras", model_name=self.model_name, status_code=500, message=msg) from exc
            logger.error("Cerebras streaming failed: %s", exc)
            raise provider_runtime_error("cerebras", self.model_name, "stream", exc, message="Cerebras streaming failed") from exc

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with native function-calling support.

        On models that support native tools (``supports_native_tools=True``),
        the tool definitions are passed directly to the API.  Tool calls are
        parsed from ``choice.message.tool_calls`` and placed in
        ``result.metadata["tool_calls"]``.

        On models that do **not** support native tools, raises
        ``NotImplementedError`` so the agent can fall back to ReAct.

        Args:
            prompt: User prompt.
            tools: List of tool definitions in OpenAI function-call format.
            config: Generation config.
            messages: Optional full message list (overrides prompt).
            **kwargs: Forwarded to the SDK.

        Returns:
            GenerationResult with ``metadata["tool_calls"]`` populated if the
            model requested one or more tools.

        Raises:
            NotImplementedError: If the model doesn't support native tools.
            RuntimeError: If the adapter is not loaded or the API call fails.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("cerebras", self.model_name, "generate_with_tools")

        model_info_dict = CEREBRAS_MODELS.get(self.model_name, {})
        if not model_info_dict.get("supports_native_tools", False):
            raise NotImplementedError(
                f"Cerebras model '{self.model_name}' does not support native tool-calling. "
                "Use ReAct strategy or choose a tool-capable model."
            )

        if config is None:
            config = GenerationConfig()

        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(est_tokens))

        result = self._do_generate(
            prompt=prompt,
            config=config,
            tools=tools,
            messages=messages,
            **kwargs,
        )

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    def supports_tool_calling(self) -> bool:
        """Return True if the loaded model supports native tool-calling."""
        return CEREBRAS_MODELS.get(self.model_name, {}).get("supports_native_tools", False)

    def supports_forced_tool_call(self) -> bool:
        """True when tools are offered: ``tool_choice`` is honoured here.

        Cerebras' OpenAI-compatible endpoint enforces the
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

    # ------------------------------------------------------------------
    # Token counting & context length
    # ------------------------------------------------------------------

    # Empirical per-family multiplier applied to tiktoken counts.
    # final = raw * mult + fixed
    _CHAT_TEMPLATE_MULTIPLIER = {
        "llama":   (1.5, 25),
        "qwen":    (1.3, 5),
        "gpt-oss": (1.3, 10),
        "zai-glm": (1.3, 10),
    }

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken with a per-family chat-template adjustment.

        The chat-template adjustment is applied to whichever base count is
        available: the BPE count when tiktoken can load its data, a length-based
        estimate when it cannot.
        """
        raw = estimate_tokens(text, model="gpt-4")
        family = CEREBRAS_MODELS.get(self.model_name, {}).get("family", "")
        mult, fixed = self._CHAT_TEMPLATE_MULTIPLIER.get(family, (1.3, 10))
        adjusted = int(raw * mult) + fixed

        return TokenCount(count=adjusted, model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return context window size for the loaded model."""
        return CEREBRAS_MODELS.get(self.model_name, {}).get("context", 128_000)

    def get_max_output(self) -> int:
        """Return maximum completion tokens for the loaded model."""
        return CEREBRAS_MODELS.get(self.model_name, {}).get("max_output", 8_192)

    def rate_limit_status(self) -> dict:
        """Return a snapshot of the rate-limit coordinator state."""
        if self._rate_limiter is None:
            return {}
        return self._rate_limiter.status()

    def cost_summary(self) -> list[dict]:
        """Return the global CostTracker summary filtered to Cerebras."""
        return [
            row for row in CostTracker.get().summary()
            if row["provider"].lower() == "cerebras"
        ]

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_models(cls) -> list[str]:
        """Return all registered Cerebras model IDs."""
        return available_models()

    @classmethod
    def list_free_tier_models(cls) -> list[str]:
        """Return model IDs callable on the Cerebras free tier."""
        return free_tier_models()

    @classmethod
    def get_model_info(cls, model_id: str) -> dict:
        """Return metadata for *model_id* (context, limits, native-tools flag, etc.)."""
        return model_info(model_id)


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.cerebras_models import CEREBRAS_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "cerebras",
            CerebrasAdapter,
            CEREBRAS_MODELS,
            env_keys=["CEREBRAS_API_KEY"],
            capabilities={Capability.chat, Capability.streaming, Capability.tools, Capability.json_schema},
            # Cerebras does not publish per-token pricing publicly; free tier available.
            # Pricing verified: https://www.cerebras.ai/pricing (2026-05-11)
            # Per-model prices added in CEREBRAS_MODELS when available.
            pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
