"""
HuggingFace Inference API adapter for effGen.

Supports two modes:
  1. Serverless Inference API — point at any public HF model ID (free tier).
  2. Dedicated Inference Endpoints — set ModelConfig.endpoint_url (or pass
     endpoint_url= to the constructor) to route to a private endpoint.

Key behaviours
--------------
- generate()         — synchronous, via InferenceClient.chat_completion().
- generate_stream()  — real SSE streaming, via stream=True.
- text_generation()  — fallback for base / non-chat models.
- ModelUnavailableError — raised on 503/404 with helpful alternative suggestions.
- Native tools       — models that advertise supports_native_tools=True in the
                       registry get function-calling; others fall back to ReAct.
- RateLimitCoordinator — wired best-effort (HF limits are per-token/tier, not
                       easily modelled — see docstring below for the known gap).
- CostTracker        — free-tier: $0; Pro-tier/Endpoints: registry rates recorded.

Rate-limit gap
--------------
HuggingFace Serverless Inference enforces limits by user tier (free/PRO/Enterprise)
and per-provider-backend (e.g. novita, sambanova) rather than simple RPM/TPM
windows.  The RateLimitCoordinator is wired with conservative defaults so it
acts as a local circuit-breaker and does not over-call the API; it does NOT
accurately model HF's actual server-side limits.  This gap is documented here
and in docs/models/hf_inference.md.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NoReturn

from effgen.models._adapter_utils import (
    annotate_reasoning_only,
    extract_reasoning_text,
    extract_reasoning_tokens,
    normalize_finish_reason,
    not_loaded_error,
    provider_runtime_error,
    reasoning_delta_text,
    warn_reasoning_only_stream,
)
from effgen.models._cost import CostTracker
from effgen.models._multimodal import require_audio_support, require_vision_support
from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models._usage import tool_calls_from_message
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    TokenCount,
    accumulate_stream_cost,
)
from effgen.models.errors import (
    BudgetExceededError,
    ModelAuthError,
    ModelNotFoundError,
    ModelUnavailableError,
)
from effgen.models.hf_inference_models import (
    HF_DEFAULT_MODEL,
    HF_MODELS,
    REGISTRY_FETCH_DATE,
    suggest_alternatives,
)
from effgen.models.latency_tracker import timed_call
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr
from effgen.utils.async_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)

_HF_MODEL_TYPE_VALUE = "hf_inference"

# HTTP status codes that indicate a model is temporarily unavailable
_UNAVAILABLE_STATUS = {503, 404}
# HTTP status codes that indicate auth failure
_AUTH_STATUS = {401, 403}


def _messages_text(messages: list[dict[str, Any]]) -> str:
    """Flatten a chat message list to the text a token estimate can measure.

    Multimodal content arrives as a list of parts; only the text parts carry
    characters to count.
    """
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "".join(parts)


class _HFModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _HF_MODEL_TYPE_VALUE


class HFInferenceAdapter(BaseModel):
    """
    Adapter for the HuggingFace Inference API (Serverless + Endpoints).

    Wraps :class:`huggingface_hub.InferenceClient` behind the standard effGen
    ``BaseModel`` interface.  Supports chat completion, streaming, and native
    function-calling on models that advertise it.

    Args:
        model_name: HuggingFace model ID (``"org/name"``), e.g.
            ``"Qwen/Qwen2.5-7B-Instruct"``.  Ignored when *endpoint_url* is set
            (the endpoint already encodes the model).  Defaults to
            ``"Qwen/Qwen2.5-7B-Instruct"``.
        api_token: HuggingFace access token.  Falls back to the ``HF_TOKEN``
            and ``HUGGINGFACE_API_KEY`` environment variables.  ``api_key`` is
            accepted as an alias.
        endpoint_url: URL of a dedicated Inference Endpoint.  When set, all
            requests are routed to that URL instead of the public Serverless API.
        timeout: Per-request timeout in seconds (default 120s).
        max_retries: Retry attempts on transient errors.
        enable_rate_limiting: Wire built-in
            :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
            Conservative defaults only — see module docstring for the gap.
        enable_cost_tracking: Record token usage in the global
            :class:`~effgen.models._cost.CostTracker`.
        warn_unknown_model: Emit a warning when the model is not in the
            bundled registry (still works — just without registry metadata).

    Example — Serverless::

        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct")
        adapter.load()

        result = adapter.generate("What is the capital of France?")
        print(result.text)

        for chunk in adapter.generate_stream("Count 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()

    Example — Dedicated Endpoint::

        adapter = HFInferenceAdapter(
            model_name="my-private-model",
            endpoint_url="https://my-endpoint.endpoints.huggingface.cloud",
        )
        adapter.load()
        result = adapter.generate("Hello")
    """

    def __init__(
        self,
        model_name: str = HF_DEFAULT_MODEL,
        api_token: str | None = None,
        endpoint_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        warn_unknown_model: bool = True,
        provider: str | None = None,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        info = HF_MODELS.get(model_name)
        if info is None and warn_unknown_model:
            logger.warning(
                "HFInferenceAdapter: model '%s' is not in the bundled registry "
                "(registry date: %s).  Proceeding with default parameters.",
                model_name,
                REGISTRY_FETCH_DATE,
            )
            info = {}

        super().__init__(
            model_name=model_name,
            model_type=_HFModelType(),  # type: ignore[arg-type]
            context_length=(info or {}).get("context", 8_192),
        )

        # ``api_key=`` is the name every other adapter uses; accept it here so
        # a credential passed under that name is used, not dropped into kwargs.
        self._api_token = api_token or api_key
        self._endpoint_url = endpoint_url
        self.timeout = timeout
        # At least one attempt: a 0/negative budget would skip the request
        # loop entirely and leave the caller with no response and no error.
        self.max_retries = max(1, int(max_retries))
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking
        self._info: dict[str, Any] = info or {}
        # Usage block from the final chunk of a streamed call, when the
        # endpoint sends one; otherwise the streamed call is estimated.
        self._last_stream_api_usage: Any = None
        # Provider routing: explicit arg > registry hint > "auto" (HF Router picks).
        # The legacy "hf-inference" backend no longer hosts most chat models — the
        # default is "auto" so the HF Router selects an available backend (Together,
        # Novita, Sambanova, etc.) per request.
        self._provider = provider or (info or {}).get("provider", "auto")

        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting:
            # Conservative defaults — HF limits are tier-based, not simple RPM/TPM.
            self._rate_limiter = RateLimitCoordinator(
                provider="hf_inference",
                model=model_name,
                rpm=10,
                rph=300,
                rpd=10_000,
                tpm=100_000,
                tph=2_000_000,
                tpd=20_000_000,
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the HuggingFace InferenceClient.

        Raises:
            RuntimeError: If ``huggingface_hub`` is not installed.
            ValueError: If no API token is available.
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed.  "
                "Install with: pip install 'effgen[hf]'"
            ) from exc

        def _resolve_token() -> str | None:
            return (
                self._api_token
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_API_KEY")
            )

        if not _resolve_token():
            raise ValueError(
                "HuggingFace API token not found.  Set the HF_TOKEN "
                "environment variable or pass api_token= to HFInferenceAdapter."
            )

        if self._endpoint_url:
            self._client = InferenceClient(
                base_url=self._endpoint_url,
                token=_resolve_token(),
                timeout=self.timeout,
            )
        else:
            client_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "token": _resolve_token(),
                "timeout": self.timeout,
            }
            if self._provider:
                client_kwargs["provider"] = self._provider
            self._client = InferenceClient(**client_kwargs)

        self._is_loaded = True

        info = self._info
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "hf_inference",
            "family": info.get("family", ""),
            "organization": info.get("organization", ""),
            "display_name": info.get("display_name", ""),
            "supports_native_tools": info.get("supports_native_tools", False),
            "supports_structured_output": info.get("supports_structured_output", False),
            "input_modalities": info.get("input_modalities", ["text"]),
            "output_modalities": info.get("output_modalities", ["text"]),
            "requires_endpoint": info.get("requires_endpoint", False),
            "endpoint_url": self._endpoint_url or "",
            "pricing_per_1m_input": info.get("pricing_per_1m_input", 0.0),
            "pricing_per_1m_output": info.get("pricing_per_1m_output", 0.0),
            "preferred_provider": info.get("preferred_provider", ""),
            "router_provider": (
                "" if self._endpoint_url else (self._provider or "auto")
            ),
            "registry_fetch_date": REGISTRY_FETCH_DATE,
        }
        logger.info(
            "HFInferenceAdapter loaded for model '%s'%s",
            self.model_name,
            f" (endpoint: {self._endpoint_url})" if self._endpoint_url else "",
        )

    def unload(self) -> None:
        """Release client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("HFInferenceAdapter unloaded")

    # ------------------------------------------------------------------
    # Token counting (approximate)
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Approximate token count (4 chars ≈ 1 token)."""
        count = max(1, len(text) // 4)
        return TokenCount(count=count, model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return the model's context window size (8192 when unknown)."""
        return self._info.get("context", 8_192)

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def _raise_for_unavailable(self, exc: Exception, context: str = "") -> NoReturn:
        """Convert an HF HTTP error into a typed effGen exception.

        Always raises: every branch ends in a typed error or the shared
        classified wrapper, so a caller can rely on this never returning.
        """
        exc_str = str(exc)
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        status_match = re.search(r"\b(401|403|404|503)\b", exc_str)
        status_text = status_match.group(1) if status_match else ""

        # Auth errors
        if status_code in {401, 403} or status_text in {"401", "403"}:
            raise ModelAuthError(
                provider="hf_inference",
                model_name=self.model_name,
                message=exc_str[:200],
            ) from exc

        # Model that doesn't exist on the Hub at all (typo, deleted, private).
        # Distinct from "exists but unavailable" — surface as ModelNotFoundError.
        if "does not exist" in exc_str or "model_not_found" in exc_str:
            raise ModelNotFoundError(
                provider="hf_inference",
                model_name=self.model_name,
                message=exc_str[:200],
            ) from exc

        # Model unavailable: 503 / 404 / "Not Found" / "Service Temporarily Unavailable"
        # Also: HF Router 400 "Model not supported by provider <x>" — the model
        # exists on Hub but no provider currently serves it.  Treat as unavailable
        # and surface helpful suggestions.
        is_unsupported_400 = (
            "Model not supported by provider" in exc_str
            or "is not supported for task" in exc_str
            or "is not supported by any provider" in exc_str
            or "model_not_supported" in exc_str
            or "no available providers" in exc_str.lower()
        )
        if (
            status_code in {404, 503}
            or status_text in {"404", "503"}
            or "Not Found" in exc_str
            or "Service Temporarily Unavailable" in exc_str
            or is_unsupported_400
        ):
            alts = suggest_alternatives(self.model_name)
            raise ModelUnavailableError(
                provider="hf_inference",
                model_name=self.model_name,
                suggestions=alts,
                message=(
                    f"Model is temporarily unavailable on HuggingFace Serverless "
                    f"Inference{' (' + context + ')' if context else ''}.  "
                    f"This is common for models that rotate in/out of the free tier."
                ),
            ) from exc

        # Everything else keeps the same classified, redacted shape as the
        # other adapters (a 402 or a transport failure lands here).
        raise provider_runtime_error(
            "hf_inference", self.model_name, context or "request", exc,
            message=f"HuggingFace Inference request failed for '{self.model_name}'",
        ) from exc

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _effgen_message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert an effGen Message to a HF/OpenAI-compatible dict.

        AudioPart is transcribed via automatic_speech_recognition before
        being injected as a TextPart so the chat model sees the transcript.
        """
        import base64

        from effgen.core.messages import AudioPart, ImagePart, TextPart, VideoPart
        from effgen.multimodal.image_pre import prepare as _preprocess_image

        role = message.role.value
        content_parts: list[dict[str, Any]] = []

        for part in message.content:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                processed = _preprocess_image(part, "hf_inference", "")
                b64 = base64.b64encode(processed.image).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{processed.mime};base64,{b64}"},
                })
            elif isinstance(part, AudioPart):
                # Transcribe via HF ASR then inject as text
                transcript = self._transcribe_audio_part(part)
                content_parts.append({
                    "type": "text",
                    "text": (
                        "[Audio transcript generated by effGen before this request]\n"
                        f"{transcript or '[no speech detected]'}\n"
                        "Use this transcript as the audio content for the user request."
                    ),
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

    def _transcribe_audio_part(self, part: Any) -> str:
        """Transcribe a single AudioPart via HF automatic_speech_recognition."""
        from effgen.multimodal.audio_pre import chunk as _chunk_audio

        chunks = _chunk_audio(part, "hf_inference", self.model_name)
        transcripts: list[str] = []
        for audio_chunk in chunks:
            text = self._call_hf_asr(audio_chunk.audio, audio_chunk.mime)
            if text:
                transcripts.append(text)
        return " ".join(transcripts).strip()

    def _call_hf_asr(self, audio_bytes: bytes, mime: str) -> str:
        """Call HF automatic_speech_recognition for a single audio chunk."""
        try:
            result = self._client.automatic_speech_recognition(
                audio=audio_bytes,
                model="openai/whisper-large-v3",
            )
            return result.text or ""
        except Exception as exc:
            logger.error("HF ASR failed: %s", exc)
            raise provider_runtime_error("hf_inference", "openai/whisper-large-v3", "transcribe", exc, message="HF audio transcription failed") from exc

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime: str = "audio/mp3",
        asr_model: str = "openai/whisper-large-v3",
    ) -> str:
        """Transcribe *audio_bytes* using the HF automatic_speech_recognition endpoint.

        Args:
            audio_bytes: Raw audio bytes.
            mime: MIME type (e.g. ``"audio/mp3"``).
            asr_model: HF model ID for ASR (default: ``"openai/whisper-large-v3"``).

        Returns:
            Transcribed text.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("hf_inference", self.model_name, "transcribe_audio")

        from effgen.core.messages import AudioPart
        from effgen.multimodal.audio_pre import chunk as _chunk_audio

        part = AudioPart(audio=audio_bytes, mime=mime)
        chunks = _chunk_audio(part, "hf_inference", self.model_name)
        transcripts: list[str] = []
        for audio_chunk in chunks:
            try:
                result = self._client.automatic_speech_recognition(
                    audio=audio_chunk.audio,
                    model=asr_model,
                )
                transcripts.append(result.text or "")
            except Exception as exc:
                logger.error("HF ASR transcription failed: %s", exc)
                raise provider_runtime_error("hf_inference", asr_model, "transcribe", exc, message="HF audio transcription failed") from exc
        return " ".join(transcripts).strip()

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if messages is not None:
            return messages

        # Handle effGen Message objects
        try:
            from effgen.core.messages import Message
            if isinstance(prompt, Message):
                return [self._effgen_message_to_dict(prompt)]
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                return [self._effgen_message_to_dict(m) for m in prompt]
        except ImportError:
            pass

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    # ------------------------------------------------------------------
    # Generation (non-streaming)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response via HuggingFace Inference API.

        Uses ``chat_completion()`` for chat models; ``text_generation()`` for
        base models (``use_text_generation=True`` in registry).

        Args:
            prompt: User prompt.
            config: Optional generation config.
            **kwargs: Forwarded to the underlying API call.

        Returns:
            GenerationResult with text, usage, and metadata.

        Raises:
            RuntimeError: If not loaded.
            ModelAuthError: On authentication failure.
            ModelUnavailableError: When the model is not available on Serverless.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("hf_inference", self.model_name, "generate")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="hf_inference",
            model_name=self.model_name,
            supports_vision=self._info.get("supports_vision", False),
            hint="Use an HF Inference model whose input_modalities include image.",
        )
        # Audio is supported via HF ASR transcription before chat completion.
        require_audio_support(
            prompt,
            provider="hf_inference",
            model_name=self.model_name,
            supports_audio=True,
            hint="Audio is transcribed via HF ASR (Whisper) before chat completion.",
        )

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(500))

        tools = kwargs.pop("tools", None)
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", "You are a helpful assistant.")

        use_text_gen = self._info.get("use_text_generation", False)

        with timed_call("hf", self.model_name):
            if use_text_gen:
                result = self._generate_text(prompt, config, **kwargs)
            else:
                msgs = self._build_messages(prompt, system_prompt, messages)
                result = self._generate_chat(msgs, config, tools=tools, **kwargs)

        if self._rate_limiter is not None:
            total = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(total)

        return result

    def _generate_chat(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Internal: call chat_completion() and assemble GenerationResult."""
        call_kwargs: dict[str, Any] = {}
        if config.max_tokens is not None:
            call_kwargs["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            call_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            call_kwargs["top_p"] = config.top_p
        if config.stop_sequences:
            call_kwargs["stop"] = config.stop_sequences
        if config.seed is not None:
            call_kwargs["seed"] = config.seed

        if tools and self._info.get("supports_native_tools"):
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        call_kwargs.update(kwargs)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat_completion(
                    messages=messages,
                    model=None if self._endpoint_url else self.model_name,
                    **call_kwargs,
                )
                break
            except Exception as exc:
                if attempt < self.max_retries and self._is_transient(exc):
                    import time
                    time.sleep(2 ** attempt)
                    continue
                self._raise_for_unavailable(exc)

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish_reason = normalize_finish_reason(choice.finish_reason)

        tool_calls = tool_calls_from_message(choice.message)

        # Usage
        usage = resp.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)

        return self._build_result(
            text=text,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            reasoning_text=extract_reasoning_text(choice.message),
            reasoning_tokens=extract_reasoning_tokens(usage),
            max_tokens=call_kwargs.get("max_tokens"),
        )

    def _generate_text(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs: Any,
    ) -> GenerationResult:
        """Internal: call text_generation() for base (non-chat) models."""
        call_kwargs: dict[str, Any] = {
            "details": True,
        }
        if config.max_tokens is not None:
            call_kwargs["max_new_tokens"] = config.max_tokens
        if config.temperature is not None:
            call_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            call_kwargs["top_p"] = config.top_p
        if config.stop_sequences:
            call_kwargs["stop_sequences"] = config.stop_sequences
        call_kwargs.update(kwargs)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.text_generation(
                    prompt,
                    model=None if self._endpoint_url else self.model_name,
                    **call_kwargs,
                )
                break
            except Exception as exc:
                if attempt < self.max_retries and self._is_transient(exc):
                    import time
                    time.sleep(2 ** attempt)
                    continue
                self._raise_for_unavailable(exc)

        if hasattr(resp, "generated_text"):
            text = resp.generated_text or ""
            details = getattr(resp, "details", None)
            output_tokens = (
                len(getattr(details, "tokens", []))
                if details
                else self._estimate_tokens(text)
            )
            input_tokens = self._estimate_tokens(prompt)
        else:
            text = str(resp)
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(text)

        total_tokens = input_tokens + output_tokens
        return self._build_result(
            text=text,
            finish_reason="stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tool_calls=[],
        )

    def _price_tokens(self, input_tokens: int, output_tokens: int) -> float | None:
        """Price a call from the registry's per-1M-token rates.

        Returns ``None`` for a model the registry carries no input price for:
        an unpriced Serverless model reports no cost rather than a ``$0`` a
        reader cannot tell apart from a genuinely free call.  A call that sent
        no input tokens is priced at ``0.0``.
        """
        info = self._info
        price_in = info.get("pricing_per_1m_input", 0.0)
        price_out = info.get("pricing_per_1m_output", 0.0)
        if not price_in:
            return None
        if not input_tokens:
            return 0.0
        return (
            input_tokens * price_in / 1_000_000
            + output_tokens * price_out / 1_000_000
        )

    def _record_cost(
        self, input_tokens: int, output_tokens: int, cost_usd: float | None
    ) -> None:
        """Record a completed call's usage on the global ledger.

        A call the registry publishes no rate for (``cost_usd`` is ``None``) is
        recorded too: its tokens are known even though its price is not, so it
        appears on the ledger as an unpriced call rather than vanishing from
        ``effgen cost`` altogether.
        """
        if not self._enable_cost_tracking:
            return
        try:
            CostTracker.get().record(
                provider="hf_inference",
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        except BudgetExceededError:
            raise
        except Exception:
            logger.debug("CostTracker recording failed for HF Inference", exc_info=True)

    @staticmethod
    def _estimate_tokens_from_chars(char_count: int) -> int:
        """Approximate a token count as four characters per token.

        Used wherever the endpoint reports no token counts of its own: a
        ``text_generation`` response without token details, and a streamed call
        whose final chunk carried no usage block.
        """
        return max(1, char_count // 4) if char_count else 0

    def _estimate_tokens(self, text: str) -> int:
        """Approximate the token count of ``text``."""
        return self._estimate_tokens_from_chars(len(text))

    def _build_result(
        self,
        text: str,
        finish_reason: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        tool_calls: list[dict[str, Any]],
        reasoning_text: str = "",
        reasoning_tokens: int = 0,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Assemble a GenerationResult and record cost."""
        cost_usd = self._price_tokens(input_tokens, output_tokens)
        self._record_cost(input_tokens, output_tokens, cost_usd)

        metadata = {
            "provider": "hf_inference",
            "model_name": self.model_name,
            "endpoint_url": self._endpoint_url or "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Canonical aliases (OpenAI-style) so downstream token/cost
            # accounting reads the same keys across every provider.
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "cost_usd": cost_usd,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }

        annotate_reasoning_only(
            metadata,
            text=text,
            reasoning_text=reasoning_text,
            reasoning_tokens=reasoning_tokens,
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=max_tokens,
            completion_tokens=output_tokens,
            tool_calls=tool_calls,
            logger=logger,
        )

        _set_span_attr(ModelAttrs.PROVIDER, "hf_inference")
        _set_span_attr(ModelAttrs.NAME, self.model_name)
        _set_span_attr(ModelAttrs.INPUT_TOKENS, input_tokens)
        _set_span_attr(ModelAttrs.OUTPUT_TOKENS, output_tokens)
        if cost_usd is not None:
            _set_span_attr(ModelAttrs.COST_USD, float(cost_usd))
        _set_span_attr(ModelAttrs.OUTCOME, "ok")

        return GenerationResult(
            text=text,
            tokens_used=total_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream tokens from HuggingFace Inference via SSE.

        Uses ``chat_completion(stream=True)`` which delivers token-by-token
        output in real time.  Falls back to ``text_generation(stream=True)``
        for base models.

        Args:
            prompt: User prompt.
            config: Optional generation config.
            **kwargs: Forwarded to the underlying API call.

        Yields:
            str: Token/chunk strings as they arrive.

        Raises:
            RuntimeError: If not loaded.
            ModelAuthError: On authentication failure.
            ModelUnavailableError: When the model is unavailable on Serverless.
        """
        if not self._is_loaded or self._client is None:
            raise not_loaded_error("hf_inference", self.model_name, "generate_stream")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="hf_inference",
            model_name=self.model_name,
            supports_vision=self._info.get("supports_vision", False),
            hint="Use an HF Inference model whose input_modalities include image.",
        )
        require_audio_support(
            prompt,
            provider="hf_inference",
            model_name=self.model_name,
            supports_audio=True,
            hint="Audio is transcribed via HF ASR (Whisper) before chat completion.",
        )

        system_prompt = kwargs.pop("system_prompt", "You are a helpful assistant.")
        messages_arg = kwargs.pop("messages", None)

        use_text_gen = self._info.get("use_text_generation", False)
        self._last_stream_api_usage = None

        if use_text_gen:
            prompt_text = prompt
        else:
            messages = self._build_messages(prompt, system_prompt, messages_arg)
            prompt_text = _messages_text(messages)

        # Only the streamed text's length is needed for a token estimate, so
        # the chunks are measured as they pass rather than accumulated.
        output_chars = 0
        with timed_call("hf", self.model_name) as _stream_timer:
            _first_token = True
            src = self._stream_text(prompt, config, **kwargs) if use_text_gen else self._stream_chat(
                messages, config, **kwargs
            )
            for token in src:
                if _first_token:
                    _stream_timer.mark_first_token()
                    _first_token = False
                output_chars += len(token)
                yield token

        self._account_stream_usage(len(prompt_text), output_chars)

    def _account_stream_usage(self, prompt_chars: int, output_chars: int) -> None:
        """Price and record the streamed call that just finished.

        The chat endpoint sends a usage block on its final chunk for some
        models; when it does, those counts are used. Otherwise the token counts
        come from the same character-based estimate the non-streaming path
        applies, over the character counts of the prompt and of the streamed
        output. The cost lands on the global ledger and on this adapter's
        cumulative totals, so a streamed call reports the same way a
        ``generate()`` call does.

        Args:
            prompt_chars: Characters sent, across every message in the prompt.
            output_chars: Characters yielded by the stream.
        """
        try:
            usage = self._last_stream_api_usage
            self._last_stream_api_usage = None
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            if not (input_tokens or output_tokens):
                input_tokens = self._estimate_tokens_from_chars(prompt_chars)
                output_tokens = self._estimate_tokens_from_chars(output_chars)
            if not (input_tokens or output_tokens):
                return
            cost_usd = self._price_tokens(input_tokens, output_tokens)
            self._record_cost(input_tokens, output_tokens, cost_usd)
            accumulate_stream_cost(
                self,
                cost_usd,
                input_tokens + output_tokens,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
        except BudgetExceededError:
            raise
        except Exception:  # noqa: BLE001 - accounting must not break a delivered stream
            logger.debug("Stream usage accounting failed for HF Inference", exc_info=True)

    def _stream_chat(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig,
        **kwargs: Any,
    ) -> Iterator[str]:
        call_kwargs: dict[str, Any] = {}
        if config.max_tokens is not None:
            call_kwargs["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            call_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            call_kwargs["top_p"] = config.top_p
        if config.stop_sequences:
            call_kwargs["stop"] = config.stop_sequences
        if config.seed is not None:
            call_kwargs["seed"] = config.seed
        call_kwargs.update(kwargs)

        reasoning_buf: list[str] = []
        yielded_text = False
        self._last_stream_finish_reason: str | None = None

        try:
            for chunk in self._client.chat_completion(
                messages=messages,
                model=None if self._endpoint_url else self.model_name,
                stream=True,
                **call_kwargs,
            ):
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    self._last_stream_api_usage = usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    if getattr(choice, "finish_reason", None):
                        self._last_stream_finish_reason = choice.finish_reason
                    reasoning_buf.append(
                        reasoning_delta_text(getattr(choice, "delta", None))
                    )
                if chunk.choices and chunk.choices[0].delta.content:
                    yielded_text = True
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            self._raise_for_unavailable(exc, context="streaming")

        warn_reasoning_only_stream(
            model_name=self.model_name,
            yielded_text=yielded_text,
            reasoning_text="".join(reasoning_buf),
            reasoning_tokens=extract_reasoning_tokens(self._last_stream_api_usage),
            finish_reason=self._last_stream_finish_reason,
            max_tokens=call_kwargs.get("max_tokens"),
            logger=logger,
        )

    def _stream_text(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs: Any,
    ) -> Iterator[str]:
        call_kwargs: dict[str, Any] = {}
        if config.max_tokens is not None:
            call_kwargs["max_new_tokens"] = config.max_tokens
        if config.temperature is not None:
            call_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            call_kwargs["top_p"] = config.top_p
        call_kwargs.update(kwargs)

        try:
            for token in self._client.text_generation(
                prompt,
                model=None if self._endpoint_url else self.model_name,
                stream=True,
                **call_kwargs,
            ):
                yield token if isinstance(token, str) else str(token)
        except Exception as exc:
            self._raise_for_unavailable(exc, context="streaming text_generation")

    # ------------------------------------------------------------------
    # Async generate
    # ------------------------------------------------------------------

    async def async_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Async version of generate() — runs blocking call in thread pool.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Returns:
            The generated text with its usage metadata.
        """
        import asyncio
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self.generate, prompt, config, **kwargs),
        )

    # ------------------------------------------------------------------
    # Tool-call generate (native tools on supported models)
    # ------------------------------------------------------------------

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        config: GenerationConfig | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with native tool-calling on supported models.

        For models with ``supports_native_tools=True`` in the registry, the
        tools list is forwarded to ``chat_completion(tools=...)``.

        For unsupported models, raises ``NotImplementedError`` — use an Agent
        with ``strategy='react'`` instead.

        Args:
            prompt: User prompt / task description.
            tools: List of tool dicts (OpenAI function-calling schema) or
                effGen Tool objects.
            config: Optional generation config.
            messages: Full conversation history; overrides prompt if provided.
            system_prompt: System prompt when building messages from *prompt*.
            **kwargs: Forwarded to the underlying API call.

        Returns:
            GenerationResult with ``tool_calls`` in metadata.
        """
        if not self._info.get("supports_native_tools"):
            raise NotImplementedError(
                f"Model '{self.model_name}' does not support native tool calling.  "
                f"Use an Agent with strategy='react' instead, or switch to "
                f"a tool-capable model like 'Qwen/Qwen2.5-7B-Instruct'."
            )

        openai_tools: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                openai_tools.append(t if "type" in t else {"type": "function", "function": t})
            else:
                try:
                    schema = t.metadata.to_json_schema()
                    openai_tools.append({"type": "function", "function": schema})
                except AttributeError:
                    openai_tools.append({"type": "function", "function": str(t)})

        return self.generate(
            prompt,
            config,
            tools=openai_tools,
            messages=messages,
            system_prompt=system_prompt,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def supports_tool_calling(self) -> bool:
        """True when the catalog marks this model as supporting native tools."""
        return bool(self._info.get("supports_native_tools", False))

    def supports_forced_tool_call(self) -> bool:
        """True when tools are offered: ``tool_choice`` is honoured here.

        The HF router's OpenAI-compatible chat endpoint enforces the choice, so
        a turn can be sent that requires a call. A model that is not offered
        tool definitions cannot be required to call one, so this follows
        :meth:`supports_tool_calling`.
        """
        return self.supports_tool_calling()

    def supports_streaming(self) -> bool:
        """True: HF Inference chat endpoints stream."""
        return True

    def get_endpoint_url(self) -> str | None:
        """Return the dedicated-endpoint URL, or ``None`` for serverless."""
        return self._endpoint_url

    # ------------------------------------------------------------------
    # Transient error helper
    # ------------------------------------------------------------------

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """Return True if the exception looks like a transient network/rate error."""
        exc_str = str(exc)
        return any(code in exc_str for code in ("429", "500", "502", "503", "504"))


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.hf_inference_models import HF_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "hf",
            HFInferenceAdapter,
            HF_MODELS,
            env_keys=["HF_TOKEN", "HUGGINGFACE_API_KEY"],
            capabilities={Capability.chat, Capability.streaming, Capability.audio_input},
            # HuggingFace Serverless Inference: many small/medium models are free via PRO
            # or community tier. Larger models may require dedicated endpoints (paid).
            # Free tier available for many open-source models via HF Inference API.
            # Pricing verified: https://huggingface.co/docs/api-inference/pricing (2026-05-11)
            pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
