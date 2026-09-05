"""
OpenAI API adapter for GPT and o-series reasoning models.

Supports:
- GPT-4o, GPT-4.1, GPT-5, GPT-5.4-nano/mini chat models
- o1, o1-mini, o3, o3-mini, o4-mini reasoning models
- reasoning_effort / max_reasoning_tokens wired through GenerationConfig
- Function / tool calling
- Streaming responses
- Automatic retries with exponential backoff
- Cost tracking via CostTracker
- OpenAI automatic prompt caching (cached_input_tokens surfaced)
- Structured outputs v2 (strict JSON schema + ModelRefusalError)
- A ``base_url`` for any server speaking the OpenAI protocol
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from effgen.models._adapter_utils import (
    FINISH_CONTENT_FILTER,
    FINISH_LENGTH,
    annotate_reasoning_only,
    estimate_tokens,
    extract_reasoning_text,
    extract_reasoning_tokens,
    model_not_found_error,
    normalize_finish_reason,
    not_loaded_error,
    provider_runtime_error,
    reasoning_delta_text,
    warn_reasoning_only_stream,
)
from effgen.models._base_url import (
    describe_endpoint,
    openai_client_base_url,
    resolve_base_url,
)
from effgen.models._multimodal import (
    require_audio_support,
    require_video_support,
    require_vision_support,
)
from effgen.models._usage import (
    accumulate_stream_tool_call_deltas,
    cost_label,
    extract_openai_usage,
    record_tracker_cost,
    stream_tool_call_entries,
    stringify_tool_arguments,
    tool_calls_from_message,
    usage_metadata,
)
from effgen.models.base import (
    FunctionCallingModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
    clear_stream_tool_calls,
    fold_call_totals,
    record_stream_tool_calls,
    record_stream_usage,
)
from effgen.models.errors import (
    ModelAuthError,
    ModelRefusalError,
    error_has_status,
)
from effgen.models.latency_tracker import timed_call
from effgen.models.openai_models import (
    OPENAI_MODELS,
    VALID_REASONING_EFFORTS,
    get_context_length,
    get_max_output,
    get_pricing,
    supports_reasoning,
    supports_vision,
)
from effgen.observability import get_logger as _get_obs_logger
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr

# Always show token/cost breakdown at INFO level
_USAGE_LOG = logging.getLogger(__name__ + ".usage")

logger = logging.getLogger(__name__)
_obs_log = _get_obs_logger(__name__)

_REASONING_UNSUPPORTED_PARAMS = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
_FIXED_SAMPLING_PREFIXES = ("gpt-5",)


def _pick_default_max_output(model_id: str) -> int:
    """Return a sensible default max_output for *model_id*.

    Reasoning models need more room for their internal chain-of-thought.
    """
    return get_max_output(model_id)


def _supports_sampling_params(model_id: str, is_reasoning_model: bool) -> bool:
    """Return whether the model accepts non-default sampling parameters."""
    return not is_reasoning_model and not model_id.startswith(_FIXED_SAMPLING_PREFIXES)


# Per-call generation kwargs that map onto a GenerationConfig field. Folding
# these into the config (rather than forwarding them raw to the API) lets the
# model-aware request builder gate them per family — e.g. emitting
# max_completion_tokens instead of the deprecated max_tokens and dropping
# temperature/top_p/stop for gpt-5 and the reasoning models that reject them.
_GENERATION_KWARG_TO_FIELD = {
    "max_tokens": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "repetition_penalty": "repetition_penalty",
    "seed": "seed",
    "stop": "stop_sequences",
    "stop_sequences": "stop_sequences",
    "reasoning_effort": "reasoning_effort",
    "max_reasoning_tokens": "max_reasoning_tokens",
}


def _fold_generation_kwargs(
    config: GenerationConfig, kwargs: dict[str, Any]
) -> GenerationConfig:
    """Move recognized generation kwargs out of *kwargs* into a copy of *config*.

    A per-call ``generate(..., temperature=0)`` (or ``max_tokens=``/``top_p=``/
    ``stop=``) must travel the same model-aware path as ``GenerationConfig`` so the
    request builder can gate it per model family. This pops every recognized
    generation kwarg from *kwargs* (mutated in place) and returns a config carrying
    those overrides; genuinely-unknown kwargs are left in *kwargs* to forward raw.
    """
    overrides: dict[str, Any] = {}
    for kwarg, field in _GENERATION_KWARG_TO_FIELD.items():
        if kwarg not in kwargs:
            continue
        value = kwargs.pop(kwarg)
        # The OpenAI API accepts a bare string for `stop`; GenerationConfig
        # carries a list, so normalize for consistent downstream handling.
        if field == "stop_sequences" and isinstance(value, str):
            value = [value]
        overrides[field] = value
    if not overrides:
        return config
    return replace(config, **overrides)


class OpenAIAdapter(FunctionCallingModel):
    """
    Adapter for OpenAI API models (chat + reasoning families).

    Attributes:
        model_name: OpenAI model identifier
        api_key: OpenAI API key (reads from OPENAI_API_KEY env if not supplied)
        base_url: Base URL of the endpoint to call. Defaults to OpenAI's own
            API; set it to talk to any server speaking the OpenAI protocol
            (vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, a gateway or a
            corporate proxy). Falls back to ``EFFGEN_BASE_URL``,
            ``OPENAI_BASE_URL`` then ``OPENAI_API_BASE``. For a self-hosted
            server prefer :class:`~effgen.models.openai_compatible_adapter.OpenAICompatibleAdapter`,
            which additionally drops the OpenAI catalog defaults and pricing.
        organization_id: OpenAI organization ID (optional)
        max_retries: Maximum retry attempts for failed requests
        timeout: Request timeout in seconds
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "openai"

    #: Whether the bundled OpenAI catalog describes this endpoint's model ids.
    #: False for a self-hosted server, whose ids are its own (see
    #: :class:`~effgen.models.openai_compatible_adapter.OpenAICompatibleAdapter`).
    _catalog_backed = True

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        organization_id: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        base_url: str | None = None,
        context_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url or resolve_base_url()
        if self._catalog_backed and model_name not in OPENAI_MODELS and not self.base_url:
            # Informational fallback, not an actionable warning — a valid new
            # or hot-swapped model id just isn't in the bundled catalog yet.
            # Surfaced at INFO so it shows with --verbose without making a
            # normal, successful run/chat turn look broken by default.
            # A custom base_url serves its own ids, which this catalog never
            # lists, so the message would be noise there.
            logger.info(
                f"Model '{model_name}' is not in the OpenAI registry. "
                f"Using conservative defaults (context=128k, pricing fallback). "
                f"Call OpenAIAdapter.list_models() for registered ids."
            )
        context = context_length if context_length is not None else get_context_length(model_name)
        super().__init__(
            model_name=model_name,
            model_type=ModelType.OPENAI,
            context_length=context,
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.organization_id = organization_id or os.getenv("OPENAI_ORG_ID")
        self.max_retries = max_retries
        self.timeout = timeout
        self.additional_kwargs = kwargs

        self.client = None
        self.total_cost = 0.0
        self.total_tokens = 0

        self._is_reasoning_model = supports_reasoning(model_name)
        self._supports_sampling_params = _supports_sampling_params(
            model_name,
            self._is_reasoning_model,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI package is not installed. Install it with: pip install openai"
            ) from e

        endpoint = describe_endpoint(self.base_url)
        try:
            logger.info(
                f"Initializing OpenAI client for model '{self.model_name}' "
                f"against {endpoint}..."
            )
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            # Always explicit: left unset, the SDK re-reads OPENAI_BASE_URL
            # itself and treats a blank one as an address rather than as no
            # override, which sends the call to ''.
            client_kwargs["base_url"] = openai_client_base_url(self.base_url)
            if self.organization_id:
                client_kwargs["organization"] = self.organization_id
            client_kwargs.update(self.additional_kwargs)

            self.client = OpenAI(**client_kwargs)

            # Light connectivity check — swallow failures, model may not be
            # listed via models.retrieve for all accounts/tiers.
            try:
                self.client.models.retrieve(self.model_name)
            except Exception as e:
                logger.debug(f"Model verify skipped: {e}")

            self._is_loaded = True
            self._metadata = {
                "model_name": self.model_name,
                "context_length": self.get_context_length(),
                "family": OPENAI_MODELS.get(self.model_name, {}).get("family", "chat"),
                "supports_reasoning": self._is_reasoning_model,
                "supports_sampling_params": self._supports_sampling_params,
                "supports_functions": True,
                "supports_streaming": True,
                "base_url": self.base_url,
            }
            logger.info(
                f"OpenAI client initialized for '{self.model_name}' against {endpoint}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client for {endpoint}: {e}")
            raise RuntimeError(
                f"OpenAI initialization failed for {endpoint}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_messages(self, prompt: str | list) -> list[dict[str, Any]]:
        """Convert *prompt* to OpenAI messages list."""
        # Handle effGen Message objects
        try:
            from effgen.core.messages import Message

            if isinstance(prompt, Message):
                return [self._message_to_openai(prompt)]

            if isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                return [self._message_to_openai(m) for m in prompt]
        except ImportError:
            pass

        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        content_parts: list[dict[str, Any]] = []
        for item in prompt:
            if isinstance(item, str):
                content_parts.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if "type" in item:
                    content_parts.append(item)
                elif "image_url" in item:
                    content_parts.append({"type": "image_url", "image_url": {"url": item["image_url"]}})
                else:
                    content_parts.append({"type": "text", "text": str(item)})
            else:
                content_parts.append({"type": "text", "text": str(item)})
        return [{"role": "user", "content": content_parts}]

    def _message_to_openai(self, message: Any) -> dict[str, Any]:
        """Convert an effGen Message to an OpenAI message dict.

        AudioPart is handled by transcribing via Whisper first, then
        injecting the transcript as a TextPart.  This keeps the chat
        completions path clean for all GPT models.
        """
        import base64

        from effgen.core.messages import (
            AudioPart,
            ImagePart,
            TextPart,
            ToolCallPart,
            ToolResultPart,
            VideoPart,
        )
        from effgen.multimodal.image_pre import prepare as _preprocess_image

        role = message.role.value
        if role == "tool":
            role = "tool"

        content_parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                processed = _preprocess_image(part, "openai", self.model_name)
                b64 = base64.b64encode(processed.image).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{processed.mime};base64,{b64}"},
                })
            elif isinstance(part, AudioPart):
                # Transcribe via Whisper and inject transcript as text
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
                # Send as a series of image frames
                for frame in part.frames:
                    b64 = base64.b64encode(frame).decode()
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.mime};base64,{b64}"},
                    })
            elif isinstance(part, ToolCallPart):
                pass  # tool calls go in a different field
            elif isinstance(part, ToolResultPart):
                pass

        # If content has a single text part, simplify to string
        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            return {"role": role, "content": content_parts[0]["text"]}

        return {"role": role, "content": content_parts}

    def _transcribe_audio_part(self, part: Any) -> str:
        """Transcribe a single AudioPart via the Whisper API.

        Uses ``/audio/transcriptions`` (Whisper-1).  For audio that exceeds
        25 MB the audio_pre chunker splits it and results are concatenated.
        """
        from effgen.multimodal.audio_pre import chunk as _chunk_audio

        chunks = _chunk_audio(part, "openai", self.model_name)
        transcripts: list[str] = []
        for audio_chunk in chunks:
            text = self._call_whisper(audio_chunk)
            if text:
                transcripts.append(text)
        return " ".join(transcripts).strip()

    def _call_whisper(self, part: Any) -> str:
        """Call the Whisper transcription endpoint for a single audio chunk."""
        import io

        from effgen.multimodal.audio_pre import _MIME_TO_FORMAT

        mime = part.mime
        fmt = _MIME_TO_FORMAT.get(mime, "mp3")
        filename = f"audio.{fmt}"

        try:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, io.BytesIO(part.audio), mime),
            )
            return response.text or ""
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            raise provider_runtime_error("openai", "whisper-1", "transcribe", exc, message="OpenAI audio transcription failed", endpoint=self.base_url) from exc

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime: str = "audio/mp3",
        model: str = "whisper-1",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe *audio_bytes* using the OpenAI Whisper API.

        This is the primary public method for standalone audio transcription.
        For audio embedded in a Message use :meth:`generate` — AudioParts
        are automatically transcribed before the chat completion call.

        Args:
            audio_bytes: Raw audio bytes.
            mime: MIME type of the audio (e.g. ``"audio/mp3"``).
            model: Whisper model ID (default ``"whisper-1"``).
            language: Optional BCP-47 language code (e.g. ``"en"``).
            prompt: Optional hint to guide transcription style.

        Returns:
            Transcribed text string.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "transcribe_audio")

        import io

        from effgen.core.messages import AudioPart
        from effgen.multimodal.audio_pre import _MIME_TO_FORMAT
        from effgen.multimodal.audio_pre import chunk as _chunk_audio

        part = AudioPart(audio=audio_bytes, mime=mime)
        chunks = _chunk_audio(part, "openai", self.model_name)
        transcripts: list[str] = []
        for audio_chunk in chunks:
            fmt = _MIME_TO_FORMAT.get(audio_chunk.mime, "mp3")
            filename = f"audio.{fmt}"
            extra: dict[str, Any] = {}
            if language:
                extra["language"] = language
            if prompt:
                extra["prompt"] = prompt
            try:
                resp = self.client.audio.transcriptions.create(
                    model=model,
                    file=(filename, io.BytesIO(audio_chunk.audio), audio_chunk.mime),
                    **extra,
                )
                transcripts.append(resp.text or "")
            except Exception as exc:
                logger.error("Whisper transcription failed: %s", exc)
                raise provider_runtime_error("openai", model, "transcribe", exc, message="OpenAI audio transcription failed", endpoint=self.base_url) from exc

        return " ".join(transcripts).strip()

    def _validate_media_support(self, prompt: Any) -> None:
        """Reject image/audio/video inputs the current model cannot handle.

        All GPT models support audio via Whisper transcription; audio capability
        is always True for OpenAI since we transparently transcribe before
        sending. Video is handled via frame-sampling + vision, so it requires a
        vision-capable model.
        """
        require_vision_support(
            prompt,
            provider="openai",
            model_name=self.model_name,
            supports_vision=supports_vision,
            hint="Use 'gpt-4o-mini' or 'gpt-4o' for image inputs.",
        )
        require_audio_support(
            prompt,
            provider="openai",
            model_name=self.model_name,
            supports_audio=True,
            hint="Audio is transcribed via Whisper before chat completion.",
        )
        require_video_support(
            prompt,
            provider="openai",
            model_name=self.model_name,
            supports_video=supports_vision(self.model_name),
            hint="Use 'gpt-4o-mini' or 'gpt-4o' for video inputs (frames sent as images).",
        )

    def _validate_reasoning_effort(self, effort: str | None) -> None:
        """Raise ValueError for invalid *effort* values."""
        if effort is not None and effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning_effort={effort!r}. "
                f"Valid values: {VALID_REASONING_EFFORTS}. "
                f"Pass None to omit."
            )

    @staticmethod
    def _requested_output_cap(request_params: dict[str, Any]) -> int | None:
        """Return the output-token cap a request asked for, whatever it is named.

        Reasoning families take ``max_completion_tokens``; the chat families
        take ``max_tokens``; the Responses API takes ``max_output_tokens``.
        """
        for key in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
            value = request_params.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _build_request_params(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``client.chat.completions.create``."""
        self._validate_reasoning_effort(config.reasoning_effort)

        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        # All current OpenAI models accept max_completion_tokens.
        # max_tokens is deprecated as of the 2024-11 API version.
        max_tokens = config.max_tokens or _pick_default_max_output(self.model_name)
        params["max_completion_tokens"] = max_tokens

        if self._is_reasoning_model:
            # Reasoning models ignore temperature / top_p / penalties — drop them.
            # reasoning_effort is passed as a top-level API parameter.
            if config.reasoning_effort is not None:
                params["reasoning_effort"] = config.reasoning_effort
            if config.max_reasoning_tokens is not None:
                # max_reasoning_tokens narrows how many tokens the model can use
                # for its internal chain-of-thought.
                params["max_completion_tokens"] = config.max_reasoning_tokens
        elif self._supports_sampling_params:
            # Chat model — include standard sampling parameters.
            params["temperature"] = config.temperature
            params["top_p"] = config.top_p
            params["presence_penalty"] = config.presence_penalty
            params["frequency_penalty"] = config.frequency_penalty

            if config.reasoning_effort is not None:
                logger.debug(
                    f"reasoning_effort={config.reasoning_effort!r} is set but "
                    f"'{self.model_name}' is not a reasoning model — dropping silently."
                )
        elif config.reasoning_effort is not None:
            logger.debug(
                f"reasoning_effort={config.reasoning_effort!r} is set but "
                f"'{self.model_name}' is not a reasoning model — dropping silently."
            )

        # GPT-5 family and reasoning models don't accept the 'stop' parameter.
        # Drop it silently so the Agent's default stop_sequences don't break calls.
        if config.stop_sequences and not self._is_reasoning_model and not self.model_name.startswith("gpt-5"):
            params["stop"] = config.stop_sequences
        if config.seed is not None:
            params["seed"] = config.seed
        if stream:
            params["stream"] = True

        return params

    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost in USD from token counts, crediting cached tokens.

        Returns ``None`` when the catalog publishes no rate for this model, so
        the call reports no price rather than a fabricated ``$0`` or a
        placeholder rate. A server the caller runs itself has no published rate
        at all, so its calls always report no price.
        """
        from effgen.models._cost import pricing_status

        if not self._catalog_backed:
            return None
        if pricing_status("openai", self.model_name) == "unpriced":
            return None
        input_price, cached_price, output_price = get_pricing(self.model_name)
        if input_price is None:
            logger.debug(f"No pricing for '{self.model_name}', defaulting to $2/$8 per 1M.")
            input_price, cached_price, output_price = 2.00, 0.50, 8.00

        non_cached = max(0, prompt_tokens - cached_tokens)
        input_cost = (non_cached / 1_000_000) * input_price
        if cached_tokens > 0 and cached_price is not None:
            input_cost += (cached_tokens / 1_000_000) * cached_price
        output_cost = (completion_tokens / 1_000_000) * (output_price or 8.00)
        return input_cost + output_cost

    def _record_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cached_tokens: int = 0,
    ) -> float | None:
        """Price this call and fold it into the adapter's running session total.

        Returns this call's cost — the value every ``GenerationResult.metadata``
        reports as ``cost_usd`` — or ``None`` when the model publishes no rate.
        An unpriced call leaves the session total where it was rather than
        adding a zero. ``self.total_cost`` (surfaced as
        ``metadata["total_cost"]``) is a different number: the cumulative cost
        across every call made on this adapter instance so far, not an alias.
        """
        cost = self._calculate_cost(prompt_tokens, completion_tokens, cached_tokens)
        fold_call_totals(self, cost, total_tokens)

        # Always print token/cost breakdown so users can see what they're spending
        _USAGE_LOG.info(
            f"[{self.model_name}] "
            f"input={prompt_tokens}tok "
            f"(cached={cached_tokens}) "
            f"output={completion_tokens}tok "
            f"| call={cost_label(cost)} session=${self.total_cost:.6f}"
        )

        record_tracker_cost(
            "openai",
            self.model_name,
            prompt_tokens,
            completion_tokens,
            log=logger,
        )
        return cost

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a completion for *prompt*.

        If ``tools`` is in *kwargs*, routes automatically to ``generate_with_tools``
        so the Agent loop can use native OpenAI function-calling without calling a
        separate method.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK, including
                ``tools`` to use native function calling.

        Returns:
            The generated text with its usage metadata.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "generate")
        if isinstance(prompt, str):
            self.validate_prompt(prompt)
        if config is None:
            config = GenerationConfig()

        self._validate_media_support(prompt)

        # Transparent routing: if tools are passed (e.g. from the Agent), use
        # generate_with_tools so native function-calling works end-to-end.
        if "tools" in kwargs:
            return self.generate_with_tools(
                prompt=prompt,
                tools=kwargs.pop("tools"),
                config=config,
                **kwargs,
            )

        config = _fold_generation_kwargs(config, kwargs)
        messages = self._create_messages(prompt)
        request_params = self._build_request_params(messages, config)
        request_params.update(kwargs)

        try:
            with timed_call("openai", self.model_name):
                response = self.client.chat.completions.create(**request_params)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            msg = str(e)
            if error_has_status(e, 401) or "invalid_api_key" in msg.lower() or "incorrect api key" in msg.lower():
                raise ModelAuthError("openai", self.model_name, msg) from e
            if error_has_status(e, 404) or "model_not_found" in msg.lower():
                raise model_not_found_error("openai", self.model_name, msg) from e
            raise provider_runtime_error("openai", self.model_name, "generate", e, message="OpenAI generation failed", endpoint=self.base_url) from e

        choice = response.choices[0]
        generated_text = choice.message.content or ""
        finish_reason = normalize_finish_reason(choice.finish_reason)

        prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
            extract_openai_usage(response.usage)
        )
        cost = self._record_cost(prompt_tokens, completion_tokens, total_tokens, cached_tokens)

        metadata: dict[str, Any] = usage_metadata(
            prompt_tokens, completion_tokens, total_tokens, cached_tokens,
            cost, self.total_cost,
        )
        annotate_reasoning_only(
            metadata,
            text=generated_text,
            reasoning_text=extract_reasoning_text(choice.message),
            reasoning_tokens=extract_reasoning_tokens(response.usage),
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=self._requested_output_cap(request_params),
            completion_tokens=completion_tokens,
            tool_calls=None,
            logger=logger,
        )

        _obs_log.model_event(
            "call.done",
            provider="openai",
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
        )
        # Emit span attributes on the current active span (set by agent loop or adapter tests)
        _set_span_attr(ModelAttrs.PROVIDER, "openai")
        _set_span_attr(ModelAttrs.NAME, self.model_name)
        _set_span_attr(ModelAttrs.INPUT_TOKENS, prompt_tokens)
        _set_span_attr(ModelAttrs.OUTPUT_TOKENS, completion_tokens)
        if cached_tokens:
            _set_span_attr(ModelAttrs.CACHED_TOKENS, cached_tokens)
        if cost is not None:
            _set_span_attr(ModelAttrs.COST_USD, float(cost))
        _set_span_attr(ModelAttrs.OUTCOME, "ok")

        return GenerationResult(
            text=generated_text,
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
        """Stream completions for *prompt*, yielding text chunks.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Returns:
            An iterator over the response's text chunks.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "generate_stream")
        if isinstance(prompt, str):
            self.validate_prompt(prompt)
        if config is None:
            config = GenerationConfig()

        self._validate_media_support(prompt)

        config = _fold_generation_kwargs(config, kwargs)
        messages = self._create_messages(prompt)
        request_params = self._build_request_params(messages, config, stream=True)
        # Ask for a final usage chunk so streamed turns are costed/counted just
        # like non-streamed ones (the chunk carries empty choices + a usage block).
        request_params.setdefault("stream_options", {"include_usage": True})
        request_params.update(kwargs)

        _usage = None
        _finish_reason = None
        _reasoning_buf: list[str] = []
        _yielded_text = False
        clear_stream_tool_calls(self)
        _tool_calls_buf: dict[int, dict[str, Any]] = {}
        try:
            with timed_call("openai", self.model_name) as _stream_timer:
                stream = self.client.chat.completions.create(**request_params)
                _first_token = True
                for chunk in stream:
                    if getattr(chunk, "usage", None) is not None:
                        _usage = chunk.usage
                    if chunk.choices:
                        _choice = chunk.choices[0]
                        if getattr(_choice, "finish_reason", None):
                            _finish_reason = _choice.finish_reason
                        _reasoning_buf.append(
                            reasoning_delta_text(getattr(_choice, "delta", None))
                        )
                        _delta = getattr(_choice, "delta", None)
                        if _delta is not None and getattr(_delta, "tool_calls", None):
                            # A tool turn streams its call and no content at all,
                            # so without this the call is lost while its tokens
                            # are still billed. Recorded as it accumulates, so a
                            # consumer knows the turn is a call before it commits
                            # any text as the answer.
                            accumulate_stream_tool_call_deltas(
                                _tool_calls_buf, _delta.tool_calls
                            )
                            record_stream_tool_calls(
                                self, stream_tool_call_entries(_tool_calls_buf)
                            )
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        if _first_token:
                            _stream_timer.mark_first_token()
                            _first_token = False
                        # An empty content delta is not a visible token: a
                        # reasoning turn that never answers can still emit one.
                        _yielded_text = _yielded_text or bool(
                            chunk.choices[0].delta.content
                        )
                        yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise provider_runtime_error("openai", self.model_name, "stream", e, message="OpenAI streaming failed", endpoint=self.base_url) from e

        self._last_stream_finish_reason = _finish_reason
        _streamed_calls = stream_tool_call_entries(_tool_calls_buf)
        record_stream_tool_calls(self, _streamed_calls)

        warn_reasoning_only_stream(
            model_name=self.model_name,
            yielded_text=_yielded_text,
            reasoning_text="".join(_reasoning_buf),
            reasoning_tokens=extract_reasoning_tokens(_usage),
            finish_reason=_finish_reason,
            max_tokens=self._requested_output_cap(request_params),
            tool_calls=_streamed_calls,
            logger=logger,
        )

        # Record real usage from the final chunk so cost/token tracking and the
        # CLI's per-turn footer reflect streamed turns too.
        if _usage is not None:
            try:
                prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                    extract_openai_usage(_usage)
                )
                cost = self._record_cost(
                    prompt_tokens, completion_tokens, total_tokens, cached_tokens
                )
                record_stream_usage(self, prompt_tokens, completion_tokens, cost)
            except Exception:  # noqa: BLE001 - usage accounting must not break streaming
                logger.debug("OpenAI stream usage recording failed", exc_info=True)

    def generate_structured(
        self,
        prompt: str,
        response_format: dict[str, Any],
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response constrained to a JSON Schema (structured outputs v2).

        Args:
            prompt: User prompt.
            response_format: OpenAI ``response_format`` dict.  Pass the output of
                ``to_openai_schema`` wrapped in the expected envelope, e.g.::

                    from effgen.models.openai_schema import to_openai_schema
                    rf = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Answer",
                            "schema": to_openai_schema(Answer),
                            "strict": True,
                        },
                    }

            system_prompt: Optional system message prepended to the conversation.
                When supplied the system prompt is placed first in the message list
                so OpenAI can cache it automatically (prefix caching).
            config: Generation configuration.
            **kwargs: Extra params forwarded to the API.

        Returns:
            GenerationResult where ``text`` contains the raw JSON string.

        Raises:
            ModelRefusalError: If the model returns a ``refusal`` instead of content.
            RuntimeError: For network / API errors.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "generate_structured")
        self.validate_prompt(prompt)
        if config is None:
            config = GenerationConfig()

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._create_messages(prompt))

        config = _fold_generation_kwargs(config, kwargs)
        request_params = self._build_request_params(messages, config)
        request_params["response_format"] = response_format
        request_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception as e:
            logger.error(f"OpenAI structured call failed: {e}")
            raise provider_runtime_error("openai", self.model_name, "structured", e, message="OpenAI structured generation failed", endpoint=self.base_url) from e

        choice = response.choices[0]
        message = choice.message

        # Check for model refusal (structured outputs may return refusal instead of content)
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ModelRefusalError(refusal_message=refusal, model_name=self.model_name)

        generated_text = message.content or ""
        finish_reason = normalize_finish_reason(choice.finish_reason)

        prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
            extract_openai_usage(response.usage)
        )
        cost = self._record_cost(prompt_tokens, completion_tokens, total_tokens, cached_tokens)

        metadata = usage_metadata(
            prompt_tokens, completion_tokens, total_tokens, cached_tokens,
            cost, self.total_cost,
        )
        metadata["tool_calls"] = tool_calls_from_message(message)
        annotate_reasoning_only(
            metadata,
            text=generated_text,
            reasoning_text=extract_reasoning_text(message),
            reasoning_tokens=extract_reasoning_tokens(response.usage),
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=self._requested_output_cap(request_params),
            completion_tokens=completion_tokens,
            tool_calls=None,
            logger=logger,
        )

        return GenerationResult(
            text=generated_text,
            tokens_used=completion_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata=metadata,
        )

    def generate_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with an explicit system prompt prepended first (for stable caching).

        Placing a long, stable system prompt at position 0 in the message list
        lets OpenAI cache the prefix automatically.  This is the recommended
        pattern for agents that reuse the same instructions across many turns.

        Args:
            prompt: User message.
            system_prompt: System instructions, placed first so caching is reliable.
            config: Generation configuration.
            **kwargs: Extra params forwarded to the API.

        Returns:
            GenerationResult with ``metadata["cached_input_tokens"]`` populated.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "generate_with_system_prompt")
        self.validate_prompt(prompt)
        if config is None:
            config = GenerationConfig()

        config = _fold_generation_kwargs(config, kwargs)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        request_params = self._build_request_params(messages, config)
        request_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception as e:
            logger.error(f"OpenAI API call with system prompt failed: {e}")
            raise provider_runtime_error("openai", self.model_name, "generate", e, message="OpenAI generation with system prompt failed", endpoint=self.base_url) from e

        choice = response.choices[0]
        generated_text = choice.message.content or ""
        finish_reason = normalize_finish_reason(choice.finish_reason)

        prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
            extract_openai_usage(response.usage)
        )
        cost = self._record_cost(prompt_tokens, completion_tokens, total_tokens, cached_tokens)

        metadata = usage_metadata(
            prompt_tokens, completion_tokens, total_tokens, cached_tokens,
            cost, self.total_cost,
        )
        metadata["tool_calls"] = tool_calls_from_message(choice.message)
        annotate_reasoning_only(
            metadata,
            text=generated_text,
            reasoning_text=extract_reasoning_text(choice.message),
            reasoning_tokens=extract_reasoning_tokens(response.usage),
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=self._requested_output_cap(request_params),
            completion_tokens=completion_tokens,
            tool_calls=None,
            logger=logger,
        )

        return GenerationResult(
            text=generated_text,
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
        """Generate with OpenAI-native function / tool calling.

        Args:
            prompt: User prompt (appended as user message if *messages* is given).
            tools: List of tool definitions in OpenAI format.
            config: Generation configuration.
            messages: Full conversation history. If provided, *prompt* is appended.
            **kwargs: Extra params forwarded to the API.

        Returns:
            The generated text, with any tool calls in its metadata.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "generate_with_tools")
        if isinstance(prompt, str):
            self.validate_prompt(prompt)
        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="openai",
            model_name=self.model_name,
            supports_vision=supports_vision,
            hint="Use 'gpt-4o-mini' or 'gpt-4o' for image inputs.",
        )

        if messages is None:
            messages = self._create_messages(prompt)
        else:
            if isinstance(prompt, str):
                messages = list(messages) + [{"role": "user", "content": prompt}]
            else:
                messages = list(messages) + self._create_messages(prompt)

        config = _fold_generation_kwargs(config, kwargs)
        request_params = self._build_request_params(messages, config)
        request_params["tools"] = tools
        request_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception as e:
            logger.error(f"OpenAI API call with tools failed: {e}")
            raise provider_runtime_error("openai", self.model_name, "generate_with_tools", e, message="OpenAI generation with tools failed", endpoint=self.base_url) from e

        choice = response.choices[0]
        message = choice.message
        finish_reason = normalize_finish_reason(choice.finish_reason)

        prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
            extract_openai_usage(response.usage)
        )
        cost = self._record_cost(prompt_tokens, completion_tokens, total_tokens, cached_tokens)

        metadata = usage_metadata(
            prompt_tokens, completion_tokens, total_tokens, cached_tokens,
            cost, self.total_cost,
        )
        metadata["tool_calls"] = tool_calls_from_message(message)
        metadata["message"] = message

        generated_text = message.content or ""
        annotate_reasoning_only(
            metadata,
            text=generated_text,
            reasoning_text=extract_reasoning_text(message),
            reasoning_tokens=extract_reasoning_tokens(response.usage),
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=self._requested_output_cap(request_params),
            completion_tokens=completion_tokens,
            tool_calls=metadata["tool_calls"],
            logger=logger,
        )

        return GenerationResult(
            text=generated_text,
            tokens_used=completion_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata=metadata,
        )

    def generate_with_native_tools(
        self,
        prompt: str,
        native_tool_specs: list[dict[str, Any]],
        function_tool_specs: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
        previous_response_id: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate using the OpenAI Responses API with server-side native tools.

        This method is the correct path for OpenAI first-party tools
        (web_search_preview, code_interpreter, file_search) which run inside
        OpenAI's infrastructure and cannot be executed locally.

        It uses ``client.responses.create`` (Responses API) rather than the
        Chat Completions API, because native tools are only supported there.

        Args:
            prompt: User message / task description.
            native_tool_specs: List of native tool spec dicts from
                ``OpenAINativeTool.to_openai_tool_spec()``.
            function_tool_specs: Optional list of regular function-call tool
                specs (OpenAI format) to include alongside native tools.
            system_prompt: Optional system instructions.
            config: Generation configuration.
            previous_response_id: Responses API conversation ID for multi-turn.
            **kwargs: Extra params forwarded to the API.

        Returns:
            GenerationResult with the final text and metadata.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "generate_with_native_tools")
        self.validate_prompt(prompt)
        if config is None:
            config = GenerationConfig()

        # Fold per-call generation kwargs into the config so they travel the
        # model-aware path: max_tokens becomes max_output_tokens below and the
        # sampling params are gated by family, instead of being merged raw into
        # the Responses API (which uses a different parameter vocabulary and
        # 400s on a raw max_tokens / temperature for gpt-5 / o-series).
        config = _fold_generation_kwargs(config, kwargs)

        # Build the tools list for the Responses API.
        # The Responses API uses a flat format for function tools:
        #   {"type": "function", "name": ..., "description": ..., "parameters": ...}
        # Unlike Chat Completions which wraps them in {"type": "function", "function": {...}}.
        tools_list: list[dict[str, Any]] = list(native_tool_specs)
        if function_tool_specs:
            for spec in function_tool_specs:
                if spec.get("type") == "function":
                    fn = spec.get("function", spec)
                    tools_list.append({
                        "type": "function",
                        "name": fn.get("name", spec.get("name", "")),
                        "description": fn.get("description", spec.get("description", "")),
                        "parameters": fn.get("parameters", spec.get("parameters", {})),
                    })
                else:
                    # Already in flat Responses API format
                    tools_list.append(spec)

        # Build the input messages
        input_messages: list[dict[str, Any]] = []
        if system_prompt:
            input_messages.append({"role": "system", "content": system_prompt})
        input_messages.append({"role": "user", "content": prompt})

        max_tokens = config.max_tokens or _pick_default_max_output(self.model_name)

        params: dict[str, Any] = {
            "model": self.model_name,
            "input": input_messages,
            "tools": tools_list,
            "max_output_tokens": max_tokens,
        }
        if any(spec.get("type", "").startswith("web_search") for spec in tools_list):
            # Ask for the URLs a web search actually returned, not just the
            # ones the model chose to cite inline: a search can run and
            # inform the answer without the model emitting a url_citation
            # annotation, and grounding should not depend on that choice.
            params["include"] = ["web_search_call.action.sources"]
        if previous_response_id:
            params["previous_response_id"] = previous_response_id
        if self._supports_sampling_params:
            params["temperature"] = config.temperature
        if config.reasoning_effort is not None and self._is_reasoning_model:
            params["reasoning"] = {"effort": config.reasoning_effort}
        params.update(kwargs)

        try:
            response = self.client.responses.create(**params)
        except Exception as e:
            logger.error(f"OpenAI Responses API call failed: {e}")
            raise provider_runtime_error("openai", self.model_name, "generate_with_tools", e, message="OpenAI native tool generation failed", endpoint=self.base_url) from e

        # Extract the output text from the response
        output_text = ""
        tool_call_results: list[dict[str, Any]] = []
        # Grounded source URLs from the web_search tool. Two kinds land here,
        # both as ``grounding_chunks`` so the Agent can fill AgentResponse
        # .sources / .citations from real provider data: URLs the model cited
        # inline (url_citation annotations, the default cited entries) and URLs
        # a search returned but the model never referenced (action.sources,
        # marked ``cited: False`` so they widen .sources without becoming a
        # .citations entry).
        grounding_chunks: list[dict[str, Any]] = []

        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for content_block in getattr(item, "content", []):
                    block_type = getattr(content_block, "type", None)
                    if block_type in ("output_text", "text"):
                        output_text += getattr(content_block, "text", "")
                    for ann in getattr(content_block, "annotations", None) or []:
                        if getattr(ann, "type", None) != "url_citation":
                            continue
                        url = getattr(ann, "url", None)
                        if not url:
                            continue
                        grounding_chunks.append({
                            "url": url,
                            "title": getattr(ann, "title", None),
                        })
            elif item_type == "web_search_call":
                action = getattr(item, "action", None)
                query = getattr(action, "query", None) if action else None
                tool_call_results.append({
                    "type": "web_search_call",
                    "id": getattr(item, "id", ""),
                    "query": query,
                })
                # A search action carries every URL it returned in
                # action.sources, independent of which (if any) the model
                # goes on to cite inline. Fold those in as recall-oriented
                # grounding: they widen response.sources so a search that
                # ran is never silently unsourced, but they are marked
                # "cited": False so they never manufacture a Citation the
                # model did not actually make (url_citation annotations
                # above remain the only source of response.citations).
                for src in getattr(action, "sources", None) or []:
                    url = getattr(src, "url", None)
                    if not url:
                        continue
                    grounding_chunks.append({"url": url, "cited": False})
            elif item_type == "code_interpreter_call":
                outputs = []
                for out in getattr(item, "outputs", []) or []:
                    outputs.append({"type": getattr(out, "type", ""), "logs": getattr(out, "logs", "")})
                tool_call_results.append({
                    "type": "code_interpreter_call",
                    "id": getattr(item, "id", ""),
                    "code": getattr(item, "code", ""),
                    "outputs": outputs,
                })
            elif item_type == "file_search_call":
                results = []
                for r in getattr(item, "results", []) or []:
                    results.append({
                        "file_id": getattr(r, "file_id", ""),
                        "filename": getattr(r, "filename", ""),
                        "score": getattr(r, "score", 0.0),
                        "text": getattr(r, "text", ""),
                    })
                tool_call_results.append({
                    "type": "file_search_call",
                    "id": getattr(item, "id", ""),
                    "results": results,
                })
            elif item_type == "function_call":
                # The Responses API names a function call "function_call" and
                # reports it flat. Both are kept — they are what the native
                # tool loop dispatches on — and the nested ``function`` block
                # every adapter reports is added beside them.
                arguments = getattr(item, "arguments", "{}")
                tool_call_results.append({
                    "type": "function_call",
                    "id": getattr(item, "id", ""),
                    "name": getattr(item, "name", ""),
                    "arguments": arguments,
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": stringify_tool_arguments(arguments),
                    },
                })

        # Usage
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        cached_tokens = 0
        if usage:
            details = getattr(usage, "input_tokens_details", None)
            if details:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0

        cost = self._record_cost(prompt_tokens, completion_tokens, total_tokens, cached_tokens)

        metadata = usage_metadata(
            prompt_tokens, completion_tokens, total_tokens, cached_tokens,
            cost, self.total_cost,
        )
        metadata.update({
            "response_id": getattr(response, "id", None),
            "tool_calls": tool_call_results,
            "native_tool_results": tool_call_results,
            "grounding_chunks": grounding_chunks,
        })

        finish_reason = self._responses_finish_reason(response)
        annotate_reasoning_only(
            metadata,
            text=output_text,
            reasoning_text="",  # the Responses API does not return the chain
            reasoning_tokens=extract_reasoning_tokens(usage),
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=self._requested_output_cap(params),
            completion_tokens=completion_tokens,
            tool_calls=tool_call_results,
            logger=logger,
        )

        return GenerationResult(
            text=output_text,
            tokens_used=completion_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata=metadata,
        )

    @staticmethod
    def _responses_finish_reason(response: Any) -> str:
        """Canonical finish reason for one Responses-API response.

        The Responses API reports a ``status`` (``"completed"`` /
        ``"incomplete"``) and, when it stopped early, an
        ``incomplete_details.reason``. A run cut off at ``max_output_tokens`` is
        the same truncation the chat-completions API calls ``"length"``, so it
        is reported as ``"length"`` — the agent grows the budget and retries on
        that reason, and would not on a bare ``"incomplete"``.
        """
        status = getattr(response, "status", "completed")
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        if reason == "max_output_tokens":
            return FINISH_LENGTH
        if reason == "content_filter":
            return FINISH_CONTENT_FILTER
        return normalize_finish_reason(status)

    def chat(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Multi-turn chat with optional tool calling.

        Args:
            messages: Full conversation history in OpenAI format.
            config: Generation configuration.
            tools: Optional tool definitions.
            **kwargs: Extra params forwarded to the API.

        Returns:
            The next assistant turn, with any tool calls in its metadata.
        """
        if not self._is_loaded:
            raise not_loaded_error("openai", self.model_name, "chat")
        if config is None:
            config = GenerationConfig()

        # Validate the last user message length (approximate).
        # Messages may be dicts or SDK Pydantic objects from a prior turn.
        last_user = ""
        for m in reversed(messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "user":
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                if isinstance(content, str):
                    last_user = content
                break
        if last_user:
            self.validate_prompt(last_user)

        config = _fold_generation_kwargs(config, kwargs)
        request_params = self._build_request_params(messages, config)
        if tools:
            request_params["tools"] = tools
        request_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise provider_runtime_error("openai", self.model_name, "chat", e, message="OpenAI chat failed", endpoint=self.base_url) from e

        choice = response.choices[0]
        message = choice.message
        prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
            extract_openai_usage(response.usage)
        )
        cost = self._record_cost(prompt_tokens, completion_tokens, total_tokens, cached_tokens)

        metadata = usage_metadata(
            prompt_tokens, completion_tokens, total_tokens, cached_tokens,
            cost, self.total_cost,
        )
        metadata["tool_calls"] = tool_calls_from_message(message)
        metadata["message"] = message

        finish_reason = normalize_finish_reason(choice.finish_reason)
        annotate_reasoning_only(
            metadata,
            text=message.content or "",
            reasoning_text=extract_reasoning_text(message),
            reasoning_tokens=extract_reasoning_tokens(response.usage),
            model_name=self.model_name,
            finish_reason=finish_reason,
            max_tokens=self._requested_output_cap(request_params),
            completion_tokens=completion_tokens,
            tool_calls=metadata["tool_calls"],
            logger=logger,
        )

        return GenerationResult(
            text=message.content or "",
            tokens_used=completion_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Capability queries
    # ------------------------------------------------------------------

    def supports_function_calling(self) -> bool:
        """True: OpenAI chat models support native function calling."""
        return True

    def supports_tool_calling(self) -> bool:
        """True when the catalog marks this model as supporting native tools."""
        return OPENAI_MODELS.get(self.model_name, {}).get("supports_native_tools", True)

    def supports_forced_tool_call(self) -> bool:
        """True when tools are offered: the Chat Completions API honours ``tool_choice``.

        A model that is not offered tool definitions cannot be required to call
        one, so this follows :meth:`supports_tool_calling` rather than standing
        on its own. The same holds for any server speaking the same protocol,
        which is why
        :class:`~effgen.models.openai_compatible_adapter.OpenAICompatibleAdapter`
        inherits it unchanged.
        """
        return self.supports_tool_calling()

    def streams_tool_calls(self) -> bool:
        """True: a streamed turn's native tool calls are recorded."""
        return True

    def is_reasoning_model(self) -> bool:
        """Return True if this is an o-series reasoning model."""
        return self._is_reasoning_model

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Count tokens with the model's tiktoken encoding.

        A model tiktoken does not know is counted with ``cl100k_base``; an
        encoding that cannot be loaded at all falls back to a length-based
        estimate rather than failing the call that asked for the count.
        """
        return TokenCount(
            count=estimate_tokens(text, model=self.model_name),
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # Context / cost helpers
    # ------------------------------------------------------------------

    def get_context_length(self) -> int:
        """Return the model's context window size in tokens."""
        return self._context_length

    def get_total_cost(self) -> float:
        """Cumulative cost (USD) charged to this adapter instance."""
        return self.total_cost

    def get_total_tokens(self) -> int:
        """Cumulative tokens (prompt + completion) used by this adapter instance."""
        return self.total_tokens

    def reset_usage_stats(self) -> None:
        """Reset the cumulative cost and token counters to zero."""
        self.total_cost = 0.0
        self.total_tokens = 0
        logger.info("Usage statistics reset")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Close the client and mark the adapter unloaded."""
        if self.client is not None:
            logger.info(
                f"Closing OpenAI client. Total cost: ${self.total_cost:.6f}, "
                f"Total tokens: {self.total_tokens}"
            )
            self.client.close()
            self.client = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_models(cls) -> list[str]:
        """Return all registered OpenAI model IDs."""
        from effgen.models.openai_models import available_models
        return available_models()

    @classmethod
    def list_reasoning_models(cls) -> list[str]:
        """Return o-series reasoning model IDs."""
        from effgen.models.openai_models import reasoning_models
        return reasoning_models()

    @classmethod
    def get_model_info(cls, model_id: str) -> dict:
        """Return registry info for *model_id*."""
        from effgen.models.openai_models import model_info
        return model_info(model_id)


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.openai_models import OPENAI_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "openai",
            OpenAIAdapter,
            OPENAI_MODELS,
            env_keys=["OPENAI_API_KEY"],
            capabilities={
                Capability.chat, Capability.streaming, Capability.tools,
                Capability.vision, Capability.audio_input, Capability.video_input,
                Capability.json_schema, Capability.thinking,
            },
            # No free tier; pay-per-token. Provider default = cheapest current text model.
            # gpt-5-nano: $0.05/$0.40; gpt-4o-mini: $0.15/$0.60; gpt-4.1: $2.00/$8.00 per 1M.
            # Pricing verified: https://platform.openai.com/docs/pricing (2026-05-11)
            pricing={"input_per_1m": 0.05, "output_per_1m": 0.40, "free_tier": False},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
