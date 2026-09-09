"""
Google Gemini API adapter.

Uses the `google-genai` SDK (google.genai). Supports Gemini Pro, Flash,
Flash-Lite, and Gemma models with:
- Native function calling / tool use
- Thinking budget (extended reasoning for supporting models)
- Multimodal inputs
- Streaming
- Cost tracking
- Rate-limit-aware retry with exponential back-off
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from typing import Any

from effgen.models._adapter_utils import (
    annotate_reasoning_only,
    merge_call_overrides,
    model_not_found_error,
    normalize_finish_reason,
    not_loaded_error,
    provider_runtime_error,
    warn_empty_stream,
    warn_reasoning_only_stream,
)
from effgen.models._multimodal import (
    require_audio_support,
    require_video_support,
    require_vision_support,
)
from effgen.models._usage import record_tracker_cost, tool_call_entry
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
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    error_has_status,
)
from effgen.models.gemini_files import FileRef
from effgen.models.gemini_models import (
    GEMINI_DEFAULT_MODEL,
    GEMINI_MODEL_ALIASES,
    GEMINI_MODELS,
)
from effgen.models.latency_tracker import timed_call
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr

logger = logging.getLogger(__name__)


def _chunk_parts(candidates: Any) -> list[Any]:
    """Return the content parts of a response chunk's first candidate.

    A chunk carrying only usage or a finish reason has no parts, and a
    ``None`` content block reads the same way, so both come back empty.
    """
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    return list(getattr(content, "parts", None) or [])


def _function_call_entry(part: Any) -> dict[str, Any] | None:
    """Return one part's function call in the documented shape, or ``None``.

    The nested ``function`` block is the shape every adapter reports; the flat
    ``name``/``arguments`` keys are kept beside it for callers written against
    the earlier shape.
    """
    call = getattr(part, "function_call", None)
    if not call or not getattr(call, "name", None):
        return None
    args: dict[str, Any] = {}
    if hasattr(call, "args"):
        try:
            args = dict(call.args)
        except Exception:  # noqa: BLE001 - a mapping-like args object still reads
            args = {key: call.args[key] for key in call.args}
    entry = tool_call_entry(call.name, args, call_id=getattr(call, "id", "") or "")
    entry["name"] = call.name
    entry["arguments"] = args
    return entry


class GeminiAdapter(FunctionCallingModel):
    """
    Adapter for Google Gemini API models.

    Wraps the `google.genai` SDK and exposes a unified effGen interface.

    Features:
    - Gemini 2.x / 3.x and Gemma 3 / 4 family support
    - Native function calling (OpenAI-format tools auto-converted)
    - thinking_budget / include_thoughts for reasoning models
    - Streaming via generate_content_stream
    - Per-call cost tracking
    - Rate-limit-aware retry (honors Retry-After from 429s)

    Attributes:
        model_name: Gemini model ID (e.g. 'gemini-3.1-flash-lite')
        api_key: Google API key (reads GOOGLE_API_KEY from env if not given)
        safety_settings: Content safety filter settings
        timeout: Per-request deadline in seconds (default 60)
        max_retries: Retries after the first request on a rate limit or a
            transient server error; 0 means one request and no retry. Left
            unset it keeps the adapter's own default, which is high because
            the free tier refuses requests often.
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "gemini"

    COST_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
        "gemini-1.5-pro": (3.5, 10.5),
        "gemini-1.5-flash": (0.35, 1.05),
        "gemini-pro": (0.5, 1.5),
        "gemini-pro-vision": (0.5, 1.5),
        "gemini-ultra": (10.0, 30.0),
        # Gemini 2.x / 3.x — free tier, cost effectively $0 on free quota
        "gemini-2.0-flash": (0.1, 0.4),
        "gemini-2.5-flash": (0.15, 0.6),
        "gemini-2.5-pro": (1.25, 5.0),
        "gemini-3.1-flash-lite": (0.01, 0.04),
    }

    CONTEXT_LENGTHS: dict[str, int] = {
        "gemini-1.5-pro": 1_000_000,
        "gemini-1.5-flash": 1_000_000,
        "gemini-pro": 32_760,
        "gemini-pro-vision": 16_384,
        "gemini-ultra": 32_760,
        **{m: info["context"] for m, info in GEMINI_MODELS.items()},
        **{
            alias: GEMINI_MODELS[c]["context"]
            for alias, c in GEMINI_MODEL_ALIASES.items()
            if c in GEMINI_MODELS
        },
    }

    MAX_RATE_LIMIT_RETRIES = 7

    def __init__(
        self,
        model_name: str = GEMINI_DEFAULT_MODEL,
        api_key: str | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        timeout: float = 60.0,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> None:
        import os
        # Resolve short/legacy aliases (e.g. the retired
        # "gemini-3.1-flash-lite-preview") to canonical registry IDs
        # (e.g. "gemini-3.1-flash-lite"). Aliases are an effGen convenience.
        canonical_name = GEMINI_MODEL_ALIASES.get(model_name, model_name)
        super().__init__(
            model_name=canonical_name,
            model_type=ModelType.GEMINI,
            context_length=self.CONTEXT_LENGTHS.get(canonical_name, 32_760),
        )
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.safety_settings = safety_settings
        self.timeout = float(timeout)
        self.max_retries = (
            self.MAX_RATE_LIMIT_RETRIES - 1 if max_retries is None
            else max(0, int(max_retries))
        )
        self.additional_kwargs = kwargs
        self.client: Any = None
        self.total_cost = 0.0
        self.total_tokens = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _http_options(self) -> Any:
        """Per-request HTTP options: the call deadline and the retry budget.

        Without a deadline the SDK waits indefinitely for an endpoint that
        never answers, regardless of what ``timeout`` was asked for; the value
        is stated in milliseconds.

        The SDK's own retry is switched off (``attempts=1``). This adapter
        retries a 429 itself, honouring the delay Gemini states in the error,
        and a second retry layer underneath multiplies the wait: five SDK
        attempts inside each of the adapter's own turns into a call that can
        sit for minutes on a rate limit.
        """
        from google.genai import types

        return types.HttpOptions(
            timeout=int(self.timeout * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        )

    def load(self) -> None:
        """Initialize the google.genai client."""
        try:
            import google.genai as genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            ) from exc

        try:
            logger.info("Initializing Gemini client for model '%s'…", self.model_name)
            self.client = genai.Client(
                api_key=self.api_key, http_options=self._http_options()
            )
            self._is_loaded = True
            self._metadata = {
                "model_name": self.model_name,
                "context_length": self.get_context_length(),
                "supports_functions": True,
                "supports_streaming": True,
                "supports_multimodal": True,
            }
            logger.info("Gemini client initialized for '%s'", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize Gemini client: %s", exc)
            raise RuntimeError(f"Gemini initialization failed: {exc}") from exc

    def unload(self) -> None:
        """Release the Gemini client."""
        if self.client is not None:
            logger.info(
                "Closing Gemini client. Total cost: $%.4f, Total tokens: %d",
                self.total_cost,
                self.total_tokens,
            )
            self.client = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_flag(self, flag: str) -> bool:
        """Return the boolean *flag* from the registered model entry (False if absent)."""
        canonical = GEMINI_MODEL_ALIASES.get(self.model_name, self.model_name)
        info = GEMINI_MODELS.get(canonical) or GEMINI_MODELS.get(self.model_name)
        if info is None:
            return False
        return bool(info.get(flag, False))

    def _model_supports_thinking(self) -> bool:
        """Return True if the registered model entry has supports_thinking=True."""
        return self._model_flag("supports_thinking")

    def _model_supports_grounding(self) -> bool:
        """Return True if the registered model entry has supports_grounding=True."""
        return self._model_flag("supports_grounding")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost_entry: tuple[float, float] | None = None
        for model_id, costs in self.COST_PER_1M_TOKENS.items():
            if model_id in self.model_name:
                cost_entry = costs
                break
        if cost_entry is None:
            cost_entry = (0.5, 1.5)
        return (prompt_tokens / 1_000_000) * cost_entry[0] + (
            completion_tokens / 1_000_000
        ) * cost_entry[1]

    def _record_to_cost_tracker(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """Record this call in the process-global cost tracker.

        Returns the catalog-priced USD cost (so it matches the model catalog and
        the dashboard), or ``None`` if tracking is unavailable.  Re-raises
        :class:`BudgetExceededError` so budget limits are enforced.
        """
        return record_tracker_cost(
            "gemini",
            self.model_name,
            prompt_tokens,
            completion_tokens,
            log=logger,
        )

    def _settle_cost(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> float | None:
        """Price this call and fold it into the adapter's running session totals.

        Prefers the catalog-priced cost from the process-global tracker (so the
        number matches ``effgen cost`` and the dashboard); falls back to the
        local per-1M estimate when tracking is unavailable. Returns this call's
        cost — the value reported as ``metadata["cost_usd"]`` — or ``None``
        when the catalog publishes no rate for the model, in which case the
        local estimate is not substituted and the session total is unchanged.
        """
        from effgen.models._cost import pricing_status

        unpriced = pricing_status("gemini", self.model_name) == "unpriced"
        cost = self._record_to_cost_tracker(prompt_tokens, completion_tokens)
        if cost is None and not unpriced:
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
        fold_call_totals(self, cost, total_tokens)
        return cost

    def _build_config(self, config: GenerationConfig | None) -> Any:
        """Build a google.genai GenerateContentConfig from an effGen GenerationConfig."""
        from google.genai import types  # type: ignore[import]

        if config is None:
            config = GenerationConfig()

        kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_tokens or 2048,
        }
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences
        if config.seed is not None:
            kwargs["seed"] = config.seed
        if config.presence_penalty:
            kwargs["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            kwargs["frequency_penalty"] = config.frequency_penalty

        # Thinking config — only for models that support it.
        if config.thinking_budget is not None:
            if self._model_supports_thinking():
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=config.thinking_budget,
                    include_thoughts=config.include_thoughts,
                )
            else:
                logger.debug(
                    "Model '%s' does not support thinking_budget; ignoring.",
                    self.model_name,
                )

        # Grounding — attach Google Search tool when requested and supported.
        if config.grounding:
            if self._model_supports_grounding():
                kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            else:
                logger.debug(
                    "Model '%s' does not support grounding; ignoring grounding=True.",
                    self.model_name,
                )

        if self.safety_settings:
            kwargs["safety_settings"] = self.safety_settings

        # Native structured-output / JSON mode. response_mime_type forces the
        # model to emit raw JSON (no markdown fences); response_schema additionally
        # constrains the shape. Gemini's response_schema accepts only an OpenAPI
        # subset and rejects standard JSON-Schema extras (e.g. additionalProperties,
        # $schema, title) at request time — not at SDK-construct time — so we strip
        # to that subset with _sanitize_schema. If the cleaned schema is empty/invalid
        # we keep JSON mode only and validate the result ourselves downstream.
        if config.response_mime_type:
            kwargs["response_mime_type"] = config.response_mime_type
            if config.response_schema:
                cleaned = self._sanitize_schema(config.response_schema)
                if isinstance(cleaned, dict) and cleaned.get("properties"):
                    try:
                        types.GenerateContentConfig(
                            response_mime_type=config.response_mime_type,
                            response_schema=cleaned,
                        )
                        kwargs["response_schema"] = cleaned
                    except Exception:
                        logger.debug(
                            "Gemini rejected response_schema dict; using JSON mode only.",
                            exc_info=True,
                        )

        return types.GenerateContentConfig(**kwargs)

    # Gemini's Schema accepts a strict subset of JSON Schema. Strip extras
    # before sending so the SDK does not raise "Unknown field for Schema".
    _GEMINI_SCHEMA_FIELDS = {
        "type", "description", "enum", "items", "properties",
        "required", "format", "nullable",
    }

    @classmethod
    def _sanitize_schema(cls, schema: Any) -> Any:
        if isinstance(schema, dict):
            cleaned: dict[str, Any] = {}
            for k, v in schema.items():
                if k not in cls._GEMINI_SCHEMA_FIELDS:
                    continue
                if k == "properties" and isinstance(v, dict):
                    cleaned[k] = {
                        name: cls._sanitize_schema(s) for name, s in v.items()
                    }
                elif k == "items":
                    cleaned[k] = cls._sanitize_schema(v)
                elif k == "required":
                    cleaned[k] = list(v) if isinstance(v, list | tuple) else v
                else:
                    cleaned[k] = v
            # Gemini rejects an ``array`` property that omits ``items``. Default a
            # missing element schema to a string so tools that declare a plain
            # ``type: array`` parameter are accepted (OpenAI/Groq tolerate the
            # omission). Covers every current and future tool without per-tool
            # patches.
            type_val = cleaned.get("type")
            if isinstance(type_val, str) and type_val.lower() == "array" \
                    and "items" not in cleaned:
                cleaned["items"] = {"type": "string"}
            return cleaned
        if isinstance(schema, list):
            return [cls._sanitize_schema(v) for v in schema]
        return schema

    @classmethod
    def _convert_tools_to_genai(cls, tools: list) -> list:
        """Convert effGen / OpenAI-style tool specs to google.genai Tool objects.

        Handles:
        - ``GeminiNativeTool`` subclasses (GoogleSearch, UrlContext, CodeExecution)
        - ``google.genai.types.Tool`` passthrough
        - OpenAI-format dicts with ``type="function"``
        """
        from google.genai import types  # type: ignore[import]

        function_declarations: list[Any] = []
        passthrough: list[Any] = []

        for t in tools:
            # Gemini first-party native tools
            try:
                from effgen.tools.builtin.gemini_native import GeminiNativeTool
                if isinstance(t, GeminiNativeTool):
                    passthrough.append(t.to_gemini_tool())
                    continue
            except ImportError:
                pass

            if isinstance(t, types.Tool):
                passthrough.append(t)
                continue
            if isinstance(t, dict):
                func = t["function"] if t.get("type") == "function" and "function" in t else t
                params = func.get("parameters") or {"type": "object", "properties": {}}
                params = cls._sanitize_schema(params)
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=params,
                    )
                )

        result = list(passthrough)
        if function_declarations:
            result.append(types.Tool(function_declarations=function_declarations))
        return result

    @staticmethod
    def _suggested_retry_delay(exc: Exception) -> float | None:
        for attr in ("retry_delay", "_details", "details"):
            obj = getattr(exc, attr, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "seconds"):
                    return float(obj.seconds) + float(getattr(obj, "nanos", 0)) / 1e9
                items = obj() if callable(obj) else obj
                for item in items or []:
                    rd = getattr(item, "retry_delay", None)
                    if rd is not None and hasattr(rd, "seconds"):
                        return float(rd.seconds) + float(getattr(rd, "nanos", 0)) / 1e9
            except Exception:
                continue
        msg = str(exc)
        if "Please retry in" in msg:
            try:
                tail = msg.split("Please retry in", 1)[1]
                num = ""
                for ch in tail.strip():
                    if ch.isdigit() or ch == ".":
                        num += ch
                    else:
                        break
                return float(num) if num else None
            except Exception:
                return None
        return None

    def _generate_with_retry(
        self,
        *,
        contents: Any,
        gen_config: Any,
        stream: bool = False,
        tools: list | None = None,
    ) -> Any:
        """Call client.models.generate_content[_stream] with rate-limit retries."""
        if tools:
            gen_config = _inject_tools(gen_config, tools)

        last_exc: Exception | None = None
        _retry_start = time.monotonic()
        _MAX_RETRY_SECONDS = 45.0
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                if stream:
                    return self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=gen_config,
                    )
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=gen_config,
                )
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                exc_cls = type(exc).__name__
                is_quota = error_has_status(exc, 429) or "quota" in exc_str.lower() or "ResourceExhausted" in exc_cls
                is_transient = (
                    any(kw in exc_cls for kw in ("ServiceUnavailable", "DeadlineExceeded", "InternalServerError", "ServerError"))
                    or "503" in exc_str
                    or "500" in exc_str
                    or "UNAVAILABLE" in exc_str
                )
                if not (is_quota or is_transient):
                    raise
                # Hard daily/project quota exhausted — no point retrying
                is_hard_quota = is_quota and (
                    "exceeded your current quota" in exc_str.lower()
                    or "per_day" in exc_str.lower()
                    or "project" in exc_str.lower() and "quota" in exc_str.lower()
                )
                if is_hard_quota:
                    raise
                if attempt >= attempts:
                    break
                if time.monotonic() - _retry_start >= _MAX_RETRY_SECONDS:
                    logger.warning("Gemini retry budget exhausted after %.1fs", _MAX_RETRY_SECONDS)
                    break
                delay = self._suggested_retry_delay(exc) if is_quota else None
                if delay is None:
                    delay = min(60.0, (2 ** attempt) + random.uniform(0, 0.5))
                else:
                    delay = max(delay + 0.5, 1.0)
                # Don't sleep past the budget
                remaining = _MAX_RETRY_SECONDS - (time.monotonic() - _retry_start)
                delay = min(delay, remaining)
                if delay <= 0:
                    break
                logger.warning(
                    "Gemini transient error on attempt %d/%d: %s — sleeping %.1fs",
                    attempt, attempts, type(exc).__name__, delay,
                )
                time.sleep(delay)
        if last_exc is None:
            raise RuntimeError(
                f"Gemini made no request for '{self.model_name}': the retry "
                f"budget was {attempts} attempts."
            )
        raise last_exc

    def _validate_media_support(self, prompt: Any) -> None:
        """Reject image/audio/video inputs the current model cannot handle.

        Video requires vision (for the frame-sampling fallback) or native
        video support.
        """
        model_info = GEMINI_MODELS.get(self.model_name, {})
        require_vision_support(
            prompt,
            provider="gemini",
            model_name=self.model_name,
            supports_vision=model_info.get("supports_vision", False),
            hint="Use a Gemini model with supports_vision=True for image inputs.",
        )
        require_audio_support(
            prompt,
            provider="gemini",
            model_name=self.model_name,
            supports_audio=model_info.get("supports_audio", False),
            hint="Use a Gemini Flash/Pro model with supports_audio=True for audio inputs.",
        )
        require_video_support(
            prompt,
            provider="gemini",
            model_name=self.model_name,
            supports_video=model_info.get("supports_vision", False) or model_info.get("supports_video", False),
            hint="Use a Gemini vision model for video inputs (frame-sampling fallback is used).",
        )

    def _prepare_content(
        self, prompt: str | list[str | dict[str, Any]]
    ) -> str | list:
        # Handle Message objects (multimodal input via effGen unified schema)
        try:
            from effgen.core.messages import Message
            from effgen.multimodal.image_pre import prepare as _preprocess_image

            if isinstance(prompt, Message):
                return self._message_to_genai_parts(prompt, _preprocess_image)
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                # Multi-turn: flatten all messages into a contents list
                from google.genai import types as _gt  # type: ignore[import]

                from ._tool_wire import call_names_by_id

                # A result part carries the id it answers but not the tool's
                # name, and a function response is keyed by name here. The name
                # is read off the call that earned the id.
                call_names = call_names_by_id(prompt)
                contents = []
                for msg in prompt:
                    # A tool result belongs to the user side of the exchange:
                    # the model asked, and this is what came back.
                    role = (
                        "user"
                        if msg.role.value in ("user", "system", "tool")
                        else "model"
                    )
                    contents.append(_gt.Content(
                        role=role,
                        parts=self._message_to_genai_parts(
                            msg, _preprocess_image, call_names=call_names,
                        ),
                    ))
                return contents
        except ImportError:
            pass

        if isinstance(prompt, str):
            return prompt
        parts: list[Any] = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "image" in item:
                import PIL.Image  # type: ignore[import]
                parts.append(PIL.Image.open(item["image"]))
            else:
                parts.append(item)
        return parts

    def _create_messages(self, prompt: Any) -> list[Any] | None:
        """The prompt as this provider's contents list, when it is a conversation.

        An assistant turn's tool call travels as a ``function_call`` part
        carrying the id the provider minted, and the result answering it as a
        ``function_response`` part quoting the same id.

        Returns:
            The contents list, or ``None`` when *prompt* is not a conversation.
        """
        try:
            from effgen.core.messages import Message
        except ImportError:  # pragma: no cover - effgen.core is always importable
            return None
        if isinstance(prompt, Message):
            prompt = [prompt]
        if not (isinstance(prompt, list) and prompt and isinstance(prompt[0], Message)):
            return None
        contents = self._prepare_content(prompt)
        return list(contents) if isinstance(contents, list) else None

    def _message_to_genai_parts(
        self, message: Any, preprocess_fn: Any,
        call_names: dict[str, str] | None = None,
    ) -> list[Any]:
        """Convert an effGen Message to a list of google.genai Part objects.

        A tool call travels as a ``function_call`` part carrying the id the
        provider minted, and its result as a ``function_response`` part quoting
        the same id. *call_names* supplies the tool name a response is keyed
        by, since a result part carries only the id.

        VideoPart handling:
          - If the model supports native video (supports_video=True in registry)
            and the VideoPart has a ``meta["video_path"]`` or ``meta["video_url"]``
            key, the video is uploaded via the Files API and referenced by URI.
          - Otherwise the frames are sent as sequential JPEG image parts (frame-
            sampling fallback that works on all Gemini vision models).
        """
        from google.genai import types as _gt  # type: ignore[import]

        from effgen.core.messages import (
            AudioPart,
            ImagePart,
            TextPart,
            ToolCallPart,
            ToolResultPart,
            VideoPart,
        )

        from ._tool_wire import result_text

        model_info = GEMINI_MODELS.get(self.model_name, {})
        supports_native_video = bool(model_info.get("supports_video", False))
        names = call_names or {}

        parts: list[Any] = []
        for part in message.content:
            if isinstance(part, ToolCallPart):
                parts.append(_gt.Part(function_call=_gt.FunctionCall(
                    id=part.tool_call_id,
                    name=part.name,
                    args=dict(part.arguments),
                )))
                continue
            if isinstance(part, ToolResultPart):
                parts.append(_gt.Part(function_response=_gt.FunctionResponse(
                    id=part.tool_call_id,
                    name=names.get(part.tool_call_id, part.tool_call_id),
                    response={"result": result_text(part.result)},
                )))
                continue
            if isinstance(part, TextPart):
                parts.append(_gt.Part.from_text(text=part.text))
            elif isinstance(part, ImagePart):
                processed = preprocess_fn(part, "gemini", self.model_name)
                parts.append(_gt.Part.from_bytes(
                    data=processed.image,
                    mime_type=processed.mime,
                ))
            elif isinstance(part, AudioPart):
                parts.append(_gt.Part.from_bytes(
                    data=part.audio,
                    mime_type=part.mime,
                ))
            elif isinstance(part, VideoPart):
                # Try native video via Files API when both the model and the
                # VideoPart supply a file reference.
                video_path = part.meta.get("video_path")
                video_url = part.meta.get("video_url")
                video_mime = part.meta.get("video_mime", "video/mp4")

                if supports_native_video and (video_path or video_url):
                    try:
                        from effgen.models.gemini_files import upload_file as _upload
                        if video_path:
                            fref = _upload(video_path, api_key=self.api_key, mime_type=video_mime)
                        else:
                            # Download URL to a temp file then upload
                            import os
                            import tempfile
                            import urllib.request
                            suffix = ".mp4" if "mp4" in video_mime else ".webm"
                            tmp_path = ""
                            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                                with urllib.request.urlopen(video_url, timeout=60) as resp:
                                    tf.write(resp.read())
                                tmp_path = tf.name
                            try:
                                fref = _upload(tmp_path, api_key=self.api_key, mime_type=video_mime)
                            finally:
                                if tmp_path:
                                    try:
                                        os.unlink(tmp_path)
                                    except OSError:
                                        pass

                        parts.append(_gt.Part.from_uri(
                            file_uri=fref.uri,
                            mime_type=fref.mime_type,
                        ))
                        logger.info(
                            "gemini_adapter: uploaded video via Files API → %s", fref.uri
                        )
                        continue
                    except Exception as exc:
                        logger.warning(
                            "gemini_adapter: native video upload failed (%s); "
                            "falling back to frame-sampling", exc,
                        )

                # Frame-sampling fallback: send each JPEG frame as an image part
                for frame in part.frames:
                    parts.append(_gt.Part.from_bytes(
                        data=frame,
                        mime_type=part.mime,
                    ))
            else:
                # Tool call / tool result — skip for Gemini (handled separately)
                pass
        return parts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str | list[str | dict[str, Any]],
        config: GenerationConfig | None = None,
        *,
        files: list[FileRef] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate text (or a tool call) from *prompt*.

        Args:
            prompt: Text prompt or multimodal content list.
            config: Generation configuration.
            files: Optional list of :class:`~effgen.models.gemini_files.FileRef`
                objects returned by :func:`~effgen.models.gemini_files.upload_file`.
                Each file is prepended to the request content so the model can
                read it before answering *prompt*.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Returns:
            The generated text with its usage metadata.
        """
        if not self._is_loaded:
            raise not_loaded_error("gemini", self.model_name, "generate")
        if isinstance(prompt, str):
            self.validate_prompt(prompt)

        gen_config = self._build_config(merge_call_overrides(
            config if config is not None else GenerationConfig(), kwargs
        ))
        self._validate_media_support(prompt)
        content = self._prepare_content(prompt)

        # Prepend uploaded files to the content so the model sees them.
        if files:
            from google.genai import types as _gt  # type: ignore[import]
            file_parts: list[Any] = []
            for fref in files:
                sdk_obj = fref.raw
                if sdk_obj is not None:
                    file_parts.append(_gt.Part.from_uri(
                        file_uri=fref.uri,
                        mime_type=fref.mime_type,
                    ))
                else:
                    file_parts.append(_gt.Part.from_uri(
                        file_uri=fref.uri,
                        mime_type=fref.mime_type,
                    ))
            if isinstance(content, list):
                content = file_parts + content
            else:
                content = file_parts + [content]

        raw_tools: list | None = None
        if kwargs.get("tools"):
            raw_tools = self._convert_tools_to_genai(kwargs.pop("tools"))

        try:
            with timed_call("gemini", self.model_name):
                response = self._generate_with_retry(
                    contents=content,
                    gen_config=gen_config,
                    tools=raw_tools,
                )

            generated_text = ""
            tool_calls: list[dict[str, Any]] = []
            thinking_text = ""

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                content_obj = getattr(candidate, "content", None)
                parts = getattr(content_obj, "parts", None) or []
                for part in parts:
                    if getattr(part, "thought", False):
                        thinking_text += getattr(part, "text", "") or ""
                    elif getattr(part, "text", None):
                        generated_text += part.text
                    # Parallel function calls
                    entry = _function_call_entry(part)
                    if entry is not None:
                        tool_calls.append(entry)
                    # Code execution result — surface output in generated text
                    cer = getattr(part, "code_execution_result", None)
                    if cer is not None:
                        output = getattr(cer, "output", None)
                        if isinstance(output, str) and output:
                            generated_text += output
                if not generated_text and not tool_calls:
                    try:
                        generated_text = response.text or ""
                    except Exception:
                        logger.debug("Failed to read response.text during generate", exc_info=True)
            else:
                try:
                    generated_text = response.text or ""
                except Exception:
                    logger.debug("Failed to read response.text during generate", exc_info=True)

            # Emit ALL tool calls as <tool_call> tokens for the agent dispatcher.
            # Parallel function calls: Gemini may return multiple functionCall
            # parts in one response turn.  Append each as a separate tag.
            if tool_calls and "<tool_call>" not in generated_text:
                import json as _json
                tc_blocks = "".join(
                    f"\n<tool_call>{_json.dumps({'name': tc['name'], 'arguments': tc['arguments']})}</tool_call>"
                    for tc in tool_calls
                )
                generated_text = (generated_text + tc_blocks).strip()

            # Token usage
            try:
                um = response.usage_metadata
                prompt_tokens = um.prompt_token_count or 0
                completion_tokens = um.candidates_token_count or 0
                total_tokens = um.total_token_count or (prompt_tokens + completion_tokens)
                thoughts_tokens = getattr(um, "thoughts_token_count", None) or 0
            except AttributeError:
                prompt_tokens = self.count_tokens(
                    prompt if isinstance(prompt, str) else str(prompt)
                ).count
                completion_tokens = self.count_tokens(generated_text).count
                total_tokens = prompt_tokens + completion_tokens
                thoughts_tokens = 0

            # Persist + price via the shared catalog-backed tracker so
            # `effgen cost` includes Gemini spend and the number matches the
            # model catalog; falls back to the local estimate if tracking is
            # unavailable.
            cost = self._settle_cost(prompt_tokens, completion_tokens, total_tokens)

            raw_finish_reason = None
            if hasattr(response, "candidates") and response.candidates:
                raw_finish_reason = response.candidates[0].finish_reason
            finish_reason = normalize_finish_reason(raw_finish_reason)

            # Extract grounding chunks (URLs + snippets) from the response.
            grounding_chunks: list[dict[str, Any]] = []
            try:
                gmd = None
                if hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    gmd = getattr(cand, "grounding_metadata", None)
                if gmd is None:
                    gmd = getattr(response, "grounding_metadata", None)
                if gmd is not None:
                    raw_chunks = getattr(gmd, "grounding_chunks", None) or []
                    for chunk in raw_chunks:
                        web = getattr(chunk, "web", None)
                        if web is not None:
                            grounding_chunks.append({
                                "url": getattr(web, "uri", None),
                                "title": getattr(web, "title", None),
                            })
                        else:
                            grounding_chunks.append({"raw": str(chunk)})
            except Exception as _ge:
                logger.debug("Could not extract grounding_chunks: %s", _ge)

            metadata: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "thoughts_token_count": thoughts_tokens,
                # ``cost_usd`` is the canonical per-call cost key shared by every
                # adapter. ``total_cost`` is a different number, not an alias: the
                # cumulative cost across every call made on this adapter instance
                # so far (including this one).
                "cost_usd": cost,
                "total_cost": self.total_cost,
                "safety_ratings": getattr(response, "safety_ratings", None),
                "tool_calls": tool_calls,
                "grounding_chunks": grounding_chunks,
                "raw_finish_reason": str(raw_finish_reason) if raw_finish_reason is not None else None,
            }
            if thinking_text:
                metadata["thinking"] = thinking_text

            # A thinking model can spend the whole output budget on its thoughts
            # and return no visible text. Report that turn the way every other
            # adapter does, naming the cap and the thinking budget it spent.
            annotate_reasoning_only(
                metadata,
                text=generated_text,
                reasoning_text=thinking_text,
                reasoning_tokens=thoughts_tokens,
                model_name=self.model_name,
                finish_reason=finish_reason,
                max_tokens=getattr(gen_config, "max_output_tokens", None),
                completion_tokens=completion_tokens,
                tool_calls=tool_calls,
                logger=logger,
            )

            # Emit span attributes on the current active span
            _set_span_attr(ModelAttrs.PROVIDER, "google")
            _set_span_attr(ModelAttrs.NAME, self.model_name)
            _set_span_attr(ModelAttrs.INPUT_TOKENS, prompt_tokens)
            _set_span_attr(ModelAttrs.OUTPUT_TOKENS, completion_tokens)
            if cost is not None:
                _set_span_attr(ModelAttrs.COST_USD, float(cost))
            _set_span_attr(ModelAttrs.OUTCOME, "ok")
            if thoughts_tokens:
                _set_span_attr(ModelAttrs.THINKING_BUDGET, thoughts_tokens)

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata=metadata,
            )

        except (ModelAuthError, ModelNotFoundError, InvalidRequestError):
            # Already a precisely-typed provider error — never re-wrap/re-classify.
            raise
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            msg = str(exc)
            lowered = msg.lower()
            if error_has_status(exc, 429) or "resource_exhausted" in lowered or "quota" in lowered:
                raise provider_runtime_error("gemini", self.model_name, "generate", exc, message="Gemini generation failed") from exc
            # Auth: 401, or a 400 INVALID_ARGUMENT specifically about the API key.
            if (
                error_has_status(exc, 401)
                or "api_key_invalid" in lowered
                or "api key not valid" in lowered
                or "permission_denied" in lowered
            ):
                raise ModelAuthError("gemini", self.model_name, msg) from exc
            if (
                "404" in msg
                or "model not found" in lowered
                or "is not found" in lowered
                or "not_found" in lowered
            ):
                raise model_not_found_error("gemini", self.model_name, msg) from exc
            # A generic 400 INVALID_ARGUMENT is a malformed/unsupported request
            # (e.g. bad schema, or combining google_search with function tools),
            # NOT an auth failure — classify it correctly so retries don't fire.
            if "invalid_argument" in lowered or "400" in msg:
                raise InvalidRequestError("gemini", self.model_name, msg) from exc
            raise provider_runtime_error("gemini", self.model_name, "generate", exc, message="Gemini generation failed") from exc

    def generate_stream(
        self,
        prompt: str | list[str | dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text chunks.

        Args:
            prompt: The prompt text, or a content list for multimodal input.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Yields:
            Successive text chunks from the model.
        """
        if not self._is_loaded:
            raise not_loaded_error("gemini", self.model_name, "generate_stream")
        if isinstance(prompt, str):
            self.validate_prompt(prompt)

        gen_config = self._build_config(merge_call_overrides(
            config if config is not None else GenerationConfig(), kwargs
        ))
        self._validate_media_support(prompt)
        content = self._prepare_content(prompt)

        raw_tools: list | None = None
        if kwargs.get("tools"):
            raw_tools = self._convert_tools_to_genai(kwargs.pop("tools"))

        _usage_metadata = None
        _raw_finish_reason = None
        clear_stream_tool_calls(self)
        _tool_calls: list[dict[str, Any]] = []
        try:
            with timed_call("gemini", self.model_name) as _stream_timer:
                response = self._generate_with_retry(
                    contents=content,
                    gen_config=gen_config,
                    stream=True,
                    tools=raw_tools,
                )
                _first_token = True
                for chunk in response:
                    if getattr(chunk, "usage_metadata", None) is not None:
                        _usage_metadata = chunk.usage_metadata
                    candidates = getattr(chunk, "candidates", None) or []
                    if candidates and getattr(candidates[0], "finish_reason", None):
                        _raw_finish_reason = candidates[0].finish_reason
                    # Read the parts rather than ``chunk.text``: that accessor
                    # drops every non-text part, so a streamed function call was
                    # lost (and the SDK warned about it on stderr) while its
                    # tokens were billed. A chunk that exposes no parts is still
                    # read through ``text``, which is where a response shape
                    # without a candidate block carries it.
                    parts = _chunk_parts(candidates)
                    if not parts:
                        try:
                            text = chunk.text
                        except Exception:
                            logger.debug(
                                "Failed to read chunk.text during streaming",
                                exc_info=True,
                            )
                            text = None
                        if text:
                            if _first_token:
                                _stream_timer.mark_first_token()
                                _first_token = False
                            yield text
                        continue
                    for part in parts:
                        text = getattr(part, "text", None)
                        if text:
                            if _first_token:
                                _stream_timer.mark_first_token()
                                _first_token = False
                            yield text
                        entry = _function_call_entry(part)
                        if entry is not None:
                            _tool_calls.append(entry)
                            # Recorded as it arrives, so a consumer streaming
                            # this turn knows it is a tool call before it
                            # commits any text as the answer.
                            record_stream_tool_calls(self, _tool_calls)
        except Exception as exc:
            logger.error("Gemini streaming failed: %s", exc)
            raise provider_runtime_error("gemini", self.model_name, "stream", exc, message="Gemini streaming failed") from exc

        record_stream_tool_calls(self, _tool_calls)

        # A stream has no metadata channel back to the caller, so a turn spent
        # entirely on thinking is reported here instead of arriving as an empty
        # iterator that reads as "the model had nothing to say".
        warn_reasoning_only_stream(
            model_name=self.model_name,
            yielded_text=not _first_token,
            reasoning_text="",  # thought parts are not streamed as text
            reasoning_tokens=getattr(_usage_metadata, "thoughts_token_count", None) or 0,
            finish_reason=_raw_finish_reason,
            max_tokens=getattr(gen_config, "max_output_tokens", None),
            tool_calls=_tool_calls,
            logger=logger,
        )
        if _usage_metadata is None:
            # A thinking model whose whole budget goes to its thoughts can end
            # the stream with no chunk at all — no content, no usage, no finish
            # reason — so the call above has nothing to key on.
            warn_empty_stream(
                model_name=self.model_name,
                yielded_text=not _first_token,
                max_tokens=getattr(gen_config, "max_output_tokens", None),
                tool_calls=_tool_calls,
                logger=logger,
            )

        # Record real usage from the streamed usage_metadata (the last chunk
        # carries the cumulative totals) so cost/token tracking and the CLI's
        # per-turn footer reflect streamed Gemini turns the same way a
        # non-streamed `generate()` call already does.
        if _usage_metadata is not None:
            try:
                prompt_tokens = _usage_metadata.prompt_token_count or 0
                completion_tokens = _usage_metadata.candidates_token_count or 0
                total_tokens = _usage_metadata.total_token_count or (prompt_tokens + completion_tokens)
                cost = self._settle_cost(prompt_tokens, completion_tokens, total_tokens)
                record_stream_usage(self, prompt_tokens, completion_tokens, cost)
            except Exception:
                logger.debug("Failed to record streaming usage for Gemini", exc_info=True)

    def build_assistant_message(self, result: GenerationResult) -> dict[str, Any]:
        """Return the assistant turn as Gemini's ``functionCall`` parts.

        Gemini's conversation is ``{"role": ..., "parts": [...]}`` and its
        assistant role is ``"model"``, so the OpenAI-shaped default would be
        rejected outright. Arguments are parsed back out of the uniform JSON
        string, because Gemini takes them as an object.

        Args:
            result: The turn to re-submit.

        Returns:
            One ``{"role": "model", "parts": [...]}`` message.
        """
        import json as _json

        parts: list[dict[str, Any]] = []
        if result.text:
            parts.append({"text": result.text})
        for call in (result.metadata or {}).get("tool_calls") or []:
            function = call.get("function", call)
            raw = function.get("arguments")
            try:
                args = _json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (ValueError, TypeError):
                args = {}
            parts.append({"functionCall": {"name": function.get("name"), "args": args}})
        return {"role": "model", "parts": parts}

    def build_tool_result_message(
        self, call_id: str, name: str, content: str
    ) -> dict[str, Any]:
        """Return one tool's result as Gemini's ``functionResponse`` part.

        Gemini matches a result to its call by **name**, not by id, so
        ``call_id`` is accepted for a uniform call site and not sent.

        Args:
            call_id: Unused here; kept so one loop serves every provider.
            name: The tool the result belongs to.
            content: What the tool returned.

        Returns:
            One ``{"role": "user", "parts": [...]}`` message.
        """
        _ = call_id
        return {
            "role": "user",
            "parts": [{"functionResponse": {"name": name, "response": {"result": content}}}],
        }

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with explicit tool list (convenience wrapper over generate).

        Args:
            prompt: The prompt to send.
            tools: Tool definitions to offer the model.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the provider SDK.

        Returns:
            The generated text, with any tool calls in its metadata.
        """
        return self.generate(prompt, config=config, tools=tools, **kwargs)

    # ------------------------------------------------------------------
    # Capabilities / token counting
    # ------------------------------------------------------------------

    def supports_function_calling(self) -> bool:
        """True: Gemini models support native function calling."""
        return True

    def supports_tool_calling(self) -> bool:
        """Alias for :meth:`supports_function_calling`."""
        return True

    def supports_message_protocol(self) -> bool:
        """True: this converter carries a tool call and a tool result through.

        An assistant turn's :class:`~effgen.core.messages.ToolCallPart` becomes
        a ``function_call`` part carrying the id the provider minted, and a
        :class:`~effgen.core.messages.ToolResultPart` becomes a
        ``function_response`` part quoting the same id.
        """
        return self.supports_tool_calling()

    def streams_tool_calls(self) -> bool:
        """True: a streamed turn's native function calls are recorded."""
        return True

    def count_tokens(self, text: str) -> TokenCount:
        """Count tokens via the Gemini API (length/4 approximation on failure)."""
        if not self._is_loaded:
            raise not_loaded_error("gemini", self.model_name, "count_tokens")
        try:
            response = self.client.models.count_tokens(
                model=self.model_name, contents=text
            )
            return TokenCount(count=response.total_tokens, model_name=self.model_name)
        except Exception as exc:
            logger.warning("Token counting failed: %s. Using approximation.", exc)
            return TokenCount(count=len(text) // 4, model_name=self.model_name)

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

    # Keep the old _create_generation_config name as a shim so any external
    # code that called it directly still works (it's not a public API but
    # let's be safe).
    def _create_generation_config(self, config: GenerationConfig | None = None) -> Any:
        return self._build_config(config)


# ---------------------------------------------------------------------------
# Module-level helper (avoids circular import if ever needed)
# ---------------------------------------------------------------------------

def _inject_tools(gen_config: Any, tools: list) -> Any:
    """Return a copy of *gen_config* with tools injected.

    When the request mixes server-side built-in tools (google_search,
    url_context, code_execution) with user function declarations, the
    Gemini API requires ``tool_config.include_server_side_tool_invocations=True``
    or it returns a 400.
    """
    from google.genai import types  # type: ignore[import]

    d = gen_config.model_dump(exclude_none=True) if hasattr(gen_config, "model_dump") else {}
    d["tools"] = tools

    has_native = False
    has_function_decls = False
    for t in tools:
        if isinstance(t, types.Tool):
            if any(getattr(t, attr, None) is not None for attr in (
                "google_search", "url_context", "code_execution",
                "google_search_retrieval", "retrieval", "computer_use",
            )):
                has_native = True
            if getattr(t, "function_declarations", None):
                has_function_decls = True

    if has_native and has_function_decls:
        existing = d.get("tool_config")
        if isinstance(existing, dict):
            existing.setdefault("include_server_side_tool_invocations", True)
            d["tool_config"] = existing
        else:
            d["tool_config"] = types.ToolConfig(include_server_side_tool_invocations=True)

    return types.GenerateContentConfig(**d)


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.gemini_models import GEMINI_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "gemini",
            GeminiAdapter,
            GEMINI_MODELS,
            env_keys=["GOOGLE_API_KEY"],
            capabilities={
                Capability.chat, Capability.streaming, Capability.tools,
                Capability.vision, Capability.audio_input, Capability.video_input,
                Capability.grounding, Capability.thinking, Capability.json_schema,
            },
            # Provider default = paid gemini-2.5-flash-lite. Gemini has a free
            # quota for Flash/Flash-Lite, but it has rate-limit and data-use
            # nuances, so cost routing keeps the published paid token prices.
            # gemini-2.5-flash-lite: $0.10/$0.40; gemini-2.5-flash: $0.30/$2.50 per 1M tokens.
            # Pricing verified: https://ai.google.dev/gemini-api/docs/pricing (2026-05-11)
            pricing={"input_per_1m": 0.10, "output_per_1m": 0.40, "free_tier": False},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
