"""
Abstract base class for all model implementations in effGen.

This module defines the interface that all model engines must implement,
ensuring consistent behavior across vLLM, Transformers, and API adapters.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Literal

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of supported model types."""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MLX = "mlx"
    MLX_VLM = "mlx_vlm"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    # Optional draft model for speculative decoding. Backends that
    # do not support speculative decoding should ignore this field.
    draft_model: Any = None
    # Reasoning model controls (OpenAI o-series). Both default to None
    # for full back-compat — non-reasoning models silently ignore them.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    max_reasoning_tokens: int | None = None
    # Gemini thinking controls. thinking_budget=None means field not sent;
    # 0 disables thinking; >0 sets token budget. include_thoughts=True
    # surfaces the reasoning trace in ModelResponse.metadata["thinking"].
    thinking_budget: int | None = None
    include_thoughts: bool = False
    # Gemini grounding. When True AND the model supports_grounding, Google
    # Search grounding is activated and grounding_chunks appear in metadata.
    grounding: bool = False
    # Anthropic extended thinking. Pass {"type": "enabled", "budget_tokens": N}.
    # None means the field is not sent (standard generation).
    thinking: dict | None = None
    # Structured-output controls. When set, backends that support a native
    # JSON/structured mode constrain generation accordingly. response_mime_type
    # (e.g. "application/json") asks the provider to emit raw JSON; response_schema
    # is an optional JSON-Schema dict the provider validates against where it can.
    # Both default to None so ordinary generation is unaffected.
    response_mime_type: str | None = None
    response_schema: dict | None = None

    def __post_init__(self) -> None:
        """Accept a bare string for ``stop_sequences``.

        Every consumer — the adapters, the local engines, the agent's own
        trimming — iterates this field, so a string given directly is walked
        character by character and cuts the text at the first single letter that
        matches. The OpenAI API accepts a bare string, so it is the shape a
        caller naturally reaches for; it becomes a one-element list here, once,
        rather than in each consumer.
        """
        from effgen.models._adapter_utils import normalize_stop_sequences

        self.stop_sequences = normalize_stop_sequences(self.stop_sequences)


@dataclass
class GenerationResult:
    """Result from a generation call.

    On a reasoning model, ``text`` can be empty even on a normal (non-error)
    return: the output-token budget is spent on hidden reasoning before any
    visible token is emitted. This is reported as ``finish_reason == "length"``
    and ``metadata["truncated"] is True`` — check either before trusting an
    empty ``text``/``str(result)``, or call through :class:`~effgen.core.agent.Agent`
    instead, which already raises a clear error in that case.
    """
    text: str
    tokens_used: int
    finish_reason: str
    model_name: str
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        """The generated text, so ``print(model.generate(...))`` shows the
        answer rather than the full dataclass repr — mirroring ``AgentResponse``.
        The unambiguous ``repr()`` still shows every field for debugging."""
        return self.text

    def _repr_html_(self) -> str:
        """Rich HTML card for Jupyter/IPython — mirrors ``AgentResponse`` so a
        raw ``load_model(...).generate(...)`` renders a tidy card (answer +
        model · time · tokens · cost) in a notebook, not a dataclass repr."""
        from effgen.ui import generation_result_html

        return generation_result_html(self)

    def _repr_markdown_(self) -> str:
        """Plain-markdown notebook view (answer + a small metric footer)."""
        from effgen.ui import generation_result_markdown

        return generation_result_markdown(self)


@dataclass
class TokenCount:
    """Token count information."""
    count: int
    model_name: str


def _stamp_latency(result: Any, elapsed_s: float) -> Any:
    """Fold per-call wall-clock latency + a truncation flag onto a
    ``GenerationResult``'s metadata.

    Adds ``latency_ms`` and ``duration_s`` (via ``setdefault``, so an engine that
    measures its own latency wins) so benchmarkers can read throughput straight off
    a raw ``generate()`` result. Also adds ``truncated`` (``finish_reason ==
    "length"``) so a caller working directly with ``model.generate()`` (no
    ``Agent`` in between) can detect a truncated/empty-from-truncation result
    without string-matching ``finish_reason`` itself, and ``tool_calls`` (empty
    for an engine that reports none) so the key is present on every result —
    including the local engines, which report tool calls as text rather than as
    a structured list. Non-``GenerationResult`` values pass through untouched.
    """
    if isinstance(result, GenerationResult):
        meta = result.metadata
        if meta is None:
            meta = {}
            result.metadata = meta
        meta.setdefault("latency_ms", round(elapsed_s * 1000.0, 1))
        meta.setdefault("duration_s", round(elapsed_s, 4))
        meta.setdefault("truncated", result.finish_reason == "length")
        meta.setdefault("tool_calls", [])
    return result


#: Serializes every read-modify-write of an adapter's cumulative ``total_cost``
#: and ``total_tokens``. One adapter commonly serves many agents at once — the
#: server hands the same instance to every request, and a thread pool of agents
#: shares one — and ``total = total + cost`` spread over a load, an add and a
#: store loses one call's money whenever two threads interleave across it. The
#: work under the lock is a few arithmetic operations, so a single process-wide
#: lock costs less than one per instance would.
_TOTALS_LOCK = threading.Lock()


def fold_call_totals(
    model: "BaseModel",
    cost: float | None = None,
    tokens: int | None = None,
) -> float:
    """Add one call's cost and tokens to *model*'s session totals.

    Args:
        model: The adapter the call was made on.
        cost: This call's USD cost, or ``None`` when the model publishes no
            price — an unpriced call leaves the total where it was rather than
            adding a zero.
        tokens: This call's total prompt+completion tokens, if known.

    Returns:
        The cumulative cost after this call, read under the same lock that
        wrote it, so the value reported alongside a call is the total that
        included it rather than one a concurrent call has since moved.
    """
    with _TOTALS_LOCK:
        if cost is not None:
            model.total_cost = getattr(model, "total_cost", 0.0) + cost
        if tokens:
            model.total_tokens = getattr(model, "total_tokens", 0) + tokens
        return getattr(model, "total_cost", 0.0)


def _stamp_cost(model: "BaseModel", result: Any) -> None:
    """Accumulate this call's cost onto the model instance's running total.

    Populates ``metadata["total_cost"]`` via presence check, so an adapter that
    already tracks its own cumulative total (OpenAI, Gemini, Anthropic) keeps
    its own bookkeeping untouched; every other adapter gets the same
    cumulative-cost field for free, derived from the ``cost_usd`` it already
    reports per call. Skipped when the result carries no ``cost_usd`` (e.g.
    local engines that do not price calls).

    **Tokens fold here too**, from the same per-call metadata. They used not to,
    so on the six adapters that rely on this wrapper — groq, cerebras, together,
    fireworks, replicate, hf_inference — ``model.total_cost`` was right while
    ``model.total_tokens`` never moved. The presence check is what keeps that
    from double-counting: the three adapters that fold their own totals also
    write ``total_cost`` into the metadata, so this returns before touching
    either counter for them.
    """
    if not isinstance(result, GenerationResult):
        return
    meta = result.metadata
    if meta is None or "total_cost" in meta:
        return
    cost = meta.get("cost_usd")
    if cost is None:
        return
    tokens = meta.get("total_tokens")
    if tokens is None:
        prompt = meta.get("prompt_tokens") or 0
        completion = meta.get("completion_tokens") or 0
        tokens = prompt + completion
    meta["total_cost"] = fold_call_totals(model, cost, tokens)


def clear_stream_usage(model: "BaseModel") -> None:
    """Drop any usage recorded by a previous streaming call on *model*.

    Called by a consumer immediately before it starts a stream, so that reading
    the usage afterwards returns this call's numbers or ``None`` — never the
    previous call's numbers for a stream that reported none.
    """
    try:
        model._last_stream_usage = None  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a model that rejects attributes still streams
        logger.debug("Could not clear stream usage", exc_info=True)


def get_stream_usage(model: "BaseModel") -> dict[str, Any] | None:
    """Return the usage of the most recent streaming call on *model*.

    The dict carries ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
    and ``cost_usd`` (``None`` when the model publishes no per-token price).
    Returns ``None`` when the call reported no usage — local engines and
    providers that omit usage from their stream.
    """
    usage = getattr(model, "_last_stream_usage", None)
    return usage if isinstance(usage, dict) else None


def record_stream_usage(
    model: "BaseModel",
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cost_usd: float | None = None,
) -> None:
    """Record the token counts and cost of the streaming call that just ended.

    ``generate()`` returns a ``GenerationResult`` whose metadata carries this
    data; ``generate_stream()`` has no return value, so an adapter that learns
    the real usage after the stream is exhausted records it here. The consumer
    reads it back with :func:`get_stream_usage`, which is what lets a streamed
    turn report its tokens and cost without a second billed call.

    Args:
        model: The adapter whose stream just ended.
        prompt_tokens: Input tokens the provider reported, when it did.
        completion_tokens: Output tokens the provider reported, when it did.
        cost_usd: What the stream cost, when the provider prices it.
    """
    if prompt_tokens is None and completion_tokens is None and cost_usd is None:
        return
    total: int | None = None
    if prompt_tokens is not None or completion_tokens is not None:
        total = int(prompt_tokens or 0) + int(completion_tokens or 0)
    try:
        model._last_stream_usage = {  # type: ignore[attr-defined]
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "cost_usd": cost_usd,
        }
    except Exception:  # noqa: BLE001 - usage accounting must not break streaming
        logger.debug("Could not record stream usage", exc_info=True)


def clear_stream_tool_calls(model: "BaseModel") -> None:
    """Drop any tool calls recorded by a previous streaming call on *model*.

    Called immediately before a stream starts, so reading the buffer afterwards
    returns this call's tool calls or an empty list — never the previous call's.
    """
    try:
        model._last_stream_tool_calls = []  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a model that rejects attributes still streams
        logger.debug("Could not clear stream tool calls", exc_info=True)


def get_stream_tool_calls(model: "BaseModel") -> list[dict[str, Any]]:
    """Return the tool calls the current or most recent stream declared.

    Each entry is in the documented ``metadata["tool_calls"]`` shape (see
    :func:`effgen.models._usage.tool_call_entry`). An adapter records a call as
    soon as its first delta arrives, so a consumer reading this mid-stream can
    tell that the turn is making a call before any text is committed as the
    answer. Returns ``[]`` when the stream declared none, or when the adapter
    does not record streamed calls (see :meth:`BaseModel.streams_tool_calls`).
    """
    calls = getattr(model, "_last_stream_tool_calls", None)
    return calls if isinstance(calls, list) else []


def record_stream_tool_calls(
    model: "BaseModel", calls: list[dict[str, Any]]
) -> None:
    """Record the tool calls a streaming call has declared so far.

    ``generate()`` returns them in ``GenerationResult.metadata["tool_calls"]``;
    ``generate_stream()`` has no return value, so an adapter records them here
    and the consumer reads them back with :func:`get_stream_tool_calls`. Called
    repeatedly as the arguments accumulate — each call replaces the buffer with
    the current state — and once more when the stream ends.

    Args:
        model: The adapter whose stream declared the calls.
        calls: The calls so far, in the documented ``tool_calls`` shape.
    """
    try:
        model._last_stream_tool_calls = list(calls)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - tool-call accounting must not break streaming
        logger.debug("Could not record stream tool calls", exc_info=True)


#: Engine types whose ``count_tokens`` runs against a local tokenizer — no
#: network call, so it is cheap enough to use for an after-the-fact estimate.
_LOCAL_ENGINE_TYPES = frozenset(
    {ModelType.TRANSFORMERS, ModelType.VLLM, ModelType.MLX, ModelType.MLX_VLM}
)


def _provider_of(model: "BaseModel") -> str | None:
    """Resolve the pricing-catalog provider name for *model*, or ``None``.

    Adapters name their provider either in ``get_metadata()`` or through their
    ``model_type``; local engines have no provider and return ``None``.
    """
    try:
        provider = (model.get_metadata() or {}).get("provider")
        if isinstance(provider, str) and provider:
            return provider
    except Exception:  # noqa: BLE001 - metadata is optional
        pass
    model_type = getattr(model, "model_type", None)
    value = getattr(model_type, "value", None)
    if isinstance(value, str) and value not in {t.value for t in _LOCAL_ENGINE_TYPES}:
        return value
    return None


def _price_estimate(
    model: "BaseModel", prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Price locally counted tokens, or return ``None`` for an unpriced model.

    Uses the same rate table the adapters bill against, and only when the
    catalog reports the model as priced — a free-tier or unpriced model returns
    ``None`` rather than a fabricated ``$0``.
    """
    provider = _provider_of(model)
    name = getattr(model, "model_name", None)
    if not provider or not name:
        return None
    try:
        from effgen.models._cost import _rate, pricing_status

        if pricing_status(provider, name) != "priced":
            return None
        input_rate, output_rate = _rate(provider, name)
    except Exception:  # noqa: BLE001 - pricing is optional; report unpriced
        return None
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


def estimate_stream_usage(
    model: "BaseModel", prompt_text: str, completion_text: str
) -> dict[str, Any]:
    """Estimate token counts for a stream whose backend reported no usage.

    Local engines are counted with their own tokenizer (offline and exact);
    anything else falls back to a four-characters-per-token approximation. The
    counts are priced at the model's catalog rate when it has one, so a stream
    the caller stopped reading early still reports a cost rather than nothing.
    The returned dict is shaped like :func:`get_stream_usage`'s and carries
    ``estimated: True`` so a caller can label the numbers as counted locally
    rather than reported by the provider.

    Args:
        model: The adapter the stream came from.
        prompt_text: The prompt that was sent.
        completion_text: The text the stream produced.

    Returns:
        The usage block, marked ``estimated`` so a reader knows the counts were
        derived locally.
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    if getattr(model, "model_type", None) in _LOCAL_ENGINE_TYPES:
        try:
            prompt_tokens = int(model.count_tokens(prompt_text).count)
            completion_tokens = int(model.count_tokens(completion_text).count)
        except Exception:  # noqa: BLE001 - fall through to the approximation
            prompt_tokens = completion_tokens = None
    if prompt_tokens is None or completion_tokens is None:
        prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
        completion_tokens = max(1, len(completion_text) // 4) if completion_text else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": _price_estimate(model, prompt_tokens, completion_tokens),
        "estimated": True,
    }


def accumulate_stream_cost(
    model: "BaseModel",
    cost: float | None,
    tokens: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    """Fold a completed streaming call's cost and tokens onto the model's totals.

    ``generate()``/``generate_batch()`` get their cumulative cost for free from
    ``_stamp_cost`` because they return a ``GenerationResult`` the wrapper can
    inspect. ``generate_stream()`` has no equivalent return value — it yields
    text chunks and the final usage is only known once the stream is exhausted
    — so every adapter that streams must call this once, after pricing the
    completed call, instead of writing ``self.total_cost += cost`` by hand.
    A ``None`` cost (no usage data — the caller never priced the call) leaves
    ``total_cost`` untouched. A real ``0.0`` (a genuine free-tier call that was
    priced and came back free) still folds in, matching what a non-streaming
    ``generate()`` call on the same free-tier adapter already reports via
    ``_stamp_cost``, so ``get_total_cost()`` reads ``0.0`` instead of leaving
    the attribute unset after a real, tracked call. ``tokens`` (the call's
    total prompt+completion tokens) folds onto ``total_tokens`` the same way a
    non-streaming call updates it, so a streamed turn's token count reaches the
    per-turn footer instead of reading zero. ``prompt_tokens``/
    ``completion_tokens``, when supplied, are also recorded as this call's own
    usage (see :func:`record_stream_usage`) so the caller can read the split and
    the cost for the turn it just streamed, not only the running totals.
    """
    fold_call_totals(model, cost, tokens)
    if prompt_tokens is not None or completion_tokens is not None:
        record_stream_usage(model, prompt_tokens, completion_tokens, cost)


def _warn_if_silently_empty(model: "BaseModel", result: Any) -> None:
    """Log a warning when a reasoning model's raw ``generate()`` call returns
    empty text because its output budget was spent on hidden reasoning before
    any visible token (``finish_reason == "length"``).

    :class:`~effgen.core.agent.Agent` already detects and escalates this case;
    a caller using ``model.generate()`` directly has no such safety net, so an
    empty ``GenerationResult`` would otherwise look like a working call that
    produced nothing.

    An adapter that saw the provider's reasoning chain reports the same turn in
    more detail (``metadata["reasoning_only"]``, naming the cap and the reasoning
    budget), so this stands down rather than saying it a second time.
    """
    if not isinstance(result, GenerationResult):
        return
    if result.text or result.finish_reason != "length":
        return
    if (result.metadata or {}).get("reasoning_only"):
        return
    from ._adapter_utils import needs_reasoning_headroom
    if not needs_reasoning_headroom(model):
        return
    logger.warning(
        "%s returned empty text after exhausting its token budget on internal "
        "reasoning (finish_reason='length'). Increase max_tokens, or check "
        "metadata['truncated'] before trusting an empty result.",
        getattr(model, "model_name", model.__class__.__name__),
    )


def _preflight_budget_check(model: "BaseModel") -> None:
    """Refuse to start a call when a configured budget is already at or over its cap.

    Runs before the provider call is made, so a call refused here never reaches
    the network and is never billed — unlike the check inside
    ``CostTracker.record()``, which runs after a call's tokens are already known
    and its cost already persisted. A tracker or budget-config read failure is
    swallowed (best-effort; the post-spend check still applies as a backstop).
    """
    try:
        from effgen.models._cost import CostTracker
    except ImportError:
        return
    provider = getattr(getattr(model, "model_type", None), "value", "") or ""
    model_name = getattr(model, "model_name", "") or ""
    CostTracker.get().check_preflight(provider, model_name)


def _redact_credentials(exc: BaseException) -> None:
    """Strip secret material from *exc* and everything it chains to.

    Applied at the boundary every engine call leaves through, so a provider
    SDK's 401 body — which commonly quotes the key that was submitted — cannot
    reach a traceback, a log record or a crash dump. The typed error the caller
    reads is already redacted; this covers the ``__cause__`` under it.
    """
    try:
        from effgen.models.errors import scrub_exception

        scrub_exception(exc)
    except Exception:  # noqa: BLE001 - redaction must not replace the error
        pass


def _redacted_call(func):
    """Wrap a method so any error it raises leaves with its secrets stripped."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except BaseException as exc:
            _redact_credentials(exc)
            raise
    wrapper.__effgen_timed__ = True
    return wrapper


def _timed_generate(func):
    """Wrap a ``generate`` method with a pre-call budget check and call latency."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _preflight_budget_check(self)
        start = time.perf_counter()
        try:
            result = func(self, *args, **kwargs)
        except BaseException as exc:
            _redact_credentials(exc)
            raise
        result = _stamp_latency(result, time.perf_counter() - start)
        _stamp_cost(self, result)
        _warn_if_silently_empty(self, result)
        return result
    wrapper.__effgen_timed__ = True
    return wrapper


def _timed_generate_batch(func):
    """Wrap a ``generate_batch`` method with a pre-call budget check so each
    result carries call latency.

    The whole batch shares one wall-clock measurement (the per-item split isn't
    knowable here); each result gets it via ``setdefault`` so an engine that
    records true per-item latency keeps its own value.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _preflight_budget_check(self)
        start = time.perf_counter()
        try:
            results = func(self, *args, **kwargs)
        except BaseException as exc:
            _redact_credentials(exc)
            raise
        elapsed = time.perf_counter() - start
        if isinstance(results, list):
            for r in results:
                _stamp_latency(r, elapsed)
                _stamp_cost(self, r)
                _warn_if_silently_empty(self, r)
        return results
    wrapper.__effgen_timed__ = True
    return wrapper


def _redacting_iter(iterator):
    """Yield from *iterator*, redacting any error it raises mid-stream."""
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except BaseException as exc:
            _redact_credentials(exc)
            raise


def _budget_gated_stream(func):
    """Wrap a ``generate_stream`` method with a pre-call budget check.

    The check runs synchronously when the caller invokes ``generate_stream``
    (not on first ``next()``), so a refusal happens before any token request
    reaches the provider. A stream that fails part-way through — the provider
    dropping the connection, a 401 arriving on the first frame — has its error
    redacted the same way a non-streaming call's does.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _preflight_budget_check(self)
        try:
            iterator = iter(func(self, *args, **kwargs))
        except BaseException as exc:
            _redact_credentials(exc)
            raise
        return _redacting_iter(iterator)
    wrapper.__effgen_timed__ = True
    return wrapper


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.

    This class defines the interface that all model engines must implement,
    including vLLM, Transformers, OpenAI, Anthropic, and Gemini adapters.

    Attributes:
        model_name: The name or identifier of the model
        model_type: The type of model engine
        context_length: Maximum context length supported by the model
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-instrument each engine's generation methods with a budget
        pre-check and call timing.

        Every concrete engine that defines ``generate``/``generate_batch``/
        ``generate_stream`` gets a pre-call check against any configured
        daily/monthly budget (refusing before the provider call is made once
        the period is already at or over its cap) and, for ``generate``/
        ``generate_batch``, its result(s) stamped with ``latency_ms``/
        ``duration_s`` (see ``_stamp_latency``) — without each engine repeating
        the bookkeeping. Abstract or already-wrapped methods are left alone, so
        this is safe across the engine hierarchy.

        ``load`` and ``generate_with_tools`` are wrapped too, with redaction
        only: a credential rejected at load time reaches a traceback the same
        way one rejected mid-call does.
        """
        super().__init_subclass__(**kwargs)
        for name, wrapper in (("generate", _timed_generate),
                              ("generate_batch", _timed_generate_batch),
                              ("generate_stream", _budget_gated_stream),
                              ("load", _redacted_call),
                              ("generate_with_tools", _redacted_call)):
            func = cls.__dict__.get(name)
            if (
                func is not None
                and callable(func)
                and not getattr(func, "__isabstractmethod__", False)
                and not getattr(func, "__effgen_timed__", False)
            ):
                setattr(cls, name, wrapper(func))

    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        context_length: int | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the base model.

        Args:
            model_name: Name or identifier of the model
            model_type: Type of model engine
            context_length: Maximum context length (auto-detected if None)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self._context_length = context_length
        self._is_loaded = False
        self._metadata: dict[str, Any] = {}
        # Session totals, folded by ``fold_call_totals``. Declared here so both
        # exist from construction: an adapter whose folds all went through
        # ``_stamp_cost`` never had ``total_tokens`` assigned at all, and
        # reading it raised AttributeError rather than reporting zero.
        self.total_cost: float = 0.0
        self.total_tokens: int = 0

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        This method should handle all initialization logic including:
        - Model weight loading
        - Device allocation
        - Tokenizer initialization
        - Configuration setup

        Raises:
            RuntimeError: If model loading fails
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any
    ) -> GenerationResult:
        """
        Generate text from a prompt synchronously.

        Args:
            prompt: Input text prompt
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult containing generated text and metadata

        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid or exceeds context length
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        Generate text from a prompt with token-by-token streaming.

        Args:
            prompt: Input text prompt
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Yields:
            str: Individual tokens or chunks of generated text

        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid or exceeds context length
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> TokenCount:
        """
        Count the number of tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object with token count and model name

        Raises:
            RuntimeError: If tokenization fails
        """
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """
        Get the maximum context length supported by the model.

        Returns:
            int: Maximum number of tokens the model can handle
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory and free resources.

        This method should:
        - Free GPU/CPU memory
        - Close any open connections
        - Clean up temporary files
        """
        pass

    def supports_tool_calling(self) -> bool:
        """
        Check if the model supports native tool/function calling.

        Native tool calling uses the model's built-in tool call format
        (e.g., chat template ``tools`` parameter, API tool_calls) instead
        of parsing free-text ReAct output.

        Returns:
            bool: True if native tool calling is supported.
        """
        return False

    def tool_call_support(self) -> str:
        """Report *how* the model receives tool definitions.

        Two mechanisms hide behind the single boolean of
        :meth:`supports_tool_calling`, and they behave differently:

        - ``"api"`` — the provider accepts tool definitions as a request
          parameter and returns any tool call as structured data. Whether a
          call is emitted is decided by the provider's tool-calling layer.
        - ``"template"`` — the definitions are rendered into the prompt by a
          local chat template. Nothing enforces the format; whether the model
          emits a call is up to the model.
        - ``"none"`` — no native tool calling. The ReAct text protocol is the
          only way to reach a tool.

        The default derives the value from :meth:`supports_tool_calling`, so a
        subclass that only implements the boolean keeps working unchanged. The
        local chat-template engines override it.

        Returns:
            str: One of ``"api"``, ``"template"`` or ``"none"``.
        """
        return "api" if self.supports_tool_calling() else "none"

    def supports_forced_tool_call(self) -> bool:
        """Whether a turn can be sent that *requires* the model to call a tool.

        Offering tools and requiring one are different capabilities. Every
        adapter that takes tool definitions can offer them; only a request layer
        that enforces the choice can require one, and a model asked to do
        something its provider does not implement fails the whole turn rather
        than answering it. So a caller with a reason to force a call — the agent
        loop after a model said it would use a tool and then did not — has to be
        able to ask first.

        The default is ``False``: an adapter advertises this only once its
        request shaping is known to carry the constraint through, so an adapter
        that has not considered it degrades to asking in words rather than
        sending a request the provider will reject.

        Returns:
            bool: True if ``tool_choice="required"`` reaches the provider in a
            form it honours.
        """
        return False

    # ------------------------------------------------------------------
    # Multi-turn tool loop
    # ------------------------------------------------------------------
    #
    # ``metadata["tool_calls"]`` has one shape on every adapter, so *reading* a
    # call is portable. Re-submitting the turn was not: the loop in the examples
    # appended ``metadata["message"]`` — written by the OpenAI adapter and by no
    # other — and a ``{"role": "tool", ...}`` message, which is the OpenAI wire
    # format and not Gemini's or Anthropic's. The two methods below make the
    # loop portable by giving each adapter a way to build its *own* provider's
    # shape, so a caller driving ``chat()``/``generate_with_tools()`` by hand
    # across providers writes one loop.
    #
    # ``Agent`` already does this internally and remains the easier route; these
    # are for a caller who wants the raw loop.

    def build_assistant_message(self, result: "GenerationResult") -> dict[str, Any]:
        """Return the assistant turn of *result*, in this provider's wire shape.

        Append it to the conversation before the tool results, so the model sees
        the call it made.

        The default is the OpenAI-compatible shape, which groq, together,
        fireworks, cerebras, hf and any OpenAI-protocol endpoint also speak. A
        provider-native message the adapter kept (``metadata["message"]``) is
        preferred when present, because it round-trips fields the uniform shape
        does not carry.

        Args:
            result: The turn to re-submit.

        Returns:
            One message dict for this provider's conversation format.
        """
        metadata = result.metadata or {}
        native = metadata.get("message")
        if isinstance(native, dict):
            return native
        message: dict[str, Any] = {"role": "assistant", "content": result.text or None}
        calls = metadata.get("tool_calls")
        if calls:
            message["tool_calls"] = calls
        return message

    def build_tool_result_message(
        self, call_id: str, name: str, content: str
    ) -> dict[str, Any]:
        """Return one tool's result, in this provider's wire shape.

        Append it after the assistant message, once per call the turn made.

        Args:
            call_id: The ``id`` of the call being answered, from
                ``metadata["tool_calls"][i]["id"]``.
            name: The tool's name — unused in the OpenAI shape, required by
                Gemini's and carried here so one call site serves every
                provider.
            content: What the tool returned, as text.

        Returns:
            One message dict for this provider's conversation format.
        """
        _ = name
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    def streams_tool_calls(self) -> bool:
        """Report whether ``generate_stream`` records the turn's tool calls.

        An adapter that returns ``True`` writes every call the stream declares
        into the buffer :func:`get_stream_tool_calls` reads, as soon as the
        call's first delta arrives. That is what lets a caller stream a turn's
        assistant text while still dispatching the calls the same turn makes;
        an adapter that returns ``False`` drops them, so a tool loop has to run
        the turn without streaming.

        Returns:
            bool: True when streamed tool calls are recorded.
        """
        return False

    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.

        Returns:
            bool: True if model is loaded and ready for inference
        """
        return self._is_loaded

    def get_total_cost(self) -> float:
        """Cumulative cost (USD) charged to this model instance.

        Accumulates each call's ``cost_usd`` since creation or the last
        :meth:`reset_cost`; local/unpriced engines stay at ``0.0``.
        """
        return getattr(self, "total_cost", 0.0)

    def reset_cost(self) -> None:
        """Reset the cumulative cost counter (see :meth:`get_total_cost`) to zero."""
        self.total_cost = 0.0

    def get_metadata(self) -> dict[str, Any]:
        """
        Get model metadata and information.

        Returns:
            Dict containing model information such as:
            - Model architecture
            - Parameter count
            - Quantization details
            - Device allocation
            - Memory usage
        """
        return self._metadata

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that a prompt is within context length limits.

        Some providers publish no context window for a model (Fireworks reports
        ``contextLength: 0`` for a custom deployment, for instance). An unknown
        window is not a zero-token window, so the length check is skipped rather
        than rejecting every prompt; the provider still enforces its own limit
        and reports a truncated result.

        Args:
            prompt: Prompt to validate

        Returns:
            bool: True if prompt is valid

        Raises:
            ValueError: If prompt exceeds a known context length
        """
        token_count = self.count_tokens(prompt)
        max_length = self.get_context_length()

        if not max_length or max_length <= 0:
            return True

        if token_count.count > max_length:
            raise ValueError(
                f"Prompt length ({token_count.count} tokens) exceeds "
                f"model context length ({max_length} tokens)"
            )

        return True

    def __enter__(self) -> BaseModel:
        """Context manager entry."""
        if not self._is_loaded:
            self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.unload()

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"type={self.model_type.value}, "
            f"loaded={self._is_loaded})"
        )


class BatchModel(BaseModel):
    """
    Extended base class for models that support batch processing.

    This class extends BaseModel with batch generation capabilities,
    useful for high-throughput scenarios.
    """

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        **kwargs: Any
    ) -> list[GenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects, one per prompt

        Raises:
            RuntimeError: If batch generation fails
            ValueError: If any prompt is invalid
        """
        pass

    def get_max_batch_size(self) -> int:
        """
        Get the maximum batch size supported.

        Returns:
            int: Maximum number of prompts that can be processed in one batch
        """
        return 1  # Default to no batching


class FunctionCallingModel(BaseModel):
    """
    Extended base class for models that support function/tool calling.

    This class extends BaseModel with function calling capabilities,
    primarily used by API adapters (OpenAI, Anthropic, Gemini).
    """

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs: Any
    ) -> GenerationResult:
        """
        Generate text with tool/function calling support.

        Args:
            prompt: Input text prompt
            tools: List of tool definitions in OpenAI function format
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with potential tool calls in metadata

        Raises:
            RuntimeError: If generation fails
            ValueError: If tools are malformed
        """
        pass

    @abstractmethod
    def supports_function_calling(self) -> bool:
        """
        Check if the model supports function calling.

        Returns:
            bool: True if function calling is supported
        """
        pass
