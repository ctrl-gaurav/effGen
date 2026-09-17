"""Shared helpers for provider adapter consistency.

Provider SDKs disagree on how they report two things that effGen treats as a
uniform contract:

* **finish reason** — OpenAI-style adapters return ``"stop"``/``"length"``/
  ``"tool_calls"``; Anthropic returns ``"end_turn"``/``"max_tokens"``/
  ``"tool_use"``; Gemini returns an enum whose ``str()`` is ``"FinishReason.STOP"``.
  :func:`normalize_finish_reason` maps all of them to one canonical set so
  downstream code (the agent loop, cost/observability) sees the same tokens
  regardless of provider.

* **error reporting** — adapters historically raised bare
  ``RuntimeError(f"... failed: {exc}")`` with the unredacted SDK message and no
  machine-readable context. :func:`build_error_context` and
  :func:`provider_runtime_error` produce a consistent, **redacted** error whose
  ``.error_context`` carries ``{provider, model, request_type, retry_status,
  remediation}``. :func:`not_loaded_error` gives the same shape to the guard
  every adapter runs when a call arrives before ``load()``.

* **per-call sampling overrides** — a caller may pass ``seed=``/``temperature=``
  and the rest of the sampling settings straight to ``generate()`` instead of
  building a ``GenerationConfig``. :func:`merge_call_overrides` folds those
  keywords into the call's config and takes them out of the keyword dict, so a
  backend whose own entry point has no parameter of that name never receives one.

* **an absent local backend** — the local engines need PyTorch and the cloud
  providers do not, so an install without it reaches the user as
  ``No module named 'torch'`` with nothing to act on.
  :func:`missing_torch_error` names the engine, the package and the way to get
  it, and is raised both when an engine module is imported and when a load is
  requested.

These are internal helpers (no public API surface change).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import replace
from typing import Any

from .errors import (
    DEVICE_MEMORY_SIGNALS,
    RETRY_NON_RETRYABLE,
    RETRY_RATE_LIMITED,
    RETRY_WILL_RETRY,
    classify_provider_error,
    context_overflow_hint,
    error_context_dict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
# tiktoken ships no BPE data: the first call for an encoding downloads it from
# openaipublic.blob.core.windows.net into a cache under the temporary directory.
# On a machine with no route to that host — an air-gapped deployment, a runner
# behind a proxy that does not allow it, a container with an empty cache — the
# download raises ``requests.ConnectionError``. That exception used to travel
# out of ``count_tokens`` and, through the pre-flight prompt check, out of every
# ``generate()`` call: a request that the provider would have answered failed
# before it was sent, reporting a name-resolution failure for a host that has
# nothing to do with the provider. A token count is an estimate, so an encoding
# that cannot be loaded degrades to the character heuristic and says so once.

#: Encodings already resolved, by encoding name. ``None`` records one that could
#: not be loaded, so the download is attempted once rather than per call.
_bpe_encodings: dict[str, Any] = {}

#: Encoding names whose unavailability has been reported.
_bpe_unavailable_warned: set[str] = set()

#: Characters per token in the fallback estimate. English prose and code both sit
#: near this ratio for the BPE vocabularies the cloud providers use.
_CHARS_PER_TOKEN = 4

#: Token counts already made, keyed by encoding and text, oldest first. A run
#: measures the prompt it is about to send more than once — the context budget
#: checks it, the adapter checks it against the context window, the calibration
#: counts it again once the provider has answered — and the prompt grows with
#: every step, so encoding it afresh each time made a run's counting grow with
#: the square of its length. An encoding's count of a text never changes, so a
#: text counted recently is answered from here.
_token_counts: OrderedDict[tuple[Any, str], int] = OrderedDict()
_token_counts_lock = threading.Lock()
_token_counts_chars = 0
#: At most this many counts are kept, holding at most this many characters of
#: text between them; the least recently used go first. Every message of every
#: conversation in flight is a separate text, short ones included, so the entry
#: bound is set well above what many concurrent agents hold at once: at 1,024,
#: thirty-two agents with ordinary histories evicted counts they were about to
#: ask for again and re-encoded ten to twenty-six texts per model call. The
#: character bound is what limits memory.
_TOKEN_COUNTS_MAX_ENTRIES = 16_384
_TOKEN_COUNTS_MAX_CHARS = 4_000_000
#: A text shorter than this is encoded every time rather than remembered.
#: Measured on this machine, remembering costs about 0.2 us and encoding a text
#: of ten to twenty-five characters costs 1.9-4.5 us, so remembering is ten to
#: twenty times cheaper even for the shortest text a prompt carries — and the
#: encoding holds the interpreter while it runs, where the lookup barely does,
#: which is what decides throughput when many agents count prompts at once. The
#: short texts are also the repeated ones: a tool's name, a tool call's
#: arguments, a one-line result. Nothing is remembered below zero characters, so
#: every count is; raise this to put a floor back.
_TOKEN_COUNTS_MIN_CHARS = 0
#: How many texts were encoded, and how many counts were answered from memory.
_token_count_stats: dict[str, int] = {"encoded": 0, "reused": 0}


def get_bpe_encoding(name: str = "cl100k_base", *, model: str | None = None) -> Any:
    """Return the tiktoken encoding for *model* (or *name*), or ``None``.

    ``None`` means the encoding is unavailable on this machine — tiktoken is not
    installed, or its BPE data is neither cached nor reachable. The first time
    that happens for an encoding it is reported at INFO with the reason; after
    that the answer is remembered, so a long run neither repeats the message nor
    retries the download on every call.
    """
    key = model or name
    if key in _bpe_encodings:
        return _bpe_encodings[key]

    encoding = None
    try:
        import tiktoken

        if model is not None:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # A model newer than the installed tiktoken: its own vocabulary
                # is unknown, and the current OpenAI-family default is the
                # closest available estimate.
                encoding = tiktoken.get_encoding(name)
        else:
            encoding = tiktoken.get_encoding(name)
    except ImportError:
        if key not in _bpe_unavailable_warned:
            _bpe_unavailable_warned.add(key)
            logger.info(
                "tiktoken is not installed, so token counts are estimated from text "
                "length. Install it for exact counts: pip install tiktoken"
            )
    except Exception as exc:  # noqa: BLE001 - any failure to obtain the data degrades the same way
        if key not in _bpe_unavailable_warned:
            _bpe_unavailable_warned.add(key)
            logger.info(
                "tiktoken could not load the '%s' encoding (%s), so token counts are "
                "estimated from text length. Pre-populate the tiktoken cache to get "
                "exact counts on a machine with no route to its data host.",
                key, type(exc).__name__,
            )

    _bpe_encodings[key] = encoding
    return encoding


def estimate_tokens(text: str, *, name: str = "cl100k_base", model: str | None = None) -> int:
    """Return a token count for *text*, exact when the BPE encoding is available.

    Falls back to a character-length estimate when it is not, so a caller always
    gets a number. Empty text is zero tokens; any non-empty text is at least one.
    The count of a text is remembered, so counting the same text again with the
    same encoding does not encode it again.

    Args:
        text: The text to count.
        name: The BPE encoding to use when no *model* is given.
        model: A model id whose own encoding is preferred over *name*.
    """
    global _token_counts_chars
    if not text:
        return 0
    encoding = get_bpe_encoding(name, model=model)
    if encoding is None:
        return max(1, len(text) // _CHARS_PER_TOKEN)
    if len(text) < _TOKEN_COUNTS_MIN_CHARS:
        return _encoded_count(encoding, text)
    key = (encoding, text)
    with _token_counts_lock:
        count = _token_counts.get(key)
        if count is not None:
            _token_counts.move_to_end(key)
            _token_count_stats["reused"] += 1
            return count
    count = _encoded_count(encoding, text)
    if len(text) <= _TOKEN_COUNTS_MAX_CHARS:
        with _token_counts_lock:
            if key not in _token_counts:
                _token_counts[key] = count
                _token_counts_chars += len(text)
                while (
                    len(_token_counts) > _TOKEN_COUNTS_MAX_ENTRIES
                    or _token_counts_chars > _TOKEN_COUNTS_MAX_CHARS
                ):
                    (_, dropped), _ = _token_counts.popitem(last=False)
                    _token_counts_chars -= len(dropped)
    return count


def _encoded_count(encoding: Any, text: str) -> int:
    """Encode *text* and return its token count, or the character estimate if the BPE refuses it."""
    with _token_counts_lock:
        _token_count_stats["encoded"] += 1
    try:
        return max(1, len(encoding.encode(text)))
    except Exception:  # noqa: BLE001 - a surrogate or control character the BPE rejects
        return max(1, len(text) // _CHARS_PER_TOKEN)


def missing_torch_error(engine: str) -> ImportError:
    """Return the error raised when the local *engine* has no PyTorch to run on.

    Args:
        engine: Name of the local engine that needs it (``"transformers"``,
            ``"vllm"``).

    Returns:
        An :class:`ImportError` naming the package, the install command and the
        cloud alternative that needs no local engine.
    """
    return ImportError(
        f"PyTorch is not installed, so the local '{engine}' engine cannot load a "
        "model. Install it with: pip install torch (see docs/installation.md for "
        "the build that matches your CUDA driver). A cloud model needs no local "
        'engine — for example: effgen run "..." -m openai:gpt-5-nano.'
    )


# Probe used to decide whether a local chat template renders tool definitions.
# The name is deliberately unusual so finding it in the rendered prompt means the
# template printed *this* definition rather than something it already contained.
TOOL_PROBE_NAME = "effgen_probe_tool"
_TOOL_PROBE = [
    {
        "type": "function",
        "function": {
            "name": TOOL_PROBE_NAME,
            "description": "capability probe",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_TOOL_PROBE_MESSAGES = [{"role": "user", "content": "test"}]


def chat_template_renders_tools(tokenizer: Any) -> bool:
    """Report whether *tokenizer*'s chat template puts tool definitions in the prompt.

    Accepting a ``tools`` keyword is not the same as using it: several templates
    (gemma-2, Phi-3.5) take the argument and discard it, so a model driven
    through them never sees the tools at all. Rendering the same messages twice —
    once plain, once with one probe tool — separates the two cases: a template
    that uses the definitions produces different text containing the probe
    tool's name.

    Args:
        tokenizer: Any object exposing ``apply_chat_template``.

    Returns:
        bool: True when the tools rendering differs from the plain one *and*
        names the probe tool. A tokenizer without a usable chat template, or one
        whose template raises, returns False.
    """
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return False
    try:
        plain = tokenizer.apply_chat_template(
            _TOOL_PROBE_MESSAGES, tokenize=False, add_generation_prompt=True
        )
        with_tools = tokenizer.apply_chat_template(
            _TOOL_PROBE_MESSAGES,
            tools=_TOOL_PROBE,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001 - a capability probe never breaks a load
        return False
    if plain is None or with_tools is None:
        return False
    rendered = str(with_tools)
    return rendered != str(plain) and TOOL_PROBE_NAME in rendered


# Canonical finish reasons. Keep this set small and OpenAI-flavoured because
# that is what the agent loop and the bulk of adapters already emit.
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"
FINISH_CONTENT_FILTER = "content_filter"
FINISH_ERROR = "error"
FINISH_UNKNOWN = "unknown"

CANONICAL_FINISH_REASONS = frozenset(
    {
        FINISH_STOP,
        FINISH_LENGTH,
        FINISH_TOOL_CALLS,
        FINISH_CONTENT_FILTER,
        FINISH_ERROR,
        FINISH_UNKNOWN,
    }
)

# Raw provider value -> canonical. Keys are lowercased and stripped of any
# ``finishreason.`` / ``stopreason.`` enum prefix before lookup.
_FINISH_REASON_MAP: dict[str, str] = {
    # OpenAI / OpenAI-compatible (groq, cerebras, together, fireworks, hf)
    "stop": FINISH_STOP,
    "length": FINISH_LENGTH,
    "tool_calls": FINISH_TOOL_CALLS,
    "function_call": FINISH_TOOL_CALLS,
    "content_filter": FINISH_CONTENT_FILTER,
    "eos": FINISH_STOP,
    "eos_token": FINISH_STOP,
    "complete": FINISH_STOP,
    "completed": FINISH_STOP,
    # Anthropic stop_reason
    "end_turn": FINISH_STOP,
    "stop_sequence": FINISH_STOP,
    "max_tokens": FINISH_LENGTH,
    "tool_use": FINISH_TOOL_CALLS,
    "pause_turn": FINISH_STOP,
    "refusal": FINISH_CONTENT_FILTER,
    "model_context_window_exceeded": FINISH_LENGTH,
    # Gemini FinishReason (str(enum) -> "finishreason.stop")
    "max_tokens_reached": FINISH_LENGTH,
    "safety": FINISH_CONTENT_FILTER,
    "recitation": FINISH_CONTENT_FILTER,
    "blocklist": FINISH_CONTENT_FILTER,
    "prohibited_content": FINISH_CONTENT_FILTER,
    "spii": FINISH_CONTENT_FILTER,
    "image_safety": FINISH_CONTENT_FILTER,
    "malformed_function_call": FINISH_TOOL_CALLS,
    "unexpected_tool_call": FINISH_TOOL_CALLS,
    "other": FINISH_UNKNOWN,
    "finish_reason_unspecified": FINISH_UNKNOWN,
    "unspecified": FINISH_UNKNOWN,
    # error / cancellation markers used internally
    "error": FINISH_ERROR,
}

# Numeric Gemini FinishReason enum values (google-genai), in case a bare int
# leaks through instead of the named enum.
_GEMINI_FINISH_INT_MAP: dict[int, str] = {
    0: FINISH_UNKNOWN,  # FINISH_REASON_UNSPECIFIED
    1: FINISH_STOP,  # STOP
    2: FINISH_LENGTH,  # MAX_TOKENS
    3: FINISH_CONTENT_FILTER,  # SAFETY
    4: FINISH_CONTENT_FILTER,  # RECITATION
    5: FINISH_UNKNOWN,  # OTHER
    6: FINISH_CONTENT_FILTER,  # BLOCKLIST
    7: FINISH_CONTENT_FILTER,  # PROHIBITED_CONTENT
    8: FINISH_CONTENT_FILTER,  # SPII
    9: FINISH_TOOL_CALLS,  # MALFORMED_FUNCTION_CALL
    10: FINISH_CONTENT_FILTER,  # IMAGE_SAFETY
}


def normalize_finish_reason(raw: Any, *, default: str = FINISH_STOP) -> str:
    """Map any provider finish/stop reason to a canonical lowercase token.

    The canonical set is :data:`CANONICAL_FINISH_REASONS`. Unknown but
    non-empty values are returned lowercased+stripped (so nothing is silently
    lost), while ``None``/empty falls back to ``default`` (a completed
    generation almost always means a normal stop).

    Args:
        raw: The provider's raw finish reason (str, enum, int, or None).
        default: Canonical value to use when ``raw`` is missing/empty.

    Returns:
        A canonical finish-reason string.
    """
    if raw is None:
        return default

    # Enum instances expose ``.name`` (e.g. google-genai FinishReason.STOP).
    name = getattr(raw, "name", None)
    if isinstance(name, str) and name:
        key = name
    elif isinstance(raw, bool):  # guard: bool is an int subclass
        return default
    elif isinstance(raw, int):
        return _GEMINI_FINISH_INT_MAP.get(raw, FINISH_UNKNOWN)
    else:
        key = str(raw)

    key = key.strip().lower()
    if not key:
        return default

    # Strip enum-repr prefixes like "finishreason." / "stopreason.".
    if "." in key:
        key = key.rsplit(".", 1)[-1]

    # A value that was only punctuation/prefix (e.g. "." or "stopreason.")
    # collapses to empty here; never surface an empty finish reason.
    if not key:
        return default

    return _FINISH_REASON_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Tool-request shaping
# ---------------------------------------------------------------------------


def apply_tool_request(
    request_params: dict[str, Any],
    tools: Any,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Put the tool definitions on an OpenAI-shaped request, or leave it alone.

    One helper for both ``generate()`` and ``generate_stream()``, because the
    two used to shape the request differently: the non-streaming path normalized
    the definitions and set ``tool_choice="auto"`` behind the catalog's
    ``supports_native_tools`` gate, while the streaming path forwarded the
    caller's ``tools=`` keyword through ``request_params.update(kwargs)``
    untouched — no gate and no ``tool_choice``. The same agent, the same turn and
    the same tools therefore produced a slightly different request depending on
    whether the turn streamed.

    The gate and ``tool_choice`` are one decision, not two: the gate decides
    which models are offered definitions at all, so copying only the
    ``tool_choice`` half would change which models receive them. Both live here.

    Args:
        request_params: The request being assembled; mutated in place.
        tools: Tool definitions — OpenAI-format dicts or effGen tool objects.
        record: The model's catalog record, or ``None`` when it is unknown.

    Returns:
        ``request_params``, for chaining.
    """
    if not tools or not (record or {}).get("supports_native_tools", False):
        return request_params
    normalized = []
    for tool in tools:
        if isinstance(tool, dict):
            normalized.append(
                tool if "type" in tool else {"type": "function", "function": tool}
            )
        else:
            normalized.append(
                {"type": "function", "function": tool.metadata.to_json_schema()}
            )
    request_params["tools"] = normalized
    request_params["tool_choice"] = "auto"
    return request_params


# ---------------------------------------------------------------------------
# generate_with_tools argument order
# ---------------------------------------------------------------------------


def normalize_tools_call_args(
    config: Any, messages: Any
) -> tuple[Any, Any]:
    """Return ``(config, messages)`` however the two were positioned.

    ``generate_with_tools`` takes ``config`` third on most adapters and took
    ``messages`` third on groq, together and fireworks. A reader of one
    signature writing ``adapter.generate_with_tools(prompt, tools, config)``
    against the other put a :class:`GenerationConfig` in the ``messages`` slot,
    and the provider SDK failed on it several layers down ("Object of type
    GenerationConfig is not JSON serializable"). The documented order is now
    ``config`` third everywhere; this reader keeps the other spelling working,
    so no caller of either shape breaks.

    The two are told apart by type, not by position: a conversation is a list
    of message mappings and a config is not, so neither can be mistaken for the
    other.

    Args:
        config: Whatever arrived in the ``config`` parameter.
        messages: Whatever arrived in the ``messages`` parameter.

    Returns:
        The pair in canonical order.
    """
    config_is_messages = isinstance(config, list)
    messages_is_config = messages is not None and not isinstance(messages, list)
    if config_is_messages or messages_is_config:
        return messages, config
    return config, messages


# ---------------------------------------------------------------------------
# Output-token budgeting
# ---------------------------------------------------------------------------

# Reasoning models (OpenAI gpt-5 family, o-series) spend part of their output
# budget on hidden internal reasoning before emitting any visible text. A small
# default budget can be entirely consumed by that reasoning, leaving zero output
# tokens and a "length" finish reason — an empty result the caller was still
# billed for. These families need a larger default and an escalation path. The
# name prefixes catch models whose adapter does not flag ``_is_reasoning_model``
# but still reason internally (notably gpt-5*, which the catalog lists as a
# non-reasoning chat model yet which burns output budget on reasoning).
_REASONING_NAME_PREFIXES = ("gpt-5", "o1", "o3", "o4")

#: Models already reported as answering "yes" from the name list alone, so the
#: line below is logged once per model rather than on every call.
_name_prefix_reasoning_reported: set[str] = set()


def needs_reasoning_headroom(model: Any) -> bool:
    """Return True if *model* spends output budget on hidden reasoning tokens.

    Such models can return an empty, billed result when ``max_tokens`` is too
    small (the budget is consumed by reasoning before any visible token), so the
    agent gives them a larger default budget and treats a ``"length"``-truncated
    empty as truncation rather than a retryable empty response.

    Three sources answer, in order, and the first that says yes decides:

    1. the adapter's own declaration (``_is_reasoning_model``, read from the
       provider catalog), which is the same statement that decides whether
       ``reasoning_effort`` travels and whether stop sequences are sent;
    2. a local model's own chat template, which is that model's statement about
       itself;
    3. the name list, which is a guess and the only source here that reads a
       model id. It fires only when the two declarations above said nothing, and
       it says so in the log, naming the model — a budget chosen from a name
       while the adapter declares the opposite is a disagreement a reader can
       act on, and it used to be silent.
    """
    if getattr(model, "_is_reasoning_model", False):
        return True
    if _local_template_reasons(model):
        return True
    name = (getattr(model, "model_name", "") or "").lower()
    name = name.split(":", 1)[-1]  # drop any "provider:" prefix
    if not name.startswith(_REASONING_NAME_PREFIXES):
        return False
    if name not in _name_prefix_reasoning_reported:
        _name_prefix_reasoning_reported.add(name)
        logger.info(
            "[budget] reasoning headroom for '%s' comes from the name list, "
            "not from a declaration: the adapter does not declare this model a "
            "reasoning model, so it gets the reasoning budget without the "
            "reasoning controls",
            name,
        )
    return True


#: The output budget a model that reasons is never given less than when the
#: framework chooses the budget itself. A reasoning model spends the first part
#: of its budget on a chain nobody sees, so a budget chosen from a declared
#: shape — a one-field schema, a short-answer style — would leave it nothing to
#: answer with and return an empty, billed result. A budget the caller pinned is
#: theirs and is never raised to this.
REASONING_OUTPUT_FLOOR = 4096

#: The smallest budget derived from a declared output schema. A schema for one
#: integer bounds the answer at a few tokens, but the model still writes an
#: envelope, and a provider that pads or repeats needs room to finish the object
#: it started rather than being cut off mid-value.
SCHEMA_BUDGET_FLOOR = 256

#: Tokens allowed for the JSON envelope itself — braces, keys, separators.
_SCHEMA_ENVELOPE_TOKENS = 64
#: Tokens allowed per declared field, by the JSON type the schema declares. An
#: array or a free string is open-ended, so both get the generous allowance; a
#: number, a boolean and an enumerated string are bounded by their own type.
_SCHEMA_FIELD_TOKENS: dict[str, int] = {
    "boolean": 16,
    "integer": 16,
    "number": 16,
    "null": 8,
    "string": 256,
    "array": 1024,
}
_SCHEMA_UNKNOWN_FIELD_TOKENS = 256
_SCHEMA_MAX_DEPTH = 4


def budget_for_output_schema(schema: Any, *, _depth: int = 0) -> int:
    """Return an output-token bound derived from a declared *schema*.

    The bound is an envelope plus one allowance per declared field, chosen from
    the JSON type the field declares and from nothing else: a bounded type gets
    a small allowance, an array or an unconstrained string gets a generous one,
    and a nested object is measured the same way. Nothing here reads the task,
    the field names, the model or a benchmark — only what the caller declared
    about the answer.

    Args:
        schema: A JSON-Schema mapping, or anything that is not one.
        _depth: Recursion depth, so a self-referential schema terminates.

    Returns:
        The bound, never below :data:`SCHEMA_BUDGET_FLOOR`.
    """
    total = _SCHEMA_ENVELOPE_TOKENS + _schema_field_tokens(schema, _depth)
    return max(SCHEMA_BUDGET_FLOOR, total)


def _schema_field_tokens(schema: Any, depth: int) -> int:
    """Tokens one schema node is allowed, summed over its declared fields."""
    if not isinstance(schema, dict) or depth > _SCHEMA_MAX_DEPTH:
        return _SCHEMA_UNKNOWN_FIELD_TOKENS
    declared = schema.get("type")
    if declared == "object" or "properties" in schema:
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return _SCHEMA_UNKNOWN_FIELD_TOKENS
        return sum(
            _schema_field_tokens(field, depth + 1) for field in properties.values()
        )
    if declared == "array":
        return _SCHEMA_FIELD_TOKENS["array"]
    if declared == "string" and schema.get("enum"):
        return 24
    if isinstance(declared, list):
        return max(
            (_SCHEMA_FIELD_TOKENS.get(str(one), _SCHEMA_UNKNOWN_FIELD_TOKENS)
             for one in declared),
            default=_SCHEMA_UNKNOWN_FIELD_TOKENS,
        )
    return _SCHEMA_FIELD_TOKENS.get(str(declared), _SCHEMA_UNKNOWN_FIELD_TOKENS)


def declared_max_output_tokens(model: Any) -> int | None:
    """The largest output *model* declares it will produce, or ``None``.

    Read from the adapter's own published attribute, never from a name. An
    adapter that publishes nothing answers ``None`` and caps nothing, which is
    what every adapter in this tree does today; the seam is here so a budget
    chosen by the framework can never exceed what a provider will accept once an
    adapter does publish it.
    """
    for attr in ("max_output_tokens", "max_output"):
        try:
            value = getattr(model, attr, None)
        except Exception:  # noqa: BLE001 - a property that raises declares nothing
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return value
    return None


def _local_template_reasons(model: Any) -> bool:
    """Whether a local model's own chat template emits a reasoning chain.

    A cloud model is flagged from the catalog, but a local engine has no
    catalog, and the name is not a reliable signal — ``Qwen3.5-2B`` reasons
    while ``Qwen3-4B-Instruct`` does not, and both start "qwen3". The template
    is the model's own statement about itself: one that opens a ``<think>``
    block spends output budget before its first visible token, so the base
    1024-token budget is consumed by the chain and the turn returns nothing.

    Measured once per tokenizer and cached on the model, because this is asked
    on the way into every call.
    """
    tokenizer = getattr(model, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    if not template or not isinstance(template, str):
        return False
    cached = getattr(model, "_reasoning_template_probe", None)
    if cached is not None and cached[0] is tokenizer:
        return bool(cached[1])
    reasons = "<think>" in template or "enable_thinking" in template
    try:
        model._reasoning_template_probe = (tokenizer, reasons)
    except Exception:  # noqa: BLE001 - a model that refuses the attribute still answers
        pass
    return reasons


def default_max_output_tokens(
    model: Any, *, base: int = 1024, reasoning: int = 4096
) -> int:
    """Pick a sensible default output-token budget for *model*.

    Reasoning families get ``reasoning`` (they burn budget on internal
    reasoning); everything else keeps the historical ``base``.

    This is the *first* budget the agent tries. A run truncated at this value
    escalates once (see ``_TRUNCATION_MAX_TOKENS_CEILING`` in
    ``effgen.core.agent_generation``) rather than starting high, so an ordinary
    task is not charged for a budget it never needed. A direct
    ``model.generate()`` call has no such escalation and should ask for more —
    see ``DIRECT_CALL_REASONING_MAX_TOKENS``.

    Args:
        model: The model id or adapter the budget is chosen for.
        base: The budget for an ordinary model.
        reasoning: The budget for a reasoning family.

    Returns:
        The output-token budget to send with the first attempt.
    """
    return reasoning if needs_reasoning_headroom(model) else base


#: Output budget for a reasoning model on a *direct* ``generate()`` /
#: ``generate_stream()`` call — one where no agent loop is watching for a
#: truncated result, so there is no second attempt at a larger budget.
#:
#: Deliberately well above the agent's escalation ceiling: a one-word answer was
#: measured consuming 4,285 output tokens, so a budget in the low thousands
#: truncates mid-thought and returns empty, still-billed text. Unused budget is
#: free — providers bill tokens generated, not the ceiling requested.
DIRECT_CALL_REASONING_MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Reasoning chains returned beside an empty answer
# ---------------------------------------------------------------------------

#: Attribute names OpenAI-compatible providers use for the chain a reasoning
#: model emits before its visible answer. Together, Groq and Cerebras all send
#: ``message.reasoning``; some deployments use ``reasoning_content``.
_REASONING_MESSAGE_ATTRS = ("reasoning", "reasoning_content")

# One warning per (model, finish reason): a batch job on a reasoning model would
# otherwise log the same line on every call.
_reasoning_only_warned: set[tuple[str, str]] = set()


def extract_reasoning_text(message: Any) -> str:
    """Return the reasoning chain attached to one chat message, or ``""``.

    The chain is the model's internal reasoning, not its answer: it is reported
    for diagnosis and never returned to the caller as the answer.
    """
    for attr in _REASONING_MESSAGE_ATTRS:
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def extract_reasoning_tokens(usage: Any) -> int:
    """Return the reasoning-token count a provider reports for one call.

    OpenAI, Groq and Cerebras nest it under
    ``usage.completion_tokens_details.reasoning_tokens`` on the chat-completions
    API and under ``usage.output_tokens_details.reasoning_tokens`` on the
    Responses API; a few report a top-level ``usage.reasoning_tokens``. Returns
    ``0`` when none is present or the value is not a positive integer.
    """
    details = getattr(usage, "completion_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    for candidate in (getattr(details, "reasoning_tokens", None),
                      getattr(output_details, "reasoning_tokens", None),
                      getattr(usage, "reasoning_tokens", None)):
        try:
            count = int(candidate)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 0


def reasoning_only_message(
    model_name: str,
    *,
    finish_reason: str,
    reasoning_tokens: int,
    reasoning_chars: int,
    max_tokens: int | None,
    completion_tokens: int | None = None,
) -> str:
    """Message for a turn whose whole output was reasoning and no visible text.

    Names the model, the output cap in force and the reasoning budget spent, so
    the caller can tell this apart from a flaky empty response and knows which
    lever to pull.

    Args:
        model_name: The model that produced the turn.
        finish_reason: The provider's finish reason for the turn.
        reasoning_tokens: Output tokens the provider billed as reasoning.
        reasoning_chars: Length of the reasoning chain, when the provider sent it.
        max_tokens: The output cap in force, or ``None`` when none was set.
        completion_tokens: Total completion tokens the provider reported.

    Returns:
        One line naming the model, the cap and the reasoning spend.
    """
    spent = reasoning_tokens or completion_tokens or 0
    budget = (
        f"a max_tokens cap of {max_tokens}" if max_tokens
        else "the provider's default max_tokens"
    )
    if finish_reason == FINISH_LENGTH:
        cause = (
            f"it spent the whole output budget on internal reasoning and hit "
            f"{budget} (finish_reason='length')"
        )
        remedy = (
            "Raise max_tokens — e.g. agent.run(task, max_tokens=8192) — or use "
            "a model that answers without an extended reasoning chain."
        )
    else:
        cause = (
            f"generation ended (finish_reason={finish_reason!r}) after the "
            f"reasoning chain and before the first visible token, under {budget}"
        )
        remedy = (
            "Raise max_tokens, drop any stop sequence the reasoning chain can "
            "match, or use a model that answers without an extended reasoning "
            "chain."
        )
    if spent and reasoning_chars:
        produced = f"{spent} reasoning tokens ({reasoning_chars} characters of reasoning)"
    elif spent:
        produced = f"{spent} reasoning tokens"
    elif reasoning_chars:
        produced = f"{reasoning_chars} characters of internal reasoning"
    else:  # pragma: no cover - one of the two is always known here
        produced = "internal reasoning"
    return (
        f"Model '{model_name}' returned no visible text: {cause}. "
        f"It produced {produced} and no answer. {remedy}"
    )


def annotate_reasoning_only(
    metadata: dict[str, Any],
    *,
    text: str,
    reasoning_text: str,
    reasoning_tokens: int,
    model_name: str,
    finish_reason: str,
    max_tokens: int | None,
    completion_tokens: int | None = None,
    tool_calls: list[Any] | None = None,
    logger: Any = None,
) -> bool:
    """Record a reasoning-only turn on *metadata* and return whether it was one.

    A reasoning-only turn is an empty ``content`` and no tool call beside either
    a populated reasoning chain or a non-zero reasoning-token count: the model
    was billed for output the caller cannot see. Providers report one signal or
    the other — Together, Groq and Cerebras send the chain text, OpenAI reports
    only the token count — so either is enough. ``metadata`` gains
    ``reasoning_tokens`` whenever the provider reports any, and — for a
    reasoning-only turn — ``reasoning_only``, ``reasoning`` (when the chain text
    is available), ``reasoning_chars`` and ``empty_response_reason`` carrying the
    message from :func:`reasoning_only_message`.

    A native tool call is a complete turn even with empty text, so it is never
    reported as reasoning-only.

    Args:
        metadata: The result metadata to annotate in place.
        text: The visible content of the turn.
        reasoning_text: The reasoning chain, when the provider sent one.
        reasoning_tokens: Output tokens the provider billed as reasoning.
        model_name: The model that produced the turn.
        finish_reason: The provider's finish reason for the turn.
        max_tokens: The output cap in force, or ``None`` when none was set.
        completion_tokens: Total completion tokens the provider reported.
        tool_calls: Native tool calls the turn made, which rule it out as
            reasoning-only.
        logger: Logger used for the heads-up, when one is passed.

    Returns:
        ``True`` when the turn was reasoning-only.
    """
    if reasoning_tokens:
        metadata["reasoning_tokens"] = reasoning_tokens
    reasoning_text = reasoning_text or ""
    if tool_calls or (text or "").strip():
        return False
    if not reasoning_text.strip() and reasoning_tokens <= 0:
        return False

    message = reasoning_only_message(
        model_name,
        finish_reason=finish_reason,
        reasoning_tokens=reasoning_tokens,
        reasoning_chars=len(reasoning_text),
        max_tokens=max_tokens,
        completion_tokens=completion_tokens,
    )
    metadata["reasoning_only"] = True
    metadata["reasoning_chars"] = len(reasoning_text)
    metadata["empty_response_reason"] = message
    if reasoning_text:
        metadata["reasoning"] = reasoning_text

    if logger is not None:
        warn_key = (model_name, finish_reason)
        if warn_key not in _reasoning_only_warned:
            _reasoning_only_warned.add(warn_key)
            logger.warning(message)
    return True


def reasoning_delta_text(delta: Any) -> str:
    """Return the reasoning fragment carried by one streaming delta, or ``""``."""
    for attr in _REASONING_MESSAGE_ATTRS:
        value = getattr(delta, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def warn_reasoning_only_stream(
    *,
    model_name: str,
    yielded_text: bool,
    reasoning_text: str,
    reasoning_tokens: int = 0,
    finish_reason: Any = None,
    max_tokens: int | None,
    tool_calls: Any = None,
    logger: Any,
) -> None:
    """Report a stream that ended without yielding a single visible token.

    A streamed turn has no metadata channel back to the caller, so the same
    message :func:`annotate_reasoning_only` records is logged instead — an empty
    iterator would otherwise look like a call that simply had nothing to say.

    Args:
        model_name: The model that produced the stream.
        yielded_text: Whether any visible token reached the caller.
        reasoning_text: The reasoning chain, when the provider sent one.
        reasoning_tokens: Output tokens the provider billed as reasoning.
        finish_reason: The provider's finish reason for the stream.
        max_tokens: The output cap in force, or ``None`` when none was set.
        tool_calls: Native tool calls the stream declared. A turn spent making
            one is complete with no visible text, so it is never reported as
            reasoning-only — the same rule :func:`annotate_reasoning_only`
            applies to a non-streamed turn.
        logger: Logger the message is written to.
    """
    if yielded_text or tool_calls:
        return
    if not (reasoning_text or "").strip() and reasoning_tokens <= 0:
        return
    finish = normalize_finish_reason(finish_reason)
    message = reasoning_only_message(
        model_name,
        finish_reason=finish,
        reasoning_tokens=reasoning_tokens,
        reasoning_chars=len(reasoning_text or ""),
        max_tokens=max_tokens,
    )
    warn_key = (model_name, f"stream:{finish}")
    if warn_key in _reasoning_only_warned:
        return
    _reasoning_only_warned.add(warn_key)
    logger.warning(message)


def warn_empty_stream(
    *,
    model_name: str,
    yielded_text: bool,
    max_tokens: int | None,
    tool_calls: Any = None,
    logger: Any,
) -> None:
    """Report a stream the provider ended without sending a single chunk.

    Some providers answer a request whose whole output budget went to internal
    reasoning with an empty stream — no content, no tool call, no usage block and
    no finish reason — so :func:`warn_reasoning_only_stream` has nothing to key
    on and the caller receives an empty iterator that reads as "the model had
    nothing to say". This states what happened and which lever to pull instead.
    Fires at most once per model per process, sharing the reasoning-only guard.

    Args:
        model_name: The model that produced the stream.
        yielded_text: Whether any visible token reached the caller.
        max_tokens: The output cap in force, or ``None`` when none was set.
        tool_calls: Native tool calls the stream declared; a turn spent making
            one is complete without visible text.
        logger: Logger the message is written to.
    """
    if yielded_text or tool_calls:
        return
    budget = (
        f"a max_tokens cap of {max_tokens}" if max_tokens
        else "the provider's default max_tokens"
    )
    warn_key = (model_name, "stream:empty")
    if warn_key in _reasoning_only_warned:
        return
    _reasoning_only_warned.add(warn_key)
    logger.warning(
        f"Model '{model_name}' streamed no tokens at all under {budget}, and the "
        f"provider reported no content, tool call or usage for the turn. A model "
        f"that reasons before answering can spend the whole output budget before "
        f"its first visible token. Raise max_tokens — e.g. "
        f"agent.run(task, max_tokens=8192) — or use a model that answers without "
        f"an extended reasoning chain."
    )


def normalize_stop_sequences(value: Any) -> list[str] | None:
    """Return *value* as a list of stop sequences, whatever shape it arrived in.

    ``stop_sequences`` is a list and every consumer iterates it, so a bare
    string — the shape the OpenAI API itself accepts, and the one a caller
    naturally writes as ``agent.run(task, stop_sequences="END")`` — is walked
    character by character and cuts the text at the first single letter that
    matches. ``"The ocean is DEEP and ENDless."`` came back as
    ``"The ocean is "`` instead of ``"The ocean is DEEP and "``.

    This is the one place that decision is made, so the agent boundary, the
    per-call keyword helpers and :class:`~effgen.models.base.GenerationConfig`
    cannot disagree about it.

    Args:
        value: ``None``, a string, or any iterable of strings.

    Returns:
        ``None`` when nothing was given, otherwise a list of strings.

    Raises:
        TypeError: The value is neither a string nor an iterable of strings.
            It is rejected here, naming what was passed, rather than reaching a
            provider as a malformed request or a consumer as a silent
            character-wise walk.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        bad = [item for item in items if not isinstance(item, str)]
        if bad:
            raise TypeError(
                "stop_sequences must contain only strings; got "
                f"{type(bad[0]).__name__}. Pass a string for a single sequence "
                'or a list of strings, e.g. stop_sequences=["\\nObservation:"].'
            )
        return items
    raise TypeError(
        f"stop_sequences must be a string or a list of strings; got "
        f"{type(value).__name__}. Pass a string for a single sequence or a list "
        'of strings, e.g. stop_sequences=["END"].'
    )


def apply_stop_sequences(text: str, stop_sequences: Any) -> str:
    """Truncate *text* at the earliest stop sequence it contains.

    Used when the stop sequences cannot be sent to the provider — a provider
    that streams a reasoning chain and the answer through one token stream
    matches them against the chain as well, which can end generation before the
    first visible token. Cutting the returned answer locally gives the same
    visible result without that collision.

    A bare string is accepted and treated as one sequence, so a config built by
    hand cannot make this walk the string character by character.
    """
    stop_sequences = normalize_stop_sequences(stop_sequences)
    if not text or not stop_sequences:
        return text
    cut = len(text)
    for sequence in stop_sequences:
        if not sequence:
            continue
        index = text.find(sequence)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


# Sampling fields a caller may override for a single call by passing the name as
# a keyword argument to ``generate()``/``generate_stream()``/``generate_batch()``
# instead of (or beside) a ``GenerationConfig``. Every name is a field of
# :class:`~effgen.models.base.GenerationConfig`; ``stop`` is the OpenAI-style
# alias for ``stop_sequences``.
CALL_OVERRIDE_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "stop_sequences",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
)

CALL_OVERRIDE_ALIASES: dict[str, str] = {"stop": "stop_sequences"}


def merge_call_overrides(config: Any, kwargs: dict[str, Any]) -> Any:
    """Return *config* with the per-call sampling keywords in *kwargs* applied.

    A per-call value supersedes the config's field for that call only — the
    config object passed in is never mutated. Recognised keys are **removed**
    from *kwargs*, so they are not also forwarded to a provider call that has no
    parameter of that name (vLLM's ``LLM.generate`` accepts neither ``seed`` nor
    ``temperature``, and mlx-lm's generate takes its sampling settings from the
    keywords this builds). Unrecognised keys stay in *kwargs* for the caller to
    forward or ignore as its backend requires.

    A bare string is accepted for ``stop``/``stop_sequences`` — the shape the
    OpenAI API takes — and becomes a one-element list, because
    ``GenerationConfig.stop_sequences`` carries a list and the backends iterate
    it: a string left as-is would be walked character by character and cut the
    text at the first matching letter.
    """
    overrides: dict[str, Any] = {}
    for key in list(kwargs):
        field = CALL_OVERRIDE_ALIASES.get(key, key)
        if field in CALL_OVERRIDE_FIELDS:
            value = kwargs.pop(key)
            if field == "stop_sequences":
                value = normalize_stop_sequences(value)
            overrides[field] = value
    if not overrides:
        return config
    return replace(config, **overrides)


def apply_call_overrides(config: Any, kwargs: dict[str, Any]) -> Any:
    """Fold per-call sampling keywords into *config*, defaulting it when absent.

    ``merge_call_overrides`` needs a config object to copy; the local engines
    are called with ``config=None`` whenever the caller passed only keywords,
    which is the ordinary case for ``generate(prompt, stop_sequences="END")``.
    Lives here rather than beside the engine so the generation and streaming
    modules can reach it without importing a sibling that pulls in torch.
    """
    from .base import GenerationConfig

    return merge_call_overrides(config if config is not None else GenerationConfig(), kwargs)


# ---------------------------------------------------------------------------
# Structured, redacted provider errors
# ---------------------------------------------------------------------------

def device_memory_hint(message: str) -> str | None:
    """Return advice for a local out-of-memory failure, or ``None``.

    A device out-of-memory failure classifies as ``resource_exhausted``, whose
    remediation says only that the request will keep failing. This adds the
    concrete levers. Callers append it to their remediation text when it
    applies.
    """
    lowered = (message or "").lower()
    if not any(s in lowered for s in DEVICE_MEMORY_SIGNALS):
        return None
    return (
        " The device ran out of memory — lower max_tokens or the batch size, "
        "load the model with quantization_bits=4/8, pick a smaller model, or "
        "free the GPU (check `nvidia-smi`)."
    )


def build_error_context(
    provider: str,
    model: str,
    request_type: str,
    exc: Exception,
) -> dict[str, str]:
    """Build the structured ``error_context`` for a provider failure.

    Classifies *exc* and returns ``{provider, model, request_type,
    retry_status, remediation, category}``. The fields are static/derived (no
    secret material), so the dict itself is always safe to log or surface. The
    retry-status + remediation text come from the single source of truth in
    :mod:`effgen.models.errors`.
    """
    cls = classify_provider_error(exc)
    return error_context_dict(provider, model, request_type, cls.category)


def provider_runtime_error(
    provider: str,
    model: str,
    request_type: str,
    exc: Exception,
    *,
    message: str | None = None,
    endpoint: str | None = None,
) -> RuntimeError:
    """Build a consistent, **redacted** :class:`RuntimeError` for a provider failure.

    The returned error carries an ``.error_context`` attribute (the dict from
    :func:`build_error_context`) and a message of the form::

        "<provider> <request_type> failed [<retry_status>]: <redacted cause>. <remediation>"

    The underlying SDK message is run through the process redactor so no API
    keys/secrets leak into logs or user-facing output. Callers should
    ``raise provider_runtime_error(...) from exc`` to preserve the traceback.

    Args:
        provider: The provider whose call failed.
        model: The model id the call targeted.
        request_type: What was attempted, such as ``generate`` or ``stream``.
        exc: The SDK exception to classify and redact.
        message: A message to use instead of the redacted cause.
        endpoint: The URL the call was sent to, when that is not the
            provider's own. Named in the remediation, because the stock advice
            to check the provider's status page is the wrong advice when the
            request went to a server the caller runs.

    Returns:
        The error to raise, carrying ``.error_context``.
    """
    # Local import keeps the module import cost low and avoids a hard
    # dependency cycle at import time.
    from ..errors import quote_for_message
    from ..observability.redact import get_redactor

    ctx = build_error_context(provider, model, request_type, exc)
    redactor = get_redactor()
    # A provider echoes the rejected request back, so the cause is bounded as
    # well as redacted: a rate-limit body alone runs to well over a kilobyte,
    # and this message is read in a terminal panel and a log line.
    cause = quote_for_message(exc) if str(exc) else exc.__class__.__name__
    head = redactor.scrub(message) if message else f"{provider} {request_type} failed"

    remediation = ctx["remediation"]
    # On a 404 / model_not_found, append the live "did you mean…/available
    # now…" hint so the user sees real alternatives instead of a raw provider
    # 404 — regardless of which adapter path (plain or tool-calling) failed.
    if ctx["category"] == "not_found" and provider and model:
        try:
            from ._catalog import suggest_for_missing

            hint = suggest_for_missing(provider, model)
            if hint:
                remediation = remediation + hint
        except Exception:  # pragma: no cover - suggestion is best-effort
            pass
    # On a rejected-as-invalid request that reads like a context-window or
    # token-rate overflow (a large preset's tool-schema payload alone can
    # push a small-context/low-rate-limit model over the line before any
    # real content is added), point toward a smaller preset or a
    # bigger-context/higher-rate-limit model instead of a bare rejection.
    elif ctx["category"] == "invalid_request":
        hint = context_overflow_hint(str(exc))
        if hint:
            remediation = remediation + hint

    # A local engine that ran out of device memory needs device-level advice,
    # not the provider-status advice the generic categories carry.
    device_hint = device_memory_hint(str(exc))
    if device_hint:
        remediation = remediation + device_hint

    # The request went somewhere other than the provider's own API, so say
    # where. Otherwise a self-hosted server that is down reads as an outage at
    # the provider whose protocol it speaks.
    if endpoint:
        remediation = f"{remediation} The call was sent to {endpoint}."

    err = RuntimeError(
        f"{head} [{ctx['retry_status']}]: {cause}. {remediation}"
    )
    err.error_context = ctx  # type: ignore[attr-defined]
    return err


def model_not_found_error(provider: str, model: str, message: str) -> Exception:
    """Build a :class:`ModelNotFoundError` carrying the catalog suggestion.

    Appends the live "did you mean… / available now…" hint from the model
    catalog to *message* so a 404 surfaces real alternatives instead of a raw
    provider error.

    Args:
        provider: The provider that rejected the id.
        model: The model id that was not found.
        message: The provider's own message, which the hint is appended to.

    Returns:
        A :class:`ModelNotFoundError` ready to raise.
    """
    from ._catalog import suggest_for_missing
    from .errors import ModelNotFoundError

    return ModelNotFoundError(
        provider, model, message + suggest_for_missing(provider, model)
    )


def not_loaded_error(
    provider: str,
    model: str = "",
    request_type: str = "request",
) -> RuntimeError:
    """Build the error every adapter raises when a call arrives before ``load()``.

    The returned :class:`RuntimeError` carries the same ``.error_context``
    shape as a provider failure, with category ``"not_loaded"`` and a
    non-retryable status: the model is missing locally, so a retry cannot
    change the outcome. Adapters raise this instead of a bare
    ``RuntimeError`` so the retry layer, the agent's error record, and the
    server envelope all see one classified shape.

    Args:
        provider: The provider the adapter speaks to.
        model: The model id the call targeted.
        request_type: What was attempted, such as ``generate`` or ``stream``.

    Returns:
        The error to raise, carrying ``.error_context``.
    """
    ctx = error_context_dict(provider, model, request_type, "not_loaded")
    where = ", ".join(p for p in (f"provider={provider}" if provider else "",
                                  f"model={model!r}" if model else "") if p)
    suffix = f" ({where})" if where else ""
    err = RuntimeError(
        f"Model is not loaded{suffix} [{ctx['retry_status']}]: "
        f"{request_type}() was called before load(). {ctx['remediation']}"
    )
    err.error_context = ctx  # type: ignore[attr-defined]
    return err


def attach_error_context(
    err: Exception,
    provider: str,
    model: str,
    request_type: str,
    *,
    source: Exception | None = None,
) -> Exception:
    """Attach a structured ``.error_context`` to an already-typed effGen error.

    Used where an adapter raises a specific typed error (auth/not-found/etc.)
    but we still want the uniform machine-readable context on it. ``source``
    is the original exception used for classification (defaults to ``err``).

    Args:
        err: The typed error to annotate, which is returned unchanged otherwise.
        provider: The provider whose call failed.
        model: The model id the call targeted.
        request_type: What was attempted, such as ``generate`` or ``stream``.
        source: The original exception to classify from, defaulting to *err*.

    Returns:
        The same exception, now carrying ``.error_context``.
    """
    if not hasattr(err, "error_context"):
        err.error_context = build_error_context(  # type: ignore[attr-defined]
            provider, model, request_type, source or err
        )
    return err


__all__ = [
    "TOOL_PROBE_NAME",
    "get_bpe_encoding",
    "estimate_tokens",
    "chat_template_renders_tools",
    "CALL_OVERRIDE_ALIASES",
    "CALL_OVERRIDE_FIELDS",
    "CANONICAL_FINISH_REASONS",
    "merge_call_overrides",
    "normalize_finish_reason",
    "apply_tool_request",
    "normalize_stop_sequences",
    "normalize_tools_call_args",
    "needs_reasoning_headroom",
    "default_max_output_tokens",
    "budget_for_output_schema",
    "declared_max_output_tokens",
    "REASONING_OUTPUT_FLOOR",
    "SCHEMA_BUDGET_FLOOR",
    "build_error_context",
    "device_memory_hint",
    "provider_runtime_error",
    "model_not_found_error",
    "not_loaded_error",
    "attach_error_context",
    "RETRY_WILL_RETRY",
    "RETRY_RATE_LIMITED",
    "RETRY_NON_RETRYABLE",
]
