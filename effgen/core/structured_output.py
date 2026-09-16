"""
Structured output support for the effGen framework.

This module allows agents to return guaranteed-valid structured data
(JSON, YAML, CSV) matching a user-provided schema.

Approaches (in priority order):
1. Grammar-constrained decoding via ``outlines`` (optional, Transformers only)
2. API-level ``response_format`` for API models (OpenAI, Gemini)
3. Post-validation with ``jsonschema`` + retry (universal fallback)

Usage::

    result = agent.run(
        "Extract entities from this text",
        output_schema={
            "type": "object",
            "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
            "required": ["entities"]
        }
    )
    data = json.loads(result.output)  # guaranteed valid

Pydantic support::

    class Entities(BaseModel):
        items: list[str]
    result = agent.run("...", output_model=Entities)
    parsed = result.metadata["parsed"]  # Entities instance
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from typing import Any

from . import ledger as _ledger

logger = logging.getLogger(__name__)


def _model_label(model: Any) -> str:
    return str(getattr(model, "model_name", "") or "")


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output constraints.

    Attributes:
        schema: JSON Schema dict that the output must conform to.
        format: Output format — "json" (default), "yaml", or "csv".
        max_retries: Maximum reprompt attempts on validation failure (the
            native JSON-mode attempt is separate and tried first).
        use_grammar: Whether to attempt grammar-constrained decoding
            (requires ``outlines`` library, Transformers only).
        time_budget_s: Wall-clock cap for all repair/reprompt attempts. When the
            budget is exhausted, repairs stop and the failure is reported clearly
            instead of looping. ``None`` disables the cap.
    """
    schema: dict[str, Any] | None = None
    format: str = "json"
    max_retries: int = 2
    use_grammar: bool = True
    time_budget_s: float | None = 30.0


@dataclass
class StructuredOutcome:
    """Result of a structured-generation attempt.

    Attributes:
        success: Whether a schema-valid object was produced.
        json_str: The validated JSON string (None on failure).
        parsed: The parsed Python object (None on failure).
        attempts: Number of *extra* model calls spent constraining the output
            (0 means the model's first answer already validated).
        method: Which strategy produced/attempted the result
            ("grammar", "native", "reprompt", or None).
        error: Human-readable failure reason (None on success).
        raw_text: The last raw model text seen, for clear failure reporting.
    """
    success: bool
    json_str: str | None = None
    parsed: Any = None
    attempts: int = 0
    method: str | None = None
    error: str | None = None
    raw_text: str | None = None


def validate_json_schema(data: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate data against a JSON Schema.

    Tries ``jsonschema`` library first, falls back to basic structural checks.

    Args:
        data: Parsed JSON data to validate.
        schema: JSON Schema dict.

    Returns:
        (is_valid, error_message)
    """
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return True, None
    except ImportError:
        logger.debug("jsonschema not installed, using basic validation")
        return _basic_validate(data, schema)
    except Exception as e:
        return False, str(e)


def _basic_validate(data: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """Basic JSON Schema validation without jsonschema library.

    Only checks type, required fields, and properties — not full spec.
    """
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            return False, f"Expected object, got {type(data).__name__}"
        required = schema.get("required", [])
        # A spec-correct schema gives ``required`` as a list of field names; a
        # malformed schema may give a string (iterating it char-by-char would
        # invent bogus "missing field" errors) or another scalar. Coerce a bare
        # string to a single-name list and ignore other non-iterables.
        if isinstance(required, str):
            required = [required]
        elif not isinstance(required, list | tuple):
            required = []
        for key in required:
            if key not in data:
                return False, f"Missing required field: {key}"
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for key, prop_schema in properties.items():
            if key in data:
                valid, err = _basic_validate(data[key], prop_schema)
                if not valid:
                    return False, f"Field '{key}': {err}"
        return True, None

    elif schema_type == "array":
        if not isinstance(data, list):
            return False, f"Expected array, got {type(data).__name__}"
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                valid, err = _basic_validate(item, items_schema)
                if not valid:
                    return False, f"Item [{i}]: {err}"
        return True, None

    elif schema_type == "string":
        return (True, None) if isinstance(data, str) else (False, f"Expected string, got {type(data).__name__}")
    elif schema_type == "integer":
        return (True, None) if isinstance(data, int) and not isinstance(data, bool) else (False, "Expected integer")
    elif schema_type == "number":
        return (True, None) if isinstance(data, int | float) and not isinstance(data, bool) else (False, "Expected number")
    elif schema_type == "boolean":
        return (True, None) if isinstance(data, bool) else (False, "Expected boolean")
    elif schema_type == "null":
        return (True, None) if data is None else (False, "Expected null")

    # No type constraint or unknown type — accept
    return True, None


def extract_json_from_text(text: str) -> str | None:
    """Extract the first JSON object or array from free text.

    Handles:
    - Markdown code fences (```json ... ```)
    - JSON embedded in prose
    - Trailing commas and unquoted keys (basic cleanup)

    Returns:
        Extracted JSON string, or None if not found.
    """
    # The first bracket-balanced object/array anywhere in the text, used both as
    # the fallback and to tell a real code fence from backticks that are part of
    # the JSON's own data.
    span = _find_balanced(text)

    # Prefer a markdown code fence: locate the fence *opener* and scan the text
    # that follows for a bracket-balanced object/array (string-aware). We do not
    # rely on a lazy ``...```...``` regex to delimit the content, because a fence
    # whose own JSON contains a ``` (e.g. as a string value) would be truncated
    # at that inner backtick. The fence is a strong "JSON is here" signal, so it
    # takes priority over a stray brace earlier in surrounding prose.
    opener = re.search(r'```[ \t]*(?:json|JSON)?[ \t]*\r?\n?', text)
    if opener is not None:
        # Backticks *inside* a JSON value that already parses are data, not a
        # fence — scanning from them would return a fragment of that object.
        backticks_are_data = (
            span is not None
            and span[0] <= opener.start() < span[1]
            and _parses_as_json(text[span[0]:span[1]])
        )
        if not backticks_are_data:
            balanced = _extract_balanced(text[opener.end():])
            if balanced is not None:
                return _clean_json(balanced)

    return _clean_json(text[span[0]:span[1]]) if span is not None else None


def _parses_as_json(text: str) -> bool:
    """Whether *text* is already valid JSON."""
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


def _extract_balanced(text: str) -> str | None:
    """Return the first bracket-balanced JSON object/array substring, or None.

    String-aware (ignores brackets/backticks inside JSON string literals).
    Starts from whichever opening bracket appears *earliest* so a top-level
    array of objects (``[{...}]``) is extracted whole rather than collapsing to
    its first inner object.
    """
    span = _find_balanced(text)
    return text[span[0]:span[1]] if span is not None else None


def _find_balanced(text: str) -> tuple[int, int] | None:
    """Return the ``(start, end)`` span of the first balanced object/array.

    ``end`` is exclusive. Returns ``None`` when nothing balances.
    """
    brace = text.find("{")
    bracket = text.find("[")
    if brace == -1 and bracket == -1:
        bracket_order: list[tuple[str, str]] = []
    elif brace == -1:
        bracket_order = [("[", "]")]
    elif bracket == -1:
        bracket_order = [("{", "}")]
    elif brace < bracket:
        bracket_order = [("{", "}"), ("[", "]")]
    else:
        bracket_order = [("[", "]"), ("{", "}")]

    for start_char, end_char in bracket_order:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching end bracket (accounting for nesting)
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return start, i + 1
    return None


def _split_string_literals(text: str) -> list[tuple[str, bool]]:
    """Split *text* into ``(segment, is_string_literal)`` spans.

    A span is marked as a string literal when it is a double-quoted JSON string
    (escapes honoured). An unterminated trailing quote marks the rest of the
    text as a literal, so a repair never edits what looks like string content.
    """
    spans: list[tuple[str, bool]] = []
    start = 0
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                spans.append((text[start:i + 1], True))
                start = i + 1
                in_string = False
        elif c == '"':
            if start < i:
                spans.append((text[start:i], False))
            start = i
            in_string = True
        i += 1
    if start < n:
        spans.append((text[start:], in_string))
    return spans


def _clean_json(text: str) -> str:
    """Repair the two JSON defects small models commonly emit.

    Removes a trailing comma before ``}``/``]`` and quotes an unquoted object
    key. Text that already parses as JSON is returned byte-for-byte unchanged,
    and both repairs are applied only outside string literals — so a string
    value containing a comma, colon, or bracket (``{"note": "first, then: go"}``)
    keeps its exact content.
    """
    try:
        json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        return text

    out: list[str] = []
    for segment, is_literal in _split_string_literals(text):
        if is_literal:
            out.append(segment)
            continue
        # Remove trailing commas before } or ]
        segment = re.sub(r',\s*([}\]])', r'\1', segment)
        # Quote unquoted keys
        segment = re.sub(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', segment)
        out.append(segment)
    return "".join(out)


def structured_generate(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    config: StructuredOutputConfig | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> StructuredOutcome:
    """Generate schema-valid output, preferring native modes over reprompting.

    Strategy order (cheapest/strongest first):

    1. Grammar-constrained decoding (``outlines``, local Transformers only) —
       the output is structurally guaranteed in one call.
    2. Provider-native JSON mode (OpenAI strict ``json_schema``; OpenAI-compatible
       ``json_object``; Gemini ``response_mime_type``) — one call, usually valid.
    3. Reprompt-with-validation fallback, bounded by ``max_retries`` **and** a
       wall-clock ``time_budget_s`` so repairs can't run away.

    Unlike a raising helper, this returns a :class:`StructuredOutcome` so callers
    can surface the attempt count and fail explicitly (raw text preserved) when no
    strategy produces a schema-valid object.

    Args:
        model: The model to generate with.
        prompt: The prompt describing what to produce.
        schema: The JSON schema the output must satisfy.
        config: Strategy, retry and time-budget settings.
        generation_kwargs: Extra keyword arguments forwarded to the model.
    """
    if config is None:
        config = StructuredOutputConfig(schema=schema)
    gen_kwargs = generation_kwargs or {}
    deadline = (
        time.monotonic() + config.time_budget_s
        if config.time_budget_s is not None
        else None
    )
    attempts = 0
    last_error: str | None = None
    last_raw: str | None = None

    # --- Strategy 1: grammar-constrained decoding (local Transformers) ---
    # Only counts as an attempt when ``outlines`` is actually present and runs;
    # a missing optional dependency is not a model call.
    if config.use_grammar and _is_transformers_model(model) and _outlines_available():
        attempts += 1
        result = _try_grammar_constrained(model, prompt, schema, gen_kwargs)
        if result is not None:
            return StructuredOutcome(
                success=True, json_str=result[0], parsed=result[1],
                attempts=attempts, method="grammar",
            )
        last_error = "grammar-constrained decoding produced no valid object"

    # --- Strategy 2: provider-native JSON mode (one call) ---
    if _supports_native_json(model):
        attempts += 1
        json_str, parsed, raw, err = _native_json_call(model, prompt, schema, gen_kwargs)
        if json_str is not None:
            return StructuredOutcome(
                success=True, json_str=json_str, parsed=parsed,
                attempts=attempts, method="native",
            )
        last_error = err or "native JSON mode output did not match schema"
        last_raw = raw

    # --- Strategy 3: reprompt-with-validation fallback (time-boxed) ---
    for attempt in range(config.max_retries):
        if deadline is not None and time.monotonic() > deadline:
            last_error = f"time budget of {config.time_budget_s}s exhausted before a valid object"
            break
        attempts += 1
        json_str, parsed, raw, err = _reprompt_once(
            model, prompt, schema, gen_kwargs, attempt, last_error,
        )
        if json_str is not None:
            return StructuredOutcome(
                success=True, json_str=json_str, parsed=parsed,
                attempts=attempts, method="reprompt",
            )
        last_error = err
        if raw is not None:
            last_raw = raw

    final_error = last_error or "could not produce schema-valid output"
    # Teachable failure: a local Transformers model that couldn't be coaxed into
    # schema-valid JSON by prompting alone can be *guaranteed* to conform with
    # grammar-constrained decoding — but only if ``outlines`` is installed. When
    # it isn't, point the user at the one-line fix instead of leaving a dead end.
    if _is_transformers_model(model) and not _outlines_available():
        final_error += (
            " — this local model did not follow the JSON schema by prompting. "
            "Install grammar-constrained decoding (`pip install effgen[grammar]`) "
            "for guaranteed schema-valid output, or use a stronger "
            "instruction-following model."
        )
    return StructuredOutcome(
        success=False, attempts=attempts, method="reprompt",
        error=final_error,
        raw_text=last_raw,
    )


def constrain_output(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    config: StructuredOutputConfig | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """Generate structured output from a model, constrained to a JSON Schema.

    Thin wrapper over :func:`structured_generate` that preserves the historical
    contract: returns ``(json_string, parsed_data)`` on success and raises
    ``ValueError`` when no strategy yields a schema-valid object.

    Args:
        model: The model instance (BaseModel subclass).
        prompt: The user prompt.
        schema: JSON Schema the output must conform to.
        config: Optional StructuredOutputConfig.
        generation_kwargs: Extra kwargs passed to model.generate().

    Returns:
        (json_string, parsed_data) — guaranteed to validate against schema.

    Raises:
        ValueError: If output cannot be constrained after all attempts.
    """
    outcome = structured_generate(model, prompt, schema, config, generation_kwargs)
    if outcome.success and outcome.json_str is not None:
        return outcome.json_str, outcome.parsed
    raise ValueError(
        f"Failed to generate valid structured output after {outcome.attempts} "
        f"attempt(s). Last error: {outcome.error}"
    )


def _is_transformers_model(model: Any) -> bool:
    """Check if model is a local Transformers engine."""
    return type(model).__name__ in ("TransformersEngine",)


def _outlines_available() -> bool:
    """Whether the grammar API this module uses can actually be imported.

    Checks the concrete symbols ``_try_grammar_constrained`` relies on (not just
    that the package exists), so an incompatible ``outlines`` build is never
    counted as a real grammar attempt — keeping the surfaced attempt count accurate.
    Supports both the current ``outlines`` 1.x API (``from_transformers`` +
    ``Generator`` + ``types.JsonSchema``) and the legacy 0.0.x API
    (``generate.json`` + ``models.Transformers``).
    """
    try:
        import outlines  # noqa: F401
    except Exception:
        return False
    # outlines 1.x
    if hasattr(outlines, "from_transformers") and hasattr(outlines, "Generator"):
        try:
            from outlines.types import JsonSchema  # noqa: F401
        except Exception:
            return False
        return True
    # legacy outlines 0.0.x
    try:
        from outlines import generate, models  # noqa: F401
    except Exception:
        return False
    return True


# Adapters that speak the OpenAI chat-completions dialect and forward a
# ``response_format`` kwarg straight into the request — they all accept the
# universal ``{"type": "json_object"}`` JSON mode.
_OPENAI_COMPATIBLE_ADAPTERS = frozenset({
    "OpenAIAdapter", "GroqAdapter", "CerebrasAdapter",
    "TogetherAdapter", "FireworksAdapter",
})


def _is_api_model(model: Any) -> bool:
    """Check if model is an API adapter (kept for backward compatibility)."""
    return type(model).__name__ in ("OpenAIAdapter", "AnthropicAdapter", "GeminiAdapter")


def _supports_native_json(model: Any) -> bool:
    """Whether *model* exposes a provider-native JSON/structured mode."""
    name = type(model).__name__
    return name in _OPENAI_COMPATIBLE_ADAPTERS or name == "GeminiAdapter"


def schema_block(schema: dict[str, Any]) -> str:
    """The schema, rendered as the sentence and fenced JSON a model is shown.

    One rendering for every place a schema is stated to a model — the agent
    loop, which says it while the answer is still being written, and the
    JSON-mode and repair calls, which say it when the answer is due — so the
    two can never describe the same schema differently.
    """
    return (
        "Respond with a single JSON object matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```"
    )


def schema_answer_instruction(schema: dict[str, Any]) -> str:
    """State *schema* as the shape the run's final answer must take.

    Used inside a tool loop, where the model may still have tools to call
    before it answers, so it describes the answer rather than demanding one
    now. :func:`_schema_prompt` is the same statement for a call whose only
    job is to produce the object.
    """
    return (
        f"{schema_block(schema)}\n"
        "When you give the final answer, it must be that JSON object."
    )


def _schema_prompt(prompt: str, schema: dict[str, Any]) -> str:
    """Append a compact schema instruction to *prompt* for JSON-mode calls."""
    return (
        f"{prompt}\n\n"
        f"{schema_block(schema)}\n"
        f"Output ONLY the JSON object — no prose, no markdown fences."
    )


def _openai_strict_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a raw JSON-Schema dict as an OpenAI strict ``json_schema`` envelope.

    Reuses the same ``$ref`` inlining + ``additionalProperties: false`` enforcement
    that :func:`effgen.models.openai_schema.to_openai_schema` applies to Pydantic
    models, so an arbitrary user schema is accepted by OpenAI structured outputs.
    """
    from ..models.openai_schema import _enforce_additional_properties, _resolve_refs

    defs: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        if key in schema:
            defs.update({k: copy.deepcopy(v) for k, v in schema[key].items()})
    resolved = _resolve_refs(copy.deepcopy(schema), defs)
    strict_schema = _enforce_additional_properties(resolved)
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": strict_schema, "strict": True},
    }


def _native_json_call(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    gen_kwargs: dict[str, Any],
) -> tuple[str | None, Any, str | None, str | None]:
    """One provider-native JSON-mode call.

    Returns ``(json_str, parsed, raw_text, error)``. On success ``json_str`` is the
    validated JSON; otherwise it is ``None`` and ``raw_text``/``error`` carry the
    last model text and the reason, so the caller can fall back.
    """
    from ..models.base import GenerationConfig
    from .agent_runtime import resolve_output_budget

    name = type(model).__name__
    config = GenerationConfig(
        temperature=gen_kwargs.get("temperature", 0.2),
        max_tokens=resolve_output_budget(
            gen_kwargs.get("max_tokens"), None, model,
            output_schema=schema, default_base=2048,
        ),
    )
    enhanced = _schema_prompt(prompt, schema)
    text: str | None = None

    try:
        if name == "OpenAIAdapter" and hasattr(model, "generate_structured"):
            # Strongest path: strict json_schema (output is schema-guaranteed).
            try:
                rf = _openai_strict_response_format(schema)
                result = _ledger.timed_model_call(
                    model.generate_structured, _model_label(model),
                    prompt, response_format=rf, config=config,
                )
                text = result.text
            except Exception as e:
                # Strict mode rejects some schema features; fall back to json_object.
                logger.debug("OpenAI strict json_schema failed, trying json_object: %s", e)
                result = _ledger.timed_model_call(
                    model.generate, _model_label(model),
                    enhanced, config=config, response_format={"type": "json_object"},
                )
                text = result.text
        elif name in _OPENAI_COMPATIBLE_ADAPTERS:
            result = _ledger.timed_model_call(
                model.generate, _model_label(model),
                enhanced, config=config, response_format={"type": "json_object"},
            )
            text = result.text
        elif name == "GeminiAdapter":
            json_config = replace(
                config, response_mime_type="application/json", response_schema=schema,
            )
            result = _ledger.timed_model_call(
                model.generate, _model_label(model), enhanced, config=json_config,
            )
            text = result.text
    except Exception as e:
        return None, None, None, f"native JSON call failed: {e}"

    if not text:
        return None, None, text, "native JSON mode returned empty output"

    json_str = extract_json_from_text(text) or text.strip()
    try:
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        return None, None, text, f"native JSON mode output was not parseable: {e}"

    valid, err = validate_json_schema(parsed, schema)
    if valid:
        return json_str, parsed, text, None
    return None, None, text, f"native JSON output did not match schema: {err}"


def _reprompt_once(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    gen_kwargs: dict[str, Any],
    attempt: int,
    previous_error: str | None,
) -> tuple[str | None, Any, str | None, str | None]:
    """One reprompt-with-validation attempt (universal fallback for any model).

    Returns ``(json_str, parsed, raw_text, error)`` — ``json_str`` is non-None only
    when the produced JSON validates against *schema*.
    """
    from ..models.base import GenerationConfig
    from .agent_runtime import resolve_output_budget

    if attempt == 0:
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Respond ONLY with the JSON object."
        )
    else:
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Your previous response was not valid JSON. "
            f"Previous error: {previous_error}\n"
            f"You MUST respond with ONLY a valid JSON object matching:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"No explanation, no markdown, ONLY the JSON."
        )

    config = GenerationConfig(
        temperature=max(0.1, gen_kwargs.get("temperature", 0.3) - (attempt * 0.1)),
        max_tokens=resolve_output_budget(
            gen_kwargs.get("max_tokens"), None, model,
            output_schema=schema, default_base=2048,
        ),
    )

    try:
        result = _ledger.timed_model_call(
            model.generate, _model_label(model), enhanced_prompt, config=config,
        )
        text = result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        logger.debug("Reprompt attempt %d failed: %s", attempt + 1, e)
        return None, None, None, str(e)

    json_str = extract_json_from_text(text)
    if json_str is None:
        return None, None, text, "no JSON object found in response"
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, None, text, f"invalid JSON: {e}"

    valid, err = validate_json_schema(parsed, schema)
    if valid:
        return json_str, parsed, text, None
    return None, None, text, f"schema validation failed: {err}"


def _grammar_max_new_tokens(gen_kwargs: dict[str, Any]) -> int:
    """Token budget for one grammar-constrained call.

    Honours an explicit ``max_new_tokens``/``max_tokens`` from the caller so a
    long structured object isn't truncated mid-JSON; otherwise a sensible default
    that comfortably fits a typical extraction record.
    """
    for key in ("max_new_tokens", "max_tokens"):
        val = gen_kwargs.get(key)
        if isinstance(val, int) and val > 0:
            return val
    return 512


def _try_grammar_constrained(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    gen_kwargs: dict[str, Any],
) -> tuple[str, Any] | None:
    """Try grammar-constrained decoding via the ``outlines`` library.

    The output is structurally guaranteed to match *schema* in one call, so even
    weak instruction-followers (small Llama/Gemma/Phi locals) emit schema-valid
    JSON. Supports both the current ``outlines`` 1.x API and the legacy 0.0.x API;
    returns ``None`` (caller falls back to reprompting) if neither is usable.
    """
    # outlines drives the underlying HF model + tokenizer directly.
    if not (hasattr(model, "model") and hasattr(model, "tokenizer")):
        return None
    try:
        import outlines
    except ImportError:
        logger.debug("outlines not installed, skipping grammar-constrained decoding")
        return None
    try:
        logger.debug("Using outlines grammar-constrained decoding for structured output")
        if hasattr(outlines, "from_transformers") and hasattr(outlines, "Generator"):
            # outlines 1.x
            from outlines.types import JsonSchema

            om = outlines.from_transformers(model.model, model.tokenizer)
            generator = outlines.Generator(om, JsonSchema(schema))
            raw = generator(prompt, max_new_tokens=_grammar_max_new_tokens(gen_kwargs))
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            json_str = raw if isinstance(raw, str) else json.dumps(raw)
            return json_str, parsed
        # legacy outlines 0.0.x
        from outlines import generate
        from outlines import models as outlines_models

        hf_model = outlines_models.Transformers(model.model, model.tokenizer)
        generator = generate.json(hf_model, json.dumps(schema))
        result = generator(prompt)
        return json.dumps(result), result
    except Exception as e:
        logger.warning(f"Grammar-constrained decoding failed: {e}")
    return None


def pydantic_model_to_schema(model_class: Any) -> dict[str, Any]:
    """Convert a Pydantic model class to a JSON Schema dict.

    Strips Pydantic-specific ``title`` fields that confuse SLMs into
    echoing the schema instead of producing data.

    Args:
        model_class: A Pydantic BaseModel subclass.

    Returns:
        JSON Schema dict.

    Raises:
        TypeError: If the class is not a Pydantic model.
    """
    if hasattr(model_class, 'model_json_schema'):
        # Pydantic v2
        schema = model_class.model_json_schema()
    elif hasattr(model_class, 'schema'):
        # Pydantic v1
        schema = model_class.schema()
    else:
        raise TypeError(
            f"{model_class} is not a Pydantic model. "
            "It must have model_json_schema() (v2) or schema() (v1)."
        )
    _strip_title_fields(schema)
    return schema


def is_pydantic_model_class(obj: Any) -> bool:
    """Return True if *obj* is a Pydantic model **class** (v1 or v2).

    A Pydantic model class exposes ``model_json_schema()`` (v2) or ``schema()``
    (v1) as a callable on the class itself.
    """
    return isinstance(obj, type) and (
        hasattr(obj, "model_json_schema") or hasattr(obj, "schema")
    )


def normalize_output_schema(schema: Any) -> dict[str, Any] | None:
    """Coerce an ``output_schema`` argument into a JSON-Schema dict.

    ``output_schema`` is documented as a JSON-Schema ``dict``, but passing a
    Pydantic model *class* directly is one of the most natural things a user
    tries. Accept both instead of letting the class reach JSON serialization
    (which fails with a cryptic ``Object of type ModelMetaclass is not JSON
    serializable``).

    Accepts:
      - ``None`` → ``None``
      - a ``dict`` → returned as-is
      - a Pydantic ``BaseModel`` subclass (v1 ``schema()`` / v2
        ``model_json_schema()``) → converted via
        :func:`pydantic_model_to_schema`

    Raises:
        TypeError: for any other type, naming the accepted shapes.
    """
    if schema is None or isinstance(schema, dict):
        return schema
    if is_pydantic_model_class(schema):
        return pydantic_model_to_schema(schema)
    raise TypeError(
        "output_schema must be a JSON-Schema dict or a Pydantic model class, "
        f"not {type(schema).__name__}. Pass output_schema={{...}} (a dict) or "
        "output_schema=YourModel (a pydantic.BaseModel subclass)."
    )


def _strip_title_fields(schema: dict[str, Any]) -> None:
    """Remove ``title`` fields from a JSON Schema in-place.

    Pydantic v2 adds ``title`` to every property and the root schema.
    SLMs often echo these titles back as data, so we strip them.
    """
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
            # Recurse into nested objects / array items
            if "properties" in prop:
                _strip_title_fields(prop)
            items = prop.get("items")
            if isinstance(items, dict):
                items.pop("title", None)
                if "properties" in items:
                    _strip_title_fields(items)
