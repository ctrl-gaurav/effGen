"""Runtime helpers for :class:`effgen.core.agent.Agent`.

Pure/utility helpers extracted from ``agent.py`` for maintainability — answer
sanitization, task/preview coercion, JSON cleanup, the sync-over-async bridge,
and small provider/number utilities. This module imports nothing from
``agent.py`` so it stays a dependency-free leaf. Behaviour is identical to the
original in-class definitions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from ..observability import get_logger as _get_obs_logger
from ..utils.async_bridge import run_coroutine_sync
from ..utils.structured_logging import (
    get_structured_logger,
)
from .structured_output import _clean_json

if TYPE_CHECKING:
    from .messages import Message

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)

# Mirrors AgentConfig.system_prompt's default (effgen/core/agent.py) — kept as
# a literal here rather than imported to preserve this module's dependency-free
# leaf status.
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
IMAGE_GROUNDING_GUIDANCE = (
    "Describe images objectively — state only what is visible, and say so "
    "explicitly when a requested detail is not shown."
)

# Guardrail-chain signatures (sorted guardrail names) already warned about for
# missing TOOL_OUTPUT injection screening — a heads-up fires once per distinct
# configuration, not on every agent built with the same preset.
_tool_output_injection_gap_warned: set[tuple[str, ...]] = set()


_PROVIDER_BY_CLASS_PREFIX: dict[str, str] = {
    "OpenAI": "openai",
    "Cerebras": "cerebras",
    "Gemini": "google",
    "Anthropic": "anthropic",
    "Groq": "groq",
    "Together": "together",
    "Fireworks": "fireworks",
    "HFInference": "hf_inference",
    "Replicate": "replicate",
    "MLXVLM": "mlx_vlm",
    # The local engines. Without these a run on this machine's own GPU fell
    # through to the family-name guesses below and was attributed to whichever
    # cloud provider serves a model of that family — on-device work reported as
    # spend at a company the user never called.
    "TransformersEngine": "transformers",
    "VLLMEngine": "vllm",
    "GGUFEngine": "gguf",
    "MLXEngine": "mlx",
}

#: The engine prefixes a model id may carry (``transformers:Qwen/...``). Mirrors
#: ``ModelLoader._LOCAL_ENGINE_PREFIXES``; kept as a literal so this module
#: stays cheap to import.
_LOCAL_ENGINE_PREFIXES: tuple[str, ...] = ("transformers", "vllm", "gguf", "mlx")


def _infer_provider_from_model(model: Any, model_name: str | None = None) -> str:
    """
    Best-effort provider inference from a model instance.

    Order of preference:
    1. ``model.provider`` / ``model.provider_name`` attribute if present.
    2. Class-name prefix mapping (``OpenAIAdapter`` → ``openai``).
    3. Heuristic mapping from the model identifier string.
    4. ``"unknown"``.
    """
    if model is None:
        return "unknown"
    for attr in ("provider", "provider_name"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val:
            return val
    cls_name = type(model).__name__
    for prefix, provider in _PROVIDER_BY_CLASS_PREFIX.items():
        if cls_name.startswith(prefix):
            return provider
    m = (model_name or "").lower()
    # An explicit engine prefix is a statement, not a guess, so it settles the
    # question before any family-name heuristic gets to see the id. Otherwise
    # "transformers:Qwen/Qwen2.5-1.5B-Instruct" matched "qwen" and reported a
    # local run as cerebras, and the span name read "cerebras:transformers:...".
    for engine in _LOCAL_ENGINE_PREFIXES:
        if m.startswith(f"{engine}:"):
            return engine
    if m.startswith(("gpt-", "o1", "o3", "o4", "text-")):
        return "openai"
    if m.startswith(("gemini", "models/gemini")):
        return "google"
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if "llama" in m or "qwen" in m or m.startswith("cerebras"):
        return "cerebras"
    if m.startswith(("mixtral", "mistral")):
        return "groq"
    return "unknown"


def model_can_require_tool_call(model: Any) -> bool:
    """Whether *model* accepts a turn that requires a tool call.

    Asks the adapter, which is the only thing that knows how its provider
    shapes a request. Anything that does not answer — a duck-typed stand-in with
    no such method, an adapter whose probe raises — is read as "no": the loop
    then asks in words instead, which every model understands, rather than
    sending a constraint the provider may reject and losing the turn.

    Args:
        model: The loaded model, or ``None``.

    Returns:
        bool: True only when the adapter says the constraint reaches the
        provider in a form it honours.
    """
    if model is None:
        return False
    try:
        return bool(model.supports_forced_tool_call())
    except Exception:
        logger.debug("supports_forced_tool_call probe failed", exc_info=True)
        return False


def resolve_output_budget(
    per_call: int | None, configured: int | None, model: Any
) -> int:
    """Return the output-token budget a generation should use.

    One order, used by every path that generates: an explicit per-call value,
    then the agent's configured default, then the model's own default. The
    streaming paths used to skip the middle step, so an agent built with
    ``AgentConfig(max_tokens=333)`` got 333 tokens through ``run()`` and the
    model default through ``stream()`` — the same agent answering at two
    different lengths depending on which method was called.

    Args:
        per_call: ``max_tokens`` passed to this call, if any.
        configured: ``AgentConfig.max_tokens``, if set.
        model: The model, asked for its default as a last resort.

    Returns:
        The budget to send.
    """
    from ..models._adapter_utils import default_max_output_tokens

    if per_call is not None:
        return per_call
    if configured is not None:
        return configured
    return default_max_output_tokens(model)


def _safe_int_or_none(value: Any) -> int | None:
    """Convert numeric dashboard metadata values without raising."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float_or_none(value: Any) -> float | None:
    """Convert numeric dashboard metadata values without raising."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Loop-bookkeeping nudges the ReAct scaffolding injects into the scratchpad to
# steer a stalling model toward a final answer. They are defined here, in ONE
# place, and imported by the injection sites (see ``agent_react``) so the set of
# strings a loop can append can never drift out of sync with the set of strings
# stripped from the user-facing answer below. Adding a new nudge => add a
# constant here and reference it at the injection site; it is then stripped for
# free. The injection sites prepend "\n" (and sometimes "Observation: "), which
# the strip patterns tolerate.
NUDGE_CONTINUE = "[Tool results computed above. Continue or provide Final Answer:]"
NUDGE_HAVE_ANSWER = "[You have the answer from the tool. Please respond with 'Final Answer:' now.]"
NUDGE_HAVE_RESULTS = (
    "[You already have results from this tool. If you have "
    "enough information, respond now with 'Final Answer:'.]"
)
NUDGE_ALREADY_COMPUTED = (
    "You already computed this. Please provide your final response "
    "using 'Final Answer:' now."
)
NUDGE_NO_TOOLS = "No tools available. Please provide your answer directly using 'Final Answer:'."
# Closing sentence of the unknown-tool observation (built by
# ``unknown_tool_observation`` below). Kept as its own constant so the text and
# the pattern that strips it cannot drift apart.
UNKNOWN_TOOL_CLOSE = "Use one of them, or answer directly using 'Final Answer:'."
NUDGE_NOT_USABLE = (
    "That was not a usable answer. Call the tool correctly or give a "
    "plain Final Answer."
)
# Sent back to a model that answered without running a tool that does work it
# cannot do in its head. It names the tool, so unlike the nudges above it is a
# template rather than a fixed string, and the pattern that strips it is
# anchored on its two fixed halves (see ``_MUST_EXECUTE_RE``). ``{tool}`` is
# filled from the agent's own tool names and from nothing else.
NUDGE_MUST_EXECUTE = (
    "You have not run a tool yet. Run the {tool} tool and answer from what it "
    "returns, not from what you expect it to return."
)

# The single line that used to close the first prompt of a run whose tools
# reached the model through a local chat template, and only then. It is
# superseded on every path by the category-selected contract in
# :mod:`effgen.prompts.tool_contract`, which says what to do with the tools
# rather than only that they exist, and which reaches provider-side tool calling
# too. Kept defined at its published name so an importer of the 1.0.0 wording
# still resolves it.
TEMPLATE_TOOL_USE_INSTRUCTION = (
    "Use the tools you have been given when they apply to the task."
)

# Closes a tool prompt when the last observation was a computed result.
CONTINUE_INSTRUCTION = (
    "Continue solving the task. If you have the final answer, state it clearly."
)
# Closes a tool prompt when the last observation was retrieved source passages.
# The passages sit immediately above and read like a finished answer, so the
# close states what to do with them: answer from them, do not return them.
#
# It says nothing about the *form* of the answer. The framework describes the
# machinery it inserted -- an observation block the model did not ask for -- and
# leaves the shape of the answer to whoever knows the question: a declared
# ``output_schema``, then the task and the caller's system prompt. This line is
# the last thing the model reads, so a form demand here silently overrules a
# caller who asked for a letter, a number or a value, and buys several times the
# output tokens for an answer that was already stated.
CONTEXT_ANSWER_INSTRUCTION = (
    "The passages above are source material, not the answer. Use them to answer "
    "the question in the form the question asks for, and do not return a passage "
    "as the answer. If they do not answer it, say so and name what is missing."
)
# Appended to the line above only when the caller asked for inline citations
# (``cite_sources``) and the passages were presented to the model as a numbered
# list. Asking for markers changes what the answer *is* — a model that obeys
# ends a one-word answer with "[1]" — so it is never added on the caller's
# behalf, and never when there is no numbered list for a marker to point at.
CONTEXT_CITATION_INSTRUCTION = (
    "Cite each passage you used inline as [1], [2], ... numbered by its order "
    "above."
)

# Literal loop-bookkeeping strings to strip — every injectable nudge above, plus a
# couple of defensive bracketed/sub-phrase variants. These must never reach a
# user-facing answer. An adjacent newline is consumed when present so removing a
# mid-line marker doesn't leave an orphan line.
_SCAFFOLD_LITERALS: tuple[str, ...] = (
    NUDGE_CONTINUE,
    NUDGE_HAVE_ANSWER,
    NUDGE_HAVE_RESULTS,
    f"[{NUDGE_ALREADY_COMPUTED}]",
    NUDGE_ALREADY_COMPUTED,
    NUDGE_NO_TOOLS,
    NUDGE_NOT_USABLE,
    "Please provide your final response using 'Final Answer:' now.",
)
_SCAFFOLD_LITERAL_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(r"[ \t]*\n?[ \t]*" + re.escape(lit)) for lit in _SCAFFOLD_LITERALS
)
# The unknown-tool observation names the action and lists the agent's tools, so
# it cannot be a fixed literal. Match the whole sentence pair instead, anchored
# on both fixed ends, so no part of it can reach a user-facing answer.
_UNKNOWN_TOOL_OBS_RE = re.compile(
    r"[ \t]*\n?[ \t]*No tool named '[^\n]*?' is available\.[ \t]*"
    r"The tools you can use are:[^\n]*?" + re.escape(UNKNOWN_TOOL_CLOSE)
)

# ``NUDGE_MUST_EXECUTE`` names a tool in its middle, so it is a template and not
# a fixed literal. Anchor on both fixed halves — the ``[^\n]*?`` covers a real
# tool name and the unrendered ``{tool}`` alike, so the constant as written and
# the constant as sent are both stripped.
_MUST_EXECUTE_RE = re.compile(
    r"[ \t]*\n?[ \t]*"
    + re.escape(NUDGE_MUST_EXECUTE.split("{tool}")[0])
    + r"[^\n]*?"
    + re.escape(NUDGE_MUST_EXECUTE.split("{tool}")[1])
)

# A line-anchored "Final Answer:" / "Answer:" label (allows quote/list prefixes).
#
# The label is often written in markdown emphasis — "**Answer:**", "__Answer:__",
# "### Answer:". The closing marker is only swallowed when it matches the one
# that opened the label, so a genuinely bold answer ("Answer: **42**") keeps
# both of its markers instead of losing the opening one and rendering the rest
# of the reply in bold.
_ANSWER_LABEL_RE = re.compile(
    r"(?:^|\n)[ \t>\-]*(\*{1,2}|_{1,2}|#{1,6})?[ \t]*"
    r"(?:final[ \t]*answer|answer)[ \t]*[:\-][ \t]*(?:\1)?[ \t]*",
    re.IGNORECASE,
)
# Trailing ReAct bleed: an Observation/Thought/Question/Action section the model
# appended after its real answer.
_TRAILING_BLEED_RE = re.compile(
    r"\n[ \t>*\-]*(?:observation|thought|question|action(?:[ \t]+input)?)[ \t]*:.*\Z",
    re.IGNORECASE | re.DOTALL,
)
# Tool-echo fragment like "[calculator({'expression': '15*15'})] → 225". The
# scaffolding always emits the Unicode arrow (see the f-strings in the ReAct/
# native loops); we deliberately do NOT match an ASCII "->" here so legitimate
# prose like "see [1] -> next" is never corrupted.
_TOOL_ECHO_RE = re.compile(r"\[[^\[\]\n]*\][ \t]*→[ \t]*")

# A brace-delimited JSON object allowing up to two levels of nesting, so a tool
# call with structured args ("{\"args\": {\"x\": 1}}") is matched whole rather
# than leaving a dangling "}" behind.
_JSON_OBJ = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
# Model tool-call syntax that some families (e.g. Llama 3.1) emit as plain text
# when a native tool call isn't routed: "<function=calc>{...}</function>",
# "<tool_call>{...}</tool_call>", and stray special tokens. These are pure
# scaffolding and must never surface in an answer.
_TOOLCALL_CONSTRUCT_RE = re.compile(
    r"<?\s*(?:function\s*=\s*[\w.\-]+|tool_call)\s*>?\s*" + _JSON_OBJ
    + r"\s*(?:</\s*(?:function|tool_call)\s*>)?",
    re.IGNORECASE | re.DOTALL,
)
_TOOLCALL_TAG_RE = re.compile(
    r"</?\s*(?:function(?:\s*=\s*[\w.\-]+)?|tool_call|invoke|function_call)\s*>"
    r"|<\|[^>|]*\|>",
    re.IGNORECASE,
)
# The same call written as nested tags rather than as JSON —
# "<function=calculator><parameter=expression>2+2</parameter></function>".
# The construct regex above needs a JSON object and finds nothing here, and
# stripping the tags alone would leave the argument values standing in the
# answer as loose prose ("calculate 2+2"), so the whole construct goes.
_XML_TAG_CALL_RE = re.compile(
    r"<(?P<tag>function|tool_call|invoke|tool|function_call)"
    r"(?:=|\s+name\s*=\s*[\"']?)[\w.\-]+[\"']?\s*>"
    r".*?</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
# The same construct with no closing tag, because the token budget ran out
# mid-call. It runs to the end of the text, so it is only applied when an
# argument tag is actually present — otherwise an answer that merely mentions
# "<function=name>" would lose everything after the mention.
_XML_TRUNCATED_CALL_RE = re.compile(
    r"<(?:function|tool_call|invoke|tool|function_call)"
    r"(?:=|\s+name\s*=\s*[\"']?)[\w.\-]+[\"']?\s*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
# An argument tag left behind by a construct that was cut short. The opening
# form has to name its argument; the closing form is bare.
_XML_ARG_TAG_RE = re.compile(
    r"<(?:parameter|param|argument|arg)"
    r"(?:\s*=\s*[\w.\-]+|\s+name\s*=\s*[\"'][\w.\-]+[\"'])\s*>"
    r"|</\s*(?:parameter|param|argument|arg)\s*>",
    re.IGNORECASE,
)
# A leaked tool call with no wrapping tag at all — just the bare tool name
# followed by its JSON arguments, e.g.
# 'order_lookup {"order_id": "ORD-1001"} \nThe order status is "shipped"' or,
# with no separator at all, 'get_order{"order_id": "ORD-1001"}'.
# Anchored to the very start of the text (never mid-sentence) so an ordinary
# answer that happens to contain "word {...}" is never touched; a tool name
# is always a lowercase snake_case identifier by effGen convention. Without a
# separating space the brace block must additionally look like a JSON argument
# object (a quoted key or an empty object), so prose or code that opens with
# something like "body{color:red}" is left alone.
#
# A model that starts to emit a tag and abandons it leaves a stray angle bracket
# on the same shape ('<wikipedia {"query": "x"}', sometimes closed with '>').
# That form is stripped too, but only when the braces look like a JSON argument
# object, so a template or markup answer opening with '<template {{ x }}>' is
# left alone.
_LEADING_TOOLCALL_ECHO_RE = re.compile(
    r"^(?:"
    r"<[a-z_][a-z0-9_]{1,63}[ \t]*(?=\{[ \t\r\n]*[\"}])"
    r"|[a-z_][a-z0-9_]{1,63}(?:[ \t]+|(?=\{[ \t\r\n]*[\"}]))"
    r")" + _JSON_OBJ + r">?[ \t\n]*"
)

# Fenced and inline code spans are stripped before scanning an answer for a
# written-out tool call, so an answer that *documents* a call ("run `calculator
# {\"expression\": \"2+2\"}`") is never mistaken for one the model tried to make.
_FENCED_CODE_RE = re.compile(r"```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# A tool name immediately followed by its JSON argument object, with or without
# the angle-bracket tag or ``function=`` prefix a family may wrap it in:
# ``<file_operations> {"operation": …}``, ``<function=calculator>{…}``,
# ``calculator {"expression": …}``. The name is only treated as a call when it
# matches a tool the agent actually has, so ordinary prose never matches.
_WRITTEN_CALL_RE = re.compile(
    r"(?:\A|(?<=[\s>*`\-]))"
    r"<?[ \t]*(?:function[ \t]*=[ \t]*)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.\-]{1,63})"
    r"[ \t]*>?[ \t\r\n]*"
    r"(?=\{[ \t\r\n]*[\"}])"
)
# The tagged shapes carry the tool name inside the JSON object instead.
_TAGGED_CALL_RE = re.compile(
    r"<tool_call>|\[TOOL_CALLS\]|<\|python_tag\|>", re.IGNORECASE,
)
_CALL_NAME_FIELD_RE = re.compile(r"\"(?:name|function)\"[ \t]*:[ \t]*\"([\w.\-]+)\"")
# A third shape, seen live on groq's 8B: the tag is there but the body is a
# query string rather than JSON, sometimes HTML-escaped —
# ``<file_operations>operation=write&path=greet.py&content=&quot;...&quot;``.
# The JSON readers above find nothing in it, so the turn used to be reported as
# a successful answer with no file written. The tag must be angle-bracketed and
# immediately followed by ``word=``, so a sentence that merely mentions a tool
# in angle brackets is not flagged.
_TAGGED_KV_CALL_RE = re.compile(
    r"<(?P<name>[A-Za-z_][A-Za-z0-9_.\-]{1,63})>[ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*="
)


def unknown_tool_observation(action: str, tool_names: list[str] | tuple[str, ...]) -> str:
    """Observation for an action that names no tool the agent holds.

    The agent has tools here — the name simply did not resolve, usually because
    the model invented one or wrapped it in extra text. Saying "no tools
    available" would be false and steers a recoverable step into answering from
    memory, so the observation names the action that failed and lists what is
    actually callable.

    Args:
        action: The unresolved tool name as the model wrote it.
        tool_names: Names of the tools the agent holds.

    Returns:
        str: The observation text to append to the scratchpad.
    """
    available = ", ".join(tool_names)
    return (
        f"No tool named '{action}' is available. "
        f"The tools you can use are: {available}. {UNKNOWN_TOOL_CLOSE}"
    )


# A call in Python call syntax — ``file_operations('write', 'greet.py', …)``.
# Deliberately **not** matched wherever it appears: ``name(...)`` is ordinary
# prose and ordinary code, and a coding agent is frequently *asked* to write or
# explain a call. It is only a written-out call when it stands alone as the
# whole answer, which is the rule below.
_PAREN_CALL_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.\-]{1,63})[ \t]*"
    r"\((?P<args>[^()]*(?:\([^()]*\)[^()]*)*)\)"
)


def _standalone_paren_call(scan: str, names: set[str]) -> str | None:
    """Return the tool named by a paren call that *is* the whole answer.

    With every such call removed, an answer that merely mentions one still has
    words left; an answer that is nothing but calls does not. That separates
    "the model is emitting a call" from "the model is writing about a call",
    which is the distinction the looser JSON shapes get for free from having a
    JSON object to anchor on.
    """
    remainder: list[str] = []
    position = 0
    found: str | None = None
    for match in _PAREN_CALL_RE.finditer(scan):
        if match.group("name") not in names:
            continue
        remainder.append(scan[position:match.start()])
        position = match.end()
        if found is None:
            found = match.group("name")
    if found is None:
        return None
    remainder.append(scan[position:])
    return found if not re.search(r"[A-Za-z]{3,}", "".join(remainder)) else None


def find_written_tool_call(text: str | None, tool_names: Any) -> str | None:
    """Return the tool whose call *text* writes out as prose, or ``None``.

    A model that is offered tools but does not emit a call the runtime can
    dispatch sometimes writes the call into its answer instead — a block such as
    ``<file_operations> {"operation": "write", …}``. Nothing runs, yet the text
    reads like work was done. This reports the name of the first such block that
    matches a tool in *tool_names*, so the caller can report the turn as failed
    rather than as an answer.

    Fenced and inline code spans are ignored, and the name has to be one the
    agent actually holds, so an answer that explains or documents a tool call is
    not flagged.
    """
    if not text or not isinstance(text, str) or not tool_names:
        return None
    names = set(tool_names)
    scan = _INLINE_CODE_RE.sub(" ", _FENCED_CODE_RE.sub(" ", text))
    for match in _WRITTEN_CALL_RE.finditer(scan):
        if match.group("name") in names:
            return match.group("name")
    # A tagged call the parser could not read (truncated or malformed JSON)
    # still names its tool in a "name" field.
    if _TAGGED_CALL_RE.search(scan):
        for field_match in _CALL_NAME_FIELD_RE.finditer(scan):
            if field_match.group(1) in names:
                return field_match.group(1)
    # A tagged call whose body is a query string rather than JSON.
    for kv_match in _TAGGED_KV_CALL_RE.finditer(scan):
        if kv_match.group("name") in names:
            return kv_match.group("name")
    # A call written as nested tags, which carries no JSON for the readers above.
    from .tool_calling import _xml_parameter_call

    xml_call = _xml_parameter_call(scan)
    if xml_call is not None and xml_call[0] in names:
        return xml_call[0]
    # Python call syntax, only when it is the entire answer.
    return _standalone_paren_call(scan, names)


def _json_object_end(text: str, start: int) -> int:
    """Index just past the JSON object opening at *start*, or ``len(text)``.

    String-aware, so a brace inside a quoted value does not close the object.
    An object that never closes (a call cut off by the token cap) runs to the
    end of the text.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def written_call_only(text: str | None, tool_names: Any) -> bool:
    """True when *text* is nothing but written-out calls for the agent's tools.

    Distinguishes a call block the model returned *as* its answer from an answer
    that recaps a call it really made: with the call blocks and the tags around
    them removed, an answer still has words left, a bare block does not.
    """
    if not find_written_tool_call(text, tool_names):
        return False
    names = set(tool_names)
    source = text or ""
    remainder: list[str] = []
    position = 0
    while position < len(source):
        match = next(
            (
                m
                for m in _WRITTEN_CALL_RE.finditer(source, position)
                if m.group("name") in names
            ),
            None,
        )
        if match is None:
            remainder.append(source[position:])
            break
        remainder.append(source[position:match.start()])
        position = _json_object_end(source, match.end())
    left = _TOOLCALL_TAG_RE.sub(" ", "".join(remainder))
    left = re.sub(r"</?[^<>\n]{0,80}>", " ", left)
    return not re.search(r"[A-Za-z]{3,}", left)

# Gemma 4 emits reasoning inside <|channel>...<channel|> (asymmetric delimiters,
# unlike the <|...|> tags above) and puts the user-facing answer *after* the
# closing tag. These strip the reasoning so it never surfaces as the answer.
_GEMMA_CHANNEL_CLOSE_RE = re.compile(r"<channel\|>")
_GEMMA_DANGLING_CHANNEL_RE = re.compile(r"<\|channel>.*$", re.DOTALL)
_GEMMA_TOOLCALL_RE = re.compile(r"<\|tool_call>.*?(?:<tool_call\|>|$)", re.DOTALL)
_GEMMA_STRAY_TOKEN_RE = re.compile(
    r"<\|?(?:channel|turn|think|tool|tool_call|tool_response|image|audio|video)\|?>"
)


def sanitize_final_answer(text: str | None) -> str | None:
    """Strip internal ReAct/tool scaffolding from a user-facing answer.

    Conservative and idempotent: removes the loop-bookkeeping strings the agent
    injects, leading ``Final Answer:``/``Answer:`` labels (returning the text
    after the *last* such label, so ``"Canberra\\nFinal Answer: Canberra"`` →
    ``"Canberra"``), trailing ``Observation:``/``Thought:``/``Question:`` bleed,
    ``[tool(args)] →`` echo prefixes, tagged tool-call syntax
    (``<function=x>{...}</function>``, ``<tool_call>{...}</tool_call>``), and a
    leading untagged tool-call echo (``tool_name {"arg": "val"}`` at the very
    start of the text, before the real answer) — without reflowing markdown
    tables, code blocks, or legitimate pipe-separated content. ``None``/non-``str``
    input is returned unchanged.
    """
    if not text or not isinstance(text, str):
        return text
    s = text
    # 1. Remove literal loop-bookkeeping strings (with any adjacent newline).
    for pat in _SCAFFOLD_LITERAL_RES:
        s = pat.sub("", s)
    # 1b. Remove the two nudges whose middle varies per agent: the unknown-tool
    # observation, which lists the agent's tools, and the execute nudge, which
    # names one of them.
    s = _UNKNOWN_TOOL_OBS_RE.sub("", s)
    s = _MUST_EXECUTE_RE.sub("", s)
    # 2. Remove tool-echo prefixes, keeping the result after the arrow.
    s = _TOOL_ECHO_RE.sub("", s)
    # 2b. Remove leaked model tool-call syntax (whole constructs, then stray tags).
    s = _TOOLCALL_CONSTRUCT_RE.sub("", s)
    s = _XML_TAG_CALL_RE.sub("", s)
    if _XML_ARG_TAG_RE.search(s):
        # An argument tag survived the closed-construct pass, so a call was cut
        # short. Only then is it safe to drop the rest of the text.
        s = _XML_TRUNCATED_CALL_RE.sub("", s)
        s = _XML_ARG_TAG_RE.sub("", s)
    s = _TOOLCALL_TAG_RE.sub("", s)
    # 2c. Remove a leading bare "tool_name {json}" echo with no wrapping tag.
    s = _LEADING_TOOLCALL_ECHO_RE.sub("", s)

    # 2d. Gemma 4 channel format: reasoning is wrapped in <|channel>...<channel|>
    #     with the answer after the close tag. Keep only the tail after the last
    #     close, drop any unclosed (truncated) trailing channel, then clear stray
    #     channel/tool special tokens. Only fires when the markers are present.
    if "<|channel>" in s or "<channel|>" in s or "<|tool_call>" in s:
        closes = list(_GEMMA_CHANNEL_CLOSE_RE.finditer(s))
        if closes:
            s = s[closes[-1].end():]
        s = _GEMMA_DANGLING_CHANNEL_RE.sub("", s)
        s = _GEMMA_TOOLCALL_RE.sub("", s)
        s = _GEMMA_STRAY_TOKEN_RE.sub("", s)
    # 3. Drop trailing Observation/Thought/Question/Action bleed.
    s = _TRAILING_BLEED_RE.sub("", s)
    # 4. If a line-anchored answer label is present, the real answer is what
    #    follows the LAST such label (when that tail is non-empty). A dangling
    #    label with nothing after it (e.g. native web-search replies sometimes
    #    end with a bare "Final Answer:") is scaffolding — drop the label and
    #    keep the content before it.
    labels = list(_ANSWER_LABEL_RE.finditer(s))
    if labels:
        tail = s[labels[-1].end():].strip()
        if tail:
            s = tail
        else:
            head = s[:labels[-1].start()].strip()
            if head:
                s = head
    # 5. Tidy separators left by removed scaffolding, without disturbing
    #    multi-line content (tables/code): collapse empty "| |" fragments and
    #    runs of spaces, then strip a dangling leading/trailing bare pipe.
    s = re.sub(r"\|[ \t]*\|", "|", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.strip()
    # Strip a dangling leading/trailing bare pipe only for single-line content,
    # so markdown tables (multi-line, pipe-delimited) are left intact.
    if "\n" not in s:
        s = re.sub(r"^\|[ \t]*", "", s)
        s = re.sub(r"[ \t]*\|$", "", s)
    return s.strip()


# A run of reference markers at the very END of an answer: "C [1]",
# "D [1], [2]", "D [1][2][3]", "C [1] , [2] ". Trailing only — a bracketed
# number inside a sentence ("...are called dormant [1].") is part of what the
# model wrote and is left alone.
_TRAILING_CITATION_RUN_RE = re.compile(r"(?:\s*\[\d+\]\s*,?)+\s*$")
_CITATION_INDEX_RE = re.compile(r"\[(\d+)\]")


def _strip_run_citation_markers(
    text: str | None, *, passages: int = 0
) -> tuple[str | None, int]:
    """Remove a trailing run of ``[n]`` reference markers from an answer.

    A pure function, kept apart from :func:`sanitize_final_answer` so the
    sanitizer stays a ``(str) -> str`` transform with no run state in it. The
    caller owns the two facts this cannot know: whether the run retrieved
    context at all, and whether the caller asked for inline citations. It is
    only correct to call when the run did retrieve and the caller did not ask.

    ``passages`` is how many passages the run's retrieval observations offered.
    When it is non-zero every index in the marker run must fall inside it, so an
    answer ending in ``[9]`` after three passages keeps its bracket — that is a
    number the answer is about, not a reference to a passage. When the
    observations offered nothing countable, the bound is not applied.

    Returns the text and how many markers were removed; ``(text, 0)`` means
    nothing changed. An answer that is *only* markers is returned untouched,
    since removing them would leave no answer at all.
    """
    if not text or not isinstance(text, str):
        return text, 0
    match = _TRAILING_CITATION_RUN_RE.search(text)
    if not match:
        return text, 0
    indices = [int(i) for i in _CITATION_INDEX_RE.findall(match.group(0))]
    if not indices:
        return text, 0
    if passages and any(i < 1 or i > passages for i in indices):
        return text, 0
    stripped = _TRAILING_CITATION_RUN_RE.sub("", text).strip()
    if not stripped:
        return text, 0
    return stripped, len(indices)


class AgentRuntimeMixin:
    """Pure/stateless helper methods mixed into :class:`Agent`."""

    if TYPE_CHECKING:
        # Contributed by :class:`~effgen.core.agent.Agent` and the ReAct mixin,
        # which own the per-run state these helpers read. Declared for the type
        # checker only — at run time they arrive through the MRO, and these
        # statements do not execute.
        config: Any
        tools: dict[str, Any]

        def _citation_prompt_state(self) -> tuple[bool, int]: ...
        def _compose_closing(self, answer_shape: str, closing: str) -> str: ...
        def _answer_shape_instruction(self) -> str: ...
        def _continuation_instruction(
            self,
            previous_actions: list[tuple[str, str]],
            *,
            cite_sources: bool = False,
            numbered_passages: int = 0,
        ) -> str: ...

    @staticmethod
    def _resolve_guardrails(guardrails: Any):
        """Resolve guardrails config to a GuardrailChain or None."""
        if guardrails is None:
            return None
        # Already a GuardrailChain
        from ..guardrails.base import GuardrailChain
        if isinstance(guardrails, GuardrailChain):
            return guardrails
        # Preset name string
        if isinstance(guardrails, str):
            from ..guardrails.presets import get_guardrail_preset
            return get_guardrail_preset(guardrails)
        return None

    @staticmethod
    def _warn_tool_output_injection_gap(guardrail_chain: Any, has_tools: bool) -> None:
        """Warn once when a tool-attached agent's guardrails skip TOOL_OUTPUT
        injection screening.

        An indirect prompt injection carried in a tool's return value (a
        scraped page, a ticket body, a retrieved document) reaches the model
        unscreened unless the chain's ``PromptInjectionGuardrail`` explicitly
        covers ``GuardrailPosition.TOOL_OUTPUT``. Logged once per distinct
        guardrail configuration at the point a tool-bearing agent is built, so
        the gap is visible at the point of choice rather than only in a
        preset's docstring.
        """
        if not has_tools or guardrail_chain is None:
            return
        from ..guardrails.base import GuardrailPosition
        from ..guardrails.injection import PromptInjectionGuardrail

        injection_guardrails = [
            g for g in guardrail_chain.guardrails
            if isinstance(g, PromptInjectionGuardrail) and g.enabled
        ]
        if not injection_guardrails:
            return
        if any(GuardrailPosition.TOOL_OUTPUT in g.positions for g in injection_guardrails):
            return

        sig = tuple(sorted(g.name for g in guardrail_chain.guardrails))
        if sig in _tool_output_injection_gap_warned:
            return
        _tool_output_injection_gap_warned.add(sig)
        logger.warning(
            "This agent has tools attached, but its guardrail configuration "
            "does not screen a tool's return value for prompt injection — an "
            "instruction embedded in a scraped page, a ticket body, or a "
            "retrieved document reaches the model unscreened. The 'strict' "
            "and 'phi' guardrail presets screen tool output too; pass "
            "PromptInjectionGuardrail(positions=[GuardrailPosition.INPUT, "
            "GuardrailPosition.TOOL_OUTPUT]) directly for a custom chain."
        )

    def _humanize_observation(self, obs: str) -> str:
        """
        Render a retrieval/search tool observation as readable passage text.

        When a knowledge-base tool's raw result (a ``{'results': [...]}`` dict)
        ends up being used as a fallback answer — e.g. a model that loops on the
        retrieval tool instead of synthesizing — dumping the Python dict repr is
        unreadable. Extract and join the passage ``content`` instead so the
        degraded answer is at least the retrieved evidence (citations are still
        attached separately to the response).
        """
        s = obs.strip()
        brace = s.find("{")
        if brace == -1 or "results" not in s:
            return obs
        # Extract a single balanced-brace object starting at the first '{' so
        # trailing scaffolding (e.g. an appended "[Tool results computed…]" nudge
        # or a following "Thought:") doesn't defeat the parse.
        depth = 0
        end = -1
        for i in range(brace, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = s[brace:end] if end != -1 else s[brace:]
        data: Any = None
        try:
            import ast
            data = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                return obs
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            passages: list[str] = []
            for item in data["results"]:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("text") or item.get("snippet")
                    if content:
                        passages.append(str(content).strip())
            if passages:
                return "\n\n".join(passages)
        return obs

    @staticmethod
    def _run_coroutine_sync(coro, timeout: float = 120.0):
        """
        Run an async coroutine from synchronous code.

        Thin wrapper over the shared :func:`effgen.utils.run_coroutine_sync`
        so every sync-to-async bridge in the codebase behaves identically:
        drive the coroutine directly when no loop is running, otherwise run it
        on a worker thread and block — never skip it.

        Args:
            coro: The coroutine to run.
            timeout: Maximum seconds to wait (default 120s).
        """
        return run_coroutine_sync(coro, timeout=timeout)

    @staticmethod
    def _clean_json_input(raw: str) -> str:
        """
        Clean malformed JSON commonly produced by SLMs before parsing.

        Handles:
        - Markdown-wrapped JSON (```json ... ```)
        - Trailing commas  ({"key": "val",})
        - Unquoted keys    ({expression: "2+2"})

        Argument values are left untouched: a repair applies only outside string
        literals, so a query like ``"Paris, France: population"`` reaches the
        tool exactly as the model wrote it.

        Returns the cleaned string (still needs json.loads).
        """
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            # Remove opening fence (with optional language tag) and closing fence
            text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            text = text.strip()

        return _clean_json(text)

    @staticmethod
    def _sanitize_tool_input(tool_input: str, max_length: int = 10000) -> str:
        """
        Sanitize tool input by stripping control characters and limiting length.

        Args:
            tool_input: Raw input string.
            max_length: Maximum allowed input length.

        Returns:
            Sanitized input string.
        """
        if not tool_input:
            return tool_input
        # Strip control characters except newline and tab
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', tool_input)
        # Limit length
        if len(sanitized) > max_length:
            logger.warning(f"Tool input truncated from {len(sanitized)} to {max_length} chars")
            sanitized = sanitized[:max_length]
        return sanitized

    @staticmethod
    def _extract_task_preview(task: Any, limit: int = 200) -> str:
        """Return a safe, short string preview of a ``run()`` task argument.

        ``task`` may be a ``str``, a ``Message``, or a ``list[ContentPart]`` —
        never assume it is subscriptable.
        """
        try:
            if isinstance(task, str):
                return task[:limit]
            from effgen.core.messages import Message, TextPart

            if isinstance(task, Message):
                return task.text[:limit]
            if isinstance(task, list | tuple):
                texts = [p.text for p in task if isinstance(p, TextPart)]
                if texts:
                    return " ".join(texts)[:limit]
                return f"<{len(task)} content part(s)>"
        except Exception:
            logger.debug("Failed to build task preview", exc_info=True)
        return str(task)[:limit]

    @staticmethod
    def _coerce_task_input(
        task: Any, inputs: Any = None,
    ) -> tuple[str, list[Any] | None]:
        """Normalise a ``run()``/``stream()`` task into ``(text, inputs)``.

        Accepts ``str``, a multimodal ``Message``, or a ``list[ContentPart]``.
        Text parts become the task string; image/audio/video parts are routed to
        the multimodal ``inputs`` path (merged with any explicit ``inputs=``).
        Any other type raises a clear ``TypeError``.
        """
        from effgen.core.messages import (
            _CONTENT_PART_TYPES,
            Message,
            TextPart,
        )

        # Fast path: the overwhelmingly common case.
        if isinstance(task, str):
            return task, inputs

        if isinstance(task, Message):
            parts: list[Any] = list(task.content)
        elif isinstance(task, list | tuple):
            parts = list(task)
        else:
            raise TypeError(
                "Agent.run(task=...) accepts a str, a Message, or a "
                "list[ContentPart]; got "
                f"{type(task).__name__}. For images/audio/video pass text plus "
                "inputs=[image_from(...)], e.g. "
                'agent.run("describe this", inputs=[image_from(path)]).'
            )

        texts: list[str] = []
        media: list[Any] = []
        for index, part in enumerate(parts):
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, TextPart):
                texts.append(part.text)
            elif isinstance(part, _CONTENT_PART_TYPES):
                media.append(part)
            else:
                raise TypeError(
                    f"Agent.run(task=...) content item {index} must be a "
                    "ContentPart (image_from/audio_from/video_from) or str; got "
                    f"{type(part).__name__}."
                )

        text_task = "\n".join(t for t in texts if t)

        merged: list[Any] = list(media)
        if inputs is not None:
            merged.extend(inputs if isinstance(inputs, list | tuple) else [inputs])

        return text_task, (merged or None)

    def _build_multimodal_prompt(self, task: str, inputs: Any) -> list[Any]:
        """Build structured Messages for adapter-native multimodal input."""
        from effgen.core.messages import ContentPart, Message, Role, TextPart

        if isinstance(inputs, tuple):
            inputs = list(inputs)
        elif not isinstance(inputs, list):
            inputs = [inputs]

        from pathlib import Path as _Path

        content: list[ContentPart] = [TextPart(text=task)]
        for index, part in enumerate(inputs):
            # Convenience: a bare path or URL string (or Path) is auto-wrapped
            # by extension into the matching image/audio/video part, so
            # inputs=["photo.png"] works without importing image_from().
            if isinstance(part, str | _Path):
                from effgen.core.multimodal import _document_extension, part_from
                from effgen.errors import InvalidMultimodalContent
                try:
                    part = part_from(part)
                except InvalidMultimodalContent as exc:
                    # A recognized document type already carries a complete hint
                    # pointing at RAG ingestion / the pdf/excel tools; appending
                    # the image_from import line would only send the user back to
                    # the media wrappers that do not fit a document.
                    if _document_extension(str(part)):
                        raise TypeError(
                            f"Agent.run(inputs=[...]) item {index}: {exc}"
                        ) from exc
                    raise TypeError(
                        f"Agent.run(inputs=[...]) item {index}: {exc}. "
                        "Import the helper with: from effgen import image_from"
                    ) from exc
            elif not hasattr(part, "type"):
                raise TypeError(
                    "Agent.run(inputs=[...]) expects effGen multimodal parts "
                    f"(image_from/audio_from/video_from); item {index} is {type(part).__name__}. "
                    "Import them with: from effgen import image_from, audio_from, video_from"
                )
            content.append(part)

        messages: list[Message] = []
        if self.config.system_prompt:
            system_text = self.config.system_prompt
            # The unconfigured default persona carries no grounding guidance
            # for images, unlike the multimodal preset's system prompt. Add
            # the same "describe only what's visible" discipline here so a
            # plain Agent(inputs=[image_from(...)]) call gets it too, without
            # touching a caller's own custom system_prompt.
            if system_text == DEFAULT_SYSTEM_PROMPT and any(
                getattr(part, "type", None) == "image" for part in content
            ):
                system_text = f"{system_text} {IMAGE_GROUNDING_GUIDANCE}"
            messages.append(Message(role=Role.SYSTEM, content=[TextPart(text=system_text)]))
        messages.append(Message(role=Role.USER, content=content))
        return messages

    def _persona_prefix(self) -> str:
        """Return the user's custom persona as a prompt prefix, or ``""``.

        When the user set a custom ``system_prompt`` (anything other than the
        default assistant prompt), the no-tool direct path and the native/hybrid
        tool path prepend it to the user turn so the persona actually steers the
        model — matching the ReAct-text and Gemini-native paths, which already
        embed it. Adapters that take a string prompt have no separate system
        slot, so prepending is the universal, family-agnostic way to deliver it.
        Returns an empty string for the default persona so default agents are
        byte-for-byte unchanged.
        """
        persona = getattr(self, "_custom_persona", None)
        return f"{persona}\n\n" if persona else ""

    def _tool_contract(self) -> str:
        """What this agent tells the model about the tools it holds, or ``""``.

        The text comes from the tools' declared categories
        (:func:`effgen.prompts.tool_contract.select_tool_contract`), so a
        calculator, a code executor and a search tool are each described for
        what they are. An agent holding no tools gets ``""`` and its prompt is
        unchanged.

        ``AgentConfig.tool_contract`` overrides the selection: any text is
        stated verbatim in the contract's position, and ``""`` states nothing at
        all, which is the way to keep a prompt free of the framework's own
        sentences without rebuilding the whole template.
        """
        declared = getattr(getattr(self, "config", None), "tool_contract", None)
        if declared is not None:
            if declared:
                logger.info("tool contract: caller-supplied")
            return str(declared)
        from ..prompts.tool_contract import select_tool_contract
        return select_tool_contract((getattr(self, "tools", None) or {}).values())

    def _native_tool_prompt(
        self, task: str, scratchpad: str, conversation_history: str,
        previous_actions: list[tuple[str, str]],
    ) -> str:
        """Build one turn's prompt for the native/hybrid tool path.

        Both the blocking loop and the streamed one call this, so the two ask
        the same model the same question for the same agent instead of
        assembling the prompt twice and drifting apart.

        The tool definitions travel outside this string — through the provider's
        tool-calling API or the chat template — so the prompt itself is the only
        place the tools can be described. The contract goes on the opening turn,
        where the model first sees them; later turns close with a continuation
        or retrieval instruction of their own, which a second statement would
        compete with.
        """
        cite_sources, numbered_passages = self._citation_prompt_state()
        closing = self._compose_closing(
            self._answer_shape_instruction(),
            self._continuation_instruction(
                previous_actions,
                cite_sources=cite_sources,
                numbered_passages=numbered_passages,
            ) if scratchpad else "",
        )
        if scratchpad:
            prompt = (
                f"{task}\n\n"
                f"Previous steps:\n{scratchpad}\n\n"
                f"{closing}"
            )
        elif closing:
            prompt = f"{task}\n\n{closing}"
        else:
            prompt = task
        # Carry prior conversation turns into the native tool-calling prompt.
        # Without this the model only sees the latest task and forgets earlier
        # turns, so a multi-turn *session* loses its context the moment any tool
        # is attached (the ReAct/template branches inject this history too).
        if conversation_history:
            prompt = f"{conversation_history}\n\n{prompt}"
        # Steer the model with the user's custom persona. This path sends a bare
        # user message (the chat template owns the system slot for tools), so
        # prepend the persona — otherwise a custom persona is dropped the moment
        # a tool is attached, even though the ReAct-text and Gemini-native paths
        # honor it. The persona leads and the contract follows: the persona is
        # who the model is, the contract describes machinery the framework
        # attached.
        prompt = f"{self._persona_prefix()}{prompt}"
        if not scratchpad and self.tools:
            contract = self._tool_contract()
            if contract:
                prompt = f"{prompt}\n\n{contract}"
        return prompt

    def _direct_prompt(self, task: str, conversation_history: str = "") -> str:
        """Build the user prompt for the no-tool direct/streaming paths.

        Default agents keep the familiar ``"Answer this question directly and
        concisely: … Answer:"`` framing byte-for-byte. When the user set a custom
        persona, that persona *is* the response contract, so the framework's own
        "answer directly / Answer:" boilerplate is dropped — it competes with the
        persona (it literally commands "Answer:", which fights a Socratic "never
        give the answer" tutor) and on the smallest models it can override the
        persona entirely. The persona then leads, followed by any conversation
        history and the raw task — exactly the shape that steers reliably across
        cloud and local families.
        """
        # A schema the caller declared is the one machine-readable statement of
        # answer shape there is, so state it while the answer is being written.
        # Without it the model answers in prose and is then asked for the same
        # answer again in JSON. It goes ahead of the "Answer:" cue, so the cue
        # stays the last thing the model reads. Empty for every run without a
        # schema, which leaves those prompts unchanged.
        _shape = getattr(self, "_answer_shape_instruction", None)
        shape = _shape() if callable(_shape) else ""
        block = f"\n\n{shape}" if shape else ""

        persona = getattr(self, "_custom_persona", None)
        if persona:
            if conversation_history:
                return f"{persona}\n\n{conversation_history}\n\n{task}{block}"
            return f"{persona}\n\n{task}{block}"
        if conversation_history:
            return (
                f"{conversation_history}\n\n"
                f"Based on the conversation above, answer this question directly "
                f"and concisely:\n\n{task}{block}\n\nAnswer:"
            )
        return f"Answer this question directly and concisely:\n\n{task}{block}\n\nAnswer:"

    @staticmethod
    def _prompt_to_task_hint(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt[:500]
        try:
            from effgen.core.messages import Message

            if isinstance(prompt, Message):
                return prompt.text[:500]
            if isinstance(prompt, list):
                text = "\n".join(p.text for p in prompt if isinstance(p, Message))
                if text:
                    return text[:500]
        except Exception:
            logger.debug("Failed to extract task hint from prompt", exc_info=True)
        return str(prompt)[:500]
