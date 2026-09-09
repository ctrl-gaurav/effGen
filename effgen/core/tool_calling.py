"""
Tool calling strategies for the effGen framework.

This module provides a strategy abstraction for how agents invoke tools:
- ReActStrategy: Parse "Action:" / "Action Input:" from free-text (legacy default)
- NativeFunctionCallingStrategy: Use model's native tool/function calling
- HybridStrategy: Try native first, fall back to ReAct on parse failure

The agent selects the appropriate strategy based on model capabilities
and the ``tool_calling_mode`` setting in AgentConfig.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from effgen.observability import get_logger as _get_obs_logger

from .structured_output import _clean_json

logger = logging.getLogger(__name__)
_obs_log = _get_obs_logger(__name__)


# Keys a tool-call envelope may carry besides the name. An object whose only
# keys come from this set is a call even without an arguments key; one that
# carries data fields alongside the name is a JSON answer, not a call.
_CALL_ENVELOPE_KEYS = frozenset({"name", "function", "type", "id", "index"})


def _json_tool_call(
    text: str, tools: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]] | None:
    """Extract a ``{"name": ..., "arguments"|"parameters": {...}}`` tool call.

    Scans string-aware balanced JSON objects rather than matching braces with a
    regex: an argument value that itself contains a brace (Llama 3.2 emits
    ``"filter_metadata": "{}"``) truncates a non-greedy ``\\{.*?\\}`` match, and
    the resulting fragment fails to parse, so a valid call is read as no call at
    all.

    An object with no arguments key is only a call when nothing but envelope
    keys sits beside the name, or when ``tools`` confirms the name is a tool
    that exists — otherwise a JSON answer such as ``{"name": "Acme Corp",
    "revenue": 5}`` would be run as a call to a tool named "Acme Corp".

    Returns the tool name and its arguments, or ``None``.
    """
    # Every shape below carries one of these keys; skip the scan otherwise.
    if '"name"' not in text and '"function"' not in text:
        return None

    from .structured_output import _extract_balanced

    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None
        blob = _extract_balanced(text[start:])
        if blob is not None:
            call = _as_tool_call(blob, tools)
            if call is not None:
                return call
        search_from = start + 1


#: Tags a chat template uses to open a call in the XML dialect, and the tags it
#: uses for one argument inside it. Both spellings of the name are accepted:
#: ``<function=NAME>`` and ``<function name="NAME">``.
_XML_CALL_TAGS = ("function", "tool_call", "invoke", "tool", "function_call")
_XML_ARGUMENT_TAGS = ("parameter", "param", "argument", "arg")

#: ``<TAG=NAME>`` / ``<TAG name="NAME">`` … ``</TAG>``. The closing tag is
#: optional so a call the token budget cut short still reads.
_XML_CALL_RE = re.compile(
    r"<(?P<tag>" + "|".join(_XML_CALL_TAGS) + r")"
    r"(?:=|\s+name\s*=\s*[\"']?)(?P<name>[\w.\-]+)[\"']?\s*>"
    r"(?P<body>.*?)(?:</(?P=tag)>|$)",
    re.DOTALL | re.IGNORECASE,
)

#: One argument inside that envelope, in either spelling.
_XML_ARGUMENT_RE = re.compile(
    r"<(?P<tag>" + "|".join(_XML_ARGUMENT_TAGS) + r")"
    r"(?:=|\s+name\s*=\s*[\"']?)(?P<key>[\w.\-]+)[\"']?\s*>"
    r"(?P<value>.*?)</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)

#: Cheap rejection: the dialect always carries an opener and an argument tag.
_XML_CALL_HINT_RE = re.compile(
    r"<(?:" + "|".join(_XML_CALL_TAGS + _XML_ARGUMENT_TAGS) + r")(?:=|\s+name\s*=)",
    re.IGNORECASE,
)


def _xml_parameter_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Extract a call written as XML tags rather than as JSON.

    Chat templates disagree about how a tool call is spelled. Many render it as
    JSON — the readers above cover those — and others render it as nested tags,
    one per argument::

        <tool_call>
        <function=calculator>
        <parameter=expression>
        4817 * 236
        </parameter>
        </function>
        </tool_call>

    The JSON readers cannot see a call in that, and a wrapper such as
    ``<tool_call>`` also stops the text being taken as a final answer, so
    without this reader the turn parses to nothing at all: the loop nudges
    itself to its iteration cap and the tool is never called, on any model whose
    template writes this shape.

    Both ways of naming a tag are accepted — ``<function=NAME>`` and
    ``<function name="NAME">`` — across the tag names templates use for a call
    and for an argument, so the reader is keyed on the shape rather than on a
    model family. Each value is the tag's text: one that is valid JSON is
    decoded, so an integer argument arrives as an ``int`` rather than as
    ``"3"``, and anything else stays the string the model wrote.

    Returns the tool name and its arguments, or ``None``.
    """
    if not _XML_CALL_HINT_RE.search(text):
        return None

    for call in _XML_CALL_RE.finditer(text):
        arguments: dict[str, Any] = {}
        for arg in _XML_ARGUMENT_RE.finditer(call.group("body")):
            value = arg.group("value").strip()
            try:
                arguments[arg.group("key")] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                arguments[arg.group("key")] = value
        if arguments:
            return call.group("name"), arguments
    return None


def _is_truncated_json_call(text: str) -> bool:
    """Whether the text looks like a tool call the generation cut short.

    A ``"name"``/``"function"`` key with no balanced object around it is a call
    that ran out of tokens; returning it as the answer would ship raw call
    syntax to the user.
    """
    if '"name"' not in text and '"function"' not in text:
        return False

    from .structured_output import _extract_balanced

    start = text.find("{")
    if start == -1:
        return True
    blob = _extract_balanced(text[start:])
    if blob is None:
        return True
    try:
        json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return True
    return False


def _as_tool_call(
    blob: str, tools: dict[str, Any] | None
) -> tuple[str, dict[str, Any]] | None:
    """Read one balanced JSON object as a tool call, or return ``None``."""
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    # OpenAI-shaped calls nest the name/arguments under "function".
    inner = data.get("function")
    call = inner if isinstance(inner, dict) else data
    name = call.get("name")
    if not isinstance(name, str) and isinstance(call.get("function"), str):
        name = call["function"]
    if not isinstance(name, str) or not name:
        return None

    args = call.get("arguments")
    if args is None:
        args = call.get("parameters")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = None
    if isinstance(args, dict):
        return name, args
    if args is not None:
        return None
    if set(call) - _CALL_ENVELOPE_KEYS and not (tools and name in tools):
        return None
    return name, {}


# The scaffolding labels that can follow a tool name on the same line. Each must
# be followed by ``:`` or ``=`` so a tool genuinely called ``read_input`` or
# ``list_args`` is never mistaken for one. An optional ``|``/``,``/``;``/``-``
# separator in front is dropped with the label.
_SAME_LINE_MARKER_RE = re.compile(
    r"\s*[|,;-]?\s*\b(?:action\s*)?(?:input|parameters|params|args|arguments)\s*[:=]",
    re.IGNORECASE,
)
_OPENING_BRACKETS = "([{"
_CLOSING_BRACKETS = ")]}"


def _bracket_depths(text: str) -> list[int]:
    """Bracket nesting depth at each character of *text*.

    An opening bracket reports the depth outside it and a closing bracket the
    depth it returns to, so a whole ``name(...)`` construct except its interior
    sits at the depth the name itself does.
    """
    depths: list[int] = []
    depth = 0
    for char in text:
        if char in _OPENING_BRACKETS:
            depths.append(depth)
            depth += 1
        elif char in _CLOSING_BRACKETS:
            depth = max(0, depth - 1)
            depths.append(depth)
        else:
            depths.append(depth)
    return depths


def action_name(raw: str) -> str:
    """Return just the tool name from the text following an ``Action:`` label.

    Models often put the whole step on one line —
    ``Action: calculator | Action Input: {"expression": "2 * 3"}`` — and reading
    to the end of the line makes the tool name the entire remainder, which
    resolves against nothing. Cutting at the first argument label recovers the
    name; the arguments are extracted separately and are unaffected.

    Only a label outside brackets ends the name. A call written as
    ``calculator(input="2 * 3")`` carries the same words inside its argument
    list, and cutting there would leave ``calculator(`` — so the whole
    construct is returned and the function-call reader downstream takes it.

    Args:
        raw: The text captured after ``Action:``/``Tool:``, already stripped of
            surrounding quotes.

    Returns:
        str: The tool name, trimmed of the argument section and trailing
        punctuation. Text with no argument label outside brackets is returned
        unchanged.
    """
    depths = _bracket_depths(raw)
    name = raw
    for marker in _SAME_LINE_MARKER_RE.finditer(raw):
        if depths[marker.start()] == 0:
            name = raw[: marker.start()]
            break
    # ``Action: calculator {"expression": "2 * 3"}`` — the arguments follow the
    # name as a bare JSON object with no label between them. Without this the
    # whole line is the tool name and resolves against nothing; the reader in
    # ``parse_call_syntax`` takes the object.
    brace = _SAME_LINE_JSON_RE.match(name)
    if brace:
        name = brace.group(1)
    return name.strip().rstrip(",;|-").strip()


# ``name {json}`` — a bare object where an argument label would be.
_SAME_LINE_JSON_RE = re.compile(r"\s*([\w.\-/]+)\s*\{")

# ``name(...)`` spanning the whole construct.
_CALL_SYNTAX_RE = re.compile(r"^\s*([\w.\-]+)\s*\((.*)\)\s*$", re.DOTALL)


def parse_call_syntax(raw: str) -> tuple[str, dict[str, Any], list[Any]] | None:
    """Read a tool call written in Python call syntax or as ``name {json}``.

    ``action_name`` recovers the *name* from these shapes; this recovers the
    *arguments*, which used to be dropped — the tool was invoked with ``{}``,
    refused the empty argument set, and the loop spent a turn on the refusal.

    Recognised, in the shapes models actually emit::

        calculator(expression="1367 * 89")   -> {"expression": "1367 * 89"}
        calculator(expression='1367 * 89')   -> {"expression": "1367 * 89"}
        calculator("1367 * 89")              -> {"__raw_input__": "1367 * 89"}
        calculator({"expression": "6*7"})    -> {"expression": "6*7"}
        calculator {"expression": "6*7"}     -> {"expression": "6*7"}

    A single positional argument becomes ``__raw_input__``, which the agent's
    existing mapper resolves against the tool's declared parameters — the same
    route a plain ``Action Input:`` value already takes, so a one-argument call
    needs no schema here. Several positional arguments are returned separately
    for a caller that has the schema to map them onto.

    Args:
        raw: The text following ``Action:``, with its surrounding quotes
            stripped but its inner quoting intact.

    Returns:
        ``(name, keyword_arguments, positional_arguments)``, or ``None`` when
        the text is not a call in either shape.
    """
    if not raw:
        return None

    brace = _SAME_LINE_JSON_RE.match(raw)
    if brace:
        from .structured_output import _extract_balanced

        blob = _extract_balanced(raw[brace.end() - 1:])
        try:
            parsed = json.loads(blob) if blob else None
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return brace.group(1), parsed, []
        return None

    call = _CALL_SYNTAX_RE.match(raw)
    if not call:
        return None
    name, inner = call.group(1), call.group(2).strip()
    if not inner:
        return name, {}, []

    # A single JSON object argument: calculator({"expression": "6*7"}).
    if inner.startswith("{"):
        try:
            parsed = json.loads(inner)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return name, parsed, []

    # Python call syntax. ``ast`` is used for the parse only; every value is
    # read with ``literal_eval``, so nothing in the model's text is executed.
    try:
        node = ast.parse(raw.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(node, ast.Call):
        return None

    keywords: dict[str, Any] = {}
    positional: list[Any] = []
    try:
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs — nothing to map it onto
                return None
            keywords[kw.arg] = ast.literal_eval(kw.value)
        for arg in node.args:
            positional.append(ast.literal_eval(arg))
    except (ValueError, TypeError):
        return None

    if len(positional) == 1 and not keywords:
        return name, {"__raw_input__": positional[0]}, []
    return name, keywords, positional


def name_positional_arguments(
    tool_name: str, positional: list[Any], tools: dict[str, Any] | None,
) -> dict[str, Any]:
    """Give several positional arguments the names the tool declares.

    ``file_operations('write', 'greet.py', 'print(1)')`` carries its values in
    the order the tool's own parameters are declared, which is the only thing
    that can name them. Without the tool the values cannot be placed, so the
    first is handed over as raw input — the same shape a plain ``Action Input:``
    takes — rather than being dropped.

    Args:
        tool_name: The name the call resolved to.
        positional: The positional values read from the call, in order.
        tools: The agent's tools by name, or ``None`` when unavailable.

    Returns:
        The values under the tool's declared parameter names, or the first value
        under ``__raw_input__`` when there is no schema to name them by.
    """
    tool = (tools or {}).get(tool_name)
    parameters = getattr(getattr(tool, "metadata", None), "parameters", None)
    if parameters:
        names = [p.name for p in parameters]
        if len(names) >= len(positional):
            return dict(zip(names, positional, strict=False))
    return {"__raw_input__": positional[0]}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCallResult:
    """Unified result from parsing a model response for tool calls.

    Attributes:
        tool_name: Name of the tool to invoke (None if final answer).
        arguments: Parsed arguments dict for the tool.
        raw_text: The original model response text.
        thought: Extracted reasoning / chain-of-thought text.
        final_answer: If the model produced a final answer instead of a tool call.
        is_tool_call: True when a valid tool call was extracted.
        call_id: The id the provider gave this call, when it gave one. A result
            has to answer the call it belongs to, and only the provider's own id
            says which that is.
        reasoning: The model's own words that arrived beside a provider-native
            call. Distinct from ``thought``, which is the reasoning a model
            wrote as text for the framework to read back: this is text the
            model addressed to nobody but itself, and it belongs on the same
            turn as the call rather than on a turn of its own.
    """
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    thought: str | None = None
    final_answer: str | None = None
    is_tool_call: bool = False
    call_id: str | None = None
    reasoning: str = ""


@dataclass
class ToolDefinition:
    """JSON Schema tool definition for native function calling.

    Mirrors the OpenAI tools format used by most model providers.
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI-style tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic-style tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


# ---------------------------------------------------------------------------
# Helper: convert BaseTool metadata to ToolDefinition
# ---------------------------------------------------------------------------

def tools_to_definitions(tools: list) -> list[ToolDefinition]:
    """Convert a list of BaseTool instances to ToolDefinition objects.

    Args:
        tools: List of BaseTool instances.

    Returns:
        List of ToolDefinition with JSON Schema parameters.
    """
    definitions: list[ToolDefinition] = []
    for tool in tools:
        meta = tool.metadata
        schema = meta.to_json_schema()
        definitions.append(ToolDefinition(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"],
        ))
    return definitions


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------

class ToolCallingStrategy(ABC):
    """Abstract base class for tool calling strategies."""

    @abstractmethod
    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Parse a model response and extract tool call or final answer.

        Args:
            text: Raw model response text.
            tools: Dict mapping tool name -> tool object (for validation).

        Returns:
            ToolCallResult with extracted information.
        """

    @abstractmethod
    def format_tools_for_prompt(self, tools: list) -> Any:
        """Prepare tool information for the model.

        For ReAct this returns a text description; for native calling it
        returns JSON Schema definitions.

        Args:
            tools: List of BaseTool instances.

        Returns:
            Formatted tool information (str or list[dict]).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""


# ---------------------------------------------------------------------------
# ReAct Strategy — extracted from Agent._parse_react_response()
# ---------------------------------------------------------------------------

class ReActStrategy(ToolCallingStrategy):
    """Parse tool calls from ReAct-formatted free text.

    This strategy is the legacy default. It extracts ``Thought:``,
    ``Action:``, ``Action Input:``, and ``Final Answer:`` fields from
    the model's textual output using regex patterns.
    """

    @property
    def name(self) -> str:
        """Strategy identifier: ``"react"``."""
        return "react"

    # -- Shared JSON-cleaning helpers (also used by Agent) -----------------

    @staticmethod
    def clean_json_input(raw: str) -> str:
        """Clean malformed JSON commonly produced by SLMs.

        Handles:
        - Markdown-wrapped JSON (```json ... ```)
        - Trailing commas  ({"key": "val",})
        - Unquoted keys    ({expression: "2+2"})

        Argument values are left untouched: a repair applies only outside string
        literals, so a query like ``"Paris, France: population"`` reaches the
        tool exactly as the model wrote it.
        """
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            text = text.strip()

        return _clean_json(text)

    # -- Core parsing ------------------------------------------------------

    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Parse ReAct formatted response.

        This is extracted from ``Agent._parse_react_response()`` with the
        same logic and patterns, returning a ``ToolCallResult`` instead of
        a plain dict.
        """
        result = ToolCallResult(raw_text=text)

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid response text for parsing: {type(text)}")
            return result

        try:
            # --- Final answer (highest priority) ---
            final_patterns = [
                r"Final Answer:\s*(.+)",
                r"^Answer:\s*(.+)",
                r"^The answer is:\s*(.+)",
            ]
            for pattern in final_patterns:
                try:
                    final_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if final_match:
                        answer = final_match.group(1).strip()
                        answer = re.split(
                            r'\n(?:Question|Thought|Action|Observation|Human):',
                            answer, maxsplit=1,
                        )[0].strip()
                        # Strip trailing hallucinated follow-up questions
                        trailing = re.search(r'([.!?])[\s]*[A-Z][^.!?]*\?', answer)
                        if trailing:
                            answer = answer[:trailing.start() + 1].strip()
                        result.final_answer = answer
                        logger.debug(f"Extracted final answer: {answer[:100]}...")
                        return result
                except Exception as e:
                    logger.warning(f"Error matching final answer pattern '{pattern}': {e}")
                    continue

            # --- Thought ---
            thought_patterns = [
                r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|Question):|$)",
                r"Thought:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            ]
            for pattern in thought_patterns:
                try:
                    thought_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if thought_match:
                        result.thought = thought_match.group(1).strip()
                        break
                except Exception as e:
                    logger.warning(f"Error matching thought pattern '{pattern}': {e}")
                    continue

            # --- Action ---
            action_patterns = [
                r"Action:\s*([^\n]+)",
                r"Tool:\s*([^\n]+)",
                r"Use tool:\s*([^\n]+)",
            ]
            for pattern in action_patterns:
                try:
                    action_match = re.search(pattern, text, re.IGNORECASE)
                    if action_match:
                        # Kept with its quoting intact: the call-syntax reader
                        # below needs it, and stripping every quote first is
                        # what turned calculator(expression="1367 * 89") into
                        # an unparseable argument list.
                        action_raw = action_match.group(1).strip()
                        action = action_raw.replace('"', '').replace("'", "")
                        # Drop a same-line "Action Input:"/"Args:" section so the
                        # name resolves against the registry.
                        action = action_name(action)

                        # "Action: Final Answer" → treat as final answer
                        if action.lower() in ["final answer", "finalanswer", "answer"]:
                            logger.debug(f"Action '{action}' detected as Final Answer indicator")
                            same_line = re.search(
                                r"Action:\s*Final\s*Answer[: \t]+([^\n]+)",
                                text, re.IGNORECASE,
                            )
                            if same_line:
                                answer_text = same_line.group(1).strip()
                                if answer_text:
                                    result.final_answer = answer_text
                                    return result

                            ai_match = re.search(
                                r"Action\s*Input:\s*(.+?)(?:\n|$)",
                                text, re.IGNORECASE,
                            )
                            if ai_match:
                                answer_text = ai_match.group(1).strip()
                                if answer_text and not answer_text.startswith(("{", "[")):
                                    result.final_answer = answer_text
                                    return result
                            break

                        # Handle a call written in Python call syntax, or a
                        # bare JSON object after the name. Read from the
                        # unmangled text so the model's own quoting survives.
                        call = parse_call_syntax(action_raw)
                        if call is not None:
                            call_name, call_kwargs, call_positional = call
                            result.tool_name = call_name
                            result.is_tool_call = True
                            result.raw_text = text
                            if call_kwargs:
                                result.arguments = call_kwargs
                            elif call_positional:
                                result.arguments = name_positional_arguments(
                                    call_name, call_positional, tools,
                                )
                            break

                        # Handle function-call format: tool_name(args)
                        func_call_match = re.match(r'^(\w+)\s*\((.+)\)$', action, re.DOTALL)
                        if func_call_match:
                            tool_name = func_call_match.group(1).strip()
                            embedded_args = func_call_match.group(2).strip().strip('"\'')
                            result.tool_name = tool_name
                            result.is_tool_call = True
                            # Parse embedded args as JSON or plain text
                            try:
                                result.arguments = json.loads(self.clean_json_input(embedded_args))
                                if not isinstance(result.arguments, dict):
                                    result.arguments = {}
                            except (json.JSONDecodeError, TypeError):
                                pass
                            # Store raw for agent to handle via _map_input_to_parameters
                            result.raw_text = text
                        else:
                            result.tool_name = action
                            result.is_tool_call = True
                        break
                except Exception as e:
                    logger.warning(f"Error matching action pattern '{pattern}': {e}")
                    continue

            # --- Action Input (skip if already set from function-call style) ---
            if result.is_tool_call and not result.arguments:
                input_patterns = [
                    r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question|Final Answer):|$)",
                    r"Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    r"Parameters?:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    # The name is trimmed at an `Args:`/`Arguments:` label too,
                    # so read the arguments from it rather than calling the tool
                    # with none.
                    r"Arg(?:ument)?s:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                ]
                for pattern in input_patterns:
                    try:
                        input_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if input_match:
                            action_input = input_match.group(1).strip()
                            action_input = re.split(r'\nObservation:', action_input, maxsplit=1)[0].strip()
                            # Try to parse as JSON
                            try:
                                cleaned = self.clean_json_input(action_input)
                                parsed = json.loads(cleaned)
                                if isinstance(parsed, dict):
                                    result.arguments = parsed
                                else:
                                    # Store raw text in a special key for agent to handle
                                    result.arguments = {"__raw_input__": action_input}
                            except (json.JSONDecodeError, TypeError):
                                result.arguments = {"__raw_input__": action_input}
                            break
                    except Exception as e:
                        logger.warning(f"Error matching action input pattern '{pattern}': {e}")
                        continue

        except Exception as e:
            logger.error(f"Critical error in ReAct parse_response: {e}", exc_info=True)

        return result

    def format_tools_for_prompt(self, tools: list) -> str:
        """Return a text description of tools (used in ReAct prompt).

        The actual formatting is delegated to ToolPromptGenerator, so
        this just returns a simple fallback description.
        """
        lines = []
        for tool in tools:
            meta = tool.metadata
            params = ", ".join(
                f"{p.name}: {p.type.value}" for p in meta.model_facing_parameters
            )
            lines.append(f"- {meta.name}: {meta.description} ({params})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native Function Calling Strategy
# ---------------------------------------------------------------------------

class NativeFunctionCallingStrategy(ToolCallingStrategy):
    """Use model's native tool/function calling via chat templates or API.

    This strategy converts BaseTool metadata to JSON Schema function
    definitions and parses structured tool calls from model responses.
    Works with:
    - Transformers models that support ``tools`` param in chat templates
    - vLLM with tool call parser
    - API models (OpenAI, Anthropic, Gemini) that return structured tool calls
    """

    @property
    def name(self) -> str:
        """Strategy identifier: ``"native"``."""
        return "native"

    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Parse tool calls from native function calling response.

        Handles multiple response formats:
        1. Structured tool_calls in metadata (API models)
        2. Qwen-style: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        3. Llama-style: <|python_tag|>function_name(...)
        4. Mistral-style: [TOOL_CALLS][{"name": ..., "arguments": ...}]
        5. Generic JSON function call format

        Args:
            text: Raw model response (may contain structured markers).
            tools: Dict mapping tool name -> tool object for validation.

        Returns:
            ToolCallResult
        """
        result = ToolCallResult(raw_text=text)

        if not text or not isinstance(text, str):
            return result

        # --- Try Gemma 4 channel format (asymmetric delimiters) ---
        # Gemma 4 wraps reasoning in <|channel>...<channel|> and tool calls in
        # <|tool_call>call:NAME{...}<tool_call|>. These markers are unknown to
        # the other parsers below, so without this branch the whole reasoning
        # trace leaks into the "final answer" and tool calls never fire.
        if "<|channel>" in text or "<|tool_call>" in text or "<channel|>" in text:
            gemma = self._parse_gemma_channels(text, tools)
            if gemma is not None:
                return gemma

        # --- Try Qwen format: <tool_call>...</tool_call> ---
        qwen_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            text, re.DOTALL,
        )
        if qwen_match:
            try:
                call_data = json.loads(qwen_match.group(1))
                tool_name = call_data.get("name") or call_data.get("function")
                arguments = call_data.get("arguments") or call_data.get("parameters") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if tool_name:
                    result.tool_name = tool_name
                    result.arguments = arguments
                    result.is_tool_call = True
                    logger.debug(f"Parsed Qwen-style tool call: {tool_name}")
                    return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse Qwen tool call JSON: {e}")

        # --- Try the XML dialect ---
        # <function=NAME><parameter=KEY>value</parameter>…</function>, the shape
        # templates use where others write JSON. Read before the JSON wrappers
        # below: none of them can see a call in it, and its wrapper tag keeps
        # the text from being read as an answer either.
        xml_call = _xml_parameter_call(text)
        if xml_call is not None:
            result.tool_name, result.arguments = xml_call
            result.is_tool_call = True
            logger.debug(f"Parsed XML-parameter tool call: {result.tool_name}")
            return result

        # --- Try the <function=NAME>{...}</function> wrapper ---
        # The name is followed by '>', not by the '(' or '{' the combined
        # pattern below requires, so this shape needs its own read.
        fn_match = re.search(r"<function=([\w.-]+)>\s*(?=\{)", text)
        if fn_match:
            from .structured_output import _extract_balanced

            blob = _extract_balanced(text[fn_match.end():])
            try:
                arguments = json.loads(blob) if blob else None
            except (json.JSONDecodeError, TypeError) as e:
                arguments = None
                logger.debug(f"Failed to parse <function=> tool call: {e}")
            if isinstance(arguments, dict):
                result.tool_name = fn_match.group(1)
                result.arguments = arguments
                result.is_tool_call = True
                logger.debug(f"Parsed <function=> tool call: {result.tool_name}")
                return result

        # --- Try Llama/Hermes format: <|python_tag|> or <function= ---
        llama_match = re.search(
            r'(?:<\|python_tag\|>|<function=)(\w+)\s*[(\{](.+?)[)\}]',
            text, re.DOTALL,
        )
        if llama_match:
            tool_name = llama_match.group(1).strip()
            args_text = llama_match.group(2).strip()
            try:
                # Try JSON parse
                if not args_text.startswith("{"):
                    args_text = "{" + args_text
                if not args_text.endswith("}"):
                    args_text = args_text + "}"
                arguments = json.loads(args_text)
                result.tool_name = tool_name
                result.arguments = arguments if isinstance(arguments, dict) else {}
                result.is_tool_call = True
                logger.debug(f"Parsed Llama-style tool call: {tool_name}")
                return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse Llama tool call: {e}")

        # --- Try Mistral format: [TOOL_CALLS][...] ---
        mistral_match = re.search(
            r'\[TOOL_CALLS\]\s*\[(.+?)\]',
            text, re.DOTALL,
        )
        if mistral_match:
            try:
                calls = json.loads("[" + mistral_match.group(1) + "]")
                if calls and isinstance(calls, list):
                    call = calls[0]  # Take first tool call
                    tool_name = call.get("name") or call.get("function")
                    arguments = call.get("arguments") or call.get("parameters") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if tool_name:
                        result.tool_name = tool_name
                        result.arguments = arguments
                        result.is_tool_call = True
                        logger.debug(f"Parsed Mistral-style tool call: {tool_name}")
                        return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse Mistral tool call: {e}")

        # --- Try generic JSON function call ---
        # {"name": "tool", "arguments": {...}}, with or without a leading
        # <|python_tag|> marker — Llama 3.2 emits the bare object.
        json_call = _json_tool_call(text, tools)
        if json_call is not None:
            tool_name, arguments = json_call
            result.tool_name = tool_name
            result.arguments = arguments
            result.is_tool_call = True
            logger.debug(f"Parsed generic JSON tool call: {tool_name}")
            return result

        # --- No tool call found — check for plain text answer ---
        # If the response doesn't contain any tool call markers, treat as final answer
        text_stripped = text.strip()
        if text_stripped and not any(marker in text for marker in [
            "<tool_call>", "<|python_tag|>", "<function=", "[TOOL_CALLS]",
            "Thought:", "Action:", "Tool:",
            "<|channel>", "<channel|>", "<|tool_call>",
        ]):
            # A bare "name"/"function" key is only evidence of a call the parse
            # above could not finish — a truncated one. When the JSON is
            # complete it is an answer (e.g. {"name": "Acme Corp", ...}), and
            # withholding it would leave the run with no answer at all.
            if _is_truncated_json_call(text):
                logger.debug("Incomplete JSON call detected, not a final answer")
                return result
            result.final_answer = text_stripped
            logger.debug("No tool call markers found, treating as final answer")

        return result

    # -- Gemma 4 channel/tool_call parsing ---------------------------------
    # Delimiters are asymmetric: open <|x> / close <x|>. Confirmed against the
    # gemma-4-*-it tokenizer special tokens (<|channel>/<channel|>,
    # <|tool_call>/<tool_call|>).
    _GEMMA_STRIP_RE = re.compile(
        r"<\|?(?:channel|turn|think|tool|tool_call|tool_response|image|audio|video)\|?>"
    )

    @staticmethod
    def _loose_json(s: str) -> dict[str, Any]:
        """Parse Gemma's loose arg blob, e.g. {query: "x"} (unquoted keys)."""
        s = (s or "").strip()
        if not s:
            return {}
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else {"__raw_input__": s}
        except (json.JSONDecodeError, TypeError):
            pass
        # Quote bare identifier keys: {query: "x"} -> {"query": "x"}
        fixed = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)', r'\1"\2"\3', s)
        try:
            v = json.loads(fixed)
            return v if isinstance(v, dict) else {"__raw_input__": s}
        except (json.JSONDecodeError, TypeError):
            return {"__raw_input__": s}

    @staticmethod
    def _resolve_tool_name(name: str, tools: dict[str, Any] | None) -> str:
        """Map a slightly-off name (e.g. search_arxiv) onto a real tool.

        Gemma often invents a verb-prefixed name when it isn't handed the tool
        schema. Fall back to the raw name when nothing plausible matches so the
        caller still surfaces a clear "tool not found".
        """
        if not tools or name in tools:
            return name
        lowered = {t.lower(): t for t in tools}
        if name.lower() in lowered:
            return lowered[name.lower()]
        for prefix in ("search_", "get_", "call_", "use_", "run_", "fetch_", "query_"):
            if name.lower().startswith(prefix):
                stem = name.lower()[len(prefix):]
                if stem in lowered:
                    return lowered[stem]
        import difflib
        close = difflib.get_close_matches(name.lower(), list(lowered), n=1, cutoff=0.7)
        return lowered[close[0]] if close else name

    def _parse_gemma_channels(
        self, text: str, tools: dict[str, Any] | None
    ) -> ToolCallResult | None:
        """Parse Gemma 4 output: strip the reasoning channel, extract tool calls."""
        result = ToolCallResult(raw_text=text)

        # Reasoning lives in <|channel>[thought]...<channel|>; keep it as thought.
        thought_match = re.search(
            r"<\|channel>\s*(?:thought)?\s*(.*?)<channel\|>", text, re.DOTALL
        )
        if thought_match:
            result.thought = thought_match.group(1).strip() or None

        # Tool call: <|tool_call> call:NAME{...} <tool_call|> (close tag optional
        # when generation is truncated).
        call_match = re.search(
            r"<\|tool_call>\s*(?:call:)?\s*([A-Za-z_]\w*)\s*(\{.*?\})"
            r"\s*(?:<tool_call\|>|$)",
            text,
            re.DOTALL,
        )
        if call_match:
            tool_name = self._resolve_tool_name(call_match.group(1).strip(), tools)
            result.tool_name = tool_name
            result.arguments = self._loose_json(call_match.group(2))
            result.is_tool_call = True
            logger.debug(f"Parsed Gemma-style tool call: {tool_name}")
            return result

        # No tool call — strip every channel block, leaving the clean answer.
        cleaned = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL)
        # Drop any unclosed trailing channel (reasoning truncated at max_tokens)
        # so half-finished thoughts never surface as the answer.
        cleaned = re.sub(r"<\|channel>.*$", "", cleaned, flags=re.DOTALL)
        cleaned = self._GEMMA_STRIP_RE.sub("", cleaned).strip()
        if cleaned:
            result.final_answer = cleaned
            logger.debug("Gemma channel format: extracted final answer")
            return result

        # Only an unclosed/empty reasoning channel (e.g. truncated at max_tokens).
        # Return an empty result so raw special tokens never leak as the answer.
        return result

    def format_tools_for_prompt(self, tools: list) -> list[dict[str, Any]]:
        """Convert tools to JSON Schema definitions for native calling.

        Returns:
            List of OpenAI-format tool definitions.
        """
        definitions = tools_to_definitions(tools)
        return [d.to_openai_format() for d in definitions]


# ---------------------------------------------------------------------------
# Hybrid Strategy
# ---------------------------------------------------------------------------

class HybridStrategy(ToolCallingStrategy):
    """Try native function calling first, fall back to ReAct on parse failure."""

    def __init__(self) -> None:
        self._native = NativeFunctionCallingStrategy()
        self._react = ReActStrategy()

    @property
    def name(self) -> str:
        """Strategy identifier: ``"hybrid"``."""
        return "hybrid"

    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Try native parsing first, then ReAct."""
        result = self._native.parse_response(text, tools)
        if result.is_tool_call or result.final_answer:
            logger.debug("Hybrid strategy: native parsing succeeded")
            return result

        logger.debug("Hybrid strategy: native parsing failed, trying ReAct")
        return self._react.parse_response(text, tools)

    def format_tools_for_prompt(self, tools: list) -> Any:
        """Use native format (JSON Schema definitions)."""
        return self._native.format_tools_for_prompt(tools)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_strategy(
    mode: str = "auto",
    model: Any | None = None,
) -> ToolCallingStrategy:
    """Create the appropriate tool calling strategy.

    Args:
        mode: One of "auto", "native", "react", "hybrid".
        model: The model instance (checked for supports_tool_calling).

    Returns:
        ToolCallingStrategy instance.
    """
    if mode == "react":
        return ReActStrategy()
    elif mode == "native":
        return NativeFunctionCallingStrategy()
    elif mode == "hybrid":
        return HybridStrategy()
    elif mode == "auto":
        # Auto-detect based on model capabilities
        if model is not None and hasattr(model, 'supports_tool_calling'):
            try:
                if model.supports_tool_calling():
                    logger.info("Auto-detected native tool calling support, using hybrid strategy")
                    return HybridStrategy()
            except Exception:
                logger.debug("Native tool-calling capability check failed; using default strategy", exc_info=True)
        logger.debug("Using ReAct strategy (model does not support native tool calling)")
        return ReActStrategy()
    else:
        logger.warning(f"Unknown tool_calling_mode '{mode}', defaulting to ReAct")
        return ReActStrategy()
