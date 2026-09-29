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


#: Logged each time a call's arguments arrived as a string rather than as an
#: object, with what the string was read as.
ARGUMENTS_AS_STRING_LOG = "[call] read arguments sent as a string (%s)"

#: The key a value the tool's parameters must still be matched to travels
#: under. The agent's input mapper binds it the way it binds a plain
#: ``Action Input:`` line.
RAW_INPUT_KEY = "__raw_input__"


def _keyword_arguments(text: str) -> dict[str, Any] | None:
    """Read ``code='…'`` / ``a=1, b="x"`` as keyword arguments, or ``None``.

    The text is parsed as the argument list of a call; every value must be a
    literal, so nothing in it is ever run. Positional values, ``**`` unpacking
    or a value that is not a literal make it not keyword syntax.
    """
    try:
        node = ast.parse(f"_({text})", mode="eval").body
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return None
    if not isinstance(node, ast.Call) or node.args or not node.keywords:
        return None
    keywords: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            return None
        value = _literal_value(kw.value)
        if value is _NOT_LITERAL:
            return None
        keywords[kw.arg] = value
    return keywords


def _declared_parameters(tool: Any) -> set[str] | None:
    """The parameter names *tool* declares, or ``None`` when it declares none."""
    parameters = getattr(getattr(tool, "metadata", None), "parameters", None)
    if not parameters:
        return None
    return {str(getattr(p, "name", "")) for p in parameters}


def read_call_arguments(raw: Any, tool: Any = None) -> dict[str, Any]:
    """Decode a tool call's arguments into a mapping, whatever shape they came in.

    Providers return a call's arguments as a JSON object text, and most of the
    time that text decodes to an object. Some serving stacks pass through what
    the model wrote instead — ``"arguments": "code='print(1)'"`` or
    ``"arguments": "56*3+35"`` — and the decoded value is then a string. That
    string is the call's argument, not an absence of one:

    * a ``dict`` is returned as it is;
    * a string is decoded as JSON; an object is returned;
    * a string that decodes to a string, or is not JSON at all, is read again as
      text: an object's JSON text is that object, keyword syntax
      (``code='…'``, ``a=1, b="x"``; values read as literals, nothing is run)
      gives those keywords — when *tool* is given, only if it declares every
      one of them — and anything else is handed on under ``__raw_input__`` for
      the agent's mapper to bind to the tool's parameter;
    * ``None`` or an empty string is no arguments, ``{}``;
    * any other decoded value (a number, a list) is handed on as its JSON text
      under ``__raw_input__``.

    Args:
        raw: The arguments as the provider or the reader produced them.
        tool: The tool the call names, when known; keyword syntax is only read
            as keywords for the parameters it declares.

    Returns:
        The arguments as a mapping. A value under ``__raw_input__`` is always a
        string.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return {RAW_INPUT_KEY: json.dumps(raw, default=str)}
    text = raw.strip()
    if not text:
        return {}
    try:
        decoded: Any = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        decoded = text
    else:
        if isinstance(decoded, dict):
            return decoded
        if decoded is None:
            return {}
        if not isinstance(decoded, str):
            logger.info(ARGUMENTS_AS_STRING_LOG, "raw value")
            return {RAW_INPUT_KEY: json.dumps(decoded)}
    value = decoded.strip()
    if not value:
        return {}
    if value.startswith("{"):
        try:
            obj = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            logger.info(ARGUMENTS_AS_STRING_LOG, "object text")
            return obj
    keywords = _keyword_arguments(value)
    declared = _declared_parameters(tool) if tool is not None else None
    if keywords and (declared is None or set(keywords) <= declared):
        logger.info(ARGUMENTS_AS_STRING_LOG, "keywords")
        return keywords
    logger.info(ARGUMENTS_AS_STRING_LOG, "raw value")
    return {RAW_INPUT_KEY: value}


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
        # A string argument is read when it names a held tool (the string is
        # then that tool's argument); otherwise only an object's JSON text is
        # a call, so a JSON answer that merely has these keys stays an answer.
        held = tools.get(name) if tools and name in tools else None
        if held is not None:
            args = read_call_arguments(args, held)
        else:
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

    A single positional argument becomes ``__raw_input__`` (as a string; a
    number or a list is given as its JSON text), which the agent's existing
    mapper resolves against the tool's declared parameters — the same route a
    plain ``Action Input:`` value already takes, so a one-argument call needs
    no schema here. Several positional arguments are returned separately for a
    caller that has the schema to map them onto.

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
        value = positional[0]
        return name, {
            RAW_INPUT_KEY: value if isinstance(value, str) else json.dumps(value)
        }, []
    return name, keywords, positional


#: The openings a chat template writes in front of a tool call. They are a
#: property of the template's call syntax, not of any one model: a turn whose
#: text opens a call with one of them meant to make that call.
CALL_OPENING_MARKERS: tuple[str, ...] = (
    "<tool_call>", "<|python_tag|>", "<function=", "[TOOL_CALLS]", "<|tool_call>",
)

_CODE_SPAN_RE = re.compile(r"```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`", re.DOTALL)


def opened_call(text: str | None) -> str | None:
    """Whether *text* opens a tool call in a template's call syntax, and how.

    Code spans are left out first, so an answer that shows a call inside a code
    block is documentation rather than a call.

    Args:
        text: The text a turn produced.

    Returns:
        ``"empty"`` when nothing follows the last opening, ``"unread"`` when
        something does, or ``None`` when the text opens no call.
    """
    if not text or not isinstance(text, str):
        return None
    if not any(marker in text for marker in CALL_OPENING_MARKERS):
        return None
    scan = _CODE_SPAN_RE.sub(" ", text) if "`" in text or "~~~" in text else text
    last = -1
    marker_len = 0
    for marker in CALL_OPENING_MARKERS:
        at = scan.rfind(marker)
        if at > last:
            last, marker_len = at, len(marker)
    if last < 0:
        return None
    body = scan[last + marker_len:]
    body = re.sub(r"</?tool_call>|<tool_call\|>", " ", body)
    return "unread" if body.strip() else "empty"


#: JSON's three words, as a Python reader meets them.
_JSON_WORDS: dict[str, Any] = {"true": True, "false": False, "null": None,
                               "True": True, "False": False, "None": None}


#: What :func:`_literal_value` returns for anything that is not a literal.
_NOT_LITERAL = object()


def _literal_value(node: ast.AST) -> Any:
    """The value a literal expression spells, or :data:`_NOT_LITERAL`.

    Only constants, the JSON words, containers of those and a signed number are
    read; anything else — a name, a call, an operator — is refused, so nothing
    in the text is ever run.
    """
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bool, type(None))
    ):
        return node.value
    if isinstance(node, ast.Name):
        return _JSON_WORDS.get(node.id, _NOT_LITERAL)
    if isinstance(node, ast.Dict):
        mapping: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                return _NOT_LITERAL
            key = _literal_value(key_node)
            value = _literal_value(value_node)
            if key is _NOT_LITERAL or value is _NOT_LITERAL:
                return _NOT_LITERAL
            try:
                mapping[key] = value
            except TypeError:
                return _NOT_LITERAL
        return mapping
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_literal_value(item) for item in node.elts]
        return _NOT_LITERAL if any(i is _NOT_LITERAL for i in items) else items
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        number = node.operand.value if isinstance(node.operand, ast.Constant) else None
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            return -number if isinstance(node.op, ast.USub) else number
    return _NOT_LITERAL


def _escape_line_breaks(body: str, quotes: str) -> str:
    """Escape the raw line breaks that sit inside string literals of *body*."""
    out: list[str] = []
    closing = ""
    escaped = False
    for char in body:
        if closing:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == closing:
                closing = ""
            elif char == "\n":
                out.append("\\n")
                continue
            elif char == "\r":
                out.append("\\r")
                continue
        elif char in quotes:
            closing = char
        out.append(char)
    return "".join(out)


#: A string value closed just before a run of ``)``/``]`` that is followed
#: only by the object's closing braces: ``…print(sum([1, 2])")}}``.
_MISPLACED_QUOTE_RE = re.compile(r'"([)\]]+)((?:\s*\})+\s*)$')


def _move_closing_quote(body: str) -> str | None:
    """*body* with a string's closing quote moved past the brackets after it.

    A model writing a program as a JSON string sometimes closes the string one
    bracket early — ``"print(sum([1, 2])")}}`` — so the brackets that end the
    program sit outside it and the object no longer parses. Only that shape,
    at the very end of the body, is moved; ``None`` when the body ends
    otherwise.
    """
    match = _MISPLACED_QUOTE_RE.search(body)
    if match is None:
        return None
    return body[:match.start()] + match.group(1) + '"' + match.group(2)


def _bracket_balance(text: str) -> tuple[int, int]:
    """Open minus closed parentheses and square brackets in *text*."""
    return text.count("(") - text.count(")"), text.count("[") - text.count("]")


def _moved_string_is_whole(data: Any, run: str) -> bool:
    """Whether moving *run* inside a string made that string's brackets whole.

    The quote is only misplaced when the string was missing exactly those
    brackets; a string that was whole already and had a stray bracket after it
    would be made wrong by the move.
    """
    values: list[Any] = [data]
    while values:
        value = values.pop()
        if isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, list):
            values.extend(value)
        elif isinstance(value, str) and value.endswith(run):
            if _bracket_balance(value) == (0, 0) and _bracket_balance(
                value[:-len(run)]
            ) != (0, 0):
                return True
    return False


def _escape_inner_quotes(body: str) -> str | None:
    """*body* with the double quotes inside its string values escaped.

    A program written as a JSON string often carries its own double quotes
    unescaped — ``"code": "f(x, key="a")"`` — so the string ends early and
    the object no longer parses. Inside a string, a quote that is followed
    (after spaces) by ``,``, ``:``, ``}``, ``]`` or the end of the body closes
    it; any other quote is taken as part of the value. ``None`` when nothing
    was escaped.
    """
    out: list[str] = []
    in_string = escaped = changed = False
    for i, char in enumerate(body):
        if not in_string:
            if char == '"':
                in_string = True
            out.append(char)
            continue
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            rest = body[i + 1:].lstrip(" \t\r\n")
            if not rest or rest[0] in ",:}]":
                in_string = False
            else:
                out.append('\\"')
                changed = True
                continue
        out.append(char)
    return "".join(out) if changed else None


def read_call_body(body: str) -> tuple[str, Any] | None:
    """Read the body of a tagged call that is not strict JSON.

    Four shapes are read, in order: JSON whose string values carry raw line
    breaks — a program written out line by line — a Python literal, where the
    arguments are quoted the way Python quotes them, JSON whose last string
    was closed before the brackets that end it (``"print(f([1])")}}``) — read
    only when moving the quote makes that string's brackets whole — and JSON
    whose string values carry unescaped double quotes (``"f(k="a")"``). A body
    that is strict JSON returns ``None``: the ordinary readers own it.

    Args:
        body: The text inside the call's tags.

    Returns:
        ``(how, value)`` where *how* names the shape that was read, or ``None``
        when neither reads it.
    """
    body = body.strip()
    if not body.startswith("{"):
        return None
    try:
        json.loads(body)
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        return "raw line breaks", json.loads(_escape_line_breaks(body, '"'))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    for candidate in (body, _escape_line_breaks(body, "\"'")):
        try:
            tree = ast.parse(candidate, mode="eval")
            value = _literal_value(tree.body)
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
            continue
        if value is not _NOT_LITERAL:
            return "python literal", value
    moved = _move_closing_quote(body)
    if moved is not None:
        run = _MISPLACED_QUOTE_RE.search(body).group(1)  # type: ignore[union-attr]
        for candidate in (moved, _escape_line_breaks(moved, '"')):
            try:
                value = json.loads(candidate)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if _moved_string_is_whole(value, run):
                return "closing quote moved", value
            break
    inner = _escape_inner_quotes(body)
    if inner is not None:
        for candidate in (inner, _escape_line_breaks(inner, '"')):
            try:
                return "inner quotes escaped", json.loads(candidate)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return None


def _read_tagged_call(
    text: str, tools: dict[str, Any] | None
) -> tuple[str, dict[str, Any], str] | None:
    """A ``<tool_call>`` whose body only a lenient reader can read.

    Accepted only when the body names a tool *tools* holds and carries its
    arguments as a mapping; anything else is left for the caller to report as a
    call that could not be read. The closing tag may be missing.

    Returns:
        ``(name, arguments, how)`` or ``None``.
    """
    if not tools or "<tool_call>" not in text:
        return None
    from .structured_output import _extract_balanced

    for opening in re.finditer(r"<tool_call>", text):
        rest = text[opening.end():]
        closed = rest.find("</tool_call>")
        body = rest[:closed] if closed >= 0 else rest
        candidates = [body]
        start = body.find("{")
        if start >= 0:
            blob = _extract_balanced(body[start:])
            if blob and blob.strip() != body.strip():
                candidates.append(blob)
        for candidate in candidates:
            read = read_call_body(candidate)
            if read is None and candidate is not body and not body[:start].strip():
                # A whole, strict JSON object with other text after it inside
                # the tag: the object is the call, the text is not part of it.
                try:
                    read = "object followed by text", json.loads(candidate)
                except (json.JSONDecodeError, TypeError, ValueError):
                    read = None
            if read is None:
                continue
            how, data = read
            if not isinstance(data, dict):
                continue
            inner = data.get("function")
            call = inner if isinstance(inner, dict) else data
            name = call.get("name")
            args = call.get("arguments", call.get("parameters"))
            if isinstance(args, str) and isinstance(name, str) and name in tools:
                args = read_call_arguments(args, tools[name])
            if isinstance(name, str) and name in tools and isinstance(args, dict):
                return name, args, how if closed >= 0 else f"{how}, no closing tag"
    return None


#: What an ``Action:`` value says when the turn takes no action: one of the
#: single words on its own, or a phrase that may run on into a reason.
_NO_ACTION_WORDS = (
    r"\(?[ \t]*(?:"
    r"(?:none|n/?a|null)(?:[ \t]+(?:needed|required|necessary))?[ \t]*\)?[ \t]*\.?"
    r"|(?:no[ _-]?action|no further action|no tool(?:[ \t]+call)?"
    r"|none[ \t]+(?:needed|required|necessary))\b[^\n]*"
    r")"
)
#: An ``Action:`` value wrapped in parentheses or brackets — ``(continue
#: reasoning)``, ``(final answer)``, ``[no action]`` — which is a placeholder
#: for an action rather than the name of one.
_PLACEHOLDER_WORDS = r"(?:\([^()\[\]\n]{1,80}\)|\[[^()\[\]\n]{1,80}\])\.?"
#: An ``Action:`` value that says the turn takes no action.
_NO_ACTION_VALUE_RE = re.compile(r"^" + _NO_ACTION_WORDS + r"$", re.IGNORECASE)
#: An ``Action:`` value that is a placeholder.
_PLACEHOLDER_VALUE_RE = re.compile(r"^" + _PLACEHOLDER_WORDS + r"$")
#: A whole ``Action:`` line that says so, in words or with a placeholder.
_NO_ACTION_LINE_RE = re.compile(
    r"^[ \t]*Action:[ \t]*(?:" + _NO_ACTION_WORDS + "|" + _PLACEHOLDER_WORDS
    + r")[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ACTION_INPUT_LINE_RE = re.compile(r"^[ \t]*Action Input:.*$", re.IGNORECASE | re.MULTILINE)


def declares_no_action(action: str, tools: dict[str, Any] | None = None) -> bool:
    """Whether an ``Action:`` value says the turn takes no action.

    ``None``, ``N/A``, ``null``, ``no action needed``, ``(no tool)`` and their
    like are the words the scaffold's own format leads a model to write when it
    has nothing to call; a phrase may run on into its reason ("No further action
    needed as the answer is clear"). A tool the agent actually holds under such
    a name is still a tool, and a call to it is still a call.

    Args:
        action: The value after ``Action:``, with its argument section removed.
        tools: The agent's tools by name.

    Returns:
        True when the value declares that no action is taken.
    """
    value = (action or "").strip()
    if not value:
        return False
    if not _NO_ACTION_VALUE_RE.match(value):
        return placeholder_action(value, tools) is not None
    held = {str(name).lower() for name in (tools or {})}
    return value.strip("().").strip().lower() not in held


def placeholder_action(action: str, tools: dict[str, Any] | None = None) -> str | None:
    """The placeholder an ``Action:`` value is, or ``None`` when it is not one.

    A value wrapped in parentheses or brackets — ``(continue reasoning)``,
    ``(final answer)``, ``[none]`` — stands in for an action without naming
    one. Models copy the shape from a transcript line that uses it, and a turn
    that writes it has taken no action. A value whose inside is the name of a
    held tool is still that tool.

    Args:
        action: The value after ``Action:``, with its argument section removed.
        tools: The agent's tools by name.

    Returns:
        The placeholder as written, or ``None``.
    """
    value = (action or "").strip()
    if not value or not _PLACEHOLDER_VALUE_RE.match(value):
        return None
    inside = value.rstrip(".").strip()[1:-1].strip().lower()
    held = {str(name).lower() for name in (tools or {})}
    if inside in held or value.lower() in held:
        return None
    return value


def text_after_declaration(text: str) -> str:
    """What a turn wrote after its last whole ``Action: None`` line.

    The ``Action Input:`` line that follows the declaration belongs to it and
    is left out.
    """
    matches = list(_NO_ACTION_LINE_RE.finditer(text or ""))
    if not matches:
        return ""
    last = matches[-1]
    rest = text[last.end():]
    rest = _ACTION_INPUT_LINE_RE.sub("", rest, count=1)
    return rest.strip()


#: A written ``Action:`` line (not ``Action Input:``) and the value after it.
_WRITTEN_ACTION_LINE_RE = re.compile(r"^[ \t]*Action:[ \t]*(\S[^\n]*)$", re.IGNORECASE | re.MULTILINE)
#: A label that states the result of a call or the run's answer. After an
#: action nothing has run yet, so whatever such a label introduces there is the
#: model's own writing, not a result.
_RESULT_OR_ANSWER_RE = re.compile(
    r"^[ \t]*(Observation:|Answer:|The answer is:)|Final Answer:",
    re.IGNORECASE | re.MULTILINE,
)
#: The labels the ReAct reader takes an answer from, which may legitimately come
#: before any action (an answer that states itself first is an answer).
_ANSWER_LABEL_RE = re.compile(
    r"Final Answer:|^[ \t]*(?:Answer:|The answer is:)", re.IGNORECASE | re.MULTILINE,
)
#: Logged each time a turn's text after its written action is discarded, so the
#: firings can be counted.
DISCARDED_AFTER_ACTION_LOG = (
    "[reader] a written action is run and what the model wrote after it is "
    "discarded (%s, %d characters)"
)


def split_at_unrun_action(
    text: str, tools: dict[str, Any] | None = None,
) -> tuple[str, str, str] | None:
    """Split a turn at the point its written action stops being the model's to write.

    A turn that writes ``Action:`` / ``Action Input:`` has asked for a tool to
    run; nothing has run yet. Anything it goes on to write under an
    ``Observation:`` label is a result the model made up, and an ``Answer`` or
    ``Final Answer`` after that is an answer built on it. Only text from the
    action on is looked at: an answer, or a quoted "Observation:", that comes
    before any action is left exactly as it is.

    ``Action: None`` (a declaration that no action is taken) is not a call and
    is skipped: the first action that names something is the one that counts.
    ``Action: Final Answer`` is the model answering, and an answer stated before
    any call is an answer; both are left to the reader's own handling.

    Args:
        text: The turn's text.
        tools: The agent's tools by name.

    Returns:
        ``(head, discarded, kind)`` — the text to read the call from (up to and
        including the action; it starts at the action when a declaration came
        before it), the text after it, and ``"observation"`` or ``"answer"`` for
        the label that began the discarded part — or ``None`` when the turn has
        no written action followed by such a label.
    """
    if not text or "action" not in text.lower():
        return None
    match = None
    declared_before = False
    for candidate in _WRITTEN_ACTION_LINE_RE.finditer(text):
        value = action_name(candidate.group(1).strip().replace('"', "").replace("'", ""))
        if declares_no_action(value, tools):
            declared_before = True
            continue
        if value.lower() in ("final answer", "finalanswer", "answer"):
            return None
        match = candidate
        break
    if match is None or _ANSWER_LABEL_RE.search(text[:match.start()]):
        return None
    after = _RESULT_OR_ANSWER_RE.search(text, match.end())
    if after is None:
        return None
    label = after.group(0).strip().lower()
    kind = "observation" if label.startswith("observation") else "answer"
    start = match.start() if declared_before else 0
    return text[start:after.start()].strip(), text[after.start():], kind


def name_positional_arguments(
    tool_name: str, positional: list[Any], tools: dict[str, Any] | None,
) -> dict[str, Any]:
    """Give several positional arguments the names the tool declares.

    ``file_operations('write', 'greet.py', 'print(1)')`` carries its values in
    the order the tool's own parameters are declared, which is the only thing
    that can name them. When they cannot be named — more values than the tool
    declares parameters, or a tool the agent does not hold — the call carries
    no arguments: keeping only the first value would run a different call from
    the one the model wrote.

    Args:
        tool_name: The name the call resolved to.
        positional: The positional values read from the call, in order.
        tools: The agent's tools by name, or ``None`` when unavailable.

    Returns:
        The values under the tool's declared parameter names, or ``{}`` when
        they cannot be named.
    """
    tool = (tools or {}).get(tool_name)
    parameters = getattr(getattr(tool, "metadata", None), "parameters", None)
    if parameters:
        names = [p.name for p in parameters]
        if len(names) >= len(positional):
            return dict(zip(names, positional, strict=False))
    logger.info(POSITIONAL_NOT_NAMED_LOG, tool_name, len(positional))
    return {}


def call_input(arguments: dict[str, Any]) -> str:
    """The input a decoded call hands the agent's executor.

    A value still waiting to be matched to the tool's parameters travels as the
    text it is, so the executor's mapper binds it; anything else as the JSON
    text of the mapping.
    """
    raw = arguments.get(RAW_INPUT_KEY) if isinstance(arguments, dict) else None
    if isinstance(raw, str) and len(arguments) == 1:
        return raw
    return json.dumps(arguments, default=str)


def missing_required_arguments(tool: Any, arguments: Any) -> list[str]:
    """The required parameters a call's arguments leave without a value.

    Only arguments that are a mapping — an object, or the JSON text of one —
    are judged: plain text is bound to the tool's parameter by the agent's
    mapper and is never refused here. A tool that declares no required
    parameter, or that takes its input through an ``execute`` of its own, is
    never refused either. A parameter the tool accepts under a synonym (an
    ``action`` passed for ``operation``) counts as supplied.

    Args:
        tool: The tool the call names.
        arguments: The call's arguments, as a mapping or as text.

    Returns:
        The names of the required parameters with no value, in declaration
        order; empty when the call can be dispatched.
    """
    parameters = getattr(getattr(tool, "metadata", None), "parameters", None) or []
    required = [str(p.name) for p in parameters if getattr(p, "required", False)]
    if not required:
        return []
    from effgen.tools.base_tool import BaseTool

    if isinstance(tool, BaseTool) and type(tool).execute is not BaseTool.execute:
        return []
    mapping: Any
    if arguments is None:
        mapping = {}
    elif isinstance(arguments, dict):
        mapping = arguments
    elif isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            mapping = {}
        else:
            try:
                mapping = json.loads(text)
            except (json.JSONDecodeError, TypeError, ValueError):
                return []
    else:
        return []
    if not isinstance(mapping, dict):
        return []
    if mapping.get(RAW_INPUT_KEY) not in (None, ""):
        return []
    supplied = {k: v for k, v in mapping.items() if v is not None}
    normalize = getattr(tool, "_normalize_selector", None)
    if callable(normalize):
        try:
            supplied = normalize(dict(supplied))
        except Exception:  # noqa: BLE001 - a tool's own hook never blocks a call
            logger.debug("selector normalisation failed", exc_info=True)
    return [name for name in required if supplied.get(name) is None]


#: The languages a fenced block may be tagged with to be run by a code tool
#: whose call the turn opened and left empty; an untagged block counts too.
_RUNNABLE_FENCE_TAGS = frozenset({"", "python", "py", "python3"})
_FENCED_BLOCK_RE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)

#: Logged when a turn's fenced block is run as the call it opened and left empty.
FENCED_BLOCK_CALL_LOG = "[call] ran the fenced block before an empty call to '%s'"


def fenced_block_call(
    text: str, tools: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """The call a turn announced with a fenced program and an empty call tag.

    A model often writes its program in a fenced block, says it will run it,
    and then opens a call it leaves empty. When the agent holds exactly one tool
    of the code-execution category, that tool declares exactly one required
    parameter and it takes a string, the last fenced block written before the
    opening is that call's argument. A block after the opening, a turn that
    opened no call, a call whose unread text names another held tool, or an
    agent holding no such tool (or several) gives ``None``.

    Args:
        text: The turn's text.
        tools: The agent's tools by name.

    Returns:
        ``(tool name, arguments)`` or ``None``.
    """
    if not text or not tools or "```" not in text:
        return None
    from effgen.tools.base_tool import ParameterType, ToolCategory

    executors = [
        (name, tool) for name, tool in tools.items()
        if getattr(getattr(tool, "metadata", None), "category", None)
        is ToolCategory.CODE_EXECUTION
    ]
    if len(executors) != 1:
        return None
    name, tool = executors[0]
    required = [
        p for p in (getattr(tool.metadata, "parameters", None) or [])
        if getattr(p, "required", False)
    ]
    if len(required) != 1 or required[0].type is not ParameterType.STRING:
        return None
    blocks = list(_FENCED_BLOCK_RE.finditer(text))
    spans = [(m.start(), m.end()) for m in blocks]
    opening = -1
    for marker in CALL_OPENING_MARKERS:
        at = text.rfind(marker)
        while at >= 0 and any(a <= at < b for a, b in spans):
            at = text.rfind(marker, 0, at)
        opening = max(opening, at)
    if opening < 0:
        return None
    after = text[opening:]
    if any(
        other != name and re.search(rf"(?<![\w-]){re.escape(str(other))}(?![\w-])", after)
        for other in tools
    ):
        # What follows the opening names another held tool: the turn asked for
        # that call, and running the program instead would be a different one.
        return None
    before = [
        m for m in blocks
        if m.end() <= opening and m.group(1).lower() in _RUNNABLE_FENCE_TAGS
        and m.group(2).strip()
    ]
    if not before:
        return None
    return str(name), {str(required[0].name): before[-1].group(2).strip("\n")}


#: Logged when a call's positional values could not be given parameter names.
POSITIONAL_NOT_NAMED_LOG = (
    "[call] positional values could not be named for '%s' (%d values); the "
    "call carries no arguments"
)


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
        read_leniently: The call was read by a reader that accepts more than
            strict JSON; ``read_how`` names the shape. The caller decides
            whether to run such a call.
        read_how: How a lenient read succeeded, for the caller's log.
        declared_no_action: The turn wrote an ``Action:`` that says it takes
            none (``Action: None``). No tool is named by it.
        after_declaration: What the turn wrote after that declaration, for the
            loop to read again; empty when nothing followed it.
        placeholder: The placeholder the declaration was written as —
            ``(continue reasoning)`` and its like — or ``""`` when it was
            written in words.
    """
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    thought: str | None = None
    final_answer: str | None = None
    is_tool_call: bool = False
    call_id: str | None = None
    reasoning: str = ""
    read_leniently: bool = False
    read_how: str = ""
    declared_no_action: bool = False
    after_declaration: str = ""
    placeholder: str = ""


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
    def parse_response(
        self, text: str, tools: dict[str, Any] | None = None, *, lenient: bool = False,
    ) -> ToolCallResult:
        """Parse a model response and extract tool call or final answer.

        Args:
            text: Raw model response text.
            tools: Dict mapping tool name -> tool object (for validation).
            lenient: Also read a tagged call whose body is not strict JSON.
                Off by default: a reader that accepts more changes which
                readings the caller ever sees, so the caller asks for it.

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

    def parse_response(
        self, text: str, tools: dict[str, Any] | None = None, *, lenient: bool = False,
    ) -> ToolCallResult:
        """Parse ReAct formatted response.

        This is extracted from ``Agent._parse_react_response()`` with the
        same logic and patterns, returning a ``ToolCallResult`` instead of
        a plain dict.

        Args:
            text: Raw model response text.
            tools: Dict mapping tool name -> tool object (for validation).
            lenient: Names a reader this format does not have; accepted so
                every strategy takes the same arguments.

        Returns:
            ToolCallResult with the call or the final answer this format
            states.
        """
        del lenient
        result = ToolCallResult(raw_text=text)

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid response text for parsing: {type(text)}")
            return result

        try:
            # --- A written action ends the turn ---
            # Nothing has run when the model writes an action, so a result or an
            # answer it writes after one is its own invention. The action is read
            # from the text up to that point and the rest is discarded, whatever
            # stop sequences the request carried.
            unrun = split_at_unrun_action(text, tools)
            if unrun is not None:
                head, discarded, kind = unrun
                action_result = self.parse_response(head, tools)
                if action_result.is_tool_call:
                    logger.info(DISCARDED_AFTER_ACTION_LOG, kind, len(discarded))
                    return action_result

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

                        # "Action: None" says the turn takes no action. It
                        # names no tool, so it is not dispatched as a call to
                        # one; what the turn wrote after it is handed to the
                        # loop to read again.
                        if declares_no_action(action, tools):
                            result.declared_no_action = True
                            result.after_declaration = text_after_declaration(text)
                            result.placeholder = placeholder_action(action, tools) or ""
                            return result

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

    def parse_response(
        self, text: str, tools: dict[str, Any] | None = None, *, lenient: bool = False,
    ) -> ToolCallResult:
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
            lenient: Also read a tagged call whose body is a Python literal or
                carries raw line breaks inside its JSON strings. Off by
                default, so a body this reader alone can read is left
                unparsed and the caller's other readers still see the turn.

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
                if not isinstance(arguments, dict):
                    arguments = read_call_arguments(
                        arguments, (tools or {}).get(str(tool_name))
                    )
                if tool_name:
                    result.tool_name = tool_name
                    result.arguments = arguments
                    result.is_tool_call = True
                    logger.debug(f"Parsed Qwen-style tool call: {tool_name}")
                    return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse Qwen tool call JSON: {e}")

        # --- A tagged call only a lenient reader can read ---
        # A program written with raw line breaks inside its JSON string, or
        # arguments quoted the way Python quotes them. The call is run only when
        # it names a tool this agent holds and carries its arguments as a
        # mapping; anything else is left to be reported as unreadable.
        #
        # Asked for, never assumed: reading a body here settles the turn, and a
        # caller that would not run such a call needs the turn to reach its
        # other readers instead — the same text often carries the call in a
        # form one of them does read.
        read = _read_tagged_call(text, tools) if lenient else None
        if read is not None:
            result.tool_name, result.arguments, result.read_how = read
            result.is_tool_call = True
            result.read_leniently = True
            return result

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
                    if not isinstance(arguments, dict):
                        arguments = read_call_arguments(
                            arguments, (tools or {}).get(str(tool_name))
                        )
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
        if text_stripped and not any(marker in text for marker in (
            *CALL_OPENING_MARKERS,
            "Thought:", "Action:", "Tool:",
            "<|channel>", "<channel|>",
        )):
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

    def parse_response(
        self, text: str, tools: dict[str, Any] | None = None, *, lenient: bool = False,
    ) -> ToolCallResult:
        """Try native parsing first, then ReAct.

        Args:
            text: Raw model response text.
            tools: Dict mapping tool name -> tool object (for validation).
            lenient: Passed to the native reader, which alone has a lenient
                body reader. It settles the turn when it succeeds, so asking
                for it decides whether the turn reaches the ReAct reader at
                all.

        Returns:
            ToolCallResult from whichever reader read the turn.
        """
        result = self._native.parse_response(text, tools, lenient=lenient)
        if result.is_tool_call or result.final_answer:
            logger.debug("Hybrid strategy: native parsing succeeded")
            return result

        logger.debug("Hybrid strategy: native parsing failed, trying ReAct")
        return self._react.parse_response(text, tools, lenient=lenient)

    def format_tools_for_prompt(self, tools: list) -> Any:
        """Use native format (JSON Schema definitions)."""
        return self._native.format_tools_for_prompt(tools)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_strategy(
    mode: str = "auto",
    model: Any | None = None,
    probe: Any | None = None,
) -> ToolCallingStrategy:
    """Create the appropriate tool calling strategy.

    Args:
        mode: One of "auto", "native", "react", "hybrid".
        model: The model instance (checked for supports_tool_calling).
        probe: What a capability probe measured for *model*
            (:class:`~effgen.models.capability_probe.ToolCallingProbe`), or
            ``None``. Read only for ``"auto"``: a model whose native tool calls
            the probe saw go unresolved, while the text frame resolved them,
            gets the ReAct strategy. Without a probe the result is what the
            model's declaration alone gives.

    Returns:
        ToolCallingStrategy instance.
    """
    if mode == "auto" and getattr(probe, "strategy", None) == "react":
        logger.info(
            "Capability probe: native tool calls did not resolve and the text "
            "frame did; using ReAct strategy"
        )
        return ReActStrategy()
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
