"""Putting back a computed result that an answer left out.

Some tools return the answer itself rather than material to build one from: a
puzzle solution as a list of moves, a graph colouring, a schedule, a sorted
list. A model handed one of those has to restate the whole thing, and a small
model routinely summarises it instead — the tool prints seven moves and the
answer is "the final answer is 7". The framework then returns an answer its own
observations do not support, and the work the tool did is thrown away even
though the run is still holding it.

So when the last thing a run executed returned a structured result and the
answer does not state all of it, the result is appended to the answer. The
model's own sentence is kept: this adds back evidence the run already had, it
does not overrule what the model said.

Three conditions keep it narrow, and each of them is load-bearing.

**Only tools that compute an answer.** A search tool returns source material
that the model still has to read and answer from, so appending three raw
passages to "who wrote this novel?" replaces an answer with a pile of evidence.
The same declared category that decides what a tool is *told* to do decides this
(:func:`~effgen.prompts.tool_contract.is_execution_tool`), so "which tools must
actually run" and "whose result is the answer" can never disagree.

**Only a structured result.** Fewer than :data:`RELAY_MIN_LINES` non-empty lines
is a value rather than a listing, and a value is what an answer is expected to
paraphrase. Every arithmetic result is one line, so a run whose tools return
single values is untouched by construction.

**Only when the whole result is missing.** The frequent failure is a partial
restatement — the first two entries and then "and so on" — which carries some of
the lines and is still not an answer. The test is therefore whether the answer
states *all* of the result's lines, not whether it states any of them. The one
exception is an answer that *is* one of those lines: a result listing several
candidates is one the model was meant to choose from, and appending it whole
would replace the choice with the menu.

The pairing of a result with the tool that produced it is read from the run's
own call records, each of which carries the tool name beside what it returned.
Recovering that pairing from the prompt text instead works only while the text
has the shape the reader expects, and a pairing that fails there fails silently:
the result is dropped rather than appended, which is the outcome this exists to
prevent.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from ..prompts.tool_contract import is_execution_tool
from .tool_call_record import MAX_RESULT_CHARS, ToolCall

logger = logging.getLogger(__name__)

#: Non-empty lines a result needs before an answer is expected to carry all of
#: it. One line is a value; two is a value and a note about it. Three is the
#: shortest thing that reads as a listing, and it is the point below which
#: requiring an answer to repeat a result verbatim would be requiring it to stop
#: answering in prose.
RELAY_MIN_LINES = 3

#: How a dispatch that produced no usable result opens, in the text the loop
#: puts in front of the model. ``Error executing tool`` is what
#: :meth:`~effgen.core.agent_tool_execution.AgentToolExecutionMixin._execute_tool`
#: returns when the call itself failed; ``Tool execution failed:`` is what it
#: returns for a tool that reported failure through its own result; and
#: ``Error:`` is what it builds from a code executor's ``stderr`` or ``error``.
#: A run holding one of those is holding a report about a failure, and a report
#: about a failure is not a result to hand back as the answer.
FAILED_RESULT_PREFIXES = (
    "error executing tool",
    "tool execution failed:",
    "error:",
)


def unrelayed_result(
    answer: str,
    calls: Sequence[ToolCall],
    tools: Mapping[str, Any],
) -> str | None:
    """The computed result *answer* dropped, or ``None`` when it dropped none.

    Reads back over *calls*, newest first, for the last result a tool of an
    execution category returned. A call that failed, returned nothing, or
    reported its failure in its output (see :data:`FAILED_RESULT_PREFIXES`) is
    passed over — a traceback is not a result to hand back as one. The search is
    then decided by the first result it does find: when that is too short to be
    a listing the run's most recent computation is a value, and an older listing
    behind it has been superseded rather than lost.

    Args:
        answer: The answer as it would be returned, already sanitized.
        calls: The run's tool-call records, oldest first.
        tools: The tools the agent holds, keyed by name, so a call can be
            matched to the category its tool declares.

    Returns:
        The result text to append, or ``None`` when nothing should be appended.
    """
    if not answer or not calls:
        return None
    for call in reversed(list(calls)):
        if not is_execution_tool(tools.get(call.name)):
            continue
        if call.error or not call.result:
            continue
        result = call.result.strip()
        if not result or result.lower().startswith(FAILED_RESULT_PREFIXES):
            continue
        if len(call.result) > MAX_RESULT_CHARS:
            # The record keeps a prefix of a result this long, and appending a
            # prefix is the partial relay this is here to prevent.
            logger.debug(
                "result relay: '%s' returned more than a record holds; "
                "not appending a shortened result",
                call.name,
            )
            return None
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if len(lines) < RELAY_MIN_LINES:
            return None
        flattened = " ".join(answer.split()).lower()
        entries = [" ".join(line.split()).lower() for line in lines]
        if flattened in entries:
            # The answer is one entry of the result, so the result was a list of
            # candidates and the model picked from it — decoding a message under
            # every shift and naming the one that reads as English, say.
            # Appending the whole listing to that replaces the choice with the
            # menu it was chosen from.
            logger.debug(
                "result relay: the answer is one entry of what '%s' returned; "
                "not appending the rest of it",
                call.name,
            )
            return None
        stated = sum(1 for entry in entries if entry in flattened)
        if stated == len(lines):
            return None
        logger.info(
            "result relay: the answer states %d of the %d lines '%s' returned; "
            "appending the result",
            stated,
            len(lines),
            call.name,
        )
        return result
    return None


def relay_result(
    answer: str,
    calls: Sequence[ToolCall],
    tools: Mapping[str, Any],
) -> str:
    """*answer*, with a computed result it dropped appended to it.

    Returns *answer* unchanged when :func:`unrelayed_result` finds nothing to
    put back, which is every run whose tools return values, whose tools return
    source material, or whose answer already states the result in full.

    Args:
        answer: The answer as it would be returned, already sanitized.
        calls: The run's tool-call records, oldest first.
        tools: The tools the agent holds, keyed by name.

    Returns:
        The answer, with the dropped result appended to it when there was one.
    """
    result = unrelayed_result(answer, calls, tools)
    if result is None:
        return answer
    return f"{answer.rstrip()}\n\n{result}"


__all__ = [
    "FAILED_RESULT_PREFIXES",
    "RELAY_MIN_LINES",
    "relay_result",
    "unrelayed_result",
]
