"""Loop policy for a tool-calling turn loop.

An agent that drives a model's tool calling has to decide more than "did the
model ask for a tool": whether the same call is repeating, whether a tool has
reproduced a result it already returned, when to stop offering tools so the
model must write prose, and whether a turn wrote its call out as text instead of
making it. :class:`NativeToolLoop` holds that policy and the state it needs, so
the blocking loop in :mod:`effgen.core.agent_react` and the streaming loop in
:mod:`effgen.core.agent_stream_native` reach the same decisions from the same
code rather than from two copies of it.

The class is state plus predicates. It never calls a model, never dispatches a
tool and never builds a prompt — the caller does all of that and tells the loop
what happened. This module imports nothing from ``agent.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..prompts.tool_contract import is_execution_tool
from ..tools.base_tool import ToolCategory
from .agent_runtime import (
    NUDGE_HAVE_ANSWER,
    NUDGE_HAVE_RESULTS,
    written_call_only,
)
from .tool_call_record import ToolCall, truncate_result

logger = logging.getLogger(__name__)

#: Prefix :meth:`Agent._execute_tool` puts on a failed dispatch. A failed call is
#: not evidence of a repeated result, so the result-based short circuit skips it.
TOOL_ERROR_PREFIX = "Error executing tool"

#: How many calls to one tool read as circling when the inputs keep changing.
#: Small models re-format the same call rather than repeating it byte for byte,
#: so the exact-pair check alone never fires for them.
#:
#: This counts *drift*, not work. It was 5, which is below the length of an
#: ordinary multi-step task: a word problem with four arithmetic steps and one
#: check spends five calls doing exactly what it was asked to do, and the guard
#: read that as a loop and ended the run holding an intermediate value. Twelve
#: distinct calls to one tool is past the length of any chain of work a single
#: tool is asked to carry, so only a model that really is circling reaches it.
FUZZY_LOOP_THRESHOLD = 12

#: The same threshold for a tool whose job is to chew through data, where
#: several calls in a row are the normal shape of the work — so the count that
#: reads as circling sits higher again.
FUZZY_LOOP_THRESHOLD_DATA = 16

#: The lowest a drift threshold may be driven by a small iteration budget. Four
#: steps and a check is five calls of legitimate work, so a guard that fires
#: below six is guarding against work. A run whose budget cannot reach this
#: floor ends at its iteration cap instead, which is what actually happened.
FUZZY_LOOP_FLOOR = 6

#: Batch tool runs allowed before the loop stops offering tools. A model that
#: answers a multi-call turn with another multi-call turn is not converging.
#:
#: Raised from 2 for the same reason as the drift threshold: a model that emits
#: two calls per turn covers a four-step task in two turns and then has its
#: tools taken away with the task unfinished. Six multi-call turns is past the
#: shape of any batched work, so the cap still catches a model that will never
#: converge and no longer catches one that was going to finish.
MAX_BATCH_TOOL_RUNS = 6

#: Calls to one tool before the loop reminds the model it already has results.
#: See :meth:`NativeToolLoop.post_tool_nudge` for why this is not 1.
NUDGE_AFTER_CALLS = 6


@dataclass
class LoopCheck:
    """What :meth:`NativeToolLoop.check_action` found about one proposed call."""

    #: ``(action, normalized_input)``, the key the exact-repeat check uses.
    pair: tuple[str, str]
    #: How many times this tool was already called in this run.
    action_call_count: int
    #: The same tool with the same input has already been dispatched.
    is_exact_loop: bool
    #: The same tool has been called enough times to read as a loop.
    is_fuzzy_loop: bool

    @property
    def is_loop(self) -> bool:
        """True when either repeat check fired."""
        return self.is_exact_loop or self.is_fuzzy_loop

    @property
    def loop_type(self) -> str:
        """A short label for the log line, or ``""`` when nothing fired."""
        if self.is_exact_loop:
            return "exact"
        if self.is_fuzzy_loop:
            return f"fuzzy ({self.action_call_count + 1} calls)"
        return ""


@dataclass
class NativeToolLoop:
    """Per-run state and the decisions a tool-calling loop makes from it.

    Args:
        tools: The tools the agent holds, by name — the same mapping the loop
            dispatches against.
        nudge_cap: The configured iteration cap, used to decide when a turn is
            close enough to the limit to ask for an answer outright.
    """

    tools: dict[str, Any]
    nudge_cap: int = 10

    #: ``(action, normalized_input)`` for every call dispatched so far.
    previous_actions: list[tuple[str, str]] = field(default_factory=list)
    #: What each dispatched ``(action, normalized_input)`` returned, so proposing
    #: that call again can be answered from the record instead of ending the
    #: run. See :meth:`cached_result`.
    results_by_pair: dict[tuple[str, str], str] = field(default_factory=dict)
    #: How many times each pair has been proposed again after it first ran. One
    #: replay is a model that did not read the observation; several mean it is
    #: not going to move on.
    replays_by_pair: dict[tuple[str, str], int] = field(default_factory=dict)
    #: ``(action, normalized_result)`` for every call that returned without error.
    previous_results: list[tuple[str, str]] = field(default_factory=list)
    #: Multi-call turns dispatched as one batch.
    batch_tool_runs: int = 0
    #: Set once a repeat left no usable partial answer: stop offering tools so
    #: the model has to write the answer from what it already has.
    force_text_answer: bool = False
    #: The tool whose call a turn wrote out as text instead of making.
    written_call: str | None = None
    #: How many turns did that.
    written_call_turns: int = 0
    #: Tools this run actually dispatched.
    executed_tools: set[str] = field(default_factory=set)
    #: One record per dispatched call, in call order — what
    #: ``AgentResponse.tool_calls`` reports back to the caller.
    calls: "list[ToolCall]" = field(default_factory=list)
    #: How many times this run has been sent back for answering without running
    #: an execution tool. Capped at one; see :meth:`note_execution_refusal`.
    execution_refusals: int = 0
    #: Set by a refusal and cleared by the turn that spends it, so it constrains
    #: exactly one turn. See :meth:`take_forced_tool_call`.
    force_tool_call: bool = False

    # ------------------------------------------------------------------
    # Offering tools
    # ------------------------------------------------------------------
    def tools_suppressed(self) -> bool:
        """True when this run should stop passing tool definitions to the model.

        Either the model has spent its allowance of multi-call turns, or a
        repeat was detected with nothing usable to fall back on. In both cases
        re-offering the same tools reproduces the same turn.
        """
        return self.batch_tool_runs >= MAX_BATCH_TOOL_RUNS or self.force_text_answer

    def note_batch_run(self) -> None:
        """Record that a turn dispatched several calls at once."""
        self.batch_tool_runs += 1

    # ------------------------------------------------------------------
    # Requiring a call
    # ------------------------------------------------------------------
    def execution_tools(self) -> list[str]:
        """The names of the held tools that do work the model cannot do itself.

        Read from each tool's declared category through
        :func:`~effgen.prompts.tool_contract.is_execution_tool`, so the set is
        whatever the tools say they are.
        """
        return [
            name for name, tool in self.tools.items() if is_execution_tool(tool)
        ]

    def note_execution_refusal(self) -> str | None:
        """Answer without running the executor: refuse it once, name the tool.

        An agent holding a code executor and answering with no call has reported
        a result nothing produced — it described what the code would print. That
        answer is not accepted the first time: the turn goes back with a nudge
        naming the tool, and the turn after it is sent requiring a call.

        **Only the first.** A model that declines twice will decline again, and
        the iteration budget buys more elsewhere; the second refusal is
        accepted, so a run cannot be spent circling on this.

        Returns:
            The tool name to name in the nudge, or ``None`` when this run holds
            no execution tool, has already dispatched a call, or has already
            been sent back once.
        """
        if self.execution_refusals or self.calls:
            return None
        names = self.execution_tools()
        if not names:
            return None
        self.execution_refusals += 1
        self.force_tool_call = True
        logger.info(
            "execution refusal: answered with no call while holding '%s'; "
            "requiring a call on the next turn",
            names[0],
        )
        return names[0]

    def take_forced_tool_call(self) -> bool:
        """Whether this turn should require a tool call. Spent on read.

        Reading clears the flag, so the constraint covers exactly one turn and
        never the turn after it — which has to be free to state the answer, and
        would otherwise be forced to call a tool it no longer needs.

        The earliest turn this can return True for is the second: nothing sets
        the flag but a refusal, and a refusal is a judgement on a turn that has
        already been generated. **Turn one is never constrained**, and that is
        deliberate rather than incidental. Given room to reason first, a small
        model writes a correct program and then calls the executor with it; a
        model emitting a native tool call usually returns empty ``content``, so
        forcing the opening turn buys the call at the cost of the reasoning that
        makes the call worth anything. An earlier attempt at this fix forced
        every turn and was reverted for exactly that.
        """
        forced, self.force_tool_call = self.force_tool_call, False
        return forced

    # ------------------------------------------------------------------
    # Repeat detection
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_input(action_input: str) -> str:
        """Return *action_input* in a form two equivalent calls share.

        JSON arguments are re-serialized with sorted keys so the same call
        written with its keys in a different order compares equal; anything that
        is not JSON is compared as trimmed text.
        """
        normalized = (action_input or "").strip()
        try:
            return json.dumps(json.loads(normalized), sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return normalized

    def fuzzy_threshold(self, action: str) -> int:
        """How many calls to *action* read as circling when the inputs differ.

        The count comes from the tool's declared category — a data-processing
        tool is expected to be called more often than one that answers a
        question — and is then bounded by the run's own iteration budget.

        Both bounds matter. A threshold the budget cannot reach is not a guard,
        it is dead code, and the run ends at its cap reporting that it ran out
        of iterations. A threshold driven below :data:`FUZZY_LOOP_FLOOR` by a
        short budget is worse: it fires on work. So the count sits one below the
        cap when the cap is the smaller of the two, which also leaves the turn
        the loop needs to ask for an answer before it gives up.
        """
        tool = self.tools.get(action)
        category = getattr(getattr(tool, "metadata", None), "category", None)
        declared = (
            FUZZY_LOOP_THRESHOLD_DATA
            if category == ToolCategory.DATA_PROCESSING
            else FUZZY_LOOP_THRESHOLD
        )
        return max(FUZZY_LOOP_FLOOR, min(declared, self.nudge_cap - 1))

    def check_action(self, action: str, action_input: str) -> LoopCheck:
        """Report whether dispatching *action* now would repeat earlier work.

        Reads state only; :meth:`record_action` is what remembers the call.
        """
        pair = (action, self.normalize_input(action_input))
        known = action in self.tools
        action_call_count = sum(1 for a, _ in self.previous_actions if a == action)
        exact_count = sum(1 for seen in self.previous_actions if seen == pair)
        return LoopCheck(
            pair=pair,
            action_call_count=action_call_count,
            is_exact_loop=exact_count >= 1 and known,
            is_fuzzy_loop=action_call_count >= self.fuzzy_threshold(action) and known,
        )

    def record_action(self, check: LoopCheck) -> None:
        """Remember the call *check* describes as dispatched."""
        self.previous_actions.append(check.pair)

    #: Times one exact call may be answered from the record before the repeat
    #: is read as a model that is not going to move on. One replay covers the
    #: common case — a model restating its plan before reading the observation
    #: — and two bound a run that would otherwise spin.
    MAX_REPLAYS_PER_PAIR = 2

    def record_pair_result(self, check: LoopCheck, tool_result: str) -> None:
        """Remember what the call *check* describes returned.

        Only a dispatch that succeeded is kept: replaying an error teaches the
        model nothing it has not already seen, and the point of the record is to
        hand back a result worth having.
        """
        if isinstance(tool_result, str) and tool_result.startswith(TOOL_ERROR_PREFIX):
            return
        self.results_by_pair.setdefault(check.pair, tool_result)

    def cached_result(self, check: LoopCheck) -> str | None:
        """Return what this exact call returned before, or ``None``.

        A model proposing a call it already made is usually not looping. It has
        restated its plan without reading the observation, or lost the result
        while re-deriving it. Both are answered by handing the recorded result
        back and letting the run continue: a pure computation is idempotent, so
        running it again returns what it returned before, and that is what the
        repeat is answered with.

        Returns ``None`` once the same pair has been replayed
        :attr:`MAX_REPLAYS_PER_PAIR` times, so a run that really is stuck still
        reaches the loop-breaking path.
        """
        result = self.results_by_pair.get(check.pair)
        if result is None:
            return None
        seen = self.replays_by_pair.get(check.pair, 0)
        if seen >= self.MAX_REPLAYS_PER_PAIR:
            return None
        self.replays_by_pair[check.pair] = seen + 1
        return result

    def record_execution(
        self,
        action: str,
        *,
        arguments: Any = None,
        result: Any = None,
        duration: float | None = None,
        error: str | None = None,
        iteration: int | None = None,
    ) -> ToolCall:
        """Remember that *action* actually ran, and what it did.

        The detail becomes one entry in :attr:`calls`, which is what
        ``AgentResponse.tool_calls`` reports. A dispatch that failed is still a
        call the run made, so it is recorded with its *error* rather than
        dropped.

        Args:
            action: The tool's registered name.
            arguments: The input as the model supplied it — text on the ReAct
                path, a parsed dict on the native one.
            result: What the tool returned, truncated for the record.
            duration: Wall-clock seconds the dispatch took, when measured.
            error: The failure message, when the caller already has one. A
                result carrying the dispatch-failure prefix supplies it
                otherwise.
            iteration: The 1-based loop iteration the call was made on.

        Returns:
            The record appended, so a caller can amend it in place.
        """
        self.executed_tools.add(action)
        text = truncate_result(result)
        if error is None and isinstance(text, str) and text.startswith(TOOL_ERROR_PREFIX):
            error = text
        call = ToolCall(
            name=action,
            arguments=arguments,
            result=text,
            duration=duration,
            error=error,
            iteration=iteration,
        )
        self.calls.append(call)
        return call

    # ------------------------------------------------------------------
    # Result repeats
    # ------------------------------------------------------------------
    @staticmethod
    def _result_key(action: str, tool_result: str) -> tuple[str, str]:
        return (action, " ".join(tool_result.split())[:500])

    def result_is_repeat(self, action: str, tool_result: str) -> bool:
        """True when *action* has already returned this result in this run.

        A tool that reproduces its own output means the answer is settled: the
        model is re-deriving something it already has. A failed dispatch is
        never a repeat.
        """
        if tool_result.startswith(TOOL_ERROR_PREFIX):
            return False
        return self._result_key(action, tool_result) in self.previous_results

    def record_result(self, action: str, tool_result: str) -> None:
        """Remember what *action* returned, unless the dispatch failed."""
        if tool_result.startswith(TOOL_ERROR_PREFIX):
            return
        self.previous_results.append(self._result_key(action, tool_result))

    # ------------------------------------------------------------------
    # Nudges
    # ------------------------------------------------------------------
    def post_tool_nudge(
        self, iteration: int, action_call_count: int, tool_result: str
    ) -> str | None:
        """Return the line to append after a tool ran, or ``None`` for silence.

        Near the cap the loop asks for the answer outright. Away from it, the
        reminder is for a model still calling the same tool long after it has
        what it needs, so it waits until the call count is genuinely unusual
        (:data:`NUDGE_AFTER_CALLS`) rather than merely plural.

        It used to fire on a tool's *second* call. On any task that needs more
        than two steps that is an instruction to stop half way, and the smallest
        models take it: they answer with whichever intermediate value is most
        recent.

        Args:
            iteration: The turn number that just ran, counted from one.
            action_call_count: How many times this tool had already been called
                before this turn.
            tool_result: What the dispatch returned, so a failed one earns no
                "you already have the answer" nudge.

        Returns:
            The line to append to the scratchpad, or ``None``.
        """
        if iteration >= self.nudge_cap - 2:
            return NUDGE_HAVE_ANSWER
        if action_call_count >= NUDGE_AFTER_CALLS and not tool_result.startswith(
            TOOL_ERROR_PREFIX
        ):
            return NUDGE_HAVE_RESULTS
        return None

    # ------------------------------------------------------------------
    # Calls written out instead of made
    # ------------------------------------------------------------------
    def is_unmade_call(self, written: str, text: str) -> bool:
        """True when a written-out call block means the work never happened.

        Either the named tool never ran in this run, or the text is nothing but
        the call — a recap beside a real answer is neither.
        """
        return written not in self.executed_tools or written_call_only(text, self.tools)

    def note_written_call(self, written: str) -> bool:
        """Record a turn that wrote its call out; True once it is time to report.

        One such turn earns a nudge. A second means the model is not going to
        make the call, so the caller reports the cause instead of billing the
        rest of the iteration budget for the same outcome.
        """
        self.written_call = self.written_call or written
        self.written_call_turns += 1
        return self.written_call_turns > 1

    def tool_ran(self, name: str | None) -> bool:
        """True when *name* was dispatched at some point in this run."""
        return bool(name) and name in self.executed_tools
