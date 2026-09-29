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

**Progress, not a count.** A turn made progress when a tool it called returned
a result the conversation does not already show in full; a failed call counts
the first time its error is seen. A turn stalled when every result it got was
already there, when a call it made was answered from the record or declined, or
when it only reasoned right after a turn that only reasoned. The drift
threshold counts only calls that brought nothing new, so a run whose every call
finds something new is not stopped by it, and
:attr:`NativeToolLoop.max_turns_without_progress` stalled turns in a row ask the
run for its answer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..prompts.tool_contract import ToolUsePolicy, is_execution_tool
from ..tools.base_tool import ToolCategory
from .agent_runtime import (
    NUDGE_HAVE_ANSWER,
    NUDGE_HAVE_RESULTS,
    written_call_only,
)
from .retrieval_requery import MAX_RETRIEVAL_REQUERIES
from .tool_call_record import ToolCall, truncate_result
from .tool_failure import INPUT_SIDE, ToolFailure

logger = logging.getLogger(__name__)

#: Prefix :meth:`Agent._execute_tool` puts on a failed dispatch. A failed call is
#: not evidence of a repeated result, so the result-based short circuit skips it.
TOOL_ERROR_PREFIX = "Error executing tool"

#: How many calls to one tool read as circling when the inputs keep changing
#: and the results do not. Small models re-format the same call rather than
#: repeating it byte for byte, so the exact-pair check alone never fires for
#: them. Only calls whose result the conversation already showed are counted: a
#: tool that keeps returning something new is doing work, however often it is
#: called.
#:
#: This counts *drift*, not work. It was 5, which is below the length of an
#: ordinary multi-step task: a word problem with four arithmetic steps and one
#: check spends five calls doing exactly what it was asked to do, and the guard
#: read that as a loop and ended the run holding an intermediate value. Twelve
#: calls to one tool that each brought nothing new is past anything a model
#: does while it is still working, so only a model that really is circling
#: reaches it.
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

#: Tool-side failures in a row after which a tool is not called again in the
#: run. The same number the agent's circuit breaker opens at, so the two agree:
#: a transient error gets its retries, a service that is down does not get ten.
TOOL_FAILURES_BEFORE_UNAVAILABLE = 3

#: Calls in a row to one tool that failed on their input after which the tool
#: is not offered again in the run: a call and three corrections. One more than
#: :data:`TOOL_FAILURES_BEFORE_UNAVAILABLE`, because a corrected input can
#: succeed where a retry against a service that is down cannot: in recorded
#: runs of small and mid-size models, a tool that had failed three times in a
#: row on its input still returned a result afterwards in 5 runs of 7,562, and
#: one that had failed four times in 1.
INPUT_ERRORS_BEFORE_WITHDRAWN = 4

#: The phrase every tool-error path of the loop logs, so its firings can be
#: counted.
TOOL_ERROR_LOG = "[tool error]"

#: What the model reads when its answer is a tool's failure.
TOOL_ERROR_NOT_AN_ANSWER = (
    "[That reports the {tool} failure; it is not an answer. Correct the call "
    "and try again, or write the answer to the question in your own words.]"
)

#: How an answer that restates a failure in the model's own words opens: the
#: ``Error:`` form tools report failures in. An answer that merely begins with
#: the word ("Error rate is 3 %", "Errors found: …") is not one.
_RESTATED_FAILURE = re.compile(r"error\s*:", re.IGNORECASE)

#: What a result that carries nothing reads as once it reached the loop. A tool
#: that returned ``None`` is reported as the first of these by the dispatch
#: layer; the others are an empty string, list or mapping written out.
_EMPTY_RESULTS = frozenset({"", "no result returned", "none", "[]", "{}", "()"})


def is_error_result(text: Any) -> bool:
    """Whether a tool observation reports a failure rather than a result.

    The dispatch layer's own prefix, a tool's failure envelope, and a tool that
    reports its failure as text beginning with ``Error`` all read as one.
    """
    value = str(text if text is not None else "").strip().lower()
    return value.startswith(("error", "tool execution failed"))


def is_usable_result(text: Any) -> bool:
    """Whether a tool observation carries something the run can use.

    Not a failure, and not structurally empty. Nothing here reads what a
    result says — "No results found." is a result.
    """
    if is_error_result(text):
        return False
    return str(text if text is not None else "").strip().lower() not in _EMPTY_RESULTS


@dataclass
class LoopCheck:
    """What :meth:`NativeToolLoop.check_action` found about one proposed call."""

    #: ``(action, normalized_input)``, the key the exact-repeat check uses.
    pair: tuple[str, str]
    #: How many times this tool was already called in this run.
    action_call_count: int
    #: The same tool with the same input has already been dispatched.
    is_exact_loop: bool
    #: The same tool has been called enough times, without a new result, to
    #: read as a loop.
    is_fuzzy_loop: bool
    #: How many of this tool's calls returned a result already shown.
    stalled_call_count: int = 0

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
            return f"fuzzy ({self.stalled_call_count + 1} calls without a new result)"
        return ""


#: :meth:`NativeToolLoop.note_lost_call`: send the turn back and require a call.
LOST_CALL_ASK = "ask"
#: Send the turn back without requiring a call.
LOST_CALL_WARN = "warn"
#: Report the run as a call that was written out instead of made.
LOST_CALL_REPORT = "report"
#: Carry on as with any turn that did nothing.
LOST_CALL_CONTINUE = "continue"


@dataclass
class _TurnMarks:
    """What the turn in progress did, as the progress policy reads it."""

    #: Calls whose result the conversation did not already show.
    new: int = 0
    #: Calls whose result it did.
    seen: int = 0
    #: Calls answered from the record or declined.
    declined: int = 0
    #: Calls that failed on the tool's own side.
    tool_failed: int = 0
    #: The turn is neither progress nor a stall: its answer was sent back by a
    #: guard, or it opened a call that could not be run.
    neutral: bool = False
    #: The turn was required to call by a guard.
    forced: bool = False
    #: The turn produced neither an action nor an answer.
    reasoned_only: bool = False
    #: The turn said it takes no action and wrote nothing usable after it.
    declared: bool = False


#: :meth:`NativeToolLoop.end_turn` found a declared no-action after progress.
ASK_AFTER_DECLARED = "declared"
#: :meth:`NativeToolLoop.end_turn` found too many stalled turns in a row.
ASK_AFTER_STALL = "stalled"
#: A repeated call was answered by asking for the answer.
ASK_AFTER_LOOP = "loop_detected"
#: A tool that reproduced its own result was answered by asking for the answer.
ASK_AFTER_REPEATED_RESULT = "repeated_tool_result"
#: Every tool the run holds failed on its input as often as the run allows.
ASK_AFTER_INPUT_ERRORS = "input_errors"


@dataclass
class NativeToolLoop:
    """Per-run state and the decisions a tool-calling loop makes from it.

    Args:
        tools: The tools the agent holds, by name — the same mapping the loop
            dispatches against.
        nudge_cap: The iteration cap the run is using — the call's own
            ``max_iterations`` when it passed one — used to decide when a turn
            is close enough to the limit to ask for an answer outright, and to
            bound the drift threshold.
        tool_use: The policy this run is on, or ``None`` to read it from each
            tool's declared category — which is what an agent whose caller
            stated no policy does. It decides one thing here:
            :meth:`execution_tools`, and through it whether an answer written
            with no call is sent back.
        max_turns_without_progress: How many stalled turns in a row, after the
            run's first new result, ask the run for its answer; ``None`` never
            asks. See :meth:`end_turn`.
        required_categories: Tool categories (``ToolCategory`` values) whose
            tools this run may not answer without calling, beyond the ones the
            tools declare themselves — what a capability probe found the model
            skips. Ignored when *tool_use* is set: a caller's stated policy
            decides for the whole run.
    """

    tools: dict[str, Any]
    nudge_cap: int = 10
    tool_use: ToolUsePolicy | None = None
    max_turns_without_progress: int | None = None
    required_categories: frozenset[str] = frozenset()

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
    #: Turns that opened a call nothing could run — a tag with nothing after
    #: it, arguments that did not read, or a call written out as text.
    lost_calls: int = 0
    #: Calls not dispatched because they left a required parameter without a
    #: value.
    missing_argument_calls: int = 0
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
    #: How many times this run has been sent back to search again after its own
    #: answer reported that what came back does not answer the question. Capped
    #: at :data:`~effgen.core.retrieval_requery.MAX_RETRIEVAL_REQUERIES`; see
    #: :meth:`take_retrieval_requery`.
    retrieval_requeries: int = 0
    #: The results the conversation still shows in full, keyed as
    #: :meth:`_result_key` keys them, with how many observations carry each.
    shown_results: dict[tuple[str, str], int] = field(default_factory=dict)
    #: Calls to each tool whose result was already shown — what the drift
    #: threshold counts.
    stalled_calls: dict[str, int] = field(default_factory=dict)
    #: Whether any turn of this run has brought a new result.
    progress_seen: bool = False
    #: Stalled turns in a row since the last new result.
    turns_without_progress: int = 0
    #: Why the run was last asked for its answer — one of the ``ASK_AFTER_*``
    #: values — or ``None`` when it has not been asked.
    ask_reason: str | None = None
    #: Dispatched calls whose result was new, not a failure and not empty.
    usable_results: int = 0
    #: Calls the loop declined to dispatch: unknown or unavailable tools,
    #: repeats answered from the record, calls on a turn that forbade one.
    declined_calls: int = 0
    #: Dispatched calls that failed on the tool's own side.
    tool_side_failures: int = 0
    #: Dispatched calls whose result reported a failure of the call's input:
    #: the tool ran and could be used, the call itself was wrong.
    input_errors: int = 0
    #: Tool-side failures in a row, per tool.
    consecutive_tool_failures: dict[str, int] = field(default_factory=dict)
    #: The tools this run stopped calling, with the failure that ended them.
    unavailable_tools: dict[str, ToolFailure] = field(default_factory=dict)
    #: Dispatches per tool, for the count a failure report states.
    dispatches_by_tool: dict[str, int] = field(default_factory=dict)
    #: The most recent tool-side failure of the run.
    last_tool_failure: ToolFailure | None = None
    #: Calls in a row, per tool, whose result reported a failure of the input.
    consecutive_input_errors: dict[str, int] = field(default_factory=dict)
    #: The input-side failure the dispatch layer reported for the call just
    #: made, when it reported one; read by :meth:`observe_result`.
    _input_failure: ToolFailure | None = field(default=None, repr=False)
    #: Answers sent back because they were a tool's error message.
    error_answers: int = 0
    #: Calls whose most recent attempt failed on the tool's own side.
    failed_pairs: set[tuple[str, str]] = field(default_factory=set)
    #: The turn that just ended wrote a placeholder action.
    last_turn_placeholder: bool = False
    _turn: _TurnMarks = field(default_factory=_TurnMarks, repr=False)
    _previous_turn_only_reasoned: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Say once which policy this run is on, so a log can be counted."""
        logger.info(
            "tool use policy: %s (%s)",
            (self.tool_use.value if self.tool_use is not None else "from tools"),
            ",".join(sorted(self.execution_tools())) or "none required",
        )

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
        """The names of the held tools this run may not answer without calling.

        With no policy stated, each tool answers for itself through
        :func:`~effgen.prompts.tool_contract.is_execution_tool`, so the set is
        whatever the tools declare: a code executor is in it, a calculator is
        not. A caller who stated a policy overrides that for the whole run —
        :attr:`~effgen.prompts.tool_contract.ToolUsePolicy.REQUIRED` puts every
        held tool in the set, which is how "always use this tool" is expressed
        for a tool whose category does not ask for it, and the other two empty
        it, which is how a caller stops the framework pushing for one it does.
        """
        if self.tool_use is ToolUsePolicy.REQUIRED:
            return list(self.tools)
        if self.tool_use is not None:
            return []
        return [
            name for name, tool in self.tools.items()
            if is_execution_tool(tool) or self._probe_requires(tool)
        ]

    def _probe_requires(self, tool: Any) -> bool:
        """Whether *tool*'s category is one the model was measured to skip."""
        if not self.required_categories:
            return False
        category = getattr(getattr(tool, "metadata", None), "category", None)
        if category is None:
            category = getattr(tool, "category", None)
        value = getattr(category, "value", category)
        return isinstance(value, str) and value in self.required_categories

    def note_execution_refusal(self) -> str | None:
        """Answer without running the executor: refuse it once, name the tool.

        An agent holding a code executor and answering with no call has reported
        a result nothing produced — it described what the code would print. That
        answer is not accepted the first time: the turn goes back with a nudge
        naming the tool, and the turn after it is sent requiring a call. A
        caller who set :attr:`~effgen.prompts.tool_contract.ToolUsePolicy.REQUIRED`
        gets the same treatment for whatever tools they attached, which is what
        "this tool must actually be used" means for a tool that does not declare
        it.

        **Only the first.** A model that declines twice will decline again, and
        the iteration budget buys more elsewhere; the second refusal is
        accepted, so a run cannot be spent circling on this.

        Returns:
            The tool name to name in the nudge, or ``None`` when this run
            requires no call, has already dispatched one, or has already been
            sent back once.
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
        if self.tool_use is None and not any(
            is_execution_tool(self.tools[name]) for name in names
        ):
            # Only a capability probe put these tools in the must-call set.
            logger.info(
                "capability probe policy: refusal fired for '%s' (%s)",
                names[0], ",".join(sorted(self.required_categories)),
            )
        return names[0]

    def take_retrieval_requery(self) -> bool:
        """Whether this run may be sent back to search again. Spent on read.

        Reading it consumes the allowance, so a run gets one further search and
        the answer it writes afterwards is accepted whatever it says. That bound
        is the point: a model that already searches several times unprompted
        does not need a policy of searching until something turns up, and a
        model that stopped after one bad query needs exactly one more.

        Returns:
            True the first time it is called on a run, False afterwards.
        """
        if self.retrieval_requeries >= MAX_RETRIEVAL_REQUERIES:
            return False
        self.retrieval_requeries += 1
        return True

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
        self._turn.forced = self._turn.forced or forced
        return forced

    # ------------------------------------------------------------------
    # Repeat detection
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_input(action_input: Any) -> str:
        """Return *action_input* in a form two equivalent calls share.

        JSON arguments are re-serialized with sorted keys so the same call
        written with its keys in a different order compares equal; anything that
        is not JSON is compared as trimmed text. A value that is not a string
        (a mapping, a number) is compared as its JSON text; this never raises.
        """
        if action_input is None:
            return ""
        if not isinstance(action_input, str):
            try:
                return json.dumps(action_input, sort_keys=True, default=str)
            except (TypeError, ValueError):
                return str(action_input).strip()
        normalized = action_input.strip()
        try:
            return json.dumps(json.loads(normalized), sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return normalized

    def fuzzy_threshold(self, action: str) -> int:
        """How many stalled calls to *action* read as circling.

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
        stalled = self.stalled_calls.get(action, 0)
        # A call whose last attempt failed on the tool's own side is a retry,
        # not a repeat: the unavailability count, not the loop guard, bounds it.
        retry = pair in self.failed_pairs
        return LoopCheck(
            pair=pair,
            action_call_count=action_call_count,
            is_exact_loop=exact_count >= 1 and known and not retry,
            is_fuzzy_loop=stalled >= self.fuzzy_threshold(action) and known,
            stalled_call_count=stalled,
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
        model is re-deriving something it already has. A failure is never a
        repeat, whoever reported it: two calls with different inputs that the
        tool rejects in the same words are two attempts, not a settled answer.
        """
        if is_error_result(tool_result):
            return False
        return self._result_key(action, tool_result) in self.previous_results

    def record_result(self, action: str, tool_result: str) -> None:
        """Remember what *action* returned, unless it reported a failure."""
        if is_error_result(tool_result):
            return
        self.previous_results.append(self._result_key(action, tool_result))

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def observe_result(
        self, action: str, tool_result: Any, *, failed_tool_side: bool = False,
    ) -> bool:
        """Judge one dispatched call's result and remember it.

        A result is new when no observation the conversation still shows in
        full carries the same text from the same tool. A failed dispatch is
        judged the same way, so a new error is progress and the same error
        again is not — except a failure on the tool's own side, which is
        neither progress nor a repeat.

        Args:
            action: The tool that ran.
            tool_result: What it returned.
            failed_tool_side: The call failed on the tool's own side.

        Returns:
            Whether the result was new.
        """
        key = self._result_key(action, str(tool_result))
        new = key not in self.shown_results
        self.shown_results[key] = self.shown_results.get(key, 0) + 1
        if not new:
            self.stalled_calls[action] = self.stalled_calls.get(action, 0) + 1
        if not failed_tool_side and is_error_result(tool_result):
            self.input_errors += 1
        input_failure, self._input_failure = self._input_failure, None
        if not failed_tool_side:
            self._count_input_error(action, tool_result, input_failure)
        if failed_tool_side:
            # The tool could not run. That is neither a result nor the model
            # going round: it is what the unavailability count is for.
            self._turn.tool_failed += 1
        elif new:
            self._turn.new += 1
            if is_usable_result(tool_result):
                self.usable_results += 1
        else:
            self._turn.seen += 1
        return new

    def _count_input_error(
        self, action: str, tool_result: Any, failure: ToolFailure | None,
    ) -> None:
        """Count a call that failed on its input; withdraw the tool at the limit.

        A result that is not a failure resets the count. At
        :data:`INPUT_ERRORS_BEFORE_WITHDRAWN` failures in a row the tool is not
        called again in this run; when it was the last tool the run could
        call, the loop asks the run for its answer.
        """
        if not is_error_result(tool_result):
            self.consecutive_input_errors[action] = 0
            return
        count = self.consecutive_input_errors.get(action, 0) + 1
        self.consecutive_input_errors[action] = count
        logger.info(
            "%s '%s' failed on its input (%d of %d in a row)",
            TOOL_ERROR_LOG, action, count, INPUT_ERRORS_BEFORE_WITHDRAWN,
        )
        if count < INPUT_ERRORS_BEFORE_WITHDRAWN or action in self.unavailable_tools:
            return
        # A failure the dispatch layer caught carries its class and message;
        # one the tool reported as text is that text.
        message = " ".join(
            str(failure.message if failure is not None else tool_result).split()
        )[:300]
        self.unavailable_tools[action] = ToolFailure(
            tool=action,
            error_type=failure.error_type if failure is not None else "",
            message=message,
            side=INPUT_SIDE,
        )
        logger.info(
            "%s '%s' failed on its input %d times in a row; it is not offered "
            "again in this run",
            TOOL_ERROR_LOG, action, count,
        )

    def withdrawn_on_input(self) -> bool:
        """Whether a tool this run stopped calling failed on its input."""
        return any(not f.tool_side for f in self.unavailable_tools.values())

    def error_answered(self, text: Any) -> str | None:
        """The tool whose failure *text* reports, or ``None``.

        An answer is a tool's error when it is a copy of the message a call of
        the run failed with, or, when the run's last call failed, the model's
        own restatement of that failure in the ``Error:`` form ("Error: the
        code timed out, please try again"). The tool named is the one whose
        error text the answer opens with, else the one that failed last. A
        sentence that mentions an error is not one; neither is an answer that
        opens with the word "Error" but not a failure's form ("Error rate is
        21 percent"), nor one written after the failed call was corrected, nor
        one in a run where no call failed.

        Args:
            text: The answer, as the model wrote it.

        Returns:
            The name of the tool whose failure the answer reports, or ``None``.
        """
        answer = " ".join(str(text or "").split())
        if len(answer) < 5 or not is_error_result(answer):
            return None
        head = answer[:60].lower()
        failed = [
            (call.name, " ".join(str(call.result or "").split()).lower())
            for call in self.calls if is_error_result(call.result)
        ]
        for name, result in reversed(failed):
            if result.startswith(head) or head.startswith(result[:60]):
                return name
        last = self.calls[-1] if self.calls else None
        if (
            last is not None and is_error_result(last.result)
            and _RESTATED_FAILURE.match(answer)
        ):
            return last.name
        return None

    def refresh_shown_results(self, steps: Any) -> None:
        """Re-read which results the conversation still shows in full.

        Called after the conversation was shortened. A result whose observation
        was cut or dropped is no longer in front of the model, so fetching it
        again is new work rather than a repeat.

        Args:
            steps: The conversation's steps, in order.
        """
        shown: dict[tuple[str, str], int] = {}
        by_call: dict[str, str] = {}
        last_tool: str | None = None
        for step in steps:
            kind = getattr(step, "kind", "")
            if kind == "action":
                last_tool = step.tool
                if step.call_id:
                    by_call[step.call_id] = step.tool
            elif kind == "observation":
                tool = by_call.get(step.call_id or "") or last_tool
                if (tool and step.compacted is None and step.declined is None):
                    key = self._result_key(tool, str(step.text))
                    shown[key] = shown.get(key, 0) + 1
        self.shown_results = shown

    def begin_turn(self) -> None:
        """Start judging a new turn."""
        self._turn = _TurnMarks()
        self.last_turn_placeholder = False

    def note_declined(self) -> None:
        """A call this turn made was answered from the record or declined."""
        self._turn.declined += 1
        self.declined_calls += 1

    def note_neutral(self) -> None:
        """This turn is neither progress nor a stall.

        A turn whose answer a guard sent back, and a turn that opened a call
        nothing could run, say nothing about whether the run is converging.
        """
        self._turn.neutral = True

    def note_reasoning_only(self, *, declared: bool = False) -> None:
        """This turn produced neither an action nor an answer.

        Args:
            declared: The turn said, in words, that it takes no action. That
                is not a turn of planning: it counts as a stall at once.
        """
        if declared:
            self._turn.declared = True
        else:
            self._turn.reasoned_only = True

    def end_turn(self) -> str | None:
        """Judge the turn that just ended, and say whether to ask for the answer.

        Counted only after the run's first new result: before it, the refusal
        guard owns a run that has not called what it must call, and taking its
        tools away would undo it. One turn of reasoning before acting is
        planning; two in a row is a stall. A turn a guard forced, a turn whose
        answer a guard sent back and a turn that lost its call are neither.

        When this returns a value, :attr:`force_text_answer` is already set:
        the next turn offers no tools and asks for the answer.

        Returns:
            :data:`ASK_AFTER_DECLARED` when the turn declared no action after
            the run made progress, :data:`ASK_AFTER_STALL` when
            :attr:`max_turns_without_progress` stalled turns ran in a row, and
            ``None`` otherwise — always ``None`` when the limit is ``None``.
        """
        turn = self._turn
        only_reasoned_before = self._previous_turn_only_reasoned
        self._previous_turn_only_reasoned = False
        if turn.new:
            self.progress_seen = True
            self.turns_without_progress = 0
            return None
        if turn.neutral or turn.forced:
            return None
        limit = self.max_turns_without_progress
        if not self.progress_seen:
            self._previous_turn_only_reasoned = turn.reasoned_only
            # Before the first result the refusal guard owns a run that has not
            # called what it must call, so only a turn that did call and had
            # every call declined counts. A call that failed on the tool's side
            # is left to the unavailability count, which gives the tool its
            # retries before the run gives up on it.
            if limit is None or turn.seen or not turn.declined:
                return None
            logger.info(
                "[progress] a turn with only declined or failed calls before "
                "any result"
            )
        elif turn.reasoned_only:
            self._previous_turn_only_reasoned = True
            if not only_reasoned_before:
                return None
        elif not (turn.seen or turn.declined or turn.declared):
            return None
        self.turns_without_progress += 1
        if limit is None or self.force_text_answer:
            return None
        if turn.declared:
            decision = ASK_AFTER_DECLARED
        elif self.turns_without_progress >= limit:
            decision = ASK_AFTER_STALL
        else:
            return None
        self.ask_for_answer(decision)
        return decision

    def ask_for_answer(self, reason: str) -> None:
        """Withdraw the tools so the next turn is asked for the answer.

        Args:
            reason: Why, one of the ``ASK_AFTER_*`` values; kept as
                :attr:`ask_reason` so a run that still writes no answer can
                report what it was stuck on.
        """
        self.force_text_answer = True
        self.ask_reason = reason

    # ------------------------------------------------------------------
    # Tools that fail on their own side
    # ------------------------------------------------------------------
    def note_dispatch(
        self, action: str, failures: list[ToolFailure],
        pair: tuple[str, str] | None = None,
    ) -> ToolFailure | None:
        """Record how one dispatched call went, as the dispatch layer saw it.

        A call that failed on the tool's side adds to that tool's count of
        failures in a row; anything else resets it. At
        :data:`TOOL_FAILURES_BEFORE_UNAVAILABLE` in a row, or when the agent's
        circuit breaker refused the call, the tool is not called again in this
        run.

        Args:
            action: The tool that was dispatched.
            failures: The failures reported while it ran.
            pair: The call's ``(action, normalized_input)``, so the same call
                can be retried after a tool-side failure without reading as a
                repeat.

        Returns:
            The tool-side failure, when the call ended in one.
        """
        self.dispatches_by_tool[action] = self.dispatches_by_tool.get(action, 0) + 1
        self._input_failure = next(
            (f for f in reversed(failures) if not f.tool_side and f.tool == action), None,
        )
        failure = next(
            (f for f in reversed(failures) if f.tool_side and f.tool == action), None,
        )
        if failure is None:
            self.consecutive_tool_failures[action] = 0
            if pair is not None:
                self.failed_pairs.discard(pair)
            return None
        if pair is not None:
            self.failed_pairs.add(pair)
        self.tool_side_failures += 1
        self.last_tool_failure = failure
        count = self.consecutive_tool_failures.get(action, 0) + 1
        self.consecutive_tool_failures[action] = count
        if action not in self.unavailable_tools and (
            count >= TOOL_FAILURES_BEFORE_UNAVAILABLE
            or failure.error_type == "CircuitOpen"
        ):
            self.unavailable_tools[action] = failure
            logger.info(
                "[tool] '%s' is unavailable for this run (%s)",
                action, failure.error_type,
            )
        return failure

    def is_unavailable(self, action: str) -> bool:
        """Whether *action* is a tool this run stopped calling."""
        return action in self.unavailable_tools

    def all_tools_unavailable(self) -> bool:
        """Whether every tool the agent holds has been given up on."""
        return bool(self.tools) and all(name in self.unavailable_tools for name in self.tools)

    def every_call_failed_tool_side(self) -> bool:
        """Whether the run dispatched calls and every one failed tool-side."""
        return bool(self.calls) and self.tool_side_failures >= len(self.calls)

    def attempted_calls(self) -> int:
        """Calls the run made: the dispatched ones and the declined ones."""
        return len(self.calls) + self.declined_calls

    def result_counts(self) -> dict[str, int]:
        """What the run's calls brought, as the response reports it."""
        return {
            "attempted": self.attempted_calls(),
            "dispatched": len(self.calls),
            "usable": self.usable_results,
            "tool_side_failures": self.tool_side_failures,
            "input_errors": self.input_errors,
        }

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
            tool_result: What the dispatch returned. A failure earns no nudge
                here: it does not hand the model an answer.

        Returns:
            The line to append to the scratchpad, or ``None``.
        """
        if is_error_result(tool_result):
            # A failure is not an answer from the tool.
            return None
        if iteration >= self.nudge_cap - 2:
            return NUDGE_HAVE_ANSWER
        if action_call_count >= NUDGE_AFTER_CALLS:
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
        return self.note_lost_call(written) == LOST_CALL_REPORT

    def note_lost_call(self, tool: str | None, *, can_ask: bool = True) -> str:
        """Record a turn that opened a call nothing could run; say what to do.

        The first such turn of a run is sent back once, with a call required on
        the next turn where the request can require one. After that, a turn
        that writes out a call for a held tool is warned the first time it
        happens and reported the second time — a model that writes the same
        call out twice is not going to make it. A turn that names no tool — a
        tag with nothing after it — cannot be reported as a call to anything,
        so the run goes on.

        Args:
            tool: The held tool the turn named, or ``None`` when it named none.
            can_ask: Whether the next turn may be asked for a call at all. A
                run whose tools are withdrawn is asking for an answer, and is
                never also asked for a call.

        Returns:
            :data:`LOST_CALL_ASK` (the forced flag is already set),
            :data:`LOST_CALL_WARN`, :data:`LOST_CALL_REPORT` or
            :data:`LOST_CALL_CONTINUE`.
        """
        self.lost_calls += 1
        self._turn.neutral = True
        if tool:
            self.written_call = self.written_call or tool
            self.written_call_turns += 1
        if self.lost_calls == 1:
            if can_ask:
                self.force_tool_call = True
                return LOST_CALL_ASK
            return LOST_CALL_WARN
        if not tool:
            return LOST_CALL_CONTINUE
        return LOST_CALL_REPORT if self.written_call_turns > 1 else LOST_CALL_WARN

    def tool_ran(self, name: str | None) -> bool:
        """True when *name* was dispatched at some point in this run."""
        return bool(name) and name in self.executed_tools
