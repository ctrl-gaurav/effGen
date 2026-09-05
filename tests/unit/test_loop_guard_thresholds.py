"""The loop guards end runs that are circling, not runs that are working.

Four mechanisms decide when a tool-calling run has stopped making progress: the
exact-repeat check, the drift threshold, the multi-call-turn cap, and the
reminder appended after a tool runs. Each of them used to fire inside the length
of an ordinary multi-step task, so a run that was going to finish was ended
holding an intermediate value.

What is pinned here:

- an exact repeat of a call that already succeeded is answered from the record,
  the run continues, and the tool is not dispatched a second time;
- the replays are capped, so a run that only ever repeats one call still breaks
  out and reports a typed partial well inside its iteration budget;
- the drift threshold sits above the length of real work and never above the
  run's own budget, which would leave it as dead code;
- a task needing several multi-call turns keeps its tools;
- the "you already have results" reminder waits for a call count that is
  genuinely unusual rather than merely plural.

Seven of these pass against the tree before the change as well, and are guards
rather than proof of it: the over-correction case (a run that only repeats one
call still ends), the two rules about a failed dispatch, the exact-repeat report
itself, the wider count a data-processing tool gets, and the threshold bounds at
the budgets where the old derivation and the new one agree.

The model is an in-process script, which is what makes the loop deterministic;
this is loop policy, not provider behavior.
"""

from __future__ import annotations

import json

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import NUDGE_HAVE_RESULTS
from effgen.core.agent_tool_loop import (
    FUZZY_LOOP_FLOOR,
    FUZZY_LOOP_THRESHOLD,
    FUZZY_LOOP_THRESHOLD_DATA,
    MAX_BATCH_TOOL_RUNS,
    NUDGE_AFTER_CALLS,
    LoopCheck,
    NativeToolLoop,
)
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from effgen.tools.builtin.calculator import Calculator


class _Scripted(BaseModel):
    """Replays a fixed list of turns and records every prompt it was sent."""

    def __init__(self, turns: list, *, native: bool = False) -> None:
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self._turns = turns
        self.calls = 0
        self.prompts: list[str] = []
        self.tools_offered: list[bool] = []
        self._native = native

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def unload(self) -> None:  # pragma: no cover - trivial
        pass

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 8192

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config, **kwargs) for p in prompts]

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
        return self.generate(prompt, config=config, tools=tools, **kwargs)

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self.generate(prompt, config=config, **kwargs).text

    def supports_function_calling(self) -> bool:
        return self._native

    def supports_tool_calling(self) -> bool:
        return self._native

    def generate(self, prompt, config=None, **kwargs):
        self.prompts.append(str(prompt))
        self.tools_offered.append(bool(kwargs.get("tools")))
        turn = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        if isinstance(turn, dict):
            return GenerationResult(
                text="", tokens_used=5, finish_reason="tool_calls",
                model_name=self.model_name, metadata=turn,
            )
        return GenerationResult(
            text=turn, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )


def _calc(expr: str) -> str:
    return (
        "Thought: next step.\nAction: calculator\n"
        f'Action Input: {{"operation": "calculate", "expression": "{expr}"}}'
    )


def _batch(*exprs: str) -> dict:
    return {"tool_calls": [
        {"id": "", "type": "function", "function": {
            "name": "calculator",
            "arguments": json.dumps({"operation": "calculate", "expression": e}),
        }}
        for e in exprs
    ]}


def _agent(turns, *, tools=None, max_iterations=10, native=False, mode="react"):
    model = _Scripted(turns, native=native)
    cfg = {
        "name": "loop-guard-test",
        "model": model,
        "tools": [Calculator()] if tools is None else tools,
        "max_iterations": max_iterations,
        "raise_on_error": False,
        "enable_memory": False,
    }
    if mode:
        cfg["tool_calling_mode"] = mode
    return Agent(config=AgentConfig(**cfg)), model


class _DeadEndSearch(BaseTool):
    """A retrieval tool that always returns the same unhelpful thing."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="search", description="Search the archive.",
            category=ToolCategory.INFORMATION_RETRIEVAL,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="query", required=True,
            )],
        ))

    async def _execute(self, query: str = "", **kwargs):
        return "No results found."


def _search(query: str) -> str:
    return (
        "Thought: look it up.\nAction: search\n"
        f'Action Input: {{"query": "{query}"}}'
    )


# --------------------------------------------------------------------------- #
# A run that is working is allowed to finish
# --------------------------------------------------------------------------- #
def test_a_four_step_task_plus_a_check_reaches_its_answer():
    """Five distinct calls are the work, not a loop.

    Four arithmetic steps and one check is the shape of an ordinary word
    problem. No guard may end it, and the answer the model finally writes is
    the answer the caller receives.
    """
    agent, model = _agent([
        _calc("120 * 3"), _calc("360 + 45"), _calc("405 / 5"), _calc("81 - 6"),
        _calc("75 * 1"), "Thought: I have it.\nFinal Answer: 75",
    ])
    resp = agent.run(
        "Explain step by step: 120 crates of 3, plus 45 more, split five ways, "
        "then six removed. How many?"
    )

    assert resp.outcome == "answered"
    assert resp.stop_reason == "final_answer"
    assert resp.output.strip() == "75"
    assert resp.tool_calls == 5
    assert model.calls == 6, "the run never reached the turn that states the answer"


def test_a_restated_call_is_answered_from_the_record_and_the_task_finishes():
    """A model that repeats its first call has not looped; it restated a plan."""
    agent, model = _agent([
        _calc("18 * 4"),    # 72
        _calc("18 * 4"),    # the exact restatement
        _calc("72 + 9"),    # 81, the step that finishes the task
        "Thought: I have it.\nFinal Answer: 81",
    ])
    resp = agent.run("Explain step by step: eighteen boxes of four, plus nine spares.")

    assert resp.outcome == "answered"
    assert resp.output.strip() == "81"
    # Two dispatches, not three: the repeat came from the record.
    assert resp.tool_calls == 2
    assert [c.name for c in resp.tool_calls] == ["calculator", "calculator"]


def test_the_replayed_result_reaches_the_model_as_an_observation():
    """The record is handed back as an observation, so the next turn can use it."""
    agent, model = _agent([
        _calc("18 * 4"), _calc("18 * 4"), "Thought: done.\nFinal Answer: 72",
    ])
    resp = agent.run("Explain step by step: eighteen boxes of four.")

    # The run reached the turn that states the answer, which is what the repeat
    # used to cost, and that turn was shown the recorded value twice: once from
    # the dispatch and once from the replay.
    assert resp.outcome == "answered"
    assert model.calls == 3
    assert model.prompts[-1].count("Observation: 72") == 2


def test_several_multi_call_turns_keep_their_tools():
    """A model that batches its calls is not punished for it."""
    agent, model = _agent([
        _batch("2 + 2", "3 + 3"), _batch("4 + 4", "5 + 5"),
        _batch("6 + 6", "7 + 7"), "Final Answer: 4, 6, 8, 10, 12, 14",
    ], native=True, mode=None)
    resp = agent.run(
        "Explain step by step and compute 2+2, 3+3, 4+4, 5+5, 6+6 and 7+7."
    )

    assert resp.outcome == "answered"
    assert resp.tool_calls == 6
    assert all(model.tools_offered), (
        "the tools were withdrawn while the task was still running: "
        f"{model.tools_offered}"
    )


def test_a_two_step_task_is_not_told_it_already_has_the_answer():
    """The reminder used to fire on a tool's second call."""
    agent, model = _agent([
        _calc("50 * 3"), _calc("150 - 20"), "Thought: I have it.\nFinal Answer: 130",
    ])
    resp = agent.run("Explain step by step: fifty crates of three, less twenty.")

    assert resp.outcome == "answered"
    assert resp.output.strip() == "130"
    assert not any(NUDGE_HAVE_RESULTS in p for p in model.prompts)


# --------------------------------------------------------------------------- #
# A run that is circling still ends
# --------------------------------------------------------------------------- #
def test_a_run_that_only_repeats_one_call_still_breaks_out():
    """The replays are capped, so the exact-repeat path cannot spin.

    This is the over-correction guard: a tool that always returns the same
    unhelpful thing must still end the run, inside the budget, with a typed
    partial rather than with a fabricated answer.
    """
    agent, model = _agent(
        [_search("ostrich egg mass")], tools=[_DeadEndSearch()], max_iterations=10,
    )
    resp = agent.run("Explain what an ostrich egg weighs.")

    assert resp.outcome == "stopped"
    assert resp.stop_reason == "loop_detected"
    assert resp.iterations < 10, "it ground on to the iteration cap"
    assert resp.metadata["error"]["type"] == "UnsynthesizedToolResult"
    assert resp.partial is not None
    assert "No results found." in resp.partial.text
    assert "No results found." not in (resp.output or "")
    # The tool ran once; every repeat after that came from the record.
    assert resp.tool_calls == 1


def test_the_replay_allowance_is_spent_once_per_call_not_once_per_run():
    """Two different calls each get their own replays; neither borrows the other's."""
    loop = NativeToolLoop(tools={"calculator": Calculator()}, nudge_cap=10)
    first = loop.check_action("calculator", '{"expression": "1+1"}')
    second = loop.check_action("calculator", '{"expression": "2+2"}')
    loop.record_action(first)
    loop.record_pair_result(first, "2")
    loop.record_action(second)
    loop.record_pair_result(second, "4")

    replayed = [loop.cached_result(first) for _ in range(4)]
    assert replayed == ["2", "2", None, None]
    # The other call is untouched by the first one's exhausted allowance.
    assert loop.cached_result(second) == "4"


def test_a_failed_dispatch_is_never_replayed():
    """Replaying an error teaches the model nothing it has not already seen."""
    loop = NativeToolLoop(tools={"calculator": Calculator()}, nudge_cap=10)
    check = loop.check_action("calculator", '{"expression": "1/0"}')
    loop.record_action(check)
    loop.record_pair_result(check, "Error executing tool 'calculator': boom")

    assert loop.cached_result(check) is None


def test_a_stuck_run_on_a_short_budget_still_reports_a_typed_partial():
    """A budget too small for the replays ends at the cap, not with a dump.

    Three iterations cannot reach the loop guard, so the iteration cap is what
    ends the run. It reports the same way: a stated outcome in ``output`` and
    the progress under ``partial``.
    """
    agent, _ = _agent(
        [_search("ostrich egg mass")], tools=[_DeadEndSearch()], max_iterations=3,
    )
    resp = agent.run("Explain what an ostrich egg weighs.")

    assert resp.outcome == "stopped"
    assert resp.stop_reason == "max_iterations_partial"
    assert resp.partial is not None
    assert "No results found." in resp.partial.text
    assert "No results found." not in (resp.output or "")


# --------------------------------------------------------------------------- #
# The thresholds, and why they are what they are
# --------------------------------------------------------------------------- #
def test_the_drift_threshold_sits_above_the_length_of_real_work():
    """Four steps and a check is five calls, so no threshold may sit at five."""
    assert FUZZY_LOOP_FLOOR > 5
    assert FUZZY_LOOP_THRESHOLD >= FUZZY_LOOP_FLOOR
    assert FUZZY_LOOP_THRESHOLD_DATA >= FUZZY_LOOP_THRESHOLD
    assert MAX_BATCH_TOOL_RUNS > 2
    assert NUDGE_AFTER_CALLS >= FUZZY_LOOP_FLOOR


class _Chunker(BaseTool):
    """A data-processing tool, the category several calls in a row are normal for."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="chunker", description="Process a chunk.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(
                name="chunk", type=ParameterType.STRING,
                description="chunk", required=True,
            )],
        ))

    async def _execute(self, chunk: str = "", **kwargs):  # pragma: no cover
        return "done"


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        (4, FUZZY_LOOP_FLOOR),               # too small to reach: the cap ends it
        (10, 9),                             # one below the budget
        (40, FUZZY_LOOP_THRESHOLD),          # the declared count
    ],
)
def test_the_drift_threshold_is_bounded_by_the_runs_own_budget(cap, expected):
    """A threshold the budget cannot reach is dead code, not a guard."""
    loop = NativeToolLoop(tools={"calculator": Calculator()}, nudge_cap=cap)
    assert loop.fuzzy_threshold("calculator") == expected


def test_a_data_processing_tool_is_allowed_more_calls():
    """The count comes from the tool's declared category, not from its name."""
    plain = NativeToolLoop(tools={"calculator": Calculator()}, nudge_cap=40)
    data = NativeToolLoop(tools={"chunker": _Chunker()}, nudge_cap=40)

    assert data.fuzzy_threshold("chunker") == FUZZY_LOOP_THRESHOLD_DATA
    assert data.fuzzy_threshold("chunker") > plain.fuzzy_threshold("calculator")


def test_the_reminder_waits_for_an_unusual_call_count():
    loop = NativeToolLoop(tools={"calculator": Calculator()}, nudge_cap=40)

    assert loop.post_tool_nudge(1, 1, "36") is None
    assert loop.post_tool_nudge(1, NUDGE_AFTER_CALLS - 1, "36") is None
    assert loop.post_tool_nudge(1, NUDGE_AFTER_CALLS, "36") == NUDGE_HAVE_RESULTS


def test_a_failed_dispatch_earns_no_you_already_have_results_reminder():
    loop = NativeToolLoop(tools={"calculator": Calculator()}, nudge_cap=40)
    error = "Error executing tool 'calculator': boom"

    assert loop.post_tool_nudge(1, NUDGE_AFTER_CALLS, error) is None


def test_the_exact_check_still_reports_a_repeat_it_cannot_replay():
    """``cached_result`` returning ``None`` must not hide the repeat itself."""
    loop = NativeToolLoop(tools={"calculator": Calculator()}, nudge_cap=10)
    check = loop.check_action("calculator", '{"expression": "1+1"}')
    assert isinstance(check, LoopCheck)
    assert check.is_exact_loop is False
    loop.record_action(check)

    again = loop.check_action("calculator", '{"expression": "1+1"}')
    assert again.is_exact_loop is True
    assert again.loop_type == "exact"
