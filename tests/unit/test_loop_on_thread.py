"""The blocking loop keeps its conversation as typed steps, not as a string.

The ReAct loop used to grow one string and re-send it, and every question about
what a run had done was answered by a regex over that string: which observations
came back, what the last thought was, whether the run had reached anything worth
reporting. The loop now builds an :class:`~effgen.core.thread.AgentThread` and
renders it wherever the prompt frame needs text, so the transcript the model
reads is unchanged and the questions are answered from the steps themselves.

What is pinned here:

* the run hands its thread back on the response, and that thread renders the
  transcript the prompt carried, byte for byte;
* a caller's own ``system_prompt_template`` still receives that transcript in
  its ``{scratchpad}`` field;
* a run resumed from a transcript continues from what that transcript said;
* the readings that were regexes are readings of steps, and they no longer
  return the framework's own lines, ReAct scaffolding, or a truncated list of
  what the tools returned.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.thread import (
    ActionStep,
    AgentThread,
    AnswerStep,
    NudgeStep,
    ObservationStep,
    ThoughtStep,
)
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.builtin.calculator import Calculator

SOURCE = Path(__file__).resolve().parents[2] / "effgen" / "core"


class _Scripted(BaseModel):
    """Says what it was told to say, in order, and records every prompt."""

    def __init__(self, turns: list[str]) -> None:
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self._turns = turns
        self.prompts: list[str] = []
        self.calls = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        self.prompts.append(prompt)
        text = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs):
        yield self.generate(prompt).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def supports_tool_calling(self) -> bool:
        return False


def _calc(expression: str) -> str:
    return (
        "Thought: compute it.\nAction: calculator\n"
        f"Action Input: {json.dumps({'expression': expression})}"
    )


def _agent(turns, **cfg) -> Agent:
    model = _Scripted(turns)
    return Agent(config=AgentConfig(
        name="thread-loop-test",
        model=model,
        tools=[Calculator()],
        tool_calling_mode="react",
        max_iterations=cfg.pop("max_iterations", 6),
        raise_on_error=False,
        enable_memory=False,
        **cfg,
    ))


def _reader() -> Agent:
    """An agent used only to call the readings, with no run behind them."""
    return Agent(config=AgentConfig(
        name="thread-reader", model=_Scripted([""]), enable_memory=False,
    ))


# --------------------------------------------------------------------------- #
# The run carries its conversation
# --------------------------------------------------------------------------- #
def test_the_run_hands_back_the_thread_it_built():
    agent = _agent([_calc("12*3"), "Thought: done.\nFinal Answer: 36"])
    response = agent.run("What is 12*3? Explain the steps.")

    thread = response.metadata["thread"]
    assert isinstance(thread, AgentThread)
    kinds = [step.kind for step in thread]
    assert kinds == ["thought", "action", "observation", "thought", "answer"]
    assert thread.steps[1].tool == "calculator"
    assert thread.steps[2].text.strip() == "36"
    assert thread.steps[-1].stop_reason == "final_answer"


def test_the_thread_renders_the_transcript_the_prompt_carried():
    """Every prompt after the first ends with the transcript up to that turn."""
    agent = _agent([_calc("12*3"), "Thought: done.\nFinal Answer: 36"])
    response = agent.run("What is 12*3? Explain the steps.")

    thread = response.metadata["thread"]
    transcript = thread.to_text()
    second_prompt = agent.model.prompts[1]

    assert transcript.startswith("\nThought: compute it.")
    assert "\nObservation: 36" in transcript
    # The frame is unchanged: the prompt still ends with the transcript as it
    # stood when that turn was assembled, and the transcript is exactly what the
    # steps render.
    assembled = "".join(step.to_text() for step in thread.steps[:3])
    assert second_prompt.endswith(assembled)


def test_a_stopped_run_carries_its_thread_and_why_it_stopped():
    agent = _agent([_calc("2+2"), _calc("2+2"), _calc("2+2")], max_iterations=6)
    response = agent.run("What is 2+2? Explain the steps.")

    assert response.success is False
    thread = response.metadata["thread"]
    assert isinstance(thread, AgentThread)
    answer = thread.steps[-1]
    assert isinstance(answer, AnswerStep)
    assert answer.stop_reason == response.stop_reason


def test_a_declined_call_is_marked_as_one():
    """A call the loop refused to make is answered by the framework, not a tool."""
    agent = _agent([_calc("2+2"), _calc("2+2"), _calc("2+2")], max_iterations=6)
    response = agent.run("What is 2+2? Explain the steps.")

    thread = response.metadata["thread"]
    declined = [
        step
        for step in thread
        if isinstance(step, ObservationStep) and step.declined
    ]
    assert declined, "the run stopped on a repeat, so a call was declined"
    # What the framework said is not reported as something a tool returned.
    assert all(step.text not in (response.metadata.get("partial_output") or "")
               for step in declined)


def test_a_saved_run_carries_its_thread_as_data():
    """A response document holds no objects, so the thread is written as data."""
    agent = _agent([_calc("12*3"), "Thought: done.\nFinal Answer: 36"])
    response = agent.run("What is 12*3? Explain the steps.")

    document = response.to_dict()
    thread = document["metadata"]["thread"]
    assert json.dumps(thread)  # a document, not an object
    assert AgentThread.from_dict(thread).to_text() == (
        response.metadata["thread"].to_text()
    )


# --------------------------------------------------------------------------- #
# The published extension point still receives a string
# --------------------------------------------------------------------------- #
def test_a_custom_template_still_receives_the_transcript():
    template = (
        "TOOLS {tools_description}\nHISTORY {conversation_history}\n"
        "TASK {task}\nPAD[{scratchpad}]"
    )
    agent = _agent(
        [_calc("5*5"), "Thought: done.\nFinal Answer: 25"],
        system_prompt_template=template,
    )
    response = agent.run("What is 5*5? Explain the steps.")

    thread = response.metadata["thread"]
    assembled = "".join(step.to_text() for step in thread.steps[:3])
    second_prompt = agent.model.prompts[1]
    assert second_prompt.startswith("TOOLS ")
    assert f"PAD[{assembled}]" == second_prompt[second_prompt.index("PAD[") :]


# --------------------------------------------------------------------------- #
# Resuming
# --------------------------------------------------------------------------- #
def test_a_resumed_run_continues_from_the_transcript_it_was_given():
    seed = "\nThought: earlier turn.\nAction: calculator\nAction Input: 2+2\nObservation: 4"
    agent = _agent(["Thought: done.\nFinal Answer: 4"])
    response = agent.run("What is 2+2? Explain the steps.", _resume_scratchpad=seed)

    first_prompt = agent.model.prompts[0]
    assert seed in first_prompt
    thread = response.metadata["thread"]
    assert thread.to_text().startswith(seed)
    assert [step.kind for step in thread][:3] == ["thought", "action", "observation"]


def test_a_resume_text_that_is_not_a_transcript_says_so(caplog):
    """Every transcript effGen writes opens at a step boundary. One that does
    not is read for what it says, and the run reports that it did so."""
    agent = _agent(["Thought: done.\nFinal Answer: 4"])
    with caplog.at_level("INFO", logger="effgen.core.agent_loop"):
        agent.run(
            "What is 2+2? Explain the steps.",
            _resume_scratchpad="Thought: earlier turn.\nObservation: 4",
        )
    assert any(
        "does not begin at a step boundary" in record.message
        for record in caplog.records
    )


# --------------------------------------------------------------------------- #
# The readings that used to be regexes over the text
# --------------------------------------------------------------------------- #
def test_a_framework_line_is_never_reported_as_the_run_s_progress():
    """The loop's own "(continue reasoning)" line is not something the run found."""
    agent = _reader()
    thread = AgentThread(steps=[
        ThoughtStep(text=""),
        NudgeStep(text="Action: (continue reasoning)", nudge_id="continue_reasoning"),
    ])
    assert agent._extract_partial_answer(thread) is None


def test_an_announced_answer_reports_the_result_not_the_scaffolding():
    agent = _reader()
    thread = AgentThread(steps=[
        ThoughtStep(text="I now know the answer."),
        ActionStep(tool="calculator", raw="2+2"),
        ObservationStep(text="4"),
    ])
    assert agent._extract_partial_answer(thread) == "4"


def test_every_result_is_reported_not_only_the_first():
    agent = _reader()
    thread = AgentThread(steps=[
        ThoughtStep(text="a"),
        ActionStep(tool="calculator", raw="1+1"),
        ObservationStep(text="2"),
        ObservationStep(text="3"),
    ])
    assert agent._extract_partial_answer(thread) == "2 | 3"


def test_a_result_that_reads_like_a_react_line_is_still_the_tool_s_words():
    agent = _reader()
    thread = AgentThread(steps=[
        ThoughtStep(text="a"),
        ActionStep(tool="web_search", raw="q"),
        ObservationStep(text="Thought: the passage begins like a ReAct line"),
        ThoughtStep(text="b"),
    ])
    assert agent._extract_partial_answer(thread) == (
        "Thought: the passage begins like a ReAct line"
    )
    assert agent._partial_result(thread, text="").last_thought == "b"


def test_a_nudge_is_not_a_result():
    from effgen.core.agent_runtime import NUDGE_NOT_USABLE

    agent = _reader()
    thread = AgentThread(steps=[
        ThoughtStep(text="a"),
        ActionStep(tool="calculator", raw="2*2"),
        ObservationStep(text="4"),
        NudgeStep(text=NUDGE_NOT_USABLE, render_as="observation", nudge_id="not_usable"),
    ])
    assert agent._extract_partial_answer(thread) == "4"


def test_an_empty_thread_reaches_nothing():
    """A guard, not a proof: a run with no steps reached nothing either way."""
    agent = _reader()
    assert agent._extract_partial_answer(AgentThread()) is None


# --------------------------------------------------------------------------- #
# The string accumulator, and the parse of it, are gone
# --------------------------------------------------------------------------- #
def test_the_blocking_loop_appends_to_no_string():
    source = (SOURCE / "agent_react.py").read_text()
    assert "scratchpad +=" not in source


#: The four expressions that used to recover a run from its rendered text. The
#: reader now reads steps, so none of them may come back. The expressions that
#: read a *model turn* are a different thing and stay where they are.
_TRANSCRIPT_REGEXES = (
    r"Thought:\s*I (?:now )?know[^.]*\.\s*(.+?)(?=\nThought:|\nAction:|\Z)",
    r"Observation:\s*(.+?)(?=\nThought:|\nAction:|\Z)",
    r"Thought:\s*(.+?)(?=\nAction:|\nObservation:|\Z)",
)


@pytest.mark.parametrize("pattern", _TRANSCRIPT_REGEXES)
def test_the_turn_reader_no_longer_parses_the_transcript(pattern):
    source = (SOURCE / "agent_react_parsing.py").read_text()
    assert pattern not in source, "a regex over the transcript came back"


def test_nothing_in_the_turn_reader_takes_a_transcript():
    """Read from the AST: a parameter named for the string is the giveaway."""
    tree = ast.parse((SOURCE / "agent_react_parsing.py").read_text())
    taking = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for argument in [*node.args.args, *node.args.kwonlyargs]
        if argument.arg == "scratchpad"
    ]
    assert taking == []
