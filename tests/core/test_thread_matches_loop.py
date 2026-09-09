"""A thread renders the transcript the agent loop actually writes.

The loop assembles its transcript by appending to a string. A thread renders
one by concatenating typed steps. These have to agree byte for byte, or moving
the loop onto the thread would change every prompt the model sees.

The runs here are driven by a scripted model and a stub tool, so the transcript
is deterministic and no model, network or card is involved.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.thread import (
    ActionStep,
    AgentThread,
    NudgeStep,
    ObservationStep,
    ThoughtStep,
)
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)


class ScriptedModel(BaseModel):
    """Says what it was told to say, in order, repeating the last turn."""

    def __init__(self, turns: list[str]) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.TRANSFORMERS)
        self.turns = turns
        self.index = 0
        self._is_loaded = True

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> GenerationResult:
        text = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        return GenerationResult(
            text=text, tokens_used=1, finish_reason="stop", model_name=self.model_name
        )

    def generate_stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


class Doubler(BaseTool):
    """Returns twice whatever number it is given."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="doubler",
                description="Doubles a number.",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="value",
                        type=ParameterType.STRING,
                        description="The number to double.",
                        required=True,
                    )
                ],
            )
        )

    async def _execute(self, **kwargs: Any) -> str:
        return str(int(kwargs.get("value", 0)) * 2)


def run_and_capture(turns: list[str], task: str, max_iterations: int = 6):
    """Run the loop and hand back the response and its final transcript."""
    agent = Agent(
        AgentConfig(
            name="thread-agreement",
            model=ScriptedModel(turns),
            tools=[Doubler()],
            max_iterations=max_iterations,
            raise_on_error=False,
            enable_memory=False,
        )
    )
    response = agent.run(task, _debug=True)
    trace = (response.metadata or {}).get("debug_trace")
    snapshots = [
        it.scratchpad_snapshot
        for it in getattr(trace, "iterations", [])
        if it.scratchpad_snapshot is not None
    ]
    return response, (snapshots[-1] if snapshots else "")


def test_a_tool_run_renders_byte_identically():
    turns = [
        'Thought: double it\nAction: doubler\nAction Input: {"value": "21"}',
        "Thought: I now know the answer.\nFinal Answer: 42",
    ]
    response, scratchpad = run_and_capture(turns, "What is 21 doubled?")
    assert response.success
    assert scratchpad

    thread = AgentThread(
        steps=[
            ThoughtStep("double it"),
            ActionStep(tool="doubler", raw='{"value": "21"}'),
            ObservationStep(text="42"),
            # The turn that answers reports no thought of its own, and the
            # loop still writes the label — so does the thread.
            ThoughtStep(""),
        ]
    )
    assert thread.to_text() == scratchpad


def test_a_turn_with_no_thought_renders_the_empty_line_the_loop_writes():
    turns = [
        'Action: doubler\nAction Input: {"value": "5"}',
        "Thought: done\nFinal Answer: 10",
    ]
    response, scratchpad = run_and_capture(turns, "What is 5 doubled?")
    assert response.success

    thread = AgentThread(
        steps=[
            ThoughtStep(""),
            ActionStep(tool="doubler", raw='{"value": "5"}'),
            ObservationStep(text="10"),
            ThoughtStep(""),
        ]
    )
    assert thread.to_text() == scratchpad
    assert "\nThought: \n" in scratchpad
    assert "None" not in scratchpad


def test_an_unknown_tool_renders_the_observation_the_loop_writes():
    turns = [
        'Thought: try it\nAction: tripler\nAction Input: {"value": "5"}',
        "Thought: done\nFinal Answer: 15",
    ]
    response, scratchpad = run_and_capture(turns, "What is 5 tripled?")
    assert response.success

    observation = scratchpad.split("\nObservation: ", 1)[1].split("\nThought:")[0]
    assert "tripler" in observation
    thread = AgentThread(
        steps=[
            ThoughtStep("try it"),
            ActionStep(tool="tripler", raw='{"value": "5"}'),
            ObservationStep(text=observation),
            ThoughtStep(""),
        ]
    )
    assert thread.to_text() == scratchpad


@pytest.mark.parametrize(
    "turns,task",
    [
        (
            [
                'Thought: double it\nAction: doubler\nAction Input: {"value": "21"}',
                "Thought: I now know the answer.\nFinal Answer: 42",
            ],
            "What is 21 doubled?",
        ),
        (
            [
                'Thought: again\nAction: doubler\nAction Input: {"value": "3"}',
                'Thought: again\nAction: doubler\nAction Input: {"value": "3"}',
            ],
            "Double three, twice.",
        ),
    ],
)
def test_a_recovered_thread_renders_the_transcript_it_was_read_from(turns, task):
    """A transcript stored before the steps were kept still renders back."""
    _, scratchpad = run_and_capture(turns, task)
    assert scratchpad
    assert AgentThread.from_scratchpad(scratchpad).to_text() == scratchpad


def test_the_message_rendering_of_a_real_run_is_well_formed():
    turns = [
        'Thought: double it\nAction: doubler\nAction Input: {"value": "21"}',
        "Thought: I now know the answer.\nFinal Answer: 42",
    ]
    _, scratchpad = run_and_capture(turns, "What is 21 doubled?")
    thread = AgentThread.from_scratchpad(scratchpad)
    messages = thread.to_messages()

    assert thread.unanswered_calls() == []
    assistant = next(
        m
        for m in messages
        if m.role.value == "assistant"
        and any(getattr(p, "type", "") == "tool_call" for p in m.content)
    )
    assert assistant.text == "double it"
    calls = {
        p.tool_call_id for m in messages for p in m.content if getattr(p, "type", "") == "tool_call"
    }
    results = [
        p.tool_call_id
        for m in messages
        for p in m.content
        if getattr(p, "type", "") == "tool_result"
    ]
    assert results and all(call_id in calls for call_id in results)


def test_a_nudge_the_loop_injects_is_attributable_in_the_thread():
    """A framework line reads as the framework's, not as the model's."""
    thread = AgentThread(
        steps=[
            ThoughtStep("go"),
            ActionStep(tool="doubler", raw='{"value": "1"}'),
            ObservationStep(text="2"),
            NudgeStep(text="[Tool results computed above.]", nudge_id="continue"),
        ]
    )
    assert thread.to_text().endswith("\n[Tool results computed above.]")
    assert thread.partial_answer() == "2"
