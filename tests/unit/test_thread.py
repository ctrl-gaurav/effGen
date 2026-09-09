"""The step model: what each step renders, and what a thread does with them.

The flat rendering is checked byte for byte, because the whole point of the
type is that a transcript rebuilt from steps is the transcript the loop wrote.
"""

from __future__ import annotations

import json

import pytest

from effgen.core.messages import ImagePart, Role, TextPart, ToolCallPart, ToolResultPart
from effgen.core.thread import (
    THREAD_SCHEMA_VERSION,
    ActionStep,
    AgentThread,
    AnswerStep,
    NudgeStep,
    ObservationStep,
    Step,
    SystemStep,
    TaskStep,
    ThoughtStep,
    step_from_dict,
)

# ---------------------------------------------------------------------------
# The flat rendering, byte for byte
# ---------------------------------------------------------------------------


def test_thought_renders_label_and_text():
    assert ThoughtStep("weigh the options").to_text() == "\nThought: weigh the options"


def test_absent_thought_renders_the_bare_label():
    """A turn that made a native call reports no thought.

    The transcript is text the model reads back, so an absent thought is an
    empty line after the label — never the word ``None``.
    """
    assert ThoughtStep().to_text() == "\nThought: "
    assert "None" not in ThoughtStep().to_text()


def test_action_renders_the_input_the_turn_wrote():
    step = ActionStep(tool="calculator", raw='{"expression": "2+2"}')
    assert step.to_text() == '\nAction: calculator\nAction Input: {"expression": "2+2"}'


def test_action_built_from_arguments_renders_them_as_json():
    step = ActionStep(tool="calculator", arguments={"expression": "2+2"})
    assert step.to_text() == '\nAction: calculator\nAction Input: {"expression": "2+2"}'


def test_action_keeps_free_text_input_under_a_raw_key():
    step = ActionStep(tool="web_search", raw="who won in 1998")
    assert step.arguments == {"__raw_input__": "who won in 1998"}
    assert step.to_text() == "\nAction: web_search\nAction Input: who won in 1998"


def test_observation_renders_the_result():
    assert ObservationStep(text="4").to_text() == "\nObservation: 4"


def test_nudge_renders_where_the_loop_put_it():
    assert NudgeStep(text="[carry on]").to_text() == "\n[carry on]"
    assert (
        NudgeStep(text="[carry on]", render_as="observation").to_text()
        == "\nObservation: [carry on]"
    )


@pytest.mark.parametrize(
    "step",
    [SystemStep(text="you are helpful"), TaskStep(text="what is 2+2"), AnswerStep(text="4")],
)
def test_frame_and_answer_steps_are_not_transcript(step):
    """The frame is assembled around a transcript, and an answer ends a run."""
    assert step.to_text() == ""


def test_to_text_is_concatenation_and_nothing_else():
    steps = [
        SystemStep(text="you are helpful"),
        TaskStep(text="what is 2+2"),
        ThoughtStep("use the tool"),
        ActionStep(tool="calculator", raw='{"expression": "2+2"}'),
        ObservationStep(text="4"),
        NudgeStep(text="[carry on]"),
        AnswerStep(text="4"),
    ]
    thread = AgentThread(steps=list(steps))
    assert thread.to_text() == "".join(step.to_text() for step in steps)
    assert thread.to_text() == (
        "\nThought: use the tool"
        '\nAction: calculator\nAction Input: {"expression": "2+2"}'
        "\nObservation: 4"
        "\n[carry on]"
    )


def test_every_step_satisfies_the_protocol():
    for step in (
        SystemStep(text="s"),
        TaskStep(text="t"),
        ThoughtStep("th"),
        ActionStep(tool="calculator"),
        ObservationStep(text="o"),
        NudgeStep(text="n"),
        AnswerStep(text="a"),
    ):
        assert isinstance(step, Step)


# ---------------------------------------------------------------------------
# The message rendering
# ---------------------------------------------------------------------------


def _worked_thread() -> AgentThread:
    return AgentThread(
        steps=[
            SystemStep(text="you are helpful", source="persona"),
            SystemStep(text="answer with a number", source="contract"),
            TaskStep(text="what is 2+2"),
            ThoughtStep("the calculator can do this"),
            ActionStep(tool="calculator", raw='{"expression": "2+2"}'),
            ObservationStep(text="4"),
            AnswerStep(text="4"),
        ]
    )


def test_several_system_steps_become_one_system_message():
    messages = _worked_thread().to_messages()
    systems = [m for m in messages if m.role is Role.SYSTEM]
    assert len(systems) == 1
    assert systems[0].text == "you are helpful\n\nanswer with a number"


def test_reasoning_and_the_call_travel_on_the_same_message():
    """Dropping the reasoning beside a native call loses why the tool was used."""
    messages = _worked_thread().to_messages()
    assistant = next(
        m
        for m in messages
        if m.role is Role.ASSISTANT
        and any(isinstance(p, ToolCallPart) for p in m.content)
    )
    assert [type(p).__name__ for p in assistant.content] == ["TextPart", "ToolCallPart"]
    assert assistant.text == "the calculator can do this"


def test_a_tool_message_answers_the_call_before_it():
    messages = _worked_thread().to_messages()
    call = next(
        p for m in messages for p in m.content if isinstance(p, ToolCallPart)
    )
    result = next(
        p for m in messages for p in m.content if isinstance(p, ToolResultPart)
    )
    assert result.tool_call_id == call.tool_call_id
    assert result.result == "4"


def test_a_provider_call_id_is_kept_and_a_missing_one_is_minted_deterministically():
    given = AgentThread(steps=[ActionStep(tool="t", call_id="call_abc"), ObservationStep(text="x")])
    assert given.to_messages()[0].content[-1].tool_call_id == "call_abc"
    assert given.to_messages()[1].content[0].tool_call_id == "call_abc"

    minted = AgentThread(steps=[ThoughtStep("a"), ActionStep(tool="t"), ObservationStep(text="x")])
    first = [m.content for m in minted.to_messages()]
    assert first[0][-1].tool_call_id == "effgen-1"
    assert minted.to_messages()[0].content[-1].tool_call_id == "effgen-1"


def test_a_task_carries_its_other_content_parts():
    part = ImagePart(image=b"\x89PNG\r\n", mime="image/png")
    messages = AgentThread(steps=[TaskStep(text="describe", parts=[part])]).to_messages()
    assert messages[0].role is Role.USER
    assert isinstance(messages[0].content[0], TextPart)
    assert isinstance(messages[0].content[1], ImagePart)


def test_a_nudge_is_attributed_to_the_framework():
    messages = AgentThread(steps=[NudgeStep(text="[carry on]", nudge_id="continue")]).to_messages()
    assert messages[0].role is Role.USER
    assert messages[0].metadata["effgen_nudge"] == "continue"


def test_a_standalone_thought_becomes_its_own_assistant_turn():
    messages = AgentThread(steps=[ThoughtStep("no tool needed"), AnswerStep(text="4")]).to_messages()
    assert [m.role for m in messages] == [Role.ASSISTANT, Role.ASSISTANT]
    assert messages[0].text == "no tool needed"


def test_summary_drops_the_bookkeeping_and_shortens_results():
    thread = AgentThread(
        steps=[
            ThoughtStep("standalone"),
            NudgeStep(text="[carry on]"),
            ActionStep(tool="t"),
            ObservationStep(text="x" * 500),
        ]
    )
    messages = thread.to_messages(summary=True)
    assert all(m.metadata.get("effgen_nudge") is None for m in messages)
    result = next(p for m in messages for p in m.content if isinstance(p, ToolResultPart))
    assert len(result.result) < 500


def test_render_picks_a_protocol_and_rejects_anything_else():
    thread = _worked_thread()
    assert thread.render("flat") == thread.to_text()
    rendered = thread.render("messages")
    assert [m.role for m in rendered] == [m.role for m in thread.to_messages()]
    with pytest.raises(ValueError, match="not a rendering this thread has"):
        thread.render("xml")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_round_trip_is_lossless_and_versioned():
    thread = _worked_thread()
    thread.metadata["run"] = "abc"
    data = thread.to_dict()
    assert data["version"] == THREAD_SCHEMA_VERSION
    again = AgentThread.from_dict(data)
    assert again.to_dict() == data
    assert again.to_text() == thread.to_text()
    assert json.loads(json.dumps(data)) == data


def test_round_trip_keeps_a_task_s_content_parts():
    thread = AgentThread(
        steps=[TaskStep(text="describe", parts=[ImagePart(image=b"\x89PNG\r\n", mime="image/png")])]
    )
    again = AgentThread.from_dict(thread.to_dict())
    assert isinstance(again.steps[0].parts[0], ImagePart)
    assert again.steps[0].parts[0].image == b"\x89PNG\r\n"
    assert again.to_dict() == thread.to_dict()
    assert json.loads(json.dumps(thread.to_dict())) == thread.to_dict()


def test_an_unknown_step_kind_is_refused_rather_than_dropped():
    with pytest.raises(ValueError, match="is not one this release knows"):
        step_from_dict({"kind": "telepathy"})


def test_a_newer_schema_is_read_with_a_warning(caplog):
    data = AgentThread(steps=[ThoughtStep("x")]).to_dict()
    data["version"] = THREAD_SCHEMA_VERSION + 1
    with caplog.at_level("WARNING"):
        AgentThread.from_dict(data)
    assert "[thread] reading schema version" in caplog.text


# ---------------------------------------------------------------------------
# Recovering a transcript written before the steps were kept
# ---------------------------------------------------------------------------


def test_from_scratchpad_recovers_the_order_and_renders_the_same_text():
    text = (
        "\nThought: use the tool"
        '\nAction: calculator\nAction Input: {"expression": "2+2"}'
        "\nObservation: 4"
    )
    thread = AgentThread.from_scratchpad(text)
    assert [step.kind for step in thread.steps] == ["thought", "action", "observation"]
    assert thread.to_text() == text


def test_from_scratchpad_keeps_a_multi_line_observation_whole():
    text = "\nObservation: line one\nline two\nline three"
    thread = AgentThread.from_scratchpad(text)
    assert thread.observations()[0].text == "line one\nline two\nline three"
    assert thread.to_text() == text


def test_from_scratchpad_keeps_a_line_the_loop_wrote_without_an_input():
    # The loop writes a bare "Action: (continue reasoning)" line when a turn
    # asked for nothing. It is a line, not a call, and re-rendering the
    # recovered thread must not invent an empty Action Input for it or move it
    # to the end of the run.
    text = "\nThought: a\nAction: (continue reasoning)\nThought: b"
    thread = AgentThread.from_scratchpad(text)
    assert [step.kind for step in thread.steps] == ["thought", "nudge", "thought"]
    assert thread.to_text() == text


def test_from_scratchpad_keeps_a_line_that_carries_no_marker():
    text = "\n[Tool results computed above. Continue:]\nThought: a"
    thread = AgentThread.from_scratchpad(text)
    assert [step.kind for step in thread.steps] == ["nudge", "thought"]
    assert thread.to_text() == text


@pytest.mark.parametrize(
    "text",
    [
        "\nThought: a",
        "\nThought: ",
        "\nThought: a\nAction: t\nAction Input: {}\nObservation: r",
        "\nObservation: one\ntwo",
        "\nThought: a\nAction: (continue reasoning)",
        "\n[a framework line]\nObservation: r\n[another one]",
        "\nAction Input: orphaned",
        "",
    ],
)
def test_from_scratchpad_renders_back_the_text_it_read(text):
    assert AgentThread.from_scratchpad(text).to_text() == text


def test_from_scratchpad_logs_what_it_recovered(caplog):
    with caplog.at_level("INFO"):
        AgentThread.from_scratchpad("\nThought: a\nObservation: b")
    assert "[thread] recovered 2 steps" in caplog.text


# ---------------------------------------------------------------------------
# Reading the run back
# ---------------------------------------------------------------------------


def test_observations_and_thoughts_come_back_in_order():
    thread = AgentThread(
        steps=[
            ThoughtStep("first"),
            ActionStep(tool="t"),
            ObservationStep(text="one"),
            ThoughtStep(""),
            ActionStep(tool="t"),
            ObservationStep(text="two"),
        ]
    )
    assert [o.text for o in thread.observations()] == ["one", "two"]
    assert thread.last_observation().text == "two"
    assert thread.last_thought().text == "first"


def test_unanswered_calls_names_the_call_a_provider_would_reject():
    answered = AgentThread(steps=[ActionStep(tool="t"), ObservationStep(text="x")])
    assert answered.unanswered_calls() == []

    dangling = AgentThread(steps=[ActionStep(tool="t")])
    assert [step.tool for step in dangling.unanswered_calls()] == ["t"]


def test_partial_answer_skips_the_framework_s_own_words_and_repeats():
    thread = AgentThread(
        steps=[
            ActionStep(tool="t"),
            ObservationStep(text="Paris"),
            NudgeStep(text="[carry on]"),
            ActionStep(tool="t"),
            ObservationStep(text="Paris"),
            ActionStep(tool="t"),
            ObservationStep(text="population 2.1m"),
            ActionStep(tool="t"),
            ObservationStep(text="Error: timed out", is_error=True),
        ]
    )
    assert thread.partial_answer() == "Paris | population 2.1m"


def test_partial_answer_falls_back_to_a_substantive_thought():
    thread = AgentThread(steps=[ThoughtStep("the answer is somewhere near forty-two")])
    assert thread.partial_answer() == "the answer is somewhere near forty-two"
    assert AgentThread(steps=[ThoughtStep("hmm")]).partial_answer() is None


def test_a_thread_is_a_sequence_of_steps():
    thread = AgentThread()
    thread.append(ThoughtStep("a"))
    thread.extend([ActionStep(tool="t"), ObservationStep(text="x")])
    assert len(thread) == 3
    assert [step.kind for step in thread] == ["thought", "action", "observation"]
