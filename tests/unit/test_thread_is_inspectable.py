"""Reading a finished run back: the accessor, the document, the rendering.

A run's conversation is the most useful thing it produces for anyone trying to
work out what happened, so three things have to hold. There is one documented
way to reach it. The whole run serialises, including the tree of what it
executed, so ``json.dumps`` on a saved run does not raise for a run that called
a tool. And the rendering is the same bytes for two runs that took the same
path, with secrets replaced, so two of them can be diffed and shared.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.multimodal import image_from
from effgen.core.thread import (
    ActionStep,
    AgentThread,
    AnswerStep,
    DelegationStep,
    ObservationStep,
    SystemStep,
    TaskStep,
    ThoughtStep,
)
from effgen.core.thread_render import RenderedStep, render_thread, thread_as_text
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

#: A key-shaped string that is not a key: the scrubber matches it on shape.
FAKE_KEY = "sk-ZZZZfakefakefakefakefakefake00"

#: A one-pixel PNG, so a test needs no fixture file on disk.
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
    b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class Calc(BaseTool):
    """A deterministic tool, so a turn's call has one right answer."""

    def __init__(self, result: str = "36") -> None:
        super().__init__(metadata=ToolMetadata(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="the expression", required=True,
            )],
        ))
        self._result = result

    async def _execute(self, **kwargs: Any) -> str:
        return self._result


class Scripted(BaseModel):
    """Replays a script and keeps every prompt it was handed."""

    def __init__(self, script: list[dict]) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.script = script
        self.prompts: list[Any] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return True

    def tool_call_support(self) -> str:
        return "api"

    def streams_tool_calls(self) -> bool:
        return False

    def supports_conversation(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt)
        turn = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        calls = turn.get("calls")
        return GenerationResult(
            text=turn.get("text", ""),
            tokens_used=5,
            finish_reason="tool_calls" if calls else "stop",
            model_name=self.model_name,
            metadata={"tool_calls": calls} if calls else {},
        )

    def generate_stream(self, prompt: Any, config: Any = None, **kwargs: Any):
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _call(name: str, arguments: str, call_id: str = "call-1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _script(call_id: str = "call-1", result: str = "36") -> list[dict]:
    return [
        {"text": "Thought: multiply.",
         "calls": [_call("calculator", '{"expression": "6*6"}', call_id)]},
        {"text": f"Thought: done.\nFinal Answer: {result}"},
    ]


def _run(*, call_id: str = "call-1", tool_result: str = "36", **config: Any):
    settings: dict[str, Any] = {
        "name": "inspectable",
        "model": Scripted(_script(call_id)),
        "tools": [Calc(tool_result)],
        "max_iterations": 4,
        "tool_calling_mode": "hybrid",
        "raise_on_error": False,
    }
    settings.update(config)
    agent = Agent(AgentConfig(**settings))
    try:
        return agent.run("what is 6*6?")
    finally:
        agent.close()


# --- one documented accessor -------------------------------------------------


def test_a_completed_run_hands_back_its_conversation_from_one_named_place() -> None:
    """``response.thread`` is the documented path, and the old one still works."""
    response = _run()

    assert isinstance(response.thread, AgentThread)
    assert response.thread is response.metadata["thread"]
    assert [s.kind for s in response.thread.steps][:2] == ["system", "task"]
    assert any(s.kind == "action" for s in response.thread.steps)


def test_the_accessor_is_none_rather_than_a_key_error_when_there_is_no_thread() -> None:
    """A response built by hand answers the accessor instead of raising."""
    from effgen.core.agent_response import AgentResponse

    assert AgentResponse(output="x").thread is None


def test_the_conversation_object_is_not_json_but_the_run_document_is() -> None:
    """``to_dict()`` is the serialisable form; the live metadata is objects."""
    response = _run()

    with pytest.raises(TypeError):
        json.dumps(response.metadata)

    document = response.to_dict()
    json.dumps(document)  # the documented path does not raise
    steps = document["metadata"]["thread"]["steps"]
    assert [s["kind"] for s in steps][:2] == ["system", "task"]


# --- the run document is JSON after a tool call ------------------------------


def test_a_run_that_called_a_tool_still_serialises() -> None:
    """The execution tree is data, so a saved run document round-trips."""
    response = _run()
    assert response.tool_calls.total == 1

    text = json.dumps(response.to_dict())
    recovered = json.loads(text)

    assert recovered["execution_tree"]
    assert recovered["tool_call_details"][0]["name"] == "calculator"


def test_the_tree_keeps_what_a_tool_call_recorded_rather_than_dropping_it() -> None:
    """Coercing the tree to data does not lose the calls it carried."""
    response = _run()
    tree = json.loads(json.dumps(response.to_dict()))["execution_tree"]

    recorded = tree.get("metadata", {}).get("tool_calls")
    assert recorded, "the finished tree records the calls the run made"
    assert [c["name"] for c in recorded] == ["calculator"]
    assert recorded[0]["result"] == "36"


# --- the rendering is stable enough to diff ----------------------------------


def test_two_runs_of_the_same_path_render_the_same_bytes() -> None:
    """Nothing that varies between runs reaches the rendering."""
    first = thread_as_text(_run(call_id="call-aaa").thread)
    second = thread_as_text(_run(call_id="call-zzz").thread)

    assert first == second
    assert "call-aaa" not in first and "call-zzz" not in second


def test_asking_for_ids_brings_them_back() -> None:
    """The ids are available to a caller who wants them, and only then."""
    thread = _run(call_id="call-aaa").thread

    assert "call-aaa" in thread_as_text(thread, include_ids=True)
    assert "call-aaa" not in thread_as_text(thread)


def test_a_run_that_did_different_work_renders_differently() -> None:
    """A stable rendering still shows a real difference."""
    assert thread_as_text(_run(tool_result="36").thread) != \
        thread_as_text(_run(tool_result="42").thread)


def test_every_step_is_rendered_in_order_with_its_own_heading() -> None:
    """The rendering names each step kind, numbered from one."""
    steps = render_thread(_run().thread)

    assert [s.position for s in steps] == list(range(1, len(steps) + 1))
    assert [s.kind for s in steps][:2] == ["system", "task"]
    assert any(s.label == "action (calculator)" for s in steps)
    assert all(isinstance(s, RenderedStep) for s in steps)


def test_a_call_renders_the_input_the_turn_actually_wrote() -> None:
    """What the model was sent is what is shown, not a summary of it."""
    text = thread_as_text(_run().thread)
    assert '"expression"' in text and "6*6" in text


# --- secrets are redacted on that path ---------------------------------------


def _leaky_thread() -> AgentThread:
    return AgentThread(steps=[
        SystemStep(text=f"You are a helper. Authenticate with {FAKE_KEY}."),
        TaskStep(text="what is 6*6?"),
        ThoughtStep(text="call the tool"),
        ActionStep(tool="calculator", raw='{"expression": "6*6"}'),
        ObservationStep(text=f"36 (billed to {FAKE_KEY})"),
        AnswerStep(text="36"),
    ])


def test_a_key_in_a_tool_result_or_an_instruction_does_not_reach_the_rendering() -> None:
    """Both places a key turns up are scrubbed, and the step still reads."""
    text = thread_as_text(_leaky_thread())

    assert FAKE_KEY not in text
    assert text.count("<REDACTED:openai_key>") == 2
    assert "36" in text and "what is 6*6?" in text


def test_redaction_can_be_turned_off_for_a_caller_that_needs_the_raw_text() -> None:
    """The scrubber is a default, not a wall."""
    assert FAKE_KEY in thread_as_text(_leaky_thread(), redact=False)


def test_a_key_reaching_a_real_run_is_redacted_in_the_command_line_document() -> None:
    """The document ``effgen run`` writes carries no key-shaped string."""
    from effgen.cli.commands.run import run_document

    response = _run(tool_result=f"36 (key {FAKE_KEY})")
    document = run_document(response)

    assert FAKE_KEY not in json.dumps(document)
    assert FAKE_KEY in json.dumps(response.to_dict()), "the library keeps what it saw"


# --- the shapes this was not written for (preamble G.3) ----------------------


def test_a_question_that_arrived_with_an_image_says_so_without_its_bytes() -> None:
    """An image step renders as an attachment, not as a wall of base64."""
    thread = AgentThread(steps=[
        TaskStep(text="what is in this picture?",
                 parts=[image_from(PNG, mime="image/png")]),
        AnswerStep(text="a single pixel"),
    ])
    steps = render_thread(thread)

    assert steps[0].detail["attachments"] == "image"
    assert steps[0].body == "what is in this picture?"
    assert "iVBOR" not in thread_as_text(thread)


def test_a_shortened_observation_says_it_was_shortened_and_by_how_much() -> None:
    """A compacted step reads as the observation it is, with its original size."""
    thread = AgentThread(steps=[
        ActionStep(tool="calculator", raw="{}"),
        ObservationStep(text="36…", compacted="elided", original_chars=4096),
    ])
    step = render_thread(thread)[1]

    assert step.detail["compacted"] == "elided"
    assert step.detail["original_chars"] == "4096"


def test_a_delegated_child_renders_its_own_steps_under_the_delegation() -> None:
    """A parent's rendering carries the conversation its child actually had."""
    child = AgentThread(steps=[TaskStep(text="half of it"), AnswerStep(text="18")])
    thread = AgentThread(steps=[
        TaskStep(text="what is 6*6, halved?"),
        DelegationStep(child_id="worker", role="sub-agent", task="half of it",
                       thread=child, output="18"),
        AnswerStep(text="18"),
    ])
    steps = render_thread(thread)

    assert [s.depth for s in steps] == [0, 0, 1, 1, 0]
    assert steps[1].label == "delegation (sub-agent:worker)"
    assert steps[1].detail["outcome"] == "ok"
    assert [s.body for s in steps[2:4]] == ["half of it", "18"]


def test_a_failed_delegation_names_the_failure() -> None:
    """A child that did not finish is still a step, with what went wrong."""
    thread = AgentThread(steps=[DelegationStep(
        child_id="worker", role="node", task="t", success=False, error="boom",
    )])
    step = render_thread(thread)[0]

    assert step.detail == {"outcome": "failed", "error": "boom"}


def test_a_conversation_read_back_from_a_checkpoint_renders_the_same() -> None:
    """A resumed run's conversation is the one it had before it was stored."""
    from effgen.core.checkpoint import Checkpoint

    thread = _run().thread
    stored = Checkpoint(
        checkpoint_id="cp-1", agent_name="inspectable", task="what is 6*6?",
        iteration=2, thread=thread.to_dict(),
    ).to_dict()
    recovered = Checkpoint.from_dict(json.loads(json.dumps(stored))).to_thread()

    assert thread_as_text(recovered) == thread_as_text(thread)


def test_the_rendering_reads_a_conversation_handed_over_as_plain_data() -> None:
    """A saved document renders without being turned back into objects first."""
    thread = _run().thread
    assert thread_as_text(thread.to_dict()) == thread_as_text(thread)


def test_no_conversation_renders_as_nothing_rather_than_raising() -> None:
    """Every caller can render whatever it has, including nothing."""
    assert render_thread(None) == []
    assert thread_as_text(None) == ""
    assert thread_as_text("not a thread") == ""


# --- the surfaces that render it ---------------------------------------------


def test_the_debug_inspector_renders_an_iteration_as_its_steps() -> None:
    """The inspector shows the conversation, not a flattened transcript."""
    from effgen.debug.inspector import DebugIteration, _conversation_text

    iteration = DebugIteration(
        iteration=1,
        scratchpad_snapshot="\nThought: multiply.\nObservation: 36",
        thread_snapshot=_leaky_thread().to_dict(),
    )
    rendered = _conversation_text(iteration)

    assert "1. system (persona)" in rendered
    assert "action (calculator)" in rendered
    assert FAKE_KEY not in rendered


def test_the_inspector_falls_back_to_the_snapshot_it_has() -> None:
    """A trace recorded before the steps were kept still shows something."""
    from effgen.debug.inspector import DebugIteration, _conversation_text

    iteration = DebugIteration(iteration=1, scratchpad_snapshot="\nThought: hello")
    assert _conversation_text(iteration) == "\nThought: hello"


def test_a_debug_run_renders_every_iteration_it_recorded() -> None:
    """What the inspector prints comes from the run's own steps."""
    from effgen.debug.inspector import _conversation_text

    agent = Agent(AgentConfig(
        name="inspectable", model=Scripted(_script()), tools=[Calc()],
        max_iterations=4, tool_calling_mode="hybrid", raise_on_error=False,
    ))
    try:
        response = agent.run("what is 6*6?", debug=True)
    finally:
        agent.close()
    trace = response.metadata["debug_trace"]

    assert trace.iterations
    for iteration in trace.iterations:
        assert "task" in _conversation_text(iteration)


def test_the_run_card_carries_the_conversation_as_a_table() -> None:
    """A shared HTML card shows every step, with secrets replaced."""
    from effgen.cli.commands.run import run_document
    from effgen.ui.report_html import build_html_report

    response = _run(tool_result=f"36 (billed to {FAKE_KEY})")
    html = build_html_report(run_document(response), kind="run")

    assert "<h2>Conversation</h2>" in html
    assert "action (calculator)" in html
    assert FAKE_KEY not in html
    assert "&lt;REDACTED:openai_key&gt;" in html


def test_a_card_for_a_run_with_no_conversation_has_no_empty_section() -> None:
    """A stored history record carries no steps and gets no heading."""
    from effgen.ui.report_html import build_html_report

    html = build_html_report(
        {"task": "t", "output": "x", "success": True, "execution_tree": {}},
        kind="run",
    )
    assert "<h2>Conversation</h2>" not in html


def test_the_dashboard_draws_a_stored_run_as_its_steps() -> None:
    """History keeps the step kinds, and the drill-in renders them in order."""
    from effgen.dashboard import STATIC_DIR

    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "renderRunSteps" in js
    assert "thread_kinds" in js
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    assert ".run-steps" in css


def test_the_stored_record_a_dashboard_reads_carries_those_kinds() -> None:
    """The shape the drill-in draws is the shape the run actually had."""
    from effgen.observability.run_log import _thread_shape

    shape = _thread_shape(_run().thread)
    assert shape["thread_kinds"][:2] == ["system", "task"]
    assert shape["thread_steps"] == len(shape["thread_kinds"])
