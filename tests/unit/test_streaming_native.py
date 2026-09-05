"""The streaming tool loop that dispatches a provider's native tool calls.

The model here is scripted in-process — a turn is a list of text deltas plus the
tool calls the adapter records for it — so the loop's decisions, event order and
reconstructed record are exercised without a network call. Live behaviour is
proven separately against real providers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models._usage import tool_call_entry
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
    clear_stream_tool_calls,
    record_stream_tool_calls,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)


class _Calculator(BaseTool):
    """A tool that multiplies two numbers, and counts how often it ran."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="calculator",
                description="Evaluate an arithmetic expression",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="expression",
                        type=ParameterType.STRING,
                        description="The expression to evaluate",
                        required=True,
                    )
                ],
            )
        )
        self.calls: list[str] = []

    async def _execute(self, **kwargs) -> str:
        expression = str(kwargs.get("expression", ""))
        self.calls.append(expression)
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - test tool


def _turn(text: str = "", calls: list[tuple[str, dict]] | None = None,
          raise_exc: Exception | None = None, raise_at: int = 1) -> dict:
    """One scripted model turn: its text deltas and the calls it declares."""
    return {
        "deltas": [text[i:i + 4] for i in range(0, len(text), 4)],
        "calls": calls or [],
        "raise": raise_exc,
        "raise_at": raise_at,
    }


class _ScriptedModel(BaseModel):
    """Replays scripted turns, recording tool calls the way an adapter does."""

    def __init__(self, turns: list[dict], *, streams_calls: bool = True,
                 supports_tools: bool = True) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self.turns = turns
        self._streams_calls = streams_calls
        self._supports_tools = supports_tools
        self.prompts: list[str] = []
        self.tool_kwargs: list[object] = []
        self._index = 0

    # -- capabilities --------------------------------------------------
    def supports_tool_calling(self) -> bool:
        return self._supports_tools

    def streams_tool_calls(self) -> bool:
        return self._streams_calls

    # -- generation ----------------------------------------------------
    def _next_turn(self, prompt: str, tools) -> dict:
        self.prompts.append(prompt)
        self.tool_kwargs.append(tools)
        turn = self.turns[min(self._index, len(self.turns) - 1)]
        self._index += 1
        return turn

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        turn = self._next_turn(prompt, kwargs.get("tools"))
        if turn["raise"] is not None:
            raise turn["raise"]
        entries = [tool_call_entry(n, json.dumps(a)) for n, a in turn["calls"]]
        return GenerationResult(
            text="".join(turn["deltas"]),
            tokens_used=7,
            finish_reason="tool_calls" if entries else "stop",
            model_name=self.model_name,
            metadata={"tool_calls": entries},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        turn = self._next_turn(prompt, kwargs.get("tools"))
        clear_stream_tool_calls(self)
        if self._streams_calls and turn["calls"]:
            # A real adapter records the call from its first delta, before any
            # content arrives, which is what the answer-commit rule reads.
            record_stream_tool_calls(
                self, [tool_call_entry(n, json.dumps(a)) for n, a in turn["calls"]]
            )
        for i, delta in enumerate(turn["deltas"]):
            if turn["raise"] is not None and i == turn["raise_at"]:
                raise turn["raise"]
            yield delta
        if turn["raise"] is not None and not turn["deltas"]:
            raise turn["raise"]

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=max(1, len(text) // 4), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 8192

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False


def _agent(model, *, tools=None, max_iterations=6, **cfg) -> Agent:
    return Agent(config=AgentConfig(
        name="native-stream-test",
        model=model,
        tools=tools if tools is not None else [_Calculator()],
        tool_calling_mode="native",
        max_iterations=max_iterations,
        **cfg,
    ))


TASK = "Work out 4817 * 236 with the calculator and explain the result."


def _events(agent, task=TASK):
    return list(agent.stream(task, include_events=True))


def _answers(events):
    return [e.text for e in events if e.kind == "answer"]


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def test_a_model_that_streams_tool_calls_takes_the_native_path():
    agent = _agent(_ScriptedModel([_turn()]))
    assert agent._can_stream_native_tools() is True


@pytest.mark.parametrize(
    ("kwargs", "cfg"),
    [
        ({"streams_calls": False}, {}),
        ({"supports_tools": False}, {}),
    ],
)
def test_a_model_without_the_capability_keeps_the_existing_path(kwargs, cfg):
    agent = _agent(_ScriptedModel([_turn()], **kwargs), **cfg)
    assert agent._can_stream_native_tools() is False


def test_a_custom_prompt_template_keeps_the_existing_path():
    agent = _agent(
        _ScriptedModel([_turn()]),
        system_prompt_template="{tools_description}{task}{scratchpad}{conversation_history}",
    )
    assert agent._can_stream_native_tools() is False


def test_an_agent_without_tools_keeps_the_direct_path():
    agent = _agent(_ScriptedModel([_turn()]), tools=[])
    assert agent._can_stream_native_tools() is False


def test_the_react_stream_still_runs_for_a_non_capable_model():
    model = _ScriptedModel(
        [_turn("Thought: I can do this\nFinal Answer: 42")], streams_calls=False
    )
    agent = _agent(model)
    assert "".join(agent.stream("What is 6*7?")) == "42"
    assert agent.last_stream_response is None


# --------------------------------------------------------------------------- #
# Incremental answers
# --------------------------------------------------------------------------- #
def _tool_then_answer(answer: str = "The product is 1136812, computed with the tool."):
    return _ScriptedModel([
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn(answer),
    ])


def test_a_tool_turn_then_an_answer_streams_more_than_one_delta():
    agent = _agent(_tool_then_answer())
    events = _events(agent)
    assert len(_answers(events)) > 1


def test_joining_the_answer_events_reproduces_the_reconstructed_output():
    agent = _agent(_tool_then_answer())
    events = _events(agent)
    assert "".join(_answers(events)) == agent.last_stream_response.output


def test_text_mode_joins_to_the_same_answer():
    agent = _agent(_tool_then_answer())
    text = "".join(agent.stream(TASK))
    assert text == agent.last_stream_response.output
    assert "Action:" not in text and "Observation:" not in text


def test_the_usage_event_is_last_and_keeps_its_keys():
    agent = _agent(_tool_then_answer())
    events = _events(agent)
    assert events[-1].kind == "usage"
    usage = events[-1].usage
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
                "latency_ms", "ttft_ms", "model_calls", "estimated"):
        assert key in usage
    assert agent.last_stream_usage == usage


def test_the_first_answer_delta_lands_before_the_stream_ends():
    agent = _agent(_tool_then_answer())
    _events(agent)
    usage = agent.last_stream_usage
    assert usage["ttft_ms"] is not None
    assert usage["ttft_ms"] < usage["latency_ms"]


def test_the_event_order_is_thought_tool_call_observation_then_answer():
    model = _ScriptedModel([
        _turn("Let me multiply those.", [("calculator", {"expression": "4817*236"})]),
        _turn("The product is 1136812."),
    ])
    agent = _agent(model)
    kinds = [e.kind for e in _events(agent)]
    assert kinds[:3] == ["thought", "tool_call", "observation"]
    assert "answer" in kinds
    assert kinds[-1] == "usage"


def test_a_tool_turns_text_is_a_thought_and_never_part_of_the_answer():
    model = _ScriptedModel([
        _turn("I should use the calculator here.",
              [("calculator", {"expression": "4817*236"})]),
        _turn("The product is 1136812."),
    ])
    agent = _agent(model)
    events = _events(agent)
    thoughts = [e.text for e in events if e.kind == "thought"]
    assert thoughts == ["I should use the calculator here."]
    assert "calculator here" not in "".join(_answers(events))


def test_text_committed_before_a_later_call_stays_in_the_answer():
    # A turn that answers and then declares a call: the text is already on
    # screen, so it stays part of the answer and the call is still dispatched.
    model = _ScriptedModel([
        _turn("Working on it. ", []),
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("The product is 1136812."),
    ])
    agent = _agent(model)
    events = _events(agent)
    joined = "".join(_answers(events))
    assert joined.startswith("Working on it.")
    assert joined == agent.last_stream_response.output


# --------------------------------------------------------------------------- #
# The reconstructed record
# --------------------------------------------------------------------------- #
def test_the_record_reports_iterations_tool_calls_and_the_reason():
    agent = _agent(_tool_then_answer())
    _events(agent)
    response = agent.last_stream_response
    assert response.success is True
    assert response.iterations == 2
    assert response.tool_calls == 1
    assert response.metadata["reason"] == "final_answer"
    assert response.metadata["tool_calling_strategy"] == "native"
    assert response.execution_time > 0


def test_the_record_feeds_the_coding_engine_the_same_shape_a_run_does(tmp_path):
    from effgen.cli.code.engine import CodeEngine
    from effgen.cli.code.permissions import PermissionMode

    agent = _agent(_tool_then_answer())
    _events(agent)
    streamed = agent.last_stream_response

    blocking = _agent(_tool_then_answer()).run(TASK)

    engine = CodeEngine(
        model="scripted", workspace=tmp_path, mode=PermissionMode.PLAN
    )
    a = engine.result_from_response("task", streamed).to_dict()
    b = engine.result_from_response("task", blocking).to_dict()
    assert set(a) == set(b)
    # Same key, same type wherever both carry a value. ``cost_usd`` is the one
    # field a run legitimately leaves unset (an unpriced model), so it is
    # compared as the nullable float it is declared to be.
    for key in a:
        if a[key] is None or b[key] is None:
            assert isinstance(a[key], type(None) | float | str | int | list | dict)
            continue
        assert type(a[key]) is type(b[key]), key
    assert isinstance(a["cost_usd"], float | type(None))
    assert isinstance(b["cost_usd"], float | type(None))


def test_a_stream_that_did_not_take_the_native_path_leaves_no_record():
    model = _ScriptedModel([_turn("Final Answer: 42")], streams_calls=False)
    agent = _agent(model)
    list(agent.stream("What is 6*7?"))
    assert agent.last_stream_response is None


# --------------------------------------------------------------------------- #
# Failure envelopes
# --------------------------------------------------------------------------- #
def test_the_iteration_cap_reports_the_typed_outcome():
    model = _ScriptedModel([
        _turn("", [("calculator", {"expression": "1+1"})]),
        _turn("", [("calculator", {"expression": "2+2"})]),
        _turn("", [("calculator", {"expression": "3+3"})]),
        _turn("", [("calculator", {"expression": "4+4"})]),
        _turn("", [("calculator", {"expression": "5+5"})]),
        _turn("", [("calculator", {"expression": "6+6"})]),
    ])
    agent = _agent(model, max_iterations=3)
    events = _events(agent)
    response = agent.last_stream_response
    assert response.success is False
    assert response.metadata["reason"].startswith("max_iterations")
    assert response.metadata["error"]["category"] == "max_iterations"
    assert events[-2].kind == "status"
    assert events[-2].text == response.output
    assert not _answers(events)


def test_a_written_out_tool_call_is_reported_not_answered():
    model = _ScriptedModel([
        _turn('<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'),
        _turn('<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'),
    ])
    agent = _agent(model, max_iterations=4)
    _events(agent)
    response = agent.last_stream_response
    assert response.success is False
    assert response.metadata["reason"] == "written_tool_call"
    assert response.metadata["error"]["tool"] == "calculator"


def test_a_failure_before_the_first_delta_falls_back_to_the_blocking_path():
    boom = RuntimeError("provider rejected the tool payload")
    model = _ScriptedModel([
        _turn("", raise_exc=boom),          # the streamed turn fails
        _turn("Recovered answer: 1136812"),  # the retried blocking turn answers
    ])
    agent = _agent(model)
    events = _events(agent)
    assert "".join(_answers(events)) == "Recovered answer: 1136812"
    assert agent.last_stream_response.success is True
    assert agent.last_stream_response.metadata.get("stream_fallback") is True


def test_a_refusal_raised_when_the_stream_is_opened_falls_back_too():
    """A pre-call refusal is a failure before the first delta, like any other.

    ``BaseModel`` wraps every ``generate_stream`` with the budget gate, which
    refuses synchronously when the caller opens the stream rather than on the
    first token — so the refusal arrives while the iterator is being built, not
    while it is being read. It still has to reach the fallback and leave a
    typed record behind.
    """
    model = _tool_then_answer()
    streamed = model.generate_stream
    state = {"refused": False}

    def _refusing_to_open(prompt, config=None, **kwargs):
        # A plain function, not a generator: it raises when it is *called*.
        if not state["refused"]:
            state["refused"] = True
            raise RuntimeError("run budget exhausted")
        return streamed(prompt, config=config, **kwargs)

    model.generate_stream = _refusing_to_open
    agent = _agent(model)
    events = _events(agent)

    assert state["refused"]
    response = agent.last_stream_response
    assert response is not None, "the refusal left no record of the turn"
    assert "".join(_answers(events)) == response.output


def test_a_failure_after_a_delta_is_raised():
    boom = RuntimeError("stream died mid-answer")
    model = _ScriptedModel(
        [_turn("Here is the answer so far", raise_exc=boom, raise_at=4)]
    )
    agent = _agent(model)
    with pytest.raises(RuntimeError, match="stream died"):
        list(agent.stream(TASK, include_events=True))


# --------------------------------------------------------------------------- #
# The shared guards
# --------------------------------------------------------------------------- #
def test_the_same_call_repeated_does_not_run_to_the_iteration_cap():
    """Without the shared guards this loops until max_iterations."""
    tool = _Calculator()
    model = _ScriptedModel([
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("", [("calculator", {"expression": "4817*236"})]),
    ])
    agent = _agent(model, tools=[tool], max_iterations=6)
    events = _events(agent)
    response = agent.last_stream_response
    assert response.iterations < 6
    assert len(tool.calls) == 1
    # The run stopped without an answer, so nothing is streamed as an answer:
    # the outcome arrives as the status event the iteration cap also uses.
    assert not _answers(events)
    assert response.outcome == "stopped"
    assert events[-2].kind == "status"
    assert events[-2].text == response.output


def test_several_calls_in_one_turn_are_dispatched_as_a_batch():
    tool = _Calculator()
    model = _ScriptedModel([
        _turn("", [("calculator", {"expression": "2*3"}),
                   ("calculator", {"expression": "4*5"})]),
        _turn("Six and twenty."),
    ])
    agent = _agent(model, tools=[tool])
    events = _events(agent)
    assert tool.calls == ["2*3", "4*5"]
    assert [e.kind for e in events].count("tool_call") == 2
    assert agent.last_stream_response.tool_calls == 2


def test_an_unknown_tool_gets_the_same_observation_the_run_loop_gives():
    model = _ScriptedModel([
        _turn("", [("nosuchtool", {"x": 1})]),
        _turn("I could not use that tool."),
    ])
    agent = _agent(model)
    events = _events(agent)
    observation = next(e for e in events if e.kind == "observation")
    assert "nosuchtool" in observation.text
    assert "calculator" in observation.text


# --------------------------------------------------------------------------- #
# Prompt parity with the blocking loop
# --------------------------------------------------------------------------- #
def test_both_loops_build_the_same_prompts_for_the_same_turns():
    task = TASK
    script = [
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("The product is 1136812."),
    ]

    streamed_model = _ScriptedModel([dict(t) for t in script])
    list(_agent(streamed_model).stream(task, include_events=True))

    blocking_model = _ScriptedModel([dict(t) for t in script])
    _agent(blocking_model).run(task)

    assert streamed_model.prompts == blocking_model.prompts
    assert streamed_model.tool_kwargs == blocking_model.tool_kwargs


def test_both_loops_replay_a_repeated_call_instead_of_ending_the_run():
    """An exact repeat is answered from the record on the streaming path too.

    The two loops share the guards, so a model that restates its call must
    reach the same answer with the same number of dispatches either way.
    """
    script = [
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("", [("calculator", {"expression": "4817*236"})]),
        _turn("The product is 1136812."),
    ]

    streamed_tool = _Calculator()
    streamed_agent = _agent(
        _ScriptedModel([dict(t) for t in script]), tools=[streamed_tool]
    )
    list(streamed_agent.stream(TASK, include_events=True))
    streamed = streamed_agent.last_stream_response

    blocking_tool = _Calculator()
    blocking = _agent(
        _ScriptedModel([dict(t) for t in script]), tools=[blocking_tool]
    ).run(TASK)

    assert streamed.outcome == blocking.outcome == "answered"
    assert "1136812" in streamed.output and "1136812" in blocking.output
    # One dispatch on each path: the repeat came from the record.
    assert streamed_tool.calls == blocking_tool.calls == ["4817*236"]


@pytest.mark.parametrize(
    ("script", "max_iterations", "reason"),
    [
        (
            [_turn("", [("calculator", {"expression": "4817*236"})])] * 4,
            6, "loop_detected",
        ),
        (
            [
                _turn("", [("calculator", {"expression": "1+1"})]),
                _turn("", [("calculator", {"expression": "2+2"})]),
                _turn("", [("calculator", {"expression": "3+3"})]),
            ],
            2, "max_iterations_partial",
        ),
    ],
    ids=["loop_detected", "max_iterations_partial"],
)
def test_both_loops_report_the_same_stopped_record(script, max_iterations, reason):
    """The same scripted turns end the same way whether streamed or not."""
    streamed_agent = _agent(
        _ScriptedModel([dict(t) for t in script]), max_iterations=max_iterations
    )
    list(streamed_agent.stream(TASK, include_events=True))
    streamed = streamed_agent.last_stream_response

    blocking = _agent(
        _ScriptedModel([dict(t) for t in script]),
        max_iterations=max_iterations, raise_on_error=False,
    ).run(TASK)

    assert streamed.stop_reason == blocking.stop_reason == reason
    assert streamed.outcome == blocking.outcome == "stopped"
    assert streamed.output == blocking.output
    assert streamed.partial is not None and blocking.partial is not None
    assert streamed.partial.text == blocking.partial.text
    assert streamed.partial.observations == blocking.partial.observations
    assert streamed.metadata["partial_output"] == blocking.metadata["partial_output"]


# --------------------------------------------------------------------------- #
# The answer emitter: what may be shown, and when
# --------------------------------------------------------------------------- #
from effgen.core.agent_runtime import sanitize_final_answer  # noqa: E402
from effgen.core.agent_stream_native import _AnswerStream  # noqa: E402


def _drive(text: str, chunk: int) -> tuple[list[str], _AnswerStream]:
    stream = _AnswerStream()
    deltas = [d for i in range(0, len(text), chunk)
              if (d := stream.push(text[i:i + chunk]))]
    tail = stream.flush()
    if tail:
        deltas.append(tail)
    return deltas, stream


@pytest.mark.parametrize("chunk", [1, 4, 11])
@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("plain prose", "The product is 1,136,812. That is the line total to expect."),
        ("leading label", "Final Answer: 1,136,812\n\nCheck the unit price and the count."),
        ("answer label", "Answer: forty two, give or take a little rounding."),
        ("double spaces", "One  two   three. Four  five."),
        ("markdown list", "Here:\n- first item\n- second item\n"),
    ],
)
def test_the_emitted_deltas_reconstruct_the_sanitized_answer(name, text, chunk):
    deltas, stream = _drive(text, chunk)
    assert "".join(deltas) == stream.emitted
    assert "".join(deltas) == (sanitize_final_answer(text) or "")
    assert not stream.diverged


@pytest.mark.parametrize("chunk", [1, 4])
def test_an_answer_label_does_not_reach_the_screen(chunk):
    """A label being typed out must never be shown and then withdrawn."""
    deltas, _ = _drive("Final Answer: 1,136,812 exactly.", chunk)
    assert deltas, "the answer after the label should still stream"
    assert not any("Final" in d for d in deltas)
    assert "".join(deltas) == "1,136,812 exactly."


@pytest.mark.parametrize("chunk", [1, 4, 40])
def test_written_out_call_syntax_never_streams(chunk):
    """Sanitizing cuts a written-out call from the middle, so nothing streams.

    The text is still delivered — once, at the end, cleaned — which is what a
    turn that was not streamed produces.
    """
    text = '<function=file_operations>{"path": "hello.py"}</function> Wrote the file.'
    deltas, stream = _drive(text, chunk)
    assert len(deltas) <= 1
    assert "".join(deltas) == (sanitize_final_answer(text) or "")
    assert "function=" not in "".join(deltas)


def test_text_already_shown_is_never_contradicted():
    """A label arriving mid-answer cannot retract what is on screen.

    The emitted text stays the answer of record, so joining the deltas still
    reproduces it — the invariant a consumer relies on.
    """
    deltas, stream = _drive("Some working out here.\nFinal Answer: 42", 4)
    assert stream.diverged
    assert "".join(deltas) == stream.emitted
    assert "42" not in "".join(deltas)


def test_the_last_word_is_held_until_more_text_follows():
    """A word can still grow, and sanitizing collapses the space after it."""
    stream = _AnswerStream()
    assert stream.push("Hello") == ""
    assert stream.push(" wor") == "Hello "
    assert stream.push("ld.") == ""
    assert stream.flush() == "world."


def test_the_streamed_turn_sends_the_agents_sampling_settings():
    """A configured sampling value must reach the model on either path.

    The blocking loop reads each one from the agent's config; a streamed turn
    that quietly used a different default would answer differently for the same
    agent, which is the drift the shared prompt and guards exist to prevent.
    """
    captured: list[object] = []
    model = _tool_then_answer()
    original = model.generate_stream

    def _spy(prompt, config=None, **kwargs):
        captured.append(config)
        return original(prompt, config=config, **kwargs)

    model.generate_stream = _spy
    agent = _agent(
        model, temperature=0.11, top_p=0.42, top_k=7, seed=1234,
        presence_penalty=0.25, frequency_penalty=0.75, repetition_penalty=1.15,
    )
    _events(agent)

    assert captured
    for config in captured:
        assert config.temperature == 0.11
        assert config.top_p == 0.42
        assert config.top_k == 7
        assert config.seed == 1234
        assert config.presence_penalty == 0.25
        assert config.frequency_penalty == 0.75
        assert config.repetition_penalty == 1.15


def test_a_per_call_sampling_override_beats_the_configured_value():
    captured: list[object] = []
    model = _tool_then_answer()
    original = model.generate_stream

    def _spy(prompt, config=None, **kwargs):
        captured.append(config)
        return original(prompt, config=config, **kwargs)

    model.generate_stream = _spy
    agent = _agent(model, top_p=0.42)
    list(agent.stream(TASK, include_events=True, top_p=0.9))

    assert captured and all(config.top_p == 0.9 for config in captured)


@pytest.mark.parametrize(
    "opener",
    ["<tool_call>", "<function=calculator>", '<function name="calculator">',
     '<invoke name="calculator">'],
    ids=["tool_call", "function-eq", "function-attr", "invoke-attr"],
)
def test_a_turn_writing_an_xml_call_streams_nothing_until_it_is_cleaned(opener):
    """A call written as nested tags is held back like any other construct.

    Sanitizing removes the construct from the middle of the text, so a delta
    emitted before the turn ends could not be extended — it would put raw
    scaffolding on screen and then have to take it back.
    """
    from effgen.core.agent_stream_native import _AnswerStream

    stream = _AnswerStream()
    assert stream.push("The product is ") != ""
    assert stream.push(opener) == ""
    assert stream.push("<parameter=expression>4817 * 236</parameter>") == ""
    tail = stream.flush()
    assert "<" not in stream.emitted + tail
    assert "parameter" not in stream.emitted + tail
