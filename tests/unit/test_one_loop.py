"""One loop, whichever method the caller used.

``run()`` and ``stream()`` used to be two implementations of the same ReAct
loop, and a third existed for a provider that streams its tool calls. They
shared the repeat guards and nothing else, so the same agent asked the same
model a different question depending on which method was called: a different
prompt frame, different sampling settings, a different set of safety checks and
a different terminal contract.

Every test here compares the two entry points on **one** agent definition and
one scripted model. Assertions are on the assembled request — the prompt the
model was handed and the parameters it travelled with — rather than on the
answer, because a request is exact and an answer is not.
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
    record_stream_tool_calls,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

TASK = "What is 12 times 3?"


class _Calculator(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="the expression", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        return "36"


class _Model(BaseModel):
    """A model with a declared capability, replaying a fixed script.

    ``kind`` is the capability profile: ``"text"`` cannot be handed tool
    definitions at all, ``"api"`` can be handed them but its adapter does not
    record the calls it streams, and ``"stream"`` does both. The middle one is
    the ordinary cloud model, and it is the one the two paths disagreed about.
    """

    def __init__(self, kind: str, script: list[dict]) -> None:
        super().__init__(model_name="probe-model", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.kind = kind
        self.script = script
        self.prompts: list[Any] = []
        self.configs: list[Any] = []
        self.kwargs: list[list[str]] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return self.kind in ("api", "stream")

    def tool_call_support(self) -> str:
        return "api" if self.kind in ("api", "stream") else "none"

    def streams_tool_calls(self) -> bool:
        return self.kind == "stream"

    def supports_message_protocol(self) -> bool:
        return False

    def generate(self, prompt, config=None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt)
        self.configs.append(config)
        self.kwargs.append(sorted(kwargs))
        turn = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        calls = turn.get("calls")
        return GenerationResult(
            text=turn.get("text", ""), tokens_used=7,
            finish_reason="tool_calls" if calls else "stop",
            model_name=self.model_name,
            metadata={"tool_calls": calls} if calls else {},
        )

    def generate_stream(self, prompt, config=None, **kwargs: Any):
        result = self.generate(prompt, config, **kwargs)
        calls = (result.metadata or {}).get("tool_calls") or []
        if self.kind == "stream" and calls:
            record_stream_tool_calls(self, calls)
        yield result.text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _call(name: str, arguments: dict) -> dict:
    return {"id": f"c-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


CALC_CALL = [_call("calculator", {"expression": "12*3"})]
ANSWER = "Final Answer: 36"
NATIVE_SCRIPT = [{"text": "I will multiply.", "calls": CALC_CALL}, {"text": ANSWER}]
TEXT_SCRIPT = [
    {"text": "Thought: multiply.\nAction: calculator\nAction Input: 12*3"},
    {"text": ANSWER},
]


def _agent(kind: str, script: list[dict] | None = None, **cfg: Any) -> Agent:
    options: dict[str, Any] = {
        "model": _Model(
            kind, script or (TEXT_SCRIPT if kind == "text" else NATIVE_SCRIPT)),
        "tools": [_Calculator()],
        "max_iterations": 4,
        "raise_on_error": False,
        "tool_calling_mode": "hybrid",
    }
    options.update(cfg)
    return Agent(AgentConfig(**options))


# ---------------------------------------------------------------------------
# One prompt frame (CE-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["text", "api", "stream"])
def test_both_entry_points_build_the_same_first_prompt(kind: str) -> None:
    """The frame is chosen from declared capability, not from the method called.

    A model that declares ``tool_call_support() == "api"`` but whose adapter
    does not stream its calls used to be sent a 2143-character ReAct scaffold
    by ``stream()`` and a 382-character native prompt by ``run()`` — for the
    same agent, the same task and the same model.
    """
    blocking = _agent(kind)
    blocking.run(TASK)
    streamed = _agent(kind)
    list(streamed.stream(TASK))

    assert blocking.model.prompts, "the blocking run made no model call"
    assert streamed.model.prompts, "the streamed run made no model call"
    assert blocking.model.prompts[0] == streamed.model.prompts[0]


@pytest.mark.parametrize("kind", ["text", "api", "stream"])
def test_the_tool_definitions_travel_the_same_way(kind: str) -> None:
    """Turn for turn, the definitions reach the provider on both paths or on neither."""
    blocking = _agent(kind)
    blocking.run(TASK)
    streamed = _agent(kind)
    list(streamed.stream(TASK))

    b_kwargs = blocking.model.kwargs
    s_kwargs = streamed.model.kwargs
    assert b_kwargs, "the blocking run made no model call"
    for turn, (before, after) in enumerate(zip(b_kwargs, s_kwargs)):
        assert ("tools" in before) == ("tools" in after), (
            f"turn {turn}: run() {before} vs stream() {after}"
        )


def test_a_react_frame_is_forced_on_both_paths_when_the_caller_asks() -> None:
    """``tool_calling_mode="react"`` reaches a streamed run too."""
    blocking = _agent("api", TEXT_SCRIPT, tool_calling_mode="react")
    blocking.run(TASK)
    streamed = _agent("api", TEXT_SCRIPT, tool_calling_mode="react")
    list(streamed.stream(TASK))

    assert "Action:" in blocking.model.prompts[0]
    assert blocking.model.prompts[0] == streamed.model.prompts[0]
    assert "tools" not in blocking.model.kwargs[0]
    assert "tools" not in streamed.model.kwargs[0]


# ---------------------------------------------------------------------------
# One set of sampling settings (CE-2)
# ---------------------------------------------------------------------------

_SAMPLING = {
    "temperature": 0.17, "top_p": 0.31, "top_k": 17, "seed": 4242,
    "presence_penalty": 0.5, "frequency_penalty": 0.25,
    "repetition_penalty": 1.15, "max_tokens": 333,
}
_FIELDS = (
    "temperature", "max_tokens", "top_p", "top_k", "seed",
    "presence_penalty", "frequency_penalty", "repetition_penalty",
    "stop_sequences",
)


@pytest.mark.parametrize("kind", ["text", "api", "stream"])
def test_a_configured_seed_and_nucleus_reach_a_streamed_run(kind: str) -> None:
    """All nine settings, both paths, one configuration.

    A caller who set ``top_p=0.31`` and streamed was getting the literal 0.9,
    and a pinned seed never reached the provider at all.
    """
    blocking = _agent(kind, **_SAMPLING)
    blocking.run(TASK)
    streamed = _agent(kind, **_SAMPLING)
    list(streamed.stream(TASK))

    before = blocking.model.configs[0]
    after = streamed.model.configs[0]
    assert before is not None and after is not None
    differing = [
        name for name in _FIELDS
        if getattr(before, name, None) != getattr(after, name, None)
    ]
    assert differing == [], {
        name: (getattr(before, name, None), getattr(after, name, None))
        for name in differing
    }
    assert after.top_p == 0.31 and after.seed == 4242 and after.max_tokens == 333


# ---------------------------------------------------------------------------
# One set of guards (CE-7)
# ---------------------------------------------------------------------------


def test_a_repeated_call_stops_a_streamed_run_where_it_stops_a_blocking_one() -> None:
    """The guards were built in two of the three loops and reached one path."""
    script = [{"text": "again", "calls": CALC_CALL}] * 4
    blocking = _agent("api", script, max_iterations=6)
    response = blocking.run(TASK)
    streamed = _agent("api", script, max_iterations=6)
    list(streamed.stream(TASK))
    record = streamed.last_stream_response

    assert record is not None
    assert record.stop_reason == response.stop_reason
    assert len(record.tool_calls or []) == len(response.tool_calls or [])
    assert len(blocking.model.prompts) == len(streamed.model.prompts)


def test_the_guards_are_built_at_exactly_one_place() -> None:
    """Two construction sites is what let one path run without the checks."""
    import pathlib

    import effgen.core.agent_loop as agent_loop

    sites = []
    for path in sorted(pathlib.Path(agent_loop.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "NativeToolLoop"):
                sites.append(path.stem)
                break
    assert sites == ["agent_loop"], sites


# ---------------------------------------------------------------------------
# One terminal contract (CE-3, CE-4, CE-8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["text", "api", "stream"])
def test_a_streamed_run_reports_the_outcome_a_blocking_run_reports(kind: str) -> None:
    blocking = _agent(kind)
    response = blocking.run(TASK)
    streamed = _agent(kind)
    list(streamed.stream(TASK))
    record = streamed.last_stream_response

    assert record is not None
    assert record.output == response.output
    assert record.success is response.success
    assert record.stop_reason == response.stop_reason
    assert "thread" in record.metadata
    assert record.metadata["prompt_protocol"] == response.metadata["prompt_protocol"]
    assert record.metadata["streamed"] is True


def test_a_streamed_run_stops_yielding_the_frameworks_own_bookkeeping() -> None:
    """A call written into the answer is reported, never handed over as one."""
    script = [{"text": "Final Answer: calculator(12*3)"}]
    blocking = _agent("text", script)
    response = blocking.run(TASK)
    streamed = _agent("text", script)
    chunks = list(streamed.stream(TASK))

    assert response.stop_reason == "written_tool_call"
    assert "".join(chunks) != "calculator(12*3)"
    assert "".join(chunks) == response.output
    assert streamed.last_stream_response.stop_reason == "written_tool_call"


@pytest.mark.parametrize("kind", ["text", "api", "stream"])
def test_the_two_modes_of_one_stream_carry_the_same_text(kind: str) -> None:
    """Joining the chunks and joining the events give the same words."""
    script = [{"text": "I will multiply.", "calls": CALC_CALL}]  # runs to the cap
    text_mode = "".join(_agent(kind, script, max_iterations=2).stream(TASK))
    events = list(_agent(kind, script, max_iterations=2).stream(
        TASK, include_events=True))
    from_events = "".join(
        event.text or "" for event in events
        if getattr(event, "kind", "") in ("answer", "status")
    )
    assert text_mode == from_events


# ---------------------------------------------------------------------------
# One safety pass (CE-6)
# ---------------------------------------------------------------------------


def test_a_streamed_answer_is_screened_by_the_output_guardrail() -> None:
    from effgen.guardrails.base import (
        Guardrail,
        GuardrailChain,
        GuardrailPosition,
        GuardrailResult,
    )

    seen: list[str] = []

    class _Watch(Guardrail):
        def __init__(self) -> None:
            super().__init__(name="watch", positions=[
                GuardrailPosition.INPUT, GuardrailPosition.OUTPUT])

        def check(self, content, **kwargs):  # type: ignore[override]
            seen.append(str(content))
            return GuardrailResult(passed=True, guardrail_name="watch")

    agent = _agent("text", guardrails=GuardrailChain([_Watch()]))
    list(agent.stream(TASK))
    assert seen == [TASK, "36"]


def test_a_blocking_output_guardrail_raises_from_the_iterator() -> None:
    from effgen.guardrails.base import (
        Guardrail,
        GuardrailChain,
        GuardrailPosition,
        GuardrailResult,
    )

    class _Block(Guardrail):
        def __init__(self) -> None:
            super().__init__(name="block", positions=[GuardrailPosition.OUTPUT])

        def check(self, content, **kwargs):  # type: ignore[override]
            return GuardrailResult(
                passed=False, guardrail_name="block", reason="not allowed")

    agent = _agent("text", guardrails=GuardrailChain([_Block()]))
    with pytest.raises(RuntimeError, match="blocked by guardrail"):
        list(agent.stream(TASK))


# ---------------------------------------------------------------------------
# The design's one sentence
# ---------------------------------------------------------------------------


def test_the_frame_does_not_read_whether_anything_is_streamed() -> None:
    """The rule that chooses the prompt frame reads declared capability only.

    ``streams_tool_calls()`` is a fact about the *adapter*; letting it choose
    the frame is how a model that could be handed tool definitions came to be
    prompted as though it could not.
    """
    from effgen.core.agent_loop import frame_for

    source = inspect.getsource(frame_for)
    assert "streams_tool_calls" not in source
    assert "emit_deltas" not in source

    for kind in ("text", "api", "stream"):
        blocking = _agent(kind)
        streamed = _agent(kind)
        assert frame_for(blocking) == frame_for(streamed)
    assert frame_for(_agent("api")) == "native"
    assert frame_for(_agent("stream")) == "native"
    assert frame_for(_agent("text")) == "react_text"


def test_only_one_policy_field_depends_on_the_method_called() -> None:
    """``emit_deltas`` is the whole of the difference between the two paths."""
    from effgen.core.agent_loop import _LoopPolicy

    agent = _agent("api")
    blocking = _LoopPolicy.for_run(agent, {}, emit_deltas=False)
    streamed = _LoopPolicy.for_run(agent, {}, emit_deltas=True)
    differing = [
        name for name in blocking.__dataclass_fields__
        if getattr(blocking, name) != getattr(streamed, name)
    ]
    assert differing == ["emit_deltas"], differing


def test_stream_says_once_that_it_ignores_a_mode(caplog) -> None:
    """Routing sub-agents through a streamed run is not supported yet."""
    from effgen.core.agent_config import AgentMode

    agent = _agent("text")
    with caplog.at_level("WARNING", logger="effgen.core.agent_streaming"):
        list(agent.stream(TASK, mode=AgentMode.SINGLE))
    assert "stream() ignores mode=" in caplog.text
