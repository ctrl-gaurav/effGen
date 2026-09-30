"""One agent object serving many conversations at the same time.

``run(..., session=...)`` and ``stream(..., session=...)`` promise that each
conversation sees only its own history and that each turn is recorded in the
conversation it belongs to — also when the calls overlap on one agent, from
many threads, from one event loop, or from streams advanced in turn by one
thread. Each conversation says a nonce of its own per turn and the scripted
model answers with every nonce its prompt showed it, so an answer naming
another conversation's nonce is a leak, and a missing own nonce is a lost turn.

Overlap is forced: every turn starts behind a barrier and the model holds each
call for a few milliseconds. The switch interval is lowered only after the
threads have started. Each scenario repeats, so a clean result is not luck.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from effgen import Agent, AgentConfig
from effgen.core.middleware import AgentMiddleware
from effgen.core.session import Session
from effgen.models.base import GenerationResult
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from tests.fixtures.mock_models import MockModel

NONCE = re.compile(r"NONCE-([0-9a-f]{2})([0-9a-f]{4})")
THREADS = 8
TURNS = 4
REPS = 3  # each concurrent scenario runs this many times


def _text(prompt: Any) -> str:
    return prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)


class EchoModel(MockModel):
    """Answers with every nonce its prompt carried, in order, once each.

    With ``tools`` it first calls the ``echo`` tool with the last nonce it was
    shown, and answers once the prompt carries the tool's result.
    """

    def __init__(self, *, tools: bool = False, delay: float = 0.003) -> None:
        super().__init__(["unused"], model_name="echo-model")
        self.tools = tools
        self.delay = delay
        self._lock = threading.Lock()

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        text = _text(prompt)
        time.sleep(self.delay)
        seen = list(dict.fromkeys(m.group(0) for m in NONCE.finditer(text)))
        if self.tools and "echoed NONCE" not in text and seen:
            reply = (
                "Thought: I will echo it.\nAction: echo\n"
                f'Action Input: {{"text": "{seen[-1]}"}}'
            )
        else:
            reply = "Final Answer: DONE " + " ".join(seen)
        with self._lock:
            self._generate_calls.append({"prompt": prompt, "config": config})
        return GenerationResult(
            text=reply,
            tokens_used=len(reply.split()),
            finish_reason="stop",
            model_name=self.model_name,
            metadata={"prompt_tokens": len(text) // 4, "completion_tokens": 4},
        )


class Echo(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="echo", description="Echo the given text back.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(name="text", type=ParameterType.STRING,
                                      description="The text to echo.", required=True)]))

    async def _execute(self, **kw: Any) -> str:
        return f"echoed {kw.get('text')}"


def _agent(*, tools: bool = False, **config: Any) -> Agent:
    return Agent(AgentConfig(
        name="shared", model=EchoModel(tools=tools), tools=[Echo()] if tools else [],
        max_iterations=6, raise_on_error=False, enable_sub_agents=False, **config,
    ))


def _task(conv: int, turn: int) -> str:
    return f"Turn {turn}: remember NONCE-{conv:02x}{turn:04x}."


@pytest.fixture(autouse=True)
def _sessions_dir(tmp_path, monkeypatch):
    """Keep saved sessions out of the developer's real session store."""
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path))
    return tmp_path


class Tally:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.answers = 0
        self.foreign: list[str] = []
        self.lost: list[str] = []
        self.errors: list[str] = []

    def check(self, conv: int, turn: int, text: str) -> None:
        owners = [int(m.group(1), 16) for m in NONCE.finditer(text)]
        with self.lock:
            self.answers += 1
            if any(o != conv for o in owners):
                self.foreign.append(f"conversation {conv} turn {turn}: {text[:120]}")
            if owners.count(conv) != turn + 1:
                self.lost.append(f"conversation {conv} turn {turn}: {text[:120]}")

    def assert_clean(self, expected: int) -> None:
        assert not self.errors, self.errors[:3]
        assert not self.foreign, self.foreign[:3]
        assert not self.lost, self.lost[:3]
        assert self.answers == expected


def _assert_sessions_hold_their_own(sessions: dict[int, Session], turns: int) -> None:
    for conv, session in sessions.items():
        users = [m for m in session.messages if m["role"] == "user"]
        owners = [int(NONCE.search(m["content"]).group(1), 16) for m in users]
        assert owners == [conv] * turns, (conv, owners)


def _threads(target, n: int) -> None:
    ts = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    previous = sys.getswitchinterval()
    for t in ts:
        t.start()
    sys.setswitchinterval(1e-5)  # only after the threads are running
    try:
        for t in ts:
            t.join(timeout=120)
    finally:
        sys.setswitchinterval(previous)


def _concurrent(agent: Agent, call) -> tuple[Tally, dict[int, Session]]:
    tally = Tally()
    sessions = {i: Session(session_id=f"conv-{i:02x}") for i in range(THREADS)}
    barrier = threading.Barrier(THREADS)

    def talk(i: int) -> None:
        try:
            for turn in range(TURNS):
                barrier.wait(timeout=60)
                tally.check(i, turn, call(agent, _task(i, turn), sessions[i]))
        except Exception as e:  # noqa: BLE001 - reported by assert_clean
            with tally.lock:
                tally.errors.append(f"{type(e).__name__}: {e}")
            barrier.abort()

    _threads(talk, THREADS)
    return tally, sessions


def _blocking(agent: Agent, task: str, session: Session) -> str:
    return str(agent.run(task, session=session).output)


def _streamed(agent: Agent, task: str, session: Session) -> str:
    return "".join(agent.stream(task, session=session))


@pytest.mark.parametrize("tools", [False, True], ids=["no-tools", "tool-loop"])
@pytest.mark.parametrize("call", [_blocking, _streamed], ids=["run", "stream"])
@pytest.mark.parametrize("rep", range(REPS))
def test_overlapping_calls_keep_each_conversation_apart(call, tools, rep):
    with _agent(tools=tools) as agent:
        tally, sessions = _concurrent(agent, call)
        tally.assert_clean(THREADS * TURNS)
        _assert_sessions_hold_their_own(sessions, TURNS)
        assert agent.session is None
        assert agent.short_term_memory.get_recent_messages(n=100) == []


@pytest.mark.parametrize("tools", [False, True], ids=["no-tools", "tool-loop"])
@pytest.mark.parametrize("rep", range(REPS))
def test_overlapping_async_calls_keep_each_conversation_apart(tools, rep):
    with _agent(tools=tools) as agent:
        tally = Tally()
        sessions = {i: Session(session_id=f"async-{i:02x}") for i in range(THREADS)}

        async def main() -> None:
            barrier = asyncio.Barrier(THREADS)

            async def talk(i: int) -> None:
                for turn in range(TURNS):
                    await barrier.wait()
                    r = await agent.run_async(_task(i, turn), session=sessions[i])
                    tally.check(i, turn, str(r.output))

            await asyncio.gather(*(talk(i) for i in range(THREADS)))

        asyncio.run(main())
        tally.assert_clean(THREADS * TURNS)
        _assert_sessions_hold_their_own(sessions, TURNS)
        assert agent.short_term_memory.get_recent_messages(n=100) == []


@pytest.mark.parametrize("tools", [False, True], ids=["no-tools", "tool-loop"])
def test_streams_advanced_in_turn_by_one_thread_stay_apart(tools):
    with _agent(tools=tools) as agent:
        tally = Tally()
        sessions = {i: Session(session_id=f"rr-{i:02x}") for i in range(THREADS)}
        for turn in range(TURNS):
            streams = {i: agent.stream(_task(i, turn), session=sessions[i]) for i in range(THREADS)}
            pieces: dict[int, list[str]] = {i: [] for i in range(THREADS)}
            while streams:
                for i in list(streams):
                    try:
                        pieces[i].append(next(streams[i]))
                    except StopIteration:
                        del streams[i]
            for i in range(THREADS):
                tally.check(i, turn, "".join(pieces[i]))
        tally.assert_clean(THREADS * TURNS)
        _assert_sessions_hold_their_own(sessions, TURNS)
        assert agent.short_term_memory.get_recent_messages(n=100) == []


def test_a_streamed_turn_is_recorded_in_the_session_it_was_given(_sessions_dir):
    with _agent() as agent:
        session = Session(session_id="streamed")
        first = "".join(agent.stream(_task(1, 0), session=session))
        second = "".join(agent.stream(_task(1, 1), session=session))
        assert "NONCE-010000" in first
        assert "NONCE-010000" in second and "NONCE-010001" in second
        assert [m["role"] for m in session.messages] == ["user", "assistant"] * 2
        assert Session.load("streamed", str(_sessions_dir)).messages[-1]["role"] == "assistant"
        assert agent.short_term_memory.get_recent_messages(n=100) == []


def test_calls_without_a_session_never_cross_with_calls_that_have_one():
    with _agent() as agent:
        tally = Tally()
        sessions = {i: Session(session_id=f"mixed-{i:02x}") for i in range(4)}
        plain_seen: list[int] = []
        barrier = threading.Barrier(6)

        def talk(i: int) -> None:
            try:
                for turn in range(TURNS):
                    barrier.wait(timeout=60)
                    if i < 4:
                        tally.check(i, turn, str(agent.run(_task(i, turn), session=sessions[i]).output))
                    else:
                        out = str(agent.run(_task(0xE0 + i, turn)).output)
                        with tally.lock:
                            plain_seen.extend(int(m.group(1), 16) for m in NONCE.finditer(out))
            except Exception as e:  # noqa: BLE001
                with tally.lock:
                    tally.errors.append(f"{type(e).__name__}: {e}")
                barrier.abort()

        _threads(talk, 6)
        tally.assert_clean(4 * TURNS)
        _assert_sessions_hold_their_own(sessions, TURNS)
        assert all(o >= 0xE0 for o in plain_seen), sorted(set(plain_seen))
        own = " ".join(str(m.content) for m in agent.short_term_memory.get_recent_messages(n=100))
        assert all(int(m.group(1), 16) >= 0xE0 for m in NONCE.finditer(own))


class _Recorder(AgentMiddleware):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.lock = threading.Lock()

    def before_model_call(self, ctx: Any) -> None:
        with self.lock:
            self.prompts.append(_text(ctx.prompt))
        return None


@pytest.mark.parametrize("rep", range(REPS))
def test_per_call_middleware_sees_only_its_own_call(rep):
    with _agent() as agent:
        recorder = _Recorder()
        barrier = threading.Barrier(2)
        errors: list[str] = []
        sessions = {i: Session(session_id=f"mw-{i}") for i in range(2)}

        def talk(i: int) -> None:
            try:
                for turn in range(TURNS):
                    barrier.wait(timeout=60)
                    if i == 0:
                        agent.run(_task(0, turn), session=sessions[0], middleware=[recorder])
                    else:
                        agent.run(_task(1, turn), session=sessions[1])
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
                barrier.abort()

        _threads(talk, 2)
        assert not errors
        owners = {int(m.group(1), 16) for p in recorder.prompts for m in NONCE.finditer(p)}
        assert 1 not in owners, "the hook ran on another call's model request"
        assert len(recorder.prompts) == TURNS, "the hook missed one of its own call's requests"
        assert agent._active_middleware is None


@pytest.mark.parametrize("rep", range(REPS))
def test_a_run_s_trace_never_holds_a_concurrent_stream_s_steps(rep):
    with _agent(tools=True) as agent:
        barrier = threading.Barrier(2)
        leaked: list[str] = []
        errors: list[str] = []

        def talk(i: int) -> None:
            try:
                for turn in range(TURNS):
                    barrier.wait(timeout=60)
                    if i == 0:
                        "".join(agent.stream(_task(0, turn)))
                    else:
                        r = agent.run(_task(1, turn))
                        trace = json.dumps(r.execution_trace, default=str)
                        if "NONCE-00" in trace:
                            leaked.append(trace[:200])
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
                barrier.abort()

        _threads(talk, 2)
        assert not errors
        assert not leaked, leaked[:2]
        assert agent._active_run_count == 0


def test_a_stream_keeps_its_cost_and_citations_to_itself():
    with _agent(tools=True) as agent:
        "".join(agent.stream(_task(0, 0)))
        "".join(agent.stream(_task(0, 1)))
        # Outside any call the agent holds no run's tallies.
        assert agent._run_cost_accum == {}
        assert agent._collected_citations == []
        assert agent._active_run_count == 0


def test_an_abandoned_stream_gives_back_its_claim_on_the_agent():
    with _agent(tools=True) as agent:
        stream = agent.stream(_task(0, 0), session=Session(session_id="abandoned"))
        next(stream)
        assert agent._active_run_count == 1
        stream.close()
        assert agent._active_run_count == 0
        assert agent.session is None


def test_a_run_blocked_by_an_input_guardrail_leaves_the_agent_s_session_alone():
    from effgen.guardrails.base import GuardrailPosition
    from effgen.guardrails.content import LengthGuardrail

    with _agent(guardrails=[LengthGuardrail(max_length=20, positions=[GuardrailPosition.INPUT])]) as agent:
        session = Session(session_id="blocked")
        blocked = agent.run("x" * 50, session=session)
        assert blocked.success is False
        assert agent.session is None
        agent.run("short one")
        assert session.messages == []
        assert agent.short_term_memory.get_recent_messages(n=10) != []


def test_an_agent_called_from_inside_another_agent_s_session_run_reads_its_own_memory():
    inner = _agent()
    inner.run(_task(0xAA, 0))  # the inner agent's own history

    class AskInner(BaseTool):
        def __init__(self) -> None:
            super().__init__(metadata=ToolMetadata(
                name="echo", description="Ask the inner agent.",
                category=ToolCategory.DATA_PROCESSING,
                parameters=[ParameterSpec(name="text", type=ParameterType.STRING,
                                          description="What to ask.", required=True)]))

        async def _execute(self, **kw: Any) -> str:
            asked = str(kw.get("text"))
            return f"echoed {asked} :: {inner.run(asked).output}"

    outer = Agent(AgentConfig(
        name="outer", model=EchoModel(tools=True), tools=[AskInner()],
        max_iterations=6, raise_on_error=False, enable_sub_agents=False,
    ))
    try:
        session = Session(session_id="outer-conv")
        outer.run(_task(0x01, 0), session=session)
        inner_memory = " ".join(
            str(m.content) for m in inner.short_term_memory.get_recent_messages(n=100)
        )
        assert "NONCE-aa0000" in inner_memory
        assert inner.session is None
        assert [m["role"] for m in session.messages] == ["user", "assistant"]
    finally:
        outer.close()
        inner.close()


def test_the_sdk_response_types_are_built_when_the_adapter_loads():
    """The SDK's deferred model build is not thread-safe; loading builds it once."""
    code = (
        "import pydantic, typing\n"
        "from effgen.models.openai_adapter import OpenAIAdapter\n"
        "from openai.types.chat import ChatCompletion, ChatCompletionChunk\n"
        "m = OpenAIAdapter(model_name='x', api_key='EMPTY', base_url='http://127.0.0.1:9/v1',"
        " max_retries=0, timeout=1)\n"
        "m.load()\n"
        "seen, todo = set(), [ChatCompletion, ChatCompletionChunk]\n"
        "while todo:\n"
        "    t = todo.pop()\n"
        "    if isinstance(t, type) and issubclass(t, pydantic.BaseModel):\n"
        "        if t in seen: continue\n"
        "        seen.add(t); assert t.__pydantic_complete__, t\n"
        "        todo.extend(f.annotation for f in t.model_fields.values())\n"
        "    else:\n"
        "        todo.extend(typing.get_args(t))\n"
        "print('built', len(seen))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "built" in out.stdout


def test_two_runs_failing_one_tool_at_once_both_count_on_the_agent_s_breaker(monkeypatch):
    """The breaker is shared by an agent's runs by design; neither failure is lost."""
    from effgen.reliability.circuit import CircuitBreaker as Core
    from effgen.utils import circuit_breaker as cb

    class SlowToBuild(Core):
        def __init__(self, *a: Any, **kw: Any) -> None:
            time.sleep(0.05)  # both runs find no circuit before either stores one
            super().__init__(*a, **kw)

    monkeypatch.setattr(cb, "_CoreCircuitBreaker", SlowToBuild)
    breaker = cb.CircuitBreaker(failure_threshold=3)
    barrier = threading.Barrier(2)

    def fail(_: int) -> None:
        barrier.wait(timeout=30)
        breaker.record_failure("flaky")

    _threads(fail, 2)
    assert breaker._get_circuit("flaky")._consecutive_failures == 2


@pytest.mark.parametrize("attr", ["session", "short_term_memory"])
def test_the_agent_s_own_session_and_memory_still_behave_as_plain_attributes(attr):
    """Outside a call, ``session`` / ``short_term_memory`` look as they always did.

    ``vars(agent)`` lists them, ``mock.patch.object`` swaps and restores them,
    and ``del`` removes them. This guards the change above against altering
    what a single-threaded caller sees; it passes on the tree before it too.
    """
    from unittest import mock

    agent = _agent()
    original = getattr(agent, attr)
    assert attr in vars(agent)
    with mock.patch.object(agent, attr, "patched"):
        assert getattr(agent, attr) == "patched"
    assert getattr(agent, attr) is original
    delattr(agent, attr)
    with pytest.raises(AttributeError):
        getattr(agent, attr)
    setattr(agent, attr, original)
    assert getattr(agent, attr) is original


@pytest.mark.parametrize("tools", [False, True], ids=["no-tools", "tool-loop"])
def test_a_streamed_turn_is_saved_to_the_session_the_agent_is_bound_to(tools, _sessions_dir):
    """``session_id=`` binds a conversation for the agent's whole life, streams included.

    Each streamed turn reaches the session file, so a new agent bound to the
    same id continues the conversation; a stream given its own ``session=``
    records there and leaves the bound session alone.
    """
    agent = Agent(AgentConfig(
        name="bound", model=EchoModel(tools=tools), tools=[Echo()] if tools else [],
        max_iterations=6, raise_on_error=False, enable_sub_agents=False,
    ), session_id="bound-conv")
    for turn in range(2):
        "".join(str(x) for x in agent.stream(_task(1, turn)))
    saved = Session.load("bound-conv", str(_sessions_dir))
    users = [m for m in saved.messages if m.get("role") == "user"]
    assert [m["content"] for m in users] == [_task(1, 0), _task(1, 1)]

    other = Session(session_id="other-conv")
    "".join(str(x) for x in agent.stream(_task(2, 0), session=other))
    assert sum(1 for m in other.messages if m.get("role") == "user") == 1
    again = Session.load("bound-conv", str(_sessions_dir))
    assert "NONCE-02" not in json.dumps(again.messages)

    fresh = Agent(AgentConfig(
        name="bound", model=EchoModel(), max_iterations=6,
        raise_on_error=False, enable_sub_agents=False,
    ), session_id="bound-conv")
    answer = str(fresh.run(_task(1, 2)).output)
    assert "NONCE-010000" in answer and "NONCE-010001" in answer
