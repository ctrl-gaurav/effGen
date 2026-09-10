"""Unit tests for streaming correctness.

Covers:
- No-tool ``Agent.stream()`` uses the direct path (no ReAct scaffold, no
  "[Max iterations reached]"); tokens stream incrementally.
- Mid-stream provider errors are *raised* at the iterator boundary, not
  buffered into a normal chunk that looks like model output.
- The final assembled answer is sanitized before it is
  handed to ``on_answer`` and stored in short-term memory.
- The ReAct/tool path also fails explicitly (raises) on a streaming error.
- A tool agent's default stream yields only the sanitized answer text — raw
  ReAct scaffolding (Thought/Action/Observation/Final Answer) is never the
  payload; ``include_events=True`` surfaces typed step events instead.
- OpenAI-compatible streaming emits a final usage chunk when the client opts
  in via ``stream_options.include_usage``, and omits it otherwise.

These use an in-process fake model (no network) — not a mock of live API
behaviour, which the project forbids.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount


class _StreamModel(BaseModel):
    """In-process model that streams ``tokens`` and may raise after ``raise_after``."""

    def __init__(self, tokens: list[str], *, raise_after: int | None = None,
                 exc: Exception | None = None):
        super().__init__(model_name="fake-stream", model_type=ModelType.OPENAI)
        self._tokens = tokens
        self._raise_after = raise_after
        self._exc = exc or RuntimeError("provider stream blew up")
        self.stream_calls = 0
        self.last_prompt: str | None = None

    def load(self) -> None:
        pass

    def generate(self, prompt, config=None, **kwargs):
        self.last_prompt = prompt
        if self._raise_after is not None:
            raise self._exc
        return GenerationResult(
            text="".join(self._tokens), tokens_used=len(self._tokens),
            finish_reason="stop", model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        self.stream_calls += 1
        self.last_prompt = prompt
        for i, tok in enumerate(self._tokens):
            if self._raise_after is not None and i == self._raise_after:
                raise self._exc
            yield tok

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def unload(self) -> None:
        pass

    def generate_batch(self, prompts, config=None, **kwargs):
        return [self.generate(p, config) for p in prompts]


def _agent(model, tools=None):
    return Agent(config=AgentConfig(name="stream-test", model=model, tools=tools or []))


# --------------------------------------------------------------------------- #
# No-tool direct streaming
# --------------------------------------------------------------------------- #

def test_no_tool_stream_is_direct_no_scaffold():
    """A no-tool agent streams the answer directly — no ReAct bookkeeping leaks."""
    model = _StreamModel(["1", " 2", " 3", " 4", " 5"])
    out = list(_agent(model).stream("Count from 1 to 5"))
    text = "".join(out)
    assert text == "1 2 3 4 5"
    assert "[Max iterations reached]" not in text
    assert "Thought:" not in text and "Action:" not in text
    assert model.stream_calls == 1  # one direct pass, not a ReAct loop


def test_no_tool_stream_increments_token_by_token():
    """Tokens are yielded one at a time (true incrementality)."""
    model = _StreamModel(["a", "b", "c", "d"])
    chunks = list(_agent(model).stream("hi"))
    assert chunks == ["a", "b", "c", "d"]


def test_final_answer_sanitized_and_stored():
    """Assembled answer is sanitized before on_answer + memory."""
    model = _StreamModel(["Final Answer: ", "Canberra"])
    captured: list[str] = []
    agent = _agent(model)
    list(agent.stream("Capital of Australia?", on_answer=captured.append))
    assert captured == ["Canberra"]  # leading "Final Answer:" label stripped
    msgs = agent.short_term_memory.get_recent_messages()
    assert any(
        getattr(m.role, "value", m.role) == "assistant" and m.content == "Canberra"
        for m in msgs
    )


# --------------------------------------------------------------------------- #
# Mid-stream errors
# --------------------------------------------------------------------------- #

def test_mid_stream_error_raises_not_buffered_direct():
    """A provider error mid-stream is raised, never yielded as a fake chunk."""
    model = _StreamModel(["ok ", "still ok "], raise_after=1,
                         exc=RuntimeError("boom"))
    received: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        for tok in _agent(model).stream("anything"):
            received.append(tok)
    # The good tokens before the failure streamed; the error did NOT become a chunk.
    assert received == ["ok "]
    assert not any("boom" in c or "[Error" in c for c in received)


def test_immediate_stream_error_raises_with_no_chunks():
    model = _StreamModel(["x"], raise_after=0, exc=ValueError("nope"))
    received: list[str] = []
    with pytest.raises(ValueError, match="nope"):
        for tok in _agent(model).stream("anything"):
            received.append(tok)
    assert received == []


def test_react_path_stream_error_raises():
    """The tool path also fails explicitly (raises), never as a chunk."""
    from effgen.tools.builtin.calculator import Calculator

    model = _StreamModel(["Thought:"], raise_after=0, exc=RuntimeError("react boom"))
    received: list[str] = []
    with pytest.raises(Exception, match="react boom"):
        for tok in _agent(model, tools=[Calculator()]).stream("15 squared?"):
            received.append(tok)
    assert not any("[Error" in c for c in received)


def test_a_tool_path_failure_is_suppressed_when_raise_on_error_is_off():
    """`raise_on_error=False` governs a streamed provider failure too.

    The text path used to raise whatever the caller asked for. A caller who
    explicitly asked not to be raised at meant it on every path, and the run's
    record carries the typed failure instead.
    """
    from effgen.tools.builtin.calculator import Calculator

    model = _StreamModel(["Thought:"], raise_after=0, exc=RuntimeError("react boom"))
    agent = Agent(config=AgentConfig(
        name="stream-test", model=model, tools=[Calculator()],
        raise_on_error=False,
    ))
    chunks = list(agent.stream("15 squared?"))
    record = agent.last_stream_response
    assert record is not None and record.success is False
    assert record.metadata["reason"] == "generation_failed"
    assert "".join(chunks) == record.output


# --------------------------------------------------------------------------- #
# Tool-agent streaming contract: raw ReAct scaffolding is NEVER the payload.
# --------------------------------------------------------------------------- #

class _ScriptedReActModel(BaseModel):
    """Streams a different scripted ReAct turn on each successive call.

    Lets us drive the tool path deterministically (a tool step, then a final
    answer) without a network model.
    """

    def __init__(self, turns: list[list[str]]):
        super().__init__(model_name="fake-react", model_type=ModelType.OPENAI)
        self._turns = turns
        self._call = 0
        self.last_prompt: str | None = None

    def load(self) -> None:
        pass

    def generate(self, prompt, config=None, **kwargs):
        text = "".join(self._turns[min(self._call, len(self._turns) - 1)])
        self._call += 1
        self.last_prompt = prompt
        return GenerationResult(
            text=text, tokens_used=1, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        turn = self._turns[min(self._call, len(self._turns) - 1)]
        self._call += 1
        self.last_prompt = prompt
        yield from turn

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def unload(self) -> None:
        pass

    def generate_batch(self, prompts, config=None, **kwargs):
        return [self.generate(p, config) for p in prompts]


_SCAFFOLD = ("Thought:", "Action:", "Action Input:", "Observation:", "Final Answer:")


def test_tool_stream_default_has_no_scaffolding_and_joins_to_answer():
    """A tool agent's default stream yields only the sanitized answer text."""
    from effgen.tools.builtin.calculator import Calculator

    model = _ScriptedReActModel([
        ["Thought: I should compute it.\n",
         "Action: calculator\n", "Action Input: 15*15\n"],
        ["Thought: I have the result.\n",
         "Final Answer: The square of 15 is 225.\n"],
    ])
    chunks = list(_agent(model, tools=[Calculator()]).stream("What is 15 squared?"))
    joined = "".join(chunks)
    for marker in _SCAFFOLD:
        assert marker not in joined, f"scaffolding leaked: {marker!r}"
    # The same answer `run()` gives for this agent and this task: a tool that
    # computed the answer itself is reported rather than summarised, which the
    # streamed path used to skip.
    blocking = _agent(
        _ScriptedReActModel([
            ["Thought: I should compute it.\n",
             "Action: calculator\n", "Action Input: 15*15\n"],
            ["Thought: I have the result.\n",
             "Final Answer: The square of 15 is 225.\n"],
        ]),
        tools=[Calculator()],
    ).run("What is 15 squared?")
    assert joined == blocking.output


def test_tool_stream_multiword_answer_not_truncated():
    """The early Final-Answer break must not clip a multi-word answer."""
    from effgen.tools.builtin.calculator import Calculator

    model = _ScriptedReActModel([
        ["Final Answer: Paris is the capital of France and a lovely city.\n"],
    ])
    joined = "".join(_agent(model, tools=[Calculator()]).stream("Capital of France?"))
    assert joined == "Paris is the capital of France and a lovely city."


def test_tool_stream_include_events_yields_typed_events():
    """include_events=True surfaces typed step events, answer text reconstructs."""
    from effgen.core.agent import StreamEvent
    from effgen.tools.builtin.calculator import Calculator

    model = _ScriptedReActModel([
        ["Thought: compute.\n", "Action: calculator\n", "Action Input: 2+2\n"],
        ["Final Answer: 4\n"],
    ])
    events = list(
        _agent(model, tools=[Calculator()]).stream("2+2?", include_events=True)
    )
    assert all(isinstance(e, StreamEvent) for e in events)
    kinds = [e.kind for e in events]
    assert "thought" in kinds and "tool_call" in kinds
    assert "observation" in kinds and "answer" in kinds
    tool_call = next(e for e in events if e.kind == "tool_call")
    assert tool_call.tool == "calculator"
    answer = "".join(e.text for e in events if e.kind == "answer")
    assert answer == "4"
    # No event's text carries raw scaffolding labels.
    for e in events:
        for marker in _SCAFFOLD:
            assert marker not in (e.text or "")


def test_tool_stream_step_limit_emits_a_plain_notice():
    """Hitting the step limit yields a plain notice, not raw bracket scaffolding."""
    from effgen.tools.builtin.calculator import Calculator

    # Never emits a Final Answer → loops to the iteration cap.
    model = _ScriptedReActModel([
        ["Thought: hmm.\n", "Action: calculator\n", "Action Input: 1+1\n"],
    ])
    agent = Agent(config=AgentConfig(
        name="limit", model=model, tools=[Calculator()], max_iterations=2,
    ))
    joined = "".join(agent.stream("loop forever"))
    assert "[Max iterations reached]" not in joined
    for marker in _SCAFFOLD:
        assert marker not in joined
    assert joined.strip()  # never a silently empty stream


# --------------------------------------------------------------------------- #
# OpenAI-compatible streaming usage
# --------------------------------------------------------------------------- #

def _compat_client():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from effgen.api.openai_compat import create_openai_router

    def runner(prompt, *, model, tools, stream, **kw):
        if stream:
            return iter(["Hello", " world", "!"])
        return "Hello world!"

    app = FastAPI()
    app.include_router(create_openai_router(runner))
    return TestClient(app)


def _usage_chunks(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            payload = json.loads(line[len("data: "):])
            if "usage" in payload:
                out.append(payload)
    return out


# --------------------------------------------------------------------------- #
# Guardrails apply on the streaming path too (not just Agent.run())
# --------------------------------------------------------------------------- #
def test_stream_redacts_input_before_model_call_no_tools():
    """A guardrail-configured no-tool agent never lets the model see raw PII."""
    model = _StreamModel(["ok"])
    agent = Agent(config=AgentConfig(
        name="stream-guardrail", model=model, guardrails="phi",
    ))
    list(agent.stream("My SSN is 219-09-9999"))
    # The model only ever saw the redacted prompt, not the raw SSN.
    assert "219-09-9999" not in model.last_prompt
    assert "SSN" in model.last_prompt


def test_stream_blocks_before_any_model_call():
    """A strict-preset block raises before generate_stream() is ever called."""
    model = _StreamModel(["should never run"])
    agent = Agent(config=AgentConfig(
        name="stream-guardrail-block", model=model, guardrails="strict",
    ))
    with pytest.raises(RuntimeError, match="Blocked by guardrail"):
        list(agent.stream("My SSN is 219-09-9999"))
    assert model.stream_calls == 0


def test_stream_no_guardrails_is_unaffected():
    """No guardrails configured: stream() behaves exactly as before."""
    model = _StreamModel(["1", " 2", " 3"])
    agent = Agent(config=AgentConfig(name="stream-no-guardrail", model=model))
    text = "".join(agent.stream("count"))
    assert text == "1 2 3"


def test_tool_stream_redacts_input_before_model_call():
    """The ReAct/tool streaming branch also applies the input guardrail."""
    from effgen.core.agent import AgentConfig
    from effgen.tools.builtin.calculator import Calculator

    model = _ScriptedReActModel([["Final Answer: done\n"]])
    agent = Agent(config=AgentConfig(
        name="tool-stream-guardrail", model=model, tools=[Calculator()],
        guardrails="phi",
    ))
    list(agent.stream("My SSN is 219-09-9999", include_events=False))
    prompt = model.last_prompt
    assert prompt is not None
    assert "219-09-9999" not in prompt


def test_compat_stream_emits_usage_when_requested():
    client = _compat_client()
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert resp.status_code == 200
    chunks = _usage_chunks(resp.text)
    assert len(chunks) == 1
    usage = chunks[0]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # OpenAI's usage-only chunk has empty choices.
    assert chunks[0]["choices"] == []


def test_compat_stream_omits_usage_by_default():
    client = _compat_client()
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert resp.status_code == 200
    assert _usage_chunks(resp.text) == []
    assert "[DONE]" in resp.text
