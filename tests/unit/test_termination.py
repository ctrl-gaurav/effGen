"""A run ends done, not possible, stuck or with its tool failed — and says which.

Each of the four endings is played by a scripted model and checked on
``run()`` and on ``stream()``, which must reach the same ending.

**Done.** A model that has its result and then writes turns that do nothing —
``Action: (continue reasoning)``, ``Action: None`` — is asked for the answer
instead of running to the iteration cap.

**Not possible.** A model that keeps calling a tool it does not hold is asked
for its answer after two such turns, and what it says is the answer; the run
reports that its tools brought nothing usable.

**Stuck.** A model that repeats itself and never answers gets one closing
request — its own calls and results, no tools — and then stops, typed.

**Tool failed.** A tool whose service fails on its own side is given up after
three attempts, and the run says which tool failed and why, instead of burning
its budget or reporting success. Bad input is not a tool failure: it never
counts against the tool.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_response import AgentResponse
from effgen.errors import RunStoppedError
from effgen.models.base import GenerationResult
from effgen.tools.base_tool import ToolCategory

sys.path.insert(0, str(Path(__file__).parent))
from test_iteration_progress import _call, _Script, _Tool  # noqa: E402

TASK = "Work out the value."

#: What the loop's in-frame answer request says, as the scripted model reads it.
_ASKS = ("respond now with 'Final Answer:'", "respond with 'Final Answer:' now")


class _Reactive(_Script):
    """Plays its script; answers whenever the request asks for the answer.

    A request asks when it carries the loop's answer request, forbids a call,
    or is the closing request — no tools offered and no scaffold in it.
    """

    def __init__(
        self, turns: list[Any], *, answers_when_asked: bool,
        answer: str = "Final Answer: 42", **kw: Any,
    ) -> None:
        super().__init__(turns, **kw)
        self.answers_when_asked = answers_when_asked
        self.answer = answer
        self.closing_requests = 0

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
        closing = "tools" not in kwargs and "Action Input" not in text
        self.closing_requests += int(closing)
        asked = (
            any(a in text[-600:] for a in _ASKS)
            or kwargs.get("tool_choice") == "none"
            or closing
        )
        if self.answers_when_asked and asked:
            self.prompts.append(text)
            self.kwargs.append(dict(kwargs))
            return GenerationResult(
                text=self.answer, tokens_used=5, finish_reason="stop",
                model_name=self.model_name, metadata={},
            )
        return super().generate(prompt, config, **kwargs)


def _calc(expr: str) -> dict[str, Any]:
    return {"text": "", "calls": [_call("calculator", expression=expr)]}


def _search(query: str) -> dict[str, Any]:
    return {"text": "", "calls": [_call("web_search", query=query)]}


def _written(expr: str) -> str:
    return f'Action: calculator\nAction Input: {{"expression": "{expr}"}}'


def _search_tool(replies: list[Any]) -> _Tool:
    return _Tool(
        "web_search", ToolCategory.INFORMATION_RETRIEVAL, replies=replies, param="query",
    )


def _agent(model: _Script, tools: list[Any], **config: Any) -> Agent:
    options: dict[str, Any] = {
        "name": "probe", "model": model, "tools": tools, "max_iterations": 10,
        "raise_on_error": False, "enable_memory": False,
        "tool_calling_mode": "hybrid",
    }
    options.update(config)
    return Agent(AgentConfig(**options))


def _both(model_factory: Any, tools_factory: Any, **config: Any) -> list[tuple[str, AgentResponse, Any, Any]]:
    """Run the scenario through ``run()`` and ``stream()``; return both."""
    out = []
    model, tools = model_factory(), tools_factory()
    out.append(("run", _agent(model, tools, **config).run(TASK), model, tools))
    model, tools = model_factory(), tools_factory()
    agent = _agent(model, tools, **config)
    for _ in agent.stream(TASK):
        pass
    streamed = agent.last_stream_response
    assert streamed is not None
    out.append(("stream", streamed, model, tools))
    return out


def _dispatches(tools: list[_Tool]) -> int:
    return sum(len(t.inputs) for t in tools)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_terminations_are_the_five_documented_values() -> None:
    from effgen.core.agent_response import TERMINATIONS

    assert TERMINATIONS == ("done", "not_possible", "stuck", "tool_failed", "error")


def test_termination_is_derived_and_saved() -> None:
    answered = AgentResponse(output="42")
    assert answered.termination == "done"
    assert answered.to_dict()["termination"] == "done"
    stuck = AgentResponse(output="x", success=False, stop_reason="loop_detected")
    assert stuck.termination == "stuck"
    failed = AgentResponse(output="x", success=False, stop_reason="tool_failed")
    assert (failed.termination, failed.outcome) == ("tool_failed", "stopped")
    error = AgentResponse(output="x", success=False, stop_reason="generation_failed")
    assert error.termination == "error"
    nothing = AgentResponse(
        output="no", metadata={"tool_results": {"attempted": 2, "usable": 0}},
    )
    assert nothing.termination == "not_possible"


# ---------------------------------------------------------------------------
# Done: a result, then turns that do nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filler", [
    "Thought: The result is 42.\nAction: (continue reasoning)\nAction Input: {}",
    "Thought: The result is 42.\nAction: None\nAction Input: None",
])
def test_a_result_then_idle_turns_is_asked_and_done(filler: str) -> None:
    for how, resp, model, _tools in _both(
        lambda: _Reactive([_calc("6*7")] + [filler] * 12, answers_when_asked=True),
        lambda: [_Tool(replies=["42"])],
    ):
        assert len(model.prompts) <= 3, how
        assert (resp.termination, resp.output, resp.success) == ("done", "42", True), how


def test_a_placeholder_action_is_logged_as_one(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    model = _Reactive(
        [_calc("6*7")] + ["Action: (continue reasoning)\nAction Input: {}"] * 5,
        answers_when_asked=True,
    )
    _agent(model, [_Tool(replies=["42"])]).run(TASK)
    assert any(
        "[progress] the turn named a placeholder action ((continue reasoning))"
        in r.getMessage() for r in caplog.records
    )


def test_a_parenthesised_held_tool_name_is_not_a_placeholder() -> None:
    from effgen.core.tool_calling import placeholder_action

    assert placeholder_action("(continue reasoning)", {"calculator": 1}) == "(continue reasoning)"
    assert placeholder_action("[final answer]", {"calculator": 1}) == "[final answer]"
    assert placeholder_action("(calculator)", {"calculator": 1}) is None
    assert placeholder_action("calculator", {"calculator": 1}) is None


def test_saying_so_at_once_is_one_call() -> None:
    for how, resp, model, _tools in _both(
        lambda: _Reactive(["Final Answer: I cannot send email with a calculator."],
                          answers_when_asked=True),
        lambda: [_Tool(replies=["42"])],
    ):
        assert len(model.prompts) == 1, how
        assert resp.termination == "done", how


def test_work_that_keeps_bringing_new_results_is_not_cut() -> None:
    for how, resp, model, _tools in _both(
        lambda: _Reactive([_search(f"query {i}") for i in range(20)], answers_when_asked=True),
        lambda: [_search_tool([])],
    ):
        assert len(model.prompts) == 8, how
        assert resp.termination == "done", how


# ---------------------------------------------------------------------------
# Stuck: the same call forever, never an answer
# ---------------------------------------------------------------------------


def test_a_run_that_never_answers_gets_one_closing_request_and_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    for how, resp, model, _tools in _both(
        lambda: _Reactive([_calc("6*7")] * 20, answers_when_asked=False),
        lambda: [_Tool(replies=["42"])],
    ):
        assert len(model.prompts) <= 6, how
        assert model.closing_requests == 1, how
        assert resp.stop_reason == "loop_detected", how
        assert resp.termination == "stuck", how
        assert resp.partial is not None and resp.partial.text == "42", how
    assert any("[closing] no answer" in r.getMessage() for r in caplog.records)


def test_a_stuck_run_raises_under_the_default() -> None:
    model = _Reactive([_calc("6*7")] * 20, answers_when_asked=False)
    with pytest.raises(RunStoppedError) as info:
        _agent(model, [_Tool(replies=["42"])], raise_on_error=True).run(TASK)
    assert info.value.response.termination == "stuck"


def test_a_closing_request_that_answers_ends_the_run_answered() -> None:
    model = _Reactive([_calc("6*7")] * 20, answers_when_asked=False)
    # The closing request carries no tool definitions; this model answers it.
    original = model.generate

    def generate(prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
        if "tools" not in kwargs and "Action Input" not in text:
            model.prompts.append(text)
            return GenerationResult(text="The value is 42.", tokens_used=5,
                                    finish_reason="stop", model_name="m", metadata={})
        return original(prompt, config, **kwargs)

    model.generate = generate  # type: ignore[method-assign]
    resp = _agent(model, [_Tool(replies=["42"])]).run(TASK)
    assert (resp.success, resp.stop_reason, resp.termination) == (True, "final_answer", "done")
    assert resp.metadata["answer_source"] == "closing_request"
    assert resp.output == "The value is 42."


def test_the_closing_request_carries_no_tools_and_no_scaffold() -> None:
    model = _Reactive([_calc("6*7")] * 20, answers_when_asked=False)
    _agent(model, [_Tool(replies=["42"])]).run(TASK)
    closing = [
        (p, k) for p, k in zip(model.prompts, model.kwargs, strict=False)
        if "tools" not in k and "Action Input" not in p
    ]
    assert len(closing) == 1
    prompt, kwargs = closing[0]
    assert "tool_choice" not in kwargs
    assert TASK in prompt and "calculator" in prompt and "42" in prompt
    assert "Final Answer" not in prompt


# ---------------------------------------------------------------------------
# Tool failed: the tool's own side is down
# ---------------------------------------------------------------------------


def test_a_tool_that_cannot_connect_ends_the_run_typed() -> None:
    for how, resp, model, tools in _both(
        lambda: _Reactive([_search(f"query {i}") for i in range(20)], answers_when_asked=True,
                          answer="Final Answer: I could not reach the search service."),
        lambda: [_search_tool([ConnectionError("connection refused")])],
    ):
        assert len(model.prompts) <= 4, how
        assert _dispatches(tools) == 3, how
        assert (resp.stop_reason, resp.outcome, resp.termination) == (
            "tool_failed", "stopped", "tool_failed"), how
        assert "web_search" in resp.output and "ConnectionError" in resp.output, how
        assert resp.metadata["error"]["kind"] == "tool"
        assert resp.metadata["unavailable_tools"] == ["web_search"]
        assert resp.partial is None, how


def test_a_tool_failure_raises_under_the_default() -> None:
    model = _Reactive([_search(f"query {i}") for i in range(20)], answers_when_asked=False)
    with pytest.raises(RunStoppedError) as info:
        _agent(model, [_search_tool([ConnectionError("refused")])], raise_on_error=True).run(TASK)
    assert info.value.stop_reason == "tool_failed"


def test_a_timeout_on_the_same_call_ends_tool_failed() -> None:
    """The same call after a tool-side failure is a retry, not a repeat."""
    for how, resp, model, tools in _both(
        lambda: _Reactive([_search("q")] * 20, answers_when_asked=False),
        lambda: [_search_tool([TimeoutError("timed out")])],
    ):
        assert len(model.prompts) <= 4, how
        assert _dispatches(tools) == 3, how
        assert resp.stop_reason == "tool_failed", how
        assert "TimeoutError" in resp.output and "3 attempts" in resp.output, how


def test_bad_input_does_not_open_the_breaker() -> None:
    tool = _Tool(replies=[ValueError("bad 1"), ValueError("bad 2"), ValueError("bad 3"), "42"])
    model = _Reactive(
        [_calc("a"), _calc("b"), _calc("c"), _calc("6*7"), "Final Answer: 42"],
        answers_when_asked=True,
    )
    resp = _agent(model, [tool]).run(TASK)
    assert [i["expression"] for i in tool.inputs] == ["a", "b", "c", "6*7"]
    assert resp.output == "42"


def test_one_tool_down_leaves_the_other_usable() -> None:
    down = _search_tool([ConnectionError("refused")])
    calc = _Tool(replies=["42"])
    model = _Reactive(
        [_search("q1"), _search("q2"), _search("q3"), _calc("6*7"), "Final Answer: 42"],
        answers_when_asked=True,
    )
    resp = _agent(model, [down, calc]).run(TASK)
    assert len(down.inputs) == 3 and len(calc.inputs) == 1
    assert (resp.success, resp.output, resp.termination) == (True, "42", "done")
    assert resp.metadata["unavailable_tools"] == ["web_search"]


def test_an_unavailable_tool_is_declined_not_dispatched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    down = _search_tool([ConnectionError("refused")])
    calc = _Tool(replies=["42"])
    model = _Reactive(
        [_search("q1"), _search("q2"), _search("q3"), _search("q4"), _calc("6*7"),
         "Final Answer: 42"],
        answers_when_asked=True,
    )
    _agent(model, [down, calc]).run(TASK)
    assert len(down.inputs) == 3
    assert any(
        "[tool] 'web_search' is unavailable for this run (ConnectionError)" in r.getMessage()
        for r in caplog.records
    )


def test_failure_side_is_read_from_the_class() -> None:
    from effgen.core.tool_failure import INPUT_SIDE, TOOL_SIDE, failure_side
    from effgen.errors import MissingCredentialsError

    class HTTPStatusError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__("status")
            self.response = type("R", (), {"status_code": status})()

    assert failure_side(ConnectionRefusedError()) == TOOL_SIDE
    assert failure_side(TimeoutError()) == TOOL_SIDE
    assert failure_side(MissingCredentialsError("t", ["KEY"])) == TOOL_SIDE
    assert failure_side(HTTPStatusError(503)) == TOOL_SIDE
    assert failure_side(HTTPStatusError(429)) == TOOL_SIDE
    assert failure_side(HTTPStatusError(404)) == INPUT_SIDE
    assert failure_side(ValueError("connection refused")) == INPUT_SIDE
    assert failure_side(class_names=["ReadTimeout", "Timeout", "RequestException"]) == TOOL_SIDE


def test_an_http_status_decides_before_the_error_class_chain() -> None:
    """urllib's HTTPError is also a URLError; its status, not the chain, says whose fault."""
    import io
    import urllib.error

    from effgen.core.tool_failure import INPUT_SIDE, TOOL_SIDE, failure_side

    def http_error(status: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError("http://x", status, "m", {}, io.BytesIO(b""))  # type: ignore[arg-type]

    assert failure_side(http_error(404)) == INPUT_SIDE
    assert failure_side(http_error(400)) == INPUT_SIDE
    assert failure_side(http_error(503)) == TOOL_SIDE
    assert failure_side(http_error(429)) == TOOL_SIDE
    assert failure_side(urllib.error.URLError("connection refused")) == TOOL_SIDE
    assert failure_side(class_names=["HTTPError", "URLError", "OSError"], status=404) == INPUT_SIDE
    # A tool's error envelope records the status urllib carries on the error itself.
    from effgen.tools.base_tool import _http_status_of

    assert _http_status_of(http_error(404)) == 404


# ---------------------------------------------------------------------------
# Not possible: the tools could not serve the task
# ---------------------------------------------------------------------------


def test_calling_a_tool_it_does_not_hold_is_asked_and_not_possible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    for how, resp, model, _tools in _both(
        lambda: _Reactive(
            [{"text": "", "calls": [_call("send_email", to=f"a{i}@b")]} for i in range(20)],
            answers_when_asked=True, answer="Final Answer: I cannot send email.",
        ),
        lambda: [_Tool(replies=["42"])],
    ):
        assert len(model.prompts) <= 4, how
        assert (resp.success, resp.termination) == (True, "not_possible"), how
        assert resp.output == "I cannot send email."
    phrases = [r.getMessage() for r in caplog.records]
    assert any("[progress] a turn with only declined or failed calls" in p for p in phrases)
    assert any("[termination] not_possible" in p for p in phrases)


# ---------------------------------------------------------------------------
# A stopped run carries results, never errors, as its progress
# ---------------------------------------------------------------------------


def test_a_stop_holding_only_errors_carries_no_partial() -> None:
    model = _Reactive([_written(f"x{i}") for i in range(20)], answers_when_asked=False,
                      native=False)
    resp = _agent(model, [_Tool(replies=["Error: cannot evaluate"])]).run(TASK)
    assert resp.outcome == "stopped"
    assert resp.partial is None
    assert "partial_output" not in resp.metadata


def test_a_stop_after_a_real_result_carries_that_result() -> None:
    model = _Reactive([_written(f"x{i}") for i in range(20)], answers_when_asked=False,
                      native=False)
    resp = _agent(model, [_Tool(replies=["42", "Error: cannot evaluate"])]).run(TASK)
    assert resp.outcome == "stopped"
    assert resp.partial is not None and resp.partial.text == "42"


# ---------------------------------------------------------------------------
# None restores the earlier loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("turns", "answers", "expected_calls", "expected_stop"), [
    ([_calc("6*7")] + ["Thought: 42.\nAction: None\nAction Input: None"] * 12, True, 10,
     "max_iterations_partial"),
    ([_calc("6*7")] * 20, False, 5, "loop_detected"),
    ([{"text": "", "calls": [_call("send_email", to=f"a{i}@b")]} for i in range(20)],
     True, 10, "max_iterations_exhausted"),
])
def test_none_turns_the_policy_off(
    turns: list[Any], answers: bool, expected_calls: int, expected_stop: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    model = _Reactive(turns, answers_when_asked=answers)
    resp = _agent(model, [_Tool(replies=["42"])], max_turns_without_progress=None).run(TASK)
    assert len(model.prompts) == expected_calls
    assert resp.stop_reason == expected_stop
    assert not any("[closing]" in r.getMessage() for r in caplog.records)


def test_the_shipped_default_is_two() -> None:
    from effgen.core.agent_config import DEFAULT_MAX_TURNS_WITHOUT_PROGRESS

    assert DEFAULT_MAX_TURNS_WITHOUT_PROGRESS == 2
    assert AgentConfig(name="a", model="m").max_turns_without_progress == 2


def test_on_the_text_scaffold_the_closing_request_is_the_ask(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A text-frame run is asked without its scaffold, not with it."""
    caplog.set_level(logging.INFO)
    model = _Reactive(
        [_written("6*7")] + ["Thought: 42.\nAction: (continue reasoning)\nAction Input: {}"] * 12,
        answers_when_asked=True, native=False,
    )
    resp = _agent(model, [_Tool(replies=["42"])]).run(TASK)
    assert len(model.prompts) == 3
    assert model.closing_requests == 1
    assert (resp.termination, resp.metadata["answer_source"]) == ("done", "closing_request")
    assert any("[closing] asking for the answer without tools (the answer request, on the "
               "text scaffold)" in r.getMessage() for r in caplog.records)


class _AnswersOnlyTheClosing(_Reactive):
    """Repeats its call until it is sent the closing request, then says *answer*."""

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
        if "tools" not in kwargs and "Action Input" not in text:
            self.prompts.append(text)
            self.kwargs.append(dict(kwargs))
            return GenerationResult(text=self.answer, tokens_used=5, finish_reason="stop",
                                    model_name=self.model_name, metadata={})
        return _Script.generate(self, prompt, config, **kwargs)


@pytest.mark.parametrize("reply", [
    "import math\nprint(math.sqrt(1764))",
    "```python\nprint(6 * 7)\n```",
    'python_exec({"code": "print(6 * 7)"})',
])
def test_a_closing_reply_that_is_a_program_is_not_the_answer(
    reply: str, caplog: pytest.LogCaptureFixture,
) -> None:
    """A run holding a code executor keeps what its code printed, not a new program."""
    caplog.set_level(logging.INFO)
    tool = _Tool("python_exec", ToolCategory.CODE_EXECUTION, replies=["42"], param="code")
    model = _AnswersOnlyTheClosing(
        [{"text": "", "calls": [_call("python_exec", code="print(6*7)")]}] * 20,
        answers_when_asked=False, answer=reply,
    )
    resp = _agent(model, [tool]).run(TASK)
    assert resp.success is False and resp.termination == "stuck"
    assert resp.partial is not None and resp.partial.text == "42"
    assert any("[closing] no answer (the reply is a program for 'python_exec'" in r.getMessage()
               for r in caplog.records)


def test_a_closing_reply_that_states_the_result_is_kept_beside_an_executor() -> None:
    tool = _Tool("python_exec", ToolCategory.CODE_EXECUTION, replies=["42"], param="code")
    model = _AnswersOnlyTheClosing(
        [{"text": "", "calls": [_call("python_exec", code="print(6*7)")]}] * 20,
        answers_when_asked=False, answer="The program printed 42, so the answer is 42.",
    )
    resp = _agent(model, [tool]).run(TASK)
    assert (resp.success, resp.termination, resp.metadata["answer_source"]) == (
        True, "done", "closing_request")


def test_effgen_code_reports_a_tool_failure_as_stopped_like_the_library() -> None:
    """``effgen code --json`` branches on the same outcome word a library run reports."""
    from effgen.cli.code.engine import CodeRunResult
    from effgen.core.agent_response import STOPPED_REASONS

    for reason in sorted(STOPPED_REASONS):
        result = CodeRunResult(
            task="t", answer="", success=False, reason=reason, model="m", provider=None,
            workspace=".", permission_mode="ask", stop_reason=reason,
        )
        assert result.outcome == "stopped", reason


@pytest.mark.parametrize("reply", ["x = 5", "total = 18", "a, b = 3, 4"])
def test_a_closing_reply_that_assigns_a_plain_value_is_an_answer(reply: str) -> None:
    """``x = 5`` states a result; only an assignment that computes something is a program."""
    tool = _Tool("python_exec", ToolCategory.CODE_EXECUTION, replies=["42"], param="code")
    model = _AnswersOnlyTheClosing(
        [{"text": "", "calls": [_call("python_exec", code="print(6*7)")]}] * 20,
        answers_when_asked=False, answer=reply,
    )
    resp = _agent(model, [tool]).run(TASK)
    assert (resp.success, resp.output, resp.metadata["answer_source"]) == (
        True, reply, "closing_request")


def test_a_closing_reply_that_computes_in_an_assignment_is_still_a_program() -> None:
    tool = _Tool("python_exec", ToolCategory.CODE_EXECUTION, replies=["42"], param="code")
    model = _AnswersOnlyTheClosing(
        [{"text": "", "calls": [_call("python_exec", code="print(6*7)")]}] * 20,
        answers_when_asked=False, answer="result = 6 * 7\nprint(result)",
    )
    resp = _agent(model, [tool]).run(TASK)
    assert resp.success is False and resp.termination == "stuck"


def test_a_code_answer_on_the_text_scaffold_is_not_lost_to_the_program_check() -> None:
    """A task whose answer is code: the rejected closing reply gives the turn back, and the
    run answers in its own frame as it did before the closing request existed."""
    code = "```python\ndef reverse(s):\n    return s[::-1]\n```"
    tool = _Tool("python_exec", ToolCategory.CODE_EXECUTION, replies=["1"], param="code")
    model = _Reactive(
        ['Action: python_exec\nAction Input: {"code": "print(1)"}'] * 20,
        answers_when_asked=True, answer=code, native=False,
    )
    resp = _agent(model, [tool]).run(TASK)
    assert resp.success is True and resp.stop_reason == "final_answer"
    assert "def reverse" in resp.output
    assert resp.metadata.get("answer_source") != "closing_request"
    assert model.closing_requests >= 1


def test_answering_after_every_call_failed_on_its_input_is_done() -> None:
    """The tool ran and rejected each expression; the model then answered itself. That is an
    answer (``done``), not a statement that the task cannot be done."""
    tool = _Tool(replies=[ValueError("Error evaluating expression: unsupported expression")])
    model = _Reactive(
        [_calc("(1+2i)*6"), _calc("(1+2*i)*6"), "Final Answer: 6+9i"],
        answers_when_asked=False,
    )
    resp = _agent(model, [tool]).run(TASK)
    assert resp.success is True and resp.output == "6+9i"
    assert resp.metadata["tool_results"]["usable"] == 0
    assert resp.metadata["tool_results"]["input_errors"] == 2
    assert resp.termination == "done"
