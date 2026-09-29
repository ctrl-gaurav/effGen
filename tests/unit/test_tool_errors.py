"""A tool error earns a correction, not a repeat.

A call can fail on its input: the model wrote an expression the tool cannot
read, a program that raises, a value of the wrong kind. What the loop does
with such a failure is played here by a scripted model, on ``run()`` and on
``stream()``:

* **It is never a repeated result.** Two calls with different inputs that the
  tool rejects in the same words are two attempts; the run keeps its tools and
  the next, corrected call runs.
* **Corrections are bounded.** A tool that fails on its input four calls in a
  row is not offered again in the run. The run is asked for its answer, and if
  it writes none it ends ``tool_failed`` with ``kind="input"``, naming the tool
  and the last error — inside the iteration cap.
* **An error is never an answer.** A final answer that is a tool's error
  message is sent back once and never reported as a success; the shortcut that
  hands a calculator's result straight back never hands back a failure; a
  failed call never earns the "you have the answer from the tool" line.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from effgen.core import agent_tool_loop
from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.errors import RunStoppedError
from effgen.tools.base_tool import ToolCategory

sys.path.insert(0, str(Path(__file__).parent))
from test_iteration_progress import _Script, _Tool  # noqa: E402

TASK = "Work out the value."
#: Calls in a row that may fail on their input before the tool is withdrawn.
INPUT_ERRORS_BEFORE_WITHDRAWN = 4
UNSUPPORTED = "Error evaluating expression: unsupported expression"
SYNTAX = "Error evaluating expression: invalid syntax (<unknown>, line 1)"


def _native(name: str, **arguments: Any) -> dict[str, Any]:
    return {"text": "", "calls": [{
        "id": f"c-{name}-{json.dumps(arguments, sort_keys=True)}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


def _react(value: str, name: str = "calculator") -> str:
    return (f"Thought: compute it\nAction: {name}\n"
            f"Action Input: {json.dumps({'expression': value})}")


def _agent(model: _Script, tools: list[Any], **config: Any) -> Agent:
    options: dict[str, Any] = {
        "name": "errors", "model": model, "tools": tools, "max_iterations": 10,
        "raise_on_error": False, "enable_memory": False,
        "tool_calling_mode": "hybrid" if model.native else "react",
    }
    options.update(config)
    return Agent(AgentConfig(**options))


def _both(model_f, tools_f, task: str = TASK, **config: Any):
    """Yield (entry, response, model, tools) for run() and stream()."""
    for entry in ("run", "stream"):
        model, tools = model_f(), tools_f()
        agent = _agent(model, tools, **config)
        if entry == "run":
            resp = agent.run(task)
        else:
            for _ in agent.stream(task):
                pass
            resp = agent.last_stream_response
        yield entry, resp, model, tools


def _sent(model: _Script) -> list[str]:
    return list(model.prompts)


def _distinct_errors(n: int = 30) -> list[str]:
    return [f"Error: cannot read input number {i}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Never a repeated result
# ---------------------------------------------------------------------------


def test_the_bound_is_a_call_and_three_corrections() -> None:
    limit = getattr(agent_tool_loop, "INPUT_ERRORS_BEFORE_WITHDRAWN", None)
    assert limit == INPUT_ERRORS_BEFORE_WITHDRAWN


def test_the_same_error_text_is_never_a_repeated_result() -> None:
    loop = NativeToolLoop({"calculator": _Tool()})
    loop.record_result("calculator", UNSUPPORTED)
    assert not loop.result_is_repeat("calculator", UNSUPPORTED)
    # A result that is not a failure still is.
    loop.record_result("calculator", "42")
    assert loop.result_is_repeat("calculator", "42")


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
def test_two_inputs_rejected_in_the_same_words_leave_the_run_its_tools(native: bool) -> None:
    turns = (
        [_native("calculator", expression="2+x"), _native("calculator", expression="2+y"),
         _native("calculator", expression="2+3"), "Final Answer: 5"]
        if native else
        [_react("2+x"), _react("2+y"), _react("2+3"), "Final Answer: 5"]
    )
    for entry, resp, _model, tools in _both(
        lambda: _Script(list(turns), native=native),
        lambda: [_Tool(replies=[UNSUPPORTED, UNSUPPORTED, "5"])],
    ):
        assert [i["expression"] for i in tools[0].inputs] == ["2+x", "2+y", "2+3"], entry
        assert resp.success and resp.output.strip() == "5", entry


def test_a_failed_call_near_the_cap_is_not_called_the_answer() -> None:
    turns = [_native("calculator", expression="1+2"), _native("calculator", expression="2+2"),
             _native("calculator", expression="3 x 3"), _native("calculator", expression="3*3"),
             "Final Answer: 9"]
    for entry, _resp, model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(replies=["3", "4", SYNTAX, "9"])],
        max_iterations=5,
    ):
        after_error = _sent(model)[3]
        assert "You have the answer from the tool" not in after_error, entry


# ---------------------------------------------------------------------------
# Bounded, then typed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
def test_a_tool_that_always_fails_on_its_input_ends_typed_inside_the_cap(native: bool) -> None:
    turns = ([_native("calculator", expression=f"v{i}") for i in range(30)] if native
             else [_react(f"v{i}") for i in range(30)])
    for entry, resp, model, tools in _both(
        lambda: _Script(list(turns), native=native),
        lambda: [_Tool(replies=_distinct_errors())],
    ):
        assert len(tools[0].inputs) == INPUT_ERRORS_BEFORE_WITHDRAWN, entry
        assert len(model.prompts) < 10, entry
        assert resp.stop_reason == "tool_failed" and not resp.success, entry
        assert resp.termination == "tool_failed", entry
        error = resp.metadata["error"]
        assert error["kind"] == "input" and error["tool"] == "calculator", entry
        assert "calculator" in resp.output and "cannot read input number 4" in resp.output, entry
        assert resp.metadata["unavailable_tools"] == ["calculator"], entry


def test_a_tool_that_raises_on_every_input_is_named_with_its_error_class() -> None:
    for entry, resp, _model, tools in _both(
        lambda: _Script([_native("calculator", expression=f"v{i}") for i in range(30)]),
        lambda: [_Tool(replies=[ValueError(f"cannot parse input {i}") for i in range(30)])],
    ):
        assert len(tools[0].inputs) == INPUT_ERRORS_BEFORE_WITHDRAWN, entry
        assert resp.metadata["error"]["kind"] == "input", entry
        assert "ValueError" in resp.output and "cannot parse input" in resp.output, entry


def test_the_typed_input_failure_raises_under_raise_on_error() -> None:
    model = _Script([_native("calculator", expression=f"v{i}") for i in range(30)])
    agent = _agent(model, [_Tool(replies=_distinct_errors())], raise_on_error=True)
    with pytest.raises(RunStoppedError) as info:
        agent.run(TASK)
    assert info.value.stop_reason == "tool_failed"


def test_after_the_tool_is_withdrawn_the_models_own_answer_stands() -> None:
    turns = [_native("calculator", expression=c) for c in "abcd"] + ["Final Answer: 7"] * 2
    for entry, resp, _model, tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(replies=_distinct_errors())],
    ):
        assert len(tools[0].inputs) == INPUT_ERRORS_BEFORE_WITHDRAWN, entry
        assert resp.success and resp.output.strip() == "7", entry
        assert resp.termination == "done", entry


def test_three_failures_then_a_correct_call_still_runs() -> None:
    turns = [_native("calculator", expression=c) for c in ("a", "b", "c", "6*7")]
    for entry, resp, _model, tools in _both(
        lambda: _Script([*turns, "Final Answer: 42"]),
        lambda: [_Tool(replies=[ValueError("bad 1"), ValueError("bad 2"),
                                ValueError("bad 3"), "42"])],
        max_turns_without_progress=None,
    ):
        assert [i["expression"] for i in tools[0].inputs] == ["a", "b", "c", "6*7"], entry
        assert resp.output.strip() == "42", entry


def test_a_withdrawn_tool_leaves_the_other_tool_usable() -> None:
    turns = [_native("calculator", expression=c) for c in "abcde"]
    for entry, resp, _model, tools in _both(
        lambda: _Script([*turns, _native("search", expression="capital"), "Final Answer: Paris"]),
        lambda: [_Tool(replies=_distinct_errors()), _Tool(name="search", replies=["Paris"])],
    ):
        assert len(tools[0].inputs) == INPUT_ERRORS_BEFORE_WITHDRAWN, entry
        assert len(tools[1].inputs) == 1, entry
        assert resp.success and resp.output.strip() == "Paris", entry


def test_the_bound_holds_with_the_progress_policy_off(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([_native("calculator", expression=f"v{i}") for i in range(30)])
    tool = _Tool(replies=_distinct_errors())
    resp = _agent(model, [tool], max_turns_without_progress=None).run(TASK)
    assert len(tool.inputs) == INPUT_ERRORS_BEFORE_WITHDRAWN
    assert resp.stop_reason == "tool_failed" and resp.metadata["error"]["kind"] == "input"
    assert any("not offered again in this run" in r.getMessage() for r in caplog.records)


def test_code_tool_failures_are_bounded_too() -> None:
    code = ToolCategory.CODE_EXECUTION
    for entry, resp, _model, tools in _both(
        lambda: _Script([_native("python_exec", code=f"print(x{i})") for i in range(30)]),
        lambda: [_Tool(name="python_exec", category=code, param="code",
                       replies=[f"Error: NameError: name 'x{i}' is not defined"
                                for i in range(30)])],
    ):
        assert len(tools[0].inputs) == INPUT_ERRORS_BEFORE_WITHDRAWN, entry
        assert resp.stop_reason == "tool_failed", entry
        assert resp.metadata["error"]["tool"] == "python_exec", entry


# ---------------------------------------------------------------------------
# Never an answer
# ---------------------------------------------------------------------------


def test_an_answer_that_is_the_tools_error_is_sent_back() -> None:
    turns = [_native("calculator", expression="6 x 7"), f"Final Answer: {UNSUPPORTED}",
             _native("calculator", expression="6*7"), "Final Answer: 42"]
    for entry, resp, model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(replies=[UNSUPPORTED, "42"])],
    ):
        assert resp.success and resp.output.strip() == "42", entry
        assert "is not an answer" in _sent(model)[2] or "not an answer" in _sent(model)[2], entry


def test_an_error_answer_given_again_is_never_a_success() -> None:
    turns = [_native("calculator", expression="6 x 7")] + [f"Final Answer: {UNSUPPORTED}"] * 4
    for entry, resp, _model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(replies=[UNSUPPORTED] * 3)],
    ):
        assert not resp.success, entry
        assert not resp.output.startswith("Error evaluating"), entry
        assert resp.stop_reason == "null_final_from_model", entry
        assert "answered with an error from" in resp.output, entry


def test_a_sentence_that_mentions_the_error_is_an_answer() -> None:
    turns = [_native("calculator", expression="1/0"),
             "Final Answer: The expression cannot be evaluated: division by zero."]
    for entry, resp, _model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(replies=["Error evaluating expression: division by zero"])],
    ):
        assert resp.success and "division by zero" in resp.output, entry


def test_the_direct_calculator_shortcut_never_returns_an_error() -> None:
    turns = ['Thought: multiply\nAction: calculator\nAction Input: {"expression": "12 x 7"}',
             'Thought: fix it\nAction: calculator\nAction Input: {"expression": "12*7"}',
             "Final Answer: 84"]
    for entry, resp, _model, tools in _both(
        lambda: _Script(list(turns), native=False),
        lambda: [_Tool(replies=[SYNTAX, "84"])],
        task="What is 12 * 7?",
    ):
        assert resp.success and resp.output.strip() == "84", entry
        assert len(tools[0].inputs) == 2, entry


def test_an_answer_restating_the_failure_in_its_own_words_is_sent_back() -> None:
    timeout = "Error: code did not finish within 30s"
    turns = [_native("python_exec", code="slow()"),
             "Error: Code execution timed out. Please try again with a smaller problem.",
             _native("python_exec", code="fast()"), "Final Answer: 7"]
    for entry, resp, model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(name="python_exec", category=ToolCategory.CODE_EXECUTION, param="code",
                       replies=[timeout, "7"])],
    ):
        assert resp.success and resp.output.strip() == "7", entry
        assert "not an answer" in _sent(model)[2], entry


def test_an_answer_opening_with_error_in_a_run_with_no_failure_is_an_answer() -> None:
    turns = [_native("calculator", expression="6*7"), "Final Answer: Error rate is 42 percent."]
    for entry, resp, _model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(replies=["42"])],
    ):
        assert resp.success and resp.output.startswith("Error rate is 42"), entry


@pytest.mark.parametrize("answer", [
    "Error rate is 21 percent.",
    "Error absoluto: 21.",
])
def test_an_answer_opening_with_error_after_the_call_was_corrected_is_an_answer(answer: str) -> None:
    turns = [_native("calculator", expression="3 x 7"), _native("calculator", expression="3*7"),
             f"Final Answer: {answer}", f"Final Answer: {answer}"]
    for entry, resp, model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(replies=[SYNTAX, "21"])],
    ):
        assert resp.success and resp.output.strip() == answer, entry
        assert not any("not an answer" in p for p in _sent(model)), entry


def test_an_answer_that_only_opens_with_the_word_error_is_an_answer() -> None:
    answer = "Errors found: f is undefined on line 1."
    turns = [_native("python_exec", code="f()"), f"Final Answer: {answer}", f"Final Answer: {answer}"]
    for entry, resp, _model, _tools in _both(
        lambda: _Script(list(turns)),
        lambda: [_Tool(name="python_exec", category=ToolCategory.CODE_EXECUTION, param="code",
                       replies=["Error: NameError: name 'f' is not defined"])],
        task="What is wrong with this code: f()",
    ):
        assert resp.success and resp.output.strip() == answer, entry
