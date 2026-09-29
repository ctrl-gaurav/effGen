"""How a tool call the model wrote becomes the call that is dispatched.

A call is read or refused, never half-read: arguments that arrive as a string
are the call's arguments, a call missing a required argument is asked for
again instead of being dispatched empty, positional values are named or the
call carries none, and at the shipped default a tagged call whose closing quote
sits one bracket early, or a program written in a fenced block before an empty
call tag, is read. Every scenario runs through ``run()`` and ``stream()``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from effgen.core import tool_calling as tc
from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_config import DEFAULT_RECOVER_LOST_TOOL_CALLS
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_iteration_progress import _Script, _Tool  # noqa: E402

TASK = "Work out the value."
CODE = ToolCategory.CODE_EXECUTION


def _py() -> _Tool:
    return _Tool("python_exec", CODE, replies=["2"], param="code")


def _calc() -> _Tool:
    return _Tool("calculator", replies=["42"])


class _NoRequired(BaseTool):
    """A tool that declares no required parameter."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="clock", description="The time.", category=ToolCategory.SYSTEM,
            parameters=[ParameterSpec(
                name="zone", type=ParameterType.STRING, description="Zone.",
                required=False,
            )],
        ))
        self.inputs: list[dict[str, Any]] = []

    async def _execute(self, **kwargs: Any) -> Any:
        self.inputs.append(dict(kwargs))
        return "noon"


def _native(name: str, arguments: str, text: str = "") -> dict[str, Any]:
    return {"text": text, "calls": [{"id": f"c-{name}", "type": "function",
                                    "function": {"name": name, "arguments": arguments}}]}


def _agent(model: _Script, tools: list[Any], **config: Any) -> Agent:
    options: dict[str, Any] = {
        "name": "reader", "model": model, "tools": tools, "max_iterations": 6,
        "raise_on_error": False, "enable_memory": False, "tool_calling_mode": "hybrid",
    }
    options.update(config)
    return Agent(AgentConfig(**options))


def _both(turns: list[Any], tools_factory: Any, **config: Any) -> list[tuple[str, Any, _Script, list[Any]]]:
    """The scenario through ``run()`` and through ``stream()``."""
    out = []
    model, tools = _Script(list(turns)), tools_factory()
    out.append(("run", _agent(model, tools, **config).run(TASK), model, tools))
    model, tools = _Script(list(turns)), tools_factory()
    agent = _agent(model, tools, **config)
    for _ in agent.stream(TASK):
        pass
    assert agent.last_stream_response is not None
    out.append(("stream", agent.last_stream_response, model, tools))
    return out


def _lines(caplog: pytest.LogCaptureFixture, phrase: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if phrase in r.getMessage()]


MISPLACED = (
    '<tool_call>\n{"name": "python_exec", "arguments": {"code": "print(sum([1, 1])")}}'
    "\n</tool_call>"
)


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------


def test_lost_calls_are_recovered_by_default() -> None:
    assert DEFAULT_RECOVER_LOST_TOOL_CALLS is True
    assert AgentConfig(name="a", model="m").recover_lost_tool_calls is True


# ---------------------------------------------------------------------------
# A closing quote one bracket early
# ---------------------------------------------------------------------------


def test_a_quote_before_the_last_bracket_is_moved_past_it() -> None:
    how, value = tc.read_call_body(
        '{"name": "python_exec", "arguments": {"code": "print(sum([1, 2])")}\n}'
    )
    assert how == "closing quote moved"
    assert value["arguments"]["code"] == "print(sum([1, 2]))"
    # A body that ends any other way is not touched.
    assert tc.read_call_body('{"name": "python_exec", "arguments": {"code": "x"') is None


@pytest.mark.parametrize("label", ["run", "stream"])
def test_the_misplaced_quote_call_runs_at_the_default(
    label: str, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    turns = [MISPLACED, "Final Answer: 2"]
    for name, response, _model, tools in _both(turns, lambda: [_py()]):
        if name != label:
            continue
        assert tools[0].inputs == [{"code": "print(sum([1, 1]))"}]
        assert response.output.strip() == "2"
    assert _lines(caplog, "(closing quote moved)")


def test_the_misplaced_quote_call_is_not_read_when_recovery_is_off() -> None:
    for _name, _response, _model, tools in _both(
        [MISPLACED, "Final Answer: 2"], lambda: [_py()], recover_lost_tool_calls=False,
    ):
        assert tools[0].inputs == []


INNER_QUOTES = (
    '<tool_call>\n{"name": "python_exec", "arguments": {"code": "import math\\n'
    'print(format(math.pi, ".2f"))"}}\n</tool_call>'
)


def test_unescaped_quotes_inside_a_string_value_are_read(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    for _name, _response, _model, tools in _both([INNER_QUOTES, "Final Answer: 3.14"], lambda: [_py()]):
        assert tools[0].inputs == [{"code": 'import math\nprint(format(math.pi, ".2f"))'}]
    assert _lines(caplog, "(inner quotes escaped)")
    # A quote followed by a comma reads as the end of the string, so this body stays unread.
    assert tc.read_call_body('{"name": "python_exec", "arguments": {"code": "print("a", "b")"}}') is None


def test_a_whole_call_object_with_text_after_it_inside_the_tag_is_read(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    turn = ('<tool_call>\n{"name": "python_exec", "arguments": {"code": "print(2)"}}\n'
            "This prints the value.\n</tool_call>")
    for _name, _response, _model, tools in _both([turn, "Final Answer: 2"], lambda: [_py()]):
        assert tools[0].inputs == [{"code": "print(2)"}]
    assert _lines(caplog, "(object followed by text)")


# ---------------------------------------------------------------------------
# Arguments sent as a string
# ---------------------------------------------------------------------------


STRING_ARGUMENTS = [
    (json.dumps("code='print(1+1)'"), "keywords"),
    (json.dumps("print(1+1)"), "raw value"),
    (json.dumps(json.dumps({"code": "print(1+1)"})), "object text"),
]


@pytest.mark.parametrize(("arguments", "kind"), STRING_ARGUMENTS)
def test_the_decoder_reads_a_string_argument(arguments: str, kind: str) -> None:
    tool = _py()
    decoded = tc.read_call_arguments(arguments, tool)
    if kind == "raw value":
        assert decoded == {"__raw_input__": "print(1+1)"}
    else:
        assert decoded == {"code": "print(1+1)"}


def test_the_decoder_never_returns_a_non_string_raw_value_or_a_non_mapping() -> None:
    assert tc.read_call_arguments("42") == {"__raw_input__": "42"}
    assert tc.read_call_arguments("[1, 2]") == {"__raw_input__": "[1, 2]"}
    assert tc.read_call_arguments(None) == {}
    assert tc.read_call_arguments("") == {}
    assert tc.read_call_arguments("null") == {}
    assert tc.read_call_arguments({"a": 1}) == {"a": 1}
    assert tc.read_call_arguments('{"a": 1}') == {"a": 1}
    # Keyword syntax is only read for parameters the tool declares.
    assert tc.read_call_arguments(json.dumps("x = 5"), _py()) == {"__raw_input__": "x = 5"}


@pytest.mark.parametrize(("arguments", "kind"), STRING_ARGUMENTS)
def test_a_string_argument_is_dispatched_on_a_single_call(
    arguments: str, kind: str, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    turns = [_native("python_exec", arguments), "Final Answer: 2"]
    for _name, _response, _model, tools in _both(turns, lambda: [_py()]):
        assert tools[0].inputs == [{"code": "print(1+1)"}]
    assert _lines(caplog, f"[call] read arguments sent as a string ({kind})")


@pytest.mark.parametrize(("arguments", "kind"), STRING_ARGUMENTS)
def test_a_string_argument_is_dispatched_in_a_batch(arguments: str, kind: str) -> None:
    turn = {"text": "", "calls": [
        {"id": "a", "type": "function", "function": {"name": "python_exec", "arguments": arguments}},
        {"id": "b", "type": "function",
         "function": {"name": "calculator", "arguments": json.dumps("6*7")}},
    ]}
    for _name, _response, _model, tools in _both(
        [turn, "Final Answer: 2"], lambda: [_py(), _calc()],
    ):
        assert tools[0].inputs == [{"code": "print(1+1)"}]
        assert tools[1].inputs == [{"expression": "6*7"}]


@pytest.mark.parametrize(("arguments", "kind"), STRING_ARGUMENTS)
def test_a_string_argument_is_dispatched_by_the_native_tool_loop(arguments: str, kind: str) -> None:
    model = _Script([_native("python_exec", arguments), "2"])
    tool = _py()
    agent = _agent(model, [tool])
    agent._run_with_gemini_native_tools(TASK, {})
    assert tool.inputs == [{"code": "print(1+1)"}]


def test_the_groq_recovery_reads_a_string_argument() -> None:
    from effgen.models.groq_adapter import _parse_failed_generation_tool_call

    entry = _parse_failed_generation_tool_call(
        '<function=python_exec>{"name": "python_exec", "arguments": "code=\'print(1)\'"}'
        "</function>"
    )
    assert entry is not None
    entry = _parse_failed_generation_tool_call('<function=python_exec>"code=\'print(1)\'"</function>')
    assert entry is not None
    assert json.loads(entry["function"]["arguments"]) == {"code": "print(1)"}


# ---------------------------------------------------------------------------
# A call missing a required argument
# ---------------------------------------------------------------------------


def test_missing_required_arguments_reads_only_mappings() -> None:
    tool = _calc()
    assert tc.missing_required_arguments(tool, "{}") == ["expression"]
    assert tc.missing_required_arguments(tool, {}) == ["expression"]
    assert tc.missing_required_arguments(tool, {"expression": None}) == ["expression"]
    assert tc.missing_required_arguments(tool, '{"expression": "1+1"}') == []
    assert tc.missing_required_arguments(tool, "1+1") == []
    assert tc.missing_required_arguments(tool, {"__raw_input__": "1+1"}) == []
    assert tc.missing_required_arguments(_NoRequired(), "{}") == []


@pytest.mark.parametrize("label", ["run", "stream"])
def test_an_empty_call_is_not_dispatched_and_a_call_is_required_next(
    label: str, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    turns = [_native("calculator", "{}"), _native("calculator", '{"expression": "6*7"}'),
             "Final Answer: 42"]
    for name, response, model, tools in _both(turns, lambda: [_calc()]):
        if name != label:
            continue
        assert tools[0].inputs == [{"expression": "6*7"}]
        assert model.kwargs[1].get("tool_choice") == "required"
        assert response.output.strip() == "42"
        assert not any("Parameter validation failed" in str(c.result)
                       for c in response.tool_calls)
    assert _lines(caplog, "[call] not dispatched: 'calculator' is missing its required argument(s) expression")


def test_a_second_empty_call_is_answered_with_what_the_tool_requires() -> None:
    turns = [_native("calculator", "{}"), _native("calculator", '{"expression": null}'),
             "Final Answer: 42"]
    for _name, response, model, tools in _both(turns, lambda: [_calc()]):
        assert tools[0].inputs == []
        assert response.output.strip() == "42"
        assert any("parameter 'expression' is required" in p for p in model.prompts[2:])


def test_a_tool_with_no_required_parameter_still_gets_an_empty_call() -> None:
    turns = [_native("clock", "{}"), "Final Answer: noon"]
    for _name, _response, _model, tools in _both(turns, lambda: [_NoRequired()]):
        assert tools[0].inputs == [{}]


def test_an_empty_call_in_a_batch_is_not_dispatched() -> None:
    turn = {"text": "", "calls": [
        {"id": "a", "type": "function", "function": {"name": "calculator", "arguments": "{}"}},
        {"id": "b", "type": "function", "function": {"name": "python_exec",
                                                    "arguments": '{"code": "print(2)"}'}},
    ]}
    for _name, response, _model, tools in _both([turn, "Final Answer: 2"], lambda: [_calc(), _py()]):
        assert tools[0].inputs == []
        assert tools[1].inputs == [{"code": "print(2)"}]
        assert [c.name for c in response.tool_calls] == ["python_exec"]


# ---------------------------------------------------------------------------
# Positional values
# ---------------------------------------------------------------------------


def test_positional_values_are_named_or_the_call_carries_none() -> None:
    tools = {"calculator": _calc()}
    assert tc.name_positional_arguments("calculator", [2, 3, 4, 6], tools) == {}
    assert tc.name_positional_arguments("lcm", [2, 3], tools) == {}
    assert tc.parse_call_syntax("calculator(42)") == ("calculator", {"__raw_input__": "42"}, [])
    assert tc.parse_call_syntax('calculator("6*7")') == ("calculator", {"__raw_input__": "6*7"}, [])


@pytest.mark.parametrize("action", ["calculator(2, 3, 4, 6)", "lcm(2, 3, 4, 6)", "lcm(12)"])
def test_several_positional_values_never_crash_the_run(action: str) -> None:
    turns = [f"Thought: need the lcm\nAction: {action}", "Final Answer: 12"]
    for _name, response, model, tools in _both(turns, lambda: [_calc()]):
        assert "has no attribute 'strip'" not in str(response.output)
        assert response.output.strip() == "12"
        assert tools[0].inputs == []
        assert len(model.prompts) == 2


def test_a_single_positional_value_still_dispatches() -> None:
    turns = ['Thought: compute\nAction: calculator("6*7")', "Final Answer: 42"]
    for _name, _response, _model, tools in _both(turns, lambda: [_calc()]):
        assert tools[0].inputs == [{"expression": "6*7"}]


# ---------------------------------------------------------------------------
# A fenced program before an empty call tag
# ---------------------------------------------------------------------------

FENCED = "Here is the code:\n```python\nprint(1+1)\n```\nNow let's run it.\n<tool_call>\n"


@pytest.mark.parametrize("label", ["run", "stream"])
def test_the_fenced_block_before_an_empty_call_runs_once(
    label: str, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    for name, response, model, tools in _both([FENCED, "Final Answer: 2"], lambda: [_py()]):
        if name != label:
            continue
        assert tools[0].inputs == [{"code": "print(1+1)"}]
        assert len(model.prompts) == 2
        assert response.output.strip() == "2"
    assert _lines(caplog, "[call] ran the fenced block before an empty call to 'python_exec'")


@pytest.mark.parametrize(("text", "tools_factory"), [
    (FENCED, lambda: [_py(), _Tool("shell", CODE, param="command")]),
    (FENCED, lambda: [_calc()]),
    ("Now let's run it.\n<tool_call>\n```python\nprint(1+1)\n```\n", lambda: [_py()]),
    ("Here is the code:\n```python\nprint(1+1)\n```\n", lambda: [_py()]),
])
def test_the_fenced_block_is_not_run_otherwise(text: str, tools_factory: Any) -> None:
    tools = {t.name: t for t in tools_factory()}
    assert tc.fenced_block_call(text, tools) is None


def test_the_fenced_block_is_not_run_for_a_call_that_names_another_tool() -> None:
    """An unread call naming a different held tool is that tool's call, not the program's."""
    turn = FENCED + '{"name": "calculator", "arguments": {"expression": "6*7"'
    tools = {t.name: t for t in (_py(), _calc())}
    assert tc.fenced_block_call(turn, tools) is None
    for _name, _response, _model, held in _both(
        [turn, "Final Answer: 42"], lambda: [_py(), _calc()],
    ):
        assert held[0].inputs == []
    # the same unread call naming the code tool itself still runs the block
    own = FENCED + '{"name": "python_exec", "arguments": {"code": "print(1+'
    assert tc.fenced_block_call(own, tools) == ("python_exec", {"code": "print(1+1)"})


def test_the_fenced_block_is_not_run_when_recovery_is_off() -> None:
    for _name, _response, _model, tools in _both(
        [FENCED, "Final Answer: 2"], lambda: [_py()], recover_lost_tool_calls=False,
    ):
        assert tools[0].inputs == []


def test_the_capability_probe_keeps_the_strict_reader() -> None:
    import inspect

    from effgen.models import capability_probe

    assert "recover_lost_tool_calls=False" in inspect.getsource(capability_probe)
