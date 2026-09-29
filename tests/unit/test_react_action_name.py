"""The tool name in an ``Action:`` line stops at the arguments that follow it.

Models regularly put a whole ReAct step on one line —
``Action: calculator | Action Input: {"expression": "1367 * 89"}`` — and reading
to the end of the line makes the tool name the entire remainder, which matches
no registered tool. The loop then tells the model it has no tools, and the model
answers from memory.

Two parsers read this text: ``ReActStrategy.parse_response`` serves ``run()``
and ``AgentReActMixin._parse_react_response`` serves ``stream()``. The same
table runs against both, so the two can never disagree about the same output.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import (
    NUDGE_NO_TOOLS,
    sanitize_final_answer,
    unknown_tool_observation,
)
from effgen.core.tool_calling import HybridStrategy, ReActStrategy, action_name
from effgen.tools import get_registry
from tests.fixtures.mock_models import MockModel

# (label, model text, expected tool name)
ACTION_SHAPES = [
    (
        "same line, pipe separator",
        (
            'Thought: I will use the calculator tool.\n\n'
            'Action: calculator | Action Input: {"expression": "1367 * 89"}'
        ),
        "calculator",
    ),
    (
        "same line, comma separator",
        'Action: calculator, Action Input: {"expression": "2 + 2"}',
        "calculator",
    ),
    (
        "same line, semicolon separator",
        'Action: calculator; Action Input: {"expression": "2 + 2"}',
        "calculator",
    ),
    (
        "same line, no separator",
        'Action: calculator Action Input: {"expression": "2 + 2"}',
        "calculator",
    ),
    (
        "same line, bare Input label",
        'Action: calculator - Input: {"expression": "2 + 2"}',
        "calculator",
    ),
    (
        "same line, Args label",
        'Action: calculator Args: {"expression": "2 + 2"}',
        "calculator",
    ),
    (
        "same line, Parameters label",
        'Action: calculator Parameters: {"expression": "2 + 2"}',
        "calculator",
    ),
    (
        "plain two-line shape",
        'Thought: use it\nAction: calculator\nAction Input: {"expression": "2 + 2"}',
        "calculator",
    ),
    (
        "function-call shape",
        'Action: calculator(expression="2 + 2")',
        "calculator",
    ),
    (
        "function-call whose argument is named 'input'",
        'Action: calculator(input="1367 * 89")',
        "calculator",
    ),
    (
        "function-call whose argument is named 'args'",
        'Action: calculator(args={"expression": "2 + 2"})',
        "calculator",
    ),
    (
        "function-call with a second argument named 'params'",
        "Action: web_search(query=weather, params=none)",
        "web_search",
    ),
    (
        "function-call followed by a same-line argument section",
        'Action: calculator(expression="2 + 2") | Action Input: {}',
        "calculator",
    ),
    (
        "tool name containing 'input'",
        'Action: read_input_file\nAction Input: {"path": "a.txt"}',
        "read_input_file",
    ),
    (
        "tool name containing 'args'",
        'Action: list_args\nAction Input: {"path": "a.txt"}',
        "list_args",
    ),
]


@pytest.fixture
def agent():
    model = MockModel(responses=["Final Answer: dummy"])
    return Agent(
        config=AgentConfig(
            name="action-name-test",
            model=model,
            tools=[],
            enable_memory=False,
            enable_sub_agents=False,
        )
    )


@pytest.mark.parametrize("label,text,expected", ACTION_SHAPES, ids=[c[0] for c in ACTION_SHAPES])
class TestActionNameAcrossParsers:

    def test_react_strategy(self, label, text, expected):
        assert ReActStrategy().parse_response(text).tool_name == expected

    def test_hybrid_strategy(self, label, text, expected):
        assert HybridStrategy().parse_response(text).tool_name == expected

    def test_mixin_parser(self, label, text, expected, agent):
        assert agent._parse_react_response(text)["action"] == expected

    def test_both_parsers_agree(self, label, text, expected, agent):
        assert (
            ReActStrategy().parse_response(text).tool_name
            == agent._parse_react_response(text)["action"]
        )


class TestArgumentsSurviveTheTrim:
    """Only the name was ever polluted; the arguments already parsed correctly."""

    def test_same_line_arguments_still_parse(self):
        text = 'Action: calculator | Action Input: {"expression": "1367 * 89"}'
        assert ReActStrategy().parse_response(text).arguments == {"expression": "1367 * 89"}

    def test_two_line_arguments_still_parse(self):
        text = 'Action: calculator\nAction Input: {"expression": "1367 * 89"}'
        assert ReActStrategy().parse_response(text).arguments == {"expression": "1367 * 89"}

    @pytest.mark.parametrize("label", ["Args", "Arguments"])
    def test_an_args_label_supplies_the_arguments_too(self, label, agent):
        """The name is cut at this label, so the arguments come from it."""
        text = f'Action: calculator\n{label}: {{"expression": "1367 * 89"}}'
        assert ReActStrategy().parse_response(text).arguments == {"expression": "1367 * 89"}
        # The mixin parser hands the raw text on; the loop parses it downstream.
        assert agent._parse_react_response(text)["action_input"] == (
            '{"expression": "1367 * 89"}'
        )


class TestBracketedArgumentsDoNotCutTheName:
    """A label inside a call's own argument list is not the end of the name."""

    @pytest.mark.parametrize(
        "raw",
        [
            'calculator(input="1367 * 89")',
            'calculator(args={"expression": "2 + 2"})',
            "web_search(query=weather, params=none)",
            '{"name": "calculator", "arguments": {"expression": "2 + 2"}}',
        ],
    )
    def test_the_construct_is_returned_whole(self, raw):
        assert action_name(raw) == raw

    def test_a_label_after_the_brackets_still_cuts(self):
        assert action_name('calculator(expression="2 + 2") | Action Input: {}') == (
            'calculator(expression="2 + 2")'
        )


class TestActionNameHelper:

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("calculator", "calculator"),
            ('calculator | Action Input: {"a": 1}', "calculator"),
            ("calculator Input: 2 + 2", "calculator"),
            ("calculator args = {}", "calculator"),
            ("read_input_file", "read_input_file"),
            ("input_reader", "input_reader"),
            ("list_args", "list_args"),
            ("", ""),
        ],
    )
    def test_trims_only_the_argument_section(self, raw, expected):
        assert action_name(raw) == expected


class TestUnknownToolObservation:
    """An action that names no held tool is reported as that, not as no tools."""

    def test_names_the_action_and_lists_the_tools(self):
        text = unknown_tool_observation("weather_lookup", ["calculator", "wikipedia"])
        assert "weather_lookup" in text
        assert "calculator, wikipedia" in text
        assert "No tools available" not in text

    def test_loop_lists_the_tools_it_holds(self):
        calc = get_registry().get_tool_sync("calculator")
        model = MockModel(
            responses=[
                'Thought: check\nAction: weather_lookup\nAction Input: {"city": "Paris"}',
                "Final Answer: done",
            ]
        )
        agent = Agent(
            config=AgentConfig(
                name="unknown-tool",
                model=model,
                tools=[calc],
                enable_memory=False,
                enable_sub_agents=False,
                max_iterations=3,
            )
        )
        seen: list[str] = []
        original = agent._generate
        agent._generate = lambda p, **kw: (seen.append(p), original(p, **kw))[1]
        agent.run("Look up the weather in Paris.")
        prompts = "\n".join(seen)
        assert "No tool named 'weather_lookup' is available" in prompts
        assert "The tools you can use are: calculator" in prompts
        assert "No tools available" not in prompts

    def test_an_agent_with_no_tools_answers_directly(self):
        """No tools means direct inference, so no observation of either kind."""
        model = MockModel(responses=["Paris is rainy today."])
        agent = Agent(
            config=AgentConfig(
                name="no-tools",
                model=model,
                tools=[],
                enable_memory=False,
                enable_sub_agents=False,
                max_iterations=3,
            )
        )
        seen: list[str] = []
        original = agent._generate
        agent._generate = lambda p, **kw: (seen.append(p), original(p, **kw))[1]
        agent.run("Look up the weather in Paris.")
        prompts = "\n".join(seen)
        assert NUDGE_NO_TOOLS not in prompts
        assert "No tool named" not in prompts

    def test_tools_lost_mid_run_fall_back_to_the_tool_less_nudge(self):
        """The loop only reaches the tool-less branch if the tools go away."""
        calc = get_registry().get_tool_sync("calculator")
        model = MockModel(
            responses=[
                'Thought: check\nAction: weather_lookup\nAction Input: {"city": "Paris"}',
                "Final Answer: done",
            ]
        )
        agent = Agent(
            config=AgentConfig(
                name="tools-vanish",
                model=model,
                tools=[calc],
                enable_memory=False,
                enable_sub_agents=False,
                max_iterations=3,
            )
        )
        seen: list[str] = []
        original = agent._generate

        def record(prompt, **kwargs):
            seen.append(prompt)
            agent.tools = {}
            return original(prompt, **kwargs)

        agent._generate = record
        agent.run("Look up the weather in Paris.")
        prompts = "\n".join(seen)
        assert NUDGE_NO_TOOLS in prompts
        assert "No tool named" not in prompts

    def test_the_observation_cannot_reach_an_answer(self):
        observation = unknown_tool_observation("weather_lookup", ["calculator", "wikipedia"])
        cleaned = sanitize_final_answer(f"The answer is 42.\nObservation: {observation}")
        assert "No tool named" not in cleaned
        assert "The tools you can use are" not in cleaned
        assert cleaned.strip().startswith("The answer is 42.")


# ---------------------------------------------------------------------------
# Arguments written in call syntax reach the tool
# ---------------------------------------------------------------------------
#
# ``action_name`` recovered the name from these shapes; the arguments were
# dropped and the tool ran with ``{}``. It refused the empty set, the loop spent
# a turn on the refusal, and the model tried again.

# (label, Action: text, expected arguments dict)
CALL_SYNTAX_SHAPES = [
    (
        "double-quoted keyword",
        'Action: calculator(expression="1367 * 89")',
        {"expression": "1367 * 89"},
    ),
    (
        "single-quoted keyword",
        "Action: calculator(expression='1367 * 89')",
        {"expression": "1367 * 89"},
    ),
    (
        "keyword named like a label",
        'Action: calculator(input="1367 * 89")',
        {"input": "1367 * 89"},
    ),
    (
        "a JSON object as the only argument",
        'Action: calculator({"expression": "1367 * 89"})',
        {"expression": "1367 * 89"},
    ),
    (
        "a bare JSON object after the name",
        'Action: calculator {"expression": "1367 * 89"}',
        {"expression": "1367 * 89"},
    ),
]


@pytest.mark.parametrize(
    "label,text,expected", CALL_SYNTAX_SHAPES, ids=[c[0] for c in CALL_SYNTAX_SHAPES]
)
class TestCallSyntaxArgumentsAcrossParsers:
    def test_react_strategy(self, label, text, expected):
        assert ReActStrategy().parse_response(text).arguments == expected

    def test_hybrid_strategy(self, label, text, expected):
        assert HybridStrategy().parse_response(text).arguments == expected

    def test_mixin_parser(self, label, text, expected, agent):
        import json

        parsed = agent._parse_react_response(text)
        assert json.loads(parsed["action_input"]) == expected


class TestPositionalCallArguments:
    """A positional call carries its values in the tool's own parameter order."""

    def test_a_single_positional_becomes_raw_input(self):
        result = ReActStrategy().parse_response('Action: calculator("1367 * 89")')
        assert result.tool_name == "calculator"
        assert result.arguments == {"__raw_input__": "1367 * 89"}

    def test_several_positionals_take_the_declared_parameter_names(self):
        tools = {name: get_registry().get_tool_sync(name) for name in ("file_operations",)}
        result = ReActStrategy().parse_response(
            'Action: file_operations("write", "greet.py", "print(1)")', tools,
        )
        assert result.tool_name == "file_operations"
        assert result.arguments["operation"] == "write"
        assert result.arguments["path"] == "greet.py"
        assert result.arguments["content"] == "print(1)"

    def test_without_the_tool_the_values_are_not_guessed_at(self):
        """No schema to name them by, so the call carries no arguments rather
        than running a different call with only the first value."""
        result = ReActStrategy().parse_response('Action: unknown_tool("a", "b")')
        assert result.arguments == {}


class TestCallSyntaxIsNotOverEager:
    """Prose and JSON answers must not be read as calls."""

    def test_a_json_answer_is_not_a_call(self):
        from effgen.core.tool_calling import parse_call_syntax

        assert parse_call_syntax('{"name": "Acme Corp", "revenue": 5}') is None

    def test_a_plain_name_is_not_a_call(self):
        from effgen.core.tool_calling import parse_call_syntax

        assert parse_call_syntax("calculator") is None

    def test_an_expression_that_is_not_a_literal_is_refused(self):
        """``ast.literal_eval`` is the only reader, so nothing is executed."""
        from effgen.core.tool_calling import parse_call_syntax

        assert parse_call_syntax('calculator(expression=__import__("os").system)') is None
