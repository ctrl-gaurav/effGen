"""Whether a run has to call the tool it is holding is a value, not a wording.

Three things are pinned here and they fail separately.

**Every category declares a policy.** ``TOOL_USE_POLICIES`` covers every member
of ``ToolCategory``, so a category added later fails this file rather than
inheriting a default nobody chose, and "which tools must actually run" is read
from the same table that says so.

**A prompt with no policy stated is the prompt that was already sent.**
``REQUIRED`` and ``AUTO`` add nothing at all, and the shipped defaults are those
two, so attaching a calculator or a code executor produces the text it always
produced. Only ``SPARING`` adds a sentence, and a caller who asked for silence
with ``tool_contract=""`` keeps silence whatever the policy is.

**The caller can say it either way.** ``tool_use="required"`` makes a tool whose
category does not ask for a call one the run may not answer without -- which is
how "use the calculator every time" is expressed -- and ``tool_use="auto"``
stops the framework pushing for one that does.
"""
from __future__ import annotations

import logging

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.prompts.tool_contract import (
    TOOL_CONTRACT_EXECUTE,
    TOOL_CONTRACT_VERIFY,
    TOOL_CONTRACTS,
    TOOL_USE_POLICIES,
    TOOL_USE_SPARING_NOTE,
    ToolUsePolicy,
    coerce_tool_use_policy,
    is_execution_tool,
    policy_for_category,
    select_tool_use_policy,
)
from effgen.tools.base_tool import ToolCategory

from .test_forced_tool_call import Recorder, forced_turns, make_tool

REFUSAL = "Thought: I can do that in my head.\nFinal Answer: 42"
CALL_THEN_ANSWER = [
    'Thought: computing.\nAction: calculator\nAction Input: {"code": "6*7"}',
    "Thought: done.\nFinal Answer: 42",
]


def build(turns, tools, **cfg):
    model = Recorder(turns)
    agent = Agent(config=AgentConfig(
        name="probe", model=model, tools=tools, max_iterations=4,
        raise_on_error=False, **cfg,
    ))
    return agent, model


# ---------------------------------------------------------------------------
# Every category declares a policy
# ---------------------------------------------------------------------------
def test_every_category_declares_a_tool_use_policy():
    """A category added later fails here rather than inheriting a default."""
    assert set(TOOL_USE_POLICIES) == set(ToolCategory)


@pytest.mark.parametrize("category", list(ToolCategory))
def test_execution_tools_are_the_ones_whose_policy_requires_a_call(category):
    """One table answers "must this run" and "is this an execution tool"."""
    expected = TOOL_USE_POLICIES[category] is ToolUsePolicy.REQUIRED
    assert is_execution_tool(make_tool("t", category)) is expected


@pytest.mark.parametrize("category", list(ToolCategory))
def test_the_execute_contract_and_the_required_policy_name_the_same_tools(category):
    """The text that says "run it" and the policy that requires it agree."""
    told_to_run = TOOL_CONTRACTS[category] is TOOL_CONTRACT_EXECUTE
    must_run = TOOL_USE_POLICIES[category] is ToolUsePolicy.REQUIRED
    assert told_to_run == must_run


def test_an_unknown_category_is_not_required_to_run():
    """Fail closed: a tool nobody declared is not one the model may not skip."""
    assert policy_for_category(None) is ToolUsePolicy.AUTO
    assert policy_for_category("something_else") is ToolUsePolicy.AUTO
    assert is_execution_tool(object()) is False


def test_a_mixed_set_takes_the_strictest_policy_present():
    """Telling a model it may skip the executor it also holds is the costly half."""
    calc = make_tool("calculator", ToolCategory.COMPUTATION)
    runner = make_tool("python_exec", ToolCategory.CODE_EXECUTION)
    assert select_tool_use_policy([calc]) is ToolUsePolicy.AUTO
    assert select_tool_use_policy([runner]) is ToolUsePolicy.REQUIRED
    assert select_tool_use_policy([calc, runner]) is ToolUsePolicy.REQUIRED
    assert select_tool_use_policy([]) is ToolUsePolicy.AUTO


@pytest.mark.parametrize("given,expected", [
    (None, None),
    ("required", ToolUsePolicy.REQUIRED),
    ("SPARING", ToolUsePolicy.SPARING),
    ("  Auto ", ToolUsePolicy.AUTO),
    (ToolUsePolicy.SPARING, ToolUsePolicy.SPARING),
])
def test_a_policy_can_be_named_by_its_string(given, expected):
    assert coerce_tool_use_policy(given) is expected


@pytest.mark.parametrize("given", ["never", "", 3, object()])
def test_a_policy_that_names_nothing_is_refused(given):
    """A silently ignored policy sends the run out on one nobody chose."""
    with pytest.raises(ValueError, match="unknown tool_use policy"):
        coerce_tool_use_policy(given)


# ---------------------------------------------------------------------------
# A prompt with no policy stated is the prompt that was already sent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", list(ToolCategory))
def test_a_default_agent_states_no_policy_sentence(category):
    agent, _ = build([REFUSAL], [make_tool("t", category)])
    assert TOOL_USE_SPARING_NOTE not in agent._tool_contract()


@pytest.mark.parametrize("policy", ["required", "auto"])
def test_required_and_auto_add_nothing_to_the_prompt(policy):
    plain, _ = build([REFUSAL], [make_tool("calculator", ToolCategory.COMPUTATION)])
    stated, _ = build([REFUSAL], [make_tool("calculator", ToolCategory.COMPUTATION)],
                      tool_use=policy)
    assert stated._tool_contract() == plain._tool_contract() == TOOL_CONTRACT_VERIFY


def test_sparing_adds_one_sentence_after_the_contract():
    agent, _ = build([REFUSAL], [make_tool("calculator", ToolCategory.COMPUTATION)],
                     tool_use="sparing")
    contract = agent._tool_contract()
    assert contract == f"{TOOL_CONTRACT_VERIFY} {TOOL_USE_SPARING_NOTE}"


def test_sparing_states_nothing_when_the_caller_asked_for_silence():
    """``tool_contract=""`` keeps its published meaning under any policy."""
    agent, _ = build([REFUSAL], [make_tool("calculator", ToolCategory.COMPUTATION)],
                     tool_contract="", tool_use="sparing")
    assert agent._tool_contract() == ""


def test_sparing_follows_a_contract_the_caller_supplied():
    """The two settings are independent: one picks the words, one the policy."""
    agent, _ = build([REFUSAL], [make_tool("calculator", ToolCategory.COMPUTATION)],
                     tool_contract="Do it my way.", tool_use="sparing")
    assert agent._tool_contract() == f"Do it my way. {TOOL_USE_SPARING_NOTE}"


def test_an_agent_with_no_tools_states_no_policy_sentence():
    agent, _ = build([REFUSAL], [], tool_use="sparing")
    assert agent._tool_contract() == ""


# ---------------------------------------------------------------------------
# The caller can say it either way
# ---------------------------------------------------------------------------
def test_which_tools_must_run_defaults_to_what_the_tools_declare():
    tools = {"calculator": make_tool("calculator", ToolCategory.COMPUTATION),
             "python_exec": make_tool("python_exec", ToolCategory.CODE_EXECUTION)}
    assert NativeToolLoop(tools).execution_tools() == ["python_exec"]


def test_a_required_policy_puts_every_held_tool_in_the_set():
    tools = {"calculator": make_tool("calculator", ToolCategory.COMPUTATION)}
    loop = NativeToolLoop(tools, tool_use=ToolUsePolicy.REQUIRED)
    assert loop.execution_tools() == ["calculator"]


@pytest.mark.parametrize("policy", [ToolUsePolicy.AUTO, ToolUsePolicy.SPARING])
def test_a_caller_can_stop_the_framework_requiring_a_call(policy):
    tools = {"python_exec": make_tool("python_exec", ToolCategory.CODE_EXECUTION)}
    assert NativeToolLoop(tools, tool_use=policy).execution_tools() == []


def test_a_calculator_agent_is_accepted_with_no_call_by_default():
    """Today's behaviour, now the stated default rather than an implication."""
    agent, model = build([REFUSAL], [make_tool("calculator", ToolCategory.COMPUTATION)])
    agent.run("What is 6 times 7?")
    assert len(model.seen) == 1


def test_a_required_calculator_agent_that_never_called_is_sent_back():
    """"Use the tool every time" for a tool whose category does not ask for it."""
    agent, model = build(
        [REFUSAL, *CALL_THEN_ANSWER],
        [make_tool("calculator", ToolCategory.COMPUTATION)],
        tool_use="required",
    )
    agent.run("What is 6 times 7?")
    assert len(model.seen) >= 2, "the answer was accepted with no call"
    assert forced_turns(model) == [1], "the turn after the refusal was not constrained"


def test_an_auto_executor_agent_is_not_sent_back():
    """A caller who says the model decides is not overruled by the category."""
    agent, model = build(
        [REFUSAL],
        [make_tool("python_exec", ToolCategory.CODE_EXECUTION)],
        tool_use="auto",
    )
    agent.run("Compute the 30th Fibonacci number.")
    assert len(model.seen) == 1
    assert forced_turns(model) == []


def test_a_policy_that_names_nothing_is_refused_at_construction():
    """A typo names the three values now, not on the first run."""
    with pytest.raises(ValueError, match="unknown tool_use policy"):
        AgentConfig(name="p", model="x", tool_use="whenever")


# ---------------------------------------------------------------------------
# The policy is greppable
# ---------------------------------------------------------------------------
def test_a_run_says_which_policy_it_is_on(caplog):
    """The firing count comes off this line, so it is not optional."""
    with caplog.at_level(logging.INFO, logger="effgen.core.agent_tool_loop"):
        NativeToolLoop({"calculator": make_tool("calculator", ToolCategory.COMPUTATION)})
        NativeToolLoop({"calculator": make_tool("calculator", ToolCategory.COMPUTATION)},
                       tool_use=ToolUsePolicy.SPARING)
    lines = [r.getMessage() for r in caplog.records if "tool use policy:" in r.getMessage()]
    assert len(lines) == 2
    assert "from tools" in lines[0]
    assert "sparing" in lines[1]
