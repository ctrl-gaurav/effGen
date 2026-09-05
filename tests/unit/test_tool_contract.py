"""What an agent tells a model about the tools it holds.

The framework attaches tool definitions; the model learns about them only from
what the prompt says. Three things are pinned here.

* The text is chosen from the tools' declared ``ToolCategory`` and from nothing
  else -- not the model, not the task, not the provider. Every category has an
  explicit entry, so one added later fails a test rather than inheriting a
  wording written for something else.
* A set of tools that maps to more than one contract gets the default. A
  specific contract asserts something about every tool the model holds, and on
  a mixed set it would be false of some of them.
* The caller wins. A persona still leads the prompt; a caller-supplied contract
  is stated verbatim; an empty one states nothing.
"""

from __future__ import annotations

import logging

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.prompts.tool_contract import (
    CONTRACT_NAMES,
    TOOL_CONTRACT_EXECUTE,
    TOOL_CONTRACT_GENERAL,
    TOOL_CONTRACT_LOOKUP,
    TOOL_CONTRACT_VERIFY,
    TOOL_CONTRACTS,
    contract_for_category,
    select_tool_contract,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)


def _tool(name: str, category) -> BaseTool:
    class _T(BaseTool):
        def __init__(self) -> None:
            super().__init__(metadata=ToolMetadata(
                name=name, description=f"The {name} tool.", category=category,
                parameters=[ParameterSpec(
                    name="query", type=ParameterType.STRING,
                    description="Input.", required=True)],
            ))

        async def _execute(self, **kw):
            return "42"
    return _T()


CALC = lambda: _tool("calculator", ToolCategory.COMPUTATION)          # noqa: E731
EXEC = lambda: _tool("python_exec", ToolCategory.CODE_EXECUTION)      # noqa: E731
SEARCH = lambda: _tool("web_search", ToolCategory.INFORMATION_RETRIEVAL)  # noqa: E731


class _Scripted(BaseModel):
    """Records every prompt it is handed and answers from a script."""

    _supports_tools = False
    _support_kind = "none"

    def __init__(self, turns) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self.turns, self.i, self.prompts = list(turns), 0, []

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def generate(self, prompt, config=None, **kw):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        text = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return GenerationResult(text=text, tokens_used=5, finish_reason="stop",
                                model_name="scripted", metadata={})

    def generate_stream(self, prompt, config=None, **kw):
        yield self.generate(prompt, config, **kw).text

    def count_tokens(self, t):
        return TokenCount(count=len(t.split()), model_name="scripted")

    def get_context_length(self) -> int:
        return 8192

    def generate_batch(self, ps, config=None, **kw):
        return [self.generate(p) for p in ps]

    def generate_with_tools(self, p, tools, config=None, **kw):
        return self.generate(p, config)

    def supports_function_calling(self) -> bool:
        return self._supports_tools

    def supports_tool_calling(self) -> bool:
        return self._supports_tools

    def tool_call_support(self) -> str:
        return self._support_kind


class _NativeApi(_Scripted):
    """A provider whose definitions travel through its tool-calling API."""

    _supports_tools = True
    _support_kind = "api"


ANSWER = "Thought: done.\nFinal Answer: 42"
CALL_TURN = (
    'Thought: check.\nAction: calculator\nAction Input: {"query": "6*7"}'
)


def _prompts(tools, *, model_cls=_NativeApi, mode="auto", stream=False, **cfg):
    model = model_cls([ANSWER])
    agent = Agent(config=AgentConfig(
        name="contract", model=model, tools=tools, tool_calling_mode=mode,
        max_iterations=2, raise_on_error=False, **cfg,
    ))
    if stream:
        list(agent.stream("What is 6*7?"))
    else:
        agent.run("What is 6*7?")
    return model.prompts


# --------------------------------------------------------------------------- #
# The mapping
# --------------------------------------------------------------------------- #
class TestTheMappingCoversTheTaxonomy:
    def test_every_declared_category_has_an_entry(self):
        """A category added later must fail here, not inherit silently."""
        assert set(TOOL_CONTRACTS) == set(ToolCategory)

    def test_every_entry_is_one_of_the_four_texts(self):
        four = {TOOL_CONTRACT_GENERAL, TOOL_CONTRACT_VERIFY,
                TOOL_CONTRACT_EXECUTE, TOOL_CONTRACT_LOOKUP}
        assert set(TOOL_CONTRACTS.values()) <= four

    def test_the_four_texts_are_distinct_and_named(self):
        assert len(CONTRACT_NAMES) == 4
        assert set(CONTRACT_NAMES.values()) == {"general", "verify", "execute", "lookup"}

    @pytest.mark.parametrize(("category", "expected"), [
        (ToolCategory.COMPUTATION, TOOL_CONTRACT_VERIFY),
        (ToolCategory.CODE_EXECUTION, TOOL_CONTRACT_EXECUTE),
        (ToolCategory.SYSTEM, TOOL_CONTRACT_EXECUTE),
        (ToolCategory.INFORMATION_RETRIEVAL, TOOL_CONTRACT_LOOKUP),
        (ToolCategory.EXTERNAL_API, TOOL_CONTRACT_LOOKUP),
        (ToolCategory.FILE_OPERATIONS, TOOL_CONTRACT_GENERAL),
        (ToolCategory.DATA_PROCESSING, TOOL_CONTRACT_GENERAL),
        (ToolCategory.COMMUNICATION, TOOL_CONTRACT_GENERAL),
    ])
    def test_each_category_selects_its_contract(self, category, expected):
        assert contract_for_category(category) is expected

    @pytest.mark.parametrize("category", [None, "information_retrieval", 7, object()])
    def test_an_unrecognised_category_falls_to_the_default(self, category):
        assert contract_for_category(category) is TOOL_CONTRACT_GENERAL


class TestSelectingForASetOfTools:
    def test_no_tools_selects_nothing(self):
        assert select_tool_contract([]) == ""

    def test_two_tools_of_one_category_are_not_mixed(self):
        tools = [_tool("a", ToolCategory.INFORMATION_RETRIEVAL),
                 _tool("b", ToolCategory.INFORMATION_RETRIEVAL)]
        assert select_tool_contract(tools) is TOOL_CONTRACT_LOOKUP

    def test_two_categories_that_share_a_contract_are_not_mixed(self):
        tools = [_tool("a", ToolCategory.CODE_EXECUTION),
                 _tool("b", ToolCategory.SYSTEM)]
        assert select_tool_contract(tools) is TOOL_CONTRACT_EXECUTE

    @pytest.mark.parametrize("pair", [
        (ToolCategory.COMPUTATION, ToolCategory.CODE_EXECUTION),
        (ToolCategory.INFORMATION_RETRIEVAL, ToolCategory.CODE_EXECUTION),
        (ToolCategory.COMPUTATION, ToolCategory.INFORMATION_RETRIEVAL),
    ])
    def test_a_mixed_set_falls_to_the_default(self, pair):
        tools = [_tool("a", pair[0]), _tool("b", pair[1])]
        assert select_tool_contract(tools) is TOOL_CONTRACT_GENERAL

    def test_a_tool_with_no_category_falls_to_the_default(self):
        assert select_tool_contract([_tool("mystery", None)]) is TOOL_CONTRACT_GENERAL

    def test_a_duck_typed_tool_with_no_metadata_raises_nothing(self):
        class _Bare:
            name = "bare"

        assert select_tool_contract([_Bare()]) is TOOL_CONTRACT_GENERAL

    def test_the_selection_logs_a_stable_phrase(self, caplog):
        with caplog.at_level(logging.INFO, logger="effgen.prompts.tool_contract"):
            select_tool_contract([CALC()])
        assert "tool contract: verify" in caplog.text


# --------------------------------------------------------------------------- #
# Reaching the prompt
# --------------------------------------------------------------------------- #
class TestTheContractReachesEveryPath:
    PATHS = [
        pytest.param({"mode": "auto", "stream": False}, id="native-blocking"),
        pytest.param({"mode": "react", "stream": False}, id="react-blocking"),
        pytest.param({"mode": "react", "stream": True}, id="react-stream"),
    ]

    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize(("tools_factory", "expected"), [
        pytest.param(lambda: [CALC()], TOOL_CONTRACT_VERIFY, id="computation"),
        pytest.param(lambda: [EXEC()], TOOL_CONTRACT_EXECUTE, id="code-execution"),
        pytest.param(lambda: [SEARCH()], TOOL_CONTRACT_LOOKUP, id="retrieval"),
        pytest.param(lambda: [CALC(), EXEC()], TOOL_CONTRACT_GENERAL, id="mixed"),
    ])
    def test_the_opening_prompt_states_it(self, path, tools_factory, expected):
        prompts = _prompts(tools_factory(), **path)
        assert prompts, "the run produced no prompt at all"
        assert expected in prompts[0]

    def test_an_agent_with_no_tools_states_none(self):
        prompts = _prompts([])
        assert not any(
            text in p
            for p in prompts
            for text in (TOOL_CONTRACT_VERIFY, TOOL_CONTRACT_GENERAL,
                         TOOL_CONTRACT_EXECUTE, TOOL_CONTRACT_LOOKUP)
        )

    def test_the_native_path_states_it_on_the_opening_turn_only(self):
        """Later turns close with an instruction of their own to follow."""
        model = _NativeApi([CALL_TURN, ANSWER])
        agent = Agent(config=AgentConfig(
            name="contract", model=model, tools=[CALC()],
            tool_calling_mode="react", max_iterations=3, raise_on_error=False,
        ))
        blocking = agent._native_tool_prompt("6*7", "", "", [])
        continued = agent._native_tool_prompt(
            "6*7", "Observation: 42", "", [("calculator", "{}")])
        assert TOOL_CONTRACT_VERIFY in blocking
        assert TOOL_CONTRACT_VERIFY not in continued

    def test_the_react_path_states_it_on_every_turn(self):
        """There the tool list is re-rendered each turn, so the contract is too."""
        model = _Scripted([CALL_TURN, ANSWER])
        _prompts_agent = Agent(config=AgentConfig(
            name="contract", model=model, tools=[CALC()],
            tool_calling_mode="react", max_iterations=3, raise_on_error=False,
        ))
        _prompts_agent.run("What is 6*7?")
        assert len(model.prompts) >= 2
        assert all(TOOL_CONTRACT_VERIFY in p for p in model.prompts)


class TestTheCallerWins:
    def test_a_persona_leads_and_the_contract_is_still_stated(self):
        persona = "You are a terse maths tutor. Never give the answer outright."
        prompts = _prompts([CALC()], system_prompt=persona)
        assert prompts[0].startswith(persona)
        assert TOOL_CONTRACT_VERIFY in prompts[0]

    def test_a_persona_no_longer_deletes_the_only_per_category_guidance(self):
        """Every shipped preset sets a persona, so this was all of them."""
        persona = "You are a terse maths tutor."
        for mode in ("auto", "react"):
            prompts = _prompts([CALC()], mode=mode, system_prompt=persona)
            assert TOOL_CONTRACT_VERIFY in prompts[0]

    def test_a_caller_supplied_contract_is_stated_verbatim(self):
        mine = "Call the tool exactly once and then stop."
        for mode in ("auto", "react"):
            prompts = _prompts([CALC()], mode=mode, tool_contract=mine)
            assert mine in prompts[0]
            assert TOOL_CONTRACT_VERIFY not in prompts[0]

    def test_an_empty_contract_states_nothing(self):
        prompts = _prompts([CALC()], tool_contract="")
        assert prompts[0] == "What is 6*7?"

    def test_a_caller_owned_template_gets_no_contract(self):
        """The caller owns the whole prompt, slot names included."""
        template = "TOOLS:\n{tools_description}\n{conversation_history}\nTASK: {task}\n{scratchpad}"
        prompts = _prompts([CALC()], system_prompt_template=template)
        assert TOOL_CONTRACT_VERIFY not in prompts[0]
        assert prompts[0].startswith("TOOLS:")


class TestNothingBranchesOnAModelOrATask:
    def test_the_same_tools_get_the_same_text_whatever_the_model_reports(self):
        """Capability decides the placement; the category decides the text."""
        opening = {
            support: _prompts([CALC()], model_cls=cls)[0]
            for support, cls in (("api", _NativeApi), ("none", _Scripted))
        }
        for text in opening.values():
            assert TOOL_CONTRACT_VERIFY in text

    def test_the_task_wording_does_not_change_the_contract(self):
        model_a, model_b = _NativeApi([ANSWER]), _NativeApi([ANSWER])
        for model, task in ((model_a, "What is 6*7?"),
                            (model_b, "Compute the sum of the first 40 primes.")):
            Agent(config=AgentConfig(
                name="contract", model=model, tools=[CALC()],
                max_iterations=2, raise_on_error=False,
            )).run(task)
        assert TOOL_CONTRACT_VERIFY in model_a.prompts[0]
        assert TOOL_CONTRACT_VERIFY in model_b.prompts[0]
