"""Who decides what an answer looks like.

The framework describes the machinery it inserted -- a block of retrieved
passages the model did not ask for -- and leaves the *form* of the answer to
whoever knows the question: a declared ``output_schema`` first, then the task
and the caller's system prompt. Two consequences are pinned here.

A schema the caller declared reaches the prompt the answer is written from,
instead of being discovered after prose has been written and paid for; and the
blocking loop and the native stream close a turn with the same text, so the
same agent does not get two different contracts depending on how it was called.
"""

from __future__ import annotations

import json

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import CONTEXT_ANSWER_INSTRUCTION
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from effgen.tools.builtin.calculator import Calculator

PASSAGES = "[Result 1] Sunlight is the source of energy for nearly all ecosystems."
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "description": "The option letter."}},
    "required": ["answer"],
    "additionalProperties": False,
}
#: A string that appears in the schema and nowhere else in a prompt.
SCHEMA_MARK = "The option letter."


def _retrieval_tool(name: str = "knowledge_search") -> BaseTool:
    class _T(BaseTool):
        def __init__(self) -> None:
            super().__init__(metadata=ToolMetadata(
                name=name, description="Look a topic up.",
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[ParameterSpec(
                    name="query", type=ParameterType.STRING,
                    description="Query.", required=True)],
            ))

        async def _execute(self, **kw):
            return PASSAGES
    return _T()


class _Scripted(BaseModel):
    """Replays a fixed list of turns and records every prompt it was given."""

    def __init__(self, turns, *, native: bool = False) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self.turns = list(turns)
        self.i = 0
        self.prompts: list[str] = []
        self._native = native

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def generate(self, prompt, config=None, **kw):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        text = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return GenerationResult(text=text, tokens_used=len(text.split()),
                                finish_reason="stop", model_name="scripted",
                                metadata={})

    def generate_stream(self, prompt, config=None, **kw):
        yield self.generate(prompt).text

    def count_tokens(self, t):
        return TokenCount(count=len(t.split()), model_name="scripted")

    def get_context_length(self) -> int:
        return 8192

    def generate_batch(self, ps, config=None, **kw):
        return [self.generate(p) for p in ps]

    def generate_with_tools(self, p, tools, config=None, **kw):
        return self.generate(p)

    def supports_function_calling(self) -> bool:
        return self._native

    def supports_tool_calling(self) -> bool:
        return self._native


SEARCH = ('Thought: look it up.\nAction: knowledge_search\n'
          'Action Input: {"query": "photosynthesis"}')
PROSE = ("Thought: the passages say sunlight.\nFinal Answer: Photosynthesis "
         "underpins food webs because sunlight powers nearly every ecosystem.")
JSON_ANSWER = 'Thought: done.\nFinal Answer: {"answer": "A"}'
TASK = "Question: what powers food webs?\nOptions:\nA. Sunlight.\nB. Land."


def _agent(model, tools, **cfg):
    return Agent(config=AgentConfig(
        name="shape", model=model, tools=tools, tool_calling_mode="react",
        max_iterations=6, raise_on_error=False, **cfg,
    ))


class TestTheCloseStatesNoForm:
    def test_the_retrieval_close_defers_to_the_question(self):
        low = CONTEXT_ANSWER_INSTRUCTION.lower()
        assert "source material" in low
        assert "in the form the question asks for" in low
        assert "your own sentences" not in low
        assert "do not copy" not in low


class TestDeclaredSchemaReachesTheLoop:
    def test_the_loop_prompt_carries_the_schema(self):
        model = _Scripted([SEARCH, JSON_ANSWER])
        resp = _agent(model, [_retrieval_tool()]).run(TASK, output_schema=SCHEMA)
        assert json.loads(resp.output) == {"answer": "A"}
        # Every prompt the loop built states the shape, including the opening
        # turn -- the model is told before it writes, not after.
        assert len(model.prompts) == 2
        assert all(SCHEMA_MARK in p for p in model.prompts)

    def test_a_schema_answer_costs_no_repair_call(self):
        model = _Scripted([SEARCH, JSON_ANSWER])
        resp = _agent(model, [_retrieval_tool()]).run(TASK, output_schema=SCHEMA)
        assert len(model.prompts) == 2
        assert resp.metadata["structured_output_attempts"] == 0
        assert resp.metadata["structured_output_method"] == "agent_output"

    def test_the_tool_free_path_carries_it_too(self):
        model = _Scripted([JSON_ANSWER])
        resp = _agent(model, []).run(TASK, output_schema=SCHEMA)
        assert json.loads(resp.output) == {"answer": "A"}
        assert len(model.prompts) == 1
        assert SCHEMA_MARK in model.prompts[0]
        # The framing's own cue stays the last thing the model reads.
        assert model.prompts[0].rstrip().endswith("Answer:")

    def test_the_agents_configured_schema_is_stated_as_well(self):
        model = _Scripted([JSON_ANSWER])
        _agent(model, [], output_schema=SCHEMA).run(TASK)
        assert SCHEMA_MARK in model.prompts[0]

    def test_a_run_without_a_schema_says_nothing_about_one(self):
        model = _Scripted([SEARCH, PROSE])
        _agent(model, [_retrieval_tool()]).run(TASK)
        assert not any("matching this schema" in p for p in model.prompts)

    def test_the_declared_shape_precedes_the_retrieval_close(self):
        model = _Scripted([SEARCH, JSON_ANSWER])
        _agent(model, [_retrieval_tool()]).run(TASK, output_schema=SCHEMA)
        last = model.prompts[-1]
        # Anchor on the whole close, not on a phrase inside it: the opening
        # contract for a retrieval tool says the same thing in the same words,
        # so a substring of either matches both and would not tell them apart.
        assert last.index(SCHEMA_MARK) < last.index(CONTEXT_ANSWER_INSTRUCTION)

    def test_a_schema_run_is_not_told_to_answer_before_it_may_call_a_tool(self):
        """The opening turn states the shape without demanding an answer now."""
        model = _Scripted([SEARCH, JSON_ANSWER])
        _agent(model, [_retrieval_tool()]).run(TASK, output_schema=SCHEMA)
        assert "Give that answer now" not in model.prompts[0]
        assert "Give that answer now" in model.prompts[1]


class TestAStreamStatesTheDeclaredShapeToo:
    """A configured schema reaches the streamed prompt, and only then.

    ``stream()`` opens no per-call scope, so it reads the agent's configured
    schema. Without this a caller who set one on the config got it honoured on
    ``run()`` and silently ignored the moment they streamed the same agent.
    """

    def test_a_configured_schema_reaches_the_streamed_prompt(self):
        model = _Scripted(["A"])
        agent = _agent(model, [], output_schema=SCHEMA)
        "".join(agent.stream(TASK))
        assert SCHEMA_MARK in model.prompts[0]
        assert model.prompts[0].rstrip().endswith("Answer:")

    def test_a_stream_without_a_schema_is_unchanged(self):
        model = _Scripted(["A"])
        "".join(_agent(model, []).stream("What powers food webs?"))
        assert model.prompts[0] == (
            "Answer this question directly and concisely:\n\n"
            "What powers food webs?\n\nAnswer:"
        )


class TestBlockingAndNativeStreamAgree:
    """The same agent gets the same closing text on both paths."""

    @pytest.mark.parametrize("tools_factory", [
        pytest.param(lambda: [_retrieval_tool()], id="retrieval"),
        pytest.param(lambda: [Calculator()], id="calculator"),
    ])
    def test_the_two_loops_build_the_same_close(self, tools_factory):
        tools = tools_factory()
        actions = [(tools[0].metadata.name, "{}")]
        agent = _agent(_Scripted([PROSE], native=True), tools)
        blocking = agent._compose_closing(
            agent._answer_shape_instruction(),
            agent._continuation_instruction(actions),
        )
        streamed = agent._native_tool_prompt(
            TASK, "Previous steps: ...", "", actions,
        )
        assert blocking
        assert streamed.endswith(blocking)

    def test_a_schema_reaches_the_streamed_close_too(self):
        tools = [_retrieval_tool()]
        agent = _agent(_Scripted([PROSE], native=True), tools,
                       output_schema=SCHEMA)
        streamed = agent._native_tool_prompt(
            TASK, "Previous steps: ...", "", [("knowledge_search", "{}")],
        )
        assert SCHEMA_MARK in streamed
        assert streamed.index(SCHEMA_MARK) < streamed.index("source material")


class TestBlockingAndTextStreamAgree:
    """``run()`` and ``stream()`` build the same ReAct prompt, turn for turn.

    The two are separate loops, so nothing but this makes them agree. Before,
    the streamed one passed neither the retrieval close nor a declared schema,
    and a caller who changed ``run`` to ``stream`` silently changed what the
    model was asked.

    ``run()`` has one stage ``stream()`` does not -- re-asking for an answer in
    a declared shape after the loop is over -- so the comparison is over the
    prompts the loop itself built, marked by where that stage begins.
    """

    @staticmethod
    def _loop_prompts(tools, *, stream: bool, **cfg) -> list[str]:
        model = _Scripted([SEARCH, JSON_ANSWER])
        agent = _agent(model, tools, **cfg)
        cut: list[int] = []
        original = agent._apply_structured_output

        def marked(*a, **kw):
            cut.append(len(model.prompts))
            return original(*a, **kw)

        agent._apply_structured_output = marked
        if stream:
            "".join(agent.stream(TASK))
        else:
            agent.run(TASK)
        return model.prompts[:cut[0]] if cut else model.prompts

    @pytest.mark.parametrize("tools_factory", [
        pytest.param(lambda: [_retrieval_tool()], id="retrieval"),
        pytest.param(lambda: [Calculator()], id="calculator"),
    ])
    @pytest.mark.parametrize("cfg", [
        pytest.param({}, id="no-schema"),
        pytest.param({"output_schema": SCHEMA}, id="declared-schema"),
    ])
    def test_every_loop_turn_is_the_same_prompt(self, tools_factory, cfg):
        blocking = self._loop_prompts(tools_factory(), stream=False, **cfg)
        streamed = self._loop_prompts(tools_factory(), stream=True, **cfg)
        assert len(blocking) == len(streamed) >= 2
        assert blocking == streamed

    def test_the_streamed_retrieval_turn_states_the_close(self):
        streamed = self._loop_prompts([_retrieval_tool()], stream=True)
        assert CONTEXT_ANSWER_INSTRUCTION in streamed[-1]

    def test_the_streamed_turns_state_a_declared_schema(self):
        streamed = self._loop_prompts(
            [_retrieval_tool()], stream=True, output_schema=SCHEMA)
        assert all(SCHEMA_MARK in p for p in streamed)


class TestTheChangeReachesNoOtherShape:
    @pytest.mark.parametrize("turns", [
        ['Thought: compute.\nAction: calculator\nAction Input: {"expression": "6*7"}',
         "Thought: done.\nFinal Answer: 42"],
    ])
    def test_a_calculator_run_says_nothing_about_form(self, turns):
        model = _Scripted(turns)
        _agent(model, [Calculator()]).run("What is 6*7?")
        joined = "\n".join(model.prompts)
        assert "source material" not in joined
        assert "matching this schema" not in joined
        assert "in the form the question asks for" not in joined
