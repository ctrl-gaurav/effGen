"""A tool call the model writes as text runs; the model's own "result" never answers.

On a native tool-calling agent the default reader falls back to ReAct text, so a
model can write ``Action: <tool>`` / ``Action Input: {...}`` instead of making a
provider-native call. If generation carries on past that point the model writes
the tool's result itself (``Observation: 5``) and an answer built on it, and a
reader that takes the first answer label it sees returns that invented result
as the run's answer: the tool never runs.

Two guards, each enough on its own:

* the request stops where the result would begin — the observation label is
  sent on every tool-holding turn whose text a ReAct-reading strategy may read,
  on ``run()``, ``run_async()`` and ``stream()`` alike; where the adapter
  declares its provider takes no stop sequence beside tools, the text is cut at
  the same place after it comes back;
* the reader runs the written action and discards whatever follows it, so an
  invented observation cannot become the answer even if the model ignores the
  stop sequence.

Other tests pin what must not change: an answer with no action
is taken as written, prose that quotes "Observation:" with no action before it
is left whole, and an agent whose reader never reads ReAct text, or which
holds no tools, is sent no stop sequence.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.tool_calling import HybridStrategy, ReActStrategy
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

#: A turn that writes a call as text, then the model's own result for it,
#: then an answer built on that result. The real tool says 9.
INVENTED = (
    "Thought: I will count them with the tool.\n"
    "Action: counter\n"
    'Action Input: {"query": "count the even numbers"}\n'
    "Observation: 5\n"
    "Thought: I now know the final answer.\n"
    "Final Answer: 5"
)
AFTER_TOOL = "Thought: the tool says 9.\nFinal Answer: 9"
#: The stable phrase logged each time the reader discards text after an action.
DISCARD_PHRASE = "a written action is run and what the model wrote after it is discarded"
OBSERVATION_STOP = ["\nObservation:"]


class _Counter(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="counter", description="Counts things.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="What to count.", required=True)],
        ))
        self.ran = 0

    async def _execute(self, **kw):
        self.ran += 1
        return "9"


class _Scripted(BaseModel):
    """Answers from a script, honouring (or ignoring) the stop list it is sent."""

    def __init__(self, turns, *, honour_stops=True, takes_stop_with_tools=True,
                 name="scripted") -> None:
        super().__init__(model_name=name, model_type=ModelType.OPENAI)
        self.turns = list(turns)
        self.i = 0
        self.honour_stops = honour_stops
        self.takes_stop_with_tools = takes_stop_with_tools
        self.configs: list[object] = []
        self.carried_tools: list[bool] = []

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def supports_stop_with_tools(self) -> bool:
        return self.takes_stop_with_tools

    def _next(self, config, kw) -> str:
        self.configs.append(config)
        tools = bool(kw.get("tools"))
        self.carried_tools.append(tools)
        stops = list(getattr(config, "stop_sequences", None) or [])
        if tools and stops and not self.takes_stop_with_tools:
            raise ValueError("400: 'stop' is not supported together with 'tools'")
        text = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        if self.honour_stops:
            for stop in stops:
                if stop in text:
                    text = text[:text.index(stop)]
        return text

    def generate(self, prompt, config=None, **kw):
        return GenerationResult(text=self._next(config, kw), tokens_used=5,
                                finish_reason="stop", model_name=self.model_name,
                                metadata={})

    def generate_stream(self, prompt, config=None, **kw):
        text = self._next(config, kw)
        for i in range(0, len(text), 3):
            yield text[i:i + 3]

    def count_tokens(self, t):
        return TokenCount(count=len(str(t).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 8192

    def generate_batch(self, ps, config=None, **kw):
        return [self.generate(p, config, **kw) for p in ps]

    def generate_with_tools(self, p, tools, config=None, **kw):
        return self.generate(p, config, tools=tools, **kw)

    def supports_function_calling(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def tool_call_support(self) -> str:
        return "api"


def _agent(model, tools, **cfg) -> Agent:
    return Agent(config=AgentConfig(
        name="written_action", model=model, tools=list(tools), max_iterations=4,
        raise_on_error=False, enable_memory=False, **cfg,
    ))


def _run(entry: str, agent: Agent, task: str = "How many even numbers are there?"):
    if entry == "run":
        return str(agent.run(task).output)
    if entry == "run_async":
        return str(asyncio.run(agent.run_async(task)).output)
    received = "".join(str(piece) for piece in agent.stream(task))
    return received


ENTRIES = ["run", "run_async", "stream"]


# --------------------------------------------------------------------------- #
# The written action runs, whatever the stop sequences did
# --------------------------------------------------------------------------- #
class TestTheWrittenActionRuns:
    @pytest.mark.parametrize("entry", ENTRIES)
    def test_an_invented_observation_never_becomes_the_answer(self, entry, caplog):
        """The model ignores every stop sequence: the reader alone must run the
        tool and drop the invented ``Observation: 5`` and ``Final Answer: 5``."""
        tool = _Counter()
        model = _Scripted([INVENTED, AFTER_TOOL], honour_stops=False)
        with caplog.at_level(logging.INFO), _agent(model, [tool]) as agent:
            answer = _run(entry, agent)
        assert tool.ran == 1, "the written action never ran"
        assert answer.strip() == "9"
        assert any(DISCARD_PHRASE in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("entry", ENTRIES)
    def test_with_the_stop_honoured_the_turn_ends_at_its_action(self, entry):
        tool = _Counter()
        model = _Scripted([INVENTED, AFTER_TOOL])
        with _agent(model, [tool]) as agent:
            answer = _run(entry, agent)
        assert tool.ran == 1
        assert answer.strip() == "9"

    def test_an_answer_written_after_an_unrun_action_is_discarded(self):
        """No observation at all: an answer after an action nothing ran is the
        model's guess at the result, and the action runs instead."""
        text = ('Action: counter\nAction Input: {"query": "evens"}\n'
                "Final Answer: 5")
        read = HybridStrategy().parse_response(text, {"counter": _Counter()})
        assert read.is_tool_call and read.tool_name == "counter"
        assert read.final_answer is None

    def test_a_call_written_after_a_declared_no_action_still_runs(self):
        """A turn that first says it takes no action and then writes a real
        call with an invented result: the call is the one that counts."""
        text = ("Thought: nothing left to do.\nAction: None\nAction Input: None\n"
                "Observation: -10.4\nThought: I will print it.\nAction: counter\n"
                'Action Input: {"query": "x"}\nObservation: 5\nAnswer: 5')
        read = HybridStrategy().parse_response(text, {"counter": _Counter()})
        assert read.is_tool_call and read.tool_name == "counter"
        assert read.arguments == {"query": "x"}
        assert read.final_answer is None

    def test_the_split_names_what_it_discarded(self):
        from effgen.core.tool_calling import split_at_unrun_action

        head, discarded, kind = split_at_unrun_action(INVENTED)
        assert head.endswith('Action Input: {"query": "count the even numbers"}')
        assert discarded.startswith("Observation: 5") and kind == "observation"
        text = 'Action: counter\nAction Input: {"query": "evens"}\nFinal Answer: 5'
        assert split_at_unrun_action(text)[2] == "answer"

    def test_the_reader_reads_the_action_from_the_text_before_the_invention(self):
        read = ReActStrategy().parse_response(INVENTED, {"counter": _Counter()})
        assert read.is_tool_call and read.tool_name == "counter"
        assert read.arguments == {"query": "count the even numbers"}
        assert read.final_answer is None


# --------------------------------------------------------------------------- #
# The observation label is sent where a written action can be read
# --------------------------------------------------------------------------- #
class TestTheObservationStopIsSent:
    @pytest.mark.parametrize("entry", ENTRIES)
    def test_every_path_sends_it_on_a_tool_holding_native_turn(self, entry):
        model = _Scripted(["Final Answer: 9"])
        with _agent(model, [_Counter()]) as agent:
            _run(entry, agent)
        assert model.carried_tools[0], "the turn was expected to carry its tools"
        assert list(model.configs[0].stop_sequences or []) == OBSERVATION_STOP

    def test_every_turn_of_the_run_sends_it(self):
        """Every turn of a tool-holding run can be read as text, so every turn
        carries it — not only the first."""
        tool = _Counter()
        model = _Scripted([INVENTED, AFTER_TOOL])
        with _agent(model, [tool]) as agent:
            agent.run("How many even numbers are there?")
        assert len(model.configs) == 2
        assert all(list(c.stop_sequences or []) == ["\nObservation:"]
                   for c in model.configs)

    def test_a_callers_own_stop_sequences_replace_it(self):
        model = _Scripted(["Final Answer: 9"])
        with _agent(model, [_Counter()]) as agent:
            agent.run("How many?", stop_sequences=["END"])
        assert model.configs[0].stop_sequences == ["END"]

    def test_it_is_read_from_the_agent_never_from_a_model_name(self):
        sent = []
        for name in ("qwen-1.5b-instruct", "some-other-model"):
            model = _Scripted(["Final Answer: 9"], name=name)
            with _agent(model, [_Counter()]) as agent:
                agent.run("How many?")
            sent.append(list(model.configs[0].stop_sequences or []))
        assert sent[0] == sent[1] == ["\nObservation:"]


class TestAProviderThatTakesNoStopBesideTools:
    @pytest.mark.parametrize("entry", ENTRIES)
    def test_the_stop_is_applied_to_the_returned_text_instead(self, entry, caplog):
        """The adapter declares its provider rejects ``stop`` beside ``tools``:
        the request goes without it (it would be refused otherwise) and the
        text is cut where the provider would have cut it."""
        tool = _Counter()
        model = _Scripted([INVENTED, AFTER_TOOL], takes_stop_with_tools=False)
        with caplog.at_level(logging.INFO), _agent(model, [tool]) as agent:
            answer = _run(entry, agent)
        assert model.carried_tools[0]
        assert not model.configs[0].stop_sequences
        assert tool.ran == 1
        assert answer.strip() == "9"
        assert any("takes no stop sequences beside tools" in r.getMessage()
                   for r in caplog.records)


# --------------------------------------------------------------------------- #
# What must not change
# --------------------------------------------------------------------------- #
class TestNothingElseIsCut:
    def test_a_genuine_final_answer_with_no_action_is_taken_unchanged(self):
        answer = "Thought: I know this.\nFinal Answer: Canberra is the capital."
        tool = _Counter()
        model = _Scripted([answer])
        with _agent(model, [tool]) as agent:
            out = agent.run("What is the capital of Australia?")
        assert tool.ran == 0
        assert str(out.output) == "Canberra is the capital."

    def test_a_quoted_observation_in_an_answer_with_no_action_is_not_cut(self):
        prose = ("Final Answer: The log says the disk is full.\n"
                 "Observation: the write failed at 02:14.\n"
                 "So the job must be re-run after cleanup.")
        read = HybridStrategy().parse_response(prose, {"counter": _Counter()})
        assert not read.is_tool_call
        assert "Observation: the write failed" in (read.final_answer or "")

    def test_an_answer_that_comes_before_an_action_is_still_the_answer(self):
        text = ("Final Answer: 9\nAction: counter\n"
                'Action Input: {"query": "double-check"}\nObservation: 9')
        read = ReActStrategy().parse_response(text, {"counter": _Counter()})
        assert not read.is_tool_call
        assert read.final_answer is not None and read.final_answer.startswith("9")

    def test_a_declared_no_action_is_left_to_its_own_reading(self):
        """``Action: None`` names no tool; the answer after it stays the answer."""
        read = ReActStrategy().parse_response(
            "Action: None\nFinal Answer: 9", {"counter": _Counter()})
        assert not read.is_tool_call
        assert read.final_answer == "9"

    def test_a_tool_free_agent_is_sent_no_stop(self):
        """With no tool to call, no written action can be run, so nothing is
        sent that could cut a long answer."""
        model = _Scripted(["Here is the summary of the report."])
        with _agent(model, []) as agent:
            agent.run("Summarise the report.")
        assert not model.configs[0].stop_sequences

    def test_an_agent_whose_reader_never_reads_react_text_is_sent_no_stop(self):
        model = _Scripted(["Final Answer: 9"])
        with _agent(model, [_Counter()], tool_calling_mode="native") as agent:
            agent.run("How many?")
        assert not model.configs[0].stop_sequences
