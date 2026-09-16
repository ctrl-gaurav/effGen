"""How long an answer may be, and who decides — one contract, every path.

Four separate things used to decide the length of an answer, and none of them
knew about the others: a constant output budget, four tool contracts that each
asked for prose, a set of stop sequences sent whether or not the prompt wrote
the labels they match, and a ``reasoning_effort`` that ``stream()`` sent and
``run()`` dropped. A multiple-choice run came back billed for 195 output tokens
and kept the letter "C".

What is pinned here:

* the settings one turn resolves reach every path identically — the blocking
  turn, the asynchronous one and the streamed one;
* the framework's own sentences say what a tool is for and never how much to
  write;
* the answer's form is stated once, last, by the caller or by the one default,
  and never names a form of its own;
* the budget follows the shape the run declared, a caller's own value always
  wins, and a model that declares it reasons is never starved;
* the four ReAct labels go out with the prompts that write them and with
  nothing else.
"""

from __future__ import annotations

import asyncio

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_loop import (
    DEFAULT_STOP_SEQUENCES,
    framework_stop_sequences,
    resolve_turn_config,
)
from effgen.core.agent_runtime import resolve_output_budget
from effgen.models._adapter_utils import REASONING_OUTPUT_FLOOR
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.prompts.answer_style import (
    ANSWER_STYLE_TEXTS,
    DEFAULT_ANSWER_STYLE,
    answer_style_text,
    resolve_answer_style,
)
from effgen.prompts.tool_contract import (
    TOOL_CONTRACT_EXECUTE,
    TOOL_CONTRACT_GENERAL,
    TOOL_CONTRACT_LOOKUP,
    TOOL_CONTRACT_VERIFY,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

ANSWER = "Thought: done.\nFinal Answer: 42"


def _tool(name: str, category: ToolCategory) -> BaseTool:
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


def CALC() -> BaseTool:
    return _tool("calculator", ToolCategory.COMPUTATION)


class _Recorder(BaseModel):
    """Answers from a script and keeps every prompt and configuration it saw."""

    _supports_tools = True
    _support_kind = "api"
    _is_reasoning_model = False

    def __init__(self, turns=(ANSWER,)) -> None:
        super().__init__(model_name="recorder", model_type=ModelType.OPENAI)
        self.turns = list(turns)
        self.i = 0
        self.prompts: list[str] = []
        self.configs: list[object] = []
        self.streamed: list[bool] = []

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def _record(self, prompt, config, streamed):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        self.configs.append(config)
        self.streamed.append(streamed)

    def generate(self, prompt, config=None, **kw):
        self._record(prompt, config, False)
        text = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return GenerationResult(text=text, tokens_used=5, finish_reason="stop",
                                model_name="recorder", metadata={})

    def generate_stream(self, prompt, config=None, **kw):
        self._record(prompt, config, True)
        text = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        yield text

    def count_tokens(self, t):
        return TokenCount(count=len(str(t).split()), model_name="recorder")

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


class _Reasons(_Recorder):
    """A model whose adapter declares that it reasons before it answers."""

    _is_reasoning_model = True


def _agent(model, **cfg) -> Agent:
    return Agent(config=AgentConfig(
        name="contract", model=model, max_iterations=2,
        raise_on_error=False, enable_memory=False, **cfg,
    ))


# --------------------------------------------------------------------------- #
# One resolved contract reaches every turn
# --------------------------------------------------------------------------- #
class TestOneResolvedContractReachesEveryTurn:
    """``run()`` used to rebuild a configuration keyword arguments could not
    carry, so a setting the caller pinned reached ``stream()`` and nothing
    else."""

    @pytest.mark.parametrize("tools", [[], [CALC()]], ids=["no-tools", "tools"])
    def test_reasoning_effort_reaches_run_as_it_reaches_stream(self, tools):
        seen = []
        for entry in ("run", "run_async", "stream"):
            model = _Reasons()
            with _agent(model, tools=list(tools)) as agent:
                if entry == "run":
                    agent.run("What is 6*7?", reasoning_effort="low")
                elif entry == "run_async":
                    asyncio.run(agent.run_async("6*7?", reasoning_effort="low"))
                else:
                    list(agent.stream("What is 6*7?", reasoning_effort="low"))
            assert model.configs, f"{entry} made no model call"
            seen.append((entry, model.configs[0].reasoning_effort))
        assert seen == [("run", "low"), ("run_async", "low"), ("stream", "low")]

    def test_a_turn_retaken_on_the_blocking_path_keeps_the_setting(self):
        """A streamed turn the loop had to take again is still the same turn."""
        model = _Reasons()
        with _agent(model, tools=[CALC()]) as agent:
            list(agent.stream("What is 6*7?", reasoning_effort="low"))
        assert model.configs
        assert all(c.reasoning_effort == "low" for c in model.configs)

    def test_the_blocking_turn_sends_the_turns_resolved_settings(self):
        """Every sampling field the run resolved, on the request it sent."""
        model = _Recorder()
        with _agent(model, tools=[CALC()], temperature=0.11, top_p=0.31,
                    seed=4242, max_tokens=333) as agent:
            agent.run("What is 6*7?")
        sent = model.configs[0]
        assert (sent.temperature, sent.top_p, sent.seed, sent.max_tokens) == (
            0.11, 0.31, 4242, 333)


# --------------------------------------------------------------------------- #
# The framework's own sentences stop demanding output
# --------------------------------------------------------------------------- #
class TestNoContractAsksForMoreOutput:
    FOUR = (TOOL_CONTRACT_GENERAL, TOOL_CONTRACT_VERIFY,
            TOOL_CONTRACT_EXECUTE, TOOL_CONTRACT_LOOKUP)

    #: Phrases that ask for text the task did not ask for. Each one was in a
    #: shipped contract and each one is paid for on every request that carries
    #: it.
    DEMANDS = (
        "in your own words",
        "say what each step gives",
        "name what is missing",
        "step by step",
    )

    @pytest.mark.parametrize("phrase", DEMANDS)
    def test_no_contract_asks_the_model_to_write_more(self, phrase):
        for contract in self.FOUR:
            assert phrase not in contract.lower(), contract

    def test_the_executing_contract_keeps_its_one_line_of_narration(self):
        """The one demand that was measured and earns its tokens.

        Taking it out cost a 7B model 25.7 accuracy points on the hardest
        coding set, against a noise band of 7.6, and it called its executor 19 %
        less often for no saving in output. A tool that does work the model
        cannot do is the case where saying what is about to be computed pays.
        """
        assert "say in one line" in TOOL_CONTRACT_EXECUTE.lower()
        for contract in (TOOL_CONTRACT_GENERAL, TOOL_CONTRACT_VERIFY,
                         TOOL_CONTRACT_LOOKUP):
            assert "say in one line" not in contract.lower()

    def test_the_four_texts_are_what_they_are(self):
        """Pinned verbatim: this text is what a request is billed for."""
        assert TOOL_CONTRACT_VERIFY == (
            "Use the tools to check the steps you are least sure of, one step "
            "per call, and correct yourself if a tool disagrees with you. Do "
            "not hand a tool the whole task at once."
        )
        assert TOOL_CONTRACT_LOOKUP == (
            "The tools bring back source material, not the answer. Answer the "
            "question yourself, in the form it asks for, from what they "
            "return, and do not return a passage as the answer."
        )
        assert TOOL_CONTRACT_EXECUTE == (
            "Use the tools to do this task rather than working it out in your "
            "head. Say in one line what you are about to compute, call the tool "
            "to compute it, and read the answer off what it returns. Do not "
            "simulate the tool yourself and do not answer from a result you did "
            "not get back from it. If the tool errors or returns something "
            "unexpected, fix the input and call it again. Finish by stating the "
            "final answer."
        )
        assert TOOL_CONTRACT_GENERAL == (
            "Work through this task one step at a time. Give a tool a single "
            "step, not the whole task at once, and use the result it returns "
            "rather than working it out again yourself. When every step is "
            "done, state the final answer."
        )

    def test_each_contract_still_says_what_the_tools_are_for(self):
        """Saying nothing is not the fix: it is what stops a tool being used."""
        for contract in self.FOUR:
            assert "tool" in contract.lower()

    def test_no_category_tip_asks_for_a_narrated_failure(self):
        from effgen.prompts.agent_system_prompt import CATEGORY_INSTRUCTIONS

        for tip in CATEGORY_INSTRUCTIONS.values():
            assert "inform the user" not in tip.lower()


# --------------------------------------------------------------------------- #
# The answer style, stated once and read last
# --------------------------------------------------------------------------- #
class TestTheAnswerStyleIsStatedOnce:
    def test_the_shipped_default_states_nothing(self):
        """Measured, not assumed: stating it by default stopped a model using
        the tools it was given, and cost accuracy where searching helps."""
        assert DEFAULT_ANSWER_STYLE is None
        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            agent.run("What is 6*7?")
        for text in ANSWER_STYLE_TEXTS.values():
            assert text not in model.prompts[0]

    def test_the_named_styles_name_no_form(self):
        """A form demand here would overrule the question that asked for one."""
        for text in ANSWER_STYLE_TEXTS.values():
            lowered = text.lower()
            for form in ("letter", "number", "sentence", "paragraph", "word"):
                assert form not in lowered, text

    def test_an_explicit_answer_style_wins_on_every_entry_point(self):
        mine = "Answer as a haiku."
        for entry in ("run", "run_async", "stream"):
            model = _Recorder()
            with _agent(model, tools=[CALC()]) as agent:
                if entry == "run":
                    agent.run("What is 6*7?", answer_style=mine)
                elif entry == "run_async":
                    asyncio.run(agent.run_async("6*7?", answer_style=mine))
                else:
                    list(agent.stream("What is 6*7?", answer_style=mine))
            assert model.prompts[0].rstrip().endswith(mine), entry

    def test_a_call_wins_over_the_agents_own_style(self):
        model = _Recorder()
        with _agent(model, tools=[CALC()], answer_style="Answer in French.") as a:
            a.run("What is 6*7?", answer_style="Answer as a haiku.")
        assert model.prompts[0].rstrip().endswith("Answer as a haiku.")

    def test_an_empty_style_states_nothing(self):
        model = _Recorder()
        with _agent(model, tools=[CALC()], answer_style="") as agent:
            agent.run("What is 6*7?")
        for text in ANSWER_STYLE_TEXTS.values():
            assert text not in model.prompts[0]

    def test_the_resolution_order_is_call_then_config_then_default(self):
        assert resolve_answer_style("mine", "theirs") == "mine"
        assert resolve_answer_style(None, "theirs") == "theirs"
        assert resolve_answer_style(None, None) == DEFAULT_ANSWER_STYLE
        assert resolve_answer_style("", "theirs") is None
        assert resolve_answer_style(None, "") is None

    def test_a_child_run_inherits_the_style_once(self):
        """A delegated run answers in the form its parent asked for."""
        parent = AgentConfig(name="parent", model=_Recorder(),
                             answer_style="Answer as a haiku.")
        model = _Recorder()
        child = AgentConfig(
            name="child", model=model, answer_style=parent.answer_style,
            max_iterations=2, raise_on_error=False, enable_memory=False,
        )
        with Agent(config=child) as agent:
            agent.run("What is 6*7?")
        built = model.prompts[0]
        assert built.count("Answer as a haiku.") == 1
        assert built.index("Answer as a haiku.") > built.index("What is 6*7?")

    def test_a_delegated_child_is_built_with_the_parents_style(self):
        """The framework's own child-config builder carries it.

        The child's configuration is assembled field by field from the
        parent's, so a field left out of that list is silently dropped: a
        parent asked for the answer and nothing else, and its sub-agent
        answered at whatever length it liked.
        """
        import inspect

        from effgen.core import sub_agent_manager

        source = inspect.getsource(sub_agent_manager)
        start = source.index("child_cfg = AgentConfig(")
        built = source[start:source.index("\n        )", start)]
        assert "answer_style=getattr(parent_cfg" in built

    def test_a_stated_style_is_the_last_thing_in_the_request(self):
        """The one place the framework will speak after the caller's task is
        the place the caller asked it to."""
        styled = _Recorder()
        with _agent(styled, tools=[CALC()], answer_style="brief") as agent:
            agent.run("What is 6*7?")
        assert styled.prompts[0].rstrip().endswith(answer_style_text("brief"))

    def test_the_contract_keeps_the_place_that_was_measured(self):
        """It closes the opening turn and is stated on no other.

        Ahead of the task, and repeated every turn, it reads as "finish by
        stating the final answer" at the top of a request whose point is to
        keep going: a 7B model on the hardest coding set went from 1.6 executor
        calls per sample to exactly 1.0 and lost 47 accuracy points.
        """
        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            agent.run("What is 6*7?")
        opening = model.prompts[0]
        assert opening.index("What is 6*7?") < opening.index(TOOL_CONTRACT_VERIFY)
        assert opening.rstrip().endswith(TOOL_CONTRACT_VERIFY)

    def test_the_style_closes_the_text_frame_with_its_label(self):
        """The ReAct format parses off a label, so the last line carries one."""
        from effgen.prompts.tool_prompt_generator import ToolPromptGenerator

        gen = ToolPromptGenerator(tools=[CALC()], model_name="recorder")
        built = gen.generate_react_prompt(
            task="What is 6*7?", answer_style="Answer briefly.",
        )
        assert built.rstrip().endswith(
            "Answer briefly.\nGive it after a 'Final Answer:' label once you have it."
        )
        assert built.count("Final Answer:' label") == 1


# --------------------------------------------------------------------------- #
# The budget follows the declared shape
# --------------------------------------------------------------------------- #
class TestTheBudgetFollowsWhatWasDeclared:
    def test_a_declared_schema_sizes_the_budget(self):
        model = _Recorder()
        with _agent(model, output_schema={
            "type": "object", "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }) as agent:
            agent.run("What is 6*7?")
        assert model.configs[0].max_tokens == 256

    def test_a_run_that_declared_nothing_keeps_the_constant(self):
        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            agent.run("What is 6*7?")
        assert model.configs[0].max_tokens == 1024

    def test_a_reasoning_model_is_never_budgeted_below_4096(self):
        model = _Reasons()
        with _agent(model, output_schema={
            "type": "object", "properties": {"n": {"type": "integer"}},
        }) as agent:
            agent.run("What is 6*7?")
        assert model.configs[0].max_tokens >= REASONING_OUTPUT_FLOOR

    @pytest.mark.parametrize("pinned", [1, 77, 100000])
    def test_an_explicit_max_tokens_wins_on_six_paths(self, pinned):
        """run, run_async, stream, a declared schema, a reasoning model, and an
        agent that pinned it on its configuration."""
        seen = []

        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            agent.run("6*7?", max_tokens=pinned)
        seen.append(("run", model.configs[0].max_tokens))

        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            asyncio.run(agent.run_async("6*7?", max_tokens=pinned))
        seen.append(("run_async", model.configs[0].max_tokens))

        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            list(agent.stream("6*7?", max_tokens=pinned))
        seen.append(("stream", model.configs[0].max_tokens))

        model = _Recorder()
        with _agent(model, output_schema={
            "type": "object", "properties": {"n": {"type": "integer"}},
        }) as agent:
            agent.run("6*7?", max_tokens=pinned)
        seen.append(("schema", model.configs[0].max_tokens))

        model = _Reasons()
        with _agent(model, tools=[CALC()]) as agent:
            agent.run("6*7?", max_tokens=pinned)
        seen.append(("reasoning", model.configs[0].max_tokens))

        model = _Recorder()
        with _agent(model, tools=[CALC()], max_tokens=pinned) as agent:
            agent.run("6*7?")
        seen.append(("config", model.configs[0].max_tokens))

        assert seen == [(name, pinned) for name, _ in seen]

    def test_a_tight_cap_still_gives_the_typed_message(self):
        """The cap the caller pinned is honoured, and an empty truncated turn
        is reported with the message the framework already had for it."""
        class _Truncates(_Recorder):
            def generate(self, prompt, config=None, **kw):
                self._record(prompt, config, False)
                return GenerationResult(
                    text="", tokens_used=8, finish_reason="length",
                    model_name="recorder", metadata={"truncated": True},
                )

        model = _Truncates()
        with _agent(model, tools=[CALC()]) as agent:
            response = agent.run("Write a report.", max_tokens=8)
        assert model.configs[0].max_tokens == 8
        text = str(response.output or "") + str(
            (response.metadata or {}).get("error", "")
        )
        assert text.strip(), "a truncated turn returned nothing to read"

    def test_the_budget_never_reads_a_model_id_for_a_declared_shape(self):
        """Only declarations decide: the same schema, two different names."""
        class _Named(_Recorder):
            def __init__(self, name):
                super().__init__()
                self.model_name = name

        schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        assert resolve_output_budget(
            None, None, _Named("qwen-1.5b"), output_schema=schema
        ) == resolve_output_budget(
            None, None, _Named("some-other-model"), output_schema=schema
        )


# --------------------------------------------------------------------------- #
# Stop sequences from the frame in use
# --------------------------------------------------------------------------- #
class TestStopSequencesFollowTheFrame:
    def test_only_the_frames_that_write_the_labels_send_them(self):
        assert framework_stop_sequences("react_text") == DEFAULT_STOP_SEQUENCES
        assert framework_stop_sequences("custom_template") == DEFAULT_STOP_SEQUENCES
        assert framework_stop_sequences("native") == ()

    def test_the_native_frame_sends_no_react_labels(self):
        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            agent.run("What is 6*7?")
        assert not model.configs[0].stop_sequences

    def test_the_text_frame_still_sends_them(self):
        class _NoTools(_Recorder):
            _supports_tools = False
            _support_kind = "none"

        model = _NoTools()
        with _agent(model, tools=[CALC()]) as agent:
            agent.run("What is 6*7?")
        assert model.configs[0].stop_sequences == list(DEFAULT_STOP_SEQUENCES)

    def test_a_callers_stop_sequences_are_always_sent(self):
        for tools in ([], [CALC()]):
            model = _Recorder()
            with _agent(model, tools=list(tools)) as agent:
                agent.run("What is 6*7?", stop_sequences=["END"])
            assert model.configs[0].stop_sequences == ["END"]

    def test_a_long_answer_is_not_cut_at_a_question_label(self):
        """The framework used to send ``"\\nQuestion:"`` on every request, so an
        answer that quoted the question it was asked came back truncated."""
        long_answer = (
            "Here is the summary you asked for.\n"
            "Question: what was asked?\n"
            "It was a request for a four-hundred word summary, and this is the "
            "rest of the answer that used to be cut off."
        )
        model = _Recorder(turns=[long_answer])
        with _agent(model) as agent:
            response = agent.run("Write a 400-word summary.")
        assert not model.configs[0].stop_sequences
        assert "rest of the answer" in str(response.output)

    def test_the_duplicate_literal_list_is_gone(self):
        """Two copies of the same four strings is how they drifted apart."""
        import inspect

        from effgen.core import agent_generation

        assert '"\\nObservation:"' not in inspect.getsource(agent_generation)


# --------------------------------------------------------------------------- #
# One declaration answers "does this model reason"
# --------------------------------------------------------------------------- #
class TestOneDeclarationAnswersWhetherAModelReasons:
    def test_the_budget_and_the_controls_read_one_declaration(self):
        """The budget picked the reasoning family from a name list while the
        adapter's own answer said otherwise, so a run got the reasoning budget
        and the stop sequences at the same time."""
        import inspect

        from effgen.models import openai_adapter

        source = inspect.getsource(openai_adapter._build_stop_source_probe) \
            if hasattr(openai_adapter, "_build_stop_source_probe") else \
            inspect.getsource(openai_adapter)
        assert 'startswith("gpt-5")' not in source

    def test_a_name_list_answer_is_logged_not_silent(self, caplog):
        import logging

        from effgen.models import _adapter_utils

        class _Named:
            model_name = "gpt-5-nano-probe-for-this-test"

        _adapter_utils._name_prefix_reasoning_reported.discard(
            _Named.model_name.lower())
        with caplog.at_level(logging.INFO, logger="effgen.models._adapter_utils"):
            assert _adapter_utils.needs_reasoning_headroom(_Named())
        assert any("comes from the name list" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Nothing branches on a task, a dataset or a model id
# --------------------------------------------------------------------------- #
class TestNothingReadsTheTaskOrAName:
    def test_the_turns_settings_do_not_read_the_task(self):
        """Two different tasks, one agent, the same request settings."""
        settings = []
        for task in ("What is 6*7?", "Which option is right? A, B or C?"):
            model = _Recorder()
            with _agent(model, tools=[CALC()]) as agent:
                agent.run(task)
            c = model.configs[0]
            settings.append((c.max_tokens, c.stop_sequences, c.temperature))
        assert settings[0] == settings[1]

    def test_the_style_line_is_the_same_whatever_the_task(self):
        """A stated style is stated verbatim; it is never read off the task."""
        mine = "Answer in one word."
        tails = []
        for task in ("What is 6*7?", "Write a 400-word summary."):
            model = _Recorder()
            with _agent(model, tools=[CALC()], answer_style=mine) as agent:
                agent.run(task)
            tails.append(model.prompts[0].rsplit("\n\n", 1)[-1])
        assert tails == [mine, mine]


class TestTheFrameIsResolvedForTheTurnNotTheRun:
    def test_a_turn_moved_onto_the_text_scaffold_gets_its_labels(self):
        """The guards can move one turn onto the frame that writes the labels."""
        from effgen.core.agent_loop import framework_stop_sequences

        assert framework_stop_sequences("react_text") == DEFAULT_STOP_SEQUENCES

    def test_resolve_turn_config_takes_the_frame_it_is_given(self):
        model = _Recorder()
        with _agent(model, tools=[CALC()]) as agent:
            native, _ = resolve_turn_config(agent, {}, frame="native")
            text, _ = resolve_turn_config(agent, {}, frame="react_text")
        assert not native.stop_sequences
        assert text.stop_sequences == list(DEFAULT_STOP_SEQUENCES)
