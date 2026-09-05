"""An agent holding a tool that must actually run gets one turn that requires it.

Three things are pinned here, and they fail separately.

**The generation parameter reaches the provider.** The agent copied exactly one
name out of its ``**kwargs`` when it built the adapter call, so every other
generation parameter was discarded between the loop and the request without an
error or a log line. A guard that decided to require a tool call recorded that
it had, and the request went out unchanged. The forwarding is now derived from
one declared set on both model-call paths, and the tests below read that set
rather than restating it, so a parameter added to it is covered here at once.

**The loop refuses one answer.** A tool whose declared category says it does work
the model cannot do in its head has not been used when the model describes what
it would have printed. The first such answer is sent back naming the tool, and
the turn after it requires a call. Only the first, only that turn, and never the
opening turn -- which has to stay free to reason, because a model emitting a
native call usually returns empty content and forcing turn one buys the call at
the cost of the work that made it worth making.

**Each adapter carries the constraint or says it cannot.** Providers spell
``tool_choice`` differently and some do not enforce it at all, so an adapter
advertises the capability and the loop asks before it constrains. An adapter
that says nothing degrades to the nudge rather than losing the turn.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_generation import MODEL_CALL_KWARGS, model_call_kwargs
from effgen.core.agent_runtime import (
    NUDGE_MUST_EXECUTE,
    model_can_require_tool_call,
    sanitize_final_answer,
)
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.prompts.tool_contract import (
    TOOL_CONTRACT_EXECUTE,
    TOOL_CONTRACTS,
    is_execution_tool,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

# A value per forwardable parameter, so a parameter added to MODEL_CALL_KWARGS
# fails here with "add a sample value" rather than silently going unchecked.
SAMPLE_VALUES: dict[str, object] = {
    "tools": [{"type": "function", "function": {"name": "python_exec"}}],
    "tool_choice": "required",
}


def make_tool(name: str, category, payload: str = "42") -> BaseTool:
    class _T(BaseTool):
        def __init__(self) -> None:
            super().__init__(metadata=ToolMetadata(
                name=name, description=f"The {name} tool.", category=category,
                parameters=[ParameterSpec(
                    name="code", type=ParameterType.STRING,
                    description="Input.", required=True)],
            ))

        async def _execute(self, **kw):
            return payload
    return _T()


class Recorder(BaseModel):
    """Answers from a script and records the kwargs of every call it received."""

    _supports_tools = True
    _support_kind = "api"
    _forces_calls = True

    def __init__(self, turns) -> None:
        super().__init__(model_name="recorder", model_type=ModelType.OPENAI)
        self.turns, self.i = list(turns), 0
        self.seen: list[dict] = []
        self.prompts: list[str] = []

    def load(self): pass

    def unload(self): pass

    def generate(self, prompt, config=None, **kw):
        self.seen.append(dict(kw))
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        text = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return GenerationResult(text=text, tokens_used=5, finish_reason="stop",
                                model_name=self.model_name, metadata={})

    def generate_stream(self, prompt, config=None, **kw):
        yield self.generate(prompt, config, **kw).text

    def generate_with_tools(self, p, tools, config=None, **kw):
        return self.generate(p, config, tools=tools, **kw)

    def count_tokens(self, t):
        return TokenCount(count=len(str(t).split()), model_name="recorder")

    def get_context_length(self):
        return 8192

    def generate_batch(self, ps, config=None, **kw):
        return [self.generate(p) for p in ps]

    def supports_function_calling(self):
        return self._supports_tools

    def supports_tool_calling(self):
        return self._supports_tools

    def tool_call_support(self):
        return self._support_kind

    def streams_tool_calls(self):
        return False

    def supports_forced_tool_call(self):
        return self._forces_calls


class Unforceable(Recorder):
    """An adapter whose provider does not enforce the choice."""

    _forces_calls = False


REFUSAL = "Thought: I will use the python_exec tool for this.\nFinal Answer: 42"
CALL_THEN_ANSWER = [
    'Thought: computing.\nAction: python_exec\nAction Input: {"code": "1+1"}',
    "Thought: done.\nFinal Answer: 42",
]


def build(turns, tools, cls=Recorder, **cfg):
    model = cls(turns)
    agent = Agent(config=AgentConfig(
        name="probe", model=model, tools=tools, max_iterations=4,
        raise_on_error=False, **cfg,
    ))
    return agent, model


def forced_turns(model) -> list[int]:
    """Indexes of the turns that carried a required-call constraint."""
    return [i for i, kw in enumerate(model.seen) if kw.get("tool_choice") == "required"]


# ---------------------------------------------------------------------------
# The generation parameter reaches the provider
# ---------------------------------------------------------------------------
def test_every_forwardable_parameter_has_a_sample_value():
    """A parameter added to the set without a value here is not being tested."""
    assert set(SAMPLE_VALUES) == set(MODEL_CALL_KWARGS), (
        "MODEL_CALL_KWARGS and SAMPLE_VALUES disagree; add the new parameter's "
        "value so the forwarding tests below actually exercise it"
    )


def test_tool_choice_is_a_forwardable_parameter():
    assert "tool_choice" in MODEL_CALL_KWARGS


def test_model_call_kwargs_keeps_only_what_it_declares():
    picked = model_call_kwargs({
        **SAMPLE_VALUES, "max_iterations": 4, "checkpoint_dir": "/tmp", "inputs": [],
    })
    assert picked == SAMPLE_VALUES


@pytest.mark.parametrize("name", sorted(MODEL_CALL_KWARGS))
def test_generate_forwards_the_parameter(name):
    """The ordinary model-call path hands the adapter what it was given."""
    agent, model = build(["Final Answer: 42"], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent._generate("hi", **SAMPLE_VALUES)
    assert model.seen, "the adapter was never called"
    assert model.seen[0].get(name) == SAMPLE_VALUES[name], (
        f"_generate dropped {name!r} between the loop and the adapter"
    )


@pytest.mark.parametrize("name", sorted(MODEL_CALL_KWARGS))
def test_generate_speculative_forwards_the_parameter(name):
    """The two-model path is the same call and must not shape it differently."""
    agent, model = build(["Final Answer: 42"], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    other = Recorder(["Final Answer: 42"])
    agent._all_models = [model, other]
    agent._generate_speculative("hi", **SAMPLE_VALUES)
    seen = model.seen + other.seen
    assert seen, "neither speculative model was called"
    assert seen[0].get(name) == SAMPLE_VALUES[name], (
        f"_generate_speculative dropped {name!r}"
    )


def test_run_accepts_tool_choice_from_a_caller():
    """A caller with a reason to require a call can say so through the public API."""
    agent, model = build(["Final Answer: 42"], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent.run("Compute 6*7.", tool_choice="required")
    assert any(kw.get("tool_choice") == "required" for kw in model.seen)


def test_a_constraint_with_no_definitions_does_not_travel():
    """Requiring a call from a request that offers none is rejected by the API."""
    assert model_call_kwargs({"tool_choice": "required"}) == {}
    assert model_call_kwargs({"tools": [], "tool_choice": "required"}) == {"tools": []}


def test_a_caller_who_requires_a_call_from_a_toolless_agent_sends_no_constraint():
    """The agent holds nothing to call, so the constraint would lose the turn."""
    model = Recorder(["Thought: done.\nFinal Answer: Paris"])
    agent = Agent(config=AgentConfig(
        name="probe", model=model, tools=[], raise_on_error=False))
    agent.run("Capital of France?", tool_choice="required")
    assert model.seen, "the adapter was never called"
    for kw in model.seen:
        assert "tool_choice" not in kw, (
            "a constraint reached the provider with no tool definitions beside it"
        )


def test_agent_bookkeeping_never_reaches_the_adapter():
    """The iteration cap and the checkpoint knobs are the agent's, not the provider's."""
    agent, model = build(["Final Answer: 42"], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent.run("Compute 6*7.", max_iterations=3, checkpoint_interval=0)
    for kw in model.seen:
        assert "max_iterations" not in kw
        assert "checkpoint_interval" not in kw


# ---------------------------------------------------------------------------
# Which tools must actually run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", list(ToolCategory))
def test_execution_tools_are_the_ones_told_to_execute(category):
    """One source of truth: the categories that get the execute contract."""
    expected = TOOL_CONTRACTS[category] is TOOL_CONTRACT_EXECUTE
    assert is_execution_tool(make_tool("t", category)) is expected


def test_a_tool_with_no_declared_category_is_not_an_execution_tool():
    """Fail closed: an unknown tool is not assumed to do work the model cannot."""
    assert is_execution_tool(make_tool("mystery", None)) is False
    assert is_execution_tool(object()) is False


# ---------------------------------------------------------------------------
# The loop refuses one answer
# ---------------------------------------------------------------------------
def test_an_execution_agent_that_never_executed_is_sent_back():
    agent, model = build([REFUSAL], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent.run("Compute the 30th Fibonacci number.")
    assert len(model.seen) >= 2, (
        "the answer was accepted with no tool call and no second turn"
    )


def test_the_turn_after_a_refusal_requires_a_call():
    agent, model = build([REFUSAL], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent.run("Compute the 30th Fibonacci number.")
    assert forced_turns(model) == [1], (
        "the turn after the refusal did not carry the constraint"
    )


def test_the_opening_turn_is_never_constrained():
    """Turn one has to be free to reason before it calls anything."""
    agent, model = build([REFUSAL], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent.run("Compute the 30th Fibonacci number.")
    assert 0 not in forced_turns(model)
    assert model.seen[0].get("tool_choice") is None


def test_the_constraint_covers_exactly_one_turn():
    """Spent on read: the turn after a forced one must be free to answer."""
    agent, model = build([REFUSAL, REFUSAL, REFUSAL, REFUSAL],
                         [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent.run("Compute the 30th Fibonacci number.")
    assert len(forced_turns(model)) == 1, (
        f"the constraint leaked onto later turns: {forced_turns(model)}"
    )


def test_only_the_first_refusal_is_refused():
    """A model that declines twice keeps declining; the budget buys more elsewhere."""
    agent, model = build([REFUSAL, REFUSAL, REFUSAL, REFUSAL],
                         [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    response = agent.run("Compute the 30th Fibonacci number.")
    assert len(model.seen) == 2, (
        f"expected one refusal and then acceptance, got {len(model.seen)} turns"
    )
    assert response.success


def test_the_nudge_names_the_tool_the_agent_holds():
    """The turn that goes back carries the nudge, with the tool's own name in it."""
    agent, model = build([REFUSAL, REFUSAL],
                         [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    agent.run("Compute the 30th Fibonacci number.")
    assert len(model.prompts) >= 2, "the answer was accepted without a second turn"
    assert NUDGE_MUST_EXECUTE.format(tool="python_exec") in model.prompts[1], (
        "the second prompt does not carry the nudge naming the tool"
    )


def test_an_agent_that_did_call_its_tool_is_not_sent_back():
    agent, model = build(CALL_THEN_ANSWER,
                         [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    response = agent.run("Compute the 30th Fibonacci number.")
    assert int(response.tool_calls) >= 1
    assert forced_turns(model) == []


def test_an_adapter_that_cannot_require_a_call_degrades_to_the_nudge():
    agent, model = build([REFUSAL], [make_tool("python_exec", ToolCategory.CODE_EXECUTION)],
                         cls=Unforceable)
    agent.run("Compute the 30th Fibonacci number.")
    assert len(model.seen) >= 2, "the answer should still be sent back once"
    assert forced_turns(model) == [], (
        "a constraint was sent to an adapter that does not advertise it"
    )


# ---------------------------------------------------------------------------
# The shapes this was not written for
# ---------------------------------------------------------------------------
def test_a_retrieval_agent_is_untouched():
    """Answering without searching is a worse answer, not an impossible one."""
    agent, model = build([REFUSAL.replace("python_exec", "web_search")],
                         [make_tool("web_search", ToolCategory.INFORMATION_RETRIEVAL, "[1] x")])
    agent.run("Who wrote Dune?")
    assert len(model.seen) == 1
    assert forced_turns(model) == []


@pytest.mark.parametrize("category", [
    ToolCategory.COMPUTATION,
    ToolCategory.INFORMATION_RETRIEVAL,
    ToolCategory.EXTERNAL_API,
    ToolCategory.FILE_OPERATIONS,
    ToolCategory.DATA_PROCESSING,
    ToolCategory.COMMUNICATION,
])
def test_no_other_category_is_sent_back(category):
    agent, model = build([REFUSAL], [make_tool("some_tool", category)])
    agent.run("Answer this.")
    assert len(model.seen) == 1, f"{category.value} was refused an answer"


def test_an_agent_with_no_tools_is_untouched():
    model = Recorder(["Thought: done.\nFinal Answer: Paris"])
    agent = Agent(config=AgentConfig(
        name="probe", model=model, tools=[], raise_on_error=False))
    agent.run("Capital of France?")
    assert len(model.seen) == 1
    assert forced_turns(model) == []


# ---------------------------------------------------------------------------
# The nudge never reaches an answer
# ---------------------------------------------------------------------------
def test_the_nudge_is_stripped_in_both_of_its_forms():
    """It names a tool, so the template and the rendered text both have to go."""
    for text in (NUDGE_MUST_EXECUTE, NUDGE_MUST_EXECUTE.format(tool="python_exec")):
        cleaned = sanitize_final_answer(f"42 {text} done")
        assert text not in cleaned
        assert "42" in cleaned


def test_the_nudge_does_not_reach_the_answer_of_a_run_that_echoes_it():
    """A small model echoes the scratchpad back; the answer must still be clean."""
    echo = f"Final Answer: 42 {NUDGE_MUST_EXECUTE.format(tool='python_exec')}"
    agent, _ = build([REFUSAL, echo],
                     [make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    response = agent.run("Compute the 30th Fibonacci number.")
    assert "You have not run a tool yet" not in str(response)


# ---------------------------------------------------------------------------
# The capability probe
# ---------------------------------------------------------------------------
def test_a_model_that_does_not_answer_the_probe_is_read_as_no():
    class Mute:
        pass

    class Exploding:
        def supports_forced_tool_call(self):
            raise RuntimeError("boom")

    assert model_can_require_tool_call(None) is False
    assert model_can_require_tool_call(Mute()) is False
    assert model_can_require_tool_call(Exploding()) is False


def test_the_base_model_default_is_fail_closed():
    """An adapter that has not considered the constraint does not advertise it."""
    assert BaseModel.supports_forced_tool_call(object()) is False


def test_the_loop_flag_is_spent_on_read():
    guards = NativeToolLoop({"python_exec": make_tool("python_exec", ToolCategory.CODE_EXECUTION)})
    assert guards.take_forced_tool_call() is False
    assert guards.note_execution_refusal() == "python_exec"
    assert guards.take_forced_tool_call() is True
    assert guards.take_forced_tool_call() is False
    assert guards.note_execution_refusal() is None


# ---------------------------------------------------------------------------
# Per-adapter: the constraint reaches the request
# ---------------------------------------------------------------------------
OPENAI_SHAPED = [
    ("openai", "effgen.models.openai_adapter", "OpenAIAdapter", "gpt-4o-mini"),
    ("openai_compatible", "effgen.models.openai_compatible_adapter",
     "OpenAICompatibleAdapter", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("groq", "effgen.models.groq_adapter", "GroqAdapter", "llama-3.1-8b-instant"),
    ("together", "effgen.models.together_adapter", "TogetherAdapter",
     "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    ("fireworks", "effgen.models.fireworks_adapter", "FireworksAdapter",
     "accounts/fireworks/models/gpt-oss-120b"),
]

TOOLS = [{
    "name": "python_exec",
    "description": "Run python",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                   "required": ["code"]},
}]


def _openai_shaped_adapter(module, cls, model):
    adapter_cls = getattr(importlib.import_module(module), cls)
    kwargs = {"model_name": model, "api_key": "k"}
    if cls == "OpenAICompatibleAdapter":
        # A self-hosted endpoint has no default host; this one is never dialled.
        kwargs["base_url"] = "http://127.0.0.1:9/v1"
    try:
        adapter = adapter_cls(enable_rate_limiting=False, **kwargs)
    except TypeError:
        adapter = adapter_cls(**kwargs)
    client = MagicMock()
    for attr in ("_client", "client"):
        if hasattr(adapter, attr):
            setattr(adapter, attr, client)
    adapter._client = client
    adapter.client = client
    adapter._is_loaded = True
    message = MagicMock()
    message.content = "ok"
    message.tool_calls = None
    message.reasoning = None
    message.reasoning_content = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    # Real integers: the cost accounting does arithmetic on these, and a
    # MagicMock token count fails the comparison rather than the assertion.
    usage = MagicMock()
    usage.prompt_tokens = 1
    usage.completion_tokens = 1
    usage.total_tokens = 2
    usage.prompt_tokens_details = None
    usage.completion_tokens_details = None
    usage.cached_tokens = 0
    response.usage = usage
    client.chat.completions.create.return_value = response
    return adapter, client


@pytest.mark.parametrize("provider,module,cls,model", OPENAI_SHAPED)
def test_tool_choice_reaches_the_request(provider, module, cls, model):
    """The word the loop sends is the word the provider receives."""
    adapter, client = _openai_shaped_adapter(module, cls, model)
    adapter.generate_with_tools("hi", TOOLS, tool_choice="required")
    sent = client.chat.completions.create.call_args.kwargs
    assert sent.get("tool_choice") == "required", (
        f"{provider} sent {sent.get('tool_choice')!r}; the caller asked for "
        "'required' and the hard-coded default won"
    )


@pytest.mark.parametrize("provider,module,cls,model", OPENAI_SHAPED)
def test_the_default_is_still_auto(provider, module, cls, model):
    """Nothing changes for a caller who does not ask."""
    adapter, client = _openai_shaped_adapter(module, cls, model)
    adapter.generate_with_tools("hi", TOOLS)
    sent = client.chat.completions.create.call_args.kwargs
    assert sent.get("tool_choice") in (None, "auto")


@pytest.mark.parametrize("provider,module,cls,model", OPENAI_SHAPED)
def test_the_adapter_advertises_what_it_carries(provider, module, cls, model):
    adapter, _ = _openai_shaped_adapter(module, cls, model)
    assert adapter.supports_forced_tool_call() == adapter.supports_tool_calling()


def test_anthropic_translates_the_word_into_its_own_shape():
    """One vocabulary in, each provider's own spelling out."""
    from effgen.models.anthropic_adapter import _translate_tool_choice

    required = _translate_tool_choice({"tools": TOOLS, "tool_choice": "required"})
    assert required["tool_choice"] == {"type": "any"}

    auto = _translate_tool_choice({"tools": TOOLS, "tool_choice": "auto"})
    assert auto["tool_choice"] == {"type": "auto"}

    native = _translate_tool_choice(
        {"tools": TOOLS, "tool_choice": {"type": "tool", "name": "python_exec"}})
    assert native["tool_choice"] == {"type": "tool", "name": "python_exec"}


def test_a_constraint_without_tools_is_dropped_rather_than_sent():
    """Requiring a call from a request that offers none is rejected by the API."""
    from effgen.models.anthropic_adapter import _translate_tool_choice

    assert "tool_choice" not in _translate_tool_choice({"tool_choice": "required"})


def test_an_adapter_that_does_not_carry_it_says_so():
    """Gemini and Replicate spell the constraint differently and do not take it."""
    for module, cls in (
        ("effgen.models.gemini_adapter", "GeminiAdapter"),
        ("effgen.models.replicate_adapter", "ReplicateAdapter"),
    ):
        adapter_cls = getattr(importlib.import_module(module), cls)
        adapter = adapter_cls.__new__(adapter_cls)
        assert adapter.supports_forced_tool_call() is False, (
            f"{cls} advertises a constraint its request shaping does not carry"
        )
