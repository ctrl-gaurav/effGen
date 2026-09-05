"""How a model receives tool definitions, and what the loop does about it.

Three things are pinned here:

* ``tool_call_support()`` reports ``"api"``/``"template"``/``"none"``, derived
  from ``supports_tool_calling()`` by default so an adapter that only implements
  the boolean keeps working.
* A local chat template that accepts a ``tools`` argument and discards it does
  **not** count as tool-calling support — the model never sees the definitions.
* Every path that hands a model tool definitions states the contract for
  them, in the same words, whichever way the definitions travel.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import TEMPLATE_TOOL_USE_INSTRUCTION
from effgen.models._adapter_utils import TOOL_PROBE_NAME, chat_template_renders_tools
from effgen.models.base import BaseModel, ModelType
from effgen.models.lazy import LazyModel
from effgen.prompts.tool_contract import TOOL_CONTRACT_VERIFY
from effgen.tools import get_registry
from tests.fixtures.mock_models import MockModel

# ── Stand-in tokenizers ──────────────────────────────────────────────────
#
# vLLM needs an offline engine and MLX runs only on Apple silicon, so neither
# engine can be loaded here. Their capability check is the shared probe below,
# exercised directly against tokenizers that stand in for the two real
# behaviours: Qwen2.5/Llama 3.x render the definitions, gemma-2 and Phi-3.5
# accept the argument and drop it.


class RendersTools:
    """A chat template that writes the tool definitions into the prompt."""

    def apply_chat_template(self, messages, tools=None, **kwargs):
        rendered = "<|user|>hello<|assistant|>"
        if tools:
            names = ", ".join(t["function"]["name"] for t in tools)
            rendered = f"<|tools|>{names}<|/tools|>" + rendered
        return rendered


class IgnoresTools:
    """A chat template that accepts ``tools`` and silently discards it."""

    def apply_chat_template(self, messages, tools=None, **kwargs):
        return "<|user|>hello<|assistant|>"


class RejectsTools:
    """A chat template with no ``tools`` parameter at all."""

    def apply_chat_template(self, messages, **kwargs):
        return "<|user|>hello<|assistant|>"


class NoChatTemplate:
    """An object that is not a tokenizer."""


class TestChatTemplateProbe:

    def test_rendering_template_reports_support(self):
        assert chat_template_renders_tools(RendersTools()) is True

    def test_template_that_discards_tools_reports_no_support(self):
        assert chat_template_renders_tools(IgnoresTools()) is False

    def test_template_that_rejects_the_argument_reports_no_support(self):
        assert chat_template_renders_tools(RejectsTools()) is False

    def test_object_without_a_chat_template_reports_no_support(self):
        assert chat_template_renders_tools(NoChatTemplate()) is False

    def test_none_reports_no_support(self):
        assert chat_template_renders_tools(None) is False

    def test_probe_name_is_what_the_template_must_echo(self):
        seen = {}

        class Recorder:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                if tools is not None:
                    seen["names"] = [t["function"]["name"] for t in tools]
                    return "with tools " + seen["names"][0]
                return "plain"

        assert chat_template_renders_tools(Recorder()) is True
        assert seen["names"] == [TOOL_PROBE_NAME]

    def test_template_that_differs_without_naming_the_tool_reports_no_support(self):
        """Padding alone is not evidence the definitions were rendered."""

        class PadsOnly:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                return "plain" + ("!" * 500 if tools else "")

        assert chat_template_renders_tools(PadsOnly()) is False


# ── The capability signal on the engines ─────────────────────────────────


def _engine_with(cls, tokenizer, tokenizer_attr):
    """Build *cls* without loading a model and give it a stand-in tokenizer."""
    engine = cls.__new__(cls)
    engine._is_loaded = True
    engine._tool_template_probe = None
    if tokenizer_attr != "tokenizer":
        engine.tokenizer = None
    setattr(engine, tokenizer_attr, tokenizer)
    return engine


def _engine_cases():
    """The local engines, or no cases at all when their backend is absent.

    The local engines import torch at module scope, so on an install without it
    this collects nothing instead of failing the whole module — the tool-call
    contracts below that use a cloud adapter still run.
    """
    try:
        from effgen.models.mlx_engine import MLXEngine
        from effgen.models.transformers_engine import TransformersEngine
        from effgen.models.vllm_engine import VLLMEngine
    except ImportError as exc:  # pragma: no cover - only on an install without torch
        return [pytest.param(None, None, marks=pytest.mark.skip(reason=str(exc)))]

    return [
        (TransformersEngine, "tokenizer"),
        (VLLMEngine, "_hf_tokenizer"),
        (MLXEngine, "tokenizer"),
    ]


@pytest.mark.parametrize("cls,attr", _engine_cases())
class TestTemplateEngineCapability:

    def test_rendering_template_is_template_support(self, cls, attr):
        engine = _engine_with(cls, RendersTools(), attr)
        assert engine.supports_tool_calling() is True
        assert engine.tool_call_support() == "template"

    def test_discarding_template_is_no_support(self, cls, attr):
        engine = _engine_with(cls, IgnoresTools(), attr)
        assert engine.supports_tool_calling() is False
        assert engine.tool_call_support() == "none"

    def test_missing_tokenizer_is_no_support(self, cls, attr):
        engine = _engine_with(cls, None, attr)
        assert engine.supports_tool_calling() is False
        assert engine.tool_call_support() == "none"

    def test_unloaded_engine_is_no_support(self, cls, attr):
        engine = _engine_with(cls, RendersTools(), attr)
        engine._is_loaded = False
        assert engine.supports_tool_calling() is False
        assert engine.tool_call_support() == "none"

    def test_result_is_cached_per_tokenizer(self, cls, attr):
        calls = []

        class Counting(RendersTools):
            def apply_chat_template(self, messages, tools=None, **kwargs):
                calls.append(tools is not None)
                return super().apply_chat_template(messages, tools=tools, **kwargs)

        engine = _engine_with(cls, Counting(), attr)
        assert engine.supports_tool_calling() is True
        rendered_once = len(calls)
        for _ in range(4):
            assert engine.supports_tool_calling() is True
        assert len(calls) == rendered_once, "probe re-rendered instead of using the cache"

    def test_a_new_tokenizer_is_re_measured(self, cls, attr):
        engine = _engine_with(cls, RendersTools(), attr)
        assert engine.supports_tool_calling() is True
        setattr(engine, attr, IgnoresTools())
        assert engine.supports_tool_calling() is False
        assert engine.tool_call_support() == "none"


# ── The default signal on every other model ──────────────────────────────


class TestDerivedDefault:

    def test_base_default_is_none(self):
        class Plain(BaseModel):
            def load(self):  # pragma: no cover - never called
                pass

            def generate(self, prompt, config=None, **kwargs):  # pragma: no cover
                pass

            def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
                pass

            def count_tokens(self, text):  # pragma: no cover
                pass

            def get_context_length(self):  # pragma: no cover
                return 2048

            def unload(self):  # pragma: no cover
                pass

        model = Plain(model_name="plain", model_type=ModelType.TRANSFORMERS)
        assert model.supports_tool_calling() is False
        assert model.tool_call_support() == "none"

    def test_an_adapter_that_only_sets_the_boolean_reports_api(self):
        """No adapter or test double needs editing for the new signal."""

        class Adapter(MockModel):
            def supports_tool_calling(self):
                return True

        assert Adapter(responses=["x"]).tool_call_support() == "api"

    def test_lazy_model_delegates_both_signals(self):
        class Adapter(MockModel):
            def supports_tool_calling(self):
                return True

        inner = Adapter(responses=["x"])
        lazy = LazyModel(inner)
        assert lazy.supports_tool_calling() is True
        assert lazy.tool_call_support() == "api"


# ── The tool-use line in the prompt ──────────────────────────────────────


class _SupportModel(MockModel):
    """A mock reporting a chosen ``tool_call_support()``."""

    _support = "none"

    def supports_tool_calling(self):
        return self._support != "none"

    def tool_call_support(self):
        return self._support


class _NoSignalModel(MockModel):
    """A model with no ``tool_call_support`` attribute at all."""

    def __getattribute__(self, name):
        if name == "tool_call_support":
            raise AttributeError(name)
        return super().__getattribute__(name)

    def supports_tool_calling(self):
        return True


def _record_prompts(model, *, with_tools: bool = True) -> list[str]:
    """Every prompt one run hands the model."""
    tools = [get_registry().get_tool_sync("calculator")] if with_tools else []
    agent = Agent(
        config=AgentConfig(
            name="prompt-test",
            model=model,
            tools=tools,
            enable_memory=False,
            enable_sub_agents=False,
            max_iterations=1,
        )
    )
    seen: list[str] = []
    original = agent._generate

    def record(prompt, **kwargs):
        seen.append(prompt)
        return original(prompt, **kwargs)

    agent._generate = record
    agent.run("Use the calculator tool to compute 1367 * 89.")
    return seen


def _prompts_from_run(support: str, *, with_tools: bool) -> list[str]:
    model = _SupportModel(responses=["Final Answer: 121663"] * 4)
    model._support = support
    return _record_prompts(model, with_tools=with_tools)


class TestWrittenCallRemediationNamesTheDelivery:
    """A written-out tool call is explained by how the definitions arrived."""

    def _detail(self, support: str):
        model = _SupportModel(responses=["Final Answer: x"])
        model._support = support
        agent = Agent(
            config=AgentConfig(
                name="written",
                model=model,
                tools=[get_registry().get_tool_sync("calculator")],
                enable_memory=False,
                enable_sub_agents=False,
            )
        )
        return agent._written_tool_call_detail("calculator", "calculator{}")

    def test_template_model_is_told_about_its_chat_template(self):
        message = self._detail("template")["message"]
        assert "chat template" in message
        assert "provider's tool-calling API" not in message

    def test_api_model_is_told_about_the_provider_api(self):
        message = self._detail("api")["message"]
        assert "provider's tool-calling API" in message
        assert "chat template" not in message

    def test_a_model_without_tool_calling_is_told_that(self):
        message = self._detail("none")["message"]
        assert "does not advertise native tool calling" in message


class TestNativeModeOnAModelWithNoDefinitions:
    """``native`` still reaches the tool when the definitions cannot be sent.

    A model whose chat template drops tool definitions is prompted as ReAct,
    because that is the only prompt that can reach a tool. Reading such a turn
    with the native reader alone finds neither a call nor an answer, and the
    run repeats the same turn to its iteration cap and ends with nothing.
    """

    REACT_TEXT = (
        "Thought: I will use the calculator.\n"
        'Action: calculator\nAction Input: {"expression": "1367 * 89"}'
    )
    NATIVE_TEXT = (
        "<tool_call>\n"
        '{"name": "calculator", "arguments": {"expression": "1367 * 89"}}\n'
        "</tool_call>"
    )

    def _run(self, support: str, mode: str, text: str):
        model = _SupportModel(responses=[text] * 12)
        model._support = support
        agent = Agent(
            config=AgentConfig(
                name="native-mode",
                model=model,
                tools=[get_registry().get_tool_sync("calculator")],
                tool_calling_mode=mode,
                max_iterations=10,
                enable_memory=False,
                enable_sub_agents=False,
            )
        )
        return agent.run("Use the calculator tool to compute 1367 * 89.")

    @pytest.mark.parametrize("mode", ["native", "hybrid", "react", "auto"])
    def test_every_mode_reaches_the_tool(self, mode):
        response = self._run("none", mode, self.REACT_TEXT)
        assert response.tool_calls == 1, f"{mode} never called the tool"
        assert response.iterations == 1, f"{mode} took {response.iterations} iterations"
        assert "121663" in str(response)

    def test_the_reported_strategy_is_still_the_one_asked_for(self):
        response = self._run("none", "native", self.REACT_TEXT)
        assert response.metadata["tool_calling_strategy"] == "native"

    @pytest.mark.parametrize("support", ["template", "api"])
    def test_a_model_that_can_receive_definitions_still_parses_native_syntax(self, support):
        response = self._run(support, "native", self.NATIVE_TEXT)
        assert response.tool_calls == 1
        assert response.iterations == 1
        assert "121663" in str(response)


class TestEveryToolPathStatesItsContract:
    """What a tool-holding prompt says about the tools, on every delivery.

    Tool definitions travel to the model outside the prompt -- through the
    provider's tool-calling API or a local chat template -- so the prompt is
    the only place they can be described. Before, a chat-template model got one
    line and a provider-side one got nothing at all, which meant the same agent
    was told two different things depending on which adapter was loaded. Now
    both get the contract for the tools they hold, and the superseded line is
    stated by neither.
    """

    def test_a_template_model_with_tools_gets_the_contract(self):
        prompts = _prompts_from_run("template", with_tools=True)
        assert any(TOOL_CONTRACT_VERIFY in p for p in prompts)

    def test_the_contract_closes_the_prompt_after_the_task(self):
        prompts = _prompts_from_run("template", with_tools=True)
        assert prompts[0].endswith("\n\n" + TOOL_CONTRACT_VERIFY)

    def test_an_api_model_gets_the_same_contract(self):
        """The path every OpenAI-protocol provider takes said nothing before."""
        prompts = _prompts_from_run("api", with_tools=True)
        assert prompts[0].endswith("\n\n" + TOOL_CONTRACT_VERIFY)

    def test_the_two_deliveries_produce_the_same_opening_prompt(self):
        assert (_prompts_from_run("template", with_tools=True)[0]
                == _prompts_from_run("api", with_tools=True)[0])

    def test_the_superseded_line_reaches_no_prompt(self):
        for support in ("template", "api"):
            prompts = _prompts_from_run(support, with_tools=True)
            assert not any(TEMPLATE_TOOL_USE_INSTRUCTION in p for p in prompts)

    def test_a_tool_free_run_states_no_contract(self):
        prompts = _prompts_from_run("template", with_tools=False)
        assert not any(TOOL_CONTRACT_VERIFY in p for p in prompts)
        assert not any(TEMPLATE_TOOL_USE_INSTRUCTION in p for p in prompts)

    def test_an_adapter_that_only_sets_the_boolean_still_gets_it(self):
        """A subclass predating the finer signal inherits ``"api"``."""
        model = MockModel(responses=["Final Answer: ok"] * 4)
        model.supports_tool_calling = lambda: True
        assert model.tool_call_support() == "api"
        assert any(TOOL_CONTRACT_VERIFY in p for p in _record_prompts(model))

    def test_a_model_object_lacking_the_method_gets_it_too(self):
        """A model that is not a ``BaseModel`` has no finer signal to read.

        It still advertises tool calling and still receives the definitions, so
        the branch that hands them over states what they are for.
        """
        model = _NoSignalModel(responses=["Final Answer: ok"] * 4)
        assert not hasattr(model, "tool_call_support")
        assert any(TOOL_CONTRACT_VERIFY in p for p in _record_prompts(model))

    def test_a_signal_that_raises_changes_nothing_about_the_run(self):
        """A probe that fails still gets the contract its branch calls for."""

        class Raises(MockModel):
            def supports_tool_calling(self):
                return True

            def tool_call_support(self):
                raise RuntimeError("probe exploded")

        model = Raises(responses=["Final Answer: ok"] * 4)
        prompts = _record_prompts(model)
        assert prompts, "the run produced no prompt at all"
        assert any(TOOL_CONTRACT_VERIFY in p for p in prompts)
