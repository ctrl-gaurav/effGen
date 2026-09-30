"""What a model does when handed a tool is measured once, kept, and acted on.

The declared tool-calling support of a model served behind a URL says how the
definitions reach it, not whether it uses them. These tests drive the probe
with scripted models whose behaviour is known — one that calls and resolves,
one that answers from memory, one whose native calls never resolve — and pin:

- the counts and the defaults derived from them;
- at most one probe per key, however many agents are built at once;
- no probe, and no request, for an adapter that declares no key;
- ``tool_calling_mode``, ``tool_use``, ``capability_probe=False`` and
  ``EFFGEN_CAPABILITY_PROBE=0`` each keep the declared behaviour;
- a probe that fails leaves nothing stored and the agent as it was;
- the must-call set grows by information tools only, never computation;
- a provider that refuses stop sequences beside tools is retried once without
  them, and the refusal is remembered.
"""
from __future__ import annotations

import json
import logging
import threading

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.core.tool_calling import HybridStrategy, ReActStrategy, get_strategy
from effgen.models import capability_probe as cp
from effgen.models._usage import tool_call_entry
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.models.errors import InvalidRequestError
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)


@pytest.fixture(autouse=True)
def _probe_on(monkeypatch, tmp_path):
    """Each test gets the probe switched on and a store of its own."""
    monkeypatch.setenv("EFFGEN_CAPABILITY_PROBE", "1")
    monkeypatch.setenv("EFFGEN_CAPABILITY_CACHE", str(tmp_path / "capabilities.json"))
    cp._FAILED.clear()
    yield
    cp._FAILED.clear()


def make_tool(name: str, category: ToolCategory, payload: str = "42") -> BaseTool:
    class _T(BaseTool):
        def __init__(self) -> None:
            super().__init__(metadata=ToolMetadata(
                name=name, description=f"The {name} tool.", category=category,
                parameters=[ParameterSpec(
                    name="query", type=ParameterType.STRING,
                    description="Input.", required=True)],
            ))

        async def _execute(self, **kw):
            return payload
    return _T()


class Scripted(BaseModel):
    """A served model whose tool behaviour is fixed by *kind*.

    ``resolve``    calls the tool natively, then answers with what it returned.
    ``skip``       answers every question from memory, with no call.
    ``skip_some``  answers the count question from memory, resolves the rest.
    ``native_broken`` in the native frame calls and never reads the result;
                   in the text frame calls and resolves.
    ``fail``       every request raises.
    """

    def __init__(self, kind: str, key: str | None = "scripted-key") -> None:
        super().__init__(model_name=f"scripted-{kind}", model_type=ModelType.OPENAI)
        self.kind = kind
        self.key = key
        self.calls = 0
        self.lock = threading.Lock()
        self.seen: list[dict] = []
        self._is_loaded = True

    def load(self): pass

    def unload(self): pass

    def capability_key(self):
        return self.key

    def supports_tool_calling(self):
        return True

    def tool_call_support(self):
        return "api"

    def supports_forced_tool_call(self):
        return True

    def streams_tool_calls(self):
        return False

    def count_tokens(self, t):
        return TokenCount(count=len(str(t).split()), model_name=self.model_name)

    def get_context_length(self):
        return 8192

    def generate_batch(self, ps, config=None, **kw):
        return [self.generate(p, config) for p in ps]

    def generate_stream(self, prompt, config=None, **kw):
        yield self.generate(prompt, config, **kw).text

    def generate_with_tools(self, p, tools, config=None, **kw):
        return self.generate(p, config, tools=tools, **kw)

    def _answer(self, text: str, calls=None) -> GenerationResult:
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop", model_name=self.model_name,
            metadata={"tool_calls": calls or [], "prompt_tokens": 20, "completion_tokens": 5},
        )

    def generate(self, prompt, config=None, **kw):
        with self.lock:
            self.calls += 1
            self.seen.append({"tools": bool(kw.get("tools")),
                              "stop": list(getattr(config, "stop_sequences", None) or [])})
        if self.kind == "fail":
            raise InvalidRequestError("scripted", self.model_name, "refused")
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
        observed = "[Result 1]" in text
        native = bool(kw.get("tools"))
        fact = text.split("[Result 1]")[-1].split("\\n")[0].split("\n")[0] if observed else ""
        if self.kind == "skip" or (self.kind == "skip_some" and "Tessavar" in text):
            return self._answer("Final Answer: It was founded in 1900 by settlers.")
        if self.kind == "native_broken" and native:
            return self._answer("", [tool_call_entry("web_search", {"query": "same"}, call_id="c1")])
        if observed:
            return self._answer(f"Final Answer: {fact.strip()}")
        if native:
            return self._answer("", [tool_call_entry("web_search", {"query": "q"}, call_id="c1")])
        return self._answer('Thought: search.\nAction: web_search\nAction Input: {"query": "q"}')


def agent_for(model, tools=None, **cfg) -> Agent:
    tools = tools if tools is not None else [
        make_tool("web_search", ToolCategory.INFORMATION_RETRIEVAL, "[Result 1] Mirela")]
    return Agent(config=AgentConfig(name="a", model=model, tools=tools, max_iterations=4,
                                    raise_on_error=False, **cfg))


# ---------------------------------------------------------------------------
# The rule the counts are read with
# ---------------------------------------------------------------------------


def test_derive_defaults_thresholds():
    assert cp.derive_defaults({"resolved": 8, "unresolved": 0, "skipped": 0}, None) == ("declared", ())
    assert cp.derive_defaults({"resolved": 7, "unresolved": 0, "skipped": 1}, None) == (
        "declared", ("information_retrieval",))
    assert cp.derive_defaults({"resolved": 2, "unresolved": 6, "skipped": 0},
                              {"resolved": 5, "unresolved": 3, "skipped": 0}) == ("react", ())
    # Unresolved natively, but the text frame is no better: stay as declared.
    assert cp.derive_defaults({"resolved": 2, "unresolved": 6, "skipped": 0},
                              {"resolved": 4, "unresolved": 4, "skipped": 0}) == ("declared", ())


def test_get_strategy_reads_the_probe_only_at_auto():
    probe = cp.ToolCallingProbe(key="k", model="m", endpoint=None, backend="B",
                                native={}, text={}, strategy="react", required_categories=())
    model = Scripted("resolve")
    assert isinstance(get_strategy("auto", model), HybridStrategy)
    assert isinstance(get_strategy("auto", model, probe=probe), ReActStrategy)
    assert isinstance(get_strategy("hybrid", model, probe=probe), HybridStrategy)
    declared = cp.ToolCallingProbe(key="k", model="m", endpoint=None, backend="B",
                                   native={}, text=None, strategy="declared",
                                   required_categories=())
    assert isinstance(get_strategy("auto", model, probe=declared), HybridStrategy)


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def test_a_model_that_resolves_is_left_as_declared():
    model = Scripted("resolve")
    agent = agent_for(model)
    assert agent._tool_calling_strategy.name == "hybrid"
    assert agent._probe_required_categories == frozenset()
    entry = next(iter(cp.read_store()["probes"].values()))
    assert entry["native"] == {"resolved": 8, "unresolved": 0, "skipped": 0}
    assert entry["text"] is None
    assert entry["requests"] == model.calls == 16


def test_a_model_that_skips_must_call_information_tools(caplog):
    model = Scripted("skip_some")
    with caplog.at_level(logging.INFO):
        agent = agent_for(model)
    assert agent._tool_calling_strategy.name == "hybrid"
    assert agent._probe_required_categories == frozenset({"information_retrieval"})
    assert "capability probe: ran scripted-skip_some native resolved=7 unresolved=0 skipped=1" in caplog.text
    assert "required_categories=information_retrieval" in caplog.text


def test_native_calls_that_never_resolve_move_auto_to_the_text_frame():
    model = Scripted("native_broken")
    agent = agent_for(model)
    probe = cp.probe_tool_calling(model)
    assert probe.source == "cached"
    assert probe.native["unresolved"] == 8 and probe.text["resolved"] == 8
    assert agent._tool_calling_strategy.name == "react"
    assert agent._tool_calling_source == "probe"


def test_at_most_one_probe_per_key_across_concurrent_agents(caplog):
    model = Scripted("resolve")
    agents: list[Agent] = []
    lock = threading.Lock()

    def build():
        a = agent_for(model)
        with lock:
            agents.append(a)

    with caplog.at_level(logging.INFO):
        threads = [threading.Thread(target=build) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        agent_for(model)
    assert len(agents) == 8
    assert model.calls == 16  # one probe's requests
    assert caplog.text.count("capability probe: ran ") == 1
    assert caplog.text.count("capability probe: cached ") == 8


def test_an_adapter_with_no_key_is_never_probed():
    model = Scripted("resolve", key=None)
    agent = agent_for(model)
    assert model.calls == 0
    assert cp.read_store()["probes"] == {}
    assert agent._tool_calling_source == "declared"


@pytest.mark.parametrize("cfg", [
    {"tool_calling_mode": "hybrid"},
    {"capability_probe": False},
])
def test_explicit_configuration_skips_the_probe(cfg):
    model = Scripted("skip")
    agent = agent_for(model, **cfg)
    assert model.calls == 0
    assert agent._probe_required_categories == frozenset()


@pytest.mark.parametrize("value", ["false", 0, None])
def test_a_capability_probe_that_is_not_a_bool_is_refused(value):
    with pytest.raises(ValueError, match="capability_probe"):
        AgentConfig(name="probe-config", model=Scripted("skip"), capability_probe=value)


def test_environment_switch_skips_the_probe(monkeypatch):
    monkeypatch.setenv("EFFGEN_CAPABILITY_PROBE", "0")
    model = Scripted("skip")
    agent_for(model)
    assert model.calls == 0


def test_a_stated_tool_use_policy_wins_over_the_probe():
    model = Scripted("skip")
    agent = agent_for(model, tool_use="auto")
    assert model.calls == 8  # the probe ran: one answer per item, no calls
    assert agent._probe_required_categories == frozenset()


def test_an_agent_without_tools_is_not_probed():
    model = Scripted("skip")
    agent_for(model, tools=[])
    assert model.calls == 0


def test_a_failing_probe_stores_nothing_and_keeps_the_declared_strategy(caplog):
    model = Scripted("fail")
    with caplog.at_level(logging.WARNING):
        agent = agent_for(model)
        agent_for(model)
    assert agent._tool_calling_strategy.name == "hybrid"
    assert cp.read_store()["probes"] == {}
    assert caplog.text.count("capability probe: could not run for scripted-fail") == 1
    assert model.calls == 1  # the second agent does not try again in this process


def test_a_corrupt_store_is_ignored_and_rewritten(tmp_path, caplog):
    path = tmp_path / "capabilities.json"
    path.write_text("{not json")
    model = Scripted("resolve")
    with caplog.at_level(logging.WARNING):
        agent_for(model)
    assert "could not be read" in caplog.text
    assert json.loads(path.read_text())["probes"]


def test_a_stored_probe_is_reused_and_refresh_measures_again():
    model = Scripted("resolve")
    first = cp.probe_tool_calling(model)
    assert first.source == "ran" and model.calls == 16
    assert cp.probe_tool_calling(model).source == "cached" and model.calls == 16
    again = cp.probe_tool_calling(model, refresh=True)
    assert again.source == "ran" and model.calls == 32


def test_a_probe_over_its_request_budget_reports_nothing(monkeypatch):
    monkeypatch.setattr(cp, "MAX_PROBE_REQUESTS", 4)
    model = Scripted("resolve")
    assert cp.probe_tool_calling(model) is None
    assert cp.read_store()["probes"] == {}


def test_the_run_says_why_it_ran_as_it_did():
    model = Scripted("skip_some")
    agent = agent_for(model)
    response = agent.run("What is the population of Tessavar?")
    info = response.metadata["tool_calling"]
    assert info["strategy"] == "hybrid" and info["source"] == "probe"
    assert info["required_categories"] == ["information_retrieval"]
    assert info["probe_key"]


# ---------------------------------------------------------------------------
# What the probe's policy moves, and what it never moves
# ---------------------------------------------------------------------------


def test_required_categories_add_information_tools_only():
    search = make_tool("web_search", ToolCategory.INFORMATION_RETRIEVAL)
    calc = make_tool("calculator", ToolCategory.COMPUTATION)
    loop = NativeToolLoop({"web_search": search, "calculator": calc},
                          required_categories=frozenset({"information_retrieval"}))
    assert loop.execution_tools() == ["web_search"]
    assert NativeToolLoop({"calculator": calc},
                          required_categories=frozenset({"information_retrieval"})
                          ).execution_tools() == []
    assert NativeToolLoop({"web_search": search}).execution_tools() == []


def test_a_skipping_model_is_sent_back_once_to_search(caplog):
    model = Scripted("skip_some")
    agent = agent_for(model)
    model.kind = "skip"
    with caplog.at_level(logging.INFO):
        agent.run("Who founded Veldmark?")
    assert caplog.text.count("execution refusal:") == 1
    assert caplog.text.count("capability probe policy: refusal fired for 'web_search'") == 1


def test_a_code_tool_refusal_is_not_counted_as_the_probe_s(caplog):
    model = Scripted("skip_some")
    agent = agent_for(model, tools=[make_tool("python_exec", ToolCategory.CODE_EXECUTION)])
    model.kind = "skip"
    with caplog.at_level(logging.INFO):
        agent.run("Print 2 + 2.")
    assert caplog.text.count("execution refusal:") == 1  # the tool's own category asks for it
    assert "capability probe policy:" not in caplog.text


def test_a_calculator_is_never_forced_by_the_probe(caplog):
    model = Scripted("skip_some")
    agent = agent_for(model, tools=[make_tool("calculator", ToolCategory.COMPUTATION)])
    model.kind = "skip"
    with caplog.at_level(logging.INFO):
        agent.run("What is 2 + 2?")
    assert "execution refusal:" not in caplog.text


# ---------------------------------------------------------------------------
# Learned: stop sequences beside tools
# ---------------------------------------------------------------------------


class RejectsStopWithTools(Scripted):
    """A provider that answers 400 to a request carrying tools and stop sequences."""

    def generate(self, prompt, config=None, **kw):
        if kw.get("tools") and getattr(config, "stop_sequences", None):
            with self.lock:
                self.calls += 1
                self.seen.append({"rejected": True})
            err = InvalidRequestError("scripted", self.model_name, "stop is not supported with tools")
            raise err
        return super().generate(prompt, config, **kw)


def test_a_refused_stop_is_retried_once_and_remembered(caplog):
    model = RejectsStopWithTools("resolve", key=None)
    agent = agent_for(model)
    with caplog.at_level(logging.INFO):
        first = agent.run("Who designed the bridge?")
    assert first.success
    rejected = sum(1 for s in model.seen if s.get("rejected"))
    assert rejected == 1
    assert caplog.text.count("rejects stop sequences beside tools") == 1
    before = len(model.seen)
    second = agent_for(model).run("Who designed the bridge?")
    assert second.success
    assert not any(s.get("rejected") for s in model.seen[before:])


def test_an_accepting_provider_is_never_retried():
    model = Scripted("resolve", key=None)
    agent_for(model).run("Who designed the bridge?")
    assert any(s.get("stop") for s in model.seen if s.get("tools"))
    assert cp.read_store()["learned"] == {}


# ---------------------------------------------------------------------------
# reasoning_effort
# ---------------------------------------------------------------------------


def test_a_pinned_effort_the_adapter_does_not_send_is_reported_once(caplog):
    from effgen.models import _adapter_utils

    _adapter_utils._reasoning_effort_warned.clear()
    model = Scripted("resolve", key=None)
    agent = agent_for(model, tools=[])
    with caplog.at_level(logging.WARNING):
        agent.run("hi", reasoning_effort="high")
        agent.run("hi", reasoning_effort="high")
    assert caplog.text.count("reasoning_effort dropped:") == 1


def test_the_compatible_adapter_sends_a_pinned_effort_with_sampling():
    from effgen.models.base import GenerationConfig
    from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter

    adapter = OpenAICompatibleAdapter("served", base_url="http://127.0.0.1:9/v1",
                                      context_length=4096)
    params = adapter._build_request_params(
        [{"role": "user", "content": "hi"}],
        GenerationConfig(temperature=0.3, reasoning_effort="low"),
    )
    assert params["reasoning_effort"] == "low"
    assert params["temperature"] == 0.3
    cp.remember_fact(adapter, "reasoning_effort", False)
    params = adapter._build_request_params(
        [{"role": "user", "content": "hi"}], GenerationConfig(reasoning_effort="low"),
    )
    assert "reasoning_effort" not in params


def test_the_groq_adapter_sends_a_pinned_effort_to_a_reasoning_model():
    pytest.importorskip("groq")
    from effgen.models.base import GenerationConfig
    from effgen.models.groq_adapter import GroqAdapter

    adapter = GroqAdapter.__new__(GroqAdapter)
    adapter.model_name = "openai/gpt-oss-20b"
    adapter._is_reasoning_model = True
    params: dict = {}
    adapter._apply_reasoning_effort(params, GenerationConfig(reasoning_effort="high"))
    assert params == {"reasoning_effort": "high"}
    adapter._is_reasoning_model = False
    params = {}
    adapter._apply_reasoning_effort(params, GenerationConfig(reasoning_effort="high"))
    assert params == {}


@pytest.mark.parametrize("module, cls", [
    ("effgen.models.anthropic_adapter", "AnthropicAdapter"),
    ("effgen.models.gemini_adapter", "GeminiAdapter"),
    ("effgen.models.together_adapter", "TogetherAdapter"),
    ("effgen.models.fireworks_adapter", "FireworksAdapter"),
    ("effgen.models.cerebras_adapter", "CerebrasAdapter"),
    ("effgen.models.hf_inference_adapter", "HFInferenceAdapter"),
    ("effgen.models.replicate_adapter", "ReplicateAdapter"),
    ("effgen.models.transformers_engine", "TransformersEngine"),
])
def test_every_other_adapter_declares_it_does_not_send_the_effort(module, cls):
    import importlib

    klass = getattr(importlib.import_module(module), cls)
    instance = klass.__new__(klass)
    assert instance.forwards_reasoning_effort() is False
    assert instance.capability_key() is None or cls == "TransformersEngine"


# ---------------------------------------------------------------------------
# Delegation and the report
# ---------------------------------------------------------------------------


def test_a_lazy_model_delegates_and_is_not_probed_before_it_loads():
    from effgen.models.lazy import LazyModel

    inner = Scripted("resolve")
    lazy = LazyModel(inner)
    inner._is_loaded = False
    assert lazy.capability_key() is None
    inner._is_loaded = True
    assert lazy.capability_key() == "scripted-key"
    assert lazy.supports_stop_with_tools() is True
    assert lazy.forwards_reasoning_effort() is False


def test_doctor_reports_the_store():
    from effgen.cli.commands.doctor import _capability_rows, _doctor_capabilities_report

    agent_for(Scripted("skip_some"))
    cp.remember_fact(Scripted("resolve", key=None), "stop_with_tools", False, detail="x")
    report = _doctor_capabilities_report()
    assert report["probes"][0]["required_categories"] == ["information_retrieval"]
    assert report["probes"][0]["native"]["skipped"] == 1
    assert report["learned"][0]["capability"] == "stop_with_tools"
    rows = _capability_rows(report)
    assert rows[0][2] == "7/0/1" and rows[1][6] == "stop_with_tools=False"


def test_persistence_off_keeps_the_store_in_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFGEN_CAPABILITY_CACHE", "off")
    monkeypatch.setattr(cp, "_MEMORY_STORE", {"probes": {}, "learned": {}})
    model = Scripted("resolve")
    agent_for(model)
    agent_for(model)
    assert model.calls == 16
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# What the probe tells the model is fixed, whatever the tool contract says
# ---------------------------------------------------------------------------


class _Recording(Scripted):
    """A scripted model that also keeps the text of every request."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.prompts: list[str] = []

    def generate(self, prompt, config=None, **kw):
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
        with self.lock:
            self.prompts.append(text)
        return super().generate(prompt, config, **kw)


@pytest.mark.parametrize("kind", ["resolve", "native_broken"])
def test_the_probe_states_its_own_contract_when_the_lookup_contract_changes(monkeypatch, kind):
    """The thresholds were calibrated against one text; a new lookup contract must not move them."""
    from effgen.prompts import tool_contract as tc

    changed = "A different sentence about search tools."
    monkeypatch.setitem(tc.TOOL_CONTRACTS, ToolCategory.INFORMATION_RETRIEVAL, changed)
    model = _Recording(kind)
    probe = cp.probe_tool_calling(model)
    assert probe is not None and probe.source == "ran"
    assert model.prompts, "the probe sent no request"
    assert not [text for text in model.prompts if changed in text]
    pinned = json.dumps(cp._PROBE_TOOL_CONTRACT)[1:-1]
    stating = [text for text in model.prompts if cp._PROBE_TOOL_CONTRACT in text or pinned in text]
    # The contract opens each run (a later turn carries the run's own steps instead).
    assert len(stating) >= 8


def test_the_probes_copy_is_the_shipped_lookup_text():
    """The copy may not drift from the shipped text without a new probe version.

    A lookup text that makes a model search less often reads differently in the
    probe, so the two move together: a new lookup contract comes with a new copy
    and a new ``PROBE_VERSION``.
    """
    from effgen.prompts.tool_contract import TOOL_CONTRACT_LOOKUP

    assert cp._PROBE_TOOL_CONTRACT == TOOL_CONTRACT_LOOKUP
