"""Input configuration for :class:`effgen.core.agent.Agent`.

Holds the agent's execution-mode enum and the :class:`AgentConfig` dataclass
that describes an agent before it runs — the model, tools, sampling defaults,
memory/routing options, and callbacks — plus the ``run()`` keyword allow-list
and the constructor guard that turns a stray model-loading option into an
actionable message. This module imports nothing from ``agent.py`` so it stays a
dependency-free leaf; ``agent.py`` re-exports these names, so
``from effgen.core.agent import AgentMode, AgentConfig`` is unchanged.
Behaviour is identical to the original in-module definitions.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models.base import BaseModel
from ..tools.base_tool import BaseTool


class AgentMode(Enum):
    """Agent execution modes."""
    SINGLE = "single"  # Single agent execution
    SUB_AGENTS = "sub_agents"  # Use sub-agents for complex tasks
    AUTO = "auto"  # Automatically decide based on router


@dataclass
class AgentConfig:
    """
    Agent configuration.

    Attributes:
        name: Agent name/identifier. Optional — defaults to the model id (or
            "agent" for a model instance) when omitted, so
            ``AgentConfig(model=...)`` works without boilerplate.
        model: Model instance or name
        tools: List of available tools
        system_prompt: System-level instructions
        max_iterations: Maximum tool-use loop iterations
        temperature: Generation temperature
        top_p: Nucleus-sampling threshold; overridden per call by run(top_p=...)
        top_k: Top-k sampling cutoff (providers that don't support it ignore it)
        seed: Sampling seed. A fixed seed plus temperature=0 reproduces a
            generation exactly on Gemini, Groq, and local engines
            (transformers/vllm/gguf/mlx). OpenAI's chat models accept
            ``seed`` and typically reproduce output, but the same
            seed+temperature=0 request can still return a different
            completion — OpenAI documents this as best-effort determinism,
            not a guarantee, especially for reasoning-tier models. Treat an
            OpenAI ``seed`` as "usually reproducible," not "always."
        presence_penalty: Penalizes tokens already present anywhere in the text
        frequency_penalty: Penalizes tokens proportionally to how often they
            already appeared (the standard anti-repetition knob for long text)
        repetition_penalty: Multiplicative repeat penalty used by local/HF engines
        mode: Default execution mode for run()/run_async() when the call
            site doesn't pass its own ``mode=``. Defaults to
            ``AgentMode.SINGLE`` so a plain ``Agent(config).run(task)`` never
            switches to sub-agent decomposition on its own; set to
            ``AgentMode.AUTO`` to have the router decide per call, or pass
            ``mode=`` on an individual ``run()`` call to override this
            default just for that call.
        enable_sub_agents: Enable sub-agent spawning
        enable_memory: Enable memory systems
        enable_streaming: Enable response streaming
        max_context_length: Maximum context window
        router_config: Configuration for sub-agent router
        sub_agent_config: Configuration for sub-agent manager
        model_config: Optional model engine configuration
        require_model: Whether a string model must load at construction time.
            Defaults to True so a typo'd id / missing key fails immediately
            instead of building a working-looking agent that only crashes on
            the first run(). Set False to defer loading (advanced use).
        provider: Optional explicit provider for a string ``model`` (e.g.
            "openai", "cerebras"). Equivalent to the "provider:model" prefix
            and the CLI ``--provider`` flag; resolves bare ids that exist on
            multiple providers.
        base_url: Endpoint for a server speaking the OpenAI protocol (vLLM,
            SGLang, TGI, llama.cpp, Ollama, LM Studio, a gateway). Giving one
            makes a string ``model`` load through the OpenAI-compatible adapter
            against that URL instead of loading the weights in this process.
            Ignored when ``model`` is already a loaded model instance.
        api_key: Credential for that endpoint. A local server that checks
            nothing needs none.
        middleware: Hooks to run around the run, each model call and each tool
            call — see :mod:`effgen.core.middleware`. Anything effGen does not
            ship as a subsystem (an approval gate, a cache, a redaction pass, a
            spend cap) goes here rather than into a patched loop.
        compaction_strategy: How the conversation is shortened when it
            approaches the context window — see
            :mod:`effgen.memory.compaction`. Accepts a strategy, a class or a
            name (``"summarize_oldest"``, ``"drop_oldest"``,
            ``"keep_first_and_last"``, ``"keep_tool_results"``). None keeps the
            default of summarizing everything but the most recent few.
        tokenizer: Anything with ``count_tokens(text)`` or ``encode(text)``,
            used to measure the history in the units the window is measured in
            rather than in characters divided by four.
        raise_on_error: When True — the default since 1.0.0 — run() raises the
            typed error on failure instead of returning an AgentResponse with
            success=False. The same failure raises regardless of which internal
            path (direct or tool loop) produced it. Set False to inspect
            ``response.success`` and ``response.metadata["reason"]`` yourself,
            which is what the CLI does so it can render a failure as a panel.
            A backend that never answered raises either way: that run produced
            no result to inspect.

            **Batch evaluation wants ``raise_on_error=False``.** Scoring a run
            that hit the iteration cap as an error rather than as a wrong answer
            measures the reporting style instead of the model, and a small model
            hits that cap often. The pairing this flag was designed for is what
            makes that safe: an ordinary failure comes back to be inspected,
            while a backend that never answered still raises, so a broken
            endpoint cannot be silently scored as a wrong answer.

            One consequence to code for: with the flag off, a failed run's
            ``response.output`` is effGen's report of what stopped it, and the
            model's own text is in ``response.metadata["partial_output"]``. Read
            that key rather than ``output`` when a partial answer is what you
            want to score.
        cite_sources: Ask the model for inline ``[1]``, ``[2]`` markers when it
            answers from retrieved passages, and number those passages for it so
            marker ``n`` is ``response.citations[n - 1]``. Off by default,
            because a marker ends up inside ``response.output``.
            ``run(cite_sources=...)`` overrides it for a single call.
        output_schema: A JSON Schema (or a Pydantic model class) every run must
            answer in. It is the one machine-readable statement of answer shape
            the framework has, so it is stated to the model inside the tool loop
            and the answer is validated against it afterwards.
            ``run(output_schema=...)`` / ``run(output_model=...)`` overrides it
            for a single call.

            The framework describes only the machinery it inserted — a block of
            retrieved passages the model did not ask for, say — and leaves the
            form of the answer to whoever knows the question: this schema first,
            then the task and ``system_prompt``. Nothing effGen appends asks for
            a shape of its own, so a task that asks for a letter gets a letter.
    """
    name: str = field(default="", kw_only=True)
    model: BaseModel | str
    tools: list[BaseTool] = field(default_factory=list)
    system_prompt: str = "You are a helpful AI assistant."
    max_iterations: int = 10
    temperature: float = 0.7
    # Default output-token budget for every run(). None leaves the budget to
    # the agent loop, which applies default_max_output_tokens(model) — 1024 for
    # an ordinary model, more for a reasoning family. run(max_tokens=...)
    # overrides it for a single call.
    max_tokens: int | None = None
    # Sampling controls. Pinned here they apply to every run(); a run(...)
    # kwarg of the same name overrides them for a single call. seed and the
    # penalties default to GenerationConfig's neutral values (no effect on
    # generation) so existing agents are unaffected until a caller sets one.
    top_p: float = 0.9
    top_k: int = 50
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    mode: AgentMode = AgentMode.SINGLE
    enable_sub_agents: bool = True
    enable_memory: bool = True
    enable_streaming: bool = False
    max_context_length: int | None = None
    router_config: dict[str, Any] = field(default_factory=dict)
    sub_agent_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] | None = None
    require_model: bool = True
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    middleware: list[Any] = field(default_factory=list)
    compaction_strategy: Any = None
    tokenizer: Any = None
    # True since 1.0.0: a failed run raises rather than returning a
    # plausible-looking string with success=False, which callers that read
    # .output without checking .success never noticed.
    raise_on_error: bool = True
    system_prompt_template: str | None = None
    verbose_tools: bool | None = None
    fallback_chain: dict[str, list] | None = None
    enable_fallback: bool = True
    max_sub_agent_depth: int = 3
    tool_calling_mode: str = "auto"  # "auto", "native", "react", "hybrid"
    output_format: str | None = None  # Global default: "json", "yaml", "csv", or None
    output_schema: dict[str, Any] | None = None  # Global default JSON Schema
    guardrails: Any = None  # GuardrailChain, preset name (str), or None
    memory_config: dict[str, Any] = field(default_factory=lambda: {
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 100,
        "long_term_backend": "sqlite",
        "long_term_persist_path": None,
        "auto_summarize": True,
    })
    # Multi-model support
    models: list[BaseModel | str] | None = None  # Additional models for routing
    speculative_execution: bool = False  # Run on 2 models, return first success
    # Human-in-the-loop
    approval_callback: Callable[[str, str], bool] | None = None
    approval_mode: str = "never"  # "always", "first_time", "never", "dangerous_only"
    approval_timeout: float = 0.0  # seconds; 0 = wait forever
    clarification_callback: Callable[[str, list[str]], int] | None = None
    input_callback: Callable[[str], str] | None = None
    # Prompt caching: keep the system prompt at a fixed position so OpenAI
    # can cache the prefix automatically across sequential calls.
    stable_system_prompt: bool = True
    # Anthropic explicit prompt caching via cache_control markers.
    # cache_system_prompt=True: Agent marks the last block of the system message
    #   with cache_control so it is cached across requests.
    # cache_tools=True: Agent marks the last tool spec with cache_control.
    # These flags have no effect when the model is not an AnthropicAdapter.
    cache_system_prompt: bool = True
    cache_tools: bool = True
    # Ask the model for inline [1], [2] citation markers when it answers from
    # retrieved passages. Off by default: a marker becomes part of ``.output``,
    # which breaks an exact-match comparison, a structured-output schema, and
    # any program that reads the answer. ``response.citations`` and
    # ``response.sources`` are populated either way. When it is on, the passages
    # are presented to the model as a numbered list and marker ``n`` is
    # ``response.citations[n - 1]``. One consequence to know about: while it is
    # off, an answer that ends in a bracketed number after a retrieval tool ran
    # is read as a marker and removed.
    cite_sources: bool = False
    # What the framework tells the model about the tools it attached. ``None``
    # selects the text from the tools' declared categories — a tool that checks
    # the model's work, one that does work the model cannot do, and one that
    # brings back material to answer from are each described for what they are
    # (``effgen.prompts.tool_contract``). Any string is stated verbatim in the
    # same position instead, and ``""`` states nothing at all — which is how to
    # keep the framework's own sentences out of a prompt without rebuilding the
    # whole scaffold through ``system_prompt_template``. A ``system_prompt``
    # persona is unaffected either way: it still leads the prompt.
    tool_contract: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model if isinstance(self.model, str) else "agent"


# Model-loading options belong to the engine (load_model), not the agent. Passing
# one straight to AgentConfig otherwise raises a cryptic dataclass
# "unexpected keyword argument 'engine'"; intercept it with an actionable hint.
_MODEL_LOAD_KWARGS = frozenset({
    "engine", "engine_config", "tensor_parallel_size", "gpu_memory_utilization",
    "apply_chat_template", "quantization", "trust_remote_code",
})

# run()'s recognized **kwargs — generation controls plus the checkpoint/debug
# knobs threaded through the tool loop. A name outside this set (and not
# starting with "_", reserved for internal call-chain bookkeeping such as
# resume()'s _resume_scratchpad) is almost always a typo, so run() rejects it
# instead of silently ignoring it.
_RUN_KWARGS = frozenset({
    "debug", "max_tokens", "temperature", "top_p", "top_k", "seed",
    "presence_penalty", "frequency_penalty", "repetition_penalty",
    "stop_sequences", "reasoning_effort", "tools", "tool_choice",
    "checkpoint_dir", "checkpoint_interval", "max_iterations",
    "middleware", "session", "cite_sources",
})


def _agentconfig_init_guard(_dataclass_init):
    """Wrap AgentConfig.__init__ to translate a stray model-loading kwarg into a
    one-line "here's how to do it" instead of a bare dataclass TypeError."""
    @functools.wraps(_dataclass_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        bad = _MODEL_LOAD_KWARGS.intersection(kwargs)
        if bad:
            opt = sorted(bad)
            raise TypeError(
                f"AgentConfig does not accept model-loading option(s) {opt}: "
                "they configure the engine, not the agent. Either load the model "
                "first — load_model(model_id, engine=\"transformers\") — and pass "
                "the instance as model=, or use "
                "create_agent(preset, model_id, engine=\"transformers\"), which "
                "routes these to load_model for you."
            )
        _dataclass_init(self, *args, **kwargs)
    return __init__


AgentConfig.__init__ = _agentconfig_init_guard(AgentConfig.__init__)
