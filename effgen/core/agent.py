"""
Core Agent implementation for effGen.

This module defines :class:`Agent` — its construction, the state one ``run()``
call is scoped to, and its lifecycle. The work a run does is contributed by
mixins in sibling modules: task-run orchestration, prompt assembly, result
assembly, model generation, the ReAct loop, tool execution, source mining and
streaming. Every public name remains importable from ``effgen.core.agent``.

The agent can:
- Execute tasks using ReAct (Reason + Act)
- Select and execute tools
- Spawn sub-agents through the router
- Manage conversation memory
- Stream responses
- Save and load state
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from ..memory.long_term import (
    JSONStorageBackend,
    LongTermMemory,
    SQLiteStorageBackend,
)
from ..memory.short_term import ShortTermMemory
from ..models.base import BaseModel
from ..models.model_loader import ModelLoader
from ..prompts.agent_system_prompt import AgentSystemPromptBuilder
from ..prompts.tool_prompt_generator import ToolPromptGenerator
from ..tools.base_tool import BaseTool
from ..tools.fallback import ToolFallbackChain
from ..utils.circuit_breaker import CircuitBreaker
from .execution_tracker import ExecutionTracker
from .router import SubAgentRouter
from .state import AgentState
from .sub_agent_manager import SubAgentManager
from .tool_calling import (
    get_strategy,
)

if TYPE_CHECKING:
    from .background import BackgroundTaskRunner

logger = logging.getLogger(__name__)
# Input config and result-assembly types live in leaf modules; re-export them so
# ``from effgen.core.agent import AgentMode, AgentConfig, AgentResponse,
# StreamEvent`` and patches against this module resolve unchanged. These lines
# stay above the generation/react mixin imports below, which do
# ``from .agent import AgentMode, AgentResponse`` and rely on those names already
# being bound on this (partially-initialized) module.
from .agent_config import (  # noqa: E402,F401  re-exported for import/patch parity
    _MODEL_LOAD_KWARGS,
    _RUN_KWARGS,
    AgentConfig,
    AgentMode,
    _agentconfig_init_guard,
)
from .agent_response import (  # noqa: E402,F401
    STOP_REASONS,
    STOPPED_REASONS,
    AgentResponse,
    PartialResult,
    StreamEvent,
)
from .agent_runtime import (  # noqa: E402
    AgentRuntimeMixin,
    sanitize_final_answer,  # noqa: F401  re-exported for import/patch parity
)


@dataclass
class _AgentCallState:
    """One ``Agent.run()`` call's sub-agent depth, citations, and cost/token
    accumulator — held in a context variable (see ``_call_state_var`` below)
    so concurrent or repeated calls that reuse one ``Agent`` instance (e.g.
    ``BatchRunner`` running many rows through a single agent) never read or
    write each other's data.
    """

    current_depth: int = 0
    collected_citations: list[dict[str, Any]] = field(default_factory=list)
    run_cost_accum: dict[str, Any] = field(default_factory=dict)
    # Whether this call asked for inline citation markers. ``None`` means no
    # call resolved it, so the agent's configured value applies (a stream never
    # enters this scope and reads the config directly).
    cite_sources: bool | None = None
    # What the run's context-retrieval observations offered: how many such
    # observations were appended, and how many passages they carried. Together
    # they gate the trailing-marker strip, so a run that never retrieved cannot
    # have a bracketed number taken off its answer.
    retrieval_observations: int = 0
    retrieval_passages: int = 0
    # The JSON Schema this call must answer in, resolved once from
    # ``run(output_schema=)``, ``run(output_model=)`` and the agent's configured
    # default, so the tool loop and the structured-output funnel read one answer
    # and a concurrent call on the same Agent cannot see another run's schema.
    # ``output_schema_resolved`` stays False when no call resolved it (a stream
    # never enters this scope), and the agent's configured schema applies.
    output_schema_resolved: bool = False
    output_schema: dict[str, Any] | None = None


# Per-call state for the three attributes above. Set for the duration of one
# run() call by Agent._agent_call_scope(); falls back to a private
# per-instance default (Agent._fallback_call_state) for any access outside
# an active run (e.g. immediately after construction).
_call_state_var: contextvars.ContextVar[_AgentCallState | None] = contextvars.ContextVar(
    "effgen_agent_call_state", default=None
)

# Overrides Agent.execution_tracker for the duration of one run() call, but
# only when a sibling run is already in flight on the same Agent instance
# (see Agent._agent_call_scope()). Otherwise a run reuses and clears the
# instance's own tracker, so a caller that grabs `agent.execution_tracker`
# before starting a single run (e.g. a live status display) keeps watching
# the object that run actually writes to.
_tracker_override_var: contextvars.ContextVar[ExecutionTracker | None] = contextvars.ContextVar(
    "effgen_agent_tracker_override", default=None
)


from .agent_generation import AgentGenerationMixin  # noqa: E402
from .agent_orchestration import AgentOrchestrationMixin  # noqa: E402
from .agent_prompting import AgentPromptingMixin  # noqa: E402
from .agent_react import AgentReActMixin  # noqa: E402
from .agent_result import AgentResultMixin  # noqa: E402
from .agent_stream_native import AgentNativeStreamMixin  # noqa: E402
from .agent_streaming import AgentStreamingMixin, _chunk_answer_text  # noqa: E402,F401


class _AgentDecompositionClient:
    """Adapts an :class:`Agent` to the ``generate(prompt, **kwargs) -> str``
    interface :class:`~effgen.core.decomposition_engine.DecompositionEngine`
    expects for LLM-assisted task decomposition, by routing through the
    agent's own model-generation path (:meth:`Agent._generate`)."""

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._agent._generate(prompt, **kwargs).get("text", "")


class Agent(
    AgentOrchestrationMixin,
    AgentPromptingMixin,
    AgentResultMixin,
    AgentGenerationMixin,
    AgentReActMixin,
    AgentStreamingMixin,
    AgentNativeStreamMixin,
    AgentRuntimeMixin,
):
    """
    Main Agent implementation with ReAct loop and sub-agent support.

    The agent can:
    - Execute tasks using ReAct (Reason + Act) pattern
    - Intelligently spawn sub-agents for complex tasks
    - Use tools to interact with external systems
    - Manage conversation memory
    - Stream responses
    - Save/load state
    """

    def __init__(self, config: AgentConfig | None = None, session_id: str | None = None) -> None:
        """
        Initialize agent.

        Args:
            config: Agent configuration
            session_id: Optional persistent session id. If provided, the
                agent loads/creates a Session in ~/.effgen/sessions/ and
                appends each run() turn to it.
        """
        # Fail fast on the bare-constructor traps and teach the fix. ``Agent()``
        # (no args) and ``Agent("groq:model")`` (a bare id) would otherwise be a
        # cryptic "missing argument" / a str stored unvalidated that crashes deep
        # in run(). Point users at the real construction paths instead.
        if config is None:
            raise TypeError(
                "Agent() needs an AgentConfig. Build one with "
                "AgentConfig(name=..., model=...), or use the preset helper "
                "create_agent('minimal', 'groq:openai/gpt-oss-20b')."
            )
        if not isinstance(config, AgentConfig):
            raise TypeError(
                "Agent(config=...) expects an AgentConfig, not "
                f"{type(config).__name__}. Build one with "
                "AgentConfig(name=..., model=...), or use the preset helper "
                "create_agent('minimal', 'groq:openai/gpt-oss-20b')."
            )
        self.config = config
        self.name = config.name
        self._closed = False

        # Tools — config.tools must be Tool instances. A bare name string is a
        # natural mistake (tool names are the idiom elsewhere: get_tool_sync(),
        # the CLI's --allowed-tools) but is not accepted here; look it up first.
        # Validated before the model loads so a config type error surfaces
        # without depending on provider credentials.
        for t in config.tools:
            if isinstance(t, str):
                raise TypeError(
                    f"AgentConfig(tools=...) expects Tool instances, not names — "
                    f"got a str {t!r}. Look it up first: "
                    f"get_registry().get_tool_sync({t!r})."
                )
            if not isinstance(t, BaseTool):
                raise TypeError(
                    f"AgentConfig(tools=...) expects Tool instances — got "
                    f"{type(t).__name__} {t!r}."
                )

        # Persistent session
        self._session_id = session_id
        self.session = None
        if session_id:
            from .session import Session as _Session
            self.session = _Session.load_or_create(session_id, agent_name=self.name)

        # Background task runner, loaded lazily.
        self._bg_runner: "BackgroundTaskRunner | None" = None

        # Last checkpoint info
        self._last_checkpoint_id: str | None = None

        # Hooks around the run, each model call and each tool call. Empty
        # for an agent that configured none, which costs one boolean test at
        # each point.
        from .middleware import MiddlewareChain
        self._middleware = MiddlewareChain(getattr(config, "middleware", None))

        # Model initialization
        self.model_loader = ModelLoader()
        if isinstance(config.model, BaseModel):
            # Model instance provided directly
            self.model = config.model
            self.model_name = getattr(config.model, 'model_name', 'custom')
        elif isinstance(config.model, str):
            # Model name provided - load it
            self.model_name = config.model
            load_kwargs: dict[str, Any] = {}
            if config.provider:
                load_kwargs["provider"] = config.provider
            # A base_url names an OpenAI-protocol server, so it both selects
            # the compatible adapter (unless a provider was named explicitly)
            # and tells it where to call.
            if config.base_url:
                load_kwargs["base_url"] = config.base_url
                load_kwargs.setdefault("provider", "openai_compatible")
            if config.api_key:
                load_kwargs["api_key"] = config.api_key
            try:
                logger.debug(f"Loading model: {self.model_name}")
                self.model = self.model_loader.load_model(
                    self.model_name,
                    engine_config=config.model_config,
                    **load_kwargs,
                )
                logger.debug(f"Model loaded successfully: {self.model_name}")
            except Exception as e:
                self.model = None
                if config.require_model:
                    # Fail fast on a typo'd id / missing key rather than
                    # returning a working-looking agent (see AgentConfig docs).
                    # Mark closed first so this never-returned, partially-built
                    # agent doesn't tail the error with a GC-without-close warning.
                    self._closed = True
                    # A typed load error (e.g. an offline cache miss or a
                    # require_gpu policy failure) already carries a clear,
                    # user-facing message and is re-raised below — log it at
                    # debug so it isn't repeated as an ERROR alongside the raise.
                    if type(e).__name__ in ("ModelNotCachedError", "GPUPlacementError"):
                        logger.debug(f"Failed to load model '{self.model_name}': {e}")
                    else:
                        logger.error(f"Failed to load model '{self.model_name}': {e}")
                    raise RuntimeError(
                        f"Failed to load model '{self.model_name}': {e}"
                    ) from e
                logger.warning(
                    f"Model loading failed for '{self.model_name}' and "
                    "require_model=False; the agent will fail on first inference. "
                    "Set require_model=True (the default) to fail fast at construction."
                )
        elif config.model is not None:
            # A non-None value that is neither a model id string nor a loaded
            # BaseModel (e.g. an int, a list, or a Pydantic class). Fail fast
            # with a clear message instead of silently building a model-less
            # agent that only crashes at run().
            self._closed = True
            raise TypeError(
                f"AgentConfig.model must be a model-id str or a loaded model "
                f"instance, not {type(config.model).__name__}. "
                "e.g. model='gpt-5-nano' or model=load_model('Qwen/Qwen2.5-1.5B-Instruct')."
            )
        else:
            # No model provided
            self.model_name = None
            self.model = None
            if config.require_model and config.model is None:
                # Fail at construction (not on the first run()) so a model-less
                # agent doesn't look usable. Set require_model=False to defer
                # model assignment (advanced/sub-agent use).
                self._closed = True
                raise ValueError(
                    f"Agent '{config.name}' was created without a model. "
                    "Pass a model id or instance, e.g. "
                    "AgentConfig(name=..., model='gpt-5-nano') or "
                    "model='Qwen/Qwen2.5-1.5B-Instruct'. "
                    "Run `effgen models list` to see options, or `effgen doctor` "
                    "to check which providers are usable. "
                    "(Set require_model=False to defer loading.)"
                )

        # Multi-model router
        self._model_router = None
        self._all_models: list[BaseModel] = []
        if self.model is not None:
            self._all_models.append(self.model)
        if config.models:
            for m in config.models:
                if isinstance(m, BaseModel):
                    self._all_models.append(m)
                elif isinstance(m, str):
                    try:
                        loaded = self.model_loader.load_model(
                            m, engine_config=config.model_config
                        )
                        self._all_models.append(loaded)
                    except Exception as e:
                        logger.warning("Failed to load additional model '%s': %s", m, e)

        if len(self._all_models) > 1:
            from ..models.router import ModelRouter
            self._model_router = ModelRouter(models=self._all_models)
            logger.info(
                "Model router enabled with %d models: %s",
                len(self._all_models),
                [getattr(m, 'model_name', str(m)) for m in self._all_models],
            )

        self._speculative_execution = config.speculative_execution

        self.tools = {tool.name: tool for tool in config.tools}

        # Validate OpenAI native tool compatibility at init time
        self._validate_native_tool_compatibility(config.tools)

        # Tool calling strategy
        self._tool_calling_strategy = get_strategy(
            mode=config.tool_calling_mode,
            model=self.model,
        )
        logger.info(f"Tool calling strategy: {self._tool_calling_strategy.name}")

        # Tool prompt generator for enhanced ReAct prompts
        self._tool_prompt_generator = ToolPromptGenerator(
            tools=config.tools,
            model_name=self.model_name or "",
        )

        # Determine verbose_tools setting: auto-detect from model size if not set
        if config.verbose_tools is not None:
            self._verbose_tools = config.verbose_tools
        else:
            self._verbose_tools = self._auto_detect_verbose()

        # Tool fallback chain
        self._fallback_chain = ToolFallbackChain(
            custom_chains=config.fallback_chain
        )
        self._enable_fallback = config.enable_fallback

        # Circuit breaker for tool failures
        self._circuit_breaker = CircuitBreaker()

        # Human-in-the-loop approval manager
        from .human_loop import ApprovalManager, ApprovalMode
        try:
            _approval_mode = ApprovalMode(config.approval_mode)
        except ValueError:
            _approval_mode = ApprovalMode.NEVER
        self._approval_manager = ApprovalManager(
            mode=_approval_mode,
            callback=config.approval_callback,
            timeout=config.approval_timeout,
        )

        # Remember whether the user supplied a custom persona via ``system_prompt``.
        # Captured *before* the tool-aware default is auto-built below, so the
        # framework's generated prompt for a plain default tool agent is never
        # mistaken for a user persona. The no-tool direct path and the
        # native/hybrid tool path use this to steer the model with the persona,
        # the same way the ReAct-text and Gemini-native paths already do.
        _user_sp = (config.system_prompt or "").strip()
        self._custom_persona: str | None = (
            _user_sp if _user_sp and _user_sp != "You are a helpful AI assistant." else None
        )

        # Auto-generate system prompt if tools are present and using default prompt
        self._system_prompt_builder = AgentSystemPromptBuilder(
            model_name=self.model_name or "",
        )
        if config.tools and config.system_prompt == "You are a helpful AI assistant.":
            self.config.system_prompt = self._build_system_prompt()

        # State management
        self.state = AgentState(agent_id=self.name)

        # Sub-agent components. Per-call depth/citation/cost state (see
        # _agent_call_scope) lives in a context variable, not on the
        # instance, so it isn't declared here.
        self.router = None
        self.sub_agent_manager = None
        if config.enable_sub_agents:
            self.router = SubAgentRouter(
                config=config.router_config,
                llm_client=_AgentDecompositionClient(self)
            )
            self.sub_agent_manager = SubAgentManager(
                parent_agent=self,
                config=config.sub_agent_config
            )

        # Execution tracker. `execution_tracker` (defined below as a
        # property) reads/clears this instance unless a concurrent sibling
        # run is in flight on this Agent, in which case it gets its own.
        self._default_execution_tracker = ExecutionTracker()
        self._active_run_lock = threading.Lock()
        self._active_run_count = 0

        # Memory system
        mem_cfg = config.memory_config or {}
        stm_max_tokens = mem_cfg.get("short_term_max_tokens", 4096)
        stm_max_messages = mem_cfg.get("short_term_max_messages", 100)
        self.short_term_memory = ShortTermMemory(
            max_tokens=stm_max_tokens,
            max_messages=stm_max_messages,
            summarization_threshold=mem_cfg.get("summarization_threshold", 0.8),
            keep_recent_messages=mem_cfg.get("keep_recent_messages", 4),
            summary_budget_ratio=mem_cfg.get("summary_budget_ratio", 0.4),
            model=self.model,
            compaction_strategy=config.compaction_strategy
            or mem_cfg.get("compaction_strategy"),
            tokenizer=config.tokenizer or mem_cfg.get("tokenizer"),
        )

        # Long-term memory (optional, requires persist path)
        self.long_term_memory: LongTermMemory | None = None
        if config.enable_memory:
            persist_path = mem_cfg.get("long_term_persist_path")
            if persist_path:
                import os
                persist_path = os.path.expanduser(persist_path)
                backend_type = mem_cfg.get("long_term_backend", "sqlite")
                if backend_type == "sqlite":
                    backend = SQLiteStorageBackend(
                        os.path.join(persist_path, "long_term.db")
                    )
                else:
                    backend = JSONStorageBackend(
                        os.path.join(persist_path, "long_term.json")
                    )
                self.long_term_memory = LongTermMemory(backend=backend)
                self.long_term_memory.start_session(name=self.name)

        # Guardrails
        self._guardrail_chain = self._resolve_guardrails(config.guardrails)
        self._warn_tool_output_injection_gap(self._guardrail_chain, bool(config.tools))

        # Hydrate short-term memory from persistent session if loaded
        if self.session and self.session.messages:
            for m in self.session.messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    self.short_term_memory.add_user_message(content)
                elif role == "assistant":
                    self.short_term_memory.add_assistant_message(content)

    def _new_short_term_memory(self) -> ShortTermMemory:
        """Build a short-term memory with this agent's configured limits."""
        mem_cfg = self.config.memory_config or {}
        return ShortTermMemory(
            max_tokens=mem_cfg.get("short_term_max_tokens", 4096),
            max_messages=mem_cfg.get("short_term_max_messages", 100),
            summarization_threshold=mem_cfg.get("summarization_threshold", 0.8),
            keep_recent_messages=mem_cfg.get("keep_recent_messages", 4),
            summary_budget_ratio=mem_cfg.get("summary_budget_ratio", 0.4),
            model=self.model,
            compaction_strategy=self.config.compaction_strategy
            or mem_cfg.get("compaction_strategy"),
            tokenizer=self.config.tokenizer or mem_cfg.get("tokenizer"),
        )

    def _hydrate_memory_from(self, session: Any, memory: ShortTermMemory) -> None:
        """Load *session*'s turns into *memory*, oldest first."""
        for message in getattr(session, "messages", None) or []:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                memory.add_user_message(content)
            elif role == "assistant":
                memory.add_assistant_message(content)

    def _enter_run_session(self, session: Any) -> tuple[Any, Any]:
        """Point this run at *session*, and return what to restore afterwards.

        The conversation the prompt is built from comes from the session rather
        than from whatever this agent answered last, so one agent can serve many
        independent conversations without their histories mixing.

        Args:
            session: A :class:`~effgen.core.session.Session`, or a session id to
                load or create.

        Returns:
            The agent's own session and short-term memory, to put back.
        """
        from .session import Session as _Session

        if isinstance(session, str):
            session = _Session.load_or_create(session, agent_name=self.name)

        previous = (self.session, self.short_term_memory)
        self.session = session
        self.short_term_memory = self._new_short_term_memory()
        self._hydrate_memory_from(session, self.short_term_memory)
        return previous

    def _exit_run_session(self, previous: tuple[Any, Any]) -> None:
        """Put back the session and memory :meth:`_enter_run_session` replaced."""
        self.session, self.short_term_memory = previous

    def _get_call_state(self) -> _AgentCallState:
        """Return the active per-call state, or a private per-instance
        fallback for access outside an active run() (e.g. right after
        construction)."""
        state = _call_state_var.get()
        if state is not None:
            return state
        fallback = getattr(self, "_fallback_call_state", None)
        if fallback is None:
            fallback = _AgentCallState()
            self._fallback_call_state = fallback
        return fallback

    @property
    def _current_depth(self) -> int:
        return self._get_call_state().current_depth

    @_current_depth.setter
    def _current_depth(self, value: int) -> None:
        self._get_call_state().current_depth = value

    @property
    def _collected_citations(self) -> list[dict[str, Any]]:
        return self._get_call_state().collected_citations

    @_collected_citations.setter
    def _collected_citations(self, value: list[dict[str, Any]]) -> None:
        self._get_call_state().collected_citations = value

    @property
    def _run_cost_accum(self) -> dict[str, Any]:
        return self._get_call_state().run_cost_accum

    @_run_cost_accum.setter
    def _run_cost_accum(self, value: dict[str, Any]) -> None:
        self._get_call_state().run_cost_accum = value

    @property
    def _retrieval_observations(self) -> int:
        return self._get_call_state().retrieval_observations

    @_retrieval_observations.setter
    def _retrieval_observations(self, value: int) -> None:
        self._get_call_state().retrieval_observations = value

    @property
    def _retrieval_passages(self) -> int:
        return self._get_call_state().retrieval_passages

    @_retrieval_passages.setter
    def _retrieval_passages(self, value: int) -> None:
        self._get_call_state().retrieval_passages = value

    def _resolve_cite_sources(self, requested: bool | None) -> bool:
        """Fix this call's citation-marker setting and record it on the run state.

        ``requested`` is the ``run(cite_sources=...)`` keyword, or ``None`` when
        the call said nothing and the agent's configured value applies.
        """
        value = bool(
            getattr(self.config, "cite_sources", False) if requested is None
            else requested
        )
        self._get_call_state().cite_sources = value
        return value

    def _set_effective_output_schema(self, schema: dict[str, Any] | None) -> None:
        """Record the schema this call must answer in on the run state."""
        state = self._get_call_state()
        state.output_schema = schema
        state.output_schema_resolved = True

    def _effective_output_schema(self) -> dict[str, Any] | None:
        """The JSON Schema this run must answer in, or ``None``.

        A ``run(output_schema=...)`` / ``run(output_model=...)`` keyword decides
        the call; without one the agent's configured schema applies, which is
        what a stream reads (it does not open a per-call scope).
        """
        state = self._get_call_state()
        if state.output_schema_resolved:
            return state.output_schema
        from .structured_output import normalize_output_schema
        return normalize_output_schema(getattr(self.config, "output_schema", None))

    def _cite_sources_requested(self) -> bool:
        """True when this run should ask the model for inline citation markers.

        A ``run(cite_sources=...)`` keyword decides the call; without one the
        agent's configured value applies, which is what a stream reads (it does
        not open a per-call scope).
        """
        override = self._get_call_state().cite_sources
        if override is None:
            return bool(getattr(self.config, "cite_sources", False))
        return bool(override)

    @property
    def execution_tracker(self) -> ExecutionTracker:
        """The tracker recording this agent's runs (per-call override aware)."""
        override = _tracker_override_var.get()
        return override if override is not None else self._default_execution_tracker

    @execution_tracker.setter
    def execution_tracker(self, value: ExecutionTracker) -> None:
        """Replace the tracker (the per-call override when one is active)."""
        if _tracker_override_var.get() is not None:
            _tracker_override_var.set(value)
        else:
            self._default_execution_tracker = value

    @contextlib.contextmanager
    def _agent_call_scope(self) -> Iterator[None]:
        """Give one run() call its own depth/citations/cost-accumulator state
        and, when a sibling run is already in flight on this Agent, its own
        execution tracker.

        A single active run reuses and clears the instance's own tracker, so
        a caller that grabbed ``agent.execution_tracker`` before starting it
        (e.g. a live status display) keeps watching the object the run
        writes to. A run that starts while another is still in flight on the
        same Agent gets an isolated tracker instead, so the two never
        interleave events, sub-agent depth, citations, or cost/token totals.
        """
        state_token = _call_state_var.set(_AgentCallState())
        with self._active_run_lock:
            concurrent = self._active_run_count > 0
            self._active_run_count += 1
        tracker_token = None
        if concurrent:
            tracker_token = _tracker_override_var.set(ExecutionTracker())
        else:
            self._default_execution_tracker.clear()
        try:
            yield
        finally:
            with self._active_run_lock:
                self._active_run_count -= 1
            if tracker_token is not None:
                _tracker_override_var.reset(tracker_token)
            _call_state_var.reset(state_token)

    def _validate_native_tool_compatibility(self, tools: list) -> None:
        """Raise ToolIncompatibleError for provider-specific tools used with the wrong model."""
        from ..models.errors import ToolIncompatibleError

        # --- OpenAI native tools ---
        try:
            from ..models.openai_adapter import OpenAIAdapter
            from ..tools.builtin.openai_native import OpenAINativeTool

            openai_native = [t for t in tools if isinstance(t, OpenAINativeTool)]
            if openai_native:
                is_openai = isinstance(self.model, OpenAIAdapter)
                if not is_openai and not isinstance(self.model, BaseModel):
                    # Only guess from the model name when the object is not a
                    # known effGen adapter (e.g. a duck-typed custom model). A
                    # concrete non-OpenAI adapter is authoritative: do NOT match
                    # on the "gpt-" prefix, or Cerebras' ``gpt-oss-*`` (and other
                    # OpenAI-compatible providers) would be mistaken for OpenAI
                    # and silently skip this fail-closed guard.
                    model_name_str = getattr(self.model, "model_name", "") or ""
                    provider = getattr(self.model, "_provider", "") or ""
                    is_openai = (
                        "openai" in provider.lower()
                        or model_name_str.startswith(("gpt-", "o1", "o3", "o4"))
                    )
                if not is_openai:
                    bad = openai_native[0]
                    current_model = getattr(self.model, "model_name", str(self.model)) if self.model else "None"
                    raise ToolIncompatibleError(
                        tool_name=bad.name,
                        model_name=current_model,
                        reason=(
                            "OpenAI native tools (web_search, code_interpreter, file_search) "
                            "are executed server-side by OpenAI and require an OpenAIAdapter. "
                            f"Current model: '{current_model}'. "
                            "Switch to an OpenAI model or remove the native tool."
                        ),
                    )
        except ImportError:
            pass

        # --- Gemini native tools ---
        try:
            from ..models.gemini_adapter import GeminiAdapter
            from ..tools.builtin.gemini_native import GeminiNativeTool

            gemini_native = [t for t in tools if isinstance(t, GeminiNativeTool)]
            if gemini_native:
                is_gemini = isinstance(self.model, GeminiAdapter)
                if not is_gemini and not isinstance(self.model, BaseModel):
                    # Name guess only for unknown duck-typed objects; a concrete
                    # non-Gemini adapter is authoritative (see OpenAI note above).
                    model_name_str = getattr(self.model, "model_name", "") or ""
                    is_gemini = model_name_str.startswith("gemini")
                if not is_gemini:
                    bad = gemini_native[0]
                    current_model = getattr(self.model, "model_name", str(self.model)) if self.model else "None"
                    raise ToolIncompatibleError(
                        tool_name=bad.name,
                        model_name=current_model,
                        reason=(
                            "Gemini native tools (google_search, url_context, code_execution) "
                            "are executed server-side by Google and require a GeminiAdapter. "
                            f"Current model: '{current_model}'. "
                            "Switch to a Gemini model or remove the native tool."
                        ),
                    )
        except ImportError:
            pass

        # --- Anthropic native tools (computer-use) ---
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..tools.builtin.anthropic_native import AnthropicNativeTool

            anthropic_native = [t for t in tools if isinstance(t, AnthropicNativeTool)]
            if anthropic_native:
                is_anthropic = isinstance(self.model, AnthropicAdapter)
                if not is_anthropic and not isinstance(self.model, BaseModel):
                    # Name guess only for unknown duck-typed objects; a concrete
                    # non-Anthropic adapter is authoritative (see OpenAI note above).
                    model_name_str = getattr(self.model, "model_name", "") or ""
                    is_anthropic = model_name_str.startswith("claude")
                if not is_anthropic:
                    bad = anthropic_native[0]
                    current_model = getattr(self.model, "model_name", str(self.model)) if self.model else "None"
                    raise ToolIncompatibleError(
                        tool_name=bad.name,
                        model_name=current_model,
                        reason=(
                            "Anthropic native computer-use tools (bash, text_editor, computer) "
                            "are executed server-side by Anthropic and require an AnthropicAdapter. "
                            f"Current model: '{current_model}'. "
                            "Switch to an Anthropic model or remove the native tool."
                        ),
                    )
        except ImportError:
            pass


    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool: Tool instance to add
        """
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the agent.

        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self.tools:
            del self.tools[tool_name]

    def reset_memory(self) -> None:
        """Clear conversation and tool history."""
        self.state.clear_history()
        self.short_term_memory.clear()
        if self.long_term_memory:
            self.long_term_memory.end_session()
            self.long_term_memory.start_session(name=self.name)

    def save_state(self, filepath: str, format: str = "json") -> None:
        """
        Save agent state.

        Args:
            filepath: Path to save to
            format: Serialization format. Only ``"json"`` is supported.
        """
        self.state.save(filepath, format)

    def load_state(self, filepath: str, format: str = "json") -> None:
        """
        Load agent state.

        Args:
            filepath: Path to load from
            format: Serialization format. Only ``"json"`` is supported.
        """
        self.state = AgentState.load(filepath, format)

    # ── Resource management ─────────────────────────────────────────────

    def close(self) -> None:
        """
        Release resources held by the agent.

        Closes SQLite connections (long-term memory), resets circuit
        breakers, and clears memory references.  Safe to call multiple
        times.
        """
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self._circuit_breaker.reset_all()
        # Stop background worker threads so they never outlive the agent.
        if getattr(self, '_bg_runner', None) is not None:
            try:
                self._bg_runner.shutdown(wait=False)
            except Exception:
                logger.debug("Failed to shut down background runner", exc_info=True)
        if self.long_term_memory is not None:
            try:
                self.long_term_memory.close()
            except Exception:
                logger.debug("Failed to close long-term memory", exc_info=True)
        logger.debug(f"Agent '{self.name}' closed")

    def __enter__(self) -> Agent:
        """Sync context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Sync context manager exit — clean up resources."""
        self.close()
        return False

    async def __aenter__(self) -> Agent:
        """Async context manager entry."""
        return self

    async def aclose(self) -> None:
        """Async-friendly alias for :meth:`close`.

        Cleanup is synchronous (it closes SQLite handles and stops worker
        threads), so this simply awaits nothing and calls :meth:`close`. It
        exists so ``await agent.aclose()`` works symmetrically with
        ``await agent.run_async(...)`` and inside ``async with agent:``.
        """
        self.close()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Async context manager exit — clean up resources."""
        await self.aclose()
        return False

    def __del__(self) -> None:
        """Warn if agent was garbage-collected without close()."""
        if getattr(self, '_closed', True):
            return
        # During interpreter shutdown module globals (including ``logger``) may
        # already be torn down to None — guard so __del__ never raises.
        if logger is None:
            return
        try:
            logger.warning(
                f"Agent '{getattr(self, 'name', '?')}' was garbage-collected "
                "without calling close(). Use 'with Agent(config) as agent:' "
                "or call agent.close() explicitly."
            )
        except Exception:
            pass

    def get_execution_summary(self) -> dict[str, Any]:
        """
        Get summary of execution.

        Returns:
            Summary dictionary
        """
        return self.execution_tracker.get_summary()

    def __repr__(self) -> str:
        """String representation."""
        return f"Agent(name={self.name}, tools={len(self.tools)}, sub_agents={self.config.enable_sub_agents})"
