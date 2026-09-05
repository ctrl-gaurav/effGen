"""Task-run orchestration for :class:`effgen.core.agent.Agent`.

Holds the synchronous :meth:`Agent.run` funnel — mode selection, the input and
output guardrail checks, structured-output application, metrics, tracing,
conversation and session persistence, and the final checkpoint — together with
the async, batch, resume, background-task and sub-agent-synthesis entry points
that wrap it. Mixed into :class:`Agent` alongside the generation, ReAct,
streaming and runtime mixins. The dataclasses come from the config and response
leaves and the sanitizer from ``agent_runtime``; this module imports nothing
from ``agent.py``.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..memory.long_term import ImportanceLevel, MemoryType
from ..models.errors import classify_provider_error
from ..observability import get_logger as _get_obs_logger
from ..observability.spans import AgentAttrs
from ..observability.tracing import (
    mark_run_error,
    set_span_attribute,
    set_span_error,
    start_agent_run,
)
from ..utils.prometheus_metrics import metrics as prom_metrics
from ..utils.structured_logging import (
    LogRunContext,
    generate_run_id,
    get_structured_logger,
)
from .agent_config import _RUN_KWARGS, AgentMode
from .agent_response import AgentResponse
from .agent_runtime import _strip_run_citation_markers, sanitize_final_answer
from .execution_tracker import EventType, ExecutionEvent

if TYPE_CHECKING:
    from .messages import Message

# A run's log records carry the agent module's logger name, so a filter or a
# structured-log consumer watching ``effgen.core.agent`` reads one stream for
# the whole run.
logger = logging.getLogger("effgen.core.agent")


def _never_reached_the_backend(response: Any) -> bool:
    """True when the run failed because nothing answered at the endpoint.

    Reads the classified category the generation path already attached, so the
    decision matches how the rest of effGen labels the same failure rather than
    re-matching the provider's wording here.
    """
    metadata = getattr(response, "metadata", None) or {}
    detail = metadata.get("error")
    return isinstance(detail, dict) and detail.get("category") == "unreachable"


_slog = get_structured_logger("effgen.core.agent")
_obs_log = _get_obs_logger("effgen.core.agent")


class AgentOrchestrationMixin:
    """The run entry points and the work that surrounds a single task run."""

    if TYPE_CHECKING:
        # Contributed by :class:`~effgen.core.agent.Agent`, which owns the
        # per-call state. Declared for the type checker only — at run time it
        # arrives through the MRO, and this statement does not execute.
        def _set_effective_output_schema(
            self, schema: dict[str, Any] | None
        ) -> None: ...

    def run(self,
            task: "str | Message | list[Any]",
            mode: AgentMode | None = None,
            context: dict[str, Any] | None = None,
            output_schema: dict[str, Any] | None = None,
            output_model: Any = None,
            inputs: list[Any] | None = None,
            **kwargs: Any) -> AgentResponse:
        """
        Execute a task.

        Args:
            task: Task description. Accepts a plain ``str``, a multimodal
                ``Message``, or a ``list[ContentPart]``; text is extracted and
                any image/audio/video parts are routed through the multimodal
                path.
            mode: Execution mode (single, sub_agents, auto). Defaults to
                ``self.config.mode`` (``AgentMode.SINGLE`` unless the config
                sets otherwise) when omitted — pass ``mode=AgentMode.AUTO``
                to let the router decide based on task complexity for this
                call only.
            context: Optional context
            output_schema: A JSON-Schema ``dict`` **or** a Pydantic
                ``BaseModel`` subclass — when provided, the final output is
                guaranteed to be valid JSON matching this schema. A model class
                is converted automatically; any other type raises ``TypeError``.
            output_model: Pydantic BaseModel class — when provided, output is
                validated and the parsed instance is stored in
                ``response.metadata["parsed"]``.

                Reasoning models and deeply nested schemas need a generous output
                budget: the model spends tokens on internal reasoning and on the
                JSON structure before it fills any values. Set ``max_tokens``
                accordingly (``run(..., max_tokens=8192)`` for a reasoning model).
                When an extraction validates but every field is empty,
                ``response.metadata["structured_output_empty"]`` is set to True.
            inputs: Optional list of multimodal content parts created by
                ``image_from``, ``audio_from``, or ``video_from``. When present,
                the agent sends a structured Message directly to the model.
            **kwargs: Additional arguments. Two shape the run itself:

                ``session`` — a :class:`~effgen.core.session.Session` or a
                session id. The run builds its prompt from that conversation's
                history and appends this turn to it, so one agent can serve many
                independent conversations rather than needing one agent object
                per conversation. Without it the agent's own memory applies, as
                before.

                ``middleware`` — hooks for this call only, appended to any on
                the config. See :mod:`effgen.core.middleware`.

                ``debug=True`` attaches a DebugTrace.

                ``cite_sources`` — ask the model for inline ``[1]``, ``[2]``
                markers when it answers from retrieved passages, overriding
                ``AgentConfig.cite_sources`` for this call. Off by default: a
                marker becomes part of ``response.output``. With it on, the
                passages are shown to the model as a numbered list and marker
                ``n`` is ``response.citations[n - 1]``.
                ``response.citations`` and ``response.sources`` are populated
                either way.

        Returns:
            AgentResponse with results
        """
        if mode is None:
            mode = self.config.mode
        unrecognized = {k for k in kwargs if k not in _RUN_KWARGS and not k.startswith("_")}
        if unrecognized:
            bad = sorted(unrecognized)[0]
            close = difflib.get_close_matches(bad, sorted(_RUN_KWARGS), n=1, cutoff=0.5)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise TypeError(
                f"run() got an unexpected keyword argument '{bad}'.{hint} "
                f"Recognized run() kwargs: {sorted(_RUN_KWARGS)}."
            )

        start_time = time.time()
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        context = context or {}

        # Accept str | Message | list[ContentPart]; route media to the
        # multimodal `inputs` path. Raises a clear TypeError otherwise.
        task, inputs = self._coerce_task_input(task, inputs)
        if inputs is not None:
            kwargs["inputs"] = inputs

        # Fail closed on an empty/whitespace-only task instead of sending it to
        # the model — a blank prompt was never a real request and still bills
        # the provider for a call the caller almost certainly didn't intend.
        if isinstance(task, str) and not task.strip() and not inputs:
            return self._stamp_run_identity(AgentResponse(
                output="",
                success=False,
                execution_time=time.time() - start_time,
                metadata={
                    "reason": "empty_task",
                    "error": {
                        "type": "InvalidRequestError",
                        "category": "invalid_input",
                        "provider": None,
                        "model": None,
                        "message": (
                            "task is empty — provide a non-empty prompt, or pass "
                            "inputs=[image_from(...)] for a caption-free "
                            "multimodal request."
                        ),
                        "retryable": False,
                    },
                },
            ), task=task, started_at=started_at)

        # Hooks around this run. A per-call middleware= list is appended to
        # the configured ones for this call only, and is what _execute_tool and
        # _generate read while the run is in flight.
        _run_middleware = kwargs.pop("middleware", None)
        _chain = getattr(self, "_middleware", None)
        if _chain is not None:
            _chain = _chain.extended_with(_run_middleware)
        _mw_run_ctx = None
        if _chain:
            from .middleware import RunContext

            _mw_run_ctx = RunContext(
                task=task if isinstance(task, str) else str(task),
                agent_name=self.name,
                mode=mode,
            )
            _short_circuit = _chain.before_run(_mw_run_ctx)
            if _short_circuit is not None:
                # A middleware answered without the model — a cache hit, or a
                # refusal. after_run still sees it, so the chain stays symmetric.
                return self._stamp_run_identity(
                    _chain.after_run(_mw_run_ctx, _short_circuit),
                    task=task, started_at=started_at,
                )
            if isinstance(task, str):
                task = _mw_run_ctx.task
        self._active_middleware = _chain
        self._active_run_context = _mw_run_ctx

        # A conversation handle for this call only. The run reads its history
        # in and writes the turn back, so one agent serves many conversations
        # without an agent object per conversation.
        _run_session = kwargs.pop("session", None)
        _previous_session = (
            self._enter_run_session(_run_session) if _run_session is not None else None
        )

        debug = kwargs.pop("debug", False)
        # Inline citation markers are requested, never assumed. Popped here so
        # it never reaches the model layer as a generation kwarg, and resolved
        # once inside the call scope below so both loops and the answer funnel
        # read the same answer.
        _cite_requested = kwargs.pop("cite_sources", None)
        # A max_tokens set on the config is the default output budget for every
        # run(); an explicit run(max_tokens=...) still overrides it per call.
        if "max_tokens" not in kwargs and self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        run_id = generate_run_id()
        # Capture checkpoint args here so the outer run() can use
        # them for the final-checkpoint write even after _run_single_agent
        # consumes them from kwargs.
        _outer_ckpt_dir = kwargs.get("checkpoint_dir") or context.get("checkpoint_dir")

        # Metrics: track request
        labels = {"agent_name": self.name}
        prom_metrics.total_requests.inc(labels=labels)
        prom_metrics.active_agents.inc(labels=labels)

        # Pre-run input guardrail check
        input_redaction: dict[str, Any] | None = None
        if self._guardrail_chain is not None:
            from ..guardrails.base import GuardrailPosition
            gr = self._guardrail_chain.check(task, position=GuardrailPosition.INPUT)
            if not gr.passed:
                prom_metrics.active_agents.dec(labels=labels)
                return self._stamp_run_identity(AgentResponse(
                    output=f"Input blocked by guardrail: {gr.reason}",
                    success=False,
                    execution_time=time.time() - start_time,
                    metadata={"guardrail_blocked": True, "guardrail_reason": gr.reason},
                ), task=task, started_at=started_at)
            if gr.modified_content is not None:
                task = gr.modified_content
                # Record what the input redaction removed so a run is auditable
                # from its response (e.g. a note de-identified before a cloud call).
                pii_types = gr.metadata.get("pii_types")
                if pii_types:
                    input_redaction = {"types": pii_types}
                    if gr.metadata.get("pii_counts"):
                        input_redaction["counts"] = gr.metadata["pii_counts"]

        # Resolve structured output schema. Accept either a JSON-Schema dict or
        # a Pydantic model class for `output_schema` (and the config default),
        # converting the class instead of letting it reach JSON serialization.
        from .structured_output import (
            is_pydantic_model_class,
            normalize_output_schema,
            pydantic_model_to_schema,
        )
        raw_schema = output_schema if output_schema is not None else self.config.output_schema
        effective_schema = normalize_output_schema(raw_schema)
        if output_model is not None and effective_schema is None:
            effective_schema = pydantic_model_to_schema(output_model)
        # If output_schema is itself a Pydantic class, treat it as the output_model
        # too so metadata["parsed"] is a typed instance — the class is right here,
        # and output_schema=Model / output_model=Model should behave the same.
        if output_model is None and is_pydantic_model_class(raw_schema):
            output_model = raw_schema

        self._warn_reasoning_budget(kwargs.get("max_tokens"), effective_schema is not None)

        # Wrap entire run in tracing span + structured log context, and give
        # this call its own depth/citations/cost/trace state so it can't
        # collide with a concurrent or prior call on this Agent instance.
        _task_preview = self._extract_task_preview(task, 200)
        with self._agent_call_scope(), \
             start_agent_run(preset=self.name, task=task, run_id=run_id) as _span, \
             LogRunContext(run_id=run_id, agent_name=self.name):
            # Track task start
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_START,
                agent_id=self.name,
                message=f"Starting task: {self._extract_task_preview(task, 100)}...",
                data={"task": task, "mode": mode.value}
            ))
            _slog.agent_event(self.name, "task_start", task=_task_preview, mode=mode.value, run_id=run_id)
            _obs_log.agent_event("run.started", agent=self.name, task=_task_preview, mode=mode.value, run_id=run_id)

            _cite_sources = self._resolve_cite_sources(_cite_requested)
            # Fix the run's answer shape on the call state so the tool loop can
            # state it to the model while the answer is being written, instead
            # of the schema being discovered only after one has been paid for.
            self._set_effective_output_schema(effective_schema)
            try:
                # Pass debug flag through kwargs
                if debug:
                    kwargs["_debug"] = True
                    kwargs["_run_id"] = run_id

                # Determine execution mode.
                #
                # routing_context carries this agent's own system_prompt down
                # into DecompositionEngine._llm_decompose (via
                # SubAgentRouter.route -> decompose), so subtask descriptions
                # generated for AUTO/SUB_AGENTS mode are written in the
                # language/register the root agent was configured for,
                # instead of the decomposition templates' hardcoded English.
                # Only added when a system_prompt is actually set, so a
                # caller inspecting `context` downstream sees no new key on
                # the common path where there is nothing to add.
                routing_context = context
                if self.config.system_prompt:
                    routing_context = {**(context or {}), "system_prompt": self.config.system_prompt}

                if mode == AgentMode.AUTO and self.config.enable_sub_agents:
                    # Use router to decide
                    routing_decision = self.router.route(task, routing_context)

                    if routing_decision.use_sub_agents:
                        response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
                    else:
                        response = self._run_single_agent(task, context, **kwargs)
                elif mode == AgentMode.SUB_AGENTS and self.config.enable_sub_agents:
                    # Force sub-agent mode
                    routing_decision = self.router.route(task, routing_context)
                    response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
                else:
                    # Single agent mode
                    response = self._run_single_agent(task, context, **kwargs)

                # Strip any internal scaffolding from the answer before it is
                # used downstream (structured output, guardrails, memory). This
                # is the single funnel covering every execution path; individual
                # assembly sites also sanitize so direct callers stay clean.
                if response.success and isinstance(response.output, str):
                    response.output = sanitize_final_answer(response.output)
                    # Nobody asked for inline citation markers, so the answer
                    # should not carry any. The instruction no longer requests
                    # them; this removes a trailing run a model produced anyway,
                    # and only for a run that actually retrieved context and an
                    # index one of its observations could have offered.
                    if not _cite_sources and self._retrieval_observations:
                        _passages = self._retrieval_passages
                        _stripped, _n_markers = _strip_run_citation_markers(
                            response.output, passages=_passages,
                        )
                        if _n_markers:
                            response.output = _stripped
                            response.metadata["citation_markers_stripped"] = _n_markers
                            logger.info(
                                "citation markers stripped from the answer: "
                                "n=%d passages=%d", _n_markers, _passages,
                            )
                    # If this run used a retrieval tool and the model echoed the
                    # raw result dict as its answer (small models sometimes paste
                    # the tool observation), render it as readable passage text.
                    if self._collected_citations:
                        response.output = self._humanize_observation(response.output)

                # Apply structured output constraint if requested
                if effective_schema and response.success and response.output:
                    response = self._apply_structured_output(
                        response, effective_schema, output_model, task,
                    )

                # Post-run output guardrail check. system_prompt= is only
                # consumed by SystemPromptLeakGuardrail (a no-op without it);
                # every other output guardrail ignores the extra kwarg.
                if self._guardrail_chain is not None and response.success and response.output:
                    from ..guardrails.base import GuardrailPosition as _GP
                    gr = self._guardrail_chain.check(
                        response.output, position=_GP.OUTPUT,
                        system_prompt=self.config.system_prompt,
                    )
                    if not gr.passed:
                        response.output = f"Output blocked by guardrail: {gr.reason}"
                        response.success = False
                        response.metadata["guardrail_blocked"] = True
                        response.metadata["guardrail_reason"] = gr.reason
                    elif gr.modified_content is not None:
                        response.output = gr.modified_content

                # Add execution metadata
                response.execution_time = time.time() - start_time
                response.execution_trace = self.execution_tracker.get_trace()
                response.metadata["run_id"] = run_id
                if input_redaction is not None:
                    response.metadata["input_redaction"] = input_redaction
                # Surface this run's cost + token usage on the result so callers
                # can budget per call without a side channel.
                self._finalize_cost_metadata(response)

                # Surface retrieved evidence: if the run consulted a knowledge
                # base / search tool, expose its passages as sources + inline
                # citations on the response (fills only what's still empty).
                self._attach_citations(response)

                # Track completion
                self.execution_tracker.track_event(ExecutionEvent(
                    type=EventType.TASK_COMPLETE,
                    agent_id=self.name,
                    message=f"Task completed in {response.execution_time:.2f}s",
                    data={
                        "execution_time": response.execution_time,
                        "tokens_used": response.tokens_used,
                        "tool_calls": response.tool_calls
                    }
                ))

                # Build the execution tree after completion is tracked so its
                # root carries a terminal status and a real duration (not a
                # still-"running" snapshot measured at serialize time).
                response.execution_tree = self.execution_tracker.generate_execution_tree()

                # Metrics: record latency and tokens
                prom_metrics.response_latency.observe(response.execution_time, labels=labels)
                if response.tokens_used:
                    prom_metrics.token_usage.observe(response.tokens_used, labels=labels)
                    prom_metrics.tokens_used.inc(response.tokens_used, labels=labels)
                # A failed response with raise_on_error set is about to be turned
                # into a raised exception below and recorded once, with a precise
                # classify_provider_error() outcome, in the except block — recording
                # it here too would double-count the same request.
                if response.success or not self.config.raise_on_error:
                    self._record_provider_metrics(
                        execution_time=response.execution_time,
                        outcome="ok" if response.success else "error",
                        prompt_tokens=response.metadata.get("prompt_tokens"),
                        completion_tokens=response.metadata.get("completion_tokens"),
                    )

                # A run that reports failure without raising still records the
                # failure on its span, so a consumer reading spans sees the same
                # outcome the response carries.
                if not response.success:
                    detail = response.metadata.get("error")
                    if isinstance(detail, dict):
                        message = (
                            f"{detail.get('type', 'AgentError')}: "
                            f"{detail.get('message', response.output)}"
                        )
                    else:
                        message = str(detail or response.output)
                    mark_run_error(message)

                # Tracing span attributes (using span constants)
                set_span_attribute(AgentAttrs.RUN_ID, run_id or "")
                set_span_attribute("effgen.tokens_used", response.tokens_used)
                set_span_attribute("effgen.tool_calls", response.tool_calls)
                set_span_attribute("effgen.success", response.success)
                set_span_attribute("effgen.latency", response.execution_time)

                _slog.agent_event(
                    self.name, "task_complete",
                    latency=response.execution_time,
                    tokens=response.tokens_used,
                    tool_calls=response.tool_calls,
                    success=response.success,
                )
                _obs_log.agent_event(
                    "run.completed",
                    agent=self.name,
                    run_id=run_id,
                    latency_ms=round(response.execution_time * 1000, 1),
                    tokens=response.tokens_used,
                    tool_calls=response.tool_calls,
                    success=response.success,
                )
                self._record_dashboard_run(response, task=task)

                # Store conversation in short-term memory for context retention
                if response.success and response.output:
                    self.short_term_memory.add_user_message(task)
                    self.short_term_memory.add_assistant_message(response.output)
                    logger.debug(
                        f"Stored conversation turn in memory "
                        f"(total: {self.short_term_memory.total_messages_added} messages)"
                    )

                    # Persist important facts to long-term memory if available
                    if self.long_term_memory and response.tool_calls > 0:
                        self.long_term_memory.add_memory(
                            content=f"Q: {task}\nA: {response.output}",
                            memory_type=MemoryType.CONVERSATION,
                            importance=ImportanceLevel.MEDIUM,
                            tags=["conversation"],
                        )

                    # Persist to session, stamping each turn with the model,
                    # token counts, cost and latency it was answered with so a
                    # stored conversation can be reviewed turn by turn.
                    if self.session is not None:
                        turn_meta = self._session_turn_metadata(response, run_id=run_id)
                        self.session.add_message("user", task, **turn_meta)
                        self.session.add_message("assistant", response.output, **turn_meta)
                        if turn_meta.get("model"):
                            self.session.metadata["model"] = turn_meta["model"]
                        self.session.metadata.setdefault("agent_name", self.name)
                        try:
                            self.session.save()
                        except Exception as _e:
                            logger.warning("Failed to save session: %s", _e)

                # Final checkpoint
                ckpt_dir = _outer_ckpt_dir
                if ckpt_dir:
                    try:
                        from .checkpoint import CheckpointManager
                        mgr = CheckpointManager(ckpt_dir)
                        cp = CheckpointManager.snapshot_agent(
                            self,
                            task=task,
                            iteration=getattr(response, "iterations", 0),
                            scratchpad="",
                            # A stopped run's ``output`` states what stopped
                            # it; resuming from that prose would re-seed the
                            # run with a report about itself, so the progress
                            # it had reached is checkpointed instead.
                            partial_output=(
                                response.partial.text
                                if getattr(response, "partial", None)
                                else response.output
                            ),
                            tool_calls=response.tool_calls,
                            tokens_used=response.tokens_used,
                            metadata={"final": True, "success": response.success},
                        )
                        self._last_checkpoint_id = mgr.save(cp)
                        response.metadata["checkpoint_id"] = self._last_checkpoint_id
                    except Exception as _e:
                        logger.warning("Failed to save final checkpoint: %s", _e)

                if _chain:
                    response = _chain.after_run(_mw_run_ctx, response)

                # A backend that never answered produced no result to report, so
                # it raises whatever raise_on_error says. Reporting it as a
                # failed run lets a dead server read like a model that could not
                # solve the task, which is how a whole batch completes against
                # nothing and looks healthy in the summary.
                if not response.success and _never_reached_the_backend(response):
                    raise self._reconstruct_error(response.metadata, response)

                # raise_on_error contract: surface a typed error on any failure
                # instead of returning success=False (same on both run paths).
                if not response.success and self.config.raise_on_error:
                    raise self._reconstruct_error(response.metadata, response)

                return self._stamp_run_identity(
                    response, task=task, started_at=started_at
                )

            except Exception as e:
                # When raise_on_error is set, propagate the typed error rather
                # than swallowing it into a success=False response. A backend
                # that was never reached propagates either way — that run has no
                # result to swallow it into.
                if (
                    self.config.raise_on_error
                    or classify_provider_error(e).category == "unreachable"
                ):
                    prom_metrics.errors.inc(labels=labels)
                    set_span_error(e)
                    self._record_provider_metrics(
                        execution_time=time.time() - start_time,
                        outcome=classify_provider_error(e).category,
                    )
                    raise
                # Track failure
                self.execution_tracker.track_event(ExecutionEvent(
                    type=EventType.TASK_FAILED,
                    agent_id=self.name,
                    message=f"Task failed: {str(e)}",
                    data={"error": str(e)}
                ))
                prom_metrics.errors.inc(labels=labels)
                set_span_error(e)
                # Build a structured, redacted error record so this catch-all
                # surfaces the same metadata["error"]={type,provider,model,...}
                # shape (and redacted message) as the inner failure paths.
                detail = self._build_error_detail(e, self.model)
                redacted_msg = detail["message"]
                _slog.agent_event(self.name, "task_failed", level=logging.ERROR, error=redacted_msg)
                _obs_log.agent_event("run.failed", level=logging.ERROR, agent=self.name, run_id=run_id, error=redacted_msg)
                response = AgentResponse(
                    output=f"Error: {redacted_msg}",
                    success=False,
                    execution_time=time.time() - start_time,
                    execution_trace=self.execution_tracker.get_trace(),
                    execution_tree=self.execution_tracker.generate_execution_tree(),
                    metadata={"reason": "run_failed", "error": detail, "run_id": run_id}
                )
                self._record_dashboard_run(response, error=redacted_msg, task=task)
                self._record_provider_metrics(
                    execution_time=response.execution_time,
                    outcome=classify_provider_error(e).category,
                )
                return self._stamp_run_identity(
                    response, task=task, started_at=started_at
                )

            finally:
                prom_metrics.active_agents.dec(labels=labels)
                self._active_middleware = None
                self._active_run_context = None
                if _previous_session is not None:
                    self._exit_run_session(_previous_session)

    def run_batch(
        self,
        queries: list[str],
        max_concurrency: int = 5,
        batch_size: int = 0,
        retry_failed: int = 1,
        timeout_per_item: float = 120.0,
        progress_callback: Callable[[int, int], None] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Run multiple queries in parallel through this agent.

        Convenience wrapper around :class:`~effgen.core.batch.BatchRunner`.

        Args:
            queries: List of query strings to execute.
            max_concurrency: Maximum number of concurrent agent runs.
            batch_size: Process queries in batches of this size (0 = all at once).
            retry_failed: Number of retries for failed queries.
            timeout_per_item: Timeout per query in seconds.
            progress_callback: Called with (completed, total) after each query.
            **run_kwargs: Extra keyword arguments forwarded to ``self.run()``.

        Returns:
            BatchResult containing all AgentResponse objects in input order.
        """
        from .batch import BatchConfig, BatchRunner

        config = BatchConfig(
            max_concurrency=max_concurrency,
            batch_size=batch_size,
            retry_failed=retry_failed,
            timeout_per_item=timeout_per_item,
            progress_callback=progress_callback,
        )
        runner = BatchRunner(self)
        return runner.run(queries, config=config, **run_kwargs)

    # ------------------------------------------------------------------ Resume
    def resume(self, checkpoint_id: str | None = None, checkpoint_dir: str = "./checkpoints", **kwargs: Any) -> "AgentResponse":
        """
        Resume execution from a checkpoint.

        Args:
            checkpoint_id: Checkpoint id (or path to a JSON file). If None,
                loads the most recent checkpoint in ``checkpoint_dir``.
            checkpoint_dir: Directory containing checkpoints.
            **kwargs: Additional run() kwargs.

        Returns:
            AgentResponse from continuing the task.
        """
        from .checkpoint import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir)
        cp = mgr.load(checkpoint_id) if checkpoint_id else mgr.load_latest()
        CheckpointManager.restore_to_agent(self, cp)
        # Seed the next run with the saved scratchpad
        kwargs.setdefault("_resume_scratchpad", cp.scratchpad)
        kwargs.setdefault("checkpoint_dir", checkpoint_dir)
        return self.run(cp.task, **kwargs)

    def run_background(self, task: str, priority: int = 5, **run_kwargs: Any) -> str:
        """Submit a task to the background runner and return its id.

        Args:
            task: The task text to run.
            priority: Queue priority; a lower number runs sooner.
            **run_kwargs: Extra keyword arguments forwarded to :meth:`run`.
        """
        if self._bg_runner is None:
            from .background import BackgroundTaskRunner
            self._bg_runner = BackgroundTaskRunner(self, max_workers=1)
        return self._bg_runner.submit(task, priority=priority, **run_kwargs)

    def get_task_status(self, task_id: str) -> Any:
        """Return the status of a background task."""
        if self._bg_runner is None:
            raise RuntimeError("No background runner active")
        return self._bg_runner.get_status(task_id)

    def get_task_result(self, task_id: str, wait: bool = False, timeout: float | None = None) -> Any:
        """Return the result of a background task (optionally blocking).

        Args:
            task_id: The id :meth:`run_background` returned.
            wait: Block until the task finishes instead of returning at once.
            timeout: Seconds to block for when *wait* is set.
        """
        if self._bg_runner is None:
            raise RuntimeError("No background runner active")
        return self._bg_runner.get_result(task_id, wait=wait, timeout=timeout)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a background task; returns ``False`` when it cannot be cancelled."""
        if self._bg_runner is None:
            return False
        return self._bg_runner.cancel(task_id)

    def pause_task(self, task_id: str) -> bool:
        """Pause a running background task; returns whether it was paused."""
        if self._bg_runner is None:
            return False
        return self._bg_runner.pause(task_id)

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused background task; returns whether it was resumed."""
        if self._bg_runner is None:
            return False
        return self._bg_runner.resume(task_id)

    def synthesize(self, synthesis_data: dict[str, Any]) -> str:
        """
        Synthesize results from sub-agents.

        Args:
            synthesis_data: Data to synthesize

        Returns:
            Synthesized output
        """
        # Build synthesis prompt
        results_text = []
        for result in synthesis_data.get("results", []):
            output = result.get("output", {})
            if isinstance(output, dict):
                results_text.append(output.get("output", str(output)))
            else:
                results_text.append(str(output))

        prompt = f"""Synthesize the following results into a comprehensive answer for: {synthesis_data['original_task']}

Results from sub-agents:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(results_text))}

Provide a well-structured, comprehensive response that integrates all findings."""

        # Generate synthesis
        response = self._generate(prompt, temperature=0.6)
        return response.get("text", "").strip()

    async def run_async(self,
                       task: "str | Message | list[Any]",
                       mode: AgentMode | None = None,
                       context: dict[str, Any] | None = None,
                       output_schema: dict[str, Any] | None = None,
                       output_model: Any = None,
                       inputs: list[Any] | None = None,
                       **kwargs: Any) -> AgentResponse:
        """
        Truly asynchronous version of run().

        Runs the synchronous run() method in a thread executor so it
        doesn't block the event loop, while remaining compatible with
        async callers.

        Args:
            task: Task description (str, Message, or list[ContentPart])
            mode: Execution mode. Defaults to ``self.config.mode`` when
                omitted (see ``run()``).
            context: Optional context
            output_schema: A JSON-Schema ``dict`` or a Pydantic ``BaseModel``
                subclass (see ``run``).
            output_model: Pydantic ``BaseModel`` class (see ``run``).
            inputs: Optional multimodal content parts (see ``run``).
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        import contextvars
        import functools
        loop = asyncio.get_running_loop()
        func = functools.partial(
            self.run, task, mode, context,
            output_schema=output_schema, output_model=output_model,
            inputs=inputs, **kwargs,
        )
        # A worker thread starts with an empty context, so the caller's context
        # is copied into it: without this the run loses the ambient state it
        # would have had synchronously, including the team/workflow execution it
        # belongs to.
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(None, functools.partial(ctx.run, func))
