"""The blocking entry to the reasoning loop, and what it reports.

:meth:`AgentReActMixin._run_single_agent` picks the run path and drives the one
loop in :mod:`effgen.core.agent_loop` with the emitter that collects rather than
streams. Beside it live the instructions the loop injects and the typed outcome
it reports when a turn writes a call out instead of making it, when the same
call keeps coming back, or when the iteration cap is reached — and sub-agent
delegation.

The surrounding concerns are inherited by :class:`AgentReActMixin`, so every
method resolves on :class:`Agent` as before: reading a turn in
:class:`~effgen.core.agent_react_parsing.AgentReActParsingMixin`, the
provider-native run paths in
:class:`~effgen.core.agent_native_tools.AgentNativeToolsMixin`, tool dispatch in
:class:`~effgen.core.agent_tool_execution.AgentToolExecutionMixin` and citation
assembly in :class:`~effgen.core.agent_citations.AgentCitationsMixin`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..observability import get_logger as _get_obs_logger
from ..tools.base_tool import ToolCategory
from ..utils.structured_logging import (
    get_structured_logger,
)
from .agent_citations import AgentCitationsMixin
from .agent_loop import (
    _budget_of,
    _Collecting,
    _decline_call,  # noqa: F401 - the loop's, re-exported where it was
    _is_request_shape_refusal,  # noqa: F401 - same
    _LoopPolicy,
    _protocol_of,
    drive,
    run_to_completion,
)
from .agent_native_tools import AgentNativeToolsMixin
from .agent_react_parsing import AgentReActParsingMixin
from .agent_tool_execution import AgentToolExecutionMixin
from .execution_tracker import EventType, ExecutionEvent
from .router import RoutingDecision, RoutingStrategy
from .thread import (
    AgentThread,
    AnswerStep,
    TaskStep,
)
from .tool_call_record import ToolCallList

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)

from .agent import AgentMode, AgentResponse  # noqa: E402
from .agent_runtime import (  # noqa: E402
    CONTEXT_ANSWER_INSTRUCTION,
    CONTEXT_CITATION_INSTRUCTION,
    CONTINUE_INSTRUCTION,
    sanitize_final_answer,
)

#: Models that complete a tool loop through the provider's tool-calling API,
#: named in the hint a written-out call block produces. Kept short and stable;
#: ``effgen models list`` marks every model that advertises tool calling.
_TOOL_CALLING_EXAMPLES = (
    "openai:gpt-5-nano, gemini:gemini-3.1-flash-lite or groq:openai/gpt-oss-120b"
)


class AgentReActMixin(
    AgentReActParsingMixin,
    AgentNativeToolsMixin,
    AgentToolExecutionMixin,
    AgentCitationsMixin,
):
    """The ReAct loop, and the surrounding tool-calling surface it inherits."""

    if TYPE_CHECKING:
        # Contributed by :class:`~effgen.core.agent.Agent`, which owns the
        # per-call state. Declared for the type checker only — at run time they
        # arrive through the MRO, and these statements do not execute.
        model: Any
        tools: dict[str, Any]

        def _effective_output_schema(self) -> dict[str, Any] | None: ...

    def _run_single_agent(self,
                         task: str,
                         context: dict[str, Any],
                         **kwargs) -> AgentResponse:
        """Run *task* on this agent alone, through the loop or directly.

        The early returns pick the run path — a tool-free agent goes straight
        to the model, a provider-hosted tool set goes to its own loop — and
        everything else is the one reasoning loop, driven here with the emitter
        that collects rather than streams. Content parts travel with the
        conversation, so an agent given a picture *and* tools drives the loop
        like any other run.

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # If no tools available, use direct inference instead of ReAct
        if not self.tools:
            return self._run_direct_inference(task, context, **kwargs)

        # A provider-hosted tool set drives its own iteration inside the
        # provider, so a run carrying content parts goes straight to the model
        # there as it always has.
        if kwargs.get("inputs") is not None and (
            self._has_native_tools() or self._has_gemini_native_tools()
        ):
            return self._run_direct_inference(task, context, **kwargs)

        # If any native OpenAI tools are present and the model supports it,
        # route through the Responses API directly (not the ReAct loop).
        if self._has_native_tools():
            return self._run_with_native_tools(task, context, **kwargs)

        # If any Gemini native tools are present, route through the Gemini
        # native-tool path which passes tool objects directly to the adapter.
        if self._has_gemini_native_tools():
            return self._run_with_gemini_native_tools(task, context, **kwargs)

        policy = _LoopPolicy.for_run(self, kwargs, emit_deltas=False)
        return run_to_completion(drive(self, task, policy, _Collecting(self)))

    def _written_tool_call_detail(
        self, tool_name: str, answer: str, *, tool_ran: bool = False,
    ) -> dict[str, Any]:
        """Return the typed error for an answer that writes out a tool call.

        The remediation depends on which tool-calling path ran: a model that was
        sent the tool definitions natively and still answered with the call as
        text needs replacing, while a model that advertises native tool calling
        but ran the ReAct text protocol only needs to be asked for the native
        path. It also names how the definitions reached the model — a provider's
        tool-calling API or a local chat template — so the advice matches what
        actually happened. *tool_ran* says whether the named tool was dispatched
        earlier in the run, which decides what the answer failed to do.
        """
        strategy = self._tool_calling_strategy.name
        model_id = (
            getattr(self.model, "model_name", None) or self.model_name or "the model"
        )
        advertises = self._model_advertises_tool_calling()
        if strategy in ("native", "hybrid") and advertises:
            delivery = (
                "rendered into the prompt by its chat template"
                if self._model_tool_call_support() == "template"
                else "sent through the provider's tool-calling API"
            )
            remedy = (
                f"'{model_id}' had the tool definitions {delivery} and answered "
                "with the call as text anyway. Run the task on a model that "
                f"calls tools — {_TOOL_CALLING_EXAMPLES} — or on a larger local "
                "model."
            )
        elif advertises:
            remedy = (
                f"This run used the ReAct text protocol, but '{model_id}' "
                "advertises native tool calling: build the agent with "
                "AgentConfig(tool_calling_mode='native') so the tool definitions "
                "reach the provider's tool-calling API."
            )
        else:
            remedy = (
                f"'{model_id}' does not advertise native tool calling. Run the "
                f"task on a model that does — {_TOOL_CALLING_EXAMPLES}."
            )
        if tool_ran:
            message = (
                f"The model returned a '{tool_name}' tool call as its answer "
                "instead of an answer, so the run has no result to report and "
                "the call as written was not carried out. "
            ) + remedy
        else:
            message = (
                f"The model wrote a '{tool_name}' tool call into its answer "
                f"instead of calling the tool, so {tool_name} never ran and "
                f"nothing the answer describes was carried out. "
            ) + remedy
        preview = " ".join((answer or "").split())[:300]
        return {
            "type": "WrittenToolCall",
            "category": "written_tool_call",
            "provider": self._model_provider(self.model),
            "model": model_id,
            "tool": tool_name,
            "tool_calling_strategy": strategy,
            "answer_preview": preview,
            "message": message,
            "retryable": False,
        }

    def _written_tool_call_response(
        self,
        tool_name: str,
        answer: str,
        *,
        iterations: int,
        tool_calls: int,
        tokens_used: int,
        tool_ran: bool = False,
        debug_trace: Any = None,
        calls: Any = (),
        thread: AgentThread | None = None,
    ) -> AgentResponse:
        """Report a turn whose answer only describes the tool call it should have made.

        The model did not do the work, so this stays a failure and claims no
        result. When tools *had* run earlier in the run, what they returned is
        carried as partial progress rather than dropped — it is not an answer,
        but it is what the run has.
        """
        detail = self._written_tool_call_detail(tool_name, answer, tool_ran=tool_ran)
        logger.warning("Tool call was written as text, not made: %s", detail["message"])
        run = thread if thread is not None else AgentThread()
        run.append(AnswerStep(text=answer or "", stop_reason="written_tool_call"))
        meta: dict[str, Any] = {
            "reason": "written_tool_call",
            "error": detail,
            "tool_calling_strategy": detail["tool_calling_strategy"],
            "thread": run,
            "prompt_protocol": _protocol_of(run),
            "context_budget": _budget_of(run),
        }
        partial = None
        if calls:
            candidate = self._partial_result(
                run,
                text=self._extract_partial_answer(run) or "",
                calls=calls,
                iterations=iterations,
                tool_calls=tool_calls,
            )
            if candidate.text.strip():
                partial = candidate
                meta["partial"] = True
                meta["partial_output"] = partial.text
        logger.info(
            "outcome failed: stop_reason=written_tool_call tool=%s observations=%d",
            tool_name or "-",
            len(partial.observations) if partial else 0,
        )
        if debug_trace is not None:
            debug_trace.total_tokens = tokens_used
            debug_trace.final_answer = None
            debug_trace.success = False
            meta["debug_trace"] = debug_trace
        return AgentResponse(
            output=detail["message"],
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=ToolCallList(list(calls), total=tool_calls),
            tokens_used=tokens_used,
            metadata=meta,
            stop_reason="written_tool_call",
            partial=partial,
        )

    def _stopped_outcome_response(
        self,
        text: str,
        *,
        action: str | None,
        reason: str,
        thread: AgentThread,
        iterations: int,
        tool_calls: int,
        tokens_used: int,
        calls: Any,
        debug_trace: Any = None,
        answer: str | None = None,
    ) -> AgentResponse:
        """Report a run the loop stopped before the model wrote an answer.

        Three things end a run this way: the model keeps asking for the same
        tool call, a tool returns a result it already returned, or the model
        answers with nothing usable after its tools ran. In all three the run
        has tool output and no answer. Returning that output as ``output`` with
        ``success=True`` presents a tool's words as if the model had written
        them, and a caller keyed on :attr:`AgentResponse.success` cannot tell
        the difference — a list of intermediate results reads exactly like an
        answer.

        So all three report the same shape: ``success=False``, an
        :attr:`~effgen.core.agent_response.AgentResponse.outcome` of
        ``"stopped"``, the outcome statement in ``output``, a typed
        ``metadata["error"]`` naming what stopped the run, and what the run had
        reached under :attr:`~effgen.core.agent_response.AgentResponse.partial`
        and ``metadata["partial_output"]``.

        Args:
            text: The flattened progress — the tool output or recovered text.
            action: The tool involved, or ``None`` when unnamed.
            reason: ``"loop_detected"``, ``"repeated_tool_result"`` or
                ``"null_final_from_model"``.
            thread: The run's conversation, read for the last thought.
            iterations: Iterations run.
            tool_calls: Tool calls made.
            tokens_used: Tokens consumed.
            calls: The recorded tool calls.
            debug_trace: The debug trace, when one is being collected.
            answer: The unusable final answer, for ``null_final_from_model``.

        Returns:
            The stopped response, carrying the progress.
        """
        retrieval = self._is_context_retrieval_tool(action) if action else False
        detail = self._repeated_tool_detail(
            action, reason, retrieval=retrieval, answer=answer
        )
        partial = self._partial_result(
            thread,
            text=text,
            calls=calls,
            iterations=iterations,
            tool_calls=tool_calls,
        )
        thread.append(AnswerStep(text=text, stop_reason=reason))
        meta: dict[str, Any] = {
            "reason": reason,
            "error": detail,
            "answer_source": reason,
            "repeated_action": action,
            "partial": True,
            "partial_output": text,
            "tool_calling_strategy": self._tool_calling_strategy.name,
            "thread": thread,
            "prompt_protocol": _protocol_of(thread),
            "context_budget": _budget_of(thread),
        }
        logger.info(
            "outcome stopped: stop_reason=%s tool=%s category=%s observations=%d",
            reason,
            action or "-",
            "INFORMATION_RETRIEVAL" if retrieval else "COMPUTATION",
            len(partial.observations),
        )
        if debug_trace is not None:
            debug_trace.total_tokens = tokens_used
            debug_trace.final_answer = None
            debug_trace.success = False
            meta["debug_trace"] = debug_trace
        return AgentResponse(
            output=detail["message"],
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=ToolCallList(list(calls), total=tool_calls),
            tokens_used=tokens_used,
            metadata=meta,
            stop_reason=reason,
            partial=partial,
        )

    #: What every stopped-outcome statement closes with. The run has progress
    #: and no answer, and both remedies are about giving the model room to write
    #: one rather than about the tool that produced the progress.
    _STOPPED_NEXT_STEP = (
        "Try a larger model, or raise max_tokens if the model is spending its "
        "budget before writing."
    )

    def _repeated_tool_detail(
        self,
        action: str | None,
        reason: str,
        *,
        retrieval: bool = True,
        answer: str | None = None,
    ) -> dict[str, Any]:
        """Return the typed outcome for a run that stopped without an answer.

        The statement names what the tool did, because that is what the reader
        has to change. A retrieval tool's output is source material the model
        was asked to write up; a computing tool's output is a number it was
        asked to explain; an unusable final answer is the model declining to
        write either. *answer* is the unusable text, quoted for
        ``null_final_from_model``.
        """
        model_id = (
            getattr(self.model, "model_name", None) or self.model_name or "the model"
        )
        action = action or "the tool"
        if reason == "null_final_from_model":
            quoted = " ".join((answer or "").split())[:80]
            message = (
                f"'{model_id}' returned an empty final answer ('{quoted}') after "
                "using tools, so the run has no answer to report. What the tools "
                "returned is reported as partial progress — tool output, not an "
                f"answer. {self._STOPPED_NEXT_STEP}"
            )
        elif retrieval:
            what = (
                "kept asking for the same information"
                if reason == "loop_detected"
                else "returned the same result again"
            )
            message = (
                f"'{model_id}' did not write an answer: the '{action}' tool "
                f"{what}, and the model would not synthesize from it even with "
                "the tools withdrawn. What was retrieved is reported as partial "
                f"progress — context, not an answer. {self._STOPPED_NEXT_STEP}"
            )
        elif reason == "loop_detected":
            message = (
                f"'{model_id}' did not write an answer: it kept asking "
                f"'{action}' for the same computation, and the run stopped "
                "rather than repeat it. The results it had are reported as "
                f"partial progress — tool output, not an answer. "
                f"{self._STOPPED_NEXT_STEP}"
            )
        else:
            message = (
                f"'{model_id}' did not write an answer: '{action}' returned the "
                "same result twice and the run stopped rather than compute it "
                "again. The results it had are reported as partial progress — "
                f"tool output, not an answer. {self._STOPPED_NEXT_STEP}"
            )
        return {
            "type": "UnsynthesizedToolResult",
            "category": reason,
            "provider": self._model_provider(self.model),
            "model": model_id,
            "repeated_tool": action,
            "message": message,
            "retryable": False,
        }

    def _iteration_cap_detail(self, cap: int, progress: str | None) -> dict[str, Any]:
        """Return the typed outcome for a run that stopped at its iteration cap.

        The loop ran out of iterations before the model wrote a final answer, so
        the run has no answer to report. What the thread holds at that point
        is tool output and reasoning: returning it as the result presents a
        retrieved passage as if the model had written it. The outcome therefore
        states what happened and what to do, and the recovered text travels
        beside it as ``metadata["partial_output"]``.
        """
        model_id = (
            getattr(self.model, "model_name", None) or self.model_name or "the model"
        )
        step = "iteration" if cap == 1 else "iterations"
        message = (
            f"Stopped after {cap} {step} without a final answer: '{model_id}' "
            "was still taking tool steps when the limit was reached."
        )
        if progress:
            message += (
                " What it had reached by then is reported as partial progress "
                "— tool output and reasoning, not an answer."
            )
        message += (
            f" Raise max_iterations above {cap} to give the run more steps, or "
            "run the task on a model that needs fewer."
        )
        return {
            "type": "MaxIterationsReached",
            "category": "max_iterations",
            "provider": self._model_provider(self.model),
            "model": model_id,
            "max_iterations": cap,
            "message": message,
            "retryable": False,
        }

    def _is_context_retrieval_tool(self, action: str) -> bool:
        """True when ``action`` is a knowledge-base/search tool whose output is
        retrieved context rather than a computed answer.

        Used to flag a fallback that returns such a tool's raw observation as
        partial, so a passage dump is not presented as a synthesized answer, and
        to pick the continuation instruction in :meth:`_continuation_instruction`.

        A tool may declare it directly with ``is_context_retrieval = True``,
        which is how a tool whose category says otherwise — a file tool narrowed
        to reading, whose output is source material — opts in. The category and
        name checks below are unchanged, so every other agent classifies exactly
        as before.
        """
        tool = self.tools.get(action)
        if getattr(tool, "is_context_retrieval", False):
            return True
        category = getattr(getattr(tool, "metadata", None), "category", None)
        if category is ToolCategory.INFORMATION_RETRIEVAL:
            return True
        return action in {"retrieval", "web_search", "search", "knowledge_base"}

    def _context_answer_instruction(
        self,
        previous_actions: list[tuple[str, str]],
        *,
        cite_sources: bool = False,
        numbered_passages: int = 0,
    ) -> str:
        """Return the answer-shaping line when the latest observation is
        retrieved context, or ``""`` for every other tool.

        A tool prompt ends with the last tool's observation, so whatever follows
        it is the final thing the model reads before answering. After a
        retrieval/search tool that observation is a block of source passages, and
        a generic close leaves the strongest recent signal a wall of text that
        reads like a finished answer: the smallest models return it verbatim,
        losing the question's scope along the way. This line states what to do
        with the passages instead. Returning ``""`` for every other tool keeps
        those prompts byte-for-byte unchanged.

        ``cite_sources`` is the caller's request for inline ``[1]``, ``[2]``
        markers, and ``numbered_passages`` is how many passages the run has
        actually numbered for the model. Markers are asked for only when both
        hold, so a marker always has a numbered list behind it. The flag is a
        parameter rather than a read of the config, which keeps this a function
        of what ran and what was asked for.
        """
        if previous_actions and self._is_context_retrieval_tool(previous_actions[-1][0]):
            logger.info("answer shape: retrieval close applied")
            if cite_sources and numbered_passages:
                return f"{CONTEXT_ANSWER_INSTRUCTION} {CONTEXT_CITATION_INSTRUCTION}"
            return CONTEXT_ANSWER_INSTRUCTION
        return ""

    def _answer_shape_instruction(self) -> str:
        """Return the schema this run must answer in, stated for the model, or
        ``""`` when the caller declared no shape.

        ``output_schema`` / ``output_model`` is the one machine-readable
        statement of shape the framework has, and without this the model never
        sees it: the answer is written as prose, and the schema is applied
        afterwards by re-prompting for the same answer in a different form. The
        line goes into the prompt the answer is written from, so the declaration
        is honoured on the first attempt rather than repaired on the second.

        Empty for every run without a schema, which keeps those prompts
        byte-for-byte unchanged.
        """
        try:
            schema = self._effective_output_schema()
        except Exception:  # pragma: no cover - defensive
            return ""
        if not schema:
            return ""
        from .structured_output import schema_answer_instruction
        logger.info("answer shape: declared schema stated in the loop prompt")
        return schema_answer_instruction(schema)

    @staticmethod
    def _compose_closing(answer_shape: str, closing: str) -> str:
        """Join the declared-shape line and the tool close into one block.

        The caller's declared shape comes first so the framework's own line
        about the machinery it inserted is not the last word on what the
        answer should look like.
        """
        return "\n\n".join(part for part in (answer_shape, closing) if part)

    def _continuation_instruction(
        self,
        previous_actions: list[tuple[str, str]],
        *,
        cite_sources: bool = False,
        numbered_passages: int = 0,
    ) -> str:
        """Return the line that closes the native/hybrid prompt after a tool ran."""
        return self._context_answer_instruction(
            previous_actions,
            cite_sources=cite_sources,
            numbered_passages=numbered_passages,
        ) or CONTINUE_INSTRUCTION

    def _citation_prompt_state(self) -> tuple[bool, int]:
        """What the prompt needs to know about citations: whether the caller
        asked for inline markers, and how many passages carry a number."""
        try:
            cite = self._cite_sources_requested()
            numbered = len(
                [e for e in self._collected_citations if e.get("cite_index")]
            )
        except Exception:  # pragma: no cover - defensive
            return False, 0
        return cite, numbered

    def _run_with_sub_agents(self,
                            task: str,
                            routing_decision: RoutingDecision,
                            context: dict[str, Any],
                            **kwargs) -> AgentResponse:
        """
        Execute task using sub-agents based on routing decision.

        Args:
            task: Task description
            routing_decision: Router's decision
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # The parent's own conversation for this run. A decomposed run never
        # built one before, so its response was the only kind with no
        # ``metadata["thread"]``; it has one now, carrying what it was asked,
        # the decomposition that chose the children, one record per child and
        # the answer it synthesised. Taken out of the keyword arguments before
        # any path can forward them to a single-agent run, which builds its own.
        thread = kwargs.pop("_run_thread", None) or AgentThread(
            steps=[TaskStep(text=task)]
        )

        if self._current_depth >= self.config.max_sub_agent_depth:
            logger.warning(f"Sub-agent depth limit reached ({self.config.max_sub_agent_depth})")
            return self._run_single_agent(task, context, **kwargs)

        self._current_depth += 1
        manager = self.sub_agent_manager
        previous_thread = getattr(manager, "parent_thread", None)
        manager.parent_thread = thread

        try:
            # Track decomposition
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_DECOMPOSITION,
                agent_id=self.name,
                message=f"Decomposed into {routing_decision.num_sub_agents} subtasks using {routing_decision.strategy.value}",
                data={
                    "strategy": routing_decision.strategy.value,
                    "num_subtasks": routing_decision.num_sub_agents,
                    "specializations": routing_decision.specializations
                }
            ))

            # Execute based on strategy
            strategy = routing_decision.strategy
            subtasks = routing_decision.decomposition

            if strategy == RoutingStrategy.PARALLEL_SUB_AGENTS:
                # Execute in parallel (use helper to handle existing event loops)
                results = self._run_coroutine_sync(
                    manager.execute_parallel(subtasks)
                )
            elif strategy == RoutingStrategy.SEQUENTIAL_SUB_AGENTS:
                # Execute sequentially
                results = manager.execute_sequential(subtasks)
            elif strategy == RoutingStrategy.HYBRID:
                # Execute with hybrid approach
                results = manager.execute_hybrid(subtasks)
            else:
                # Default to sequential
                results = manager.execute_sequential(subtasks)

            # Synthesize results
            synthesis = manager.synthesize_results(
                results,
                task,
                strategy
            )

            # Calculate totals
            total_tokens = synthesis["metrics"]["total_tokens_used"]
            total_tool_calls = synthesis["metrics"]["total_tool_calls"]

            answered = synthesis["successful"] > 0
            output = (
                sanitize_final_answer(synthesis["final_output"])
                or synthesis["final_output"]
            )
            stop_reason = "final_answer" if answered else "sub_agent_failed"
            thread.append(AnswerStep(text=output, stop_reason=stop_reason))
            logger.info(
                "[thread] the decomposed run recorded %d delegation(s), "
                "%d carrying the child's own conversation",
                len(thread.delegations()), len(thread.child_threads()),
            )
            return AgentResponse(
                output=output,
                success=answered,
                mode=AgentMode.SUB_AGENTS,
                iterations=len(subtasks),
                tool_calls=total_tool_calls,
                tokens_used=total_tokens,
                routing_decision=routing_decision,
                metadata={
                    "synthesis": synthesis,
                    "failed_subtasks": synthesis["failed"],
                    "thread": thread,
                },
                stop_reason=stop_reason,
            )
        finally:
            self._current_depth -= 1
            manager.parent_thread = previous_thread




