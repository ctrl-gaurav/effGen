"""The ReAct reasoning loop for :class:`Agent`.

Holds the loop itself — the scratchpad, the instructions and nudges it appends,
the stop it reports when a turn writes a call out instead of making it or when
the iteration cap is reached — and sub-agent delegation. The surrounding
concerns live beside it and are inherited by :class:`AgentReActMixin`, so every
method resolves on :class:`Agent` as before: reading a turn in
:class:`~effgen.core.agent_react_parsing.AgentReActParsingMixin`, the
provider-native run paths in
:class:`~effgen.core.agent_native_tools.AgentNativeToolsMixin`, tool dispatch in
:class:`~effgen.core.agent_tool_execution.AgentToolExecutionMixin` and citation
assembly in :class:`~effgen.core.agent_citations.AgentCitationsMixin`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from ..observability import get_logger as _get_obs_logger
from ..observability.spans import ModelAttrs, ToolAttrs
from ..observability.tracing import (
    stamp_call_cost as _stamp_call_cost,
)
from ..observability.tracing import (
    start_agent_iteration,
    start_model_call,
    start_tool_call,
)
from ..tools.base_tool import ToolCategory
from ..utils.prometheus_metrics import metrics as prom_metrics
from ..utils.structured_logging import (
    get_structured_logger,
)
from .agent_citations import AgentCitationsMixin
from .agent_native_tools import AgentNativeToolsMixin
from .agent_react_parsing import AgentReActParsingMixin
from .agent_tool_execution import AgentToolExecutionMixin
from .agent_tool_loop import NativeToolLoop
from .execution_tracker import EventType, ExecutionEvent
from .router import RoutingDecision, RoutingStrategy
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
    NUDGE_ALREADY_COMPUTED,
    NUDGE_CONTINUE,
    NUDGE_HAVE_RESULTS,
    NUDGE_MUST_EXECUTE,
    NUDGE_NO_TOOLS,
    NUDGE_NOT_USABLE,
    _infer_provider_from_model,
    find_written_tool_call,
    model_can_require_tool_call,
    sanitize_final_answer,
    unknown_tool_observation,
)

#: Models that complete a tool loop through the provider's tool-calling API,
#: named in the hint a written-out call block produces. Kept short and stable;
#: ``effgen models list`` marks every model that advertises tool calling.
_TOOL_CALLING_EXAMPLES = (
    "openai:gpt-5-nano, gemini:gemini-3.1-flash-lite or groq:llama-3.3-70b-versatile"
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
        # per-call state. Declared for the type checker only — at run time it
        # arrives through the MRO, and these statements do not execute.
        model: Any

        def _effective_output_schema(self) -> dict[str, Any] | None: ...

        # Contributed by :class:`~effgen.core.agent_runtime.AgentRuntimeMixin`,
        # which holds the prompt assembly both tool loops share.
        def _tool_contract(self) -> str: ...

        def _native_tool_prompt(
            self, task: str, scratchpad: str, conversation_history: str,
            previous_actions: list[tuple[str, str]],
        ) -> str: ...

    def _run_single_agent(self,
                         task: str,
                         context: dict[str, Any],
                         **kwargs) -> AgentResponse:
        """
        Execute task using single agent with ReAct loop or direct inference.

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # Extract debug flags (set by run())
        debug = kwargs.pop("_debug", False)
        run_id = kwargs.pop("_run_id", "")
        # Pop custom kwargs so they don't leak to the model layer.
        # We re-read them locally below before they would propagate further.
        _ckpt_interval_arg = kwargs.pop("checkpoint_interval", 0) or 0
        _ckpt_dir_arg = kwargs.pop("checkpoint_dir", None)
        _resume_scratchpad_arg = kwargs.pop("_resume_scratchpad", None)

        # Structured multimodal inputs must reach adapters as Message parts.
        # The ReAct prompt is text-only, so use direct inference for these calls
        # even when the preset includes tools.
        if kwargs.get("inputs") is not None:
            return self._run_direct_inference(task, context, **kwargs)

        # If no tools available, use direct inference instead of ReAct
        if not self.tools:
            return self._run_direct_inference(task, context, **kwargs)

        # If any native OpenAI tools are present and the model supports it,
        # route through the Responses API directly (not the ReAct loop).
        if self._has_native_tools():
            return self._run_with_native_tools(task, context, **kwargs)

        # If any Gemini native tools are present, route through the Gemini
        # native-tool path which passes tool objects directly to the adapter.
        if self._has_gemini_native_tools():
            return self._run_with_gemini_native_tools(task, context, **kwargs)

        iterations = 0
        tool_calls = 0
        tokens_used = 0
        scratchpad = ""
        # An explicit ``max_iterations=None`` — what an optional flag forwards when
        # the user did not set it — must fall back to the configured cap rather
        # than reach the loop comparison as None.
        _requested_iterations = kwargs.get("max_iterations")
        max_iterations: int = (
            self.config.max_iterations
            if _requested_iterations is None
            else int(_requested_iterations)
        )

        # Debug trace collector
        debug_trace = None
        if debug:
            from ..debug.inspector import DebugTrace
            debug_trace = DebugTrace(
                task=task, agent_name=self.name, run_id=run_id,
            )

        # Format conversation history
        conversation_history = self._format_conversation_history()

        # ReAct loop. The repeat guards — which calls have been dispatched, which
        # results have already come back, when to stop offering tools and when a
        # written-out call has been seen once too often — live in the loop policy
        # the streaming loop shares, so both reach the same decisions.
        guards = NativeToolLoop(self.tools, nudge_cap=self.config.max_iterations)

        # Optional periodic checkpointing
        _ckpt_interval = _ckpt_interval_arg
        _ckpt_dir = _ckpt_dir_arg
        _ckpt_mgr = None
        if _ckpt_interval and _ckpt_dir:
            try:
                from .checkpoint import CheckpointManager as _CM
                _ckpt_mgr = _CM(_ckpt_dir)
            except Exception as _e:
                logger.warning("Failed to init CheckpointManager: %s", _e)
        # Allow resuming with a seeded scratchpad
        if _resume_scratchpad_arg:
            scratchpad = _resume_scratchpad_arg
        while iterations < max_iterations:
            iterations += 1
            iter_start = time.time()
            if _ckpt_mgr is not None and iterations > 1 and (iterations - 1) % _ckpt_interval == 0:
                try:
                    from .checkpoint import CheckpointManager as _CM2
                    cp = _CM2.snapshot_agent(
                        self,
                        task=task,
                        iteration=iterations,
                        scratchpad=scratchpad,
                        tool_calls=tool_calls,
                        tokens_used=tokens_used,
                        metadata={"interval": _ckpt_interval},
                    )
                    self._last_checkpoint_id = _ckpt_mgr.save(cp)
                except Exception as _e:
                    logger.warning("Periodic checkpoint failed: %s", _e)

            # Determine if we should use native tool calling prompt format
            use_native_prompt = (
                self._tool_calling_strategy.name in ("native", "hybrid")
                and self.model is not None
                and hasattr(self.model, 'supports_tool_calling')
                and self.model.supports_tool_calling()
            )

            # Build prompt
            _cite_sources, _numbered_passages = self._citation_prompt_state()
            _answer_shape = self._answer_shape_instruction()
            gen_kwargs = dict(kwargs)
            # After 2 multi-tool batches, or once a loop with no usable partial
            # answer was detected, stop passing tools to force synthesis.
            if guards.tools_suppressed():
                use_native_prompt = False
            if use_native_prompt and not self.config.system_prompt_template:
                # Native/hybrid mode: use a simple user message and pass
                # tool definitions via the chat template's tools parameter.
                # The model will produce native tool call tokens (e.g.
                # <tool_call> for Qwen, [TOOL_CALLS] for Mistral). The prompt is
                # assembled by the same function the streamed loop calls, so the
                # two paths cannot say different things to the same model.
                prompt = self._native_tool_prompt(
                    task, scratchpad, conversation_history, guards.previous_actions,
                )
                # Pass tool definitions for the chat template
                tool_defs = self._tool_calling_strategy.format_tools_for_prompt(
                    list(self.tools.values())
                )
                if isinstance(tool_defs, list):
                    gen_kwargs["tools"] = tool_defs
            elif self.config.system_prompt_template:
                # User-provided custom template
                tools_description = self._get_tools_description()
                prompt = self.config.system_prompt_template.format(
                    tools_description=tools_description,
                    conversation_history=conversation_history,
                    task=task,
                    scratchpad=scratchpad
                )
            else:
                # ReAct mode: use enhanced ToolPromptGenerator
                prompt = self._tool_prompt_generator.generate_react_prompt(
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                    system_prompt=self.config.system_prompt,
                    verbose=self._verbose_tools,
                    closing_instruction=self._context_answer_instruction(
                        guards.previous_actions,
                        cite_sources=_cite_sources,
                        numbered_passages=_numbered_passages,
                    ),
                    answer_shape=_answer_shape,
                    tool_contract=self._tool_contract(),
                )

            # A turn that answered while holding a tool doing work the model
            # cannot do in its head was sent back once (see the acceptance check
            # below); this is the turn that follows, and it is required to call.
            #
            # The constraint only exists where the definitions travel as a
            # request parameter the provider enforces. On the ReAct-text path
            # there is nothing to constrain — the tools are prose in the prompt —
            # and on an adapter that does not advertise it, sending it anyway
            # loses the turn. Both degrade to the nudge already in the
            # scratchpad, which is the whole of the ask for them.
            #
            # The flag is spent whether or not it could be used, so a turn that
            # could not be constrained does not leak the constraint onto a later
            # one.
            if guards.take_forced_tool_call():
                if "tools" in gen_kwargs and model_can_require_tool_call(self.model):
                    gen_kwargs["tool_choice"] = "required"
                    logger.info(
                        "forced tool call: requiring a call on iteration %d",
                        iterations,
                    )
                else:
                    logger.info(
                        "forced tool call: nudge only on iteration %d "
                        "(no request-level constraint available here)",
                        iterations,
                    )

            # Debug: log first iteration prompt to see if history is included
            if iterations == 1 and conversation_history:
                logger.info(f"[Memory] Including conversation history ({len(self.short_term_memory.messages)} messages)")

            # Track reasoning step
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.REASONING_STEP,
                agent_id=self.name,
                message=f"Iteration {iterations}: Reasoning...",
                data={"iteration": iterations}
            ))

            # Generate response inside tracing span
            with start_agent_iteration(preset=self.name, iteration=iterations):
                model_name = getattr(self, "model_name", None) or "unknown"
                provider = _infer_provider_from_model(self.model, model_name)
                with start_model_call(provider=provider, model=model_name) as _mspan:
                    response = self._generate(prompt, **gen_kwargs)
                    # Annotate span with token counts from response
                    _meta = response.get("metadata") or {}
                    _in_tok = _meta.get("prompt_tokens", 0) or 0
                    _out_tok = response.get("tokens_used", 0) or 0
                    _cached = _meta.get("cached_input_tokens", 0) or 0
                    try:
                        _mspan.set_attribute(ModelAttrs.INPUT_TOKENS, int(_in_tok))
                        _mspan.set_attribute(ModelAttrs.OUTPUT_TOKENS, int(_out_tok))
                        if _cached:
                            _mspan.set_attribute(ModelAttrs.CACHED_TOKENS, int(_cached))
                        _mspan.set_attribute(ModelAttrs.OUTCOME, "ok" if response.get("finish_reason") != "error" else "error")
                        _stamp_call_cost(_mspan, _meta)
                    except Exception:
                        logger.debug("Failed to set model span attributes", exc_info=True)
                iter_tokens = response.get("tokens_used", 0)
                tokens_used += iter_tokens

            _slog.iteration_event(iterations, "generate", tokens=iter_tokens)
            _obs_log.event("agent.iteration.generate", iteration=iterations, tokens=iter_tokens, model=getattr(self, "model_name", "unknown"))

            if response.get("finish_reason") == "error":
                return self._generation_failure_response(
                    response,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    tokens=tokens_used,
                    debug_trace=debug_trace,
                )

            # Debug: Log the raw response
            logger.info(f"[Iteration {iterations}] Raw model output: {response['text'][:300]}...")
            logger.debug(f"[Iteration {iterations}] Full model output: {response['text']}")

            # Parse response using strategy. If the adapter returned a native
            # tool call (empty text + structured tool_calls in metadata), use
            # it directly — no text parsing needed.
            native_tool_calls = response.get("tool_calls") or []

            # Whether this turn's own text was written before any observation
            # this turn produced. A batch of calls is dispatched after the model
            # has finished writing, so its text cannot state a result the batch
            # returned; the answer recovery below has to know that.
            dispatched_calls_this_turn = False

            # Execute ALL native tool calls in one batch (OpenAI/Cerebras can
            # return multiple tool_calls in a single response).
            if len(native_tool_calls) > 1 and self.tools:
                batch_observations: list[str] = []
                for _tc in native_tool_calls:
                    _fn = _tc.get("function", _tc)
                    _tname = _fn.get("name", "")
                    _targs = _fn.get("arguments", {})
                    if isinstance(_targs, str):
                        try:
                            _targs = json.loads(_targs)
                        except (json.JSONDecodeError, TypeError):
                            _targs = {"__raw_input__": _targs}
                    if _tname in self.tools:
                        with start_tool_call(tool_name=_tname, tool_input=str(_targs)[:500]) as _btspan:
                            _obs = self._execute_tool(_tname, json.dumps(_targs))
                            try:
                                _btspan.set_attribute(ToolAttrs.STATUS, "ok")
                            except Exception:
                                logger.debug("Failed to set tool span status", exc_info=True)
                        tool_calls += 1
                        guards.record_execution(_tname)
                        batch_observations.append(f"[{_tname}({_targs})] → {_obs}")
                        scratchpad += f"\nAction: {_tname}\nAction Input: {json.dumps(_targs)}\nObservation: {_obs}"
                    else:
                        batch_observations.append(f"[{_tname}] → Tool not found")
                # After batch execution, nudge model to synthesize a final answer.
                scratchpad += f"\n{NUDGE_CONTINUE}"
                guards.note_batch_run()
                parsed = {"thought": "", "action": None, "action_input": None, "final_answer": None}
                dispatched_calls_this_turn = True
                cur_observation = "\n".join(batch_observations)
                logger.info(f"[Batch native tool calls] {len(native_tool_calls)} calls executed (batch run #{guards.batch_tool_runs})")
            elif native_tool_calls:
                strategy_result = self._parse_native_tool_calls(native_tool_calls)
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)
            else:
                parse_strategy = self._text_parse_strategy(use_native_prompt)
                strategy_result = parse_strategy.parse_response(
                    response["text"], tools=self.tools,
                )
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)

            # Debug: Log what was parsed
            logger.info(f"[Iteration {iterations}] Parsed - Action: {parsed.get('action')}, Input: {parsed.get('action_input')}, Final: {parsed.get('final_answer')}")

            # Add to scratchpad. A turn that made a native tool call reports no
            # thought, and the scratchpad is prompt text the model reads back —
            # so an absent thought is an empty line, never the word "None".
            scratchpad += f"\nThought: {parsed.get('thought') or ''}"

            # Capture debug iteration data
            cur_observation = None  # filled later if tool runs

            def _build_response(
                output: str,
                success: bool = True,
                _tokens_used: int = tokens_used,
                _iterations: int = iterations,
                _tool_calls: int = tool_calls,
                _iter_start: float = iter_start,
                _scratchpad: str = scratchpad,
                **extra_meta: Any,
            ) -> AgentResponse:
                """Helper to build response and attach debug trace."""
                if success:
                    raw_answer = output
                    output = sanitize_final_answer(output) or output
                    # An answer that writes out a call for a tool this agent
                    # holds means the tool never ran: the turn describes work
                    # that did not happen, so it is reported as a failure.
                    # Sanitizing a tagged call can leave its arguments behind as
                    # a bare JSON fragment, so the text as the model wrote it is
                    # scanned as well as the cleaned answer.
                    written = find_written_tool_call(
                        output, self.tools
                    ) or find_written_tool_call(raw_answer, self.tools)
                    if written and guards.is_unmade_call(written, raw_answer):
                        return self._written_tool_call_response(
                            written,
                            output,
                            iterations=_iterations,
                            tool_calls=_tool_calls,
                            tokens_used=_tokens_used,
                            tool_ran=guards.tool_ran(written),
                            debug_trace=debug_trace,
                            calls=guards.calls,
                            scratchpad=_scratchpad,
                        )
                meta: dict[str, Any] = {
                    "reason": "final_answer",
                    "tool_calling_strategy": self._tool_calling_strategy.name,
                }
                meta.update(extra_meta)
                if debug_trace is not None:
                    debug_trace.total_tokens = _tokens_used
                    debug_trace.total_latency = time.time() - (_iter_start - (_iterations - 1) * 0.001)
                    debug_trace.final_answer = output if success else None
                    debug_trace.success = success
                    meta["debug_trace"] = debug_trace
                return AgentResponse(
                    output=output,
                    success=success,
                    mode=AgentMode.SINGLE,
                    iterations=_iterations,
                    tool_calls=ToolCallList(list(guards.calls), total=_tool_calls),
                    tokens_used=_tokens_used,
                    metadata=meta,
                )

            # Check for final answer
            final_answer = parsed.get("final_answer")
            if final_answer and tool_calls > 0 and final_answer.strip().lower() in {
                "none",
                "null",
                "n/a",
                "na",
            }:
                partial = self._extract_partial_answer(scratchpad)
                if partial:
                    return self._stopped_outcome_response(
                        sanitize_final_answer(partial) or partial,
                        action=guards.calls[-1].name if guards.calls else None,
                        reason="null_final_from_model",
                        scratchpad=scratchpad,
                        iterations=iterations,
                        tool_calls=tool_calls,
                        tokens_used=tokens_used,
                        calls=guards.calls,
                        debug_trace=debug_trace,
                        answer=final_answer,
                    )

            # A "final answer" that is purely leaked tool-call syntax /
            # scaffolding (sanitizes to nothing) is not a real answer — keep
            # looping so the tool actually runs or a partial is extracted. When
            # what leaked is a call for a tool this agent holds, the model is
            # writing the call instead of making it: nudge once, then report it
            # rather than billing the rest of the iteration budget for the same
            # outcome.
            if final_answer and not (sanitize_final_answer(final_answer) or "").strip():
                written = find_written_tool_call(final_answer, self.tools)
                if written and guards.is_unmade_call(written, final_answer):
                    if guards.note_written_call(written):
                        return self._written_tool_call_response(
                            guards.written_call,
                            final_answer,
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            tool_ran=guards.tool_ran(guards.written_call),
                            debug_trace=debug_trace,
                            calls=guards.calls,
                            scratchpad=scratchpad,
                        )
                logger.info(
                    "Discarding scaffolding-only final answer; continuing loop"
                )
                scratchpad += f"\nObservation: {NUDGE_NOT_USABLE}"
                final_answer = None

            if final_answer:
                # An agent holding a tool that does work the model cannot do in
                # its head has not answered by saying what the tool would have
                # returned: nothing computed that result. The first such answer
                # is not accepted — the turn goes back naming the tool, and the
                # turn after it is required to call one where that can be
                # required. Only the first: a model that declines twice will
                # decline again, and the iterations buy more elsewhere.
                refused_tool = guards.note_execution_refusal()
                if refused_tool is not None:
                    scratchpad += (
                        "\nObservation: "
                        + NUDGE_MUST_EXECUTE.format(tool=refused_tool)
                    )
                    continue

                # Record final debug iteration
                if debug_trace is not None:
                    from ..debug.inspector import DebugIteration
                    debug_trace.iterations.append(DebugIteration(
                        iteration=iterations,
                        raw_prompt=prompt[:2000],
                        raw_response=response["text"][:2000],
                        thought=parsed.get("thought", ""),
                        final_answer=final_answer,
                        tokens_used=iter_tokens,
                        latency=time.time() - iter_start,
                        scratchpad_snapshot=scratchpad,
                    ))
                return _build_response(final_answer)

            # Check if model is stating an answer without "Final Answer:" keyword
            # This happens when model provides result after tool execution.
            #
            # Only when the model has actually seen a result. A turn that
            # dispatched its own calls wrote its text first and the observations
            # came back after, so that text states a plan, not an answer — and
            # accepting it throws away every result the turn just fetched. A
            # model asked to reason step by step writes "3 * 60 = 180" while it
            # calls the calculator, and the loop used to stop there and return
            # the working as the answer.
            if tool_calls > 0 and not parsed.get("action") and not dispatched_calls_this_turn:
                # No action and we've used tools - model might be stating the answer
                response_text = response["text"].strip()
                # Check for answer-like patterns
                if any(phrase in response_text.lower() for phrase in ["the answer is", "the result is", "the sum is", "equals", "="]):
                    logger.info("Detected answer statement without 'Final Answer:' keyword")
                    if debug_trace is not None:
                        from ..debug.inspector import DebugIteration
                        debug_trace.iterations.append(DebugIteration(
                            iteration=iterations,
                            raw_prompt=prompt[:2000],
                            raw_response=response_text[:2000],
                            thought=parsed.get("thought", ""),
                            final_answer=response_text,
                            tokens_used=iter_tokens,
                            latency=time.time() - iter_start,
                            scratchpad_snapshot=scratchpad,
                        ))
                    return _build_response(response_text)

            # Execute action if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                # Repeat detection: the same call again, or the same tool
                # enough times with drifting inputs that it reads as a loop.
                check = guards.check_action(action, action_input)
                action_call_count = check.action_call_count
                # An exact repeat of a call that already succeeded is answered
                # from the record. A pure computation is idempotent, so running
                # it again returns what it returned before; the record supplies
                # that, and the run carries on with the step it was on.
                #
                # Treating the second identical call as proof the model was
                # stuck ended the run holding whatever the last observation
                # happened to be, which on a multi-step task is an intermediate
                # value. A model repeats a call because it restated its plan
                # before reading the observation; handing the result back lets
                # it finish the task instead.
                replay = None
                if check.is_exact_loop and not check.is_fuzzy_loop:
                    replay = guards.cached_result(check)
                if replay is not None:
                    logger.info(
                        "[Repeat] '%s' was already called with this input; "
                        "replaying the recorded result instead of ending the run",
                        action,
                    )
                    scratchpad += (
                        f"\nAction: {action}"
                        f"\nAction Input: {action_input}"
                        f"\nObservation: {replay}"
                    )
                    cur_observation = replay
                    nudge = guards.post_tool_nudge(
                        iterations, action_call_count, replay
                    )
                    if nudge:
                        scratchpad += f"\n{nudge}"
                    continue
                if check.is_loop:
                    logger.info(
                        f"[Loop detected] Repeated action '{action}' ({check.loop_type}) — "
                        f"the run stops offering this tool"
                    )
                    # Extract the last successful observation from scratchpad
                    partial = self._extract_partial_answer(scratchpad)
                    # What a tool returned is not an answer, whatever the tool
                    # was: a retrieved passage is source material, and a
                    # computed number is usually an intermediate one, so
                    # handing either back loses the question it belonged to.
                    # The model already has both in the scratchpad. Stop
                    # offering tools and spend one turn asking it to state the
                    # answer from what it has, before falling back to the
                    # progress itself.
                    if partial and not guards.force_text_answer:
                        logger.info(
                            "[Loop synthesis] '%s' is repeating; asking for an "
                            "answer stated from the observations so far",
                            action,
                        )
                        guards.force_text_answer = True
                        scratchpad += (
                            f"\nAction: {action}"
                            f"\nAction Input: {action_input}"
                            f"\nObservation: {NUDGE_HAVE_RESULTS}"
                        )
                        continue
                    if partial and self._is_context_retrieval_tool(action):
                        return self._stopped_outcome_response(
                            partial,
                            action=action,
                            reason="loop_detected",
                            scratchpad=scratchpad,
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            calls=guards.calls,
                            debug_trace=debug_trace,
                        )
                    if partial:
                        # The tool's results are not an answer either: the model
                        # never wrote one, so the run reports that it stopped and
                        # carries what the tools returned as partial progress.
                        return self._stopped_outcome_response(
                            sanitize_final_answer(partial) or partial,
                            action=action,
                            reason="loop_detected",
                            scratchpad=scratchpad,
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            calls=guards.calls,
                            debug_trace=debug_trace,
                        )
                    # No partial answer to fall back on — every attempt of this
                    # action failed or was denied, so simply nudging and
                    # re-offering the same tool just repeats the loop until
                    # max_iterations (the model keeps retrying the tool it was
                    # just told is already computed). Stop offering tools for
                    # the rest of this run so the model must respond in prose.
                    guards.force_text_answer = True
                    scratchpad += (
                        f"\nAction: {action}"
                        f"\nAction Input: {action_input}"
                        f"\nObservation: {NUDGE_ALREADY_COMPUTED}"
                    )
                    continue

                guards.record_action(check)

                # Check if tool is available (handle no-tool mode without raising)
                if not self.tools or action not in self.tools:
                    # The action names no tool the agent holds. With tools
                    # attached, say which ones are callable — telling a model
                    # that owns a calculator there are "no tools available"
                    # sends it off to do the work itself. With no tools at all,
                    # answering directly is the only option left.
                    observation = (
                        unknown_tool_observation(action, list(self.tools))
                        if self.tools
                        else NUDGE_NO_TOOLS
                    )
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {observation}"
                else:
                    # Execute tool inside tracing span
                    tool_start = time.time()
                    with start_tool_call(tool_name=action, tool_input=str(action_input)) as _tspan:
                        tool_result = self._execute_tool(action, action_input)
                        try:
                            _tspan.set_attribute(ToolAttrs.STATUS, "ok")
                        except Exception:
                            logger.debug("Failed to set tool span status", exc_info=True)
                    tool_elapsed = time.time() - tool_start
                    tool_calls += 1
                    guards.record_execution(
                        action,
                        arguments=action_input,
                        result=tool_result,
                        duration=tool_elapsed,
                        iteration=iterations,
                    )
                    # Keep the result against the exact call that produced it,
                    # so proposing that call again is answered from the record.
                    guards.record_pair_result(check, tool_result)
                    cur_observation = tool_result

                    # Metrics for tool execution
                    tool_labels = {"tool_name": action, "agent_name": self.name}
                    prom_metrics.tool_calls.inc(labels=tool_labels)
                    prom_metrics.tool_execution_time.observe(tool_elapsed, labels=tool_labels)
                    _slog.tool_event(action, "executed", latency=tool_elapsed)
                    _obs_log.tool_event("executed", tool=action, latency_ms=round(tool_elapsed * 1000, 1))

                    # Add observation to scratchpad
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    # Log the observation for debugging
                    logger.info(f"Tool result added to scratchpad: {tool_result[:100]}...")

                    if self._should_return_direct_calculator_result(task, action, action_input):
                        logger.info(
                            "Returning direct calculator result for simple arithmetic task"
                        )
                        return _build_response(
                            tool_result,
                            _tool_calls=tool_calls,
                            answer_source="direct_calculator_result",
                        )

                    # Result-based short-circuit: a model often re-derives a
                    # result it already has (e.g. "15^2" then "15*15", both
                    # 225) with slightly different inputs, so the exact-input
                    # loop guard never fires. A tool that reproduces its own
                    # output means the model is re-deriving rather than moving
                    # on, and re-offering it produces the same turn again.
                    if guards.result_is_repeat(action, tool_result):
                        # What the tool returned is not the answer, whatever
                        # the tool is: a retrieved passage is source material,
                        # and a repeated number is usually an intermediate one.
                        # The observation is in the scratchpad, so stop
                        # offering tools and give the model one turn to state
                        # the answer from it, falling back to the progress
                        # itself only when that turn produces nothing.
                        if not guards.force_text_answer:
                            logger.info(
                                "[Loop synthesis] Tool '%s' repeated a result; "
                                "asking for an answer stated from it",
                                action,
                            )
                            guards.force_text_answer = True
                            scratchpad += f"\n{NUDGE_HAVE_RESULTS}"
                            continue
                        logger.info(
                            "[Loop efficiency] Tool '%s' reproduced an identical "
                            "result; stopping the run",
                            action,
                        )
                        if self._is_context_retrieval_tool(action):
                            return self._stopped_outcome_response(
                                tool_result,
                                action=action,
                                reason="repeated_tool_result",
                                scratchpad=scratchpad,
                                iterations=iterations,
                                tool_calls=tool_calls,
                                tokens_used=tokens_used,
                                calls=guards.calls,
                                debug_trace=debug_trace,
                            )
                        # A tool that recomputes a number it already returned has
                        # not answered the question the caller asked — the model
                        # never wrote the answer up — so the run reports that it
                        # stopped and carries the result as partial progress.
                        return self._stopped_outcome_response(
                            sanitize_final_answer(tool_result) or tool_result,
                            action=action,
                            reason="repeated_tool_result",
                            scratchpad=scratchpad,
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            calls=guards.calls,
                            debug_trace=debug_trace,
                        )
                    guards.record_result(action, tool_result)

                    nudge = guards.post_tool_nudge(
                        iterations, action_call_count, tool_result
                    )
                    if nudge:
                        scratchpad += f"\n{nudge}"

            else:
                # A turn that produced neither an action nor an answer, but did
                # write out a call for a tool this agent holds, is the same
                # failure the answer path reports: the model is writing the call
                # instead of making it. Say so once, and on a second such turn
                # report the cause rather than grinding to the iteration cap
                # and reporting only that the cap was reached.
                written = find_written_tool_call(response["text"], self.tools)
                if written and guards.is_unmade_call(written, response["text"]):
                    if guards.note_written_call(written):
                        return self._written_tool_call_response(
                            guards.written_call,
                            response["text"],
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            tool_ran=guards.tool_ran(guards.written_call),
                            debug_trace=debug_trace,
                            calls=guards.calls,
                            scratchpad=scratchpad,
                        )
                    scratchpad += f"\nObservation: {NUDGE_NOT_USABLE}"
                # No action specified, prompt to continue
                scratchpad += "\nAction: (continue reasoning)"

            # Record debug iteration
            if debug_trace is not None:
                from ..debug.inspector import DebugIteration
                debug_trace.iterations.append(DebugIteration(
                    iteration=iterations,
                    raw_prompt=prompt[:2000],
                    raw_response=response["text"][:2000],
                    thought=parsed.get("thought", ""),
                    action=parsed.get("action"),
                    action_input=parsed.get("action_input"),
                    observation=cur_observation,
                    tokens_used=iter_tokens,
                    latency=time.time() - iter_start,
                    scratchpad_snapshot=scratchpad,
                ))

        # Max iterations reached. When every turn wrote its tool call out as
        # text and nothing ran, the cap is a symptom: report the cause instead.
        partial_answer = self._extract_partial_answer(scratchpad)
        if guards.written_call and not partial_answer:
            return self._written_tool_call_response(
                guards.written_call,
                "",
                iterations=iterations,
                tool_calls=tool_calls,
                tokens_used=tokens_used,
                tool_ran=guards.tool_ran(guards.written_call),
                debug_trace=debug_trace,
                calls=guards.calls,
                scratchpad=scratchpad,
            )
        # The run stopped without a final answer. Whatever the scratchpad holds
        # is a tool observation or a half-finished thought — source material, not
        # something the model wrote as its answer — so it is reported as progress
        # under ``partial_output`` and the outcome itself states what happened
        # and what to do about it.
        if partial_answer:
            partial_answer = sanitize_final_answer(partial_answer) or partial_answer
        detail = self._iteration_cap_detail(max_iterations, partial_answer)
        reason = (
            "max_iterations_partial" if partial_answer else "max_iterations_exhausted"
        )
        meta: dict[str, Any] = {
            "reason": reason,
            "error": detail,
            "tool_calling_strategy": self._tool_calling_strategy.name,
        }
        cap_partial = None
        if partial_answer:
            cap_partial = self._partial_result(
                scratchpad,
                text=partial_answer,
                calls=guards.calls,
                iterations=iterations,
                tool_calls=tool_calls,
            )
            meta["partial"] = True
            meta["partial_output"] = partial_answer
        logger.info(
            "outcome stopped: stop_reason=%s observations=%d",
            reason,
            len(cap_partial.observations) if cap_partial else 0,
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
            tool_calls=ToolCallList(list(guards.calls), total=tool_calls),
            tokens_used=tokens_used,
            metadata=meta,
            stop_reason=reason,
            partial=cap_partial,
        )

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
        scratchpad: str = "",
    ) -> AgentResponse:
        """Report a turn whose answer only describes the tool call it should have made.

        The model did not do the work, so this stays a failure and claims no
        result. When tools *had* run earlier in the run, what they returned is
        carried as partial progress rather than dropped — it is not an answer,
        but it is what the run has.
        """
        detail = self._written_tool_call_detail(tool_name, answer, tool_ran=tool_ran)
        logger.warning("Tool call was written as text, not made: %s", detail["message"])
        meta: dict[str, Any] = {
            "reason": "written_tool_call",
            "error": detail,
            "tool_calling_strategy": detail["tool_calling_strategy"],
        }
        partial = None
        if calls:
            candidate = self._partial_result(
                scratchpad,
                text=self._extract_partial_answer(scratchpad) or "",
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
        scratchpad: str,
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
            scratchpad: The run's scratchpad, read for the last thought.
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
            scratchpad,
            text=text,
            calls=calls,
            iterations=iterations,
            tool_calls=tool_calls,
        )
        meta: dict[str, Any] = {
            "reason": reason,
            "error": detail,
            "answer_source": reason,
            "repeated_action": action,
            "partial": True,
            "partial_output": text,
            "tool_calling_strategy": self._tool_calling_strategy.name,
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
        the run has no answer to report. What the scratchpad holds at that point
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
        if self._current_depth >= self.config.max_sub_agent_depth:
            logger.warning(f"Sub-agent depth limit reached ({self.config.max_sub_agent_depth})")
            return self._run_single_agent(task, context, **kwargs)

        self._current_depth += 1

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
                    self.sub_agent_manager.execute_parallel(subtasks)
                )
            elif strategy == RoutingStrategy.SEQUENTIAL_SUB_AGENTS:
                # Execute sequentially
                results = self.sub_agent_manager.execute_sequential(subtasks)
            elif strategy == RoutingStrategy.HYBRID:
                # Execute with hybrid approach
                results = self.sub_agent_manager.execute_hybrid(subtasks)
            else:
                # Default to sequential
                results = self.sub_agent_manager.execute_sequential(subtasks)

            # Synthesize results
            synthesis = self.sub_agent_manager.synthesize_results(
                results,
                task,
                strategy
            )

            # Calculate totals
            total_tokens = synthesis["metrics"]["total_tokens_used"]
            total_tool_calls = synthesis["metrics"]["total_tool_calls"]

            answered = synthesis["successful"] > 0
            return AgentResponse(
                output=sanitize_final_answer(synthesis["final_output"]) or synthesis["final_output"],
                success=answered,
                mode=AgentMode.SUB_AGENTS,
                iterations=len(subtasks),
                tool_calls=total_tool_calls,
                tokens_used=total_tokens,
                routing_decision=routing_decision,
                metadata={
                    "synthesis": synthesis,
                    "failed_subtasks": synthesis["failed"]
                },
                stop_reason="final_answer" if answered else "sub_agent_failed",
            )
        finally:
            self._current_depth -= 1
