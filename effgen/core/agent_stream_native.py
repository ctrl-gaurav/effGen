"""Streaming tool loop driven by a provider's native tool calling.

``Agent.stream()`` has always had two shapes: with no tools it streams the
model's answer directly, and with tools it runs the prompt-based ReAct scaffold
and hands back the parsed answer as one block. This module adds the third: for a
model whose adapter reports its streamed tool calls, the loop dispatches those
calls while the assistant's text streams through as it arrives.

The loop mirrors the blocking one in :mod:`effgen.core.agent_react` — same
prompt pieces, same dispatch, same repeat guards (they come from the shared
:class:`~effgen.core.agent_tool_loop.NativeToolLoop`), same failure vocabulary —
and reconstructs the :class:`~effgen.core.agent_response.AgentResponse` a
non-streamed turn would have returned, so a caller that needs the per-turn record
reads :attr:`Agent.last_stream_response` instead of running the task twice.

Two rules decide what may be shown, and when:

* **The answer-commit rule.** A turn's text is held back until the turn can no
  longer become a tool call. Once the adapter has recorded a tool call the text
  is delivered as a ``thought`` and never enters the answer; once a text delta
  has arrived with no call declared the turn is committed to answering and every
  later delta is passed through as it arrives.
* **Sanitize before emitting.** Only the settled prefix of
  :func:`~effgen.core.agent_runtime.sanitize_final_answer` applied to the text so
  far is emitted, so what reaches the screen is what the answer ends up being.
  The final word of the text so far is held back until more arrives, because it
  can still grow.

Together they give the invariant a consumer can rely on: for a stream that
answered, joining the ``answer`` events reproduces
``last_stream_response.output`` exactly. A stream that stopped at its iteration
cap, or one whose model wrote its call out as text, reports the typed outcome in
``output`` the same way ``run()`` does — that text is a ``status`` event, not an
answer delta.

This module imports nothing from ``agent.py``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from ..models.base import (
    GenerationConfig,
    clear_stream_tool_calls,
    clear_stream_usage,
    get_stream_tool_calls,
)
from .agent_config import AgentMode
from .agent_response import AgentResponse, PartialResult, StreamEvent
from .agent_runtime import (
    NUDGE_ALREADY_COMPUTED,
    NUDGE_CONTINUE,
    NUDGE_HAVE_RESULTS,
    NUDGE_NOT_USABLE,
    find_written_tool_call,
    resolve_output_budget,
    sanitize_final_answer,
    unknown_tool_observation,
)
from .agent_tool_loop import NativeToolLoop
from .tool_call_record import ToolCallList

logger = logging.getLogger(__name__)


#: Labels that make sanitizing promote what follows them and discard what came
#: before. Once the label is resolved the text after it grows monotonically, so
#: only the label itself has to be held back.
_ANSWER_LABELS = ("final answer:", "answer:")

#: Tool-call syntax a model sometimes writes out instead of calling. Sanitizing
#: removes these constructs from the middle of the text, so nothing can be
#: emitted from a turn that contains one — what survives is not an extension of
#: what came before it.
#:
#: These are **openers**, and the list is not a complete account of how local
#: models emit calls. Measured on 2026-08-13 over the five families:
#: Qwen2.5 emits ``<tool_call>{...}</tool_call>``, which is here; Llama 3.2
#: emits a **bare JSON object** — ``{"name": ..., "parameters": {...}}`` — with
#: no opener at all; Mistral-Small-24B, Phi-3.5-mini and gemma-2-2b report
#: ``tool_call_support() == "none"`` and never take a template tool path. A
#: suffix scan cannot hold back the Llama shape without holding back every
#: answer that starts with ``{``, which is what structured output looks like.
#: That is why the streamed tool loop is still gated on ``"api"`` rather than
#: widened to template models: widening it would need a way to tell an
#: opener-emitting family from a bare-JSON one, and streamed call recording on
#: the local engines, neither of which exists.
#:
#: The tag openers cover the XML dialect too — a call written as nested tags
#: rather than as JSON (``<function=x><parameter=y>…``) opens with one of these,
#: in either spelling of the name, so such a turn is held back and delivered
#: sanitized rather than streamed with its scaffolding on screen.
_CALL_CONSTRUCTS = (
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call>",
    "<function=",
    "<function ",
    "<function_call",
    "<invoke",
    "[tool_calls]",
    "<|python_tag|>",
)

_REWRITE_MARKERS = _ANSWER_LABELS + _CALL_CONSTRUCTS


def _contains(text: str, markers: tuple[str, ...]) -> bool:
    """True when *text* holds any of *markers*."""
    low = text.lower()
    return any(marker in low for marker in markers)


def _ends_mid_marker(text: str) -> bool:
    """True when *text* ends with something that could still become a marker."""
    low = text.lower()
    for marker in _REWRITE_MARKERS:
        for length in range(min(len(marker) - 1, len(low)), 0, -1):
            if low.endswith(marker[:length]):
                return True
    return False


class _AnswerStream:
    """Turns raw model text into sanitized answer deltas, in order.

    Fed the text as it arrives, it hands back only the part of the sanitized
    answer that has settled. Two things are withheld: the trailing word, because
    sanitizing collapses runs of spaces and strips the tail, and everything at
    all while the text could still turn into a construct that rewrites what came
    before it. :meth:`flush` releases the remainder once the model has stopped.

    ``emitted`` is always exactly what the consumer received, so a caller can
    use it as the answer and know the two agree.
    """

    def __init__(self) -> None:
        self.raw = ""
        self.emitted = ""
        #: Set when sanitizing rewrote text that had already been delivered.
        #: Nothing further is emitted, since a delta cannot be taken back;
        #: ``emitted`` stays the answer of record.
        self.diverged = False

    def push(self, text: str) -> str:
        """Add *text* to the answer and return the next delta (``""`` if none)."""
        self.raw += text
        return self._advance(hold_tail=True)

    def flush(self) -> str:
        """Release whatever is left once no more text is coming."""
        return self._advance(hold_tail=False)

    def _advance(self, hold_tail: bool) -> str:
        if self.diverged:
            return ""
        if hold_tail and _ends_mid_marker(self.raw):
            # The tail could still be the first half of a marker — an answer
            # label being typed out, say. Emitting it now would put half a
            # label on screen and then have to take it back.
            return ""
        if hold_tail and _contains(self.raw, _CALL_CONSTRUCTS):
            # The turn is writing tool-call syntax into its text. Sanitizing
            # cuts that construct out of the middle, so nothing here extends
            # what came before it: hold the whole turn and deliver the cleaned
            # text once the model stops, which is what a non-streamed turn does.
            return ""
        safe = sanitize_final_answer(self.raw) or ""
        if hold_tail and _contains(safe, _ANSWER_LABELS):
            # Sanitizing has not resolved the label yet: a bare "Final Answer:"
            # with nothing after it is still the label, not the answer. Once the
            # answer follows, what survives is the text after the label — which
            # grows monotonically from there.
            return ""
        if not safe.startswith(self.emitted):
            # Sanitizing changed text already on screen — a construct that
            # rewrites its predecessor arriving after the answer was under way.
            # It cannot be withdrawn, so stop advancing and keep what the
            # consumer already has.
            logger.debug("Streamed answer diverged from its sanitized form")
            self.diverged = True
            return ""
        settled = safe
        if hold_tail:
            stripped = safe.rstrip()
            cut = max(
                stripped.rfind(" "), stripped.rfind("\n"), stripped.rfind("\t")
            ) + 1
            settled = safe[:cut] if cut > 0 else ""
        if len(settled) <= len(self.emitted):
            return ""
        delta = settled[len(self.emitted):]
        self.emitted = settled
        return delta


class AgentNativeStreamMixin:
    """The streamed tool loop, and the record it leaves behind."""

    if TYPE_CHECKING:
        # Contributed by the sibling mixins this one is combined with on
        # :class:`~effgen.core.agent.Agent`. Declared for the type checker only
        # — at run time they arrive through the MRO, and these statements do
        # not execute.
        def _extract_partial_answer(self, scratchpad: str) -> str | None: ...

        def _partial_result(
            self,
            scratchpad: str,
            *,
            text: str,
            calls: Any = (),
            iterations: int = 0,
            tool_calls: int = 0,
        ) -> PartialResult: ...

        def _is_context_retrieval_tool(self, action: str) -> bool: ...

        def _answer_shape_instruction(self) -> str: ...

        def _native_tool_prompt(
            self, task: str, scratchpad: str, conversation_history: str,
            previous_actions: list[tuple[str, str]],
        ) -> str: ...

        @staticmethod
        def _compose_closing(answer_shape: str, closing: str) -> str: ...

        def _repeated_tool_detail(
            self,
            action: str | None,
            reason: str,
            *,
            retrieval: bool = True,
            answer: str | None = None,
        ) -> dict[str, Any]: ...

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    def _can_stream_native_tools(self) -> bool:
        """True when this agent's next streamed turn can dispatch native calls.

        Every condition has to hold: the agent carries ordinary tools, no
        provider-side native tool is attached (those have their own run paths),
        the strategy asks for native calling, the definitions travel through the
        provider's tool-calling API rather than a chat template, the adapter
        records the calls it streams, and no custom system-prompt template owns
        the prompt. Anything else keeps today's path.
        """
        if not self.tools or self.model is None:
            return False
        if self.config.system_prompt_template:
            return False
        try:
            if self._has_native_tools() or self._has_gemini_native_tools():
                return False
        except Exception:  # noqa: BLE001 - an unreadable tool set is not eligible
            logger.debug("Native-tool probe failed", exc_info=True)
            return False
        if self._tool_calling_strategy.name not in ("native", "hybrid"):
            return False
        if self._model_tool_call_support() != "api":
            return False
        probe = getattr(self.model, "streams_tool_calls", None)
        try:
            return bool(probe and probe())
        except Exception:  # noqa: BLE001 - a capability probe never breaks a run
            logger.debug("streams_tool_calls probe failed", exc_info=True)
            return False

    @property
    def last_stream_response(self) -> AgentResponse | None:
        """The record of the most recent streamed tool loop, or ``None``.

        Set once the iterator is exhausted, and shaped exactly like the
        :class:`~effgen.core.agent_response.AgentResponse` the same task would
        have produced through :meth:`run` — ``output``, ``success``,
        ``iterations``, ``tool_calls``, ``tokens_used``, ``execution_time`` and
        the ``reason`` / ``error`` / ``partial`` metadata — so a caller that
        renders a turn can report it without running the task again. It is
        ``None`` after a stream that did not take the native tool path (a
        tool-free stream, or a model whose adapter does not record streamed
        calls).
        """
        return getattr(self, "_last_stream_response", None)

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    def _stream_native_tools(
        self,
        task: str,
        on_thought: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_observation: Callable[[str], None] | None = None,
        on_answer: Callable[[str], None] | None = None,
        include_events: bool = False,
        _usage_acc: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "Iterator[str] | Iterator[StreamEvent]":
        """Stream one task through the model's native tool calling.

        Yields answer-text deltas (or typed events with *include_events*) while
        dispatching each tool call the model makes, and leaves the turn's record
        in :attr:`last_stream_response`.
        """
        started = time.perf_counter()
        # An explicit ``max_iterations=None`` — what an optional flag forwards when
        # the user did not set it — must fall back to the configured cap rather
        # than reach the loop comparison as None.
        _requested_iterations = kwargs.get("max_iterations")
        max_iterations: int = (
            self.config.max_iterations
            if _requested_iterations is None
            else int(_requested_iterations)
        )
        guards = NativeToolLoop(self.tools, nudge_cap=self.config.max_iterations)
        answer = _AnswerStream()
        scratchpad = ""
        iterations = 0
        tool_calls = 0
        committed = False  # has any answer delta reached the consumer this run

        conversation_history = self._format_conversation_history()
        # The same sampling settings the blocking loop sends: a value configured
        # on the agent, or pinned per call, has to reach the model whichever path
        # the turn takes. No stop sequences — there is no text scaffold to trim
        # here, and the reasoning families reject them.
        # A cap pinned on the call wins, then one configured on the agent — the
        # same order ``run()`` resolves, so a turn does not get a different
        # output budget for taking the streamed path.
        max_tokens = resolve_output_budget(
            kwargs.get("max_tokens"), self.config.max_tokens, self.model
        )
        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            seed=kwargs.get("seed", self.config.seed),
            presence_penalty=kwargs.get(
                "presence_penalty", self.config.presence_penalty
            ),
            frequency_penalty=kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            ),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.config.repetition_penalty
            ),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )

        def _emit(text: str) -> "StreamEvent | str":
            return StreamEvent(kind="answer", text=text) if include_events else text

        while iterations < max_iterations:
            iterations += 1
            prompt = self._native_tool_prompt(
                task, scratchpad, conversation_history, guards.previous_actions
            )
            gen_kwargs: dict[str, Any] = {}
            # Once the guards stop offering tools the prompt stays the native
            # one rather than switching to the ReAct scaffold the blocking loop
            # falls back to: that scaffold's ``Thought:``/``Action:`` shape is
            # written to the screen here, which is exactly what this path exists
            # to avoid. Withholding the definitions is what forces the answer.
            if not guards.tools_suppressed():
                tool_defs = self._tool_calling_strategy.format_tools_for_prompt(
                    list(self.tools.values())
                )
                if isinstance(tool_defs, list):
                    gen_kwargs["tools"] = tool_defs

            clear_stream_usage(self.model)
            clear_stream_tool_calls(self.model)
            raw = ""
            turn_committed = False
            failure: Exception | None = None
            stream_iter: Any = None
            try:
                # Opening the stream belongs inside the guard: the budget gate
                # wrapping ``generate_stream`` refuses synchronously, before the
                # iterator exists, so a refusal raised here would otherwise
                # leave the loop with no record and no fallback.
                stream_iter = self.model.generate_stream(
                    prompt, config=gen_config, **gen_kwargs
                )
                for token in stream_iter:
                    if not token:
                        continue
                    raw += token
                    if not turn_committed:
                        if get_stream_tool_calls(self.model):
                            # The turn is making a call; its text is reasoning.
                            continue
                        turn_committed = True
                        delta = answer.push(raw)
                    else:
                        delta = answer.push(token)
                    if delta:
                        committed = True
                        yield _emit(delta)
            except Exception as exc:  # noqa: BLE001 - handled below
                logger.debug("Native streaming turn failed", exc_info=True)
                failure = exc
            finally:
                close_stream = getattr(stream_iter, "close", None)
                if close_stream is not None:
                    close_stream()

            if failure is not None:
                if committed:
                    # Part of the answer is already on screen; the contract says
                    # a mid-stream failure is raised, not disguised as an end.
                    self._last_stream_response = self._native_stream_failure(
                        task, failure, iterations=iterations, tool_calls=tool_calls,
                        usage_acc=_usage_acc, started=started,
                    )
                    raise failure
                # Nothing has been delivered yet, so the whole task can be run
                # again on the non-streaming path, which retries provider-side
                # rejections the stream has no retry for. The answer then
                # arrives in one block — today's behaviour.
                yield from self._native_stream_fallback(
                    task, failure, include_events=include_events,
                    on_answer=on_answer, usage_acc=_usage_acc, started=started,
                    **kwargs,
                )
                return

            if _usage_acc is not None:
                self._fold_stream_usage(_usage_acc, prompt, raw)

            calls = get_stream_tool_calls(self.model)

            if not calls:
                # ---- the turn is the answer -------------------------------
                delta = answer.flush()
                if delta:
                    committed = True
                    yield _emit(delta)
                if not turn_committed or not raw.strip():
                    scratchpad += "\nThought: "
                    scratchpad += "\nAction: (continue reasoning)"
                    continue
                text = answer.emitted
                if not text.strip():
                    # The turn produced only scaffolding. When what leaked is a
                    # call for a tool this agent holds, the model is writing the
                    # call instead of making it: say so once, then report it.
                    written = find_written_tool_call(raw, self.tools)
                    if written and guards.is_unmade_call(written, raw):
                        if guards.note_written_call(written):
                            self._last_stream_response = self._native_written_call(
                                task, guards, raw, iterations=iterations,
                                tool_calls=tool_calls, usage_acc=_usage_acc,
                                started=started, scratchpad=scratchpad,
                            )
                            yield from self._yield_outcome(
                                self._last_stream_response, include_events
                            )
                            return
                    scratchpad += f"\nObservation: {NUDGE_NOT_USABLE}"
                    continue
                written = find_written_tool_call(
                    text, self.tools
                ) or find_written_tool_call(raw, self.tools)
                if written and guards.is_unmade_call(written, raw):
                    self._last_stream_response = self._native_written_call(
                        task, guards, raw, iterations=iterations,
                        tool_calls=tool_calls, usage_acc=_usage_acc,
                        started=started, written=written, scratchpad=scratchpad,
                    )
                    yield from self._yield_outcome(
                        self._last_stream_response, include_events
                    )
                    return
                if on_answer:
                    on_answer(text)
                self.short_term_memory.add_user_message(task)
                self.short_term_memory.add_assistant_message(text)
                self._last_stream_response = self._native_stream_response(
                    task, output=text, success=True, iterations=iterations,
                    tool_calls=tool_calls, usage_acc=_usage_acc, started=started,
                    meta={"reason": "final_answer"},
                )
                return

            # ---- the turn made tool calls ---------------------------------
            thought = "" if turn_committed else raw.strip()
            if thought:
                if on_thought:
                    on_thought(thought)
                if include_events:
                    yield StreamEvent(kind="thought", text=thought)
            scratchpad += f"\nThought: {thought}"

            if len(calls) > 1:
                # Several calls in one turn are dispatched as a batch and
                # answered together, matching the blocking loop.
                for call in calls:
                    name, args = _call_parts(call)
                    if on_tool_call:
                        on_tool_call(name, json.dumps(args))
                    if include_events:
                        yield StreamEvent(
                            kind="tool_call", tool=name, tool_input=json.dumps(args)
                        )
                    if name in self.tools:
                        observation = self._execute_tool(name, json.dumps(args))
                        tool_calls += 1
                        guards.record_execution(
                            name, arguments=args, result=observation,
                            iteration=iterations,
                        )
                        scratchpad += (
                            f"\nAction: {name}"
                            f"\nAction Input: {json.dumps(args)}"
                            f"\nObservation: {observation}"
                        )
                    else:
                        observation = unknown_tool_observation(name, list(self.tools))
                    if on_observation:
                        on_observation(observation)
                    if include_events:
                        yield StreamEvent(
                            kind="observation", tool=name, text=observation
                        )
                scratchpad += f"\n{NUDGE_CONTINUE}"
                guards.note_batch_run()
                continue

            parsed = self._tool_call_result_to_dict(
                self._parse_native_tool_calls(calls)
            )
            action = parsed.get("action")
            action_input = parsed.get("action_input")
            if not action or action_input is None:
                scratchpad += "\nAction: (continue reasoning)"
                continue

            if on_tool_call:
                on_tool_call(action, str(action_input))
            if include_events:
                yield StreamEvent(
                    kind="tool_call", tool=action, tool_input=str(action_input)
                )

            check = guards.check_action(action, action_input)
            # An exact repeat of a call that already succeeded is answered from
            # the record and the run carries on, the same as on the blocking
            # path: a pure computation is idempotent, so running it again
            # returns what it returned before.
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
                if on_observation:
                    on_observation(replay)
                if include_events:
                    yield StreamEvent(kind="observation", tool=action, text=replay)
                nudge = guards.post_tool_nudge(
                    iterations, check.action_call_count, replay
                )
                if nudge:
                    scratchpad += f"\n{nudge}"
                continue
            if check.is_loop:
                logger.info(
                    "[Loop detected] Repeated action '%s' (%s) while streaming",
                    action, check.loop_type,
                )
                partial = self._extract_partial_answer(scratchpad)
                # What a tool returned is not an answer, whatever the tool was.
                # Stop offering tools and spend one turn asking the model to
                # state the answer from the observations it already has, before
                # falling back to the progress itself.
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
                if partial:
                    retrieval = self._is_context_retrieval_tool(action)
                    yield from self._finish_stopped(
                        task,
                        partial if retrieval else (
                            sanitize_final_answer(partial) or partial
                        ),
                        guards, scratchpad=scratchpad, action=action,
                        reason="loop_detected", iterations=iterations,
                        tool_calls=tool_calls, usage_acc=_usage_acc,
                        started=started, include_events=include_events,
                    )
                    return
                guards.force_text_answer = True
                scratchpad += (
                    f"\nAction: {action}"
                    f"\nAction Input: {action_input}"
                    f"\nObservation: {NUDGE_ALREADY_COMPUTED}"
                )
                continue

            guards.record_action(check)

            if action not in self.tools:
                observation = unknown_tool_observation(action, list(self.tools))
                scratchpad += (
                    f"\nAction: {action}"
                    f"\nAction Input: {action_input}"
                    f"\nObservation: {observation}"
                )
                if on_observation:
                    on_observation(observation)
                if include_events:
                    yield StreamEvent(
                        kind="observation", tool=action, text=observation
                    )
                continue

            observation = self._execute_tool(action, action_input)
            tool_calls += 1
            guards.record_execution(
                action, arguments=action_input, result=observation,
                iteration=iterations,
            )
            # Keep the result against the exact call that produced it, so
            # proposing that call again is answered from the record.
            guards.record_pair_result(check, observation)
            scratchpad += (
                f"\nAction: {action}"
                f"\nAction Input: {action_input}"
                f"\nObservation: {observation}"
            )
            if on_observation:
                on_observation(observation)
            if include_events:
                yield StreamEvent(kind="observation", tool=action, text=observation)

            if self._should_return_direct_calculator_result(task, action, action_input):
                yield from self._finish_with_recovered(
                    task, observation, guards, iterations=iterations,
                    tool_calls=tool_calls, usage_acc=_usage_acc, started=started,
                    include_events=include_events, on_answer=on_answer,
                    answer_stream=answer, answer_source="direct_calculator_result",
                    scratchpad=scratchpad,
                )
                return

            if guards.result_is_repeat(action, observation):
                # A repeated result means the model is re-deriving, not that the
                # task is answered — so give it one turn to state the answer
                # from the observation before falling back to the observation.
                if not guards.force_text_answer:
                    logger.info(
                        "[Loop synthesis] Tool '%s' repeated a result; "
                        "asking for an answer stated from it",
                        action,
                    )
                    guards.force_text_answer = True
                    scratchpad += f"\n{NUDGE_HAVE_RESULTS}"
                    continue
                retrieval = self._is_context_retrieval_tool(action)
                yield from self._finish_stopped(
                    task,
                    observation if retrieval else (
                        sanitize_final_answer(observation) or observation
                    ),
                    guards, scratchpad=scratchpad, action=action,
                    reason="repeated_tool_result", iterations=iterations,
                    tool_calls=tool_calls, usage_acc=_usage_acc, started=started,
                    include_events=include_events,
                )
                return
            guards.record_result(action, observation)

            nudge = guards.post_tool_nudge(
                iterations, check.action_call_count, observation
            )
            if nudge:
                scratchpad += f"\n{nudge}"

        # ---- the iteration cap ------------------------------------------
        partial_answer = self._extract_partial_answer(scratchpad)
        if guards.written_call and not partial_answer:
            self._last_stream_response = self._native_written_call(
                task, guards, "", iterations=iterations, tool_calls=tool_calls,
                usage_acc=_usage_acc, started=started, scratchpad=scratchpad,
            )
            yield from self._yield_outcome(self._last_stream_response, include_events)
            return
        if partial_answer:
            partial_answer = sanitize_final_answer(partial_answer) or partial_answer
        detail = self._iteration_cap_detail(max_iterations, partial_answer)
        reason = (
            "max_iterations_partial" if partial_answer else "max_iterations_exhausted"
        )
        meta: dict[str, Any] = {"reason": reason, "error": detail}
        cap_partial = None
        if partial_answer:
            cap_partial = self._partial_result(
                scratchpad, text=partial_answer, calls=guards.calls,
                iterations=iterations, tool_calls=tool_calls,
            )
            meta["partial"] = True
            meta["partial_output"] = partial_answer
        logger.info(
            "outcome stopped: stop_reason=%s observations=%d",
            reason, len(cap_partial.observations) if cap_partial else 0,
        )
        self._last_stream_response = self._native_stream_response(
            task, output=detail["message"], success=False, iterations=iterations,
            tool_calls=tool_calls, usage_acc=_usage_acc, started=started, meta=meta,
            calls=guards.calls, partial=cap_partial,
        )
        yield from self._yield_outcome(self._last_stream_response, include_events)

    # ------------------------------------------------------------------
    # Terminal shapes
    # ------------------------------------------------------------------
    def _yield_outcome(
        self, response: AgentResponse, include_events: bool
    ) -> "Iterator[str] | Iterator[StreamEvent]":
        """Deliver a terminal notice that is not an answer.

        The iteration cap and a written-out call have no answer to stream, so
        the typed outcome travels as a ``status`` event (plain text without
        events), exactly as the ReAct streaming path already reports its step
        limit.
        """
        if include_events:
            yield StreamEvent(kind="status", text=response.output)
        else:
            yield response.output

    def _finish_with_recovered(
        self,
        task: str,
        text: str,
        guards: NativeToolLoop,
        *,
        iterations: int,
        tool_calls: int,
        usage_acc: dict[str, Any] | None,
        started: float,
        include_events: bool,
        on_answer: Callable[[str], None] | None,
        answer_stream: _AnswerStream,
        answer_source: str,
        scratchpad: str = "",
        **extra_meta: Any,
    ) -> "Iterator[str] | Iterator[StreamEvent]":
        """End the run on an answer the loop recovered rather than streamed.

        A repeated call or a tool that reproduced its own result ends the run
        with an observation as the answer; it was never streamed, so it is
        emitted here as answer deltas and folded into the same buffer, keeping
        the join invariant true.
        """
        output = sanitize_final_answer(text) or text
        written = find_written_tool_call(output, self.tools) or find_written_tool_call(
            text, self.tools
        )
        if written and guards.is_unmade_call(written, text):
            self._last_stream_response = self._native_written_call(
                task, guards, text, iterations=iterations, tool_calls=tool_calls,
                usage_acc=usage_acc, started=started, written=written,
                scratchpad=scratchpad,
            )
            yield from self._yield_outcome(self._last_stream_response, include_events)
            return
        delta = answer_stream.push(output)
        delta += answer_stream.flush()
        if delta:
            yield StreamEvent(kind="answer", text=delta) if include_events else delta
        final = answer_stream.emitted
        if on_answer:
            on_answer(final)
        if final:
            self.short_term_memory.add_user_message(task)
            self.short_term_memory.add_assistant_message(final)
        meta: dict[str, Any] = {"reason": "final_answer", "answer_source": answer_source}
        meta.update(extra_meta)
        self._last_stream_response = self._native_stream_response(
            task, output=final, success=True, iterations=iterations,
            tool_calls=tool_calls, usage_acc=usage_acc, started=started, meta=meta,
            calls=guards.calls,
        )

    def _finish_stopped(
        self,
        task: str,
        text: str,
        guards: NativeToolLoop,
        *,
        scratchpad: str,
        action: str | None,
        reason: str,
        iterations: int,
        tool_calls: int,
        usage_acc: dict[str, Any] | None,
        started: float,
        include_events: bool,
    ) -> "Iterator[str] | Iterator[StreamEvent]":
        """End a streamed run the loop stopped before the model wrote an answer.

        The blocking loop's counterpart is
        :meth:`~effgen.core.agent_react.AgentReActMixin._stopped_outcome_response`,
        and the two report the same thing: the outcome statement in ``output``,
        the tool results under ``partial``. The statement is not an answer, so it
        is not streamed as answer deltas — it travels as the ``status`` event the
        iteration cap already uses, and the stream ends.
        """
        retrieval = self._is_context_retrieval_tool(action) if action else False
        detail = self._repeated_tool_detail(action, reason, retrieval=retrieval)
        partial = self._partial_result(
            scratchpad, text=text, calls=guards.calls,
            iterations=iterations, tool_calls=tool_calls,
        )
        meta: dict[str, Any] = {
            "reason": reason,
            "error": detail,
            "answer_source": reason,
            "repeated_action": action,
            "partial": True,
            "partial_output": text,
        }
        logger.info(
            "outcome stopped: stop_reason=%s tool=%s category=%s observations=%d",
            reason, action or "-",
            "INFORMATION_RETRIEVAL" if retrieval else "COMPUTATION",
            len(partial.observations),
        )
        self._last_stream_response = self._native_stream_response(
            task, output=detail["message"], success=False, iterations=iterations,
            tool_calls=tool_calls, usage_acc=usage_acc, started=started, meta=meta,
            calls=guards.calls, partial=partial,
        )
        yield from self._yield_outcome(self._last_stream_response, include_events)

    def _native_stream_fallback(
        self,
        task: str,
        failure: Exception,
        *,
        include_events: bool,
        on_answer: Callable[[str], None] | None,
        usage_acc: dict[str, Any] | None,
        started: float,
        **kwargs: Any,
    ) -> "Iterator[str] | Iterator[StreamEvent]":
        """Re-run *task* without streaming after a failure that reached nobody.

        A streamed call has no retry layer, while the blocking path has two, so
        a provider-side rejection that ``run()`` would have retried away must not
        end the turn. Nothing has been delivered at this point, so the task is
        run once more and its answer is emitted as deltas.
        """
        logger.info(
            "Streaming tool turn failed before any output; retrying without "
            "streaming (%s)", type(failure).__name__,
        )
        from .agent_streaming import _chunk_answer_text

        run_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"temperature", "max_tokens", "top_p", "seed", "max_iterations",
                     "reasoning_effort"}
        }
        response = self.run(task, **run_kwargs)
        if usage_acc is not None:
            usage_acc["total_tokens"] = (
                usage_acc.get("total_tokens") or 0
            ) + int(response.tokens_used or 0)
            cost = (response.metadata or {}).get("cost_usd")
            if cost is not None:
                usage_acc["cost_usd"] = (usage_acc.get("cost_usd") or 0.0) + float(cost)
            usage_acc["model_calls"] = usage_acc.get("model_calls", 0) + 1
        response.execution_time = time.perf_counter() - started
        (response.metadata or {}).setdefault("stream_fallback", True)
        self._last_stream_response = response
        if response.success and response.output:
            if on_answer:
                on_answer(response.output)
            for chunk in _chunk_answer_text(response.output):
                yield StreamEvent(kind="answer", text=chunk) if include_events else chunk
            return
        yield from self._yield_outcome(response, include_events)

    def _native_written_call(
        self,
        task: str,
        guards: NativeToolLoop,
        text: str,
        *,
        iterations: int,
        tool_calls: int,
        usage_acc: dict[str, Any] | None,
        started: float,
        written: str | None = None,
        scratchpad: str = "",
    ) -> AgentResponse:
        """Build the record for a turn that wrote its tool call out as text.

        The model did not do the work, so this stays a failure. When tools had
        run earlier in the run, what they returned travels as partial progress
        rather than being dropped.
        """
        name = written or guards.written_call or ""
        detail = self._written_tool_call_detail(
            name, text, tool_ran=guards.tool_ran(name)
        )
        logger.warning("Tool call was written as text, not made: %s", detail["message"])
        meta: dict[str, Any] = {"reason": "written_tool_call", "error": detail}
        partial = None
        if guards.calls:
            candidate = self._partial_result(
                scratchpad, text=self._extract_partial_answer(scratchpad) or "",
                calls=guards.calls, iterations=iterations, tool_calls=tool_calls,
            )
            if candidate.text.strip():
                partial = candidate
                meta["partial"] = True
                meta["partial_output"] = partial.text
        logger.info(
            "outcome failed: stop_reason=written_tool_call tool=%s observations=%d",
            name or "-", len(partial.observations) if partial else 0,
        )
        return self._native_stream_response(
            task, output=detail["message"], success=False, iterations=iterations,
            tool_calls=tool_calls, usage_acc=usage_acc, started=started,
            meta=meta, calls=guards.calls, partial=partial,
        )

    def _native_stream_failure(
        self,
        task: str,
        failure: Exception,
        *,
        iterations: int,
        tool_calls: int,
        usage_acc: dict[str, Any] | None,
        started: float,
    ) -> AgentResponse:
        """Build the record for a stream that failed after it began answering."""
        detail = self._build_error_detail(failure, self.model)
        return self._native_stream_response(
            task, output=str(detail.get("message") or failure), success=False,
            iterations=iterations, tool_calls=tool_calls, usage_acc=usage_acc,
            started=started, meta={"reason": "generation_failed", "error": detail},
        )

    def _native_stream_response(
        self,
        task: str,
        *,
        output: str,
        success: bool,
        iterations: int,
        tool_calls: int,
        usage_acc: dict[str, Any] | None,
        started: float,
        meta: dict[str, Any],
        calls: Any = (),
        partial: PartialResult | None = None,
    ) -> AgentResponse:
        """Assemble the :class:`AgentResponse` a streamed turn reconstructs."""
        usage = dict(usage_acc or {})
        metadata: dict[str, Any] = {
            "tool_calling_strategy": self._tool_calling_strategy.name,
            "streamed": True,
        }
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
            metadata[key] = usage.get(key)
        metadata["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        metadata.update(meta)
        return AgentResponse(
            output=output,
            success=success,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=ToolCallList(list(calls), total=tool_calls),
            tokens_used=int(usage.get("total_tokens") or 0),
            execution_time=time.perf_counter() - started,
            metadata=metadata,
            task=task,
            model=getattr(self.model, "model_name", None) or self.model_name,
            provider=self._model_provider(self.model),
            partial=partial,
        )


def _call_parts(call: dict[str, Any]) -> tuple[str, Any]:
    """Return ``(name, arguments)`` for one recorded tool call.

    Arguments arrive as the JSON string the model streamed; text that is not
    JSON is passed through under ``__raw_input__`` so the tool layer reports it
    rather than the loop dropping the call.
    """
    fn = call.get("function", call)
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {"__raw_input__": args}
    if not isinstance(args, dict):
        args = {}
    return name, args
