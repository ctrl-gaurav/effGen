"""Streaming run orchestration for :class:`effgen.core.agent.Agent`.

Extracted from ``agent.py`` without behaviour change: the incremental
:meth:`Agent.stream` entry point, the no-tool direct-stream and tool-loop
stream implementations, the per-run usage folding, and the
:attr:`Agent.last_stream_usage` accessor. Mixed into :class:`Agent` alongside
the generation, ReAct, and runtime mixins. The dataclasses come from the config
and response leaves and the sanitizer from ``agent_runtime``; this module
imports nothing from ``agent.py``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from ..models.base import GenerationConfig
from .agent_config import AgentMode
from .agent_response import StreamEvent
from .agent_runtime import (
    NUDGE_NO_TOOLS,
    resolve_output_budget,
    sanitize_final_answer,
    unknown_tool_observation,
)
from .agent_tool_loop import NativeToolLoop
from .result_relay import unrelayed_result
from .tool_call_record import ToolCall, truncate_result

if TYPE_CHECKING:
    from .agent_config import AgentConfig
    from .messages import Message

logger = logging.getLogger(__name__)


def _chunk_answer_text(answer: str) -> Iterator[str]:
    """Yield *answer* as word-sized deltas whose concatenation is ``answer``.

    Each chunk is a run of non-whitespace plus its trailing whitespace, so
    ``"".join(_chunk_answer_text(s)) == s`` exactly (``sanitize_final_answer``
    has already stripped leading/trailing whitespace). This gives a streaming
    feel for an answer that was produced behind ReAct scaffolding without
    re-emitting any of that scaffolding.
    """
    import re as _re

    chunks = _re.findall(r"\S+\s*", answer)
    if not chunks:  # whitespace-only (shouldn't happen post-sanitize)
        if answer:
            yield answer
        return
    yield from chunks


class AgentStreamingMixin:
    """Streaming-run methods for :class:`Agent`."""

    if TYPE_CHECKING:
        # Supplied by the class this is mixed into. Declared so a reader of the
        # streaming code can see where `self.config` comes from, and so the
        # type checker does not count each use as an undefined attribute. Only
        # the attributes this module actually reads are declared; the other
        # mixins declare their own.
        config: AgentConfig
        tools: dict[str, Any]

        def _tool_contract(self) -> str: ...

        def _citation_prompt_state(self) -> tuple[bool, int]: ...

        def _answer_shape_instruction(self) -> str: ...

        def _context_answer_instruction(
            self,
            previous_actions: list[tuple[str, str]],
            *,
            cite_sources: bool = False,
            numbered_passages: int = 0,
        ) -> str: ...

    def _fold_stream_usage(
        self, acc: dict[str, Any], prompt_text: str, completion_text: str
    ) -> None:
        """Fold the model call that just finished streaming into *acc*.

        Reads the usage the adapter recorded for that call; when the backend
        reported none (a local engine, or a provider that omits usage from its
        stream) the counts are estimated from the prompt and completion text and
        the accumulator is marked estimated. Summing across calls means a
        tool-using stream reports the whole run, not just its last model call.
        """
        from ..models.base import (
            clear_stream_usage,
            estimate_stream_usage,
            get_stream_usage,
        )

        usage = get_stream_usage(self.model)
        clear_stream_usage(self.model)
        if usage is None:
            usage = estimate_stream_usage(self.model, prompt_text, completion_text)
        if usage.get("estimated"):
            acc["estimated"] = True
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if value is not None:
                acc[key] = (acc.get(key) or 0) + int(value)
        cost = usage.get("cost_usd")
        if cost is not None:
            acc["cost_usd"] = (acc.get("cost_usd") or 0.0) + float(cost)
        acc["model_calls"] = acc.get("model_calls", 0) + 1

    def _stream_direct(self, task: str, on_answer: Callable[[str], None] | None = None,
                       include_events: bool = False,
                       _usage_acc: dict[str, Any] | None = None,
                       **kwargs) -> "Iterator[str] | Iterator[StreamEvent]":
        """Stream a model answer directly, without the ReAct scaffold.

        Used by ``stream()`` when the agent has no tools. The prompt mirrors
        ``_run_direct_inference`` so streamed and non-streamed answers match.
        Tokens are yielded as they arrive (true incrementality); the assembled
        answer is sanitized before it is stored in memory and handed to
        ``on_answer``. A mid-stream provider error is raised
        (typed + redacted) rather than yielded as a chunk, so a consumer can
        tell success from failure. With ``include_events`` the same deltas are
        wrapped as :class:`StreamEvent` ``answer`` records.
        """
        # Mirror ``_run_direct_inference``: a custom persona leads the prompt and
        # owns the response contract; default agents keep the familiar framing.
        # Otherwise a custom persona (e.g. an `effgen chat --persona` tutor) is
        # silently ignored on the tool-free streaming path that chat uses.
        conversation_history = self._format_conversation_history()
        prompt = self._direct_prompt(task, conversation_history)

        # No ReAct stop sequences here: there is no scaffold to trim, and the
        # GPT-5/reasoning families reject `stop`. reasoning_effort is threaded
        # through so callers can request "minimal" for trivial prompts.
        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=resolve_output_budget(
                kwargs.get("max_tokens"), self.config.max_tokens, self.model
            ),
            top_p=kwargs.get("top_p", 0.9),
            stop_sequences=kwargs.get("stop_sequences"),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )

        from ..models.base import clear_stream_usage

        accumulated = ""
        clear_stream_usage(self.model)
        stream_iter = self.model.generate_stream(prompt, config=gen_config)
        try:
            for token in stream_iter:
                accumulated += token
                if token:
                    yield StreamEvent(kind="answer", text=token) if include_events else token
        except Exception:
            logger.debug("Streaming generation failed", exc_info=True)
            raise
        finally:
            close_stream = getattr(stream_iter, "close", None)
            if close_stream is not None:
                close_stream()

        if _usage_acc is not None:
            self._fold_stream_usage(_usage_acc, prompt, accumulated)

        answer = sanitize_final_answer(accumulated) or accumulated.strip()
        if answer:
            self.short_term_memory.add_user_message(task)
            self.short_term_memory.add_assistant_message(answer)
        if on_answer:
            on_answer(answer)

    def stream(self,
               task: "str | Message | list[Any]",
               mode: AgentMode | None = None,
               context: dict[str, Any] | None = None,
               on_thought: Callable[[str], None] | None = None,
               on_tool_call: Callable[[str, str], None] | None = None,
               on_observation: Callable[[str], None] | None = None,
               on_answer: Callable[[str], None] | None = None,
               inputs: list[Any] | None = None,
               include_events: bool = False,
               **kwargs: Any) -> "Iterator[str] | Iterator[StreamEvent]":
        """
        Stream a response incrementally using real model streaming.

        Streaming contract (stable):

        - **Default (text mode).** Iterating yields successive **answer-text**
          ``str`` deltas. Joining every chunk
          (``"".join(agent.stream(task))``) reconstructs the *sanitized* final
          answer — on both the no-tool and the tool path. Internal ReAct
          scaffolding (``Thought:``/``Action:``/``Observation:``/
          ``Final Answer:``) is **never** part of the text payload; on a tool
          agent the intermediate steps are delivered to the ``on_thought`` /
          ``on_tool_call`` / ``on_observation`` callbacks (and, with
          ``include_events=True``, as typed events) — not as text.
        - **Typed events (opt-in).** ``stream(..., include_events=True)`` yields
          :class:`StreamEvent` objects instead of plain text — ``answer`` deltas
          plus ``thought`` / ``tool_call`` / ``observation`` / ``status`` events
          — so a presentation layer can render live progress without parsing the
          text stream. Concatenating the ``text`` of the ``answer`` events still
          reconstructs the sanitized final answer.
        - **Usage (event mode only).** The final event of an
          ``include_events=True`` stream is a ``usage`` event whose ``usage``
          dict carries the run's token counts, cost and timings — see
          :attr:`last_stream_usage`, which holds the same dict after any
          stream (text mode included) so a text-mode consumer can read it
          without a second billed call. Text mode still yields answer text
          only, so ``"".join(agent.stream(task))`` is unchanged.
        - The iterator simply **ending** is the terminal "done" signal; there is
          no sentinel value to test for.
        - A provider/model failure raises a typed error from the iterator (it is
          not silently swallowed into an empty stream).

        Args:
            task: Task description. Accepts a ``str``, a ``Message``, or a
                ``list[ContentPart]`` (text is extracted).
            mode: Execution mode
            context: Optional context
            on_thought: Callback for thought tokens
            on_tool_call: Callback(tool_name, tool_input) when a tool is called
            on_observation: Callback for tool observation text
            on_answer: Callback for final answer tokens
            inputs: Multimodal content parts. Streaming is text-only today; if
                media parts are supplied a clear error points to ``run()``.
            include_events: When True, yield typed :class:`StreamEvent` objects
                instead of plain answer-text ``str`` deltas (opt-in; see above).
            **kwargs: Additional arguments

        Yields:
            ``str`` answer-text deltas by default, or :class:`StreamEvent`
            objects when ``include_events=True`` (see the streaming contract).
        """
        usage_acc: dict[str, Any] = {}
        started = time.perf_counter()
        ttft: float | None = None
        # Cleared up front so a stream that does not reconstruct a response —
        # a tool-free stream, or a model whose calls are not streamed — never
        # leaves the previous stream's record readable as if it were this one's.
        self._last_stream_response = None
        for item in self._stream_impl(
            task,
            mode=mode,
            context=context,
            on_thought=on_thought,
            on_tool_call=on_tool_call,
            on_observation=on_observation,
            on_answer=on_answer,
            inputs=inputs,
            include_events=include_events,
            _usage_acc=usage_acc,
            **kwargs,
        ):
            if ttft is None:
                is_answer_text = (
                    bool(item.text) and item.kind == "answer"
                    if isinstance(item, StreamEvent)
                    else bool(item)
                )
                if is_answer_text:
                    ttft = time.perf_counter() - started
            yield item

        usage = dict(usage_acc)
        # Every key is always present so a consumer can read the dict without
        # probing for optional fields; an unpriced model reports cost as None.
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
            usage.setdefault(key, None)
        usage.setdefault("model_calls", 0)
        usage.setdefault("estimated", False)
        usage["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        usage["ttft_ms"] = round(ttft * 1000.0, 1) if ttft is not None else None
        self._last_stream_usage = usage
        # A reconstructed per-turn record is built before these run-level
        # timings exist, so it is completed here rather than carrying its own
        # narrower numbers.
        response = getattr(self, "_last_stream_response", None)
        if response is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                        "cost_usd"):
                if usage.get(key) is not None:
                    response.metadata[key] = usage[key]
            response.metadata["latency_ms"] = usage["latency_ms"]
            response.metadata["ttft_ms"] = usage["ttft_ms"]
            response.tokens_used = int(
                usage.get("total_tokens") or response.tokens_used or 0
            )
        if include_events:
            yield StreamEvent(kind="usage", usage=usage)

    @property
    def last_stream_usage(self) -> dict[str, Any] | None:
        """Usage of the most recent completed :meth:`stream` call, or ``None``.

        Set once the stream iterator is exhausted (it is unknown before that).
        Keys: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
        ``cost_usd`` (``None`` for a model with no published price),
        ``latency_ms``, ``ttft_ms`` (time to the first answer token),
        ``model_calls`` (more than one on a tool-using run) and ``estimated``
        (``True`` when the token counts were counted locally because the
        backend reported none). This is the same dict the terminal ``usage``
        :class:`StreamEvent` carries.
        """
        return getattr(self, "_last_stream_usage", None)

    def _stream_impl(self,
                     task: "str | Message | list[Any]",
                     mode: AgentMode | None = None,
                     context: dict[str, Any] | None = None,
                     on_thought: Callable[[str], None] | None = None,
                     on_tool_call: Callable[[str, str], None] | None = None,
                     on_observation: Callable[[str], None] | None = None,
                     on_answer: Callable[[str], None] | None = None,
                     inputs: list[Any] | None = None,
                     include_events: bool = False,
                     _usage_acc: dict[str, Any] | None = None,
                     **kwargs) -> "Iterator[str] | Iterator[StreamEvent]":
        """Produce the stream payload; :meth:`stream` adds the usage accounting."""
        # Accept str | Message | list[ContentPart]; streaming is text-only, so
        # surface a clear error if media is supplied rather than dropping it.
        task, _stream_inputs = self._coerce_task_input(task, inputs)
        if _stream_inputs is not None:
            raise TypeError(
                "Agent.stream() is text-only; multimodal inputs are not "
                "supported while streaming. Use agent.run(task, inputs=[...]) "
                "for image/audio/video input."
            )

        if self.model is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        # Pre-stream input guardrail check, mirroring run()'s pre-run check —
        # a guardrail-configured agent must never let the model see a raw
        # input on the streaming path either. A block raises (stream() has no
        # success=False return to fall back on); a redaction replaces `task`
        # before it reaches either the direct or the tool-loop branch below,
        # so the model prompt and short-term memory only ever see the
        # modified content.
        if self._guardrail_chain is not None:
            from ..guardrails.base import GuardrailPosition
            gr = self._guardrail_chain.check(task, position=GuardrailPosition.INPUT)
            if not gr.passed:
                raise RuntimeError(f"Blocked by guardrail: {gr.reason}")
            if gr.modified_content is not None:
                task = gr.modified_content

        context = context or {}

        # Fast path: with no tools there is nothing for the ReAct loop to do, so
        # stream the model's answer directly. The ReAct scaffold otherwise forces
        # the model to emit Thought/Action/Final Answer bookkeeping that wastes
        # latency (acute on reasoning models) and leaks into the streamed output —
        # and small models that write "Action: Final Answer" instead of
        # "Final Answer:" loop to max-iterations and never surface an answer.
        if not self.tools:
            yield from self._stream_direct(
                task, on_answer=on_answer, include_events=include_events,
                _usage_acc=_usage_acc, **kwargs
            )
            return

        # With a model whose adapter records the tool calls it streams, the loop
        # can dispatch those calls natively while the assistant's text streams
        # through as it arrives — the same loop ``run()`` drives, rather than the
        # ReAct text scaffold below. Every other model keeps that scaffold.
        if self._can_stream_native_tools():
            yield from self._stream_native_tools(
                task,
                on_thought=on_thought,
                on_tool_call=on_tool_call,
                on_observation=on_observation,
                on_answer=on_answer,
                include_events=include_events,
                _usage_acc=_usage_acc,
                **kwargs,
            )
            return

        max_iterations = self.config.max_iterations
        scratchpad = ""
        iterations = 0
        tool_calls = 0
        # ``(action, normalized_input)`` for every call this stream dispatched,
        # in the shape the blocking loop keeps. The closing instruction is
        # chosen from the last call's tool, so without this the streamed prompt
        # could not state it and the two paths asked the model different
        # questions about the same observations.
        previous_actions: list[tuple[str, str]] = []
        # What each dispatched call returned, in the shape the blocking loop's
        # records take, so both paths decide from the same evidence whether the
        # answer dropped a result a tool computed.
        executed_calls: list[ToolCall] = []

        # Build conversation history
        conversation_history = self._format_conversation_history()

        default_stop_sequences = [
            "\nObservation:",
            "\nQuestion:",
            "\nHuman:",
            "\nUser:",
        ]

        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=resolve_output_budget(
                kwargs.get("max_tokens"), self.config.max_tokens, self.model
            ),
            top_p=kwargs.get("top_p", 0.9),
            stop_sequences=kwargs.get("stop_sequences", default_stop_sequences),
        )

        while iterations < max_iterations:
            iterations += 1

            # Build prompt
            tools_desc = self._get_tools_description()
            if self.config.system_prompt_template:
                prompt = self.config.system_prompt_template.format(
                    tools_description=tools_desc,
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                )
            else:
                cite_sources, numbered_passages = self._citation_prompt_state()
                prompt = self._tool_prompt_generator.generate_react_prompt(
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                    system_prompt=self.config.system_prompt,
                    verbose=self._verbose_tools,
                    closing_instruction=self._context_answer_instruction(
                        previous_actions,
                        cite_sources=cite_sources,
                        numbered_passages=numbered_passages,
                    ),
                    answer_shape=self._answer_shape_instruction(),
                    tool_contract=self._tool_contract(),
                )

            # Stream tokens from the model into a buffer. The raw ReAct
            # scaffolding (Thought/Action/Observation/Final Answer) is internal
            # bookkeeping and is NEVER yielded as the user-facing payload — only
            # the parsed, sanitized final answer is (text deltas in the default
            # mode; an "answer" StreamEvent in event mode). Stop sequences and an
            # early Final-Answer break still bound generation so a small model
            # that ignores `stop` cannot run away.
            accumulated = ""
            from ..models.base import clear_stream_usage
            clear_stream_usage(self.model)
            stream_iter = self.model.generate_stream(prompt, config=gen_config)
            try:
                for token in stream_iter:
                    accumulated += token

                    hit_stop = False
                    for stop_seq in default_stop_sequences:
                        if stop_seq in accumulated:
                            accumulated = accumulated[:accumulated.index(stop_seq)]
                            hit_stop = True
                            break

                    # Break early once the Final Answer line is *complete* to
                    # avoid runaway generation (transformers streaming ignores
                    # stop_sequences). "Complete" means the model ended the
                    # answer line (a newline after non-empty answer text) or
                    # started a new ReAct block — NOT merely "a few characters
                    # appeared", which would truncate a multi-word answer.
                    if not hit_stop and "Final Answer:" in accumulated:
                        fa_pos = accumulated.rindex("Final Answer:")
                        after_fa = accumulated[fa_pos + len("Final Answer:"):]
                        if after_fa.lstrip("\n").strip() and (
                            "\n" in after_fa.lstrip("\n")
                            or any(
                                m in after_fa
                                for m in ("Thought:", "Observation:", "Question:")
                            )
                        ):
                            hit_stop = True

                    if hit_stop:
                        break

            except Exception:
                # Fail explicitly: raise the typed (already-redacted) provider error
                # at the iterator boundary so a consumer iterating stream() can tell
                # success from failure, instead of receiving the error text as a
                # normal chunk that looks like model output.
                logger.debug("Streaming generation failed", exc_info=True)
                raise
            finally:
                close_stream = getattr(stream_iter, "close", None)
                if close_stream is not None:
                    close_stream()

            if _usage_acc is not None:
                self._fold_stream_usage(_usage_acc, prompt, accumulated)

            # Parse the accumulated response
            parsed = self._parse_react_response(accumulated)
            thought = parsed.get("thought", "")
            scratchpad += f"\nThought: {thought}"

            if thought:
                if on_thought:
                    on_thought(thought)
                if include_events:
                    yield StreamEvent(kind="thought", text=thought)

            # Check for final answer
            if parsed.get("final_answer"):
                answer = sanitize_final_answer(parsed["final_answer"]) or parsed["final_answer"]
                # A tool that computed the answer itself is answered by
                # summarising it far too often, and the result the run is still
                # holding is then lost. Put it back, exactly as run() does.
                appended = unrelayed_result(answer, executed_calls, self.tools)
                if appended is not None:
                    answer = f"{answer.rstrip()}\n\n{appended}"
                if on_answer:
                    on_answer(answer)
                # Store in memory
                if answer:
                    self.short_term_memory.add_user_message(task)
                    self.short_term_memory.add_assistant_message(answer)
                # Emit the sanitized answer as the user-facing payload. Text mode
                # re-chunks it character-preservingly so joining the deltas
                # reproduces the answer exactly; event mode emits one answer event.
                if answer:
                    if include_events:
                        yield StreamEvent(kind="answer", text=answer)
                    else:
                        yield from _chunk_answer_text(answer)
                return

            # Execute tool if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                if on_tool_call:
                    on_tool_call(action, action_input)
                if include_events:
                    yield StreamEvent(
                        kind="tool_call", tool=action, tool_input=str(action_input)
                    )
                previous_actions.append(
                    (action, NativeToolLoop.normalize_input(action_input))
                )

                if action in self.tools:
                    tool_result = self._execute_tool(action, action_input)
                    tool_calls += 1
                    executed_calls.append(ToolCall(
                        name=action,
                        arguments=action_input,
                        result=truncate_result(tool_result),
                        iteration=iterations,
                    ))

                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    if on_observation:
                        on_observation(str(tool_result))
                    if include_events:
                        yield StreamEvent(
                            kind="observation", tool=action, text=str(tool_result)
                        )
                else:
                    # Same observation the non-streaming loop uses, so run() and
                    # stream() tell the model the same thing about the same text.
                    observation = (
                        unknown_tool_observation(action, list(self.tools))
                        if self.tools
                        else NUDGE_NO_TOOLS
                    )
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {observation}"
                    if on_observation:
                        on_observation(observation)
                    if include_events:
                        yield StreamEvent(
                            kind="observation",
                            tool=action,
                            text=observation,
                        )
            else:
                scratchpad += "\nAction: (continue reasoning)"

        # Step limit reached without a Final Answer: surface a clear terminal
        # notice (never raw scaffolding) so the stream is not silently empty.
        limit_msg = (
            "I wasn't able to finish this within the step limit. "
            "Try simplifying the request or raising max_iterations."
        )
        if include_events:
            yield StreamEvent(kind="status", text=limit_msg)
        else:
            yield limit_msg
