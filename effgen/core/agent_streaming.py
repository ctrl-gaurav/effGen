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

from ..models._adapter_utils import apply_stop_sequences
from .agent_config import AgentMode
from .agent_loop import (
    _Deltas,
    _LoopPolicy,
    drive,
    resolve_turn_config,
)
from .agent_response import StreamEvent
from .agent_runtime import sanitize_final_answer

if TYPE_CHECKING:
    from .agent_config import AgentConfig
    from .messages import Message

logger = logging.getLogger(__name__)

#: Set once a streamed run was given a ``mode``. Routing sub-agents through a
#: streamed loop is not supported yet, and saying so on every call would drown
#: the log of a chat session.
_STREAM_MODE_WARNED: set[bool] = set()


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
        model: Any
        name: str
        _guardrail_chain: Any

        def _reconstruct_error(
            self, metadata: dict[str, Any] | None, response: Any = None,
        ) -> Exception: ...

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

        # The same nine settings ``run()`` resolves, from the same place: a
        # value pinned on the call, then one configured on the agent, then the
        # model's own default. A model that matches stop sequences against its
        # own reasoning chain is sent none and has them applied to the text it
        # returns, which is what the blocking path does with them too.
        gen_config, local_stops = resolve_turn_config(self, kwargs)

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

        if local_stops:
            accumulated = apply_stop_sequences(accumulated, list(local_stops))
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
          ``str`` deltas. On a tool agent, joining every chunk
          (``"".join(agent.stream(task))``) reconstructs the *sanitized* final
          answer, and internal ReAct scaffolding
          (``Thought:``/``Action:``/``Observation:``/``Final Answer:``) is
          **never** part of the text payload: the intermediate steps are
          delivered to the ``on_thought`` / ``on_tool_call`` /
          ``on_observation`` callbacks (and, with ``include_events=True``, as
          typed events) — not as text. An agent with **no tools** has one turn
          and nothing to decide, so its text is the model's own, delivered as it
          is written; the sanitized form of it is what :meth:`run` returns.
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
        - **The run's record.** After any stream that entered the loop,
          :attr:`last_stream_response` holds the
          :class:`~effgen.core.agent_response.AgentResponse` the same task would
          have produced through :meth:`run` — the same ``output``,
          ``success``, ``stop_reason`` and metadata. A run that ends without an
          answer (its step limit, a repeated call, a call the model wrote out
          instead of making) reports the same typed outcome ``run()`` reports,
          delivered as a terminal notice rather than as answer text.
        - A provider/model failure raises a typed error from the iterator (it is
          not silently swallowed into an empty stream) unless the agent was
          built with ``raise_on_error=False``, which suppresses it here as it
          does on :meth:`run`.

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
        # Cleared up front so a stream that does not reach the loop — a
        # tool-free stream — never leaves the previous stream's record readable
        # as if it were this one's.
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
            response.execution_time = usage["latency_ms"] / 1000.0
            response.metadata["duration_s"] = round(response.execution_time, 4)
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
        """Produce the stream payload; :meth:`stream` adds the usage accounting.

        The loop is the one :meth:`run` drives. The only difference is the
        emitter: this one hands the consumer each step as it happens and the
        answer as it settles, where ``run()``'s collects and returns.
        """
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
        if mode is not None and not _STREAM_MODE_WARNED:
            _STREAM_MODE_WARNED.add(True)
            logger.warning(
                "[loop] stream() ignores mode=%s; sub-agent routing is not "
                "streamed and the task runs on this agent alone", mode,
            )

        # Fast path: with no tools there is nothing for the loop to do, so
        # stream the model's answer directly. The scaffold otherwise forces the
        # model to emit bookkeeping that wastes latency (acute on reasoning
        # models) and leaks into the streamed output.
        if not self.tools:
            yield from self._stream_direct(
                task, on_answer=on_answer, include_events=include_events,
                _usage_acc=_usage_acc, **kwargs
            )
            return

        from ..utils.structured_logging import LogRunContext, generate_run_id

        run_id = str(kwargs.pop("_run_id", "") or "") or generate_run_id()
        kwargs["_run_id"] = run_id
        prompt_task: Any = task
        policy = _LoopPolicy.for_run(self, kwargs, emit_deltas=True)
        emitter = _Deltas(
            self,
            include_events=include_events,
            on_thought=on_thought,
            on_tool_call=on_tool_call,
            on_observation=on_observation,
            usage_acc=_usage_acc,
        )
        with LogRunContext(run_id=run_id, agent_name=self.name):
            driver = drive(self, prompt_task, policy, emitter)
            while True:
                try:
                    item = next(driver)
                except StopIteration as stop:
                    response = stop.value
                    break
                yield item

            response.metadata["run_id"] = run_id
            response.metadata.setdefault("streamed", True)
            if emitter.turns_retaken:
                # A turn the stream could not finish was taken again on the
                # blocking path. The key is what a caller has always read to
                # tell a fallback from a clean stream.
                response.metadata["stream_fallback"] = True
            self._last_stream_response = response
            yield from self._finish_stream(
                prompt_task, response, emitter, on_answer=on_answer,
            )

    def _finish_stream(
        self,
        task: Any,
        response: Any,
        emitter: Any,
        *,
        on_answer: Callable[[str], None] | None = None,
    ) -> "Iterator[str] | Iterator[StreamEvent]":
        """Deliver what the loop ended with, and screen it on the way out.

        An answer is put through the OUTPUT guardrail before it is finished —
        the same screening ``run()`` applies — then handed to the consumer, the
        callback and short-term memory. A run that ended without an answer has
        the typed outcome ``run()`` reports, delivered as a terminal notice
        rather than as answer text.
        """
        if response.success and response.output:
            if self._guardrail_chain is not None:
                from ..guardrails.base import GuardrailPosition as _GP
                gr = self._guardrail_chain.check(
                    response.output, position=_GP.OUTPUT,
                    system_prompt=self.config.system_prompt,
                )
                if not gr.passed:
                    response.metadata["guardrail_blocked"] = True
                    response.metadata["guardrail_reason"] = gr.reason
                    response.success = False
                    raise RuntimeError(
                        f"Output blocked by guardrail: {gr.reason}. "
                        "An iterator has no blocked answer to hand back, so "
                        "relax the guardrail or call agent.run(), which "
                        "returns the blocked outcome as a response."
                    )
                if gr.modified_content is not None:
                    response.output = gr.modified_content
                    response.metadata["guardrail_modified"] = True
            text = str(response.output)
            if on_answer:
                on_answer(text)
            if text:
                self.short_term_memory.add_user_message(task)
                self.short_term_memory.add_assistant_message(text)
            yield from emitter.answer(text)
            return
        # A provider that never answered has no result to report, so a caller
        # who left ``raise_on_error`` at its default gets the typed error rather
        # than a notice. A caller who turned it off asked not to be raised at,
        # on this path as much as on the blocking one.
        reason = str((response.metadata or {}).get("reason") or "")
        if reason == "generation_failed" and self.config.raise_on_error:
            raise self._reconstruct_error(response.metadata, response)
        yield from emitter.status(str(response.output or ""))
