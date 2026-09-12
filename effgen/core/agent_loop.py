"""One reasoning loop, driven once per run, blocking or streamed.

The ReAct loop used to be written three times: once for :meth:`Agent.run`, once
for the prompt-scaffold branch of :meth:`Agent.stream`, and once for the branch
that dispatches a provider's streamed tool calls. They shared the repeat guards
and nothing else, so the same agent asked the same model a different question
depending on which method the caller used — a different prompt frame, different
sampling settings, a different set of safety checks and a different terminal
contract.

This module is the loop. Four pieces, and only the last of them knows whether
anything is being streamed:

* :class:`_LoopPolicy` — every question the three loops used to answer
  differently, decided once at the top of a run. ``frame`` is chosen from the
  model's *declared* capability and the caller's own configuration; it does not
  read whether the adapter streams its tool calls, and it does not read
  ``emit_deltas``.
* :class:`~effgen.core.thread.AgentThread` — the conversation, on every path.
* :func:`step` — one iteration: build the prompt, take the turn, read it, apply
  the guards, dispatch the calls, append the steps. It never yields and never
  prints.
* an emitter — :class:`_Collecting` for a blocking run, which returns the
  answer, or :class:`_Deltas` for a streamed one, which hands the consumer the
  answer as it settles. The emitter is the *only* thing that differs between
  ``run()`` and ``stream()``.

The one asymmetry that is real: a streamed turn cannot be un-emitted, so **a
turn whose outcome a later check can still revise is accumulated rather than
streamed** and delivered in one piece. A model whose adapter does not record
the tool calls it streams is accumulated for the same reason.

Everything here is private. This module imports nothing from ``agent.py``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..models._adapter_utils import apply_stop_sequences, normalize_stop_sequences
from ..models.base import (
    GenerationConfig,
    clear_stream_tool_calls,
    clear_stream_usage,
    get_stream_tool_calls,
)
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
from ..utils.prometheus_metrics import metrics as prom_metrics
from ..utils.structured_logging import get_structured_logger
from .agent_config import AgentMode
from .agent_response import AgentResponse, StreamEvent
from .agent_runtime import (
    CONTINUE_REASONING_LINE,
    NUDGE_ALREADY_COMPUTED,
    NUDGE_CONTINUE,
    NUDGE_HAVE_RESULTS,
    NUDGE_MUST_EXECUTE,
    NUDGE_NO_TOOLS,
    NUDGE_NOT_USABLE,
    NUDGE_SEARCH_AGAIN,
    PROTOCOL_REFUSED,
    PROTOCOL_SPLIT,
    _count_tool_parts,
    _infer_provider_from_model,
    find_written_tool_call,
    model_can_require_tool_call,
    resolve_output_budget,
    sanitize_final_answer,
    unknown_tool_observation,
)
from .agent_tool_loop import NativeToolLoop
from .execution_tracker import EventType, ExecutionEvent
from .result_relay import relay_result
from .retrieval_requery import (
    REQUERY_FORCES_TOOL_CALL,
    REQUERY_MIN_ITERATIONS_LEFT,
    should_requery,
)
from .thread import (
    ActionStep,
    AgentThread,
    AnswerStep,
    NudgeStep,
    ObservationStep,
    Step,
    SystemStep,
    TaskStep,
    ThoughtStep,
)
from .thread_budget import (
    UNBOUNDED_CONTEXT_BUDGET,
    count_prompt_tokens,
    resolve_context_budget,
)
from .thread_compaction import resolve_policy
from .tool_call_record import ToolCallList

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Generator

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
_obs_log = _get_obs_logger(__name__)


# --- The policy --------------------------------------------------------------

#: The stop sequences the ReAct scaffold needs trimming at. They are a property
#: of the frame, not of the path, so every path sends the same list for the
#: same frame.
DEFAULT_STOP_SEQUENCES = (
    "\nObservation:",
    "\nQuestion:",
    "\nHuman:",
    "\nUser:",
)


@dataclass(frozen=True)
class _LoopPolicy:
    """Everything the loop decides once, before the first prompt is built.

    ``emit_deltas`` is the only field whose value depends on which method the
    caller used. Every other field is a function of the agent's configuration
    and the model's declared capability, so a run and a streamed run of the same
    agent send the same request.
    """

    #: ``"native"``, ``"custom_template"`` or ``"react_text"``, chosen from the
    #: declared tool-calling strategy, the model's own
    #: ``supports_tool_calling()`` and whether the caller supplied a template.
    frame: str
    max_iterations: int
    #: The sampling settings for one turn, resolved once. The blocking emitter
    #: does not read it — ``_generate`` rebuilds the identical configuration for
    #: its first attempt, which is what keeps ``run()`` byte-for-byte unchanged.
    gen_config: GenerationConfig
    #: Applied to the returned text instead of being sent, for a model that
    #: matches stop sequences against its own reasoning chain.
    local_stop_sequences: tuple[str, ...] | None
    #: The caller's keyword arguments, as the loop forwards them to the model.
    call_kwargs: dict[str, Any]
    emit_deltas: bool
    retries: int
    checkpoint_interval: int
    checkpoint_dir: Any
    resume_scratchpad: str | None
    #: The run's steps as a resumed checkpoint stored them. Preferred over
    #: *resume_scratchpad*, which is the same conversation with its structure
    #: rendered away.
    resume_thread: Any
    debug: bool
    run_id: str
    raise_on_error: bool
    #: The task's non-text content parts — an image, a clip — validated once,
    #: before the first turn. They ride on the thread's task step, so a run
    #: holding both a picture and a tool drives the loop instead of falling out
    #: of it into a single direct call.
    task_parts: tuple[Any, ...] = ()
    #: Steps a parent run chose for this one to start with — a projection of
    #: the parent's thread (:mod:`effgen.core.thread_projection`). They open the
    #: run's conversation in front of its question, so the child answers with
    #: the background its parent gave it rather than with that background pasted
    #: into the question itself.
    prior_steps: tuple[Any, ...] = ()

    @property
    def tools_travel_as_parameter(self) -> bool:
        """Whether this run's frame sends the tool definitions as a parameter."""
        return self.frame == "native"

    @classmethod
    def for_run(
        cls, agent: Any, kwargs: dict[str, Any], *, emit_deltas: bool
    ) -> _LoopPolicy:
        """Decide the policy for one run.

        *kwargs* is consumed: the loop's own bookkeeping arguments are popped
        out of it so they cannot reach the model layer, exactly as the blocking
        loop popped them before.
        """
        debug = bool(kwargs.pop("_debug", False))
        run_id = str(kwargs.pop("_run_id", "") or "")
        checkpoint_interval = kwargs.pop("checkpoint_interval", 0) or 0
        checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        resume = kwargs.pop("_resume_scratchpad", None)
        resume_thread = kwargs.pop("_resume_thread", None)
        prior_steps = tuple(kwargs.pop("_prior_steps", None) or ())
        # Content parts belong to the conversation, not to the model call, so
        # they are taken out of the caller's keyword arguments here and put on
        # the thread's task step instead.
        raw_inputs = kwargs.pop("inputs", None)
        task_parts = (
            tuple(agent._content_parts_from_inputs(raw_inputs))
            if raw_inputs is not None else ()
        )
        # A ``max_tokens`` configured on the agent is the default budget for
        # every run. ``run()`` already writes it into its kwargs; doing it here
        # as well means a streamed run resolves the same budget rather than
        # falling through to the model's own default.
        if kwargs.get("max_tokens") is None and agent.config.max_tokens is not None:
            kwargs["max_tokens"] = agent.config.max_tokens
        requested = kwargs.get("max_iterations")
        max_iterations = (
            agent.config.max_iterations if requested is None else int(requested)
        )
        gen_config, local_stops = resolve_turn_config(agent, kwargs)
        policy = cls(
            frame=frame_for(agent),
            max_iterations=max_iterations,
            gen_config=gen_config,
            local_stop_sequences=local_stops,
            call_kwargs=kwargs,
            emit_deltas=emit_deltas,
            retries=3,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            resume_scratchpad=resume,
            resume_thread=resume_thread,
            debug=debug,
            run_id=run_id,
            raise_on_error=bool(agent.config.raise_on_error),
            task_parts=task_parts,
            prior_steps=prior_steps,
        )
        logger.info(
            "[loop] frame=%s tools_as_parameter=%s streamed=%s max_iterations=%d",
            policy.frame, policy.tools_travel_as_parameter,
            policy.emit_deltas, policy.max_iterations,
        )
        return policy


def frame_for(agent: Any, *, tools_suppressed: bool = False) -> str:
    """Which prompt frame this agent's turns are written in.

    Read from the declared tool-calling strategy, the model's own
    ``supports_tool_calling()`` and whether the caller supplied a template of
    their own. Nothing here reads whether the adapter *streams* its tool calls,
    and nothing reads whether deltas are being emitted — a model that can be
    handed tool definitions is handed them whichever method the caller used.

    *tools_suppressed* is the guards' decision to stop offering tools for the
    rest of a run; the frame then falls back to the text scaffold, which is the
    only frame that can ask for an answer without offering a call.
    """
    native = (
        agent._tool_calling_strategy.name in ("native", "hybrid")
        and agent.model is not None
        and hasattr(agent.model, "supports_tool_calling")
        and agent.model.supports_tool_calling()
    )
    if tools_suppressed:
        native = False
    if native and not agent.config.system_prompt_template:
        return "native"
    if agent.config.system_prompt_template:
        return "custom_template"
    return "react_text"


def resolve_turn_config(
    agent: Any, kwargs: dict[str, Any]
) -> tuple[GenerationConfig, tuple[str, ...] | None]:
    """The sampling settings one turn goes out with, and any local trimming.

    The same resolution the blocking generation path applies to its first
    attempt: a value pinned on the call, then one configured on the agent, then
    the model's own default. Returned as a configuration rather than applied, so
    a streamed turn and a blocking turn of the same run are provably identical.

    A model that matches stop sequences against its own reasoning chain is sent
    none and has them applied to the text it returns; those are the second
    element of the pair.
    """
    config = agent.config
    requested_stops = normalize_stop_sequences(
        kwargs.get("stop_sequences", list(DEFAULT_STOP_SEQUENCES))
    )
    local_stops: tuple[str, ...] | None = None
    try:
        interleaves = agent._interleaves_reasoning(agent.model)
    except Exception:  # noqa: BLE001 - a capability probe never breaks a run
        logger.debug("reasoning-interleave probe failed", exc_info=True)
        interleaves = False
    if interleaves:
        local_stops = tuple(requested_stops or ())
    gen_config = GenerationConfig(
        temperature=kwargs.get("temperature", config.temperature),
        max_tokens=resolve_output_budget(
            kwargs.get("max_tokens"), config.max_tokens, agent.model
        ),
        top_p=kwargs.get("top_p", config.top_p),
        top_k=kwargs.get("top_k", config.top_k),
        seed=kwargs.get("seed", config.seed),
        presence_penalty=kwargs.get("presence_penalty", config.presence_penalty),
        frequency_penalty=kwargs.get("frequency_penalty", config.frequency_penalty),
        repetition_penalty=kwargs.get(
            "repetition_penalty", config.repetition_penalty
        ),
        stop_sequences=(None if local_stops else requested_stops),
        reasoning_effort=kwargs.get("reasoning_effort"),
    )
    return gen_config, local_stops


#: The nine settings a caller can pin on an agent or on a call. Resolved in one
#: place, so "the same agent samples the same way whichever method was called"
#: is a fact rather than a claim.
SAMPLING_FIELDS = (
    "temperature", "max_tokens", "top_p", "top_k", "seed",
    "presence_penalty", "frequency_penalty", "repetition_penalty",
    "stop_sequences",
)


# --- The run's mutable state -------------------------------------------------


@dataclass
class _RunState:
    """What one run accumulates as it goes.

    ``iter_start`` and ``prompt`` are the current turn's, kept for the debug
    trace's latency arithmetic and its record of what the model was sent.
    """

    thread: AgentThread
    guards: NativeToolLoop
    conversation_history: str = ""
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    resolved_to_messages: bool = False
    frame_carried_by_messages: bool = False
    debug_trace: Any = None
    checkpoints: Any = None
    iter_start: float = 0.0
    prompt: Any = ""
    #: How many prompt tokens this run may send, resolved once before the first
    #: turn. ``None`` when the run is unbounded — the caller asked for that, or
    #: the model declares no window a budget could be derived from.
    budget: Any = None
    #: What the run gives up when it reaches the budget.
    compaction: Any = None


@dataclass(frozen=True)
class _StepOutcome:
    """What one iteration decided.

    ``kind`` is ``"continue"`` when the run goes round again and ``"response"``
    when the iteration produced the run's terminal
    :class:`~effgen.core.agent_response.AgentResponse` — an answer, a stop, or a
    failure. The response carries which of those it was in its ``stop_reason``.
    """

    kind: str
    response: AgentResponse | None = None


CONTINUE = _StepOutcome("continue")


# --- The emitters ------------------------------------------------------------


class _Emitter(Protocol):
    """How a turn reaches the caller.

    Every method is a generator so a streamed run can hand the consumer
    something; on a blocking run every one of them yields nothing at all.
    ``model_turn`` returns the turn in the shape the generation layer returns
    it, so nothing downstream can tell how it was taken.
    """

    def model_turn(
        self, agent: Any, prompt: Any, policy: _LoopPolicy,
        gen_kwargs: dict[str, Any], *, streamable: bool,
    ) -> Generator[Any, None, dict[str, Any]]: ...
    def thought(self, text: str) -> Iterator[Any]: ...
    def tool_call(self, tool: str, tool_input: str) -> Iterator[Any]: ...
    def observation(self, tool: str, text: str) -> Iterator[Any]: ...
    def answer(self, text: str) -> Iterator[Any]: ...
    def status(self, text: str) -> Iterator[Any]: ...


class _Collecting:
    """``run()``'s emitter: the answer is the return value.

    Every model turn goes through ``_generate``, so the retries, the stop-
    sequence trimming and the cost accounting are exactly what they were, and
    every step method hands the caller nothing at all.
    """

    emits_deltas = False

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def model_turn(
        self, agent: Any, prompt: Any, policy: _LoopPolicy,
        gen_kwargs: dict[str, Any], *, streamable: bool,
    ) -> Generator[Any, None, dict[str, Any]]:
        turn: dict[str, Any] = agent._generate(prompt, **gen_kwargs)
        return turn
        yield  # pragma: no cover - makes this a generator function

    @staticmethod
    def _nothing(*_args: Any, **_kwargs: Any) -> Iterator[Any]:
        return iter(())

    thought = tool_call = observation = answer = status = _nothing


class _Deltas:
    """``stream()``'s emitter: the answer reaches the consumer as it settles.

    A turn is streamed only when nothing downstream can still revise it: the
    frame hands the tool definitions to the provider, the adapter records the
    calls it streams, and no check is holding the turn back. Every other turn is
    accumulated through ``_generate`` — the same call ``run()`` makes, with the
    same retries — and its answer is delivered in one piece.
    """

    emits_deltas = True

    def __init__(
        self,
        agent: Any,
        *,
        include_events: bool = False,
        on_thought: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_observation: Callable[[str], None] | None = None,
        usage_acc: dict[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.include_events = include_events
        self.on_thought = on_thought
        self.on_tool_call = on_tool_call
        self.on_observation = on_observation
        self.usage_acc = usage_acc
        self.answer_stream = _AnswerStream()
        #: Whether any answer text has reached the consumer.
        self.committed = False
        #: Turns that could not be streamed to the end and were taken again on
        #: the blocking path, which carries the retries a stream has none of.
        self.turns_retaken = 0

    # -- helpers ----------------------------------------------------------
    def _answer_event(self, text: str) -> StreamEvent | str:
        return StreamEvent(kind="answer", text=text) if self.include_events else text

    def reset_answer(self) -> None:
        """Start the answer over — a turn was discarded before it was shown."""
        self.answer_stream = _AnswerStream()

    # -- the protocol -----------------------------------------------------
    def model_turn(
        self, agent: Any, prompt: Any, policy: _LoopPolicy,
        gen_kwargs: dict[str, Any], *, streamable: bool,
    ) -> Generator[Any, None, dict[str, Any]]:
        if not streamable:
            response: dict[str, Any] = agent._generate(prompt, **gen_kwargs)
            self._fold_response_usage(response)
            return response
        return (yield from self._streamed_turn(agent, prompt, policy, gen_kwargs))

    def thought(self, text: str) -> Iterator[Any]:
        if not text:
            return
        if self.on_thought:
            self.on_thought(text)
        if self.include_events:
            yield StreamEvent(kind="thought", text=text)

    def tool_call(self, tool: str, tool_input: str) -> Iterator[Any]:
        if self.on_tool_call:
            self.on_tool_call(tool, tool_input)
        if self.include_events:
            yield StreamEvent(kind="tool_call", tool=tool, tool_input=tool_input)

    def observation(self, tool: str, text: str) -> Iterator[Any]:
        if self.on_observation:
            self.on_observation(str(text))
        if self.include_events:
            yield StreamEvent(kind="observation", tool=tool, text=str(text))

    def answer(self, text: str) -> Iterator[Any]:
        """Deliver whatever of *text* the consumer has not already seen.

        The loop's terminal answer is the answer of record. When it extends what
        was streamed, the remainder is delivered; when nothing has been streamed
        yet — an accumulated turn, which is every turn on a frame that is read
        as a whole — the whole answer is delivered here. A terminal answer that
        *rewrites* text already on screen cannot be taken back, so the consumer
        keeps what it was sent and the divergence is logged.
        """
        already = self.answer_stream.emitted
        if already and text.startswith(already):
            delta = text[len(already):]
        elif not self.committed:
            delta = text
        else:
            logger.info(
                "[loop] the final answer differs from the text already "
                "streamed; the consumer keeps what it was sent"
            )
            return
        self.answer_stream.emitted = text
        if delta:
            self.committed = True
            yield self._answer_event(delta)

    def status(self, text: str) -> Iterator[Any]:
        """Deliver a terminal notice that is not an answer.

        The iteration cap, a loop the guards stopped and a call the model wrote
        out instead of making have no answer to stream. The typed outcome
        travels as a ``status`` event, and as plain text without events, so both
        modes of one stream carry the same words.
        """
        if not text:
            return
        if self.include_events:
            yield StreamEvent(kind="status", text=text)
        else:
            yield text

    # -- the streamed turn ------------------------------------------------
    def _streamed_turn(
        self, agent: Any, prompt: Any, policy: _LoopPolicy,
        gen_kwargs: dict[str, Any],
    ) -> Generator[Any, None, dict[str, Any]]:
        from .agent_generation import model_call_kwargs

        model = agent.model
        clear_stream_usage(model)
        clear_stream_tool_calls(model)
        raw = ""
        turn_committed = False
        failure: Exception | None = None
        stream_iter: Any = None
        emitted_here = False
        try:
            # Opening the stream belongs inside the guard: the budget gate
            # wrapping ``generate_stream`` refuses synchronously, before the
            # iterator exists, so a refusal raised here would otherwise leave
            # the loop with no record and no fallback.
            stream_iter = model.generate_stream(
                prompt, config=policy.gen_config, **model_call_kwargs(gen_kwargs)
            )
            for token in stream_iter:
                if not token:
                    continue
                raw += token
                if not turn_committed:
                    if get_stream_tool_calls(model):
                        # The turn is making a call; its text is reasoning.
                        continue
                    turn_committed = True
                    delta = self.answer_stream.push(raw)
                else:
                    delta = self.answer_stream.push(token)
                if delta:
                    self.committed = True
                    emitted_here = True
                    yield self._answer_event(delta)
        except Exception as exc:  # noqa: BLE001 - handled below
            logger.debug("Streamed turn failed", exc_info=True)
            failure = exc
        finally:
            close_stream = getattr(stream_iter, "close", None)
            if close_stream is not None:
                close_stream()

        if failure is not None:
            if emitted_here:
                # Part of this turn is already on screen; the contract says a
                # mid-stream failure is raised, not disguised as an end.
                raise failure
            # Nothing of this turn reached anyone, so it can be taken again on
            # the blocking path, which retries provider-side rejections a
            # stream has no retry for.
            logger.info(
                "[loop] the streamed turn failed before any output; the turn "
                "is taken again without streaming (%s)", type(failure).__name__,
            )
            self.turns_retaken += 1
            retaken: dict[str, Any] = agent._generate(prompt, **gen_kwargs)
            self._fold_response_usage(retaken)
            return retaken

        calls = list(get_stream_tool_calls(model) or [])
        if not calls and not raw.strip():
            # A turn that streamed nothing at all is the empty response the
            # blocking path retries, at a rising temperature, and reports as a
            # failure when it stays empty. Nothing reached anyone, so the turn
            # is taken again there rather than read as a turn that said nothing.
            logger.info(
                "[loop] the streamed turn produced no text and no call; the "
                "turn is taken again without streaming"
            )
            self.turns_retaken += 1
            empty: dict[str, Any] = agent._generate(prompt, **gen_kwargs)
            self._fold_response_usage(empty)
            return empty
        before = int((self.usage_acc or {}).get("completion_tokens") or 0)
        if self.usage_acc is not None:
            agent._fold_stream_usage(self.usage_acc, prompt, raw)
        after = int((self.usage_acc or {}).get("completion_tokens") or 0)
        if policy.local_stop_sequences:
            raw = apply_stop_sequences(raw, list(policy.local_stop_sequences))
        return {
            "text": raw,
            "tool_calls": calls,
            "tokens_used": max(after - before, 0),
            "finish_reason": "tool_calls" if calls else "stop",
            "metadata": {"streamed": True},
        }

    def _fold_response_usage(self, response: dict[str, Any]) -> None:
        """Fold a blocking turn taken inside a streamed run into the usage."""
        acc = self.usage_acc
        if acc is None:
            return
        meta = response.get("metadata") or {}
        got = meta.get("prompt_tokens")
        out = meta.get("completion_tokens", response.get("tokens_used"))
        total = meta.get("total_tokens")
        if total is None and (got is not None or out is not None):
            total = int(got or 0) + int(out or 0)
        for key, value in (("prompt_tokens", got), ("completion_tokens", out),
                           ("total_tokens", total)):
            if value is not None:
                acc[key] = (acc.get(key) or 0) + int(value)
        cost = meta.get("cost_usd")
        if cost is not None:
            acc["cost_usd"] = (acc.get("cost_usd") or 0.0) + float(cost)
        acc["model_calls"] = acc.get("model_calls", 0) + 1


# --- Turning raw model text into answer deltas -------------------------------

#: Labels that make sanitizing promote what follows them and discard what came
#: before. Once the label is resolved the text after it grows monotonically, so
#: only the label itself has to be held back.
_ANSWER_LABELS = ("final answer:", "answer:")

#: Tool-call syntax a model sometimes writes out instead of calling. Sanitizing
#: removes these constructs from the middle of the text, so nothing can be
#: emitted from a turn that contains one — what survives is not an extension of
#: what came before it. The tag openers cover the XML dialect too
#: (``<function=x><parameter=y>…``), so such a turn is held back and delivered
#: sanitized rather than streamed with its scaffolding on screen.
#:
#: These are **openers**, and not a complete account of how a local chat
#: template writes a call: measured on 2026-08-13 over five local families,
#: some open with one of these tags and others emit a bare JSON object with no
#: opener at all — and a suffix scan cannot hold the second shape back without
#: holding back every answer that starts with ``{``, which is what structured
#: output looks like. That is why a turn is streamed only when the adapter
#: records the calls it streams.
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


# --- The terminal contract ---------------------------------------------------


@dataclass
class _TurnSnapshot:
    """The counters an answer reports, read once the turn has been parsed.

    The blocking loop built its response from a closure whose defaults were
    bound at that same point — after the turn's own tool batch had been
    dispatched, before anything else this turn does. Kept as a value so the
    same numbers reach the same response.
    """

    tokens_used: int
    iterations: int
    tool_calls: int
    iter_start: float
    thread: AgentThread


def build_response(
    agent: Any,
    policy: _LoopPolicy,
    state: _RunState,
    snapshot: _TurnSnapshot,
    output: str,
    success: bool = True,
    *,
    tool_calls: int | None = None,
    **extra_meta: Any,
) -> AgentResponse:
    """The run's terminal contract, built once for every path.

    An answer that writes out a call for a tool this agent holds is reported as
    the failure it is; a result the run is still holding and did not report is
    put back; the conversation and the protocol it went out on travel on the
    metadata. Nothing here depends on whether anything was streamed.

    Args:
        agent: The agent the run belongs to.
        policy: The run's policy, for the frame it was prompted in.
        state: The run's conversation, guards and counters.
        snapshot: The counters as they stood at the top of this turn.
        output: The answer the turn produced.
        success: Whether the run answered.
        tool_calls: The call count to report, when it is not the snapshot's.
        **extra_meta: Extra keys for ``AgentResponse.metadata``.

    Returns:
        The run's terminal response.
    """
    thread = state.thread
    guards = state.guards
    tokens_used = snapshot.tokens_used
    iterations = snapshot.iterations
    n_tool_calls = snapshot.tool_calls if tool_calls is None else tool_calls
    if success:
        raw_answer = output
        output = sanitize_final_answer(output) or output
        # An answer that writes out a call for a tool this agent holds means
        # the tool never ran: the turn describes work that did not happen, so it
        # is reported as a failure. Sanitizing a tagged call can leave its
        # arguments behind as a bare JSON fragment, so the text as the model
        # wrote it is scanned as well as the cleaned answer.
        written = find_written_tool_call(
            output, agent.tools
        ) or find_written_tool_call(raw_answer, agent.tools)
        if written and guards.is_unmade_call(written, raw_answer):
            reported: AgentResponse = agent._written_tool_call_response(
                written,
                output,
                iterations=iterations,
                tool_calls=n_tool_calls,
                tokens_used=tokens_used,
                tool_ran=guards.tool_ran(written),
                debug_trace=state.debug_trace,
                calls=guards.calls,
                thread=snapshot.thread,
            )
            return reported
        # A tool that computed the answer itself — a move sequence, a colouring,
        # a sorted list — is answered by summarising it far too often, and the
        # result the run is still holding is then lost. Put it back, after the
        # scaffolding strip above so a result carrying an "Answer:" line of its
        # own is not read as a label on the answer.
        output = relay_result(output, guards.calls, agent.tools)
    if success:
        thread.append(AnswerStep(text=output, stop_reason="final_answer"))
    meta = _terminal_meta(agent, thread, "final_answer")
    meta.update(extra_meta)
    if state.debug_trace is not None:
        state.debug_trace.total_tokens = tokens_used
        state.debug_trace.total_latency = time.time() - (
            snapshot.iter_start - (iterations - 1) * 0.001
        )
        state.debug_trace.final_answer = output if success else None
        state.debug_trace.success = success
        meta["debug_trace"] = state.debug_trace
    return AgentResponse(
        output=output,
        success=success,
        mode=AgentMode.SINGLE,
        iterations=iterations,
        tool_calls=ToolCallList(list(guards.calls), total=n_tool_calls),
        tokens_used=tokens_used,
        metadata=meta,
    )


# --- One iteration -----------------------------------------------------------


def _stopped(
    agent: Any, state: _RunState, text: str, *,
    action: str | None, reason: str, answer: str | None = None,
) -> _StepOutcome:
    """End the run with progress and no answer, from the run's own counters."""
    return _StepOutcome("response", agent._stopped_outcome_response(
        text, action=action, reason=reason, thread=state.thread,
        iterations=state.iterations, tool_calls=state.tool_calls,
        tokens_used=state.tokens_used, calls=state.guards.calls,
        debug_trace=state.debug_trace, answer=answer,
    ))


def _written_call(
    agent: Any, state: _RunState, tool: Any, text: str,
) -> _StepOutcome:
    """End the run on a call the turn wrote out instead of making."""
    return _StepOutcome("response", agent._written_tool_call_response(
        tool, text, iterations=state.iterations, tool_calls=state.tool_calls,
        tokens_used=state.tokens_used, tool_ran=state.guards.tool_ran(tool),
        debug_trace=state.debug_trace, calls=state.guards.calls,
        thread=state.thread,
    ))


#: How many rounds of compaction one turn may take before the run is declared
#: not to fit. Each round gives up a rung and the prompt is rebuilt, so a
#: policy that is releasing anything at all converges well inside this; the cap
#: is what stops a policy that reports progress it did not make from looping.
MAX_COMPACTION_ROUNDS = 8


def _stamp_budget(state: _RunState) -> None:
    """Put the run's budget on the thread, where every terminal path reads it.

    On the thread rather than only on the response, so the key survives a
    checkpoint and a stored session turn the same way the protocol does.
    """
    state.thread.metadata["context_budget"] = (
        UNBOUNDED_CONTEXT_BUDGET if state.budget is None else state.budget.as_dict()
    )


def _compact_once(agent: Any, state: _RunState, thread: AgentThread) -> bool:
    """Give up one rung of the thread, and say what it cost.

    Args:
        agent: The agent the run belongs to.
        state: The run's conversation, budget and policy.
        thread: The run's conversation, changed in place.

    Returns:
        Whether anything was given up.
    """
    budget = state.budget
    before = (budget.stats.summarisation or {}).get("total_tokens", 0)
    if not state.compaction.compact(thread, budget):
        logger.info(
            "[context] the thread cannot be brought under the budget: "
            "%d tokens allowed, %d measured",
            budget.budget_tokens, budget.last_measured,
        )
        return False
    after = (budget.stats.summarisation or {}).get("total_tokens", 0)
    state.tokens_used += max(0, after - before)
    budget.stats.firings += 1
    logger.info(
        "[context] compacted the thread: %d tokens allowed, %d measured, "
        "%d observations shortened, %d steps dropped, about %d tokens released",
        budget.budget_tokens, budget.last_measured,
        budget.stats.observations_shortened, budget.stats.steps_dropped,
        budget.stats.tokens_dropped,
    )
    return True


def _cannot_fit(
    agent: Any, policy: _LoopPolicy, state: _RunState, thread: AgentThread
) -> _StepOutcome | None:
    """End the run on a conversation that will not fit what it may send.

    Raises the typed error when the caller asked for failures to raise, and
    otherwise hands back the equivalent failure response — the same choice
    every other failure in this loop makes.

    Args:
        agent: The agent the run belongs to.
        policy: The run's policy, for whether a failure raises.
        state: The run's conversation and counters.
        thread: The run's conversation.

    Returns:
        The terminal outcome, when the run reports rather than raises.

    Raises:
        ContextBudgetExceededError: When the run raises its failures.
    """
    from ..models.errors import ContextBudgetExceededError

    budget = state.budget
    error = ContextBudgetExceededError(
        budget_tokens=budget.budget_tokens,
        measured_tokens=budget.last_measured,
        window_tokens=budget.window_tokens,
        budget_source=budget.source,
        provider=str(_infer_provider_from_model(
            agent.model, getattr(agent, "model_name", None) or "unknown"
        ) or "effgen"),
        model_name=str(getattr(agent, "model_name", None) or ""),
    )
    if policy.raise_on_error:
        raise error
    failure = agent._generation_failure_response(
        {"metadata": {"error_detail": {
            "type": type(error).__name__,
            "category": "invalid_request",
            "provider": error.provider,
            "model": error.model_name,
            "message": error.message,
            "retryable": False,
        }}},
        iterations=state.iterations,
        tool_calls=state.tool_calls,
        tokens=state.tokens_used,
        debug_trace=state.debug_trace,
    )
    failure.metadata["thread"] = thread
    failure.metadata["prompt_protocol"] = _protocol_of(thread)
    failure.metadata["context_budget"] = _budget_of(thread)
    return _StepOutcome("response", failure)


def _note_debug_turn(state: _RunState, response: dict[str, Any], **fields: Any) -> None:
    """Append one iteration to the debug trace, when one is being collected."""
    if state.debug_trace is None:
        return
    from ..debug.inspector import DebugIteration
    state.debug_trace.iterations.append(DebugIteration(
        iteration=state.iterations,
        raw_prompt=state.prompt[:2000],
        raw_response=str(response.get("text") or "")[:2000],
        tokens_used=response.get("tokens_used", 0),
        latency=time.time() - state.iter_start,
        scratchpad_snapshot=state.thread.to_text(),
        thread_snapshot=state.thread.to_dict(),
        **fields,
    ))



def step(
    agent: Any,
    task: str,
    policy: _LoopPolicy,
    state: _RunState,
    emitter: Any,
) -> Generator[Any, None, _StepOutcome]:
    """Take one turn: prompt, generate, read, guard, dispatch, record.

    Everything the three loops used to do differently is a field of *policy* or
    a call on the guards, so there is nothing left for a second copy of this
    function to disagree about. It yields only what *emitter* yields, and on a
    blocking run that is nothing at all.

    Args:
        agent: The agent the run belongs to.
        task: The task, as the caller wrote it.
        policy: The run's policy, decided once before the first turn.
        state: The run's conversation, guards and counters.
        emitter: How this turn reaches the caller.

    Returns:
        Whether the run goes round again, or the response it ended with.
    """
    thread = state.thread
    guards = state.guards
    kwargs = policy.call_kwargs
    state.iterations += 1
    iterations = state.iterations
    state.iter_start = iter_start = time.time()
    conversation_history = state.conversation_history

    _write_periodic_checkpoint(agent, task, policy, state)

    # The frame for this turn: the run's frame, unless the guards have stopped
    # offering tools, which no frame but the text scaffold can express.
    turn_frame = frame_for(agent, tools_suppressed=guards.tools_suppressed())
    _cite_sources, _numbered_passages = agent._citation_prompt_state()
    _answer_shape = agent._answer_shape_instruction()
    gen_kwargs = dict(kwargs)
    turn_protocol = "flat"
    flat_prompt = ""
    prompt: Any = ""

    # Whether the persona and the session's earlier turns can travel as their
    # own messages this turn. A string frame has one role, so on a model whose
    # adapter does not take a conversation they stay in the text, exactly as
    # they were.
    carry_roles = agent._model_carries_a_conversation()

    def build_prompt() -> None:
        """Assemble this turn's request from the thread as it stands.

        Every frame reaches the model through this one site, so the request a
        budget measures is the request that is about to be sent rather than an
        estimate of something near it. It is called again after each round of
        compaction, which is why the two lines it logs are guarded to fire once
        per run: a turn rebuilt eight times still reports its frame once.
        """
        nonlocal prompt, turn_protocol, flat_prompt
        transcript = thread.to_text()
        # The tool contract is not a reason on its own: the string frame
        # already states it, in its own place. What a string frame cannot
        # express is the caller's own persona and the turns of a session that
        # happened before this run — those are what move the request onto roles.
        frame_roles = carry_roles and bool(
            thread.persona_text() or thread.prior_turns()
        )
        task_step = thread.task()
        frame_parts = bool(task_step.parts) if task_step is not None else False

        if turn_frame == "native":
            # Native/hybrid mode: use a simple user message and pass tool
            # definitions via the chat template's tools parameter. The model then
            # writes its calls as the tokens its own template declares.
            prompt = agent._native_tool_prompt(
                task, transcript,
                "" if frame_roles else conversation_history,
                guards.previous_actions,
                frame_owns_roles=frame_roles,
            )
            tool_defs = agent._tool_calling_strategy.format_tools_for_prompt(
                list(agent.tools.values())
            )
            if isinstance(tool_defs, list):
                gen_kwargs["tools"] = tool_defs
            if frame_roles or frame_parts:
                if not state.frame_carried_by_messages:
                    state.frame_carried_by_messages = True
                    logger.info(
                        "[frame] the run's frame travels as messages: "
                        "persona=%s earlier turns=%d content parts=%d",
                        bool(thread.persona_text()) if frame_roles else False,
                        len(thread.prior_turns()) if frame_roles else 0,
                        len(task_step.parts) if task_step is not None else 0,
                    )
                prompt = agent._frame_as_messages(
                    prompt, thread, carry_roles=frame_roles,
                )
            # The same turn, said as the conversation it was, when the caller asked
            # for that and the model declares it carries the shape. The flat prompt
            # above stays built either way: it is what a refusal falls back to, and
            # building it keeps the two renderings assembled from the same state.
            flat_prompt = prompt
            turn_protocol = agent._resolve_prompt_protocol(
                tools_travel_as_parameter="tools" in gen_kwargs,
                conversation_carries_earlier_turns=bool(thread.prior_turns()),
            )
            if turn_protocol == "messages":
                if not state.resolved_to_messages:
                    state.resolved_to_messages = True
                    logger.info("[protocol] the run sends the conversation as messages")
                prompt = agent._native_tool_messages(
                    thread, guards.previous_actions,
                    split_reasoning=(
                        agent._message_protocol_probe() == PROTOCOL_SPLIT
                    ),
                )
        elif turn_frame == "custom_template":
            turn_protocol = agent._resolve_prompt_protocol(
                tools_travel_as_parameter=False,
                conversation_carries_earlier_turns=bool(thread.prior_turns()),
            )
            # User-provided custom template
            tools_description = agent._get_tools_description()
            prompt = agent.config.system_prompt_template.format(
                tools_description=tools_description,
                conversation_history=conversation_history,
                task=task,
                scratchpad=transcript,
            )
        else:
            turn_protocol = agent._resolve_prompt_protocol(
                tools_travel_as_parameter=False,
                conversation_carries_earlier_turns=bool(thread.prior_turns()),
            )
            # ReAct mode: use enhanced ToolPromptGenerator
            prompt = agent._tool_prompt_generator.generate_react_prompt(
                task=task,
                scratchpad=transcript,
                conversation_history=conversation_history,
                system_prompt=agent.config.system_prompt,
                verbose=agent._verbose_tools,
                closing_instruction=agent._context_answer_instruction(
                    guards.previous_actions,
                    cite_sources=_cite_sources,
                    numbered_passages=_numbered_passages,
                ),
                answer_shape=_answer_shape,
                tool_contract=agent._tool_contract(),
            )
        if frame_parts and not isinstance(prompt, list):
            # A picture cannot ride in a string. The template already states the
            # persona and the earlier turns, so only the parts are added here.
            if not state.frame_carried_by_messages:
                state.frame_carried_by_messages = True
                logger.info(
                    "[frame] the run's frame travels as messages: "
                    "persona=False earlier turns=0 content parts=%d",
                    len(task_step.parts) if task_step is not None else 0,
                )
            prompt = agent._frame_as_messages(prompt, thread, carry_roles=False)

    build_prompt()
    # The conversation is brought back under the run's budget here, before
    # anything is sent: a prompt that will not fit is not a request worth
    # paying for. Each round gives up one rung of the policy's ladder and the
    # turn is rebuilt, so what is measured is always the request itself.
    if state.budget is not None:
        rounds = 0
        while state.budget.exceeded(prompt) and rounds < MAX_COMPACTION_ROUNDS:
            if not _compact_once(agent, state, thread):
                outcome = _cannot_fit(agent, policy, state, thread)
                if outcome is not None:
                    return outcome
                break
            rounds += 1
            build_prompt()
        if state.budget.exceeded(prompt):
            outcome = _cannot_fit(agent, policy, state, thread)
            if outcome is not None:
                return outcome
        _stamp_budget(state)
    if turn_protocol == "messages":
        logger.info(
            "[protocol] messages turn: %d messages, %d tool calls, "
            "%d tool results",
            len(prompt),
            *_count_tool_parts(prompt),
        )
    state.prompt = prompt

    # A turn that answered while holding a tool doing work the model cannot do
    # in its head was sent back once (see the acceptance check below); this is
    # the turn that follows, and it is required to call.
    #
    # The constraint only exists where the definitions travel as a request
    # parameter the provider enforces. On the ReAct-text path there is nothing
    # to constrain — the tools are prose in the prompt — and on an adapter that
    # does not advertise it, sending it anyway loses the turn. Both degrade to
    # the nudge already in the thread, which is the whole of the ask for them.
    #
    # The flag is spent whether or not it could be used, so a turn that could
    # not be constrained does not leak the constraint onto a later one.
    if guards.take_forced_tool_call():
        if "tools" in gen_kwargs and model_can_require_tool_call(agent.model):
            gen_kwargs["tool_choice"] = "required"
            logger.info("forced tool call: requiring a call on iteration %d",
                        iterations)
        else:
            logger.info("forced tool call: nudge only on iteration %d "
                        "(no request-level constraint available here)", iterations)

    if iterations == 1 and conversation_history:
        logger.info(
            f"[Memory] Including conversation history "
            f"({len(agent.short_term_memory.messages)} messages)"
        )

    agent.execution_tracker.track_event(ExecutionEvent(
        type=EventType.REASONING_STEP, agent_id=agent.name,
        message=f"Iteration {iterations}: Reasoning...",
        data={"iteration": iterations},
    ))

    # Whether this turn may reach the consumer while it is still being written.
    streamable = _turn_is_streamable(agent, policy, state, turn_frame)

    with start_agent_iteration(preset=agent.name, iteration=iterations):
        model_name = getattr(agent, "model_name", None) or "unknown"
        provider = _infer_provider_from_model(agent.model, model_name)
        with start_model_call(provider=provider, model=model_name) as _mspan:
            response = yield from emitter.model_turn(
                agent, prompt, policy, gen_kwargs, streamable=streamable,
            )
            # A model whose template will not take an assistant turn carrying
            # both text and a call says so by rejecting the request. That is the
            # probe: it costs nothing on a model that takes the shape, and what
            # it learns is remembered for the rest of the process rather than
            # re-learned every run. Both the retry and the fall-back are real
            # calls and are counted as such.
            if turn_protocol == "messages" and _is_request_shape_refusal(response):
                if agent._message_protocol_probe() is None:
                    logger.info(
                        "[protocol] the model refused an assistant turn "
                        "carrying both text and a tool call; retrying "
                        "with the reasoning as its own turn"
                    )
                    agent._record_message_protocol_probe(PROTOCOL_SPLIT)
                    response = yield from emitter.model_turn(
                        agent,
                        agent._native_tool_messages(
                            thread, guards.previous_actions,
                            split_reasoning=True,
                        ),
                        policy, gen_kwargs, streamable=streamable,
                    )
                if _is_request_shape_refusal(response):
                    logger.warning(
                        "[protocol] the model refused the message "
                        "protocol; the run sends the flat transcript"
                    )
                    agent._record_message_protocol_probe(PROTOCOL_REFUSED)
                    turn_protocol = "flat"
                    response = yield from emitter.model_turn(
                        agent, flat_prompt, policy, gen_kwargs,
                        streamable=streamable,
                    )
            # The provider refused a prompt our own count said would fit. Its
            # number is the better one — it has just been shown to be right and
            # ours wrong in the direction that matters — so take it, give up one
            # more rung, and send once more. A run already at what it cannot do
            # without says so with the typed error instead of paying for the
            # same refusal again.
            if state.budget is not None and _is_context_overflow(response):
                stated = _stated_context_limit(response)
                lowered = (
                    state.budget.lower_to(stated) if stated else False
                )
                if lowered:
                    logger.info(
                        "[context] the provider counted %d tokens where we "
                        "estimated %d",
                        stated or 0, state.budget.last_measured,
                    )
                if _compact_once(agent, state, thread):
                    build_prompt()
                    _stamp_budget(state)
                    response = yield from emitter.model_turn(
                        agent, prompt, policy, gen_kwargs, streamable=streamable,
                    )
                if _is_context_overflow(response):
                    state.budget.last_measured = max(
                        state.budget.last_measured,
                        count_prompt_tokens(prompt, model=agent.model),
                    )
                    outcome = _cannot_fit(agent, policy, state, thread)
                    if outcome is not None:
                        return outcome
            _meta = response.get("metadata") or {}
            _in_tok = _meta.get("prompt_tokens", 0) or 0
            _out_tok = response.get("tokens_used", 0) or 0
            _cached = _meta.get("cached_input_tokens", 0) or 0
            # The provider counted the prompt we just sent. Paired with our own
            # estimate of it, that is a correction for the next turn which
            # costs nothing and is exact one turn late.
            if state.budget is not None and _in_tok:
                state.budget.observe_reported(
                    reported=int(_in_tok),
                    estimated=count_prompt_tokens(prompt, model=agent.model),
                )
            try:
                _mspan.set_attribute(ModelAttrs.INPUT_TOKENS, int(_in_tok))
                _mspan.set_attribute(ModelAttrs.OUTPUT_TOKENS, int(_out_tok))
                if _cached:
                    _mspan.set_attribute(ModelAttrs.CACHED_TOKENS, int(_cached))
                _mspan.set_attribute(
                    ModelAttrs.OUTCOME,
                    "ok" if response.get("finish_reason") != "error" else "error",
                )
                _stamp_call_cost(_mspan, _meta)
            except Exception:
                logger.debug("Failed to set model span attributes", exc_info=True)
        # The protocol this run's turns actually went out on. A turn that fell
        # back does not un-send the ones that did not, so the first turn the
        # model took as messages settles it.
        if turn_protocol == "messages":
            thread.metadata["prompt_protocol"] = "messages"
        iter_tokens = response.get("tokens_used", 0)
        state.tokens_used += iter_tokens

    _slog.iteration_event(iterations, "generate", tokens=iter_tokens)
    _obs_log.event("agent.iteration.generate", iteration=iterations,
                   tokens=iter_tokens,
                   model=getattr(agent, "model_name", "unknown"))

    if response.get("finish_reason") == "error":
        failure = agent._generation_failure_response(
            response,
            iterations=iterations,
            tool_calls=state.tool_calls,
            tokens=state.tokens_used,
            debug_trace=state.debug_trace,
        )
        failure.metadata["thread"] = thread
        failure.metadata["prompt_protocol"] = _protocol_of(thread)
        failure.metadata["context_budget"] = _budget_of(thread)
        return _StepOutcome("response", failure)

    logger.info(
        f"[Iteration {iterations}] Raw model output: {response['text'][:300]}..."
    )
    logger.debug(f"[Iteration {iterations}] Full model output: {response['text']}")

    # Parse response using strategy. If the adapter returned a native tool call
    # (empty text + structured tool_calls in metadata), use it directly — no
    # text parsing needed.
    native_tool_calls = response.get("tool_calls") or []

    # Whether this turn's own text was written before any observation this turn
    # produced. A batch of calls is dispatched after the model has finished
    # writing, so its text cannot state a result the batch returned; the answer
    # recovery below has to know that.
    dispatched_calls_this_turn = False

    if len(native_tool_calls) > 1 and agent.tools:
        batch_observations: list[str] = []
        # What the model said while asking for the batch. It explains the whole
        # batch, so it goes on the first call of it — the turn the model was
        # speaking on.
        _batch_reasoning = str(response.get("text") or "")
        yield from emitter.thought(_batch_reasoning.strip())
        for _tc in native_tool_calls:
            _fn = _tc.get("function", _tc)
            _tname = _fn.get("name", "")
            _targs = _fn.get("arguments", {})
            if isinstance(_targs, str):
                try:
                    _targs = json.loads(_targs)
                except (json.JSONDecodeError, TypeError):
                    _targs = {"__raw_input__": _targs}
            _call_id = _tc.get("id") or None
            _declined: str | None = None
            yield from emitter.tool_call(_tname, json.dumps(_targs, default=str))
            if _tname in agent.tools:
                _batch_start = time.time()
                with start_tool_call(
                    tool_name=_tname, tool_input=str(_targs)[:500]
                ) as _btspan:
                    _obs = agent._execute_tool(_tname, json.dumps(_targs))
                    try:
                        _btspan.set_attribute(ToolAttrs.STATUS, "ok")
                    except Exception:
                        logger.debug(
                            "Failed to set tool span status", exc_info=True
                        )
                _batch_elapsed = time.time() - _batch_start
                state.tool_calls += 1
                # The record carries what the call returned and how long it
                # took, as it does on the one-call-per-turn path: a batched call
                # is a call the run made, and a record with no result in it
                # cannot say what the run was holding when it answered.
                guards.record_execution(
                    _tname,
                    arguments=_targs,
                    result=_obs,
                    duration=_batch_elapsed,
                    iteration=iterations,
                )
                batch_observations.append(f"[{_tname}({_targs})] → {_obs}")
            else:
                # A call naming a tool this agent does not hold is still a call
                # the model made. Answering it says which tools are callable;
                # dropping it left the turn with a call nothing replied to and
                # the model with no idea its request had been refused.
                _declined = "unknown_tool"
                _obs = (
                    unknown_tool_observation(_tname, list(agent.tools))
                    if agent.tools
                    else NUDGE_NO_TOOLS
                )
                logger.info(
                    "[Batch] '%s' is not a tool this agent holds; the "
                    "call is answered rather than dropped",
                    _tname,
                )
                batch_observations.append(f"[{_tname}] → {_obs}")
            thread.append(
                ActionStep(
                    tool=_tname,
                    arguments=dict(_targs) if isinstance(_targs, dict) else {},
                    raw=json.dumps(_targs),
                    call_id=_call_id,
                    reasoning=_batch_reasoning,
                )
            )
            _batch_reasoning = ""
            thread.append(
                ObservationStep(
                    text=str(_obs), call_id=_call_id,
                    is_error=_declined is not None, declined=_declined,
                )
            )
            yield from emitter.observation(_tname, str(_obs))
        # After batch execution, nudge model to synthesize a final answer.
        thread.append(
            NudgeStep(text=NUDGE_CONTINUE, render_as="raw", nudge_id="continue")
        )
        guards.note_batch_run()
        parsed = {
            "thought": "", "action": None, "action_input": None,
            "final_answer": None,
        }
        dispatched_calls_this_turn = True
        cur_observation: Any = "\n".join(batch_observations)
        logger.info(
            f"[Batch native tool calls] {len(native_tool_calls)} calls "
            f"executed (batch run #{guards.batch_tool_runs})"
        )
    elif native_tool_calls:
        strategy_result = agent._parse_native_tool_calls(
            native_tool_calls, response.get("text") or "",
        )
        parsed = agent._tool_call_result_to_dict(strategy_result)
    else:
        parse_strategy = agent._text_parse_strategy(turn_frame == "native")
        strategy_result = parse_strategy.parse_response(
            response["text"], tools=agent.tools,
        )
        parsed = agent._tool_call_result_to_dict(strategy_result)

    logger.info(
        f"[Iteration {iterations}] Parsed - Action: {parsed.get('action')}, "
        f"Input: {parsed.get('action_input')}, Final: {parsed.get('final_answer')}"
    )

    # Record the turn's own reasoning. A turn that made a native tool call
    # reports no thought, and the transcript is prompt text the model reads back
    # — so an absent thought renders as the bare label and an empty line, never
    # as the word "None".
    thread.append(ThoughtStep(text=parsed.get("thought") or ""))
    if not dispatched_calls_this_turn:
        yield from emitter.thought(
            str(parsed.get("thought") or parsed.get("reasoning") or "").strip()
        )

    # Capture debug iteration data
    cur_observation = None  # filled later if tool runs

    snapshot = _TurnSnapshot(
        tokens_used=state.tokens_used,
        iterations=iterations,
        tool_calls=state.tool_calls,
        iter_start=iter_start,
        thread=AgentThread(steps=list(thread.steps)),
    )

    # Check for final answer
    final_answer: Any = parsed.get("final_answer")
    if final_answer and state.tool_calls > 0 and final_answer.strip().lower() in {
        "none", "null", "n/a", "na",
    }:
        partial = agent._extract_partial_answer(thread)
        if partial:
            return _stopped(
                agent, state, sanitize_final_answer(partial) or partial,
                action=guards.calls[-1].name if guards.calls else None,
                reason="null_final_from_model", answer=final_answer,
            )

    # A "final answer" that is purely leaked tool-call syntax / scaffolding
    # (sanitizes to nothing) is not a real answer — keep looping so the tool
    # actually runs or a partial is extracted. When what leaked is a call for a
    # tool this agent holds, the model is writing the call instead of making it:
    # nudge once, then report it rather than billing the rest of the iteration
    # budget for the same outcome.
    if final_answer and not (sanitize_final_answer(final_answer) or "").strip():
        written = find_written_tool_call(final_answer, agent.tools)
        if written and guards.is_unmade_call(written, final_answer):
            if guards.note_written_call(written):
                return _written_call(
                    agent, state, guards.written_call, final_answer,
                )
        logger.info("Discarding scaffolding-only final answer; continuing loop")
        thread.append(
            NudgeStep(
                text=NUDGE_NOT_USABLE,
                render_as="observation",
                nudge_id="not_usable",
            )
        )
        final_answer = None

    if final_answer:
        # An agent holding a tool that does work the model cannot do in its head
        # has not answered by saying what the tool would have returned: nothing
        # computed that result. The first such answer is not accepted — the turn
        # goes back naming the tool, and the turn after it is required to call
        # one where that can be required. Only the first: a model that declines
        # twice will decline again, and the iterations buy more elsewhere.
        refused_tool = guards.note_execution_refusal()
        if refused_tool is not None:
            thread.append(
                NudgeStep(
                    text=NUDGE_MUST_EXECUTE.format(tool=refused_tool),
                    render_as="observation",
                    nudge_id="must_execute",
                )
            )
            return CONTINUE

        # A run whose search came back without what the question asked for has
        # one more query to spend before that answer is taken.
        if _requery(agent, state, policy, final_answer):
            if emitter.emits_deltas:
                emitter.reset_answer()
            return CONTINUE

        _note_debug_turn(
            state, response, thought=parsed.get("thought", ""),
            final_answer=final_answer,
        )
        return _StepOutcome("response", build_response(
            agent, policy, state, snapshot, final_answer,
        ))

    # Check if model is stating an answer without "Final Answer:" keyword. This
    # happens when model provides result after tool execution.
    #
    # Only when the model has actually seen a result. A turn that dispatched its
    # own calls wrote its text first and the observations came back after, so
    # that text states a plan, not an answer — and accepting it throws away
    # every result the turn just fetched.
    if (
        state.tool_calls > 0
        and not parsed.get("action")
        and not dispatched_calls_this_turn
    ):
        response_text = response["text"].strip()
        if any(
            phrase in response_text.lower()
            for phrase in [
                "the answer is", "the result is", "the sum is", "equals", "=",
            ]
        ):
            if _requery(agent, state, policy, response_text):
                if emitter.emits_deltas:
                    emitter.reset_answer()
                return CONTINUE
            logger.info(
                "Detected answer statement without 'Final Answer:' keyword"
            )
            _note_debug_turn(
                state, {"text": response_text, "tokens_used": iter_tokens},
                thought=parsed.get("thought", ""), final_answer=response_text,
            )
            return _StepOutcome("response", build_response(
                agent, policy, state, snapshot, response_text,
            ))

    # Execute action if present
    if parsed.get("action") and parsed.get("action_input"):
        action = str(parsed["action"])
        action_input: Any = parsed["action_input"]
        # The provider's own id for this call and the words the model said
        # beside it. Both are empty on the ReAct-text path, where the thought is
        # already its own step and no provider minted an id, so the transcript
        # is unchanged there.
        call_id = parsed.get("call_id") or None
        reasoning = str(parsed.get("reasoning") or "")

        yield from emitter.tool_call(action, str(action_input))

        # Repeat detection: the same call again, or the same tool enough times
        # with drifting inputs that it reads as a loop.
        check = guards.check_action(action, action_input)
        action_call_count = check.action_call_count
        # An exact repeat of a call that already succeeded is answered from the
        # record. A pure computation is idempotent, so running it again returns
        # what it returned before; the record supplies that, and the run carries
        # on with the step it was on.
        replay = None
        if check.is_exact_loop and not check.is_fuzzy_loop:
            replay = guards.cached_result(check)
        if replay is not None:
            logger.info(
                "[Repeat] '%s' was already called with this input; "
                "replaying the recorded result instead of ending the run",
                action,
            )
            thread.append(ActionStep(
                tool=action, raw=str(action_input),
                call_id=call_id, reasoning=reasoning,
            ))
            # The reply is the tool's own recorded result, not the framework's
            # words, so the observation is not a decline.
            thread.append(ObservationStep(text=str(replay), call_id=call_id))
            yield from emitter.observation(action, str(replay))
            cur_observation = replay
            nudge = guards.post_tool_nudge(iterations, action_call_count, replay)
            if nudge:
                thread.append(
                    NudgeStep(
                        text=str(nudge), render_as="raw", nudge_id="post_tool"
                    )
                )
            return CONTINUE
        if check.is_loop:
            logger.info(
                f"[Loop detected] Repeated action '{action}' ({check.loop_type}) — "
                f"the run stops offering this tool"
            )
            # Read the last successful observation out of the thread
            partial = agent._extract_partial_answer(thread)
            # What a tool returned is not an answer, whatever the tool was: a
            # retrieved passage is source material, and a computed number is
            # usually an intermediate one, so handing either back loses the
            # question it belonged to. The model already has both in the
            # transcript. Stop offering tools and spend one turn asking it to
            # state the answer from what it has, before falling back to the
            # progress itself.
            if partial and not guards.force_text_answer:
                logger.info(
                    "[Loop synthesis] '%s' is repeating; asking for an "
                    "answer stated from the observations so far",
                    action,
                )
                guards.force_text_answer = True
                _decline_call(
                    thread, action, action_input,
                    call_id=call_id, reasoning=reasoning,
                    reason="loop_detected", text=NUDGE_HAVE_RESULTS,
                )
                return CONTINUE
            if partial:
                # The run ends on one of the two branches below, and the call
                # the model just made is part of what happened: recording it
                # with the reply it got leaves the conversation with no call
                # nothing answered.
                _decline_call(
                    thread, action, action_input,
                    call_id=call_id, reasoning=reasoning,
                    reason="loop_detected", text=NUDGE_HAVE_RESULTS,
                )
            if partial and agent._is_context_retrieval_tool(action):
                return _stopped(
                    agent, state, partial, action=action, reason="loop_detected",
                )
            if partial:
                # The tool's results are not an answer either: the model never
                # wrote one, so the run reports that it stopped and carries what
                # the tools returned as partial progress.
                return _stopped(
                    agent, state, sanitize_final_answer(partial) or partial,
                    action=action, reason="loop_detected",
                )
            # No partial answer to fall back on — every attempt of this action
            # failed or was denied, so simply nudging and re-offering the same
            # tool just repeats the loop until max_iterations. Stop offering
            # tools for the rest of this run so the model must respond in prose.
            guards.force_text_answer = True
            _decline_call(
                thread, action, action_input,
                call_id=call_id, reasoning=reasoning,
                reason="already_computed", text=NUDGE_ALREADY_COMPUTED,
            )
            return CONTINUE

        guards.record_action(check)

        # Check if tool is available (handle no-tool mode without raising)
        if not agent.tools or action not in agent.tools:
            # The action names no tool the agent holds. With tools attached, say
            # which ones are callable — telling a model that owns a calculator
            # there are "no tools available" sends it off to do the work itself.
            # With no tools at all, answering directly is the only option left.
            observation = (
                unknown_tool_observation(action, list(agent.tools))
                if agent.tools
                else NUDGE_NO_TOOLS
            )
            thread.append(ActionStep(
                tool=action, raw=str(action_input),
                call_id=call_id, reasoning=reasoning,
            ))
            thread.append(ObservationStep(
                text=str(observation), call_id=call_id, declined="unknown_tool",
            ))
            yield from emitter.observation(action, str(observation))
        else:
            # Execute tool inside tracing span
            tool_start = time.time()
            with start_tool_call(
                tool_name=action, tool_input=str(action_input)
            ) as _tspan:
                tool_result = agent._execute_tool(action, action_input)
                try:
                    _tspan.set_attribute(ToolAttrs.STATUS, "ok")
                except Exception:
                    logger.debug("Failed to set tool span status", exc_info=True)
            tool_elapsed = time.time() - tool_start
            state.tool_calls += 1
            guards.record_execution(
                action,
                arguments=action_input,
                result=tool_result,
                duration=tool_elapsed,
                iteration=iterations,
            )
            # Keep the result against the exact call that produced it, so
            # proposing that call again is answered from the record.
            guards.record_pair_result(check, tool_result)
            cur_observation = tool_result

            labels = {"tool_name": action, "agent_name": agent.name}
            prom_metrics.tool_calls.inc(labels=labels)
            prom_metrics.tool_execution_time.observe(tool_elapsed, labels=labels)
            _slog.tool_event(action, "executed", latency=tool_elapsed)
            _obs_log.tool_event("executed", tool=action,
                                latency_ms=round(tool_elapsed * 1000, 1))

            # Record the call and what it returned
            thread.append(ActionStep(
                tool=action, raw=str(action_input),
                call_id=call_id, reasoning=reasoning,
            ))
            thread.append(ObservationStep(text=str(tool_result), call_id=call_id))
            yield from emitter.observation(action, str(tool_result))

            logger.info(
                f"Tool result added to the thread: {tool_result[:100]}..."
            )

            if agent._should_return_direct_calculator_result(
                task, action, action_input
            ):
                logger.info(
                    "Returning direct calculator result for simple arithmetic task"
                )
                # A turn that ends the run here is still a turn the model took,
                # so the debug trace records it rather than reporting a run with
                # no iterations at all.
                _note_debug_turn(
                    state, response, thought=reasoning, action=action,
                    action_input=str(action_input), observation=str(tool_result),
                    final_answer=tool_result,
                )
                return _StepOutcome("response", build_response(
                    agent, policy, state, snapshot, tool_result,
                    tool_calls=state.tool_calls,
                    answer_source="direct_calculator_result",
                ))

            # Result-based short-circuit: a model often re-derives a result it
            # already has (e.g. "15^2" then "15*15", both 225) with slightly
            # different inputs, so the exact-input loop guard never fires. A
            # tool that reproduces its own output means the model is re-deriving
            # rather than moving on, and re-offering it produces the same turn.
            if guards.result_is_repeat(action, tool_result):
                # What the tool returned is not the answer, whatever the tool
                # is: a retrieved passage is source material, and a repeated
                # number is usually an intermediate one. The observation is in
                # the transcript, so stop offering tools and give the model one
                # turn to state the answer from it, falling back to the progress
                # itself only when that turn produces nothing.
                if not guards.force_text_answer:
                    logger.info(
                        "[Loop synthesis] Tool '%s' repeated a result; "
                        "asking for an answer stated from it",
                        action,
                    )
                    guards.force_text_answer = True
                    thread.append(
                        NudgeStep(
                            text=NUDGE_HAVE_RESULTS,
                            render_as="raw",
                            nudge_id="have_results",
                        )
                    )
                    return CONTINUE
                logger.info(
                    "[Loop efficiency] Tool '%s' reproduced an identical "
                    "result; stopping the run",
                    action,
                )
                if agent._is_context_retrieval_tool(action):
                    return _stopped(
                        agent, state, tool_result, action=action,
                        reason="repeated_tool_result",
                    )
                # A tool that recomputes a number it already returned has not
                # answered the question the caller asked — the model never wrote
                # the answer up — so the run reports that it stopped and carries
                # the result as partial progress.
                return _stopped(
                    agent, state, sanitize_final_answer(tool_result) or tool_result,
                    action=action, reason="repeated_tool_result",
                )
            guards.record_result(action, tool_result)

            nudge = guards.post_tool_nudge(
                iterations, action_call_count, tool_result
            )
            if nudge:
                thread.append(
                    NudgeStep(
                        text=str(nudge), render_as="raw", nudge_id="post_tool"
                    )
                )

    else:
        # A turn that produced neither an action nor an answer, but did write
        # out a call for a tool this agent holds, is the same failure the answer
        # path reports: the model is writing the call instead of making it. Say
        # so once, and on a second such turn report the cause rather than
        # grinding to the iteration cap and reporting only that.
        written = find_written_tool_call(response["text"], agent.tools)
        if written and guards.is_unmade_call(written, response["text"]):
            if guards.note_written_call(written):
                return _written_call(
                    agent, state, guards.written_call, response["text"],
                )
            thread.append(
                NudgeStep(
                    text=NUDGE_NOT_USABLE,
                    render_as="observation",
                    nudge_id="not_usable",
                )
            )
        # No action specified, prompt to continue
        thread.append(
            NudgeStep(
                text=CONTINUE_REASONING_LINE,
                render_as="raw",
                nudge_id="continue_reasoning",
            )
        )

    _note_debug_turn(
        state, response, thought=parsed.get("thought", ""),
        action=parsed.get("action"), action_input=parsed.get("action_input"),
        observation=cur_observation,
    )
    return CONTINUE


# --- The driver --------------------------------------------------------------


def _frame_steps(agent: Any, task: str, policy: _LoopPolicy) -> list[Step]:
    """The steps a run starts with: how it is framed, and what it was asked.

    The persona and the tool contract are system steps, the session's earlier
    messages are the turns they were, and the question is a task step carrying
    any content parts it arrived with. None of them renders into the flat
    transcript, so a run that carries none of them sends the bytes it always
    did; a run that carries some of them can state them as roles instead of
    pasting them into the question.

    Args:
        agent: The agent the run belongs to.
        task: The task, as the caller wrote it.
        policy: The run's policy, which holds the task's content parts.

    Returns:
        The steps, in the order they belong in.
    """
    steps: list[Step] = []
    persona = getattr(agent, "_custom_persona", None)
    if persona:
        steps.append(SystemStep(text=str(persona), source="persona"))
    if agent.tools:
        contract = agent._tool_contract()
        if contract:
            steps.append(SystemStep(text=contract, source="contract"))
    prior = agent._prior_turn_steps()
    if prior:
        logger.info(
            "[thread] the run carries %d earlier turn(s) of this session", len(prior)
        )
        steps.extend(prior)
    if policy.prior_steps:
        logger.info(
            "[thread] the run opens with %d step(s) projected from its parent",
            len(policy.prior_steps),
        )
        steps.extend(policy.prior_steps)
    steps.append(TaskStep(text=task, parts=list(policy.task_parts)))
    return steps


def drive(
    agent: Any,
    task: str,
    policy: _LoopPolicy,
    emitter: Any,
) -> Generator[Any, None, AgentResponse]:
    """Run *task* to a terminal response, yielding whatever *emitter* yields.

    The generator's return value is the run's
    :class:`~effgen.core.agent_response.AgentResponse` — the same object on
    every path. :func:`run_to_completion` is how a blocking caller reads it.

    Args:
        agent: The agent the run belongs to.
        task: The task, as the caller wrote it.
        policy: The run's policy, decided once before the first turn.
        emitter: How each turn reaches the caller.

    Returns:
        The run's terminal response.
    """
    state = _RunState(
        thread=AgentThread(steps=_frame_steps(agent, task, policy)),
        # The repeat guards — which calls have been dispatched, which results
        # have already come back, when to stop offering tools and when a
        # written-out call has been seen once too often. One construction site
        # for every path, so all of them reach the same decisions.
        guards=NativeToolLoop(
            agent.tools,
            nudge_cap=agent.config.max_iterations,
            tool_use=agent._declared_tool_use(),
        ),
    )
    # Which protocol the run's turns went out on. Stamped here so every response
    # carries the key whatever ended the run, and raised to "messages" by the
    # first turn the model took that way.
    state.thread.metadata["prompt_protocol"] = "flat"
    # How many prompt tokens this run may send, and what it gives up to stay
    # inside that. Resolved once, before the first turn, so every turn of one
    # run is measured against the same ceiling.
    state.budget = resolve_context_budget(
        agent.config.context_budget,
        model=agent.model,
        window_override=agent.config.max_context_length,
        output_tokens=policy.gen_config.max_tokens,
    )
    state.compaction = resolve_policy(agent.config.compaction)
    if state.budget is not None:
        logger.info(
            "[context] the run is bounded at %d tokens (window %s, source %s)",
            state.budget.budget_tokens,
            state.budget.window_tokens,
            state.budget.source,
        )
    _stamp_budget(state)
    if policy.debug:
        from ..debug.inspector import DebugTrace
        state.debug_trace = DebugTrace(
            task=task, agent_name=agent.name, run_id=policy.run_id,
        )
    # The earlier turns as text, for a frame that can only take one string.
    # They are steps on the thread either way; this is the rendering a prompt
    # template's conversation-history field receives.
    state.conversation_history = state.thread.history_text()
    if policy.checkpoint_interval and policy.checkpoint_dir:
        try:
            from .checkpoint import CheckpointManager as _CM
            state.checkpoints = _CM(policy.checkpoint_dir)
        except Exception as _e:
            logger.warning("Failed to init CheckpointManager: %s", _e)
    # A resumed run continues from the steps its checkpoint stored. The frame
    # this run built stays at the front; only the steps the earlier run took are
    # carried over, so a resume onto a different agent is framed by that agent.
    resumed = _resumed_steps(policy)
    if resumed:
        state.thread.extend(resumed)
        logger.info("[thread] resumed a run from %d saved steps", len(resumed))
        if policy.resume_thread is None and state.thread.to_text() != (
            policy.resume_scratchpad or ""
        ):
            logger.info(
                "[thread] resumed transcript does not begin at a step "
                "boundary; the run continues from what could be read"
            )

    while state.iterations < policy.max_iterations:
        outcome = yield from step(agent, task, policy, state, emitter)
        if outcome.kind == "response":
            assert outcome.response is not None
            return outcome.response

    return _iteration_cap_response(agent, policy, state)


def run_to_completion(driver: Generator[Any, None, AgentResponse]) -> AgentResponse:
    """Drain a driver that yields nothing and hand back its response."""
    try:
        while True:
            next(driver)
    except StopIteration as stop:
        response: AgentResponse = stop.value
        return response


def _iteration_cap_response(
    agent: Any, policy: _LoopPolicy, state: _RunState
) -> AgentResponse:
    """Report a run whose iteration budget ran out before an answer.

    When every turn wrote its tool call out as text and nothing ran, the cap is
    a symptom: the cause is reported instead.
    """
    thread = state.thread
    guards = state.guards
    partial_answer = agent._extract_partial_answer(thread)
    if guards.written_call and not partial_answer:
        outcome = _written_call(agent, state, guards.written_call, "")
        reported: AgentResponse | None = outcome.response
        if reported is not None:
            return reported
    # The run stopped without a final answer. Whatever the thread holds is a
    # tool observation or a half-finished thought — source material, not
    # something the model wrote as its answer — so it is reported as progress
    # under ``partial_output`` and the outcome itself states what happened and
    # what to do about it.
    if partial_answer:
        partial_answer = sanitize_final_answer(partial_answer) or partial_answer
    detail = agent._iteration_cap_detail(policy.max_iterations, partial_answer)
    reason = (
        "max_iterations_partial" if partial_answer else "max_iterations_exhausted"
    )
    thread.append(AnswerStep(text=partial_answer or "", stop_reason=reason))
    meta = _terminal_meta(agent, thread, reason)
    meta["error"] = detail
    cap_partial = None
    if partial_answer:
        cap_partial = agent._partial_result(
            thread,
            text=partial_answer,
            calls=guards.calls,
            iterations=state.iterations,
            tool_calls=state.tool_calls,
        )
        meta["partial"] = True
        meta["partial_output"] = partial_answer
    logger.info(
        "outcome stopped: stop_reason=%s observations=%d",
        reason,
        len(cap_partial.observations) if cap_partial else 0,
    )
    if state.debug_trace is not None:
        state.debug_trace.total_tokens = state.tokens_used
        state.debug_trace.final_answer = None
        state.debug_trace.success = False
        meta["debug_trace"] = state.debug_trace
    return AgentResponse(
        output=detail["message"],
        success=False,
        mode=AgentMode.SINGLE,
        iterations=state.iterations,
        tool_calls=ToolCallList(list(guards.calls), total=state.tool_calls),
        tokens_used=state.tokens_used,
        metadata=meta,
        stop_reason=reason,
        partial=cap_partial,
    )


# --- Pieces the loop leans on ------------------------------------------------


def _turn_is_streamable(
    agent: Any, policy: _LoopPolicy, state: _RunState, turn_frame: str
) -> bool:
    """Whether this turn's text may reach the consumer as it is written.

    Three conditions, and the log line names the one that failed. The turn's
    tool definitions must travel to the provider, the adapter must record the
    calls it streams — otherwise a call written into the text cannot be told
    from an answer — and no check may still be able to revise the turn.
    """
    if not policy.emit_deltas:
        return False
    if turn_frame != "native":
        logger.info(
            "[loop] turn %d accumulated rather than streamed: the %s frame is "
            "read as a whole before any of it is an answer",
            state.iterations, turn_frame,
        )
        return False
    probe = getattr(agent.model, "streams_tool_calls", None)
    try:
        streams_calls = bool(probe and probe())
    except Exception:  # noqa: BLE001 - a capability probe never breaks a run
        logger.debug("streams_tool_calls probe failed", exc_info=True)
        streams_calls = False
    if not streams_calls:
        logger.info(
            "[loop] the model does not stream its tool calls; the turn is "
            "accumulated"
        )
        return False
    guards = state.guards
    # A turn that could still be sent back to search is accumulated rather than
    # streamed. The decision needs the whole answer, and on this path tokens
    # reach the consumer as they arrive — so a turn judged after the fact would
    # already be on screen, and neither withdrawing it nor emitting a second
    # answer is something a consumer can be asked to handle.
    if (
        guards.retrieval_requeries == 0
        and not guards.tools_suppressed()
        and policy.max_iterations - state.iterations >= REQUERY_MIN_ITERATIONS_LEFT
        and bool(guards.calls)
        and agent._is_context_retrieval_tool(guards.calls[-1].name)
    ):
        logger.info(
            "[loop] turn %d accumulated rather than streamed: the run may "
            "still be sent back to search", state.iterations,
        )
        return False
    return True


def _requery(agent: Any, state: _RunState, policy: _LoopPolicy, answer: str) -> bool:
    """Send this run back for one more search, or leave the answer alone.

    True when the answer was not accepted and the thread now asks for a
    different query. Called from every point that would otherwise accept an
    answer, so which of them the model happened to reach — with an answer label
    or without one — does not decide whether the run gets its second search.
    """
    guards = state.guards
    if not should_requery(
        sanitize_final_answer(answer) or answer,
        guards.calls,
        agent._is_context_retrieval_tool,
        tools_suppressed=guards.tools_suppressed(),
        iterations_left=policy.max_iterations - state.iterations,
        requery_spent=guards.retrieval_requeries > 0,
    ):
        return False
    if not guards.take_retrieval_requery():
        return False
    state.thread.append(
        NudgeStep(
            text=NUDGE_SEARCH_AGAIN,
            render_as="observation",
            nudge_id="search_again",
        )
    )
    if REQUERY_FORCES_TOOL_CALL:
        guards.force_tool_call = True
    return True


def _resumed_steps(policy: _LoopPolicy) -> list[Any]:
    """The steps a resumed run carries over, from whichever shape it was given.

    A checkpoint this release writes carries the run's steps, so they come back
    as they were. One written before a run's steps were kept carries only the
    transcript; it is read back, which is lossy in the ways
    :func:`effgen.core._compat.thread_from_saved` documents but renders the same
    bytes, so the resumed run sees the prompt it would have seen.

    The frame steps a saved thread opens with are left behind: the run being
    resumed builds its own frame from the agent it is resuming onto, and two
    personas in one prompt is not the conversation either run had.
    """
    from .thread import ActionStep, NudgeStep, ObservationStep, ThoughtStep

    saved: Any = None
    if policy.resume_thread is not None:
        from ._compat import thread_from_saved

        saved = thread_from_saved({"thread": policy.resume_thread}, label="resume")
    elif policy.resume_scratchpad:
        saved = AgentThread.from_scratchpad(policy.resume_scratchpad)
    if saved is None:
        return []
    carried = (ThoughtStep, ActionStep, ObservationStep, NudgeStep)
    return [step for step in saved.steps if isinstance(step, carried)]


def _write_periodic_checkpoint(
    agent: Any, task: str, policy: _LoopPolicy, state: _RunState
) -> None:
    """Save the run's transcript every ``checkpoint_interval`` iterations."""
    interval = policy.checkpoint_interval
    if state.checkpoints is None or state.iterations <= 1 or not interval:
        return
    if (state.iterations - 1) % interval != 0:
        return
    try:
        from .checkpoint import CheckpointManager as _CM2
        cp = _CM2.snapshot_agent(
            agent,
            task=task,
            iteration=state.iterations,
            thread=state.thread,
            tool_calls=state.tool_calls,
            tokens_used=state.tokens_used,
            metadata={"interval": interval},
        )
        agent._last_checkpoint_id = state.checkpoints.save(cp)
    except Exception as _e:
        logger.warning("Periodic checkpoint failed: %s", _e)


def _decline_call(
    thread: AgentThread,
    tool: str,
    action_input: Any,
    *,
    call_id: str | None,
    reasoning: str,
    reason: str,
    text: str,
) -> None:
    """Record a call the loop chose not to dispatch, with the reply it got.

    The loop declines a call for three reasons: it has the result already, the
    same call keeps coming back, or the tool is not one this agent holds. In
    each case the model asked for something and is owed an answer — and a
    conversation carrying a call that nothing replies to is rejected outright
    by a provider, so the reply is not optional once the run is held as
    messages.

    Args:
        thread: The run's conversation.
        tool: The tool the turn named.
        action_input: The input the turn wrote, as it wrote it.
        call_id: The provider's id for the call, when the turn carried one.
        reasoning: What the model said beside the call.
        reason: Why the call was declined, kept on the observation.
        text: The reply the model reads.
    """
    logger.info("[Declined call] '%s' was not dispatched: %s", tool, reason)
    thread.append(ActionStep(
        tool=tool, raw=str(action_input), call_id=call_id, reasoning=reasoning,
    ))
    thread.append(ObservationStep(text=text, call_id=call_id, declined=reason))


def _terminal_meta(agent: Any, thread: AgentThread, reason: str) -> dict[str, Any]:
    """The metadata every terminal response carries, whatever ended the run."""
    task = thread.task()
    return {
        "reason": reason,
        "tool_calling_strategy": agent._tool_calling_strategy.name,
        "thread": thread,
        "prompt_protocol": _protocol_of(thread),
        # Always present, the way the protocol is: a reader should not have to
        # know whether a run was bounded to ask whether it was.
        "context_budget": _budget_of(thread),
        # A run that carries a picture or a recording says so whether or not the
        # agent holds tools: a caller reading the key should not have to know
        # which path answered it.
        "multimodal_inputs": bool(task is not None and task.parts),
    }


def _budget_of(thread: AgentThread) -> dict[str, Any]:
    """What the run's budget did, as every terminal path reports it.

    Read off the thread, which is where the loop stamps it, so a response built
    somewhere other than the loop carries the same mapping without having to be
    handed the budget object.

    Args:
        thread: The run's conversation.

    Returns:
        The budget mapping, or the one a run with no budget in force reports.
    """
    return dict(thread.metadata.get("context_budget") or UNBOUNDED_CONTEXT_BUDGET)


def _protocol_of(thread: AgentThread) -> str:
    """Which protocol a run's turns went out on.

    Args:
        thread: The run's conversation.

    Returns:
        ``"messages"`` when at least one turn reached the model as a message
        list, ``"flat"`` otherwise.
    """
    return str(thread.metadata.get("prompt_protocol") or "flat")


def _failure_message(response: dict[str, Any]) -> str:
    """What a failed turn said, as the generation layer recorded it."""
    if response.get("finish_reason") != "error":
        return ""
    meta = response.get("metadata") or {}
    detail = meta.get("error_detail") or {}
    return str(detail.get("message") or meta.get("error") or "")


def _is_context_overflow(response: dict[str, Any]) -> bool:
    """Whether a turn failed because its prompt was larger than the window.

    Read through the same phrase list the rest of effGen reads a too-long
    prompt with, so the loop and the error surface agree about what one is.

    Args:
        response: What the generation layer returned for the turn.

    Returns:
        True when the failure was the prompt not fitting.
    """
    from ..models.errors import _CONTEXT_WINDOW_SIGNALS

    message = _failure_message(response).lower()
    return bool(message) and any(s in message for s in _CONTEXT_WINDOW_SIGNALS)


#: What a provider says the model would have accepted. Every wording this
#: matches states the window before it states what was asked for, so the first
#: number is the one to believe.
_STATED_LIMIT_RE = re.compile(
    r"(?:maximum context length is|context length \()\s*(\d{3,})", re.IGNORECASE
)


def _stated_context_limit(response: dict[str, Any]) -> int | None:
    """The window a refusal named, when it named one.

    Args:
        response: What the generation layer returned for the turn.

    Returns:
        The token count the provider stated, or ``None``.
    """
    match = _STATED_LIMIT_RE.search(_failure_message(response))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - the pattern only matches digits
        return None


def _is_request_shape_refusal(response: dict[str, Any]) -> bool:
    """Whether a failed turn failed because the provider refused the request.

    An invalid-request failure is the one a differently shaped request could
    fix. Auth, a missing model, a rate limit and a transport failure are not,
    and retrying them in another shape only spends the budget again.

    Args:
        response: What the generation layer returned for the turn.

    Returns:
        True when the failure was the request itself being refused.
    """
    if response.get("finish_reason") != "error":
        return False
    detail = (response.get("metadata") or {}).get("error_detail") or {}
    return str(detail.get("category", "")) == "invalid_request"
