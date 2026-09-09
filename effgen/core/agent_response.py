"""Result-assembly types for :class:`effgen.core.agent.Agent`.

Holds :class:`AgentResponse`, the object every ``run()`` returns (the answer
text, success flag, token/cost/timing metadata, execution trace, and the dict/
render helpers), and :class:`StreamEvent`, the typed record
``stream(..., include_events=True)`` yields. This module imports only
:class:`AgentMode` from the config leaf and references
:class:`~effgen.core.router.RoutingDecision` under ``TYPE_CHECKING``, so it does
not import ``agent.py``; ``agent.py`` re-exports both names above the
generation/react mixin imports, keeping ``from effgen.core.agent import
AgentResponse, StreamEvent`` unchanged. Behaviour is identical to the original
in-module definitions.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from .agent_config import AgentMode
from .tool_call_record import ToolCall, ToolCallList, coerce_tool_calls

if TYPE_CHECKING:
    from .router import RoutingDecision

__all__ = [
    "STOPPED_REASONS",
    "STOP_REASONS",
    "AgentResponse",
    "PartialResult",
    "StreamEvent",
    "ToolCall",
    "ToolCallList",
]

#: Every value :attr:`AgentResponse.stop_reason` can take. The vocabulary is
#: closed: a run either answered (``final_answer``), was stopped by the loop
#: before the model wrote an answer, or failed outright.
STOP_REASONS = (
    # answered
    "final_answer",
    # stopped — the loop ended the run before the model wrote an answer
    "max_iterations_partial",
    "max_iterations_exhausted",
    "loop_detected",
    "repeated_tool_result",
    "null_final_from_model",
    # failed — the run could not be carried out
    "written_tool_call",
    "generation_failed",
    "structured_output_failed",
    "empty_task",
    "guardrail_blocked",
    "run_failed",
    "sub_agent_failed",
)

#: The subset of :data:`STOP_REASONS` that means the loop stopped a run which
#: was otherwise proceeding. These are the responses whose
#: :attr:`AgentResponse.outcome` is ``"stopped"``.
STOPPED_REASONS = frozenset({
    "max_iterations_partial",
    "max_iterations_exhausted",
    "loop_detected",
    "repeated_tool_result",
    "null_final_from_model",
})


@dataclass(frozen=True)
class PartialResult:
    """What a run had reached when it stopped without an answer.

    Progress, not an answer. A run that ends at the iteration cap, on a repeated
    tool call, or on a tool that reproduced its own result has tool observations
    and reasoning but nothing the model wrote as its answer. Those observations
    travel here so a caller can use them deliberately, instead of arriving in
    :attr:`AgentResponse.output` where an answer would be.

    Attributes:
        observations: Every tool result of the run, in call order, exactly as
            the tools returned them (bounded only by the tool-call record's own
            cap). Not deduplicated and not reworded.
        last_observation: The final entry of :attr:`observations`, or ``None``
            when the run made no tool call.
        last_thought: The model's last substantive reasoning line, when the run
            left one.
        text: The flattened one-line form, also carried as
            ``metadata["partial_output"]``.
        iterations: How many loop iterations the run had made.
        tool_calls: How many tool calls the run had made.
    """

    observations: tuple[str, ...] = ()
    last_observation: str | None = None
    last_thought: str | None = None
    text: str = ""
    iterations: int = 0
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the payload as plain data, for JSON output and saved runs."""
        return {
            "observations": list(self.observations),
            "last_observation": self.last_observation,
            "last_thought": self.last_thought,
            "text": self.text,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartialResult":
        """Rebuild a payload from :meth:`to_dict` output."""
        return cls(
            observations=tuple(data.get("observations") or ()),
            last_observation=data.get("last_observation"),
            last_thought=data.get("last_thought"),
            text=data.get("text") or "",
            iterations=int(data.get("iterations") or 0),
            tool_calls=int(data.get("tool_calls") or 0),
        )


@dataclass
class StreamEvent:
    """A typed event yielded by ``Agent.stream(..., include_events=True)``.

    The default ``stream()`` yields plain answer-text ``str`` deltas. Opting into
    events instead surfaces the agent's progress as structured records a
    presentation layer can render (spinner labels, per-tool ticks) without ever
    parsing raw ReAct scaffolding out of the text payload.

    ``kind`` is one of:

    - ``"answer"``   — a sanitized final-answer text delta (``text``).
    - ``"thought"``  — the model's reasoning for a step (``text``); display-only.
    - ``"tool_call"``— a tool invocation (``tool`` + ``tool_input``).
    - ``"observation"`` — a tool result (``text`` + ``tool``).
    - ``"status"``   — a terminal/limit notice (``text``), e.g. step-limit hit.
    - ``"usage"``    — the last event of the stream, carrying the run's token
      counts, cost and timings in ``usage`` (see
      :attr:`Agent.last_stream_usage` for the keys). Emitted once the answer is
      complete, so a caller can report what a streamed turn cost without
      running the prompt again.
    """

    kind: str
    text: str = ""
    tool: str | None = None
    tool_input: str | None = None
    usage: dict[str, Any] | None = None


def _metadata_as_data(metadata: dict[str, Any]) -> dict[str, Any]:
    """The response's metadata with the run's conversation written as data.

    ``metadata["thread"]`` carries the run's steps as objects, which is what a
    caller inspecting a finished run wants. A saved document cannot hold an
    object, so the thread is written through its own serialisation and every
    other entry is left exactly as the run put it there.
    """
    thread = metadata.get("thread")
    if thread is None or not hasattr(thread, "to_dict"):
        return metadata
    return {**metadata, "thread": thread.to_dict()}


@dataclass
class AgentResponse:
    """
    Response from agent execution.

    Attributes:
        output: The answer text on a successful run. **On a failed run this is
            effGen's report of what stopped the run, not the model's words** —
            what the run had reached is kept in ``metadata["partial_output"]``.
            See ``metadata`` below for the per-reason detail. Callers that read
            ``output`` without checking :attr:`success` are the reason
            ``raise_on_error`` defaults to ``True``; a batch caller that turns
            it off has to make that check itself.
        success: Whether execution succeeded
        mode: Execution mode used
        iterations: Number of iterations performed
        tool_calls: The tool calls the run made, as
            :class:`~effgen.core.tool_call_record.ToolCall` records — which tool,
            with what arguments, what came back, how long it took and whether it
            failed. Iterate it for the calls; it also compares and casts as the
            number of calls, so ``tool_calls == 2`` and ``tool_calls > 0`` read
            as they always did. :attr:`tool_call_count` says the number plainly.
        tokens_used: Total tokens consumed
        execution_time: Time taken in seconds
        execution_trace: Full execution trace
        execution_tree: Hierarchical execution tree
        task: The task text this response answers, as the agent received it
        model: Model id the run was answered on
        provider: Provider that served the model, when one is known
        started_at: UTC ISO-8601 timestamp of when the run started
        stop_reason: What ended the run, one of :data:`STOP_REASONS`. Present on
            every response — a run that answered reports ``"final_answer"`` —
            and always equal to ``metadata["reason"]``. Read
            :attr:`outcome` to branch on the three cases without listing the
            reasons yourself.
        partial: What the run had reached when it stopped without writing an
            answer, as a :class:`PartialResult`; ``None`` on an answered run and
            on a stopped run that had reached nothing. Its ``text`` is also
            carried as ``metadata["partial_output"]``.
        routing_decision: Routing decision (if sub-agents used)
        metadata: Additional metadata. Always includes ``reason``, which is
            :attr:`stop_reason` under its original key, one of:

            - ``"final_answer"`` — the model produced an answer (``success=True``).
              A finer ``answer_source`` may also be present (e.g.
              ``loop_detected``, ``direct_calculator_result``) for heuristically
              recovered answers.
            - ``"max_iterations_partial"`` — the tool loop hit its iteration cap
              before producing a final answer (``success=False``,
              ``partial=True``). The run has no answer, so ``output`` is the
              typed outcome: what stopped the run and what to do about it.
              What the run had reached — tool observations and reasoning, which
              the model never wrote up as an answer — is kept beside it in
              ``metadata["partial_output"]``. ``metadata["error"]`` has the
              structured shape below with ``category="max_iterations"`` and the
              cap in ``max_iterations``.
            - ``"max_iterations_exhausted"`` — the loop hit the same cap with no
              recoverable progress (``success=False``); ``output`` and
              ``metadata["error"]`` are as above, without ``partial_output``.
            - ``"generation_failed"`` — the model/provider call failed
              (``success=False``); ``metadata["error"]`` is a structured dict
              ``{type, category, provider, model, message, retryable}`` and is
              identical whether the failure happened on the direct or tool path.
            - ``"empty_task"`` — the task was empty or whitespace-only
              (``success=False``); rejected before any model call, so nothing
              is billed. ``metadata["error"]`` has the same shape as above with
              ``provider``/``model`` set to ``None``.
            - ``"written_tool_call"`` — the model wrote a tool call into its
              answer instead of calling the tool (``success=False``), either for
              a tool that never ran in this run or as an answer that is nothing
              but the call block. An answer that recaps a call the run really
              made is not this failure. ``metadata["error"]`` adds ``tool`` (the
              tool whose call was written out), ``tool_calling_strategy`` and a
              short ``answer_preview`` to the structured shape above.

            A run that went through the tool loop also carries
            ``tool_calling_strategy`` — ``"react"``, ``"native"``, ``"hybrid"``,
            ``"openai_native"`` or ``"gemini_native"`` — naming the tool-calling
            path that produced the result.

            - ``"loop_detected"`` — the model kept asking for the same tool
              call and the loop stopped the run (``success=False``,
              ``partial=True``). Whatever the tools had returned is in
              :attr:`partial` and ``metadata["partial_output"]``.
            - ``"repeated_tool_result"`` — a tool returned a result it had
              already returned and the loop stopped the run (``success=False``),
              carrying the same payload.
            - ``"null_final_from_model"`` — the model answered with nothing
              usable ("N/A", "none") after using tools (``success=False``),
              carrying the same payload.

            Success rule: ``success`` is ``True`` only when a run finished with a
            real answer (``final_answer``); a run the loop stopped — at the
            iteration cap, on a repeated call, on a repeated result, or on an
            empty final answer — reports ``success=False`` and never puts the
            text it recovered where an answer goes. ``success`` is never
            ``True`` with empty output.
    """
    output: str
    success: bool = True
    mode: AgentMode = AgentMode.SINGLE
    iterations: int = 0
    tool_calls: ToolCallList = field(default_factory=ToolCallList)
    tokens_used: int = 0
    execution_time: float = 0.0
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    execution_tree: dict[str, Any] = field(default_factory=dict)
    routing_decision: RoutingDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[Any] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    task: str | None = None
    model: str | None = None
    provider: str | None = None
    started_at: str | None = None
    stop_reason: str | None = None
    partial: PartialResult | None = None

    def __post_init__(self) -> None:
        """Normalize ``tool_calls`` and settle ``stop_reason``.

        Every path that builds a response, including a saved run read back and
        the many that report only a count, arrives here, so the field a caller
        sees is always the same type.

        ``stop_reason`` is settled here rather than at each of the two dozen
        places a response is built, so it is never ``None`` on a response a
        caller receives. An explicit value wins; otherwise it is taken from
        ``metadata["reason"]``, and failing that derived from what the response
        already says. The two are then kept equal, because ``metadata["reason"]``
        is what earlier releases read.
        """
        self.tool_calls = coerce_tool_calls(self.tool_calls)
        metadata = self.metadata if isinstance(self.metadata, dict) else None
        if not self.stop_reason:
            recorded = metadata.get("reason") if metadata else None
            if recorded:
                self.stop_reason = str(recorded)
            elif self.success:
                self.stop_reason = "final_answer"
            elif metadata and metadata.get("guardrail_blocked"):
                self.stop_reason = "guardrail_blocked"
            else:
                self.stop_reason = "run_failed"
        if metadata is not None and not metadata.get("reason"):
            metadata["reason"] = self.stop_reason

    @property
    def outcome(self) -> str:
        """What became of the run: ``"answered"``, ``"stopped"`` or ``"failed"``.

        - ``"answered"`` — :attr:`output` is the answer the model wrote.
        - ``"stopped"`` — the loop ended the run before the model wrote an
          answer. :attr:`output` says what happened; :attr:`partial` holds what
          the run had reached, when it had reached anything.
        - ``"failed"`` — the run could not be carried out at all (a provider
          failure, an empty task, a blocked run, a model that wrote its tool
          call out instead of making it).

        Derived from :attr:`success` and :attr:`stop_reason` rather than stored,
        so it cannot drift from them.
        """
        if self.success:
            return "answered"
        return "stopped" if self.stop_reason in STOPPED_REASONS else "failed"

    def mark_failed(self, reason: str, error: dict[str, Any] | None = None) -> None:
        """Turn an assembled response into a failure with *reason*.

        A few paths decide a run failed after the response was built — the
        structured-output step is one. Recording that means changing
        :attr:`success`, :attr:`stop_reason` and ``metadata["reason"]`` together;
        doing it here keeps them from disagreeing.
        """
        self.success = False
        self.stop_reason = reason
        if isinstance(self.metadata, dict):
            self.metadata["reason"] = reason
            if error is not None:
                self.metadata["error"] = error

    @property
    def tool_call_count(self) -> int:
        """How many tool calls the run made.

        The same number ``tool_calls`` compares as; named for code that wants
        to say so plainly.
        """
        return self.tool_calls.total

    def __str__(self) -> str:
        """The answer text — so ``print(result)`` shows the answer, not a repr.

        The full structured view stays available via ``repr(result)`` and
        :meth:`to_dict`.
        """
        return self.output if self.output is not None else ""

    def __await__(self) -> Generator[Any, None, NoReturn]:
        """Fail clearly if someone ``await``s the result of the sync ``run()``.

        ``Agent.run()`` is synchronous and returns this object directly, so
        ``await agent.run(...)`` would otherwise raise the opaque
        ``object AgentResponse can't be used in 'await' expression``. Point the
        caller at the async entry point instead.
        """
        raise TypeError(
            "Agent.run() is synchronous and already returns the AgentResponse — "
            "don't await it. Use `result = agent.run(...)`, or for async code "
            "`result = await agent.run_async(...)`."
        )
        # Unreachable; makes this a generator function so it's a valid __await__.
        yield  # pragma: no cover

    def __repr__(self) -> str:
        """A detailed-but-compact developer view.

        The default dataclass repr dumps the entire execution trace and tree,
        which is unreadable. This keeps the useful structured fields and a
        truncated answer preview, and summarizes the trace by length.
        """
        preview = (self.output or "").replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "…"
        reason = self.metadata.get("reason") if isinstance(self.metadata, dict) else None
        return (
            f"AgentResponse(success={self.success}, output={preview!r}, "
            f"mode={self.mode.value!r}, iterations={self.iterations}, "
            f"tool_calls={self.tool_calls}, tokens_used={self.tokens_used}, "
            f"execution_time={self.execution_time:.2f}s, "
            f"trace_steps={len(self.execution_trace)}, reason={reason!r})"
        )

    def _repr_html_(self) -> str:
        """Rich HTML card for Jupyter/IPython (answer + metrics + step trace)."""
        from effgen.ui import response_html

        return response_html(self)

    def show(self, console: Any = None) -> "AgentResponse":
        """Print a compact human view: the answer plus a one-line metric footer.

        Renders markdown, fenced code, and tables in the answer. Returns
        ``self`` so it can be chained. Use :meth:`trace` for the full reasoning.
        """
        from effgen.ui import response_show

        response_show(self, console=console)
        return self

    def trace(self, console: Any = None) -> "AgentResponse":
        """Print the full step-by-step reasoning trace. Returns ``self``."""
        from effgen.ui import response_trace

        response_trace(self, console=console)
        return self

    @property
    def text(self) -> str:
        """Read-only alias for :attr:`output` (familiar from other SDKs)."""
        return self.output

    @property
    def content(self) -> str:
        """Read-only alias for :attr:`output` (familiar from other SDKs)."""
        return self.output

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        The document identifies the run it came from — ``task``, ``model``,
        ``provider`` and ``started_at`` — so a saved result can be read back
        without the caller having to keep a separate note of what produced it.
        """
        return {
            "task": self.task,
            "model": self.model,
            "provider": self.provider,
            "started_at": self.started_at,
            "output": self.output,
            "success": self.success,
            "stop_reason": self.stop_reason,
            "outcome": self.outcome,
            "partial": self.partial.to_dict() if self.partial else None,
            "mode": self.mode.value,
            "iterations": self.iterations,
            # The count stays an int under its original key so a reader of a
            # saved run keeps working; the calls themselves are alongside it.
            "tool_calls": self.tool_calls.total,
            "tool_call_details": self.tool_calls.to_list(),
            "tokens_used": self.tokens_used,
            "execution_time": round(self.execution_time, 2),
            "execution_trace": self.execution_trace,
            "execution_tree": self.execution_tree,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "metadata": _metadata_as_data(self.metadata),
            "citations": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.citations],
            "sources": self.sources,
        }
