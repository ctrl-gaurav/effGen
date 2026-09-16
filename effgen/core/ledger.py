"""What a run spent: model calls, tool calls, tokens, and where its time went.

Every :meth:`~effgen.core.agent.Agent.run` and :meth:`~effgen.core.agent.Agent.stream`
keeps a ledger. It counts the model calls and tool calls the run made, the
tokens each model call reported, and splits the run's wall time into four parts:

- ``model_wait_s`` — time spent inside a model call (a blocking ``generate``, or
  pulling the next piece of a stream), plus the agent's own back-off sleeps
  between retries of one;
- ``tool_wait_s`` — time spent inside a tool's own ``execute``;
- ``caller_wait_s`` — time the run spent waiting on the caller: a stream
  suspended while the consumer handles a piece, or a human approving a tool call;
- ``framework_s`` — everything else: prompt building, parsing, guards,
  middleware, guardrails, telemetry and bookkeeping.

A run that starts another run — a decomposed run's sub-agents, a tool that runs
an agent, a workflow's nodes, a team's members — attaches that run's ledger as a
child. A child's calls are never counted in the parent's own numbers;
:meth:`RunLedger.total` adds them in exactly once.

The ledger is read from ``response.ledger`` (a :class:`RunLedger`) or, as plain
data, from ``response.metadata["ledger"]``.

The recorder for the run in progress is held in a context variable, so
concurrent runs, including concurrent runs on one agent, never share one.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Callable, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CallRecord", "RunLedger", "StepLedger"]

#: Counters that add up across a run's children and across a resumed run's
#: earlier part. ``wall_s`` and ``iterations`` describe one run and do not.
_ADDITIVE = (
    "llm_calls",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "total_tokens",
    "unpriced_calls",
    "estimated_calls",
    "unreported_calls",
    "model_wait_s",
    "tool_wait_s",
    "caller_wait_s",
    "framework_s",
)

_ROUND = 6


def _r(value: float) -> float:
    return round(value, _ROUND)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


@dataclass
class CallRecord:
    """One model call or one tool call a run made.

    Attributes:
        kind: ``"model"`` or ``"tool"``.
        name: The model id or the tool name.
        wait_s: Seconds spent waiting on the call. For a model call through an
            adapter that marks its provider request, the time inside that
            request (retries included); otherwise the whole call.
        call_s: For a model call, seconds from entering the adapter to leaving
            it; ``call_s - wait_s`` is the adapter's own work, counted as
            framework time.
        iteration: The loop iteration the call belongs to (0 before the first).
        prompt_tokens: Prompt tokens the backend reported; ``None`` when it
            reported none.
        completion_tokens: Completion tokens the backend reported.
        cached_input_tokens: Prompt tokens the provider served from its cache.
        cache_write_tokens: Prompt tokens the provider wrote into its cache,
            on a provider that reports writes and bills them above the input
            rate; ``None`` when the provider reports none.
        cost_usd: What the call cost; ``None`` when unpriced or unknown.
        estimated: The token counts were estimated locally, not reported.
        stream: The call was a stream.
        outcome: ``"ok"``, or the error category of a call that raised.
    """

    kind: str
    name: str
    wait_s: float
    iteration: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    estimated: bool = False
    stream: bool = False
    outcome: str = "ok"
    call_s: float | None = None
    #: Whether the backend's result carried a ``cost_usd`` key at all.
    priced_key: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The call as plain data."""
        return {
            "kind": self.kind,
            "name": self.name,
            "wait_s": _r(self.wait_s),
            "call_s": None if self.call_s is None else _r(self.call_s),
            "iteration": self.iteration,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": self.cost_usd,
            "estimated": self.estimated,
            "stream": self.stream,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallRecord:
        """Rebuild a call from :meth:`to_dict` output."""
        return cls(
            kind=str(data.get("kind") or "model"),
            name=str(data.get("name") or ""),
            wait_s=float(data.get("wait_s") or 0.0),
            call_s=data.get("call_s"),
            iteration=int(data.get("iteration") or 0),
            prompt_tokens=_int_or_none(data.get("prompt_tokens")),
            completion_tokens=_int_or_none(data.get("completion_tokens")),
            cached_input_tokens=_int_or_none(data.get("cached_input_tokens")),
            cache_write_tokens=_int_or_none(data.get("cache_write_tokens")),
            cost_usd=data.get("cost_usd"),
            estimated=bool(data.get("estimated")),
            stream=bool(data.get("stream")),
            outcome=str(data.get("outcome") or "ok"),
        )


@dataclass
class StepLedger:
    """What one loop iteration spent.

    ``wall_s`` runs from the start of the iteration to the start of the next
    one (or the end of the run), so the steps' ``framework_s`` add up to the
    run's own less the time before the first iteration began.
    """

    iteration: int
    wall_s: float = 0.0
    model_wait_s: float = 0.0
    tool_wait_s: float = 0.0
    caller_wait_s: float = 0.0
    child_wait_s: float = 0.0
    framework_s: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data."""
        return {
            "iteration": self.iteration,
            "wall_s": _r(self.wall_s),
            "model_wait_s": _r(self.model_wait_s),
            "tool_wait_s": _r(self.tool_wait_s),
            "caller_wait_s": _r(self.caller_wait_s),
            "child_wait_s": _r(self.child_wait_s),
            "framework_s": _r(self.framework_s),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepLedger:
        """Rebuild a step from :meth:`to_dict` output."""
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class RunLedger:
    """What one run spent, and where its time went.

    The counters on the object are the run's **own**: the calls this run made
    itself. :meth:`total` adds every child run once; :meth:`cumulative` adds, on
    a resumed run, what the run had spent before its checkpoint.

    Attributes:
        kind: ``"run"``, ``"stream"``, ``"workflow"`` or ``"team"``.
        name: The agent, workflow or team name.
        llm_calls: Model requests made, retries and failed calls included.
        tool_calls: Tool executions (a fallback tool that ran counts as one).
        prompt_tokens: Prompt tokens reported by the backend, summed.
        completion_tokens: Completion tokens reported by the backend, summed.
        cached_input_tokens: Prompt tokens served from a provider cache.
        cache_write_tokens: Prompt tokens written into a provider cache, which
            the providers that report them bill above the input rate. A run
            that only ever writes is spending more than one that never cached.
        cost_usd: Cost of the priced calls; ``None`` when no call was priced.
        unpriced_calls: Calls whose model has no published price.
        estimated_calls: Calls whose token counts were estimated locally.
        unreported_calls: Calls whose backend reported no token counts.
        iterations: Loop iterations the run took.
        wall_s: Seconds from the start of the run to the end of it.
        model_wait_s: Seconds inside model calls.
        tool_wait_s: Seconds inside tool executions.
        caller_wait_s: Seconds waiting on the caller.
        child_wait_s: Seconds inside child runs that were not already inside
            one of this run's model or tool calls.
        framework_s: ``wall_s`` less the four waits above.
        process_peak_rss_mb: The process's peak resident memory when the run
            ended. A process-wide number: never added across runs.
        steps: One :class:`StepLedger` per iteration.
        calls: One :class:`CallRecord` per model or tool call.
        children: The ledgers of runs this run started.
        resumed_from: On a resumed run, ``{"checkpoint_id": ..., "prior": {...}}``
            with what the run had spent before the checkpoint. A value in
            ``prior`` is ``None`` when the checkpoint did not record it.
    """

    kind: str = "run"
    name: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    unpriced_calls: int = 0
    estimated_calls: int = 0
    unreported_calls: int = 0
    iterations: int = 0
    wall_s: float = 0.0
    model_wait_s: float = 0.0
    tool_wait_s: float = 0.0
    caller_wait_s: float = 0.0
    child_wait_s: float = 0.0
    framework_s: float = 0.0
    process_peak_rss_mb: float | None = None
    steps: list[StepLedger] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    children: list[RunLedger] = field(default_factory=list)
    resumed_from: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens of the run's own calls."""
        return self.prompt_tokens + self.completion_tokens

    def own(self) -> dict[str, Any]:
        """The run's own counters, children and earlier parts excluded."""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "unpriced_calls": self.unpriced_calls,
            "estimated_calls": self.estimated_calls,
            "unreported_calls": self.unreported_calls,
            "iterations": self.iterations,
            "wall_s": _r(self.wall_s),
            "model_wait_s": _r(self.model_wait_s),
            "tool_wait_s": _r(self.tool_wait_s),
            "caller_wait_s": _r(self.caller_wait_s),
            "child_wait_s": _r(self.child_wait_s),
            "framework_s": _r(self.framework_s),
        }

    def total(self) -> dict[str, Any]:
        """The run's own counters with every child's :meth:`total` added once.

        ``wall_s``, ``iterations`` and ``child_wait_s`` stay the run's own; the
        waits and ``framework_s`` add up across the tree, so a child's framework
        time is reported once, on the total, and never as the parent's.
        """
        out = self.own()
        for child in self.children:
            _add_into(out, child.total())
        for key in ("model_wait_s", "tool_wait_s", "caller_wait_s", "framework_s"):
            out[key] = _r(out[key])
        return out

    def cumulative(self) -> dict[str, Any]:
        """:meth:`total` plus what a resumed run spent before its checkpoint.

        On a run that was not resumed this equals :meth:`total`. A counter the
        checkpoint did not record makes the cumulative value ``None``, because
        the true figure is unknown.
        """
        out = self.total()
        prior = (self.resumed_from or {}).get("prior") or {}
        if not prior:
            return out
        for key in _ADDITIVE:
            if key not in out:
                continue
            before = prior.get(key)
            if before is None:
                out[key] = None
            elif out[key] is not None:
                out[key] = out[key] + before
        before_cost = prior.get("cost_usd")
        if before_cost is not None:
            out["cost_usd"] = (out.get("cost_usd") or 0.0) + before_cost
        return out

    def to_dict(self) -> dict[str, Any]:
        """The ledger as plain data: own counters, steps, calls, children, total."""
        own = self.own()
        data: dict[str, Any] = {"kind": self.kind, "name": self.name, **own}
        data["process_peak_rss_mb"] = self.process_peak_rss_mb
        data["steps"] = [s.to_dict() for s in self.steps]
        data["calls"] = [c.to_dict() for c in self.calls]
        data["children"] = [c.to_dict() for c in self.children]
        data["total"] = self.total() if self.children else own.copy()
        data["resumed_from"] = self.resumed_from
        if self.resumed_from:
            data["cumulative"] = self.cumulative()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunLedger:
        """Rebuild a ledger from :meth:`to_dict` output."""
        simple = {
            k: data[k] for k in cls.__dataclass_fields__
            if k in data and k not in ("steps", "calls", "children")
        }
        ledger = cls(**simple)
        ledger.steps = [StepLedger.from_dict(s) for s in data.get("steps") or []]
        ledger.calls = [CallRecord.from_dict(c) for c in data.get("calls") or []]
        ledger.children = [cls.from_dict(c) for c in data.get("children") or []]
        return ledger


def _add_into(out: dict[str, Any], other: dict[str, Any]) -> None:
    for key in _ADDITIVE:
        value = other.get(key)
        if value is not None and out.get(key) is not None:
            out[key] = out[key] + value
    other_cost = other.get("cost_usd")
    if other_cost is not None:
        out["cost_usd"] = (out.get("cost_usd") or 0.0) + other_cost


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - not on Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS bytes.
    return round(peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0, 1)


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Length of the intersection of two merged interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


class _Recorder:
    """The mutable ledger of the run in progress."""

    __slots__ = (
        "kind", "name", "parent", "t0", "calls", "spans", "caller",
        "marks", "children", "prior", "closed", "retry_wait", "inner",
        "backoff_total",
    )

    def __init__(self, kind: str, name: str, parent: _Recorder | None) -> None:
        self.kind = kind
        self.name = name
        self.parent = parent
        self.t0 = perf_counter()
        self.calls: list[CallRecord] = []
        #: (start, end) of each model or tool call, for the child-overlap sum.
        self.spans: list[tuple[float, float]] = []
        #: iteration -> seconds waiting on the caller.
        self.caller: dict[int, float] = {}
        #: (iteration, start time) in order.
        self.marks: list[tuple[int, float]] = []
        #: (ledger, start, end) of each child run.
        self.children: list[tuple[RunLedger, float, float]] = []
        self.prior: dict[str, Any] | None = None
        self.closed = False
        #: iteration -> seconds of back-off between retries of a model call.
        self.retry_wait: dict[int, float] = {}
        #: Seconds inside provider requests during the model call in progress,
        #: or ``None`` when its adapter marked none.
        self.inner: float | None = None
        self.backoff_total = 0.0

    # -- recording --------------------------------------------------------
    @property
    def iteration(self) -> int:
        return self.marks[-1][0] if self.marks else 0

    def mark_iteration(self, number: int) -> None:
        self.marks.append((number, perf_counter()))

    def model_call(
        self,
        name: str,
        start: float,
        end: float,
        metadata: Any = None,
        *,
        outcome: str = "ok",
        stream: bool = False,
        wait_s: float | None = None,
    ) -> CallRecord:
        record = CallRecord(
            kind="model", name=name,
            wait_s=(end - start) if wait_s is None else wait_s,
            call_s=end - start,
            iteration=self.iteration, stream=stream, outcome=outcome,
        )
        if isinstance(metadata, dict) and metadata:
            _apply_usage(record, metadata)
        self.calls.append(record)
        self.spans.append((start, end))
        return record

    def tool_call(self, name: str, start: float, end: float, *, outcome: str = "ok") -> None:
        self.calls.append(CallRecord(
            kind="tool", name=name, wait_s=end - start,
            iteration=self.iteration, outcome=outcome,
        ))
        self.spans.append((start, end))

    def caller_wait(self, seconds: float) -> None:
        it = self.iteration
        self.caller[it] = self.caller.get(it, 0.0) + seconds

    def backoff(self, seconds: float) -> None:
        it = self.iteration
        self.retry_wait[it] = self.retry_wait.get(it, 0.0) + seconds
        self.backoff_total += seconds

    def fill_stream_usage(self, usage: dict[str, Any]) -> None:
        """Put a finished stream's usage on the stream call it belongs to."""
        for record in reversed(self.calls):
            if record.kind == "model" and record.stream:
                if record.prompt_tokens is None and record.completion_tokens is None:
                    _apply_usage(record, usage)
                    record.estimated = bool(usage.get("estimated"))
                return

    def attach_child(self, ledger: RunLedger, start: float, end: float) -> None:
        self.children.append((ledger, start, end))

    # -- closing ----------------------------------------------------------
    def snapshot(
        self, iterations: int | None = None, *, rss: bool = True, breakdown: bool = True,
    ) -> RunLedger:
        """The ledger as it stands now; the recorder keeps recording.

        ``breakdown=False`` leaves the per-iteration steps out, for a caller
        that reads only the run's counters and calls.
        """
        if not breakdown:
            return self._counters(iterations)
        end = perf_counter()
        ledger = RunLedger(kind=self.kind, name=self.name, resumed_from=self.prior)
        ledger.wall_s = end - self.t0
        cost = 0.0
        priced = False
        steps: dict[int, StepLedger] = {}
        for record in self.calls:
            step = steps.get(record.iteration)
            if step is None:
                step = steps[record.iteration] = StepLedger(iteration=record.iteration)
            if record.kind == "model":
                ledger.llm_calls += 1
                ledger.model_wait_s += record.wait_s
                step.llm_calls += 1
                step.model_wait_s += record.wait_s
                if record.prompt_tokens is None and record.completion_tokens is None:
                    ledger.unreported_calls += 1
                else:
                    p = record.prompt_tokens or 0
                    c = record.completion_tokens or 0
                    ledger.prompt_tokens += p
                    ledger.completion_tokens += c
                    step.prompt_tokens += p
                    step.completion_tokens += c
                if record.cached_input_tokens:
                    ledger.cached_input_tokens += record.cached_input_tokens
                    step.cached_input_tokens += record.cached_input_tokens
                if record.cache_write_tokens:
                    ledger.cache_write_tokens += record.cache_write_tokens
                    step.cache_write_tokens += record.cache_write_tokens
                if record.estimated:
                    ledger.estimated_calls += 1
                if record.cost_usd is not None:
                    cost += float(record.cost_usd)
                    priced = True
                elif record.priced_key:
                    ledger.unpriced_calls += 1
            else:
                ledger.tool_calls += 1
                ledger.tool_wait_s += record.wait_s
                step.tool_calls += 1
                step.tool_wait_s += record.wait_s
        ledger.cost_usd = round(cost, 8) if priced else None
        for it, seconds in self.retry_wait.items():
            ledger.model_wait_s += seconds
            steps.setdefault(it, StepLedger(iteration=it)).model_wait_s += seconds
        for it, seconds in self.caller.items():
            ledger.caller_wait_s += seconds
            steps.setdefault(it, StepLedger(iteration=it)).caller_wait_s += seconds

        child_extra_by_step: dict[int, float] = {}
        if self.children:
            ledger.children = [c[0] for c in self.children]
            child_spans = _union([(s, e) for _, s, e in self.children])
            own_spans = _union(self.spans)
            extra = sum(e - s for s, e in child_spans) - _overlap(child_spans, own_spans)
            ledger.child_wait_s = max(extra, 0.0)
            if self.marks:
                for s, e in child_spans:
                    it = self._iteration_at(s)
                    part = (e - s) - _overlap([(s, e)], own_spans)
                    child_extra_by_step[it] = child_extra_by_step.get(it, 0.0) + max(part, 0.0)

        ledger.framework_s = max(
            ledger.wall_s - ledger.model_wait_s - ledger.tool_wait_s
            - ledger.caller_wait_s - ledger.child_wait_s,
            0.0,
        )

        # Step walls come from the iteration marks; a run that never marked
        # an iteration (one direct model call) is one step spanning the run.
        if self.marks:
            bounds = [t for _, t in self.marks[1:]] + [end]
            for (it, start), stop in zip(self.marks, bounds):
                step = steps.setdefault(it, StepLedger(iteration=it))
                step.wall_s = stop - start
        elif steps or iterations:
            step = steps.pop(0, None) or StepLedger(iteration=1)
            step.iteration = 1
            step.wall_s = ledger.wall_s
            steps = {1: step}
        for it, step in steps.items():
            step.child_wait_s = child_extra_by_step.get(it, 0.0)
            step.framework_s = max(
                step.wall_s - step.model_wait_s - step.tool_wait_s
                - step.caller_wait_s - step.child_wait_s,
                0.0,
            )
        ledger.steps = [
            steps[k] for k in sorted(steps)
            if k or steps[k].llm_calls or steps[k].tool_calls
        ]
        ledger.calls = list(self.calls)
        ledger.iterations = (
            int(iterations) if iterations is not None
            else (self.marks[-1][0] if self.marks else (1 if self.calls else 0))
        )
        ledger.process_peak_rss_mb = _peak_rss_mb() if rss else None
        return ledger

    def _counters(self, iterations: int | None) -> RunLedger:
        """The run's counters and calls as they stand, without the step breakdown."""
        ledger = RunLedger(kind=self.kind, name=self.name, resumed_from=self.prior)
        ledger.wall_s = perf_counter() - self.t0
        cost = 0.0
        priced = False
        for record in self.calls:
            if record.kind == "model":
                ledger.llm_calls += 1
                ledger.model_wait_s += record.wait_s
                if record.prompt_tokens is None and record.completion_tokens is None:
                    ledger.unreported_calls += 1
                else:
                    ledger.prompt_tokens += record.prompt_tokens or 0
                    ledger.completion_tokens += record.completion_tokens or 0
                ledger.cached_input_tokens += record.cached_input_tokens or 0
                ledger.cache_write_tokens += record.cache_write_tokens or 0
                ledger.estimated_calls += 1 if record.estimated else 0
                if record.cost_usd is not None:
                    cost += float(record.cost_usd)
                    priced = True
                elif record.priced_key:
                    ledger.unpriced_calls += 1
            else:
                ledger.tool_calls += 1
                ledger.tool_wait_s += record.wait_s
        ledger.cost_usd = round(cost, 8) if priced else None
        ledger.model_wait_s += self.backoff_total
        ledger.caller_wait_s = sum(self.caller.values())
        if self.children:
            ledger.children = [c[0] for c in self.children]
            child_spans = _union([(s, e) for _, s, e in self.children])
            extra = sum(e - s for s, e in child_spans) - _overlap(child_spans, _union(self.spans))
            ledger.child_wait_s = max(extra, 0.0)
        ledger.framework_s = max(
            ledger.wall_s - ledger.model_wait_s - ledger.tool_wait_s
            - ledger.caller_wait_s - ledger.child_wait_s,
            0.0,
        )
        ledger.calls = self.calls
        ledger.iterations = int(iterations) if iterations is not None else len(self.marks)
        return ledger

    def _iteration_at(self, when: float) -> int:
        current = 0
        for it, start in self.marks:
            if start > when:
                break
            current = it
        return current

    def close(self, iterations: int | None = None) -> RunLedger:
        """Finish the ledger and hand it to the run that started this one."""
        ledger = self.snapshot(iterations)
        self.closed = True
        parent = self.parent
        while parent is not None and parent.closed:
            parent = parent.parent
        if parent is not None:
            parent.attach_child(ledger, self.t0, perf_counter())
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[ledger] closed %s '%s': llm_calls=%d tool_calls=%d "
                "framework_ms=%.2f model_ms=%.2f tool_ms=%.2f children=%d",
                self.kind, self.name, ledger.llm_calls, ledger.tool_calls,
                ledger.framework_s * 1000, ledger.model_wait_s * 1000,
                ledger.tool_wait_s * 1000, len(ledger.children),
            )
        return ledger


def _apply_usage(record: CallRecord, meta: dict[str, Any]) -> None:
    record.prompt_tokens = _int_or_none(meta.get("prompt_tokens"))
    record.completion_tokens = _int_or_none(meta.get("completion_tokens"))
    cached = _int_or_none(meta.get("cached_input_tokens"))
    if cached is not None:
        record.cached_input_tokens = cached
    written = _int_or_none(meta.get("cache_write_tokens"))
    if written is not None:
        record.cache_write_tokens = written
    if "cost_usd" in meta:
        record.priced_key = True
        cost = meta.get("cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            record.cost_usd = float(cost)
    if meta.get("estimated_usage") or meta.get("estimated"):
        record.estimated = True


# --- The run in progress -------------------------------------------------------

_active: ContextVar[_Recorder | None] = ContextVar("effgen_run_ledger", default=None)


def current() -> _Recorder | None:
    """The recorder of the run in progress in this context, or ``None``."""
    rec = _active.get()
    while rec is not None and rec.closed:
        rec = rec.parent
    return rec


def open_run(kind: str = "run", name: str = "") -> _Recorder | None:
    """Start a ledger for a run; the run in progress, if any, becomes its parent."""
    return _Recorder(kind, name, current())


@contextlib.contextmanager
def activate(rec: _Recorder | None) -> Iterator[None]:
    """Make *rec* the recorder calls in this context are counted against."""
    if rec is None:
        yield
        return
    token = _active.set(rec)
    try:
        yield
    finally:
        _active.reset(token)


def enter(rec: _Recorder | None) -> Token[_Recorder | None] | None:
    """Make *rec* current until :func:`leave`; for a generator's single step."""
    return None if rec is None else _active.set(rec)


def leave(token: Token[_Recorder | None] | None) -> None:
    """Undo :func:`enter`."""
    if token is not None:
        _active.reset(token)


def mark_iteration(number: int) -> None:
    """Record that the run in progress started loop iteration *number*."""
    rec = current()
    if rec is not None:
        rec.mark_iteration(number)


def _outcome_of(exc: BaseException) -> str:
    try:
        from ..models.errors import classify_provider_error

        return str(classify_provider_error(exc).category or "error")  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - classification is best-effort
        return "error"


def timed_model_call(call: Callable[..., Any], name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a blocking model ``generate`` and count it on the run in progress.

    The call's model wait is the time inside the provider requests its adapter
    marked with :func:`provider_request`, plus the adapter's back-off between
    them; an adapter that marks none (a local engine) is waited on whole.

    Args:
        call: The bound generation method to call.
        name: The model id the call is recorded under.
        *args: Positional arguments for *call*.
        **kwargs: Keyword arguments for *call*.

    Returns:
        Whatever *call* returned.
    """
    rec = current()
    if rec is None:
        return call(*args, **kwargs)
    outer_inner = rec.inner
    rec.inner = None
    slept_before = rec.backoff_total
    start = perf_counter()
    outcome = "ok"
    result: Any = None
    try:
        result = call(*args, **kwargs)
    except BaseException as exc:
        outcome = _outcome_of(exc)
        raise
    finally:
        end = perf_counter()
        inner = rec.inner
        rec.inner = outer_inner
        if inner is None:
            # The adapter's own back-off is already counted; do not count it twice.
            wait = max((end - start) - (rec.backoff_total - slept_before), 0.0)
        else:
            wait = inner
        metadata = None
        if outcome == "ok":
            metadata = (
                result.get("metadata") if isinstance(result, dict)
                else getattr(result, "metadata", None)
            )
        rec.model_call(name, start, end, metadata, outcome=outcome, wait_s=wait)
    return result


@contextlib.contextmanager
def provider_request() -> Iterator[None]:
    """Count the enclosed provider request as model wait for the call in progress."""
    rec = current()
    if rec is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        rec.inner = (rec.inner or 0.0) + (perf_counter() - start)


def backoff_sleep(sleep: Callable[[float], Any], seconds: float) -> None:
    """Sleep between retries of a model call, counting the sleep as model wait."""
    rec = current()
    if rec is None:
        sleep(seconds)
        return
    start = perf_counter()
    try:
        sleep(seconds)
    finally:
        rec.backoff(perf_counter() - start)


class StreamClock:
    """Times a model stream: only the time spent pulling its next piece."""

    __slots__ = ("rec", "name", "start", "wait", "outcome")

    def __init__(self, name: str) -> None:
        self.rec = current()
        self.name = name
        self.start = perf_counter()
        self.wait = 0.0
        self.outcome = "ok"

    def open(self, factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Open the stream, counting the time the request takes to start.

        Args:
            factory: The model's ``generate_stream`` method.
            *args: Positional arguments for *factory*.
            **kwargs: Keyword arguments for *factory*.

        Returns:
            The stream iterator *factory* returned.
        """
        started = perf_counter()
        try:
            return factory(*args, **kwargs)
        except BaseException as exc:
            self.outcome = _outcome_of(exc)
            raise
        finally:
            self.wait += perf_counter() - started

    def pieces(self, iterator: Any) -> Iterator[Any]:
        """Yield the stream's pieces, counting only the time inside ``next``."""
        it = iter(iterator)
        while True:
            started = perf_counter()
            try:
                piece = next(it)
            except StopIteration:
                self.wait += perf_counter() - started
                return
            except BaseException as exc:
                self.wait += perf_counter() - started
                self.outcome = _outcome_of(exc)
                raise
            self.wait += perf_counter() - started
            yield piece

    def finish(self) -> None:
        """Count the stream as one model call on the run in progress."""
        if self.rec is not None:
            self.rec.model_call(
                self.name, self.start, perf_counter(),
                outcome=self.outcome, stream=True, wait_s=self.wait,
            )
            self.rec = None


def note_stream_usage(usage: dict[str, Any]) -> None:
    """Attach a finished stream's reported (or estimated) usage to its call."""
    rec = current()
    if rec is not None and isinstance(usage, dict):
        rec.fill_stream_usage(usage)


@contextlib.contextmanager
def tool_wait(name: str) -> Iterator[None]:
    """Count the enclosed tool execution on the run in progress."""
    rec = current()
    if rec is None:
        yield
        return
    start = perf_counter()
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        rec.tool_call(name, start, perf_counter(), outcome=outcome)


@contextlib.contextmanager
def caller_wait() -> Iterator[None]:
    """Count the enclosed wait on the caller (an approval, a prompt)."""
    rec = current()
    if rec is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        rec.caller_wait(perf_counter() - start)


def attach(response: Any, rec: _Recorder | None, *, close: bool) -> RunLedger | None:
    """Write *rec*'s ledger onto *response*'s metadata and return it.

    With ``close`` the ledger is final, is handed to the parent run and is
    written to ``metadata["ledger"]``; without it this only returns a snapshot
    for the telemetry recorded while the run is still finishing.
    """
    if rec is None or rec.closed:
        return None
    iterations = getattr(response, "iterations", None)
    if not close:
        return rec.snapshot(iterations, rss=False, breakdown=False)
    ledger = rec.close(iterations)
    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, dict):
        metadata["ledger"] = ledger.to_dict()
    return ledger
