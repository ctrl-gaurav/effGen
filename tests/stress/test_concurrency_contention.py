"""Contention scenarios: one shared object, many threads, exact counts.

Every scenario lives in :data:`SCENARIOS` as a row, so covering a new shared
object is a function plus an entry rather than a new module. Each returns a
:class:`~tests.stress.contention.ContentionReport` whose measurements are
equalities — a lost cost, a lost token, a double count, a read that failed
while another thread was swapping state — and the single parametrized case
below fails on any difference and prints the whole table.

``EFFGEN_CONTENTION_SCALE`` multiplies the worker counts and iteration counts,
so the same code is both the short form that runs in a couple of minutes and a
long, heavily-overlapped run.

All cases carry ``expensive`` and ``slow``; the CI lanes deselect both. The
offline regression tests for the defects these scenarios found are in
``tests/unit/test_concurrency_shared_state.py``, which CI does run.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from .contention import (
    ContentionReport,
    barrier_map,
    collect_exceptions,
    tight_switching,
)
from .test_endurance_soak import (
    LIVE_FAMILIES,
    _first_keyed_family,
    _free_port,
    _spawn_server,
    _wait_for_server,
)

pytestmark = [pytest.mark.expensive, pytest.mark.slow, pytest.mark.timeout(0)]


def _scale() -> int:
    """How many times over to run each scenario (``EFFGEN_CONTENTION_SCALE``)."""
    try:
        value = int(os.environ.get("EFFGEN_CONTENTION_SCALE", "1"))
    except ValueError:
        return 1
    return max(1, value)


# ---------------------------------------------------------------------------
# 1-2 — the tool registry every agent shares
# ---------------------------------------------------------------------------


def scenario_tool_registry_first_access() -> ContentionReport:
    """Threads reaching a cold registry together must all see the same catalog.

    The built-in catalog is discovered on first access. A thread that arrives
    while another is still discovering must wait for it, not be handed the
    empty registry that discovery has not filled yet.
    """
    from effgen.tools import registry as reg_mod

    workers, rounds = 32, 20 * _scale()
    report = ContentionReport("tool_registry_first_access", workers, rounds)
    divergent = 0
    empty_views = 0
    errors: list[str] = []
    counts_seen: set[int] = set()

    for _ in range(rounds):
        reg_mod.reset_registry()
        registry = reg_mod.get_registry()

        def view(_i: int, registry=registry) -> object:
            try:
                return len(registry.list_tools())
            except Exception as exc:  # noqa: BLE001 - counted, not raised
                return exc

        results = barrier_map(workers, view)
        errors.extend(collect_exceptions(results))
        counts = [r for r in results if isinstance(r, int)]
        empty_views += sum(1 for c in counts if c == 0)
        counts_seen.update(counts)
        if counts and min(counts) != max(counts):
            divergent += 1

    reg_mod.reset_registry()
    report.add("rounds_with_divergent_views", divergent, 0)
    report.add("threads_that_saw_an_empty_registry", empty_views, 0)
    report.add("exceptions", len(errors), 0, detail=str(errors[:2]))
    report.notes["tool_counts_seen"] = sorted(counts_seen)
    return report


def scenario_tool_registry_instance_identity() -> ContentionReport:
    """One tool, one instance, however many threads and loops ask for it.

    ``get_tool_sync`` drives the async loader on the calling thread's own event
    loop, so this also crosses loops: the registry's internal locking has to
    hold when the callers are not sharing one.
    """
    from effgen.tools import get_registry

    workers, per_worker = 16, 40 * _scale()
    registry = get_registry()
    registry.list_tools()
    report = ContentionReport("tool_registry_instance_identity", workers, per_worker)
    identities: set[int] = set()
    lock = threading.Lock()

    def work(_i: int) -> object:
        try:
            for _ in range(per_worker):
                tool = registry.get_tool_sync("calculator")
                with lock:
                    identities.add(id(tool))
            return None
        except Exception as exc:  # noqa: BLE001 - counted, not raised
            return exc

    errors = collect_exceptions(barrier_map(workers, work))
    report.add("exceptions", len(errors), 0, detail=str(errors[:2]))
    report.add("distinct_calculator_instances", len(identities), 1)
    return report


# ---------------------------------------------------------------------------
# 3-5 — the provider registry, its snapshot/restore round trip, its middleware
# ---------------------------------------------------------------------------


#: Fewest lookups the reset-under-reads scenario has to have performed for its
#: "no failures" result to mean anything. A run well under this measured the
#: readers rather than the swap.
_MINIMUM_READS = 2_000


def scenario_provider_registry_reset_under_reads() -> ContentionReport:
    """A model that exists throughout resolves throughout.

    ``reset()`` and ``restore()`` replace every registration. A lookup running
    at the same time asks for a model that is registered before the swap and
    after it, so every failure is the swap being observable, not the model
    being absent.
    """
    from effgen.models.registry import ProviderRegistry

    ProviderRegistry.register_builtins()
    opening = ProviderRegistry.snapshot()
    provider = "groq" if "groq" in opening else sorted(opening)[0]
    model_id = sorted(opening[provider]["models"])[0]
    target = f"{provider}:{model_id}"

    readers, cycles = 8, 200 * _scale()
    report = ContentionReport("provider_registry_reset_under_reads", readers, cycles)
    failures: list[str] = []
    empty_lists = 0
    reads = 0
    stop = threading.Event()
    lock = threading.Lock()

    def churn(_i: int) -> None:
        try:
            for _ in range(cycles):
                ProviderRegistry.reset()
                ProviderRegistry.restore(opening)
        finally:
            stop.set()

    def reader(_i: int) -> None:
        nonlocal reads, empty_lists
        local_reads = 0
        local_empty = 0
        local_failures: list[str] = []
        while not stop.is_set():
            local_reads += 1
            if not ProviderRegistry.list_providers():
                local_empty += 1
            try:
                ProviderRegistry.lookup(target)
            except Exception as exc:  # noqa: BLE001 - counted, not raised
                local_failures.append(f"{type(exc).__name__}: {str(exc)[:80]}")
        with lock:
            reads += local_reads
            empty_lists += local_empty
            failures.extend(local_failures)

    def worker(index: int) -> None:
        churn(index) if index == 0 else reader(index)

    barrier_map(readers + 1, worker, timeout=1800.0)
    ProviderRegistry.restore(opening)

    report.add("lookup_failures", len(failures), 0, detail=str(failures[:2]))
    report.add("reads_that_saw_no_providers", empty_lists, 0)
    # Zero failures out of a handful of reads proves nothing. Readers block
    # while a swap holds the registry, so the volume is far below what an
    # unguarded registry lets through; this floor only rules out a run that
    # measured almost nothing.
    report.add("reads_were_sampled", reads >= _MINIMUM_READS, True, detail=f"{reads} reads")
    report.notes["reads"] = reads
    report.notes["swap_cycles"] = cycles
    return report


def scenario_provider_registry_snapshot_round_trip() -> ContentionReport:
    """snapshot → inject → reset → restore, run by many threads at once.

    Each worker adds its own provider, drops the registry to its built-in
    state, then puts the opening snapshot back. However the rounds interleave,
    the registry has to end where it started and no reader may see a registry
    with none of the built-ins in it.
    """
    from effgen.models.registry import ProviderRegistry

    ProviderRegistry.register_builtins()
    opening = ProviderRegistry.snapshot()
    workers, rounds = 8, 200 * _scale()
    report = ContentionReport("provider_registry_snapshot_round_trip", workers, rounds)
    empty = 0
    lock = threading.Lock()

    class _Adapter:
        """Stand-in adapter class; the registry only stores the reference."""

    def work(index: int) -> object:
        nonlocal empty
        local_empty = 0
        try:
            for _ in range(rounds):
                ProviderRegistry.register(
                    f"contention-{index}", _Adapter, {f"m-{index}": {"context": 1}},
                )
                if not ProviderRegistry.list_providers():
                    local_empty += 1
                ProviderRegistry.reset()
                ProviderRegistry.restore(opening)
            return None
        except Exception as exc:  # noqa: BLE001 - counted, not raised
            return exc
        finally:
            with lock:
                empty += local_empty

    errors = collect_exceptions(barrier_map(workers, work, timeout=1800.0))
    ProviderRegistry.restore(opening)
    closing = ProviderRegistry.snapshot()

    report.add("exceptions", len(errors), 0, detail=str(errors[:2]))
    report.add("reads_that_saw_no_providers", empty, 0)
    report.add("providers_after", sorted(closing), sorted(opening))
    report.add(
        "models_after",
        sum(len(rec["models"]) for rec in closing.values()),
        sum(len(rec["models"]) for rec in opening.values()),
    )
    report.add(
        "injected_providers_left_behind",
        [name for name in closing if name.startswith("contention-")],
        [],
    )
    return report


def scenario_provider_registry_middleware_identity() -> ContentionReport:
    """One circuit breaker and one bulkhead per provider, however many askers.

    Two breakers for one provider means each holds half the failures, so the
    threshold that should have opened the circuit is never reached.
    """
    from effgen.models.registry import ProviderRegistry

    ProviderRegistry.register_builtins()
    provider = sorted(ProviderRegistry.list_providers())[0]
    workers, rounds = 32, 40 * _scale()
    report = ContentionReport("provider_registry_middleware_identity", workers, rounds)
    breaker_dupes = 0
    bulkhead_dupes = 0

    for _ in range(rounds):
        record = ProviderRegistry.get_provider_info(provider)
        record.pop("_circuit_breaker", None)
        record.pop("_bulkhead", None)

        def ask(_i: int) -> tuple[int, int]:
            return (
                id(ProviderRegistry.get_circuit_breaker(provider)),
                id(ProviderRegistry.get_bulkhead(provider)),
            )

        results = barrier_map(workers, ask)
        if len({r[0] for r in results}) > 1:
            breaker_dupes += 1
        if len({r[1] for r in results}) > 1:
            bulkhead_dupes += 1

    report.add("rounds_with_more_than_one_breaker", breaker_dupes, 0)
    report.add("rounds_with_more_than_one_bulkhead", bulkhead_dupes, 0)
    return report


# ---------------------------------------------------------------------------
# 6-8 — money and tokens: one adapter, one cost tracker
# ---------------------------------------------------------------------------

_PRICE = 0.001

#: Credential handed to an adapter that is only used to fold numbers. It keeps
#: these scenarios independent of whatever the machine has in its environment.
_OFFLINE_KEY = "offline-scenario-no-call-is-made"


@contextmanager
def _no_configured_budget() -> Iterator[None]:
    """Point the budget config at an empty file for the duration of the block."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "budget.json")
        Path(path).write_text("{}", encoding="utf-8")
        previous = os.environ.get("EFFGEN_BUDGET_CONFIG")
        os.environ["EFFGEN_BUDGET_CONFIG"] = path
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("EFFGEN_BUDGET_CONFIG", None)
            else:
                os.environ["EFFGEN_BUDGET_CONFIG"] = previous


def scenario_adapter_totals_non_streaming() -> ContentionReport:
    """A shared adapter's session total is the sum of the calls made on it."""
    from effgen.models.base import GenerationResult, _stamp_cost
    from effgen.models.openai_adapter import OpenAIAdapter

    workers, per_worker = 16, 200 * _scale()
    # The scenario folds numbers onto the adapter and never calls the service,
    # but the constructor requires a credential, so one is supplied here rather
    # than read from whatever the machine happens to have set.
    adapter = OpenAIAdapter(model_name="gpt-5-nano", api_key=_OFFLINE_KEY)
    adapter.total_cost = 0.0
    report = ContentionReport("adapter_totals_non_streaming", workers, per_worker)

    def work(_i: int) -> None:
        for _ in range(per_worker):
            _stamp_cost(
                adapter,
                GenerationResult(
                    text="x", tokens_used=15, finish_reason="stop",
                    model_name="gpt-5-nano", metadata={"cost_usd": _PRICE},
                ),
            )

    with tight_switching():
        barrier_map(workers, work, timeout=1800.0)
    calls = workers * per_worker
    report.add("total_cost", round(adapter.total_cost, 6), round(calls * _PRICE, 6))
    report.notes["calls"] = calls
    return report


def scenario_adapter_totals_streaming() -> ContentionReport:
    """The same for streamed calls, which fold their totals after the stream."""
    from effgen.models.base import accumulate_stream_cost
    from effgen.models.openai_adapter import OpenAIAdapter

    workers, per_worker = 16, 200 * _scale()
    adapter = OpenAIAdapter(model_name="gpt-5-nano", api_key=_OFFLINE_KEY)
    adapter.total_cost = 0.0
    adapter.total_tokens = 0
    report = ContentionReport("adapter_totals_streaming", workers, per_worker)

    def work(_i: int) -> None:
        for _ in range(per_worker):
            accumulate_stream_cost(
                adapter, _PRICE, tokens=15, prompt_tokens=10, completion_tokens=5,
            )

    with tight_switching():
        barrier_map(workers, work, timeout=1800.0)
    calls = workers * per_worker
    report.add("total_cost", round(adapter.total_cost, 6), round(calls * _PRICE, 6))
    report.add("total_tokens", adapter.total_tokens, calls * 15)
    report.notes["calls"] = calls
    return report


def scenario_cost_tracker_records() -> ContentionReport:
    """One cost tracker, many agents: every call is counted exactly once."""
    from effgen.models._cost import CostTracker

    workers, per_worker = 16, 500 * _scale()
    tracker = CostTracker(storage=None)
    report = ContentionReport("cost_tracker_records", workers, per_worker)

    def work(_i: int) -> None:
        for _ in range(per_worker):
            tracker.record("openai", "gpt-5-nano", 10, 5, cost_usd=_PRICE)

    # The scenario books thousands of dollars of synthetic spend, so it must
    # not consult the budget the machine's owner configured for real work.
    with _no_configured_budget(), tight_switching():
        barrier_map(workers, work, timeout=1800.0)
    calls = workers * per_worker
    stats = tracker._data[("openai", "gpt-5-nano")]
    report.add("requests", stats.requests, calls)
    report.add("prompt_tokens", stats.prompt_tokens, calls * 10)
    report.add("completion_tokens", stats.completion_tokens, calls * 5)
    report.add("total_cost_usd", round(stats.total_cost_usd, 6), round(calls * _PRICE, 6))
    return report


def scenario_cost_store_inserts() -> ContentionReport:
    """Every recorded call reaches the store, in memory and on disk.

    A tracker that logs the insert failure and returns is a call that was billed
    by the provider and is missing from the user's spend.
    """
    from effgen.models._cost_store import SQLiteCostStore

    workers, per_worker = 8, 100 * _scale()
    report = ContentionReport("cost_store_inserts", workers, per_worker)

    with tempfile.TemporaryDirectory() as directory:
        stores = {
            "memory": SQLiteCostStore(":memory:"),
            "file": SQLiteCostStore(os.path.join(directory, "costs.sqlite")),
        }
        for label, store in stores.items():
            errors: list[str] = []
            lock = threading.Lock()

            def work(_i: int, store=store, errors=errors, lock=lock) -> None:
                local: list[str] = []
                for _ in range(per_worker):
                    try:
                        store.insert("openai", "gpt-5-nano", 10, 5, _PRICE, 1.0)
                    except Exception as exc:  # noqa: BLE001 - counted, not raised
                        local.append(f"{type(exc).__name__}: {exc}")
                with lock:
                    errors.extend(local)

            barrier_map(workers, work, timeout=1800.0)
            rows = store.query_all()
            report.add(f"{label}_rows", len(rows), workers * per_worker)
            report.add(f"{label}_insert_errors", len(errors), 0, detail=str(errors[:2]))
            report.add(
                f"{label}_recorded_cost",
                round(sum(r.cost_usd for r in rows), 6),
                round(workers * per_worker * _PRICE, 6),
            )
    return report


# ---------------------------------------------------------------------------
# 10-12 — the session store and the run history
# ---------------------------------------------------------------------------


def scenario_session_store_same_id() -> ContentionReport:
    """Concurrent writers of one session leave one whole session behind.

    Last writer wins is the expected outcome. A file that will not parse, or
    one holding a blend of two writers' histories, is not.
    """
    from effgen.core.session import Session

    workers, rounds = 8, 40 * _scale()
    report = ContentionReport("session_store_same_id", workers, rounds)
    unreadable = 0
    blended = 0
    save_errors: list[str] = []

    with tempfile.TemporaryDirectory() as directory:
        for round_index in range(rounds):
            session_id = f"shared-{round_index}"
            written = []
            for worker in range(workers):
                session = Session(session_id=session_id, agent_name=f"agent-{worker}")
                for turn in range((worker + 1) * 40):
                    session.add_message("user", f"worker-{worker}-turn-{turn}" * 8)
                written.append(session)

            def save(index: int, written=written) -> object:
                try:
                    written[index].save(directory)
                    return None
                except Exception as exc:  # noqa: BLE001 - counted, not raised
                    return exc

            save_errors.extend(collect_exceptions(barrier_map(workers, save)))
            try:
                loaded = Session.load(session_id, directory)
            except Exception as exc:  # noqa: BLE001 - counted, not raised
                unreadable += 1
                save_errors.append(f"load {type(exc).__name__}: {str(exc)[:80]}")
                continue
            if len(loaded.messages) not in {len(s.messages) for s in written}:
                blended += 1

    report.add("sessions_that_would_not_load", unreadable, 0)
    report.add("sessions_holding_a_blend", blended, 0)
    report.add("save_errors", len(save_errors), 0, detail=str(save_errors[:2]))
    return report


def scenario_session_store_distinct_ids() -> ContentionReport:
    """Many writers, many sessions: the listing finds all of them, none broken."""
    from effgen.core.session import Session, SessionManager

    workers, per_worker = 16, 25 * _scale()
    report = ContentionReport("session_store_distinct_ids", workers, per_worker)

    with tempfile.TemporaryDirectory() as directory:
        def work(index: int) -> None:
            for n in range(per_worker):
                session = Session(session_id=f"s-{index}-{n}", agent_name="agent")
                session.add_message("user", "hello")
                session.add_message("assistant", "hi", cost_usd=_PRICE, run_id=f"r{index}{n}")
                session.save(directory)

        barrier_map(workers, work, timeout=1800.0)
        listed, unreadable = SessionManager(directory).scan()

    report.add("sessions_listed", len(listed), workers * per_worker)
    report.add("unreadable_files", len(unreadable), 0, detail=str(unreadable[:2]))
    return report


def scenario_run_history_appends() -> ContentionReport:
    """Every recorded run is one whole JSON line in the history file."""
    from effgen.observability import run_log

    workers, per_worker = 16, 100 * _scale()
    report = ContentionReport("run_history_appends", workers, per_worker)

    tag = os.urandom(4).hex()
    with tempfile.TemporaryDirectory() as directory:
        # EFFGEN_RUN_HISTORY_DIR takes precedence over EFFGEN_HOME, so both are
        # pointed here — another test in the same process may have set either.
        previous = {key: os.environ.get(key)
                    for key in ("EFFGEN_HOME", "EFFGEN_RUN_HISTORY_DIR")}
        os.environ["EFFGEN_HOME"] = directory
        os.environ["EFFGEN_RUN_HISTORY_DIR"] = os.path.join(directory, "runs")
        try:
            def work(index: int) -> None:
                for n in range(per_worker):
                    run_log.record_run(
                        task=f"task-{index}-{n}" * 20,
                        model="groq:openai/gpt-oss-20b",
                        output=f"answer-{index}-{n}" * 20,
                        run_id=f"{tag}-{index}-{n}",
                    )

            barrier_map(workers, work, timeout=1800.0)
            lines: list[str] = []
            for path in Path(run_log.history_dir()).glob("*.jsonl"):
                lines.extend(path.read_text(encoding="utf-8").splitlines())
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    parsed = 0
    broken = 0
    ids: set[str] = set()
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            broken += 1
            continue
        parsed += 1
        run_id = str(record.get("run_id", ""))
        if run_id.startswith(tag):
            ids.add(run_id)

    total = workers * per_worker
    report.add("lines_written", len(lines), total)
    report.add("lines_that_parse", parsed, total)
    report.add("truncated_or_interleaved_lines", broken, 0)
    report.add("distinct_run_ids", len(ids), total)
    return report


# ---------------------------------------------------------------------------
# 13 — trace isolation
# ---------------------------------------------------------------------------


def scenario_trace_isolation() -> ContentionReport:
    """Concurrent runs do not appear in each other's spans.

    Each worker opens its own agent-run scope and records tool spans inside it.
    Every buffered span has to carry the run id and the agent of the worker that
    produced it — a span filed under another run's id is a trace a user cannot
    read.
    """
    import asyncio

    from effgen.observability import tracing

    workers, spans = 8, 50 * _scale()
    report = ContentionReport("trace_isolation", workers, spans)
    tracing._SPAN_BUFFER.clear()
    drifted = 0
    lock = threading.Lock()

    def work(index: int) -> str:
        nonlocal drifted
        local_drift = 0
        with tracing.start_agent_run(f"thread-agent-{index}", "task"):
            context = tracing._RUN_CONTEXT.get()
            run_id = (context or {}).get("run_id")
            for _ in range(spans):
                with tracing.start_tool_call("calculator", "1+1"):
                    pass
                now = tracing._RUN_CONTEXT.get()
                if (now or {}).get("run_id") != run_id:
                    local_drift += 1
        with lock:
            drifted += local_drift
        return run_id

    thread_run_ids = barrier_map(workers, work, timeout=1800.0)

    async def task(index: int) -> str:
        with tracing.start_agent_run(f"task-agent-{index}", "task"):
            context = tracing._RUN_CONTEXT.get()
            run_id = (context or {}).get("run_id")
            for _ in range(spans):
                with tracing.start_tool_call("calculator", "1+1"):
                    await asyncio.sleep(0)
            after = tracing._RUN_CONTEXT.get()
            assert (after or {}).get("run_id") == run_id
        return run_id

    async def gather() -> list[str]:
        return list(await asyncio.gather(*(task(i) for i in range(workers))))

    task_run_ids = asyncio.run(gather())

    by_run: dict[str, set[str]] = {}
    for record in tracing.get_recent_spans(limit=1_000_000):
        if record.get("kind") != "agent":
            continue
        by_run.setdefault(record.get("run_id") or "", set()).add(record.get("agent") or "")
    mixed = {run: agents for run, agents in by_run.items() if len(agents) > 1}

    report.add("distinct_thread_run_ids", len(set(thread_run_ids)), workers)
    report.add("distinct_task_run_ids", len(set(task_run_ids)), workers)
    report.add("run_id_drifted_mid_run", drifted, 0)
    report.add("run_ids_holding_two_agents", len(mixed), 0, detail=str(list(mixed)[:2]))
    return report


# ---------------------------------------------------------------------------
# 14-15 — real models: a live server under load, live agents sharing one registry
# ---------------------------------------------------------------------------


#: Requests per minute each provider's account allows for the model used here.
#: The scenario measures the server under concurrency, so it paces itself to
#: what the tier permits rather than measuring the provider's rate limiter.
_REQUESTS_PER_MINUTE = {"groq": 30, "openai": 500, "gemini": 15}


def _is_rate_limit(exc: BaseException) -> bool:
    """Whether *exc* is the account's tier refusing the call.

    Classified through the shipped classifier rather than by matching provider
    prose, so a new provider's wording does not quietly turn a quota refusal
    into a counted shared-state failure.
    """
    from effgen.models.errors import classify_provider_error

    try:
        if classify_provider_error(exc).rate_limited:
            return True
    except Exception:  # noqa: BLE001 - a classifier failure is not a rate limit
        pass
    text = str(exc)
    return "rate_limited" in text or "RateLimitExceeded" in text


def scenario_live_server_concurrency() -> ContentionReport:
    """A running server answering many requests at once keeps them separate.

    Each request asks for a marker only it was given, so an answer landing on
    the wrong response is visible rather than inferred.

    A response that is not a 200 is not automatically a defect: an upstream
    quota refusal is a real outcome, and what matters is that it arrives as a
    typed error envelope rather than as a 200 carrying the error as content, a
    hang, or an empty body. Every request therefore has to end in one of those
    two shapes, and every 200 has to carry its own answer and its own run id.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    import httpx

    family = _first_keyed_family()
    if family is None:
        pytest.skip("no cloud provider key available")

    workers, per_worker = 8, 8 * _scale()
    report = ContentionReport("live_server_concurrency", workers, per_worker)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    api_key = "contention-" + os.urandom(8).hex()
    # Every worker calls at the same instant, so the interval each one waits is
    # the whole pool's budget divided out.
    rpm = _REQUESTS_PER_MINUTE.get(family.provider, 60)
    pace = max(family.min_interval_s, workers * 60.0 / rpm)

    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ, EFFGEN_API_KEY=api_key, EFFGEN_HOME=home)
        log_path = Path(home) / "server.log"
        proc = _spawn_server(port, env, log_path)
        statuses: dict[int, int] = {}
        run_ids: list[str] = []
        answered = 0
        mismatched = 0
        malformed_refusals = 0
        usage_total = 0
        lock = threading.Lock()
        try:
            _wait_for_server(base, proc, log_path=log_path)
            client = httpx.Client(timeout=300.0)
            headers = {"Authorization": f"Bearer {api_key}"}

            def work(index: int) -> object:
                nonlocal answered, mismatched, malformed_refusals, usage_total
                local_ids: list[str] = []
                local_status: dict[int, int] = {}
                local_answered = 0
                local_mismatch = 0
                local_malformed = 0
                local_usage = 0
                try:
                    for n in range(per_worker):
                        marker = f"{index:02d}{n:02d}{os.urandom(3).hex()}"
                        response = client.post(
                            f"{base}/v1/chat/completions",
                            headers=headers,
                            json={
                                "model": family.model,
                                "messages": [{
                                    "role": "user",
                                    "content": (
                                        "Repeat this token exactly and write nothing "
                                        f"else: {marker}"
                                    ),
                                }],
                                "max_tokens": family.max_tokens,
                            },
                        )
                        local_status[response.status_code] = (
                            local_status.get(response.status_code, 0) + 1
                        )
                        if response.status_code != 200:
                            body = response.json()
                            if "error" not in body or not body["error"].get("type"):
                                local_malformed += 1
                            time.sleep(pace)
                            continue
                        body = response.json()
                        answer = body["choices"][0]["message"]["content"] or ""
                        local_answered += 1
                        if marker not in answer:
                            local_mismatch += 1
                        local_usage += int((body.get("usage") or {}).get("total_tokens") or 0)
                        meta = body.get("effgen") or {}
                        if meta.get("run_id"):
                            local_ids.append(meta["run_id"])
                        time.sleep(pace)
                    return None
                except Exception as exc:  # noqa: BLE001 - counted, not raised
                    return exc
                finally:
                    with lock:
                        run_ids.extend(local_ids)
                        answered += local_answered
                        mismatched += local_mismatch
                        malformed_refusals += local_malformed
                        usage_total += local_usage
                        for code, count in local_status.items():
                            statuses[code] = statuses.get(code, 0) + count

            errors = collect_exceptions(barrier_map(workers, work, timeout=3600.0))
            requests = workers * per_worker
            report.add("transport_errors", len(errors), 0, detail=str(errors[:2]))
            report.add("responses_received", sum(statuses.values()), requests)
            report.add("refusals_without_a_typed_envelope", malformed_refusals, 0)
            report.add("distinct_run_ids", len(set(run_ids)), len(run_ids))
            report.add("answers_carrying_their_own_marker", answered - mismatched, answered)
            # A run where the provider refused everything would satisfy every
            # row above without the server having answered anything, so the
            # scenario states how much real traffic it needs to have measured.
            #
            # When the shortfall IS the provider refusing — the upstream tier is
            # per project while these workers are not — there is nothing here to
            # assert about effGen, so this is a skip and not a red. The sibling
            # scenario excludes quota refusals the same way; it can read them off
            # the exception, while a refusal that arrives through the server is a
            # status code. Any other shortfall still fails the row below.
            if answered < requests // 2:
                refused = sum(
                    count for code, count in statuses.items()
                    if code in (402, 429, 502, 503, 504)
                )
                if refused and answered + refused >= requests // 2:
                    pytest.skip(
                        f"the provider refused {refused} of {requests} calls "
                        f"(status codes {sorted(c for c in statuses if c in (402, 429, 502, 503, 504))}); "
                        "too little traffic was served to measure contention"
                    )
            report.add(
                "enough_answers_to_measure",
                answered >= requests // 2,
                True,
                detail=f"{answered} of {requests} requests answered",
            )
            report.notes["statuses"] = dict(sorted(statuses.items()))
            report.notes["seconds_between_calls_per_worker"] = round(pace, 2)
            report.notes["total_tokens_billed"] = usage_total
            report.notes["model"] = family.model

            forged = client.post(
                f"{base}/v1/chat/completions",
                headers={"Authorization": "Bearer not-the-key"},
                json={"model": family.model,
                      "messages": [{"role": "user", "content": "hi"}]},
            )
            report.add("forged_token_status_after_load", forged.status_code, 401)
            client.close()
            assert proc.poll() is None, f"server exited with {proc.returncode} under load"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                proc.kill()
                proc.wait(timeout=20)
    return report


def scenario_live_agents_share_state() -> ContentionReport:
    """Many agents, one registry, one tracker: every run stays its own.

    Each run asks for a token only it was given — random, so two runs cannot
    collide on a substring — which makes an answer that belongs to another run
    visible in the text rather than inferred from a counter.
    """
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models._cost import CostTracker

    families = [f for f in LIVE_FAMILIES if f.available]
    if not families:
        pytest.skip("no cloud provider key available")

    workers, per_worker = 12, 8 * _scale()
    report = ContentionReport("live_agents_share_state", workers, per_worker)
    # Workers are dealt round-robin, so each family carries several of them and
    # they all call at once. A per-worker interval therefore paces nothing: four
    # workers waiting 4.2s each still put ~57 requests a minute on a tier that
    # allows 15, and the run then measures the provider's rate limiter rather
    # than the shared state it is here to measure. The family's budget is
    # divided across the workers dealt to it, the same way
    # ``scenario_live_server_concurrency`` divides its own.
    share = {
        family.model: sum(1 for i in range(workers) if families[i % len(families)] is family)
        for family in families
    }
    pace = {
        family.model: max(
            family.min_interval_s,
            share[family.model] * 60.0 / _REQUESTS_PER_MINUTE.get(family.provider, 60),
        )
        for family in families
    }
    tracker = CostTracker(storage=None)
    answers: list[tuple[str, str]] = []
    refusals: list[int] = []
    lock = threading.Lock()

    def work(index: int) -> object:
        family = families[index % len(families)]
        local: list[tuple[str, str]] = []
        refused = 0
        try:
            agent = Agent(config=AgentConfig(
                model=family.model,
                name=f"agent-{index}",
                max_tokens=family.max_tokens,
                enable_memory=False,
            ))
            try:
                for n in range(per_worker):
                    marker = f"{index:02d}{n:02d}{os.urandom(3).hex()}"
                    try:
                        response = agent.run(
                            f"Repeat this token exactly and write nothing else: {marker}"
                        )
                    except Exception as exc:  # noqa: BLE001 - classified below
                        if not _is_rate_limit(exc):
                            raise
                        # The account's tier refused this call. Skip the
                        # iteration rather than abandoning the worker, so the
                        # completed count stays exact and the shared-state
                        # assertions still see the rest of the run.
                        refused += 1
                        if pace[family.model]:
                            time.sleep(pace[family.model])
                        continue
                    usage = response.metadata or {}
                    tracker.record(
                        family.model.split(":")[0],
                        family.model.split(":", 1)[1],
                        int(usage.get("prompt_tokens") or 0),
                        int(usage.get("completion_tokens") or 0),
                        cost_usd=usage.get("cost_usd") or 0.0,
                    )
                    local.append((marker, str(response)))
                    if pace[family.model]:
                        time.sleep(pace[family.model])
            finally:
                agent.close()
            return None
        except Exception as exc:  # noqa: BLE001 - counted, not raised
            return exc
        finally:
            with lock:
                answers.extend(local)
                refusals.append(refused)

    errors = collect_exceptions(barrier_map(workers, work, timeout=3600.0))
    # A quota refusal is a real outcome of calling a rate-limited account, not a
    # shared-state defect, and it is the account's tier that decides whether one
    # arrives. Refused iterations are counted and subtracted, the way
    # ``scenario_live_server_concurrency`` treats a non-200: what this scenario
    # asserts is that no run's answer reaches another run, that the tracker
    # recorded exactly the runs that completed, and that nothing failed for any
    # other reason.
    refused_total = sum(refusals)
    runs = workers * per_worker - refused_total
    foreign = sum(
        1 for marker, text in answers
        if marker not in text and any(m in text for m, _ in answers if m != marker)
    )
    recorded = sum(stats.requests for stats in tracker._data.values())

    report.add("run_failures", len(errors), 0, detail=str(errors[:2]))
    report.add("runs_completed", len(answers), runs)
    report.add("answers_carrying_another_run_marker", foreign, 0)
    report.add("cost_tracker_requests", recorded, len(answers))
    report.notes["families"] = [f.model for f in families]
    report.notes["rate_limited_calls"] = refused_total
    return report


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, Callable[[], ContentionReport]] = {
    "tool_registry_first_access": scenario_tool_registry_first_access,
    "tool_registry_instance_identity": scenario_tool_registry_instance_identity,
    "provider_registry_reset_under_reads": scenario_provider_registry_reset_under_reads,
    "provider_registry_snapshot_round_trip": scenario_provider_registry_snapshot_round_trip,
    "provider_registry_middleware_identity": scenario_provider_registry_middleware_identity,
    "adapter_totals_non_streaming": scenario_adapter_totals_non_streaming,
    "adapter_totals_streaming": scenario_adapter_totals_streaming,
    "cost_tracker_records": scenario_cost_tracker_records,
    "cost_store_inserts": scenario_cost_store_inserts,
    "session_store_same_id": scenario_session_store_same_id,
    "session_store_distinct_ids": scenario_session_store_distinct_ids,
    "run_history_appends": scenario_run_history_appends,
    "trace_isolation": scenario_trace_isolation,
}

LIVE_SCENARIOS: dict[str, Callable[[], ContentionReport]] = {
    "live_server_concurrency": scenario_live_server_concurrency,
    "live_agents_share_state": scenario_live_agents_share_state,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_shared_state_holds_under_contention(name: str) -> None:
    """Every counted quantity is exactly what the work performed adds up to."""
    report = SCENARIOS[name]()
    print("\n" + report.table())
    report.assert_exact()


@pytest.mark.live
@pytest.mark.parametrize("name", sorted(LIVE_SCENARIOS))
def test_shared_state_holds_under_live_contention(name: str) -> None:
    """The same, driven through real provider calls and a running server."""
    report = LIVE_SCENARIOS[name]()
    print("\n" + report.table())
    report.assert_exact()
