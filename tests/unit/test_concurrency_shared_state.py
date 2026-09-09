"""State shared by concurrent agents stays consistent.

One process runs many agents: they share the tool registry, the provider
registry, an adapter instance, the cost tracker and the session store. These
cases pin the behaviour each of those has to hold when more than one thread is
inside it — a lookup that never sees a half-replaced registry, a session total
that equals the calls made, a session file that is one writer's whole history.

Bounded (each runs in well under a second) so the ordinary offline lane keeps
them; the long, heavily-overlapped versions are the contention scenarios in
``tests/stress/test_concurrency_contention.py``.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import re
import stat
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from effgen.core.session import Session
from effgen.models._cost import CostTracker
from effgen.models._cost_store import SQLiteCostStore
from effgen.models.base import GenerationResult, _stamp_cost, accumulate_stream_cost
from effgen.models.registry import ProviderRegistry
from effgen.tools.base_tool import BaseTool, ToolCategory, ToolMetadata, ToolResult
from effgen.tools.registry import ToolRegistry

PRICE = 0.001


def _run_together(workers: int, fn, *, interleaved: bool = False, timeout: float = 300.0):
    """Run ``fn(i)`` on *workers* threads that start at the same instant.

    Args:
        workers: Number of threads.
        fn: Called once per worker with the worker's index.
        interleaved: Shorten the interpreter's thread switch interval for the
            duration of the run. A read-modify-write that is not atomic still
            gives the right answer most of the time at the default 5 ms
            interval, because a worker usually finishes its loop body inside
            one slice; shortening it makes the interleaving a loaded machine
            produces anyway happen reliably here. It is switched on only once
            every thread has been started — at 1 us the interpreter switches
            often enough that starting the pool's threads is itself slow, and
            on a small runner they stop reaching the barrier in time.
        timeout: Seconds a worker waits at the barrier for the others.

    Returns:
        One result per worker, in worker order.
    """
    barrier = threading.Barrier(workers, timeout=timeout)
    previous = sys.getswitchinterval()

    def run(index: int):
        barrier.wait()
        return fn(index)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, index) for index in range(workers)]
        if interleaved:
            sys.setswitchinterval(1e-6)
        try:
            return [future.result() for future in futures]
        finally:
            sys.setswitchinterval(previous)


# ---------------------------------------------------------------------------
# The tool registry every agent shares
# ---------------------------------------------------------------------------


def _metadata(name: str) -> ToolMetadata:
    return ToolMetadata(
        name=name, description="test tool", category=ToolCategory.COMPUTATION,
        parameters=[],
    )


class _Simple(BaseTool):
    """A tool with no dependencies."""

    def __init__(self) -> None:
        super().__init__(_metadata("simple_probe"))

    async def _execute(self, **_kwargs: object) -> ToolResult:
        return ToolResult(success=True, data="ok")


class _NeedsSimple(BaseTool):
    """A tool that declares another tool as a dependency."""

    def __init__(self) -> None:
        super().__init__(_metadata("needs_simple_probe"))
        self._dependencies = ["simple_probe"]

    async def _execute(self, **_kwargs: object) -> ToolResult:
        return ToolResult(success=True, data="ok")


class _SlowDiscoveryRegistry(ToolRegistry):
    """A registry whose catalog discovery takes measurable time."""

    def __init__(self) -> None:
        super().__init__()
        self.discovery_started = threading.Event()
        self.may_finish = threading.Event()
        self._plugins_discovered = True

    def discover_builtin_tools(self) -> None:  # type: ignore[override]
        self.discovery_started.set()
        self.may_finish.wait(timeout=30.0)
        self.register_tool(_Simple)


def test_a_thread_arriving_during_discovery_waits_for_the_catalog() -> None:
    """A second caller sees the discovered catalog, never the empty registry."""
    registry = _SlowDiscoveryRegistry()
    first: list[list[str]] = []
    second: list[list[str]] = []

    discoverer = threading.Thread(target=lambda: first.append(registry.list_tools()))
    discoverer.start()
    assert registry.discovery_started.wait(timeout=30.0)

    latecomer = threading.Thread(target=lambda: second.append(registry.list_tools()))
    latecomer.start()
    # Long enough that a registry which does not wait would already have
    # answered with nothing.
    latecomer.join(timeout=0.5)
    still_waiting = latecomer.is_alive()

    registry.may_finish.set()
    discoverer.join(timeout=30.0)
    latecomer.join(timeout=30.0)

    assert still_waiting, "the second caller was answered while discovery was running"
    assert first == [["simple_probe"]]
    assert second == [["simple_probe"]], (
        f"a caller arriving during discovery saw {second} instead of the catalog"
    )


def test_a_tool_that_declares_a_dependency_loads() -> None:
    """Loading a dependency goes through the loader, so it cannot wait on itself."""
    registry = ToolRegistry()
    registry._builtins_discovered = True
    registry._plugins_discovered = True
    registry.register_tool(_Simple)
    registry.register_tool(_NeedsSimple)

    async def load() -> BaseTool:
        return await asyncio.wait_for(registry.get_tool("needs_simple_probe"), timeout=10)

    tool = asyncio.run(load())
    assert tool.metadata.name == "needs_simple_probe"
    assert "simple_probe" in registry._instances, "the dependency was not loaded"


def test_one_tool_instance_however_many_threads_ask() -> None:
    """Concurrent lookups from separate event loops share one instance."""
    registry = ToolRegistry()
    registry._builtins_discovered = True
    registry._plugins_discovered = True
    registry.register_tool(_Simple)

    identities = _run_together(8, lambda _i: id(registry.get_tool_sync("simple_probe")))
    assert len(set(identities)) == 1, f"{len(set(identities))} instances handed out"


class _CountedInit(BaseTool):
    """A tool that records how many times it was initialized."""

    initializations = 0
    _counter_lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__(_metadata("counted_init_probe"))

    async def initialize(self) -> None:
        # Long enough that every waiting thread is inside the loader while this
        # one runs, which is when a second initialize would start.
        await asyncio.sleep(0.05)
        with _CountedInit._counter_lock:
            _CountedInit.initializations += 1

    async def _execute(self, **_kwargs: object) -> ToolResult:
        return ToolResult(success=True, data="ok")


def test_a_tool_is_initialized_once_however_many_threads_ask() -> None:
    """Twelve callers on twelve event loops produce one initialization.

    ``initialize()`` is where a tool opens its connection, starts its worker or
    registers its handler. The loader lock is per event loop, so without a
    cross-loop claim each thread runs it again and the tool ends up holding one
    set of those resources per caller.
    """
    _CountedInit.initializations = 0
    registry = ToolRegistry()
    registry._builtins_discovered = True
    registry._plugins_discovered = True
    registry.register_tool(_CountedInit)

    tools = _run_together(12, lambda _i: registry.get_tool_sync("counted_init_probe"))

    assert _CountedInit.initializations == 1, (
        f"the tool was initialized {_CountedInit.initializations} times"
    )
    assert len({id(t) for t in tools}) == 1


class _FailsTwice(BaseTool):
    """A tool whose first two initializations raise."""

    attempts = 0

    def __init__(self) -> None:
        super().__init__(_metadata("fails_twice_probe"))

    async def initialize(self) -> None:
        _FailsTwice.attempts += 1
        if _FailsTwice.attempts < 3:
            raise RuntimeError("not ready")

    async def _execute(self, **_kwargs: object) -> ToolResult:
        return ToolResult(success=True, data="ok")


def test_a_failed_initialization_lets_the_next_caller_try_again() -> None:
    """Claiming a tool for initialization does not strand it when that fails."""
    _FailsTwice.attempts = 0
    registry = ToolRegistry()
    registry._builtins_discovered = True
    registry._plugins_discovered = True
    registry.register_tool(_FailsTwice)

    for _ in range(4):
        try:
            registry.get_tool_sync("fails_twice_probe")
        except RuntimeError:
            pass

    assert _FailsTwice.attempts == 3, "a failed attempt blocked every later one"
    assert "fails_twice_probe" in registry._initialized_tools
    assert not registry._initializing, "a claim was left behind"


def test_the_loader_locks_do_not_accumulate_with_calls() -> None:
    """The per-loop loader locks live and die with their event loops.

    ``get_tool_sync`` runs each call on an event loop of its own. Holding those
    locks in a plain dict keyed on the loop's id keeps one entry per call the
    loader ever ran, and hands a new loop the lock of a dead one that happened
    to land at the same address.
    """
    registry = ToolRegistry()
    registry._builtins_discovered = True
    registry._plugins_discovered = True
    registry.register_tool(_Simple)

    # Every call takes the loader path, because cleanup drops the instance.
    for _ in range(200):
        registry.get_tool_sync("simple_probe")
        asyncio.run(registry.cleanup_all())

    gc.collect()
    assert len(registry._loop_locks) <= 2, (
        f"{len(registry._loop_locks)} loader locks survived 200 calls"
    )


# ---------------------------------------------------------------------------
# The provider registry
# ---------------------------------------------------------------------------


@pytest.fixture
def provider_registry_restored():
    """Put the provider registry back exactly as it was after the test."""
    ProviderRegistry.register_builtins()
    snapshot = ProviderRegistry.snapshot()
    try:
        yield snapshot
    finally:
        ProviderRegistry.restore(snapshot)


def test_reset_never_publishes_a_partly_built_registry(
    provider_registry_restored, monkeypatch
) -> None:
    """Mid-reset, a reader still sees every provider that was registered.

    The observation is taken from inside one adapter module's registration hook,
    which runs part-way through the rebuild — the exact moment a registry that
    empties itself first has lost the providers it has not re-added yet.
    """
    from effgen.models import groq_adapter

    before = ProviderRegistry.list_providers()
    seen: list[list[str]] = []
    original = groq_adapter._register

    def observing_register() -> None:
        seen.append(ProviderRegistry.list_providers())
        original()

    monkeypatch.setattr(groq_adapter, "_register", observing_register)
    ProviderRegistry.reset()

    assert seen, "the registration hook did not run"
    assert seen[0] == before, (
        f"a reader part-way through reset() saw {len(seen[0])} providers "
        f"instead of {len(before)}"
    )
    assert ProviderRegistry.list_providers() == before


def test_restore_never_publishes_a_partly_built_registry(
    provider_registry_restored,
) -> None:
    """The same for restore(): the snapshot is installed in one step."""
    snapshot = provider_registry_restored
    before = ProviderRegistry.list_providers()
    seen: list[list[str]] = []

    class _Watching(dict):
        """Reports what a reader would see while restore() walks the snapshot."""

        def items(self):
            for name, record in super().items():
                seen.append(ProviderRegistry.list_providers())
                yield name, record

    ProviderRegistry.restore(_Watching(snapshot))

    assert seen, "the snapshot was not walked"
    assert seen[-1] == before, (
        f"a reader part-way through restore() saw {len(seen[-1])} providers "
        f"instead of {len(before)}"
    )
    assert ProviderRegistry.list_providers() == before


def test_lookups_hold_while_the_registry_is_replaced(provider_registry_restored) -> None:
    """A model registered before and after a swap resolves throughout it."""
    snapshot = provider_registry_restored
    provider = "groq" if "groq" in snapshot else sorted(snapshot)[0]
    target = f"{provider}:{sorted(snapshot[provider]['models'])[0]}"
    failures: list[str] = []
    stop = threading.Event()

    def swap(_i: int) -> None:
        try:
            for _ in range(90):
                ProviderRegistry.reset()
                ProviderRegistry.restore(snapshot)
        finally:
            stop.set()

    def read(_i: int) -> None:
        local: list[str] = []
        # Bounded as well as stop-driven: a reader that spins only on the event
        # would run until the suite's timeout if the swapper ever died, and the
        # count is high enough that the swapper finishes first on any machine.
        for _ in range(2_000_000):
            if stop.is_set():
                break
            try:
                ProviderRegistry.lookup(target)
            except Exception as exc:  # noqa: BLE001 - collected, not raised
                local.append(f"{type(exc).__name__}: {exc}")
        failures.extend(local)

    # Interleaved: at the default switch interval a reader often runs a whole
    # burst inside one slice and misses the window a swap is visible in, so a
    # registry that empties itself first still passes about half the time.
    _run_together(4, lambda i: swap(i) if i == 0 else read(i), interleaved=True)
    assert not failures, f"{len(failures)} lookups failed during the swap: {failures[:2]}"


def test_concurrent_restores_leave_the_catalog_whole(provider_registry_restored) -> None:
    """Restoring the same snapshot from several threads is not a race."""
    snapshot = provider_registry_restored
    expected_models = sum(len(rec["models"]) for rec in snapshot.values())

    def restore(_i: int) -> object:
        try:
            for _ in range(140):
                ProviderRegistry.restore(snapshot)
        except Exception as exc:  # noqa: BLE001 - collected, not raised
            return exc
        return None

    # Interleaved, and enough rounds that the threads are still overlapping
    # after the barrier: a catalog rewritten with an unguarded ``del`` raises in
    # every worker but the one that got there first, and this has to see that.
    errors = [
        r for r in _run_together(6, restore, interleaved=True)
        if isinstance(r, BaseException)
    ]
    assert not errors, f"restore raised under contention: {errors[:2]}"
    after = ProviderRegistry.snapshot()
    assert sum(len(rec["models"]) for rec in after.values()) == expected_models


def test_one_circuit_breaker_and_bulkhead_per_provider(provider_registry_restored) -> None:
    """The lazily-created middleware is created once, not once per caller."""
    provider = sorted(ProviderRegistry.list_providers())[0]
    record = ProviderRegistry.get_provider_info(provider)
    record.pop("_circuit_breaker", None)
    record.pop("_bulkhead", None)

    pairs = _run_together(16, lambda _i: (
        id(ProviderRegistry.get_circuit_breaker(provider)),
        id(ProviderRegistry.get_bulkhead(provider)),
    ))
    assert len({p[0] for p in pairs}) == 1
    assert len({p[1] for p in pairs}) == 1


# ---------------------------------------------------------------------------
# Money and tokens
# ---------------------------------------------------------------------------


class _Totals:
    """Stands in for an adapter: only the two counters matter here."""

    def __init__(self) -> None:
        self.total_cost = 0.0
        self.total_tokens = 0


def test_a_shared_adapter_counts_every_call_it_priced() -> None:
    """``total_cost`` equals the calls folded onto it, with none lost."""
    adapter = _Totals()
    workers, calls = 16, 400

    def work(_i: int) -> None:
        for _ in range(calls):
            _stamp_cost(adapter, GenerationResult(
                text="x", tokens_used=15, finish_reason="stop", model_name="m",
                metadata={"cost_usd": PRICE},
            ))

    _run_together(workers, work, interleaved=True)
    assert round(adapter.total_cost, 6) == round(workers * calls * PRICE, 6)


def test_a_shared_adapter_counts_every_streamed_call() -> None:
    """The same for streamed calls, which fold their totals after the stream."""
    adapter = _Totals()
    calls = 300

    def work(_i: int) -> None:
        for _ in range(calls):
            accumulate_stream_cost(
                adapter, PRICE, tokens=15, prompt_tokens=10, completion_tokens=5,
            )

    _run_together(8, work, interleaved=True)
    assert round(adapter.total_cost, 6) == round(8 * calls * PRICE, 6)
    assert adapter.total_tokens == 8 * calls * 15


def test_the_cost_tracker_counts_every_recorded_call(monkeypatch, tmp_path) -> None:
    """Requests, tokens and cost all equal what was recorded."""
    budget = tmp_path / "budget.json"
    budget.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(budget))
    tracker = CostTracker(storage=None)
    calls = 300

    def work(_i: int) -> None:
        for _ in range(calls):
            tracker.record("openai", "gpt-5-nano", 10, 5, cost_usd=PRICE)

    _run_together(8, work, interleaved=True)
    stats = tracker._data[("openai", "gpt-5-nano")]
    assert stats.requests == 8 * calls
    assert stats.prompt_tokens == 8 * calls * 10
    assert stats.completion_tokens == 8 * calls * 5
    assert round(stats.total_cost_usd, 6) == round(8 * calls * PRICE, 6)


def test_an_in_memory_cost_store_records_calls_from_every_thread() -> None:
    """An in-memory store is one database, not one per thread.

    A thread with a database of its own has no table to write to, and the
    tracker logs that failure and returns — so the call is billed by the
    provider and missing from the user's spend.
    """
    store = SQLiteCostStore(":memory:")
    calls = 25
    errors: list[str] = []

    def work(_i: int) -> None:
        for _ in range(calls):
            try:
                store.insert("openai", "gpt-5-nano", 10, 5, PRICE, 1.0)
            except Exception as exc:  # noqa: BLE001 - collected, not raised
                errors.append(f"{type(exc).__name__}: {exc}")

    _run_together(4, work)
    assert not errors, f"inserts failed: {errors[:2]}"
    rows = store.query_all()
    assert len(rows) == 4 * calls
    assert round(sum(row.cost_usd for row in rows), 6) == round(4 * calls * PRICE, 6)
    store.close()


def test_a_cost_store_still_records_after_it_is_closed() -> None:
    """Reopening a store leaves it writable, in memory as well as on disk.

    An in-memory database is created empty with its connection, so a store that
    reopens one has to lay the schema down again. Without that, every call
    recorded after a close is logged as a failure and dropped from the user's
    spend while the provider still bills it.
    """
    with tempfile.TemporaryDirectory() as directory:
        stores = {
            "memory": SQLiteCostStore(":memory:"),
            "file": SQLiteCostStore(os.path.join(directory, "costs.sqlite")),
        }
        for label, store in stores.items():
            store.insert("openai", "gpt-5-nano", 10, 5, PRICE, 1.0)
            store.close()
            store.insert("openai", "gpt-5-nano", 10, 5, PRICE, 1.0)
            assert store.query_all(), f"the {label} store recorded nothing after close()"
            store.close()


# ---------------------------------------------------------------------------
# The session store
# ---------------------------------------------------------------------------


def test_each_session_write_uses_its_own_temporary_file(tmp_path, monkeypatch) -> None:
    """Two writers of one session never share the file they write through.

    A shared temporary name lets two writes interleave into it, and the rename
    then publishes the mix — or finds nothing, because the other writer already
    moved it away.
    """
    sources: list[str] = []
    real_replace = os.replace

    def recording_replace(src, dst, **kwargs):
        sources.append(str(src))
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", recording_replace)

    for index in range(2):
        session = Session(session_id="shared", agent_name=f"agent-{index}")
        session.add_message("user", "hello")
        session.save(str(tmp_path))

    assert len(sources) == 2
    assert sources[0] != sources[1], (
        f"both writers published through the same file: {sources[0]}"
    )


def test_concurrent_writers_of_one_session_leave_it_readable(tmp_path) -> None:
    """Last writer wins, whole — never a truncated or blended session file."""
    written = []
    for index in range(6):
        session = Session(session_id="shared", agent_name=f"agent-{index}")
        for turn in range((index + 1) * 30):
            session.add_message("user", f"worker-{index}-turn-{turn}" * 8)
        written.append(session)

    errors: list[str] = []

    def save(index: int) -> None:
        try:
            written[index].save(str(tmp_path))
        except Exception as exc:  # noqa: BLE001 - collected, not raised
            errors.append(f"{type(exc).__name__}: {exc}")

    for _ in range(10):
        _run_together(6, save)
        loaded = Session.load("shared", str(tmp_path))
        assert len(loaded.messages) in {len(s.messages) for s in written}, (
            "the stored session holds a blend of two writers' histories"
        )
    assert not errors, f"saves failed: {errors[:2]}"


def test_a_session_write_that_fails_leaves_the_previous_one_in_place(tmp_path) -> None:
    """A failed write publishes nothing and leaves no temporary behind."""
    session = Session(session_id="kept", agent_name="agent")
    session.add_message("user", "first")
    path = session.save(str(tmp_path))
    original = json.loads(open(path).read())

    class _Unserializable:
        def __repr__(self) -> str:
            raise RuntimeError("cannot render")

    session.add_message("user", "second", broken=_Unserializable())
    with pytest.raises(RuntimeError, match="cannot render"):
        session.save(str(tmp_path))

    assert json.loads(open(path).read()) == original
    leftovers = [p for p in os.listdir(tmp_path) if p != "kept.json"]
    assert not leftovers, f"a failed write left {leftovers} behind"


def test_many_writers_of_distinct_sessions_all_land(tmp_path) -> None:
    """Every session written concurrently is listed and readable afterwards."""
    from effgen.core.session import SessionManager

    per_worker = 20

    def work(index: int) -> None:
        for n in range(per_worker):
            session = Session(session_id=f"s-{index}-{n}", agent_name="agent")
            session.add_message("user", "hello")
            session.save(str(tmp_path))

    _run_together(8, work)
    listed, unreadable = SessionManager(str(tmp_path)).scan()
    assert not unreadable, f"unreadable session files: {unreadable[:2]}"
    assert len(listed) == 8 * per_worker


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def test_concurrent_runs_do_not_appear_in_each_others_traces() -> None:
    """Every buffered span carries the run and agent that produced it."""
    from effgen.observability import tracing

    tracing._SPAN_BUFFER.clear()

    def work(index: int) -> str:
        with tracing.start_agent_run(f"agent-{index}", "task"):
            run_id = (tracing._RUN_CONTEXT.get() or {}).get("run_id")
            for _ in range(20):
                with tracing.start_tool_call("calculator", "1+1"):
                    pass
            assert (tracing._RUN_CONTEXT.get() or {}).get("run_id") == run_id
        return run_id

    run_ids = _run_together(6, work)
    assert len(set(run_ids)) == 6

    agents_by_run: dict[str, set[str]] = {}
    for record in tracing.get_recent_spans(limit=100_000):
        if record.get("kind") == "agent":
            agents_by_run.setdefault(record.get("run_id") or "", set()).add(
                record.get("agent") or ""
            )
    mixed = {run: agents for run, agents in agents_by_run.items() if len(agents) > 1}
    assert not mixed, f"spans from different agents share a run id: {mixed}"


def test_run_history_records_one_whole_line_per_run(tmp_path, monkeypatch) -> None:
    """Concurrent writers never interleave inside a history line."""
    from effgen.observability import run_log

    # Both, and in this order: EFFGEN_RUN_HISTORY_DIR takes precedence, and
    # something else in the suite may already have pointed it elsewhere.
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    per_worker = 40
    tag = os.urandom(4).hex()

    def work(index: int) -> None:
        for n in range(per_worker):
            run_log.record_run(
                task=f"task-{index}-{n}" * 20,
                model="groq:openai/gpt-oss-20b",
                output=f"answer-{index}-{n}" * 20,
                run_id=f"{tag}-{index}-{n}",
            )

    _run_together(8, work)

    lines: list[str] = []
    for path in run_log.history_dir().glob("*.jsonl"):
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    parsed = [json.loads(line) for line in lines]
    mine = [record for record in parsed if str(record.get("run_id", "")).startswith(tag)]
    assert len(parsed) == len(lines), "a history line was truncated or interleaved"
    assert len(mine) == 8 * per_worker
    assert len({record["run_id"] for record in mine}) == 8 * per_worker


def test_the_session_store_helper_writes_the_whole_payload(tmp_path) -> None:
    """The shared atomic write publishes complete contents, not a prefix."""
    from effgen.utils.atomic_file import atomic_write_text

    target = tmp_path / "state.json"
    payloads = ["x" * 200_000, "y" * 5, "z" * 90_000]

    def work(index: int) -> None:
        for _ in range(20):
            atomic_write_text(target, payloads[index % len(payloads)])

    _run_together(6, work)
    assert target.read_text(encoding="utf-8") in payloads
    assert list(os.listdir(tmp_path)) == ["state.json"]


def test_an_atomic_write_keeps_the_permissions_the_file_had(tmp_path) -> None:
    """Replacing a file does not change who can read it.

    The replacement is a different file renamed into place, so its mode is the
    one the destination ends up with. A new file gets what an ordinary write
    would have given it, and a mode the owner chose is carried across.
    """
    from effgen.utils.atomic_file import atomic_write_text

    fresh = tmp_path / "fresh.json"
    atomic_write_text(fresh, "{}")
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    assert stat.S_IMODE(fresh.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode), (
        "a new file did not get the mode an ordinary write would have given it"
    )

    kept = tmp_path / "kept.json"
    kept.write_text("{}", encoding="utf-8")
    os.chmod(kept, 0o600)
    atomic_write_text(kept, '{"second": true}')
    assert stat.S_IMODE(kept.stat().st_mode) == 0o600, "the owner's mode was discarded"


def test_a_session_kept_readable_stays_readable_when_it_is_rewritten(tmp_path) -> None:
    """Saving a session again does not narrow the permissions it had."""
    session = Session(session_id="modes", agent_name="agent")
    session.add_message("user", "first")
    path = Path(session.save(str(tmp_path)))
    first = stat.S_IMODE(path.stat().st_mode)

    session.add_message("user", "second")
    session.save(str(tmp_path))
    assert stat.S_IMODE(path.stat().st_mode) == first


def test_the_temporary_directory_is_left_clean() -> None:
    """The helper writes beside its destination, not into the system temp dir."""
    from effgen.utils.atomic_file import atomic_write_text

    with tempfile.TemporaryDirectory() as directory:
        target = os.path.join(directory, "sub", "state.txt")
        os.makedirs(os.path.dirname(target))
        atomic_write_text(target, "hello")
        assert os.listdir(os.path.dirname(target)) == ["state.txt"]


# ---------------------------------------------------------------------------
# Gates: keep a new call site from reintroducing either pattern
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "effgen"

#: ``base.fold_call_totals`` is where an adapter's cumulative cost and token
#: counters are written, and it holds the lock that makes the write whole. The
#: helper's own body is the one place allowed to assign them.
_TOTALS_HELPER = "effgen/models/base.py"


def _source_files() -> list[Path]:
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


def test_adapter_session_totals_are_folded_through_the_locked_helper() -> None:
    """No adapter folds a call onto the running totals with a bare ``+=``.

    ``self.total_cost += cost`` reads, adds and stores as separate steps, so two
    calls on one adapter can each read the same starting value and one call's
    money is dropped. Every adapter goes through the helper that holds the lock.
    """
    pattern = re.compile(r"\.total_(cost|tokens)\s*(\+=|=\s*getattr\()")
    offenders: list[str] = []
    for path in sorted((_PACKAGE_ROOT / "models").glob("*_adapter.py")) + [
        _PACKAGE_ROOT / "models" / "base.py"
    ]:
        relative = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        source = path.read_text(encoding="utf-8")
        if relative == _TOTALS_HELPER:
            # The helper's own body is where the counters are written, under
            # the lock; everything before it is ordinary module code.
            source = source.split("def fold_call_totals(", 1)[0]
        for number, line in enumerate(source.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert not offenders, (
        "fold a call's cost and tokens through effgen.models.base.fold_call_totals, "
        "which holds the lock that keeps the totals whole:\n" + "\n".join(offenders)
    )


def test_state_files_are_published_through_the_shared_atomic_write() -> None:
    """No module publishes state through a temporary named after the target.

    A name derived from the destination is shared by every writer of that file,
    so two writes interleave into it and the rename publishes the mix.
    """
    pattern = re.compile(r"""["'][^"']*\.tmp["']|with_suffix\(\s*["'][^"']*\.tmp["']""")
    allowed = {"effgen/utils/atomic_file.py"}
    offenders: list[str] = []
    for path in _source_files():
        relative = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        if relative in allowed:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line) and "os.replace" not in line:
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert not offenders, (
        "publish state with effgen.utils.atomic_file.atomic_write_text, which "
        "writes through a temporary of the writer's own:\n" + "\n".join(offenders)
    )
