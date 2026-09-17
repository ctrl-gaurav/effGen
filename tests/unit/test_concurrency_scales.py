"""Many agents at once pay for the same bookkeeping as one.

A process running one agent and a process running sixty-four run the same code.
What separates them is the bookkeeping every model call does — writing the spend
ledger, reading the budget, counting the prompt's tokens, appending the run's
history line — and each of those is shared. A shared thing that is held while it
waits on a disk turns concurrent agents into a queue.

These cases pin the shape of that bookkeeping rather than its speed, because a
wall-clock assertion on a shared machine measures the machine:

* concurrent recorded calls are written together rather than queueing for a
  transaction each, every recorded call reaches the ledger exactly once, and it
  is in the file by the time recording it returns;
* the budget's lock is not held while the ledger is read, so a caller arriving
  during a reading is not stopped by it, and spend recorded during a reading is
  not lost from it;
* the budget file is read from disk when it changes and not once per call;
* a token count is remembered however short the text is;
* a turn measures its prompt against the budget once;
* no lock on the run path is held across a call that goes to a disk or a socket.

The last is checked by reading the source, with an allowed list naming each
region that legitimately holds one.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from effgen.models import _adapter_utils, _cost
from effgen.models._cost import CostTracker, _load_budget
from effgen.models._cost_store import SQLiteCostStore


def _forget_budget_file() -> None:
    """Drop any remembered reading of the budget file."""
    getattr(_cost, "reset_budget_config_cache", lambda: None)()


def _write_stats(store: SQLiteCostStore) -> dict[str, int]:
    """What the ledger says it has written."""
    stats = getattr(store, "write_stats", None)
    assert stats is not None, "the ledger does not say what it wrote"
    return stats

# --------------------------------------------------------------------------
# the spend ledger
# --------------------------------------------------------------------------

#: Long enough that a caller waiting for one would be obvious, short enough that
#: the whole file still runs in well under a second.
WRITE_DELAY_S = 0.05


class _SlowConnection:
    """A SQLite connection whose writes take *delay* and say who made them."""

    def __init__(self, conn: sqlite3.Connection, seen: list[str], delay: float) -> None:
        self._conn = conn
        self._seen = seen
        self._delay = delay

    def _note(self, sql: str) -> None:
        if sql.strip().upper().startswith("INSERT"):
            self._seen.append(threading.current_thread().name)
            time.sleep(self._delay)

    def execute(self, sql: str, *args: Any) -> Any:
        self._note(sql)
        return self._conn.execute(sql, *args)

    def executemany(self, sql: str, *args: Any) -> Any:
        self._note(sql)
        return self._conn.executemany(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def __enter__(self) -> Any:
        return self._conn.__enter__()

    def __exit__(self, *args: Any) -> Any:
        return self._conn.__exit__(*args)


@pytest.fixture
def slow_store(tmp_path, monkeypatch):
    """A cost store whose writes are slow, and the threads that made them."""
    seen: list[str] = []
    opened = SQLiteCostStore._open

    def open_slowly(self: SQLiteCostStore) -> Any:
        return _SlowConnection(opened(self), seen, WRITE_DELAY_S)

    monkeypatch.setattr(SQLiteCostStore, "_open", open_slowly)
    store = SQLiteCostStore(str(tmp_path / "costs.sqlite"))
    try:
        yield store, seen
    finally:
        store.close()


def test_concurrent_calls_do_not_queue_for_a_transaction_each(slow_store) -> None:
    """Sixteen calls recorded at once cost a few writes, not sixteen in a row."""
    store, seen = slow_store
    workers = 16
    ready = threading.Barrier(workers + 1)

    def record() -> None:
        ready.wait(timeout=30)
        store.insert("openai", "m", 10, 5, 0.001)

    threads = [threading.Thread(target=record) for _ in range(workers)]
    for thread in threads:
        thread.start()
    ready.wait(timeout=30)
    start = time.monotonic()
    for thread in threads:
        thread.join(timeout=60)
    elapsed = time.monotonic() - start

    assert store.count() == workers
    assert len(seen) < workers / 2, f"{len(seen)} transactions for {workers} calls"
    assert elapsed < workers * WRITE_DELAY_S / 2, (
        f"{workers} concurrent calls took {elapsed:.3f}s against {WRITE_DELAY_S}s per "
        "write: they queued for a transaction each"
    )


@pytest.mark.parametrize("ending", ["os._exit(0)", "os.kill(os.getpid(), signal.SIGKILL)",
                                    "os.kill(os.getpid(), signal.SIGTERM)"])
def test_a_recorded_call_is_in_the_file_when_recording_returns(tmp_path, ending) -> None:
    """A process that ends right after recording, however it ends, has not lost a call.

    Guards against moving the write off the caller's thread: a process that
    exits without running its shutdown hooks, or is killed, then drops the calls
    still waiting to be written.
    """
    import effgen

    path = tmp_path / "costs.sqlite"
    # The child imports the same effgen this test did, whichever copy that is.
    root = str(Path(effgen.__file__).resolve().parents[1])
    code = textwrap.dedent(f"""
        import os, signal, sys, threading
        sys.meta_path = [f for f in sys.meta_path
                         if "__editable__" not in type(f).__module__]
        sys.path.insert(0, {root!r})
        from effgen.models._cost_store import SQLiteCostStore
        store = SQLiteCostStore({str(path)!r})
        def record():
            for _ in range(50):
                store.insert("openai", "m", 1, 1, 0.001)
        threads = [threading.Thread(target=record) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        {ending}
    """)
    subprocess.run([sys.executable, "-c", code], timeout=120, check=False,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})

    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 200
    finally:
        conn.close()


def test_a_read_is_not_held_up_by_calls_being_recorded(tmp_path) -> None:
    """A budget read while many threads record returns in about one write's time."""
    store = SQLiteCostStore(str(tmp_path / "costs.sqlite"))
    stop = threading.Event()

    def record() -> None:
        while not stop.is_set():
            store.insert("openai", "m", 1, 1, 0.0)

    threads = [threading.Thread(target=record) for _ in range(16)]
    for thread in threads:
        thread.start()
    slowest = 0.0
    try:
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            start = time.monotonic()
            store.spend_since(0.0)
            slowest = max(slowest, time.monotonic() - start)
            time.sleep(0.05)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
        store.close()

    assert slowest < 0.5, f"a read waited {slowest:.2f}s behind calls being recorded"


def test_a_store_that_recorded_leaves_no_thread_behind(tmp_path) -> None:
    """A store costs nothing once its caller is done with it."""
    before = threading.active_count()
    for index in range(20):
        store = SQLiteCostStore(str(tmp_path / f"costs-{index}.sqlite"))
        store.insert("openai", "m", 1, 1, 0.001)
        assert store.count() == 1
        del store
    assert threading.active_count() <= before


def test_every_recorded_call_reaches_the_ledger_once(tmp_path) -> None:
    store = SQLiteCostStore(str(tmp_path / "costs.sqlite"))
    workers, per_worker = 16, 40
    ready = threading.Barrier(workers)

    def record(index: int) -> None:
        ready.wait(timeout=30)
        for _step in range(per_worker):
            store.insert("openai", f"m{index}", 1, 1, 0.001, timestamp=time.time())

    threads = [threading.Thread(target=record, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    try:
        assert store.count() == workers * per_worker
        assert store.spend_since(0.0) == pytest.approx(workers * per_worker * 0.001)
        assert _write_stats(store)["events"] == workers * per_worker
    finally:
        store.close()


def test_a_read_of_the_ledger_sees_the_calls_recorded_before_it(tmp_path) -> None:
    """A reader gets the whole ledger, not the part that happens to be written."""
    store = SQLiteCostStore(str(tmp_path / "costs.sqlite"))
    try:
        for _ in range(50):
            store.insert("openai", "m", 1, 1, 0.002)
        assert store.spend_since(0.0) == pytest.approx(0.1)
        assert store.count() == 50
        assert store.query_all() and len(store.query_all()) == 50
    finally:
        store.close()


def test_a_closed_ledger_keeps_what_was_recorded_before_it_closed(tmp_path) -> None:
    path = tmp_path / "costs.sqlite"
    store = SQLiteCostStore(str(path))
    for _ in range(5):
        store.insert("openai", "m", 1, 1, 0.01)
    store.close()

    reopened = SQLiteCostStore(str(path))
    try:
        assert reopened.count() == 5
    finally:
        reopened.close()


def test_the_ledger_writes_many_calls_in_one_transaction(slow_store) -> None:
    """Concurrent calls are written together rather than one transaction each."""
    store, _seen = slow_store
    workers = 16
    ready = threading.Barrier(workers)

    def record(_index: int) -> None:
        ready.wait(timeout=30)
        for _ in range(8):
            store.insert("openai", "m", 1, 1, 0.001)

    threads = [threading.Thread(target=record, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    store.flush()

    events = _write_stats(store)["events"]
    batches = _write_stats(store)["batches"]
    assert events == workers * 8
    assert batches < events, (batches, events)


def test_each_ledger_connection_keeps_a_bounded_page_cache(tmp_path) -> None:
    """A connection per thread must not mean a full default page cache per thread."""
    store = SQLiteCostStore(str(tmp_path / "costs.sqlite"))
    sizes: list[int] = []

    def read() -> None:
        store.spend_since(0.0)
        sizes.append(store._conn().execute("PRAGMA cache_size;").fetchone()[0])

    try:
        threads = [threading.Thread(target=read) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        store.close()

    # A negative size is KiB; SQLite's own default is -2000.
    assert len(sizes) == 4
    assert all(-1024 <= size < 0 for size in sizes), sizes


# --------------------------------------------------------------------------
# the budget
# --------------------------------------------------------------------------


class _BlockingStore:
    """A ledger whose read blocks until it is let go."""

    def __init__(self) -> None:
        self.reading = threading.Event()
        self.release = threading.Event()
        self.reads = 0

    def spend_today(self) -> float:
        self.reads += 1
        self.reading.set()
        self.release.wait(timeout=30)
        return 0.25

    def spend_month(self) -> float:
        return self.spend_today()

    def query_today(self):  # pragma: no cover - the aggregate above is used
        return []


@pytest.fixture
def budget(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"daily": 1000.0}))
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(path))
    _forget_budget_file()
    yield path
    _forget_budget_file()


def test_the_budget_lock_is_free_while_the_ledger_is_read(budget) -> None:
    """A reading in flight does not stop another caller touching the budget."""
    store = _BlockingStore()
    tracker = CostTracker(storage=store)

    # One reading completes, so there is a previous number to stand on.
    store.release.set()
    assert tracker._period_spend("daily") == pytest.approx(0.25)
    store.release.clear()
    store.reading.clear()
    # Expired a moment ago, as a reading under steady load is when the next
    # caller arrives.
    tracker._period_spend_cache["daily"] = (time.monotonic() - 1.5, 0.25)

    reader = threading.Thread(target=tracker._period_spend, args=("daily",))
    reader.start()
    try:
        assert store.reading.wait(timeout=10), "the reading never started"
        # The lock guarding the cached reading is not held across the read, so
        # this returns rather than waiting for the reader.
        done = threading.Event()

        def fold() -> None:
            tracker._add_period_spend(0.5)
            done.set()

        threading.Thread(target=fold, daemon=True).start()
        assert done.wait(timeout=5), "the budget lock was held while the ledger was read"

        # And a second caller gets an answer rather than queueing behind it.
        answered: list[float] = []

        def ask() -> None:
            answered.append(tracker._period_spend("daily"))

        asking = threading.Thread(target=ask, daemon=True)
        asking.start()
        asking.join(timeout=5)
        assert answered, "a caller arriving during a reading waited for it"
    finally:
        store.release.set()
        reader.join(timeout=10)

    assert store.reads == 2, store.reads


class _SpendStore(_BlockingStore):
    """A ledger whose read blocks and then returns a fixed total."""

    def __init__(self, total: float) -> None:
        super().__init__()
        self.total = total
        self.inserted: list[dict[str, Any]] = []

    def insert(self, **kwargs: Any) -> None:
        self.inserted.append(kwargs)

    def spend_today(self) -> float:
        self.reads += 1
        self.reading.set()
        self.release.wait(timeout=30)
        return self.total


def test_spend_recorded_while_the_ledger_is_read_is_not_lost(budget) -> None:
    """A call recorded during a reading still counts once the reading lands.

    The reading may have summed the ledger before that call reached it, so the
    call is added to the reading when it is published.
    """
    from effgen.models.errors import BudgetExceededError

    budget.write_text(json.dumps({"daily": 1.0}))
    _forget_budget_file()
    store = _SpendStore(total=0.0)      # summed before the call below landed
    tracker = CostTracker(storage=store)
    tracker._period_spend_cache["daily"] = (time.monotonic() - 1.5, 0.0)

    reader = threading.Thread(target=tracker._period_spend, args=("daily",))
    reader.start()
    assert store.reading.wait(timeout=10), "the reading never started"
    recorder = threading.Thread(
        target=tracker.record, args=("openai", "m", 1, 1), kwargs={"cost_usd": 0.6})
    recorder.start()
    recorder.join(timeout=0.5)
    store.release.set()
    reader.join(timeout=10)
    recorder.join(timeout=10)

    with pytest.raises(BudgetExceededError):
        tracker.record("openai", "m", 1, 1, cost_usd=0.6)


def test_a_reading_too_old_to_trust_is_not_used_while_a_new_one_is_made(budget) -> None:
    """After a quiet spell a caller waits for the new reading rather than the old one."""
    from effgen.models.errors import BudgetExceededError

    budget.write_text(json.dumps({"daily": 1.0}))
    _forget_budget_file()
    store = _SpendStore(total=5.0)      # another process spent this meanwhile
    tracker = CostTracker(storage=store)
    tracker._period_spend_cache["daily"] = (time.monotonic() - 3600.0, 0.0)

    reader = threading.Thread(target=tracker._period_spend, args=("daily",))
    reader.start()
    assert store.reading.wait(timeout=10), "the reading never started"
    outcome: list[str] = []

    def ask() -> None:
        try:
            tracker.check_preflight("openai", "m")
            outcome.append("allowed")
        except BudgetExceededError:
            outcome.append("refused")

    asking = threading.Thread(target=ask)
    asking.start()
    asking.join(timeout=0.5)
    answered_early = list(outcome)
    store.release.set()
    reader.join(timeout=10)
    asking.join(timeout=10)

    assert answered_early == [], "an hour-old reading was used while a new one was made"
    assert outcome == ["refused"]


def _age(path: Path, seconds: float = 60.0) -> None:
    """Make *path* look as if it was last written *seconds* ago."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_the_budget_file_is_read_once_however_many_calls_check_it(budget, monkeypatch) -> None:
    # A budget set earlier, not one being written right now. The file's change
    # time cannot be set back, so the window is shortened and waited out.
    monkeypatch.setattr(_cost, "_BUDGET_RACY_WINDOW_S", 0.05)
    time.sleep(0.1)
    _forget_budget_file()
    assert _load_budget() == {"daily": 1000.0}
    stats = getattr(_cost, "_budget_config_stats", None)
    assert stats is not None, "the budget loader does not say when it read the file"

    reads = stats["read"]
    for _ in range(200):
        assert _load_budget() == {"daily": 1000.0}

    assert stats["read"] == reads, f"{stats['read'] - reads} reads for 200 checks"
    assert stats["reused"] >= 200


def test_the_budget_file_is_read_again_when_it_changes(budget) -> None:
    assert _load_budget() == {"daily": 1000.0}
    time.sleep(0.01)
    budget.write_text(json.dumps({"daily": 2.0}))

    assert _load_budget() == {"daily": 2.0}


def test_a_budget_rewritten_within_the_same_second_is_seen(budget) -> None:
    """A rewrite of the same size, stamped with the same time, is still read.

    A file system stamps modification times at a granularity of its own, so two
    writes close together can leave ``stat`` saying nothing changed.
    """
    budget.write_text(json.dumps({"daily": 1.0}))
    before = os.stat(budget)
    assert _load_budget() == {"daily": 1.0}

    with open(budget, "r+", encoding="utf-8") as handle:   # same file, same size
        handle.write(json.dumps({"daily": 7.0}))
        handle.truncate()
    os.utime(budget, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(budget)
    assert (after.st_mtime_ns, after.st_size, after.st_ino) == (
        before.st_mtime_ns, before.st_size, before.st_ino)

    assert _load_budget() == {"daily": 7.0}


def test_a_budget_rewritten_with_an_old_time_stamp_put_back_is_seen(budget) -> None:
    """``cp -p``, ``touch -r`` and unpacked archives set a file's time stamp back."""
    budget.write_text(json.dumps({"daily": 1.0}))
    _age(budget)
    before = os.stat(budget)
    time.sleep(0.05)
    assert _load_budget() == {"daily": 1.0}

    with open(budget, "r+", encoding="utf-8") as handle:   # same file, same size
        handle.write(json.dumps({"daily": 7.0}))
        handle.truncate()
    os.utime(budget, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert _load_budget() == {"daily": 7.0}


def test_a_budget_rewritten_between_tests_is_seen_by_the_next(budget, tmp_path) -> None:
    """Each test's own file is read, whatever the previous one left behind."""
    for cap in (100.0, 0.01, 100.0):
        budget.write_text(json.dumps({"daily": cap}))
        assert _load_budget() == {"daily": cap}


def test_a_caller_may_change_the_budget_it_was_given(budget) -> None:
    """Each caller gets its own mapping, as it always did."""
    first = _load_budget()
    first["daily"] = 0.0

    assert _load_budget() == {"daily": 1000.0}


def test_a_budget_file_that_is_not_a_mapping_is_treated_as_no_budget(budget) -> None:
    budget.write_text(json.dumps([1, 2, 3]))

    assert _load_budget() == {}


# --------------------------------------------------------------------------
# counting a prompt
# --------------------------------------------------------------------------


def test_a_short_text_is_counted_once_and_then_remembered() -> None:
    """The texts a tool call and its result are made of are short and repeated."""
    text = f"step-result-{time.time_ns()}"
    before = dict(_adapter_utils._token_count_stats)
    counts = [_adapter_utils.estimate_tokens(text) for _ in range(20)]
    after = dict(_adapter_utils._token_count_stats)

    assert len(set(counts)) == 1
    assert after["encoded"] - before["encoded"] <= 1, (
        f"a {len(text)}-character text was encoded "
        f"{after['encoded'] - before['encoded']} times"
    )
    assert after["reused"] - before["reused"] >= 19


def test_a_turn_measures_its_prompt_against_the_budget_once(monkeypatch) -> None:
    """A budget check counts every message, so asking twice counts them twice."""
    from effgen.core import agent_loop, thread_budget

    counted = {"n": 0}
    real = thread_budget.count_prompt_tokens

    def counting(prompt: Any, *, model: Any = None) -> int:
        counted["n"] += 1
        return real(prompt, model=model)

    monkeypatch.setattr(thread_budget, "count_prompt_tokens", counting)

    budget = thread_budget.ContextBudget(budget_tokens=100_000)
    prompt = [{"role": "user", "content": "how many tokens is this prompt"}]
    source = Path(agent_loop.__file__).read_text(encoding="utf-8")

    # The loop measures the prompt it built, then acts on that measurement; it
    # does not measure the same prompt again to decide what to do about it.
    assert budget.exceeded(prompt) is False
    assert counted["n"] == 1
    assert source.count("state.budget.exceeded(prompt)") == 2, (
        "a prompt is measured once per version of it: once before compaction and "
        "once after each round"
    )
    assert "over_budget = state.budget.exceeded(prompt)" in source


# --------------------------------------------------------------------------
# the run history
# --------------------------------------------------------------------------


def test_the_run_history_directory_is_made_once_per_process(tmp_path, monkeypatch) -> None:
    from effgen.observability import run_log

    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    getattr(run_log, "_history_dirs_made", set()).clear()
    made = {"n": 0}
    real = Path.mkdir

    def counting(self: Path, *args: Any, **kwargs: Any) -> None:
        if str(self).endswith("runs"):
            made["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", counting)
    for index in range(10):
        run_log._append_to_file({"ts": "2026-09-16T00:00:00", "run_id": str(index)})

    assert made["n"] == 1, f"the directory was created {made['n']} times for 10 runs"
    written = (tmp_path / "runs" / "2026-09-16.jsonl").read_text().splitlines()
    assert len(written) == 10


def test_a_history_directory_removed_under_a_running_process_is_made_again(
        tmp_path, monkeypatch) -> None:
    import shutil

    from effgen.observability import run_log

    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    getattr(run_log, "_history_dirs_made", set()).clear()
    run_log._append_to_file({"ts": "2026-09-16T00:00:00", "run_id": "before"})
    shutil.rmtree(tmp_path / "runs")

    run_log._append_to_file({"ts": "2026-09-16T00:00:01", "run_id": "after"})

    written = (tmp_path / "runs" / "2026-09-16.jsonl").read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in written] == ["after"]


# --------------------------------------------------------------------------
# no lock held across a disk or a socket, on the run path
# --------------------------------------------------------------------------

#: The modules every model call passes through. A lock held across I/O in one of
#: these is held while other agents are trying to make their own calls.
RUN_PATH_MODULES = (
    "effgen/models/_cost.py",
    "effgen/models/_cost_store.py",
    "effgen/models/_adapter_utils.py",
    "effgen/models/base.py",
    "effgen/models/_usage.py",
    "effgen/observability/run_log.py",
    "effgen/observability/tracing_buffer.py",
    "effgen/core/ledger.py",
    "effgen/core/agent_loop.py",
    "effgen/core/agent_runtime.py",
    "effgen/core/agent_generation.py",
    "effgen/tools/registry.py",
)

#: The regions that hold a lock across a call that can block, and why each is
#: allowed to. Keyed by ``file:function``; a region not named here fails the
#: test, and a name here that no longer matches a region fails it too.
ALLOWED_LOCKED_IO = {
    "effgen/models/_cost_store.py:_init_schema":
        "once, when the store is constructed, before anything can be recorded",
    "effgen/models/_cost_store.py:_drain":
        "the batch write itself; the lock around it is only the in-memory store's "
        "shared connection",
    "effgen/models/_cost_store.py:_write_through":
        "the batch write itself, by whichever caller took the write token; nobody "
        "waits on that token — a caller that finds it taken waits for the batch to "
        "end, which is one transaction however many calls are waiting",
    "effgen/models/_cost_store.py:_conn":
        "an in-memory store opening its one shared connection, once",
    "effgen/tools/registry.py:_ensure_builtins":
        "built-in tool discovery, once per process, at the first lookup",
    "effgen/tools/registry.py:get_tool":
        "a tool's own initialize(), once per tool, under an asyncio lock of the "
        "calling thread's event loop, so no other thread waits on it",
    "effgen/models/_cost_store.py:query_since":
        "a read, and the lock is taken only by an in-memory store, which shares "
        "one connection; a file-backed store gives each thread its own",
    "effgen/models/_cost_store.py:spend_since":
        "a read, and the lock is taken only by an in-memory store; the budget "
        "read that reaches it is made by one caller at a time",
    "effgen/models/_cost_store.py:_count_rows":
        "a read, and the lock is taken only by an in-memory store",
    "effgen/models/_cost_store.py:count_since":
        "a read, and the lock is taken only by an in-memory store",
    "effgen/models/_cost_store.py:query_all":
        "a report, asked for by a person rather than by a run",
    "effgen/models/_cost_store.py:prune":
        "asked for by a person, not by a run",
    "effgen/models/_cost_store.py:cleanup":
        "asked for by a person, not by a run",
}

_FILE_IO = {
    "open", "read", "read_text", "read_bytes", "write", "write_text", "write_bytes",
    "readline", "readlines", "writelines", "flush", "fsync", "mkdir", "makedirs",
    "stat", "lstat", "exists", "unlink", "remove", "rename", "replace", "listdir",
    "walk", "glob", "rglob", "iterdir", "touch", "chmod", "rmtree", "scandir",
}
_DB_IO = {
    "execute", "executemany", "executescript", "commit", "rollback", "connect",
    "fetchone", "fetchall", "fetchmany", "cursor",
}
_NET_IO = {"send", "sendall", "recv", "urlopen", "do_handshake"}
_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "request", "stream"}
_HTTP_RECEIVERS = ("client", "session", "http", "requests", "urllib", "socket")


def _call_name(node: ast.Call) -> tuple[str, str]:
    parts: list[str] = []
    current: Any = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return ".".join(parts), (parts[-1] if parts else "")


def _blocking(dotted: str, attribute: str) -> str | None:
    receiver = dotted.rsplit(".", 1)[0].lower() if "." in dotted else ""
    root = dotted.split(".", 1)[0]
    if attribute in _DB_IO:
        return "database"
    if attribute in _NET_IO:
        return "socket"
    if attribute in _FILE_IO and root not in ("json", "pickle", "yaml", "io"):
        return "file"
    if attribute in _HTTP_VERBS and any(word in receiver for word in _HTTP_RECEIVERS):
        return "http"
    if attribute == "sleep" and root in ("time", "asyncio"):
        return "sleep"
    return None


def _lock_name(item: ast.withitem) -> str | None:
    node = item.context_expr
    if isinstance(node, ast.Call):
        node = node.func
    parts: list[str] = []
    current: Any = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    dotted = ".".join(parts)
    words = ("lock", "mutex", "semaphore", "condition", "_exclusive", "_pending")
    return dotted if any(word in dotted.lower() for word in words) else None


def _is_acquire(node: ast.AST) -> bool:
    """``<lock>.acquire(...)``, as a statement or as an ``if`` test."""
    if isinstance(node, ast.Expr):
        node = node.value
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "acquire":
        return False
    return _lock_name(ast.withitem(context_expr=node.func.value)) is not None


def _regions_holding_io(path: Path) -> dict[str, list[str]]:
    """Each function's lock regions that reach a blocking call.

    A region is a ``with <lock>:`` block, the body of ``if <lock>.acquire(...):``,
    or the statements after a bare ``<lock>.acquire()``. A call inside it counts
    when it blocks itself, or when it calls a function of the same module that
    does (one level down), which is how a region usually reaches its I/O.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, list[str]] = {}
    direct: dict[str, list[str]] = {}

    def own_calls(function: ast.AST) -> list[ast.Call]:
        calls: list[ast.Call] = []
        stack = list(ast.iter_child_nodes(function))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Call):
                calls.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return calls

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for call in own_calls(node):
                why = _blocking(*_call_name(call))
                if why:
                    direct.setdefault(node.name, []).append(why)

    def reasons_in(nodes: list[ast.AST]) -> list[str]:
        reasons = []
        for top in nodes:
            for inner in ast.walk(top):
                if not isinstance(inner, ast.Call):
                    continue
                dotted, attribute = _call_name(inner)
                why = _blocking(dotted, attribute)
                if why:
                    reasons.append(f"{dotted} ({why}, line {inner.lineno})")
                    continue
                local = dotted.split(".")
                own = len(local) == 1 or (len(local) == 2 and local[0] in ("self", "cls"))
                if own and attribute in direct:
                    kinds = ", ".join(sorted(set(direct[attribute])))
                    reasons.append(f"{dotted} -> {kinds} (line {inner.lineno})")
        return reasons

    def walk(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, child.name)
                continue
            region: list[ast.AST] = []
            if isinstance(child, (ast.With, ast.AsyncWith)) and any(
                _lock_name(item) for item in child.items
            ):
                region = list(child.body)
            elif isinstance(child, ast.If) and _is_acquire(child.test):
                region = list(child.body)
            body = getattr(child, "body", None)
            if isinstance(body, list):
                for index, statement in enumerate(body):
                    if _is_acquire(statement):
                        region.extend(body[index + 1:])
            reasons = reasons_in(region)
            if reasons:
                found.setdefault(function, []).extend(reasons)
            walk(child, function)

    for statement_index, statement in enumerate(tree.body):
        if _is_acquire(statement):
            reasons = reasons_in(tree.body[statement_index + 1:])
            if reasons:
                found.setdefault("<module>", []).extend(reasons)
    walk(tree, "<module>")
    return found


def test_no_lock_on_the_run_path_is_held_across_a_disk_or_a_socket() -> None:
    root = Path(__file__).resolve().parents[2]
    offenders: dict[str, list[str]] = {}
    seen: set[str] = set()
    for relative in RUN_PATH_MODULES:
        path = root / relative
        assert path.exists(), relative
        for function, reasons in _regions_holding_io(path).items():
            key = f"{relative}:{function}"
            seen.add(key)
            if key not in ALLOWED_LOCKED_IO:
                offenders[key] = reasons

    assert not offenders, (
        "a lock on the run path is held across a call that can block:\n"
        + "\n".join(f"  {key}: {value}" for key, value in sorted(offenders.items()))
    )
    stale = sorted(set(ALLOWED_LOCKED_IO) - seen)
    assert not stale, f"the allowed list names regions that no longer hold I/O: {stale}"
