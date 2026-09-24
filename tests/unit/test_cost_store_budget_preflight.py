"""The budget preflight must not read the whole ledger.

Every model call runs a budget preflight when a budget is configured, and the
ledger it reads gains a row per model call and is never pruned in normal
operation. These tests pin the three things that keeps bounded: the query is
answered from an index, the total is summed in SQLite rather than in Python,
and a reading is reused for a moment instead of being re-read per call. They
also pin what must not change: the totals the budget gate and ``effgen cost``
report, and the fact that a store which does not offer the aggregate still
works.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time

import pytest

from effgen.models._cost import CostTracker
from effgen.models._cost_store import CostEvent, SQLiteCostStore

# The retention constants arrived with the size ceiling. Imported inside the
# tests that use them so the rest of the module still loads — and still fails
# on its own assertions — against a tree that does not have them yet.

DAY = 86400.0


def _ledger(path, rows, span_days=60.0, cost=1e-6):
    """Write *rows* events spread evenly over the last *span_days*."""
    store = SQLiteCostStore(str(path))
    now = time.time()
    conn = sqlite3.connect(str(path))
    with conn:
        conn.executemany(
            "INSERT INTO cost_events (provider, model, prompt_tokens, "
            "completion_tokens, cost_usd, timestamp) VALUES (?,?,?,?,?,?)",
            [("openai", "gpt-5-nano", 10, 5, cost,
              now - (i / max(1, rows)) * span_days * DAY) for i in range(rows)],
        )
    conn.close()
    return store


def _budget(tmp_path, monkeypatch, **caps):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(caps))
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(path))
    return path


# ---------------------------------------------------------------- query shape


def test_timestamp_index_exists_and_is_used(tmp_path):
    """The budget query seeks an index instead of walking the table.

    The pre-change schema indexed ``(provider, model, timestamp)``. A filter on
    time alone cannot seek into an index that leads with ``provider``, so the
    SQLite walked every distinct prefix — the cost of a check grew with the
    ledger rather than with the window it asked about.
    """
    store = _ledger(tmp_path / "c.sqlite", 2000)
    conn = sqlite3.connect(str(tmp_path / "c.sqlite"))
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='cost_events'")]
    assert "idx_cost_events_timestamp" in names

    plan = " ".join(str(r[-1]) for r in conn.execute(
        "EXPLAIN QUERY PLAN SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_events "
        "WHERE timestamp >= ?", (time.time() - DAY,)))
    conn.close()
    store.close()
    assert "idx_cost_events_timestamp" in plan
    assert "SCAN cost_events" not in plan


def test_spend_query_does_not_examine_the_whole_table(tmp_path):
    """Work stays proportional to the window, not to the ledger.

    Counted in VDBE steps rather than milliseconds: a wall-clock assertion on a
    shared machine is flaky, and the thing worth pinning is that the rows
    outside the window are never visited at all.
    """
    path = tmp_path / "c.sqlite"
    store = _ledger(path, 60_000, span_days=60.0)
    since = time.time() - DAY  # 1/60th of the ledger

    steps = [0]

    def bump():
        steps[0] += 1
        return 0

    conn = sqlite3.connect(str(path))
    conn.set_progress_handler(bump, 1)
    conn.execute("SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_events "
                 "WHERE timestamp >= ?", (since,)).fetchall()
    conn.set_progress_handler(None, 0)
    conn.close()
    store.close()

    # A full pass over 60,000 rows costs well over 100,000 steps; one day's
    # slice is ~1,000 rows. The margin is wide on purpose — this fails on a
    # scan and passes on any sane index.
    assert steps[0] < 30_000, f"budget sum examined too much: {steps[0]} steps"


def test_spend_since_matches_the_rows_it_summarises(tmp_path):
    """The aggregate returns exactly what summing the rows returns."""
    path = tmp_path / "c.sqlite"
    store = _ledger(path, 5000, span_days=60.0, cost=3e-6)
    for window in (DAY, 7 * DAY, 30 * DAY):
        since = time.time() - window
        rows = sum(e.cost_usd for e in store.query_since(since))
        assert store.spend_since(since) == pytest.approx(rows, abs=1e-12)
    assert store.spend_today() == pytest.approx(
        sum(e.cost_usd for e in store.query_today()), abs=1e-12)
    assert store.spend_month() == pytest.approx(
        sum(e.cost_usd for e in store.query_month()), abs=1e-12)
    store.close()


def test_index_is_added_to_a_ledger_written_before_it_existed(tmp_path):
    """An existing ledger gains the index when it is next opened."""
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute(
            "CREATE TABLE cost_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "provider TEXT NOT NULL, model TEXT NOT NULL, prompt_tokens INTEGER "
            "NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, "
            "cost_usd REAL NOT NULL DEFAULT 0.0, timestamp REAL NOT NULL)")
        conn.execute("INSERT INTO cost_events (provider, model, cost_usd, timestamp) "
                     "VALUES ('openai','gpt-5-nano', 0.25, ?)", (time.time(),))
    conn.close()

    store = SQLiteCostStore(str(path))
    assert store.spend_today() == pytest.approx(0.25)
    conn = sqlite3.connect(str(path))
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='cost_events'")]
    conn.close()
    store.close()
    assert "idx_cost_events_timestamp" in names


# ---------------------------------------------------------------- the tracker


def test_preflight_prefers_the_aggregate_over_reading_rows(tmp_path, monkeypatch):
    """The tracker asks the store for a total, not for the rows behind it."""
    _budget(tmp_path, monkeypatch, daily=1e9)
    store = _ledger(tmp_path / "c.sqlite", 500)
    calls = {"rows": 0, "sum": 0}
    real_since, real_spend = store.query_since, store.spend_since

    def counting_rows(since):
        calls["rows"] += 1
        return real_since(since)

    def counting_spend(since):
        calls["sum"] += 1
        return real_spend(since)

    monkeypatch.setattr(store, "query_since", counting_rows)
    monkeypatch.setattr(store, "spend_since", counting_spend)
    tracker = CostTracker(storage=store)
    tracker.check_preflight("openai", "gpt-5-nano")
    store.close()
    assert calls["sum"] >= 1
    assert calls["rows"] == 0, "the preflight still materialised rows"


def test_a_store_without_the_aggregate_still_works(tmp_path, monkeypatch):
    """A third-party store that only offers the row queries keeps working.

    The aggregate is an optimization, not a new requirement on the duck type,
    so an implementation that predates it must still answer the budget check.
    This passes on a tree without the aggregate as well: it guards against
    the fallback being dropped, not against the defect.
    """
    _budget(tmp_path, monkeypatch, daily=1e9, monthly=1e9)
    now = time.time()

    class RowsOnlyStore:
        def query_today(self):
            return [CostEvent("openai", "m", 1, 1, 0.25, now)]

        def query_month(self):
            return [CostEvent("openai", "m", 1, 1, 0.25, now),
                    CostEvent("openai", "m", 1, 1, 0.75, now - 10 * DAY)]

        def query_since(self, since):
            return self.query_month()

    tracker = CostTracker(storage=RowsOnlyStore())
    assert tracker._period_spend("daily") == pytest.approx(0.25)
    assert tracker._period_spend("monthly") == pytest.approx(1.0)


def test_repeated_preflights_read_the_ledger_once(tmp_path, monkeypatch):
    """A burst of calls costs one read, not one read per call.

    This is the concurrency defect in miniature: sixteen agents all asked the
    same question of the same file at the same moment.
    """
    _budget(tmp_path, monkeypatch, daily=1e9)
    store = _ledger(tmp_path / "c.sqlite", 200)
    reads = {"n": 0}
    real = store.spend_since

    def counting(since):
        reads["n"] += 1
        return real(since)

    monkeypatch.setattr(store, "spend_since", counting)
    tracker = CostTracker(storage=store)
    for _ in range(50):
        tracker.check_preflight("openai", "gpt-5-nano")
    store.close()
    assert reads["n"] == 1, f"{reads['n']} ledger reads for 50 preflights"


def test_recording_spend_updates_the_cached_reading_without_a_read(tmp_path, monkeypatch):
    """New spend is visible to the next check at once, and costs no read.

    The cache may only ever delay a *reassuring* answer, never a refusal, so a
    priced call is folded into the cached total before the post-spend check
    reads it. Folding rather than dropping is what keeps a paid workload —
    preflight, call, record, preflight, ... — at one ledger read a second
    instead of one per call: with a few hundred thousand events inside the
    window a read costs tens of milliseconds, and dropping the cache paid it
    on every call.
    """
    _budget(tmp_path, monkeypatch, daily=1e9)
    store = _ledger(tmp_path / "c.sqlite", 10, cost=0.0)
    reads = {"n": 0}
    real = store.spend_since

    def counting(since):
        reads["n"] += 1
        return real(since)

    monkeypatch.setattr(store, "spend_since", counting)
    tracker = CostTracker(storage=store)
    assert tracker._period_spend("daily") == pytest.approx(0.0)
    assert reads["n"] == 1

    tracker.record("openai", "gpt-5-nano", 100, 100, cost_usd=5.0)
    # No sleep: if this only passed after the TTL expired, the fold would not
    # be doing the work.
    assert tracker._period_spend("daily") == pytest.approx(5.0)
    assert reads["n"] == 1, f"{reads['n']} ledger reads; recorded spend was not folded in"
    store.close()


def test_a_paid_workload_reads_the_ledger_once_per_reading(tmp_path, monkeypatch):
    """Preflight, record, preflight, record ... costs one read per period.

    The reading is held for longer than the test runs so that what is counted
    is the number of reads the sequence needs, not how fast this machine is;
    the cached total must still equal what the rows say at the end.
    """
    _budget(tmp_path, monkeypatch, daily=1e9)
    monkeypatch.setattr(CostTracker, "_PERIOD_SPEND_TTL_S", 3600.0)
    store = _ledger(tmp_path / "c.sqlite", 200)
    reads = {"n": 0}
    real = store.spend_since

    def counting(since):
        reads["n"] += 1
        return real(since)

    monkeypatch.setattr(store, "spend_since", counting)
    tracker = CostTracker(storage=store)
    for _ in range(50):
        tracker.check_preflight("openai", "gpt-5-nano")
        tracker.record("openai", "gpt-5-nano", 100, 100, cost_usd=0.01)
    from_rows = real(time.time() - DAY)
    assert reads["n"] == 1, f"{reads['n']} ledger reads for 50 priced calls"
    assert tracker._period_spend("daily") == pytest.approx(from_rows, abs=1e-9)
    store.close()


def test_the_budget_still_refuses_a_call_over_its_cap(tmp_path, monkeypatch):
    """The gate the cache sits in front of still closes.

    Passes without the cache too; it guards against the cache ever delaying a
    refusal.
    """
    from effgen.models.errors import BudgetExceededError

    _budget(tmp_path, monkeypatch, daily=1.0)
    store = _ledger(tmp_path / "c.sqlite", 4, span_days=0.5, cost=0.5)  # $2.00 today
    tracker = CostTracker(storage=store)
    with pytest.raises(BudgetExceededError):
        tracker.check_preflight("openai", "gpt-5-nano")
    store.close()


def test_totals_agree_with_the_rows_to_the_cent(tmp_path, monkeypatch):
    """What the budget gate sees is what ``effgen cost`` reports."""
    _budget(tmp_path, monkeypatch, daily=1e9)
    path = tmp_path / "c.sqlite"
    store = _ledger(path, 3000, span_days=0.9, cost=0.000123)  # all inside today
    tracker = CostTracker(storage=store)
    from_rows = sum(e.cost_usd for e in store.query_today())
    assert tracker._period_spend("daily") == pytest.approx(from_rows, abs=1e-9)
    assert round(store.spend_today(), 2) == round(from_rows, 2)
    store.close()


# ---------------------------------------------------------------- retention


def test_the_size_warning_fires_once_not_per_call(tmp_path, caplog):
    """With folding off, crossing the ceiling says so one time and names the command."""
    path = tmp_path / "c.sqlite"
    from effgen.models._cost_store import RETENTION_WARN_ROWS

    store = _ledger(path, 5)
    store._max_rows = 0
    store._rows = RETENTION_WARN_ROWS - 1
    with caplog.at_level(logging.WARNING, logger="effgen.models._cost_store"):
        for _ in range(20):
            store.insert("openai", "gpt-5-nano", 1, 1, 0.0)
    store.close()
    hits = [r for r in caplog.records if "effgen cost prune" in r.getMessage()]
    assert len(hits) == 1, f"retention warning fired {len(hits)} times"


def test_no_warning_below_the_ceiling(tmp_path, caplog):
    """Guards the other direction: a tree that never warns passes this too."""
    store = _ledger(tmp_path / "c.sqlite", 5)
    with caplog.at_level(logging.WARNING, logger="effgen.models._cost_store"):
        for _ in range(10):
            store.insert("openai", "gpt-5-nano", 1, 1, 0.0)
    store.close()
    assert not [r for r in caplog.records if "prune" in r.getMessage()]


def test_prune_by_age_keeps_the_recent_rows(tmp_path):
    path = tmp_path / "c.sqlite"
    from effgen.models._cost_store import RETENTION_MAX_AGE_DAYS

    store = _ledger(path, 600, span_days=180.0)
    before = store.count()
    kept_expected = store.count_since(time.time() - RETENTION_MAX_AGE_DAYS * DAY)
    deleted = store.prune()
    assert store.count() == before - deleted == kept_expected
    assert store.count_since(time.time() - RETENTION_MAX_AGE_DAYS * DAY) == store.count()
    store.close()


def test_prune_by_row_count_keeps_the_newest(tmp_path):
    path = tmp_path / "c.sqlite"
    store = _ledger(path, 500, span_days=60.0)
    newest = max(e.timestamp for e in store.query_all())
    deleted = store.prune(keep_rows=100)
    assert deleted == 400
    assert store.count() == 100
    assert max(e.timestamp for e in store.query_all()) == pytest.approx(newest)
    store.close()


def test_prune_refuses_two_bounds_at_once(tmp_path):
    store = _ledger(tmp_path / "c.sqlite", 10)
    with pytest.raises(ValueError):
        store.prune(max_age_days=1, keep_rows=1)
    store.close()


def test_count_and_count_since_agree_with_the_rows(tmp_path):
    store = _ledger(tmp_path / "c.sqlite", 400, span_days=40.0)
    assert store.count() == len(store.query_all())
    since = time.time() - 10 * DAY
    assert store.count_since(since) == len(store.query_since(since))
    store.close()
