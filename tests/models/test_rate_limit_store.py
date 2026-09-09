"""
Unit tests for SQLiteRateLimitStore and the SQLite-backed RateLimitCoordinator.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models._rate_limit_store import SQLiteRateLimitStore
from effgen.models.groq_adapter import GroqAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """In-memory SQLite store (no file I/O, safe for unit tests)."""
    s = SQLiteRateLimitStore(":memory:")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Schema / basic operations
# ---------------------------------------------------------------------------

def test_schema_creation_idempotent(store):
    """Creating the store twice against the same DB doesn't fail."""
    s2 = SQLiteRateLimitStore(":memory:")
    s2.close()


def test_insert_and_query(store):
    now = time.time()
    store.insert_event("groq", "llama-3.3-70b", "request", now, 0)
    rows = store.query_window("groq", "llama-3.3-70b", "request", now - 1)
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(now, abs=0.001)
    assert rows[0][1] == 0


def test_insert_token_event(store):
    now = time.time()
    store.insert_event("cerebras", "llama3.1-8b", "tokens", now, 500)
    rows = store.query_window("cerebras", "llama3.1-8b", "tokens", now - 1)
    assert len(rows) == 1
    assert rows[0][1] == 500


def test_query_respects_window_cutoff(store):
    now = time.time()
    store.insert_event("groq", "m", "request", now - 120, 0)  # old
    store.insert_event("groq", "m", "request", now - 30, 0)   # within window
    rows = store.query_window("groq", "m", "request", now - 60)
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(now - 30, abs=0.001)


def test_query_different_providers_isolated(store):
    now = time.time()
    store.insert_event("groq", "m", "request", now, 0)
    store.insert_event("openai", "m", "request", now, 0)
    groq_rows = store.query_window("groq", "m", "request", now - 1)
    openai_rows = store.query_window("openai", "m", "request", now - 1)
    assert len(groq_rows) == 1
    assert len(openai_rows) == 1


def test_cleanup_removes_old_events(store):
    now = time.time()
    store.insert_event("g", "m", "request", now - 100, 0)
    store.insert_event("g", "m", "request", now - 10, 0)
    deleted = store.cleanup(max_age_seconds=50)
    assert deleted == 1
    rows = store.query_window("g", "m", "request", now - 200)
    assert len(rows) == 1  # only the recent one remains


def test_cleanup_no_old_events(store):
    now = time.time()
    store.insert_event("g", "m", "request", now - 5, 0)
    deleted = store.cleanup(max_age_seconds=3600)
    assert deleted == 0


# ---------------------------------------------------------------------------
# RateLimitCoordinator with SQLite storage
# ---------------------------------------------------------------------------

def _make_coordinator(store, rpm: int = 5) -> RateLimitCoordinator:
    return RateLimitCoordinator(
        provider="test",
        model="model",
        rpm=rpm, rph=100, rpd=1000,
        tpm=100_000, tph=1_000_000, tpd=10_000_000,
        storage=store,
    )


def test_coordinator_in_memory_backcompat():
    """No storage arg → in-memory mode, no SQLite involved."""
    coord = RateLimitCoordinator(
        provider="test", model="m",
        rpm=5, rph=100, rpd=1000,
        tpm=100_000, tph=1_000_000, tpd=10_000_000,
    )
    asyncio.run(coord.acquire())
    coord.record(actual_tokens=100)
    assert coord.total_requests == 1
    assert coord.status()["storage"] == "memory"


def test_coordinator_sqlite_records_events(store):
    coord = _make_coordinator(store, rpm=10)
    asyncio.run(coord.acquire())
    rows = store.query_window("test", "model", "request", time.time() - 5)
    assert len(rows) == 1

    coord.record(actual_tokens=200)
    assert coord.total_requests == 1
    rows = store.query_window("test", "model", "request", time.time() - 5)
    assert len(rows) == 1


def test_coordinator_sqlite_updates_reserved_token_event(store):
    coord = _make_coordinator(store, rpm=10)
    asyncio.run(coord.acquire(tokens_estimate=50))

    token_rows = store.query_window("test", "model", "tokens", time.time() - 5)
    assert len(token_rows) == 1
    assert token_rows[0][1] == 50

    coord.record(actual_tokens=200)
    token_rows = store.query_window("test", "model", "tokens", time.time() - 5)
    assert len(token_rows) == 1
    assert token_rows[0][1] == 200


def test_coordinator_sqlite_throttles_at_limit(store):
    """When rpm=2 and 2 events are in-window, acquire throttles before reserving."""
    coord = _make_coordinator(store, rpm=2)
    base = time.time()
    # Pre-fill the store with 2 events within the last 60 s
    store.insert_event("test", "model", "request", base - 10, 0)
    store.insert_event("test", "model", "request", base - 5, 0)

    current = [base]

    def fake_time():
        return current[0]

    async def fake_sleep(seconds):
        current[0] += seconds + 0.001

    with patch("effgen.models._rate_limit.time.time", side_effect=fake_time), \
         patch("effgen.models._rate_limit.asyncio.sleep", side_effect=fake_sleep):
        asyncio.run(coord.acquire())

    assert coord.total_throttled >= 1
    rows = store.query_window("test", "model", "request", base - 60)
    assert len(rows) == 3


def test_coordinator_sqlite_atomic_acquire_before_record(store):
    """A second coordinator sees the first coordinator's acquire immediately."""
    c1 = _make_coordinator(store, rpm=1)
    c2 = _make_coordinator(store, rpm=1)
    base = time.time()
    current = [base]

    def fake_time():
        return current[0]

    async def fake_sleep(seconds):
        current[0] += seconds + 0.001

    with patch("effgen.models._rate_limit.time.time", side_effect=fake_time), \
         patch("effgen.models._rate_limit.asyncio.sleep", side_effect=fake_sleep):
        asyncio.run(c1.acquire())
        asyncio.run(c2.acquire())

    assert c2.total_throttled >= 1
    rows = store.query_window("test", "model", "request", base - 60)
    assert len(rows) == 2


def test_coordinator_status_includes_sqlite_counts(store):
    coord = _make_coordinator(store)
    now = time.time()
    store.insert_event("test", "model", "request", now - 10, 0)
    store.insert_event("test", "model", "request", now - 5, 0)
    s = coord.status()
    assert s["storage"] == "sqlite"
    assert s["sqlite_req_minute"] == 2


def test_coordinator_cleanup_storage(store):
    coord = _make_coordinator(store)
    now = time.time()
    store.insert_event("test", "model", "request", now - 200000, 0)
    deleted = coord.cleanup_storage()
    assert deleted == 1


def test_coordinator_cleanup_noop_for_memory():
    coord = RateLimitCoordinator(
        provider="test", model="m",
        rpm=5, rph=100, rpd=1000,
        tpm=100_000, tph=1_000_000, tpd=10_000_000,
    )
    assert coord.cleanup_storage() == 0


def test_groq_adapter_accepts_sqlite_rate_limit_storage(store):
    adapter = GroqAdapter(
        "openai/gpt-oss-20b",
        api_key="fake-key",
        rate_limit_storage=store,
    )
    assert adapter._rate_limiter is not None
    assert adapter._rate_limiter.status()["storage"] == "sqlite"
