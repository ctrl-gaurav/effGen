"""Unit tests for LatencyTracker — rolling window, p50, TTFT recording."""

from __future__ import annotations

import time

import pytest

from effgen.models.latency_tracker import LatencyTracker, timed_call


@pytest.fixture(autouse=True)
def fresh_tracker():
    """Each test gets a fresh global tracker instance."""
    LatencyTracker.reset()
    yield
    LatencyTracker.reset()


# ---------------------------------------------------------------------------
# Basic recording and p50
# ---------------------------------------------------------------------------

def test_p50_none_when_empty():
    t = LatencyTracker()
    assert t.p50("cerebras", "llama3.1-8b") is None


def test_p50_single_record():
    t = LatencyTracker()
    t.record("cerebras", "llama3.1-8b", 350.0)
    assert t.p50("cerebras", "llama3.1-8b") == 350.0


def test_p50_multiple_records():
    t = LatencyTracker()
    for ms in [100.0, 200.0, 300.0, 400.0, 500.0]:
        t.record("groq", "openai/gpt-oss-20b", ms)
    # median of [100,200,300,400,500] = 300
    assert t.p50("groq", "openai/gpt-oss-20b") == 300.0


def test_p50_separate_providers_independent():
    t = LatencyTracker()
    t.record("cerebras", "m1", 100.0)
    t.record("groq", "m2", 999.0)
    assert t.p50("cerebras", "m1") == 100.0
    assert t.p50("groq", "m2") == 999.0


def test_rolling_window_evicts_oldest():
    t = LatencyTracker(window=3)
    t.record("x", "m", 1000.0)
    t.record("x", "m", 1000.0)
    t.record("x", "m", 1000.0)
    # Now add a new value — oldest 1000 is evicted
    t.record("x", "m", 10.0)
    # window = [1000, 1000, 10] → median = 1000
    p = t.p50("x", "m")
    assert p is not None
    # After 3 more fast records, oldest 1000s are gone
    t.record("x", "m", 10.0)
    t.record("x", "m", 10.0)
    p2 = t.p50("x", "m")
    assert p2 is not None and p2 < 1000.0


# ---------------------------------------------------------------------------
# TTFT recording
# ---------------------------------------------------------------------------

def test_record_ttft_separate_from_total():
    t = LatencyTracker()
    t.record("cerebras", "m", total_ms=500.0, ttft_ms=120.0)
    assert t.p50("cerebras", "m") == 500.0
    assert t.p50_ttft("cerebras", "m") == 120.0


def test_record_ttft_none_by_default():
    t = LatencyTracker()
    t.record("cerebras", "m", total_ms=500.0)
    assert t.p50_ttft("cerebras", "m") is None


def test_record_ttft_helper():
    t = LatencyTracker()
    t.record_ttft("groq", "m", 95.0)
    assert t.p50("groq", "m") is None
    assert t.p50_ttft("groq", "m") == 95.0


def test_has_data():
    t = LatencyTracker()
    assert not t.has_data("cerebras", "m")
    t.record("cerebras", "m", 100.0)
    assert t.has_data("cerebras", "m")


def test_is_empty_tracks_total_latency_only():
    t = LatencyTracker()
    assert t.is_empty()
    t.record_ttft("groq", "m", 95.0)
    assert t.is_empty()
    t.record("groq", "m", 150.0)
    assert not t.is_empty()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

def test_singleton():
    a = LatencyTracker.get()
    b = LatencyTracker.get()
    assert a is b


def test_reset_clears_singleton():
    a = LatencyTracker.get()
    a.record("x", "m", 1.0)
    LatencyTracker.reset()
    b = LatencyTracker.get()
    assert b is not a
    assert b.p50("x", "m") is None


# ---------------------------------------------------------------------------
# timed_call context manager
# ---------------------------------------------------------------------------

def test_timed_call_records_total():
    t = LatencyTracker()
    with timed_call("cerebras", "llama3.1-8b", tracker=t):
        time.sleep(0.05)  # 50ms
    p = t.p50("cerebras", "llama3.1-8b")
    assert p is not None and 30 < p < 500  # generous window for CI


def test_timed_call_marks_first_token():
    t = LatencyTracker()
    with timed_call("groq", "openai/gpt-oss-20b", tracker=t) as tc:
        time.sleep(0.02)
        tc.mark_first_token()
        time.sleep(0.05)
    assert t.p50_ttft("groq", "openai/gpt-oss-20b") is not None


# ---------------------------------------------------------------------------
# all_stats
# ---------------------------------------------------------------------------

def test_all_stats():
    t = LatencyTracker()
    t.record("a", "m1", 100.0, 50.0)
    t.record("b", "m2", 200.0)
    stats = t.all_stats()
    assert ("a", "m1") in stats
    assert ("b", "m2") in stats
    assert stats[("a", "m1")]["p50_total_ms"] == 100.0
    assert stats[("a", "m1")]["p50_ttft_ms"] == 50.0
    assert stats[("b", "m2")]["p50_ttft_ms"] is None
