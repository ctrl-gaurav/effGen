"""LatencyTracker — rolling-window TTFT and total-latency recorder.

Tracks time-to-first-token (TTFT) and total latency per (provider, model)
with an in-memory rolling window.  Exposes p50 for use by LatencyBasedPolicy.
Persistence is deferred to a later release.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW = 50  # rolling-window size per (provider, model)


class LatencyTracker:
    """Thread-safe rolling-window latency tracker.

    One global instance is shared across adapters (accessed via
    :meth:`get`).  Adapters that want an isolated tracker can
    instantiate directly.

    Args:
        window: Maximum number of recent records kept per (provider, model).
    """

    _instance: LatencyTracker | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
        self._window = window
        self._totals: dict[tuple[str, str], deque[float]] = {}
        self._ttfts: dict[tuple[str, str], deque[float]] = {}
        self._rw_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "LatencyTracker":
        """Return the process-global LatencyTracker instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the global instance (useful in tests)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        total_ms: float,
        ttft_ms: float | None = None,
    ) -> None:
        """Record a latency observation.

        Args:
            provider:  Provider name (e.g. ``"cerebras"``).
            model:     Model ID (e.g. ``"llama3.1-8b"``).
            total_ms:  Total call duration in milliseconds.
            ttft_ms:   Time-to-first-token in milliseconds for streaming
                       calls.  Pass ``None`` for non-streaming calls.
        """
        key = (provider, model)
        with self._rw_lock:
            if key not in self._totals:
                self._totals[key] = deque(maxlen=self._window)
            self._totals[key].append(total_ms)
            if ttft_ms is not None:
                if key not in self._ttfts:
                    self._ttfts[key] = deque(maxlen=self._window)
                self._ttfts[key].append(ttft_ms)
        logger.debug(
            "LatencyTracker.record: %s/%s total=%.1fms ttft=%s",
            provider, model, total_ms,
            f"{ttft_ms:.1f}ms" if ttft_ms is not None else "n/a",
        )

    def record_ttft(self, provider: str, model: str, ttft_ms: float) -> None:
        """Record only a TTFT observation without affecting total p50 latency.

        Args:
            provider: The provider that served the stream.
            model: The model id the stream used.
            ttft_ms: Milliseconds from request to first token.
        """
        key = (provider, model)
        with self._rw_lock:
            if key not in self._ttfts:
                self._ttfts[key] = deque(maxlen=self._window)
            self._ttfts[key].append(ttft_ms)
        logger.debug(
            "LatencyTracker.record_ttft: %s/%s ttft=%.1fms",
            provider, model, ttft_ms,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def p50(self, provider: str, model: str) -> float | None:
        """Return p50 total latency in ms, or ``None`` if no data yet."""
        key = (provider, model)
        with self._rw_lock:
            totals = list(self._totals.get(key, []))
        if not totals:
            return None
        return statistics.median(totals)

    def p50_ttft(self, provider: str, model: str) -> float | None:
        """Return p50 TTFT in ms for streaming observations, or ``None``."""
        key = (provider, model)
        with self._rw_lock:
            ttfts = list(self._ttfts.get(key, []))
        if not ttfts:
            return None
        return statistics.median(ttfts)

    def all_stats(self) -> dict[tuple[str, str], dict[str, float | None]]:
        """Return a dict of all tracked (provider, model) → stats."""
        with self._rw_lock:
            keys = set(self._totals) | set(self._ttfts)
        result = {}
        for key in keys:
            result[key] = {
                "p50_total_ms": self.p50(key[0], key[1]),
                "p50_ttft_ms": self.p50_ttft(key[0], key[1]),
                "n": self._count(key),
                "n_ttft": self._count_ttft(key),
            }
        return result

    def _count(self, key: tuple[str, str]) -> int:
        with self._rw_lock:
            return len(self._totals.get(key, []))

    def _count_ttft(self, key: tuple[str, str]) -> int:
        with self._rw_lock:
            return len(self._ttfts.get(key, []))

    def has_data(self, provider: str, model: str) -> bool:
        """Return True if at least one observation exists for this pair."""
        return self.p50(provider, model) is not None

    def is_empty(self) -> bool:
        """Return True when no total-latency observations have been recorded."""
        with self._rw_lock:
            return not any(self._totals.values())


# ---------------------------------------------------------------------------
# Context-manager helpers for timing calls
# ---------------------------------------------------------------------------

class _TimedCall:
    """Context manager that records total latency on exit."""

    def __init__(
        self,
        tracker: LatencyTracker,
        provider: str,
        model: str,
    ) -> None:
        self._tracker = tracker
        self._provider = provider
        self._model = model
        self._t0: float = 0.0
        self._ttft_ms: float | None = None
        self._closed = False

    def mark_first_token(self) -> None:
        """Call this when the first token arrives (streaming only)."""
        self._ttft_ms = (time.perf_counter() - self._t0) * 1000.0

    def __enter__(self) -> "_TimedCall":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        if self._closed:
            return
        self._closed = True
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        self._tracker.record(self._provider, self._model, total_ms, self._ttft_ms)


def timed_call(
    provider: str,
    model: str,
    tracker: LatencyTracker | None = None,
) -> _TimedCall:
    """Return a context manager that records latency for (provider, model).

    Usage::

        with timed_call("cerebras", "llama3.1-8b") as t:
            result = adapter._do_generate(...)
        # latency recorded automatically

    For streaming::

        with timed_call("groq", "openai/gpt-oss-20b") as t:
            for chunk in stream:
                if first_chunk:
                    t.mark_first_token()
                yield chunk

    Args:
        provider: The provider the call goes to.
        model: The model id the call uses.
        tracker: The tracker to record into, defaulting to the process-wide one.
    """
    if tracker is None:
        tracker = LatencyTracker.get()
    return _TimedCall(tracker, provider, model)
