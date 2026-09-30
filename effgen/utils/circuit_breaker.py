"""
Per-tool circuit breaker for tool execution.

Tracks tool failure rates and temporarily disables tools that fail too
frequently, preventing repeated wasteful calls.

This is a thin, per-tool *manager* over the single canonical circuit-breaker
state machine in :mod:`effgen.reliability.circuit`. There is exactly one
``CircuitBreaker`` finite-state-machine implementation in effGen
(``reliability.circuit.CircuitBreaker``); this module keeps the ergonomic
"one breaker tracks many tools by name" surface that the agent and the
top-level ``effgen.CircuitBreaker`` export use, delegating each tool's circuit
to that shared implementation. :class:`CircuitState` is re-exported from there
so callers compare against the same enum.
"""

from __future__ import annotations

import json
import logging
import time

from effgen.reliability.circuit import CircuitBreaker as _CoreCircuitBreaker
from effgen.reliability.circuit import CircuitState

__all__ = ["CircuitBreaker", "CircuitState"]

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Track tool failure rates and temporarily disable tools that fail too often.

    After ``failure_threshold`` consecutive failures, the tool is marked OPEN
    (skipped). After ``cooldown_seconds``, it transitions to HALF_OPEN and
    allows one test call. A success resets the circuit to CLOSED.

    Each tool name gets its own underlying
    :class:`effgen.reliability.circuit.CircuitBreaker` instance, so circuits are
    independent per tool.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        cooldown_seconds: Seconds before an open circuit allows a retry.
        persist_path: Optional JSON file to persist/restore circuit state.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        persist_path: str | None = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._circuits: dict[str, _CoreCircuitBreaker] = {}
        self._persist_path = persist_path
        if persist_path:
            self._load_state()

    def _get_circuit(self, tool_name: str) -> _CoreCircuitBreaker:
        circuit = self._circuits.get(tool_name)
        if circuit is None:
            # setdefault keeps whichever circuit was stored first, so two runs
            # of one agent failing the same tool at once count both failures
            # on the one circuit rather than one of them on a discarded copy.
            circuit = self._circuits.setdefault(tool_name, _CoreCircuitBreaker(
                name=tool_name,
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.cooldown_seconds,
                half_open_probes=1,
            ))
        return circuit

    def is_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available (circuit not open).

        Returns True for CLOSED or HALF_OPEN (test call allowed). An OPEN
        circuit whose cooldown has elapsed advances to HALF_OPEN and is then
        available for a single probe call.
        """
        return self._get_circuit(tool_name).is_call_permitted()

    def record_success(self, tool_name: str) -> None:
        """Record a successful tool execution, resetting the circuit."""
        self._get_circuit(tool_name).on_success()
        self._save_state()

    def record_failure(self, tool_name: str) -> None:
        """Record a tool failure. May open the circuit."""
        self._get_circuit(tool_name).on_failure()
        self._save_state()

    def get_state(self, tool_name: str) -> CircuitState:
        """Get the current circuit state for a tool.

        This is a non-advancing read: it returns the raw stored state and does
        not itself trigger an OPEN→HALF_OPEN transition (use
        :meth:`is_available` for that).
        """
        return self._get_circuit(tool_name)._state

    def reset(self, tool_name: str) -> None:
        """Manually reset a tool's circuit to CLOSED."""
        if tool_name in self._circuits:
            self._circuits[tool_name].reset()
            logger.debug(f"Circuit for '{tool_name}' manually reset")

    def reset_all(self) -> None:
        """Reset all circuits."""
        self._circuits.clear()

    # ------------------------------------------------------------------
    # Optional persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Save circuit breaker state to a JSON file (if configured)."""
        if not self._persist_path:
            return
        data = {}
        for name, circuit in list(self._circuits.items()):
            data[name] = {
                "consecutive_failures": circuit._consecutive_failures,
                "last_failure_time": circuit._opened_at,
                "state": circuit._state.value,
            }
        try:
            with open(self._persist_path, "w") as f:
                json.dump(data, f)
        except Exception:  # best-effort state persistence; never break the call path
            pass

    def _load_state(self) -> None:
        """Load circuit breaker state from a JSON file (if present)."""
        if not self._persist_path:
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            for name, state in data.items():
                circuit = self._get_circuit(name)
                circuit._consecutive_failures = int(state["consecutive_failures"])
                circuit._state = CircuitState(state["state"])
                # The underlying breaker measures recovery from a monotonic-clock
                # timestamp, which is meaningless across process restarts. For a
                # restored OPEN circuit, restart the recovery window from "now"
                # so it cools down after ``cooldown_seconds`` instead of getting
                # stuck open (a stale monotonic value would never elapse).
                if circuit._state == CircuitState.OPEN:
                    circuit._opened_at = time.monotonic()
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):  # missing/corrupt state file; start fresh
            pass
