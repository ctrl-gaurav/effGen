"""The data model behind execution tracking.

Four types and the id minting they share:

- ``EventType`` — the kinds of event a run emits;
- ``ExecutionEvent`` — one recorded event, with its parent link;
- ``ExecutionStatus`` — a snapshot of progress at a point in time;
- ``ExecutionNode`` — one node of the execution tree.

Each carries a ``to_dict`` whose key order is what consumers read, so the
serialised shapes are part of the contract.

Import these from :mod:`effgen.core.execution_tracker`; this module is the
implementation.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


def as_data(value: Any) -> Any:
    """*value* as JSON-serialisable data, however it was recorded.

    An event's ``data`` is whatever the code that emitted it had to hand, so a
    node's metadata can end up holding a record object — the list of tool calls
    a finished run made, for instance. A serialised tree has to be data, or a
    caller writing the run out with :func:`json.dumps` gets ``TypeError`` for
    the whole document. Anything carrying its own ``to_dict`` is asked for it,
    a dataclass is read field by field, a set becomes a list, an enum becomes
    its value, and anything else is rendered as its string form rather than
    dropped.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): as_data(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [as_data(item) for item in value]
    if isinstance(value, set | frozenset):
        return [as_data(item) for item in value]
    if isinstance(value, Enum):
        return as_data(value.value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return as_data(to_dict())
        except Exception:  # noqa: BLE001 - a record that cannot describe itself
            return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return as_data(asdict(value))
        except Exception:  # noqa: BLE001 - a field that cannot be copied
            return str(value)
    return str(value)


# Monotonic sequence so every event id is unique even when several events are
# created within the same millisecond. A bare ``time.time()*1000`` id collides
# under a tight tool loop, which would make parent/child links ambiguous.
_EVENT_SEQ = itertools.count(1)
_EVENT_SEQ_LOCK = threading.Lock()


def _next_event_id() -> str:
    """Return a process-unique event id (``evt_<ms>_<seq>``)."""
    with _EVENT_SEQ_LOCK:
        seq = next(_EVENT_SEQ)
    return f"evt_{int(time.time() * 1000)}_{seq}"


class EventType(Enum):
    """Types of execution events."""
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    ROUTING_DECISION = "routing_decision"
    TASK_DECOMPOSITION = "task_decomposition"
    SUB_AGENT_SPAWN = "sub_agent_spawn"
    SUB_AGENT_START = "sub_agent_start"
    SUB_AGENT_PROGRESS = "sub_agent_progress"
    SUB_AGENT_COMPLETE = "sub_agent_complete"
    SUB_AGENT_FAILED = "sub_agent_failed"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TOOL_CALL_FAILED = "tool_call_failed"
    RESULT_SYNTHESIS = "result_synthesis"
    REASONING_STEP = "reasoning_step"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ExecutionEvent:
    """
    Represents a single execution event.

    Attributes:
        type: Event type
        timestamp: When event occurred
        agent_id: ID of agent that generated event
        message: Human-readable message
        data: Additional event data
        parent_event_id: ID of parent event (for hierarchy)
        event_id: Unique event identifier
    """
    type: EventType
    timestamp: float = field(default_factory=time.time)
    agent_id: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    parent_event_id: str | None = None
    event_id: str = field(default_factory=_next_event_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "message": self.message,
            "data": as_data(self.data),
            "parent_event_id": self.parent_event_id,
            "event_id": self.event_id
        }


@dataclass
class ExecutionStatus:
    """
    Current execution status snapshot.

    Attributes:
        active_agents: Currently running agent IDs
        completed_subtasks: Number of completed subtasks
        total_subtasks: Total number of subtasks
        pending_subtasks: Number of pending subtasks
        failed_subtasks: Number of failed subtasks
        current_operations: Description of current operations
        progress_percentage: Overall progress (0-100)
        elapsed_time: Time elapsed since start
        estimated_remaining: Estimated time remaining
    """
    active_agents: list[str] = field(default_factory=list)
    completed_subtasks: int = 0
    total_subtasks: int = 0
    pending_subtasks: int = 0
    failed_subtasks: int = 0
    current_operations: list[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "active_agents": self.active_agents,
            "completed_subtasks": self.completed_subtasks,
            "total_subtasks": self.total_subtasks,
            "pending_subtasks": self.pending_subtasks,
            "failed_subtasks": self.failed_subtasks,
            "current_operations": self.current_operations,
            "progress_percentage": round(self.progress_percentage, 1),
            "elapsed_time": round(self.elapsed_time, 2),
            "estimated_remaining": round(self.estimated_remaining, 2) if self.estimated_remaining else None
        }


@dataclass
class ExecutionNode:
    """
    Node in execution tree.

    Represents a single agent or subtask execution in the hierarchy.
    """
    node_id: str
    node_type: str  # "agent", "subtask", "tool"
    name: str
    status: str  # "pending", "running", "completed", "failed"
    started_at: float | None = None
    completed_at: float | None = None
    parent_id: str | None = None
    children: list["ExecutionNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_duration(self) -> float | None:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.get_duration(),
            "parent_id": self.parent_id,
            "children": [child.to_dict() for child in self.children],
            "metadata": as_data(self.metadata)
        }
