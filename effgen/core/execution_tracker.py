"""
Execution tracking system for transparent agent operation visibility.

Tracks all execution events including:
- Task start/completion
- Routing decisions
- Task decomposition
- Sub-agent spawning and execution
- Tool calls
- Result synthesis
- Errors and retries
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


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
    agent_id: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    parent_event_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time()*1000)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "message": self.message,
            "data": self.data,
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
    active_agents: List[str] = field(default_factory=list)
    completed_subtasks: int = 0
    total_subtasks: int = 0
    pending_subtasks: int = 0
    failed_subtasks: int = 0
    current_operations: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
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
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    parent_id: Optional[str] = None
    children: List["ExecutionNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
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
            "metadata": self.metadata
        }


class ExecutionTracker:
    """
    Track and display execution progress with full transparency.

    Provides real-time visibility into:
    - Parent agent reasoning
    - Sub-agent spawning
    - Subtask assignments
    - Tool executions
    - Intermediate results
    - Errors and retries
    - Final synthesis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize execution tracker.

        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.events: List[ExecutionEvent] = []
        self.nodes: Dict[str, ExecutionNode] = {}
        self.root_node_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.active_agents: set = set()
        self.active_tools: set = set()

    def track_event(self, event: ExecutionEvent):
        """
        Record execution event.

        Args:
            event: Execution event to track
        """
        self.events.append(event)

        # Update internal state based on event type
        self._update_state(event)

        # Update execution tree
        self._update_tree(event)

    def _update_state(self, event: ExecutionEvent):
        """Update internal state based on event."""
        if event.type == EventType.TASK_START:
            if self.start_time is None:
                self.start_time = event.timestamp

        elif event.type in [EventType.TASK_COMPLETE, EventType.TASK_FAILED]:
            self.end_time = event.timestamp

        elif event.type == EventType.SUB_AGENT_START:
            if event.agent_id:
                self.active_agents.add(event.agent_id)

        elif event.type in [EventType.SUB_AGENT_COMPLETE, EventType.SUB_AGENT_FAILED]:
            if event.agent_id:
                self.active_agents.discard(event.agent_id)

        elif event.type == EventType.TOOL_CALL_START:
            tool_name = event.data.get("tool_name", "unknown")
            self.active_tools.add(tool_name)

        elif event.type in [EventType.TOOL_CALL_COMPLETE, EventType.TOOL_CALL_FAILED]:
            tool_name = event.data.get("tool_name", "unknown")
            self.active_tools.discard(tool_name)

    def _update_tree(self, event: ExecutionEvent):
        """Update execution tree based on event."""
        if event.type == EventType.TASK_START:
            # Create root node
            node = ExecutionNode(
                node_id=event.event_id,
                node_type="task",
                name=event.data.get("task", "Main Task"),
                status="running",
                started_at=event.timestamp,
                metadata=event.data
            )
            self.nodes[node.node_id] = node
            if self.root_node_id is None:
                self.root_node_id = node.node_id

        elif event.type == EventType.SUB_AGENT_SPAWN:
            # Create sub-agent node
            agent_id = event.agent_id or event.event_id
            node = ExecutionNode(
                node_id=agent_id,
                node_type="agent",
                name=event.data.get("agent_name", agent_id),
                status="pending",
                parent_id=event.parent_event_id or self.root_node_id,
                metadata=event.data
            )
            self.nodes[node.node_id] = node

            # Add to parent's children
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.append(node)

        elif event.type == EventType.SUB_AGENT_START:
            # Update agent node status
            if event.agent_id and event.agent_id in self.nodes:
                self.nodes[event.agent_id].status = "running"
                self.nodes[event.agent_id].started_at = event.timestamp

        elif event.type == EventType.SUB_AGENT_COMPLETE:
            # Update agent node status
            if event.agent_id and event.agent_id in self.nodes:
                self.nodes[event.agent_id].status = "completed"
                self.nodes[event.agent_id].completed_at = event.timestamp

        elif event.type == EventType.SUB_AGENT_FAILED:
            # Update agent node status
            if event.agent_id and event.agent_id in self.nodes:
                self.nodes[event.agent_id].status = "failed"
                self.nodes[event.agent_id].completed_at = event.timestamp

        elif event.type == EventType.TOOL_CALL_START:
            # Create tool node
            tool_id = event.event_id
            node = ExecutionNode(
                node_id=tool_id,
                node_type="tool",
                name=event.data.get("tool_name", "Tool"),
                status="running",
                started_at=event.timestamp,
                parent_id=event.agent_id or self.root_node_id,
                metadata=event.data
            )
            self.nodes[node.node_id] = node

            # Add to parent's children
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.append(node)

        elif event.type == EventType.TOOL_CALL_COMPLETE:
            # Find and update tool node
            tool_id = event.data.get("tool_call_id", event.event_id)
            if tool_id in self.nodes:
                self.nodes[tool_id].status = "completed"
                self.nodes[tool_id].completed_at = event.timestamp

    def get_live_status(self) -> ExecutionStatus:
        """
        Get current execution status.

        Returns:
            ExecutionStatus with current state
        """
        # Count subtasks by status
        completed = 0
        failed = 0
        total = 0

        for node in self.nodes.values():
            if node.node_type in ["agent", "subtask"]:
                total += 1
                if node.status == "completed":
                    completed += 1
                elif node.status == "failed":
                    failed += 1

        pending = total - completed - failed

        # Calculate progress
        progress = (completed / total * 100) if total > 0 else 0.0

        # Get current operations
        current_ops = []
        for agent_id in self.active_agents:
            if agent_id in self.nodes:
                current_ops.append(self.nodes[agent_id].name)

        # Calculate elapsed time
        elapsed = 0.0
        if self.start_time:
            end = self.end_time or time.time()
            elapsed = end - self.start_time

        # Estimate remaining time
        estimated_remaining = None
        if progress > 0 and progress < 100:
            estimated_remaining = (elapsed / progress) * (100 - progress)

        return ExecutionStatus(
            active_agents=list(self.active_agents),
            completed_subtasks=completed,
            total_subtasks=total,
            pending_subtasks=pending,
            failed_subtasks=failed,
            current_operations=current_ops,
            progress_percentage=progress,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining
        )

    def generate_execution_tree(self) -> Dict[str, Any]:
        """
        Generate visual execution tree.

        Returns:
            Execution tree as nested dictionary
        """
        if self.root_node_id and self.root_node_id in self.nodes:
            return self.nodes[self.root_node_id].to_dict()
        return {}

    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get full execution trace.

        Returns:
            List of all events as dictionaries
        """
        return [event.to_dict() for event in self.events]

    def format_for_display(self, format: str = "rich") -> str:
        """
        Format execution trace for user display.

        Args:
            format: Output format (rich, markdown, plain)

        Returns:
            Formatted string
        """
        if format == "rich":
            return self._format_rich()
        elif format == "markdown":
            return self._format_markdown()
        elif format == "plain":
            return self._format_plain()
        elif format == "json":
            return json.dumps({
                "events": self.get_trace(),
                "tree": self.generate_execution_tree(),
                "status": self.get_live_status().to_dict()
            }, indent=2)
        else:
            return self._format_plain()

    def _format_rich(self) -> str:
        """Format with rich terminal output."""
        lines = []
        lines.append("=" * 60)
        lines.append("EXECUTION TRACE")
        lines.append("=" * 60)

        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            icon = self._get_event_icon(event.type)
            lines.append(f"{timestamp} {icon} [{event.type.value}] {event.message}")
            if event.agent_id:
                lines.append(f"  Agent: {event.agent_id}")

        lines.append("=" * 60)
        status = self.get_live_status()
        lines.append(f"Status: {status.progress_percentage:.1f}% complete")
        lines.append(f"Elapsed: {status.elapsed_time:.2f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_markdown(self) -> str:
        """Format as markdown."""
        lines = []
        lines.append("# Execution Trace\n")

        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            lines.append(f"**{timestamp}** - {event.type.value}")
            lines.append(f"> {event.message}\n")

        status = self.get_live_status()
        lines.append("\n## Summary\n")
        lines.append(f"- Progress: {status.progress_percentage:.1f}%")
        lines.append(f"- Completed: {status.completed_subtasks}/{status.total_subtasks}")
        lines.append(f"- Elapsed: {status.elapsed_time:.2f}s")

        return "\n".join(lines)

    def _format_plain(self) -> str:
        """Format as plain text."""
        lines = []
        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            lines.append(f"[{timestamp}] {event.type.value}: {event.message}")
        return "\n".join(lines)

    def _get_event_icon(self, event_type: EventType) -> str:
        """Get icon for event type."""
        icons = {
            EventType.TASK_START: "🎯",
            EventType.TASK_COMPLETE: "✅",
            EventType.TASK_FAILED: "❌",
            EventType.ROUTING_DECISION: "🧠",
            EventType.TASK_DECOMPOSITION: "🔀",
            EventType.SUB_AGENT_SPAWN: "🚀",
            EventType.SUB_AGENT_START: "🔍",
            EventType.SUB_AGENT_COMPLETE: "✅",
            EventType.SUB_AGENT_FAILED: "❌",
            EventType.TOOL_CALL_START: "🔧",
            EventType.TOOL_CALL_COMPLETE: "✓",
            EventType.TOOL_CALL_FAILED: "✗",
            EventType.RESULT_SYNTHESIS: "✨",
            EventType.REASONING_STEP: "💭",
            EventType.ERROR: "⚠️",
            EventType.WARNING: "⚠️",
            EventType.INFO: "ℹ️"
        }
        return icons.get(event_type, "•")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.

        Returns:
            Summary dictionary with key metrics
        """
        status = self.get_live_status()

        # Count events by type
        event_counts = {}
        for event in self.events:
            event_type = event.type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Get tool usage
        tool_calls = [e for e in self.events if e.type == EventType.TOOL_CALL_START]
        tool_usage = {}
        for call in tool_calls:
            tool_name = call.data.get("tool_name", "unknown")
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "status": status.to_dict(),
            "tool_usage": tool_usage,
            "execution_tree": self.generate_execution_tree(),
            "duration": status.elapsed_time
        }

    def clear(self):
        """Clear all tracking data."""
        self.events = []
        self.nodes = {}
        self.root_node_id = None
        self.start_time = None
        self.end_time = None
        self.active_agents = set()
        self.active_tools = set()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics from execution.

        Returns:
            Performance metrics dictionary
        """
        if not self.events:
            return {"error": "No execution data available"}

        metrics = {
            "timing": self._calculate_timing_metrics(),
            "throughput": self._calculate_throughput_metrics(),
            "resource_usage": self._calculate_resource_usage(),
            "efficiency": self._calculate_efficiency_metrics(),
            "bottlenecks": self._identify_bottlenecks()
        }

        return metrics

    def _calculate_timing_metrics(self) -> Dict[str, Any]:
        """Calculate timing-related metrics."""
        status = self.get_live_status()

        # Agent timing
        agent_times = []
        for node in self.nodes.values():
            if node.node_type == "agent" and node.get_duration():
                agent_times.append(node.get_duration())

        # Tool timing
        tool_times = []
        for node in self.nodes.values():
            if node.node_type == "tool" and node.get_duration():
                tool_times.append(node.get_duration())

        return {
            "total_elapsed": status.elapsed_time,
            "agent_execution": {
                "total": sum(agent_times) if agent_times else 0,
                "average": sum(agent_times) / len(agent_times) if agent_times else 0,
                "min": min(agent_times) if agent_times else 0,
                "max": max(agent_times) if agent_times else 0
            },
            "tool_execution": {
                "total": sum(tool_times) if tool_times else 0,
                "average": sum(tool_times) / len(tool_times) if tool_times else 0,
                "min": min(tool_times) if tool_times else 0,
                "max": max(tool_times) if tool_times else 0
            }
        }

    def _calculate_throughput_metrics(self) -> Dict[str, Any]:
        """Calculate throughput metrics."""
        status = self.get_live_status()

        if status.elapsed_time == 0:
            return {"error": "No elapsed time"}

        return {
            "subtasks_per_second": status.completed_subtasks / status.elapsed_time if status.elapsed_time > 0 else 0,
            "events_per_second": len(self.events) / status.elapsed_time if status.elapsed_time > 0 else 0,
            "tool_calls_per_second": len([e for e in self.events if e.type == EventType.TOOL_CALL_START]) / status.elapsed_time if status.elapsed_time > 0 else 0
        }

    def _calculate_resource_usage(self) -> Dict[str, Any]:
        """Calculate resource usage metrics."""
        # Count different resource types
        tool_calls = sum(1 for e in self.events if e.type == EventType.TOOL_CALL_START)
        agents_spawned = sum(1 for e in self.events if e.type == EventType.SUB_AGENT_SPAWN)

        # Get peak concurrent agents
        peak_agents = 0
        current_agents = 0
        for event in sorted(self.events, key=lambda e: e.timestamp):
            if event.type == EventType.SUB_AGENT_START:
                current_agents += 1
                peak_agents = max(peak_agents, current_agents)
            elif event.type in [EventType.SUB_AGENT_COMPLETE, EventType.SUB_AGENT_FAILED]:
                current_agents = max(0, current_agents - 1)

        return {
            "total_tool_calls": tool_calls,
            "total_agents_spawned": agents_spawned,
            "peak_concurrent_agents": peak_agents,
            "total_events_tracked": len(self.events)
        }

    def _calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        status = self.get_live_status()

        if status.total_subtasks == 0:
            return {"error": "No subtasks"}

        success_rate = status.completed_subtasks / status.total_subtasks if status.total_subtasks > 0 else 0
        failure_rate = status.failed_subtasks / status.total_subtasks if status.total_subtasks > 0 else 0

        # Calculate parallel efficiency (how well we're using parallelism)
        agent_times = [
            node.get_duration() for node in self.nodes.values()
            if node.node_type == "agent" and node.get_duration()
        ]
        total_agent_time = sum(agent_times) if agent_times else 0
        wall_clock_time = status.elapsed_time if status.elapsed_time > 0 else 1

        parallel_efficiency = (total_agent_time / wall_clock_time) / len(agent_times) if agent_times and wall_clock_time > 0 else 0
        parallel_efficiency = min(1.0, parallel_efficiency)  # Cap at 100%

        return {
            "success_rate": round(success_rate, 3),
            "failure_rate": round(failure_rate, 3),
            "parallel_efficiency": round(parallel_efficiency, 3),
            "avg_time_per_subtask": round(status.elapsed_time / status.completed_subtasks, 2) if status.completed_subtasks > 0 else 0
        }

    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Find slowest agents
        agent_durations = [
            (node.name, node.get_duration())
            for node in self.nodes.values()
            if node.node_type == "agent" and node.get_duration()
        ]

        if agent_durations:
            agent_durations.sort(key=lambda x: x[1], reverse=True)
            avg_duration = sum(d for _, d in agent_durations) / len(agent_durations)

            for name, duration in agent_durations[:3]:  # Top 3 slowest
                if duration > avg_duration * 1.5:
                    bottlenecks.append({
                        "type": "slow_agent",
                        "name": name,
                        "duration": round(duration, 2),
                        "severity": "high" if duration > avg_duration * 2 else "medium"
                    })

        # Find failed operations
        failed_events = [e for e in self.events if "failed" in e.type.value]
        if len(failed_events) > 2:
            bottlenecks.append({
                "type": "high_failure_rate",
                "count": len(failed_events),
                "severity": "high"
            })

        # Check for idle time (gaps in execution)
        status = self.get_live_status()
        if status.total_subtasks > 0 and len(status.active_agents) == 0 and status.pending_subtasks > 0:
            bottlenecks.append({
                "type": "idle_with_pending_work",
                "pending": status.pending_subtasks,
                "severity": "medium"
            })

        return bottlenecks

    def export_trace(self, filepath: str, format: str = "json"):
        """
        Export execution trace to file.

        Args:
            filepath: Path to save trace
            format: Export format (json, csv, html)
        """
        import json
        import csv

        if format == "json":
            data = {
                "trace": self.get_trace(),
                "tree": self.generate_execution_tree(),
                "status": self.get_live_status().to_dict(),
                "summary": self.get_summary(),
                "metrics": self.get_performance_metrics()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            with open(filepath, 'w', newline='') as f:
                fieldnames = ["timestamp", "type", "agent_id", "message"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for event in self.events:
                    writer.writerow({
                        "timestamp": event.timestamp,
                        "type": event.type.value,
                        "agent_id": event.agent_id or "",
                        "message": event.message
                    })

        elif format == "html":
            html = self._generate_html_trace()
            with open(filepath, 'w') as f:
                f.write(html)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_html_trace(self) -> str:
        """Generate HTML visualization of trace."""
        from datetime import datetime

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Execution Trace</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".event { padding: 10px; margin: 5px 0; border-left: 3px solid #ddd; }",
            ".task_start { border-color: #4CAF50; }",
            ".task_complete { border-color: #2196F3; }",
            ".task_failed { border-color: #f44336; }",
            ".sub_agent_start { border-color: #FF9800; }",
            ".tool_call { border-color: #9C27B0; }",
            ".timestamp { color: #666; font-size: 0.9em; }",
            ".summary { background: #f5f5f5; padding: 15px; margin: 20px 0; }",
            "</style>",
            "</head><body>",
            "<h1>Execution Trace</h1>"
        ]

        # Add summary
        status = self.get_live_status()
        html_parts.append('<div class="summary">')
        html_parts.append(f"<h2>Summary</h2>")
        html_parts.append(f"<p>Total Events: {len(self.events)}</p>")
        html_parts.append(f"<p>Completed: {status.completed_subtasks}/{status.total_subtasks}</p>")
        html_parts.append(f"<p>Elapsed Time: {status.elapsed_time:.2f}s</p>")
        html_parts.append(f"<p>Progress: {status.progress_percentage:.1f}%</p>")
        html_parts.append('</div>')

        # Add events
        html_parts.append("<h2>Events</h2>")
        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            css_class = event.type.value.replace("_", " ")
            html_parts.append(f'<div class="event {event.type.value}">')
            html_parts.append(f'<span class="timestamp">{timestamp}</span> ')
            html_parts.append(f'<strong>{event.type.value}</strong>: {event.message}')
            if event.agent_id:
                html_parts.append(f' <em>(Agent: {event.agent_id})</em>')
            html_parts.append('</div>')

        html_parts.append("</body></html>")

        return "\n".join(html_parts)

    def get_critical_path(self) -> List[Dict[str, Any]]:
        """
        Identify the critical path in execution (longest dependency chain).

        Returns:
            List of nodes on critical path
        """
        if not self.root_node_id or self.root_node_id not in self.nodes:
            return []

        def calculate_path_length(node_id: str, memo: Dict[str, float]) -> float:
            if node_id in memo:
                return memo[node_id]

            if node_id not in self.nodes:
                return 0

            node = self.nodes[node_id]
            if not node.children:
                duration = node.get_duration() or 0
                memo[node_id] = duration
                return duration

            max_child_path = max(
                calculate_path_length(child.node_id, memo)
                for child in node.children
            )

            total = (node.get_duration() or 0) + max_child_path
            memo[node_id] = total
            return total

        # Find critical path
        memo = {}
        def find_critical_path(node_id: str) -> List[Dict[str, Any]]:
            if node_id not in self.nodes:
                return []

            node = self.nodes[node_id]
            path = [{
                "node_id": node.node_id,
                "name": node.name,
                "duration": node.get_duration(),
                "type": node.node_type
            }]

            if node.children:
                # Find child with longest path
                child_paths = [
                    (child, calculate_path_length(child.node_id, memo))
                    for child in node.children
                ]
                if child_paths:
                    critical_child = max(child_paths, key=lambda x: x[1])[0]
                    path.extend(find_critical_path(critical_child.node_id))

            return path

        return find_critical_path(self.root_node_id)

    def __repr__(self) -> str:
        """String representation."""
        return f"ExecutionTracker(events={len(self.events)}, nodes={len(self.nodes)})"
