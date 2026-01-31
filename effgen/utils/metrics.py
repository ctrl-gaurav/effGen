"""
Performance metrics and monitoring utilities for effGen framework.

This module provides comprehensive performance tracking including execution time,
token usage, cost calculation, throughput metrics, resource monitoring, and metrics
export capabilities.

Features:
    - Execution time tracking with high precision
    - Token usage tracking across multiple models
    - Cost calculation based on model pricing
    - Throughput metrics (tokens/sec, requests/sec)
    - System resource monitoring (CPU, Memory, GPU)
    - Metrics aggregation and statistics
    - Export to JSON, CSV, and other formats
    - Real-time metrics visualization support
    - Integration with monitoring platforms (W&B, TensorBoard)

Example:
    Basic usage:
        >>> from effgen.utils.metrics import MetricsCollector
        >>> metrics = MetricsCollector()
        >>> with metrics.track_execution("agent_run"):
        ...     agent.run(task)
        >>> print(metrics.get_summary())

    Token tracking:
        >>> metrics.record_tokens("gpt-4", prompt_tokens=100, completion_tokens=50)
        >>> print(f"Total cost: ${metrics.get_total_cost():.4f}")

    Export metrics:
        >>> metrics.export_to_json("metrics.json")
        >>> metrics.export_to_csv("metrics.csv")
"""

import time
import json
import csv
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import statistics

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


# Model pricing information (cost per 1K tokens)
# Updated as of 2024 - these are approximate values
MODEL_PRICING = {
    # OpenAI Models
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0},

    # Anthropic Models
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    "claude-2.1": {"prompt": 0.008, "completion": 0.024},
    "claude-2": {"prompt": 0.008, "completion": 0.024},
    "claude-instant": {"prompt": 0.0008, "completion": 0.0024},

    # Google Models
    "gemini-pro": {"prompt": 0.00025, "completion": 0.0005},
    "gemini-pro-vision": {"prompt": 0.00025, "completion": 0.0005},
    "palm-2": {"prompt": 0.0005, "completion": 0.001},

    # Default for unknown models (local/SLM models typically free)
    "default": {"prompt": 0.0, "completion": 0.0},
}


@dataclass
class ExecutionMetric:
    """
    Represents a single execution metric.

    Attributes:
        name: Name of the operation
        start_time: When the operation started
        end_time: When the operation ended
        duration: Duration in seconds
        success: Whether the operation succeeded
        metadata: Additional metadata
    """
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, success: bool = True, **metadata) -> None:
        """
        Mark the execution as finished.

        Args:
            success: Whether the execution succeeded
            **metadata: Additional metadata to record
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.metadata.update(metadata)


@dataclass
class TokenMetric:
    """
    Represents token usage for a model call.

    Attributes:
        model: Name of the model
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total number of tokens
        cost: Cost in USD
        timestamp: When the call was made
        metadata: Additional metadata
    """
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceSnapshot:
    """
    Represents a snapshot of system resource usage.

    Attributes:
        timestamp: When the snapshot was taken
        cpu_percent: CPU usage percentage
        memory_percent: Memory usage percentage
        memory_used_mb: Memory used in MB
        disk_usage_percent: Disk usage percentage
        gpu_metrics: GPU metrics if available
    """
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    gpu_metrics: Optional[List[Dict[str, Any]]] = None


class MetricsCollector:
    """
    Comprehensive metrics collector for tracking performance and resource usage.

    This class provides a centralized way to collect, aggregate, and export
    various metrics related to agent execution, model usage, and system resources.
    """

    def __init__(
        self,
        enable_resource_monitoring: bool = True,
        resource_sample_interval: float = 1.0,
        max_history_size: int = 10000,
    ):
        """
        Initialize the metrics collector.

        Args:
            enable_resource_monitoring: Whether to monitor system resources
            resource_sample_interval: Interval for sampling resources (seconds)
            max_history_size: Maximum number of metrics to keep in history
        """
        self.execution_metrics: List[ExecutionMetric] = []
        self.token_metrics: List[TokenMetric] = []
        self.resource_snapshots: deque = deque(maxlen=max_history_size)
        self.custom_metrics: Dict[str, List[float]] = defaultdict(list)

        self.enable_resource_monitoring = enable_resource_monitoring
        self.resource_sample_interval = resource_sample_interval
        self.max_history_size = max_history_size

        self._active_executions: Dict[str, ExecutionMetric] = {}
        self._lock = threading.Lock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        if enable_resource_monitoring:
            self.start_resource_monitoring()

    def start_resource_monitoring(self) -> None:
        """Start background thread for resource monitoring."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._monitoring_thread.start()

    def stop_resource_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)

    def _monitor_resources(self) -> None:
        """Background task for monitoring system resources."""
        while not self._stop_monitoring.is_set():
            try:
                snapshot = self._capture_resource_snapshot()
                self.resource_snapshots.append(snapshot)
            except Exception as e:
                # Silently fail to avoid disrupting the main application
                pass

            self._stop_monitoring.wait(self.resource_sample_interval)

    def _capture_resource_snapshot(self) -> ResourceSnapshot:
        """
        Capture current system resource usage.

        Returns:
            ResourceSnapshot with current metrics
        """
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # GPU metrics (if available)
        gpu_metrics = None
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_metrics = [
                    {
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature,
                    }
                    for gpu in gpus
                ]
            except Exception:
                pass

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            gpu_metrics=gpu_metrics,
        )

    @contextmanager
    def track_execution(self, name: str, **metadata):
        """
        Context manager for tracking execution time.

        Args:
            name: Name of the operation
            **metadata: Additional metadata to record

        Example:
            >>> with metrics.track_execution("agent_run", agent_id="001"):
            ...     agent.run(task)
        """
        metric = ExecutionMetric(
            name=name,
            start_time=time.time(),
            metadata=metadata
        )

        execution_id = f"{name}_{id(metric)}"
        with self._lock:
            self._active_executions[execution_id] = metric

        try:
            yield metric
            metric.finish(success=True)
        except Exception as e:
            metric.finish(success=False, error=str(e), error_type=type(e).__name__)
            raise
        finally:
            with self._lock:
                self._active_executions.pop(execution_id, None)
                self.execution_metrics.append(metric)

                # Limit history size
                if len(self.execution_metrics) > self.max_history_size:
                    self.execution_metrics = self.execution_metrics[-self.max_history_size:]

    def record_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        **metadata
    ) -> TokenMetric:
        """
        Record token usage for a model call.

        Args:
            model: Name of the model
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            **metadata: Additional metadata

        Returns:
            TokenMetric instance with recorded data
        """
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        metric = TokenMetric(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            metadata=metadata
        )

        with self._lock:
            self.token_metrics.append(metric)

            # Limit history size
            if len(self.token_metrics) > self.max_history_size:
                self.token_metrics = self.token_metrics[-self.max_history_size:]

        return metric

    def record_custom_metric(self, name: str, value: float) -> None:
        """
        Record a custom metric value.

        Args:
            name: Name of the metric
            value: Value to record
        """
        with self._lock:
            self.custom_metrics[name].append(value)

            # Limit history size
            if len(self.custom_metrics[name]) > self.max_history_size:
                self.custom_metrics[name] = self.custom_metrics[name][-self.max_history_size:]

    @staticmethod
    def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for a model call.

        Args:
            model: Name of the model
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
            Cost in USD
        """
        # Normalize model name
        model_lower = model.lower()

        # Find matching pricing
        pricing = None
        for key, value in MODEL_PRICING.items():
            if key.lower() in model_lower:
                pricing = value
                break

        if pricing is None:
            pricing = MODEL_PRICING["default"]

        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    def get_total_cost(self, model: Optional[str] = None) -> float:
        """
        Get total cost across all model calls.

        Args:
            model: Optional model name to filter by

        Returns:
            Total cost in USD
        """
        with self._lock:
            if model:
                return sum(
                    m.cost for m in self.token_metrics
                    if model.lower() in m.model.lower()
                )
            return sum(m.cost for m in self.token_metrics)

    def get_total_tokens(self, model: Optional[str] = None) -> Dict[str, int]:
        """
        Get total token usage.

        Args:
            model: Optional model name to filter by

        Returns:
            Dictionary with prompt, completion, and total tokens
        """
        with self._lock:
            metrics = self.token_metrics
            if model:
                metrics = [m for m in metrics if model.lower() in m.model.lower()]

            return {
                "prompt_tokens": sum(m.prompt_tokens for m in metrics),
                "completion_tokens": sum(m.completion_tokens for m in metrics),
                "total_tokens": sum(m.total_tokens for m in metrics),
            }

    def get_execution_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for execution metrics.

        Args:
            name: Optional operation name to filter by

        Returns:
            Dictionary with execution statistics
        """
        with self._lock:
            metrics = self.execution_metrics
            if name:
                metrics = [m for m in metrics if m.name == name]

            if not metrics:
                return {
                    "count": 0,
                    "success_rate": 0.0,
                    "avg_duration": 0.0,
                    "min_duration": 0.0,
                    "max_duration": 0.0,
                    "total_duration": 0.0,
                }

            durations = [m.duration for m in metrics if m.duration is not None]
            successes = sum(1 for m in metrics if m.success)

            stats = {
                "count": len(metrics),
                "success_rate": successes / len(metrics) if metrics else 0.0,
                "total_duration": sum(durations),
            }

            if durations:
                stats.update({
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "median_duration": statistics.median(durations),
                    "stdev_duration": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                })
            else:
                stats.update({
                    "avg_duration": 0.0,
                    "min_duration": 0.0,
                    "max_duration": 0.0,
                    "median_duration": 0.0,
                    "stdev_duration": 0.0,
                })

            return stats

    def get_throughput(self, window_seconds: float = 60.0) -> Dict[str, float]:
        """
        Calculate throughput metrics over a time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Dictionary with throughput metrics
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        with self._lock:
            # Token throughput
            recent_token_metrics = [
                m for m in self.token_metrics
                if m.timestamp >= cutoff_time
            ]
            total_tokens = sum(m.total_tokens for m in recent_token_metrics)
            tokens_per_second = total_tokens / window_seconds if window_seconds > 0 else 0

            # Request throughput
            recent_executions = [
                m for m in self.execution_metrics
                if m.start_time >= cutoff_time
            ]
            requests_per_second = len(recent_executions) / window_seconds if window_seconds > 0 else 0

            return {
                "tokens_per_second": tokens_per_second,
                "requests_per_second": requests_per_second,
                "window_seconds": window_seconds,
            }

    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get statistics for resource usage.

        Returns:
            Dictionary with resource usage statistics
        """
        with self._lock:
            snapshots = list(self.resource_snapshots)

        if not snapshots:
            return {
                "cpu": {"avg": 0.0, "min": 0.0, "max": 0.0},
                "memory": {"avg": 0.0, "min": 0.0, "max": 0.0},
            }

        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]

        stats = {
            "cpu": {
                "avg": statistics.mean(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0.0,
            },
            "memory": {
                "avg": statistics.mean(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": memory_values[-1] if memory_values else 0.0,
            },
        }

        # Add GPU stats if available
        if snapshots[-1].gpu_metrics:
            gpu_stats = []
            for i, gpu_metric in enumerate(snapshots[-1].gpu_metrics):
                gpu_loads = [
                    s.gpu_metrics[i]['load']
                    for s in snapshots
                    if s.gpu_metrics and len(s.gpu_metrics) > i
                ]
                gpu_memory = [
                    s.gpu_metrics[i]['memory_percent']
                    for s in snapshots
                    if s.gpu_metrics and len(s.gpu_metrics) > i
                ]

                gpu_stats.append({
                    "id": gpu_metric['id'],
                    "name": gpu_metric['name'],
                    "load": {
                        "avg": statistics.mean(gpu_loads) if gpu_loads else 0.0,
                        "min": min(gpu_loads) if gpu_loads else 0.0,
                        "max": max(gpu_loads) if gpu_loads else 0.0,
                        "current": gpu_metric['load'],
                    },
                    "memory": {
                        "avg": statistics.mean(gpu_memory) if gpu_memory else 0.0,
                        "min": min(gpu_memory) if gpu_memory else 0.0,
                        "max": max(gpu_memory) if gpu_memory else 0.0,
                        "current": gpu_metric['memory_percent'],
                    },
                })

            stats["gpu"] = gpu_stats

        return stats

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all metrics.

        Returns:
            Dictionary with complete metrics summary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "execution": self.get_execution_stats(),
            "tokens": self.get_total_tokens(),
            "cost": {
                "total": self.get_total_cost(),
                "by_model": self._get_cost_by_model(),
            },
            "throughput": self.get_throughput(),
            "resources": self.get_resource_stats(),
            "custom_metrics": {
                name: {
                    "count": len(values),
                    "avg": statistics.mean(values) if values else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                }
                for name, values in self.custom_metrics.items()
            },
        }

    def _get_cost_by_model(self) -> Dict[str, float]:
        """Get total cost grouped by model."""
        with self._lock:
            cost_by_model = defaultdict(float)
            for metric in self.token_metrics:
                cost_by_model[metric.model] += metric.cost
            return dict(cost_by_model)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.execution_metrics.clear()
            self.token_metrics.clear()
            self.resource_snapshots.clear()
            self.custom_metrics.clear()
            self._active_executions.clear()

    def export_to_json(self, filepath: Union[str, Path], pretty: bool = True) -> None:
        """
        Export metrics to JSON file.

        Args:
            filepath: Path to output file
            pretty: Whether to pretty-print JSON
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "execution_metrics": [
                {
                    "name": m.name,
                    "start_time": m.start_time,
                    "end_time": m.end_time,
                    "duration": m.duration,
                    "success": m.success,
                    "metadata": m.metadata,
                }
                for m in self.execution_metrics
            ],
            "token_metrics": [
                {
                    "model": m.model,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                    "cost": m.cost,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.token_metrics
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2 if pretty else None)

    def export_to_csv(self, filepath: Union[str, Path], metric_type: str = "execution") -> None:
        """
        Export metrics to CSV file.

        Args:
            filepath: Path to output file
            metric_type: Type of metrics to export ("execution" or "tokens")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if metric_type == "execution":
            with open(filepath, 'w', newline='') as f:
                if not self.execution_metrics:
                    return

                writer = csv.DictWriter(
                    f,
                    fieldnames=['name', 'start_time', 'end_time', 'duration', 'success']
                )
                writer.writeheader()
                for metric in self.execution_metrics:
                    writer.writerow({
                        'name': metric.name,
                        'start_time': metric.start_time,
                        'end_time': metric.end_time,
                        'duration': metric.duration,
                        'success': metric.success,
                    })

        elif metric_type == "tokens":
            with open(filepath, 'w', newline='') as f:
                if not self.token_metrics:
                    return

                writer = csv.DictWriter(
                    f,
                    fieldnames=['model', 'prompt_tokens', 'completion_tokens',
                               'total_tokens', 'cost', 'timestamp']
                )
                writer.writeheader()
                for metric in self.token_metrics:
                    writer.writerow({
                        'model': metric.model,
                        'prompt_tokens': metric.prompt_tokens,
                        'completion_tokens': metric.completion_tokens,
                        'total_tokens': metric.total_tokens,
                        'cost': metric.cost,
                        'timestamp': metric.timestamp,
                    })

    def __del__(self):
        """Cleanup when collector is destroyed."""
        self.stop_resource_monitoring()


# Singleton instance for global metrics collection
_global_metrics = None
_global_metrics_lock = threading.Lock()


def get_global_metrics() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance

    Example:
        >>> metrics = get_global_metrics()
        >>> with metrics.track_execution("task"):
        ...     do_work()
    """
    global _global_metrics
    if _global_metrics is None:
        with _global_metrics_lock:
            if _global_metrics is None:
                _global_metrics = MetricsCollector()
    return _global_metrics


__all__ = [
    'MODEL_PRICING',
    'ExecutionMetric',
    'TokenMetric',
    'ResourceSnapshot',
    'MetricsCollector',
    'get_global_metrics',
]
