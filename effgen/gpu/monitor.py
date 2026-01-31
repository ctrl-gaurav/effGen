"""
GPU Monitor for effGen Framework

This module provides real-time GPU monitoring capabilities including VRAM usage,
utilization, temperature, power consumption, and active process tracking. It includes
an alert system for threshold breaches and comprehensive metrics logging.

Author: effGen Team
License: MIT
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Thread, Event, Lock
from typing import Dict, List, Optional, Callable, Any
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU monitoring will use CPU fallback.")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn(
        "pynvml not available. Advanced GPU monitoring features disabled. "
        "Install with: pip install nvidia-ml-py3"
    )


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of GPU metrics"""
    VRAM_USAGE = "vram_usage"
    VRAM_UTILIZATION = "vram_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    TEMPERATURE = "temperature"
    POWER_USAGE = "power_usage"
    FAN_SPEED = "fan_speed"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class GPUMetrics:
    """Metrics for a single GPU at a point in time"""
    device_id: int
    timestamp: float
    vram_total: int  # bytes
    vram_used: int  # bytes
    vram_free: int  # bytes
    gpu_utilization: float  # 0.0 to 1.0
    memory_utilization: float  # 0.0 to 1.0
    temperature: Optional[float] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    power_limit: Optional[float] = None  # Watts
    fan_speed: Optional[float] = None  # 0.0 to 1.0
    processes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def vram_used_gb(self) -> float:
        """VRAM used in GB"""
        return self.vram_used / (1024**3)

    @property
    def vram_total_gb(self) -> float:
        """Total VRAM in GB"""
        return self.vram_total / (1024**3)

    @property
    def vram_free_gb(self) -> float:
        """Free VRAM in GB"""
        return self.vram_free / (1024**3)


@dataclass
class Alert:
    """Alert for threshold breach"""
    device_id: int
    metric_type: MetricType
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MonitorConfig:
    """Configuration for GPU monitor"""
    update_interval: float = 1.0  # seconds
    enable_alerts: bool = True
    enable_logging: bool = True
    log_interval: float = 60.0  # seconds

    # Alert thresholds
    vram_warning_threshold: float = 0.8  # 80%
    vram_critical_threshold: float = 0.95  # 95%
    gpu_utilization_warning: float = 0.9  # 90%
    temperature_warning: float = 80.0  # Celsius
    temperature_critical: float = 90.0  # Celsius
    power_warning_threshold: float = 0.9  # 90% of power limit


class GPUMonitor:
    """
    Real-time GPU monitoring with alerts and metrics logging.

    This class provides:
    - Real-time VRAM usage tracking
    - GPU utilization monitoring
    - Temperature and power monitoring
    - Active process tracking
    - Alert system on threshold breach
    - Comprehensive metrics logging

    Example:
        >>> config = MonitorConfig(update_interval=2.0)
        >>> monitor = GPUMonitor(config)
        >>> monitor.start()
        >>>
        >>> # Get current metrics
        >>> metrics = monitor.get_metrics()
        >>> for device_id, metric in metrics.items():
        ...     print(f"GPU {device_id}: {metric.vram_used_gb:.2f}GB used")
        >>>
        >>> # Add alert callback
        >>> def on_alert(alert):
        ...     print(f"ALERT: {alert.message}")
        >>> monitor.add_alert_callback(on_alert)
        >>>
        >>> monitor.stop()
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize GPU monitor.

        Args:
            config: Monitor configuration (uses defaults if None)
        """
        self.config = config or MonitorConfig()
        self._lock = Lock()
        self._running = False
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        self._log_thread: Optional[Thread] = None

        # Metrics storage
        self._current_metrics: Dict[int, GPUMetrics] = {}
        self._metrics_history: Dict[int, List[GPUMetrics]] = {}
        self._max_history_size = 1000

        # Alerts
        self._alerts: List[Alert] = []
        self._max_alerts = 100
        self._alert_callbacks: List[Callable[[Alert], None]] = []

        # NVML initialization
        self._nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                logger.info("NVML initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")

        # Discover devices
        self._device_handles: Dict[int, Any] = {}
        self._discover_devices()

        logger.info(f"GPUMonitor initialized for {len(self._device_handles)} device(s)")

    def _discover_devices(self) -> None:
        """Discover available GPU devices"""
        if self._nvml_initialized:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self._device_handles[i] = handle
                    self._metrics_history[i] = []
                    logger.info(f"Discovered GPU {i}")
            except Exception as e:
                logger.error(f"Error discovering NVML devices: {e}")

        elif TORCH_AVAILABLE and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                self._device_handles[i] = None  # PyTorch doesn't use handles
                self._metrics_history[i] = []
                logger.info(f"Discovered GPU {i} (PyTorch)")

    def start(self) -> None:
        """Start monitoring in background thread"""
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True
        self._stop_event.clear()

        # Start monitoring thread
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # Start logging thread if enabled
        if self.config.enable_logging:
            self._log_thread = Thread(target=self._log_loop, daemon=True)
            self._log_thread.start()

        logger.info("GPU monitoring started")

    def stop(self) -> None:
        """Stop monitoring"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Wait for threads to finish
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        if self._log_thread:
            self._log_thread.join(timeout=5.0)

        logger.info("GPU monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                self._update_metrics()
                if self.config.enable_alerts:
                    self._check_thresholds()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

            # Sleep with interruptible wait
            self._stop_event.wait(self.config.update_interval)

    def _log_loop(self) -> None:
        """Periodic logging loop"""
        while self._running:
            try:
                self._log_metrics()
            except Exception as e:
                logger.error(f"Error in log loop: {e}")

            # Sleep with interruptible wait
            self._stop_event.wait(self.config.log_interval)

    def _update_metrics(self) -> None:
        """Update metrics for all devices"""
        timestamp = time.time()

        for device_id in self._device_handles:
            try:
                metrics = self._collect_metrics(device_id, timestamp)

                with self._lock:
                    self._current_metrics[device_id] = metrics

                    # Add to history
                    self._metrics_history[device_id].append(metrics)

                    # Limit history size
                    if len(self._metrics_history[device_id]) > self._max_history_size:
                        self._metrics_history[device_id].pop(0)

            except Exception as e:
                logger.error(f"Error collecting metrics for GPU {device_id}: {e}")

    def _collect_metrics(self, device_id: int, timestamp: float) -> GPUMetrics:
        """
        Collect metrics for a specific device.

        Args:
            device_id: GPU device ID
            timestamp: Current timestamp

        Returns:
            GPUMetrics object
        """
        if self._nvml_initialized:
            return self._collect_nvml_metrics(device_id, timestamp)
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            return self._collect_torch_metrics(device_id, timestamp)
        else:
            # CPU fallback
            return GPUMetrics(
                device_id=device_id,
                timestamp=timestamp,
                vram_total=0,
                vram_used=0,
                vram_free=0,
                gpu_utilization=0.0,
                memory_utilization=0.0
            )

    def _collect_nvml_metrics(self, device_id: int, timestamp: float) -> GPUMetrics:
        """Collect metrics using NVML (most detailed)"""
        handle = self._device_handles[device_id]

        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total = mem_info.total
        vram_used = mem_info.used
        vram_free = mem_info.free

        # Utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu / 100.0
        mem_util = utilization.memory / 100.0

        # Temperature
        try:
            temperature = float(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            ))
        except:
            temperature = None

        # Power
        try:
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except:
            power_usage = None
            power_limit = None

        # Fan speed
        try:
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle) / 100.0
        except:
            fan_speed = None

        # Processes
        processes = []
        try:
            proc_infos = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in proc_infos:
                processes.append({
                    'pid': proc.pid,
                    'used_memory': proc.usedGpuMemory,
                })
        except:
            pass

        return GPUMetrics(
            device_id=device_id,
            timestamp=timestamp,
            vram_total=vram_total,
            vram_used=vram_used,
            vram_free=vram_free,
            gpu_utilization=gpu_util,
            memory_utilization=mem_util,
            temperature=temperature,
            power_usage=power_usage,
            power_limit=power_limit,
            fan_speed=fan_speed,
            processes=processes
        )

    def _collect_torch_metrics(self, device_id: int, timestamp: float) -> GPUMetrics:
        """Collect metrics using PyTorch (basic info)"""
        torch.cuda.set_device(device_id)

        # Memory info
        vram_total = torch.cuda.get_device_properties(device_id).total_memory
        vram_reserved = torch.cuda.memory_reserved(device_id)
        vram_allocated = torch.cuda.memory_allocated(device_id)
        vram_free = vram_total - vram_reserved

        return GPUMetrics(
            device_id=device_id,
            timestamp=timestamp,
            vram_total=vram_total,
            vram_used=vram_allocated,
            vram_free=vram_free,
            gpu_utilization=0.0,  # Not available in PyTorch
            memory_utilization=vram_allocated / vram_total if vram_total > 0 else 0.0
        )

    def _check_thresholds(self) -> None:
        """Check metrics against thresholds and generate alerts"""
        with self._lock:
            for device_id, metrics in self._current_metrics.items():
                # VRAM utilization
                if metrics.memory_utilization >= self.config.vram_critical_threshold:
                    self._create_alert(
                        device_id,
                        MetricType.VRAM_UTILIZATION,
                        AlertLevel.CRITICAL,
                        f"GPU {device_id} VRAM critically high: "
                        f"{metrics.memory_utilization*100:.1f}%",
                        metrics.memory_utilization,
                        self.config.vram_critical_threshold
                    )
                elif metrics.memory_utilization >= self.config.vram_warning_threshold:
                    self._create_alert(
                        device_id,
                        MetricType.VRAM_UTILIZATION,
                        AlertLevel.WARNING,
                        f"GPU {device_id} VRAM usage high: "
                        f"{metrics.memory_utilization*100:.1f}%",
                        metrics.memory_utilization,
                        self.config.vram_warning_threshold
                    )

                # GPU utilization
                if (metrics.gpu_utilization >= self.config.gpu_utilization_warning
                    and metrics.gpu_utilization > 0):
                    self._create_alert(
                        device_id,
                        MetricType.GPU_UTILIZATION,
                        AlertLevel.WARNING,
                        f"GPU {device_id} utilization high: "
                        f"{metrics.gpu_utilization*100:.1f}%",
                        metrics.gpu_utilization,
                        self.config.gpu_utilization_warning
                    )

                # Temperature
                if metrics.temperature is not None:
                    if metrics.temperature >= self.config.temperature_critical:
                        self._create_alert(
                            device_id,
                            MetricType.TEMPERATURE,
                            AlertLevel.CRITICAL,
                            f"GPU {device_id} temperature critically high: "
                            f"{metrics.temperature:.1f}°C",
                            metrics.temperature,
                            self.config.temperature_critical
                        )
                    elif metrics.temperature >= self.config.temperature_warning:
                        self._create_alert(
                            device_id,
                            MetricType.TEMPERATURE,
                            AlertLevel.WARNING,
                            f"GPU {device_id} temperature high: "
                            f"{metrics.temperature:.1f}°C",
                            metrics.temperature,
                            self.config.temperature_warning
                        )

                # Power usage
                if (metrics.power_usage is not None
                    and metrics.power_limit is not None
                    and metrics.power_limit > 0):
                    power_ratio = metrics.power_usage / metrics.power_limit
                    if power_ratio >= self.config.power_warning_threshold:
                        self._create_alert(
                            device_id,
                            MetricType.POWER_USAGE,
                            AlertLevel.WARNING,
                            f"GPU {device_id} power usage high: "
                            f"{metrics.power_usage:.1f}W / {metrics.power_limit:.1f}W",
                            power_ratio,
                            self.config.power_warning_threshold
                        )

    def _create_alert(
        self,
        device_id: int,
        metric_type: MetricType,
        level: AlertLevel,
        message: str,
        value: float,
        threshold: float
    ) -> None:
        """Create and process an alert"""
        alert = Alert(
            device_id=device_id,
            metric_type=metric_type,
            level=level,
            message=message,
            value=value,
            threshold=threshold
        )

        # Add to alerts list
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts.pop(0)

        # Log alert
        if level == AlertLevel.CRITICAL:
            logger.critical(message)
        elif level == AlertLevel.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

        # Call callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _log_metrics(self) -> None:
        """Log current metrics"""
        with self._lock:
            if not self._current_metrics:
                return

            metrics_summary = []
            for device_id, metrics in sorted(self._current_metrics.items()):
                summary = (
                    f"GPU {device_id}: "
                    f"VRAM {metrics.vram_used_gb:.2f}/{metrics.vram_total_gb:.2f}GB "
                    f"({metrics.memory_utilization*100:.1f}%), "
                    f"Util {metrics.gpu_utilization*100:.1f}%"
                )

                if metrics.temperature is not None:
                    summary += f", Temp {metrics.temperature:.1f}°C"

                if metrics.power_usage is not None:
                    summary += f", Power {metrics.power_usage:.1f}W"

                metrics_summary.append(summary)

            logger.info("GPU Metrics: " + " | ".join(metrics_summary))

    def get_metrics(
        self,
        device_id: Optional[int] = None
    ) -> Dict[int, GPUMetrics]:
        """
        Get current metrics.

        Args:
            device_id: Specific device ID, or None for all devices

        Returns:
            Dictionary mapping device IDs to GPUMetrics
        """
        with self._lock:
            if device_id is not None:
                if device_id in self._current_metrics:
                    return {device_id: self._current_metrics[device_id]}
                else:
                    return {}
            return self._current_metrics.copy()

    def get_metrics_history(
        self,
        device_id: int,
        limit: Optional[int] = None
    ) -> List[GPUMetrics]:
        """
        Get metrics history for a device.

        Args:
            device_id: GPU device ID
            limit: Maximum number of metrics to return (most recent)

        Returns:
            List of GPUMetrics ordered by timestamp (oldest to newest)
        """
        with self._lock:
            if device_id not in self._metrics_history:
                return []

            history = self._metrics_history[device_id]
            if limit:
                return history[-limit:]
            return history.copy()

    def get_alerts(
        self,
        device_id: Optional[int] = None,
        level: Optional[AlertLevel] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """
        Get alerts.

        Args:
            device_id: Filter by device ID
            level: Filter by alert level
            limit: Maximum number of alerts to return (most recent)

        Returns:
            List of Alert objects
        """
        with self._lock:
            alerts = self._alerts.copy()

            # Filter by device
            if device_id is not None:
                alerts = [a for a in alerts if a.device_id == device_id]

            # Filter by level
            if level is not None:
                alerts = [a for a in alerts if a.level == level]

            # Apply limit
            if limit:
                alerts = alerts[-limit:]

            return alerts

    def clear_alerts(self) -> None:
        """Clear all alerts"""
        with self._lock:
            self._alerts.clear()
        logger.info("Alerts cleared")

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add callback for alerts.

        Args:
            callback: Function that takes Alert object as parameter
        """
        self._alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")

    def remove_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Remove alert callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
            logger.info(f"Removed alert callback: {callback.__name__}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current GPU status.

        Returns:
            Dictionary with summary information
        """
        with self._lock:
            summary = {
                'num_devices': len(self._device_handles),
                'monitoring_active': self._running,
                'total_alerts': len(self._alerts),
                'devices': {}
            }

            for device_id, metrics in self._current_metrics.items():
                summary['devices'][device_id] = {
                    'vram_used_gb': metrics.vram_used_gb,
                    'vram_total_gb': metrics.vram_total_gb,
                    'memory_utilization': metrics.memory_utilization,
                    'gpu_utilization': metrics.gpu_utilization,
                    'temperature': metrics.temperature,
                    'power_usage': metrics.power_usage,
                    'num_processes': len(metrics.processes)
                }

            return summary

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

    def __del__(self):
        """Cleanup on deletion"""
        if self._running:
            self.stop()

        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"GPUMonitor(devices={len(self._device_handles)}, "
            f"running={self._running}, "
            f"alerts={len(self._alerts)})"
        )
