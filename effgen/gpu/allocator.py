"""
GPU Allocator for effGen Framework

This module provides intelligent GPU allocation and resource management for
multi-GPU deployments. It supports various allocation strategies, tensor/pipeline
parallelism, and prevents resource overallocation.

Author: effGen Team
License: MIT
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from threading import Lock
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU allocation will use CPU fallback.")


logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """GPU allocation strategies"""
    GREEDY = "greedy"  # Fill GPUs sequentially
    BALANCED = "balanced"  # Distribute evenly across GPUs
    OPTIMIZE = "optimize"  # Minimize inter-GPU communication
    PRIORITY = "priority"  # Allocate based on task priority


class ParallelismType(Enum):
    """Types of model parallelism"""
    TENSOR = "tensor"  # Tensor parallelism (split layers across GPUs)
    PIPELINE = "pipeline"  # Pipeline parallelism (different layers on different GPUs)
    DATA = "data"  # Data parallelism (replicate model, split batches)
    NONE = "none"  # No parallelism


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    device_id: int
    name: str
    total_memory: int  # in bytes
    available_memory: int  # in bytes
    utilization: float  # 0.0 to 1.0
    temperature: Optional[float] = None  # in Celsius
    power_usage: Optional[float] = None  # in Watts

    @property
    def used_memory(self) -> int:
        """Calculate used memory in bytes"""
        return self.total_memory - self.available_memory

    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization (0.0 to 1.0)"""
        if self.total_memory == 0:
            return 0.0
        return self.used_memory / self.total_memory


@dataclass
class AllocationRequest:
    """Request for GPU allocation"""
    requester_id: str
    memory_required: int  # in bytes
    num_gpus: int = 1
    strategy: AllocationStrategy = AllocationStrategy.BALANCED
    parallelism_type: ParallelismType = ParallelismType.NONE
    preferred_devices: Optional[List[int]] = None
    priority: int = 0  # Higher priority = more important
    allow_shared: bool = True  # Allow sharing GPU with other tasks


@dataclass
class Allocation:
    """Represents an active GPU allocation"""
    requester_id: str
    device_ids: List[int]
    memory_allocated: Dict[int, int]  # device_id -> memory in bytes
    parallelism_type: ParallelismType
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class GPUAllocator:
    """
    Intelligent GPU allocation and resource management.

    This class handles:
    - Device discovery and monitoring
    - Smart allocation based on various strategies
    - Tensor and pipeline parallelism support
    - Resource tracking and deallocation
    - Prevention of overallocation

    Example:
        >>> allocator = GPUAllocator()
        >>> request = AllocationRequest(
        ...     requester_id="agent_1",
        ...     memory_required=8 * 1024**3,  # 8GB
        ...     num_gpus=2,
        ...     strategy=AllocationStrategy.BALANCED
        ... )
        >>> allocation = allocator.allocate(request)
        >>> if allocation:
        ...     print(f"Allocated GPUs: {allocation.device_ids}")
        >>> allocator.deallocate("agent_1")
    """

    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize GPU allocator.

        Args:
            enable_monitoring: Enable real-time GPU monitoring
        """
        self._lock = Lock()
        self._allocations: Dict[str, Allocation] = {}
        self._devices: Dict[int, GPUInfo] = {}
        self._enable_monitoring = enable_monitoring

        # Initialize devices
        self._discover_devices()

        logger.info(
            f"GPUAllocator initialized with {len(self._devices)} GPU(s). "
            f"CUDA available: {self.is_cuda_available()}"
        )

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        if not TORCH_AVAILABLE:
            return False
        return torch.cuda.is_available()

    def _discover_devices(self) -> None:
        """Discover available GPU devices"""
        if not self.is_cuda_available():
            logger.warning("CUDA not available. Running in CPU mode.")
            return

        try:
            num_devices = torch.cuda.device_count()

            for device_id in range(num_devices):
                props = torch.cuda.get_device_properties(device_id)

                # Get current memory info
                torch.cuda.set_device(device_id)
                total_memory = props.total_memory
                reserved = torch.cuda.memory_reserved(device_id)
                allocated = torch.cuda.memory_allocated(device_id)
                available = total_memory - reserved

                gpu_info = GPUInfo(
                    device_id=device_id,
                    name=props.name,
                    total_memory=total_memory,
                    available_memory=available,
                    utilization=0.0  # Will be updated by monitor
                )

                self._devices[device_id] = gpu_info

                logger.info(
                    f"Discovered GPU {device_id}: {gpu_info.name} "
                    f"with {gpu_info.total_memory / (1024**3):.2f} GB VRAM"
                )

        except Exception as e:
            logger.error(f"Error discovering GPU devices: {e}")

    def get_device_info(self, device_id: Optional[int] = None) -> Dict[int, GPUInfo]:
        """
        Get information about GPU devices.

        Args:
            device_id: Specific device ID, or None for all devices

        Returns:
            Dictionary mapping device IDs to GPUInfo objects
        """
        self._refresh_device_info()

        if device_id is not None:
            if device_id in self._devices:
                return {device_id: self._devices[device_id]}
            else:
                raise ValueError(f"Device {device_id} not found")

        return self._devices.copy()

    def _refresh_device_info(self) -> None:
        """Refresh device memory information"""
        if not self.is_cuda_available():
            return

        try:
            for device_id in self._devices:
                torch.cuda.set_device(device_id)
                total = self._devices[device_id].total_memory
                reserved = torch.cuda.memory_reserved(device_id)
                self._devices[device_id].available_memory = total - reserved
        except Exception as e:
            logger.error(f"Error refreshing device info: {e}")

    def allocate(self, request: AllocationRequest) -> Optional[Allocation]:
        """
        Allocate GPUs based on the request.

        Args:
            request: Allocation request specifying requirements

        Returns:
            Allocation object if successful, None if allocation failed
        """
        with self._lock:
            # Check if requester already has an allocation
            if request.requester_id in self._allocations:
                logger.warning(
                    f"Requester {request.requester_id} already has an allocation. "
                    f"Deallocate first before requesting new allocation."
                )
                return None

            # Refresh device info
            self._refresh_device_info()

            # Select devices based on strategy
            selected_devices = self._select_devices(request)

            if not selected_devices:
                logger.warning(
                    f"Could not allocate {request.num_gpus} GPU(s) for "
                    f"{request.requester_id} with {request.memory_required / (1024**3):.2f} GB"
                )
                return None

            # Create allocation
            memory_per_device = {}
            for device_id in selected_devices:
                if request.parallelism_type == ParallelismType.TENSOR:
                    # Tensor parallelism: split memory across devices
                    memory_per_device[device_id] = request.memory_required // len(selected_devices)
                else:
                    # Other types: full memory on each device
                    memory_per_device[device_id] = request.memory_required

            allocation = Allocation(
                requester_id=request.requester_id,
                device_ids=selected_devices,
                memory_allocated=memory_per_device,
                parallelism_type=request.parallelism_type
            )

            # Update allocations
            self._allocations[request.requester_id] = allocation

            logger.info(
                f"Allocated GPU(s) {selected_devices} to {request.requester_id} "
                f"with parallelism: {request.parallelism_type.value}"
            )

            return allocation

    def _select_devices(self, request: AllocationRequest) -> List[int]:
        """
        Select GPU devices based on allocation strategy.

        Args:
            request: Allocation request

        Returns:
            List of selected device IDs
        """
        if not self._devices:
            logger.warning("No GPU devices available")
            return []

        # Filter available devices
        available_devices = []
        for device_id, info in self._devices.items():
            # Check if device is in preferred list (if specified)
            if request.preferred_devices and device_id not in request.preferred_devices:
                continue

            # Check if device has enough memory
            required_memory = request.memory_required
            if request.parallelism_type == ParallelismType.TENSOR:
                # Tensor parallelism splits memory across devices
                required_memory = request.memory_required // request.num_gpus

            # Account for existing allocations
            used_memory = sum(
                alloc.memory_allocated.get(device_id, 0)
                for alloc in self._allocations.values()
            )

            free_memory = info.total_memory - used_memory

            if free_memory >= required_memory:
                available_devices.append((device_id, free_memory, info.utilization))

        if len(available_devices) < request.num_gpus:
            logger.warning(
                f"Only {len(available_devices)} GPU(s) available, "
                f"but {request.num_gpus} requested"
            )
            return []

        # Apply allocation strategy
        if request.strategy == AllocationStrategy.GREEDY:
            return self._greedy_selection(available_devices, request.num_gpus)
        elif request.strategy == AllocationStrategy.BALANCED:
            return self._balanced_selection(available_devices, request.num_gpus)
        elif request.strategy == AllocationStrategy.OPTIMIZE:
            return self._optimized_selection(available_devices, request.num_gpus)
        elif request.strategy == AllocationStrategy.PRIORITY:
            return self._priority_selection(available_devices, request)
        else:
            logger.warning(f"Unknown strategy {request.strategy}, using BALANCED")
            return self._balanced_selection(available_devices, request.num_gpus)

    def _greedy_selection(
        self,
        available_devices: List[Tuple[int, int, float]],
        num_gpus: int
    ) -> List[int]:
        """
        Greedy selection: Fill GPUs sequentially.

        Args:
            available_devices: List of (device_id, free_memory, utilization)
            num_gpus: Number of GPUs to select

        Returns:
            List of selected device IDs
        """
        # Sort by device ID
        sorted_devices = sorted(available_devices, key=lambda x: x[0])
        return [device_id for device_id, _, _ in sorted_devices[:num_gpus]]

    def _balanced_selection(
        self,
        available_devices: List[Tuple[int, int, float]],
        num_gpus: int
    ) -> List[int]:
        """
        Balanced selection: Select GPUs with lowest utilization.

        Args:
            available_devices: List of (device_id, free_memory, utilization)
            num_gpus: Number of GPUs to select

        Returns:
            List of selected device IDs
        """
        # Sort by utilization (lower is better), then by free memory (higher is better)
        sorted_devices = sorted(
            available_devices,
            key=lambda x: (x[2], -x[1])  # utilization asc, free_memory desc
        )
        return [device_id for device_id, _, _ in sorted_devices[:num_gpus]]

    def _optimized_selection(
        self,
        available_devices: List[Tuple[int, int, float]],
        num_gpus: int
    ) -> List[int]:
        """
        Optimized selection: Minimize inter-GPU communication by selecting adjacent GPUs.

        Args:
            available_devices: List of (device_id, free_memory, utilization)
            num_gpus: Number of GPUs to select

        Returns:
            List of selected device IDs
        """
        if num_gpus == 1:
            return self._balanced_selection(available_devices, num_gpus)

        # Sort by device ID
        sorted_devices = sorted(available_devices, key=lambda x: x[0])
        device_ids = [d[0] for d in sorted_devices]

        # Find contiguous sequences of GPUs
        best_sequence = []
        best_score = float('inf')

        for i in range(len(device_ids) - num_gpus + 1):
            sequence = device_ids[i:i + num_gpus]

            # Check if sequence is contiguous
            is_contiguous = all(
                sequence[j+1] - sequence[j] == 1
                for j in range(len(sequence) - 1)
            )

            if is_contiguous:
                # Score based on average utilization
                devices_info = [d for d in sorted_devices if d[0] in sequence]
                avg_utilization = sum(d[2] for d in devices_info) / len(devices_info)

                if avg_utilization < best_score:
                    best_score = avg_utilization
                    best_sequence = sequence

        # If no contiguous sequence found, fall back to balanced
        if not best_sequence:
            return self._balanced_selection(available_devices, num_gpus)

        return best_sequence

    def _priority_selection(
        self,
        available_devices: List[Tuple[int, int, float]],
        request: AllocationRequest
    ) -> List[int]:
        """
        Priority-based selection: Allocate best GPUs to high priority tasks.

        Args:
            available_devices: List of (device_id, free_memory, utilization)
            request: Allocation request with priority

        Returns:
            List of selected device IDs
        """
        # High priority tasks get GPUs with most free memory and lowest utilization
        sorted_devices = sorted(
            available_devices,
            key=lambda x: (-x[1], x[2])  # free_memory desc, utilization asc
        )

        return [device_id for device_id, _, _ in sorted_devices[:request.num_gpus]]

    def deallocate(self, requester_id: str) -> bool:
        """
        Deallocate GPU resources for a requester.

        Args:
            requester_id: ID of the requester to deallocate

        Returns:
            True if deallocation successful, False otherwise
        """
        with self._lock:
            if requester_id not in self._allocations:
                logger.warning(f"No allocation found for {requester_id}")
                return False

            allocation = self._allocations.pop(requester_id)

            logger.info(
                f"Deallocated GPU(s) {allocation.device_ids} from {requester_id}"
            )

            # Clear GPU cache if using PyTorch
            if self.is_cuda_available():
                try:
                    for device_id in allocation.device_ids:
                        torch.cuda.set_device(device_id)
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error clearing GPU cache: {e}")

            return True

    def get_allocation(self, requester_id: str) -> Optional[Allocation]:
        """
        Get allocation for a specific requester.

        Args:
            requester_id: ID of the requester

        Returns:
            Allocation object if found, None otherwise
        """
        return self._allocations.get(requester_id)

    def list_allocations(self) -> Dict[str, Allocation]:
        """
        List all active allocations.

        Returns:
            Dictionary mapping requester IDs to Allocation objects
        """
        return self._allocations.copy()

    def get_available_memory(self, device_id: int) -> int:
        """
        Get available memory on a specific GPU.

        Args:
            device_id: GPU device ID

        Returns:
            Available memory in bytes
        """
        self._refresh_device_info()

        if device_id not in self._devices:
            raise ValueError(f"Device {device_id} not found")

        # Account for existing allocations
        used_memory = sum(
            alloc.memory_allocated.get(device_id, 0)
            for alloc in self._allocations.values()
        )

        return self._devices[device_id].total_memory - used_memory

    def can_allocate(self, request: AllocationRequest) -> bool:
        """
        Check if allocation request can be fulfilled.

        Args:
            request: Allocation request

        Returns:
            True if request can be fulfilled, False otherwise
        """
        with self._lock:
            self._refresh_device_info()
            selected_devices = self._select_devices(request)
            return len(selected_devices) >= request.num_gpus

    def get_total_memory(self, device_id: Optional[int] = None) -> int:
        """
        Get total memory across GPUs.

        Args:
            device_id: Specific device ID, or None for all devices

        Returns:
            Total memory in bytes
        """
        if device_id is not None:
            if device_id not in self._devices:
                raise ValueError(f"Device {device_id} not found")
            return self._devices[device_id].total_memory

        return sum(info.total_memory for info in self._devices.values())

    def reset(self) -> None:
        """Clear all allocations and reset state"""
        with self._lock:
            requester_ids = list(self._allocations.keys())
            for requester_id in requester_ids:
                self.deallocate(requester_id)

            logger.info("GPU allocator reset complete")

    def __repr__(self) -> str:
        """String representation"""
        num_devices = len(self._devices)
        num_allocations = len(self._allocations)
        total_memory = self.get_total_memory() / (1024**3)

        return (
            f"GPUAllocator(devices={num_devices}, "
            f"allocations={num_allocations}, "
            f"total_memory={total_memory:.2f}GB)"
        )
