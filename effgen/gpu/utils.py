"""
GPU Utility Functions for effGen Framework

This module provides utility functions for GPU operations including memory estimation,
device selection, compatibility checks, and various helper functions for GPU management.

Author: effGen Team
License: MIT
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU utilities will use CPU fallback.")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


logger = logging.getLogger(__name__)


# Memory size constants (in bytes)
KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB


def is_gpu_available() -> bool:
    """
    Check if GPU is available.

    Returns:
        True if at least one GPU is available, False otherwise
    """
    if not TORCH_AVAILABLE:
        return False

    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_device_count() -> int:
    """
    Get number of available GPU devices.

    Returns:
        Number of GPU devices (0 if no GPUs available)
    """
    if not is_gpu_available():
        return 0

    return torch.cuda.device_count()


def get_device_name(device_id: int) -> str:
    """
    Get name of GPU device.

    Args:
        device_id: GPU device ID

    Returns:
        Device name string

    Raises:
        ValueError: If device_id is invalid
    """
    if not is_gpu_available():
        raise RuntimeError("No GPU devices available")

    if device_id < 0 or device_id >= get_device_count():
        raise ValueError(f"Invalid device ID: {device_id}")

    return torch.cuda.get_device_properties(device_id).name


def get_device_capability(device_id: int) -> Tuple[int, int]:
    """
    Get CUDA compute capability of device.

    Args:
        device_id: GPU device ID

    Returns:
        Tuple of (major, minor) version numbers

    Raises:
        ValueError: If device_id is invalid
    """
    if not is_gpu_available():
        raise RuntimeError("No GPU devices available")

    if device_id < 0 or device_id >= get_device_count():
        raise ValueError(f"Invalid device ID: {device_id}")

    props = torch.cuda.get_device_properties(device_id)
    return (props.major, props.minor)


def get_total_memory(device_id: int) -> int:
    """
    Get total memory of GPU device.

    Args:
        device_id: GPU device ID

    Returns:
        Total memory in bytes

    Raises:
        ValueError: If device_id is invalid
    """
    if not is_gpu_available():
        return 0

    if device_id < 0 or device_id >= get_device_count():
        raise ValueError(f"Invalid device ID: {device_id}")

    return torch.cuda.get_device_properties(device_id).total_memory


def get_free_memory(device_id: int) -> int:
    """
    Get free memory on GPU device.

    Args:
        device_id: GPU device ID

    Returns:
        Free memory in bytes

    Raises:
        ValueError: If device_id is invalid
    """
    if not is_gpu_available():
        return 0

    if device_id < 0 or device_id >= get_device_count():
        raise ValueError(f"Invalid device ID: {device_id}")

    torch.cuda.set_device(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    reserved = torch.cuda.memory_reserved(device_id)

    return total - reserved


def get_allocated_memory(device_id: int) -> int:
    """
    Get allocated memory on GPU device.

    Args:
        device_id: GPU device ID

    Returns:
        Allocated memory in bytes

    Raises:
        ValueError: If device_id is invalid
    """
    if not is_gpu_available():
        return 0

    if device_id < 0 or device_id >= get_device_count():
        raise ValueError(f"Invalid device ID: {device_id}")

    torch.cuda.set_device(device_id)
    return torch.cuda.memory_allocated(device_id)


def format_memory_size(size_bytes: int) -> str:
    """
    Format memory size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "4.5 GB", "512 MB")
    """
    if size_bytes >= TB:
        return f"{size_bytes / TB:.2f} TB"
    elif size_bytes >= GB:
        return f"{size_bytes / GB:.2f} GB"
    elif size_bytes >= MB:
        return f"{size_bytes / MB:.2f} MB"
    elif size_bytes >= KB:
        return f"{size_bytes / KB:.2f} KB"
    else:
        return f"{size_bytes} B"


def estimate_model_memory(
    num_parameters: int,
    dtype: str = "float32",
    quantization: Optional[str] = None,
    overhead_factor: float = 1.2
) -> int:
    """
    Estimate memory required for a model.

    Args:
        num_parameters: Number of model parameters
        dtype: Data type ('float32', 'float16', 'bfloat16')
        quantization: Quantization type ('4bit', '8bit', None)
        overhead_factor: Overhead multiplier for activations, gradients, etc.

    Returns:
        Estimated memory in bytes
    """
    # Bytes per parameter based on dtype
    bytes_per_param = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1,
    }.get(dtype.lower(), 4)

    # Apply quantization
    if quantization:
        if '4bit' in quantization.lower():
            bytes_per_param = 0.5
        elif '8bit' in quantization.lower():
            bytes_per_param = 1

    # Calculate base memory
    base_memory = num_parameters * bytes_per_param

    # Apply overhead factor
    total_memory = int(base_memory * overhead_factor)

    return total_memory


def estimate_batch_memory(
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int = 1,
    dtype: str = "float32"
) -> int:
    """
    Estimate memory required for batch processing.

    Args:
        sequence_length: Length of input sequence
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        batch_size: Batch size
        dtype: Data type ('float32', 'float16', 'bfloat16')

    Returns:
        Estimated memory in bytes
    """
    bytes_per_element = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
    }.get(dtype.lower(), 4)

    # Estimate activations memory
    # Rough approximation: batch_size * seq_len * hidden_size * num_layers * 12
    # (12 is approximate factor for all activations in transformer)
    activations = batch_size * sequence_length * hidden_size * num_layers * 12

    total_memory = activations * bytes_per_element

    return int(total_memory)


def select_best_device(
    memory_required: int,
    exclude_devices: Optional[List[int]] = None,
    prefer_empty: bool = True
) -> Optional[int]:
    """
    Select best GPU device based on available memory.

    Args:
        memory_required: Required memory in bytes
        exclude_devices: List of device IDs to exclude
        prefer_empty: Prefer devices with more free memory

    Returns:
        Best device ID, or None if no suitable device found
    """
    if not is_gpu_available():
        return None

    exclude_devices = exclude_devices or []
    best_device = None
    best_free_memory = 0

    for device_id in range(get_device_count()):
        if device_id in exclude_devices:
            continue

        free_memory = get_free_memory(device_id)

        if free_memory >= memory_required:
            if prefer_empty and free_memory > best_free_memory:
                best_device = device_id
                best_free_memory = free_memory
            elif not prefer_empty and best_device is None:
                best_device = device_id
                break

    return best_device


def select_devices_for_tensor_parallel(
    memory_required_per_device: int,
    num_devices: int,
    prefer_contiguous: bool = True
) -> Optional[List[int]]:
    """
    Select devices for tensor parallel model.

    Args:
        memory_required_per_device: Memory required on each device in bytes
        num_devices: Number of devices needed
        prefer_contiguous: Prefer contiguous device IDs for better communication

    Returns:
        List of device IDs, or None if not enough devices available
    """
    if not is_gpu_available():
        return None

    total_devices = get_device_count()
    if total_devices < num_devices:
        return None

    # Find devices with enough memory
    suitable_devices = []
    for device_id in range(total_devices):
        if get_free_memory(device_id) >= memory_required_per_device:
            suitable_devices.append(device_id)

    if len(suitable_devices) < num_devices:
        return None

    # If prefer contiguous, try to find contiguous sequence
    if prefer_contiguous:
        for i in range(len(suitable_devices) - num_devices + 1):
            devices = suitable_devices[i:i + num_devices]
            # Check if contiguous
            if all(devices[j+1] - devices[j] == 1 for j in range(len(devices) - 1)):
                return devices

    # Otherwise, return first num_devices
    return suitable_devices[:num_devices]


def calculate_optimal_tensor_parallel_size(
    model_memory: int,
    available_devices: Optional[List[int]] = None
) -> int:
    """
    Calculate optimal tensor parallel size for a model.

    Args:
        model_memory: Total model memory required in bytes
        available_devices: List of available device IDs (None = all devices)

    Returns:
        Optimal number of devices for tensor parallelism
    """
    if not is_gpu_available():
        return 1

    if available_devices is None:
        available_devices = list(range(get_device_count()))

    if not available_devices:
        return 1

    # Get minimum free memory across available devices
    min_free_memory = min(
        get_free_memory(device_id) for device_id in available_devices
    )

    # Calculate how many devices needed
    if min_free_memory >= model_memory:
        return 1  # Fits on single device

    # Calculate minimum devices needed
    devices_needed = math.ceil(model_memory / min_free_memory)

    # Ensure power of 2 for optimal tensor parallelism
    tensor_parallel_size = 2 ** math.ceil(math.log2(devices_needed))

    # Cap at number of available devices
    tensor_parallel_size = min(tensor_parallel_size, len(available_devices))

    return tensor_parallel_size


def clear_cache(device_id: Optional[int] = None) -> None:
    """
    Clear GPU memory cache.

    Args:
        device_id: Specific device to clear, or None for all devices
    """
    if not is_gpu_available():
        return

    if device_id is not None:
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        logger.debug(f"Cleared cache for GPU {device_id}")
    else:
        for device_id in range(get_device_count()):
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
        logger.debug("Cleared cache for all GPUs")


def synchronize(device_id: Optional[int] = None) -> None:
    """
    Synchronize GPU operations.

    Args:
        device_id: Specific device to synchronize, or None for all devices
    """
    if not is_gpu_available():
        return

    if device_id is not None:
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize(device_id)
    else:
        torch.cuda.synchronize()


def reset_peak_memory_stats(device_id: Optional[int] = None) -> None:
    """
    Reset peak memory statistics.

    Args:
        device_id: Specific device, or None for all devices
    """
    if not is_gpu_available():
        return

    if device_id is not None:
        torch.cuda.reset_peak_memory_stats(device_id)
    else:
        for device_id in range(get_device_count()):
            torch.cuda.reset_peak_memory_stats(device_id)


def get_memory_summary(device_id: int) -> Dict[str, Union[int, str]]:
    """
    Get comprehensive memory summary for a device.

    Args:
        device_id: GPU device ID

    Returns:
        Dictionary with memory statistics
    """
    if not is_gpu_available():
        return {
            'available': False,
            'error': 'No GPU available'
        }

    if device_id < 0 or device_id >= get_device_count():
        return {
            'available': False,
            'error': f'Invalid device ID: {device_id}'
        }

    torch.cuda.set_device(device_id)

    total = torch.cuda.get_device_properties(device_id).total_memory
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    free = total - reserved

    return {
        'available': True,
        'device_id': device_id,
        'device_name': get_device_name(device_id),
        'total_memory': total,
        'total_memory_formatted': format_memory_size(total),
        'allocated_memory': allocated,
        'allocated_memory_formatted': format_memory_size(allocated),
        'reserved_memory': reserved,
        'reserved_memory_formatted': format_memory_size(reserved),
        'free_memory': free,
        'free_memory_formatted': format_memory_size(free),
        'utilization': allocated / total if total > 0 else 0.0
    }


def check_compute_capability(
    device_id: int,
    min_major: int = 7,
    min_minor: int = 0
) -> bool:
    """
    Check if device meets minimum compute capability.

    Args:
        device_id: GPU device ID
        min_major: Minimum major version
        min_minor: Minimum minor version

    Returns:
        True if device meets requirements, False otherwise
    """
    if not is_gpu_available():
        return False

    try:
        major, minor = get_device_capability(device_id)
        return (major > min_major) or (major == min_major and minor >= min_minor)
    except Exception as e:
        logger.error(f"Error checking compute capability: {e}")
        return False


def supports_bfloat16(device_id: int) -> bool:
    """
    Check if device supports bfloat16.

    Args:
        device_id: GPU device ID

    Returns:
        True if bfloat16 is supported, False otherwise
    """
    # bfloat16 requires compute capability 8.0+
    return check_compute_capability(device_id, min_major=8, min_minor=0)


def supports_flash_attention(device_id: int) -> bool:
    """
    Check if device supports Flash Attention.

    Args:
        device_id: GPU device ID

    Returns:
        True if Flash Attention is supported, False otherwise
    """
    # Flash Attention requires compute capability 7.5+ (Turing or newer)
    return check_compute_capability(device_id, min_major=7, min_minor=5)


def supports_tensor_cores(device_id: int) -> bool:
    """
    Check if device has Tensor Cores.

    Args:
        device_id: GPU device ID

    Returns:
        True if device has Tensor Cores, False otherwise
    """
    # Tensor Cores available from compute capability 7.0+ (Volta or newer)
    return check_compute_capability(device_id, min_major=7, min_minor=0)


def get_optimal_dtype(device_id: int, prefer_bfloat16: bool = True) -> str:
    """
    Get optimal dtype for a device.

    Args:
        device_id: GPU device ID
        prefer_bfloat16: Prefer bfloat16 over float16 when available

    Returns:
        Optimal dtype string ('bfloat16', 'float16', or 'float32')
    """
    if not is_gpu_available():
        return 'float32'

    try:
        if prefer_bfloat16 and supports_bfloat16(device_id):
            return 'bfloat16'
        elif supports_tensor_cores(device_id):
            return 'float16'
        else:
            return 'float32'
    except Exception as e:
        logger.error(f"Error determining optimal dtype: {e}")
        return 'float32'


def get_device_info_string(device_id: int) -> str:
    """
    Get formatted device information string.

    Args:
        device_id: GPU device ID

    Returns:
        Formatted string with device information
    """
    if not is_gpu_available():
        return "No GPU available"

    try:
        name = get_device_name(device_id)
        total_mem = get_total_memory(device_id)
        free_mem = get_free_memory(device_id)
        major, minor = get_device_capability(device_id)

        info = (
            f"GPU {device_id}: {name} | "
            f"Compute {major}.{minor} | "
            f"Memory: {format_memory_size(free_mem)} / {format_memory_size(total_mem)} free"
        )

        features = []
        if supports_tensor_cores(device_id):
            features.append("Tensor Cores")
        if supports_bfloat16(device_id):
            features.append("BF16")
        if supports_flash_attention(device_id):
            features.append("Flash Attention")

        if features:
            info += f" | Features: {', '.join(features)}"

        return info

    except Exception as e:
        return f"Error getting device info: {e}"


def print_device_info(device_id: Optional[int] = None) -> None:
    """
    Print device information to console.

    Args:
        device_id: Specific device ID, or None for all devices
    """
    if not is_gpu_available():
        print("No GPU devices available")
        return

    if device_id is not None:
        print(get_device_info_string(device_id))
    else:
        print(f"Found {get_device_count()} GPU device(s):")
        for i in range(get_device_count()):
            print(f"  {get_device_info_string(i)}")


def validate_device_id(device_id: int) -> bool:
    """
    Validate that device ID is valid.

    Args:
        device_id: GPU device ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not is_gpu_available():
        return False

    return 0 <= device_id < get_device_count()


def get_device_or_cpu(device_id: Optional[int] = None) -> str:
    """
    Get device string for PyTorch, with CPU fallback.

    Args:
        device_id: GPU device ID, or None for auto-select

    Returns:
        Device string (e.g., 'cuda:0', 'cpu')
    """
    if not is_gpu_available():
        return 'cpu'

    if device_id is None:
        # Auto-select device with most free memory
        device_id = select_best_device(memory_required=0, prefer_empty=True)
        if device_id is None:
            return 'cpu'

    if validate_device_id(device_id):
        return f'cuda:{device_id}'
    else:
        logger.warning(f"Invalid device ID {device_id}, falling back to CPU")
        return 'cpu'


def estimate_tokens_per_second(
    model_parameters: int,
    batch_size: int,
    device_ids: List[int],
    dtype: str = "float16"
) -> float:
    """
    Rough estimation of tokens per second for inference.

    Args:
        model_parameters: Number of model parameters
        batch_size: Batch size
        device_ids: List of GPU device IDs
        dtype: Model dtype

    Returns:
        Estimated tokens per second
    """
    if not is_gpu_available() or not device_ids:
        # CPU inference is much slower
        return model_parameters / 1e9  # Very rough estimate

    # Rough FLOPS estimates by compute capability
    device_id = device_ids[0]
    major, minor = get_device_capability(device_id)

    # Rough TFLOPS estimates for different architectures
    # These are very approximate and for FP16/BF16
    if major >= 9:  # Ada/Hopper
        tflops = 300
    elif major >= 8:  # Ampere
        tflops = 150
    elif major >= 7:  # Turing/Volta
        tflops = 100
    else:
        tflops = 50

    # Adjust for dtype
    if dtype == 'float32':
        tflops /= 2
    elif '8bit' in dtype.lower():
        tflops *= 2
    elif '4bit' in dtype.lower():
        tflops *= 4

    # Multiply by number of GPUs (assuming perfect scaling, which is optimistic)
    tflops *= len(device_ids)

    # Very rough formula: tokens/sec ≈ TFLOPS * 1e12 / (model_params * 2 * batch_size)
    # The factor of 2 accounts for both forward and backward pass (though we only do forward)
    tokens_per_second = (tflops * 1e12) / (model_parameters * 2)

    return tokens_per_second


# Convenience functions for common operations

def gb_to_bytes(gb: float) -> int:
    """Convert GB to bytes"""
    return int(gb * GB)


def bytes_to_gb(bytes_val: int) -> float:
    """Convert bytes to GB"""
    return bytes_val / GB


def mb_to_bytes(mb: float) -> int:
    """Convert MB to bytes"""
    return int(mb * MB)


def bytes_to_mb(bytes_val: int) -> float:
    """Convert bytes to MB"""
    return bytes_val / MB
