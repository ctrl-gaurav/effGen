"""
Hardware platform detection for effGen framework.

Provides detection functions for Apple Silicon, CUDA, and MLX availability.
All imports are lazy to avoid hard dependencies on optional packages.
"""

from __future__ import annotations

import functools
import logging
import platform
import subprocess
import sys
from enum import Enum

logger = logging.getLogger(__name__)


class HardwarePlatform(Enum):
    """Detected hardware platform."""
    APPLE_SILICON = "apple_silicon"
    CUDA = "cuda"
    CPU_ONLY = "cpu"


@functools.lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M-series Mac)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


@functools.lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """Check if CUDA GPU is available. Lazy-imports torch."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def is_mlx_available() -> bool:
    """Check if MLX framework is installed and usable."""
    if not is_apple_silicon():
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def is_mlx_vlm_available() -> bool:
    """Check if MLX-VLM (vision-language) is installed."""
    if not is_mlx_available():
        return False
    try:
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        return False


def get_unified_memory_gb() -> float:
    """Get total unified memory in GB (macOS Apple Silicon).

    On macOS, uses sysctl hw.memsize. Returns 0.0 on non-macOS.
    """
    if sys.platform != "darwin":
        return 0.0
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (subprocess.TimeoutExpired, ValueError, OSError) as e:
        logger.debug(f"Failed to get unified memory size: {e}")
    return 0.0


def get_best_local_backend() -> str:
    """Determine the best local inference backend for the current hardware.

    Returns:
        str: One of "mlx", "vllm", "transformers"
    """
    # Apple Silicon with MLX available
    if is_apple_silicon() and is_mlx_available():
        if not is_cuda_available():
            logger.info("Apple Silicon detected with MLX available, recommending MLX backend")
            return "mlx"

    # CUDA available - check for vLLM
    if is_cuda_available():
        try:
            import vllm  # noqa: F401
            logger.info("CUDA detected with vLLM available, recommending vLLM backend")
            return "vllm"
        except ImportError:
            logger.info("CUDA detected without vLLM, recommending Transformers backend")
            return "transformers"

    # CPU-only fallback
    logger.info("No GPU acceleration detected, recommending Transformers backend (CPU)")
    return "transformers"


def detect_platform() -> HardwarePlatform:
    """Detect the current hardware platform.

    Returns:
        HardwarePlatform enum value
    """
    if is_apple_silicon() and is_mlx_available():
        return HardwarePlatform.APPLE_SILICON
    if is_cuda_available():
        return HardwarePlatform.CUDA
    return HardwarePlatform.CPU_ONLY
