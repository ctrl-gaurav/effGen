from effgen.hardware.platform import (
    is_apple_silicon,
    is_cuda_available,
    is_mlx_available,
    is_mlx_vlm_available,
    get_unified_memory_gb,
    get_best_local_backend,
    HardwarePlatform,
)

__all__ = [
    "is_apple_silicon",
    "is_cuda_available",
    "is_mlx_available",
    "is_mlx_vlm_available",
    "get_unified_memory_gb",
    "get_best_local_backend",
    "HardwarePlatform",
]
