"""How much GPU memory is actually free right now.

Both the model loader and the transformers engine size their placement
decisions from this, and they have to agree: if one reads the card's capacity
and the other reads what is unused, the same machine gets two different answers
about whether a model fits.
"""

from __future__ import annotations


def free_vram_gb() -> float:
    """Return free VRAM in GB across the visible CUDA devices, or 0.0 if none.

    Reports currently-free memory rather than total capacity, so a decision
    made from it accounts for whatever else is already resident on the card.

    The reading comes from NVML, the driver's own view (the one ``nvidia-smi``
    prints), mapped through ``CUDA_VISIBLE_DEVICES`` the way torch maps it.
    Asking CUDA instead (``torch.cuda.mem_get_info``) creates a CUDA context on
    every device it is asked about, which holds memory on each of them for the
    life of the process and leaves a forked vLLM engine core unable to use the
    GPU. CUDA is asked only for a device NVML cannot describe.

    Returns:
        Free memory in gibibytes, summed over the visible devices.
    """
    # Imported here rather than at module scope: torch is heavy, and a caller
    # that never touches a GPU should not pay for it at import time.
    import torch

    if not cuda_device_visible():
        return 0.0
    free_bytes = 0
    for index in range(torch.cuda.device_count()):
        free = _nvml_free_bytes(index)
        if free is None:
            try:
                free = int(torch.cuda.mem_get_info(index)[0])
            except Exception:  # noqa: BLE001 - one unreadable device is not fatal
                free = 0
        free_bytes += free
    return free_bytes / (1024**3)


def _nvml_free_bytes(index: int) -> int | None:
    """Free bytes on visible device *index* from NVML, or None when NVML cannot say."""
    import torch

    handler = getattr(torch.cuda, "_get_pynvml_handler", None)
    if handler is None:
        return None
    try:
        import pynvml

        return int(pynvml.nvmlDeviceGetMemoryInfo(handler(index)).free)
    except Exception:  # noqa: BLE001 - no NVML reading; the caller asks CUDA
        return None


def cuda_device_visible() -> bool:
    """Whether a CUDA device is visible, asked without starting the CUDA driver.

    vLLM starts its engine core in a forked process, and a parent that has
    already started the CUDA driver leaves that child unable to use the GPU at
    all ("Cannot re-initialize CUDA in forked subprocess").
    ``torch.cuda.is_available()`` starts the driver; ``torch.cuda.device_count()``
    reads the device list through NVML where it can and does not, so this is the
    question to ask before handing a model to vLLM in this process.

    Returns:
        True when at least one CUDA device is visible to this process.
    """
    import torch

    try:
        return bool(torch.cuda.device_count() > 0)
    except Exception:  # noqa: BLE001 - no readable device list means no device
        return False
