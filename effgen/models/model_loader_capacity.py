"""VRAM inspection and automatic quantization/parallelism choices.

Reads free VRAM and picks a quantization scheme and tensor-parallel size for a
model. Mixed into :class:`~effgen.models.model_loader.ModelLoader`; not usable
on its own.
"""

from __future__ import annotations

import logging

from ._vram import cuda_device_visible, free_vram_gb

logger = logging.getLogger("effgen.models.model_loader")


class ModelLoaderCapacityMixin:
    """VRAM inspection and automatic quantization/parallelism selection."""

    @staticmethod
    def _free_vram_gb() -> float:
        """Free VRAM (GB) across the visible CUDA devices.

        Uses the currently-free memory, not the card's total capacity, so
        quantization and offload decisions reflect what is actually available
        after any other tenants on the GPU.

        Returns:
            Free memory in gibibytes, summed over the visible devices.
        """
        return free_vram_gb()

    def _auto_select_quantization(self, model_name: str) -> str | None:
        """
        Automatically select quantization based on free VRAM.

        Args:
            model_name: Model identifier

        Returns:
            Quantization method or None
        """
        # Asked without starting the CUDA driver, so a vLLM engine core forked
        # later can still reach the GPU.
        if not cuda_device_visible():
            return None

        # Base the decision on free VRAM, not total capacity, so a busy GPU
        # doesn't silently overcommit and offload to CPU.
        available_vram = self._free_vram_gb()
        logger.info(f"Free VRAM: {available_vram:.2f} GB")

        # Estimate model size (rough heuristics from the id).
        if "70b" in model_name.lower() or "65b" in model_name.lower():
            # Large models need quantization
            if available_vram < 80:
                logger.info("Using AWQ quantization for large model")
                return "awq"
        elif "13b" in model_name.lower() or "7b" in model_name.lower():
            # Medium models might benefit from quantization
            if available_vram < 24:
                logger.info("Using AWQ quantization for medium model")
                return "awq"

        # No quantization needed
        logger.info("Sufficient free VRAM, no quantization needed")
        return None

    def _auto_select_quantization_bits(self) -> int | None:
        """
        Automatically select quantization bits for Transformers.

        Returns:
            Quantization bits (4, 8) or None
        """
        # Only the Transformers engine asks this, and it places the model with
        # torch.cuda.is_available() right after, so the two must agree: a host
        # whose driver lists a GPU that torch cannot use runs on CPU, where a
        # bit width chosen from the GPU's free memory does not apply.
        import torch

        if not torch.cuda.is_available():
            return None

        # Base the decision on free VRAM, not total capacity.
        available_vram = self._free_vram_gb()

        if available_vram < 16:
            logger.info("Low free VRAM detected, using 4-bit quantization")
            return 4
        elif available_vram < 32:
            logger.info("Medium free VRAM detected, using 8-bit quantization")
            return 8

        logger.info("Sufficient free VRAM, no quantization")
        return None

    def _auto_select_tensor_parallel(self, model_name: str) -> int:
        """
        Automatically select tensor parallel size based on available GPUs and model size.

        For tensor parallelism to work, the number of attention heads must be divisible
        by the tensor parallel size. Small models often have fewer attention heads,
        so we need to be conservative.

        Args:
            model_name: Model identifier to help determine appropriate parallelism

        Returns:
            Number of GPUs to use for tensor parallelism
        """
        import torch

        # Asked without starting the CUDA driver, so a vLLM engine core forked
        # later can still reach the GPU.
        if not cuda_device_visible():
            return 1

        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")

        # For small models (indicated by size in name), use fewer GPUs
        # Small models have fewer attention heads which limits parallelism options
        model_lower = model_name.lower()

        # Check for small model indicators
        # Note: We check for specific sizes like "1.7b", "3b-" etc. to handle various naming conventions
        small_model_indicators = [
            "0.5b", "1b", "1.5b", "1.7b", "2b", "3b", "4b",
            "-0.5b", "-1b", "-1.5b", "-1.7b", "-2b", "-3b", "-4b",
            "_0.5b", "_1b", "_1.5b", "_1.7b", "_2b", "_3b", "_4b",
        ]
        if any(size in model_lower for size in small_model_indicators):
            # Small models: use at most 1 GPU (attention heads typically 12-16)
            # 12 heads divisible by: 1, 2, 3, 4, 6, 12
            # 16 heads divisible by: 1, 2, 4, 8, 16
            tp_size = 1  # Conservative: use 1 GPU for small models
            logger.info(f"Small model detected, using tensor_parallel_size={tp_size}")
            return tp_size
        elif any(size in model_lower for size in ["7b", "8b"]):
            # Medium models: typically 32 heads, can use up to 4 GPUs
            # 32 heads divisible by: 1, 2, 4, 8, 16, 32
            tp_size = min(num_gpus, 4)
            logger.info(f"Medium model detected, using tensor_parallel_size={tp_size}")
            return tp_size
        elif any(size in model_lower for size in ["13b", "14b"]):
            # Larger models: typically 40 heads
            # 40 heads divisible by: 1, 2, 4, 5, 8, 10, 20, 40
            tp_size = min(num_gpus, 4)
            logger.info(f"13B+ model detected, using tensor_parallel_size={tp_size}")
            return tp_size
        elif any(size in model_lower for size in ["30b", "33b", "34b", "65b", "70b"]):
            # Large models: can benefit from more parallelism
            # 64/80 heads divisible by: 1, 2, 4, 8, 16, etc.
            tp_size = min(num_gpus, 8)
            logger.info(f"Large model detected, using tensor_parallel_size={tp_size}")
            return tp_size

        # Default: conservative approach, use 1 GPU unless we know the model
        logger.info("Unknown model size, using tensor_parallel_size=1 for safety")
        return 1
