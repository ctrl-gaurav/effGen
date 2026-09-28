"""
HuggingFace Transformers engine implementation as fallback.

This module provides a fallback inference engine using HuggingFace Transformers
with features including:
- Automatic quantization with bitsandbytes
- Flash Attention support
- Multi-GPU device mapping
- Memory optimization techniques
- CPU fallback support
"""

from __future__ import annotations

# A few names below are no longer read by this module itself. They stay bound
# here so an existing ``from effgen.models.transformers_engine import ...``
# keeps resolving them.
import logging
import os  # noqa: F401
import threading
import warnings
from collections.abc import Iterator  # noqa: F401
from contextlib import contextmanager  # noqa: F401
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - only on an install without torch
    from effgen.models._adapter_utils import missing_torch_error

    raise missing_torch_error("transformers") from exc

from transformers import (
    AutoModelForCausalLM,  # noqa: F401
    AutoTokenizer,  # noqa: F401
    BitsAndBytesConfig,  # noqa: F401
)
from transformers import (
    GenerationConfig as HFGenerationConfig,  # noqa: F401
)
from transformers import (
    set_seed as _hf_set_seed,  # noqa: F401
)

from effgen.models._adapter_utils import (
    attach_error_context,  # noqa: F401
    chat_template_renders_tools,
    normalize_finish_reason,  # noqa: F401
    not_loaded_error,  # noqa: F401
    provider_runtime_error,  # noqa: F401
)
from effgen.models.base import (
    BatchModel,
    GenerationConfig,  # noqa: F401
    GenerationResult,  # noqa: F401
    ModelType,
    TokenCount,  # noqa: F401
)
from effgen.models.errors import ModelNotFoundError  # noqa: F401
from effgen.models.transformers_engine_generation import TransformersGenerationMixin
from effgen.models.transformers_engine_loading import TransformersLoadingMixin
from effgen.models.transformers_engine_placement import TransformersPlacementMixin
from effgen.models.transformers_engine_sampling import TransformersSamplingMixin
from effgen.models.transformers_engine_streaming import TransformersStreamingMixin
from effgen.models.transformers_engine_support import (  # noqa: F401  re-exported
    _COMMON_END_MARKERS,
    _TOOL_CALL_DELIMITERS,
    GPUPlacementError,
    ModelNotCachedError,
    _is_cache_miss_error,
    _list_cached_model_repos,
    _offline_mode_active,
    _quiet_model_load,
    _reraise_if_classified,
    _strip_special_tokens_keep_tool_calls,
)

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='accelerate')
warnings.filterwarnings('ignore', message='.*Some parameters are on the meta device.*')

logger = logging.getLogger(__name__)


class TransformersEngine(
    TransformersPlacementMixin,
    TransformersSamplingMixin,
    TransformersLoadingMixin,
    TransformersGenerationMixin,
    TransformersStreamingMixin,
    BatchModel,
):
    """
    HuggingFace Transformers-based model engine.

    This engine serves as a fallback when vLLM is unavailable or incompatible.
    It supports a wider range of models and edge cases.

    Features:
    - Automatic quantization (4-bit, 8-bit with bitsandbytes)
    - Flash Attention 2 support
    - Auto device mapping for multi-GPU
    - Memory optimization (gradient checkpointing, mixed precision)
    - CPU fallback
    - Streaming generation

    Attributes:
        model_name: HuggingFace model identifier or path
        quantization_bits: Quantization level (None, 4, 8)
        device_map: Device mapping strategy ('auto', 'balanced', or custom)
        use_flash_attention: Whether to use Flash Attention 2
        torch_dtype: Torch data type for model weights
    """

    # BaseModel.__init_subclass__ instruments the generation entry points by
    # reading this class's own __dict__: a pre-call budget check, call latency
    # and cost stamped on the result, and credential redaction on a load
    # failure. Binding the names here keeps that wrapping while the bodies
    # live beside the rest of their concern.
    load = TransformersLoadingMixin.load
    generate = TransformersGenerationMixin.generate
    generate_batch = TransformersGenerationMixin.generate_batch
    generate_stream = TransformersStreamingMixin.generate_stream

    def __init__(
        self,
        model_name: str,
        quantization_bits: int | None = None,
        device_map: str = "auto",
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
        max_memory: dict[int, str] | None = None,
        offload_folder: str | None = None,
        require_gpu: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize Transformers engine.

        Args:
            model_name: HuggingFace model ID or local path
            quantization_bits: Quantization level (None, 4, or 8)
            device_map: Device mapping ('auto', 'balanced', 'sequential', or dict)
            use_flash_attention: Enable Flash Attention 2 if available
            torch_dtype: Data type (None for auto, or torch.float16, torch.bfloat16)
            trust_remote_code: Whether to trust remote code
            low_cpu_mem_usage: Use low CPU memory during loading
            max_memory: Maximum memory per device (e.g., {0: "20GB", "cpu": "30GB"})
            offload_folder: Folder for offloading weights
            require_gpu: Fail with a clear error if the model cannot be placed
                entirely on the GPU (rather than falling back to CPU offload).
                Use this on hardware where a silent CPU fallback is unacceptable.
            **kwargs: Additional model loading arguments
        """
        super().__init__(
            model_name=model_name,
            model_type=ModelType.TRANSFORMERS
        )

        self.quantization_bits = quantization_bits
        self.device_map = device_map
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        self.require_gpu = require_gpu

        # Extra arguments for the tokenizer's chat template, e.g.
        # ``chat_template_kwargs={"enable_thinking": False}`` to turn off a
        # reasoning model's hidden chain. The switch is a property of the
        # template rather than of generation, so it cannot be reached through
        # GenerationConfig; passed here it reaches apply_chat_template and
        # nothing else. Held out of additional_kwargs below, because anything
        # left there is forwarded to ``from_pretrained`` and the model class
        # rejects a keyword it does not know.
        self.chat_template_kwargs: dict[str, Any] = dict(
            kwargs.get("chat_template_kwargs") or {}
        )

        # Filter out parameters that shouldn't be passed to model loading
        # These are vLLM-specific or other incompatible parameters
        self.additional_kwargs = {k: v for k, v in kwargs.items()
                                  if k not in ['quantization', 'engine', 'backend', 'device',
                                               'require_gpu',
                                               'use_tqdm', 'tensor_parallel_size',
                                               'apply_chat_template', 'system_prompt',
                                               'chat_template_kwargs',
                                               'gpu_memory_utilization', 'max_num_seqs',
                                               'max_num_batched_tokens']}

        self.model = None
        self.tokenizer = None
        self.device = None
        # Cached chat-template tool-rendering probe, paired with the tokenizer it
        # was measured on so a reload re-measures instead of reusing a stale
        # answer. The agent loop asks once per iteration; rendering the template
        # twice per turn is pure waste.
        self._tool_template_probe: tuple[Any, bool] | None = None
        # HuggingFace "fast" (Rust) tokenizers are NOT thread-safe: two threads
        # encoding/decoding on the same tokenizer raise "Already borrowed". A
        # single local model can't parallelize across threads on one GPU anyway,
        # so serialize every tokenizer-touching call on this engine. Reentrant so
        # a method that internally counts tokens while holding the lock is safe.
        self._tokenizer_lock = threading.RLock()
        # Set after a CUDA device-side assert; reloads weights on GPU 0 only.
        self._pin_device_map_for_cuda = False
        self._retrying_after_cuda_assert = False
        # Whether the loaded weights have already answered the device-map probe.
        # The probe is a full forward pass, so it runs once per set of weights
        # rather than before every call.
        self._device_map_probe_passed = False

    def get_context_length(self) -> int:
        """
        Get maximum context length.

        Returns:
            int: Maximum context length in tokens
        """
        if self._context_length is not None:
            return self._context_length
        return 2048  # Default fallback

    def get_max_batch_size(self) -> int:
        """
        Get maximum batch size.

        Returns:
            int: Maximum batch size (conservative estimate)
        """
        # Conservative batch size based on available VRAM
        if self.device == "cuda":
            return 8
        else:
            return 1  # CPU is slow, use minimal batch size

    def capability_key(self) -> str | None:
        """Weights, their revision and the chat template: the probe's cache key.

        ``None`` until the model is loaded, so a model that has not loaded is
        not probed.
        """
        from .capability_probe import local_capability_key

        return local_capability_key(self)

    def supports_tool_calling(self) -> bool:
        """Check if the model supports native tool calling via chat template.

        True when the tokenizer's chat template actually **renders** tool
        definitions into the prompt, which Qwen2.5 and Llama 3.x do. Templates
        that accept a ``tools`` argument and discard it (gemma-2, Phi-3.5) report
        False: the model never sees the tools, so the ReAct text protocol is the
        only path that reaches one.
        """
        if not self._is_loaded or self.tokenizer is None:
            return False
        tokenizer = self.tokenizer
        cached = self._tool_template_probe
        if cached is not None and cached[0] is tokenizer:
            return cached[1]
        supported = chat_template_renders_tools(tokenizer)
        self._tool_template_probe = (tokenizer, supported)
        return supported

    def tool_call_support(self) -> str:
        """``"template"`` when the chat template renders tools, else ``"none"``.

        A local chat template only writes the definitions into the prompt; there
        is no provider-side tool-calling layer, so nothing obliges the model to
        answer with a call.
        """
        return "template" if self.supports_tool_calling() else "none"
