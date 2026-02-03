"""
Smart model loader with automatic detection and fallback.

This module provides intelligent model loading with:
- Automatic model type detection (HuggingFace vs API)
- Transformers-first with vLLM as optional production backend
- GPU allocation and VRAM management
- Quantization decision based on available memory
- Model validation and health checks
"""

import logging
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch

from effgen.models.base import BaseModel, ModelType
from effgen.models.vllm_engine import VLLMEngine
from effgen.models.transformers_engine import TransformersEngine
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.anthropic_adapter import AnthropicAdapter
from effgen.models.gemini_adapter import GeminiAdapter

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Smart model loader with automatic detection and configuration.

    This class handles:
    1. Model type detection (local, HuggingFace, or API)
    2. Engine selection (Transformers default, vLLM optional, or API adapter)
    3. GPU allocation and memory management
    4. Automatic quantization decisions
    5. Fallback strategies
    6. Model validation

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load_model("meta-llama/Llama-2-7b-hf")
        >>> # Uses Transformers by default, can specify vLLM with engine='vllm'

        >>> model = loader.load_model("gpt-4")
        >>> # Automatically uses OpenAI adapter
    """

    # API model prefixes for automatic detection
    OPENAI_MODELS = [
        "gpt-3.5", "gpt-4", "text-davinci", "text-curie",
        "text-babbage", "text-ada"
    ]

    ANTHROPIC_MODELS = [
        "claude-3", "claude-2", "claude-instant"
    ]

    GEMINI_MODELS = [
        "gemini-pro", "gemini-ultra", "gemini-flash", "gemini-1.5"
    ]

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_device: str = "auto",
        force_engine: Optional[str] = None,
    ):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models
            default_device: Default device allocation ('auto', 'cuda', 'cpu')
            force_engine: Force specific engine ('vllm', 'transformers', or None for auto)
        """
        # Expand ~ to full path and use environment variable if set
        default_cache = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
        self.cache_dir = os.path.expanduser(cache_dir) if cache_dir else default_cache
        self.default_device = default_device
        self.force_engine = force_engine

        self.loaded_models: Dict[str, BaseModel] = {}

    def load_model(
        self,
        model_name: str,
        engine_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseModel:
        """
        Load a model with automatic detection and configuration.

        Args:
            model_name: Model identifier (HuggingFace ID, local path, or API model name)
            engine_config: Optional engine-specific configuration
            **kwargs: Additional model parameters

        Returns:
            Loaded model instance ready for inference

        Raises:
            ValueError: If model_name is invalid or unsupported
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading model: {model_name}")

        # Check if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded, returning cached instance")
            return self.loaded_models[model_name]

        # Detect model type
        model_type = self._detect_model_type(model_name)
        logger.info(f"Detected model type: {model_type.value}")

        # Load based on type
        if model_type == ModelType.OPENAI:
            model = self._load_openai_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.ANTHROPIC:
            model = self._load_anthropic_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.GEMINI:
            model = self._load_gemini_model(model_name, engine_config, **kwargs)
        else:
            # HuggingFace model - use Transformers by default, vLLM optional
            model = self._load_huggingface_model(model_name, engine_config, **kwargs)

        # Load the model
        model.load()

        # Validate
        self._validate_model(model)

        # Cache the loaded model
        self.loaded_models[model_name] = model

        logger.info(f"Model '{model_name}' loaded successfully")
        return model

    def _detect_model_type(self, model_name: str) -> ModelType:
        """
        Detect the type of model based on its name.

        Args:
            model_name: Model identifier

        Returns:
            ModelType enum value
        """
        model_lower = model_name.lower()

        # Check API models
        for prefix in self.OPENAI_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.OPENAI

        for prefix in self.ANTHROPIC_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.ANTHROPIC

        for prefix in self.GEMINI_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.GEMINI

        # Check if it's a local path
        if os.path.exists(model_name):
            logger.info(f"Detected local model path: {model_name}")
            return ModelType.TRANSFORMERS  # Default to Transformers for local models

        # Assume HuggingFace model ID
        return ModelType.TRANSFORMERS  # Default to Transformers for HuggingFace models

    def _load_openai_model(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> OpenAIAdapter:
        """
        Load OpenAI model.

        Args:
            model_name: OpenAI model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            OpenAIAdapter instance
        """
        logger.info(f"Loading OpenAI model: {model_name}")

        params = config or {}
        params.update(kwargs)

        return OpenAIAdapter(model_name=model_name, **params)

    def _load_anthropic_model(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AnthropicAdapter:
        """
        Load Anthropic model.

        Args:
            model_name: Anthropic model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            AnthropicAdapter instance
        """
        logger.info(f"Loading Anthropic model: {model_name}")

        params = config or {}
        params.update(kwargs)

        return AnthropicAdapter(model_name=model_name, **params)

    def _load_gemini_model(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> GeminiAdapter:
        """
        Load Gemini model.

        Args:
            model_name: Gemini model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            GeminiAdapter instance
        """
        logger.info(f"Loading Gemini model: {model_name}")

        params = config or {}
        params.update(kwargs)

        return GeminiAdapter(model_name=model_name, **params)

    def _load_huggingface_model(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[VLLMEngine, TransformersEngine]:
        """
        Load HuggingFace model with Transformers-first strategy.

        Args:
            model_name: HuggingFace model ID or local path
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            TransformersEngine or VLLMEngine instance
        """
        params = config or {}
        params.update(kwargs)

        # Check if vLLM engine is explicitly requested
        if self.force_engine == "vllm":
            logger.info("Using vLLM engine (explicitly requested)")
            try:
                return self._load_with_vllm(model_name, params)
            except Exception as e:
                logger.warning(f"vLLM loading failed: {e}")
                logger.info("Falling back to Transformers...")
                return self._load_with_transformers(model_name, params)

        # Default to Transformers (more compatible, easier setup)
        logger.info("Using Transformers engine (default)")
        return self._load_with_transformers(model_name, params)

    def _load_with_vllm(
        self,
        model_name: str,
        params: Dict[str, Any]
    ) -> VLLMEngine:
        """
        Load model with vLLM.

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            VLLMEngine instance

        Raises:
            RuntimeError: If vLLM is unavailable or loading fails
        """
        logger.info(f"Attempting to load with vLLM: {model_name}")

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, vLLM requires GPU")

        # Determine quantization if not specified
        if "quantization" not in params:
            params["quantization"] = self._auto_select_quantization(model_name)

        # Determine tensor parallel size if not specified
        if "tensor_parallel_size" not in params:
            params["tensor_parallel_size"] = self._auto_select_tensor_parallel(model_name)

        # Only set download directory if explicitly specified (let vLLM use its default otherwise)
        # This avoids potential issues with path handling
        if "download_dir" not in params and self.cache_dir != os.path.expanduser("~/.cache/huggingface"):
            params["download_dir"] = self.cache_dir

        return VLLMEngine(model_name=model_name, **params)

    def _load_with_transformers(
        self,
        model_name: str,
        params: Dict[str, Any]
    ) -> TransformersEngine:
        """
        Load model with Transformers.

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            TransformersEngine instance
        """
        logger.info(f"Loading with Transformers: {model_name}")

        # Determine quantization if not specified
        if "quantization_bits" not in params:
            params["quantization_bits"] = self._auto_select_quantization_bits()

        # Set device map
        if "device_map" not in params:
            params["device_map"] = "auto" if torch.cuda.is_available() else None

        return TransformersEngine(model_name=model_name, **params)

    def _auto_select_quantization(self, model_name: str) -> Optional[str]:
        """
        Automatically select quantization based on available VRAM.

        Args:
            model_name: Model identifier

        Returns:
            Quantization method or None
        """
        if not torch.cuda.is_available():
            return None

        # Get available VRAM (GB)
        available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Available VRAM: {available_vram:.2f} GB")

        # Estimate model size (rough heuristics)
        # This is a simplified approach - in production, you'd want a more sophisticated method
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
        logger.info("Sufficient VRAM available, no quantization needed")
        return None

    def _auto_select_quantization_bits(self) -> Optional[int]:
        """
        Automatically select quantization bits for Transformers.

        Returns:
            Quantization bits (4, 8) or None
        """
        if not torch.cuda.is_available():
            return None

        # Get available VRAM
        available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if available_vram < 16:
            logger.info("Low VRAM detected, using 4-bit quantization")
            return 4
        elif available_vram < 32:
            logger.info("Medium VRAM detected, using 8-bit quantization")
            return 8

        logger.info("Sufficient VRAM available, no quantization")
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
        if not torch.cuda.is_available():
            return 1

        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")

        # For small models (indicated by size in name), use fewer GPUs
        # Small models have fewer attention heads which limits parallelism options
        model_lower = model_name.lower()

        # Check for small model indicators
        if any(size in model_lower for size in ["0.5b", "1b", "1.5b", "2b", "3b"]):
            # Small models: use at most 1-2 GPUs (attention heads typically 12-16)
            # 12 heads divisible by: 1, 2, 3, 4, 6, 12
            # 16 heads divisible by: 1, 2, 4, 8, 16
            tp_size = min(num_gpus, 1)  # Conservative: use 1 GPU for small models
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
        logger.info(f"Unknown model size, using tensor_parallel_size=1 for safety")
        return 1

    def _validate_model(self, model: BaseModel) -> None:
        """
        Validate that the model is properly loaded and functional.

        Args:
            model: Model instance to validate

        Raises:
            RuntimeError: If validation fails
        """
        logger.info("Validating model...")

        if not model.is_loaded():
            raise RuntimeError("Model validation failed: model not loaded")

        # Test token counting
        try:
            test_text = "Hello, world!"
            token_count = model.count_tokens(test_text)
            logger.info(f"Token counting works: '{test_text}' = {token_count.count} tokens")
        except Exception as e:
            logger.warning(f"Token counting validation failed: {e}")

        # Test context length
        try:
            context_length = model.get_context_length()
            logger.info(f"Context length: {context_length} tokens")
        except Exception as e:
            logger.warning(f"Context length validation failed: {e}")

        logger.info("Model validation passed")

    def unload_model(self, model_name: str) -> None:
        """
        Unload a specific model from memory.

        Args:
            model_name: Name of model to unload
        """
        if model_name in self.loaded_models:
            logger.info(f"Unloading model: {model_name}")
            model = self.loaded_models[model_name]
            model.unload()
            del self.loaded_models[model_name]
            logger.info(f"Model '{model_name}' unloaded")
        else:
            logger.warning(f"Model '{model_name}' not found in loaded models")

    def unload_all(self) -> None:
        """
        Unload all loaded models.
        """
        logger.info("Unloading all models...")
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        logger.info("All models unloaded")

    def get_loaded_models(self) -> Dict[str, BaseModel]:
        """
        Get dictionary of all loaded models.

        Returns:
            Dict mapping model names to model instances
        """
        return self.loaded_models.copy()

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model.

        Args:
            model_name: Name of the model

        Returns:
            Model metadata dict or None if not loaded
        """
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            return model.get_metadata()
        return None


# Convenience function for quick model loading
def load_model(
    model_name: str,
    engine: Optional[str] = None,
    engine_config: Optional[Dict[str, Any]] = None,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    apply_chat_template: bool = True,
    **kwargs
) -> BaseModel:
    """
    Convenience function to quickly load a model.

    Args:
        model_name: Model identifier
        engine: Engine to use ('vllm', 'transformers', or None for auto)
        engine_config: Optional engine configuration
        tensor_parallel_size: Number of GPUs for tensor parallelism (vLLM only).
                             If not specified, auto-detected based on model size.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0, vLLM only).
                               Default is 0.90. Lower this if you get CUDA OOM errors.
        apply_chat_template: Whether to automatically apply chat templates for
                            instruction-tuned models (default: True, vLLM only).
                            This ensures proper formatting for models like Qwen-Instruct.
        **kwargs: Additional parameters (e.g., quantization="4bit", trust_remote_code=True)

    Returns:
        Loaded model instance

    Example:
        >>> from effgen.models import load_model
        >>> # Default uses Transformers engine
        >>> model = load_model("meta-llama/Llama-2-7b-hf")
        >>> result = model.generate("Hello, how are you?")

        >>> # Explicitly use vLLM for production (5-10x faster)
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm")

        >>> # With tensor parallelism for large models
        >>> model = load_model("meta-llama/Llama-2-70b-hf", engine="vllm", tensor_parallel_size=4)

        >>> # Lower GPU memory usage if getting OOM errors
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm", gpu_memory_utilization=0.7)

        >>> # Disable chat template for raw text generation
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm", apply_chat_template=False)
    """
    # Pass tensor_parallel_size to kwargs if specified
    if tensor_parallel_size is not None:
        kwargs["tensor_parallel_size"] = tensor_parallel_size

    # Pass gpu_memory_utilization to kwargs if specified
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    # Pass apply_chat_template for vLLM
    kwargs["apply_chat_template"] = apply_chat_template

    loader = ModelLoader(force_engine=engine)
    return loader.load_model(model_name, engine_config, **kwargs)
