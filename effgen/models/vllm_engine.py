"""
vLLM engine implementation for fast inference with multi-GPU support.

This module provides high-performance inference using vLLM with features including:
- Tensor parallelism for multi-GPU deployment
- Quantization support (4-bit, 8-bit)
- Dynamic batching for throughput optimization
- Streaming token generation
- Graceful fallback handling
"""

import logging
from typing import Iterator, Optional, List, Dict, Any
import torch

from effgen.models.base import (
    BaseModel,
    BatchModel,
    ModelType,
    GenerationConfig,
    GenerationResult,
    TokenCount
)

logger = logging.getLogger(__name__)


class VLLMEngine(BatchModel):
    """
    vLLM-based model engine for fast inference.

    This engine uses vLLM for optimized inference with PagedAttention,
    continuous batching, and efficient memory management.

    Features:
    - Multi-GPU tensor parallelism
    - Quantization (AWQ, GPTQ, SqueezeLLM)
    - Dynamic batching
    - KV cache optimization
    - Streaming generation

    Attributes:
        model_name: HuggingFace model identifier or path
        tensor_parallel_size: Number of GPUs for tensor parallelism
        quantization: Quantization method (None, 'awq', 'gptq', 'squeezellm')
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = False,
        download_dir: Optional[str] = None,
        dtype: str = "auto",
        seed: int = 0,
        max_num_seqs: int = 256,
        max_num_batched_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        **kwargs
    ):
        """
        Initialize vLLM engine.

        Args:
            model_name: HuggingFace model ID or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            quantization: Quantization method ('awq', 'gptq', 'squeezellm', or None)
            max_model_len: Maximum sequence length (auto-detected if None)
            gpu_memory_utilization: GPU memory fraction to use (0.0-1.0)
            trust_remote_code: Whether to trust remote code from HuggingFace
            download_dir: Directory to download model to
            dtype: Data type for model weights ('auto', 'float16', 'bfloat16')
            seed: Random seed for reproducibility
            max_num_seqs: Maximum number of sequences in a batch
            max_num_batched_tokens: Maximum number of tokens per batch
            use_tqdm: Whether to show tqdm progress bar during generation (default: True)
            **kwargs: Additional vLLM engine arguments
        """
        super().__init__(
            model_name=model_name,
            model_type=ModelType.VLLM,
            context_length=max_model_len
        )

        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.dtype = dtype
        self.seed = seed
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.use_tqdm = use_tqdm
        self.additional_kwargs = kwargs

        self.llm = None
        self.tokenizer = None

    def load(self) -> None:
        """
        Load the model using vLLM.

        Raises:
            RuntimeError: If vLLM is not installed or model loading fails
            ValueError: If configuration is invalid
        """
        try:
            from vllm import LLM
            from vllm import SamplingParams
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. Please install it with: "
                "pip install vllm"
            ) from e

        # Validate GPU availability
        if not torch.cuda.is_available() and self.tensor_parallel_size > 0:
            raise RuntimeError("CUDA is not available but tensor_parallel_size > 0")

        if self.tensor_parallel_size > torch.cuda.device_count():
            logger.warning(
                f"tensor_parallel_size ({self.tensor_parallel_size}) exceeds "
                f"available GPUs ({torch.cuda.device_count()}). "
                f"Reducing to {torch.cuda.device_count()}"
            )
            self.tensor_parallel_size = torch.cuda.device_count()

        try:
            logger.info(f"Loading model '{self.model_name}' with vLLM...")
            logger.info(
                f"Configuration: tensor_parallel={self.tensor_parallel_size}, "
                f"quantization={self.quantization}, dtype={self.dtype}"
            )

            # Build vLLM engine arguments
            engine_args = {
                "model": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": self.trust_remote_code,
                "dtype": self.dtype,
                "seed": self.seed,
                "max_num_seqs": self.max_num_seqs,
            }

            if self.quantization:
                engine_args["quantization"] = self.quantization

            if self.max_model_len:
                engine_args["max_model_len"] = self.max_model_len

            if self.download_dir:
                engine_args["download_dir"] = self.download_dir

            if self.max_num_batched_tokens:
                engine_args["max_num_batched_tokens"] = self.max_num_batched_tokens

            # Add any additional kwargs
            engine_args.update(self.additional_kwargs)

            # Initialize vLLM engine
            self.llm = LLM(**engine_args)

            # Get tokenizer for token counting
            self.tokenizer = self.llm.get_tokenizer()

            # Store metadata
            self._context_length = self.llm.llm_engine.model_config.max_model_len
            self._metadata = {
                "model_name": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "quantization": self.quantization,
                "dtype": self.dtype,
                "max_model_len": self._context_length,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            }

            self._is_loaded = True
            logger.info(f"Model '{self.model_name}' loaded successfully with vLLM")

        except Exception as e:
            logger.error(f"Failed to load model with vLLM: {e}")
            raise RuntimeError(f"vLLM model loading failed: {e}") from e

    def _create_sampling_params(
        self,
        config: Optional[GenerationConfig] = None
    ) -> "SamplingParams":
        """
        Create vLLM SamplingParams from GenerationConfig.

        Args:
            config: Generation configuration

        Returns:
            vLLM SamplingParams object
        """
        from vllm import SamplingParams

        if config is None:
            config = GenerationConfig()

        return SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            repetition_penalty=config.repetition_penalty,
            seed=config.seed,
        )

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        self.validate_prompt(prompt)

        sampling_params = self._create_sampling_params(config)

        try:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=self.use_tqdm, **kwargs)
            output = outputs[0]

            generated_text = output.outputs[0].text
            tokens_used = len(output.outputs[0].token_ids)
            finish_reason = output.outputs[0].finish_reason

            return GenerationResult(
                text=generated_text,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata={
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": tokens_used,
                    "total_tokens": len(output.prompt_token_ids) + tokens_used,
                }
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        self.validate_prompt(prompt)

        sampling_params = self._create_sampling_params(config)

        try:
            # vLLM's streaming interface
            for output in self.llm.generate([prompt], sampling_params, use_tqdm=self.use_tqdm, **kwargs):
                # Stream each token as it's generated
                for token_output in output.outputs:
                    yield token_output.text

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If any prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Validate all prompts
        for prompt in prompts:
            self.validate_prompt(prompt)

        sampling_params = self._create_sampling_params(config)

        try:
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=self.use_tqdm, **kwargs)

            results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                tokens_used = len(output.outputs[0].token_ids)
                finish_reason = output.outputs[0].finish_reason

                results.append(GenerationResult(
                    text=generated_text,
                    tokens_used=tokens_used,
                    finish_reason=finish_reason,
                    model_name=self.model_name,
                    metadata={
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": tokens_used,
                        "total_tokens": len(output.prompt_token_ids) + tokens_used,
                    }
                ))

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise RuntimeError(f"Batch generation failed: {e}") from e

    def count_tokens(self, text: str) -> TokenCount:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        try:
            tokens = self.tokenizer.encode(text)
            return TokenCount(count=len(tokens), model_name=self.model_name)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            raise RuntimeError(f"Token counting failed: {e}") from e

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
            int: Maximum number of sequences per batch
        """
        return self.max_num_seqs

    def unload(self) -> None:
        """
        Unload the model and free GPU memory.
        """
        if self.llm is not None:
            logger.info(f"Unloading model '{self.model_name}'...")
            del self.llm
            self.llm = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        logger.info(f"Model '{self.model_name}' unloaded successfully")
