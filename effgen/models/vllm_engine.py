"""
vLLM engine implementation for fast inference with multi-GPU support.

This module provides high-performance inference using vLLM with features including:
- Tensor parallelism for multi-GPU deployment
- Quantization support (4-bit, 8-bit)
- Dynamic batching for throughput optimization
- Streaming token generation
- Graceful fallback handling
- Automatic chat template application for instruction-tuned models
"""

import logging
from typing import Iterator, Optional, List
import torch

from effgen.models.base import (
    BatchModel,
    ModelType,
    GenerationConfig,
    GenerationResult,
    TokenCount
)

logger = logging.getLogger(__name__)


def _is_cuda_oom_error(error: Exception) -> bool:
    """Check if an exception is a CUDA out-of-memory error."""
    error_str = str(error).lower()
    oom_indicators = [
        "cuda out of memory",
        "out of memory",
        "oom",
        "cudamalloc failed",
        "failed to allocate",
        "memory allocation",
        "insufficient memory",
        "cuda error: out of memory",
    ]
    return any(indicator in error_str for indicator in oom_indicators)


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
        trust_remote_code: bool = True,
        download_dir: Optional[str] = None,
        dtype: str = "auto",
        seed: int = 0,
        max_num_seqs: int = 256,
        max_num_batched_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        apply_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize vLLM engine.

        Args:
            model_name: HuggingFace model ID or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            quantization: Quantization method ('awq', 'gptq', 'squeezellm', or None)
            max_model_len: Maximum sequence length (auto-detected if None)
            gpu_memory_utilization: GPU memory fraction to use (0.0-1.0). Default is 0.90.
                                   Lower this value if you encounter CUDA out-of-memory errors.
            trust_remote_code: Whether to trust remote code from HuggingFace (default: True)
            download_dir: Directory to download model to
            dtype: Data type for model weights ('auto', 'float16', 'bfloat16')
            seed: Random seed for reproducibility
            max_num_seqs: Maximum number of sequences in a batch
            max_num_batched_tokens: Maximum number of tokens per batch
            use_tqdm: Whether to show tqdm progress bar during generation (default: True)
            apply_chat_template: Whether to automatically apply the model's chat template
                                to prompts (default: True). This is essential for instruction-tuned
                                models like Qwen-Instruct, Llama-Instruct, etc.
            system_prompt: Optional system prompt to use when applying chat template.
                          If None and apply_chat_template is True, no system message is added.
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
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt
        self.additional_kwargs = kwargs

        self.llm = None
        self.tokenizer = None
        self._hf_tokenizer = None  # Separate HuggingFace tokenizer for chat template

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
                f"quantization={self.quantization}, dtype={self.dtype}, "
                f"gpu_memory_utilization={self.gpu_memory_utilization}"
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

            # Get tokenizer for token counting (vLLM's internal tokenizer)
            self.tokenizer = self.llm.get_tokenizer()

            # Load a separate HuggingFace tokenizer for chat template support
            # This is needed because vLLM's tokenizer may not have full chat template support
            if self.apply_chat_template:
                try:
                    from transformers import AutoTokenizer
                    self._hf_tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code
                    )
                    if hasattr(self._hf_tokenizer, 'chat_template') and self._hf_tokenizer.chat_template:
                        logger.info("Chat template support enabled for this model")
                    else:
                        logger.warning(
                            f"Model '{self.model_name}' does not have a chat template. "
                            "Prompts will be passed directly without formatting."
                        )
                        self._hf_tokenizer = None
                except Exception as e:
                    logger.warning(f"Failed to load HuggingFace tokenizer for chat template: {e}")
                    self._hf_tokenizer = None

            # Store metadata
            self._context_length = self.llm.llm_engine.model_config.max_model_len
            self._metadata = {
                "model_name": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "quantization": self.quantization,
                "dtype": self.dtype,
                "max_model_len": self._context_length,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "apply_chat_template": self.apply_chat_template,
            }

            self._is_loaded = True
            logger.info(f"Model '{self.model_name}' loaded successfully with vLLM")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load model with vLLM: {error_msg}")

            # Provide helpful error message for CUDA OOM
            if _is_cuda_oom_error(e):
                gpu_mem = self.gpu_memory_utilization
                logger.error(
                    f"\n{'='*60}\n"
                    f"CUDA OUT OF MEMORY ERROR\n"
                    f"{'='*60}\n"
                    f"The model '{self.model_name}' could not fit in GPU memory.\n\n"
                    f"Current gpu_memory_utilization: {gpu_mem}\n\n"
                    f"Suggestions to resolve this:\n"
                    f"1. Lower gpu_memory_utilization (e.g., 0.7 or 0.8)\n"
                    f"2. Use quantization: quantization='awq' or 'gptq'\n"
                    f"3. Reduce max_model_len if you don't need full context\n"
                    f"4. Use a smaller model\n"
                    f"5. Free up GPU memory by stopping other processes\n"
                    f"{'='*60}"
                )
                raise RuntimeError(
                    f"CUDA out of memory while loading '{self.model_name}'. "
                    f"Try lowering gpu_memory_utilization (current: {gpu_mem}) or use quantization."
                ) from e

            raise RuntimeError(f"vLLM model loading failed: {error_msg}") from e

    def _format_prompt_with_chat_template(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format a prompt using the model's chat template.

        This is essential for instruction-tuned models (e.g., Qwen-Instruct, Llama-Instruct)
        which expect prompts in a specific format with special tokens.

        Args:
            prompt: The raw user prompt/message
            system_prompt: Optional system prompt to include

        Returns:
            Formatted prompt string ready for generation
        """
        # If chat template is disabled or not available, return raw prompt
        if not self.apply_chat_template or self._hf_tokenizer is None:
            return prompt

        # Check if the prompt already looks like it has chat template applied
        # (contains special tokens like <|im_start|>, [INST], <s>, etc.)
        chat_template_indicators = [
            '<|im_start|>', '<|im_end|>',  # Qwen format
            '[INST]', '[/INST]',            # Llama/Mistral format
            '<|begin_of_text|>', '<|start_header_id|>',  # Llama 3 format
            '<|user|>', '<|assistant|>',    # Generic format
            '### Human:', '### Assistant:',  # Vicuna format
        ]

        if any(indicator in prompt for indicator in chat_template_indicators):
            logger.debug("Prompt already has chat template markers, skipping template application")
            return prompt

        try:
            # Build messages list
            messages = []

            # Add system prompt if provided
            effective_system_prompt = system_prompt or self.system_prompt
            if effective_system_prompt:
                messages.append({"role": "system", "content": effective_system_prompt})

            # Add user message
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            formatted_prompt = self._hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            logger.debug(f"Applied chat template to prompt (length: {len(prompt)} -> {len(formatted_prompt)})")
            return formatted_prompt

        except Exception as e:
            logger.warning(f"Failed to apply chat template, using raw prompt: {e}")
            return prompt

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
        system_prompt: Optional[str] = None,
        skip_chat_template: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt to use (overrides instance-level system_prompt)
            skip_chat_template: If True, skip chat template even if apply_chat_template is enabled.
                               Useful when passing pre-formatted prompts.
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Apply chat template if enabled and not skipped
        if not skip_chat_template:
            formatted_prompt = self._format_prompt_with_chat_template(prompt, system_prompt)
        else:
            formatted_prompt = prompt

        self.validate_prompt(formatted_prompt)

        sampling_params = self._create_sampling_params(config)

        try:
            outputs = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=self.use_tqdm, **kwargs)
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
                    "chat_template_applied": not skip_chat_template and self.apply_chat_template,
                }
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Generation failed: {error_msg}")

            # Provide helpful error for CUDA OOM during generation
            if _is_cuda_oom_error(e):
                logger.error(
                    f"CUDA out of memory during generation. "
                    f"Try reducing max_tokens in GenerationConfig or use a smaller prompt."
                )

            raise RuntimeError(f"Generation failed: {error_msg}") from e

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        skip_chat_template: bool = False,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt to use
            skip_chat_template: If True, skip chat template application
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Apply chat template if enabled and not skipped
        if not skip_chat_template:
            formatted_prompt = self._format_prompt_with_chat_template(prompt, system_prompt)
        else:
            formatted_prompt = prompt

        self.validate_prompt(formatted_prompt)

        sampling_params = self._create_sampling_params(config)

        try:
            # vLLM's streaming interface
            for output in self.llm.generate([formatted_prompt], sampling_params, use_tqdm=self.use_tqdm, **kwargs):
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
        system_prompt: Optional[str] = None,
        skip_chat_template: bool = False,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            system_prompt: Optional system prompt to use for all prompts
            skip_chat_template: If True, skip chat template application
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If any prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Apply chat template to all prompts if enabled
        if not skip_chat_template:
            formatted_prompts = [
                self._format_prompt_with_chat_template(p, system_prompt)
                for p in prompts
            ]
        else:
            formatted_prompts = prompts

        # Validate all formatted prompts
        for prompt in formatted_prompts:
            self.validate_prompt(prompt)

        sampling_params = self._create_sampling_params(config)

        try:
            outputs = self.llm.generate(formatted_prompts, sampling_params, use_tqdm=self.use_tqdm, **kwargs)

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
                        "chat_template_applied": not skip_chat_template and self.apply_chat_template,
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

        if self._hf_tokenizer is not None:
            del self._hf_tokenizer
            self._hf_tokenizer = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        logger.info(f"Model '{self.model_name}' unloaded successfully")
