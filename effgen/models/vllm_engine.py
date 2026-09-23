"""
vLLM engine implementation for fast inference with multi-GPU support.

This module provides high-performance inference using vLLM with features including:
- Tensor parallelism for multi-GPU deployment
- Quantization support (4-bit, 8-bit)
- Dynamic batching for throughput optimization
- Streaming token generation
- Raw-prompt fallback for a model that has no chat template
- Automatic chat template application for instruction-tuned models
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - only on an install without torch
    from effgen.models._adapter_utils import missing_torch_error

    raise missing_torch_error("vllm") from exc

from effgen.models._adapter_utils import (
    attach_error_context,
    chat_template_renders_tools,
    merge_call_overrides,
    not_loaded_error,
    provider_runtime_error,
)
from effgen.models.base import BatchModel, GenerationConfig, GenerationResult, ModelType, TokenCount

logger = logging.getLogger(__name__)


@dataclass
class _Waiting:
    """One call waiting for the engine, and what the engine made of it."""

    prompt: str
    params: Any
    kwargs: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    output: Any = None
    error: BaseException | None = None


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
        quantization: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        download_dir: str | None = None,
        dtype: str = "auto",
        seed: int = 0,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int | None = None,
        use_tqdm: bool = True,
        apply_chat_template: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any
    ) -> None:
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
        # Extra arguments for the chat template, e.g.
        # ``chat_template_kwargs={"enable_thinking": False}`` on a reasoning
        # model. Same spelling as the Transformers engine, so switching engines
        # does not change how the switch is reached.
        self.chat_template_kwargs: dict[str, Any] = dict(
            kwargs.pop("chat_template_kwargs", None) or {}
        )
        self.system_prompt = system_prompt
        self.additional_kwargs = kwargs

        self.llm = None
        self.tokenizer = None
        self._hf_tokenizer = None  # Separate HuggingFace tokenizer for chat template
        # Cached chat-template tool-rendering probe, paired with the tokenizer it
        # was measured on so a reload re-measures instead of reusing a stale answer.
        self._tool_template_probe: tuple[Any, bool] | None = None
        # One engine, many agents. vLLM's in-process client is not safe to call
        # from two threads at once — a second call mid-flight corrupts the
        # socket to its engine core and takes the process down — so calls take
        # turns on the engine, and whoever takes it next sends every call that
        # arrived meanwhile as one batch, which vLLM schedules together.
        self._engine_lock = threading.Lock()
        self._waiting_lock = threading.Lock()
        self._waiting: list[_Waiting] = []
        # A fast tokenizer refuses concurrent use ("Already borrowed").
        self._tokenizer_lock = threading.Lock()

    def load(self) -> None:
        """
        Load the model using vLLM.

        Raises:
            RuntimeError: If vLLM is not installed or model loading fails
            ValueError: If configuration is invalid
        """
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            import importlib.util

            # Distinguish "package not installed" from "package installed but
            # failed to import" (almost always a CUDA/torch ABI mismatch, e.g.
            # a missing libcudart .so). Reporting the latter as "not installed"
            # sends users down the wrong path (reinstalling vLLM won't fix it).
            try:
                pkg_present = importlib.util.find_spec("vllm") is not None
            except (ImportError, ValueError):
                pkg_present = False
            if not pkg_present:
                raise RuntimeError(
                    "vLLM is not installed. Please install it with: pip install vllm"
                ) from e
            raise RuntimeError(
                f"vLLM is installed but failed to import "
                f"({type(e).__name__}: {e}). This is usually a CUDA/torch ABI "
                "mismatch (for example a missing libcudart shared library), not a "
                "missing package — verify your torch build matches the CUDA runtime "
                "vLLM was compiled against. Reinstalling vLLM alone will not fix it."
            ) from e

        # Validate GPU availability without starting the CUDA driver here: vLLM
        # forks its engine core, and a parent that started the driver leaves
        # the child unable to reach the GPU (see cuda_device_visible).
        from effgen.models._vram import cuda_device_visible

        if not cuda_device_visible() and self.tensor_parallel_size > 0:
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
                raise attach_error_context(
                    RuntimeError(
                        f"CUDA out of memory while loading '{self.model_name}'. "
                        f"Try lowering gpu_memory_utilization (current: "
                        f"{gpu_mem}) or use quantization."
                    ),
                    "vllm", self.model_name, "load", source=e,
                ) from e

            raise provider_runtime_error(
                "vllm", self.model_name, "load", e,
                message="vLLM model loading failed",
            ) from e

    def _format_prompt_with_chat_template(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
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
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            template_kwargs.update(getattr(self, "chat_template_kwargs", None) or {})
            if tools:
                template_kwargs["tools"] = tools
            try:
                formatted_prompt = self._hf_tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError:
                # A template that does not declare ``tools`` still has to render
                # the turn; the model simply does not see the definitions, which
                # is what ``tool_call_support() == "none"`` already reports.
                template_kwargs.pop("tools", None)
                formatted_prompt = self._hf_tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )

            logger.debug(f"Applied chat template to prompt (length: {len(prompt)} -> {len(formatted_prompt)})")
            return formatted_prompt

        except Exception as e:
            logger.warning(f"Failed to apply chat template, using raw prompt: {e}")
            return prompt

    def _create_sampling_params(
        self,
        config: GenerationConfig | None = None,
        call_kwargs: dict[str, Any] | None = None,
    ) -> "SamplingParams":  # noqa: F821
        """
        Create vLLM SamplingParams from GenerationConfig.

        Args:
            config: Generation configuration
            call_kwargs: The call's keyword arguments. A sampling keyword found
                here (``seed``, ``temperature``, ``max_tokens``, ...) supersedes
                the config's field for this call and is removed from the dict —
                ``LLM.generate()`` has no parameter of that name and would
                raise ``TypeError`` if one were forwarded to it.

        Returns:
            vLLM SamplingParams object
        """
        from vllm import SamplingParams

        if config is None:
            config = GenerationConfig()
        if call_kwargs:
            config = merge_call_overrides(config, call_kwargs)

        # Normalize deterministic generation: vLLM treats temperature=0 as greedy
        # decoding. Clamp temperature<=0 to exactly 0.0 (vLLM rejects negatives)
        # and disable top_p/top_k so the result is consistent with the other
        # backends' greedy path.
        if config.temperature is None or config.temperature <= 0:
            temperature = 0.0
            top_p = 1.0
            top_k = -1
        else:
            temperature = config.temperature
            top_p = config.top_p
            top_k = config.top_k

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        **kwargs: Any
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt to use (overrides instance-level system_prompt)
            skip_chat_template: If True, skip chat template even if apply_chat_template is enabled.
                               Useful when passing pre-formatted prompts.
            **kwargs: Per-call sampling overrides (``seed``, ``temperature``,
                ``top_p``, ``top_k``, ``max_tokens``, ``stop``, the penalties)
                take precedence over the same field on *config*; any other
                keyword is forwarded to vLLM's ``LLM.generate()``.

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise not_loaded_error("vllm", self.model_name, "generate")

        # Tool definitions belong to the chat template, not to vLLM: the agent
        # attaches them to every call and ``LLM.generate()`` has no ``tools``
        # parameter, so forwarding them raised ``got an unexpected keyword
        # argument 'tools'`` and every tool-using turn failed. Rendered into the
        # prompt here, the way the Transformers engine does it.
        tools_for_template = kwargs.pop("tools", None)

        # Apply chat template if enabled and not skipped
        if not skip_chat_template:
            formatted_prompt = self._format_prompt_with_chat_template(
                prompt, system_prompt, tools_for_template,
            )
        else:
            formatted_prompt = prompt

        self.validate_prompt(formatted_prompt)

        sampling_params = self._create_sampling_params(config, kwargs)
        kwargs.setdefault("use_tqdm", self.use_tqdm)

        try:
            output = self._submit(formatted_prompt, sampling_params, kwargs)

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
                    "CUDA out of memory during generation. "
                    "Try reducing max_tokens in GenerationConfig or use a smaller prompt."
                )

            raise provider_runtime_error(
                "vllm", self.model_name, "generate", e,
                message="Generation failed",
            ) from e

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt to use
            skip_chat_template: If True, skip chat template application
            **kwargs: Per-call sampling overrides (``seed``, ``temperature``,
                ``top_p``, ``top_k``, ``max_tokens``, ``stop``, the penalties)
                take precedence over the same field on *config*; any other
                keyword is forwarded to vLLM's ``LLM.generate()``.

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise not_loaded_error("vllm", self.model_name, "generate_stream")

        # Tool definitions belong to the chat template, not to vLLM: the agent
        # attaches them to every call and ``LLM.generate()`` has no ``tools``
        # parameter, so forwarding them raised ``got an unexpected keyword
        # argument 'tools'`` and every tool-using turn failed. Rendered into the
        # prompt here, the way the Transformers engine does it.
        tools_for_template = kwargs.pop("tools", None)

        # Apply chat template if enabled and not skipped
        if not skip_chat_template:
            formatted_prompt = self._format_prompt_with_chat_template(
                prompt, system_prompt, tools_for_template,
            )
        else:
            formatted_prompt = prompt

        self.validate_prompt(formatted_prompt)

        sampling_params = self._create_sampling_params(config, kwargs)
        kwargs.setdefault("use_tqdm", self.use_tqdm)

        try:
            # ``LLM.generate()`` returns whole outputs, so the stream is the
            # finished text, delivered once.
            output = self._submit(formatted_prompt, sampling_params, kwargs)
            for token_output in output.outputs:
                yield token_output.text

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise provider_runtime_error(
                "vllm", self.model_name, "generate_stream", e,
                message="Streaming generation failed",
            ) from e

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        **kwargs: Any
    ) -> list[GenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            system_prompt: Optional system prompt to use for all prompts
            skip_chat_template: If True, skip chat template application
            **kwargs: Per-call sampling overrides (``seed``, ``temperature``,
                ``top_p``, ``top_k``, ``max_tokens``, ``stop``, the penalties)
                take precedence over the same field on *config*; any other
                keyword is forwarded to vLLM's ``LLM.generate()``.

        Returns:
            List of GenerationResult objects

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If any prompt exceeds context length
        """
        if not self._is_loaded:
            raise not_loaded_error("vllm", self.model_name, "generate_batch")

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

        sampling_params = self._create_sampling_params(config, kwargs)
        kwargs.setdefault("use_tqdm", self.use_tqdm)

        try:
            with self._engine_lock:
                outputs = self.llm.generate(formatted_prompts, sampling_params, **kwargs)

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
            raise provider_runtime_error(
                "vllm", self.model_name, "generate_batch", e,
                message="Batch generation failed",
            ) from e

    def _submit(self, prompt: str, sampling_params: Any, kwargs: dict[str, Any]) -> Any:
        """Run one prompt through the engine, batched with the calls waiting beside it.

        Returns:
            The ``RequestOutput`` vLLM produced for *prompt*.
        """
        mine = _Waiting(prompt, sampling_params, dict(kwargs))
        with self._waiting_lock:
            self._waiting.append(mine)
        with self._engine_lock:
            if not mine.done:
                with self._waiting_lock:
                    batch, self._waiting = self._waiting, []
                self._run_waiting(batch)
        if mine.error is not None:
            raise mine.error
        return mine.output

    def _run_waiting(self, batch: list[_Waiting]) -> None:
        """Send *batch* to the engine; calls with the same extra arguments go together.

        Each prompt keeps its own sampling settings — ``LLM.generate()`` takes
        one ``SamplingParams`` per prompt. Called with the engine lock held.
        """
        groups: dict[str, list[_Waiting]] = {}
        for waiting in batch:
            key = repr(sorted(waiting.kwargs.items(), key=lambda kv: kv[0]))
            groups.setdefault(key, []).append(waiting)
        for group in groups.values():
            if len(group) > 1:
                logger.debug(
                    "[vllm] %d concurrent calls went to the engine as one batch", len(group)
                )
            # A call on its own goes exactly as it always has: one prompt, one
            # ``SamplingParams``.
            params = group[0].params if len(group) == 1 else [w.params for w in group]
            try:
                outputs = self.llm.generate(
                    [w.prompt for w in group], params, **group[0].kwargs
                )
                for waiting, output in zip(group, outputs):
                    waiting.output = output
            except BaseException as exc:  # noqa: BLE001 - handed to every caller in the group
                for waiting in group:
                    waiting.error = exc
            finally:
                for waiting in group:
                    waiting.done = True

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
            raise not_loaded_error("vllm", self.model_name, "count_tokens")

        try:
            with self._tokenizer_lock:
                tokens = self.tokenizer.encode(text)
            return TokenCount(count=len(tokens), model_name=self.model_name)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            raise provider_runtime_error(
                "vllm", self.model_name, "count_tokens", e,
                message="Token counting failed",
            ) from e

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

    def supports_tool_calling(self) -> bool:
        """Check if the model supports native tool calling.

        True when the HuggingFace tokenizer's chat template actually **renders**
        tool definitions into the prompt. A template that accepts a ``tools``
        argument and discards it reports False: the model never sees the tools,
        so the ReAct text protocol is the only path that reaches one.
        """
        tokenizer = self._hf_tokenizer or self.tokenizer
        if not self._is_loaded or tokenizer is None:
            return False
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

    def unload(self) -> None:
        """
        Unload the model and free GPU memory.

        The device memory the weights occupied is returned to the GPU, so the
        next model to load sees it as free.
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

        # Drop the cached capability probe with the tokenizer it measured, so
        # unload releases it and a later load re-measures.
        self._tool_template_probe = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            from effgen.gpu.utils import release_cached_memory

            release_cached_memory()

        self._is_loaded = False
        logger.info(f"Model '{self.model_name}' unloaded successfully")
