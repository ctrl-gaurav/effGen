"""
MLX engine implementation for Apple Silicon inference.

This module provides inference on Apple Silicon (M-series) Macs using the MLX framework.
Features:
- Native Apple Silicon GPU acceleration via Metal
- Unified memory architecture (no CPU-GPU transfer overhead)
- Quantized model support (4-bit, 8-bit)
- LoRA adapter loading
- Streaming token generation
- Chat template support
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from typing import Any

from effgen.models._adapter_utils import (
    chat_template_renders_tools,
    merge_call_overrides,
    not_loaded_error,
    provider_runtime_error,
)
from effgen.models.base import (
    BatchModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)

logger = logging.getLogger(__name__)


def _build_mlx_gen_kwargs(config: GenerationConfig, greedy: bool) -> dict[str, Any]:
    """Build sampling kwargs for mlx_lm ``generate``/``stream_generate``.

    mlx-lm dropped the flat ``temp``/``top_p``/``repetition_penalty`` kwargs
    (accepted by ``generate_step`` in <=0.19) in favour of a ``sampler`` callable
    (``make_sampler``) plus ``logits_processors`` (``make_logits_processors``).
    We build the modern form when those helpers are importable and fall back to
    the legacy flat kwargs otherwise, so effGen works across mlx-lm versions.

    ``seed`` is applied via ``mx.random.seed`` (stable across versions) rather
    than passed through, since the newer API no longer accepts a ``seed`` kwarg.
    """
    temp = 0.0 if greedy else (config.temperature or 0.0)
    top_p = 1.0 if greedy else (config.top_p if config.top_p is not None else 1.0)
    rep_penalty = config.repetition_penalty

    if config.seed is not None:
        try:  # deterministic sampling; harmless if unsupported
            import mlx.core as mx

            mx.random.seed(config.seed)
        except Exception:  # pragma: no cover - best effort
            pass

    kwargs: dict[str, Any] = {"max_tokens": config.max_tokens or 512}

    try:
        from mlx_lm.sample_utils import make_sampler
    except Exception:  # pragma: no cover - very old mlx-lm: legacy flat kwargs
        kwargs.update({"temp": temp, "top_p": top_p})
        if rep_penalty is not None:
            kwargs["repetition_penalty"] = rep_penalty
        if config.seed is not None:
            kwargs["seed"] = config.seed
        return kwargs

    kwargs["sampler"] = make_sampler(temp=temp, top_p=top_p)
    if rep_penalty is not None and rep_penalty != 1.0:
        try:
            from mlx_lm.sample_utils import make_logits_processors

            kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=rep_penalty
            )
        except Exception:  # pragma: no cover - processor helper unavailable
            pass
    return kwargs


class MLXEngine(BatchModel):
    """
    MLX-based model engine for Apple Silicon inference.

    This engine uses Apple's MLX framework for optimized inference on M-series chips,
    leveraging unified memory for efficient model loading and generation.

    Features:
    - Native Metal GPU acceleration
    - Unified memory (no CPU-GPU transfer)
    - 4-bit and 8-bit quantization
    - LoRA adapter support
    - Streaming generation
    - Chat template auto-application

    Attributes:
        model_name: HuggingFace model ID, mlx-community model ID, or local path
        apply_chat_template: Whether to auto-apply chat templates for instruct models
        adapter_path: Optional path to LoRA adapter weights
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int | None = None,
        trust_remote_code: bool = True,
        apply_chat_template: bool = True,
        system_prompt: str | None = None,
        adapter_path: str | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MLX engine.

        Args:
            model_name: HuggingFace model ID (e.g., 'mlx-community/Mistral-7B-Instruct-v0.3-4bit'),
                        or local path to MLX model weights
            max_tokens: Maximum context length (auto-detected if None)
            trust_remote_code: Whether to trust remote code from HuggingFace
            apply_chat_template: Whether to auto-apply chat template for instruct models (default: True)
            system_prompt: Optional system prompt for chat template
            adapter_path: Optional path to LoRA adapter weights
            lazy_load: Whether to use MLX lazy loading for reduced memory usage
            **kwargs: Additional parameters (ignored for compatibility)
        """
        # Use ModelType.MLX if available, else create a compatible fallback
        try:
            model_type = ModelType.MLX
        except AttributeError:
            model_type = ModelType.TRANSFORMERS  # Temporary fallback

        super().__init__(
            model_name=model_name,
            model_type=model_type,
            context_length=max_tokens,
        )

        self.max_tokens_limit = max_tokens
        self.trust_remote_code = trust_remote_code
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt
        self.adapter_path = adapter_path
        self.lazy_load = lazy_load

        self.model = None
        self.tokenizer = None
        # Cached chat-template tool-rendering probe, paired with the tokenizer it
        # was measured on so a reload re-measures instead of reusing a stale answer.
        self._tool_template_probe: tuple[Any, bool] | None = None

    def load(self) -> None:
        """
        Load the model using MLX.

        Raises:
            RuntimeError: If MLX is not installed, not on Apple Silicon, or loading fails
        """
        # Check platform
        from effgen.hardware.platform import is_apple_silicon

        if not is_apple_silicon():
            raise RuntimeError(
                "MLX requires Apple Silicon (M-series) Mac. "
                "Use engine='vllm' or engine='transformers' on this platform."
            )

        # Lazy import mlx-lm
        try:
            from mlx_lm import load as mlx_load
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm is not installed. Install with: pip install 'effgen[mlx]' "
                "or: pip install mlx-lm"
            ) from e

        try:
            logger.info(f"Loading model '{self.model_name}' with MLX...")

            # Load model and tokenizer
            load_kwargs: dict[str, Any] = {}
            if self.adapter_path:
                load_kwargs["adapter_path"] = self.adapter_path
            if self.lazy_load:
                load_kwargs["lazy"] = True

            self.model, self.tokenizer = mlx_load(
                self.model_name,
                **load_kwargs,
            )

            # Detect context length from model config
            self._context_length = self._detect_context_length()

            # Store metadata
            self._metadata = {
                "model_name": self.model_name,
                "engine": "mlx",
                "adapter_path": self.adapter_path,
                "max_context_length": self._context_length,
                "apply_chat_template": self.apply_chat_template,
                "platform": "apple_silicon",
            }

            # Log memory info
            try:
                from effgen.hardware.platform import get_unified_memory_gb

                mem_gb = get_unified_memory_gb()
                if mem_gb > 0:
                    logger.info(f"Unified memory available: {mem_gb:.1f} GB")
                    self._metadata["unified_memory_gb"] = mem_gb
            except Exception:
                logger.debug("Failed to read unified memory info", exc_info=True)

            self._is_loaded = True
            logger.info(f"Model '{self.model_name}' loaded successfully with MLX")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load model with MLX: {error_msg}")
            raise provider_runtime_error(
                "mlx", self.model_name, "load", e,
                message="MLX model loading failed",
            ) from e

    def _detect_context_length(self) -> int:
        """Detect the model's maximum context length from its config."""
        if self.max_tokens_limit:
            return self.max_tokens_limit

        # Try to read from model config (common HuggingFace config fields)
        config = getattr(self.model, "config", None) or getattr(
            self.model, "args", None
        )
        if config is not None:
            # Try various config attribute names
            for attr in [
                "max_position_embeddings",
                "max_sequence_length",
                "n_positions",
                "seq_length",
                "max_seq_len",
                "sliding_window",
            ]:
                val = getattr(config, attr, None)
                if val is not None and isinstance(val, int) and val > 0:
                    logger.debug(f"Detected context length {val} from config.{attr}")
                    return val

        # Check tokenizer model_max_length
        if self.tokenizer is not None:
            max_len = getattr(self.tokenizer, "model_max_length", None)
            if max_len and isinstance(max_len, int) and max_len < 1_000_000:
                return max_len

        logger.debug("Could not detect context length, using default 4096")
        return 4096

    def _format_prompt_with_chat_template(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: Any = None,
    ) -> str:
        """
        Format a prompt using the model's chat template.

        This is essential for instruction-tuned models (e.g., Qwen-Instruct, Llama-Instruct)
        which expect prompts in a specific format with special tokens.

        Args:
            prompt: The raw user prompt
            system_prompt: Optional system prompt override
            tools: Native tool schemas to render into the template, for a
                model whose chat template accepts them. A template that does
                not is rendered without them rather than failing.

        Returns:
            Formatted prompt string
        """
        if not self.apply_chat_template or self.tokenizer is None:
            return prompt

        # Check if already formatted
        chat_template_indicators = [
            "<|im_start|>",
            "<|im_end|>",  # Qwen format
            "[INST]",
            "[/INST]",  # Llama/Mistral format
            "<|begin_of_text|>",
            "<|start_header_id|>",  # Llama 3 format
            "<|user|>",
            "<|assistant|>",  # Generic format
            "### Human:",
            "### Assistant:",  # Vicuna format
        ]
        if any(indicator in prompt for indicator in chat_template_indicators):
            logger.debug(
                "Prompt already has chat template markers, skipping template application"
            )
            return prompt

        # Check if tokenizer supports chat templates
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        try:
            messages: list[dict[str, str]] = []
            effective_system_prompt = system_prompt or self.system_prompt
            if effective_system_prompt:
                messages.append({"role": "system", "content": effective_system_prompt})
            messages.append({"role": "user", "content": prompt})

            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # Inject native tool schemas so instruct models (Qwen, Gemma, …) emit
            # their trained tool-call tokens. Without this the model never learns
            # the tools exist and just answers in prose — the reason a "calculator
            # agent" would silently do (sometimes wrong) mental math instead.
            if tools:
                template_kwargs["tools"] = tools
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError as e:
                # Older/simple templates don't accept a tools= kwarg — fall back.
                if tools:
                    logger.debug(
                        f"Chat template does not accept tools param: {e}; "
                        "falling back to plain template"
                    )
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                else:
                    raise
            logger.debug(
                f"Applied chat template (length: {len(prompt)} -> {len(formatted)})"
            )
            return formatted
        except Exception as e:
            logger.warning(f"Failed to apply chat template, using raw prompt: {e}")
            return prompt

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt override
            skip_chat_template: If True, skip chat template application
            **kwargs: Per-call sampling overrides (``seed``, ``temperature``,
                ``top_p``, ``max_tokens``, ``stop``, ``repetition_penalty``)
                take precedence over the same field on *config*.

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._is_loaded:
            raise not_loaded_error("mlx", self.model_name, "generate")

        from mlx_lm import generate as mlx_generate

        if config is None:
            config = GenerationConfig()
        # A per-call sampling keyword supersedes the config's field for this call
        # and is consumed here, so it is not also forwarded to mlx-lm.
        config = merge_call_overrides(config, kwargs)

        # Native tool schemas are passed to the chat template, not to mlx-lm's
        # generate(), so pull them out of kwargs before building sampling args.
        tools = kwargs.get("tools")

        # Apply chat template
        if not skip_chat_template:
            formatted_prompt = self._format_prompt_with_chat_template(
                prompt, system_prompt, tools=tools
            )
        else:
            formatted_prompt = prompt

        self.validate_prompt(formatted_prompt)

        try:
            # Build MLX generation kwargs. Normalize deterministic generation:
            # temperature<=0 means greedy (temp=0 in mlx-lm); clamp negatives.
            _greedy = config.temperature is None or config.temperature <= 0
            gen_kwargs = _build_mlx_gen_kwargs(config, _greedy)

            generated_text = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                verbose=False,
                **gen_kwargs,
            )

            # Handle stop sequences
            if config.stop_sequences:
                for stop_seq in config.stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text[
                            : generated_text.index(stop_seq)
                        ]
                        break

            # Count tokens for metadata
            prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
            completion_tokens = len(self.tokenizer.encode(generated_text))

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason="stop",
                model_name=self.model_name,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "chat_template_applied": not skip_chat_template
                    and self.apply_chat_template,
                    "engine": "mlx",
                },
            )

        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise provider_runtime_error(
                "mlx", self.model_name, "generate", e,
                message="MLX generation failed",
            ) from e

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt override
            skip_chat_template: If True, skip chat template application
            **kwargs: Per-call sampling overrides (``seed``, ``temperature``,
                ``top_p``, ``max_tokens``, ``stop``, ``repetition_penalty``)
                take precedence over the same field on *config*.

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._is_loaded:
            raise not_loaded_error("mlx", self.model_name, "generate_stream")

        from mlx_lm import stream_generate

        if config is None:
            config = GenerationConfig()
        # A per-call sampling keyword supersedes the config's field for this call
        # and is consumed here, so it is not also forwarded to mlx-lm.
        config = merge_call_overrides(config, kwargs)

        # Native tool schemas go to the chat template, not mlx-lm generate().
        tools = kwargs.get("tools")

        # Apply chat template
        if not skip_chat_template:
            formatted_prompt = self._format_prompt_with_chat_template(
                prompt, system_prompt, tools=tools
            )
        else:
            formatted_prompt = prompt

        self.validate_prompt(formatted_prompt)

        try:
            # Normalize deterministic generation: temperature<=0 means greedy.
            _greedy = config.temperature is None or config.temperature <= 0
            gen_kwargs = _build_mlx_gen_kwargs(config, _greedy)

            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                **gen_kwargs,
            ):
                # stream_generate yields dicts or objects with 'text' field
                if isinstance(response, dict):
                    text = response.get("text", "")
                elif hasattr(response, "text"):
                    text = response.text
                else:
                    text = str(response)

                if text:
                    # Check for stop sequences
                    if config.stop_sequences:
                        should_stop = False
                        for stop_seq in config.stop_sequences:
                            if stop_seq in text:
                                text = text[: text.index(stop_seq)]
                                if text:
                                    yield text
                                should_stop = True
                                break
                        if should_stop:
                            return
                    yield text

        except Exception as e:
            logger.error(f"MLX streaming generation failed: {e}")
            raise provider_runtime_error(
                "mlx", self.model_name, "generate_stream", e,
                message="MLX streaming generation failed",
            ) from e

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        **kwargs: Any,
    ) -> list[GenerationResult]:
        """
        Generate text for multiple prompts.

        Note: MLX does not support continuous batching like vLLM.
        Prompts are processed sequentially.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            system_prompt: Optional system prompt for all prompts
            skip_chat_template: If True, skip chat template application
            **kwargs: Per-call sampling overrides (``seed``, ``temperature``,
                ``top_p``, ``max_tokens``, ``stop``, ``repetition_penalty``)
                take precedence over the same field on *config*.

        Returns:
            List of GenerationResult objects
        """
        if not self._is_loaded:
            raise not_loaded_error("mlx", self.model_name, "generate_batch")

        if len(prompts) > 1:
            logger.info(
                f"Processing {len(prompts)} prompts sequentially "
                "(MLX does not support continuous batching)"
            )

        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(
                    prompt=prompt,
                    config=config,
                    system_prompt=system_prompt,
                    skip_chat_template=skip_chat_template,
                    **kwargs,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append(
                    GenerationResult(
                        text="",
                        tokens_used=0,
                        finish_reason="error",
                        model_name=self.model_name,
                        metadata={"error": str(e), "batch_index": i},
                    )
                )

        return results

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
            raise not_loaded_error("mlx", self.model_name, "count_tokens")

        try:
            tokens = self.tokenizer.encode(text)
            return TokenCount(count=len(tokens), model_name=self.model_name)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            raise provider_runtime_error(
                "mlx", self.model_name, "count_tokens", e,
                message="Token counting failed",
            ) from e

    def get_context_length(self) -> int:
        """Get maximum context length."""
        if self._context_length is not None:
            return self._context_length
        return 4096

    def get_max_batch_size(self) -> int:
        """Get maximum batch size. MLX processes sequentially."""
        return 1

    def capability_key(self) -> str | None:
        """Weights, their revision and the chat template: the probe's cache key.

        ``None`` until the model is loaded, so a model that has not loaded is
        not probed.
        """
        from .capability_probe import local_capability_key

        return local_capability_key(self)

    def supports_tool_calling(self) -> bool:
        """Check if model supports native tool calling via chat template.

        True when the chat template actually **renders** tool definitions into
        the prompt. A template that accepts a ``tools`` argument and discards it
        reports False: the model never sees the tools, so the ReAct text
        protocol is the only path that reaches one.
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

    def unload(self) -> None:
        """Unload the model and free memory."""
        if self.model is not None:
            logger.info(f"Unloading MLX model '{self.model_name}'...")
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Drop the cached capability probe with the tokenizer it measured, so
        # unload releases it and a later load re-measures.
        self._tool_template_probe = None

        gc.collect()

        self._is_loaded = False
        logger.info(f"Model '{self.model_name}' unloaded successfully")
