"""
MLX-VLM engine for vision-language model inference on Apple Silicon.

Extends MLXEngine with multimodal capabilities for processing images and text.
Supports 30+ VLM architectures via mlx-vlm including Qwen2-VL, LLaVA,
Phi-3 Vision, Pixtral, PaliGemma, and more.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from typing import Any

from effgen.models.base import GenerationConfig, GenerationResult, ModelType, TokenCount
from effgen.models.mlx_engine import MLXEngine

logger = logging.getLogger(__name__)


class MLXVLMEngine(MLXEngine):
    """
    MLX-VLM engine for vision-language models on Apple Silicon.

    Extends MLXEngine with image understanding capabilities.
    Supports Qwen2-VL, LLaVA, Phi-3 Vision, Pixtral, PaliGemma, and 30+ more
    architectures through the mlx-vlm library.

    Key differences from MLXEngine:
    - Uses ``mlx_vlm.load`` which returns (model, processor) instead of (model, tokenizer)
    - ``generate()`` accepts an ``images`` kwarg for multimodal inputs
    - Uses VLM-specific chat template formatting via ``mlx_vlm.prompt_utils``
    - Tool calling is not supported (VLMs typically lack this capability)

    Attributes:
        processor: The VLM processor (handles both text and image preprocessing)
        vlm_config: Model configuration loaded via mlx-vlm utilities
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int | None = None,
        trust_remote_code: bool = True,
        apply_chat_template: bool = True,
        system_prompt: str | None = None,
        **kwargs,
    ):
        """
        Initialize MLX-VLM engine.

        Args:
            model_name: HuggingFace model ID (e.g., 'mlx-community/Qwen2-VL-2B-Instruct-4bit'),
                        or local path to MLX-VLM model weights
            max_tokens: Maximum context length (auto-detected if None)
            trust_remote_code: Whether to trust remote code from HuggingFace
            apply_chat_template: Whether to auto-apply chat template (default: True)
            system_prompt: Optional system prompt for chat template
            **kwargs: Additional parameters (ignored gracefully for compatibility)
        """
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.processor = None
        self.vlm_config = None

        # Override model type to MLX_VLM if available
        try:
            self.model_type = ModelType.MLX_VLM
        except AttributeError:
            pass  # Will use parent's fallback

    def load(self) -> None:
        """
        Load the vision-language model using mlx-vlm.

        Uses ``mlx_vlm.load`` which returns (model, processor) and
        ``mlx_vlm.utils.load_config`` for model configuration.

        The processor is stored separately, and a tokenizer reference is
        extracted from it for ``count_tokens`` compatibility.

        Raises:
            RuntimeError: If not on Apple Silicon, mlx-vlm is not installed,
                          or model loading fails
        """
        from effgen.hardware.platform import is_apple_silicon

        if not is_apple_silicon():
            raise RuntimeError(
                "MLX-VLM requires Apple Silicon (M-series) Mac. "
                "Use a different engine on this platform."
            )

        try:
            from mlx_vlm import load as vlm_load
            from mlx_vlm.utils import load_config
        except ImportError as e:
            raise RuntimeError(
                "mlx-vlm is not installed. Install with: pip install 'effgen[mlx-vlm]' "
                "or: pip install mlx-vlm"
            ) from e

        try:
            logger.info(f"Loading VLM '{self.model_name}' with MLX-VLM...")

            self.model, self.processor = vlm_load(self.model_name)
            self.vlm_config = load_config(self.model_name)

            # Extract tokenizer from processor for count_tokens compatibility.
            # Different processor implementations expose the tokenizer differently.
            if hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer
            elif hasattr(self.processor, "get_tokenizer"):
                self.tokenizer = self.processor.get_tokenizer()
            else:
                # Some processors implement the tokenizer interface directly
                self.tokenizer = self.processor

            self._context_length = self._detect_context_length()

            self._metadata = {
                "model_name": self.model_name,
                "engine": "mlx_vlm",
                "max_context_length": self._context_length,
                "apply_chat_template": self.apply_chat_template,
                "platform": "apple_silicon",
                "supports_vision": True,
            }

            # Log memory info
            try:
                from effgen.hardware.platform import get_unified_memory_gb

                mem_gb = get_unified_memory_gb()
                if mem_gb > 0:
                    logger.info(f"Unified memory available: {mem_gb:.1f} GB")
                    self._metadata["unified_memory_gb"] = mem_gb
            except Exception:
                pass

            self._is_loaded = True
            logger.info(f"VLM '{self.model_name}' loaded successfully with MLX-VLM")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load VLM with MLX-VLM: {error_msg}")
            raise RuntimeError(f"MLX-VLM model loading failed: {error_msg}") from e

    def _format_prompt_with_chat_template(
        self,
        prompt: str,
        system_prompt: str | None = None,
        num_images: int = 0,
    ) -> str:
        """
        Format a prompt using the VLM-specific chat template.

        When images are present, uses ``mlx_vlm.prompt_utils.apply_chat_template``
        which handles image token insertion. For text-only prompts, falls back
        to the parent implementation.

        Args:
            prompt: The raw user prompt
            system_prompt: Optional system prompt override
            num_images: Number of images being passed (0 for text-only)

        Returns:
            Formatted prompt string with appropriate image tokens
        """
        if num_images == 0:
            return super()._format_prompt_with_chat_template(prompt, system_prompt)

        if not self.apply_chat_template:
            return prompt

        try:
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_template

            formatted = vlm_apply_template(
                self.processor,
                self.vlm_config,
                prompt,
                num_images=num_images,
            )
            logger.debug(
                f"Applied VLM chat template (length: {len(prompt)} -> {len(formatted)}, "
                f"images: {num_images})"
            )
            return formatted
        except Exception as e:
            logger.warning(
                f"Failed to apply VLM chat template, falling back to parent: {e}"
            )
            return super()._format_prompt_with_chat_template(prompt, system_prompt)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        images: list | None = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text from a prompt and optional images.

        When no images are provided, delegates entirely to the parent MLXEngine
        for text-only generation. When images are present, uses mlx-vlm's
        multimodal generation pipeline.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt override
            skip_chat_template: If True, skip chat template application
            images: List of image paths (str), URLs (str), or PIL Image objects.
                    Pass None or empty list for text-only generation.
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # If no images, delegate to text-only parent
        if not images:
            return super().generate(
                prompt=prompt,
                config=config,
                system_prompt=system_prompt,
                skip_chat_template=skip_chat_template,
                **kwargs,
            )

        from mlx_vlm import generate as vlm_generate

        if config is None:
            config = GenerationConfig()

        try:
            # Format prompt with VLM-specific template that handles image tokens
            if not skip_chat_template:
                formatted_prompt = self._format_prompt_with_chat_template(
                    prompt,
                    system_prompt,
                    num_images=len(images),
                )
            else:
                formatted_prompt = prompt

            # Build generation kwargs
            gen_kwargs: dict[str, Any] = {
                "max_tokens": config.max_tokens or 512,
                "temp": config.temperature,
                "verbose": False,
            }

            output = vlm_generate(
                self.model,
                self.processor,
                formatted_prompt,
                images,
                **gen_kwargs,
            )

            # Handle stop sequences
            if config.stop_sequences:
                for stop_seq in config.stop_sequences:
                    if stop_seq in output:
                        output = output[: output.index(stop_seq)]
                        break

            # Count tokens for metadata
            prompt_tokens = 0
            completion_tokens = 0
            if self.tokenizer is not None:
                try:
                    prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
                    completion_tokens = len(self.tokenizer.encode(output))
                except Exception:
                    # Token counting may fail for prompts with image tokens
                    completion_tokens = len(output.split()) * 2  # rough estimate

            return GenerationResult(
                text=output,
                tokens_used=completion_tokens,
                finish_reason="stop",
                model_name=self.model_name,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "num_images": len(images),
                    "engine": "mlx_vlm",
                    "chat_template_applied": not skip_chat_template
                    and self.apply_chat_template,
                },
            )

        except Exception as e:
            logger.error(f"MLX-VLM generation failed: {e}")
            raise RuntimeError(f"MLX-VLM generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        images: list | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        For text-only prompts (no images), delegates to the parent MLXEngine's
        streaming implementation. For multimodal prompts with images, generates
        the full response and yields it at once since mlx-vlm does not support
        streaming for image inputs.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt override
            skip_chat_template: If True, skip chat template application
            images: List of image paths, URLs, or PIL Image objects
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks
        """
        if not images:
            yield from super().generate_stream(
                prompt=prompt,
                config=config,
                system_prompt=system_prompt,
                skip_chat_template=skip_chat_template,
                **kwargs,
            )
            return

        # VLM with images: generate full response and yield at once
        # (mlx-vlm does not have native streaming for image inputs)
        result = self.generate(
            prompt=prompt,
            config=config,
            system_prompt=system_prompt,
            skip_chat_template=skip_chat_template,
            images=images,
            **kwargs,
        )
        yield result.text

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        skip_chat_template: bool = False,
        images_list: list[list] | None = None,
        **kwargs,
    ) -> list[GenerationResult]:
        """
        Generate text for multiple prompts, optionally with per-prompt images.

        Processes prompts sequentially since MLX does not support continuous batching.
        Each prompt can have its own set of images via the ``images_list`` parameter.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            system_prompt: Optional system prompt for all prompts
            skip_chat_template: If True, skip chat template application
            images_list: Optional list of image lists, one per prompt.
                         Use None entries for text-only prompts in the batch.
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects, one per prompt
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if images_list is None:
            # No images at all, delegate to parent batch
            return super().generate_batch(
                prompts=prompts,
                config=config,
                system_prompt=system_prompt,
                skip_chat_template=skip_chat_template,
                **kwargs,
            )

        if len(images_list) != len(prompts):
            raise ValueError(
                f"images_list length ({len(images_list)}) must match "
                f"prompts length ({len(prompts)})"
            )

        if len(prompts) > 1:
            logger.info(
                f"Processing {len(prompts)} VLM prompts sequentially "
                "(MLX does not support continuous batching)"
            )

        results = []
        for i, prompt in enumerate(prompts):
            try:
                images = images_list[i] if images_list[i] else None
                result = self.generate(
                    prompt=prompt,
                    config=config,
                    system_prompt=system_prompt,
                    skip_chat_template=skip_chat_template,
                    images=images,
                    **kwargs,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"VLM batch item {i} failed: {e}")
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

    def supports_tool_calling(self) -> bool:
        """VLMs typically do not support native tool calling."""
        return False

    def unload(self) -> None:
        """Unload the model, processor, and free memory."""
        if self.model is not None:
            logger.info(f"Unloading MLX-VLM model '{self.model_name}'...")
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.vlm_config = None

        gc.collect()

        self._is_loaded = False
        logger.info(f"VLM '{self.model_name}' unloaded successfully")
