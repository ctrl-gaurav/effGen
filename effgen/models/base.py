"""
Abstract base class for all model implementations in effGen.

This module defines the interface that all model engines must implement,
ensuring consistent behavior across vLLM, Transformers, and API adapters.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Enumeration of supported model types."""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result from a generation call."""
    text: str
    tokens_used: int
    finish_reason: str
    model_name: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TokenCount:
    """Token count information."""
    count: int
    model_name: str


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.

    This class defines the interface that all model engines must implement,
    including vLLM, Transformers, OpenAI, Anthropic, and Gemini adapters.

    Attributes:
        model_name: The name or identifier of the model
        model_type: The type of model engine
        context_length: Maximum context length supported by the model
    """

    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        context_length: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name or identifier of the model
            model_type: Type of model engine
            context_length: Maximum context length (auto-detected if None)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self._context_length = context_length
        self._is_loaded = False
        self._metadata: Dict[str, Any] = {}

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        This method should handle all initialization logic including:
        - Model weight loading
        - Device allocation
        - Tokenizer initialization
        - Configuration setup

        Raises:
            RuntimeError: If model loading fails
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt synchronously.

        Args:
            prompt: Input text prompt
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult containing generated text and metadata

        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid or exceeds context length
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text from a prompt with token-by-token streaming.

        Args:
            prompt: Input text prompt
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Yields:
            str: Individual tokens or chunks of generated text

        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid or exceeds context length
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> TokenCount:
        """
        Count the number of tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object with token count and model name

        Raises:
            RuntimeError: If tokenization fails
        """
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """
        Get the maximum context length supported by the model.

        Returns:
            int: Maximum number of tokens the model can handle
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory and free resources.

        This method should:
        - Free GPU/CPU memory
        - Close any open connections
        - Clean up temporary files
        """
        pass

    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.

        Returns:
            bool: True if model is loaded and ready for inference
        """
        return self._is_loaded

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata and information.

        Returns:
            Dict containing model information such as:
            - Model architecture
            - Parameter count
            - Quantization details
            - Device allocation
            - Memory usage
        """
        return self._metadata

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that a prompt is within context length limits.

        Args:
            prompt: Prompt to validate

        Returns:
            bool: True if prompt is valid

        Raises:
            ValueError: If prompt exceeds context length
        """
        token_count = self.count_tokens(prompt)
        max_length = self.get_context_length()

        if token_count.count > max_length:
            raise ValueError(
                f"Prompt length ({token_count.count} tokens) exceeds "
                f"model context length ({max_length} tokens)"
            )

        return True

    def __enter__(self):
        """Context manager entry."""
        if not self._is_loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"type={self.model_type.value}, "
            f"loaded={self._is_loaded})"
        )


class BatchModel(BaseModel):
    """
    Extended base class for models that support batch processing.

    This class extends BaseModel with batch generation capabilities,
    useful for high-throughput scenarios.
    """

    @abstractmethod
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
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects, one per prompt

        Raises:
            RuntimeError: If batch generation fails
            ValueError: If any prompt is invalid
        """
        pass

    def get_max_batch_size(self) -> int:
        """
        Get the maximum batch size supported.

        Returns:
            int: Maximum number of prompts that can be processed in one batch
        """
        return 1  # Default to no batching


class FunctionCallingModel(BaseModel):
    """
    Extended base class for models that support function/tool calling.

    This class extends BaseModel with function calling capabilities,
    primarily used by API adapters (OpenAI, Anthropic, Gemini).
    """

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text with tool/function calling support.

        Args:
            prompt: Input text prompt
            tools: List of tool definitions in OpenAI function format
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with potential tool calls in metadata

        Raises:
            RuntimeError: If generation fails
            ValueError: If tools are malformed
        """
        pass

    @abstractmethod
    def supports_function_calling(self) -> bool:
        """
        Check if the model supports function calling.

        Returns:
            bool: True if function calling is supported
        """
        pass
