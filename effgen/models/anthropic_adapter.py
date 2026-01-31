"""
Anthropic Claude API adapter.

This module provides integration with Anthropic's Claude API, supporting:
- Claude 3 (Opus, Sonnet, Haiku) and newer models
- Tool use API
- Extended context (200K tokens)
- Thinking/chain-of-thought extraction
- Cost tracking
- Streaming responses
"""

import logging
from typing import Iterator, Optional, List, Dict, Any
import os

from effgen.models.base import (
    BaseModel,
    FunctionCallingModel,
    ModelType,
    GenerationConfig,
    GenerationResult,
    TokenCount
)

logger = logging.getLogger(__name__)


class AnthropicAdapter(FunctionCallingModel):
    """
    Adapter for Anthropic Claude API models.

    Provides a unified interface for Claude models with support for
    tool use, extended context, and advanced features.

    Features:
    - Support for Claude 3 family (Opus, Sonnet, Haiku) and newer
    - Tool use API integration
    - Extended context windows (up to 200K tokens)
    - Thinking/reasoning extraction
    - Cost tracking and usage monitoring
    - Streaming responses
    - Automatic retries

    Attributes:
        model_name: Anthropic model identifier (e.g., 'claude-3-opus-20240229')
        api_key: Anthropic API key (reads from env if not provided)
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    """

    # Cost per million tokens (input/output) as of 2024
    COST_PER_1M_TOKENS = {
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-3-sonnet-20240229": (3.0, 15.0),
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-3-5-sonnet-20240620": (3.0, 15.0),
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
    }

    # Context lengths for models
    CONTEXT_LENGTHS = {
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-sonnet-20241022": 200000,
    }

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs
    ):
        """
        Initialize Anthropic adapter.

        Args:
            model_name: Anthropic model identifier
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
            **kwargs: Additional Anthropic client parameters
        """
        super().__init__(
            model_name=model_name,
            model_type=ModelType.ANTHROPIC,
            context_length=self.CONTEXT_LENGTHS.get(model_name, 200000)
        )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self.max_retries = max_retries
        self.timeout = timeout
        self.additional_kwargs = kwargs

        self.client = None
        self.total_cost = 0.0
        self.total_tokens = 0

    def load(self) -> None:
        """
        Initialize the Anthropic client.

        Raises:
            RuntimeError: If Anthropic package is not installed
            ValueError: If API key is invalid
        """
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "Anthropic package is not installed. Install it with: "
                "pip install anthropic"
            ) from e

        try:
            logger.info(f"Initializing Anthropic client for model '{self.model_name}'...")

            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }

            client_kwargs.update(self.additional_kwargs)

            self.client = Anthropic(**client_kwargs)

            self._is_loaded = True

            self._metadata = {
                "model_name": self.model_name,
                "context_length": self.get_context_length(),
                "supports_tools": True,
                "supports_streaming": True,
                "supports_vision": "claude-3" in self.model_name,
            }

            logger.info(f"Anthropic client initialized for '{self.model_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise RuntimeError(f"Anthropic initialization failed: {e}") from e

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for the API call.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            float: Estimated cost in USD
        """
        # Find matching cost entry
        cost_entry = None
        for model_id, costs in self.COST_PER_1M_TOKENS.items():
            if self.model_name == model_id:
                cost_entry = costs
                break

        if cost_entry is None:
            logger.warning(f"Unknown cost for model '{self.model_name}', using Sonnet pricing")
            cost_entry = (3.0, 15.0)  # Default to Sonnet pricing

        input_cost = (prompt_tokens / 1_000_000) * cost_entry[0]
        output_cost = (completion_tokens / 1_000_000) * cost_entry[1]

        return input_cost + output_cost

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt
            **kwargs: Additional Anthropic parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If client is not initialized or request fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "max_tokens": config.max_tokens or 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }

            if system_prompt:
                request_params["system"] = system_prompt

            if config.stop_sequences:
                request_params["stop_sequences"] = config.stop_sequences

            # Add any additional kwargs
            request_params.update(kwargs)

            # Make API call
            response = self.client.messages.create(**request_params)

            # Extract response
            generated_text = ""
            thinking_content = []

            for content_block in response.content:
                if content_block.type == "text":
                    generated_text += content_block.text
                elif content_block.type == "thinking":
                    # Extended thinking block (Claude's reasoning)
                    thinking_content.append(content_block.thinking)

            finish_reason = response.stop_reason

            # Get token usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens

            logger.info(
                f"Generated {completion_tokens} tokens. "
                f"Cost: ${cost:.4f}. Total cost: ${self.total_cost:.4f}"
            )

            metadata = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "total_cost": self.total_cost,
            }

            if thinking_content:
                metadata["thinking"] = thinking_content

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            system_prompt: Optional system prompt
            **kwargs: Additional Anthropic parameters

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If client is not initialized or request fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "max_tokens": config.max_tokens or 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }

            if system_prompt:
                request_params["system"] = system_prompt

            if config.stop_sequences:
                request_params["stop_sequences"] = config.stop_sequences

            # Add any additional kwargs
            request_params.update(kwargs)

            # Make streaming API call
            with self.client.messages.stream(**request_params) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text with tool use support.

        Args:
            prompt: Input text prompt
            tools: List of tool definitions in Anthropic tool format
            config: Generation configuration
            system_prompt: Optional system prompt
            **kwargs: Additional Anthropic parameters

        Returns:
            GenerationResult with potential tool uses in metadata

        Raises:
            RuntimeError: If client is not initialized or request fails
            ValueError: If prompt or tools are invalid
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "max_tokens": config.max_tokens or 4096,
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }

            if system_prompt:
                request_params["system"] = system_prompt

            # Add any additional kwargs
            request_params.update(kwargs)

            # Make API call
            response = self.client.messages.create(**request_params)

            # Extract response
            generated_text = ""
            tool_uses = []
            thinking_content = []

            for content_block in response.content:
                if content_block.type == "text":
                    generated_text += content_block.text
                elif content_block.type == "tool_use":
                    tool_uses.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input,
                    })
                elif content_block.type == "thinking":
                    thinking_content.append(content_block.thinking)

            finish_reason = response.stop_reason

            # Get token usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens

            metadata = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "total_cost": self.total_cost,
                "tool_uses": tool_uses,
            }

            if thinking_content:
                metadata["thinking"] = thinking_content

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Anthropic API call with tools failed: {e}")
            raise RuntimeError(f"Generation with tools failed: {e}") from e

    def supports_function_calling(self) -> bool:
        """
        Check if the model supports tool use.

        Returns:
            bool: True (all Claude 3+ models support tool use)
        """
        return "claude-3" in self.model_name

    def count_tokens(self, text: str) -> TokenCount:
        """
        Count tokens in text using Anthropic's token counting.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        try:
            # Use Anthropic's token counting API
            response = self.client.messages.count_tokens(
                model=self.model_name,
                messages=[{"role": "user", "content": text}]
            )

            return TokenCount(
                count=response.input_tokens,
                model_name=self.model_name
            )
        except Exception as e:
            # Fallback to simple estimation (rough approximation)
            logger.warning(f"Token counting API failed: {e}. Using approximation.")
            # Approximate: 1 token ≈ 4 characters for English text
            estimated_tokens = len(text) // 4
            return TokenCount(count=estimated_tokens, model_name=self.model_name)

    def get_context_length(self) -> int:
        """
        Get maximum context length.

        Returns:
            int: Maximum context length in tokens
        """
        return self._context_length

    def get_total_cost(self) -> float:
        """
        Get total cost of all API calls.

        Returns:
            float: Total cost in USD
        """
        return self.total_cost

    def get_total_tokens(self) -> int:
        """
        Get total tokens used across all API calls.

        Returns:
            int: Total token count
        """
        return self.total_tokens

    def reset_usage_stats(self) -> None:
        """
        Reset usage statistics (cost and token count).
        """
        self.total_cost = 0.0
        self.total_tokens = 0
        logger.info("Usage statistics reset")

    def unload(self) -> None:
        """
        Close the Anthropic client connection.
        """
        if self.client is not None:
            logger.info(
                f"Closing Anthropic client. Total cost: ${self.total_cost:.4f}, "
                f"Total tokens: {self.total_tokens}"
            )
            self.client.close()
            self.client = None

        self._is_loaded = False
