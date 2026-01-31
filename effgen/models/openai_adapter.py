"""
OpenAI API adapter for GPT models.

This module provides integration with OpenAI's API, supporting:
- GPT-3.5, GPT-4, GPT-4-turbo models
- Function/tool calling
- Streaming responses
- Automatic retries with exponential backoff
- Cost tracking
- Rate limiting
"""

import logging
import time
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


class OpenAIAdapter(FunctionCallingModel):
    """
    Adapter for OpenAI API models.

    Provides a unified interface for OpenAI's GPT models with support for
    function calling, streaming, and comprehensive error handling.

    Features:
    - Support for GPT-3.5, GPT-4, GPT-4-turbo
    - Function/tool calling
    - Streaming responses
    - Automatic retries with exponential backoff
    - Cost tracking and usage monitoring
    - Rate limit handling

    Attributes:
        model_name: OpenAI model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')
        api_key: OpenAI API key (reads from env if not provided)
        organization_id: OpenAI organization ID (optional)
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    """

    # Cost per 1K tokens (input/output) as of 2024
    COST_PER_1K_TOKENS = {
        "gpt-4-turbo-preview": (0.01, 0.03),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-3.5-turbo-16k": (0.003, 0.004),
    }

    # Context lengths for models
    CONTEXT_LENGTHS = {
        "gpt-4-turbo-preview": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
    }

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs
    ):
        """
        Initialize OpenAI adapter.

        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization_id: OpenAI organization ID
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
            **kwargs: Additional OpenAI client parameters
        """
        super().__init__(
            model_name=model_name,
            model_type=ModelType.OPENAI,
            context_length=self.CONTEXT_LENGTHS.get(model_name, 4096)
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.organization_id = organization_id or os.getenv("OPENAI_ORG_ID")
        self.max_retries = max_retries
        self.timeout = timeout
        self.additional_kwargs = kwargs

        self.client = None
        self.total_cost = 0.0
        self.total_tokens = 0

    def load(self) -> None:
        """
        Initialize the OpenAI client.

        Raises:
            RuntimeError: If OpenAI package is not installed
            ValueError: If API key is invalid
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI package is not installed. Install it with: "
                "pip install openai"
            ) from e

        try:
            logger.info(f"Initializing OpenAI client for model '{self.model_name}'...")

            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }

            if self.organization_id:
                client_kwargs["organization"] = self.organization_id

            client_kwargs.update(self.additional_kwargs)

            self.client = OpenAI(**client_kwargs)

            # Test the connection
            try:
                self.client.models.retrieve(self.model_name)
            except Exception as e:
                logger.warning(f"Could not verify model '{self.model_name}': {e}")

            self._is_loaded = True

            self._metadata = {
                "model_name": self.model_name,
                "context_length": self.get_context_length(),
                "supports_functions": True,
                "supports_streaming": True,
            }

            logger.info(f"OpenAI client initialized for '{self.model_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"OpenAI initialization failed: {e}") from e

    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Convert prompt to OpenAI messages format.

        Args:
            prompt: Input prompt

        Returns:
            List of message dicts
        """
        return [{"role": "user", "content": prompt}]

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for the API call.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            float: Estimated cost in USD
        """
        # Find matching cost entry (handle model variants)
        cost_entry = None
        for model_prefix, costs in self.COST_PER_1K_TOKENS.items():
            if self.model_name.startswith(model_prefix):
                cost_entry = costs
                break

        if cost_entry is None:
            logger.warning(f"Unknown cost for model '{self.model_name}', using default")
            cost_entry = (0.01, 0.03)  # Default to GPT-4-turbo pricing

        input_cost = (prompt_tokens / 1000) * cost_entry[0]
        output_cost = (completion_tokens / 1000) * cost_entry[1]

        return input_cost + output_cost

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
            **kwargs: Additional OpenAI parameters

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

        messages = self._create_messages(prompt)

        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "presence_penalty": config.presence_penalty,
                "frequency_penalty": config.frequency_penalty,
            }

            if config.stop_sequences:
                request_params["stop"] = config.stop_sequences

            if config.seed is not None:
                request_params["seed"] = config.seed

            # Add any additional kwargs
            request_params.update(kwargs)

            # Make API call
            response = self.client.chat.completions.create(**request_params)

            # Extract response
            choice = response.choices[0]
            generated_text = choice.message.content or ""
            finish_reason = choice.finish_reason

            # Get token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens

            logger.info(
                f"Generated {completion_tokens} tokens. "
                f"Cost: ${cost:.4f}. Total cost: ${self.total_cost:.4f}"
            )

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "total_cost": self.total_cost,
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
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
            **kwargs: Additional OpenAI parameters

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

        messages = self._create_messages(prompt)

        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "presence_penalty": config.presence_penalty,
                "frequency_penalty": config.frequency_penalty,
                "stream": True,
            }

            if config.stop_sequences:
                request_params["stop"] = config.stop_sequences

            if config.seed is not None:
                request_params["seed"] = config.seed

            # Add any additional kwargs
            request_params.update(kwargs)

            # Make streaming API call
            stream = self.client.chat.completions.create(**request_params)

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

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
            config: Generation configuration
            **kwargs: Additional OpenAI parameters

        Returns:
            GenerationResult with potential tool calls in metadata

        Raises:
            RuntimeError: If client is not initialized or request fails
            ValueError: If prompt or tools are invalid
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        messages = self._create_messages(prompt)

        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "tools": tools,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
            }

            # Add any additional kwargs
            request_params.update(kwargs)

            # Make API call
            response = self.client.chat.completions.create(**request_params)

            # Extract response
            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            # Get token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens

            # Extract tool calls if present
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })

            generated_text = message.content or ""

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "total_cost": self.total_cost,
                    "tool_calls": tool_calls,
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API call with tools failed: {e}")
            raise RuntimeError(f"Generation with tools failed: {e}") from e

    def supports_function_calling(self) -> bool:
        """
        Check if the model supports function calling.

        Returns:
            bool: True (all supported OpenAI models support function calling)
        """
        return True

    def count_tokens(self, text: str) -> TokenCount:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object

        Raises:
            RuntimeError: If tiktoken is not available
        """
        try:
            import tiktoken
        except ImportError as e:
            raise RuntimeError(
                "tiktoken is not installed. Install it with: pip install tiktoken"
            ) from e

        try:
            # Get encoding for model
            encoding = tiktoken.encoding_for_model(self.model_name)
            tokens = encoding.encode(text)
            return TokenCount(count=len(tokens), model_name=self.model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
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
        Close the OpenAI client connection.
        """
        if self.client is not None:
            logger.info(
                f"Closing OpenAI client. Total cost: ${self.total_cost:.4f}, "
                f"Total tokens: {self.total_tokens}"
            )
            self.client.close()
            self.client = None

        self._is_loaded = False
