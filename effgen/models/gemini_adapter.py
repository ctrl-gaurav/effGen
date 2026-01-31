"""
Google Gemini API adapter.

This module provides integration with Google's Gemini API, supporting:
- Gemini Pro, Flash, and Ultra models
- Multimodal support (text, images, video)
- Function calling
- Grounding with Google Search
- Cost tracking
- Streaming responses
"""

import logging
from typing import Iterator, Optional, List, Dict, Any, Union
import os
from pathlib import Path

from effgen.models.base import (
    BaseModel,
    FunctionCallingModel,
    ModelType,
    GenerationConfig,
    GenerationResult,
    TokenCount
)

logger = logging.getLogger(__name__)


class GeminiAdapter(FunctionCallingModel):
    """
    Adapter for Google Gemini API models.

    Provides a unified interface for Gemini models with support for
    multimodal inputs, function calling, and grounding.

    Features:
    - Support for Gemini Pro, Flash, Ultra models
    - Multimodal inputs (text, images, video, audio)
    - Function calling support
    - Grounding with Google Search
    - Cost tracking and usage monitoring
    - Streaming responses
    - Safety settings configuration

    Attributes:
        model_name: Gemini model identifier (e.g., 'gemini-pro', 'gemini-pro-vision')
        api_key: Google API key (reads from env if not provided)
        safety_settings: Content safety filter settings
    """

    # Cost per 1M tokens (input/output) - Free tier available
    COST_PER_1M_TOKENS = {
        "gemini-1.5-pro": (3.5, 10.5),
        "gemini-1.5-flash": (0.35, 1.05),
        "gemini-pro": (0.5, 1.5),
        "gemini-pro-vision": (0.5, 1.5),
        "gemini-ultra": (10.0, 30.0),
    }

    # Context lengths for models
    CONTEXT_LENGTHS = {
        "gemini-1.5-pro": 1000000,  # 1M tokens
        "gemini-1.5-flash": 1000000,
        "gemini-pro": 32760,
        "gemini-pro-vision": 16384,
        "gemini-ultra": 32760,
    }

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize Gemini adapter.

        Args:
            model_name: Gemini model identifier
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            safety_settings: Content safety filter settings
            **kwargs: Additional Gemini client parameters
        """
        super().__init__(
            model_name=model_name,
            model_type=ModelType.GEMINI,
            context_length=self.CONTEXT_LENGTHS.get(model_name, 32760)
        )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.safety_settings = safety_settings
        self.additional_kwargs = kwargs

        self.client = None
        self.model = None
        self.total_cost = 0.0
        self.total_tokens = 0

    def load(self) -> None:
        """
        Initialize the Gemini client.

        Raises:
            RuntimeError: If google-generativeai package is not installed
            ValueError: If API key is invalid
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai package is not installed. Install it with: "
                "pip install google-generativeai"
            ) from e

        try:
            logger.info(f"Initializing Gemini client for model '{self.model_name}'...")

            # Configure API key
            genai.configure(api_key=self.api_key)

            # Initialize model
            generation_config = {}
            if self.safety_settings:
                generation_config["safety_settings"] = self.safety_settings

            generation_config.update(self.additional_kwargs)

            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                **generation_config
            )

            self.client = genai

            self._is_loaded = True

            self._metadata = {
                "model_name": self.model_name,
                "context_length": self.get_context_length(),
                "supports_functions": True,
                "supports_streaming": True,
                "supports_multimodal": "vision" in self.model_name or "1.5" in self.model_name,
            }

            logger.info(f"Gemini client initialized for '{self.model_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise RuntimeError(f"Gemini initialization failed: {e}") from e

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
            if model_id in self.model_name:
                cost_entry = costs
                break

        if cost_entry is None:
            logger.warning(f"Unknown cost for model '{self.model_name}', using Pro pricing")
            cost_entry = (0.5, 1.5)  # Default to Pro pricing

        input_cost = (prompt_tokens / 1_000_000) * cost_entry[0]
        output_cost = (completion_tokens / 1_000_000) * cost_entry[1]

        return input_cost + output_cost

    def _prepare_content(
        self,
        prompt: Union[str, List[Union[str, Dict[str, Any]]]]
    ) -> Union[str, List]:
        """
        Prepare content for Gemini API.

        Supports both text-only and multimodal inputs.

        Args:
            prompt: Text string or list of content parts (text, images, etc.)

        Returns:
            Formatted content for Gemini API
        """
        if isinstance(prompt, str):
            return prompt

        # Handle multimodal content
        parts = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Handle image, video, or other content types
                if "image" in item:
                    # Load image
                    import PIL.Image
                    image = PIL.Image.open(item["image"])
                    parts.append(image)
                elif "video" in item:
                    # Video content
                    parts.append(item)
                else:
                    parts.append(item)
            else:
                parts.append(item)

        return parts

    def _create_generation_config(
        self,
        config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Create Gemini generation config.

        Args:
            config: Our generation configuration

        Returns:
            Gemini generation config dict
        """
        if config is None:
            config = GenerationConfig()

        gen_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_tokens or 2048,
        }

        if config.stop_sequences:
            gen_config["stop_sequences"] = config.stop_sequences

        return gen_config

    def generate(
        self,
        prompt: Union[str, List[Union[str, Dict[str, Any]]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text or multimodal content
            config: Generation configuration
            **kwargs: Additional Gemini parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If client is not initialized or request fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        if isinstance(prompt, str):
            self.validate_prompt(prompt)

        generation_config = self._create_generation_config(config)
        content = self._prepare_content(prompt)

        try:
            # Generate content
            response = self.model.generate_content(
                content,
                generation_config=generation_config,
                **kwargs
            )

            # Extract response
            generated_text = response.text

            # Get token usage
            try:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
            except AttributeError:
                # Fallback if usage metadata not available
                logger.warning("Usage metadata not available, using estimates")
                prompt_tokens = self.count_tokens(
                    prompt if isinstance(prompt, str) else str(prompt)
                ).count
                completion_tokens = self.count_tokens(generated_text).count
                total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens

            logger.info(
                f"Generated {completion_tokens} tokens. "
                f"Cost: ${cost:.4f}. Total cost: ${self.total_cost:.4f}"
            )

            # Get finish reason
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

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
                    "safety_ratings": getattr(response, "safety_ratings", None),
                }
            )

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: Union[str, List[Union[str, Dict[str, Any]]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text or multimodal content
            config: Generation configuration
            **kwargs: Additional Gemini parameters

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If client is not initialized or request fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        if isinstance(prompt, str):
            self.validate_prompt(prompt)

        generation_config = self._create_generation_config(config)
        content = self._prepare_content(prompt)

        try:
            # Generate content with streaming
            response = self.model.generate_content(
                content,
                generation_config=generation_config,
                stream=True,
                **kwargs
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text with function calling support.

        Args:
            prompt: Input text prompt
            tools: List of tool definitions in Gemini function format
            config: Generation configuration
            **kwargs: Additional Gemini parameters

        Returns:
            GenerationResult with potential function calls in metadata

        Raises:
            RuntimeError: If client is not initialized or request fails
            ValueError: If prompt or tools are invalid
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        generation_config = self._create_generation_config(config)

        try:
            # Convert tools to Gemini format
            from google.generativeai.types import FunctionDeclaration, Tool

            function_declarations = []
            for tool in tools:
                if "function" in tool:
                    func = tool["function"]
                    function_declarations.append(
                        FunctionDeclaration(
                            name=func["name"],
                            description=func.get("description", ""),
                            parameters=func.get("parameters", {}),
                        )
                    )

            gemini_tools = [Tool(function_declarations=function_declarations)]

            # Generate content with tools
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                tools=gemini_tools,
                **kwargs
            )

            # Extract response
            generated_text = response.text if hasattr(response, "text") else ""

            # Extract function calls
            function_calls = []
            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        fc = part.function_call
                        function_calls.append({
                            "name": fc.name,
                            "arguments": dict(fc.args),
                        })

            # Get token usage
            try:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
            except AttributeError:
                prompt_tokens = self.count_tokens(prompt).count
                completion_tokens = self.count_tokens(generated_text).count if generated_text else 0
                total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens

            # Get finish reason
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

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
                    "function_calls": function_calls,
                }
            )

        except Exception as e:
            logger.error(f"Gemini API call with tools failed: {e}")
            raise RuntimeError(f"Generation with tools failed: {e}") from e

    def supports_function_calling(self) -> bool:
        """
        Check if the model supports function calling.

        Returns:
            bool: True for most Gemini models
        """
        return True

    def count_tokens(self, text: str) -> TokenCount:
        """
        Count tokens in text using Gemini's token counting.

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
            # Use Gemini's token counting
            result = self.model.count_tokens(text)
            return TokenCount(count=result.total_tokens, model_name=self.model_name)
        except Exception as e:
            # Fallback to estimation
            logger.warning(f"Token counting failed: {e}. Using approximation.")
            # Approximate: 1 token ≈ 4 characters
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
        Close the Gemini client connection.
        """
        if self.model is not None:
            logger.info(
                f"Closing Gemini client. Total cost: ${self.total_cost:.4f}, "
                f"Total tokens: {self.total_tokens}"
            )
            self.model = None
            self.client = None

        self._is_loaded = False
