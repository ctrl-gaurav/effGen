"""
Model infrastructure for effGen framework.

This package provides a unified interface for various model backends including:
- vLLM for fast local inference
- HuggingFace Transformers as fallback
- OpenAI API adapter
- Anthropic Claude API adapter
- Google Gemini API adapter

Example:
    >>> from effgen.models import load_model
    >>>
    >>> # Load a local HuggingFace model (tries vLLM, falls back to Transformers)
    >>> model = load_model("meta-llama/Llama-2-7b-hf")
    >>>
    >>> # Load an API model
    >>> gpt4 = load_model("gpt-4")
    >>> claude = load_model("claude-3-opus-20240229")
    >>>
    >>> # Generate text
    >>> result = model.generate("What is the capital of France?")
    >>> print(result.text)
"""

from effgen.models.base import (
    BaseModel,
    BatchModel,
    FunctionCallingModel,
    ModelType,
    GenerationConfig,
    GenerationResult,
    TokenCount,
)

from effgen.models.vllm_engine import VLLMEngine
from effgen.models.transformers_engine import TransformersEngine
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.anthropic_adapter import AnthropicAdapter
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.models.model_loader import ModelLoader, load_model

__all__ = [
    # Base classes
    "BaseModel",
    "BatchModel",
    "FunctionCallingModel",
    "ModelType",

    # Data classes
    "GenerationConfig",
    "GenerationResult",
    "TokenCount",

    # Engine implementations
    "VLLMEngine",
    "TransformersEngine",

    # API adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",

    # Loader
    "ModelLoader",
    "load_model",
]
