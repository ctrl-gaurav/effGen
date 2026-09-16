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
    >>> model = load_model("Qwen/Qwen2.5-1.5B-Instruct")
    >>>
    >>> # Load an API model
    >>> gpt = load_model("openai:gpt-5-nano")
    >>> gemini = load_model("gemini:gemini-3.1-flash-lite")
    >>>
    >>> # Generate text
    >>> result = model.generate("What is the capital of France?")
    >>> print(result.text)
"""

from typing import Any

from effgen.models._catalog import (
    ModelRecord,
    nearest_alternatives,
    snapshot_age_days,
    stale_providers,
    warn_if_stale,
)
from effgen.models._catalog import (
    build_records as build_catalog_records,
)
from effgen.models._catalog import (
    list_models as list_catalog_models,
)
from effgen.models._catalog import (
    load_snapshot as load_catalog_snapshot,
)
from effgen.models._catalog import (
    lookup as lookup_catalog_model,
)
from effgen.models._catalog import (
    save_snapshot as save_catalog_snapshot,
)
from effgen.models._cost import CostTracker
from effgen.models._cost_store import SQLiteCostStore
from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models._rate_limit_store import SQLiteRateLimitStore
from effgen.models.anthropic_adapter import AnthropicAdapter, StreamChunk
from effgen.models.anthropic_cache import (
    MAX_CACHE_BREAKPOINTS,
    apply_cache_to_last_tool,
    apply_cache_to_system,
    get_min_cache_tokens,
    mark_cached,
)
from effgen.models.anthropic_models import ANTHROPIC_MODELS
from effgen.models.anthropic_models import get_model_info as get_anthropic_model_info
from effgen.models.auth import check_keys
from effgen.models.base import (
    BaseModel,
    BatchModel,
    FunctionCallingModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    PromptCachePolicy,
    TokenCount,
)
from effgen.models.batching import ContinuousBatcher
from effgen.models.capabilities import (
    MODEL_CAPABILITIES,
    Capability,
    ModelCapability,
    estimate_capability,
    get_model_capability,
    list_registered_models,
    register_model_capability,
)
from effgen.models.cerebras_adapter import CerebrasAdapter
from effgen.models.errors import (
    AllCandidatesExhaustedError,
    AmbiguousModelError,
    BudgetExceededError,
    ErrorClass,
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelRefusalError,
    ModelTimeoutError,
    ModelUnavailableError,
    NoCandidateWithinBudgetError,
    ProviderTransientError,
    classify_provider_error,
)
from effgen.models.fireworks_adapter import FireworksAdapter
from effgen.models.fireworks_models import FIREWORKS_MODELS
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.models.gemini_files import FileRef, upload_file
from effgen.models.groq_adapter import GroqAdapter
from effgen.models.groq_models import GROQ_MODELS
from effgen.models.hf_inference_adapter import HFInferenceAdapter
from effgen.models.hf_inference_models import HF_MODELS
from effgen.models.latency_tracker import LatencyTracker
from effgen.models.lazy import LazyModel
from effgen.models.model_loader import ModelLoader, load_model
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter
from effgen.models.openai_schema import to_openai_schema
from effgen.models.pool import ModelPool, PoolConfig
from effgen.models.registry import ProviderRegistry, list_models, list_providers, lookup
from effgen.models.replicate_adapter import ReplicateAdapter
from effgen.models.replicate_models import REPLICATE_MODELS
from effgen.models.router import (
    ComplexityEstimate,
    ComplexityLevel,
    ModelRouter,
    NoCandidateError,
    PolicyBasedRouter,
    ProviderModelPair,
    RouterDecision,
    RouterEvent,
    RoutingConfig,
    RoutingContext,
    RoutingDecision,
    RoutingPolicy,
    estimate_complexity,
)
from effgen.models.routing.cost import CostBasedPolicy
from effgen.models.routing.first_available import FirstAvailablePolicy
from effgen.models.routing.latency import LatencyBasedPolicy
from effgen.models.routing.retry import RetryPolicy
from effgen.models.together_adapter import TogetherAdapter
from effgen.models.together_models import TOGETHER_MODELS

# Local-inference engines (Transformers, vLLM, MLX) are resolved lazily by the
# ``__getattr__`` below: they pull torch / transformers / vLLM, so importing
# ``effgen.models`` (which happens transitively whenever any model submodule is
# imported, e.g. via ``from effgen import Agent``) must not load them. They are
# imported only when the engine class itself is accessed or a local model is
# loaded. MLX engines are additionally Apple-Silicon-only.
_LAZY_ENGINES: dict[str, tuple[str, str]] = {
    "TransformersEngine": ("effgen.models.transformers_engine", "TransformersEngine"),
    "VLLMEngine": ("effgen.models.vllm_engine", "VLLMEngine"),
    "MLXEngine": ("effgen.models.mlx_engine", "MLXEngine"),
    "MLXVLMEngine": ("effgen.models.mlx_vlm_engine", "MLXVLMEngine"),
    "MLXVLMAdapter": ("effgen.models.mlx_vlm", "MLXVLMAdapter"),
    "MLX_VLM_MODELS": ("effgen.models.mlx_vlm", "RECOMMENDED_MODELS"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ENGINES.get(name)
    if target is None:
        raise AttributeError(f"module 'effgen.models' has no attribute {name!r}")
    module_path, attr = target
    from importlib import import_module
    try:
        value = getattr(import_module(module_path), attr)
    except ImportError as exc:
        # MLX is Apple-only; surface as a normal missing attribute on other
        # platforms rather than crashing the import.
        raise AttributeError(
            f"module 'effgen.models' has no attribute {name!r} "
            f"(optional engine '{module_path}' is unavailable: {exc})"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_ENGINES) | set(globals()))


__all__ = [
    # Base classes
    "BaseModel",
    "BatchModel",
    "FunctionCallingModel",
    "ModelType",

    # Data classes
    "GenerationConfig",
    "GenerationResult",
    "PromptCachePolicy",
    "TokenCount",

    # Engine implementations
    "VLLMEngine",
    "TransformersEngine",
    "MLXEngine",
    "MLXVLMEngine",
    "MLXVLMAdapter",
    "MLX_VLM_MODELS",

    # API adapters
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "CerebrasAdapter",
    "GroqAdapter",
    "TogetherAdapter",
    "FireworksAdapter",
    "ReplicateAdapter",
    "HFInferenceAdapter",

    # Model registries
    "GROQ_MODELS",
    "TOGETHER_MODELS",
    "FIREWORKS_MODELS",
    "REPLICATE_MODELS",
    "HF_MODELS",

    # Anthropic streaming
    "StreamChunk",

    # Loader
    "ModelLoader",
    "load_model",

    # Complexity-based router (v0.2.3, back-compat)
    "ModelRouter",
    "RoutingConfig",
    "RoutingDecision",
    "ComplexityEstimate",
    "ComplexityLevel",
    "estimate_complexity",

    # Policy-based router (v0.2.4+)
    "PolicyBasedRouter",
    "RoutingPolicy",
    "RoutingContext",
    "RouterDecision",
    "RouterEvent",
    "ProviderModelPair",
    "NoCandidateError",
    "FirstAvailablePolicy",
    "CostBasedPolicy",
    "LatencyBasedPolicy",
    "RetryPolicy",

    # Latency + cost tracking (v0.2.4+)
    "LatencyTracker",
    "CostTracker",
    "SQLiteCostStore",

    # Capabilities
    "Capability",
    "ModelCapability",
    "MODEL_CAPABILITIES",
    "register_model_capability",
    "get_model_capability",
    "estimate_capability",
    "list_registered_models",

    # Pool
    "ModelPool",
    "PoolConfig",

    "LazyModel",
    "ContinuousBatcher",

    # Rate-limit coordination
    "RateLimitCoordinator",
    "SQLiteRateLimitStore",

    # Errors
    "ModelRefusalError",
    "ModelAuthError",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "ModelNotFoundError",
    "AmbiguousModelError",
    "NoCandidateWithinBudgetError",
    "ProviderTransientError",
    "AllCandidatesExhaustedError",
    "InvalidRequestError",
    "BudgetExceededError",
    "ErrorClass",
    "classify_provider_error",

    # Provider registry + auth
    "ProviderRegistry",
    "list_providers",
    "list_models",
    "lookup",
    "check_keys",

    # Normalized, drift-aware model catalog (uniform across providers)
    "ModelRecord",
    "list_catalog_models",
    "lookup_catalog_model",
    "build_catalog_records",
    "nearest_alternatives",
    "stale_providers",
    "warn_if_stale",
    "snapshot_age_days",
    "load_catalog_snapshot",
    "save_catalog_snapshot",

    # Schema helpers
    "to_openai_schema",

    # Anthropic registry
    "ANTHROPIC_MODELS",
    "get_anthropic_model_info",

    # Anthropic cache helpers
    "mark_cached",
    "apply_cache_to_system",
    "apply_cache_to_last_tool",
    "get_min_cache_tokens",
    "MAX_CACHE_BREAKPOINTS",

    # Gemini Files API
    "FileRef",
    "upload_file",
]
