"""Every engine says whether its provider has a prompt cache, or says nothing.

A provider only serves a prompt prefix from its cache if the request reaches it
in a form it recognises, and what that takes differs: some match the rendered
prefix themselves, some want the caller to mark where the stable part ends.
Nothing in the run may decide that from a provider name or a model id, so the
adapter declares it and the run reads the declaration.

Offline: no adapter is constructed with a key and no request is made. What is
asserted is the declaration itself.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import effgen.models
from effgen.models.base import BaseModel, ModelType, PromptCachePolicy
from effgen.models.lazy import LazyModel


def _engine_classes() -> list[type[BaseModel]]:
    """Every concrete ``BaseModel`` subclass the package ships."""
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.iter_modules(effgen.models.__path__):
        try:
            module = importlib.import_module(f"effgen.models.{info.name}")
        except Exception:  # noqa: BLE001 - an engine whose SDK is absent is skipped
            continue
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == module.__name__
            ):
                found[f"{obj.__module__}.{name}"] = obj
    return list(found.values())


def test_the_default_is_no_cache():
    """An engine that says nothing caches nothing, so its requests never change."""
    assert BaseModel.prompt_cache_policy(object()) is None  # type: ignore[arg-type]
    assert BaseModel.supports_suppressed_tool_call(object()) is False  # type: ignore[arg-type]


def test_every_engine_either_declares_a_policy_or_declares_none():
    """No engine may answer with something that is not a policy."""
    checked = 0
    for cls in _engine_classes():
        method = getattr(cls, "prompt_cache_policy", None)
        assert callable(method), f"{cls.__name__} lost prompt_cache_policy"
        checked += 1
        if inspect.isabstract(cls):
            # An intermediate base declares nothing of its own; the concrete
            # engines under it are each checked in their own right.
            continue
        stub = cls.__new__(cls)
        stub.model_name = "m"
        stub.base_url = None
        stub._catalog_backed = False
        try:
            policy = cls.prompt_cache_policy(stub)
        except Exception:  # noqa: BLE001 - an engine needing a live client is read statically
            continue
        assert policy is None or isinstance(policy, PromptCachePolicy), (
            f"{cls.__name__} answered with {policy!r}"
        )
        if policy is not None:
            assert policy.style in ("automatic", "explicit")
            if policy.style == "explicit":
                assert policy.max_breakpoints >= 1, (
                    f"{cls.__name__} places breakpoints but allows none"
                )
            else:
                assert policy.max_breakpoints == 0
            if policy.min_prefix_tokens is not None:
                assert policy.min_prefix_tokens > 0
    assert checked > 5, "the engine sweep found almost nothing"


def test_the_cloud_adapters_declare_what_they_read():
    """Each provider effGen reads a cached-token count from says it does."""
    from effgen.models.anthropic_adapter import AnthropicAdapter
    from effgen.models.cerebras_adapter import CerebrasAdapter
    from effgen.models.fireworks_adapter import FireworksAdapter
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.models.groq_adapter import GroqAdapter
    from effgen.models.hf_inference_adapter import HFInferenceAdapter
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.models.together_adapter import TogetherAdapter

    for cls, model, style in (
        (OpenAIAdapter, "gpt-4o-mini", "automatic"),
        (AnthropicAdapter, "claude-sonnet-4-6", "explicit"),
        (GeminiAdapter, "gemini-2.5-flash", "automatic"),
        (GroqAdapter, "openai/gpt-oss-20b", "automatic"),
        (TogetherAdapter, "any", "automatic"),
        (FireworksAdapter, "any", "automatic"),
        (CerebrasAdapter, "any", "automatic"),
        (HFInferenceAdapter, "any", "automatic"),
    ):
        stub: Any = cls.__new__(cls)
        stub.model_name = model
        stub.base_url = None
        stub._catalog_backed = True
        policy = cls.prompt_cache_policy(stub)
        assert policy is not None, f"{cls.__name__} declares no policy"
        assert policy.style == style, f"{cls.__name__} declares {policy.style}"
        assert policy.reports_cached_tokens, f"{cls.__name__} reads the hit but hides it"

    stub = AnthropicAdapter.__new__(AnthropicAdapter)
    stub.model_name = "claude-sonnet-4-6"
    policy = AnthropicAdapter.prompt_cache_policy(stub)
    assert policy.reports_cache_writes
    assert policy.max_breakpoints == 4
    assert "5m" in policy.ttl_options


def test_an_endpoint_the_caller_points_at_claims_no_minimum():
    """A server the caller runs has its own rules; none is claimed for it."""
    from effgen.models.openai_adapter import OpenAIAdapter

    hosted: Any = OpenAIAdapter.__new__(OpenAIAdapter)
    hosted.model_name = "gpt-4o-mini"
    hosted.base_url = None
    hosted._catalog_backed = True
    own: Any = OpenAIAdapter.__new__(OpenAIAdapter)
    own.model_name = "a-model-only-that-server-serves"
    own.base_url = "http://127.0.0.1:8400/v1"
    own._catalog_backed = True

    assert OpenAIAdapter.prompt_cache_policy(hosted).min_prefix_tokens == 1024
    assert OpenAIAdapter.prompt_cache_policy(own).min_prefix_tokens is None


def test_lazy_forwards_the_declarations_it_wraps():
    """A wrapped adapter keeps every capability it declared."""

    class Declaring(BaseModel):
        def __init__(self) -> None:
            super().__init__(model_name="declaring", model_type=ModelType.OPENAI)
            self._is_loaded = True

        def load(self) -> None:
            self._is_loaded = True

        def unload(self) -> None:
            self._is_loaded = False

        def generate(self, prompt, config=None, **kwargs): ...
        def generate_stream(self, prompt, config=None, **kwargs): ...
        def count_tokens(self, text): ...
        def get_context_length(self) -> int: return 8192
        def supports_tool_calling(self) -> bool: return True
        def supports_forced_tool_call(self) -> bool: return True
        def supports_suppressed_tool_call(self) -> bool: return True
        def prompt_cache_policy(self) -> PromptCachePolicy:
            return PromptCachePolicy(style="automatic", reports_cached_tokens=True)

    inner = Declaring()
    wrapped = LazyModel(inner, idle_timeout=None)
    assert wrapped.prompt_cache_policy() == inner.prompt_cache_policy()
    assert wrapped.supports_suppressed_tool_call() is True
    assert wrapped.supports_forced_tool_call() is True
