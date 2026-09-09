"""
Tests for effgen.models.registry.ProviderRegistry.

Covers: registration idempotency, list_providers, list_models, lookup,
AmbiguousModelError, provider-prefix parsing.
"""

from __future__ import annotations

import pytest

from effgen.models.errors import AmbiguousModelError
from effgen.models.registry import ProviderRegistry, list_models, list_providers, lookup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeAdapter:
    pass


class _OtherAdapter:
    pass


FAKE_MODELS = {
    "alpha-7b": {"context": 8_192, "supports_native_tools": True},
    "beta-13b": {"context": 32_768, "supports_native_tools": False},
}

OVERLAP_MODELS = {
    "alpha-7b": {"context": 4_096},  # same id as FAKE_MODELS["alpha-7b"]
    "gamma-3b": {"context": 2_048},
}


@pytest.fixture(autouse=True)
def reset_registry():
    """Start each test from an empty registry, and hand back a full one.

    The tests here register fake providers and assert on the exact provider
    list, so they need the registry empty. ``reset()`` on teardown puts the
    real providers back, so nothing that runs later sees an empty registry.
    """
    ProviderRegistry.clear()
    yield
    ProviderRegistry.reset()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_single_provider(self):
        ProviderRegistry.register("fakeprov", _FakeAdapter, FAKE_MODELS, env_keys=["FAKE_API_KEY"])
        assert "fakeprov" in ProviderRegistry.list_providers()

    def test_idempotent_registration(self):
        ProviderRegistry.register("fakeprov", _FakeAdapter, FAKE_MODELS, env_keys=["FAKE_API_KEY"])
        ProviderRegistry.register("fakeprov", _FakeAdapter, FAKE_MODELS, env_keys=["FAKE_API_KEY"])
        assert ProviderRegistry.list_providers().count("fakeprov") == 1

    def test_re_registration_overwrites(self):
        ProviderRegistry.register("fakeprov", _FakeAdapter, FAKE_MODELS, env_keys=["OLD_KEY"])
        ProviderRegistry.register("fakeprov", _OtherAdapter, FAKE_MODELS, env_keys=["NEW_KEY"])
        info = ProviderRegistry.get_provider_info("fakeprov")
        assert info["adapter_cls"] is _OtherAdapter
        assert info["env_keys"] == ["NEW_KEY"]

    def test_register_multiple_providers(self):
        ProviderRegistry.register("prov_a", _FakeAdapter, FAKE_MODELS)
        ProviderRegistry.register("prov_b", _OtherAdapter, OVERLAP_MODELS)
        providers = ProviderRegistry.list_providers()
        assert "prov_a" in providers
        assert "prov_b" in providers


# ---------------------------------------------------------------------------
# list_providers / list_models
# ---------------------------------------------------------------------------

class TestListMethods:
    def test_list_providers_sorted(self):
        ProviderRegistry.register("zebra", _FakeAdapter, {})
        ProviderRegistry.register("apple", _OtherAdapter, {})
        providers = list_providers()
        assert providers == sorted(providers)

    def test_list_models_returns_all(self):
        ProviderRegistry.register("fakeprov", _FakeAdapter, FAKE_MODELS)
        models = list_models("fakeprov")
        assert len(models) == len(FAKE_MODELS)
        ids = {m["model_id"] for m in models}
        assert ids == set(FAKE_MODELS)

    def test_list_models_includes_model_id_key(self):
        ProviderRegistry.register("fakeprov", _FakeAdapter, FAKE_MODELS)
        for m in list_models("fakeprov"):
            assert "model_id" in m

    def test_list_models_unknown_provider_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            list_models("nonexistent")


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def setup_method(self):
        ProviderRegistry.register("fakeprov", _FakeAdapter, FAKE_MODELS, env_keys=["FAKE_KEY"])
        ProviderRegistry.register("otherprov", _OtherAdapter, OVERLAP_MODELS)

    def test_lookup_unique_model(self):
        prov, cls, info = lookup("beta-13b")
        assert prov == "fakeprov"
        assert cls is _FakeAdapter
        assert info["context"] == 32_768

    def test_lookup_with_provider_prefix(self):
        prov, cls, info = lookup("fakeprov:alpha-7b")
        assert prov == "fakeprov"
        assert cls is _FakeAdapter

    def test_lookup_explicit_provider_kwarg(self):
        prov, cls, info = lookup("alpha-7b", provider="otherprov")
        assert prov == "otherprov"
        assert cls is _OtherAdapter

    def test_lookup_unknown_model_raises(self):
        with pytest.raises(KeyError):
            lookup("nonexistent-model-xyz")

    def test_lookup_ambiguous_raises(self):
        # "alpha-7b" exists in both fakeprov and otherprov
        with pytest.raises(AmbiguousModelError) as exc_info:
            lookup("alpha-7b")
        err = exc_info.value
        assert err.model_id == "alpha-7b"
        assert "fakeprov" in err.providers
        assert "otherprov" in err.providers

    def test_lookup_ambiguous_resolved_by_prefix(self):
        prov, cls, info = lookup("otherprov:alpha-7b")
        assert prov == "otherprov"

    def test_lookup_nonexistent_provider_prefix_raises(self):
        with pytest.raises(KeyError):
            lookup("unknownprov:alpha-7b")

    def test_lookup_unique_model_in_otherprov(self):
        prov, cls, info = lookup("gamma-3b")
        assert prov == "otherprov"


# ---------------------------------------------------------------------------
# AmbiguousModelError
# ---------------------------------------------------------------------------

class TestAmbiguousModelError:
    def test_error_message_contains_providers(self):
        err = AmbiguousModelError("my-model", ["groq", "together", "fireworks"])
        assert "my-model" in str(err)
        assert "groq" in str(err)
        assert "together" in str(err)
        assert "fireworks" in str(err)

    def test_error_attributes(self):
        err = AmbiguousModelError("model-x", ["prov1", "prov2"])
        assert err.model_id == "model-x"
        assert set(err.providers) == {"prov1", "prov2"}


# ---------------------------------------------------------------------------
# Real adapters self-register on import
# ---------------------------------------------------------------------------

class TestAdapterSelfRegistration:
    def test_all_9_providers_registered_after_imports(self):
        ProviderRegistry.register_builtins()
        providers = ProviderRegistry.list_providers()
        expected = {"anthropic", "cerebras", "fireworks", "gemini", "groq", "hf", "openai", "replicate", "together"}
        assert expected.issubset(set(providers)), f"Missing: {expected - set(providers)}"

    def test_groq_prefix_lookup(self):
        ProviderRegistry.register_builtins()
        prov, cls, info = lookup("groq:openai/gpt-oss-120b")
        assert prov == "groq"
        assert "context" in info
