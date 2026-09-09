"""Unit tests for CostBasedPolicy and pricing registry.

Tests cover:
- Cheapest candidate wins.
- Free-tier providers rank ahead of equally-priced paid providers.
- Budget guard raises NoCandidateWithinBudgetError when no candidate fits.
- Provider pricing registry correctly populated.
- Per-model pricing overrides provider-level defaults.
"""

from __future__ import annotations

import pytest

from effgen.models.capabilities import Capability
from effgen.models.errors import NoCandidateWithinBudgetError
from effgen.models.registry import ProviderRegistry
from effgen.models.router import NoCandidateError, ProviderModelPair, RoutingContext
from effgen.models.routing.cost import CostBasedPolicy, _get_model_pricing

# ---------------------------------------------------------------------------
# Fixtures — isolated registry per test
# ---------------------------------------------------------------------------

class _DummyAdapter:
    pass


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Run each test with a fresh ProviderRegistry."""
    snapshot = ProviderRegistry.snapshot()
    ProviderRegistry.clear()
    yield
    ProviderRegistry.restore(snapshot)


def _reg(
    name: str,
    models: dict,
    *,
    key_env: str | None = None,
    caps: set | None = None,
    pricing: dict | None = None,
    monkeypatch,
):
    """Register a provider and optionally set its env key."""
    if key_env:
        monkeypatch.setenv(key_env, "dummy-key")
    ProviderRegistry.register(
        name,
        _DummyAdapter,
        models,
        env_keys=[key_env] if key_env else [],
        capabilities=caps or {Capability.chat},
        pricing=pricing or {"input_per_1m": 1.0, "output_per_1m": 1.0, "free_tier": False},
    )


# ---------------------------------------------------------------------------
# Tests: cheapest wins
# ---------------------------------------------------------------------------

def test_cheapest_wins(monkeypatch):
    """CostBasedPolicy selects the lowest-cost provider."""
    _reg("cheap", {"cheap-model": {}}, key_env="CHEAP_KEY",
         pricing={"input_per_1m": 0.1, "output_per_1m": 0.1, "free_tier": False},
         monkeypatch=monkeypatch)
    _reg("expensive", {"exp-model": {}}, key_env="EXP_KEY",
         pricing={"input_per_1m": 5.0, "output_per_1m": 5.0, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    candidates = [
        ProviderModelPair("cheap", "cheap-model"),
        ProviderModelPair("expensive", "exp-model"),
    ]
    ctx = RoutingContext(prompt_tokens_estimate=1000, required_capabilities={Capability.chat})
    decision = policy.select(candidates, ctx)
    assert decision.chosen.provider == "cheap"
    assert decision.policy_name == "cost_based"


def test_cheapest_wins_reversed_order(monkeypatch):
    """Order of candidates list does not affect cheapest selection."""
    _reg("cheap", {"cheap-model": {}}, key_env="CHEAP_KEY",
         pricing={"input_per_1m": 0.1, "output_per_1m": 0.1, "free_tier": False},
         monkeypatch=monkeypatch)
    _reg("expensive", {"exp-model": {}}, key_env="EXP_KEY",
         pricing={"input_per_1m": 5.0, "output_per_1m": 5.0, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    # expensive listed first
    candidates = [
        ProviderModelPair("expensive", "exp-model"),
        ProviderModelPair("cheap", "cheap-model"),
    ]
    ctx = RoutingContext(prompt_tokens_estimate=1000, required_capabilities={Capability.chat})
    decision = policy.select(candidates, ctx)
    assert decision.chosen.provider == "cheap"


# ---------------------------------------------------------------------------
# Tests: free-tier tie-breaking
# ---------------------------------------------------------------------------

def test_free_tier_beats_paid_at_equal_cost(monkeypatch):
    """When two providers have equal cost, free-tier ranks first."""
    _reg("paid_zero", {"pm": {}}, key_env="PAID_KEY",
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": False},
         monkeypatch=monkeypatch)
    _reg("free_zero", {"fm": {}}, key_env="FREE_KEY",
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    candidates = [
        ProviderModelPair("paid_zero", "pm"),
        ProviderModelPair("free_zero", "fm"),
    ]
    ctx = RoutingContext(prompt_tokens_estimate=100, required_capabilities={Capability.chat})
    decision = policy.select(candidates, ctx)
    assert decision.chosen.provider == "free_zero"


def test_free_tier_provider_routes_at_zero_effective_cost(monkeypatch):
    """Groq-style free tiers route at zero while retaining paid list prices."""
    _reg("groq", {
        "openai/gpt-oss-20b": {
            "pricing_per_1m_input": 0.05,
            "pricing_per_1m_output": 0.08,
        }
    }, key_env="GROQ_KEY",
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    ctx = RoutingContext(
        prompt_tokens_estimate=1000,
        user_budget_usd=0.0,
        required_capabilities={Capability.chat},
    )
    decision = policy.select([ProviderModelPair("groq", "openai/gpt-oss-20b")], ctx)
    assert decision.chosen.provider == "groq"
    assert decision.score == pytest.approx(0.0)


def test_model_free_tier_false_overrides_provider_free_tier(monkeypatch):
    """Restricted models do not tie-break as free even on free-tier providers."""
    _reg("cerebras", {
        "restricted": {"free_tier": False},
        "callable": {"free_tier": True},
    }, key_env="CEREBRAS_KEY",
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    ctx = RoutingContext(required_capabilities={Capability.chat})
    decision = policy.select([
        ProviderModelPair("cerebras", "restricted"),
        ProviderModelPair("cerebras", "callable"),
    ], ctx)
    assert decision.chosen == ProviderModelPair("cerebras", "callable")


def test_free_tier_tiebreak_is_deterministic(monkeypatch):
    """Same context and candidates always choose the documented provider priority."""
    _reg("hf", {"hf-model": {}}, key_env="HF_KEY",
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
         monkeypatch=monkeypatch)
    _reg("groq", {"groq-model": {}}, key_env="GROQ_KEY",
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    candidates = [
        ProviderModelPair("hf", "hf-model"),
        ProviderModelPair("groq", "groq-model"),
    ]
    ctx = RoutingContext(prompt_tokens_estimate=100, required_capabilities={Capability.chat})
    choices = [policy.select(candidates, ctx).chosen for _ in range(5)]
    assert choices == [ProviderModelPair("groq", "groq-model")] * 5


def test_free_tier_prefers_known_published_list_price(monkeypatch):
    """Known paid list prices beat unknown prices inside the same free tier."""
    _reg("groq", {
        "unknown-price": {},
        "known-cheap": {
            "pricing_per_1m_input": 0.05,
            "pricing_per_1m_output": 0.08,
        },
    }, key_env="GROQ_KEY",
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    ctx = RoutingContext(prompt_tokens_estimate=100, required_capabilities={Capability.chat})
    decision = policy.select([
        ProviderModelPair("groq", "unknown-price"),
        ProviderModelPair("groq", "known-cheap"),
    ], ctx)
    assert decision.chosen == ProviderModelPair("groq", "known-cheap")


# ---------------------------------------------------------------------------
# Tests: capability filtering
# ---------------------------------------------------------------------------

def test_capability_filtering_eliminates_providers(monkeypatch):
    """Providers missing required capabilities are eliminated."""
    _reg("no-tools", {"m1": {}}, key_env="K1",
         caps={Capability.chat},
         pricing={"input_per_1m": 0.01, "output_per_1m": 0.01, "free_tier": False},
         monkeypatch=monkeypatch)
    _reg("has-tools", {"m2": {}}, key_env="K2",
         caps={Capability.chat, Capability.tools},
         pricing={"input_per_1m": 5.0, "output_per_1m": 5.0, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    candidates = [
        ProviderModelPair("no-tools", "m1"),
        ProviderModelPair("has-tools", "m2"),
    ]
    ctx = RoutingContext(
        prompt_tokens_estimate=100,
        required_capabilities={Capability.chat, Capability.tools},
    )
    decision = policy.select(candidates, ctx)
    # no-tools is eliminated despite being cheaper
    assert decision.chosen.provider == "has-tools"
    reasons = dict(decision.eliminated)
    assert any("missing capabilities" in r for r in reasons.values())


# ---------------------------------------------------------------------------
# Tests: budget guard
# ---------------------------------------------------------------------------

def test_budget_guard_raises_when_no_candidate_fits(monkeypatch):
    """NoCandidateWithinBudgetError raised when cheapest cost > budget."""
    _reg("pricey", {"m": {}}, key_env="K",
         pricing={"input_per_1m": 10.0, "output_per_1m": 10.0, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    candidates = [ProviderModelPair("pricey", "m")]
    ctx = RoutingContext(
        prompt_tokens_estimate=1_000_000,  # 1M tokens → $10 minimum
        user_budget_usd=0.001,
        required_capabilities={Capability.chat},
    )
    with pytest.raises(NoCandidateWithinBudgetError) as exc_info:
        policy.select(candidates, ctx)

    err = exc_info.value
    assert err.user_budget_usd == pytest.approx(0.001)
    assert err.cheapest_cost_usd > 0.001
    assert err.cheapest_pair == ("pricey", "m")


def test_budget_guard_error_includes_cheapest_pair(monkeypatch):
    """NoCandidateWithinBudgetError.cheapest_pair identifies the cheapest option."""
    _reg("p1", {"m1": {}}, key_env="K1",
         pricing={"input_per_1m": 5.0, "output_per_1m": 5.0, "free_tier": False},
         monkeypatch=monkeypatch)
    _reg("p2", {"m2": {}}, key_env="K2",
         pricing={"input_per_1m": 2.0, "output_per_1m": 2.0, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    candidates = [
        ProviderModelPair("p1", "m1"),
        ProviderModelPair("p2", "m2"),
    ]
    ctx = RoutingContext(
        prompt_tokens_estimate=1_000_000,
        user_budget_usd=0.001,
        required_capabilities={Capability.chat},
    )
    with pytest.raises(NoCandidateWithinBudgetError) as exc_info:
        policy.select(candidates, ctx)

    assert exc_info.value.cheapest_pair[0] == "p2"  # p2 is cheaper


def test_zero_budget_paid_only_raises_with_cheapest_cost_in_message(monkeypatch):
    """A $0 budget with paid-only candidates raises and reports cheapest cost."""
    _reg("openai", {
        "gpt-4o": {"input_price_per_1m": 2.5, "output_price_per_1m": 10.0},
        "gpt-4o-mini": {"input_price_per_1m": 0.15, "output_price_per_1m": 0.60},
    }, key_env="OPENAI_KEY",
         pricing={"input_per_1m": 0.05, "output_per_1m": 0.40, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy(expected_output_tokens=1000)
    ctx = RoutingContext(
        prompt_tokens_estimate=1000,
        user_budget_usd=0.0,
        required_capabilities={Capability.chat},
    )
    with pytest.raises(NoCandidateWithinBudgetError) as exc_info:
        policy.select([
            ProviderModelPair("openai", "gpt-4o"),
            ProviderModelPair("openai", "gpt-4o-mini"),
        ], ctx)

    message = str(exc_info.value)
    assert exc_info.value.cheapest_cost_usd == pytest.approx(0.00075)
    assert "Cheapest available: $0.000750" in message


def test_unlimited_budget_selects_cheapest(monkeypatch):
    """With no budget constraint, cheapest is still selected."""
    _reg("cheap", {"cm": {}}, key_env="CK",
         pricing={"input_per_1m": 0.5, "output_per_1m": 0.5, "free_tier": False},
         monkeypatch=monkeypatch)
    _reg("pricey", {"pm": {}}, key_env="PK",
         pricing={"input_per_1m": 10.0, "output_per_1m": 10.0, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    candidates = [
        ProviderModelPair("cheap", "cm"),
        ProviderModelPair("pricey", "pm"),
    ]
    ctx = RoutingContext(
        prompt_tokens_estimate=1000,
        user_budget_usd=None,  # unlimited
        required_capabilities={Capability.chat},
    )
    decision = policy.select(candidates, ctx)
    assert decision.chosen.provider == "cheap"


# ---------------------------------------------------------------------------
# Tests: no API key
# ---------------------------------------------------------------------------

def test_no_api_key_eliminates_provider(monkeypatch):
    """Provider without configured key is eliminated even if cheapest."""
    _reg("no-key", {"m1": {}},
         key_env=None,  # no key
         pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
         monkeypatch=monkeypatch)
    _reg("has-key", {"m2": {}}, key_env="HK",
         pricing={"input_per_1m": 5.0, "output_per_1m": 5.0, "free_tier": False},
         monkeypatch=monkeypatch)

    # Override: no-key registered without env var, so not available
    # but has-key has a key set
    ProviderRegistry._providers["no-key"]["env_keys"] = ["NO_SUCH_KEY_XYZ"]

    policy = CostBasedPolicy()
    candidates = [
        ProviderModelPair("no-key", "m1"),
        ProviderModelPair("has-key", "m2"),
    ]
    ctx = RoutingContext(required_capabilities={Capability.chat})
    decision = policy.select(candidates, ctx)
    assert decision.chosen.provider == "has-key"


def test_no_candidates_raises_no_candidate_error(monkeypatch):
    """NoCandidateError raised when all providers lack keys."""
    _reg("p1", {"m1": {}},
         pricing={"input_per_1m": 1.0, "output_per_1m": 1.0, "free_tier": False},
         monkeypatch=monkeypatch)
    ProviderRegistry._providers["p1"]["env_keys"] = ["MISSING_KEY_XYZ"]

    policy = CostBasedPolicy()
    candidates = [ProviderModelPair("p1", "m1")]
    ctx = RoutingContext(required_capabilities={Capability.chat})
    with pytest.raises(NoCandidateError):
        policy.select(candidates, ctx)


def test_standard_routing_skips_dedicated_endpoint_models(monkeypatch):
    """Non-serverless catalog entries are not selected by the standard router."""
    _reg("together", {
        "dedicated-only": {"serverless": False, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0},
        "serverless-chat": {"serverless": True, "pricing_per_1m_input": 0.05, "pricing_per_1m_output": 0.20},
    }, key_env="TOGETHER_KEY",
         pricing={"input_per_1m": 0.03, "output_per_1m": 0.12, "free_tier": False},
         monkeypatch=monkeypatch)

    policy = CostBasedPolicy()
    ctx = RoutingContext(required_capabilities={Capability.chat})
    decision = policy.select([
        ProviderModelPair("together", "dedicated-only"),
        ProviderModelPair("together", "serverless-chat"),
    ], ctx)
    assert decision.chosen == ProviderModelPair("together", "serverless-chat")
    assert any(reason == "requires dedicated endpoint" for _, reason in decision.eliminated)


# ---------------------------------------------------------------------------
# Tests: per-model pricing override
# ---------------------------------------------------------------------------

def test_per_model_pricing_overrides_provider_default(monkeypatch):
    """Per-model pricing_per_1m_input takes precedence over provider default."""
    # Register with expensive provider default but one model has cheap per-model price
    ProviderRegistry.register(
        "mixed",
        _DummyAdapter,
        {
            "cheap-model": {"pricing_per_1m_input": 0.1, "pricing_per_1m_output": 0.1},
            "default-model": {},
        },
        env_keys=["MX_KEY"],
        capabilities={Capability.chat},
        pricing={"input_per_1m": 10.0, "output_per_1m": 10.0, "free_tier": False},
    )
    monkeypatch.setenv("MX_KEY", "key")

    inp, out, _ = _get_model_pricing("mixed", "cheap-model")
    assert inp == pytest.approx(0.1)
    assert out == pytest.approx(0.1)

    inp2, out2, _ = _get_model_pricing("mixed", "default-model")
    assert inp2 == pytest.approx(10.0)  # falls back to provider default
    assert out2 == pytest.approx(10.0)


def test_openai_style_pricing_keys(monkeypatch):
    """input_price_per_1m / output_price_per_1m (OpenAI style) are also honored."""
    ProviderRegistry.register(
        "oai-style",
        _DummyAdapter,
        {
            "mymodel": {"input_price_per_1m": 2.5, "output_price_per_1m": 10.0},
        },
        env_keys=["OAI_KEY"],
        capabilities={Capability.chat},
        pricing={"input_per_1m": 99.0, "output_per_1m": 99.0, "free_tier": False},
    )
    monkeypatch.setenv("OAI_KEY", "key")

    inp, out, _ = _get_model_pricing("oai-style", "mymodel")
    assert inp == pytest.approx(2.5)
    assert out == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: ProviderRegistry.get_pricing
# ---------------------------------------------------------------------------

def test_registry_get_pricing_returns_correct_values(monkeypatch):
    """ProviderRegistry.get_pricing returns the registered pricing dict."""
    ProviderRegistry.register(
        "test-prov",
        _DummyAdapter,
        {},
        pricing={"input_per_1m": 3.0, "output_per_1m": 7.5, "free_tier": True},
    )
    p = ProviderRegistry.get_pricing("test-prov")
    assert p["input_per_1m"] == pytest.approx(3.0)
    assert p["output_per_1m"] == pytest.approx(7.5)
    assert p["free_tier"] is True


def test_registry_get_pricing_unknown_provider():
    """get_pricing returns zeros for unknown provider."""
    p = ProviderRegistry.get_pricing("not-registered-xyz")
    assert p["input_per_1m"] == 0.0
    assert p["output_per_1m"] == 0.0
    assert p["free_tier"] is False


def test_register_without_pricing_defaults_to_zero():
    """Registering without pricing= defaults to 0.0/0.0/False."""
    ProviderRegistry.register("no-pricing", _DummyAdapter, {})
    p = ProviderRegistry.get_pricing("no-pricing")
    assert p["input_per_1m"] == 0.0
    assert p["output_per_1m"] == 0.0
    assert p["free_tier"] is False


# ---------------------------------------------------------------------------
# Tests: cost estimation math
# ---------------------------------------------------------------------------

def test_cost_estimation_math(monkeypatch):
    """RouterDecision.score reflects correct cost estimate."""
    ProviderRegistry.register(
        "math-test",
        _DummyAdapter,
        {"m": {}},
        env_keys=["MT_KEY"],
        capabilities={Capability.chat},
        pricing={"input_per_1m": 2.0, "output_per_1m": 8.0, "free_tier": False},
    )
    monkeypatch.setenv("MT_KEY", "key")

    policy = CostBasedPolicy(expected_output_tokens=1000)
    candidates = [ProviderModelPair("math-test", "m")]
    ctx = RoutingContext(
        prompt_tokens_estimate=500_000,  # 0.5M input tokens
        required_capabilities={Capability.chat},
    )
    decision = policy.select(candidates, ctx)
    # cost = 2.0 * 500000 / 1e6 + 8.0 * 1000 / 1e6 = 1.0 + 0.008 = 1.008
    expected = 2.0 * 500_000 / 1_000_000 + 8.0 * 1_000 / 1_000_000
    assert decision.score == pytest.approx(expected, rel=1e-6)
