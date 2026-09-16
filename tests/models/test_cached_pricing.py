"""A cached prompt token is priced at the rate its provider publishes, or not at all.

A run that hits a cache and is billed as though every token were fresh over-states
what it cost; a run credited a discount its provider never published under-states
it. Both are wrong, and the second is worse, so a provider with no published
cached rate is billed exactly as it was before.

Offline: the catalog and the pricing functions, no provider call.
"""

from __future__ import annotations

import pytest

from effgen.models._catalog import lookup
from effgen.models._cost import CostTracker, cache_rates, call_cost


def test_a_published_cached_rate_makes_a_hit_cheaper():
    """gpt-4o-mini publishes a cached-input rate, so a hit costs less."""
    fresh = call_cost("openai", "gpt-4o-mini", 1000, 100)
    hit = call_cost("openai", "gpt-4o-mini", 1000, 100, cached_tokens=800)
    assert fresh is not None and hit is not None
    assert hit < fresh
    cached_rate, _ = cache_rates("openai", "gpt-4o-mini")
    assert cached_rate is not None
    record = lookup("gpt-4o-mini", "openai")
    expected = (
        200 * record.price_in_per_1m
        + 800 * cached_rate
        + 100 * record.price_out_per_1m
    ) / 1_000_000
    assert hit == pytest.approx(expected)


def test_a_provider_with_no_published_rate_is_billed_exactly_as_before():
    """Tokens cached, no published rate: no saving is invented."""
    cached_rate, write_rate = cache_rates("groq", "openai/gpt-oss-20b")
    assert cached_rate is None and write_rate is None
    fresh = call_cost("groq", "openai/gpt-oss-20b", 1000, 100)
    hit = call_cost("groq", "openai/gpt-oss-20b", 1000, 100, cached_tokens=900)
    assert fresh == hit


def test_anthropic_prices_a_read_below_and_a_write_above_the_input_rate():
    from effgen.models.anthropic_models import (
        CACHE_READ_PRICE_MULTIPLIER,
        CACHE_WRITE_PRICE_MULTIPLIER,
    )

    read_rate, write_rate = cache_rates("anthropic", "claude-sonnet-4-6")
    record = lookup("claude-sonnet-4-6", "anthropic")
    assert read_rate == pytest.approx(record.price_in_per_1m * CACHE_READ_PRICE_MULTIPLIER)
    assert write_rate == pytest.approx(record.price_in_per_1m * CACHE_WRITE_PRICE_MULTIPLIER)

    fresh = call_cost("anthropic", "claude-sonnet-4-6", 10_000, 100)
    read = call_cost("anthropic", "claude-sonnet-4-6", 10_000, 100, cached_tokens=9_000)
    written = call_cost(
        "anthropic", "claude-sonnet-4-6", 10_000, 100, cache_write_tokens=9_000
    )
    assert read < fresh < written


def test_the_anthropic_adapter_prices_its_own_call_the_same_way():
    from typing import Any

    from effgen.models.anthropic_adapter import AnthropicAdapter

    stub: Any = AnthropicAdapter.__new__(AnthropicAdapter)
    stub.model_name = "claude-sonnet-4-6"
    fresh = AnthropicAdapter._calculate_cost(stub, 10_000, 100)
    read = AnthropicAdapter._calculate_cost(stub, 10_000, 100, 9_000, 0)
    written = AnthropicAdapter._calculate_cost(stub, 10_000, 100, 0, 9_000)
    assert read < fresh < written
    assert read == pytest.approx(
        call_cost("anthropic", "claude-sonnet-4-6", 10_000, 100, cached_tokens=9_000)
    )


def test_cached_tokens_are_never_counted_outside_the_prompt():
    """A provider reporting more cached tokens than prompt tokens cannot
    make a call cost less than nothing."""
    weird = call_cost("openai", "gpt-4o-mini", 100, 10, cached_tokens=10_000)
    only_cached = call_cost("openai", "gpt-4o-mini", 100, 10, cached_tokens=100)
    assert weird == pytest.approx(only_cached)
    assert weird >= 0


def test_the_tracker_records_the_discounted_cost():
    tracker = CostTracker(storage=None)
    fresh = tracker.record(
        provider="openai", model="gpt-4o-mini",
        prompt_tokens=1000, completion_tokens=100,
    )
    hit = tracker.record(
        provider="openai", model="gpt-4o-mini",
        prompt_tokens=1000, completion_tokens=100, cached_tokens=900,
    )
    assert fresh is not None and hit is not None and hit < fresh
