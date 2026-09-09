"""Cost/pricing completeness & accuracy tests.

These lock in the fix that makes the cost tracker read pricing from the
normalized model catalog (the single source of truth shared with the adapters)
instead of a stale hand-maintained rate table.  The headline regression is that
priced providers like Groq must no longer report ``$0.000000``, and the
per-token rates must match the catalog the CLI shows.
"""

from __future__ import annotations

import pytest

from effgen.api.openai_compat import count_tokens
from effgen.models import _catalog as cat
from effgen.models._cost import (
    CostTracker,
    _rate,
    call_cost,
    pricing_status,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    CostTracker.reset()
    yield
    CostTracker.reset()


class TestCatalogDrivenRates:
    def test_groq_is_priced_not_free(self):
        """A priced Groq model must report a real per-token rate, not $0."""
        rin, rout = _rate("groq", "openai/gpt-oss-20b")
        assert rin > 0 and rout > 0
        assert pricing_status("groq", "openai/gpt-oss-20b") == "priced"

    def test_rate_matches_catalog(self):
        """The tracker rate must equal the catalog price (no placeholder drift)."""
        for prov, model in [
            ("groq", "openai/gpt-oss-20b"),
            ("openai", "gpt-5-nano"),
            ("gemini", "gemini-3.1-flash-lite"),
        ]:
            rec = cat.lookup(model, prov)
            assert rec is not None, f"{prov}/{model} missing from catalog"
            assert rec.price_in_per_1m is not None
            rin, rout = _rate(prov, model)
            assert rin == pytest.approx(rec.price_in_per_1m)
            assert rout == pytest.approx(rec.price_out_per_1m or 0.0)

    def test_openai_nano_not_overcharged(self):
        """The old placeholder $1/$3 over-charged nano ~10x; catalog is ~$0.05/$0.40."""
        rin, rout = _rate("openai", "gpt-5-nano")
        assert rin < 0.5 and rout < 2.0

    def test_record_computes_catalog_cost(self):
        t = CostTracker(storage=None)
        rin, rout = _rate("openai", "gpt-5-nano")
        expected = (1000 * rin + 500 * rout) / 1_000_000
        cost = t.record("openai", "gpt-5-nano", prompt_tokens=1000, completion_tokens=500)
        assert cost == pytest.approx(expected)
        assert cost > 0


class TestPricingStatus:
    def test_cerebras_free_tier(self):
        assert pricing_status("cerebras", "gpt-oss-120b") == "free"
        assert _rate("cerebras", "gpt-oss-120b") == (0.0, 0.0)

    def test_unknown_id_on_priced_provider_never_silently_zero(self):
        # A model absent from the catalog but on a known priced provider must
        # NOT silently report $0. It reports no cost at all instead: the
        # provider's "*" entry is a placeholder, and billing an unknown id at
        # it — a fine-tuned `ft:` id, anything newer than the last refresh —
        # published an invented number as though it were real.
        rin, rout = _rate("openai", "totally-made-up-model-xyz")
        assert rin > 0 and rout > 0, "the estimate itself is still available"
        assert pricing_status("openai", "totally-made-up-model-xyz") == "unpriced"
        assert call_cost("openai", "totally-made-up-model-xyz", 1000, 1000) is None

    def test_a_fine_tuned_id_is_not_billed_at_the_placeholder(self):
        model = "ft:gpt-4o-mini-2024-07-18:acme::abc123"
        assert pricing_status("openai", model) == "unpriced"
        assert call_cost("openai", model, 1000, 1000) is None

    def test_a_catalogued_id_is_still_priced(self):
        """The change must not cost a known model its real published rate."""
        assert pricing_status("openai", "gpt-4o-mini") == "priced"
        assert call_cost("openai", "gpt-4o-mini", 1000, 1000) > 0

    def test_unknown_provider_is_unpriced(self):
        assert pricing_status("no-such-provider", "model") == "unpriced"

    def test_replicate_is_metered(self):
        # Replicate bills per GPU-second; catalog marks it metered, token rate 0.
        assert pricing_status("replicate", "meta/meta-llama-3-8b-instruct") == "metered"
        assert _rate("replicate", "meta/meta-llama-3-8b-instruct") == (0.0, 0.0)

    def test_summary_carries_pricing(self):
        t = CostTracker(storage=None)
        t.record("groq", "openai/gpt-oss-20b", 100, 50)
        row = t.summary()[0]
        assert row["pricing"] == "priced"
        assert row["cost_usd"] > 0


class TestBackCompatRates:
    def test_cerebras_test_id_still_free(self):
        # Deprecated test id absent from the live catalog → legacy fallback ($0).
        assert _rate("cerebras", "llama3.1-8b") == (0.0, 0.0)

    def test_unknown_provider_zero(self):
        assert _rate("unknown_provider", "model") == (0.0, 0.0)


class TestTokenCounting:
    def test_count_tokens_beats_len_over_4_for_repetition(self):
        # A real BPE tokenizer collapses repetition that len//4 over-counts.
        # tiktoken downloads its BPE data on first use, so on a machine with an
        # empty cache and no route to the data host there is no tokenizer to
        # measure — the count is then the heuristic this case compares against.
        from effgen.models._adapter_utils import get_bpe_encoding

        if get_bpe_encoding() is None:
            pytest.skip("the cl100k_base BPE data is not available on this machine")
        text = "x" * 200
        assert count_tokens(text) < len(text) // 4

    def test_count_tokens_empty_is_zero(self):
        assert count_tokens("") == 0

    def test_count_tokens_positive(self):
        assert count_tokens("The capital of France is Paris.") > 0
