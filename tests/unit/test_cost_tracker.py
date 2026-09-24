"""Unit tests for CostTracker (effgen/models/_cost.py)."""

from __future__ import annotations

import threading

import pytest

from effgen.models._cost import CostTracker, _rate


# Reset global tracker before each test
@pytest.fixture(autouse=True)
def reset_tracker():
    CostTracker.reset()
    yield
    CostTracker.reset()


class TestRateLookup:
    def test_cerebras_is_free(self):
        rate = _rate("cerebras", "llama3.1-8b")
        assert rate == (0.0, 0.0)

    def test_cerebras_wildcard_is_free(self):
        rate = _rate("cerebras", "any-model")
        assert rate == (0.0, 0.0)

    def test_openai_gpt4o_mini_has_rate(self):
        inp, out = _rate("openai", "gpt-4o-mini")
        assert inp > 0
        assert out > 0

    def test_unknown_provider_returns_zero(self):
        rate = _rate("unknown_provider", "model")
        assert rate == (0.0, 0.0)


class TestCostTrackerBasic:
    def test_singleton(self):
        t1 = CostTracker.get()
        t2 = CostTracker.get()
        assert t1 is t2

    def test_reset_creates_new_instance(self):
        t1 = CostTracker.get()
        CostTracker.reset()
        t2 = CostTracker.get()
        assert t1 is not t2

    def test_record_cerebras_returns_zero(self):
        cost = CostTracker.get().record("cerebras", "llama3.1-8b", 100, 50)
        assert cost == 0.0

    def test_record_openai_returns_nonzero(self):
        cost = CostTracker.get().record("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost > 0

    def test_record_accumulates(self):
        tracker = CostTracker.get()
        tracker.record("cerebras", "llama3.1-8b", 10, 5)
        tracker.record("cerebras", "llama3.1-8b", 20, 10)
        totals = tracker.total_tokens("cerebras", "llama3.1-8b")
        assert totals["prompt"] == 30
        assert totals["completion"] == 15
        assert totals["total"] == 45

    def test_total_cost_all_providers(self):
        tracker = CostTracker.get()
        tracker.record("cerebras", "llama3.1-8b", 100, 50)
        tracker.record("openai", "gpt-4o-mini", 1_000, 500)
        total = tracker.total_cost()
        assert total >= 0  # cerebras is 0, openai is positive

    def test_total_cost_filtered_by_provider(self):
        tracker = CostTracker.get()
        tracker.record("cerebras", "llama3.1-8b", 100, 50)
        tracker.record("openai", "gpt-4o-mini", 1_000_000, 500_000)
        assert tracker.total_cost("cerebras") == 0.0
        assert tracker.total_cost("openai") > 0.0

    def test_total_cost_filtered_by_model(self):
        tracker = CostTracker.get()
        tracker.record("openai", "gpt-4o-mini", 1_000_000, 0)
        tracker.record("openai", "gpt-4", 1_000_000, 0)
        cost_mini = tracker.total_cost(provider="openai", model="gpt-4o-mini")
        cost_gpt4 = tracker.total_cost(provider="openai", model="gpt-4")
        assert cost_mini > 0
        assert cost_gpt4 > cost_mini  # gpt-4 is more expensive

    def test_summary_empty_on_fresh_tracker(self):
        assert CostTracker.get().summary() == []

    def test_summary_has_correct_fields(self):
        tracker = CostTracker.get()
        tracker.record("cerebras", "qwen-3-235b-a22b-instruct-2507", 50, 30)
        rows = tracker.summary()
        assert len(rows) == 1
        row = rows[0]
        assert row["provider"] == "cerebras"
        assert row["model"] == "qwen-3-235b-a22b-instruct-2507"
        assert row["requests"] == 1
        assert row["prompt_tokens"] == 50
        assert row["completion_tokens"] == 30
        assert row["total_tokens"] == 80
        assert row["cost_usd"] == 0.0

    def test_multiple_models_summary(self):
        tracker = CostTracker.get()
        tracker.record("cerebras", "llama3.1-8b", 10, 5)
        tracker.record("cerebras", "qwen-3-235b-a22b-instruct-2507", 20, 10)
        rows = tracker.summary()
        assert len(rows) == 2

    def test_reset_stats_clears_data(self):
        tracker = CostTracker.get()
        tracker.record("cerebras", "llama3.1-8b", 100, 50)
        assert len(tracker.summary()) == 1
        tracker.reset_stats()
        assert len(tracker.summary()) == 0


class TestCostTrackerThreadSafety:
    def test_concurrent_records(self):
        """Multiple threads recording simultaneously must not corrupt state."""
        tracker = CostTracker.get()
        results = []

        def record_batch():
            for _ in range(100):
                cost = tracker.record("cerebras", "llama3.1-8b", 10, 5)
                results.append(cost)

        threads = [threading.Thread(target=record_batch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        totals = tracker.total_tokens("cerebras", "llama3.1-8b")
        assert totals["prompt"] == 5 * 100 * 10
        assert totals["completion"] == 5 * 100 * 5
        assert all(c == 0.0 for c in results)  # Cerebras free


class TestCostTrackerCerebrasRates:
    """All 4 Cerebras models must be $0 regardless of token count."""

    @pytest.mark.parametrize("model", [
        "llama3.1-8b",
        "qwen-3-235b-a22b-instruct-2507",
        "gpt-oss-120b",
        "zai-glm-4.7",
    ])
    def test_cerebras_model_is_free(self, model):
        cost = CostTracker.get().record("cerebras", model, 1_000_000, 1_000_000)
        assert cost == 0.0


class TestBudgetConfigOverride:
    """EFFGEN_BUDGET_CONFIG redirects the budget file (mirrors EFFGEN_COST_DB),
    so a sandbox or CI run is not affected by the developer's real
    ~/.effgen/budget.json."""

    def test_override_points_at_a_budget_file(self, tmp_path, monkeypatch):
        import json as _json

        from effgen.models._cost import _load_budget
        cfg = tmp_path / "budget.json"
        cfg.write_text(_json.dumps({"daily": 5.0}))
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))
        assert _load_budget() == {"daily": 5.0}

    def test_override_missing_file_reads_as_no_budget(self, tmp_path, monkeypatch):
        from effgen.models._cost import _load_budget
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(tmp_path / "absent.json"))
        assert _load_budget() == {}

    def test_configured_budget_is_enforced_via_override(self, tmp_path, monkeypatch):
        import json as _json

        from effgen.models.errors import BudgetExceededError
        cfg = tmp_path / "budget.json"
        cfg.write_text(_json.dumps({"daily": 0.01}))
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))
        with pytest.raises(BudgetExceededError):
            # 1M gpt-4o-mini output tokens costs well over $0.01.
            CostTracker.get().record("openai", "gpt-4o-mini", 0, 1_000_000)


class TestPreflightBudgetGate:
    """check_preflight() refuses a call before it is billed, unlike the
    post-spend check inside record() which only fires once a call's tokens
    (and therefore its cost) are already known."""

    def test_raises_when_spend_already_at_cap(self, tmp_path, monkeypatch):
        import json as _json

        from effgen.models.errors import BudgetExceededError
        cfg = tmp_path / "budget.json"
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))

        tracker = CostTracker.get()
        # Spend $0.05 while no budget is configured yet.
        tracker.record("openai", "gpt-4o-mini", cost_usd=0.05)
        # Configure a cap already below that spend.
        cfg.write_text(_json.dumps({"daily": 0.01}))

        with pytest.raises(BudgetExceededError):
            tracker.check_preflight("openai", "gpt-4o-mini")

    def test_does_not_bill_anything(self, tmp_path, monkeypatch):
        """A refused preflight check must not add to recorded spend."""
        import json as _json

        from effgen.models.errors import BudgetExceededError
        cfg = tmp_path / "budget.json"
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))

        tracker = CostTracker.get()
        tracker.record("openai", "gpt-4o-mini", cost_usd=0.05)
        cfg.write_text(_json.dumps({"daily": 0.01}))
        before = tracker.total_cost()

        with pytest.raises(BudgetExceededError):
            tracker.check_preflight("openai", "gpt-4o-mini")

        assert tracker.total_cost() == before

    def test_passes_when_under_budget(self, tmp_path, monkeypatch):
        import json as _json
        cfg = tmp_path / "budget.json"
        cfg.write_text(_json.dumps({"daily": 100.0}))
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))

        CostTracker.get().check_preflight("openai", "gpt-4o-mini")  # no raise

    def test_passes_when_no_budget_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(tmp_path / "absent.json"))
        CostTracker.get().check_preflight("openai", "gpt-4o-mini")  # no raise

    def test_model_generate_refuses_before_the_call_when_over_budget(self, tmp_path, monkeypatch):
        """BaseModel.generate() runs the preflight check before invoking the
        engine's own generate body (see effgen.models.base)."""
        import json as _json

        from effgen.models.errors import BudgetExceededError
        from tests.fixtures.mock_models import MockModel

        cfg = tmp_path / "budget.json"
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))

        tracker = CostTracker.get()
        tracker.record("openai", "gpt-4o-mini", cost_usd=1.0)
        cfg.write_text(_json.dumps({"daily": 0.01}))

        model = _priced(MockModel)(responses=["hi"], model_name="gpt-4o-mini")
        with pytest.raises(BudgetExceededError):
            model.generate("hello")
        # The engine's own generate body never ran.
        assert model.call_count == 0

    def test_a_local_engine_runs_when_the_cap_is_spent(self, tmp_path, monkeypatch):
        """A model run on this machine is billed by nobody, so a spent cap lets it run."""
        import json as _json

        from tests.fixtures.mock_models import MockModel

        cfg = tmp_path / "budget.json"
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))
        CostTracker.get().record("openai", "gpt-4o-mini", cost_usd=1.0)
        cfg.write_text(_json.dumps({"daily": 0.01}))

        model = MockModel(responses=["hi"])
        assert model.generate("hello").text == "hi"
        assert model.call_count == 1

    def test_model_generate_stream_refuses_before_the_call_when_over_budget(
        self, tmp_path, monkeypatch,
    ):
        import json as _json

        from effgen.models.errors import BudgetExceededError
        from tests.fixtures.mock_models import MockModel

        cfg = tmp_path / "budget.json"
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))

        tracker = CostTracker.get()
        tracker.record("openai", "gpt-4o-mini", cost_usd=1.0)
        cfg.write_text(_json.dumps({"daily": 0.01}))

        model = _priced(MockModel)(responses=["hi"], model_name="gpt-4o-mini")
        with pytest.raises(BudgetExceededError):
            model.generate_stream("hello")
        assert model.call_count == 0

    def test_model_generate_unaffected_when_under_budget(self, tmp_path, monkeypatch):
        import json as _json

        from tests.fixtures.mock_models import MockModel

        cfg = tmp_path / "budget.json"
        cfg.write_text(_json.dumps({"daily": 100.0}))
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(cfg))

        model = MockModel(responses=["hi"])
        result = model.generate("hello")
        assert result.text == "hi"
        assert model.call_count == 1


def _priced(model_cls):
    """*model_cls* declared as a model a pricing provider bills."""
    return type("PricedMock", (model_cls,), {"_provider": "openai"})


class TestFormatUsd:
    """format_usd() preserves significant digits for sub-cent amounts instead
    of a fixed :.4f that rounds a tiny cap away."""

    def test_zero(self):
        from effgen.models._cost import format_usd
        assert format_usd(0.0) == "$0.0000"

    def test_ordinary_amount_uses_four_decimals(self):
        from effgen.models._cost import format_usd
        assert format_usd(1.5) == "$1.5000"
        assert format_usd(0.005) == "$0.0050"

    def test_sub_cent_amount_keeps_significant_digits(self):
        from effgen.models._cost import format_usd
        # Under a fixed :.4f this rounds to $0.0001 — losing the cap entirely.
        assert format_usd(0.00005) == "$0.00005"

    def test_very_small_amount_still_visible(self):
        from effgen.models._cost import format_usd
        result = format_usd(0.0000023)
        assert result != "$0.0000"
        assert "23" in result or "2.3" in result
