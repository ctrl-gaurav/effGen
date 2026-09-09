"""Tests for SQLiteCostStore."""

from __future__ import annotations

import json
import time
import warnings

import pytest

from effgen.models import _cost as cost_mod
from effgen.models._cost import CostTracker
from effgen.models._cost_store import CostEvent, SQLiteCostStore
from effgen.models.errors import BudgetExceededError


@pytest.fixture(autouse=True)
def isolated_budget_path(tmp_path, monkeypatch):
    """Keep budget alert tests away from the real user config.

    Point both the default path and the ``EFFGEN_BUDGET_CONFIG`` override (the
    root conftest sets the override to isolate cost tests from a host-configured
    budget) at this test's temp file, so a budget written here is the one read.
    """
    budget_file = tmp_path / "budget.json"
    monkeypatch.setattr(cost_mod, "_BUDGET_CONFIG_PATH", budget_file)
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(budget_file))


# ---------------------------------------------------------------------------
# SQLiteCostStore unit tests
# ---------------------------------------------------------------------------

class TestCostStoreLocation:
    """Where cost events land when no explicit path is given."""

    def test_effgen_home_relocates_the_database(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EFFGEN_COST_DB", raising=False)
        monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "state"))

        store = SQLiteCostStore()

        assert store._path == str(tmp_path / "state" / "costs.sqlite")
        assert (tmp_path / "state").is_dir()

    def test_explicit_env_path_wins_over_effgen_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("EFFGEN_COST_DB", str(tmp_path / "explicit.sqlite"))

        assert SQLiteCostStore()._path == str(tmp_path / "explicit.sqlite")


class TestSQLiteCostStore:
    """Unit tests for SQLiteCostStore using an in-memory DB."""

    def setup_method(self):
        self.store = SQLiteCostStore(":memory:")

    def test_schema_creation_idempotent(self):
        """Schema init runs twice without error."""
        self.store._init_schema()
        self.store._init_schema()

    def test_insert_and_query_today(self):
        now = time.time()
        self.store.insert("cerebras", "llama3.1-8b", 50, 20, 0.0, now)
        rows = self.store.query_today()
        assert len(rows) == 1
        row = rows[0]
        assert row.provider == "cerebras"
        assert row.model == "llama3.1-8b"
        assert row.prompt_tokens == 50
        assert row.completion_tokens == 20
        assert row.cost_usd == 0.0

    def test_insert_multiple_and_aggregate(self):
        now = time.time()
        self.store.insert("openai", "gpt-4o-mini", 100, 50, 0.00005, now)
        self.store.insert("openai", "gpt-4o-mini", 200, 80, 0.0001, now)
        self.store.insert("groq", "openai/gpt-oss-20b", 30, 15, 0.0, now)
        rows = self.store.query_today()
        assert len(rows) == 3
        total = sum(r.cost_usd for r in rows)
        assert abs(total - 0.00015) < 1e-9

    def test_query_week_excludes_old(self):
        old_ts = time.time() - 8 * 86400  # 8 days ago
        recent_ts = time.time() - 2 * 86400  # 2 days ago
        self.store.insert("openai", "gpt-4o", 100, 50, 0.01, old_ts)
        self.store.insert("groq", "openai/gpt-oss-20b", 30, 15, 0.0, recent_ts)
        week_rows = self.store.query_week()
        assert len(week_rows) == 1
        assert week_rows[0].provider == "groq"

    def test_query_month_excludes_old(self):
        old_ts = time.time() - 31 * 86400
        recent_ts = time.time() - 20 * 86400
        self.store.insert("openai", "gpt-4o", 100, 50, 0.01, old_ts)
        self.store.insert("groq", "openai/gpt-oss-20b", 30, 15, 0.0, recent_ts)
        month_rows = self.store.query_month()
        assert len(month_rows) == 1
        assert month_rows[0].provider == "groq"

    def test_query_all_returns_everything(self):
        old_ts = time.time() - 100 * 86400
        now = time.time()
        self.store.insert("openai", "gpt-4o", 100, 50, 0.01, old_ts)
        self.store.insert("groq", "openai/gpt-oss-20b", 30, 15, 0.0, now)
        all_rows = self.store.query_all()
        assert len(all_rows) == 2

    def test_cleanup_removes_old_events(self):
        old_ts = time.time() - 200 * 86400
        now = time.time()
        self.store.insert("openai", "gpt-4o", 100, 50, 0.01, old_ts)
        self.store.insert("groq", "openai/gpt-oss-20b", 30, 15, 0.0, now)
        deleted = self.store.cleanup(max_age_seconds=86400)  # keep only last 1 day
        assert deleted == 1
        remaining = self.store.query_all()
        assert len(remaining) == 1
        assert remaining[0].provider == "groq"

    def test_query_since_filters_correctly(self):
        base = time.time()
        self.store.insert("openai", "gpt-4o", 100, 50, 0.01, base - 3600)
        self.store.insert("groq", "llama", 30, 15, 0.0, base - 100)
        rows = self.store.query_since(base - 200)
        assert len(rows) == 1
        assert rows[0].provider == "groq"

    def test_event_dataclass_fields(self):
        now = time.time()
        self.store.insert("fireworks", "llama-v3", 200, 100, 0.002, now)
        rows = self.store.query_today()
        assert isinstance(rows[0], CostEvent)
        assert rows[0].provider == "fireworks"
        assert rows[0].model == "llama-v3"
        assert rows[0].prompt_tokens == 200
        assert rows[0].completion_tokens == 100
        assert abs(rows[0].cost_usd - 0.002) < 1e-9
        assert abs(rows[0].timestamp - now) < 1.0

    def test_default_timestamp_set_automatically(self):
        before = time.time()
        self.store.insert("groq", "llama", 10, 5, 0.0)
        after = time.time()
        rows = self.store.query_all()
        assert len(rows) == 1
        assert before <= rows[0].timestamp <= after


# ---------------------------------------------------------------------------
# CostTracker with storage integration
# ---------------------------------------------------------------------------

class TestCostTrackerWithStorage:
    """Integration tests for CostTracker using in-memory SQLiteCostStore."""

    def setup_method(self):
        CostTracker.reset()
        self.store = SQLiteCostStore(":memory:")
        self.tracker = CostTracker(storage=self.store)

    def test_record_writes_to_db(self):
        self.tracker.record("groq", "openai/gpt-oss-20b", 50, 20)
        rows = self.store.query_today()
        assert len(rows) == 1
        assert rows[0].provider == "groq"
        assert rows[0].model == "openai/gpt-oss-20b"

    def test_record_cost_zero_for_free_tier(self):
        cost = self.tracker.record("cerebras", "llama3.1-8b", 100, 50)
        assert cost == 0.0
        rows = self.store.query_today()
        assert rows[0].cost_usd == 0.0

    def test_record_non_zero_cost_stored(self):
        cost = self.tracker.record("openai", "gpt-4o-mini", 1000000, 1000000)
        assert cost > 0
        rows = self.store.query_today()
        assert rows[0].cost_usd > 0

    def test_memory_summary_consistent_with_db(self):
        self.tracker.record("groq", "openai/gpt-oss-20b", 50, 20)
        self.tracker.record("groq", "openai/gpt-oss-20b", 30, 10)
        summary = self.tracker.summary()
        assert len(summary) == 1
        assert summary[0]["requests"] == 2
        db_rows = self.store.query_today()
        assert len(db_rows) == 2

    def test_in_memory_tracker_no_storage(self):
        """Passing storage=None gives a pure in-memory tracker (back-compat)."""
        t = CostTracker(storage=None)
        # Cerebras is a genuine free tier → $0 cost.
        cost = t.record("cerebras", "gpt-oss-120b", 50, 20)
        assert cost == 0.0
        assert len(t.summary()) == 1

    def test_singleton_uses_real_db(self):
        """CostTracker.get() creates an SQLite-backed instance."""
        CostTracker.reset()
        t = CostTracker.get()
        assert t._storage is not None

    def test_legacy_singleton_aliases(self):
        assert CostTracker.instance() is CostTracker.get()
        assert CostTracker.get_instance() is CostTracker.get()

    def test_record_accepts_token_aliases_and_cost_override(self):
        cost = self.tracker.record(
            "replicate",
            "model-x",
            input_tokens=12,
            output_tokens=5,
            cost_usd=0.123,
        )
        rows = self.store.query_today()
        assert cost == 0.123
        assert rows[0].prompt_tokens == 12
        assert rows[0].completion_tokens == 5
        assert rows[0].cost_usd == 0.123


# ---------------------------------------------------------------------------
# Budget alert tests
# ---------------------------------------------------------------------------

class TestBudgetAlerts:
    """Tests for budget warning and BudgetExceededError."""

    def setup_method(self):
        CostTracker.reset()
        self.store = SQLiteCostStore(":memory:")
        self.tracker = CostTracker(storage=self.store)

    def teardown_method(self):
        # Always clear the budget config so tests don't pollute each other
        try:
            cost_mod._BUDGET_CONFIG_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    def _set_budget(self, daily_usd: float) -> None:
        cost_mod._BUDGET_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cost_mod._BUDGET_CONFIG_PATH.write_text(json.dumps({"daily": daily_usd}))

    def test_no_budget_no_error(self):
        """Without a budget config, record never raises."""
        cost_mod._BUDGET_CONFIG_PATH.unlink(missing_ok=True)
        # gpt-4o-mini is cheap but non-zero
        cost = self.tracker.record("openai", "gpt-4o-mini", 100, 50)
        assert cost >= 0

    def test_warning_at_80_percent(self):
        self._set_budget(0.001)
        # Seed store just below 80%, then cross the warning threshold.
        self.store.insert("openai", "gpt-4o-mini", 1000000, 0, 0.0007999, time.time())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.tracker.record("openai", "gpt-4o-mini", 1_000, 0)
        warning_messages = [str(x.message) for x in w]
        assert any("budget warning" in msg.lower() for msg in warning_messages), (
            f"Expected budget warning; got: {warning_messages}"
        )

    def test_warning_not_repeated_above_80_percent(self):
        self._set_budget(0.001)
        self.store.insert("openai", "gpt-4o-mini", 1000000, 0, 0.00085, time.time())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.tracker.record("openai", "gpt-4o-mini", 100, 50)
        warning_messages = [str(x.message) for x in w]
        assert not any("budget warning" in msg.lower() for msg in warning_messages)

    def test_budget_exceeded_raises(self):
        self._set_budget(0.000001)
        # Seed above budget
        self.store.insert("openai", "gpt-4o-mini", 1000000, 0, 0.000002, time.time())

        with pytest.raises(BudgetExceededError) as exc_info:
            self.tracker.record("openai", "gpt-4o-mini", 100, 50)
        err = exc_info.value
        assert err.period == "daily"
        assert err.actual_usd > err.budget_usd

    def test_budget_exceeded_contains_provider_info(self):
        self._set_budget(0.000001)
        self.store.insert("openai", "gpt-4o-mini", 1000000, 0, 0.000002, time.time())

        with pytest.raises(BudgetExceededError) as exc_info:
            self.tracker.record("openai", "gpt-4o-mini", 100, 50)
        err = exc_info.value
        assert err.provider == "openai"
        assert err.model == "gpt-4o-mini"

    def test_zero_cost_record_allowed_when_budget_already_exceeded(self):
        self._set_budget(0.000001)
        self.store.insert("openai", "gpt-4o", 100, 50, 0.000002, time.time())

        # A genuine free-tier call (cerebras, $0) must still be allowed so the
        # router can fail over to a free provider after the budget is hit.
        cost = self.tracker.record("cerebras", "gpt-oss-120b", 100, 50)

        assert cost == 0.0
        rows = self.store.query_today()
        assert rows[-1].provider == "cerebras"

    def test_monthly_budget_exceeded_raises(self):
        cost_mod._BUDGET_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cost_mod._BUDGET_CONFIG_PATH.write_text(json.dumps({"monthly": 0.000001}))
        self.store.insert("openai", "gpt-4o-mini", 1000000, 0, 0.000002, time.time())

        with pytest.raises(BudgetExceededError) as exc_info:
            self.tracker.record("openai", "gpt-4o-mini", 100, 50)
        assert exc_info.value.period == "monthly"

    def test_budget_exceeded_is_retriable_by_retry_policy(self):
        """Router's RetryPolicy must classify BudgetExceededError as retriable."""
        from effgen.models.routing.retry import RetryPolicy
        policy = RetryPolicy()
        err = BudgetExceededError(0.001, 0.002, "daily", "openai", "gpt-4o")
        assert policy.is_retriable(err)

    def test_budget_exceeded_triggers_failover_reason(self):
        """_exc_to_reason maps BudgetExceededError to budget_exceeded_daily."""
        from effgen.models.router import _exc_to_reason
        err = BudgetExceededError(0.001, 0.002, "daily", "openai", "gpt-4o")
        reason = _exc_to_reason(err)
        assert reason == "budget_exceeded_daily"
