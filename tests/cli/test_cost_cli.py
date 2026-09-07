"""Tests for the 'effgen cost' CLI subcommand."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from effgen.models import _cost as cost_mod

if TYPE_CHECKING:
    from effgen.cli import CLIInterface


@pytest.fixture(autouse=True)
def isolated_budget_path(tmp_path, monkeypatch):
    """Keep CLI budget tests on a private budget file — not the shared
    session-wide isolation path from conftest, nor the developer's real
    ~/.effgen/budget.json. ``EFFGEN_BUDGET_CONFIG`` is the one override every
    read/write/display path honors (see ``_budget_config_path()``)."""
    monkeypatch.setattr(cost_mod, "_BUDGET_CONFIG_PATH", tmp_path / "budget.json")
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(tmp_path / "budget.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli() -> "CLIInterface":
    """Return a CLIInterface with a no-op console so tests don't print to stdout."""
    from effgen.cli import CLIInterface
    cli = CLIInterface()
    cli.console = None  # suppress Rich output
    return cli


def _args(**kwargs):
    """Build a minimal argparse-like namespace."""
    ns = MagicMock()
    # argparse always provides these flags as real values; default them so the
    # MagicMock doesn't auto-vivify them as truthy mocks.
    ns.output_json = False
    ns.output = None
    ns.report = None
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _clear_budget():
    from pathlib import Path
    env_path = os.environ.get("EFFGEN_BUDGET_CONFIG")
    if env_path:
        Path(env_path).unlink(missing_ok=True)
    try:
        cost_mod._BUDGET_CONFIG_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _run_cost_cmd(subcmd: str | None, store):
    """Call _handle_cost_command, injecting *store* via patching the module import."""
    from effgen.cli import _handle_cost_command
    from effgen.models import _cost_store as cost_store_mod
    cli = _make_cli()
    args = _args(cost_command=subcmd)
    with patch.object(cost_store_mod, "SQLiteCostStore", return_value=store):
        with patch("effgen.models._cost_store.SQLiteCostStore", return_value=store):
            return _handle_cost_command(args, cli)


# ---------------------------------------------------------------------------
# set-budget / clear-budget
# ---------------------------------------------------------------------------

class TestBudgetManagementCLI:
    def teardown_method(self):
        _clear_budget()

    def test_set_budget_via_cost_subcommand(self, tmp_path):
        """'effgen cost set-budget 2.5' writes budget.json."""
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="set-budget", amount=2.5)
        code = _handle_cost_command(args, cli)
        assert code == 0
        budget_path = tmp_path / "budget.json"
        cfg = json.loads(budget_path.read_text())
        assert cfg.get("daily") == 2.5

    def test_config_set_budget_daily(self, tmp_path):
        """'effgen config set budget.daily 1.5' writes the budget file."""
        cli = _make_cli()
        args = _args(config_command="set", key="budget.daily", value="1.5")
        cli._config_set(args)
        budget_path = tmp_path / "budget.json"
        cfg = json.loads(budget_path.read_text())
        assert cfg.get("daily") == 1.5

    def test_config_set_budget_monthly(self, tmp_path):
        cli = _make_cli()
        args = _args(config_command="set", key="budget.monthly", value="10")
        cli._config_set(args)
        budget_path = tmp_path / "budget.json"
        cfg = json.loads(budget_path.read_text())
        assert cfg.get("monthly") == 10.0

    def test_config_set_unknown_key_no_crash(self):
        cli = _make_cli()
        args = _args(config_command="set", key="unknown.key", value="xyz")
        cli._config_set(args)  # Should not raise

    def test_clear_budget(self, tmp_path):
        """'effgen cost clear-budget' removes the budget limits."""
        budget_path = tmp_path / "budget.json"
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        budget_path.write_text(json.dumps({"daily": 5.0}))

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="clear-budget")
        code = _handle_cost_command(args, cli)
        assert code == 0
        if budget_path.exists():
            cfg = json.loads(budget_path.read_text())
            assert "daily" not in cfg

    def test_set_budget_honors_effgen_budget_config_override(self, tmp_path, monkeypatch):
        """`set-budget` writes to EFFGEN_BUDGET_CONFIG's target, not the
        developer's real ~/.effgen/budget.json, when the override is set."""
        override_path = tmp_path / "override" / "budget.json"
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(override_path))
        real_home_path = tmp_path / "budget.json"  # what the fixture set _BUDGET_CONFIG_PATH to

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="set-budget", amount=3.0)
        code = _handle_cost_command(args, cli)
        assert code == 0
        assert json.loads(override_path.read_text()).get("daily") == 3.0
        assert not real_home_path.exists()

    def test_config_set_budget_honors_effgen_budget_config_override(self, tmp_path, monkeypatch):
        """`config set budget.daily` writes to EFFGEN_BUDGET_CONFIG's target too."""
        override_path = tmp_path / "override" / "budget.json"
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(override_path))
        real_home_path = tmp_path / "budget.json"

        cli = _make_cli()
        args = _args(config_command="set", key="budget.daily", value="4.0")
        cli._config_set(args)
        assert json.loads(override_path.read_text()).get("daily") == 4.0
        assert not real_home_path.exists()

    def test_cost_display_honors_effgen_budget_config_override(self, tmp_path, monkeypatch):
        """The `cost` display reads the same EFFGEN_BUDGET_CONFIG path as the
        writers, so what a user sets is what they see (no read/write split)."""
        override_path = tmp_path / "override" / "budget.json"
        override_path.parent.mkdir(parents=True)
        override_path.write_text(json.dumps({"daily": 7.0}))
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(override_path))

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="today", output_json=True)
        with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
            with patch("builtins.print") as mock_print:
                code = _handle_cost_command(args, cli)
        assert code == 0
        printed = "".join(str(c.args[0]) for c in mock_print.call_args_list)
        assert json.loads(printed)["daily_budget_usd"] == 7.0


# ---------------------------------------------------------------------------
# Cost report subcommands
# ---------------------------------------------------------------------------

class TestCostReportCLI:
    """Test effgen cost today/week/by-provider output generation."""

    def setup_method(self):
        _clear_budget()

    def teardown_method(self):
        _clear_budget()

    def test_today_empty_store(self):
        """'effgen cost today' with no data returns exit code 0."""
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="today")
        with patch("effgen.models._cost_store.SQLiteCostStore.__init__", return_value=None):
            with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
                code = _handle_cost_command(args, cli)
        assert code == 0

    def test_week_empty_store(self):
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="week")
        with patch("effgen.models._cost_store.SQLiteCostStore.query_week", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_by_provider_empty_store(self):
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="by-provider")
        with patch("effgen.models._cost_store.SQLiteCostStore.query_all", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_by_provider_groups_models(self, capsys):
        from effgen.cli import _handle_cost_command
        from effgen.models._cost_store import CostEvent
        cli = _make_cli()
        args = _args(cost_command="by-provider")
        events = [
            CostEvent("openai", "gpt-4o-mini", 10, 5, 0.001, time.time()),
            CostEvent("openai", "gpt-4o", 20, 5, 0.002, time.time()),
            CostEvent("groq", "llama-3.1-8b-instant", 5, 5, 0.0, time.time()),
        ]
        with patch("effgen.models._cost_store.SQLiteCostStore.query_all", return_value=events):
            code = _handle_cost_command(args, cli)
        out = capsys.readouterr().out
        assert code == 0
        assert out.count("openai") == 1
        assert "all models" in out
        assert "$   0.003000" in out

    def test_today_with_real_inmemory_store(self):
        """Actually use an in-memory store and verify exit code 0."""
        from effgen.cli import _handle_cost_command
        from effgen.models._cost_store import SQLiteCostStore
        cli = _make_cli()
        # Use real in-memory store and patch the constructor in the handler
        store = SQLiteCostStore(":memory:")
        store.insert("groq", "llama-3.1-8b-instant", 100, 50, 0.0, time.time())
        store.insert("openai", "gpt-4o-mini", 200, 80, 0.00005, time.time())
        args = _args(cost_command="today")
        with patch("effgen.models._cost_store.SQLiteCostStore") as mock_cls:
            mock_cls.return_value = store
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_default_subcommand_shows_today(self):
        """effgen cost with no subcommand (None) defaults to today."""
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command=None)
        with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_invalid_subcommand_returns_error(self):
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="nonexistent")
        with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 1

    def test_budget_display_with_configured_budget(self, tmp_path):
        """Budget bar appears in output when daily budget is set."""
        budget_cfg = tmp_path / "budget.json"
        budget_cfg.write_text(json.dumps({"daily": 5.0}))

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="today")
        with patch("effgen.models._cost._BUDGET_CONFIG_PATH", budget_cfg):
            with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
                code = _handle_cost_command(args, cli)
        assert code == 0

    def test_no_spend_yet_reports_the_cap_already_in_force(self, tmp_path, capsys):
        """With a cap set, the empty report states it instead of asking for one."""
        budget_cfg = tmp_path / "budget.json"
        budget_cfg.write_text(json.dumps({"daily": 1.0}))

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="today")
        with patch("effgen.models._cost._BUDGET_CONFIG_PATH", budget_cfg):
            with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
                assert _handle_cost_command(args, cli) == 0
        out = capsys.readouterr().out
        assert "Daily cap in force" in out
        assert "Then set a cap with" not in out

    def test_no_spend_and_no_cap_still_offers_to_set_one(self, tmp_path, capsys):
        budget_cfg = tmp_path / "absent.json"

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="today")
        with patch("effgen.models._cost._BUDGET_CONFIG_PATH", budget_cfg):
            with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
                assert _handle_cost_command(args, cli) == 0
        out = capsys.readouterr().out
        assert "Then set a cap with: effgen cost set-budget 1.00" in out
        assert "Daily cap in force" not in out


# ---------------------------------------------------------------------------
# BudgetExceededError integration
# ---------------------------------------------------------------------------

class TestBudgetExceededIntegration:
    """Tests how the retry policy classifies BudgetExceededError."""

    def test_budget_exceeded_error_is_retriable(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.routing.retry import RetryPolicy
        policy = RetryPolicy()
        err = BudgetExceededError(1.0, 2.0, "daily", "openai", "gpt-4o")
        assert policy.is_retriable(err)

    def test_budget_exceeded_not_in_non_retriable(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.routing.retry import _NON_RETRIABLE
        assert not isinstance(BudgetExceededError(1.0, 2.0, "daily"), _NON_RETRIABLE)

    def test_budget_exceeded_failover_reason_daily(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.router import _exc_to_reason
        err = BudgetExceededError(1.0, 2.0, "daily", "openai", "gpt-4o")
        assert _exc_to_reason(err) == "budget_exceeded_daily"

    def test_budget_exceeded_failover_reason_monthly(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.router import _exc_to_reason
        err = BudgetExceededError(1.0, 2.0, "monthly", "together", "llama-3")
        assert _exc_to_reason(err) == "budget_exceeded_monthly"

    def test_budget_exceeded_exported_from_models(self):
        from effgen.models import BudgetExceededError
        assert BudgetExceededError is not None

    def test_budget_exceeded_error_message_format(self):
        from effgen.models.errors import BudgetExceededError
        err = BudgetExceededError(1.0, 1.5, "daily", "openai", "gpt-4o")
        msg = str(err)
        assert "daily" in msg.lower()
        assert "1.5" in msg or "1.50" in msg
        # The error is raised to a plain Agent.run() caller too, where no
        # failover happens — the message must describe failover as something a
        # router *can* do, not something that unconditionally will happen.
        assert "failover" in msg.lower()
        assert "fallback_chain" in msg
        assert "router will attempt" not in msg.lower()
