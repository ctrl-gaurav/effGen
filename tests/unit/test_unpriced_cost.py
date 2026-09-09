"""A model with no published price reports no cost, never a fabricated ``$0``.

``$0.00`` is a real answer for a free tier. For a model the catalog carries no
rate for, the same number states a price nobody published and reads as "this
run was free": the run card prints ``$0.000000``, a bake-off ranks the model as
the cheapest contender, and a budget report counts zero spend for a call the
provider may well have billed.
These tests pin the distinction end to end — the pricing helper, the tracker,
the budget gate, the adapters' own pricing paths, the per-run accumulator and
the workflow tab.
"""

from __future__ import annotations

import warnings

import pytest

from effgen.models._cost import (
    CostTracker,
    call_cost,
    pricing_status,
    reset_unpriced_budget_warnings,
)
from effgen.models._usage import cost_label, usage_metadata

# Ids chosen so each pricing status is represented by a real catalog entry.
PRICED = ("groq", "openai/gpt-oss-20b")
FREE = ("cerebras", "gpt-oss-120b")
UNPRICED = ("groq", "allam-2-7b")


@pytest.fixture(autouse=True)
def _clear_unpriced_warnings():
    reset_unpriced_budget_warnings()
    yield
    reset_unpriced_budget_warnings()


class TestCallCost:
    def test_the_three_statuses_are_what_the_catalog_says(self):
        assert pricing_status(*PRICED) == "priced"
        assert pricing_status(*FREE) == "free"
        assert pricing_status(*UNPRICED) == "unpriced"

    def test_a_priced_model_is_a_real_number(self):
        assert call_cost(*PRICED, 1000, 1000) > 0

    def test_a_free_tier_is_zero_because_zero_is_the_truth(self):
        assert call_cost(*FREE, 1000, 1000) == 0.0

    def test_an_unpriced_model_reports_nothing(self):
        assert call_cost(*UNPRICED, 1000, 1000) is None

    def test_free_and_unpriced_do_not_collapse_to_the_same_value(self):
        assert call_cost(*FREE, 500, 500) != call_cost(*UNPRICED, 500, 500)


class TestTrackerRecord:
    def test_record_returns_none_for_an_unpriced_model(self):
        tracker = CostTracker(storage=None)
        assert tracker.record(*UNPRICED, 100, 50) is None

    def test_record_returns_zero_for_a_free_tier(self):
        tracker = CostTracker(storage=None)
        assert tracker.record(*FREE, 100, 50) == 0.0

    def test_an_explicit_cost_override_still_wins(self):
        # Replicate and HF price outside the token table and pass the number in.
        tracker = CostTracker(storage=None)
        assert tracker.record(*UNPRICED, 100, 50, cost_usd=0.25) == 0.25

    def test_tokens_are_still_recorded_when_the_price_is_unknown(self):
        tracker = CostTracker(storage=None)
        tracker.record(*UNPRICED, 100, 50)
        assert tracker.total_tokens()["total"] == 150

    def test_summary_reports_no_cost_and_counts_the_unpriced_calls(self):
        tracker = CostTracker(storage=None)
        tracker.record(*UNPRICED, 100, 50)
        tracker.record(*UNPRICED, 10, 5)
        row = tracker.summary()[0]
        assert row["cost_usd"] is None
        assert row["unpriced_requests"] == 2
        assert row["pricing"] == "unpriced"

    def test_summary_of_a_priced_model_is_unchanged(self):
        tracker = CostTracker(storage=None)
        tracker.record(*PRICED, 100, 50)
        row = tracker.summary()[0]
        assert row["cost_usd"] > 0
        assert row["unpriced_requests"] == 0


class TestBudgetGateWithAnUnknownPrice:
    """The gate refuses spend it can measure; it does not refuse on a price
    nobody published — it says so instead."""

    @pytest.fixture
    def budget(self, tmp_path, monkeypatch):
        config = tmp_path / "budget.json"
        config.write_text('{"daily": 0.01}')
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(config))
        return config

    def test_an_unpriced_call_is_allowed_and_says_why_it_is_not_counted(self, budget):
        tracker = CostTracker(storage=None)
        with pytest.warns(UserWarning, match=r"no published price for 'groq:allam-2-7b'"):
            assert tracker.record(*UNPRICED, 1000, 1000) is None

    def test_the_heads_up_names_the_model_once_not_once_per_call(self, budget):
        tracker = CostTracker(storage=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                tracker.record(*UNPRICED, 10, 10)
        unpriced_warnings = [
            w for w in caught if "no published price" in str(w.message)
        ]
        assert len(unpriced_warnings) == 1

    def test_the_heads_up_names_a_provider_the_refresh_command_accepts(self, budget):
        """The remediation line has to be a command that runs.

        The ledger keys HF Inference spend under ``hf_inference`` while
        ``effgen models refresh`` knows the provider as ``hf``, so the raw key
        would send the user to "Unknown provider".
        """
        from effgen.models._catalog import known_providers
        from effgen.models._cost import _refresh_hint

        hint = _refresh_hint("hf_inference")
        assert "--provider hf`" in hint
        for provider in ("groq", "openai", "hf_inference"):
            named = _refresh_hint(provider).split("--provider ")[1].split("`")[0]
            assert named in known_providers()
        # A local engine has no catalog to refresh, so it gets no hint at all.
        assert _refresh_hint("transformers") == ""

    def test_no_budget_configured_means_no_heads_up(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(tmp_path / "absent.json"))
        tracker = CostTracker(storage=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.record(*UNPRICED, 10, 10)
        assert not [w for w in caught if "no published price" in str(w.message)]


class TestUsageMetadata:
    def test_an_unknown_cost_travels_as_none(self):
        meta = usage_metadata(10, 5, 15, 0, None, 0.0)
        assert meta["cost_usd"] is None

    def test_cost_label_says_unpriced_instead_of_a_dollar_amount(self):
        assert cost_label(None) == "unpriced"
        assert cost_label(0.0) == "$0.000000"


class TestAdapterPricingPaths:
    """The adapters that price a call from their own table, not the tracker."""

    def test_openai_reports_nothing_when_the_catalog_has_no_rate(self, monkeypatch):
        from effgen.models import _cost as cost_mod
        from effgen.models.openai_adapter import OpenAIAdapter

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
        monkeypatch.setattr(cost_mod, "pricing_status", lambda p, m: "unpriced")
        adapter = OpenAIAdapter("gpt-4o-mini")
        assert adapter._calculate_cost(1000, 1000) is None
        assert adapter._record_cost(1000, 1000, 2000) is None
        # An unpriced call adds nothing to the session total.
        assert adapter.total_cost == 0.0

    def test_openai_still_prices_a_model_the_catalog_carries(self, monkeypatch):
        from effgen.models.openai_adapter import OpenAIAdapter

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
        adapter = OpenAIAdapter("gpt-4o-mini")
        assert adapter._calculate_cost(1000, 1000) > 0

    def test_anthropic_reports_nothing_when_the_catalog_has_no_rate(self, monkeypatch):
        pytest.importorskip("anthropic")
        from effgen.models import _cost as cost_mod
        from effgen.models.anthropic_adapter import AnthropicAdapter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        monkeypatch.setattr(cost_mod, "pricing_status", lambda p, m: "unpriced")
        adapter = AnthropicAdapter("claude-3-haiku-20240307")
        assert adapter._calculate_cost(1000, 1000) is None
        assert adapter._record_usage(1000, 1000) is None
        assert adapter.total_cost == 0.0

    def test_hf_inference_reports_nothing_without_a_registry_price(self, monkeypatch):
        pytest.importorskip("huggingface_hub")
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        monkeypatch.setenv("HF_TOKEN", "test-token-not-used")
        adapter = HFInferenceAdapter("meta-llama/Llama-3.1-8B-Instruct")
        monkeypatch.setattr(adapter, "_info", {}, raising=False)
        assert adapter._price_tokens(1000, 500) is None
        monkeypatch.setattr(
            adapter, "_info", {"pricing_per_1m_input": 1.0, "pricing_per_1m_output": 2.0},
            raising=False,
        )
        assert adapter._price_tokens(1000, 500) > 0

    def test_replicate_reports_nothing_without_a_registry_rate(self, monkeypatch):
        pytest.importorskip("replicate")
        from effgen.models.replicate_adapter import ReplicateAdapter

        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token-not-used")
        adapter = ReplicateAdapter("meta/meta-llama-3-8b-instruct")
        metrics = {"predict_time": 2.0, "input_token_count": 10, "output_token_count": 20}
        monkeypatch.setattr(adapter, "_info", {}, raising=False)
        assert adapter._price_metrics(metrics)[3] is None
        monkeypatch.setattr(
            adapter, "_info", {"cost_per_second_usd": 0.000225}, raising=False
        )
        assert adapter._price_metrics(metrics)[3] > 0


class TestUnpricedCallsStillReachTheLedger:
    """An unknown price must not delete the call from the spend record.

    HF Inference and Replicate price a call from their own registry rather
    than the token table, so they hand the tracker a number. When they have no
    rate the call is still real: its tokens belong on the ledger, marked
    unpriced, not dropped from ``effgen cost`` entirely.
    """

    @pytest.fixture
    def tracker(self, monkeypatch):
        from effgen.models._cost import CostTracker

        fresh = CostTracker(storage=None)
        monkeypatch.setattr(CostTracker, "get", classmethod(lambda cls: fresh))
        return fresh

    def test_hf_inference_records_an_unpriced_call(self, monkeypatch, tracker):
        pytest.importorskip("huggingface_hub")
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        monkeypatch.setenv("HF_TOKEN", "test-token-not-used")
        adapter = HFInferenceAdapter("meta-llama/Llama-3.1-8B-Instruct")
        adapter._record_cost(120, 30, None)
        row = tracker.summary()[0]
        assert row["total_tokens"] == 150
        assert row["cost_usd"] is None
        assert row["unpriced_requests"] == 1

    def test_replicate_records_an_unpriced_prediction(self, monkeypatch, tracker):
        pytest.importorskip("replicate")
        from effgen.models.replicate_adapter import ReplicateAdapter

        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token-not-used")
        # An id neither the registry nor the catalog carries a rate for: every
        # registry model has a per-second or per-token rate, so "no rate" means
        # a model effGen has not seen.
        adapter = ReplicateAdapter("acme/not-a-real-model")
        adapter._record_cost(40, 60, None)
        row = tracker.summary()[0]
        assert row["total_tokens"] == 100
        assert row["cost_usd"] is None

    def test_a_priced_call_still_records_its_number(self, monkeypatch, tracker):
        pytest.importorskip("replicate")
        from effgen.models.replicate_adapter import ReplicateAdapter

        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token-not-used")
        adapter = ReplicateAdapter("meta/meta-llama-3-8b-instruct")
        adapter._record_cost(40, 60, 0.0125)
        assert tracker.summary()[0]["cost_usd"] == pytest.approx(0.0125)


class TestCostReportDocument:
    """``effgen cost --json`` states the same thing its table does."""

    def _rows(self, capsys, events):
        import json
        from unittest.mock import MagicMock, patch

        from effgen.cli import CLIInterface, _handle_cost_command

        cli = CLIInterface()
        cli.console = None
        args = MagicMock()
        args.cost_command = "today"
        args.output_json = True
        args.output = None
        args.report = None
        with patch(
            "effgen.models._cost_store.SQLiteCostStore.query_today", return_value=events
        ):
            assert _handle_cost_command(args, cli) == 0
        return json.loads(capsys.readouterr().out)["rows"]

    def test_an_unpriced_row_reports_no_cost_not_a_zero(self, capsys):
        import time

        from effgen.models._cost_store import CostEvent

        now = time.time()
        rows = self._rows(capsys, [
            CostEvent(*PRICED, 100, 50, 0.0002, now),
            CostEvent(*FREE, 100, 50, 0.0, now),
            CostEvent(*UNPRICED, 100, 50, 0.0, now),
        ])
        by_model = {r["model"]: r for r in rows}
        assert by_model[UNPRICED[1]]["cost_usd"] is None
        assert by_model[UNPRICED[1]]["cost_label"] == "unpriced"
        # A free tier is a measured $0, and a priced model keeps its number.
        assert by_model[FREE[1]]["cost_usd"] == 0.0
        assert by_model[FREE[1]]["cost_label"] == "free"
        assert by_model[PRICED[1]]["cost_usd"] == pytest.approx(0.0002)


class TestPerRunAccumulator:
    """``Agent.run`` folds every call's cost onto the response metadata."""

    def _agent(self):
        from effgen.core.agent import Agent, AgentConfig
        from tests.conftest import MockModel

        return Agent(config=AgentConfig(model=MockModel(responses=["Final Answer: 4"])))

    def test_a_run_of_unpriced_calls_reports_no_cost_at_all(self):
        from effgen.core.agent_response import AgentResponse

        agent = self._agent()
        agent._run_cost_accum = {}
        agent._accumulate_run_cost({"cost_usd": None, "total_tokens": 30})
        response = AgentResponse(output="4", success=True)
        agent._finalize_cost_metadata(response)
        assert "cost_usd" not in response.metadata
        assert response.metadata["unpriced_calls"] == 1

    def test_a_mixed_run_reports_the_priced_calls_and_flags_the_rest(self):
        from effgen.core.agent_response import AgentResponse

        agent = self._agent()
        agent._run_cost_accum = {}
        agent._accumulate_run_cost({"cost_usd": 0.002, "total_tokens": 30})
        agent._accumulate_run_cost({"cost_usd": None, "total_tokens": 10})
        response = AgentResponse(output="4", success=True)
        agent._finalize_cost_metadata(response)
        assert response.metadata["cost_usd"] == 0.002
        assert response.metadata["unpriced_calls"] == 1

    def test_a_priced_run_carries_no_unpriced_flag(self):
        from effgen.core.agent_response import AgentResponse

        agent = self._agent()
        agent._run_cost_accum = {}
        agent._accumulate_run_cost({"cost_usd": 0.002, "total_tokens": 30})
        response = AgentResponse(output="4", success=True)
        agent._finalize_cost_metadata(response)
        assert response.metadata["cost_usd"] == 0.002
        assert "unpriced_calls" not in response.metadata


class _CostAgent:
    """Minimal agent whose runs report a chosen ``cost_usd`` (or none)."""

    def __init__(self, name: str, cost: float | None):
        self.name = name
        self._cost = cost

    def run(self, task: str, **kwargs):
        from effgen.core.agent_response import AgentResponse

        meta = {} if self._cost is None else {"cost_usd": self._cost}
        return AgentResponse(output=f"{self.name}:{task}", success=True, metadata=meta)

    async def run_async(self, task: str, **kwargs):
        return self.run(task, **kwargs)


class TestWorkflowCostTab:
    def _dag(self, *agents):
        from effgen.core.workflow import WorkflowDAG, WorkflowNode

        dag = WorkflowDAG("tab")
        previous = None
        for agent in agents:
            dag.add_node(WorkflowNode(id=agent.name, agent=agent))
            if previous is not None:
                dag.connect(previous, agent.name)
            previous = agent.name
        return dag

    def test_a_workflow_of_unpriced_nodes_reports_no_cost(self):
        result = self._dag(_CostAgent("a", None), _CostAgent("b", None)).run("go")
        assert result.metadata["cost_usd"] is None
        node_costs = [n["metadata"].get("cost_usd") for n in result.node_results]
        assert node_costs == [None, None]

    def test_a_partly_priced_workflow_reports_what_it_knows(self):
        result = self._dag(_CostAgent("a", 0.001), _CostAgent("b", None)).run("go")
        assert result.metadata["cost_usd"] == pytest.approx(0.001)

    def test_a_priced_workflow_is_unchanged(self):
        result = self._dag(_CostAgent("a", 0.001), _CostAgent("b", 0.002)).run("go")
        assert result.metadata["cost_usd"] == pytest.approx(0.003)


class TestTeamCostTab:
    def _orchestrator(self, *agents):
        from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern

        orch = MultiAgentOrchestrator()
        orch.create_team(
            "team", list(agents), pattern=OrchestrationPattern.SEQUENTIAL
        )
        return orch

    def test_a_team_of_unpriced_members_reports_no_cost(self):
        orch = self._orchestrator(_CostAgent("a", None), _CostAgent("b", None))
        result = orch.assign_task("go", "team")
        assert result.metadata["cost_usd"] is None

    def test_a_partly_priced_team_reports_what_it_knows(self):
        orch = self._orchestrator(_CostAgent("a", 0.001), _CostAgent("b", None))
        result = orch.assign_task("go", "team")
        assert result.metadata["cost_usd"] == pytest.approx(0.001)
