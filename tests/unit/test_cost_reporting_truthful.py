"""Cost and latency reports tell the caller what is known and nothing else.

A model with no published price reports ``None`` on every surface — never a
``0.0`` that reads like a free call. A free tier reports ``0.0``; a priced model
its cost. The spend cap refuses only what it can count, and when it refuses the
caller gets :class:`BudgetExceededError` itself. The latency a run reports is
the wall time the caller waited. The cost ledger holds itself to its ceiling.

Everything here runs against a scripted OpenAI-protocol endpoint or in-process
stand-ins; no provider is called.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from effgen.models import _cost
from effgen.models._cost import CostTracker, reset_budget_config_cache
from effgen.models._cost_store import SQLiteCostStore
from effgen.models.errors import BudgetExceededError

# --------------------------------------------------------------------- fixtures


class _Endpoint(BaseHTTPRequestHandler):
    """An OpenAI-protocol server that answers every id it is asked for."""

    requests = 0

    def log_message(self, *args) -> None:
        pass

    def _json(self, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._json({"object": "list", "data": [{"id": "served-model"}]})

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests += 1
        if request.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            chunk = {"id": "x", "object": "chat.completion.chunk", "model": request["model"],
                     "choices": [{"index": 0, "delta": {"content": "Final Answer: 4"}}]}
            done = {"id": "x", "object": "chat.completion.chunk", "model": request["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(done)}\n\n"
                             "data: [DONE]\n\n".encode())
            self.wfile.flush()
            self.close_connection = True
            return
        self._json({"id": "x", "object": "chat.completion", "model": request["model"],
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": "Final Answer: 4"}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20,
                              "total_tokens": 70}})


@pytest.fixture()
def endpoint():
    _Endpoint.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A private ledger and budget file, and a fresh tracker over them."""
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    monkeypatch.setenv("EFFGEN_COST_DB", str(tmp_path / "costs.sqlite"))
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(tmp_path / "budget.json"))
    # Sessions and run history too: a run here saves both, and a directory set
    # in the environment would otherwise be shared with every later test.
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("EFFGEN_COST_MAX_ROWS", raising=False)
    for var in ("EFFGEN_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    reset_budget_config_cache()
    CostTracker.reset()
    _cost.reset_unpriced_budget_warnings()
    if hasattr(_cost, "reset_spend_cap_decisions"):
        _cost.reset_spend_cap_decisions()
    yield tmp_path
    CostTracker.reset()
    reset_budget_config_cache()


def _budget(home: Path, daily: float) -> None:
    (home / "budget.json").write_text(json.dumps({"daily": daily}))
    reset_budget_config_cache()


def _spend(amount: float) -> None:
    """Record priced spend earlier today, as another run of the day would have."""
    CostTracker.get()._storage.insert(provider="openai", model="gpt-4o-mini",
                                      prompt_tokens=1, completion_tokens=1,
                                      cost_usd=amount, timestamp=time.time())


def _served_agent(base_url: str, model: str = "served-model", **cfg):
    from effgen import Agent, AgentConfig

    return Agent(AgentConfig(name="served", model=model, base_url=base_url, tools=[],
                             max_iterations=2, **cfg))


def _priced_agent(base_url: str, **cfg):
    """The scripted endpoint spoken to as OpenAI, under an id OpenAI prices."""
    from effgen import Agent, AgentConfig
    from effgen.models.openai_adapter import OpenAIAdapter

    model = OpenAIAdapter(model_name="gpt-4o-mini", api_key="sk-scripted", base_url=base_url)
    model.load()
    return Agent(AgentConfig(name="priced", model=model, tools=[], max_iterations=2, **cfg))


def _local_model():
    from effgen.models.base import BaseModel, GenerationResult, ModelType

    class LocalEngine(BaseModel):
        def __init__(self):
            super().__init__(model_name="local-engine", model_type=ModelType.TRANSFORMERS)
            self._is_loaded = True

        def load(self):
            self._is_loaded = True

        def unload(self):
            pass

        def generate(self, prompt, config=None, **kw):
            return GenerationResult(text="Final Answer: local", tokens_used=3,
                                    finish_reason="stop", model_name="local-engine")

        def generate_stream(self, prompt, config=None, **kw):
            yield "Final Answer: local"

        def get_context_length(self):
            return 4096

        def count_tokens(self, text):
            return max(1, len(text) // 4)

    return LocalEngine()


# ------------------------------------------------------ priced, free, unpriced


def test_the_tracker_reports_none_zero_and_a_cost(home):
    tracker = CostTracker.get()
    assert tracker.record("openai_compatible", "served-model", 50, 20) is None
    assert tracker.record("cerebras", "llama3.1-8b", 50, 20) == 0.0
    assert tracker.record("openai", "gpt-4o-mini", 50, 20) > 0.0
    rows = {row["provider"]: row for row in tracker.summary()}
    assert rows["openai_compatible"]["cost_usd"] is None
    assert rows["cerebras"]["cost_usd"] == 0.0
    assert rows["openai"]["cost_usd"] > 0.0


def test_total_cost_is_none_when_every_call_is_unpriced(home):
    tracker = CostTracker.get()
    tracker.record("openai_compatible", "served-model", 50, 20)
    assert tracker.total_cost() is None
    assert tracker.total_cost("openai_compatible") is None
    tracker.record("cerebras", "llama3.1-8b", 50, 20)
    assert tracker.total_cost("cerebras") == 0.0
    assert tracker.total_cost() == 0.0          # a free call is known to cost nothing
    tracker.record("openai", "gpt-4o-mini", 50, 20)
    assert tracker.total_cost() == pytest.approx(tracker.total_cost("openai"))
    assert CostTracker(storage=None).total_cost() == 0.0   # nothing recorded


def test_a_served_model_named_like_a_catalog_model_is_not_charged(home, endpoint):
    """A server the caller runs is never billed at OpenAI's rate for a shared id."""
    response = _served_agent(endpoint, model="gpt-4o-mini").run("2+2?")
    assert response.success
    assert response.metadata.get("cost_usd") is None
    assert response.metadata.get("unpriced_calls") == 1
    assert response.metadata["ledger"]["cost_usd"] is None
    tracker = CostTracker.get()
    assert tracker.total_cost() is None
    (row,) = tracker.summary()
    assert row["provider"] == "openai_compatible"
    assert row["cost_usd"] is None
    assert response.provider == "openai_compatible"


def test_the_ledger_file_records_an_unpriced_call_as_unpriced(home):
    tracker = CostTracker.get()
    tracker.record("openai_compatible", "served-model", 50, 20)
    tracker.record("cerebras", "llama3.1-8b", 50, 20)
    events = {e.provider: e for e in tracker._storage.query_all()}
    assert events["openai_compatible"].cost_usd is None
    assert events["openai_compatible"].unpriced_calls == 1
    assert events["cerebras"].cost_usd == 0.0
    assert events["cerebras"].unpriced_calls == 0
    assert tracker._storage.spend_today() == 0.0


def test_a_ledger_from_before_call_counts_opens_and_reads_as_priced_calls(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cost_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "provider TEXT NOT NULL, model TEXT NOT NULL, prompt_tokens INTEGER NOT NULL "
                 "DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL "
                 "DEFAULT 0.0, timestamp REAL NOT NULL)")
    conn.execute("INSERT INTO cost_events (provider, model, prompt_tokens, completion_tokens, "
                 "cost_usd, timestamp) VALUES ('openai', 'gpt-4o-mini', 5, 5, 0.25, ?)",
                 (time.time(),))
    conn.commit()
    conn.close()
    store = SQLiteCostStore(path)
    store.insert("openai_compatible", "served-model", 1, 1, None)
    old, new = store.query_all()
    assert (old.cost_usd, old.calls, old.unpriced_calls) == (0.25, 1, 0)
    assert (new.cost_usd, new.calls, new.unpriced_calls) == (None, 1, 1)
    assert store.spend_today() == pytest.approx(0.25)
    store.close()


def test_the_cost_report_says_unpriced_rather_than_zero(home, capsys):
    from argparse import Namespace

    from effgen.cli.commands.cost import _handle_cost_command

    tracker = CostTracker.get()
    tracker.record("openai_compatible", "served-model", 50, 20)
    tracker.record("openai_compatible", "served-model", 50, 20)

    class _Cli:
        console = None
        _human_to_stderr = False

        def print(self, *a, **k):
            pass

        print_error = print_success = print_header = print

    code = _handle_cost_command(Namespace(cost_command="today", output_json=True, output=None,
                                          report=None), _Cli())
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    (row,) = document["rows"]
    assert row["cost_usd"] is None and row["cost_label"] == "unpriced"
    assert row["requests"] == 2 and row["unpriced_requests"] == 2
    assert document["total_cost_usd"] is None


def test_the_prometheus_series_count_an_unpriced_call_apart_from_spend(home, endpoint):
    from effgen.observability import metrics

    metrics.reset_all()
    _served_agent(endpoint).run("2+2?")
    labels = {"provider": "openai_compatible", "model": "served-model"}
    assert metrics.model_unpriced_calls_total.get(labels) == 1.0
    assert metrics.model_cost_usd_total.get(labels) == 0.0
    text = metrics.export_metrics()
    assert 'effgen_model_unpriced_calls_total{model="served-model",provider="openai_compatible"} 1' \
        in text.replace(".0", "")
    metrics.record_model_cost(provider="openai", model="gpt-4o-mini", cost_usd=0.002)
    assert metrics.model_cost_usd_total.get(
        {"provider": "openai", "model": "gpt-4o-mini"}) == pytest.approx(0.002)
    metrics.reset_all()


def test_the_run_store_does_not_turn_unpriced_runs_into_zero(tmp_path, monkeypatch):
    from effgen.observability import run_log

    records = [
        {"run_id": "a", "execution_id": "x1", "execution_kind": "team", "cost_usd": None,
         "input_tokens": 5, "output_tokens": 5, "ts": "2026-09-23T00:00:01", "status": "ok"},
        {"run_id": "b", "execution_id": "x1", "execution_kind": "team", "cost_usd": None,
         "input_tokens": 5, "output_tokens": 5, "ts": "2026-09-23T00:00:02", "status": "ok"},
        {"run_id": "c", "execution_id": "x2", "execution_kind": "team", "cost_usd": 0.5,
         "input_tokens": 5, "output_tokens": 5, "ts": "2026-09-23T00:00:03", "status": "ok"},
        {"run_id": "d", "execution_id": "x2", "execution_kind": "team", "cost_usd": None,
         "input_tokens": 5, "output_tokens": 5, "ts": "2026-09-23T00:00:04", "status": "ok"},
    ]
    monkeypatch.setattr(run_log, "read_runs", lambda **kw: list(reversed(records)))
    by_id = {e["id"]: e for e in run_log.read_executions(limit=10)}
    assert by_id["x1"]["cost_usd"] is None
    assert by_id["x2"]["cost_usd"] == pytest.approx(0.5)


def test_the_run_card_labels_a_run_without_a_price(home, endpoint):
    from effgen.ui.report_html_run import _run_body

    response = _served_agent(endpoint).run("2+2?")
    _, _, body = _run_body(response.to_dict() if hasattr(response, "to_dict") else {
        "metadata": response.metadata, "model": response.model,
        "provider": response.provider, "success": response.success})
    assert "unpriced" in body
    assert "$0.00" not in body


# ------------------------------------------------------------------ the warning


def test_the_price_warning_fires_once_per_model_over_many_calls(home, endpoint):
    _budget(home, 50.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        barrier = threading.Barrier(8)

        def many() -> None:
            barrier.wait()
            for _ in range(5):
                CostTracker.get().record("openai_compatible", "served-model", 5, 5)

        threads = [threading.Thread(target=many) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for _ in range(10):
            _served_agent(endpoint).run("2+2?")
        CostTracker.get().record("openai_compatible", "another-model", 5, 5)
    fired = [w for w in caught if "no published price" in str(w.message)]
    assert len(fired) == 2, [str(w.message) for w in fired]
    assert "effgen models refresh" not in str(fired[0].message)


# ------------------------------------------------------------------- spend cap


def test_the_exemptions_come_from_the_catalog_and_the_provider():
    assert _cost.spend_cap_exemption(None, "local-engine") == "local"
    assert _cost.spend_cap_exemption("", "local-engine") == "local"
    assert _cost.spend_cap_exemption("cerebras", "llama3.1-8b") == "free"
    assert _cost.spend_cap_exemption("openai_compatible", "gpt-4o-mini") == "unmetered"
    assert _cost.spend_cap_exemption("openai", "gpt-4o-mini") is None
    # An id a pricing provider does not list is still billed by it.
    assert _cost.spend_cap_exemption("openai", "a-model-released-yesterday") is None
    # HF Inference records its calls as ``hf_inference``; the catalog knows it
    # as ``hf``, which bills, so a spent cap refuses it.
    assert _cost.spend_cap_exemption("hf_inference", "meta-llama/Llama-3.1-8B-Instruct") is None
    assert _cost.spend_cap_exemption("hf_inference", "an-unlisted/model") is None


def test_a_spent_cap_refuses_hf_inference_before_the_call(home):
    from effgen.models.hf_inference_adapter import HFInferenceAdapter

    _spend(2.0)
    _budget(home, 1.0)
    model = HFInferenceAdapter("meta-llama/Llama-3.1-8B-Instruct", api_token="hf_not_a_key")
    # Not loaded: were the cap to let the call through, generate() would fail on
    # the missing client instead of refusing it.
    with pytest.raises(BudgetExceededError):
        model.generate("hi")
    assert _cost.spend_cap_decisions() == {"refused": 1}


def test_a_spent_cap_lets_a_served_model_run(home, endpoint):
    _spend(2.0)
    _budget(home, 1.0)
    response = _served_agent(endpoint).run("2+2?")
    assert response.success and "4" in response.output
    stream = "".join(str(t) for t in _served_agent(endpoint).stream("2+2?"))
    assert "4" in stream
    batch = _served_agent(endpoint, raise_on_error=False).run_batch(["a?", "b?", "c?"])
    assert batch.succeeded == 3 and batch.failed == 0
    assert _cost.spend_cap_decisions().get("allowed_unmetered", 0) >= 5
    assert "refused" not in _cost.spend_cap_decisions()


def test_a_spent_cap_lets_a_local_engine_run(home):
    from effgen import Agent, AgentConfig

    _spend(2.0)
    _budget(home, 1.0)
    response = Agent(AgentConfig(name="local", model=_local_model(), tools=[],
                                 max_iterations=2)).run("hi")
    assert response.success
    assert _cost.spend_cap_decisions() == {"allowed_local": 1}


@pytest.mark.parametrize("raise_on_error", [True, False])
def test_a_refusal_is_the_typed_error_never_a_failed_generation(home, endpoint, raise_on_error):
    _spend(2.0)
    _budget(home, 1.0)
    agent = _priced_agent(endpoint, raise_on_error=raise_on_error)
    with pytest.raises(BudgetExceededError) as info:
        agent.run("2+2?")
    assert info.value.period == "daily"
    assert _Endpoint.requests == 0, "a refused call reached the endpoint"
    assert _cost.spend_cap_decisions().get("refused", 0) >= 1


def test_a_batch_stops_at_a_spent_cap_with_the_typed_error(home, endpoint):
    _spend(2.0)
    _budget(home, 1.0)
    agent = _priced_agent(endpoint, raise_on_error=False)
    with pytest.raises(BudgetExceededError):
        agent.run_batch(["a?", "b?", "c?"], max_concurrency=1)
    assert _Endpoint.requests == 0


def test_a_streamed_refusal_is_the_typed_error(home, endpoint):
    _spend(2.0)
    _budget(home, 1.0)
    with pytest.raises(BudgetExceededError):
        "".join(str(t) for t in _priced_agent(endpoint).stream("2+2?"))


def test_the_server_answers_a_spent_cap_with_429():
    from effgen.api.openai_compat_errors import _classify_http

    status, err_type, code = _classify_http(BudgetExceededError(1.0, 2.0))
    assert (status, err_type, code) == (429, "insufficient_quota", "budget_exceeded")


# --------------------------------------------------------------------- latency


def test_reported_latency_is_the_runs_wall_including_post_run_work(home, endpoint, tmp_path):
    from effgen.core.session import Session

    agent = _served_agent(endpoint)
    started = time.perf_counter()
    response = agent.run("2+2?", checkpoint_dir=str(tmp_path / "ck"),
                         session=Session(session_id="s1"))
    outer = time.perf_counter() - started
    ledger = response.metadata["ledger"]
    assert response.execution_time == pytest.approx(ledger["wall_s"], abs=1e-3)
    assert response.metadata["duration_s"] == pytest.approx(ledger["wall_s"], abs=1e-3)
    split = sum(ledger[k] for k in ("model_wait_s", "tool_wait_s", "caller_wait_s",
                                    "child_wait_s", "framework_s"))
    assert split == pytest.approx(ledger["wall_s"], abs=1e-3)
    assert 0.0 <= outer - response.execution_time < 0.005


# ------------------------------------------------------------------- retention


def test_the_ledger_folds_at_its_ceiling_and_keeps_every_total(tmp_path, caplog):
    store = SQLiteCostStore(tmp_path / "c.sqlite", max_rows=500)
    start = time.time() - 86400.0
    expected = {"cost": 0.0, "calls": 0, "unpriced": 0, "prompt": 0}
    with caplog.at_level(logging.INFO, logger="effgen.models._cost_store"):
        for i in range(3000):
            cost = None if i % 4 == 0 else 0.001
            store.insert("p", f"m{i % 3}", 2, 1, cost, timestamp=start + i * 28.8)
            expected["cost"] += cost or 0.0
            expected["calls"] += 1
            expected["unpriced"] += cost is None
            expected["prompt"] += 2
            assert store._rows is None or store._rows <= 500
    events = store.query_all()
    assert len(events) <= 500
    assert sum(e.cost_usd or 0.0 for e in events) == pytest.approx(expected["cost"])
    assert sum(e.calls for e in events) == expected["calls"]
    assert sum(e.unpriced_calls for e in events) == expected["unpriced"]
    assert sum(e.prompt_tokens for e in events) == expected["prompt"]
    folds = [r for r in caplog.records if "cost ledger: folded" in r.getMessage()]
    assert folds and store.fold_stats["folds"] == len(folds)
    assert sum(r.levelno == logging.WARNING for r in folds) == 1
    store.close()


def test_folding_off_keeps_every_row(tmp_path):
    store = SQLiteCostStore(tmp_path / "c.sqlite", max_rows=0)
    for i in range(300):
        store.insert("p", "m", 1, 1, 0.001, timestamp=time.time() - i)
    assert store.count() == 300
    assert store.fold_stats["folds"] == 0
    store.close()


# ------------------------------------------------------------ a sub-agent's cost


def test_an_agent_inside_a_sync_tool_rolls_its_cost_into_the_parent(home, endpoint):
    """A plain function tool runs on a worker thread in the caller's context."""
    import asyncio

    from effgen import tool
    from effgen.core import ledger as run_ledger

    child_costs = []

    @tool
    def helper(question: str) -> str:
        """Ask a helper agent."""
        response = _priced_agent(endpoint).run(question)
        child_costs.append(response.metadata.get("cost_usd"))
        return response.output

    recorder = run_ledger.open_run("run", "parent")
    with run_ledger.activate(recorder):
        asyncio.run(helper.execute(question="2+2?"))
    parent = recorder.close()
    assert child_costs and child_costs[0] > 0
    assert len(parent.children) == 1
    assert parent.total()["cost_usd"] == pytest.approx(child_costs[0])
