"""What a run spent, and where its time went.

Every run keeps a ledger of the model calls and tool calls it made, the tokens
those calls reported, and its wall time split into model, tool, caller and
framework time. These tests pin that the split moves with a delay planted on
purpose, that the counts add up across a run's children exactly once, and that
every place a run is reported — the response, the run document, the run store,
the Prometheus series and the run card — carries it.

Every import below is a name that existed before the ledger did; the ledger's
own module is imported inside the tests, so this file collects against an
earlier release and fails on its assertions.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig, AgentMode
from effgen.core.middleware import AgentMiddleware
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

PROMPT_TOKENS = 11
COMPLETION_TOKENS = 5
CACHED_TOKENS = 3
CACHE_WRITE_TOKENS = 7
ACTION = 'Thought: I will look it up.\nAction: echo\nAction Input: {"value": "x"}'


class _Model(BaseModel):
    """Answers from the prompt alone: asks for the tool until its result is in."""

    def __init__(self, *, delay: float = 0.0, answer: str = "Final Answer: echo-x") -> None:
        super().__init__(model_name="ledger-model", model_type=ModelType.TRANSFORMERS)
        self.delay = delay
        self.answer = answer
        self._is_loaded = True

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def _text(self, prompt: Any) -> str:
        if "echo-x" in str(prompt) or not self._wants_tool(prompt):
            return self.answer
        return ACTION

    @staticmethod
    def _wants_tool(prompt: Any) -> bool:
        return "Action Input" in str(prompt) or "echo" in str(prompt)

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        if self.delay:
            time.sleep(self.delay)
        return GenerationResult(
            text=self._text(prompt), tokens_used=COMPLETION_TOKENS, finish_reason="stop",
            model_name=self.model_name,
            metadata={
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                "cached_input_tokens": CACHED_TOKENS,
                "cache_write_tokens": CACHE_WRITE_TOKENS,
            },
        )

    def generate_stream(self, prompt: Any, config: Any = None, **kwargs: Any):
        for word in self._text(prompt).split(" "):
            yield word + " "

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


class _Echo(BaseTool):
    def __init__(self, delay: float = 0.0) -> None:
        super().__init__(metadata=ToolMetadata(
            name="echo", description="Echo a value back.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(
                name="value", type=ParameterType.STRING, description="The value.",
                required=True,
            )],
        ))
        self.delay = delay

    async def _execute(self, **kwargs: Any) -> str:
        if self.delay:
            time.sleep(self.delay)
        return f"echo-{kwargs.get('value')}"


class _SlowHook(AgentMiddleware):
    """Framework overhead planted on purpose: a sleep before every model call."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def before_model_call(self, ctx: Any) -> None:
        time.sleep(self.seconds)


def _agent(*, tools: bool = False, delay: float = 0.0, tool_delay: float = 0.0,
           name: str = "ledger", answer: str = "Final Answer: echo-x", **cfg: Any) -> Agent:
    return Agent(AgentConfig(
        name=name, model=_Model(delay=delay, answer=answer),
        tools=[_Echo(tool_delay)] if tools else [],
        enable_memory=False, enable_sub_agents=cfg.pop("enable_sub_agents", False),
        raise_on_error=False, **cfg,
    ))


@pytest.fixture(autouse=True)
def _private_state(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "sessions"))


def _median_of(runs: int, make) -> tuple[float, float, Any]:
    ledgers = [make() for _ in range(runs)]
    return (
        statistics.median(led.framework_s for led in ledgers),
        statistics.median(led.model_wait_s for led in ledgers),
        ledgers[-1],
    )


# ------------------------------------------------------------------ counting
def test_a_tool_run_counts_its_calls_tokens_and_cached_tokens():
    response = _agent(tools=True).run("echo x")
    ledger = response.ledger

    assert response.success
    assert (ledger.llm_calls, ledger.tool_calls, ledger.iterations) == (2, 1, 2)
    assert ledger.prompt_tokens == 2 * PROMPT_TOKENS
    assert ledger.completion_tokens == 2 * COMPLETION_TOKENS
    assert ledger.cached_input_tokens == 2 * CACHED_TOKENS
    # The flat keys a caller already reads say the same thing.
    assert response.metadata["prompt_tokens"] == ledger.prompt_tokens
    assert response.metadata["completion_tokens"] == ledger.completion_tokens
    assert response.tokens_used == ledger.total_tokens
    assert [s.llm_calls for s in ledger.steps] == [1, 1]
    assert [s.tool_calls for s in ledger.steps] == [1, 0]
    assert [c.kind for c in ledger.calls] == ["model", "tool", "model"]


def test_a_run_with_no_tools_is_one_call_in_one_step():
    ledger = _agent(answer="The answer is 4.").run("what is 2+2?").ledger
    assert (ledger.llm_calls, ledger.tool_calls, len(ledger.steps)) == (1, 0, 1)
    assert ledger.framework_s + ledger.model_wait_s <= ledger.wall_s + 1e-6


def test_the_split_adds_up_to_the_wall_time():
    ledger = _agent(tools=True, delay=0.01, tool_delay=0.01).run("echo x").ledger
    parts = ledger.model_wait_s + ledger.tool_wait_s + ledger.caller_wait_s + ledger.framework_s
    # Each figure is written to six decimal places.
    assert abs(parts - ledger.wall_s) < 1e-5
    assert ledger.tool_wait_s >= 0.009


# ------------------------------------------------------------------ the split moves with planted overhead
def test_a_delay_planted_in_framework_code_moves_framework_time_only():
    planted = 0.02

    def run(with_hook: bool):
        middleware = [_SlowHook(planted)] if with_hook else None
        return _agent(tools=True).run("echo x", middleware=middleware).ledger

    fw_base, model_base, _ = _median_of(3, lambda: run(False))
    fw_hook, model_hook, ledger = _median_of(3, lambda: run(True))
    expected = planted * ledger.llm_calls
    assert abs((fw_hook - fw_base) - expected) < 0.015, (fw_hook - fw_base, expected)
    assert abs(model_hook - model_base) < 0.005


def test_a_slow_model_moves_model_time_only():
    delay = 0.03
    fw_fast, model_fast, _ = _median_of(3, lambda: _agent(tools=True).run("echo x").ledger)
    fw_slow, model_slow, ledger = _median_of(
        3, lambda: _agent(tools=True, delay=delay).run("echo x").ledger,
    )
    expected = delay * ledger.llm_calls
    assert abs((model_slow - model_fast) - expected) < 0.015
    assert abs(fw_slow - fw_fast) < 0.01


def test_a_stream_consumer_that_pauses_is_caller_time_not_framework_time():
    agent = _agent(answer="one two three four five")
    pause = 0.02
    chunks = 0
    for _ in agent.stream("count to five"):
        time.sleep(pause)
        chunks += 1
    ledger = agent.last_stream_ledger

    assert chunks == 5 and ledger.llm_calls == 1
    assert abs(ledger.caller_wait_s - pause * chunks) < 0.02
    assert ledger.framework_s < 0.02


def test_a_streamed_tool_run_carries_its_ledger_and_keeps_its_usage_keys():
    agent = _agent(tools=True)
    assert "".join(agent.stream("echo x")).strip() == "echo-x"
    ledger = agent.last_stream_ledger

    assert (ledger.llm_calls, ledger.tool_calls) == (2, 1)
    assert agent.last_stream_response.metadata["ledger"]["llm_calls"] == 2
    assert set(agent.last_stream_usage) == {
        "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
        "model_calls", "estimated", "latency_ms", "ttft_ms",
    }


# ------------------------------------------------------------------ children counted once
def test_a_decomposed_run_counts_each_child_once_and_reports_its_whole_token_total():
    from effgen.core.router import RoutingDecision, RoutingStrategy
    from effgen.core.task import SubTask

    agent = _agent(name="parent", answer="Final Answer: done", enable_sub_agents=True)
    decision = RoutingDecision(
        use_sub_agents=True, strategy=RoutingStrategy.SEQUENTIAL_SUB_AGENTS,
        num_sub_agents=2,
        decomposition=[
            SubTask(id="st_1", description="research the market", expected_output="x"),
            SubTask(id="st_2", description="draft the summary", expected_output="x"),
        ],
    )
    agent.router.route = lambda task, context=None: decision
    try:
        response = agent.run("research, then summarise", mode=AgentMode.SUB_AGENTS)
    finally:
        agent.close()
    ledger = response.ledger
    total = ledger.total()

    assert len(ledger.children) == 2
    assert all(child.llm_calls == 1 for child in ledger.children)
    # The parent's own call is its synthesis; the children's calls are not in it.
    assert ledger.llm_calls == 1
    assert total["llm_calls"] == 3
    assert total["prompt_tokens"] == 3 * PROMPT_TOKENS
    # tokens_used is the whole run: the parent's call and both children's.
    assert response.tokens_used == 3 * (PROMPT_TOKENS + COMPLETION_TOKENS)


def test_a_decomposed_run_adds_each_childs_tokens_to_the_token_counters_once():
    from effgen.core.router import RoutingDecision, RoutingStrategy
    from effgen.core.task import SubTask
    from effgen.utils.prometheus_metrics import metrics as prom_metrics

    agent = _agent(name="parent", answer="Final Answer: done", enable_sub_agents=True)
    decision = RoutingDecision(
        use_sub_agents=True, strategy=RoutingStrategy.PARALLEL_SUB_AGENTS,
        num_sub_agents=2,
        decomposition=[
            SubTask(id="st_1", description="research the market", expected_output="x"),
            SubTask(id="st_2", description="draft the summary", expected_output="x"),
        ],
    )
    agent.router.route = lambda task, context=None: decision
    prom_metrics.reset()
    try:
        response = agent.run("research, then summarise", mode=AgentMode.SUB_AGENTS)
    finally:
        agent.close()
    counted = sum(
        float(line.rsplit(" ", 1)[1])
        for line in prom_metrics.export().splitlines()
        if line.startswith("effgen_tokens_used_total{")
    )
    prom_metrics.reset()

    # Each sub-agent's run records its own tokens, so the parent's run adds
    # only the tokens of its own call, and the counter reads the whole task once.
    assert len(response.ledger.children) == 2
    assert counted == response.ledger.total()["total_tokens"] == 3 * (PROMPT_TOKENS + COMPLETION_TOKENS)


def test_a_streamed_calls_cached_prompt_tokens_reach_the_ledger(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    usage = {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46,
             "prompt_tokens_details": {"cached_tokens": 32}}

    class _OpenAIProtocolStream(BaseHTTPRequestHandler):
        """Streams one answer; the last event carries usage, as the protocol sends it."""

        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            base = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
            events = [
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "four"},
                                      "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append({**base, "choices": [], "usage": usage})
            for event in events:
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIProtocolStream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = Agent(AgentConfig(
            name="cached", model="served-model",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            enable_memory=False, enable_sub_agents=False, raise_on_error=False,
        ))
        text = "".join(agent.stream("what is 2+2?"))
    finally:
        server.shutdown()
        server.server_close()
    ledger = agent.last_stream_ledger

    assert "four" in text
    assert (ledger.llm_calls, ledger.prompt_tokens, ledger.completion_tokens) == (1, 40, 6)
    assert ledger.cached_input_tokens == 32
    # The stream's usage keys a caller reads are unchanged.
    assert agent.last_stream_usage["prompt_tokens"] == 40


def test_a_workflow_ledger_holds_every_node_run_once():
    from effgen.core.workflow import WorkflowDAG, WorkflowNode

    dag = WorkflowDAG("three")
    for node_id in ("a", "b", "c"):
        dag.add_node(WorkflowNode(
            id=node_id, agent=_agent(name=f"node-{node_id}", answer="Final Answer: ok"),
        ))
    dag.connect("a", "b")
    dag.connect("b", "c")
    result = dag.run("go")

    ledger = result.metadata["ledger"]
    assert result.success
    assert ledger["kind"] == "workflow" and len(ledger["children"]) == 3
    assert ledger["llm_calls"] == 0
    assert ledger["total"]["llm_calls"] == 3
    assert ledger["total"]["prompt_tokens"] == sum(
        child["total"]["prompt_tokens"] for child in ledger["children"]
    )
    json.dumps(result.to_dict())


def test_a_team_ledger_holds_every_member_run():
    from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern

    orchestrator = MultiAgentOrchestrator()
    orchestrator.create_team(
        "pair",
        [_agent(name="first", answer="Final Answer: one"),
         _agent(name="second", answer="Final Answer: two")],
        pattern=OrchestrationPattern.SEQUENTIAL,
    )
    response = orchestrator.assign_task("do the work", "pair")

    ledger = response.metadata["ledger"]
    assert ledger["kind"] == "team" and len(ledger["children"]) >= 2
    assert ledger["total"]["llm_calls"] == sum(
        child["total"]["llm_calls"] for child in ledger["children"]
    )


def test_a_resumed_run_reports_its_own_calls_and_the_whole_task(tmp_path):
    from effgen.core.checkpoint import CheckpointManager

    first = _agent(answer="Final Answer: done").run("q", checkpoint_dir=str(tmp_path))
    checkpoint_id = first.metadata["checkpoint_id"]
    saved = CheckpointManager(str(tmp_path)).load(checkpoint_id)
    assert saved.ledger["llm_calls"] == 1

    resumed = _agent(answer="Final Answer: done").resume(
        checkpoint_id=checkpoint_id, checkpoint_dir=str(tmp_path),
    )
    ledger = resumed.ledger
    assert ledger.llm_calls == 1
    assert ledger.resumed_from["checkpoint_id"] == checkpoint_id
    assert ledger.cumulative()["llm_calls"] == 2
    assert ledger.cumulative()["total_tokens"] == 2 * (PROMPT_TOKENS + COMPLETION_TOKENS)


def test_a_checkpoint_written_without_a_ledger_resumes_with_calls_unknown(tmp_path):
    from effgen.core.checkpoint import Checkpoint, CheckpointManager

    manager = CheckpointManager(str(tmp_path))
    checkpoint_id = manager.save(Checkpoint(
        checkpoint_id="", agent_name="ledger", task="q", iteration=1, tokens_used=40,
    ))
    ledger = _agent(answer="Final Answer: done").resume(
        checkpoint_id=checkpoint_id, checkpoint_dir=str(tmp_path),
    ).ledger
    assert ledger.cumulative()["llm_calls"] is None
    assert ledger.cumulative()["total_tokens"] == 40 + PROMPT_TOKENS + COMPLETION_TOKENS


def test_concurrent_runs_on_one_agent_keep_separate_ledgers():
    import threading

    agent = _agent(tools=True, delay=0.005)
    served = {"calls": 0}
    lock = threading.Lock()
    generate = agent.model.generate

    def counting(*args: Any, **kwargs: Any) -> GenerationResult:
        with lock:
            served["calls"] += 1
        return generate(*args, **kwargs)

    agent.model.generate = counting
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: agent.run("echo x"), range(16)))
    # How many calls a run needs depends on what the shared agent remembers;
    # what must hold is that each ledger is its own run's and none is lost.
    for response in responses:
        ledger = response.ledger
        assert ledger.prompt_tokens == response.metadata["prompt_tokens"]
        assert ledger.tool_calls == response.tool_calls
        assert ledger.llm_calls * PROMPT_TOKENS == ledger.prompt_tokens
    assert sum(r.ledger.llm_calls for r in responses) == served["calls"]


# ------------------------------------------------------------------ every consumer renders it
def test_the_ledger_is_plain_data_in_the_run_document():
    from effgen.cli.commands.run import run_document
    from effgen.core.ledger import RunLedger

    response = _agent(tools=True).run("echo x")
    document = json.loads(json.dumps(run_document(response)))
    ledger = document["metadata"]["ledger"]

    assert ledger["llm_calls"] == 2 and ledger["tool_calls"] == 1
    assert {"framework_s", "model_wait_s", "tool_wait_s", "steps", "calls", "total"} <= set(ledger)
    assert RunLedger.from_dict(ledger).to_dict() == ledger


def test_the_run_store_record_carries_calls_and_the_time_split():
    from effgen.observability import run_log

    run_log.clear()
    _agent(tools=True).run("echo x")
    record = run_log.read_runs(limit=1)[0]

    assert record["llm_calls"] == 2 and record["tool_calls"] == 1
    assert record["cached_input_tokens"] == 2 * CACHED_TOKENS
    assert record["cache_write_tokens"] == 2 * CACHE_WRITE_TOKENS
    for key in ("model_wait_s", "tool_wait_s", "framework_s"):
        assert isinstance(record[key], float)
    run_log.clear()


def test_prometheus_observes_each_model_call_and_the_run_framework_time():
    from effgen.observability import metrics

    metrics.reset_all()
    agent = _agent(tools=True)
    ledger = agent.run("echo x").ledger
    text = metrics.export_metrics()

    counts = [
        line for line in text.splitlines()
        if line.startswith("effgen_model_call_latency_seconds_count{")
    ]
    assert sum(float(line.rsplit(" ", 1)[1]) for line in counts) == ledger.llm_calls == 2
    assert 'effgen_run_framework_seconds_count{agent="ledger"} 1' in text
    provider = agent._model_provider(agent.model)
    assert metrics.tokens_total.get(
        labels={"provider": provider, "model": "ledger-model", "kind": "cached"}
    ) == 2 * CACHED_TOKENS
    assert metrics.tokens_total.get(
        labels={"provider": provider, "model": "ledger-model", "kind": "cache_write"}
    ) == 2 * CACHE_WRITE_TOKENS
    metrics.reset_all()


def test_the_run_card_shows_the_calls_and_the_time_split():
    from effgen.cli.commands.run import run_document
    from effgen.observability import run_log
    from effgen.ui.report_html_run import _run_body

    run_log.clear()
    response = _agent(tools=True).run("echo x")
    _, _, body = _run_body(run_document(response))
    assert "Model calls" in body and "Framework time" in body

    _, _, stored = _run_body(run_log.read_runs(limit=1)[0])
    assert "Model calls" in stored and "Framework time" in stored
    run_log.clear()


# ------------------------------------------------------------------ nothing a caller reads changed
_VOLATILE = {"latency_ms", "duration_s", "run_id", "thread"}


def test_the_ledger_adds_one_metadata_key_and_changes_no_other(monkeypatch):
    from effgen.core import ledger as ledger_module

    with_ledger = _agent(tools=True).run("echo x")
    monkeypatch.setattr(ledger_module, "open_run", lambda *a, **k: None)
    without = _agent(tools=True).run("echo x")

    assert set(with_ledger.metadata) - set(without.metadata) == {"ledger"}
    assert set(without.metadata) <= set(with_ledger.metadata)
    for key in set(without.metadata) - _VOLATILE:
        assert with_ledger.metadata[key] == without.metadata[key], key
    assert with_ledger.tokens_used == without.tokens_used
    assert without.ledger is None


# ------------------------------------------------------------------ what a cache cost and saved


def test_the_ledger_counts_what_the_cache_served_and_what_it_cost_to_fill():
    """Both halves, per call and per step, and once across a run's children.

    A run that only ever writes a cache is spending more than one that never
    cached at all, and nothing shows that unless the writes are counted apart
    from the reads.
    """
    ledger = _agent(tools=True).run("echo x").ledger

    assert ledger.cached_input_tokens == 2 * CACHED_TOKENS
    assert ledger.cache_write_tokens == 2 * CACHE_WRITE_TOKENS
    assert [c.cache_write_tokens for c in ledger.calls if c.kind == "model"] == (
        [CACHE_WRITE_TOKENS] * 2
    )
    assert sum(step.cache_write_tokens for step in ledger.steps) == 2 * CACHE_WRITE_TOKENS
    assert ledger.own()["cache_write_tokens"] == 2 * CACHE_WRITE_TOKENS
    assert ledger.total()["cache_write_tokens"] == 2 * CACHE_WRITE_TOKENS


def test_a_cache_write_survives_the_document_round_trip():
    from effgen.core.ledger import RunLedger

    response = _agent(tools=True).run("echo x")
    document = response.metadata["ledger"]
    assert document["cache_write_tokens"] == 2 * CACHE_WRITE_TOKENS
    assert RunLedger.from_dict(document).cache_write_tokens == 2 * CACHE_WRITE_TOKENS
    assert RunLedger.from_dict(document).to_dict() == document


def test_the_run_card_names_what_the_cache_served():
    from effgen.cli.commands.run import run_document
    from effgen.ui.report_html_run import _run_body

    response = _agent(tools=True).run("echo x")
    _, _, body = _run_body(run_document(response))
    assert "Cached prompt tokens" in body
    assert f"{2 * CACHE_WRITE_TOKENS} written" in body


def test_a_provider_that_reports_no_cache_gets_no_card_and_no_counters():
    """Zero is not reported as a hit rate of zero; nothing is claimed at all."""
    from effgen.cli.commands.run import run_document
    from effgen.ui.report_html_run import _run_body

    class _Quiet(_Model):
        def generate(self, prompt: Any, config: Any = None, **kwargs: Any):
            result = super().generate(prompt, config, **kwargs)
            result.metadata.pop("cached_input_tokens", None)
            result.metadata.pop("cache_write_tokens", None)
            return result

    agent = Agent(AgentConfig(
        name="quiet", model=_Quiet(), tools=[], max_iterations=3,
        raise_on_error=False, enable_memory=False,
    ))
    response = agent.run("say hello")
    assert response.ledger.cached_input_tokens == 0
    assert response.ledger.cache_write_tokens == 0
    _, _, body = _run_body(run_document(response))
    assert "Cached prompt tokens" not in body
