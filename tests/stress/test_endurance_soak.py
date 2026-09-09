"""Long-run endurance soaks measured as trend lines.

Each scenario drives one subsystem for many iterations while
:mod:`tests.stress.soak` samples resident memory, open file descriptors,
thread count, direct child processes, reserved device memory and the live
object count. The assertion is on the fitted slope over the post-warmup window
as well as the total delta, so a resource that grows a little on every
iteration fails even when a single before/after reading would not.

Every budget lives in :data:`BUDGETS` and every iteration count in
:data:`ITERATIONS`, so a new scenario is a row plus the function that performs
one iteration.

``EFFGEN_SOAK_SCALE`` multiplies the iteration counts: scale 1 is the short
form that runs in minutes, and the same code at a higher scale is the hours-long
run. Set ``EFFGEN_SOAK_OUT`` to a directory to have each scenario write its
trend table and per-sample CSV there.

All cases carry ``expensive`` and ``slow``; the CI lanes deselect both.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from .soak import (
    GrowthBudget,
    ResourceSampler,
    SoakReport,
    assert_bounded,
    scale_iterations,
)

# The suite-wide per-test timeout is sized for ordinary tests. A soak runs for
# minutes at scale 1 and for hours at a raised scale, so it opts out of that cap
# rather than reporting a cut-off run as a failure; the lane markers above are
# what keep these cases out of CI.
pytestmark = [pytest.mark.expensive, pytest.mark.slow, pytest.mark.timeout(0)]


# ---------------------------------------------------------------------------
# The table: one row per scenario
# ---------------------------------------------------------------------------

ITERATIONS: dict[str, int] = {
    "agent_construct_close": 400,
    "agent_construct_dropped": 400,
    "repl_worker_churn": 300,
    "session_store_growth": 400,
    "conversation_memory_growth": 1_000,
    "agent_run_live": 120,
    # Enough iterations that the CUDA caching allocator's settling (it stops
    # growing its reserve by roughly the 30th generation, measured over 600)
    # lands inside the discarded warmup window rather than in the fit.
    "agent_run_local": 200,
    "model_load_unload": 12,
    "server_requests": 400,
}

BUDGETS: dict[str, GrowthBudget] = {
    "agent_construct_close": GrowthBudget(
        rss_mb_per_1k=40, rss_mb_total=48,
        fds_per_1k=2, fds_total=4,
        threads_per_1k=2, threads_total=2,
        children_per_1k=1, children_total=0,
        gpu_mb_per_1k=0, gpu_mb_total=0,
        gc_objects_per_1k=5_000, gc_objects_total=20_000,
    ),
    "agent_construct_dropped": GrowthBudget(
        rss_mb_per_1k=40, rss_mb_total=48,
        fds_per_1k=2, fds_total=4,
        threads_per_1k=2, threads_total=2,
        children_per_1k=1, children_total=0,
        gpu_mb_per_1k=0, gpu_mb_total=0,
        gc_objects_per_1k=5_000, gc_objects_total=20_000,
    ),
    "repl_worker_churn": GrowthBudget(
        rss_mb_per_1k=40, rss_mb_total=48,
        fds_per_1k=4, fds_total=8,
        threads_per_1k=2, threads_total=4,
        children_per_1k=1, children_total=1,
        gpu_mb_per_1k=0, gpu_mb_total=0,
        gc_objects_per_1k=8_000, gc_objects_total=20_000,
    ),
    "session_store_growth": GrowthBudget(
        rss_mb_per_1k=40, rss_mb_total=48,
        fds_per_1k=2, fds_total=4,
        threads_per_1k=2, threads_total=2,
        children_per_1k=1, children_total=0,
        gpu_mb_per_1k=0, gpu_mb_total=0,
        gc_objects_per_1k=8_000, gc_objects_total=20_000,
    ),
    "conversation_memory_growth": GrowthBudget(
        rss_mb_per_1k=40, rss_mb_total=48,
        fds_per_1k=2, fds_total=4,
        threads_per_1k=2, threads_total=2,
        children_per_1k=1, children_total=0,
        gpu_mb_per_1k=0, gpu_mb_total=0,
        gc_objects_per_1k=8_000, gc_objects_total=20_000,
    ),
    "agent_run_live": GrowthBudget(
        rss_mb_per_1k=60, rss_mb_total=64,
        fds_per_1k=8, fds_total=12,
        threads_per_1k=4, threads_total=8,
        children_per_1k=1, children_total=1,
        gpu_mb_per_1k=0, gpu_mb_total=0,
        gc_objects_per_1k=20_000, gc_objects_total=50_000,
    ),
    "agent_run_local": GrowthBudget(
        rss_mb_per_1k=120, rss_mb_total=128,
        fds_per_1k=8, fds_total=12,
        threads_per_1k=4, threads_total=8,
        children_per_1k=1, children_total=1,
        gpu_mb_per_1k=64, gpu_mb_total=32,
        gc_objects_per_1k=20_000, gc_objects_total=50_000,
    ),
    # A dozen heavy cycles carry no meaningful per-1k slope, so this row
    # asserts the totals and leaves the slopes reported but unbounded. The
    # per-cycle device residue is asserted directly inside the scenario.
    "model_load_unload": GrowthBudget(
        rss_mb_per_1k=None, rss_mb_total=256,
        fds_per_1k=None, fds_total=8,
        threads_per_1k=None, threads_total=4,
        children_per_1k=None, children_total=1,
        gpu_mb_per_1k=None, gpu_mb_total=32,
        gc_objects_per_1k=None, gc_objects_total=50_000,
    ),
    # Sampled on the server process, which this process cannot walk: the
    # object count and device memory are not readable from outside and are
    # reported as sentinels rather than asserted.
    "server_requests": GrowthBudget(
        rss_mb_per_1k=40, rss_mb_total=128,
        fds_per_1k=12, fds_total=16,
        threads_per_1k=4, threads_total=8,
        children_per_1k=1, children_total=2,
        gpu_mb_per_1k=None, gpu_mb_total=None,
        gc_objects_per_1k=None, gc_objects_total=None,
    ),
}

#: Samples discarded before the fit: imports, lazy caches and the allocator's
#: first growth all land there and would otherwise be read as a slope.
WARMUP_FRACTION = 0.25

@dataclass(frozen=True)
class LiveFamily:
    """A cloud model the live scenarios run against.

    ``max_tokens`` is per family because a reasoning model spends part of the
    budget before its first visible token — a budget that answers on one family
    returns nothing on another. ``min_interval_s`` paces the calls to what the
    account's tier allows, so a soak measures resource trends rather than the
    provider's rate limiter.
    """

    model: str
    key_var: str
    max_tokens: int
    min_interval_s: float = 0.0

    @property
    def provider(self) -> str:
        return self.model.split(":", 1)[0]

    @property
    def available(self) -> bool:
        return bool(os.environ.get(self.key_var))


#: A family whose key is absent skips rather than failing.
LIVE_FAMILIES: list[LiveFamily] = [
    # A reasoning model: a 16-token budget is spent entirely on hidden
    # reasoning and returns no answer at all.
    LiveFamily("groq:openai/gpt-oss-20b", "GROQ_API_KEY", max_tokens=4096),
    LiveFamily("openai:gpt-5-nano", "OPENAI_API_KEY", max_tokens=4096),
    # The free tier allows 15 requests per minute for this model.
    LiveFamily("gemini:gemini-3.1-flash-lite", "GOOGLE_API_KEY",
               max_tokens=16, min_interval_s=4.2),
]

LOCAL_MODEL = "transformers:Qwen/Qwen2.5-1.5B-Instruct"


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_cuda(), reason="requires a CUDA GPU")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _sample_every(iterations: int) -> int:
    """Take roughly 40 readings whatever the scale, at least one per iteration."""
    return max(1, iterations // 40)


def _write_evidence(report: SoakReport) -> None:
    out = os.environ.get("EFFGEN_SOAK_OUT")
    if not out:
        return
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    stem = report.scenario.replace("/", "_").replace(":", "_")
    (directory / f"{stem}.txt").write_text(report.table() + "\n")
    (directory / f"{stem}.csv").write_text(report.samples_csv() + "\n")
    (directory / f"{stem}.json").write_text(report.to_json() + "\n")


def run_soak(
    scenario: str,
    one_iteration: Callable[[int], None],
    *,
    budget_key: str | None = None,
    iterations: int | None = None,
    setup: Callable[[], None] | None = None,
    pid: int | None = None,
) -> SoakReport:
    """Drive *one_iteration* and return the sampled trend."""
    key = budget_key or scenario
    total = iterations if iterations is not None else scale_iterations(ITERATIONS[key])
    every = _sample_every(total)
    if setup is not None:
        setup()
    sampler = ResourceSampler(scenario, pid=pid)
    sampler.sample(0)
    for i in range(total):
        one_iteration(i)
        if (i + 1) % every == 0:
            sampler.sample(i + 1)
    report = sampler.report(warmup_fraction=WARMUP_FRACTION)
    print("\n" + report.table())
    _write_evidence(report)
    return report


def assert_scenario(report: SoakReport, scenario: str) -> None:
    assert_bounded(report, BUDGETS[scenario])


# ---------------------------------------------------------------------------
# Agent construction and teardown
# ---------------------------------------------------------------------------


def _agent_config(model: str, *, with_tools: bool = False):
    from effgen.core.agent import AgentConfig
    from effgen.tools import get_registry

    tools = []
    if with_tools:
        registry = get_registry()
        tools = [registry.get_tool_sync(name) for name in ("calculator", "python_repl")]
    return AgentConfig(model=model, name="soak-agent", tools=tools, max_iterations=2)


def _first_keyed_family() -> LiveFamily | None:
    """The first family whose key is present, or ``None`` if none is."""
    for family in LIVE_FAMILIES:
        if family.available:
            return family
    return None


@pytest.mark.live
def test_agent_construct_and_close_is_bounded() -> None:
    """Building and closing an agent many times settles, it does not accumulate.

    No model call is made — this measures what construction and teardown alone
    leave behind (an adapter, its HTTP client, the tool registry wiring).
    """
    family = _first_keyed_family()
    if family is None:
        pytest.skip("no cloud provider key available")
    model = family.model
    from effgen.core.agent import Agent

    def one(_i: int) -> None:
        agent = Agent(config=_agent_config(model, with_tools=True))
        agent.close()

    report = run_soak("agent_construct_close", one)
    assert_scenario(report, "agent_construct_close")


class _CountingHandler(logging.Handler):
    """Counts records instead of keeping them.

    An agent collected without ``close()`` logs a warning saying so, and any
    handler that retains records — pytest's capture among them — then holds one
    record per iteration. That is the harness accumulating, not the agent, so
    the scenario counts the warnings and keeps none.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0
        self.unclosed = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        if "without calling close()" in record.getMessage():
            self.unclosed += 1


@pytest.mark.live
def test_agent_constructed_and_dropped_is_bounded() -> None:
    """An agent dropped without ``close()`` is reclaimed, and says it was.

    Forgetting the teardown is the common case in a script, so the resources an
    agent holds must be released when it becomes unreachable — and the run must
    say that it happened rather than passing silently.
    """
    family = _first_keyed_family()
    if family is None:
        pytest.skip("no cloud provider key available")
    model = family.model
    from effgen.core.agent import Agent

    agent_logger = logging.getLogger("effgen.core.agent")
    counter = _CountingHandler()
    previous_handlers = agent_logger.handlers[:]
    previous_propagate = agent_logger.propagate
    agent_logger.handlers = [counter]
    agent_logger.propagate = False

    def one(_i: int) -> None:
        Agent(config=_agent_config(model, with_tools=True))  # deliberately not closed

    try:
        report = run_soak("agent_construct_dropped", one)
    finally:
        agent_logger.handlers = previous_handlers
        agent_logger.propagate = previous_propagate
    assert_scenario(report, "agent_construct_dropped")

    # Every dropped agent is collected, and each one reports that it was not
    # closed. A count far below the iteration count would mean agents are being
    # kept alive somewhere instead.
    iterations = report.iterations - 1
    print(f"unclosed-agent warnings: {counter.unclosed} for {iterations} drops")
    assert counter.unclosed >= iterations * 0.9, (
        f"only {counter.unclosed} of {iterations} dropped agents were collected"
    )


# ---------------------------------------------------------------------------
# REPL worker churn
# ---------------------------------------------------------------------------


def test_repl_worker_churn_is_bounded() -> None:
    """Sustained REPL traffic over rotating session ids holds a bounded pool."""
    from effgen.tools.builtin.python_repl import PythonREPL

    cap = 4
    baseline_children = ResourceSampler("repl_baseline").sample(0).children
    repl = PythonREPL(max_sessions=cap)
    loop = asyncio.new_event_loop()

    def one(i: int) -> None:
        result = loop.run_until_complete(
            repl.execute(code=f"x = {i}\nx + 1", session_id=f"soak-{i % 40}", timeout=60)
        )
        assert result.success is True, result.error
        assert result.output["result"] == i + 1
        if i % 25 == 24:
            repl.reset_session(f"soak-{i % 40}")

    try:
        report = run_soak("repl_worker_churn", one)
        assert_scenario(report, "repl_worker_churn")

        settled = report.samples[-1].children - baseline_children
        assert settled <= cap, f"{settled} workers alive for a cap of {cap}"
        loop.run_until_complete(repl.cleanup())
        time.sleep(0.5)
        final = ResourceSampler("repl_worker_churn_final").sample(0).children
        assert final - baseline_children == 0, (
            f"{final - baseline_children} workers survived cleanup()"
        )
    finally:
        loop.run_until_complete(repl.cleanup())
        loop.close()


# ---------------------------------------------------------------------------
# Session store growth
# ---------------------------------------------------------------------------


def test_session_store_growth_is_bounded(tmp_path) -> None:
    """A store that keeps growing stays linear to read and flat in memory.

    Every listing reads the whole store, so the cost per session file must not
    rise as the store fills — a quadratic listing turns into a hang for a user
    who never deletes a session.
    """
    from effgen.core.session import Session, SessionManager

    store = tmp_path / "sessions"
    store.mkdir()
    manager = SessionManager(str(store))
    scan_ms_per_file: list[tuple[int, float]] = []

    def one(i: int) -> None:
        session = Session(session_id=f"soak-{i:06d}", agent_name="soak")
        session.add_user_message(f"question {i} " * 20)
        session.add_assistant_message(f"answer {i} " * 20)
        session.save(str(store))
        Session.load(f"soak-{i:06d}", str(store))
        if i % 20 == 0:
            started = time.monotonic()
            listed = manager.list_sessions()
            elapsed_ms = (time.monotonic() - started) * 1000
            assert len(listed) == i + 1, f"listed {len(listed)} of {i + 1} sessions"
            scan_ms_per_file.append((i + 1, elapsed_ms / (i + 1)))

    report = run_soak("session_store_growth", one)
    assert_scenario(report, "session_store_growth")

    # Cost per file, first quarter of the run against the last.
    assert len(scan_ms_per_file) >= 8, "not enough listings to compare"
    quarter = max(1, len(scan_ms_per_file) // 4)
    early = sum(ms for _, ms in scan_ms_per_file[:quarter]) / quarter
    late = sum(ms for _, ms in scan_ms_per_file[-quarter:]) / quarter
    print(f"scan cost per session file: {early:.4f} ms early -> {late:.4f} ms late")
    assert late <= early * 3, (
        f"listing cost per file grew {early:.4f} -> {late:.4f} ms "
        f"over {scan_ms_per_file[-1][0]} sessions — the scan is not linear"
    )


def test_conversation_memory_growth_is_bounded() -> None:
    """A session that never ends keeps its stored context inside its budget.

    Older turns are folded into summaries, and every summary is replayed into
    the next prompt — so the summaries have to be bounded too, or the prompt
    grows until it passes the model's context window and every further call is
    refused.
    """
    from effgen.memory.short_term import ShortTermMemory

    max_tokens = 4096
    memory = ShortTermMemory(max_tokens=max_tokens, max_messages=100)
    stored: list[int] = []

    def one(i: int) -> None:
        memory.add_user_message(f"Question {i}: " + "context padding words " * 12)
        memory.add_assistant_message(f"Answer {i}: " + "response words here " * 12)
        stored.append(memory.get_token_count())

    report = run_soak("conversation_memory_growth", one)
    assert_scenario(report, "conversation_memory_growth")

    turns = len(stored)
    print(
        f"stored tokens over {turns} turns: start {stored[0]} "
        f"peak {max(stored)} end {stored[-1]} (budget {max_tokens}); "
        f"{len(memory.summaries)} summaries holding "
        f"{memory.summary_token_count()} tokens"
    )
    assert max(stored) <= max_tokens, (
        f"stored context peaked at {max(stored)} tokens against a "
        f"max_tokens of {max_tokens} over {turns} turns"
    )
    # The tail must not sit above the start: that is the growth shape.
    tail = stored[-turns // 4 :]
    head = stored[turns // 4 : turns // 2]
    assert max(tail) <= max(head) + 256, (
        f"stored context still rising: {max(head)} tokens mid-run, "
        f"{max(tail)} at the end"
    )


# ---------------------------------------------------------------------------
# Live agent runs
# ---------------------------------------------------------------------------


def _failure_reason(response) -> str:
    """The classified reason a run failed, from the response metadata."""
    error = (response.metadata or {}).get("error")
    if isinstance(error, dict):
        return f"{error.get('type', '?')}: {str(error.get('message', ''))[:160]}"
    return str(error or response.output)[:160]


@pytest.mark.live
@pytest.mark.parametrize("family", LIVE_FAMILIES, ids=lambda f: f.provider)
def test_agent_run_live_is_bounded(family: LiveFamily) -> None:
    """Repeated runs on one agent hold steady across a long session."""
    if not family.available:
        pytest.skip(f"{family.key_var} not set")
    from effgen.core.agent import Agent

    agent = Agent(config=_agent_config(family.model))
    failures: list[str] = []
    next_call_at = 0.0

    def one(i: int) -> None:
        nonlocal next_call_at
        wait = next_call_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        next_call_at = time.monotonic() + family.min_interval_s
        response = agent.run(
            f"Reply with the number {i % 97}. Digits only.",
            max_tokens=family.max_tokens,
        )
        if not response.success:
            failures.append(f"run {i}: {_failure_reason(response)}")

    iterations = scale_iterations(ITERATIONS["agent_run_live"])
    try:
        report = run_soak(f"agent_run_live-{family.provider}", one,
                          budget_key="agent_run_live", iterations=iterations)
        assert_scenario(report, "agent_run_live")
    finally:
        agent.close()
    # A provider-side rate limit is not a leak, so a few failures are tolerated;
    # a mostly-failing run did not exercise what the measurement claims to.
    print(f"failed runs: {len(failures)} of {iterations}")
    for line in failures[:5]:
        print("  " + line)
    assert len(failures) <= iterations * 0.2, (
        f"{len(failures)} of {iterations} runs failed:\n" + "\n".join(failures[:5])
    )


@requires_gpu
def test_agent_run_local_is_bounded() -> None:
    """Repeated runs against a resident local model hold steady on the GPU."""
    from effgen.core.agent import Agent

    agent = Agent(config=_agent_config(LOCAL_MODEL))
    try:
        def one(i: int) -> None:
            response = agent.run(
                f"Reply with the number {i % 97}. Digits only.", max_tokens=16
            )
            assert response.success, response.error

        report = run_soak("agent_run_local", one)
        assert_scenario(report, "agent_run_local")
    finally:
        agent.close()


@requires_gpu
def test_model_load_unload_cycles_return_device_memory() -> None:
    """Every load/unload cycle returns the device memory it took."""
    import gc

    import torch

    from effgen.models import load_model

    def reserved() -> float:
        return sum(
            torch.cuda.memory_reserved(d) for d in range(torch.cuda.device_count())
        ) / 1024**2

    # A model another scenario in this process left resident would sit in the
    # baseline, and every cycle would then measure below it and pass whatever it
    # found. So the floor is taken after one complete cycle of this scenario's
    # own, and the peak of that cycle proves the floor is one a residue would
    # show above.
    warmup = load_model(LOCAL_MODEL)
    assert (warmup.generate("Reply with one word: hello", max_tokens=8).text or "").strip()
    peak = reserved()
    warmup.unload()
    del warmup
    gc.collect()
    torch.cuda.empty_cache()
    baseline = reserved()
    assert peak > baseline + 32, (
        f"loading the model did not raise reserved memory above the floor "
        f"({baseline:.0f} -> {peak:.0f} MB): a cycle that failed to release "
        f"would not be visible"
    )
    residues: list[float] = []

    def one(i: int) -> None:
        model = load_model(LOCAL_MODEL)
        result = model.generate("Reply with one word: hello", max_tokens=8)
        assert (result.text or "").strip(), f"cycle {i} produced no output"
        model.unload()
        del model
        gc.collect()
        after = reserved()
        residues.append(after - baseline)
        assert after <= baseline + 32, (
            f"cycle {i}: {after - baseline:.0f}MB still reserved "
            f"(baseline {baseline:.0f}MB)"
        )

    report = run_soak("model_load_unload", one)
    print("reserved above baseline per cycle (MB): "
          + ", ".join(f"{r:.0f}" for r in residues))
    assert_scenario(report, "model_load_unload")


# ---------------------------------------------------------------------------
# Server under sustained load
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """A port nothing is listening on right now.

    A fixed port that another process already holds would leave the spawned
    server unable to bind while the health check still answers — from the other
    listener — and the run would sample a dead process and report every metric
    as unreadable, which reads as a pass.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_server(port: int, env: dict[str, str], log_path: Path) -> subprocess.Popen:
    """Start ``effgen serve`` with its output going to a file.

    Deliberately not a pipe. Nothing reads the server's output while a scenario
    runs, so a pipe fills its buffer after a few hundred log lines and the
    server blocks forever inside a write it can never finish — every request
    after that times out against a process that is still alive and still
    listening. A file has no such ceiling, and the failure paths below still
    get the whole log.
    """
    with open(log_path, "w") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "effgen.cli", "serve", "--port", str(port)],
            env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
        )


def _server_log(proc: subprocess.Popen, log_path: Path | None) -> str:
    """Whatever the server has written so far."""
    if log_path is not None:
        try:
            return log_path.read_text(errors="replace")
        except OSError:
            return ""
    return proc.stdout.read() if proc.stdout else ""


def _wait_for_server(
    base: str,
    proc: subprocess.Popen,
    deadline_s: float = 90.0,
    log_path: Path | None = None,
) -> None:
    import httpx

    end = time.monotonic() + deadline_s
    last: Exception | None = None
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise AssertionError(
                f"server exited with {proc.returncode} before answering:\n"
                + _server_log(proc, log_path)[-2000:]
            )
        try:
            httpx.get(f"{base}/health", timeout=5.0)
            return
        except Exception as exc:  # server not up yet
            last = exc
            time.sleep(1.0)
    raise AssertionError(f"server did not start within {deadline_s}s: {last}")


@pytest.mark.live
def test_server_under_sustained_load_is_bounded(tmp_path) -> None:
    """A server answering a long stream of requests holds a bounded footprint."""
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    import httpx

    family = _first_keyed_family()
    if family is None:
        pytest.skip("no cloud provider key available")
    model = family.model

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    api_key = "soak-" + os.urandom(8).hex()
    env = dict(os.environ, EFFGEN_API_KEY=api_key, EFFGEN_HOME=str(tmp_path))
    log_path = tmp_path / "server.log"
    proc = _spawn_server(port, env, log_path)
    statuses: dict[int, int] = {}
    try:
        _wait_for_server(base, proc, log_path=log_path)
        client = httpx.Client(timeout=120.0)
        headers = {"Authorization": f"Bearer {api_key}"}

        def one(i: int) -> None:
            try:
                response = client.post(
                    f"{base}/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user",
                             "content": f"Reply with the number {i % 97}. Digits only."}
                        ],
                        "max_tokens": 16,
                    },
                )
            except httpx.HTTPError:
                statuses[-1] = statuses.get(-1, 0) + 1
                return
            statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
            if response.status_code == 200:
                assert response.json()["choices"], "200 with no choices"
            else:
                # Anything that is not a success is a typed envelope, never an
                # empty body or an HTML page.
                assert "error" in response.json(), response.text[:200]

        report = run_soak("server_requests", one, pid=proc.pid)
        print(f"statuses: {dict(sorted(statuses.items()))}")
        # An exited server reads as every metric unreadable, and an unreadable
        # metric is skipped rather than failed — so confirm the readings came
        # from a process that was alive for the whole run.
        assert proc.poll() is None, f"server exited with {proc.returncode} during the load"
        assert report.trends["rss_mb"].last > 0, (
            "resident memory was never readable on the server process"
        )
        assert_scenario(report, "server_requests")
        assert statuses.get(-1, 0) == 0, f"{statuses.get(-1)} transport failures"

        forged = client.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": "Bearer not-the-key"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert forged.status_code == 401, (
            f"forged token accepted after the load: {forged.status_code}"
        )
        client.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            proc.wait(timeout=20)
