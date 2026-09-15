"""A run's own overhead has a ceiling, and the work that grew with a run is counted.

The model here is served over the OpenAI protocol from a thread in this process
and answers each request from the request alone — a tool call per step until the
task's steps are done, then an answer — so a run's time is effGen's own work plus
a local socket round trip. On that workload:

* the framework time a run's ledger reports stays under a ceiling per shape: a
  single answer, one tool call, ten steps, a hundred steps across ten tools, and
  sixteen one-tool runs at once. Each ceiling is several times what the shape
  takes on the machine it was set on, because a shared machine's clock is noisy.
  The sixteen runs wait 40 ms on each model call, as a served model keeps them
  waiting, so they overlap the way concurrent agents do instead of all queueing
  for the interpreter at the same instant;
* the work that grew with a run's length or with an answer's size is counted
  rather than timed: how many times prompts are encoded to count their tokens,
  how many pattern scans reading an answer costs, and how many cost records and
  history lines a run writes.

Every import below is a name that existed before these ceilings did, so the file
collects against an earlier tree and fails there on its assertions.
"""

from __future__ import annotations

import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from typing import Any

import pytest

from effgen.core import agent_runtime, tool_calling
from effgen.core.agent import Agent, AgentConfig
from effgen.models import _adapter_utils
from effgen.models._cost_store import SQLiteCostStore
from effgen.observability import run_log
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

#: Framework milliseconds a run of each shape may take, as the median of its runs.
FRAMEWORK_CEILING_MS = {
    "none": 4.2,
    "one_tool": 9.8,
    "ten_step": 54.8,
    "hundred_step": 503.6,
    "sixteen_at_once": 78.7,
}
#: Steps, tools, and how many runs the median is taken over.
SHAPES = {
    "none": (0, 0, 15),
    "one_tool": (1, 1, 10),
    "ten_step": (10, 1, 5),
    "hundred_step": (100, 10, 3),
}

_STEPS = re.compile(r"STEPS=(\d+)")
_DONE = re.compile(r"step-result-(\d+)")
_MODEL_MS = re.compile(r"MODEL_MS=(\d+)")
_serial = count(1)


def _request_text(request: dict[str, Any]) -> str:
    parts = []
    for message in request.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        parts.append(str(content or ""))
    return "\n".join(parts)


class _ScriptedModel(BaseHTTPRequestHandler):
    """Answers from the request alone: a call per step until the task's steps are done."""

    protocol_version = "HTTP/1.1"
    # Headers and body leave in one write with Nagle's algorithm off, so no
    # response waits on the client's delayed acknowledgement.
    disable_nagle_algorithm = True

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        head = (
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        )
        self.wfile.write(head.encode() + body)

    def do_GET(self) -> None:
        self._send({"object": "list", "data": [{"id": "scripted", "object": "model"}]})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        text = _request_text(request)
        latency = _MODEL_MS.search(text)
        if latency:
            time.sleep(int(latency.group(1)) / 1000)
        steps = _STEPS.search(text)
        wanted = int(steps.group(1)) if steps else 0
        done = max((int(k) for k in _DONE.findall(text)), default=0)
        tools = request.get("tools") or []
        if tools and done < wanted:
            names = [(t.get("function") or {}).get("name") for t in tools]
            message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{done + 1}", "type": "function",
                "function": {
                    "name": names[done % len(names)],
                    "arguments": json.dumps({"step": str(done + 1)}),
                },
            }]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": f"Final Answer: finished after {done} steps."}
            finish = "stop"
        prompt_tokens = len(text) // 4 + 1
        self._send({
            "id": "scripted", "object": "chat.completion", "created": 0,
            "model": request.get("model"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 8,
                      "total_tokens": prompt_tokens + 8},
        })


class _StepTool(BaseTool):
    """Returns the result of the step it is given, at once."""

    def __init__(self, name: str) -> None:
        super().__init__(metadata=ToolMetadata(
            name=name,
            description="Record one step of the task and return its result.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(
                name="step", type=ParameterType.STRING,
                description="The number of the step.", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        return f"step-result-{kwargs.get('step')}"


class _CountingEncoding:
    """Stands in for a BPE encoding: a token per word, and a count of every text encoded."""

    name = "counting"

    def __init__(self) -> None:
        self.encoded = 0
        self._lock = threading.Lock()

    def encode(self, text: str, **_: Any) -> list[str]:
        with self._lock:
            self.encoded += 1
        return text.split()


class _EveryName(dict):
    """An encoding table that answers every encoding name with one encoding."""

    def __init__(self, encoding: _CountingEncoding) -> None:
        super().__init__()
        self._encoding = encoding

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: str) -> _CountingEncoding:
        return self._encoding


class _CountingPattern:
    """A compiled pattern that counts every scan made with it."""

    _SCANS = ("sub", "subn", "search", "match", "fullmatch", "finditer", "findall", "split")

    def __init__(self, pattern: re.Pattern[str], scans: list[int]) -> None:
        self._pattern = pattern
        self._scans = scans

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._pattern, name)
        if name not in self._SCANS:
            return attribute

        def scan(*args: Any, **kwargs: Any) -> Any:
            self._scans[0] += 1
            return attribute(*args, **kwargs)

        return scan


class _CountingRe:
    """The ``re`` module as a reader's module sees it, counting every scan made through it."""

    def __init__(self, scans: list[int]) -> None:
        self._scans = scans

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(re, name)
        if name not in _CountingPattern._SCANS:
            return attribute

        def scan(*args: Any, **kwargs: Any) -> Any:
            self._scans[0] += 1
            return attribute(*args, **kwargs)

        return scan


@pytest.fixture(scope="module")
def model_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedModel)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def encoding(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    fake = _CountingEncoding()
    monkeypatch.setattr(_adapter_utils, "_bpe_encodings", _EveryName(fake))
    return fake


@pytest.fixture
def writes(monkeypatch):
    counts = {"cost_records": 0, "history_lines": 0}
    lock = threading.Lock()
    insert = SQLiteCostStore.insert
    append = run_log._append_to_file

    def counted_insert(self: SQLiteCostStore, *args: Any, **kwargs: Any) -> None:
        with lock:
            counts["cost_records"] += 1
        insert(self, *args, **kwargs)

    def counted_append(record: dict[str, Any]) -> None:
        with lock:
            counts["history_lines"] += 1
        append(record)

    monkeypatch.setattr(SQLiteCostStore, "insert", counted_insert)
    monkeypatch.setattr(run_log, "_append_to_file", counted_append)
    return counts


@pytest.fixture
def pattern_scans(monkeypatch):
    scans = [0]
    for module in (agent_runtime, tool_calling):
        for name, value in list(vars(module).items()):
            if isinstance(value, re.Pattern):
                monkeypatch.setattr(module, name, _CountingPattern(value, scans))
            elif isinstance(value, tuple) and value and all(isinstance(v, re.Pattern) for v in value):
                monkeypatch.setattr(module, name, tuple(_CountingPattern(v, scans) for v in value))
        # A pattern written inline (``re.sub(r"...", ...)``) is scanned through the module.
        monkeypatch.setattr(module, "re", _CountingRe(scans))
    return scans


def _agent(url: str, steps: int, tools: int) -> Agent:
    return Agent(AgentConfig(
        name="budgeted", model="scripted", base_url=url,
        tools=[_StepTool("step_tool" if tools == 1 else f"step_tool_{i}") for i in range(tools)],
        max_iterations=max(10, steps + 5), enable_memory=False, enable_sub_agents=False,
        raise_on_error=False,
    ))


def _task(steps: int, model_ms: int = 0) -> str:
    # Every task is new, so no run repeats a prompt an earlier run sent.
    latency = f" MODEL_MS={model_ms}." if model_ms else ""
    return (
        f"Work through the task one step at a time. STEPS={steps}. RUN={next(_serial)}.{latency} "
        "Call the tool once for each step, in order, then give the final answer."
    )


def _run(url: str, steps: int, tools: int):
    agent = _agent(url, steps, tools)
    try:
        return agent.run(_task(steps))
    finally:
        agent.close()


def _sixteen_at_once(url: str, pool: ThreadPoolExecutor) -> list:
    """Sixteen one-tool runs started together on *pool*, each model call taking 40 ms.

    The agents are built first: building an agent is not part of a run, and on
    sixteen threads it would compete with the runs being measured.
    """
    agents = [_agent(url, 1, 1) for _ in range(16)]
    tasks = [_task(1, model_ms=40) for _ in agents]
    try:
        return list(pool.map(lambda pair: pair[0].run(pair[1]), zip(agents, tasks)))
    finally:
        for agent in agents:
            agent.close()


# ------------------------------------------------------------------ ceilings
@pytest.mark.parametrize("shape", list(SHAPES))
def test_framework_time_per_run_stays_under_its_ceiling(shape, model_url, encoding, writes):
    steps, tools, runs = SHAPES[shape]
    _run(model_url, steps, tools)  # the first run in a process pays for lazy imports
    responses = [_run(model_url, steps, tools) for _ in range(runs)]

    for response in responses:
        assert response.success, response.output
        assert (response.ledger.llm_calls, response.ledger.tool_calls) == (steps + 1, steps)
    framework_ms = statistics.median(1000 * r.ledger.framework_s for r in responses)
    assert framework_ms <= FRAMEWORK_CEILING_MS[shape], (shape, framework_ms)


def test_sixteen_runs_at_once_each_stay_under_the_ceiling(model_url, encoding, writes):
    with ThreadPoolExecutor(max_workers=16) as pool:
        _sixteen_at_once(model_url, pool)
        responses = [r for _ in range(3) for r in _sixteen_at_once(model_url, pool)]

    for response in responses:
        assert (response.ledger.llm_calls, response.ledger.tool_calls) == (2, 1)
    framework_ms = statistics.median(1000 * r.ledger.framework_s for r in responses)
    assert framework_ms <= FRAMEWORK_CEILING_MS["sixteen_at_once"], framework_ms


# ------------------------------------------------------------------ counted work
@pytest.mark.parametrize(("steps", "tools"), [(0, 0), (10, 1), (100, 10)])
def test_each_request_is_encoded_once_however_long_the_run(steps, tools, model_url, encoding, writes):
    _run(model_url, steps, tools)
    before = encoding.encoded
    response = _run(model_url, steps, tools)
    encoded = encoding.encoded - before

    # A request is measured by the context budget, checked against the context
    # window and counted again for calibration. It is the same text each time,
    # so it is encoded the first time and read back after.
    assert response.ledger.llm_calls == steps + 1
    assert encoded <= response.ledger.llm_calls + 1, (encoded, response.ledger.llm_calls)


def test_a_run_writes_a_cost_record_per_model_call_and_one_history_line(model_url, encoding, writes):
    before = dict(writes)
    response = _run(model_url, 10, 1)

    assert writes["cost_records"] - before["cost_records"] == response.ledger.llm_calls == 11
    assert writes["history_lines"] - before["history_lines"] == 1


def test_a_remembered_token_count_equals_a_fresh_one(encoding):
    text = f"word{next(_serial)} " + "word " * 399
    first = _adapter_utils.estimate_tokens(text)
    encoded = encoding.encoded

    assert _adapter_utils.estimate_tokens(text) == first == 400
    assert encoding.encoded == encoded


def test_remembered_token_counts_hold_a_bounded_amount_of_text(encoding, monkeypatch):
    monkeypatch.setattr(_adapter_utils, "_TOKEN_COUNTS_MAX_CHARS", 10_000)
    monkeypatch.setattr(_adapter_utils, "_TOKEN_COUNTS_MAX_ENTRIES", 8)
    for i in range(40):
        _adapter_utils.estimate_tokens(f"text{next(_serial)} " + "x" * 1_000 + str(i))

    assert len(_adapter_utils._token_counts) <= 8
    assert _adapter_utils._token_counts_chars <= 10_000


# ------------------------------------------------------------------ reading an answer
#: A long answer holding nothing any scaffolding pattern needs: no label, tag,
#: brace, bracket, pipe, colon, dash, parenthesis, backtick or run of spaces.
_PLAIN = (
    "The committee met twice this year and approved the plan in full. Each of the "
    "four sites reported its figures on time, and the totals agree with the survey.\n"
)


def test_a_plain_answer_costs_no_pattern_scans(pattern_scans):
    text = _PLAIN * 120

    assert agent_runtime.sanitize_final_answer(text) == text.strip()
    assert agent_runtime.find_written_tool_call(text, ["calculator", "web_search"]) is None
    assert pattern_scans[0] == 0


def test_an_answer_with_a_label_is_scanned_only_for_the_label(pattern_scans):
    text = _PLAIN * 20 + "Final Answer: the plan was approved."

    assert agent_runtime.sanitize_final_answer(text) == "the plan was approved."
    assert pattern_scans[0] == 1


#: Answers holding the brackets and signs a written call is made of, but no call and none of
#: the words the patterns need, beside the most pattern scans reading each one may cost: only
#: the readers whose every character is present run.
_SIGNED = [
    ("The value on the left is smaller when x < y holds for every row.", 1),
    ("Each total is x > y = z (rounded) for every row in the survey.", 1),
    ("Each total is x < y > z = w (rounded) for every row in the survey.", 5),
]


@pytest.mark.parametrize(("text", "most"), _SIGNED)
def test_an_answer_with_brackets_and_signs_is_scanned_only_by_readers_it_could_match(text, most, pattern_scans):
    assert agent_runtime.sanitize_final_answer(text) == text
    assert agent_runtime.find_written_tool_call(text, ["calculator", "web_search"]) is None
    assert pattern_scans[0] <= most, pattern_scans[0]


def test_reading_twenty_kilobytes_of_plain_answer_takes_under_two_milliseconds():
    text = _PLAIN * 120
    agent_runtime.sanitize_final_answer(text)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        agent_runtime.sanitize_final_answer(text)
        agent_runtime.find_written_tool_call(text, ["calculator"])
        times.append(time.perf_counter() - start)

    assert statistics.median(times) < 0.002, statistics.median(times)
