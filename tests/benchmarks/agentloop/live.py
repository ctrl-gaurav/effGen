"""Running one set against one endpoint, and writing what it cost.

The records this writes are the same shape as the ones ``records.py`` reads, so
a fresh run and an already-recorded one go through the same reader, the same
tables and the same counterfactual with no special case. Three fields are added
per sample — ``model_wall_s``, ``framework_wall_s``, ``framework_cpu_s`` — and
they are additive, so a reader still loads a record that has none of them.

Two rules the runner enforces on itself rather than trusting the caller:

* the endpoint is an argument, never an environment variable. If any of the
  three variables that redirect OpenAI-protocol traffic is set — set to
  *anything*, including the empty string — the run refuses to start and names
  it. A stale one silently redirects every call in the process and the failures
  read as an outage at the provider;
* the run records where it went. Base URL, the model the server lists back, the
  card, the settings and the revision of this tree go into a manifest beside the
  records. A run that cannot say where it went is not evidence.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import INSTRUMENT_VERSION, timing
from .harness.benchmarks import find_spec, load_benchmark
from .harness.tools.builtin import TOOLS
from .harness.types import Sample

#: The variables that redirect OpenAI-protocol traffic. Any of them being set is
#: refused, including to the empty string, which is the shape that does the
#: damage without looking like it is set at all.
ENDPOINT_ENV_VARS = ("EFFGEN_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")

#: The generation settings every recorded run used. Changing one makes a fresh
#: run incomparable to the records, so they are defaults rather than opinions.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 1024
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_CONTEXT_LENGTH = 16384

logger = logging.getLogger("agentloop.live")


class EndpointEnvironmentSet(RuntimeError):
    """One of the variables that redirect model traffic is set."""


def check_endpoint_env(environ: dict[str, str] | None = None) -> None:
    """Refuse to run while a redirecting variable is set."""
    env = os.environ if environ is None else environ
    present = [name for name in ENDPOINT_ENV_VARS if name in env]
    if present:
        shown = ", ".join(f"{name}={env[name]!r}" for name in present)
        raise EndpointEnvironmentSet(
            f"{shown} is set, which redirects every OpenAI-protocol call in this "
            f"process. Run: unset {' '.join(present)}"
        )


# ------------------------------------------------------------------ the plan


@dataclass
class LiveRun:
    """Everything one cell needs to run, with no defaults that reach the network."""

    bench: str
    model: str
    base_url: str
    out_dir: Path
    api_key: str = "EMPTY"
    n: int | None = None
    offset: int = 0
    concurrency: int = 8
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int | None = DEFAULT_SEED
    context_length: int = DEFAULT_CONTEXT_LENGTH
    timeout_s: float = 300.0
    max_retries: int = DEFAULT_MAX_RETRIES
    capture_logs: bool = False
    capture_prompts: bool = False
    tool_cache_dir: Path | None = None
    samples_path: Path | None = None
    resume: bool = True
    label: str = ""
    #: Set when the agent is built by something other than the library under
    #: measurement. Used by the instrument's own tests to plant a known cost.
    agent_factory: Callable[..., Any] | None = None


# ---------------------------------------------------------------- the samples


def load_samples(bench: str, path: Path, n: int | None, offset: int) -> list[Sample]:
    """Read a vendored sample file, in the order the recorded runs used."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                Sample(
                    sample_id=row["sample_id"],
                    question=row["question"],
                    answer=row.get("answer"),
                    context=row.get("context") or "",
                    meta=dict(row.get("meta") or {}),
                )
            )
    if offset:
        rows = rows[offset:]
    if n is not None and n > 0:
        rows = rows[:n]
    if not rows:
        raise ValueError(f"{path}: no samples left after offset={offset} n={n}")
    return rows


def default_samples_path(bench: str) -> Path:
    return Path(__file__).parent / "fixtures" / "samples" / f"{bench}.jsonl"


# ------------------------------------------------------------------ the tools


class ToolCache:
    """Replayed answers for a tool that reaches a third party.

    Two arms of a comparison that get different search results are not a
    comparison. The cache is keyed on the exact query text, so the second arm
    sees what the first arm saw. Deterministic tools are not cached.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _path(self, name: str, query: str) -> Path:
        import hashlib

        digest = hashlib.sha256(f"{name}\0{query}".encode()).hexdigest()[:32]
        return self.dir / f"{name}-{digest}.json"

    def wrap(self, spec):
        import dataclasses as _dc

        inner = spec.func
        arg = spec.arg_names[0] if spec.arg_names else "query"

        def cached(**kwargs: Any) -> str:
            query = str(kwargs.get(arg, ""))
            path = self._path(spec.name, query)
            if path.exists():
                with self._lock:
                    self.hits += 1
                return json.loads(path.read_text())["result"]
            result = inner(**kwargs)
            path.write_text(json.dumps({"query": query, "result": result}))
            with self._lock:
                self.misses += 1
            return result

        return _dc.replace(spec, func=cached)


#: Tools whose answer depends on a third party and is therefore cached when a
#: cache is given. The others compute the same answer every time.
CACHEABLE_TOOLS = ("web_search",)


def build_tool_specs(bench: str, cache: ToolCache | None):
    specs = [TOOLS[name] for name in find_spec(bench).tools if name in TOOLS]
    if cache is None:
        return specs
    return [cache.wrap(s) if s.name in CACHEABLE_TOOLS else s for s in specs]


# ------------------------------------------------------------- log capture


class _PerSampleLog(logging.Handler):
    """Collect log lines per worker thread, so a line belongs to a sample.

    A shared log file cannot answer "how many of the samples this path fired on
    became correct", because the lines from four workers are interleaved and
    carry no sample id. Scoping the handler to the thread does answer it.
    """

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        lines = getattr(self._local, "lines", None)
        if lines is None:
            return
        try:
            lines.append(self.format(record))
        except Exception:
            pass

    def start(self) -> None:
        self._local.lines = []

    def finish(self) -> str:
        lines = getattr(self._local, "lines", None) or []
        self._local.lines = None
        return "\n".join(lines)


# ------------------------------------------------------------------ the run


@dataclass
class RunOutcome:
    directory: Path
    summary: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return float(self.summary["accuracy"])


def probe_endpoint(base_url: str, model: str, timeout_s: float = 10.0) -> list[str]:
    """Ask the endpoint what it serves, and check the wanted model is on the list."""
    import httpx

    url = base_url.rstrip("/") + "/models"
    with httpx.Client(timeout=timeout_s) as client:
        response = client.get(url)
    response.raise_for_status()
    served = [entry.get("id") for entry in (response.json().get("data") or [])]
    if model not in served:
        raise RuntimeError(
            f"{url} serves {served}, which does not include {model!r}. "
            "The run would have measured a different model from the one named."
        )
    return served


def _gpu_line() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip()
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def _head_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_cell(plan: LiveRun) -> RunOutcome:
    """Run one set against one endpoint and write records, summary and manifest."""
    check_endpoint_env()
    if not plan.base_url:
        raise ValueError(
            "no base_url. The endpoint is always an argument here; there is no "
            "default and no environment fallback."
        )

    benchmark = load_benchmark(plan.bench)
    samples_path = plan.samples_path or default_samples_path(plan.bench)
    samples = load_samples(plan.bench, samples_path, plan.n, plan.offset)

    out_dir = Path(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    done_ids: set[str] = set()
    existing: list[dict[str, Any]] = []
    if plan.resume and records_path.exists():
        with open(records_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing.append(row)
                done_ids.add(row["sample_id"])
        samples = [s for s in samples if s.sample_id not in done_ids]
    elif records_path.exists():
        records_path.unlink()

    served = probe_endpoint(plan.base_url, plan.model, timeout_s=min(plan.timeout_s, 30))

    cache = ToolCache(plan.tool_cache_dir) if plan.tool_cache_dir else None
    tool_specs = build_tool_specs(plan.bench, cache)
    max_tokens = plan.max_tokens
    if benchmark.min_max_tokens:
        # A set whose right answer is simply long is not measuring reasoning if
        # the budget cuts every answer off. The floor is applied once, here, so
        # every run of this set shares it.
        max_tokens = max(max_tokens, benchmark.min_max_tokens)

    timing.install()
    handler = _PerSampleLog() if plan.capture_logs else None
    if handler is not None:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)
        for noisy in ("httpx", "httpcore", "openai", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    started_at = time.time()
    build = _AgentBuilder(plan, tool_specs, benchmark, max_tokens)
    written: list[dict[str, Any]] = list(existing)
    write_lock = threading.Lock()

    def one(sample: Sample) -> dict[str, Any]:
        if handler is not None:
            handler.start()
        attempt: dict[str, Any] = {
            "output": "",
            "tool_calls": [],
            "tool_call_count": None,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": 0.0,
            "error": None,
            "stop_reason": None,
            "success": None,
            "outcome": None,
            "iterations": None,
        }
        # The agent is built inside the timed block on purpose: building it is
        # part of what the framework costs per task, and it is rebuilt per sample
        # so a sample cannot see the ones before it.
        with timing.measure() as timed:
            try:
                agent = build.agent()
                response = agent.run(benchmark.user_prompt(sample, bool(tool_specs)))
                from .agent_binding import (
                    read_iterations,
                    read_outcome,
                    read_response,
                    read_stop_reason,
                )

                text, calls, count = read_response(response)
                attempt["output"] = text or ""
                attempt["tool_calls"] = calls
                attempt["tool_call_count"] = count
                attempt["stop_reason"] = read_stop_reason(response)
                success = getattr(response, "success", None)
                attempt["success"] = None if success is None else bool(success)
                attempt["outcome"] = read_outcome(response)
                attempt["iterations"] = read_iterations(response)
                if plan.capture_prompts:
                    attempt["messages"] = _captured_prompts(response)
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("sample %s failed: %s", sample.sample_id, attempt["error"])
                logger.debug(traceback.format_exc())

        attempt["latency_s"] = round(timed.wall_s, 3)
        attempt["llm_calls"] = timed.usage.calls
        attempt["prompt_tokens"] = timed.usage.prompt_tokens
        attempt["completion_tokens"] = timed.usage.completion_tokens
        attempt["total_tokens"] = timed.usage.total_tokens
        attempt["tools_used"] = (
            attempt["tool_call_count"]
            if attempt["tool_call_count"] is not None
            else len(attempt["tool_calls"])
        )
        attempt["model_wall_s"] = round(timed.model_wall_s, 4)
        attempt["framework_wall_s"] = round(timed.framework_wall_s, 4)
        attempt["framework_cpu_s"] = round(timed.framework_cpu_s, 4)
        attempt["streaming_seen"] = timed.usage.streaming_seen
        attempt["uncounted_calls"] = timed.usage.uncounted
        if handler is not None:
            attempt["log"] = handler.finish()

        try:
            score, prediction = benchmark.score(sample, attempt["output"])
        except Exception as exc:
            logger.warning("scoring %s failed: %s", sample.sample_id, exc)
            score, prediction = 0.0, None
        score = float(score)
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "ground_truth": sample.answer,
            "prediction": prediction,
            "score": score,
            "correct": score >= 0.5,
            "output": attempt["output"],
            "meta": sample.meta,
            "attempt": attempt,
        }

    def record(row: dict[str, Any]) -> None:
        with write_lock:
            with open(records_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            written.append(row)
            done = len(written)
            if done % 10 == 0 or done == len(samples) + len(existing):
                logger.info(
                    "%d/%d | running accuracy %.2f%%",
                    done,
                    len(samples) + len(existing),
                    sum(r["score"] for r in written) / done * 100,
                )

    state, error = "done", None
    try:
        if plan.concurrency <= 1:
            for sample in samples:
                record(one(sample))
        else:
            with ThreadPoolExecutor(max_workers=plan.concurrency) as pool:
                for row in pool.map(one, samples):
                    record(row)
    except KeyboardInterrupt:
        state, error = "failed", "interrupted"
    except Exception as exc:  # a failure of the run, not of a sample
        state, error = "failed", f"{type(exc).__name__}: {exc}"
        logger.error("run failed: %s", error)
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
        build.teardown()

    manifest = {
        "instrument_version": INSTRUMENT_VERSION,
        "run_id": out_dir.name,
        "label": plan.label,
        "started_at": started_at,
        "finished_at": time.time(),
        "base_url": plan.base_url,
        "model": plan.model,
        "served_models": served,
        "benchmark": plan.bench,
        "samples_file": str(samples_path),
        "requested_n": plan.n,
        "offset": plan.offset,
        "concurrency": plan.concurrency,
        "max_steps": plan.max_steps,
        "endpoint": {
            "temperature": plan.temperature,
            "top_p": plan.top_p,
            "max_tokens": max_tokens,
            "seed": plan.seed,
            "context_length": plan.context_length,
            "max_retries": plan.max_retries,
            "timeout_s": plan.timeout_s,
        },
        "tool_cache": (
            None
            if cache is None
            else {"dir": str(cache.dir), "hits": cache.hits, "misses": cache.misses}
        ),
        "capture_logs": plan.capture_logs,
        "capture_prompts": plan.capture_prompts,
        "gpu": _gpu_line(),
        "revision": _head_revision(),
        "python": platform.python_version(),
        "host": platform.node(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    summary = _summarise(written, manifest, state, error, started_at)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return RunOutcome(directory=out_dir, summary=summary, records=written)


def _captured_prompts(response) -> list[dict[str, Any]]:
    metadata = getattr(response, "metadata", None) or {}
    for key in ("messages", "prompt", "prompts"):
        value = metadata.get(key)
        if value:
            return value if isinstance(value, list) else [{"role": "user", "content": value}]
    trace = getattr(response, "execution_trace", None) or []
    return [event for event in trace if isinstance(event, dict)]


class _AgentBuilder:
    """One model per process, one fresh agent per sample."""

    def __init__(self, plan: LiveRun, tool_specs, benchmark, max_tokens: int) -> None:
        self.plan = plan
        self.benchmark = benchmark
        self.max_tokens = max_tokens
        self._model = None
        self._tools = None
        self._lock = threading.Lock()
        self._tool_specs = tool_specs

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from effgen.models import load_model

                    from .agent_binding import build_tools

                    self._model = load_model(
                        self.plan.model,
                        provider="openai_compatible",
                        base_url=self.plan.base_url,
                        api_key=self.plan.api_key,
                        # Without this the adapter plans against a window twice
                        # the one the server was started with.
                        context_length=self.plan.context_length,
                        timeout=int(self.plan.timeout_s),
                        max_retries=self.plan.max_retries,
                    )
                    self._tools = build_tools(list(self._tool_specs))
        return self._model, self._tools

    def agent(self):
        if self.plan.agent_factory is not None:
            return self.plan.agent_factory(self)
        model, tools = self._ensure()
        from effgen.core.agent import Agent, AgentConfig

        from .agent_binding import agent_config_kwargs

        config = AgentConfig(
            name="measured_agent",
            model=model,
            tools=tools,
            system_prompt=self.benchmark.system_prompt(bool(tools)),
            max_iterations=self.plan.max_steps,
            **agent_config_kwargs(
                AgentConfig,
                temperature=self.plan.temperature,
                top_p=self.plan.top_p,
                max_tokens=self.max_tokens,
                seed=self.plan.seed,
            ),
        )
        return Agent(config=config)

    def teardown(self) -> None:
        if self._model is not None:
            try:
                self._model.unload()
            except Exception:
                pass


def _summarise(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    state: str,
    error: str | None,
    started_at: float,
) -> dict[str, Any]:
    n = max(len(rows), 1)
    attempts = [row.get("attempt") or {} for row in rows]

    def mean(key: str) -> float:
        return sum(float(a.get(key) or 0.0) for a in attempts) / n

    def mean_optional(key: str) -> float | None:
        values = [float(a[key]) for a in attempts if a.get(key) is not None]
        return sum(values) / len(values) if values else None

    reasons: Counter[str] = Counter()
    for attempt in attempts:
        reason = attempt.get("stop_reason")
        reasons[str(reason) if reason else "None"] += 1

    return {
        **manifest,
        "state": state,
        "error": error,
        "completed": len(rows),
        "total": len(rows),
        "wall_time_s": round(time.time() - started_at, 2),
        "accuracy": round(sum(float(r.get("score") or 0.0) for r in rows) / n * 100, 4),
        "exact_correct": sum(1 for r in rows if r.get("correct")),
        "errors": sum(1 for a in attempts if a.get("error")),
        "empty_outputs": sum(1 for r in rows if not str(r.get("output") or "").strip()),
        "avg_latency_s": round(mean("latency_s"), 3),
        "avg_llm_calls": round(mean("llm_calls"), 3),
        "avg_tool_calls": round(mean("tools_used"), 3),
        "avg_prompt_tokens": round(mean("prompt_tokens"), 1),
        "avg_completion_tokens": round(mean("completion_tokens"), 1),
        "avg_total_tokens": round(mean("total_tokens"), 1),
        "avg_model_wall_s": _round_optional(mean_optional("model_wall_s")),
        "avg_framework_wall_s": _round_optional(mean_optional("framework_wall_s")),
        "avg_framework_cpu_s": _round_optional(mean_optional("framework_cpu_s")),
        "streaming_seen": any(a.get("streaming_seen") for a in attempts),
        "stop_reasons": dict(reasons.most_common()),
    }


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def new_run_id(bench: str, label: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{label}" if label else ""
    return f"{bench}{tag}-{stamp}-{uuid.uuid4().hex[:6]}"


__all__ = [
    "CACHEABLE_TOOLS",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SEED",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "ENDPOINT_ENV_VARS",
    "EndpointEnvironmentSet",
    "LiveRun",
    "RunOutcome",
    "ToolCache",
    "build_tool_specs",
    "check_endpoint_env",
    "default_samples_path",
    "load_samples",
    "new_run_id",
    "probe_endpoint",
    "run_cell",
]
