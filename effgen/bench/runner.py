"""Running a bench suite and building its table.

:func:`run_suite` gives every task a fresh agent (sharing one loaded model),
runs the tasks with bounded concurrency, scores each answer, and reads what
each run spent off the run's ledger. The result is a plain run document — the
same one ``effgen bench run --json`` prints and ``effgen bench compare`` reads.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scoring import make_scorer
from .suite import AGENT_SETTINGS, BenchSuite, BenchTask

__all__ = [
    "COUNTERS",
    "RUN_SCHEMA",
    "TABLE_FIELDS",
    "BenchConfig",
    "BenchRunError",
    "build_table",
    "read_run",
    "run_suite",
    "write_run",
]

logger = logging.getLogger(__name__)

RUN_SCHEMA = "effgen.bench.run/1"

#: Per-task counters read from each run's ledger, in table order.
COUNTERS = (
    "llm_calls",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "cached_input_tokens",
    "wall_s",
    "framework_s",
    "model_wait_s",
    "tool_wait_s",
)

#: Every key of a run's ``table``, in the order the text table prints them.
TABLE_FIELDS = (
    "accuracy",
    "tasks",
    "errors",
    *COUNTERS,
    "cost_usd",
    "unpriced_calls",
    "elapsed_s",
    "stop_reasons",
)

#: Stop reasons that mean the run could not be carried out at all: the model
#: was never reached, or the run broke. They are counted as errors.
_ERROR_REASONS = frozenset({"generation_failed", "run_failed"})


class BenchRunError(RuntimeError):
    """A bench run that could not start: no model, an unknown tool, a load failure."""


@dataclass
class BenchConfig:
    """How to run a suite: the model, the endpoint, and the settings the caller overrides.

    Attributes:
        model: The model id (``provider:model``, a served model's id, or a local one).
        base_url: An OpenAI-protocol endpoint; the model is reached there.
        api_key: The endpoint's credential, if it needs one.
        engine: A local engine (``transformers``, ``vllm``) for a local model.
        concurrency: Tasks in flight at once.
        label: A name for the run, shown by ``compare``.
        settings: Agent settings that override the suite's
            (keys from :data:`~effgen.bench.suite.AGENT_SETTINGS`).
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    engine: str | None = None
    concurrency: int = 4
    label: str = ""
    settings: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        """The configuration as recorded in a run document (no credential)."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "engine": self.engine,
            "concurrency": self.concurrency,
            "label": self.label,
            "api_key_set": bool(self.api_key),
        }


def _load_model(config: BenchConfig) -> Any:
    from effgen.models import load_model

    kwargs: dict[str, Any] = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.engine:
        kwargs["engine"] = config.engine
    try:
        return load_model(config.model, **kwargs)
    except Exception as exc:  # noqa: BLE001 - reported as the run's own error
        raise BenchRunError(f"Could not load model {config.model!r} ({exc}). Check the model id and --base-url.") from exc


def _resolve_tools(names: list[str]) -> list[Any]:
    if not names:
        return []
    from effgen.tools.registry import get_registry

    registry = get_registry()
    registry.discover_builtin_tools()
    tools = []
    for name in names:
        try:
            tools.append(registry.get_tool_sync(name))
        except KeyError as exc:
            known = ", ".join(sorted(registry.list_tools()))
            raise BenchRunError(f"unknown tool {name!r}. Use one of the available tools: {known}.") from exc
    return tools


def _ledger_totals(response: Any) -> dict[str, Any]:
    ledger = (getattr(response, "metadata", None) or {}).get("ledger") or {}
    return ledger.get("total") or ledger


def _run_one(
    task: BenchTask,
    *,
    model: Any,
    tools: list[Any],
    settings: dict[str, Any],
    seed: int | None,
    scorer: Callable[[str, Any, dict[str, Any]], float],
) -> dict[str, Any]:
    from effgen.core.agent import Agent, AgentConfig

    record: dict[str, Any] = {"id": task.id, "expected": task.expected}
    started = time.perf_counter()
    response = None
    try:
        config = AgentConfig(
            name="bench",
            model=model,
            tools=list(tools),
            raise_on_error=False,
            enable_memory=False,
            enable_sub_agents=False,
            seed=seed,
            **settings,
        )
        with Agent(config) as agent:
            response = agent.run(task.prompt)
    except Exception as exc:  # noqa: BLE001 - a task that raised is a counted error
        record["error"] = f"{type(exc).__name__}: {exc}"
    outer = time.perf_counter() - started

    metadata = (getattr(response, "metadata", None) or {}) if response is not None else {}
    success = bool(getattr(response, "success", False))
    answer = ""
    if response is not None:
        answer = str(response.output or "") if success else str(metadata.get("partial_output") or "")
    stop_reason = getattr(response, "stop_reason", None) if response is not None else "run_failed"
    if "error" not in record and stop_reason in _ERROR_REASONS:
        record["error"] = str(metadata.get("error") or stop_reason)
    record.setdefault("error", None)

    try:
        score = scorer(answer, task.expected, task.to_dict())
    except Exception as exc:  # noqa: BLE001 - a scorer that raised scores 0
        score = 0.0
        record["scorer_error"] = f"{type(exc).__name__}: {exc}"

    totals = _ledger_totals(response) if response is not None else {}
    record.update({
        "score": float(score),
        "answered": success,
        "outcome": getattr(response, "outcome", None) if response is not None else "failed",
        "stop_reason": stop_reason,
        "answer": answer,
    })
    for key in COUNTERS:
        value = totals.get(key)
        record[key] = value if value is not None else (outer if key == "wall_s" else 0)
    record["cost_usd"] = totals.get("cost_usd")
    record["unpriced_calls"] = int(totals.get("unpriced_calls") or 0)
    logger.info(
        "bench: task finished id=%s score=%.3g stop_reason=%s llm_calls=%s tool_calls=%s",
        task.id, record["score"], stop_reason, record["llm_calls"], record["tool_calls"],
    )
    return record


def build_table(records: list[dict[str, Any]], elapsed_s: float | None = None) -> dict[str, Any]:
    """The run's table from its task records: every key in :data:`TABLE_FIELDS`.

    Counters carry ``{"mean": per-task mean, "total": sum}``. ``cost_usd`` is
    ``None`` when no task's calls were priced; ``accuracy`` is the mean score
    in percent.
    """
    n = len(records)
    table: dict[str, Any] = {
        "accuracy": (100.0 * sum(r["score"] for r in records) / n) if n else None,
        "tasks": n,
        "errors": sum(1 for r in records if r.get("error")),
    }
    for key in COUNTERS:
        total = sum(float(r.get(key) or 0) for r in records)
        if key.endswith("_s"):
            table[key] = {"mean": round(total / n, 6) if n else None, "total": round(total, 6)}
        else:
            table[key] = {"mean": round(total / n, 4) if n else None, "total": int(total)}
    priced = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    table["cost_usd"] = (
        {"mean": sum(priced) / n, "total": sum(priced)} if priced and n else None
    )
    table["unpriced_calls"] = sum(int(r.get("unpriced_calls") or 0) for r in records)
    table["elapsed_s"] = round(elapsed_s, 3) if elapsed_s is not None else None
    table["stop_reasons"] = dict(Counter(str(r.get("stop_reason")) for r in records).most_common())
    return table


def run_suite(
    suite: BenchSuite,
    config: BenchConfig,
    *,
    records_path: str | Path | None = None,
    on_task: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run every task of *suite* and return the run document.

    Args:
        suite: The loaded suite.
        config: The model, endpoint and overrides.
        records_path: When given, each task record is appended to this JSON
            Lines file as it finishes, so an interrupted run leaves what it did.
        on_task: ``on_task(done, total, record)`` after each task finishes.

    Returns:
        ``{"schema", "state", "label", "suite", "config", "effgen_version",
        "started_at", "finished_at", "table", "tasks"}``. ``state`` is
        ``"complete"``; an interrupted run raises ``KeyboardInterrupt`` after
        writing what it finished to *records_path*.

    Raises:
        BenchRunError: The model could not be loaded or a tool is unknown.
    """
    from effgen import __version__

    if not config.model:
        raise BenchRunError("No model was given. Pass --model, or set 'model' in the suite.")
    scorer = make_scorer(suite.scorer, base_dir=suite.path.parent if suite.path else None)
    tools = _resolve_tools(suite.tools)
    settings = {k: v for k, v in suite.agent.items() if k in AGENT_SETTINGS}
    settings.update({k: v for k, v in (config.settings or {}).items()
                     if k in AGENT_SETTINGS and v is not None})
    model = _load_model(config)

    sink = None
    if records_path is not None:
        Path(records_path).parent.mkdir(parents=True, exist_ok=True)
        sink = open(records_path, "w", encoding="utf-8")  # noqa: SIM115 - closed below
    lock = threading.Lock()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    clock = time.perf_counter()
    records: dict[int, dict[str, Any]] = {}
    total = len(suite.tasks)
    logger.info("bench: run started suite=%s tasks=%d model=%s", suite.name, total, config.model)

    def _finish(index: int, record: dict[str, Any]) -> None:
        with lock:
            records[index] = record
            if sink is not None:
                sink.write(json.dumps(record, default=str) + "\n")
                sink.flush()
            done = len(records)
        if on_task is not None:
            try:
                on_task(done, total, record)
            except Exception:  # noqa: BLE001 - a progress callback never loses a record
                logger.debug("bench: on_task callback failed", exc_info=True)

    work: queue.SimpleQueue[tuple[int, BenchTask]] = queue.SimpleQueue()
    for index, task in enumerate(suite.tasks):
        work.put((index, task))
    stop = threading.Event()

    def _worker() -> None:
        while not stop.is_set():
            try:
                index, task = work.get_nowait()
            except queue.Empty:
                return
            _finish(index, _run_one(task, model=model, tools=tools, settings=settings,
                                    seed=suite.seed, scorer=scorer))

    # Daemon workers: an interrupted run returns to its caller at once instead
    # of waiting at interpreter exit for requests that are still in flight.
    workers = [
        threading.Thread(target=_worker, name=f"effgen-bench-{i}", daemon=True)
        for i in range(max(1, min(int(config.concurrency), total)))
    ]
    try:
        for thread in workers:
            thread.start()
        for thread in workers:
            while thread.is_alive():
                thread.join(timeout=0.2)
    except KeyboardInterrupt:
        stop.set()
        logger.info("bench: run interrupted after %d of %d tasks", len(records), total)
        raise
    finally:
        stop.set()
        if sink is not None:
            with lock:
                sink.close()
                sink = None
        if not any(thread.is_alive() for thread in workers):
            unload = getattr(model, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001 - freeing memory is best effort
                    logger.debug("bench: model unload failed", exc_info=True)

    missing = [i for i in range(total) if i not in records]
    if missing:
        raise BenchRunError(f"{len(missing)} task(s) produced no record. Re-run the suite.")
    elapsed = time.perf_counter() - clock
    ordered = [records[i] for i in range(total)]
    table = build_table(ordered, elapsed)
    logger.info("bench: run complete suite=%s tasks=%d errors=%d accuracy=%.2f",
                suite.name, total, table["errors"], table["accuracy"] or 0.0)
    return {
        "schema": RUN_SCHEMA,
        "state": "complete",
        "label": config.label or f"{suite.name}@{config.model}",
        "suite": suite.describe(),
        "config": {**config.describe(), "settings": settings, "seed": suite.seed},
        "effgen_version": __version__,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "table": table,
        "tasks": ordered,
    }


def write_run(document: dict[str, Any], out_dir: str | Path) -> Path:
    """Write *document* to ``<out_dir>/run.json`` and return the path."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "run.json"
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return path


def read_run(path: str | Path) -> dict[str, Any]:
    """Read a run document from a ``run.json`` file or the directory holding one.

    Raises:
        BenchRunError: The file is missing, is not a bench run document, or
            the run did not complete.
    """
    source = Path(path)
    if source.is_dir():
        source = source / "run.json"
    if not source.is_file():
        raise BenchRunError(f"There is no bench run at {path}. Pass a run.json file or the directory holding one.")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchRunError(f"{source} is not valid JSON ({exc}). Re-run the suite to write it again.") from exc
    if not isinstance(document, dict) or document.get("schema") != RUN_SCHEMA:
        raise BenchRunError(f"{source}: not an effgen bench run document ({RUN_SCHEMA}). Pass a run.json that effgen bench run wrote.")
    if document.get("state") != "complete":
        raise BenchRunError(f"{source}: the run did not complete (state {document.get('state')!r}). Re-run the suite before comparing it.")
    return document
