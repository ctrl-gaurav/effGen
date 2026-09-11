"""Agent run history — an in-memory ring plus an append-only file store.

Every completed agent run is appended to a process-local ring buffer (the hot
read path for the dashboard) and, unless history is disabled, to a daily
JSONL file under the effGen state directory::

    ~/.effgen/runs/2026-07-18.jsonl

The files are the durable source of truth: runs recorded by the CLI, by a
script, and by the server all land in the same place and survive a restart.
:func:`get_recent_runs` reads the ring, so live views stay scoped to the
current process; :func:`read_runs` and :func:`get_run` read the durable
history, merging the ring with the files and de-duplicating on ``run_id``.

Locations honor, in order, ``EFFGEN_RUN_HISTORY_DIR``, ``EFFGEN_HOME`` (as
``$EFFGEN_HOME/runs``), then ``~/.effgen/runs``. Set ``EFFGEN_RUN_HISTORY=0``
to keep history in memory only. Files older than
``EFFGEN_RUN_HISTORY_MAX_DAYS`` (default 30) are removed when the first record
of a process is written.

Writes are best-effort: a history write that fails is logged at debug level and
never propagates into the caller's run.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_RUNS = 200

#: Longest task/output preview persisted with a run record.
PREVIEW_CHARS = 500

DEFAULT_MAX_DAYS = 30

_lock: threading.Lock = threading.Lock()
_runs: deque[dict[str, Any]] = deque(maxlen=MAX_RUNS)
_pruned = False


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
def history_dir() -> Path:
    """Return the directory holding daily run-history files."""
    explicit = os.environ.get("EFFGEN_RUN_HISTORY_DIR")
    if explicit:
        return Path(os.path.expanduser(explicit)).absolute()
    home = os.environ.get("EFFGEN_HOME")
    if home:
        return Path(os.path.expanduser(home)).absolute() / "runs"
    return Path(os.path.expanduser("~/.effgen/runs"))


def history_enabled() -> bool:
    """Whether run records are written to disk."""
    return os.environ.get("EFFGEN_RUN_HISTORY", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _now_iso() -> str:
    """Local time as ISO-8601 with an explicit UTC offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _preview(text: Any, limit: int = PREVIEW_CHARS) -> str | None:
    if text is None:
        return None
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def _thread_shape(thread: Any) -> dict[str, Any]:
    """The schema version and ordered step kinds of *thread*, or nothing.

    A record says which conversation a run had by its shape, so a listing can
    tell a two-step run from a twenty-step one and can tell which release wrote
    the steps, without this bounded store growing a transcript.
    """
    if thread is None:
        return {}
    data = thread
    to_dict = getattr(thread, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
    if not isinstance(data, dict):
        return {}
    steps = data.get("steps")
    if not isinstance(steps, list):
        return {}
    kinds = [str(s.get("kind")) for s in steps if isinstance(s, dict)]
    return {
        "thread_version": data.get("version"),
        "thread_steps": len(kinds),
        "thread_kinds": kinds,
    }



def record_run(
    *,
    model: str = "unknown",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    duration_s: float | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
    stop_reason: str | None = None,
    outcome: str | None = None,
    run_id: str | None = None,
    task: str | None = None,
    output: str | None = None,
    provider: str | None = None,
    session_id: str | None = None,
    agent: str | None = None,
    execution_id: str | None = None,
    execution_kind: str | None = None,
    execution_name: str | None = None,
    parent_agent: str | None = None,
    role: str | None = None,
    thread: Any = None,
) -> dict[str, Any]:
    """Append a run record to the ring buffer and the daily history file.

    Parameters
    ----------
    model:
        Model identifier used for the run.
    input_tokens:
        Number of input (prompt) tokens consumed, if known.
    output_tokens:
        Number of output (completion) tokens generated, if known.
    duration_s:
        Wall-clock duration of the run in seconds.
    cost_usd:
        Estimated cost in USD, if known.
    error:
        Message if the run failed; ``None`` on success. Stored truncated to
        ``PREVIEW_CHARS``, so a caller may pass the whole message.
    stop_reason:
        What ended the run, when the caller knows it (``"final_answer"``,
        ``"loop_detected"``, …). Stored as given.
    outcome:
        ``"answered"``, ``"stopped"`` or ``"failed"``. A stopped run is
        recorded with ``status="stopped"`` rather than ``"error"``: the run was
        carried out, it just never reached an answer.
    run_id:
        Identifier shared with the run's trace spans, so a record can be
        opened and joined to its trace.
    task:
        The task text; stored truncated to ``PREVIEW_CHARS``.
    output:
        The answer text; stored truncated to ``PREVIEW_CHARS``.
    provider:
        Provider that served the model, if known.
    session_id:
        Session the run belongs to, when it was run with one.
    agent:
        Name of the agent that performed the run.
    execution_id:
        Identifier shared by every run of one team or workflow execution.
        Defaults to the execution the caller is running inside, so a sub-agent
        run is grouped with its siblings without the caller passing anything.
    execution_kind:
        ``"team"`` or ``"workflow"``; defaults from the current execution.
    execution_name:
        Team or workflow name; defaults from the current execution.
    parent_agent:
        Agent that delegated this run; defaults from the current execution.
    role:
        Role the agent played in the execution (``"manager"``, ``"worker"``,
        ``"node"``, …); defaults from the current execution.
    thread:
        The run's conversation, as an :class:`~effgen.core.thread.AgentThread`
        or as the data one serialises to. The record keeps its schema version
        and the kinds of its steps in order — the shape of the conversation,
        not a truncated rendering of its text. This store is a bounded ring of
        previews; the conversation itself lives on the run's checkpoint, on the
        session turn and on ``response.metadata["thread"]``.

    Returns the record that was stored.
    """
    # Each field falls back to the execution in force, so a caller that names
    # only one of them keeps the rest of the correlation instead of dropping it.
    try:
        from .tracing import current_execution

        execution = current_execution() or {}
    except Exception:  # noqa: BLE001 - correlation is best-effort
        execution = {}
    record: dict[str, Any] = {
        "ts": _now_iso(),
        "run_id": run_id,
        "status": (
            "stopped" if outcome == "stopped" else ("error" if error else "ok")
        ),
        "stop_reason": stop_reason,
        "model": model,
        "provider": provider,
        "agent": agent,
        "session_id": session_id,
        "execution_id": execution_id or execution.get("execution_id"),
        "execution_kind": execution_kind or execution.get("execution_kind"),
        "execution_name": execution_name or execution.get("execution_name"),
        "parent_agent": parent_agent or execution.get("parent_agent"),
        "role": role or execution.get("role"),
        "task": _preview(task),
        "output": _preview(output),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": duration_s,
        "cost_usd": cost_usd,
        "error": _preview(error),
        **_thread_shape(thread),
    }
    with _lock:
        _runs.append(record)
    _append_to_file(record)
    return record


def _append_to_file(record: dict[str, Any]) -> None:
    """Append one record to today's history file; failures are non-fatal."""
    if not history_enabled():
        return
    try:
        directory = history_dir()
        directory.mkdir(parents=True, exist_ok=True)
        _prune_once(directory)
        day = record["ts"][:10]
        line = json.dumps(record, default=str, ensure_ascii=False)
        with open(directory / f"{day}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - history must never fail a user's run
        logger.debug("Run history write failed", exc_info=True)


def _prune_once(directory: Path) -> None:
    """Drop history files older than the retention window (once per process)."""
    global _pruned
    if _pruned:
        return
    _pruned = True
    try:
        max_days = int(os.environ.get("EFFGEN_RUN_HISTORY_MAX_DAYS", DEFAULT_MAX_DAYS))
    except ValueError:
        max_days = DEFAULT_MAX_DAYS
    if max_days <= 0:
        return
    cutoff = (datetime.now().date() - timedelta(days=max_days)).isoformat()
    try:
        for path in directory.glob("*.jsonl"):
            if path.stem < cutoff:
                path.unlink()
    except OSError:
        logger.debug("Run history prune failed", exc_info=True)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _read_files(*, limit: int) -> list[dict[str, Any]]:
    """Read up to *limit* records from the newest history files (newest first)."""
    directory = history_dir()
    if not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    try:
        files = sorted(directory.glob("*.jsonl"), key=lambda p: p.stem, reverse=True)
    except OSError:
        return []
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            logger.debug("Unreadable run history file %s", path)
            continue
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                # One damaged line never hides the rest of the history.
                logger.debug("Skipping unparseable run history line in %s", path)
        if len(out) >= limit:
            break
    return out


def _sort_key(record: dict[str, Any]) -> str:
    return str(record.get("ts") or "")


def _identity(record: dict[str, Any]) -> str:
    """A key that matches the same run whether it came from the ring or a file.

    ``run_id`` identifies a run on its own. Without one, the stored fields are
    compared instead, so a run present in both the ring and today's file is
    still reported once rather than twice.
    """
    run_id = record.get("run_id")
    if run_id:
        return f"id:{run_id}"
    return "fields:" + "|".join(
        str(record.get(field) or "")
        for field in ("ts", "model", "status", "session_id", "task", "error")
    )


def read_runs(
    *,
    limit: int = 50,
    status: str | None = None,
    model: str | None = None,
    search: str | None = None,
    since: str | None = None,
    until: str | None = None,
    session_id: str | None = None,
    execution_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return stored run records, newest first, after applying filters.

    Merges the in-memory ring with the history files and de-duplicates on
    ``run_id``. ``since``/``until`` accept a date (``2026-07-18``) or any
    ISO-8601 prefix and are compared against the record timestamp. ``search``
    matches the task, output, model, run id, session id, or error text.
    ``execution_id`` narrows to the runs of one team or workflow execution.

    Args:
        limit: Most records to return.
        status: Keep only runs that ended with this status.
        model: Keep only runs against this model id.
        search: Keep only runs whose text matches this term.
        since: Keep only runs at or after this date or ISO-8601 prefix.
        until: Keep only runs at or before this date or ISO-8601 prefix.
        session_id: Keep only runs from this session.
        execution_id: Keep only runs from one team or workflow execution.
    """
    with _lock:
        merged = list(_runs)
    merged.extend(_read_files(limit=max(limit * 4, 200)))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in sorted(merged, key=_sort_key, reverse=True):
        key = _identity(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    needle = (search or "").lower()
    out: list[dict[str, Any]] = []
    for record in unique:
        if status and str(record.get("status") or "") != status:
            continue
        if model and model.lower() not in str(record.get("model") or "").lower():
            continue
        if session_id and str(record.get("session_id") or "") != session_id:
            continue
        if execution_id and str(record.get("execution_id") or "") != execution_id:
            continue
        ts = str(record.get("ts") or "")
        if since and ts < since:
            continue
        if until and ts[: len(until)] > until:
            continue
        if needle:
            haystack = " ".join(
                str(record.get(f) or "")
                for f in (
                    "task", "output", "model", "run_id", "session_id", "error",
                    "agent", "execution_id", "execution_name",
                )
            ).lower()
            if needle not in haystack:
                continue
        out.append(record)
        if len(out) >= limit:
            break
    return out


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return the record with *run_id*, or ``None`` when it is not stored."""
    for record in read_runs(limit=10000, search=run_id):
        if str(record.get("run_id")) == run_id:
            return record
    return None


def read_executions(*, limit: int = 10, scan: int = 400) -> list[dict[str, Any]]:
    """Return recent multi-agent executions reconstructed from stored runs.

    Each entry carries the execution ``id``, ``kind`` (``team``/``workflow``),
    ``name``, first/last timestamps, totals, and the member runs in the order
    they were recorded. Because it reads the durable history rather than a
    process-local buffer, a team or workflow run from a script or the CLI is
    visible to a reader in another process.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for record in read_runs(limit=scan):
        execution_id = record.get("execution_id")
        if not execution_id:
            continue
        entry = grouped.setdefault(str(execution_id), {
            "id": str(execution_id),
            "kind": record.get("execution_kind"),
            "name": record.get("execution_name"),
            "started": record.get("ts"),
            "updated": record.get("ts"),
            "runs": [],
            "cost_usd": 0.0,
            "tokens": 0,
            "failed": 0,
        })
        entry["runs"].append(record)
        ts = str(record.get("ts") or "")
        if ts and ts < str(entry["started"] or ts):
            entry["started"] = ts
        if ts and ts > str(entry["updated"] or ""):
            entry["updated"] = ts
        try:
            entry["cost_usd"] += float(record.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            # A record with an unusable cost contributes nothing to the total
            # rather than failing the whole grouping.
            pass
        for field in ("input_tokens", "output_tokens"):
            try:
                entry["tokens"] += int(record.get(field) or 0)
            except (TypeError, ValueError):
                # As above: an unusable token count is left out of the sum.
                pass
        # A stopped run reached no answer either, so it counts here — the
        # status word distinguishes the two, the tally does not need to.
        if record.get("status") in ("error", "stopped"):
            entry["failed"] += 1
        if entry["kind"] is None:
            entry["kind"] = record.get("execution_kind")
        if entry["name"] is None:
            entry["name"] = record.get("execution_name")

    executions = sorted(grouped.values(), key=lambda e: str(e["updated"] or ""), reverse=True)
    for entry in executions:
        entry["runs"].reverse()  # oldest first, the order they ran in
        entry["cost_usd"] = round(entry["cost_usd"], 6)
    return executions[:limit]


def get_recent_runs(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most-recent *limit* runs of this process (newest first).

    Reads the in-memory ring only, so live views stay scoped to current
    activity. Use :func:`read_runs` for the durable history, which spans
    processes and restarts.

    Returns an empty list when this process has recorded no runs.
    """
    with _lock:
        runs = list(_runs)
    return list(reversed(runs))[:limit]


def cleanup_runs(older_than_days: int = 30) -> int:
    """Delete history files older than *older_than_days*; returns the count."""
    directory = history_dir()
    if not directory.is_dir():
        return 0
    cutoff = (datetime.now().date() - timedelta(days=older_than_days)).isoformat()
    removed = 0
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.debug("Could not remove run history file %s", path)
    return removed


def clear() -> None:
    """Clear the in-memory ring (the history files are left untouched)."""
    global _pruned
    with _lock:
        _runs.clear()
    _pruned = False
