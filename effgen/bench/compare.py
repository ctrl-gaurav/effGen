"""Comparing two bench runs, with the noise band beside every delta.

Two runs of the same suite are paired task by task. For each measured field
the delta is the mean of the per-task differences ``b - a``, and the band is
**two standard errors of those differences**::

    d_i  = value_b(task i) - value_a(task i)        over the k shared tasks
    band = 2 * stdev(d) / sqrt(k)                    (stdev with k - 1)

The band is computed from the two runs' own disagreement, never from a fixed
constant: two runs that agree on every task have a band of zero, and two runs
whose answers flip on many tasks have a wide one. A delta whose size is within
its band is not a measured difference, and the output says so. Stop reasons
are compared as counts, with the band scaled to a count (``2 * stdev(d) *
sqrt(k)``, ``d_i`` being the change in whether task *i* ended that way).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .runner import COUNTERS

__all__ = ["COMPARE_SCHEMA", "METRICS", "compare_runs", "paired_band"]

COMPARE_SCHEMA = "effgen.bench.compare/1"

#: The per-task fields compared, with how each is read from a task record.
METRICS: tuple[tuple[str, Callable[[dict[str, Any]], float | None]], ...] = (
    ("accuracy", lambda r: 100.0 * float(r.get("score") or 0.0)),
    ("errors", lambda r: 1.0 if r.get("error") else 0.0),
    *((key, (lambda k: lambda r: float(r.get(k) or 0.0))(key)) for key in COUNTERS),
    ("cost_usd", lambda r: None if r.get("cost_usd") is None else float(r["cost_usd"])),
)


def paired_band(diffs: list[float]) -> float | None:
    """Two standard errors of the mean of *diffs*; ``None`` below two tasks."""
    k = len(diffs)
    if k < 2:
        return None
    mean = sum(diffs) / k
    variance = sum((d - mean) ** 2 for d in diffs) / (k - 1)
    return 2.0 * math.sqrt(variance) / math.sqrt(k)


def _row(field: str, a: list[float], b: list[float], *, scale: float = 1.0) -> dict[str, Any]:
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    k = len(diffs)
    band = paired_band(diffs)
    delta = (sum(diffs) / k) * scale if k else None
    if band is not None:
        band *= scale
    return {
        "field": field,
        "a": (sum(a) / k) * scale if k else None,
        "b": (sum(b) / k) * scale if k else None,
        "delta": delta,
        "band": band,
        "within_band": None if band is None or delta is None else abs(delta) <= band + 1e-12,
    }


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare run documents *a* and *b* (as :func:`~effgen.bench.runner.read_run` returns).

    Returns:
        ``{"schema", "a", "b", "paired", "only_a", "only_b", "same_tasks",
        "method", "rows", "stop_reasons"}``. Each row carries ``a``, ``b``
        (means over the paired tasks), ``delta`` (``b - a``), ``band`` and
        ``within_band``; ``band`` is ``None`` when fewer than two tasks pair.

    Raises:
        ValueError: The runs share no task.
    """
    by_id_a = {r["id"]: r for r in a.get("tasks") or []}
    by_id_b = {r["id"]: r for r in b.get("tasks") or []}
    shared = [i for i in by_id_a if i in by_id_b]
    if not shared:
        raise ValueError("The two runs share no task id. Use two runs of the same suite.")
    ra = [by_id_a[i] for i in shared]
    rb = [by_id_b[i] for i in shared]

    rows: list[dict[str, Any]] = []
    for field, read in METRICS:
        read_a = [read(r) for r in ra]
        read_b = [read(r) for r in rb]
        va = [v for v in read_a if v is not None]
        vb = [v for v in read_b if v is not None]
        if len(va) < len(read_a) or len(vb) < len(read_b):
            rows.append({"field": field, "a": None, "b": None, "delta": None, "band": None,
                         "within_band": None, "note": "not every paired task was priced"})
            continue
        rows.append(_row(field, va, vb))

    k = len(shared)
    reasons = sorted({str(r.get("stop_reason")) for r in ra + rb})
    stops = []
    for reason in reasons:
        in_a = [1.0 if str(r.get("stop_reason")) == reason else 0.0 for r in ra]
        in_b = [1.0 if str(r.get("stop_reason")) == reason else 0.0 for r in rb]
        row = _row(reason, in_a, in_b, scale=float(k))
        row["field"] = reason
        stops.append(row)

    def _about(doc: dict[str, Any]) -> dict[str, Any]:
        suite = doc.get("suite") or {}
        config = doc.get("config") or {}
        return {"label": doc.get("label"), "model": config.get("model"),
                "suite": suite.get("name"), "fingerprint": suite.get("fingerprint"),
                "tasks": len(doc.get("tasks") or []), "settings": config.get("settings"),
                "finished_at": doc.get("finished_at")}

    return {
        "schema": COMPARE_SCHEMA,
        "a": _about(a),
        "b": _about(b),
        "paired": k,
        "only_a": len(by_id_a) - k,
        "only_b": len(by_id_b) - k,
        "same_tasks": (a.get("suite") or {}).get("fingerprint")
        == (b.get("suite") or {}).get("fingerprint"),
        "method": "band = 2 x standard error of the per-task differences (b - a), "
                  "paired on task id",
        "rows": rows,
        "stop_reasons": stops,
    }
