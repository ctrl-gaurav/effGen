"""Where each monitor panel's numbers come from.

Reads the five sources ``effgen top`` assembles a screen from: the durable run
history, the running server's dashboard aggregation, the local cost ledger and
the physical GPUs. :func:`collect_snapshot` returns one document with every
panel in it, which both rendering paths consume without touching a source
themselves. ``effgen.cli.monitor`` stays the import path for every name here.

Nothing here raises. A source that cannot be read comes back as a panel with
``available`` false and an ``unavailable_reason`` naming what failed, so one
missing source never blanks the screen or hides the panels that did read.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

from effgen.cli.monitor_format import (
    SCOPE_ACTIVITY,
    SCOPE_GPU,
    SCOPE_HTTP_STATUS,
    SCOPE_SPEND,
    SCOPE_TRAFFIC,
    _preview,
)

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_ACTIVITY_LIMIT = 12
BURN_WINDOW_S = 3600.0
_HTTP_TIMEOUT_S = 3.0


# -------------------------------------------------------------------------
# Source resolution
# -------------------------------------------------------------------------


def resolve_server_url(url: str | None = None, port: int | None = None) -> str:
    """Resolve the server base URL from *url*, *port*, the environment, or the default.

    Precedence: an explicit ``--url``, then ``--port`` on localhost, then
    ``EFFGEN_SERVER_URL``, then ``http://127.0.0.1:8000`` (the port
    ``effgen serve`` binds by default). A bare ``host:port`` gains an
    ``http://`` scheme; a trailing slash is dropped.
    """
    candidate = url or None
    if candidate is None and port is not None:
        candidate = f"http://127.0.0.1:{int(port)}"
    if candidate is None:
        candidate = os.environ.get("EFFGEN_SERVER_URL") or ""
    candidate = candidate.strip() or DEFAULT_SERVER_URL
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def resolve_api_key(api_key: str | None = None) -> str | None:
    """Resolve the server API key from *api_key* or ``EFFGEN_API_KEY``."""
    key = (api_key or os.environ.get("EFFGEN_API_KEY") or "").strip()
    return key or None


def _fetch_json(url: str, *, api_key: str | None, timeout: float = _HTTP_TIMEOUT_S) -> tuple[
    dict[str, Any] | None, str | None
]:
    """GET *url* and decode JSON. Returns ``(document, error_message)``.

    A URL problem is reported the same way an unreachable server is, so a
    mistyped ``--url`` leaves the panels that need no server intact instead of
    failing the whole command.
    """
    if not url.startswith(("http://", "https://")):
        return None, f"unsupported URL scheme: {url}"
    try:
        request = urllib.request.Request(url, method="GET")  # noqa: S310 - http(s) only, checked above
    except ValueError as exc:
        return None, f"invalid server URL {url} ({exc})"
    request.add_header("Accept", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = response.read().decode("utf-8", errors="replace")
        return json.loads(payload), None
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return None, (
                f"HTTP {exc.code} — the server requires an API key. Pass --api-key, "
                "set EFFGEN_API_KEY, or start the server with EFFGEN_PUBLIC_DASHBOARD=1."
            )
        return None, f"HTTP {exc.code} from {url}"
    except urllib.error.URLError as exc:
        return None, f"no server reachable at {url} ({exc.reason})"
    except TimeoutError:
        return None, f"no response from {url} within {timeout:.0f}s"
    except json.JSONDecodeError:
        return None, f"{url} did not return JSON"
    except ValueError as exc:
        return None, f"invalid server URL {url} ({exc})"
    except http.client.HTTPException as exc:
        # A host with spaces or control characters (InvalidURL) only fails once
        # a connection is attempted, and a port serving something that is not
        # HTTP raises here too. Neither is a ValueError or an OSError. The
        # detail can quote raw bytes, so it is flattened to one line rather than
        # allowed to break the panel it is rendered in.
        detail = " ".join(f"{type(exc).__name__}: {exc}".split())
        return None, f"{url} did not answer as an HTTP server ({_preview(detail, 120)})"
    except OSError as exc:
        return None, f"could not read {url} ({exc})"


# -------------------------------------------------------------------------
# Panel collection
# -------------------------------------------------------------------------


def _collect_activity(limit: int) -> dict[str, Any]:
    """Recent completed runs from the durable, cross-process run history."""
    try:
        from effgen.observability.run_log import read_runs

        runs = read_runs(limit=max(1, limit))
    except Exception as exc:  # noqa: BLE001 - a missing history must not blank the screen
        return {
            "scope": SCOPE_ACTIVITY,
            "available": False,
            "unavailable_reason": f"run history unreadable ({exc})",
            "runs": [],
        }
    rows = [
        {
            "ts": run.get("ts"),
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "model": run.get("model"),
            "agent": run.get("agent"),
            "duration_s": run.get("duration_s"),
            # ``None`` means the model has no published price; it is rendered as
            # an em dash and never summed as zero.
            "cost_usd": run.get("cost_usd"),
            "task": _preview(run.get("task"), 60),
            "error": run.get("error"),
        }
        for run in runs
    ]
    return {
        "scope": SCOPE_ACTIVITY,
        "available": True,
        "unavailable_reason": None,
        "runs": rows,
    }


def _collect_server(url: str, api_key: str | None) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Traffic and per-model panels from the server's dashboard aggregation."""
    document, error = _fetch_json(f"{url}/dashboard/data.json", api_key=api_key)
    if document is None:
        unavailable = {
            "scope": SCOPE_TRAFFIC,
            "available": False,
            "unavailable_reason": error,
            "server_url": url,
        }
        return (
            {**unavailable},
            {**unavailable, "rows": []},
            False,
        )

    metrics = document.get("metrics") or {}
    slo = document.get("slo") or {}
    requests = int(metrics.get("total_requests") or 0)
    errors = int(metrics.get("total_errors") or 0)
    traffic = {
        "scope": SCOPE_TRAFFIC,
        "available": True,
        "unavailable_reason": None,
        "server_url": url,
        "requests": requests,
        "errors": errors,
        "error_rate": (errors / requests) if requests else 0.0,
        "tokens": int(metrics.get("total_tokens") or 0),
        "cost_usd": metrics.get("cost_usd"),
        "avg_latency_s": metrics.get("avg_latency_s"),
        "p50_latency_s": slo.get("p50_latency_s"),
        "p95_latency_s": slo.get("p95_latency_s"),
        "p99_latency_s": slo.get("p99_latency_s"),
        "availability": slo.get("availability"),
        "by_status": document.get("by_status") or {},
        "by_status_scope": SCOPE_HTTP_STATUS,
    }
    by_model = {
        "scope": SCOPE_TRAFFIC,
        "available": True,
        "unavailable_reason": None,
        "server_url": url,
        "rows": list(document.get("by_model") or []),
    }
    return traffic, by_model, True


def _collect_spend() -> dict[str, Any]:
    """24-hour spend, daily budget, and a trailing-hour burn rate from the ledger."""
    panel: dict[str, Any] = {
        "scope": SCOPE_SPEND,
        "available": False,
        "unavailable_reason": None,
        "total_cost_usd": 0.0,
        "requests": 0,
        "daily_budget_usd": None,
        "budget_used_pct": None,
        "burn_rate_usd_per_hour": None,
        "burn_window_s": BURN_WINDOW_S,
        "unpriced_models": [],
        "rows": [],
    }
    try:
        from effgen.models._cost import _budget_config_path, pricing_status
        from effgen.models._cost_store import SQLiteCostStore

        store = SQLiteCostStore()
        events = store.query_today()
        recent = store.query_since(time.time() - BURN_WINDOW_S)
    except Exception as exc:  # noqa: BLE001 - an unreadable ledger must not blank the screen
        panel["unavailable_reason"] = f"cost ledger unreadable ({exc})"
        return panel

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        key = (event.provider, event.model)
        row = aggregate.setdefault(
            key,
            {
                "provider": event.provider,
                "model": event.model,
                "requests": 0,
                "tokens": 0,
                "cost_usd": 0.0,
            },
        )
        row["requests"] += int(getattr(event, "calls", 1) or 1)
        row["tokens"] += int(event.prompt_tokens or 0) + int(event.completion_tokens or 0)
        row["cost_usd"] += float(event.cost_usd or 0.0)

    rows = sorted(aggregate.values(), key=lambda r: r["cost_usd"], reverse=True)
    unpriced: list[str] = []
    for row in rows:
        if row["cost_usd"] > 0:
            row["cost_label"] = f"${row['cost_usd']:.6f}"
            continue
        try:
            status = pricing_status(row["provider"], row["model"])
        except Exception:  # noqa: BLE001 - an unknown model is simply unlabelled
            status = None
        if status == "unpriced":
            row["cost_label"] = "—"
            unpriced.append(f"{row['provider']}:{row['model']}")
        elif status == "free":
            row["cost_label"] = "free"
        else:
            row["cost_label"] = f"${row['cost_usd']:.6f}"

    total_cost = sum(float(r["cost_usd"]) for r in rows)
    burn = sum(float(e.cost_usd or 0.0) for e in recent) / (BURN_WINDOW_S / 3600.0)

    daily_budget = None
    try:
        budget_path = _budget_config_path()
        if budget_path.exists():
            daily_budget = json.loads(budget_path.read_text()).get("daily")
    except Exception:  # noqa: BLE001 - an unreadable budget file just means no budget shown
        daily_budget = None

    panel.update(
        {
            "available": True,
            "total_cost_usd": round(total_cost, 8),
            "requests": sum(int(r["requests"]) for r in rows),
            "daily_budget_usd": daily_budget,
            "budget_used_pct": (
                round(100.0 * total_cost / float(daily_budget), 2)
                if daily_budget
                else None
            ),
            "burn_rate_usd_per_hour": round(burn, 8),
            "unpriced_models": unpriced,
            "rows": [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "requests": r["requests"],
                    "tokens": r["tokens"],
                    "cost_usd": round(float(r["cost_usd"]), 8),
                    "cost_label": r["cost_label"],
                }
                for r in rows
            ],
        }
    )
    return panel


def _collect_gpu() -> dict[str, Any]:
    """Per-device memory and utilisation for the physical GPUs on this host."""
    try:
        from effgen.gpu.cuda_compat import per_gpu_status

        devices = per_gpu_status()
    except Exception as exc:  # noqa: BLE001 - no GPU layer is a normal state, not an error
        return {
            "scope": SCOPE_GPU,
            "available": False,
            "unavailable_reason": f"GPU status unavailable ({exc})",
            "devices": [],
        }
    if not devices:
        return {
            "scope": SCOPE_GPU,
            "available": False,
            "unavailable_reason": "no CUDA devices detected on this host",
            "devices": [],
        }
    gib = 1024**3
    return {
        "scope": SCOPE_GPU,
        "available": True,
        "unavailable_reason": None,
        "devices": [
            {
                "index": device.index,
                "name": device.name,
                "total_gb": round(device.total_bytes / gib, 2),
                "used_gb": round(device.used_bytes / gib, 2),
                "free_gb": round(device.free_bytes / gib, 2),
                "memory_used_pct": (
                    round(100.0 * device.used_bytes / device.total_bytes, 1)
                    if device.total_bytes
                    else None
                ),
                "utilization_pct": device.utilization_pct,
            }
            for device in devices
        ],
    }


def collect_snapshot(
    *,
    url: str | None = None,
    port: int | None = None,
    api_key: str | None = None,
    limit: int = DEFAULT_ACTIVITY_LIMIT,
    interval: float | None = None,
    gpu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect one full monitor snapshot.

    Returns a document with a ``header`` plus the ``activity``, ``traffic``,
    ``by_model``, ``spend`` and ``gpu`` panels. Each panel carries a ``scope``
    describing the window and process it measures, and an ``available`` flag
    with an ``unavailable_reason`` when its source could not be read. Pass a
    previously collected *gpu* panel to reuse it instead of re-sampling.

    Args:
        url: Base URL of the server to read, when one is running.
        port: Port on localhost to read, used when *url* is absent.
        api_key: Credential for the server's authenticated routes.
        limit: Most activity rows to collect.
        interval: The refresh interval reported in each panel's scope.
        gpu: A previously collected GPU panel to reuse instead of re-sampling.
    """
    server_url = resolve_server_url(url, port)
    key = resolve_api_key(api_key)
    traffic, by_model, reachable = _collect_server(server_url, key)
    return {
        "ts": datetime.now(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "header": {
            "version": _effgen_version(),
            "hostname": socket.gethostname(),
            "server_url": server_url,
            "server_reachable": reachable,
            "server_detail": traffic.get("unavailable_reason"),
            "interval_s": interval,
        },
        "activity": _collect_activity(limit),
        "traffic": traffic,
        "by_model": by_model,
        "spend": _collect_spend(),
        "gpu": gpu if gpu is not None else _collect_gpu(),
    }


def _effgen_version() -> str:
    try:
        from effgen import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - the version is cosmetic here
        return "unknown"
