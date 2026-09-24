"""Assemble a multi-agent execution into a node-link graph.

A team or workflow run issues one execution id that every member run carries
(see :func:`effgen.observability.tracing.execution_scope`). This module reads
those correlated records back and returns the graph they describe: agents and
the tools they reached as nodes, delegation/handoff and tool use as edges, each
node annotated with the status, model, cost, tokens and duration of the work it
did.

Two sources are merged:

* the durable run store (:mod:`effgen.observability.run_log`), which records
  runs from every process — a team started from a script or the CLI is visible
  to a reader elsewhere, such as the dashboard;
* the in-process span buffer, which adds the tool calls and the model calls of
  work happening in this process right now.

Nodes and edges are ordered deterministically, so re-reading a live execution
returns the graph in the same order rather than reshuffling it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_topology"]

#: Status ranked by how much it should dominate a node that did several things.
_STATUS_RANK = {"ok": 0, "running": 1, "skipped": 2, "error": 3}


def _worse(current: str, candidate: str) -> str:
    """Return whichever status a viewer needs to see more."""
    return candidate if _STATUS_RANK.get(candidate, 0) > _STATUS_RANK.get(current, 0) else current


def _node(nodes: dict[str, dict[str, Any]], node_id: str, node_type: str) -> dict[str, Any]:
    """Return the node with *node_id*, creating it on first use."""
    node = nodes.get(node_id)
    if node is None:
        node = {
            "id": node_id,
            "label": node_id.split(":", 1)[1] if node_id.startswith("tool:") else node_id,
            "type": node_type,
            "status": "ok",
            "model": None,
            "role": None,
            "runs": 0,
            "calls": 0,
            # None until a run on this node reports a price: an unpriced model
            # costs an unknown amount, not $0.
            "cost_usd": None,
            "tokens": 0,
            "duration_s": 0.0,
            "agent": None,
            "note": None,
            "task": None,
            "output": None,
            "error": None,
        }
        nodes[node_id] = node
    return node


def _add_edge(
    edges: dict[tuple[str, str, str], dict[str, Any]],
    source: str,
    target: str,
    kind: str,
) -> dict[str, Any]:
    """Return the edge for this (source, target, kind), creating it on first use."""
    key = (source, target, kind)
    edge = edges.get(key)
    if edge is None:
        edge = {"source": source, "target": target, "kind": kind, "count": 0}
        edges[key] = edge
    return edge


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _identity(agent: Any, role: Any) -> str:
    """The graph identity of a run: its workflow node id, else the agent name.

    Workflow edges are drawn between node ids, so a node keeps its id as its
    identity and carries the agent name alongside it.
    """
    role_text = str(role or "")
    if role_text.startswith("node:"):
        return role_text.split(":", 1)[1] or str(agent or "node")
    return str(agent or "agent")


#: Roles whose runs were handed out by another agent rather than following it.
_DELEGATED_ROLES = frozenset({"worker", "sub-agent"})


def _edge_kind(role: Any) -> str:
    """The relationship an edge represents, named by the receiving run's role."""
    return "delegation" if str(role or "") in _DELEGATED_ROLES else "handoff"


def _apply_run(
    nodes: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str, str], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    """Fold one stored run record into the graph."""
    role = record.get("role")
    agent = _identity(record.get("agent"), role)
    node = _node(nodes, agent, "manager" if role == "manager" else "agent")
    node["agent"] = record.get("agent")
    node["runs"] += 1
    node["role"] = role or node["role"]
    node["model"] = record.get("model") or node["model"]
    if record.get("cost_usd") is not None:
        node["cost_usd"] = round(_float(node["cost_usd"]) + _float(record.get("cost_usd")), 8)
    node["tokens"] += _int(record.get("input_tokens")) + _int(record.get("output_tokens"))
    node["duration_s"] = round(node["duration_s"] + _float(record.get("duration_s")), 3)
    if node["task"] is None:
        node["task"] = record.get("task")
    if record.get("output"):
        node["output"] = record.get("output")
    if record.get("status") == "error":
        node["status"] = _worse(node["status"], "error")
        node["error"] = record.get("error") or node["error"]

    parent = record.get("parent_agent")
    if parent and str(parent) != agent:
        _node(nodes, str(parent), "manager" if role in _DELEGATED_ROLES else "agent")
        # The child's role names the relationship, so the edge kind does not
        # depend on which of the two runs was stored first.
        _add_edge(edges, str(parent), agent, _edge_kind(role))["count"] += 1


def _apply_span(
    nodes: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str, str], dict[str, Any]],
    span: dict[str, Any],
    agent_by_run: dict[str, str],
) -> None:
    """Fold one buffered span into the graph."""
    kind = span.get("kind")
    if kind == "agent" and span.get("agent"):
        agent = _identity(span.get("agent"), span.get("role"))
        node = _node(nodes, agent, "manager" if span.get("role") == "manager" else "agent")
        node["role"] = span.get("role") or node["role"]
        parent = span.get("parent_agent")
        if parent and str(parent) != agent:
            _add_edge(edges, str(parent), agent, _edge_kind(span.get("role")))
        if span.get("status") == "skipped":
            node["status"] = _worse(node["status"], "skipped")
            node["note"] = node["note"] or span.get("note")
        if span.get("error"):
            node["status"] = _worse(node["status"], "error")
            node["error"] = node["error"] or str(span["error"])
        return

    if kind == "tool" and span.get("tool"):
        agent = agent_by_run.get(str(span.get("run_id") or ""))
        tool_id = f"tool:{span['tool']}"
        node = _node(nodes, tool_id, "tool")
        node["calls"] += 1
        node["duration_s"] = round(node["duration_s"] + _float(span.get("duration_ms")) / 1000.0, 3)
        if span.get("error"):
            node["status"] = _worse(node["status"], "error")
            node["error"] = node["error"] or str(span["error"])
        if agent:
            _add_edge(edges, agent, tool_id, "tool_use")["count"] += 1


def build_topology(*, limit: int = 6, span_limit: int = 400) -> dict[str, Any]:
    """Return recent multi-agent executions as node-link graphs.

    Args:
        limit: Maximum number of executions to return, newest first.
        span_limit: How many buffered spans to read for live tool/model detail.

    Returns:
        ``{"executions": [...], "count": n}``. Each execution carries its
        ``id``, ``kind``, ``name``, timestamps, totals, ``status``, ``nodes``
        and ``edges``. The list is empty when nothing multi-agent has run yet —
        which a caller should report as such rather than as an empty graph.
    """
    from . import run_log
    from .tracing import get_recent_spans

    executions = run_log.read_executions(limit=max(1, int(limit)))
    try:
        spans = get_recent_spans(limit=max(1, int(span_limit)))
    except Exception:  # noqa: BLE001 - a graph is still useful without live spans
        spans = []

    spans_by_execution: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        execution_id = span.get("execution_id")
        if execution_id:
            spans_by_execution.setdefault(str(execution_id), []).append(span)

    # Spans can describe an execution whose runs have not been stored yet (work
    # still in flight). Add those so a live execution appears while it runs.
    known = {str(e["id"]) for e in executions}
    for execution_id, group in spans_by_execution.items():
        if execution_id in known:
            continue
        first = group[0]
        executions.append({
            "id": execution_id,
            "kind": first.get("execution_kind"),
            "name": first.get("execution_name"),
            "started": first.get("ts"),
            "updated": first.get("ts"),
            "runs": [],
            "cost_usd": None,
            "tokens": 0,
            "failed": 0,
        })

    out: list[dict[str, Any]] = []
    for execution in executions[: max(1, int(limit))]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}

        for record in execution["runs"]:
            _apply_run(nodes, edges, record)

        group = list(reversed(spans_by_execution.get(str(execution["id"]), [])))
        agent_by_run = {
            str(s.get("run_id")): _identity(s.get("agent"), s.get("role"))
            for s in group
            if s.get("kind") == "agent" and s.get("run_id") and s.get("agent")
        }
        for span in group:
            _apply_span(nodes, edges, span, agent_by_run)

        node_list = sorted(
            nodes.values(),
            key=lambda n: (n["type"] != "manager", n["type"] == "tool", n["id"]),
        )
        out.append({
            "id": execution["id"],
            "kind": execution.get("kind") or "team",
            "name": execution.get("name") or execution["id"],
            "started": execution.get("started"),
            "updated": execution.get("updated"),
            "cost_usd": execution.get("cost_usd"),
            "tokens": execution.get("tokens", 0),
            "status": "error" if execution.get("failed") else "ok",
            "nodes": node_list,
            "edges": list(edges.values()),
        })

    return {"executions": out, "count": len(out)}
