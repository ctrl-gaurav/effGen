"""The run card: one agent run rendered as a shareable document.

Shows the task, the model it ran on, the answer or the typed error, the
step-by-step tool trace with per-step durations, sources and citations, and the
run's tokens, cost and latency.

Two documents render through this builder: the full ``effgen run --json``
result, and a stored run-history record, which names its answer ``output`` and
its timing ``duration_s`` and keeps no step trace.
"""

from __future__ import annotations

from typing import Any

from .report_html_components import _badge, _bar_chart, _cards, _link, _table
from .report_html_format import (
    _DASH,
    _esc,
    _int,
    _mapping,
    _secs,
    _sequence,
    _truncate,
    _usd,
)


def _children(node: Any) -> list[Any]:
    """The child nodes of a tree node, ignoring a malformed ``children`` value."""
    children = node.get("children") if isinstance(node, dict) else None
    return children if isinstance(children, list) else []


def _sort_key(value: Any) -> float:
    """A start time as a number, placing an unusable one first."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _tool_steps(tree: Any) -> list[dict[str, Any]]:
    """Flatten a run's execution tree into its tool nodes, in start order."""
    steps: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if str(node.get("node_type") or "") == "tool":
            steps.append(node)
        for child in _children(node):
            walk(child)

    for child in _children(tree):
        walk(child)
    steps.sort(key=lambda n: _sort_key(n.get("started_at")))
    return steps


#: Engine prefixes that mark an ``engine:model_id`` string as running on the
#: caller's own hardware, mirroring the model loader's local engines.
_LOCAL_ENGINES = ("transformers:", "vllm:", "gguf:", "mlx:")


def _unpriced_label(model: str, provider: str, total_tokens: Any = None) -> str:
    """What to show in place of a cost the run does not carry.

    A run on local hardware has no price to report. A hosted run reaches here
    either because it never completed a billed call — a run that failed on the
    first request spends nothing, which says nothing about the model's price —
    or because the model was billed and the catalog has no rate for it, a gap in
    the pricing data rather than a free call. The three are not labelled alike,
    so a card never reads as "this model publishes no rate" for a run that
    simply never got that far.
    """
    if model.startswith(_LOCAL_ENGINES) or not provider:
        return "unpriced (local)"
    if not total_tokens:
        return "no billed call"
    return "unpriced (no published rate)"


def _run_body(data: dict[str, Any]) -> tuple[str, str, str]:
    metadata = _mapping(data.get("metadata"))
    # A stored history record names its answer `output` and its timing
    # `duration_s`, so both the full run document and the history record
    # render through the same path.
    summary_only = "success" not in data and "status" in data
    success = (
        str(data.get("status")) == "ok" if summary_only else bool(data.get("success"))
    )
    task = str(data.get("task") or metadata.get("task") or "")
    model = str(data.get("model") or metadata.get("model") or "")
    provider = str(data.get("provider") or metadata.get("provider") or "")
    run_id = str(data.get("run_id") or metadata.get("run_id") or "")
    duration = data.get("execution_time")
    if duration is None:
        duration = data.get("duration_s", metadata.get("duration_s"))
    cost = metadata.get("cost_usd", data.get("cost_usd"))
    prompt_tokens = metadata.get("prompt_tokens", data.get("input_tokens"))
    completion_tokens = metadata.get("completion_tokens", data.get("output_tokens"))
    total_tokens = metadata.get("total_tokens", data.get("tokens_used"))
    if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens

    target = model or "the configured model"
    if provider:
        target = f"{target} ({provider})"

    # A run the loop stopped has no answer but is not an outright failure: it
    # reads as its own outcome, naming what stopped it, so the progress below is
    # not taken for a result.
    run_outcome = str(data.get("outcome") or "")
    stop_reason = str(data.get("stop_reason") or metadata.get("reason") or "")
    stopped = run_outcome == "stopped" or (
        not success and bool(metadata.get("partial"))
    )
    if success:
        verdict = _badge("succeeded", "ok")
    elif stopped:
        verdict = _badge(
            f"stopped ({stop_reason})" if stop_reason else "stopped", "warn"
        )
    else:
        verdict = _badge("failed", "err")

    parts: list[str] = [
        '<div class="verdict">'
        '<div class="verdict-label">Task</div>'
        f'<div class="verdict-value task-text" id="run-task">{_esc(task) or _DASH}</div>'
        f'<p class="verdict-note">{verdict} '
        f"on {_esc(target)}"
        + (f" · run {_esc(run_id)}" if run_id else "")
        + "</p></div>"
    ]

    if task:
        parts.append(
            '<div class="actions">'
            '<button type="button" class="copy-btn" data-copy-from="run-task">'
            "Copy task</button>"
            '<button type="button" class="copy-btn" data-copy-from="run-command">'
            "Copy command</button>"
            f'<span hidden id="run-command">{_esc(_run_command(task, model))}</span>'
            "</div>"
        )

    steps = _tool_steps(data.get("execution_tree"))
    tool_calls = data.get("tool_calls")
    if tool_calls is None:
        tool_calls = len(steps) if steps else None
    parts.append(_cards([
        ("Duration", _secs(duration, 2), f"{_int(data.get('iterations'))} iterations"
         if data.get("iterations") is not None else ""),
        ("Tool calls", _int(tool_calls), f"{len(steps)} recorded steps" if steps else ""),
        (
            "Tokens",
            _int(total_tokens),
            f"{_int(prompt_tokens)} in / {_int(completion_tokens)} out"
            if prompt_tokens is not None or completion_tokens is not None else "",
        ),
        # A model with no published price records no cost at all. Reporting
        # $0.00 for it would read as a free cloud call rather than a local run.
        (
            "Cost",
            _usd(cost, 6) if cost is not None
            else _unpriced_label(model, provider, total_tokens),
            "",
        ),
    ]))

    error = metadata.get("error")
    if not success and isinstance(error, dict):
        parts.append("<h2>Outcome</h2>" if stopped else "<h2>Error</h2>")
        parts.append(_table(
            ["Field", "Value"],
            [
                ["Type", _esc(error.get("type") or _DASH)],
                ["Category", _esc(error.get("category") or _DASH)],
                ["Provider", _esc(error.get("provider") or _DASH)],
                ["Model", _esc(error.get("model") or _DASH)],
                ["Retryable", _esc("yes" if error.get("retryable") else "no")],
                ["Message", f'<span class="wrap">{_esc(error.get("message") or _DASH)}</span>'],
            ],
        ))
    elif not success and data.get("error"):
        parts.append("<h2>Error</h2>")
        parts.append(f'<div class="prose">{_esc(data.get("error"))}</div>')

    answer = str(data.get("output") or "")
    if stopped:
        # The run has no answer; ``output`` is the outcome already shown above.
        # What it did reach is shown as progress, under its own heading.
        progress = str(metadata.get("partial_output") or "")
        parts.append("<h2>Partial progress</h2>")
        if progress.strip():
            parts.append(
                '<p class="empty">Tool output and reasoning the run had reached '
                "when the cap stopped it — not an answer.</p>"
            )
            parts.append(f'<div class="prose">{_esc(progress)}</div>')
        else:
            parts.append(
                '<p class="empty">The run reached the cap with nothing to report.</p>'
            )
    else:
        parts.append("<h2>Answer</h2>")
        if answer.strip():
            parts.append(f'<div class="prose">{_esc(answer)}</div>')
            if summary_only:
                parts.append(
                    '<p class="empty">Rendered from stored run history, which keeps a '
                    "truncated answer and no step trace. Export at run time with "
                    "<code>effgen run --card</code> for the full answer, the tool "
                    "trace and sources.</p>"
                )
        else:
            parts.append('<p class="empty">The run produced no answer text.</p>')

    if steps:
        parts.append("<h2>Steps</h2>")
        parts.append(_bar_chart(
            "Step duration",
            [
                (
                    f"{idx}. {node.get('name') or 'step'}",
                    node.get("duration"),
                    _secs(node.get("duration"), 3),
                )
                for idx, node in enumerate(steps, start=1)
            ],
            role="accent2",
        ))
        rows = []
        for idx, node in enumerate(steps, start=1):
            meta = _mapping(node.get("metadata"))
            failed = str(node.get("status") or "") == "failed"
            outcome = meta.get("error") if failed else meta.get("result")
            rows.append([
                f'<span class="num">{idx}</span>',
                _esc(node.get("name") or _DASH),
                f'<span class="mono">{_esc(_truncate(meta.get("tool_input") or _DASH, 300))}</span>',
                _badge("failed" if failed else "ok", "err" if failed else "ok"),
                f'<span class="mono">{_esc(_truncate(outcome if outcome is not None else _DASH, 300))}</span>',
                f'<span class="num">{_secs(node.get("duration"), 3)}</span>',
            ])
        parts.append(_table(
            ["#", "Tool", "Input", "Status", "Result", "Duration"], rows,
        ))

    parts.append(_conversation(metadata))

    sources = [s for s in _sequence(data.get("sources")) if s]
    if sources:
        parts.append("<h2>Sources</h2>")
        parts.append(
            '<ol class="sources">'
            + "".join(f"<li>{_link(s)}</li>" for s in sources)
            + "</ol>"
        )

    citations = _sequence(data.get("citations"))
    if citations:
        parts.append("<h2>Citations</h2>")
        rows = []
        for c in citations:
            if not isinstance(c, dict):
                rows.append([_DASH, f'<span class="wrap">{_esc(c)}</span>'])
                continue
            quote = str(c.get("quote") or "")
            rows.append([
                f'<span class="num">{_esc(c.get("index", _DASH))}</span>',
                _link(c.get("source"))
                + (f'<p class="quote">{_esc(_truncate(quote, 400))}</p>' if quote else ""),
            ])
        parts.append(_table(["#", "Source"], rows))

    subtitle = _truncate(task, 160) if task else "Agent run"
    return "Run Card", subtitle, "".join(parts)


def _conversation(metadata: Any) -> str:
    """The run's conversation as a table of steps, or the empty string.

    A card is read to find out what the model was actually sent, so the steps
    are rendered as themselves — the frame, the question, each thought, each
    call with the input it was given, each result and how the run ended — with
    secrets redacted and nothing in them that differs between two runs that
    took the same path.
    """
    from ..core.thread_render import render_thread

    steps = render_thread(_mapping(metadata).get("thread"), max_chars=2000)
    if not steps:
        return ""
    rows = []
    for step in steps:
        detail = " ".join(f"{k}={v}" for k, v in step.detail.items())
        rows.append([
            f'<span class="num">{"&nbsp;" * (step.depth * 4)}{step.position}</span>',
            _esc(step.label) + (f'<p class="quote">{_esc(detail)}</p>' if detail else ""),
            f'<span class="wrap">{_esc(step.body) or _DASH}</span>',
        ])
    note = (
        '<p class="empty">Every step the run took, in order — what the model '
        'was framed by, what it was asked, what it called and what came back. '
        "Secrets are redacted.</p>"
    )
    return "<h2>Conversation</h2>" + note + _table(["#", "Step", "Content"], rows)


def _run_command(task: str, model: str) -> str:
    """The ``effgen run`` invocation that reproduces this run."""
    import shlex

    argv = ["effgen", "run", shlex.quote(task)]
    if model:
        argv.extend(["-m", shlex.quote(model)])
    return " ".join(argv)
