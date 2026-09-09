"""Terminal view for ``effgen battle`` — several models racing one prompt.

On an interactive terminal each contender gets a column that fills in as its
answer streams, headed by the model id and footed by its live token count,
elapsed time and cost, ending in a verdict panel.

When stdout is not a terminal, or ``--json``/``--no-animation``/``NO_COLOR``
apply, no live frame is drawn: the battle runs to completion and prints one
structured result — the JSON document, or a Markdown report carrying each
model's full answer, the tally and the verdict.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any

from effgen.eval.battle import BattleResult, Contender, fmt_cost, run_battle
from effgen.ui.render import json_ensure_ascii
from effgen.ui.tables import console_is_interactive
from effgen.ui.theme import color_enabled, get_console

#: Rows a column borrows for its border, metrics and spacing. The rest of the
#: panel height is answer text.
_COLUMN_CHROME_ROWS = 8
#: Bounds on how tall a column grows, so the race fits a short terminal and does
#: not sprawl on a tall one.
_MIN_COLUMN_HEIGHT = 12
_MAX_COLUMN_HEIGHT = 28


def _state_label(contender: Contender) -> str:
    """Describe what a contender is doing, so a slow column is never ambiguous.

    A model being loaded reads as loading rather than as one that has stalled —
    a local model can spend seconds there before its first token.
    """
    if contender.state == "loading":
        return "loading model…"
    if contender.state == "streaming":
        return "generating…" if contender.answer else "waiting for first token…"
    if contender.state == "failed":
        return "failed"
    if contender.state == "done":
        return "done"
    return "queued"


def _footer(contender: Contender) -> str:
    """One line of live measurements for a column footer."""
    parts: list[str] = []
    if contender.load_s is not None:
        parts.append(f"load {contender.load_s:.1f}s")
    if contender.ttft_s is not None:
        parts.append(f"ttft {contender.ttft_s:.2f}s")
    if contender.latency_s is not None:
        parts.append(f"total {contender.latency_s:.2f}s")
    if contender.total_tokens is not None:
        tokens = f"{contender.total_tokens} tok"
        if contender.estimated_tokens:
            tokens += " (local count)"
        parts.append(tokens)
    if contender.state == "done":
        parts.append(fmt_cost(contender.cost_usd))
    elif not parts:
        parts.append(_state_label(contender))
    return " · ".join(parts)


def _column(contender: Contender, *, width: int, height: int) -> Any:
    """Render one contender as a bordered panel.

    The measurements sit at the top of the panel, where a narrow column wraps
    them instead of cutting them off. An answer longer than the column shows its
    most recent text, so a column that is still writing keeps moving; the whole
    answer is in the result the command prints or saves.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    if contender.error:
        answer: Any = Text(contender.error, style="red")
    elif contender.answer:
        budget = max(width, 1) * max(height - _COLUMN_CHROME_ROWS, 1)
        answer = Text(contender.answer[-budget:])
    else:
        answer = Text(_state_label(contender), style="dim")

    border = {
        "failed": "red",
        "done": "green",
        "streaming": "cyan",
    }.get(contender.state, "dim")
    return Panel(
        Group(Text(_footer(contender), style="dim"), Text(""), answer),
        title=contender.model,
        subtitle=_state_label(contender),
        border_style=border,
        width=width,
        height=height,
    )


def _live_view(contenders: list[Contender], console: Any) -> Any:
    """Lay the contenders out side by side, sized to the terminal."""
    from rich.columns import Columns

    total = max(console.width, 40)
    width = max(24, (total - 2 * len(contenders)) // max(1, len(contenders)))
    # Leave room under the columns for the verdict panel that follows them.
    height = min(_MAX_COLUMN_HEIGHT, max(_MIN_COLUMN_HEIGHT, console.height - 10))
    return Columns(
        [_column(c, width=width, height=height) for c in contenders],
        equal=True,
        expand=True,
    )


def render_verdict(result: BattleResult, console: Any) -> None:
    """Print the measured outcomes, and any judged pick, after the race."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    for key in ("fastest", "cheapest", "longest"):
        entry = result.verdict.get(key)
        if entry:
            table.add_row(key.capitalize(), f"{entry['model']} — {entry['detail']}")
    judge = result.verdict.get("judge")
    if judge:
        if judge.get("winner"):
            reason = judge.get("reasoning") or ""
            table.add_row(
                "Judged pick",
                f"{judge['winner']} — {reason} (judged by {judge['judge_model']})",
            )
        elif judge.get("error"):
            table.add_row("Judged pick", f"unavailable — {judge['error']}")
    failed = [c for c in result.contenders if not c.ok]
    if failed:
        table.add_row(
            "Did not answer",
            ", ".join(f"{c.model} ({c.error})" for c in failed),
        )
    spent = fmt_cost(result.total_cost()) if result.finishers else "—"
    table.add_row(
        "Race",
        f"{len(result.finishers)}/{len(result.contenders)} answered in "
        f"{result.wall_s:.2f}s · total {spent}",
    )
    console.print(Panel(table, title="Verdict", border_style="cyan"))


class _Columns:
    """Holds each contender in the column position its model was named in.

    A contender reports itself as it changes; the columns stay in the order the
    models were given rather than the order they happen to report in. Each
    contender keeps a column of its own, so entering one model id twice shows
    two independent racers instead of two views of the same one.
    """

    def __init__(self, models: list[str]) -> None:
        self._slots: list[Contender] = [Contender(model=m) for m in models]
        self._assigned: dict[int, int] = {}

    def update(self, contender: Contender) -> None:
        slot = self._assigned.get(id(contender))
        if slot is None:
            taken = set(self._assigned.values())
            slot = next(
                (i for i, c in enumerate(self._slots)
                 if c.model == contender.model and i not in taken),
                None,
            )
            if slot is None:
                return
            self._assigned[id(contender)] = slot
        self._slots[slot] = contender

    def current(self) -> list[Contender]:
        return list(self._slots)


def _run_live(
    prompt: str,
    models: list[str],
    *,
    console: Any,
    temperature: float | None,
    max_tokens: int | None,
    system_prompt: str | None,
    judge: str | None,
) -> BattleResult:
    """Race the models with a live side-by-side view, then print the verdict."""
    from rich.live import Live

    columns = _Columns(models)
    pending = threading.Event()

    def _on_update(contender: Contender) -> None:
        columns.update(contender)
        pending.set()

    with Live(console=console, auto_refresh=False, transient=False) as live:
        holder: dict[str, Any] = {}

        def _race() -> None:
            try:
                holder["result"] = run_battle(
                    prompt,
                    models,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    judge=judge,
                    on_update=_on_update,
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                holder["error"] = exc
            finally:
                pending.set()

        def _draw() -> None:
            live.update(_live_view(columns.current(), console), refresh=True)

        # The race owns the worker threads; this thread owns the screen, so a
        # redraw never happens from inside a contender's callback.
        racer = threading.Thread(target=_race, name="battle-race", daemon=True)
        racer.start()
        _draw()
        while racer.is_alive():
            if pending.wait(timeout=0.1):
                pending.clear()
                _draw()
        racer.join()
        _draw()

    if "error" in holder:
        raise holder["error"]
    result = holder["result"]
    render_verdict(result, console)
    return result


def run_battle_command(args: Any) -> int:
    """Entry point for ``effgen battle``."""
    from effgen.cli import progress as _progress

    prompt = (getattr(args, "prompt", "") or "").strip()
    if not prompt:
        print("A battle needs a prompt: effgen battle \"<prompt>\" -m a,b", file=sys.stderr)
        return 2

    models = [m.strip() for m in (getattr(args, "models", "") or "").split(",") if m.strip()]
    if len(models) < 2:
        print(
            "A battle needs at least two models to compare "
            "(-m openai:gpt-5-nano,groq:openai/gpt-oss-20b). "
            "Use `effgen models list` to see available ids.",
            file=sys.stderr,
        )
        return 2

    temperature = getattr(args, "temperature", None)
    max_tokens = getattr(args, "max_tokens", None)
    system_prompt = getattr(args, "system_prompt", None)
    judge = getattr(args, "judge", None)
    json_mode = bool(getattr(args, "output_json", False))

    console = get_console(theme_name=getattr(args, "theme", None))
    animated = _progress.animation_enabled(
        quiet=getattr(args, "quiet", False),
        no_animation=getattr(args, "no_animation", False),
    )
    live_capable = (
        not json_mode
        and animated
        and color_enabled()
        and console_is_interactive(console)
    )

    try:
        if live_capable:
            result = _run_live(
                prompt,
                models,
                console=console,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                judge=judge,
            )
        else:
            result = run_battle(
                prompt,
                models,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                judge=judge,
            )
    except KeyboardInterrupt:
        print("Battle interrupted.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if json_mode:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=json_ensure_ascii()))
    elif not live_capable:
        print(result.to_markdown())

    output = getattr(args, "output", None)
    if output:
        _write_battle_artifact(output, result)

    report = getattr(args, "report", None)
    if report:
        _write_battle_report(report, result)

    # A battle reports a comparison; it exits non-zero only when no contender
    # answered, which means there is nothing to compare.
    return 0 if result.finishers else 1


def _write_battle_artifact(path: str, result: BattleResult) -> None:
    """Save the battle as JSON or Markdown, chosen by the file extension."""
    from pathlib import Path

    target = Path(path)
    text = result.to_markdown() if target.suffix.lower() in (".md", ".markdown") else result.to_json()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"Saved battle to {target}", file=sys.stderr)


def _write_battle_report(path: str, result: BattleResult) -> None:
    """Save the battle as a self-contained HTML report."""
    from effgen.ui.report_html import write_html_report

    write_html_report(path, result.to_dict(), kind="battle")
    print(f"Wrote HTML report to {path}", file=sys.stderr)
