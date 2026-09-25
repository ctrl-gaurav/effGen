"""The ``effgen bench`` command: ``init``, ``run`` and ``compare``.

``run`` loads a suite file, runs it against the model the caller names and
prints the run's table — accuracy, tasks, errors, LLM and tool calls, prompt,
completion and cached-input tokens, wall, framework, model and tool time, cost
and stop reasons — then saves the run so ``compare`` can read it. ``compare``
pairs two saved runs task by task and prints a noise band beside every delta.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effgen.cli import progress as _progress
from effgen.ui.render import json_ensure_ascii
from effgen.ui.tables import render_table

#: Row labels of the run table, one per key of the run document's ``table``.
TABLE_LABELS: dict[str, str] = {
    "accuracy": "Accuracy (%)",
    "tasks": "Tasks",
    "errors": "Errors",
    "llm_calls": "LLM calls",
    "tool_calls": "Tool calls",
    "prompt_tokens": "Prompt tokens",
    "completion_tokens": "Completion tokens",
    "cached_input_tokens": "Cached input tokens",
    "wall_s": "Wall time (s)",
    "framework_s": "Framework time (s)",
    "model_wait_s": "Model wait (s)",
    "tool_wait_s": "Tool wait (s)",
    "cost_usd": "Cost (USD)",
    "unpriced_calls": "Unpriced calls",
    "elapsed_s": "Elapsed, whole run (s)",
    "stop_reasons": "Stop reasons",
}


def _fmt(field: str, value: Any) -> str:
    if value is None:
        return "-"
    if field == "accuracy":
        return f"{value:.2f}"
    if field == "cost_usd":
        return f"{value:.6f}"
    if field.endswith("_s"):
        return f"{value:.3f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def table_rows(table: dict[str, Any]) -> list[list[str]]:
    """The run table as ``[label, per task, total]`` rows, one per table key."""
    rows: list[list[str]] = []
    for key, label in TABLE_LABELS.items():
        value = table.get(key)
        if key == "stop_reasons":
            text = ", ".join(f"{k} {v}" for k, v in (value or {}).items()) or "-"
            rows.append([label, text, ""])
        elif key == "cost_usd":
            if value is None:
                rows.append([label, "unpriced", "unpriced"])
            else:
                rows.append([label, _fmt(key, value["mean"]), _fmt(key, value["total"])])
        elif isinstance(value, dict):
            rows.append([label, _fmt(key, value.get("mean")), _fmt(key, value.get("total"))])
        elif key == "accuracy":
            rows.append([label, _fmt(key, value), ""])
        else:
            rows.append([label, "", _fmt(key, value)])
    return rows


def _emit_json(document: dict[str, Any]) -> None:
    print(json.dumps(document, indent=2, default=str, ensure_ascii=json_ensure_ascii()))


def _handle_init(args: Any, cli: Any) -> int:
    from effgen.bench import example_suite_text

    target = Path(args.path)
    if target.exists() and not args.force:
        cli.print_error(f"{target} exists; pass --force to overwrite it")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(example_suite_text(), encoding="utf-8")
    cli.print_success(f"Wrote a starter suite to {target}")
    cli.print(f"Run it with: effgen bench run {target} --model <model id> [--base-url <endpoint>]")
    return 0


def _handle_run(args: Any, cli: Any) -> int:
    from effgen.bench import (
        BenchConfig,
        BenchRunError,
        SuiteError,
        default_runs_dir,
        load_suite,
        run_suite,
        write_run,
    )

    json_mode = bool(getattr(args, "output_json", False))
    if json_mode:
        cli._human_to_stderr = True
    try:
        suite = load_suite(args.suite, n=args.n, seed=args.seed)
    except (SuiteError, OSError) as exc:
        cli.print_error(f"Cannot load the suite: {exc}")
        return 2

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            cli.print_error(f"--api-key-env {args.api_key_env}: that variable is not set")
            return 2
    config = BenchConfig(
        model=args.model or suite.model or "",
        base_url=args.base_url or suite.base_url,
        api_key=api_key,
        engine=args.engine,
        concurrency=args.concurrency,
        label=args.label or "",
        settings={"temperature": args.temperature, "max_tokens": args.max_tokens,
                  "max_iterations": args.max_iterations},
    )
    if not config.model:
        cli.print_error("No model: pass --model, or set 'model' in the suite file")
        return 2

    out_dir: Path | None = None
    if not args.no_save:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(args.out) if args.out else default_runs_dir() / f"{suite.name}-{stamp}"

    cli.print(f"Suite {suite.name}: {len(suite.tasks)} of {suite.total_tasks} tasks, "
              f"seed {suite.seed}, scorer {suite.scorer}, tools {suite.tools or 'none'}")
    cli.print(f"Model {config.model}" + (f" at {config.base_url}" if config.base_url else ""))
    animate = not json_mode and _progress.animation_enabled(
        quiet=getattr(args, "quiet", False),
        no_animation=getattr(args, "no_animation", False),
    )
    try:
        with _progress.StepProgress(cli.console, total=len(suite.tasks), description="Bench",
                                    animate=animate) as bar:
            document = run_suite(
                suite, config,
                records_path=(out_dir / "records.jsonl") if out_dir else None,
                on_task=lambda done, total, _record: bar.update(done, total),
            )
    except BenchRunError as exc:
        cli.print_error(str(exc))
        return 2
    except KeyboardInterrupt:
        cli.print_warning("Interrupted; the tasks that finished are in "
                          f"{out_dir / 'records.jsonl' if out_dir else 'no file (--no-save)'}")
        return 130

    table = document["table"]
    if out_dir is not None:
        path = write_run(document, out_dir)
        document["saved_to"] = str(path)
    render_table(
        columns=["Metric", "Per task (mean)", "Total"],
        rows=table_rows(table),
        console=None if json_mode else cli.console,
        title=f"effgen bench: {document['label']}",
        justify=["left", "right", "right"],
        file=sys.stderr if json_mode else None,
    )
    if out_dir is not None:
        cli.print(f"Saved: {out_dir / 'run.json'}")
    if json_mode:
        _emit_json(document)

    errors = int(table.get("errors") or 0)
    if errors > args.max_errors:
        cli.print_error(
            f"{errors} of {table['tasks']} tasks never produced a run (errors > --max-errors "
            f"{args.max_errors}); their scores are not a measurement of the model"
        )
        return 1
    return 0


def compare_rows(result: dict[str, Any]) -> list[list[str]]:
    """The comparison as ``[field, A, B, delta, band, reading]`` rows."""
    rows: list[list[str]] = []
    for row in [*result["rows"], *[dict(r, stop=True) for r in result["stop_reasons"]]]:
        field = row["field"]
        label = f"stop: {field}" if row.get("stop") else TABLE_LABELS.get(field, field)
        key = "tasks" if row.get("stop") else field
        if row.get("delta") is None:
            reading = row.get("note") or "n/a"
            rows.append([label, _fmt(key, row.get("a")), _fmt(key, row.get("b")), "-", "-", reading])
            continue
        band = row.get("band")
        if band is None:
            reading = "no band (fewer than 2 paired tasks)"
        else:
            reading = "within noise" if row["within_band"] else "beyond band"
        delta = row["delta"]
        if row.get("stop"):
            rows.append([label, f"{row['a']:.0f}", f"{row['b']:.0f}", f"{delta:+.0f}",
                         "-" if band is None else f"±{band:.1f}", reading])
        else:
            sign = "+" if delta >= 0 else ""
            rows.append([label, _fmt(key, row["a"]), _fmt(key, row["b"]),
                         f"{sign}{_fmt(key, delta)}",
                         "-" if band is None else f"±{_fmt(key, band)}", reading])
    return rows


def _handle_compare(args: Any, cli: Any) -> int:
    from effgen.bench import BenchRunError, compare_runs, read_run

    json_mode = bool(getattr(args, "output_json", False))
    if json_mode:
        cli._human_to_stderr = True
    try:
        a = read_run(args.a)
        b = read_run(args.b)
        result = compare_runs(a, b)
    except (BenchRunError, ValueError) as exc:
        cli.print_error(str(exc))
        return 2

    cli.print(f"A: {result['a']['label']}  ({result['a']['model']})")
    cli.print(f"B: {result['b']['label']}  ({result['b']['model']})")
    line = f"Paired on {result['paired']} tasks"
    if result["only_a"] or result["only_b"]:
        line += f" (A only {result['only_a']}, B only {result['only_b']}: left out)"
    cli.print(line)
    if not result["same_tasks"]:
        cli.print_warning("The runs' task sets or scorers differ; only the shared task ids are compared")
    render_table(
        columns=["Field (mean per task)", "A", "B", "Delta (B-A)", "Band (2 SE, ~95%)", "Reading"],
        rows=compare_rows(result),
        console=None if json_mode else cli.console,
        justify=["left", "right", "right", "right", "right", "left"],
        caption=("Band: two standard errors of the per-task differences between the two runs "
                 "(about 95% for one field). A delta within its band is not a measured "
                 "difference. Two runs of the same configuration still put about 1 field in 20 "
                 "outside its band by chance, so reading many fields at once, expect some."),
        file=sys.stderr if json_mode else None,
    )
    if json_mode:
        _emit_json(result)
    return 0


def _handle_bench_command(args: Any, cli: Any) -> int:
    """Handle ``effgen bench`` and its subcommands."""
    sub = getattr(args, "bench_command", None)
    if sub == "init":
        return _handle_init(args, cli)
    if sub == "run":
        return _handle_run(args, cli)
    if sub == "compare":
        return _handle_compare(args, cli)
    cli.print_error("Usage: effgen bench {init,run,compare} ... (see effgen bench --help)")
    return 2
