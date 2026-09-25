"""Measure what an agent costs on tasks you choose: ``effgen bench``.

A **suite** is a small YAML or JSON file — tasks, tools, a scorer, ``n`` and a
seed (see :mod:`effgen.bench.suite`). :func:`run_suite` runs it against a
configured model and returns a run document whose ``table`` holds accuracy,
task count, errors, LLM calls, tool calls, prompt, completion and cached-input
tokens, wall, framework, model and tool time, cost and the stop-reason
distribution, all read from each run's ledger. :func:`compare_runs` pairs two
runs task by task and puts a noise band beside every delta.

The command line is ``effgen bench init | run | compare``; this package is what
it calls::

    from effgen.bench import BenchConfig, load_suite, run_suite

    suite = load_suite("suite.yaml")
    doc = run_suite(suite, BenchConfig(model="my-model",
                                       base_url="http://127.0.0.1:8000/v1"))
    print(doc["table"]["accuracy"], doc["table"]["llm_calls"]["mean"])
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from .compare import COMPARE_SCHEMA, compare_runs, paired_band
from .runner import (
    RUN_SCHEMA,
    TABLE_FIELDS,
    BenchConfig,
    BenchRunError,
    build_table,
    read_run,
    run_suite,
    write_run,
)
from .scoring import BUILTIN_SCORERS, make_scorer
from .suite import BenchSuite, BenchTask, SuiteError, load_suite

__all__ = [
    "BUILTIN_SCORERS",
    "COMPARE_SCHEMA",
    "RUN_SCHEMA",
    "TABLE_FIELDS",
    "BenchConfig",
    "BenchRunError",
    "BenchSuite",
    "BenchTask",
    "SuiteError",
    "build_table",
    "compare_runs",
    "default_runs_dir",
    "example_suite_text",
    "load_suite",
    "make_scorer",
    "paired_band",
    "read_run",
    "run_suite",
    "write_run",
]


def example_suite_text() -> str:
    """The starter suite shipped with effGen, as the text ``effgen bench init`` writes."""
    return (resources.files(__package__) / "examples" / "starter.yaml").read_text(encoding="utf-8")


def default_runs_dir() -> Path:
    """Where ``effgen bench run`` saves runs when ``--out`` is not given.

    ``EFFGEN_BENCH_DIR``, else ``$EFFGEN_HOME/bench``, else ``~/.effgen/bench``.
    """
    import os

    explicit = os.environ.get("EFFGEN_BENCH_DIR")
    if explicit:
        return Path(os.path.expanduser(explicit)).absolute()
    home = os.environ.get("EFFGEN_HOME")
    if home:
        return Path(os.path.expanduser(home)).absolute() / "bench"
    return Path(os.path.expanduser("~/.effgen/bench"))
