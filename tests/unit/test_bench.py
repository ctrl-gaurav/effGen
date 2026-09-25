"""``effgen bench``: suite files, scorers, the run table, and the compare band.

Runs go to an OpenAI-protocol stub (``bench_stub``) that knows each task's
answer, calls the calculator once when tools are offered, and reports usage as a
fixed function of the request — so every number the table prints is known.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from effgen.bench import (
    TABLE_FIELDS,
    BenchConfig,
    BenchRunError,
    SuiteError,
    compare_runs,
    example_suite_text,
    load_suite,
    paired_band,
    read_run,
    run_suite,
    write_run,
)
from effgen.bench.scoring import make_scorer

from .bench_stub import serve

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "cli" / "bench.md"


@pytest.fixture(autouse=True)
def _private_home(tmp_path, monkeypatch):
    for name in ("EFFGEN_HOME", "EFFGEN_BENCH_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EFFGEN_COST_DB", str(tmp_path / "home" / "costs.sqlite"))
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "home" / "runs"))
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "home" / "sessions"))
    for name in ("EFFGEN_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)


def _write(tmp_path: Path, text: str, name: str = "suite.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _starter(tmp_path: Path) -> Path:
    return _write(tmp_path, example_suite_text(), "starter.yaml")


def _stub_for(suite, wrong=()):
    return serve([t.to_dict() for t in suite.tasks], wrong=set(wrong))


# ---------------------------------------------------------------------------
# The suite file
# ---------------------------------------------------------------------------

def test_the_shipped_starter_suite_loads(tmp_path):
    suite = load_suite(_starter(tmp_path))
    assert suite.name == "starter"
    assert len(suite.tasks) == suite.total_tasks == 6
    assert suite.tools == ["calculator"] and suite.scorer == "number" and suite.seed == 42
    assert suite.agent["max_iterations"] == 5


def test_n_and_seed_pick_the_same_tasks_every_time(tmp_path):
    path = _starter(tmp_path)
    first = [t.id for t in load_suite(path, n=3).tasks]
    again = [t.id for t in load_suite(path, n=3).tasks]
    other = [t.id for t in load_suite(path, n=3, seed=1).tasks]
    assert first == again and len(first) == 3
    assert other != first
    order = [t.id for t in load_suite(path).tasks]
    assert first == [i for i in order if i in first], "selected tasks keep file order"


def test_without_a_seed_n_takes_the_first_tasks(tmp_path):
    path = _write(tmp_path, "tasks:\n" + "".join(
        f"  - {{id: t{i}, input: q{i}, expected: {i}}}\n" for i in range(5)))
    assert [t.id for t in load_suite(path, n=2).tasks] == ["t0", "t1"]


@pytest.mark.parametrize(("text", "message"), [
    ("tasks: []\nscorrer: exact\n", "unknown key"),
    ("tasks:\n  - {id: a, input: q, expected: 1}\n  - {id: a, input: r, expected: 2}\n", "duplicate id"),
    ("tasks:\n  - {id: a, input: '', expected: 1}\n", "non-empty"),
    ("n: 5\ntasks:\n  - {id: a, input: q, expected: 1}\n", "holds only 1"),
    ("tasks:\n  - {id: a, input: q}\n", "missing on: a"),
    ("tasks:\n  - {id: a, input: q, expected: 1, answer: 2}\n", "unknown key"),
    ("agent: {temprature: 0.1}\ntasks:\n  - {id: a, input: q, expected: 1}\n", "unknown: temprature"),
    ("scorer: fuzzy\ntasks:\n  - {id: a, input: q, expected: 1}\n", "unknown scorer"),
    ("tasks: []\n", "no tasks"),
    ("n: 0\ntasks:\n  - {id: a, input: q, expected: 1}\n", "at least 1"),
])
def test_a_malformed_suite_is_refused_with_the_reason(tmp_path, text, message):
    with pytest.raises(SuiteError, match=message):
        load_suite(_write(tmp_path, text))


def test_tasks_can_come_from_a_jsonl_file_with_its_own_keys(tmp_path):
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"qid": "x1", "question": "Pick one", "gold": "B", "opts": "A. no\nB. yes"})
        + "\n", encoding="utf-8")
    path = _write(tmp_path, "tasks_file: data.jsonl\nscorer: choice\n"
                  "fields: {id: qid, input: question, expected: gold, context: opts}\n")
    task = load_suite(path).tasks[0]
    assert (task.id, task.expected) == ("x1", "B")
    assert task.prompt == "Pick one\n\nA. no\nB. yes"


def test_a_json_suite_loads_like_a_yaml_one(tmp_path):
    path = _write(tmp_path, json.dumps({"tasks": [{"input": "q", "expected": "a"}]}), "s.json")
    assert load_suite(path).tasks[0].id == "task-1"


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("scorer", "answer", "expected", "score"), [
    ("exact", "Paris", "paris", 1.0),
    ("exact", "Thinking...\nFinal answer: Paris.", ["Lyon", "Paris"], 1.0),
    ("exact", "Paris, France", "Paris", 0.0),
    ("contains", "It is PARIS for sure", "paris", 1.0),
    ("number", "So 1,250 x 38.\nAnswer: 47,500", 47500, 1.0),
    ("number", "Answer: 25.5 litres", "25.5", 1.0),
    ("number", "Answer: 25.6", 25.5, 0.0),
    ("number", "no number here", 3, 0.0),
    ("regex", "The code is AB-1234.", r"ab-\d{4}", 1.0),
    ("choice", "The answer is C.", "C", 1.0),
    ("choice", "Answer: (d)", "D", 1.0),
    ("choice", "B", "C", 0.0),
    ({"name": "number", "tolerance": 0.05}, "Answer: 102", 100, 1.0),
])
def test_builtin_scorers(scorer, answer, expected, score):
    assert make_scorer(scorer)(answer, expected, {}) == score


def test_a_scorer_of_the_callers_own_is_loaded_from_a_file(tmp_path):
    (tmp_path / "mine.py").write_text(
        "def score(answer, expected, task):\n    return answer.strip().endswith(str(expected))\n",
        encoding="utf-8")
    path = _write(tmp_path, "scorer: {callable: mine.py:score}\n"
                  "tasks:\n  - {id: a, input: q, expected: done}\n")
    suite = load_suite(path)
    scorer = make_scorer(suite.scorer, base_dir=tmp_path)
    assert scorer("all done", "done", {}) == 1.0 and scorer("not yet", "done", {}) == 0.0


# ---------------------------------------------------------------------------
# A run: every number in the table
# ---------------------------------------------------------------------------

def test_a_run_reports_the_ledger_table_the_endpoint_saw(tmp_path):
    suite = load_suite(_starter(tmp_path))
    server, url = _stub_for(suite, wrong={"fuel"})
    try:
        doc = run_suite(suite, BenchConfig(model="bench-stub", base_url=url, concurrency=3))
    finally:
        server.shutdown()
    requests = server.RequestHandlerClass.requests
    table = doc["table"]
    assert tuple(table) == TABLE_FIELDS
    assert table["tasks"] == 6 and table["errors"] == 0
    assert table["accuracy"] == pytest.approx(500 / 6)
    assert table["llm_calls"]["total"] == len(requests) == 12
    assert table["tool_calls"]["total"] == 6
    assert table["stop_reasons"] == {"final_answer": 6}
    # The stub's usage is a function of the request; the table carries its sum.
    def usage(request):
        text = "\n".join(str(m.get("content") or "") for m in request["messages"])
        return len(text) // 4 + 1
    prompt = sum(usage(r) for r in requests)
    assert table["prompt_tokens"]["total"] == prompt
    assert table["completion_tokens"]["total"] == 9 * 12
    assert table["cached_input_tokens"]["total"] == sum(usage(r) // 2 for r in requests)
    assert table["cost_usd"] is None and table["unpriced_calls"] == 12
    for key in ("wall_s", "framework_s", "model_wait_s"):
        assert table[key]["total"] > 0
    by_id = {r["id"]: r for r in doc["tasks"]}
    assert by_id["fuel"]["score"] == 0.0 and by_id["crates"]["score"] == 1.0
    assert [r["id"] for r in doc["tasks"]] == [t.id for t in suite.tasks]
    assert doc["suite"]["fingerprint"] == suite.fingerprint()
    assert doc["config"]["seed"] == 42 and doc["state"] == "complete"


def test_a_run_whose_endpoint_is_down_counts_every_task_as_an_error(tmp_path):
    suite = load_suite(_starter(tmp_path), n=2)
    server, url = _stub_for(suite)
    server.shutdown()
    server.server_close()
    doc = run_suite(suite, BenchConfig(model="bench-stub", base_url=url))
    assert doc["table"]["errors"] == 2
    assert all(r["error"] for r in doc["tasks"])


def test_an_unknown_tool_stops_the_run_before_any_task(tmp_path):
    path = _write(tmp_path, "tools: [no_such_tool]\ntasks:\n  - {id: a, input: q, expected: 1}\n")
    with pytest.raises(BenchRunError, match="unknown tool 'no_such_tool'"):
        run_suite(load_suite(path), BenchConfig(model="bench-stub", base_url="http://127.0.0.1:9/v1"))


def test_read_run_refuses_what_is_not_a_complete_run(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "run.json").write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(BenchRunError, match="not an effgen bench run"):
        read_run(tmp_path / "a")
    write_run({"schema": "effgen.bench.run/1", "state": "interrupted"}, tmp_path / "b")
    with pytest.raises(BenchRunError, match="did not complete"):
        read_run(tmp_path / "b")


# ---------------------------------------------------------------------------
# Compare: the band comes from the runs' own disagreement
# ---------------------------------------------------------------------------

def _doc(scores, calls=None, stops=None, fingerprint="f"):
    tasks = []
    for i, s in enumerate(scores):
        tasks.append({"id": f"t{i}", "score": s, "error": None,
                      "llm_calls": (calls or [1] * len(scores))[i], "tool_calls": 0,
                      "prompt_tokens": 10, "completion_tokens": 2, "cached_input_tokens": 0,
                      "wall_s": 1.0, "framework_s": 0.1, "model_wait_s": 0.9, "tool_wait_s": 0.0,
                      "cost_usd": None,
                      "stop_reason": (stops or ["final_answer"] * len(scores))[i]})
    return {"schema": "effgen.bench.run/1", "state": "complete", "label": "x",
            "suite": {"name": "s", "fingerprint": fingerprint}, "config": {"model": "m"},
            "tasks": tasks}


def test_the_band_is_two_standard_errors_of_the_per_task_differences():
    a = _doc([1, 1, 0, 0, 1, 0, 1, 1], calls=[1, 2, 3, 4, 1, 2, 3, 4])
    b = _doc([1, 0, 1, 0, 1, 1, 1, 1], calls=[2, 2, 3, 5, 1, 2, 4, 4])
    result = compare_runs(a, b)
    rows = {r["field"]: r for r in result["rows"]}
    d = [0, -100, 100, 0, 0, 100, 0, 0]
    mean = sum(d) / 8
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 7)
    assert rows["accuracy"]["delta"] == pytest.approx(12.5)
    assert rows["accuracy"]["band"] == pytest.approx(2 * sd / math.sqrt(8))
    assert rows["accuracy"]["within_band"] is True
    calls = [1, 0, 0, 1, 0, 0, 1, 0]
    cm = sum(calls) / 8
    csd = math.sqrt(sum((x - cm) ** 2 for x in calls) / 7)
    assert rows["llm_calls"]["delta"] == pytest.approx(0.375)
    assert rows["llm_calls"]["band"] == pytest.approx(2 * csd / math.sqrt(8))
    assert result["paired"] == 8


def test_identical_runs_have_a_zero_band_and_every_delta_within_it():
    result = compare_runs(_doc([1, 0, 1]), _doc([1, 0, 1]))
    for row in result["rows"] + result["stop_reasons"]:
        if row["delta"] is not None:
            assert row["delta"] == 0 and row["band"] == 0 and row["within_band"] is True


def test_a_consistent_shift_lands_beyond_the_band():
    a = _doc([1] * 10, calls=[1] * 10)
    b = _doc([1] * 10, calls=[3, 3, 3, 3, 3, 3, 3, 3, 3, 4])
    row = {r["field"]: r for r in compare_runs(a, b)["rows"]}["llm_calls"]
    assert row["delta"] == pytest.approx(2.1) and row["within_band"] is False


def test_stop_reasons_are_compared_as_counts_with_a_count_band():
    a = _doc([1, 1, 1, 1], stops=["final_answer"] * 4)
    b = _doc([1, 1, 1, 0], stops=["final_answer"] * 3 + ["max_iterations_partial"])
    stops = {r["field"]: r for r in compare_runs(a, b)["stop_reasons"]}
    assert stops["max_iterations_partial"]["delta"] == pytest.approx(1.0)
    assert stops["max_iterations_partial"]["band"] == pytest.approx(2 * 0.5 * 2)


def test_runs_that_share_no_task_are_refused():
    b = _doc([1])
    b["tasks"][0]["id"] = "other"
    with pytest.raises(ValueError, match="share no task"):
        compare_runs(_doc([1]), b)


def test_a_single_paired_task_has_no_band():
    assert paired_band([1.0]) is None
    row = compare_runs(_doc([1]), _doc([0]))["rows"][0]
    assert row["band"] is None and row["within_band"] is None


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def _cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "effgen.cli", *args], cwd=cwd, capture_output=True, text=True,
        timeout=300, env=_cli_env(),
    )


def _cli_env():
    import os
    env = dict(os.environ)
    env["EFFGEN_NO_DOTENV"] = "1"
    env["NO_COLOR"] = "1"
    return env


def test_init_writes_the_starter_and_will_not_overwrite(tmp_path):
    first = _cli("bench", "init", str(tmp_path / "s.yaml"))
    assert first.returncode == 0, first.stderr
    assert (tmp_path / "s.yaml").read_text(encoding="utf-8") == example_suite_text()
    again = _cli("bench", "init", str(tmp_path / "s.yaml"))
    assert again.returncode == 1 and "exists" in (again.stdout + again.stderr)


def test_run_json_carries_every_field_the_table_prints(tmp_path):
    from effgen.cli.commands.bench import TABLE_LABELS, table_rows

    path = _starter(tmp_path)
    suite = load_suite(path, n=3)
    server, url = _stub_for(suite)
    try:
        proc = _cli("bench", "run", str(path), "--n", "3", "-m", "bench-stub", "--base-url", url,
                    "--out", str(tmp_path / "run"), "--json", "--no-animation")
    finally:
        server.shutdown()
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = json.loads(proc.stdout)
    assert tuple(doc["table"]) == TABLE_FIELDS == tuple(TABLE_LABELS)
    for label in TABLE_LABELS.values():
        assert label in proc.stderr, f"the text table is missing {label!r}"
    assert len(table_rows(doc["table"])) == len(TABLE_FIELDS)
    saved = json.loads((tmp_path / "run" / "run.json").read_text())
    assert saved["table"] == doc["table"]
    # The printed document is the saved one plus where it was saved.
    assert Path(doc.pop("saved_to")) == (tmp_path / "run" / "run.json")
    assert "saved_to" not in saved and saved == doc
    assert len((tmp_path / "run" / "records.jsonl").read_text().splitlines()) == 3


def test_run_exits_nonzero_when_tasks_never_reached_the_model(tmp_path):
    path = _starter(tmp_path)
    proc = _cli("bench", "run", str(path), "--n", "2", "-m", "bench-stub",
                "--base-url", "http://127.0.0.1:9/v1", "--no-save", "--no-animation")
    assert proc.returncode == 1
    assert "never produced a run" in proc.stdout + proc.stderr


def test_compare_prints_a_band_beside_every_delta(tmp_path):
    write_run(_doc([1, 0, 1, 1], calls=[1, 2, 1, 1]), tmp_path / "a")
    write_run(_doc([1, 1, 1, 0], calls=[1, 2, 2, 1]), tmp_path / "b")
    proc = _cli("bench", "compare", str(tmp_path / "a"), str(tmp_path / "b"))
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if re.search(r"[+-]\d", ln)]
    assert lines, proc.stdout
    for line in lines:
        assert "±" in line and ("within noise" in line or "beyond band" in line), line
    as_json = json.loads(_cli("bench", "compare", str(tmp_path / "a"), str(tmp_path / "b"),
                              "--json").stdout)
    assert all("band" in r for r in as_json["rows"] + as_json["stop_reasons"])


def test_the_suite_in_the_documentation_runs_as_written(tmp_path):
    text = DOCS.read_text(encoding="utf-8")
    fence = re.search(r"Save this as `(?P<name>[^`]+)`:\s*```yaml\n(?P<body>.*?)```", text, re.S)
    assert fence, "the quick-start suite fence is missing from the documentation"
    path = _write(tmp_path, fence["body"], fence["name"])
    suite = load_suite(path)
    server, url = _stub_for(suite)
    try:
        proc = _cli("bench", "run", fence["name"], "--model", "bench-stub", "--base-url", url,
                    "--out", "runs/a", "--no-animation", cwd=tmp_path)
    finally:
        server.shutdown()
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = read_run(tmp_path / "runs" / "a")
    assert doc["table"]["tasks"] == 3 and doc["table"]["accuracy"] == 100.0


def test_an_interrupted_run_exits_at_once_with_130(tmp_path):
    """Ctrl-C returns promptly even while every worker waits on a slow endpoint."""
    import os
    import signal
    import time

    path = _starter(tmp_path)
    suite = load_suite(path)
    server, url = serve([t.to_dict() for t in suite.tasks], delay_s=60.0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "effgen.cli", "bench", "run", str(path), "-m", "bench-stub",
             "--base-url", url, "--out", str(tmp_path / "run"), "--no-animation"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_cli_env(),
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        )
        deadline = time.time() + 60
        while time.time() < deadline and not server.RequestHandlerClass.requests:
            time.sleep(0.2)
        assert server.RequestHandlerClass.requests, "the run never reached the endpoint"
        started = time.time()
        os.kill(proc.pid, signal.SIGINT)
        out, _ = proc.communicate(timeout=30)
        took = time.time() - started
    finally:
        server.shutdown()
    assert proc.returncode == 130, out[-2000:]
    assert took < 15, f"the interrupted run took {took:.1f}s to exit"
    assert "Interrupted" in out
