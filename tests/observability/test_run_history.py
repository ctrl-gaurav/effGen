"""Tests for the durable agent run history store."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from effgen.observability import run_log


@pytest.fixture(autouse=True)
def history_dir(tmp_path, monkeypatch):
    """Point run history at a private directory and start from an empty ring."""
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("EFFGEN_RUN_HISTORY", raising=False)
    run_log.clear()
    yield tmp_path / "runs"
    run_log.clear()


def test_record_writes_a_daily_jsonl_file(history_dir):
    run_log.record_run(model="gpt-5-nano", run_id="abc123", task="hello", cost_usd=0.001)

    files = list(history_dir.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].stem == datetime.now().astimezone().date().isoformat()
    record = json.loads(files[0].read_text().strip())
    assert record["run_id"] == "abc123"
    assert record["model"] == "gpt-5-nano"
    assert record["task"] == "hello"
    assert record["status"] == "ok"


def test_records_survive_a_cleared_in_memory_ring(history_dir):
    """A restart drops the ring; the files remain the source of truth."""
    run_log.record_run(model="m1", run_id="r1", task="first")
    run_log.clear()

    runs = run_log.read_runs(limit=10)
    assert [r["run_id"] for r in runs] == ["r1"]


def test_timestamp_is_iso8601_with_an_explicit_offset(history_dir):
    run_log.record_run(model="m1", run_id="r1")
    ts = run_log.read_runs(limit=1)[0]["ts"]

    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    # Sortable across midnight: the date is part of the value.
    assert ts[:10] == datetime.now().astimezone().date().isoformat()


def test_failed_run_is_recorded_with_error_status(history_dir):
    run_log.record_run(model="bogus", run_id="r-bad", error="model not found")

    failed = run_log.read_runs(limit=10, status="error")
    assert [r["run_id"] for r in failed] == ["r-bad"]
    assert failed[0]["error"] == "model not found"
    assert run_log.read_runs(limit=10, status="ok") == []


def test_filters_by_model_session_and_search(history_dir):
    run_log.record_run(model="gpt-5-nano", run_id="r1", task="refund policy",
                       session_id="s-1")
    run_log.record_run(model="openai/gpt-oss-20b", run_id="r2", task="escalation",
                       session_id="s-2")

    assert [r["run_id"] for r in run_log.read_runs(model="gpt-oss")] == ["r2"]
    assert [r["run_id"] for r in run_log.read_runs(session_id="s-1")] == ["r1"]
    assert [r["run_id"] for r in run_log.read_runs(search="refund")] == ["r1"]
    assert [r["run_id"] for r in run_log.read_runs(search="r2")] == ["r2"]


def test_date_filters_bound_the_result(history_dir):
    run_log.record_run(model="m", run_id="today")
    today = datetime.now().astimezone().date()
    old = (today - timedelta(days=3)).isoformat()
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"{old}.jsonl").write_text(
        json.dumps({"ts": f"{old}T09:00:00+00:00", "run_id": "old", "model": "m",
                    "status": "ok"}) + "\n"
    )
    run_log.clear()

    assert {r["run_id"] for r in run_log.read_runs()} == {"today", "old"}
    assert [r["run_id"] for r in run_log.read_runs(since=today.isoformat())] == ["today"]
    assert [r["run_id"] for r in run_log.read_runs(until=old)] == ["old"]


def test_get_run_returns_the_matching_record(history_dir):
    run_log.record_run(model="m", run_id="find-me", task="t")

    assert run_log.get_run("find-me")["task"] == "t"
    assert run_log.get_run("absent") is None


def test_duplicate_run_ids_are_returned_once(history_dir):
    """The ring and the file hold the same record; a reader sees one."""
    run_log.record_run(model="m", run_id="dup")

    assert len(run_log.read_runs(limit=10)) == 1


def test_a_run_without_an_id_is_also_returned_once(history_dir):
    """De-duplication cannot rely on a run id that a caller may not supply."""
    run_log.record_run(model="m", task="no id here")

    assert len(run_log.read_runs(limit=10)) == 1


def test_distinct_runs_without_ids_are_both_kept(history_dir):
    run_log.record_run(model="m", task="first")
    run_log.record_run(model="m", task="second")

    assert {r["task"] for r in run_log.read_runs(limit=10)} == {"first", "second"}


def test_task_and_output_previews_are_bounded(history_dir):
    long_text = "x" * (run_log.PREVIEW_CHARS * 3)
    run_log.record_run(model="m", run_id="r", task=long_text, output=long_text)

    record = run_log.read_runs(limit=1)[0]
    assert len(record["task"]) == run_log.PREVIEW_CHARS
    assert len(record["output"]) == run_log.PREVIEW_CHARS


def test_history_can_be_kept_in_memory_only(history_dir, monkeypatch):
    monkeypatch.setenv("EFFGEN_RUN_HISTORY", "0")
    run_log.record_run(model="m", run_id="r1")

    assert not history_dir.exists() or not list(history_dir.glob("*.jsonl"))
    assert [r["run_id"] for r in run_log.read_runs()] == ["r1"]
    assert run_log.history_enabled() is False


def test_retention_prunes_stale_files_on_write(history_dir):
    history_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().astimezone().date()
    stale = (today - timedelta(days=45)).isoformat()
    (history_dir / f"{stale}.jsonl").write_text("{}\n")

    run_log.record_run(model="m", run_id="fresh")

    assert [p.stem for p in history_dir.glob("*.jsonl")] == [today.isoformat()]


def test_cleanup_removes_files_outside_the_requested_window(history_dir, monkeypatch):
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_MAX_DAYS", "365")
    history_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().astimezone().date()
    stale = (today - timedelta(days=45)).isoformat()
    (history_dir / f"{stale}.jsonl").write_text("{}\n")
    run_log.record_run(model="m", run_id="fresh")

    assert run_log.cleanup_runs(older_than_days=30) == 1
    assert [p.stem for p in history_dir.glob("*.jsonl")] == [today.isoformat()]


def test_a_damaged_line_does_not_hide_the_rest(history_dir):
    run_log.record_run(model="m", run_id="good")
    day = datetime.now().astimezone().date().isoformat()
    with open(history_dir / f"{day}.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    run_log.clear()

    assert [r["run_id"] for r in run_log.read_runs()] == ["good"]


def test_history_dir_follows_effgen_home(tmp_path, monkeypatch):
    monkeypatch.delenv("EFFGEN_RUN_HISTORY_DIR", raising=False)
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "state"))

    assert run_log.history_dir() == (tmp_path / "state" / "runs").absolute()


def _agent_with(model):
    from effgen.core.agent import Agent, AgentConfig

    return Agent(config=AgentConfig(name="rec", model=model))


def test_provider_is_taken_from_the_serving_adapter(history_dir):
    """A ``provider:model`` id names its provider without the caller doing so."""
    from tests.fixtures.mock_models import MockModel

    class ServedModel(MockModel):
        def get_metadata(self):
            return {**super().get_metadata(), "provider": "groq"}

    agent = _agent_with(ServedModel(["ok"], model_name="openai/gpt-oss-20b"))
    agent.run("hello")

    record = run_log.read_runs()[0]
    assert record["provider"] == "groq"


def test_a_local_engine_run_records_no_provider(history_dir):
    """Local engines are not served by a provider, so the field stays unset."""
    from tests.fixtures.mock_models import MockModel

    agent = _agent_with(MockModel(["ok"], model_name="Qwen/Qwen2.5-1.5B-Instruct"))
    agent.run("hello")

    record = run_log.read_runs()[0]
    assert record["provider"] is None
