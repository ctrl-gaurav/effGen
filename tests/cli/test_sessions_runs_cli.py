"""Tests for the `effgen sessions` and `effgen runs` review commands."""

from __future__ import annotations

import argparse
import json

import pytest

from effgen.cli._main import (
    _handle_runs_command,
    _handle_sessions_command,
    create_parser,
)
from effgen.core.session import Session, SessionManager
from effgen.observability import run_log
from tests._harness.offline_assets import assert_self_contained

MARKUP_TEXT = "check [bold]v2[/bold] and [REDACTED] ref [1]"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Keep sessions and run history inside the test's own directory."""
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "state" / "sessions"))
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "state" / "runs"))
    run_log.clear()
    yield
    run_log.clear()


def _cli():
    from effgen.cli import CLIInterface

    cli = CLIInterface()
    cli.console = None  # plain output, no Rich markup rendering
    return cli


def _rich_cli():
    """A CLI with its Rich console live — the path that parses console markup.

    Output routed through that console drops ``[bold]``-style spans, so the
    commands that emit stored data must not use it. Tests that assert content
    survives intact are only meaningful against this console.
    """
    from effgen.cli import CLIInterface

    cli = CLIInterface()
    assert cli.console is not None, "Rich console unavailable; markup path untested"
    return cli


def _args(**kwargs):
    defaults = {"output_json": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_session(session_id="support-1", *, content=MARKUP_TEXT, cost=0.0002):
    session = Session(session_id=session_id, agent_name="cli-agent")
    session.add_message("user", content, model="gpt-5-nano", run_id="r1")
    session.add_message(
        "assistant", "answer " + content,
        model="gpt-5-nano", run_id="r1", cost_usd=cost, tokens_used=42, latency_ms=120.0,
    )
    session.metadata["model"] = "gpt-5-nano"
    session.save()
    return session


# ---------------------------------------------------------------- export
def test_export_text_is_byte_identical_to_the_store(capsys):
    _make_session()
    mgr = SessionManager()

    assert _handle_sessions_command(
        _args(session_command="export", session_id="support-1", format="text"),
        _rich_cli(),
    ) == 0

    printed = capsys.readouterr().out
    assert printed.rstrip("\n") == mgr.export("support-1", format="text").rstrip("\n")
    assert "[user]" in printed
    assert "[bold]v2[/bold]" in printed
    assert "[REDACTED]" in printed
    assert "[1]" in printed


def test_export_json_round_trips_message_content(capsys):
    _make_session()

    assert _handle_sessions_command(
        _args(session_command="export", session_id="support-1", format="json"),
        _rich_cli(),
    ) == 0

    data = json.loads(capsys.readouterr().out)
    assert data["messages"][0]["content"] == MARKUP_TEXT


def test_console_markup_path_would_damage_the_record(capsys):
    """The renderer these commands avoid does drop bracketed content.

    Guards the fix: if stored data were ever routed back through the console,
    the export assertions above would fail rather than quietly pass.
    """
    _rich_cli().print("[user] check [bold]v2[/bold]")

    damaged = capsys.readouterr().out
    assert "[bold]" not in damaged
    assert "[user]" not in damaged


def test_show_writes_stored_content_verbatim(capsys):
    """Reading a conversation must not damage it either."""
    _make_session()

    _handle_sessions_command(
        _args(session_command="show", session_id="support-1", last=None), _rich_cli()
    )

    out = capsys.readouterr().out
    assert MARKUP_TEXT in out


def test_runs_show_writes_stored_task_verbatim(capsys):
    run_log.record_run(model="m", run_id="r-markup", task=MARKUP_TEXT,
                       output="[green]done[/green]")

    _handle_runs_command(_args(runs_command="show", run_id="r-markup"), _rich_cli())

    out = capsys.readouterr().out
    assert MARKUP_TEXT in out
    assert "[green]done[/green]" in out


# ---------------------------------------------------------------- show
def test_show_renders_speaker_labels_and_per_turn_cost(capsys):
    _make_session()

    assert _handle_sessions_command(
        _args(session_command="show", session_id="support-1", last=None), _cli()
    ) == 0

    out = capsys.readouterr().out
    assert "user" in out and "assistant" in out
    assert MARKUP_TEXT in out
    assert "gpt-5-nano" in out
    assert "$0.0002" in out
    assert "chat --session-id support-1" in out


def test_show_last_limits_the_messages(capsys):
    _make_session()

    _handle_sessions_command(
        _args(session_command="show", session_id="support-1", last=1), _cli()
    )

    out = capsys.readouterr().out
    assert "showing last 1" in out
    assert out.count("--- ") == 1


def test_show_json_is_pure_json(capsys):
    _make_session()

    _handle_sessions_command(
        _args(session_command="show", session_id="support-1", last=1, output_json=True),
        _cli(),
    )

    data = json.loads(capsys.readouterr().out)
    assert len(data["messages"]) == 1


def test_show_missing_session_exits_one(capsys):
    assert _handle_sessions_command(
        _args(session_command="show", session_id="absent", last=None), _cli()
    ) == 1
    assert "sessions list" in capsys.readouterr().out


# ---------------------------------------------------------------- list
def _list_args(**kwargs):
    base = {
        "session_command": "list", "search": None, "since": None, "until": None,
        "model": None, "limit": 50,
    }
    base.update(kwargs)
    return _args(**base)


def test_list_shows_model_and_cost(capsys):
    _make_session()

    _handle_sessions_command(_list_args(), _cli())

    out = capsys.readouterr().out
    assert "gpt-5-nano" in out
    assert "$0.0002" in out


def test_list_filters_by_search_model_and_date(capsys):
    _make_session("support-1")
    _make_session("other-2", content="unrelated topic", cost=0.0)

    _handle_sessions_command(_list_args(search="unrelated"), _cli())
    out = capsys.readouterr().out
    assert "other-2" in out and "support-1" not in out

    _handle_sessions_command(_list_args(model="gpt-5-nano"), _cli())
    out = capsys.readouterr().out
    assert "support-1" in out and "other-2" in out

    _handle_sessions_command(_list_args(until="2000-01-01"), _cli())
    out = capsys.readouterr().out
    assert "support-1" not in out and "other-2" not in out


def test_list_names_unreadable_session_files(tmp_path, capsys):
    _make_session()
    (tmp_path / "state" / "sessions" / "broken.json").write_text("{not json")

    _handle_sessions_command(_list_args(), _cli())

    out = capsys.readouterr().out
    assert "broken.json" in out
    assert "could not be read" in out


def test_list_json_reports_unreadable_files(tmp_path, capsys):
    _make_session()
    (tmp_path / "state" / "sessions" / "broken.json").write_text("{not json")

    _handle_sessions_command(_list_args(output_json=True), _cli())

    data = json.loads(capsys.readouterr().out)
    assert [u["file"] for u in data["unreadable"]] == ["broken.json"]
    assert len(data["sessions"]) == 1


def test_browse_is_non_interactive_when_piped(capsys, monkeypatch):
    _make_session()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)

    assert _handle_sessions_command(
        _args(session_command="browse", limit=20), _cli()
    ) == 0
    assert "support-1" in capsys.readouterr().out


# ---------------------------------------------------------------- runs
def _runs_args(**kwargs):
    base = {
        "runs_command": "list", "status": None, "model": None, "search": None,
        "session_filter": None, "since": None, "until": None, "limit": 20,
        "output_json": False,
    }
    base.update(kwargs)
    return _args(**base)


def test_runs_list_shows_recorded_runs(capsys):
    run_log.record_run(model="gpt-5-nano", run_id="r-ok", task="refund policy",
                       cost_usd=0.0001, duration_s=1.5, session_id="support-1")

    assert _handle_runs_command(_runs_args(), _cli()) == 0

    out = capsys.readouterr().out
    assert "r-ok" in out and "gpt-5-nano" in out and "refund policy" in out


def test_runs_list_status_failed_selects_errors(capsys):
    run_log.record_run(model="m", run_id="r-ok", task="fine")
    run_log.record_run(model="m", run_id="r-bad", task="broke", error="boom")

    _handle_runs_command(_runs_args(status="failed"), _cli())

    out = capsys.readouterr().out
    assert "r-bad" in out and "r-ok" not in out


def test_runs_list_json_is_pure_json(capsys):
    run_log.record_run(model="m", run_id="r1", task="t")

    _handle_runs_command(_runs_args(output_json=True), _cli())

    data = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in data["runs"]] == ["r1"]
    assert data["persisted"] is True


def test_runs_show_prints_task_answer_and_session_link(capsys):
    run_log.record_run(model="gpt-5-nano", run_id="r1", task=MARKUP_TEXT,
                       output="the answer", session_id="support-1", cost_usd=0.0003)

    assert _handle_runs_command(
        _args(runs_command="show", run_id="r1"), _cli()
    ) == 0

    out = capsys.readouterr().out
    assert MARKUP_TEXT in out  # written verbatim, brackets intact
    assert "the answer" in out
    assert "sessions show support-1" in out


def test_sub_cent_cost_is_not_shown_as_zero():
    """A priced run must stay distinguishable from a free one."""
    from effgen.cli._main import _fmt_cost

    assert _fmt_cost(0.000027) == "$0.000027"
    assert _fmt_cost(0.0) == "$0.00"
    assert _fmt_cost(None) == "—"
    assert _fmt_cost(0.0012) == "$0.0012"
    assert _fmt_cost(2.5) == "$2.50"


def test_runs_list_distinguishes_no_matches_from_no_history(capsys):
    # Nothing recorded at all: the hint is to go run an agent.
    _handle_runs_command(_runs_args(), _cli())
    assert "No runs recorded yet" in capsys.readouterr().out

    # Runs exist but the filters exclude them: the hint is to widen them.
    run_log.record_run(model="m", run_id="r1", task="t")
    _handle_runs_command(_runs_args(search="nothing-matches-this"), _cli())
    assert "No runs match those filters" in capsys.readouterr().out


def test_runs_show_missing_run_exits_one(capsys):
    assert _handle_runs_command(
        _args(runs_command="show", run_id="nope"), _cli()
    ) == 1
    assert "runs list" in capsys.readouterr().out


def test_runs_and_sessions_subcommands_are_registered():
    parser = create_parser()
    args = parser.parse_args(["runs", "list", "--status", "failed"])
    assert args.command == "runs" and args.status == "failed"

    args = parser.parse_args(["sessions", "show", "abc", "--last", "3"])
    assert args.session_command == "show" and args.last == 3


def test_runs_show_card_writes_a_labeled_summary_card(tmp_path, capsys):
    run_log.record_run(model="openai/gpt-oss-20b", provider="groq", run_id="r-card",
                       task="Reply with exactly: OK", output="OK",
                       input_tokens=313, output_tokens=2, duration_s=0.27,
                       cost_usd=1.6e-05)
    out = tmp_path / "card.html"

    assert _handle_runs_command(
        _args(runs_command="show", run_id="r-card", card=str(out)), _cli()
    ) == 0

    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "Reply with exactly: OK" in html and "openai/gpt-oss-20b" in html
    # The card states the limits of what stored history holds.
    assert "truncated answer and no step trace" in html
    # Self-contained: nothing is fetched when the file is opened.
    assert_self_contained(html, "the run card")
    assert "Summary card written to" in capsys.readouterr().out


def test_runs_show_card_reports_an_unwritable_path(tmp_path, capsys):
    run_log.record_run(model="m", run_id="r-card2", task="t", output="a")
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory", encoding="utf-8")

    assert _handle_runs_command(
        _args(runs_command="show", run_id="r-card2",
              card=str(blocked / "card.html")), _cli()
    ) == 1
    assert "--card" in capsys.readouterr().out
