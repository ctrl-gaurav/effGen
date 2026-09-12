"""Presentation tests for the shared results-command output helpers.

The listing/results commands (``batch``, ``compare``, ``eval``, ``cost``,
``models``, ``prompts``) share one table/empty-state/glyph vocabulary. These
tests pin the two shared helpers (:func:`check_mark`, :func:`empty_state`) and
assert the byte-compatibility contract for the non-interactive paths: on a piped
or redirected stream the empty state and status glyphs are not drawn, so the
machine-readable bytes carry no ANSI colour and no terminal-only markers.

No live model calls: the helpers are driven directly and the command surfaces
are exercised with a forced-terminal / plain Rich console.
"""

from __future__ import annotations

import io

import pytest

from effgen.ui import tables as T
from effgen.ui.theme import get_console


class _AsciiStream(io.StringIO):
    """A stream that only encodes ASCII, to exercise the glyph fallback."""

    encoding = "ascii"


class _Utf8Stream(io.StringIO):
    encoding = "utf-8"


@pytest.fixture(autouse=True)
def _no_color_off(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    yield


# ---------------------------------------------------------------------------
# check_mark — the shared capability cell
# ---------------------------------------------------------------------------


def test_check_mark_present_is_success_glyph_on_utf8():
    assert T.check_mark(True, _Utf8Stream()) == "✓"


def test_check_mark_absent_is_empty():
    assert T.check_mark(False, _Utf8Stream()) == ""
    assert T.check_mark(False, _AsciiStream()) == ""


def test_check_mark_ascii_fallback_does_not_raise():
    cell = T.check_mark(True, _AsciiStream())
    assert cell == "+"  # ASCII stand-in for ✓
    cell.encode("ascii")  # must not raise on a non-UTF-8 console


# ---------------------------------------------------------------------------
# empty_state — the shared "nothing here yet" block (interactive only)
# ---------------------------------------------------------------------------


def test_empty_state_renders_title_message_and_hints():
    con = get_console(file=io.StringIO(), force_terminal=True, width=70)
    T.empty_state(con, title="No spend recorded",
                  message="No spend recorded yet.",
                  hints=["Run an agent to start tracking", "Set a cap"])
    out = con.file.getvalue()
    assert "No spend recorded" in out
    assert "No spend recorded yet." in out
    assert "Run an agent to start tracking" in out and "Set a cap" in out
    assert "→" in out  # arrow glyph leads each hint
    assert "\x1b" in out  # styled on a colour terminal


def test_empty_state_no_color_drops_colour():
    import os

    os.environ["NO_COLOR"] = "1"
    try:
        con = get_console(file=io.StringIO(), force_terminal=True, width=70)
        T.empty_state(con, title="Empty", message="nothing", hints=["do a thing"])
        out = con.file.getvalue()
    finally:
        os.environ.pop("NO_COLOR", None)
    # NO_COLOR drops every foreground colour (cyan heading, magenta arrow,
    # dim message reduce to structure); the text is intact.
    for colour in ("36m", "35m", "32m", "38;5", "38;2"):
        assert colour not in out
    assert "Empty" in out and "nothing" in out and "do a thing" in out


# ---------------------------------------------------------------------------
# console_is_interactive — the gate every command uses to keep piped output plain
# ---------------------------------------------------------------------------


def test_console_is_interactive_false_off_a_terminal():
    plain = get_console(file=io.StringIO(), width=70)  # not a terminal
    assert T.console_is_interactive(plain) is False


def test_console_is_interactive_true_on_a_forced_terminal():
    con = get_console(file=io.StringIO(), force_terminal=True, width=70)
    assert T.console_is_interactive(con) is True


# ---------------------------------------------------------------------------
# Non-TTY command contract: empty state / summary glyphs are gated off a pipe
# ---------------------------------------------------------------------------


def _run_cli(argv, monkeypatch):
    """Run the CLI in-process with a non-terminal stdout; return captured text."""
    import contextlib
    import sys

    from effgen.cli._main import main

    monkeypatch.setattr(sys, "argv", ["effgen", *argv])
    buf = io.StringIO()
    # A StringIO is not a terminal, so every command takes its piped/plain path.
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            main()
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buf.getvalue()


def test_eval_list_non_tty_is_plain(monkeypatch):
    code, out = _run_cli(["eval", "--suite", "list"], monkeypatch)
    assert code == 0
    assert "\x1b" not in out  # no ANSI on a piped stream
    assert "Available Evaluation Suites" in out


def test_prompts_list_non_tty_has_no_ansi(monkeypatch):
    code, out = _run_cli(["prompts", "list"], monkeypatch)
    assert code == 0
    assert "\x1b" not in out  # rich strips colour off a non-terminal


def test_batch_no_input_non_tty_error_has_no_ansi(monkeypatch):
    code, out = _run_cli(["batch"], monkeypatch)
    assert code == 1
    assert "\x1b" not in out


def test_cost_non_tty_has_no_ansi(monkeypatch):
    code, out = _run_cli(["cost"], monkeypatch)
    assert code == 0
    assert "\x1b" not in out


# ---------------------------------------------------------------------------
# public surface anchor
# ---------------------------------------------------------------------------


def test_public_surface_anchor_unchanged():
    import effgen

    # 1.0.0 added 17 names: the OpenAI-compatible adapter and its
    # BackendUnreachableError, the middleware surface, the tool-call records,
    # the compaction strategies, and the workflow checkpoint stores. 1.1.0
    # added 9 more: AgentThread, the Step protocol, and the seven kinds of
    # step a run's conversation is made of. Growing this number is a
    # deliberate act — update it in the same commit that widens the surface,
    # and say why.
    assert len(effgen.__all__) == 239
