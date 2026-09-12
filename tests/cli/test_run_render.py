"""Presentation tests for the shared run/chat output layer.

These pin the byte-compatibility contract for the non-interactive output paths
(piped, ``NO_COLOR``, non-UTF-8) and verify the shared streaming/answer/summary
helpers behave the same way on every surface. No live model calls: the streaming
renderer is driven with a fixed token iterator and the answer/summary helpers
with a small fake response, so the checks are deterministic.

The interactive live-markdown rendering itself is proven end-to-end with real
models; here it is exercised against a forced-terminal Rich console.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import pytest

from effgen.cli import progress as P
from effgen.ui import render as R
from effgen.ui.theme import get_console

TOKENS = ["# Fruits\n", "\n", "- apple\n", "- banana\n"]


@pytest.fixture(autouse=True)
def _no_color_off(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    yield


class _AsciiStream(io.StringIO):
    """A stream that only encodes ASCII, to exercise the glyph fallback."""

    encoding = "ascii"


# ---------------------------------------------------------------------------
# stream_answer — raw passthrough (the piped / non-TTY contract)
# ---------------------------------------------------------------------------


def test_stream_answer_raw_is_byte_clean(capsys):
    text = P.stream_answer(None, iter(TOKENS), animate=False, interactive=False,
                           trailing_newline=True)
    out = capsys.readouterr().out
    assert text == "".join(TOKENS)
    assert out == "".join(TOKENS) + "\n"  # exact tokens + one trailing newline
    assert "\x1b" not in out  # zero ANSI on the piped path


def test_stream_answer_raw_no_trailing_newline():
    buf = io.StringIO()
    import sys
    old = sys.stdout
    sys.stdout = buf
    try:
        P.stream_answer(None, iter(["a", "b"]), animate=False, interactive=False,
                        trailing_newline=False)
    finally:
        sys.stdout = old
    assert buf.getvalue() == "ab"  # caller adds its own newline


def test_stream_answer_skips_empty_tokens(capsys):
    P.stream_answer(None, iter(["x", "", None, "y"]), animate=False, interactive=False,
                    trailing_newline=True)  # type: ignore[list-item]
    assert capsys.readouterr().out == "xy\n"


# ---------------------------------------------------------------------------
# stream_answer — animated live-markdown region (interactive colour TTY)
# ---------------------------------------------------------------------------


def test_stream_answer_animate_renders_markdown_live():
    con = get_console(file=io.StringIO(), force_terminal=True, width=60)
    text = P.stream_answer(con, iter(TOKENS), animate=True, interactive=True,
                           label="assistant")
    out = con.file.getvalue()
    assert text == "".join(TOKENS)  # full text returned for persistence
    assert "\x1b" in out  # styled output on a colour terminal
    assert "assistant" in out  # the label header
    assert "Fruits" in out and "apple" in out and "banana" in out
    # The heading rendered as markdown (bold + underline), not raw '#'.
    assert "\x1b[1;4m" in out


def test_stream_answer_plain_interactive_renders_once():
    # Interactive but not animating (e.g. NO_COLOR): collect, render one block.
    con = get_console(file=io.StringIO(), force_terminal=True, width=60)
    text = P.stream_answer(con, iter(TOKENS), animate=False, interactive=True,
                           render_plain=True, label="assistant")
    out = con.file.getvalue()
    assert text == "".join(TOKENS)
    assert "assistant" in out
    assert "apple" in out and "banana" in out


# ---------------------------------------------------------------------------
# thinking_status — one spinner, no-op off a terminal
# ---------------------------------------------------------------------------


def test_thinking_status_noop_without_console():
    with P.thinking_status(None, animate=True):
        pass  # no console → no-op, no exception


def test_thinking_status_noop_when_not_animating():
    con = get_console(file=io.StringIO(), force_terminal=True, width=40)
    with P.thinking_status(con, animate=False):
        pass
    assert con.file.getvalue() == ""  # nothing drawn when animation is off


# ---------------------------------------------------------------------------
# answer_surface — one answer, framed (run) and inline (chat)
# ---------------------------------------------------------------------------


@dataclass
class _Resp:
    success: bool = True
    execution_time: float = 3.2
    tool_calls: int = 2
    tokens_used: int = 1204
    metadata: dict = field(default_factory=dict)
    output: str = "# Title\n\n- alpha\n- beta"
    execution_trace: list = field(default_factory=list)


def _render(fn, *, no_color=False, tty=True, **kw):
    if no_color:
        os.environ["NO_COLOR"] = "1"
    try:
        con = get_console(file=io.StringIO(), force_terminal=tty, width=70)
        fn(con)
        return con.file.getvalue()
    finally:
        os.environ.pop("NO_COLOR", None)


def test_answer_surface_framed_draws_panel():
    out = _render(lambda c: R.answer_surface("# Hi\n\n- x", framed=True,
                                             title="Agent Response", console=c))
    assert "Agent Response" in out  # panel title
    assert "─" in out or "—" in out  # panel border box-drawing
    assert "Hi" in out and "x" in out


def test_answer_surface_inline_has_no_panel():
    out = _render(lambda c: R.answer_surface("hello", framed=False,
                                             label="assistant", console=c))
    assert "assistant" in out
    assert "╭" not in out and "╰" not in out  # inline: no panel chrome


def test_answer_surface_partial_uses_warning_role():
    # A partial result uses the warning border, distinct from a plain success.
    ok = _render(lambda c: R.answer_surface("x", success=True, framed=True, console=c))
    partial = _render(lambda c: R.answer_surface("x", success=False, partial=True,
                                                 framed=True, console=c))
    assert ok != partial  # different border role → different bytes on a TTY


def test_answer_surface_non_tty_has_zero_ansi():
    # The piped / non-TTY path degrades to plain box-drawing with no ANSI.
    out = _render(lambda c: R.answer_surface("# H\n\n- a", framed=True,
                                             title="Agent Response", console=c),
                  tty=False)
    assert "\x1b" not in out
    assert "Agent Response" in out and "a" in out


# ---------------------------------------------------------------------------
# summary_line — one builder, run + chat presets, ASCII fallback
# ---------------------------------------------------------------------------


def test_summary_line_run_matches_progress_reexport():
    resp = _Resp(metadata={"cost_usd": 0.0006})
    assert R.summary_line(resp, mode="run") == P.summary_line(resp)
    plain, markup = R.summary_line(resp, mode="run")
    assert plain == "✓ Done in 3.2s · 2 tools · 1,204 tokens · $0.0006"
    assert "[green]" in markup


def test_summary_line_chat_preset_and_session_total():
    footer, markup = R.summary_line(mode="chat", elapsed=3.2, tokens=318,
                                    cost=0.0003, session=(4, 1020, 0.0012))
    assert footer == markup
    assert footer.startswith("· 3.2s · 318 tok · $0.0003")
    assert "(session: 4 turns · 1,020 tok · $0.0012)" in footer


def test_summary_line_chat_no_session_on_first_turn():
    footer, _ = R.summary_line(mode="chat", elapsed=1.0, tokens=10, cost=0.0,
                               session=(1, 10, 0.0))
    assert footer == "· 1.0s · 10 tok"
    assert "session" not in footer


def test_summary_line_ascii_glyph_fallback():
    stream = _AsciiStream()
    plain, _ = R.summary_line(_Resp(), mode="run", stream=stream)
    assert plain.startswith("+ Done in")  # ASCII stand-in for ✓
    # At the print boundary the typographic separators fold to ASCII too, so the
    # whole line encodes without raising.
    R.ascii_fold(plain, stream).encode("ascii")


def test_ascii_fold_is_identity_on_utf8():
    class _Utf8(io.StringIO):
        encoding = "utf-8"

    line = "✓ Done in 3.2s · 2 tools — partial…"
    assert R.ascii_fold(line, _Utf8()) == line  # byte-unchanged on a UTF-8 stream


def test_ascii_fold_swaps_typography():
    folded = R.ascii_fold("a · b — c … █", _AsciiStream())
    folded.encode("ascii")  # must not raise
    assert "·" not in folded and "—" not in folded and "…" not in folded


# ---------------------------------------------------------------------------
# trace / timeline glyphs — ASCII-safe on a non-UTF-8 stream
# ---------------------------------------------------------------------------


def test_trace_lines_ascii_safe():
    stream = _AsciiStream()
    trace = [
        {"type": "reasoning_step", "message": "step"},
        {"type": "tool_call_start", "data": {"tool_name": "calc", "tool_input": "1+1"}},
        {"type": "tool_call_complete", "data": {"result": "2"}},
        {"type": "sub_agent_start", "data": {"agent_name": "helper"}},
        {"type": "task_decomposition", "message": "plan"},
    ]
    # Glyphs fall back at the source; the print boundary folds the remaining
    # typography (bars, ellipsis) — together the lines encode without raising.
    for _style, text in P.execution_trace_lines(trace, stream=stream):
        R.ascii_fold(text, stream).encode("ascii")
    for _style, text in P.execution_timeline_lines(trace, stream=stream):
        R.ascii_fold(text, stream).encode("ascii")


def test_trace_lines_utf8_keep_emoji():
    # On a UTF-8 stream the emitted characters are unchanged.
    trace = [{"type": "tool_call_start",
              "data": {"tool_name": "calc", "tool_input": "1+1"}}]
    text = P.execution_trace_lines(trace)[0][1]
    assert text.startswith("🔧 ")


# ---------------------------------------------------------------------------
# a terminal that cannot encode the CLI's typography still renders
# ---------------------------------------------------------------------------


def _ascii_console():
    """A forced-terminal effGen console writing to an ASCII-only stream."""
    stream = _AsciiStream()
    return get_console(file=stream, force_terminal=True, width=100), stream


def test_spinner_frame_falls_back_to_ascii():
    assert P._spinner_frame(_AsciiStream()) in P._BRAILLE_ASCII
    assert P._spinner_frame(io.StringIO()) in P._BRAILLE_ASCII  # no encoding -> ASCII


def test_spinner_frame_keeps_braille_on_utf8():
    class _Utf8(io.StringIO):
        encoding = "utf-8"

    assert P._spinner_frame(_Utf8()) in P._BRAILLE


def test_status_and_thinking_renderables_encode_on_ascii():
    console, stream = _ascii_console()
    state = P._StatusState("gpt-5-nano", reasoning=True, hint="Ctrl-C to cancel",
                           stream=stream)
    console.print(P._StatusRenderable(state, stream))
    state.set_step("Running calculator…")
    console.print(P._StatusRenderable(state, stream))
    console.print(P._ThinkingRenderable("Thinking…", 0.0, stream))
    stream.getvalue().encode("ascii")  # must not raise
    assert "reasoning" in stream.getvalue()
    assert "Running calculator..." in stream.getvalue()


def test_reasoning_label_drops_the_glyph_on_ascii():
    class _Utf8(io.StringIO):
        encoding = "utf-8"

    ascii_label = P._StatusState("gpt-5-nano", True, stream=_AsciiStream())._idle_label()
    # The glyph is dropped at the source; the ellipsis folds at the print boundary.
    assert ascii_label == "gpt-5-nano reasoning…"
    R.ascii_fold(ascii_label, _AsciiStream()).encode("ascii")  # must not raise
    assert P._StatusState("gpt-5-nano", True, stream=_Utf8())._idle_label().startswith("🧠 ")


def test_typography_fold_covers_the_input_chevron():
    assert R.ascii_fold("model · 1 tool › ", _AsciiStream()) == "model - 1 tool > "


def test_chat_banner_and_prompt_encode_on_ascii():
    from effgen.cli.chat import ChatREPL

    console, stream = _ascii_console()

    class _CLI:
        def __init__(self, console):
            self.console = console

        def _human_stream(self):
            return stream

        def print(self, text):
            console.print(text)

    repl = ChatREPL.__new__(ChatREPL)
    repl.cli = _CLI(console)
    repl._color = True
    repl.preset = "math"
    repl.model_id = "openai:gpt-5-nano"
    repl.agent = None  # tool_count is derived from the agent
    repl._banner_line("effGen v0.0.0 · chat")
    repl._banner_line("Session: s1  ·  resuming 3 prior message(s)")
    prompt = repl._prompt_str()
    prompt.encode("ascii")
    stream.getvalue().encode("ascii")
    assert prompt.endswith("> ")
    # the multi-line continuation prompt folds too
    assert R.ascii_fold("… ", stream) == "... "


def test_landing_renders_on_ascii():
    from effgen.cli import landing

    console, stream = _ascii_console()
    landing._render(console)
    stream.getvalue().encode("ascii")  # must not raise
    assert "What next?" in stream.getvalue()


def test_first_run_welcome_folds_and_is_identity_on_utf8():
    from effgen.cli.onboarding import welcome_text

    class _Utf8(io.StringIO):
        encoding = "utf-8"

    welcome_text(_AsciiStream()).encode("ascii")  # must not raise
    assert "👋 Welcome to effGen — " in welcome_text(_Utf8())


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
