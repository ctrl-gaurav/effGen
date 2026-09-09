"""Terminal output for the ``effgen code`` session.

:mod:`effgen.cli.code.repl` owns the session itself and mixes these methods into
:class:`~effgen.cli.code.repl.CodeREPL`, so ``effgen.cli.code.repl`` stays the
import path callers use. Holds what the session prints around a turn: the line
and status writers, the startup banner, the prompt string, the answer surface,
and the per-turn footer. Each renders through the shared console when one is
available and falls back to plain lines when it is not.
"""

from __future__ import annotations

from typing import Any

from effgen.cli.code.permissions import MODE_DESCRIPTIONS
from effgen.cli.code.render import print_plain, print_status


class CodeViewMixin:
    """The presentation half of :class:`~effgen.cli.code.repl.CodeREPL`.

    Every method reads state ``CodeREPL.__init__`` sets — ``self.cli``,
    ``self.model_id``, ``self.mode``, ``self.project``, ``self.workspace`` and
    the session counters — and prints through :mod:`effgen.cli.code.render`.
    """

    def _say(self, text: str = "", *, style: str | None = None) -> None:
        """Print one line of session output verbatim (see :func:`print_plain`)."""
        print_plain(self.cli, text, style=style)

    def _status(self, kind: str, text: str) -> None:
        """Print a success/warning/error line with its glyph, without markup."""
        print_status(self.cli, kind, text)

    # ------------------------------------------------------------------
    # Banner / prompt
    # ------------------------------------------------------------------
    def _banner_line(self, text: str, style: str | None = None) -> None:
        self._say(text, style=style)

    def _banner(self) -> None:
        from effgen import __version__

        self._banner_line(f"\neffGen v{__version__} · code", style="bold cyan")
        from effgen.cli import progress as _progress

        self._banner_line(f"Model: {_progress.short_model_label(self.model_id)}")
        self._banner_line(f"Workspace: {self.workspace}")
        self._banner_line(self.project.summary_line())
        if self.project.brief_path:
            self._banner_line(f"Project instructions: {self.project.brief_path}")
        self._banner_line(
            f"Permissions: {self.mode.value} — {MODE_DESCRIPTIONS[self.mode]}"
        )
        self._banner_line(
            "Describe a change and press Enter.  End a line with \\ for multi-line.\n"
            "Slash commands (type / for the menu): /help  /plan  /review  /run  "
            "/test  /diff  /apply  /undo  /context  /add  /model  /git  /exit"
        )
        if self._model_defaulted and "/" in (self.model_id or "") and ":" not in (self.model_id or ""):
            self._banner_line(
                f"Loading a local model ({self.model_id}); the first run downloads it. "
                "For a fast keyed model:  /model groq:openai/gpt-oss-20b",
                style="dim",
            )

    def _prompt_str(self) -> str:
        from effgen.cli import progress as _progress

        label = _progress.short_model_label(self.model_id)
        bits = ["code", label, self.mode.value]
        if self.context_files:
            bits.append(f"{len(self.context_files)} file" + ("s" if len(self.context_files) != 1 else ""))
        return " · ".join(bits) + " › "

    # ------------------------------------------------------------------
    # The answer and the per-turn footer
    # ------------------------------------------------------------------
    def _render_answer(self, result: Any, *, already_streamed: bool = False) -> None:
        """Print the turn's answer, and any partial progress beside it.

        With *already_streamed* the answer text is on screen already — it was
        written there as the model produced it — so only what did not stream is
        printed: the partial-progress panel and, for a turn that failed, the
        error panel carrying the typed outcome.
        """
        from effgen.cli.commands.run import PARTIAL_PROGRESS_TITLE
        from effgen.ui.render import answer_surface

        console = self.cli._human()
        stopped = not result.success and result.partial
        progress = getattr(result, "partial_output", "") if stopped else ""
        if not result.success and not result.partial:
            self.cli.print_error_panel(
                result.answer or "The run produced no answer.", title="Error"
            )
        elif console:
            if not already_streamed:
                answer_surface(
                    result.answer,
                    success=result.success,
                    partial=result.partial,
                    framed=False,
                    label="agent",
                    console=console,
                )
            if progress:
                answer_surface(
                    progress,
                    success=False,
                    partial=True,
                    framed=True,
                    title=PARTIAL_PROGRESS_TITLE,
                    console=console,
                )
        else:
            self._say(result.answer)
            if progress:
                self._say(f"{PARTIAL_PROGRESS_TITLE}:\n{progress}")

    def _print_footer(self, elapsed: float, tokens: int, cost: float) -> None:
        from effgen.ui.render import ascii_fold, summary_line

        footer, _ = summary_line(
            mode="chat",
            elapsed=elapsed,
            tokens=tokens,
            cost=cost,
            session=(self.turns, self.session_tokens, self.session_cost),
        )
        self._say(ascii_fold(footer, self.cli._human_stream()), style="dim")
