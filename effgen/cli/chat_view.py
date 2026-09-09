"""What a chat session prints around a turn.

:mod:`effgen.cli.chat` owns the session itself and mixes these methods into
:class:`~effgen.cli.chat.ChatREPL`, so ``effgen.cli.chat`` stays the import path
callers use. Holds the render layer: which console human-facing chrome goes to,
the startup banner, the model/tool-aware prompt string, the answer surface, and
the per-turn footer.

Two pieces of session state decide every surface here. ``self._color`` (off under
``NO_COLOR``) decides whether styling is applied at all; ``self.interactive``
decides whether chrome joins the answer on stdout or is routed to stderr so a
piped caller reads answers only. Text carrying the separators is ASCII-folded
against the stream it is written to, so a terminal that cannot encode them prints
the surface instead of failing on it.
"""

from __future__ import annotations

import sys
from typing import Any

from effgen.cli import progress as _progress
from effgen.cli.chat_session import _history_dir


class ChatViewMixin:
    """The presentation half of :class:`~effgen.cli.chat.ChatREPL`."""

    # ------------------------------------------------------------------
    # Output helpers (color-aware, non-TTY-aware)
    # ------------------------------------------------------------------
    def _chrome_console(self) -> Any:
        """The console human-facing chrome should print to (stderr when piped)."""
        human = getattr(self.cli, "_human", None)
        if callable(human):
            try:
                return human()
            except Exception:  # noqa: BLE001
                pass
        return getattr(self.cli, "console", None)

    def _human_stream(self) -> Any:
        """The text stream human output goes to, for the typography fallback.

        A CLI stand-in need not carry the hook; fall back to ``sys.stdout``, which
        is where the prompt and banner are written when it is absent.
        """
        getter = getattr(self.cli, "_human_stream", None)
        return getter() if callable(getter) else sys.stdout

    def _banner_line(self, text: str, style: str | None = None) -> None:
        """Print one banner line without number/repr auto-highlighting.

        Styling is applied only when color is enabled, so versions/counts read
        as one token and a ``NO_COLOR`` banner carries no escape codes. The text
        is ASCII-folded when the console cannot encode the separators, so a
        non-UTF-8 terminal prints the banner instead of failing on it.
        """
        from effgen.ui.render import ascii_fold

        text = ascii_fold(text, self._human_stream())
        console = getattr(self.cli, "console", None)
        if console:
            markup = f"[{style}]{text}[/{style}]" if (style and self._color) else text
            console.print(markup, highlight=False)
        else:
            self.cli.print(text)

    # ------------------------------------------------------------------
    # Prompt / banner
    # ------------------------------------------------------------------
    def _prompt_str(self) -> str:
        """A model/tool-aware input prompt, e.g. ``math · gpt-5-nano · 1 tool › ``.

        Kept as plain text (no ANSI) so ``readline`` width math and ↑/↓ history
        stay correct across terminals, and ASCII-folded when the terminal cannot
        encode the separators.
        """
        from effgen.ui.render import ascii_fold

        label = _progress.short_model_label(self.model_id)
        bits = []
        if self.preset:
            bits.append(self.preset)
        bits.append(label)
        n = self.tool_count
        if n:
            bits.append(f"{n} tool" + ("s" if n != 1 else ""))
        return ascii_fold(" · ".join(bits) + " › ", self._human_stream())

    def _banner(self) -> None:
        from effgen import __version__

        self._banner_line(f"\neffGen v{__version__} · chat", style="bold cyan")
        label = _progress.short_model_label(self.model_id)
        meta = f"Model: {label}"
        if self.provider:
            meta += f"  (provider: {self.provider})"
        if self.preset:
            meta = f"Preset: {self.preset}  ·  " + meta
        self._banner_line(meta)
        if self.system_prompt:
            self._banner_line("Persona: custom (steers every reply)")
        if self.tool_names:
            self._banner_line(f"Tools: {', '.join(self.tool_names)}")
        if self.guardrails:
            self._banner_line(f"Guardrails: {self.guardrails}")
        # Show that we're continuing a named session (and how many turns it has).
        if self.session_id:
            try:
                from effgen.core.session import Session

                prior = len(Session.load_or_create(self.session_id).messages)
            except Exception:  # noqa: BLE001
                prior = 0
            if prior:
                self._banner_line(
                    f"Session: {self.session_id}  ·  resuming {prior} prior message(s)"
                )
            else:
                self._banner_line(f"Session: {self.session_id}  ·  new (will be saved)")
        self._banner_line(
            "Type your message and press Enter.  "
            "End a line with \\ for multi-line input.\n"
            "Slash commands (type / for the menu): /help  /model  /tools  /status  "
            "/cost  /trace  /reset  /save  /session  /load  /doctor  /exit"
        )
        # A first-timer running bare `effgen chat` gets a local model by default;
        # name the download and offer a fast keyed alternative so the wait is not
        # a surprise.
        if self._model_defaulted and "/" in self.model_id and ":" not in self.model_id:
            self._banner_line(
                f"Loading a local model ({self.model_id}); the first run downloads it. "
                "For a fast keyed model:  effgen chat -m groq:openai/gpt-oss-20b",
                style="dim",
            )
        # Friendly resume: if there are saved conversations, point at the most
        # recent one so a returning user can pick up where they left off. A
        # single non-spammy line — we never auto-load (that would silently mix
        # an old context into a fresh session).
        try:
            saved = sorted(_history_dir().glob("chat_*.json"), reverse=True)
        except OSError:
            saved = []
        if saved:
            last = saved[0].name[len("chat_"):-len(".json")] or saved[0].name
            self._banner_line(
                f"Resume: {len(saved)} saved — `/load` to list, "
                f"or `/load {last}` for the most recent."
            )

    # ------------------------------------------------------------------
    # Answer rendering
    # ------------------------------------------------------------------
    def _show_answer(self, answer: str) -> None:
        """Print a finished answer, rendering markdown when a rich console is up."""
        from effgen.ui.render import answer_surface

        answer_surface(answer, framed=False, label="assistant", console=self.console)

    def _show_progress(self, text: str) -> None:
        """Print what a stopped turn had reached, labelled as progress.

        Rendered under the reply rather than as part of it: it is tool output and
        reasoning the model never wrote up as an answer. A piped session gets it
        on stdout beside the reply it belongs to, as plain text.
        """
        from effgen.cli.commands.run import PARTIAL_PROGRESS_TITLE
        from effgen.ui.render import answer_surface

        if not self.interactive:
            sys.stdout.write(f"\n{PARTIAL_PROGRESS_TITLE}:\n{text}\n")
            return
        self.cli.print_warning(f"{PARTIAL_PROGRESS_TITLE}:")
        answer_surface(text, framed=False, console=self.console)

    # ------------------------------------------------------------------
    # The per-turn footer
    # ------------------------------------------------------------------
    def _footer_text(
        self, elapsed: float, tok: int, cost: float, interrupted: bool
    ) -> str:
        """Build the per-turn footer, appending a running session total over time.

        Shares the one summary builder with ``effgen run``; the ``chat`` preset
        keeps the compact per-turn shape and the running session total.
        """
        from effgen.ui.render import summary_line

        footer, _ = summary_line(
            mode="chat",
            elapsed=elapsed,
            tokens=tok,
            cost=cost,
            interrupted=interrupted,
            session=(self.turns, self.session_tokens, self.session_cost),
        )
        return footer

    def _print_footer(
        self, elapsed: float, dtok: int, dcost: float, interrupted: bool
    ) -> None:
        # The tool path has exact per-turn accounting from the response; prefer it.
        rtok = getattr(self, "_turn_response_tokens", 0)
        rcost = getattr(self, "_turn_response_cost", 0.0)
        self._turn_response_tokens = 0
        self._turn_response_cost = 0.0
        tok = rtok or max(dtok, 0)
        cost = rcost or max(dcost, 0.0)

        from effgen.ui.render import ascii_fold

        footer = ascii_fold(self._footer_text(elapsed, tok, cost, interrupted),
                            self.cli._human_stream())
        console = self._chrome_console()
        if console:
            console.print(
                f"[dim]{footer}[/dim]" if self._color else footer, highlight=False
            )
        else:
            print(footer, file=None if self.interactive else sys.stderr)
