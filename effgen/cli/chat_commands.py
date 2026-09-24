"""The chat command table, its dispatcher, and the commands that report state.

:mod:`effgen.cli.chat` owns the session itself and mixes these methods into
:class:`~effgen.cli.chat.ChatREPL`, so ``effgen.cli.chat`` stays the import path
callers use.

:data:`_SLASH_COMMANDS` and :meth:`ChatCommandsMixin._dispatch` are in one module
so the help text, the tab-completer and the routing table describe the same
command set. The commands that change what the session *is* — ``/model``,
``/save``, ``/load``, ``/session`` — live in :mod:`effgen.cli.chat_session`; what
stays here reports the session's state (``/help``, ``/status``, ``/cost``,
``/tools``, ``/trace``, ``/doctor``) or resets it (``/reset``, ``/clear``).
"""

from __future__ import annotations

import os

from effgen.cli import progress as _progress

# Slash commands surfaced in /help and tab-completion. Kept in one place so the
# help text, the completer, and the dispatcher never drift apart.
_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help (or type just / for the menu)",
    "/model": "Hot-swap the active model:  /model gpt-5-nano",
    "/tools": "List tools, or toggle one:  /tools calculator",
    "/status": "Show the session state: model, persona, tools, running totals",
    "/cost": "Session token + cost total",
    "/reset": "Clear the conversation memory",
    "/save": "Save this chat to a file:  /save [name]  (see /session to resume)",
    "/session": "Show or name the resumable session id (used by `effgen sessions`)",
    "/load": "Load a saved conversation:  /load [name|number]",
    "/trace": "Show the last turn's reasoning/tool steps",
    "/doctor": "Run a quick environment check",
    "/clear": "Clear the screen",
    "/exit": "Leave chat (also: /quit, exit, quit)",
}


class ChatCommandsMixin:
    """The command half of :class:`~effgen.cli.chat.ChatREPL`."""

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, text: str) -> str | None:
        """Route a line. Returns 'exit', 'handled', or None (treat as a message)."""
        low = text.lower()
        # Back-compat bare words from the old chat.
        if low in ("exit", "quit"):
            return "exit"
        if low == "help":
            self._cmd_help()
            return "handled"
        if low == "clear":
            self._cmd_reset()
            return "handled"
        if low in ("history", "sessions"):
            self._list_saved()
            return "handled"

        if not text.startswith("/"):
            return None

        # A bare "/" opens the command menu, so slash commands stay discoverable
        # after the start banner scrolls away.
        if text.strip() == "/":
            self._cmd_help()
            return "handled"

        parts = text[1:].split(maxsplit=1)
        cmd = ("/" + parts[0]).lower() if parts else "/"
        arg = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/help": lambda: self._cmd_help(),
            "/exit": lambda: "exit",
            "/quit": lambda: "exit",
            "/model": lambda: self._cmd_model(arg),
            "/tools": lambda: self._cmd_tools(arg),
            "/status": lambda: self._cmd_status(),
            "/session": lambda: self._cmd_session(arg),
            "/cost": lambda: self._cmd_cost(),
            "/reset": lambda: self._cmd_reset(),
            "/save": lambda: self._cmd_save(arg),
            "/load": lambda: self._cmd_load(arg),
            "/trace": lambda: self._cmd_trace(),
            "/doctor": lambda: self._cmd_doctor(),
            "/clear": lambda: self._cmd_clear_screen(),
        }
        handler = handlers.get(cmd)
        if handler is None:
            import difflib

            typed = text.split()[0]
            close = difflib.get_close_matches(cmd, list(_SLASH_COMMANDS), n=3)
            hint = (
                f" Did you mean: {', '.join(close)}?"
                if close
                else " Type /help for the list."
            )
            self.cli.print_warning(f"Unknown command '{typed}'.{hint}")
            return "handled"
        result = handler()
        return result if result == "exit" else "handled"

    # ------------------------------------------------------------------
    # Slash command implementations
    # ------------------------------------------------------------------
    def _cmd_help(self) -> None:
        # A rich table only on an interactive color terminal; a piped or
        # NO_COLOR run gets plain lines with no box-drawing or escape codes.
        if self.console and self.interactive and self._color:
            from rich.table import Table

            table = Table(title="Chat commands", show_header=True, header_style="cyan")
            table.add_column("Command", style="green", no_wrap=True)
            table.add_column("Does")
            for cmd, desc in _SLASH_COMMANDS.items():
                table.add_row(cmd, desc)
            self.console.print(table)
        else:
            self.cli.print("Chat commands:")
            for cmd, desc in _SLASH_COMMANDS.items():
                self.cli.print(f"  {cmd:9s} {desc}")

    def _cmd_tools(self, arg: str) -> None:
        registry = self.cli.tool_registry
        registry.discover_builtin_tools()
        if not arg:
            active = sorted(getattr(self.agent, "tools", {}) or {})
            if active:
                self.cli.print(f"Active tools ({len(active)}): {', '.join(active)}")
            else:
                self.cli.print("No tools active. Add one:  /tools calculator")
            sample = ", ".join(sorted(registry.list_tools())[:12])
            self.cli.print(f"Available (sample): {sample} …  (effgen tools list)")
            return
        name = arg.split()[0]
        if name not in set(registry.list_tools()):
            self.cli.print_warning(f"No tool named '{name}'. See: effgen tools list")
            return
        if name in self.tool_names:
            self.tool_names.remove(name)
            self._rebuild()
            self.cli.print_success(f"Removed tool '{name}' ({self.tool_count} active)")
        else:
            self.tool_names.append(name)
            before = self.tool_count
            self._rebuild()
            if self.tool_count <= before and name not in (getattr(self.agent, "tools", {}) or {}):
                # filter_incompatible_tools dropped it for this model.
                self.tool_names.remove(name)
                self.cli.print_warning(
                    f"Tool '{name}' is not compatible with {self.model_id}."
                )
            else:
                self.cli.print_success(f"Added tool '{name}' ({self.tool_count} active)")

    def _cmd_status(self) -> None:
        """Show the session state a long chat needs to stay legible."""
        from effgen.ui.render import format_cost

        self.cli.print_header("Session")
        self.cli.print(f"Model: {self.model_id}")
        if self.provider:
            self.cli.print(f"Provider: {self.provider}")
        if self.preset:
            self.cli.print(f"Preset: {self.preset}")
        if self.system_prompt:
            self.cli.print("Persona: custom (set with --persona/--system-prompt)")
        if self.guardrails:
            self.cli.print(f"Guardrails: {self.guardrails}")
        active = sorted(getattr(self.agent, "tools", {}) or {})
        self.cli.print(f"Tools: {', '.join(active) if active else '(none)'}")
        if self.session_id:
            self.cli.print(
                f"Session: {self.session_id}  (resumable via `effgen sessions`)"
            )
        else:
            self.cli.print("Session: not persisted  (start one with /session <id>)")
        cost = format_cost(self.session_cost) if self.session_cost else "$0.00"
        self.cli.print(
            f"Totals: {self.turns} turns · {self.session_tokens:,} tok · {cost}"
        )

    def _cmd_cost(self) -> None:
        self.cli.print_header("Session cost")
        from effgen.ui.render import format_cost

        self.cli.print(f"Turns: {self.turns}")
        self.cli.print(f"Tokens: {self.session_tokens:,}")
        # A $0.00 here is accurate for free/local sessions; for cheap cloud turns
        # the adaptive formatter keeps sub-cent spend visible instead of $0.0000.
        self.cli.print(f"Cost: {format_cost(self.session_cost) if self.session_cost else '$0.00'}")
        # Cross-check against the process-wide cost tracker when keyed providers
        # reported real usage (local models stay $0).
        try:
            from effgen.models._cost import CostTracker

            tracker = CostTracker.get()
            total = tracker.total_cost()
            # None: every call so far was on a model with no published price.
            if total is not None and total > 0:
                self.cli.print(f"(process total across all models: {format_cost(total)})")
        except Exception:  # noqa: BLE001
            pass

    def _cmd_reset(self) -> None:
        try:
            self.agent.reset_memory()
        except Exception:  # noqa: BLE001
            pass
        self.last_trace = None
        self.last_answer = ""
        self.cli.print_success("Conversation memory cleared.")

    def _cmd_clear_screen(self) -> None:
        try:
            os.system("cls" if os.name == "nt" else "clear")  # noqa: S605,S607
        except Exception:  # noqa: BLE001
            pass

    def _cmd_trace(self) -> None:
        if not self.last_trace:
            if self.last_answer:
                self.cli.print(
                    "Last turn was a direct streamed answer (no tool steps). "
                    "Add a tool with /tools to see reasoning steps."
                )
            else:
                self.cli.print("No turn to trace yet.")
            return
        from effgen.ui.render import ascii_fold

        self.cli.print_header(
            ascii_fold("Last turn — reasoning trace", self._human_stream())
        )
        lines = _progress.execution_trace_lines(
            self.last_trace, stream=self.cli._human_stream()
        )
        if not lines:
            self.cli.print("(no detailed steps recorded for this turn)")
            return
        for style, text in lines:
            text = ascii_fold(text, self.cli._human_stream())
            if self.console and self._color:
                self.console.print(f"[{style}]{text}[/{style}]", highlight=False)
            else:
                self.cli.print(text)

    def _cmd_doctor(self) -> None:
        from effgen.cli._main import _handle_doctor_command

        class _DoctorArgs:
            doctor_provider = self.provider
            live = False
            output_json = False

        try:
            _handle_doctor_command(_DoctorArgs())
        except Exception as e:  # noqa: BLE001
            self.cli.print_error(f"doctor failed: {e}")

    # ------------------------------------------------------------------
    # Friendly model errors
    # ------------------------------------------------------------------
    def _teach_model_error(self, exc: Exception) -> None:
        msg = str(exc).lower()
        if "not found" in msg or "404" in msg or "model_not_found" in msg:
            self.cli.print(
                "Tip: check the id with `effgen models list`, or hot-swap here "
                "with `/model <id>`."
            )
        elif "api key" in msg or "auth" in msg or "401" in msg:
            self.cli.print(
                "Tip: run `effgen doctor` (or `/doctor`) to see which provider "
                "keys are set."
            )
