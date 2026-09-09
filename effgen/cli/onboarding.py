"""Onboarding, tips, and friendly guidance for the command-line interface.

This is a *presentation layer only*: nothing here changes what an agent does or
calls a model. It makes the CLI feel welcoming and self-teaching:

* a rotating, non-spammy ``💡 Tip:`` line surfaced at natural moments,
* "did you mean …?" fuzzy suggestions for mistyped names (subcommands, config
  keys, presets, …) shared with the model/tool/preset suggesters elsewhere,
* a one-time first-run welcome that points at ``effgen doctor`` and a hello
  agent,
* a shared "errors that teach" formatter (one-line cause + fix + doc pointer),
* friendly empty-state helpers.

Everything is opt-out and quiet by default in the wrong context. Tips and the
welcome only appear on an interactive terminal, never under ``--quiet`` /
non-interactive / CI, and tips honor ``EFFGEN_TIPS=0``. State (the rotating tip
index, the "already welcomed" flag) lives under ``~/.effgen/`` like the rest of
effGen's user state.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Reuse the CLI-wide environment helpers so the whole tool agrees on what
# "interactive" and "CI" mean.
from effgen.cli.progress import is_ci

__all__ = [
    "TIPS",
    "state_dir",
    "is_interactive",
    "tips_enabled",
    "pick_tip",
    "maybe_print_tip",
    "first_run_pending",
    "mark_welcomed",
    "maybe_show_first_run_welcome",
    "suggest",
    "did_you_mean",
    "teach",
]


# ---------------------------------------------------------------------------
# User-state directory (shared with budget/cost/session stores at ~/.effgen)
# ---------------------------------------------------------------------------

def state_dir() -> Path:
    """Return effGen's per-user state directory (``~/.effgen`` by default).

    Honors ``EFFGEN_HOME`` so tests (and unusual installs) can redirect state
    without touching the real home directory.
    """
    override = os.environ.get("EFFGEN_HOME", "").strip()
    base = Path(override).expanduser() if override else Path.home() / ".effgen"
    return base


def _state_file(name: str) -> Path:
    return state_dir() / name


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing/corrupt state is non-fatal
        return {}


def _write_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception:  # noqa: BLE001 - never let a tip/welcome break a command
        pass


# ---------------------------------------------------------------------------
# Interactivity gating
# ---------------------------------------------------------------------------

def is_interactive() -> bool:
    """True only when a human is plausibly watching an interactive terminal."""
    if is_ci():
        return False
    try:
        return bool(sys.stdout.isatty()) and bool(sys.stdin.isatty())
    except Exception:  # noqa: BLE001 - a stream without isatty() is non-interactive
        return False


# ---------------------------------------------------------------------------
# Tip system
# ---------------------------------------------------------------------------

# Curated, genuinely-useful tips. Kept short, actionable, and free of internal
# jargon. They rotate so the user sees variety, and are shown sparingly.
TIPS: tuple[str, ...] = (
    "add --stream to watch tokens arrive live as the model answers.",
    "effgen doctor --live --cheap confirms which providers actually work.",
    "effgen models refresh updates the local model catalog from each provider.",
    "pass --provider to pick the backend, e.g. effgen run \"hi\" -m gpt-5-nano --provider openai.",
    "effgen run --preset coding (or math/research/rag) starts with a tuned tool set.",
    "pip install 'effgen[vllm]' enables faster local GPU inference.",
    "effgen cost shows your spend; effgen cost set-budget caps it.",
    "use a provider prefix to be explicit: -m groq:openai/gpt-oss-20b.",
    "effgen tools list shows every built-in tool; tools info <name> explains one.",
    "--explain shows the full step-by-step reasoning trace for a run.",
    "effgen chat opens an interactive session with the same agent.",
    "effgen code writes and runs code in your workspace, showing each edit as a diff.",
    "effgen code --plan proposes the change and its diffs without writing anything.",
    "set EFFGEN_TIPS=0 to silence these tips.",
)


def tips_enabled(*, quiet: bool = False) -> bool:
    """Whether a tip may be shown for this invocation."""
    if quiet:
        return False
    # EFFGEN_TIPS=0 / false / no / off silences tips entirely.
    raw = os.environ.get("EFFGEN_TIPS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return is_interactive()


def pick_tip(index: int | None = None) -> str:
    """Return a tip body. With *index* given, return that tip (wrapping)."""
    if not TIPS:
        return ""
    if index is None:
        index = 0
    return TIPS[index % len(TIPS)]


# Show a tip roughly once every few eligible runs so it never feels spammy.
_TIP_THROTTLE = 3


def maybe_print_tip(*, quiet: bool = False, force: bool = False, stream: Any = None) -> bool:
    """Print one rotating ``💡 Tip:`` line at a natural moment, if appropriate.

    Returns True if a tip was printed. Non-spammy: only shows on an interactive
    terminal (not ``--quiet``/CI/piped), honors ``EFFGEN_TIPS=0``, rotates
    through :data:`TIPS` without repeating, and only fires on roughly every
    third eligible invocation (use *force* to bypass the throttle in tests).

    Args:
        quiet: Suppress the tip, for a command run with ``--quiet``.
        force: Print regardless of the throttle, used in tests.
        stream: The stream the tip is written to.
    """
    if not tips_enabled(quiet=quiet):
        return False

    path = _state_file("tips.json")
    state = _read_json(path)
    count = int(state.get("count", 0)) + 1
    index = int(state.get("index", 0))

    show = force or (count % _TIP_THROTTLE == 0)
    if show:
        body = pick_tip(index)
        state["index"] = (index + 1) % max(1, len(TIPS))
        out = stream if stream is not None else sys.stdout
        try:
            print(f"\n💡 Tip: {body}", file=out)
        except Exception:  # noqa: BLE001
            show = False
    state["count"] = count
    _write_json(path, state)
    return bool(show)


# ---------------------------------------------------------------------------
# First-run welcome
# ---------------------------------------------------------------------------

_WELCOME_TEXT = """\
{welcome}Welcome to effGen — a framework for building agents with small (and cloud) models.

Get started in five steps:
  1. effgen doctor              # check which providers are ready to use
  2. effgen quickstart          # a 2-minute guided first run
  3. effgen run "What is 25 * 17?" -m gpt-5-nano --provider openai
  4. effgen quickstart --init my-agent   # a project: config, .env template,
                                         # example, daily spend cap
  5. effgen code                # a coding agent that writes, runs and fixes code

Or in Python:
  from effgen import create_agent
  agent = create_agent("math")
  print(agent.run("What is 25 * 17?"))

(This welcome shows only once. See effgen --help for everything.)
"""


def welcome_text(stream: Any = None) -> str:
    """Return the first-run welcome, glyphed and ASCII-folded for *stream*."""
    from effgen.ui.palette import glyph
    from effgen.ui.render import ascii_fold

    mark = glyph("welcome", stream)
    return ascii_fold(_WELCOME_TEXT.format(welcome=f"{mark} " if mark else ""), stream)


def first_run_pending() -> bool:
    """True if the one-time welcome has not been shown yet."""
    return not _state_file(".welcomed").exists()


def mark_welcomed() -> None:
    """Record the one-time welcome flag so it never shows again."""
    flag = _state_file(".welcomed")
    try:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("1\n")
    except Exception:  # noqa: BLE001 - never let state IO break a command
        pass


def _render_welcome_panel() -> bool:
    """Render the welcome inside a branded panel on the real terminal.

    Returns False (so the caller prints the plain text instead) when ``rich`` is
    unavailable or no console can be built.
    """
    from effgen.ui.theme import get_console, rich_available

    if not rich_available():
        return False
    console = get_console()
    if console is None:
        return False
    from rich.panel import Panel
    from rich.text import Text

    from effgen.ui import brand

    stream = getattr(console, "file", None)
    title = brand.wordmark(stream=stream)
    console.print(
        Panel(
            Text(welcome_text(stream).strip()),
            title=f"[effgen.title]{title}[/effgen.title]",
            border_style="effgen.accent",
            padding=(1, 2),
        )
    )
    return True


def maybe_show_first_run_welcome(*, quiet: bool = False, stream: Any = None) -> bool:
    """Show the one-time welcome on first interactive use; return True if shown.

    Records a flag under ``~/.effgen`` so it never shows again. Silent under
    ``--quiet``/non-interactive/CI (but still records the flag so a later
    interactive run does not surprise the user with a stale welcome). On a real
    terminal the welcome is drawn in a branded panel; when an explicit *stream*
    is given (or ``rich`` is absent) the plain text is written to it.
    """
    flag = _state_file(".welcomed")
    if flag.exists():
        return False
    # Mark as welcomed regardless, so we never nag across contexts.
    mark_welcomed()
    if quiet or not is_interactive():
        return False
    if stream is None and _render_welcome_panel():
        return True
    out = stream if stream is not None else sys.stdout
    try:
        print(welcome_text(out), file=out)
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# "Did you mean …?" — shared fuzzy suggester
# ---------------------------------------------------------------------------

def suggest(name: str, candidates: Iterable[str], *, n: int = 1, cutoff: float = 0.6) -> list[str]:
    """Return up to *n* candidates closest to *name* (case-insensitive).

    Args:
        name: The name that was not recognized.
        candidates: The names it could have been.
        n: Most suggestions to return.
        cutoff: Least similarity a candidate must reach.
    """
    if not name or not candidates:
        return []
    pool = list(candidates)
    lower_map: dict[str, str] = {}
    for c in pool:
        lower_map.setdefault(str(c).lower(), str(c))
    matches = difflib.get_close_matches(str(name).lower(), list(lower_map), n=n, cutoff=cutoff)
    out: list[str] = []
    for m in matches:
        original = lower_map[m]
        if original not in out:
            out.append(original)
    # Also catch a clean substring/prefix relationship the ratio may miss
    # (e.g. "mod" -> "models"), which users type all the time.
    if len(out) < n:
        nl = str(name).lower()
        for c in pool:
            cl = str(c).lower()
            if c not in out and (cl.startswith(nl) or nl.startswith(cl)) and len(nl) >= 2:
                out.append(str(c))
                if len(out) >= n:
                    break
    return out[:n]


def did_you_mean(name: str, candidates: Iterable[str], *, n: int = 1, cutoff: float = 0.6) -> str:
    """Format a "Did you mean 'x'?" clause, or ``""`` if nothing is close.

    Args:
        name: The name that was not recognized.
        candidates: The names it could have been.
        n: Most suggestions to name in the clause.
        cutoff: Least similarity a candidate must reach.

    Returns:
        The clause to append to an error, or ``""`` when nothing is close.
    """
    matches = suggest(name, candidates, n=n, cutoff=cutoff)
    if not matches:
        return ""
    if len(matches) == 1:
        return f"Did you mean '{matches[0]}'?"
    quoted = ", ".join(f"'{m}'" for m in matches)
    return f"Did you mean one of: {quoted}?"


# ---------------------------------------------------------------------------
# Errors that teach
# ---------------------------------------------------------------------------

def teach(cause: str, fix: str | None = None, doc: str | None = None) -> str:
    """Format a consistent, teaching error: cause + how to fix + a doc pointer.

    Produces a compact multi-line block other CLI code can print, so every
    user-facing error reads the same way:

        <cause>
          Fix: <fix>
          Docs: <doc>

    Args:
        cause: What went wrong, in one line.
        fix: The next step that resolves it.
        doc: A documentation pointer for the reader who wants more.

    Returns:
        The formatted block, ready to print.
    """
    lines = [cause.rstrip()]
    if fix:
        lines.append(f"  Fix: {fix}")
    if doc:
        lines.append(f"  Docs: {doc}")
    return "\n".join(lines)
