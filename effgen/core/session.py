"""
Persistent conversation sessions for effGen.

A Session stores a conversation history (and optional memory snapshot)
in ~/.effgen/sessions/<session_id>.json so it can be reloaded by any
agent process. Sessions are keyed by UUID by default, but callers may
provide their own session_id (e.g. "user-123").
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from effgen.utils.atomic_file import atomic_write_text

logger = logging.getLogger(__name__)


def _default_session_dir() -> str:
    """Resolve where sessions live.

    Honors ``EFFGEN_SESSIONS_DIR`` (explicit) then ``EFFGEN_HOME`` (the base
    effGen state dir), falling back to ``~/.effgen/sessions``. Resolved lazily so
    tests and callers can point it elsewhere via the environment.
    """
    explicit = os.environ.get("EFFGEN_SESSIONS_DIR")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    home = os.environ.get("EFFGEN_HOME")
    if home:
        return os.path.join(os.path.abspath(os.path.expanduser(home)), "sessions")
    return os.path.expanduser("~/.effgen/sessions")


# Module-level constant kept for backward compatibility. New code should call
# _default_session_dir() so EFFGEN_SESSIONS_DIR / EFFGEN_HOME are honored at
# call time rather than import time.
DEFAULT_SESSION_DIR = _default_session_dir()


def _message_list(value: Any) -> list[Any]:
    """Return *value* as a list of messages, or empty when it is not one.

    A session file is JSON a previous build (or a hand edit) wrote, so
    ``messages`` may be any type. Every reader below counts, sums or renders it,
    and none of them should crash on a file whose shape drifted.
    """
    return value if isinstance(value, list) else []


def _message_metadata(message: Any) -> dict[str, Any]:
    """Return a message's metadata mapping, or empty when it is not a mapping."""
    if not isinstance(message, dict):
        return {}
    meta = message.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _last_message_field(messages: Any, key: str) -> Any:
    """Return *key* from the most recent message metadata that carries it."""
    for m in reversed(_message_list(messages)):
        value = _message_metadata(m).get(key)
        if value:
            return value
    return None


def _sum_message_costs(messages: Any) -> float | None:
    """Total recorded cost across a session's turns, or ``None`` if unpriced.

    A turn stamps the same per-run cost on both the user and the assistant
    message, so the reply side of each turn is what gets counted, and costs
    sharing a ``run_id`` are counted once. A cost that is not a finite number is
    not a cost: it is skipped rather than summed into a total that would then be
    reported as real spend.
    """
    total = 0.0
    seen_runs: set[str] = set()
    priced = False
    for m in _message_list(messages):
        if not isinstance(m, dict) or m.get("role") == "user":
            continue
        meta = _message_metadata(m)
        cost = meta.get("cost_usd")
        if isinstance(cost, bool) or not isinstance(cost, int | float):
            continue
        if not math.isfinite(cost):
            continue
        run_id = meta.get("run_id")
        if run_id:
            # A run id read off disk may be any JSON value, including an
            # unhashable one; key the de-duplication on its text.
            key = str(run_id)
            if key in seen_runs:
                continue
            seen_runs.add(key)
        total += float(cost)
        priced = True
    return round(total, 6) if priced else None


@dataclass
class Session:
    """
    A persistent conversation session.

    Attributes:
        session_id: Stable identifier (UUID by default).
        agent_name: Name of the owning agent (informational).
        messages: Conversation history as list of {role, content, timestamp}.
        memory: Optional memory snapshot (e.g. ShortTermMemory.to_dict()).
        metadata: Free-form metadata.
        created_at / updated_at: ISO timestamps.
        keep_thread_history: Whether every turn keeps the full conversation its
            run had. ``False``, the default, keeps it on the latest turn only
            and reduces each earlier one to its shape as a new turn arrives —
            :meth:`last_thread` reads only the latest, so nothing that is read
            is lost, and a session file grows with the conversation rather than
            with its square. ``True`` restores keeping all of them.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    keep_thread_history: bool = False

    # ------------------------------------------------------------------ messages
    def add_message(self, role: str, content: str, **meta: Any) -> None:
        """Append a message with a timestamp and refresh ``updated_at``.

        A turn arriving with the conversation its run had takes that record
        over from the turn before it, unless :attr:`keep_thread_history` says
        every turn keeps its own.

        Args:
            role: Who the message is from.
            content: The message text.
            **meta: Extra fields stored alongside the message.
        """
        if meta.get("thread") is not None and not self.keep_thread_history:
            self._reduce_earlier_threads()
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": meta or {},
        })
        self.updated_at = datetime.now().isoformat()

    def _reduce_earlier_threads(self) -> None:
        """Replace every stored thread with its shape, keeping the text.

        The shape — the version it was written at, how many steps it had and
        which kinds, in order — is the same reduction the run store makes and
        is made for the same reason: a listing can tell a two-step turn from a
        twenty-step one without the file carrying every transcript twice. What
        the turn said is in its ``content``, where it always was.
        """
        for message in _message_list(self.messages):
            if not isinstance(message, dict):
                continue
            metadata = message.get("metadata")
            if not isinstance(metadata, dict):
                continue
            saved = metadata.get("thread")
            if not isinstance(saved, dict) or not isinstance(saved.get("steps"), list):
                continue
            metadata["thread"] = {
                "version": saved.get("version"),
                "steps": len(saved["steps"]),
                "kinds": [
                    str(step.get("kind"))
                    for step in saved["steps"]
                    if isinstance(step, dict)
                ],
            }

    def add_user_message(self, content: str) -> None:
        """Append a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message."""
        self.add_message("assistant", content)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """Return the session as a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Rebuild a session from ``to_dict()`` output (unknown fields ignored)."""
        # Load forgivingly so sessions saved by older/newer effGen builds (which may
        # carry since-renamed fields) still reload instead of raising a cryptic
        # TypeError. See effgen.core._compat.load_from_dict.
        from ._compat import load_from_dict

        return load_from_dict(cls, data, label="Session")

    def save(self, sessions_dir: str | None = None) -> str:
        """Write the session to ``<sessions_dir>/<id>.json`` atomically; returns the path."""
        sessions_dir = sessions_dir or _default_session_dir()
        os.makedirs(sessions_dir, exist_ok=True)
        path = os.path.join(sessions_dir, f"{self.session_id}.json")
        self.updated_at = datetime.now().isoformat()
        # Write through a temporary file of this writer's own and rename it into
        # place, so a crash mid-write cannot leave a truncated file and two
        # writers of the same session cannot publish a mix of both.
        return atomic_write_text(
            path, json.dumps(self.to_dict(), indent=2, default=str)
        )

    @classmethod
    def load(
        cls,
        session_id: str,
        sessions_dir: str | None = None,
    ) -> "Session":
        """Load the session *session_id* from disk (``FileNotFoundError`` if absent)."""
        sessions_dir = sessions_dir or _default_session_dir()
        path = os.path.join(sessions_dir, f"{session_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session not found: {session_id}")
        with open(path) as f:
            raw = f.read()
        from ..errors import CorruptStateError

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise CorruptStateError("session", path, str(e)) from e
        try:
            return cls.from_dict(data)
        except ValueError as e:
            # Valid JSON that is not a session document (an array, a scalar, a
            # file missing a required field) — name the file, not the parser.
            raise CorruptStateError("session", path, str(e)) from e

    # ------------------------------------------------------------------ threads
    def last_thread(self) -> Any:
        """The run's steps from the most recent turn that recorded any.

        A turn stores the conversation its run had under the assistant
        message's ``thread`` metadata, so a session can be continued with the
        earlier run's structure rather than with a reading of its text. A
        session whose turns were written before that — or one whose last turn
        recorded no steps — hands back an empty thread.

        Returns:
            The thread, as :class:`~effgen.core.thread.AgentThread`.
        """
        from ._compat import thread_from_saved

        for message in reversed(_message_list(self.messages)):
            if not isinstance(message, dict) or message.get("role") == "user":
                continue
            saved = _message_metadata(message).get("thread")
            if saved:
                return thread_from_saved(
                    {"thread": saved}, label=f"session {self.session_id}"
                )
        from .thread import AgentThread

        return AgentThread()

    @classmethod
    def load_or_create(
        cls,
        session_id: str | None,
        agent_name: str = "",
        sessions_dir: str | None = None,
    ) -> "Session":
        """Load *session_id* when it exists, otherwise create a new session.

        Args:
            session_id: The session to load, or ``None`` to always create one.
            agent_name: Agent recorded on a session that is created here.
            sessions_dir: Where sessions live, defaulting to the state directory.

        Returns:
            The loaded or newly created session.
        """
        if session_id:
            try:
                return cls.load(session_id, sessions_dir)
            except FileNotFoundError:
                return cls(session_id=session_id, agent_name=agent_name)
        return cls(agent_name=agent_name)


class SessionManager:
    """
    Filesystem-backed session store living in ~/.effgen/sessions/.

    Provides list / get / delete / cleanup operations used by the CLI.
    """

    def __init__(self, sessions_dir: str | None = None) -> None:
        self.sessions_dir = os.path.abspath(
            os.path.expanduser(sessions_dir or _default_session_dir())
        )
        os.makedirs(self.sessions_dir, exist_ok=True)

    def scan(self) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Return ``(sessions, unreadable)`` for the store.

        ``sessions`` holds one summary per readable session file, newest first.
        ``unreadable`` names every file that could not be parsed, with the
        reason, so a listing never under-counts without saying so.
        """
        out: list[dict[str, Any]] = []
        unreadable: list[dict[str, str]] = []
        for fname in sorted(os.listdir(self.sessions_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.sessions_dir, fname)) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("session file is not a JSON object")
                messages = _message_list(data.get("messages"))
                meta = data.get("metadata")
                meta = meta if isinstance(meta, dict) else {}
                out.append({
                    "session_id": str(data.get("session_id") or fname[:-5]),
                    "agent_name": data.get("agent_name", ""),
                    "messages": len(messages),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "model": meta.get("model") or _last_message_field(messages, "model"),
                    "cost_usd": _sum_message_costs(messages),
                })
            except (OSError, json.JSONDecodeError, ValueError) as e:
                unreadable.append({"file": fname, "reason": str(e)})
        # Sort on the timestamp as text: a file may carry any JSON value there,
        # and comparing a string against a number would fail the whole listing.
        out.sort(key=lambda d: str(d.get("updated_at") or ""), reverse=True)
        return out, unreadable

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return a summary of every readable session, newest first.

        Files that cannot be parsed are omitted here; use :meth:`scan` to also
        get the list of unreadable files.
        """
        sessions, unreadable = self.scan()
        for entry in unreadable:
            logger.debug(
                "Skipping unreadable session file %s: %s", entry["file"], entry["reason"]
            )
        return sessions

    def get(self, session_id: str) -> Session:
        """Load and return the session *session_id*."""
        return Session.load(session_id, self.sessions_dir)

    def delete(self, session_id: str) -> bool:
        """Delete the stored session file; returns whether it existed."""
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def export(self, session_id: str, format: str = "json") -> str:
        """Render a session as ``json`` or plain ``text``."""
        session = self.get(session_id)
        if format == "json":
            return json.dumps(session.to_dict(), indent=2, default=str)
        if format == "text":
            lines = [f"Session: {session.session_id}", f"Agent: {session.agent_name}", ""]
            for m in _message_list(session.messages):
                if isinstance(m, dict):
                    lines.append(f"[{m.get('role')}] {m.get('content')}")
                else:
                    lines.append(f"[?] {m}")
            return "\n".join(lines)
        raise ValueError(f"Unsupported export format: {format}")

    def cleanup(self, older_than_days: int = 30) -> int:
        """Delete sessions whose updated_at is older than the cutoff."""
        cutoff = datetime.now() - timedelta(days=older_than_days)
        removed = 0
        for entry in self.list_sessions():
            updated = entry.get("updated_at")
            if not isinstance(updated, str) or not updated:
                # A timestamp of another type is not a timestamp; leave the
                # session alone rather than deleting it on a guess.
                continue
            try:
                ts = datetime.fromisoformat(updated)
            except ValueError:
                continue
            if ts < cutoff:
                if self.delete(entry["session_id"]):
                    removed += 1
        return removed
