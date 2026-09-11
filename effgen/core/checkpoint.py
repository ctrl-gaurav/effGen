"""
Agent checkpointing for effGen framework.

Provides CheckpointManager which can serialize agent execution state
(the run's steps, iterations, tool history, partial results, memory) to disk
as human-readable JSON. Supports filesystem (default) and SQLite backends.

A checkpoint carries the run's conversation as :class:`~effgen.core.thread.AgentThread`
data under ``thread``, versioned by that thread's own schema version, and the
transcript that thread renders to under ``scratchpad``. Both are written, so a
checkpoint this build writes still resumes on a build that only knows the
transcript, and a checkpoint an older build wrote still resumes here — see
:func:`effgen.core._compat.thread_from_saved` for what the second direction loses.

Checkpoints are JSON-serializable only (no pickle) for security.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _read_checkpoint_json(path: str) -> dict:
    """Read+parse a checkpoint JSON file, raising a clear error if corrupt."""
    from ..errors import CorruptStateError

    with open(path) as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise CorruptStateError("checkpoint", path, str(e)) from e
    if not isinstance(data, dict):
        raise CorruptStateError(
            "checkpoint",
            path,
            f"expected a JSON object of fields, got {type(data).__name__}",
        )
    return data


def _load_checkpoint(source: str, data: Any) -> "Checkpoint":
    """Build a Checkpoint from parsed data, naming *source* when it cannot."""
    from ..errors import CorruptStateError

    try:
        return Checkpoint.from_dict(data)
    except ValueError as e:
        raise CorruptStateError("checkpoint", source, str(e)) from e


def _thread_fields(thread: Any) -> tuple[dict[str, Any], str]:
    """Return ``(serialised thread, the transcript it renders)`` for *thread*.

    Accepts a thread, the data one serialises to, or nothing at all, so a caller
    that has either shape — or neither — gets a checkpoint it can resume.
    """
    if thread is None:
        return {}, ""
    if isinstance(thread, dict):
        try:
            from .thread import AgentThread

            return dict(thread), AgentThread.from_dict(thread).to_text()
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                "Checkpoint: saved thread data could not be read back (%s); "
                "storing it as given, with no transcript beside it.", e
            )
            return dict(thread), ""
    to_dict = getattr(thread, "to_dict", None)
    to_text = getattr(thread, "to_text", None)
    if callable(to_dict) and callable(to_text):
        return to_dict(), to_text()
    logger.warning(
        "Checkpoint: thread=%s is neither a thread nor thread data; the "
        "checkpoint records no steps. Pass response.metadata['thread'].",
        type(thread).__name__,
    )
    return {}, ""


@dataclass
class Checkpoint:
    """A single checkpoint snapshot."""

    checkpoint_id: str
    agent_name: str
    task: str
    iteration: int
    model: str = ""
    scratchpad: str = ""
    thread: dict[str, Any] = field(default_factory=dict)
    partial_output: str | None = None
    tool_calls: int = 0
    tokens_used: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    tool_states: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the checkpoint as a JSON-serializable dict."""
        return asdict(self)

    def to_thread(self) -> Any:
        """Return the run's conversation as an :class:`~effgen.core.thread.AgentThread`.

        A checkpoint written by this build hands back exactly the steps it was
        given. One written before a run's steps were kept is reconstructed from
        its transcript, which is lossy in the ways
        :func:`effgen.core._compat.thread_from_saved` documents.

        Returns:
            The thread, empty when the checkpoint recorded no progress.
        """
        from ._compat import thread_from_saved

        return thread_from_saved(self.to_dict(), label=f"checkpoint {self.checkpoint_id}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Rebuild a checkpoint from ``to_dict()`` output (unknown fields ignored)."""
        # Load forgivingly so checkpoints written by a different effGen build (which
        # may carry since-renamed fields) still resume instead of raising a cryptic
        # TypeError. See effgen.core._compat.load_from_dict.
        from ._compat import load_from_dict

        return load_from_dict(cls, data, label="Checkpoint")


class CheckpointManager:
    """
    Save and restore agent state to filesystem (JSON) or SQLite.

    Usage:
        mgr = CheckpointManager("./checkpoints")
        cp_id = mgr.save(checkpoint)
        cp = mgr.load(cp_id)         # by id
        cp = mgr.load_latest()       # most recent
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        backend: str = "filesystem",
    ) -> None:
        self.backend = backend
        self.checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if backend == "sqlite":
            self.db_path = os.path.join(self.checkpoint_dir, "checkpoints.db")
            self._init_sqlite()
        elif backend != "filesystem":
            raise ValueError(f"Unsupported backend: {backend}")

    # ------------------------------------------------------------------ sqlite
    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    agent_name TEXT,
                    task TEXT,
                    iteration INTEGER,
                    created_at TEXT,
                    data TEXT
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ save
    def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint and return its id."""
        if not checkpoint.checkpoint_id:
            checkpoint.checkpoint_id = self._new_id(checkpoint.agent_name)

        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.agent_name,
                        checkpoint.task,
                        checkpoint.iteration,
                        checkpoint.created_at,
                        json.dumps(checkpoint.to_dict()),
                    ),
                )
                conn.commit()
        else:
            path = os.path.join(self.checkpoint_dir, f"{checkpoint.checkpoint_id}.json")
            with open(path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            # Maintain a "latest.json" pointer for convenience
            latest = os.path.join(self.checkpoint_dir, "latest.json")
            with open(latest, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)

        return checkpoint.checkpoint_id

    # ------------------------------------------------------------------ load
    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load a checkpoint by id, or by file path."""
        # Allow passing a path directly
        if os.path.sep in checkpoint_id or checkpoint_id.endswith(".json"):
            if not os.path.exists(checkpoint_id):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
            return _load_checkpoint(checkpoint_id, _read_checkpoint_json(checkpoint_id))

        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT data FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
                if row is None:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
                try:
                    data = json.loads(row[0])
                except (json.JSONDecodeError, ValueError) as e:
                    from ..errors import CorruptStateError
                    raise CorruptStateError("checkpoint", self.db_path, str(e)) from e
                return _load_checkpoint(self.db_path, data)

        path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return _load_checkpoint(path, _read_checkpoint_json(path))

    def load_latest(self) -> Checkpoint:
        """Return the most recently created checkpoint."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT data FROM checkpoints ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    raise FileNotFoundError("No checkpoints found")
                try:
                    data = json.loads(row[0])
                except (json.JSONDecodeError, ValueError) as e:
                    from ..errors import CorruptStateError
                    raise CorruptStateError("checkpoint", self.db_path, str(e)) from e
                return _load_checkpoint(self.db_path, data)

        latest = os.path.join(self.checkpoint_dir, "latest.json")
        if os.path.exists(latest):
            return _load_checkpoint(latest, _read_checkpoint_json(latest))

        files = sorted(
            (f for f in os.listdir(self.checkpoint_dir) if f.endswith(".json")),
            key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_dir, f)),
            reverse=True,
        )
        if not files:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        return self.load(files[0].replace(".json", ""))

    # ------------------------------------------------------------------ list/delete
    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return a list of checkpoint summaries."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT checkpoint_id, agent_name, task, iteration, created_at "
                    "FROM checkpoints ORDER BY created_at DESC"
                ).fetchall()
            return [
                {
                    "checkpoint_id": r[0],
                    "agent_name": r[1],
                    "task": r[2],
                    "iteration": r[3],
                    "created_at": r[4],
                }
                for r in rows
            ]

        results: list[dict[str, Any]] = []
        for fname in os.listdir(self.checkpoint_dir):
            if not fname.endswith(".json") or fname == "latest.json":
                continue
            try:
                with open(os.path.join(self.checkpoint_dir, fname)) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"expected a JSON object of fields, got {type(data).__name__}"
                    )
                results.append({
                    "checkpoint_id": str(data.get("checkpoint_id") or fname[:-5]),
                    "agent_name": data.get("agent_name"),
                    "task": data.get("task"),
                    "iteration": data.get("iteration"),
                    "created_at": data.get("created_at"),
                })
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.debug("Skipping unreadable checkpoint file %s: %s", fname, e)
                continue
        # Sort on the timestamp as text: a file may carry any JSON value there,
        # and comparing a string against a number would fail the whole listing.
        results.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
        return results

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by id. Returns True if removed."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )
                conn.commit()
                return cur.rowcount > 0
        path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _new_id(agent_name: str) -> str:
        ts = int(time.time() * 1000)
        return f"{agent_name}-{ts}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def snapshot_agent(
        agent: Any,
        task: str,
        iteration: int,
        scratchpad: str = "",
        partial_output: str | None = None,
        tool_calls: int = 0,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
        thread: Any = None,
    ) -> Checkpoint:
        """Build a Checkpoint by snapshotting an Agent's serializable state.

        Tool *instances* are stored only by name + class (config, not
        instances), keeping the checkpoint JSON-safe.

        Args:
            agent: The agent whose state is captured.
            task: The task the agent is working on.
            iteration: The reasoning iteration reached so far.
            scratchpad: The reasoning transcript accumulated so far. Left out
                when *thread* is given, which renders it.
            partial_output: Any answer text produced before the snapshot.
            tool_calls: How many tool calls the run has made.
            tokens_used: Tokens the run has consumed.
            metadata: Extra context to store with the checkpoint.
            thread: The run's conversation, as an
                :class:`~effgen.core.thread.AgentThread` or as the data one
                serialises to. Stored whole, so resuming gets the run's steps
                back rather than a reading of their text.

        Returns:
            The checkpoint, ready to persist.
        """
        thread_data, rendered = _thread_fields(thread)
        if not scratchpad:
            scratchpad = rendered

        memory_dict: dict[str, Any] = {}
        try:
            stm = getattr(agent, "short_term_memory", None)
            if stm is not None and hasattr(stm, "to_dict"):
                memory_dict["short_term"] = stm.to_dict()
        except Exception:
            logger.debug("Failed to snapshot short-term memory for checkpoint", exc_info=True)

        # Record the model id so `resume` can reuse it (and warn on a mismatch)
        # rather than silently resuming on a different model.
        model_id = getattr(agent, "model_name", None) or ""

        tool_states: dict[str, Any] = {}
        for tname, tool in getattr(agent, "tools", {}).items():
            tool_states[tname] = {
                "name": tname,
                "class": type(tool).__name__,
                "module": type(tool).__module__,
            }

        return Checkpoint(
            checkpoint_id="",
            agent_name=getattr(agent, "name", "agent"),
            task=task,
            iteration=iteration,
            model=model_id,
            scratchpad=scratchpad,
            thread=thread_data,
            partial_output=partial_output,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            memory=memory_dict,
            tool_states=tool_states,
            metadata=metadata or {},
        )

    @staticmethod
    def restore_to_agent(agent: Any, checkpoint: Checkpoint) -> None:
        """
        Restore checkpoint state into an existing agent (memory only).

        Tool instances are *not* recreated — the agent must already be
        constructed with the same tools. Scratchpad / iteration are
        consumed by Agent.resume() to seed the next run.
        """
        stm_data = checkpoint.memory.get("short_term") if checkpoint.memory else None
        if stm_data:
            try:
                from ..memory.short_term import ShortTermMemory
                agent.short_term_memory = ShortTermMemory.from_dict(stm_data)
            except Exception:
                logger.debug("Failed to restore short-term memory from checkpoint", exc_info=True)
