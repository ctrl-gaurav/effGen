"""Durable checkpoints for workflow runs.

A workflow that spans several agents can run for minutes and cost real money.
When the process dies part way through — a crash, a kill, a machine reboot —
the work already paid for is gone and the run starts from the top. A
checkpoint store records what has completed so a later run can pick up from
there.

The unit of progress is the level, not the node: nodes in one topological
level run concurrently, so a checkpoint is written when a level finishes and
records every node that reached a terminal state. Resuming replays no
completed node, and retries anything that failed or never started.

What is stored is the *state* of a run, never the graph itself. Agents hold
sockets, model handles and credentials, none of which survive a process
boundary, so the caller rebuilds the same :class:`~effgen.core.workflow.WorkflowDAG`
and hands it the run id::

    from effgen import FileCheckpointStore, WorkflowDAG

    store = FileCheckpointStore()
    dag = build_my_workflow()          # the same graph as before
    result = dag.run("draft the memo", checkpoint=store, run_id="memo-42")

Re-running that line after a crash reuses whatever "memo-42" had finished. A
run id that the store has never seen simply starts from the beginning, so the
same call works for both cases and there is no separate resume path to get
wrong.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..utils.atomic_file import atomic_write_text

__all__ = [
    "CheckpointStore",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "WorkflowCheckpoint",
    "default_checkpoint_dir",
]


def _jsonable(value: Any) -> Any:
    """Return *value* if json can write it, else its string form.

    A node's output is usually text, but a node is free to return anything its
    agent produced. Refusing to checkpoint because one node returned an object
    would lose the whole run's progress over a detail, so anything unknown is
    stored as text and the loss is confined to that one value.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _thread_data(value: Any) -> Any:
    """One node's conversation as plain data, however it was handed in.

    Args:
        value: An :class:`~effgen.core.thread.AgentThread`, the dict one
            serialises to, or ``None``.

    Returns:
        The serialised thread, or ``None``.
    """
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        written: Any = value.to_dict()
        return written
    return _jsonable(value)


@dataclass
class WorkflowCheckpoint:
    """The state of one workflow run, as far as it got.

    Attributes:
        run_id: The caller's identifier for this run. Two runs sharing an id
            are the same run continued, which is the whole mechanism.
        workflow: The workflow's name, recorded so a mismatch can be reported.
        node_ids: Every node in the graph when the checkpoint was written.
            Resuming into a graph with different nodes is refused.
        completed: ``{node_id: output}`` for nodes that finished successfully.
        skipped: Node ids that were skipped, with the reason.
        failed: ``{node_id: error}`` for nodes that ran and failed.
        outputs: The output map as the run had it, keyed as the run keys it.
        threads: ``{node_id: thread}`` as plain data — each node's own
            conversation, the steps it took rather than a rendering of them, so
            a resumed run can still say how a node it is not re-running reached
            its answer, and a failed node can be read back after the process
            that ran it is gone.
        tasks: ``{node_id: task}`` — what each node was actually asked, which
            the task string alone does not record once upstream outputs have
            been folded into it.
        updated_at: Unix time of the last write.
        metadata: Anything the caller wants carried alongside.
    """

    run_id: str
    workflow: str = ""
    node_ids: list[str] = field(default_factory=list)
    completed: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    threads: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, str] = field(default_factory=dict)
    updated_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def thread_for(self, node_id: str) -> Any:
        """One node's conversation, rebuilt from what was stored.

        Args:
            node_id: The node to read.

        Returns:
            The node's :class:`~effgen.core.thread.AgentThread`, or ``None``
            when nothing was stored for it — an older checkpoint, a node that
            is not an agent, or a node that never ran.
        """
        from .thread import AgentThread

        stored = self.threads.get(node_id)
        if isinstance(stored, AgentThread):
            return stored
        if not isinstance(stored, dict):
            return None
        return AgentThread.from_dict(stored)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "run_id": self.run_id,
            "workflow": self.workflow,
            "node_ids": list(self.node_ids),
            "completed": {k: _jsonable(v) for k, v in self.completed.items()},
            "skipped": dict(self.skipped),
            "failed": dict(self.failed),
            "outputs": {k: _jsonable(v) for k, v in self.outputs.items()},
            "threads": {k: _thread_data(v) for k, v in self.threads.items()},
            "tasks": {k: str(v) for k, v in self.tasks.items()},
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowCheckpoint:
        """Rebuild a checkpoint from :meth:`to_dict` output.

        Unknown keys are ignored and missing ones take their defaults, so a
        checkpoint written by an older version still loads.
        """
        return cls(
            run_id=str(data.get("run_id", "")),
            workflow=str(data.get("workflow", "")),
            node_ids=list(data.get("node_ids") or []),
            completed=dict(data.get("completed") or {}),
            skipped=dict(data.get("skipped") or {}),
            failed=dict(data.get("failed") or {}),
            outputs=dict(data.get("outputs") or {}),
            threads=dict(data.get("threads") or {}),
            tasks={k: str(v) for k, v in (data.get("tasks") or {}).items()},
            updated_at=float(data.get("updated_at") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )

    def is_done(self, node_id: str) -> bool:
        """Whether *node_id* reached a terminal state that resuming should keep.

        A failed node is not done: resuming retries it, which is the point of
        resuming after the failure has been addressed.
        """
        return node_id in self.completed or node_id in self.skipped


@runtime_checkable
class CheckpointStore(Protocol):
    """Where workflow checkpoints are kept.

    Implement this to keep checkpoints somewhere else — a database, an object
    store, a queue. Three methods, and only :meth:`save` is on the hot path.
    """

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Write *checkpoint*, replacing any earlier state for its run id."""
        ...

    def load(self, run_id: str) -> WorkflowCheckpoint | None:
        """Return the checkpoint for *run_id*, or None if there is none."""
        ...

    def delete(self, run_id: str) -> None:
        """Remove the checkpoint for *run_id*. Not an error if absent."""
        ...


class InMemoryCheckpointStore:
    """A checkpoint store that lives and dies with the process.

    Useful in tests and for a resume that only has to survive a failed node
    rather than a failed process. It deliberately does not persist: for that,
    use :class:`FileCheckpointStore`.
    """

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowCheckpoint] = {}

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Write *checkpoint*, replacing any earlier state for its run id."""
        self._runs[checkpoint.run_id] = checkpoint

    def load(self, run_id: str) -> WorkflowCheckpoint | None:
        """Return the checkpoint for *run_id*, or None if there is none."""
        return self._runs.get(run_id)

    def delete(self, run_id: str) -> None:
        """Remove the checkpoint for *run_id*. Not an error if absent."""
        self._runs.pop(run_id, None)

    def list_runs(self) -> list[str]:
        """Return the run ids this store holds."""
        return sorted(self._runs)


def default_checkpoint_dir() -> Path:
    """Return the directory file checkpoints are written to.

    ``EFFGEN_WORKFLOW_DIR`` overrides it, which is what test suites and
    containers set so a run cannot write into a developer's home directory.
    """
    override = os.getenv("EFFGEN_WORKFLOW_DIR")
    if override:
        return Path(override)
    return Path.home() / ".effgen" / "workflows"


class FileCheckpointStore:
    """A checkpoint store backed by one JSON file per run.

    Writes go through the shared atomic-write helper, so a crash leaves either
    the previous checkpoint or the new one and never a half-written file —
    which matters, because the crash this guards against is exactly the one
    likely to happen mid-write.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else default_checkpoint_dir()

    def _path(self, run_id: str) -> Path:
        # A run id reaches the filesystem, so it may not walk out of the
        # directory it is meant to be confined to.
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
        if not safe or safe.strip(".") == "":
            safe = "run"
        return self.directory / f"{safe}.json"

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Write *checkpoint* atomically, replacing any earlier state."""
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint.updated_at = time.time()
        atomic_write_text(
            self._path(checkpoint.run_id),
            json.dumps(checkpoint.to_dict(), indent=2),
        )

    def load(self, run_id: str) -> WorkflowCheckpoint | None:
        """Return the checkpoint for *run_id*, or None if there is none.

        A file that is unreadable or not valid JSON reads as no checkpoint: a
        corrupt file should cost the run its saved progress, not its ability
        to run at all.
        """
        path = self._path(run_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return WorkflowCheckpoint.from_dict(data)

    def delete(self, run_id: str) -> None:
        """Remove the checkpoint for *run_id*. Not an error if absent."""
        self._path(run_id).unlink(missing_ok=True)

    def list_runs(self) -> list[str]:
        """Return the run ids this store holds."""
        if not self.directory.is_dir():
            return []
        found = []
        for path in sorted(self.directory.glob("*.json")):
            checkpoint = self.load(path.stem)
            if checkpoint is not None and checkpoint.run_id:
                found.append(checkpoint.run_id)
        return found
