"""Workspace resolution, agent construction and the run record for ``effgen code``.

:class:`CodeEngine` assembles a coding agent from parts that already ship: the
``coding`` preset's system prompt and iteration budget, its four tools with a
permission gate in front of them (:mod:`effgen.cli.code.tools`), and the ReAct
loop that feeds every tool result back into the next step. One run returns a
:class:`CodeRunResult` carrying the answer, what the agent did to the workspace,
what was withheld, and the usual token/cost/timing numbers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effgen.tools.builtin._fs import WORKSPACE_ENV_VAR, default_workspace

from .edits import EditJournal, ProposedEdit, UndoOutcome
from .git_actions import (
    CommitOutcome,
    commit_paths,
    other_staged_paths,
    relative_to_repo,
    suggest_message,
    untracked_among,
)
from .permissions import ActionRecord, PermissionGate, PermissionMode
from .project import ProjectContext, ReviewSubject, build_project_context
from .tools import build_code_tools, build_review_tools

logger = logging.getLogger(__name__)

#: The preset whose prompt, tools and iteration budget ``effgen code`` builds on.
CODE_PRESET = "coding"

#: Where a coding session's own state lives on a session record, beside the
#: conversation every ``effgen`` session stores. Additive, so ``effgen
#: sessions`` and the history views read a coding session unchanged.
CODING_METADATA_KEY = "coding"

#: Added to the preset prompt so the model works inside the one workspace root
#: and treats a gate refusal as a real constraint rather than something to retry.
_WORKSPACE_PROMPT = (
    "You are working inside the directory {workspace}. Read and write files "
    "with relative paths inside it; never write outside it. To create or change "
    "a file, use the file tool's write operation — not shell redirection such as "
    "'cat > file', 'tee', or 'sed -i' — so each change is shown as a diff before "
    "it is applied and can be undone. Run code with the code execution tool, "
    "which starts in that directory, so a relative path such as 'main.py' reads "
    "the file you just wrote. The Python REPL tool evaluates short expressions in "
    "a restricted namespace with a small import allow-list; it cannot import "
    "project files, so use the code execution tool for anything that touches the "
    "workspace. Base every claim on a tool's real output. If a tool reports that "
    "an action was not permitted, tell the user instead of retrying the same "
    "action or answering as if it had succeeded."
)

#: The system prompt a read-only review run uses in place of
#: :data:`_WORKSPACE_PROMPT`, which instructs the model to write files.
REVIEW_PROMPT = (
    "You are reviewing code in {workspace}. This run is read-only: you cannot "
    "write a file, run code or run a shell command, and no such tool is "
    "attached. Your tools read — the file tool reads, lists and searches files "
    "in the workspace, and the git tool reports the repository's status, log, "
    "branches and stat-level diffs. The change under review is included with "
    "the request, so read a file only when you need the code around a hunk, and "
    "never read the same file twice — a second read returns what you already "
    "have. Answer with the review itself: the correctness risks first, one "
    "finding per bullet, each naming the file and the line it is on, then "
    "anything else worth raising. Say plainly when something you would need is "
    "not visible to you rather than guessing at it. Describe the change you "
    "would make; do not attempt to make it."
)

#: Stop reasons that end a run holding a tool result the model never wrote up:
#: the last observation after a repeated call, or what the run had reached when
#: the model returned no answer. The coding surfaces name which one it was, so a
#: repeated file read is not mistaken for a review. ``direct_calculator_result``
#: is not here — that is a computed result the loop returned deliberately.
RECOVERED_ANSWER_SOURCES: frozenset[str] = frozenset(
    {"loop_detected", "repeated_tool_result", "null_final_from_model"}
)

#: How each of those reads in the run report.
RECOVERED_ANSWER_LABELS: dict[str, str] = {
    "loop_detected": "the last tool result, after the model repeated the same call",
    "repeated_tool_result": "a tool result the model asked for twice",
    "null_final_from_model": "what the run had reached when the model returned no answer",
}

#: The default question a review answers when the caller gave no task.
REVIEW_TASK = (
    "Review the change below. Report the correctness risks first — one finding "
    "per bullet, each naming the file and the line it is on — then anything "
    "else worth raising."
)

#: What the run reports when every action it took failed and nothing was
#: written. The model's own paragraph travels under ``partial_output``.
_ALL_ACTIONS_FAILED = (
    "The turn changed nothing: every action it took returned an error. What the "
    "model wrote about it is reported as partial progress, not as a result. "
    "Read the action log for the failures, then re-run with the cause fixed."
)


def _every_action_failed(actions: list[Any]) -> bool:
    """True when the turn ran at least one action and none of them succeeded.

    Only *allowed* actions count: an action the gate withheld, declined or
    refused never ran, so a turn that proposed edits under ``plan`` mode has
    nothing to have failed and is not this case.
    """
    ran = [a for a in actions if a.decision == "allowed"]
    return bool(ran) and all(a.outcome == "error" for a in ran)


def resolve_workspace(explicit: str | None = None) -> Path:
    """Return the directory a coding run is confined to.

    An explicit ``-w/--workspace`` wins; otherwise ``EFFGEN_WORKSPACE`` when it
    is set; otherwise the current directory. The directory is created if it does
    not exist, so a fresh workspace can be named on the command line.

    Raises:
        OSError: The directory could not be created.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    configured = default_workspace()
    if configured is not None:
        return configured
    return Path.cwd().resolve()


def workspace_execution_note(workspace: Path) -> str | None:
    """Return a note when executed code cannot reach *workspace*, else ``None``.

    A sandbox that shields ``/tmp`` with a private ``tmpfs`` but cannot confine
    writes leaves a workspace inside the system temp directory written by the
    file tool yet invisible to the code the sandbox runs — an import of a file
    the agent just created would fail with a confusing "no such module". This
    runs one short probe through the configured sandbox and reports the
    mismatch up front instead of leaving it to surface mid-loop. Any error in
    the probe returns ``None``: it informs, it never blocks a run.

    Where writes are confined, the workspace is bound into the sandbox
    read-write whatever directory it lives in, so this returns ``None``.
    """
    import asyncio

    from effgen.security.sandbox import SandboxConfig, get_sandbox

    probe = "import os\nprint('visible' if os.path.isdir(os.getcwd()) else 'hidden')\n"

    async def _probe() -> str:
        config = SandboxConfig.from_env()
        config.workdir = str(workspace)
        sandbox = await get_sandbox(config)
        result = await sandbox.run(code=probe, language="python", config=config)
        return (result.stdout or "").strip()

    try:
        verdict = asyncio.run(_probe())
    except Exception:  # noqa: BLE001 - a diagnostic never breaks the run
        return None
    if verdict != "hidden":
        return None
    return (
        f"Code executed in the sandbox cannot see {workspace}: this sandbox "
        "gives executed code a private temp directory but cannot bind the "
        "workspace into it, so a workspace under the system temp directory is "
        "not readable from it. Files written there are real, but running them "
        "will fail. Pass -w/--workspace with a directory outside the system "
        "temp directory."
    )


@contextmanager
def workspace_env(workspace: Path) -> Iterator[Path]:
    """Set ``EFFGEN_WORKSPACE`` to *workspace* for the duration of the block.

    Every file and shell tool resolves its root from that variable, so setting it
    once makes the whole run agree on a single directory even when the caller
    named it with ``-w``. The previous value is restored on exit.
    """
    previous = os.environ.get(WORKSPACE_ENV_VAR)
    os.environ[WORKSPACE_ENV_VAR] = str(workspace)
    try:
        yield workspace
    finally:
        if previous is None:
            os.environ.pop(WORKSPACE_ENV_VAR, None)
        else:
            os.environ[WORKSPACE_ENV_VAR] = previous


def undo_workspace(workspace: Path, count: int = 1) -> tuple[list[UndoOutcome], int]:
    """Reverse the last *count* applied edits in *workspace*.

    Returns the outcomes performed (newest first) and how many edits remain on
    the stack afterwards. An empty stack yields no outcomes.
    """
    journal = EditJournal(Path(workspace))
    outcomes: list[UndoOutcome] = []
    for _ in range(max(1, count)):
        outcome = journal.undo()
        if outcome is None:
            break
        outcomes.append(outcome)
    return outcomes, len(journal)


@dataclass
class CommitPlan:
    """What a commit of the run's edits would cover, before it is confirmed.

    Built by :meth:`CodeEngine.plan_commit` so the surface can show the user the
    repository, the exact paths and the message before anything is asked or run.
    A non-empty :attr:`error` means there is nothing to commit and why.
    """

    repo_root: Path | None = None
    branch: str = ""
    paths: list[str] = field(default_factory=list)
    message: str = ""
    other_staged: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def subject(self) -> str:
        """The message's first line."""
        return self.message.splitlines()[0] if self.message else ""

    def describe(self) -> str:
        """One line naming what would be committed and where."""
        if self.error:
            return self.error
        return (
            f"Commit {len(self.paths)} file(s) to {self.repo_root} on "
            f"{self.branch}: {', '.join(self.paths)}"
        )


@dataclass
class CodeRunResult:
    """The outcome of one ``effgen code`` task."""

    task: str
    answer: str
    success: bool
    reason: str
    model: str
    provider: str | None
    workspace: str
    permission_mode: str
    #: The tool-calling path the turn ran through — ``react``, ``native``,
    #: ``hybrid`` or a provider-native path. Empty when the run failed before
    #: the tool loop.
    tool_calling: str = ""
    #: Where the answer came from when the loop recovered one rather than the
    #: model writing it (``loop_detected``, ``repeated_tool_result``, ...).
    #: Empty when the model wrote the answer.
    answer_source: str = ""
    #: What ended the agent run, as the response reported it. Empty when the
    #: result was assembled from something that carries no stop reason.
    stop_reason: str = ""
    #: True when the run held no tool that writes, runs or executes anything.
    read_only: bool = False
    #: What a read-only review was asked to look at, or ``None``.
    review: dict[str, Any] | None = None
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float | None = None
    duration_s: float = 0.0
    partial: bool = False
    #: What the run had reached when the loop stopped it — tool output and
    #: reasoning, never an answer. Empty when the run answered or failed
    #: outright.
    partial_output: str = ""
    actions: list[ActionRecord] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    # Every edit the run proposed, in order. ``applied`` is false for the ones
    # that were only proposed (plan mode, a decline, a withheld action).
    diffs: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    repo: dict[str, Any] | None = None
    commit: dict[str, Any] | None = None

    # ``effgen.ui.render.summary_line`` reads a run's metrics off these names,
    # so the coding footer is the same one ``effgen run`` prints.
    @property
    def execution_time(self) -> float:
        """Wall-clock seconds the run took."""
        return self.duration_s

    @property
    def tokens_used(self) -> int:
        """Total prompt + completion tokens for the run."""
        return self.tokens

    @property
    def metadata(self) -> dict[str, Any]:
        """The subset of run metadata the shared renderers read."""
        return {"reason": self.reason, "cost_usd": self.cost_usd, "partial": self.partial}

    @property
    def withheld(self) -> list[ActionRecord]:
        """Actions the permission mode did not allow."""
        return [a for a in self.actions if a.decision == "withheld"]

    @property
    def refused(self) -> list[ActionRecord]:
        """Actions blocked because their target was outside the workspace."""
        return [a for a in self.actions if a.decision == "refused"]

    @property
    def hit_iteration_cap(self) -> bool:
        """True when the loop stopped at its iteration cap without an answer."""
        return self.reason in ("max_iterations_partial", "max_iterations_exhausted")

    @property
    def outcome(self) -> str:
        """``"answered"``, ``"stopped"`` or ``"failed"`` for this run.

        The same three words :attr:`effgen.core.agent.AgentResponse.outcome`
        uses, so a script reading ``effgen code --json`` branches the same way
        it would on a library run.
        """
        if self.success:
            return "answered"
        from effgen.core.agent_response import STOPPED_REASONS

        return "stopped" if (self.stop_reason or self.reason) in STOPPED_REASONS else "failed"

    @property
    def recovered_answer(self) -> bool:
        """True when the loop recovered the answer instead of the model writing it.

        The run stops holding the last tool observation when a model keeps
        repeating a call, and holding its own text when the model returns none.
        Neither is an answer, so the surfaces name which one the run was
        holding rather than showing it as a result.
        """
        return self.answer_source in RECOVERED_ANSWER_SOURCES

    def to_dict(self) -> dict[str, Any]:
        """Return the result as the JSON document ``--json`` prints."""
        return {
            "task": self.task,
            "answer": self.answer,
            "partial_output": self.partial_output,
            "success": self.success,
            "reason": self.reason,
            "stop_reason": self.stop_reason,
            "outcome": self.outcome,
            "model": self.model,
            "provider": self.provider,
            "workspace": self.workspace,
            "permission_mode": self.permission_mode,
            "tool_calling": self.tool_calling,
            "answer_source": self.answer_source,
            "read_only": self.read_only,
            "review": self.review,
            "repo": self.repo,
            "commit": self.commit,
            "files_written": list(self.files_written),
            "diffs": [dict(d) for d in self.diffs],
            "actions": [a.to_dict() for a in self.actions],
            "withheld": [a.to_dict() for a in self.withheld],
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
        }


class CodeEngine:
    """Builds and runs the coding agent behind ``effgen code``.

    Args:
        model: Model id, optionally ``provider:model`` prefixed.
        provider: Explicit provider for a bare model id.
        workspace: The directory the run is confined to.
        mode: The active :class:`~effgen.cli.code.permissions.PermissionMode`.
        mode_explicit: True when the caller named the mode on the command line.
        interactive: Whether a human can be asked to confirm an action.
        max_iterations: Override the preset's iteration cap.
        temperature: Override the preset's temperature.
        max_tokens: Per-call output-token cap, when the model needs one raised.
        confirm: Confirmation callback, passed through to the gate.
        on_event: Called with each :class:`ActionRecord` as it is decided.
        on_diff: Called with each :class:`ProposedEdit` before the write is
            decided, so its diff can be shown before it touches disk.
        project: The workspace's repository state, layout and project brief,
            appended to the system prompt. ``None`` builds it from the workspace.
        review: What a read-only review is looking at. Set it and the run holds
            only reading tools, writes nothing, and carries the subject as
            context instead of reaching for a shell to find it.
        session_id: A persistent conversation session to continue and append to.
            The stored turns are recalled into the agent's memory and each new
            turn is written back.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        workspace: Path,
        mode: PermissionMode,
        mode_explicit: bool = False,
        interactive: bool = False,
        max_iterations: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        confirm: Callable[[str], str] | None = None,
        on_event: Callable[[ActionRecord], None] | None = None,
        on_diff: Callable[[ProposedEdit], None] | None = None,
        project: ProjectContext | None = None,
        review: ReviewSubject | None = None,
        session_id: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.workspace = Path(workspace)
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.project = project
        self.review = review
        self.session_id = session_id
        self.gate = PermissionGate(
            mode,
            self.workspace,
            mode_explicit=mode_explicit,
            interactive=interactive,
            confirm=confirm,
            on_event=on_event,
            on_diff=on_diff,
            journal=EditJournal(self.workspace),
        )
        self._agent: Any = None

    @property
    def mode(self) -> PermissionMode:
        """The permission mode this run is using."""
        return self.gate.mode

    @property
    def read_only(self) -> bool:
        """True when this run holds no tool that writes, runs or executes."""
        return self.review is not None

    def system_prompt(self) -> str:
        """Return the preset prompt, the mode instruction and the project context.

        A review run replaces the workspace instruction — which tells the model
        to write files through the file tool — with the read-only review brief.
        """
        from effgen.presets import get_preset

        base = get_preset(CODE_PRESET).system_prompt
        instruction = (
            REVIEW_PROMPT if self.read_only else _WORKSPACE_PROMPT
        ).format(workspace=self.workspace)
        parts = [base, instruction]
        if self.project is not None:
            parts.append(self.project.as_prompt())
        return "\n\n".join(parts)

    def compose_review_task(self, task: str) -> str:
        """Return *task* with the review subject in front of it.

        The subject is handed to the model as context because the tools a review
        holds cannot produce a diff: the only route to one is a shell, and a
        read-only run has none.
        """
        if self.review is None or not self.review.text:
            return task
        return (
            f"{task or REVIEW_TASK}\n\n--- the change under review "
            f"({self.review.describe().lower()}) ---\n{self.review.text}"
        )

    def load_project(self, *, refresh: bool = False) -> ProjectContext:
        """Return the workspace's project context, building it once on demand.

        *refresh* rebuilds it — the repository moved on, files were added — and
        updates the live agent's system prompt so the next turn sees the new
        state. Rebuilding the context is the point; an agent that carries no
        editable configuration simply keeps the prompt it was built with.
        """
        if self.project is None or refresh:
            self.project = build_project_context(self.workspace)
            config = getattr(self._agent, "config", None)
            if config is not None and hasattr(config, "system_prompt"):
                config.system_prompt = self.system_prompt()
        return self.project

    def build_agent(self) -> Any:
        """Construct the agent, reusing the ``coding`` preset's configuration.

        A review run is given the read-only tool set; a ``session_id`` attaches
        the persistent session, whose stored turns the agent recalls.
        """
        from effgen.core.agent import Agent, AgentConfig
        from effgen.presets import get_preset

        preset = get_preset(CODE_PRESET)
        config = AgentConfig(
            name="code-agent",
            # A failed turn is reported in the session, not as a traceback.
            raise_on_error=False,
            model=self.model,
            provider=self.provider,
            tools=(
                build_review_tools(self.gate) if self.read_only
                else build_code_tools(self.gate)
            ),
            system_prompt=self.system_prompt(),
            max_iterations=(
                self.max_iterations if self.max_iterations is not None
                else preset.max_iterations
            ),
            temperature=(
                self.temperature if self.temperature is not None else preset.temperature
            ),
            max_tokens=self.max_tokens,
            enable_memory=preset.enable_memory,
            enable_sub_agents=False,
        )
        self._agent = (
            Agent(config=config, session_id=self.session_id)
            if self.session_id else Agent(config=config)
        )
        return self._agent

    def run(self, task: str) -> CodeRunResult:
        """Run *task* to completion and return its :class:`CodeRunResult`.

        ``EFFGEN_WORKSPACE`` is set for the duration of the run so every tool
        resolves the same root, and restored afterwards. A review run carries
        its subject into the task.
        """
        agent = self._agent or self.build_agent()
        composed = self.compose_review_task(task) if self.read_only else task
        with workspace_env(self.workspace):
            response = agent.run(composed)
        return self.result_from_response(task, response)

    # -- the persistent session ---------------------------------------------

    def coding_state(
        self,
        *,
        files_in_context: Iterable[str] = (),
        files_written: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Return the coding half of a session record.

        The conversation is stored by the session itself; this is what a coding
        session additionally needs to pick up where it left off — where it was
        working, under which permissions, on which files.
        """
        from datetime import datetime

        return {
            "workspace": str(self.workspace),
            "permission_mode": self.mode.value,
            "files_in_context": list(files_in_context),
            "files_written": list(files_written),
            "model": self.model,
            "provider": self.provider,
            "updated_at": datetime.now().isoformat(),
        }

    def save_coding_state(
        self,
        *,
        files_in_context: Iterable[str] = (),
        files_written: Iterable[str] = (),
    ) -> None:
        """Write the coding state onto the attached session, if there is one.

        Additive: everything else in the record is left as it is, so listing,
        ``sessions show`` and the history views read the session unchanged. A
        store that cannot be written is logged and does not fail the turn.
        """
        session = getattr(self._agent, "session", None)
        if session is None:
            return
        session.metadata[CODING_METADATA_KEY] = self.coding_state(
            files_in_context=files_in_context, files_written=files_written
        )
        try:
            session.save()
        except Exception as exc:  # noqa: BLE001 - a store failure never fails a turn
            logger.warning("Could not save the coding session state: %s", exc)

    @contextmanager
    def review_turn(self, subject: ReviewSubject) -> Iterator[None]:
        """Make the live agent read-only for the duration of the block.

        The agent's tool dict, its system prompt and the gate's mode are swapped
        for the review set, the review brief and ``plan``, and all three are put
        back in ``finally`` — including when the turn raises — so the next turn
        has its writing tools again. The loop reads ``self.tools`` per call, so
        the swap takes effect for the turn that runs inside the block.
        """
        agent = self._agent
        previous_review = self.review
        previous_mode = self.gate.mode
        previous_tools = dict(getattr(agent, "tools", {}) or {}) if agent else {}
        config = getattr(agent, "config", None)
        previous_prompt = getattr(config, "system_prompt", None)
        self.review = subject
        self.gate.mode = PermissionMode.PLAN
        try:
            if agent is not None:
                agent.tools = {t.metadata.name: t for t in build_review_tools(self.gate)}
                if config is not None and hasattr(config, "system_prompt"):
                    config.system_prompt = self.system_prompt()
            yield
        finally:
            self.review = previous_review
            self.gate.mode = previous_mode
            if agent is not None:
                agent.tools = previous_tools
                if config is not None and previous_prompt is not None:
                    config.system_prompt = previous_prompt

    # -- git ----------------------------------------------------------------

    def plan_commit(
        self, rel_paths: list[str], *, task: str = "", message: str | None = None
    ) -> CommitPlan:
        """Describe the commit that would record *rel_paths*, without running it.

        *rel_paths* are workspace-relative (what the run reports as written) and
        are mapped to repository-relative paths. Nothing is staged or committed
        here; the caller shows the plan, then calls :meth:`perform_commit`.

        Args:
            rel_paths: Workspace-relative paths the run reported as written.
            task: The task text, used as the commit body when no *message* is given.
            message: A commit message to use as-is instead of a suggested one.

        Returns:
            A :class:`CommitPlan` naming the repository, branch, mapped paths and
            message; a non-empty ``error`` means there is nothing to commit.
        """
        repo = (self.project.repo if self.project is not None else None)
        if repo is None:
            repo = build_project_context(self.workspace, include_brief=False).repo
        if repo is None:
            return CommitPlan(
                error=(
                    f"{self.workspace} is not inside a git repository, so there is "
                    "nothing to commit to."
                )
            )
        paths = relative_to_repo(self.workspace, repo.root, rel_paths)
        if not paths:
            return CommitPlan(
                repo_root=repo.root,
                branch=repo.branch,
                error="No files were written in this repository, so there is nothing to commit.",
            )
        if message and message.strip():
            text = message.strip()
        else:
            text = suggest_message(paths, task, untracked_among(repo.root, paths))
        return CommitPlan(
            repo_root=repo.root,
            branch=repo.branch,
            paths=paths,
            message=text,
            other_staged=other_staged_paths(repo.root, paths),
        )

    def perform_commit(self, plan: CommitPlan) -> CommitOutcome:
        """Ask the gate about *plan*, and commit when it is allowed.

        The gate decides exactly as it does for a write or a shell command: in
        ``ask`` mode the human answers a y/N prompt, in ``plan`` mode the commit
        is withheld, and only ``--yes`` commits without asking. The decision is
        recorded on the run's action log either way.
        """
        if plan.error or plan.repo_root is None:
            return CommitOutcome(False, plan.message, plan.error or "Nothing to commit.")
        # Phrased like the other gated actions ("Write main.py (+2/-0)"), so the
        # confirm prompt and the action tick read the same way.
        question = (
            f"Commit {len(plan.paths)} file(s) to {plan.repo_root.name} on "
            f"{plan.branch} as \"{plan.subject}\""
        )
        decision = self.gate.request("git", question, target=", ".join(plan.paths))
        if not decision.allowed:
            return CommitOutcome(
                False, plan.message,
                decision.reason or "not confirmed; nothing was committed.",
                paths=list(plan.paths),
            )
        outcome = commit_paths(plan.repo_root, plan.paths, plan.message)
        self.gate.note_outcome(
            decision.record,
            "ok" if outcome.success else "error",
            outcome.detail or (f"commit {outcome.commit}" if outcome.commit else ""),
        )
        return outcome

    def result_from_response(self, task: str, response: Any) -> CodeRunResult:
        """Assemble a :class:`CodeRunResult` from *response* and the gate's log.

        The agent run is separated from record assembly so the REPL, which drives
        the agent itself (under a live status render, and sometimes with the mode
        overridden for a single turn), builds the same result the single-shot
        path does.
        """
        metadata = response.metadata or {}
        answer = response.output or ""
        success = bool(response.success)
        reason = str(metadata.get("reason", ""))
        partial = bool(metadata.get("partial"))
        partial_output = str(metadata.get("partial_output") or "")
        answer_source = str(metadata.get("answer_source", "") or "")
        error = metadata.get("error")

        actions = list(self.gate.actions)
        if success and _every_action_failed(actions) and not self.gate.files_written:
            # Every action the turn decided on came back an error and nothing
            # was written, yet the model wrote a paragraph about it. The
            # paragraph is truthful and there is nothing in the text to
            # distinguish it from an answer — but a turn whose every action
            # failed and which changed nothing did not do what it was asked, so
            # it is reported as the partial outcome the iteration cap uses
            # rather than as a completed success.
            failed = actions[-1]
            success = False
            partial = True
            partial_output = answer
            reason = "all_actions_failed"
            answer = _ALL_ACTIONS_FAILED
            error = error or {
                "type": "NoActionSucceeded",
                "category": "all_actions_failed",
                "message": _ALL_ACTIONS_FAILED,
                "failed_actions": len(actions),
                "last_action": failed.summary,
                "last_error": failed.detail or failed.reason,
                "retryable": True,
            }

        return CodeRunResult(
            task=task,
            answer=answer,
            success=success,
            reason=reason,
            model=self.model,
            provider=self.provider or getattr(response, "provider", None),
            workspace=str(self.workspace),
            permission_mode=self.mode.value,
            tool_calling=str(metadata.get("tool_calling_strategy", "") or ""),
            answer_source=answer_source,
            stop_reason=str(getattr(response, "stop_reason", "") or ""),
            read_only=self.read_only,
            review=self.review.to_dict() if self.review is not None else None,
            iterations=int(getattr(response, "iterations", 0) or 0),
            tool_calls=int(getattr(response, "tool_calls", 0) or 0),
            tokens=int(getattr(response, "tokens_used", 0) or 0),
            cost_usd=metadata.get("cost_usd"),
            duration_s=float(getattr(response, "execution_time", 0.0) or 0.0),
            partial=partial,
            partial_output=partial_output,
            actions=actions,
            files_written=self.gate.files_written,
            diffs=list(self.gate.edits),
            error=error,
            repo=(
                self.project.repo.to_dict()
                if self.project is not None and self.project.repo is not None
                else None
            ),
        )
