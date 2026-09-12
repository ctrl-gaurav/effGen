"""What a parent run lets a child run start with.

A parent that hands work to a sub-agent, a team stage or a workflow node has to
decide what the child knows. The way that has always been done is
concatenation: paste the earlier answers into the child's question and let the
child work out which part is the job and which part is background. That reads
badly to a model, grows without bound, and leaves nothing a caller can inspect
afterwards — the child's conversation was one long string it was asked.

A projection is the other way round. It **selects** steps from the parent's
:class:`~effgen.core.thread.AgentThread` and the child starts its own thread
with them, as the steps they were. The child's question stays the question it
was given, the selection is the parent's decision rather than a fixed rule, and
what the child was shown is recorded on the child's own thread where a reader,
a checkpoint and a context budget can all see it.

Four are built in, and a caller with another rule subclasses
:class:`ThreadProjection`:

- :class:`NoParentContext` — nothing. The child sees only its own question.
  This is the default everywhere, and it is what every pattern did before
  projections existed, so a caller who asks for nothing gets the run they
  always got.
- :class:`ParentTask` — the job the parent was given, as one user turn, so a
  child doing a piece of it knows what the whole of it was.
- :class:`ParentAnswers` — the parent's task, plus what each child that has
  already finished answered, as assistant turns.
- :class:`LastCycles` — the parent's task, plus the last few complete cycles of
  the parent's own work, as the thoughts, calls and results they were.

Every projection returns a thread a provider will accept: a call whose result
was not selected is dropped with it, so the child never opens on a tool call
nothing answered.

The child's context budget counts these steps like any others
(:mod:`effgen.core.thread_budget`), and its compaction policy gives them up
before it gives up the child's own question
(:mod:`effgen.core.thread_compaction`) — a projection is bounded by the
parent's choice on the way in, and by the child's budget once it is there.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .thread import (
    ActionStep,
    AgentThread,
    DelegationStep,
    ObservationStep,
    Step,
    ThoughtStep,
    TurnStep,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LastCycles",
    "NoParentContext",
    "ParentAnswers",
    "ParentTask",
    "ThreadProjection",
]


def _pair_safe(steps: list[Step]) -> list[Step]:
    """Drop any half of a call/result pair whose other half is not here.

    A conversation carrying a tool call nothing answered, or a result answering
    a call that is not in it, is one a provider rejects. A selection is free to
    cut anywhere, so the pairing is settled here rather than in each rule.

    Args:
        steps: The selected steps, in order.

    Returns:
        The same steps with every unpaired call and unpaired result removed.
    """
    call_ids = {
        step.call_id for step in steps
        if isinstance(step, ActionStep) and step.call_id
    }
    answered = {
        step.call_id for step in steps
        if isinstance(step, ObservationStep) and step.call_id
    }
    kept: list[Step] = []
    for step in steps:
        if isinstance(step, ActionStep) and (step.call_id or "") not in answered:
            continue
        if isinstance(step, ObservationStep) and (step.call_id or "") not in call_ids:
            continue
        kept.append(step)
    return kept


class ThreadProjection(ABC):
    """Which of a parent run's steps a child run starts with.

    Subclass this and implement :meth:`select` to write a rule of your own. The
    surrounding :meth:`project` settles the call/result pairing and logs what
    was carried, so a rule only has to say what it wants.
    """

    #: A stable name for logs and for a caller reading back what was used.
    name: str = "projection"

    @abstractmethod
    def select(self, parent: AgentThread, *, task: str) -> list[Step]:
        """The steps to carry from *parent* into the child's thread.

        Args:
            parent: The parent run's conversation, as far as it has got.
            task: The question the child is about to be asked, so a rule may
                take it into account.

        Returns:
            The selected steps, oldest first. They are copied into the child's
            thread, not shared with the parent's.
        """

    def project(self, parent: AgentThread | None, *, task: str) -> list[Step]:
        """The steps the child opens with, ready to be framed.

        Args:
            parent: The parent's conversation, or ``None`` when there is no
                parent run to project from.
            task: The question the child is about to be asked.

        Returns:
            The carried steps, with every unpaired tool call removed. An empty
            list means the child starts on its own question alone.
        """
        if parent is None:
            return []
        carried = _pair_safe(list(self.select(parent, task=task)))
        if carried:
            logger.info(
                "[projection] the child opens with %d step(s) chosen by %s",
                len(carried), self.name,
            )
        return carried


class NoParentContext(ThreadProjection):
    """Carry nothing: the child sees only the question it was asked.

    The default for every pattern, and what all of them did before projections
    existed.
    """

    name = "none"

    def select(self, parent: AgentThread, *, task: str) -> list[Step]:
        """Nothing.

        Args:
            parent: The parent's conversation, ignored.
            task: The child's question, ignored.

        Returns:
            An empty list.
        """
        return []


def _task_turn(parent: AgentThread) -> list[Step]:
    """The parent's question as one user turn, or nothing when it has none."""
    task_step = parent.task()
    if task_step is None or not task_step.text.strip():
        return []
    return [TurnStep(text=task_step.text, role="user")]


class ParentTask(ThreadProjection):
    """Carry the job the parent was given, as one user turn.

    A child working on a piece of a larger job answers it differently when it
    knows what the larger job was — and a task stated as an earlier turn is
    background, where the same text pasted into the child's question competes
    with it.
    """

    name = "parent-task"

    def select(self, parent: AgentThread, *, task: str) -> list[Step]:
        """The parent's task step, restated as a user turn.

        Args:
            parent: The parent's conversation.
            task: The child's question, ignored.

        Returns:
            One turn, or an empty list when the parent has no task step.
        """
        return _task_turn(parent)


class ParentAnswers(ThreadProjection):
    """Carry the parent's task and what its finished children answered.

    This is the rule a sequence of children wants: the third child sees what
    the first two produced, as turns of the conversation it is joining, rather
    than as a block of text in front of its own question. A child that failed
    is carried as what went wrong, because the next child usually should not
    repeat it.

    Args:
        include_task: Whether to open with the parent's own question.
        limit: How many of the most recent finished children to carry, or
            ``None`` for all of them.
    """

    name = "parent-answers"

    def __init__(self, *, include_task: bool = True, limit: int | None = None) -> None:
        self.include_task = include_task
        self.limit = limit

    def select(self, parent: AgentThread, *, task: str) -> list[Step]:
        """The parent's task, then one assistant turn per finished child.

        Args:
            parent: The parent's conversation.
            task: The child's question, ignored.

        Returns:
            The carried turns, oldest first.
        """
        steps: list[Step] = _task_turn(parent) if self.include_task else []
        done = parent.delegations()
        if self.limit is not None:
            done = done[-self.limit:] if self.limit > 0 else []
        for step in done:
            body = step.output if step.success else (step.error or "did not complete")
            if not str(body).strip():
                continue
            steps.append(
                TurnStep(text=f"{step.child_id}: {body}", role="assistant")
            )
        return steps


class LastCycles(ThreadProjection):
    """Carry the parent's task and the last *n* complete cycles of its own work.

    The rule for a child that is continuing the parent's reasoning rather than
    doing a separate piece of it. A cycle is a thought, the call it led to and
    the result that answered it; an incomplete one at the end is left behind
    rather than carried as a call nothing replied to.

    Args:
        n: How many of the most recent complete cycles to carry.
        include_task: Whether to open with the parent's own question.

    Raises:
        ValueError: *n* is negative. A negative count has no reading that is
            different from zero, so it is refused rather than guessed at. Pass
            0 to carry no cycles.
    """

    name = "last-cycles"

    def __init__(self, n: int = 2, *, include_task: bool = True) -> None:
        if int(n) < 0:
            raise ValueError(
                f"LastCycles(n={n}) cannot carry a negative number of cycles. "
                f"Pass n=0 to carry none, or a positive count."
            )
        self.n = int(n)
        self.include_task = include_task

    def select(self, parent: AgentThread, *, task: str) -> list[Step]:
        """The parent's task, then its last *n* answered cycles.

        Args:
            parent: The parent's conversation.
            task: The child's question, ignored.

        Returns:
            The carried steps, oldest first.
        """
        steps: list[Step] = _task_turn(parent) if self.include_task else []
        if self.n == 0:
            return steps
        answered = {
            step.call_id for step in parent.steps
            if isinstance(step, ObservationStep) and step.call_id
        }
        cycles: list[list[Step]] = []
        current: list[Step] = []
        for step in parent.steps:
            if isinstance(step, ThoughtStep):
                current = [step]
                continue
            if isinstance(step, ActionStep):
                if (step.call_id or "") in answered:
                    current.append(step)
                else:
                    current = []
                continue
            if isinstance(step, ObservationStep):
                if current:
                    current.append(step)
                    cycles.append(current)
                current = []
        for cycle in cycles[-self.n:]:
            steps.extend(cycle)
        return steps


#: The built-in rules by the name a caller may pass instead of an instance.
_BY_NAME: dict[str, type[ThreadProjection]] = {
    "none": NoParentContext,
    "parent-task": ParentTask,
    "parent-answers": ParentAnswers,
    "last-cycles": LastCycles,
}


def resolve_projection(value: Any) -> ThreadProjection:
    """The projection a caller asked for, whichever way they said it.

    Args:
        value: ``None`` for the default, one of the built-in names, a
            :class:`ThreadProjection` subclass, or an instance of one.

    Returns:
        The projection to use.

    Raises:
        ValueError: The value names no built-in rule. The message lists the
            names that exist, so the caller can pick one.
        TypeError: The value is not a name, a class or a projection. Pass a
            :class:`ThreadProjection` instance, or one of the built-in names.
    """
    if value is None:
        return NoParentContext()
    if isinstance(value, ThreadProjection):
        return value
    if isinstance(value, type) and issubclass(value, ThreadProjection):
        return value()
    if isinstance(value, str):
        cls = _BY_NAME.get(value)
        if cls is None:
            raise ValueError(
                f"{value!r} is not a projection this release knows. "
                f"Pass one of {sorted(_BY_NAME)}, or a ThreadProjection instance."
            )
        return cls()
    raise TypeError(
        f"a projection cannot be built from {type(value).__name__}. "
        f"Pass a ThreadProjection instance, a subclass, or one of "
        f"{sorted(_BY_NAME)}."
    )


def delegation_of(
    child_id: str,
    *,
    role: str,
    task: str,
    response: Any,
    output: str | None = None,
) -> DelegationStep:
    """The record a parent keeps of one child's work.

    Args:
        child_id: What the parent calls this piece of work.
        role: What the child was doing, in the parent's own vocabulary.
        task: The question the child was asked.
        response: Whatever the child returned. Its ``metadata["thread"]`` is
            kept when it has one; anything else is recorded without a thread,
            which is what a workflow node that is not an agent produces.
        output: The answer text, when the parent reads it from somewhere other
            than ``response.output``.

    Returns:
        The step to append to the parent's thread.
    """
    meta = getattr(response, "metadata", None) or {}
    thread = meta.get("thread") if isinstance(meta, dict) else None
    if not isinstance(thread, AgentThread):
        thread = None
    success = bool(getattr(response, "success", True))
    text = output if output is not None else str(getattr(response, "output", "") or "")
    error: str | None = None
    if not success:
        detail = meta.get("error") if isinstance(meta, dict) else None
        if isinstance(detail, dict):
            error = f"{detail.get('type', 'AgentError')}: {detail.get('message', text)}"
        else:
            error = str(detail or text or "the child did not complete")
    return DelegationStep(
        child_id=str(child_id),
        role=role,
        task=task,
        thread=thread,
        output=text,
        success=success,
        error=error,
    )
