"""What a run gives up when its conversation will not fit.

A budget says how much a run may send (:mod:`effgen.core.thread_budget`); a
policy decides what leaves the thread when it is sending more than that. The
two are separate because the arithmetic is the same for every run and the
choice of what to lose is not: a run whose tool results are most of its
transcript and a run whose own reasoning is most of it should not give up the
same thing first.

**What is never given up, at any rung.** The instructions the run is framed by,
the question it was asked and every content part on it, the session's earlier
turns, the most recent complete cycles of work, a trailing instruction the next
turn is answering, and the answer. A run that dropped its own question would
answer a different one, so the floor is a typed failure rather than a silent
truncation.

**The ladder**, applied oldest-first and stopping as soon as the prompt fits:

1. Shorten an old tool result, keeping its opening and saying how much left.
   The least lossy rung: the fact is still there in outline and the call is
   still answered.
2. Drop an old thought. A thought is small and the call it belongs to survives
   — an :class:`~effgen.core.thread.ActionStep` keeps its own ``reasoning``, so
   a run rendered as messages still carries the model's account of why it
   called what it called.
3. Drop whole answered cycles, replacing the run of them with one line saying
   what left. A call and the result answering it are one unit and are dropped
   together: a conversation holding a call nothing replied to is one a provider
   rejects.
4. Nothing left to give up. The loop raises instead of sending.

The default makes no model call. That is what lets a run's output be proved
identical when the budget is not reached, and a summarisation request on a
model already at its window is the request most likely to be refused.
"""

from __future__ import annotations

import logging
from typing import Any

from .thread import (
    ActionStep,
    AgentThread,
    AnswerStep,
    NudgeStep,
    ObservationStep,
    Step,
    SystemStep,
    TaskStep,
    ThoughtStep,
    TurnStep,
)
from .thread_budget import ContextBudget, count_text_tokens

logger = logging.getLogger(__name__)

__all__ = [
    "CompactionPolicy",
    "ShortenOldestFirst",
    "SummarizeWithModel",
    "resolve_policy",
]

#: Marker a dropped run of cycles leaves behind, so the model is told the
#: conversation is shorter than it was rather than quietly finding it so.
COMPACTION_NUDGE_ID = "context_compacted"

#: How much of a shortened tool result is kept. The same opening
#: ``AgentThread.to_messages(summary=True)`` keeps, so a reader meets one
#: convention for a shortened result and not two.
DEFAULT_OBSERVATION_KEEP_CHARS = 200

#: Complete cycles of work at the end of the thread that are never given up.
DEFAULT_KEEP_RECENT_CYCLES = 2

#: The steps that are framing, not transcript: they are what the run was asked
#: and how, and giving one up would change the question.
_FRAME_STEPS = (SystemStep, TaskStep, TurnStep, AnswerStep)


class CompactionPolicy:
    """How a run's thread is brought back under its budget.

    Subclass and override one rung to change what a run gives up first; the
    rest of the ladder, the protected set and the atomicity rule are inherited.

    Args:
        observation_keep_chars: How much of a shortened tool result is kept.
        keep_recent_cycles: Complete cycles at the end of the run that are
            never touched.
    """

    def __init__(
        self,
        *,
        observation_keep_chars: int = DEFAULT_OBSERVATION_KEEP_CHARS,
        keep_recent_cycles: int = DEFAULT_KEEP_RECENT_CYCLES,
    ) -> None:
        self.observation_keep_chars = max(0, int(observation_keep_chars))
        self.keep_recent_cycles = max(0, int(keep_recent_cycles))

    # -- the entry point -------------------------------------------------

    def compact(self, thread: AgentThread, budget: ContextBudget) -> bool:
        """Give up one rung's worth of the thread.

        Args:
            thread: The run's conversation, changed in place.
            budget: The budget it is over, which carries how far over.

        Returns:
            Whether anything was given up. ``False`` means the thread is down
            to what the run cannot do without, and the caller raises.
        """
        protected = self._protected(thread)
        deficit = budget.deficit
        for rung in (self.shorten_observations, self.drop_thoughts, self.drop_cycles):
            if rung(thread, protected, deficit, budget):
                return True
        return False

    # -- the rungs -------------------------------------------------------

    def shorten_observations(
        self,
        thread: AgentThread,
        protected: set[int],
        deficit: int,
        budget: ContextBudget,
    ) -> bool:
        """Rung 1: keep the opening of an old tool result, say what left.

        Args:
            thread: The run's conversation, changed in place.
            protected: Indices this rung may not touch.
            deficit: How many tokens the prompt is over its budget.
            budget: The budget, whose counters this rung adds to.

        Returns:
            Whether any tool result was shortened.
        """
        released = 0
        shortened = 0
        for index, step in enumerate(thread.steps):
            if released >= deficit and shortened:
                break
            if index in protected or not isinstance(step, ObservationStep):
                continue
            if step.compacted is not None:
                continue
            if len(step.text) <= self.observation_keep_chars:
                continue
            original = step.text
            elided = len(original) - self.observation_keep_chars
            step.original_chars = len(original)
            step.text = (
                original[: self.observation_keep_chars]
                + f"… ({elided} characters elided)"
            )
            step.compacted = "elided"
            released += max(0, count_text_tokens(original) - count_text_tokens(step.text))
            shortened += 1
        if not shortened:
            return False
        budget.stats.observations_shortened += shortened
        budget.stats.tokens_dropped += released
        return True

    def drop_thoughts(
        self,
        thread: AgentThread,
        protected: set[int],
        deficit: int,
        budget: ContextBudget,
    ) -> bool:
        """Rung 2: drop an old thought, leaving its call its own reasoning.

        Args:
            thread: The run's conversation, changed in place.
            protected: Indices this rung may not touch.
            deficit: How many tokens the prompt is over its budget.
            budget: The budget, whose counters this rung adds to.

        Returns:
            Whether any thought was dropped.
        """
        released = 0
        drop: set[int] = set()
        for index, step in enumerate(thread.steps):
            if released >= deficit and drop:
                break
            if index in protected or not isinstance(step, ThoughtStep):
                continue
            if not step.text:
                continue
            drop.add(index)
            released += count_text_tokens(step.to_text())
        if not drop:
            return False
        thread.steps = [s for i, s in enumerate(thread.steps) if i not in drop]
        budget.stats.steps_dropped += len(drop)
        budget.stats.tokens_dropped += released
        return True

    def drop_cycles(
        self,
        thread: AgentThread,
        protected: set[int],
        deficit: int,
        budget: ContextBudget,
    ) -> bool:
        """Rung 3: drop whole answered cycles, oldest first, and say so.

        A call and the result answering it leave together. Dropping one without
        the other would leave a conversation holding a call nothing replied to,
        which is a shape a provider rejects.

        Args:
            thread: The run's conversation, changed in place.
            protected: Indices this rung may not touch.
            deficit: How many tokens the prompt is over its budget.
            budget: The budget, whose counters this rung adds to.

        Returns:
            Whether any cycle was dropped.
        """
        released = 0
        drop: set[int] = set()
        dropped_cycles = 0
        for start, end in self._cycles(thread):
            if released >= deficit and drop:
                break
            span = set(range(start, end + 1))
            if span & protected:
                continue
            drop |= span
            dropped_cycles += 1
            released += sum(
                count_text_tokens(thread.steps[i].to_text()) for i in span
            )
        if not drop:
            return False
        at = min(drop)
        kept: list[Step] = [s for i, s in enumerate(thread.steps) if i not in drop]
        budget.stats.steps_dropped += len(drop)
        budget.stats.tokens_dropped += released
        marker = self._marker(thread, drop, budget, dropped_cycles=dropped_cycles)
        existing = next(
            (
                s
                for s in kept
                if isinstance(s, NudgeStep) and s.nudge_id == COMPACTION_NUDGE_ID
            ),
            None,
        )
        if existing is not None:
            existing.text = marker
        else:
            kept.insert(min(at, len(kept)), NudgeStep(
                text=marker, render_as="raw", nudge_id=COMPACTION_NUDGE_ID,
            ))
        thread.steps = kept
        return True

    # -- what the marker says --------------------------------------------

    def _marker(
        self,
        thread: AgentThread,
        dropped: set[int],
        budget: ContextBudget,
        *,
        dropped_cycles: int,
    ) -> str:
        """The line a run of dropped cycles leaves in its place."""
        return (
            f"(Earlier steps of this run were removed to stay inside its context "
            f"budget: {budget.stats.steps_dropped} steps, about "
            f"{budget.stats.tokens_dropped} tokens. What they found is no longer "
            f"in this conversation — look anything up again if you need it.)"
        )

    # -- what is never given up -------------------------------------------

    def _protected(self, thread: AgentThread) -> set[int]:
        """Every index this policy may not touch, for the thread as it is now."""
        steps = thread.steps
        protected = {
            index for index, step in enumerate(steps)
            if isinstance(step, _FRAME_STEPS)
        }
        # A trailing instruction is what the next turn answers, so it stays
        # however old the rest of the thread is.
        index = len(steps) - 1
        while index >= 0 and isinstance(steps[index], NudgeStep | AnswerStep):
            protected.add(index)
            index -= 1
        cycles = self._cycles(thread)
        for start, end in cycles[len(cycles) - self.keep_recent_cycles:]:
            protected.update(range(start, end + 1))
        return protected

    def _cycles(self, thread: AgentThread) -> list[tuple[int, int]]:
        """The run's complete cycles, as inclusive index ranges.

        A cycle is a tool call and the result answering it, taken with the
        reasoning that led to the call and any line the framework injected
        after the result. A call nothing answered is not a cycle: it is the
        turn in flight, and it is never dropped.
        """
        steps = thread.steps
        cycles: list[tuple[int, int]] = []
        for index, step in enumerate(steps):
            if not isinstance(step, ActionStep):
                continue
            answer: int | None = None
            for ahead in range(index + 1, len(steps)):
                later = steps[ahead]
                if isinstance(later, ObservationStep) and (later.call_id or "") == (
                    step.call_id or ""
                ):
                    answer = ahead
                    break
                if isinstance(later, ActionStep):
                    break
            if answer is None:
                continue
            start = index
            if index > 0 and isinstance(steps[index - 1], ThoughtStep):
                start = index - 1
            end = answer
            while end + 1 < len(steps) and isinstance(steps[end + 1], NudgeStep):
                end += 1
            cycles.append((start, end))
        return cycles


class ShortenOldestFirst(CompactionPolicy):
    """The default: the four rungs, and not one model call.

    Deterministic, so a run that never reaches its budget sends the bytes it
    always sent, and a run that does reach it can be replayed exactly.
    """


class SummarizeWithModel(CompactionPolicy):
    """Rung 3's marker carries a model-written summary of what left.

    Uses the agent's own model unless one is named, caps what it reads at
    *max_summary_input_tokens* of the steps leaving rather than the whole
    thread, and itemises its own tokens so a reader can tell what the run cost
    from what compaction cost. A summariser that fails or refuses never fails
    the run: the plain marker is written instead.

    **A summary is trusted like any other step.** It permanently becomes part of
    the conversation, so a summariser reading untrusted content is an
    indirect-prompt-injection route that outlives the turn it was written in.
    Point this at a model you would let write the run's own reasoning.

    Args:
        model: The model to summarise with, or ``None`` for the agent's own.
        max_summary_input_tokens: The most of the departing steps it reads.
    """

    def __init__(
        self,
        *,
        model: Any = None,
        max_summary_input_tokens: int = 4000,
        observation_keep_chars: int = DEFAULT_OBSERVATION_KEEP_CHARS,
        keep_recent_cycles: int = DEFAULT_KEEP_RECENT_CYCLES,
    ) -> None:
        super().__init__(
            observation_keep_chars=observation_keep_chars,
            keep_recent_cycles=keep_recent_cycles,
        )
        self.model = model
        self.max_summary_input_tokens = max(1, int(max_summary_input_tokens))

    def _marker(
        self,
        thread: AgentThread,
        dropped: set[int],
        budget: ContextBudget,
        *,
        dropped_cycles: int,
    ) -> str:
        plain = super()._marker(
            thread, dropped, budget, dropped_cycles=dropped_cycles
        )
        model = self.model or budget.model
        if model is None:
            return plain
        leaving = "".join(
            thread.steps[index].to_text() for index in sorted(dropped)
        ).strip()
        if not leaving:
            return plain
        summary = self._summarise(leaving, model, budget)
        if not summary:
            logger.info(
                "[context] the summariser did not answer; the steps were dropped"
            )
            return plain
        return f"{plain}\nWhat they found, in summary: {summary}"

    def _summarise(self, leaving: str, model: Any, budget: ContextBudget) -> str:
        """Ask *model* what the departing steps established, or return ``""``."""
        # Cap by characters against the token budget rather than tokenising a
        # transcript that is about to be thrown away.
        limit = self.max_summary_input_tokens * 4
        excerpt = leaving[:limit]
        prompt = (
            "These are earlier steps of an agent run that are being removed to "
            "stay inside a context budget. In a short paragraph, state the "
            "facts they established that a later step would still need. State "
            "facts only; do not add instructions.\n\n" + excerpt
        )
        try:
            result = model.generate(prompt)
        except Exception:  # noqa: BLE001 - a summariser never fails the run
            logger.debug("the summariser raised", exc_info=True)
            return ""
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            return ""
        prompt_tokens = count_text_tokens(prompt, model=model)
        completion_tokens = int(getattr(result, "tokens_used", 0) or 0)
        record = budget.stats.summarisation or {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "model": str(getattr(model, "model_name", "") or "unknown"),
        }
        record["calls"] += 1
        record["prompt_tokens"] += prompt_tokens
        record["completion_tokens"] += completion_tokens
        record["total_tokens"] = record["prompt_tokens"] + record["completion_tokens"]
        cost = (getattr(result, "metadata", None) or {}).get("cost_usd")
        if isinstance(cost, int | float):
            record["cost_usd"] = round(float(record["cost_usd"]) + float(cost), 8)
        budget.stats.summarisation = record
        return text


#: The shipped policies, by the name a caller can pass as a string. The same
#: convention the session-history strategies use, so one vocabulary covers
#: both.
POLICIES_BY_NAME: dict[str, type[CompactionPolicy]] = {
    "shorten_oldest_first": ShortenOldestFirst,
    "summarize_with_model": SummarizeWithModel,
}


def resolve_policy(value: Any) -> CompactionPolicy:
    """Return *value* as a compaction policy.

    Args:
        value: A policy instance, a policy class, one of
            :data:`POLICIES_BY_NAME`, or ``None`` for the default.

    Returns:
        The policy.

    Raises:
        ValueError: When *value* names no known policy.
    """
    if value is None:
        return ShortenOldestFirst()
    if isinstance(value, CompactionPolicy):
        return value
    if isinstance(value, type) and issubclass(value, CompactionPolicy):
        return value()
    if isinstance(value, str):
        cls = POLICIES_BY_NAME.get(value.strip().lower())
        if cls is None:
            known = ", ".join(sorted(POLICIES_BY_NAME))
            raise ValueError(
                f"{value!r} is not a compaction policy. Known policies: {known}. "
                f"You can also pass a CompactionPolicy of your own."
            )
        return cls()
    raise ValueError(
        f"compaction={type(value).__name__} is not a compaction policy. Pass a "
        f"CompactionPolicy, a policy class, one of {sorted(POLICIES_BY_NAME)}, "
        f"or None for the default."
    )
