"""What a run gives up when its conversation will not fit, and what it keeps.

What is pinned here:

* the question, the instructions the run is framed by, the session's earlier
  turns, the most recent complete cycles, a trailing instruction and the answer
  are never given up, at any rung;
* **a call and the result answering it leave together** — over two hundred
  randomly generated threads, compacting to nothing still leaves no call
  unanswered, because a conversation holding one is a shape a provider rejects;
* the frame survives on the same terms: over two hundred more generated
  threads, and over a question carrying a picture, the question, the
  framing instructions, the earlier turns and the answer are all still
  there once the policy has given up everything it can;
* the rungs fire in order: an old tool result is shortened before an old
  thought is dropped, and a whole cycle goes only when neither is left;
* a thread with nothing left to give up says so, which is what makes the loop
  raise instead of sending;
* the shortened result is still the result — the transcript is what the model
  was sent, and the step still answers its own call;
* a summarising policy that cannot reach its model writes the plain marker and
  the run carries on.
"""

from __future__ import annotations

import random

import pytest

from effgen.core.thread import (
    ActionStep,
    AgentThread,
    AnswerStep,
    NudgeStep,
    ObservationStep,
    SystemStep,
    TaskStep,
    ThoughtStep,
    TurnStep,
)
from effgen.core.thread_budget import ContextBudget
from effgen.core.thread_compaction import (
    COMPACTION_NUDGE_ID,
    CompactionPolicy,
    ShortenOldestFirst,
    SummarizeWithModel,
    resolve_policy,
)

LONG = "result text " * 200


def _thread(cycles: int = 6, *, frame: bool = True, chars: int = 2400) -> AgentThread:
    """A run with *cycles* complete cycles of work, framed the way a run is."""
    steps: list = []
    if frame:
        steps += [
            SystemStep(text="you are helpful", source="persona"),
            SystemStep(text="you hold one tool", source="contract"),
            TurnStep(text="what did we say before?", role="user"),
            TurnStep(text="we said this", role="assistant"),
            TaskStep(text="the question the run is answering"),
        ]
    for index in range(cycles):
        steps.append(ThoughtStep(text=f"reasoning number {index} about the work"))
        steps.append(ActionStep(tool="search", raw=f"query {index}"))
        steps.append(ObservationStep(text=f"passage {index}: " + "x" * chars))
    return AgentThread(steps=steps)


def _budget(deficit: int = 400) -> ContextBudget:
    """A budget already measured as *deficit* tokens over."""
    budget = ContextBudget(budget_tokens=100, window_tokens=4096)
    budget.last_measured = 100 + deficit
    return budget


def _kept(thread: AgentThread, kind: type) -> list:
    return [step for step in thread.steps if isinstance(step, kind)]


# --------------------------------------------------------------------------- #
# What is never given up
# --------------------------------------------------------------------------- #


def test_the_frame_and_the_question_are_never_given_up() -> None:
    thread = _thread(cycles=8)
    policy = ShortenOldestFirst()
    budget = _budget(deficit=100_000)
    for _ in range(20):
        if not policy.compact(thread, budget):
            break
    assert len(_kept(thread, SystemStep)) == 2
    assert len(_kept(thread, TaskStep)) == 1
    assert len(_kept(thread, TurnStep)) == 2
    assert thread.task() is not None
    assert thread.task().text == "the question the run is answering"


def test_the_most_recent_cycles_survive_everything() -> None:
    thread = _thread(cycles=8)
    policy = ShortenOldestFirst(keep_recent_cycles=2)
    budget = _budget(deficit=100_000)
    for _ in range(20):
        if not policy.compact(thread, budget):
            break
    kept = _kept(thread, ObservationStep)
    assert len(kept) >= 2
    # The two most recent results are whole; nothing older than them is.
    assert kept[-1].compacted is None
    assert kept[-2].compacted is None


def test_a_trailing_instruction_and_the_answer_survive() -> None:
    thread = _thread(cycles=6)
    thread.append(NudgeStep(text="answer now, please", nudge_id="continue"))
    thread.append(AnswerStep(text="the answer", stop_reason="final_answer"))
    policy = ShortenOldestFirst()
    budget = _budget(deficit=100_000)
    for _ in range(20):
        if not policy.compact(thread, budget):
            break
    assert any(
        isinstance(s, NudgeStep) and s.nudge_id == "continue" for s in thread.steps
    )
    assert len(_kept(thread, AnswerStep)) == 1


# --------------------------------------------------------------------------- #
# The ladder, in order
# --------------------------------------------------------------------------- #


def test_an_old_result_is_shortened_before_anything_is_dropped() -> None:
    thread = _thread(cycles=6)
    before = len(thread.steps)
    budget = _budget(deficit=200)
    assert ShortenOldestFirst().compact(thread, budget) is True
    assert len(thread.steps) == before  # nothing left the thread
    shortened = [s for s in _kept(thread, ObservationStep) if s.compacted == "elided"]
    assert shortened
    assert shortened[0].original_chars > len(shortened[0].text)
    assert "characters elided" in shortened[0].text
    assert budget.stats.observations_shortened == len(shortened)


def test_an_old_thought_goes_next_and_its_call_keeps_its_reasoning() -> None:
    thread = _thread(cycles=6, chars=10)  # results too short for rung 1
    policy = ShortenOldestFirst()
    budget = _budget(deficit=200)
    assert policy.compact(thread, budget) is True
    assert len(_kept(thread, ThoughtStep)) < 6
    # Every call still stands, with the reasoning it carries of its own.
    assert len(_kept(thread, ActionStep)) == 6
    assert thread.unanswered_call_ids() == []


def test_whole_cycles_go_last_and_leave_one_line_saying_so() -> None:
    thread = _thread(cycles=8, chars=10)
    policy = ShortenOldestFirst()
    budget = _budget(deficit=100_000)
    rounds = 0
    while policy.compact(thread, budget) and rounds < 20:
        rounds += 1
    markers = [
        s for s in thread.steps
        if isinstance(s, NudgeStep) and s.nudge_id == COMPACTION_NUDGE_ID
    ]
    assert len(markers) == 1, "a run of dropped cycles leaves one line, not many"
    assert "context budget" in markers[0].text
    assert budget.stats.steps_dropped > 0


def test_a_thread_with_nothing_left_to_give_up_says_so() -> None:
    """This is what makes the loop raise instead of sending."""
    thread = AgentThread(steps=[
        SystemStep(text="framing"),
        TaskStep(text="the question"),
    ])
    assert ShortenOldestFirst().compact(thread, _budget(deficit=100_000)) is False


def test_a_run_with_no_frame_still_keeps_its_recent_work() -> None:
    thread = _thread(cycles=4, frame=False)
    policy = ShortenOldestFirst()
    budget = _budget(deficit=100_000)
    while policy.compact(thread, budget):
        pass
    assert thread.steps  # never emptied
    assert thread.unanswered_call_ids() == []


# --------------------------------------------------------------------------- #
# Atomicity — the invariant a provider enforces
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(200))
def test_no_call_is_ever_left_unanswered(seed: int) -> None:
    """Over randomly shaped threads, compacted as far as the policy will go."""
    rng = random.Random(seed)
    steps: list = [TaskStep(text="q")]
    if rng.random() < 0.5:
        steps.insert(0, SystemStep(text="framing"))
    for index in range(rng.randint(1, 9)):
        if rng.random() < 0.8:
            steps.append(ThoughtStep(text=f"thought {index}"))
        steps.append(ActionStep(tool=rng.choice(["a", "b"]), raw=f"in {index}"))
        if rng.random() < 0.9:  # sometimes the call is still in flight
            steps.append(ObservationStep(text="o" * rng.randint(1, 900)))
        if rng.random() < 0.2:
            steps.append(NudgeStep(text="carry on", nudge_id="continue"))
    thread = AgentThread(steps=steps)
    before = set(thread.unanswered_call_ids())
    policy = ShortenOldestFirst()
    budget = _budget(deficit=100_000)
    rounds = 0
    while policy.compact(thread, budget) and rounds < 30:
        rounds += 1
        assert set(thread.unanswered_call_ids()) <= before, (
            "compaction left a call nothing replied to"
        )
    assert set(thread.unanswered_call_ids()) <= before


@pytest.mark.parametrize("seed", range(200))
def test_the_frame_survives_every_shape_of_thread(seed: int) -> None:
    """The frame is kept whatever the conversation around it looks like.

    The fixed-shape test above pins one thread. This one generates the thread —
    how many cycles, which of them the model reasoned about, whether the run
    carries a persona, earlier turns or an answer — and asserts the same thing
    over every one of them: what the run was asked, and the framing it was
    asked in, is still there after the policy has given up everything it can.
    """
    rng = random.Random(10_000 + seed)
    steps: list = []
    personas = rng.randint(0, 2)
    for index in range(personas):
        steps.append(SystemStep(text=f"framing {index}", source="persona"))
    turns = rng.randint(0, 4)
    for index in range(turns):
        steps.append(TurnStep(
            text=f"earlier turn {index}",
            role="user" if index % 2 == 0 else "assistant",
        ))
    task = TaskStep(text=f"the question, seed {seed}")
    steps.append(task)
    for index in range(rng.randint(1, 10)):
        if rng.random() < 0.7:
            steps.append(ThoughtStep(text=f"reasoning {index} " * rng.randint(1, 30)))
        steps.append(ActionStep(tool=rng.choice(["search", "calculator"]),
                                raw=f"query {index}"))
        if rng.random() < 0.9:
            steps.append(ObservationStep(text="p" * rng.randint(1, 3000)))
        if rng.random() < 0.25:
            steps.append(NudgeStep(text="carry on", nudge_id="continue"))
    answered = rng.random() < 0.4
    if answered:
        steps.append(AnswerStep(text="the answer"))
    thread = AgentThread(steps=steps)

    policy = ShortenOldestFirst()
    budget = _budget(deficit=1_000_000)
    rounds = 0
    while policy.compact(thread, budget) and rounds < 50:
        rounds += 1

    assert len(_kept(thread, SystemStep)) == personas, "a framing instruction left"
    assert len(_kept(thread, TurnStep)) == turns, "an earlier turn of the session left"
    assert len(_kept(thread, AnswerStep)) == (1 if answered else 0)
    kept_task = thread.task()
    assert kept_task is not None and kept_task.text == task.text
    assert _kept(thread, TaskStep) == [task], "the question was replaced, not kept"


@pytest.mark.parametrize("seed", range(60))
def test_a_thread_of_content_parts_keeps_them(seed: int) -> None:
    """A question carrying a picture keeps the picture, however far it compacts."""
    rng = random.Random(20_000 + seed)
    from effgen.core.messages import ImagePart, TextPart

    parts = [TextPart(text="look at this")] + [
        ImagePart(image=b"\x89PNG\r\n" + bytes(index), mime="image/png")
        for index in range(rng.randint(1, 3))
    ]
    steps: list = [SystemStep(text="framing"), TaskStep(text="what is this?", parts=parts)]
    for index in range(rng.randint(2, 8)):
        steps.append(ActionStep(tool="search", raw=f"q{index}"))
        steps.append(ObservationStep(text="p" * rng.randint(500, 3000)))
    thread = AgentThread(steps=steps)
    policy = ShortenOldestFirst()
    budget = _budget(deficit=1_000_000)
    rounds = 0
    while policy.compact(thread, budget) and rounds < 50:
        rounds += 1
    task = thread.task()
    assert task is not None
    assert len(task.parts) == len(parts), "a content part on the question left"


# --------------------------------------------------------------------------- #
# What a shortened result still is
# --------------------------------------------------------------------------- #


def test_the_transcript_is_what_the_model_was_sent() -> None:
    thread = _thread(cycles=6)
    ShortenOldestFirst().compact(thread, _budget(deficit=400))
    # to_text is a pure concatenation of each step's own text, so a compacted
    # thread renders exactly the bytes the next prompt carries.
    assert thread.to_text() == "".join(step.to_text() for step in thread.steps)
    for step in _kept(thread, ObservationStep):
        assert step.to_text().endswith(step.text)


def test_a_shortened_result_still_answers_its_own_call() -> None:
    thread = _thread(cycles=6)
    ShortenOldestFirst().compact(thread, _budget(deficit=400))
    messages = thread.to_messages()
    call_ids = {
        part.tool_call_id
        for message in messages for part in message.content
        if getattr(part, "type", "") == "tool_call"
    }
    result_ids = {
        part.tool_call_id
        for message in messages for part in message.content
        if getattr(part, "type", "") == "tool_result"
    }
    assert call_ids == result_ids


def test_a_shortened_result_round_trips_as_plain_data() -> None:
    thread = _thread(cycles=6)
    ShortenOldestFirst().compact(thread, _budget(deficit=400))
    back = AgentThread.from_dict(thread.to_dict())
    assert back.to_text() == thread.to_text()
    assert [s.to_dict() for s in back.steps] == [s.to_dict() for s in thread.steps]


def test_a_reader_that_ignores_the_flag_still_reads_the_result() -> None:
    """The flag is why a later release can add a shortening this one has not."""
    step = ObservationStep.from_dict(
        {"kind": "observation", "text": "kept", "call_id": "c1"}
    )
    assert (step.compacted, step.original_chars) == (None, 0)
    assert step.text == "kept"


# --------------------------------------------------------------------------- #
# Resolving a policy, and the summarising one
# --------------------------------------------------------------------------- #


def test_a_policy_resolves_from_a_name_an_instance_or_a_class() -> None:
    assert isinstance(resolve_policy(None), ShortenOldestFirst)
    assert isinstance(resolve_policy("shorten_oldest_first"), ShortenOldestFirst)
    assert isinstance(resolve_policy("summarize_with_model"), SummarizeWithModel)
    assert isinstance(resolve_policy(SummarizeWithModel), SummarizeWithModel)
    mine = ShortenOldestFirst(keep_recent_cycles=5)
    assert resolve_policy(mine) is mine
    with pytest.raises(ValueError):
        resolve_policy("nothing_of_the_sort")
    with pytest.raises(ValueError):
        resolve_policy(17)


def test_a_summariser_that_answers_puts_its_words_in_the_marker() -> None:
    class _Summariser:
        model_name = "summariser"
        calls = 0

        def generate(self, prompt, config=None, **kwargs):
            type(self).calls += 1

            class _R:
                text = "the run had looked up four passages about harbours."
                tokens_used = 11
                metadata = {"cost_usd": 0.0001}

            return _R()

    summariser = _Summariser()
    thread = _thread(cycles=8, chars=10)
    policy = SummarizeWithModel(model=summariser)
    budget = _budget(deficit=100_000)
    rounds = 0
    while policy.compact(thread, budget) and rounds < 20:
        rounds += 1
    marker = next(
        s for s in thread.steps
        if isinstance(s, NudgeStep) and s.nudge_id == COMPACTION_NUDGE_ID
    )
    assert "harbours" in marker.text
    record = budget.stats.summarisation
    assert record is not None
    assert record["calls"] >= 1
    assert record["total_tokens"] == (
        record["prompt_tokens"] + record["completion_tokens"]
    )
    assert record["model"] == "summariser"


def test_a_summariser_that_fails_never_fails_the_run() -> None:
    class _Broken:
        model_name = "broken"

        def generate(self, prompt, config=None, **kwargs):
            raise RuntimeError("the summariser is down")

    thread = _thread(cycles=8, chars=10)
    policy = SummarizeWithModel(model=_Broken())
    budget = _budget(deficit=100_000)
    rounds = 0
    while policy.compact(thread, budget) and rounds < 20:
        rounds += 1
    marker = next(
        s for s in thread.steps
        if isinstance(s, NudgeStep) and s.nudge_id == COMPACTION_NUDGE_ID
    )
    assert "context budget" in marker.text
    assert budget.stats.summarisation is None


def test_a_policy_of_ones_own_overrides_one_rung_and_inherits_the_rest() -> None:
    class _NeverShortens(ShortenOldestFirst):
        def shorten_observations(self, thread, protected, deficit, budget):
            return False

    thread = _thread(cycles=6)
    before = len(_kept(thread, ThoughtStep))
    policy: CompactionPolicy = _NeverShortens()
    assert policy.compact(thread, _budget(deficit=400)) is True
    assert not [s for s in _kept(thread, ObservationStep) if s.compacted]
    assert len(_kept(thread, ThoughtStep)) < before
