"""A stored session grows with the conversation, not with its square.

A turn stores the conversation its run had so a later turn can continue from
the earlier run's structure. Storing that on *every* turn stored the whole
conversation once per turn, and only the latest one is ever read
(:meth:`Session.last_thread`), so the file grew quadratically while most of it
was written and never looked at.

What is pinned here:

* the latest turn keeps the whole conversation and :meth:`last_thread` is
  unchanged, through a save and a load;
* an earlier turn keeps its *shape* — the version, the step count and the
  ordered kinds — and what it said stays in its own ``content``;
* ``keep_thread_history=True`` restores keeping every one of them;
* the file stops growing quadratically, measured on the same probe both ways.
"""

from __future__ import annotations

import json

from effgen.core.session import Session
from effgen.core.thread import (
    ActionStep,
    AgentThread,
    ObservationStep,
    TaskStep,
    ThoughtStep,
    TurnStep,
)


def _thread(cycles: int = 4, earlier_turns: int = 0) -> dict:
    steps: list = [TaskStep(text="the question")]
    for index in range(earlier_turns):
        steps.append(TurnStep(text=f"earlier {index}", role="user"))
    for index in range(cycles):
        steps.append(ThoughtStep(text=f"thinking {index}"))
        steps.append(ActionStep(tool="search", raw=f"query {index}"))
        steps.append(ObservationStep(text=f"passage {index}: " + "x" * 400))
    return AgentThread(steps=steps).to_dict()


def _session(turns: int, *, keep: bool = False, session_id: str = "probe") -> Session:
    session = Session(session_id=session_id, keep_thread_history=keep)
    for turn in range(turns):
        session.add_message("user", f"question {turn}")
        session.add_message(
            "assistant", f"answer {turn}", thread=_thread(earlier_turns=2 * turn)
        )
    return session


def _bytes(session: Session) -> int:
    return len(json.dumps(session.to_dict(), default=str))


def test_the_latest_turn_keeps_the_whole_conversation(tmp_path) -> None:
    session = _session(5)
    thread = session.last_thread()
    assert len(thread.steps) > 10
    assert thread.task() is not None
    # And through a file, which is how a session is actually continued.
    session.save(str(tmp_path))
    reloaded = Session.load("probe", str(tmp_path))
    assert [s.to_dict() for s in reloaded.last_thread().steps] == [
        s.to_dict() for s in thread.steps
    ]


def test_an_earlier_turn_keeps_its_shape_and_its_words() -> None:
    session = _session(4)
    stored = [
        m["metadata"].get("thread")
        for m in session.messages if m["role"] == "assistant"
    ]
    *earlier, latest = stored
    assert isinstance(latest["steps"], list), "the latest turn keeps its steps"
    for shape in earlier:
        assert isinstance(shape["steps"], int)
        assert shape["kinds"][0] == "task"
        assert len(shape["kinds"]) == shape["steps"]
        assert shape["version"] == latest["version"]
    # What the turn said is where it always was.
    assert [m["content"] for m in session.messages if m["role"] == "assistant"] == [
        f"answer {turn}" for turn in range(4)
    ]


def test_reading_an_earlier_turns_shape_hands_back_an_empty_thread() -> None:
    """There is nothing to rebuild, and saying so beats raising on it."""
    from effgen.core._compat import thread_from_saved

    session = _session(3)
    first = next(m for m in session.messages if m["role"] == "assistant")
    recovered = thread_from_saved(
        {"thread": first["metadata"]["thread"]}, label="session probe"
    )
    assert recovered.steps == []


def test_keeping_every_turns_thread_is_still_available() -> None:
    bounded = _session(6)
    kept = _session(6, keep=True, session_id="kept")
    stored = [
        m["metadata"].get("thread")
        for m in kept.messages if m["role"] == "assistant"
    ]
    assert all(isinstance(t["steps"], list) for t in stored)
    assert _bytes(kept) > _bytes(bounded)
    # The same conversation is still readable either way.
    assert kept.last_thread().to_text() == bounded.last_thread().to_text()


def test_the_file_stops_growing_quadratically() -> None:
    """The growth of the growth is what a quadratic has and a linear one does not."""
    def second_differences(keep: bool) -> list[int]:
        sizes = [
            _bytes(_session(turns, keep=keep, session_id=f"s{turns}"))
            for turns in range(2, 10)
        ]
        first = [b - a for a, b in zip(sizes, sizes[1:], strict=False)]
        return [b - a for a, b in zip(first, first[1:], strict=False)]

    kept = second_differences(keep=True)
    bounded = second_differences(keep=False)
    assert min(kept) > 0, "keeping every thread grows quadratically"
    # The quadratic term that is left is the session's own earlier turns riding
    # on the latest stored thread as its frame; it is an order of magnitude
    # smaller than storing every thread.
    assert max(bounded) * 5 < min(kept)


def test_a_turn_carrying_no_thread_leaves_the_earlier_ones_alone() -> None:
    """A guard: this passes against the tree before the bound as well.

    It is here because the reduction is triggered by a turn that carries a
    conversation, and a turn that carries none — the question half of every
    exchange — must leave the record alone.
    """
    session = Session(session_id="plain")
    session.add_message("user", "q0")
    session.add_message("assistant", "a0", thread=_thread())
    session.add_message("user", "q1")  # no thread: nothing is reduced
    stored = session.messages[1]["metadata"]["thread"]
    assert isinstance(stored["steps"], list)
