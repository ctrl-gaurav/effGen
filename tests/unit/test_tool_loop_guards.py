"""The repeat guards a tool-calling loop applies between turns.

:class:`~effgen.core.agent_tool_loop.NativeToolLoop` is the policy both the
blocking loop and the streaming loop consult, so it is exercised here directly:
each decision is driven from the state that produces it rather than inferred
from a whole run.
"""

from __future__ import annotations

import pytest

from effgen.core.agent_runtime import NUDGE_HAVE_ANSWER, NUDGE_HAVE_RESULTS
from effgen.core.agent_tool_loop import (
    FUZZY_LOOP_THRESHOLD,
    FUZZY_LOOP_THRESHOLD_DATA,
    MAX_BATCH_TOOL_RUNS,
    NUDGE_AFTER_CALLS,
    NativeToolLoop,
)
from effgen.tools.base_tool import ToolCategory


class _Meta:
    def __init__(self, category):
        self.category = category
        self.name = "t"


class _Tool:
    def __init__(self, category=ToolCategory.COMPUTATION, name="calculator"):
        self.metadata = _Meta(category)
        self.name = name


#: A budget generous enough for the declared drift thresholds to be reachable.
#: :meth:`NativeToolLoop.fuzzy_threshold` bounds the count by the run's own
#: iteration cap, so a loop built with a cap of 10 would answer 9 whatever the
#: tool's category says.
ROOMY_CAP = 40


def _loop(cap=ROOMY_CAP, **tools):
    return NativeToolLoop(tools or {"calculator": _Tool()}, nudge_cap=cap)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------
def test_json_arguments_compare_equal_whatever_their_key_order():
    loop = _loop()
    a = loop.normalize_input('{"b": 2, "a": 1}')
    b = loop.normalize_input('{"a": 1, "b": 2}')
    assert a == b


def test_non_json_input_is_compared_as_trimmed_text():
    loop = _loop()
    assert loop.normalize_input("  6*7 ") == "6*7"


# ---------------------------------------------------------------------------
# Repeat detection
# ---------------------------------------------------------------------------
def test_the_same_call_twice_is_an_exact_loop():
    loop = _loop()
    first = loop.check_action("calculator", '{"expression": "6*7"}')
    assert not first.is_loop
    loop.record_action(first)
    again = loop.check_action("calculator", '{"expression": "6*7"}')
    assert again.is_exact_loop
    assert again.loop_type == "exact"


def test_a_call_the_agent_does_not_hold_is_never_a_loop():
    loop = _loop()
    check = loop.check_action("nosuchtool", "{}")
    loop.record_action(check)
    assert not loop.check_action("nosuchtool", "{}").is_loop


def test_enough_differing_calls_to_one_tool_is_a_fuzzy_loop():
    loop = _loop()
    for i in range(FUZZY_LOOP_THRESHOLD):
        check = loop.check_action("calculator", f'{{"expression": "{i}+1"}}')
        assert not check.is_fuzzy_loop
        loop.record_action(check)
    check = loop.check_action("calculator", '{"expression": "99+1"}')
    assert check.is_fuzzy_loop
    assert check.loop_type == f"fuzzy ({FUZZY_LOOP_THRESHOLD + 1} calls)"


def test_a_data_processing_tool_gets_the_wider_threshold():
    loop = NativeToolLoop(
        {"cruncher": _Tool(ToolCategory.DATA_PROCESSING, "cruncher")},
        nudge_cap=ROOMY_CAP,
    )
    assert loop.fuzzy_threshold("cruncher") == FUZZY_LOOP_THRESHOLD_DATA
    for i in range(FUZZY_LOOP_THRESHOLD_DATA):
        loop.record_action(loop.check_action("cruncher", f'{{"n": {i}}}'))
    assert loop.check_action("cruncher", '{"n": 99}').is_fuzzy_loop


# ---------------------------------------------------------------------------
# Result repeats
# ---------------------------------------------------------------------------
def test_a_tool_that_reproduces_its_result_is_a_repeat():
    loop = _loop()
    loop.record_result("calculator", "42")
    assert loop.result_is_repeat("calculator", "42")
    assert not loop.result_is_repeat("calculator", "43")


def test_whitespace_differences_do_not_hide_a_repeated_result():
    loop = _loop()
    loop.record_result("calculator", "the  answer\nis 42")
    assert loop.result_is_repeat("calculator", "the answer is 42")


def test_a_failed_dispatch_is_neither_recorded_nor_a_repeat():
    loop = _loop()
    loop.record_result("calculator", "Error executing tool calculator: boom")
    assert loop.previous_results == []
    assert not loop.result_is_repeat(
        "calculator", "Error executing tool calculator: boom"
    )


# ---------------------------------------------------------------------------
# Offering tools
# ---------------------------------------------------------------------------
def test_tools_are_suppressed_after_the_batch_allowance():
    loop = _loop()
    for _ in range(MAX_BATCH_TOOL_RUNS):
        assert not loop.tools_suppressed()
        loop.note_batch_run()
    assert loop.tools_suppressed()


def test_forcing_a_text_answer_suppresses_tools():
    loop = _loop()
    loop.force_text_answer = True
    assert loop.tools_suppressed()


# ---------------------------------------------------------------------------
# Nudges
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("iteration", "call_count", "result", "expected"),
    [
        (ROOMY_CAP - 2, 0, "42", NUDGE_HAVE_ANSWER),
        (2, NUDGE_AFTER_CALLS, "42", NUDGE_HAVE_RESULTS),
        (2, NUDGE_AFTER_CALLS - 1, "42", None),
        (2, 0, "42", None),
        (2, NUDGE_AFTER_CALLS, "Error executing tool calculator: boom", None),
    ],
)
def test_the_post_tool_nudge_matches_the_loop_state(
    iteration, call_count, result, expected
):
    assert _loop().post_tool_nudge(iteration, call_count, result) == expected


def test_the_drift_threshold_never_exceeds_the_runs_own_budget():
    """A count the iteration cap cannot reach is dead code, not a guard."""
    assert _loop(cap=10).fuzzy_threshold("calculator") == 9
    assert _loop(cap=ROOMY_CAP).fuzzy_threshold("calculator") == FUZZY_LOOP_THRESHOLD


# ---------------------------------------------------------------------------
# Calls written out instead of made
# ---------------------------------------------------------------------------
def test_a_call_for_a_tool_that_never_ran_is_unmade():
    loop = _loop()
    assert loop.is_unmade_call("calculator", "calculator {\"expression\": \"6*7\"}")


def test_a_recap_of_a_call_that_really_ran_is_not_unmade():
    loop = _loop()
    loop.record_execution("calculator")
    assert not loop.is_unmade_call(
        "calculator", "I used calculator and the answer is 42."
    )


def test_the_first_written_call_is_a_warning_and_the_second_is_reportable():
    loop = _loop()
    assert loop.note_written_call("calculator") is False
    assert loop.written_call == "calculator"
    assert loop.note_written_call("calculator") is True


def test_tool_ran_reports_only_dispatched_tools():
    loop = _loop()
    assert not loop.tool_ran("calculator")
    assert not loop.tool_ran(None)
    loop.record_execution("calculator")
    assert loop.tool_ran("calculator")
