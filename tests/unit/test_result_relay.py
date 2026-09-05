"""A result a tool computed survives into the answer.

Some tools return the answer itself rather than material to build one from: a
puzzle solution as a list of moves, a colouring, a sorted list. The model then
has to restate the whole thing, and a small model summarises it instead — the
tool prints seven moves and the answer is "the final answer is 7". The run had
the result and returned an answer its own observations do not support.

The relay puts the result back, and almost everything pinned here is about
keeping it narrow. It reads a tool's declared category, so it never touches a
search result and never touches a single value; it requires the answer to be
missing the *whole* result, because a partial restatement is the frequent
failure and is still not an answer; it steps over a call that failed, because a
traceback is not a result; and it declines when the answer *is* one entry of the
result, because a listing of candidates is one the model was meant to choose
from.

``run()`` and ``stream()`` reach the same answer, from the same records.
"""
from __future__ import annotations

import logging

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.result_relay import (
    FAILED_RESULT_PREFIXES,
    RELAY_MIN_LINES,
    relay_result,
    unrelayed_result,
)
from effgen.core.tool_call_record import MAX_RESULT_CHARS, ToolCall
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.prompts.tool_contract import is_execution_tool
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

#: Seven moves, one per line: what a solver prints and what an answer has to
#: carry.
MOVES = "\n".join(
    f"Move disk {disk} from {src} to {dst}"
    for disk, src, dst in [
        (1, "A", "C"), (2, "A", "B"), (1, "C", "B"), (3, "A", "C"),
        (1, "B", "A"), (2, "B", "C"), (1, "A", "C"),
    ]
)

#: Three passages a search returns. Material for an answer, not the answer.
PASSAGES = (
    "[1] The novel was published in 1965.\n"
    "[2] It was first serialised in a magazine.\n"
    "[3] It won the inaugural award for best novel."
)


def make_tool(name: str, category: ToolCategory, payload: str) -> BaseTool:
    class _Tool(BaseTool):
        def __init__(self) -> None:
            super().__init__(metadata=ToolMetadata(
                name=name, description=f"The {name} tool.", category=category,
                parameters=[ParameterSpec(
                    name="input", type=ParameterType.STRING,
                    description="Input.", required=True)],
            ))

        async def _execute(self, **kwargs):
            return payload

    return _Tool()


def executor(payload: str = MOVES) -> BaseTool:
    return make_tool("python_exec", ToolCategory.CODE_EXECUTION, payload)


def searcher(payload: str = PASSAGES) -> BaseTool:
    return make_tool("web_search", ToolCategory.INFORMATION_RETRIEVAL, payload)


class Script(BaseModel):
    """Answers from a fixed list of turns."""

    def __init__(self, turns) -> None:
        super().__init__(model_name="script", model_type=ModelType.OPENAI)
        self.turns, self.i = list(turns), 0

    def load(self): pass

    def unload(self): pass

    def generate(self, prompt, config=None, **kwargs):
        text = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return GenerationResult(text=text, tokens_used=5, finish_reason="stop",
                                model_name=self.model_name, metadata={})

    def generate_stream(self, prompt, config=None, **kwargs):
        yield self.generate(prompt, config, **kwargs).text

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):
        return self.generate(prompt, config, tools=tools, **kwargs)

    def count_tokens(self, text):
        return TokenCount(count=len(str(text).split()), model_name="script")

    def get_context_length(self):
        return 8192

    def generate_batch(self, prompts, config=None, **kwargs):
        return [self.generate(p) for p in prompts]

    def supports_function_calling(self):
        return False

    def supports_tool_calling(self):
        return False

    def tool_call_support(self):
        return "none"

    def streams_tool_calls(self):
        return False


class Batching(Script):
    """Returns two provider-side tool calls on its first turn, then answers."""

    def generate(self, prompt, config=None, **kwargs):
        result = super().generate(prompt, config, **kwargs)
        if self.i == 1:
            result.metadata = {"tool_calls": [
                {"function": {"name": "python_exec", "arguments": '{"input": "go"}'}},
                {"function": {"name": "web_search", "arguments": '{"input": "go"}'}},
            ]}
        return result

    def supports_function_calling(self):
        return True

    def supports_tool_calling(self):
        return True

    def tool_call_support(self):
        return "api"


def call_turn(name: str, arg: str) -> str:
    return f"Thought: I will run it.\nAction: {name}\nAction Input: {arg}"


def agent(turns, tools) -> Agent:
    return Agent(config=AgentConfig(
        name="probe", model=Script(turns), tools=tools,
        max_iterations=5, raise_on_error=False,
    ))


def held(*tools: BaseTool) -> dict[str, BaseTool]:
    return {tool.metadata.name: tool for tool in tools}


def ran(name: str, result: str, **kwargs) -> ToolCall:
    return ToolCall(name=name, arguments="x", result=result, **kwargs)


# --------------------------------------------------------------- the decision


def test_a_computed_listing_the_answer_dropped_comes_back() -> None:
    answer = relay_result(
        "The final answer is 7.", [ran("python_exec", MOVES)], held(executor())
    )
    assert answer.startswith("The final answer is 7.")
    assert answer.endswith(MOVES)


def test_a_partial_restatement_is_not_enough() -> None:
    """Two of seven moves and "and so on" carries a line and is not an answer."""
    head = "\n".join(MOVES.splitlines()[:2])
    stated = f"The moves are:\n{head}\n... and so on."
    assert relay_result(stated, [ran("python_exec", MOVES)], held(executor())).endswith(
        MOVES
    )


def test_an_answer_that_already_states_the_result_is_left_alone() -> None:
    stated = f"The moves are:\n{MOVES}"
    assert relay_result(
        stated, [ran("python_exec", MOVES)], held(executor())
    ) == stated


def test_reformatted_lines_still_count_as_stated() -> None:
    """A model that bullets the lines has restated them, not dropped them."""
    stated = "The moves are:\n" + "\n".join(
        f"- {line}   " for line in MOVES.splitlines()
    )
    assert relay_result(
        stated, [ran("python_exec", MOVES)], held(executor())
    ) == stated


def test_a_search_result_is_never_relayed() -> None:
    """Passages are material to answer from, so appending them is not an answer."""
    assert relay_result(
        "Frank Herbert.", [ran("web_search", PASSAGES)], held(searcher())
    ) == "Frank Herbert."


@pytest.mark.parametrize("category", list(ToolCategory), ids=lambda c: c.name)
def test_the_relay_follows_the_declared_execution_categories(
    category: ToolCategory,
) -> None:
    """One place decides which tools compute an answer, and this reads it.

    A category added to the execute contract is relayed from here at once, and a
    category removed from it stops being relayed, with no second list to update.
    """
    tool = make_tool("t", category, MOVES)
    relayed = relay_result("A short answer.", [ran("t", MOVES)], held(tool))
    assert (relayed != "A short answer.") is is_execution_tool(tool)


def test_a_result_shorter_than_a_listing_is_a_value_and_stays_put() -> None:
    short = "\n".join(MOVES.splitlines()[: RELAY_MIN_LINES - 1])
    assert relay_result(
        "The answer is 7.", [ran("python_exec", short)], held(executor())
    ) == "The answer is 7."


@pytest.mark.parametrize("prefix", FAILED_RESULT_PREFIXES)
def test_a_failure_is_not_a_result(prefix: str) -> None:
    failure = f"{prefix} Traceback (most recent call last):\n  line 3\n  boom"
    assert relay_result(
        "The answer is 7.", [ran("python_exec", failure)], held(executor())
    ) == "The answer is 7."


def test_a_failed_call_does_not_hide_the_result_behind_it() -> None:
    """The tree steps back over the failures a run collected before answering."""
    calls = [
        ran("python_exec", MOVES),
        ran("python_exec", "Error: Traceback (most recent call last):\n a\n b"),
    ]
    assert relay_result("The final answer is 7.", calls, held(executor())).endswith(
        MOVES
    )


def test_a_recorded_error_is_stepped_over_too() -> None:
    calls = [
        ran("python_exec", MOVES),
        ran("python_exec", "1\n2\n3", error="Error executing tool 'python_exec'"),
    ]
    assert relay_result("The final answer is 7.", calls, held(executor())).endswith(
        MOVES
    )


def test_a_later_single_value_supersedes_an_earlier_listing() -> None:
    """The run's most recent computation is a value, so there is nothing to add."""
    calls = [ran("python_exec", MOVES), ran("python_exec", "7")]
    assert relay_result(
        "The final answer is 7.", calls, held(executor())
    ) == "The final answer is 7."


def test_an_answer_that_is_one_entry_of_the_result_is_a_choice_from_it() -> None:
    """A listing of candidates is one the model was meant to pick from."""
    candidates = "Uif nfttbhf\nThe message\nRgd ldrrzfd"
    assert relay_result(
        "The message", [ran("python_exec", candidates)], held(executor())
    ) == "The message"


def test_a_result_too_long_to_record_in_full_is_not_relayed_in_part() -> None:
    """Appending a shortened result is the partial relay this exists to stop."""
    long_result = "\n".join(f"line {i}" for i in range(MAX_RESULT_CHARS))
    call = ToolCall(name="python_exec", result=long_result[: MAX_RESULT_CHARS + 20])
    assert relay_result("Done.", [call], held(executor())) == "Done."


def test_nothing_ran_and_nothing_is_added() -> None:
    assert relay_result("Paris.", [], held(executor())) == "Paris."
    assert unrelayed_result("", [ran("python_exec", MOVES)], held(executor())) is None


def test_a_call_to_a_tool_the_agent_does_not_hold_is_not_relayed() -> None:
    """The category comes from a tool, and an unheld name has none to read."""
    assert relay_result(
        "The final answer is 7.", [ran("mystery", MOVES)], held(executor())
    ) == "The final answer is 7."


def test_every_firing_says_so_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="effgen.core.result_relay"):
        relay_result("The final answer is 7.", [ran("python_exec", MOVES)],
                     held(executor()))
    firings = [r for r in caplog.records if "result relay:" in r.getMessage()]
    assert len(firings) == 1
    assert "states 0 of the 7 lines" in firings[0].getMessage()


# ------------------------------------------------------------------- the loop


def test_a_run_returns_the_result_its_tool_computed() -> None:
    response = agent(
        [call_turn("python_exec", "solve(3)"), "Final Answer: The final answer is 7."],
        [executor()],
    ).run("Solve it and list the moves.")
    assert str(response).endswith(MOVES)
    assert int(response.tool_calls) == 1


def test_a_run_over_a_search_tool_answers_with_the_answer() -> None:
    """The shape this is not written for: nothing about it changes."""
    response = agent(
        [call_turn("web_search", "who wrote it"), "Final Answer: Frank Herbert."],
        [searcher()],
    ).run("Who wrote it?")
    assert str(response) == "Frank Herbert."


def test_a_mixed_toolset_relays_the_executor_and_not_the_searcher() -> None:
    response = agent(
        [call_turn("web_search", "the rules"),
         call_turn("python_exec", "solve(3)"),
         "Final Answer: The final answer is 7."],
        [searcher(), executor()],
    ).run("Look the rules up, then solve it.")
    answer = str(response)
    assert answer.endswith(MOVES)
    assert "inaugural award" not in answer


def test_a_mixed_toolset_relays_the_executor_run_before_the_search() -> None:
    """The computed result is the one that was lost, wherever the search sits."""
    answer = str(agent(
        [call_turn("python_exec", "solve(3)"),
         call_turn("web_search", "the rules"),
         "Final Answer: The final answer is 7."],
        [searcher(), executor()],
    ).run("Solve it, then check the rules."))
    assert answer.endswith(MOVES)
    assert "inaugural award" not in answer


def test_the_answer_states_the_result_without_a_final_answer_label() -> None:
    """A turn that states the answer in prose reaches the same relay."""
    answer = str(agent(
        [call_turn("python_exec", "solve(3)"), "The answer is 7."],
        [executor()],
    ).run("Solve it and list the moves."))
    assert answer.endswith(MOVES)


def test_a_streamed_run_and_a_blocking_run_give_the_same_answer() -> None:
    turns = [call_turn("python_exec", "solve(3)"),
             "Final Answer: The final answer is 7."]
    blocking = str(agent(turns, [executor()]).run("Solve it and list the moves."))
    streamed = "".join(agent(turns, [executor()]).stream("Solve it and list the moves."))
    assert streamed.endswith(MOVES)
    assert " ".join(streamed.split()) == " ".join(blocking.split())


def test_a_batched_call_records_what_it_returned() -> None:
    """A call the run made carries its result, however the turn dispatched it.

    A turn that returns several calls at once has them dispatched in one batch.
    Without the result on each record, such a run reports calls with nothing in
    them and nothing downstream can say what it was holding when it answered —
    the relay included, which is why an agent that batches its calls used to
    lose the result it computed.
    """
    model = Batching(["", "Final Answer: The final answer is 7."])
    response = Agent(config=AgentConfig(
        name="probe", model=model, tools=[executor(), searcher()],
        max_iterations=4, raise_on_error=False,
    )).run("Solve it and list the moves.")
    assert [(c.name, c.result) for c in response.tool_calls] == [
        ("python_exec", MOVES), ("web_search", PASSAGES),
    ]
    assert str(response).endswith(MOVES)
