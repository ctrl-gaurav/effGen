"""A run whose search came back silent gets one more query before it gives up.

After a search, the last thing a model reads is a block of passages and a line
telling it to answer from them and to say so if they do not answer the question.
A run that obeys that line has, so far, had nothing else to do: the loops offer
no way to search again with different words, so one badly-worded query ends the
run. What is pinned here is the missing move and, just as importantly, its
bounds.

Four things fail separately.

**The claim, not the prose around it.** The test that reads an answer as
"the material does not answer this" reads what the answer commits to. A model
that answers a multiple-choice question from its own knowledge and then remarks
that the passages were not useful has answered; the remark is not the claim.

**One further query, never two.** A model that declines twice is believed the
second time, and the run keeps that answer.

**"Not found" stays an answer.** A second search that returns the same passages
must end with the model's own abstention and a successful run — not with the
repeat guard ending the run and reporting that a tool reproduced its result.

**Every path reaches the same decision.** The blocking loop and both streamed
loops agree on the same scripted turns, and a stream that re-queries still
yields exactly one answer.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import NUDGE_SEARCH_AGAIN
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.core.retrieval_requery import (
    MAX_RETRIEVAL_REQUERIES,
    declines_from_context,
    should_requery,
)
from effgen.core.tool_call_record import ToolCall
from effgen.models._usage import tool_call_entry
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
    clear_stream_tool_calls,
    record_stream_tool_calls,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

# --------------------------------------------------------------------------- #
# Material
# --------------------------------------------------------------------------- #
QUESTION = (
    "In what year was the Rhodesian Ridgeback breed standard first approved "
    "by the Kennel Union of Southern Africa?"
)

SILENT = (
    "[Result 1] Rhodesian Ridgeback\n"
    "A dog breed developed in Southern Africa, known for the ridge along its "
    "back.\n\n"
    "[Result 2] Breed information\n"
    "Ridgebacks are loyal and aloof to strangers.\n\n"
    "[Result 3] Breed club\n"
    "The club organises shows across the region."
)
CARRIES_THE_FACT = (
    "[Result 1] Drafting the standard\n"
    "The first standard for the breed was drawn up in Bulawayo and approved in "
    "1922."
)

DECLINING = (
    "Thought: the passages are about the breed, not about the standard.\n"
    "Final Answer: The sources provided do not specify the year the breed "
    "standard was approved."
)
ANSWERED = "Thought: the passages carry it.\nFinal Answer: 1922"
SEARCH_CALL = (
    'Thought: I will look this up.\n'
    'Action: web_search\n'
    'Action Input: {"query": "ridgeback standard approval year"}'
)
# The different query the nudge asks for. Repeating the first one byte for byte
# is answered from the run's own record instead of being dispatched again, which
# is the repeat policy doing its job and not this rule failing to fire.
SEARCH_AGAIN = (
    'Thought: I will try different words.\n'
    'Action: web_search\n'
    'Action Input: {"query": "Kennel Union of Southern Africa breed standard 1920s"}'
)


FIRST_QUERY = ("web_search", {"query": "ridgeback standard approval year"})
SECOND_QUERY = (
    "web_search",
    {"query": "Kennel Union of Southern Africa breed standard 1920s"},
)
NATIVE_REQUERY_TURNS = [
    ("", [FIRST_QUERY]),
    ("The sources provided do not specify the year.", []),
    ("", [SECOND_QUERY]),
    ("The standard was approved in 1922.", []),
]

class _Search(BaseTool):
    """A search whose results are scripted, and which records every query."""

    def __init__(self, results: list[str]) -> None:
        super().__init__(metadata=ToolMetadata(
            name="web_search",
            description="Search the web. Args: query (string).",
            category=ToolCategory.INFORMATION_RETRIEVAL,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="Search query", required=True)],
        ))
        self.results = results
        self.queries: list[str] = []

    async def _execute(self, **kwargs) -> str:
        self.queries.append(str(kwargs.get("query") or ""))
        return self.results[min(len(self.queries) - 1, len(self.results) - 1)]


class _Calculator(BaseTool):
    """A tool that computes rather than retrieves — the shape this leaves alone."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="python_exec",
            description="Run a short Python program and return what it printed.",
            category=ToolCategory.CODE_EXECUTION,
            parameters=[ParameterSpec(
                name="code", type=ParameterType.STRING,
                description="The program", required=True)],
        ))
        self.calls: list[str] = []

    async def _execute(self, **kwargs) -> str:
        self.calls.append(str(kwargs.get("code", "")))
        return "gamma\ndelta\nepsilon"


class _Scripted(BaseModel):
    """Replays scripted text turns and records the kwargs of every call."""

    _forces_calls = True

    def __init__(self, turns: list[str]) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self.turns = list(turns)
        self.index = 0
        self.prompts: list[str] = []
        self.seen: list[dict] = []

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        self.prompts.append(str(prompt))
        self.seen.append(dict(kwargs))
        text = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):
        return self.generate(prompt, config, tools=tools, **kwargs)

    def count_tokens(self, text) -> TokenCount:
        return TokenCount(count=max(1, len(str(text)) // 4),
                          model_name=self.model_name)

    def get_context_length(self) -> int:
        return 8192

    def supports_function_calling(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def tool_call_support(self) -> str:
        return "api"

    def streams_tool_calls(self) -> bool:
        return False

    def supports_forced_tool_call(self) -> bool:
        return self._forces_calls


class _NativeScripted(_Scripted):
    """The same script, delivered as native tool calls and streamed deltas."""

    def __init__(self, turns: list[tuple[str, list[tuple[str, dict]]]]) -> None:
        super().__init__([t for t, _ in turns])
        self.native = list(turns)

    def streams_tool_calls(self) -> bool:
        return True

    def _turn(self, prompt, kwargs):
        self.prompts.append(str(prompt))
        self.seen.append(dict(kwargs))
        turn = self.native[min(self.index, len(self.native) - 1)]
        self.index += 1
        return turn

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        text, calls = self._turn(prompt, kwargs)
        entries = [tool_call_entry(n, json.dumps(a)) for n, a in calls]
        return GenerationResult(
            text=text, tokens_used=5,
            finish_reason="tool_calls" if entries else "stop",
            model_name=self.model_name, metadata={"tool_calls": entries},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        text, calls = self._turn(prompt, kwargs)
        clear_stream_tool_calls(self)
        if calls:
            record_stream_tool_calls(
                self, [tool_call_entry(n, json.dumps(a)) for n, a in calls]
            )
        for i in range(0, len(text), 6):
            yield text[i:i + 6]


def _agent(model, tools, **cfg) -> Agent:
    return Agent(config=AgentConfig(
        name="requery-test", model=model, tools=tools,
        max_iterations=cfg.pop("max_iterations", 6),
        raise_on_error=False, **cfg,
    ))


# --------------------------------------------------------------------------- #
# The claim an answer commits to
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "Final Answer: Unknown",
    "Final Answer: not specified in the provided sources",
    "Answer: The information is not available.",
    "Final Answer: cannot be determined from the passages",
    "The sources provided do not specify the year the standard was approved.",
    ("The passages do not mention the catalogue number. Further research would "
     "be needed to answer this."),
])
def test_an_answer_that_reports_the_material_is_silent_is_read_as_one(text):
    assert declines_from_context(text) is True


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "Final Answer: 1922",
    "Final Answer: B",
    # The claim is a value; the prose above it is not the answer.
    ("The sources do not state this outright, but the standard was approved in "
     "1922.\nFinal Answer: 1922"),
    # An answer from the model's own knowledge, with a remark about the
    # passages after it. The remark is not the claim.
    "The largest planet is Jupiter. The retrieved passages did not mention it.",
    # A comparative remark that names no material at all.
    "Final Answer: C is not the most precise answer among the options provided",
])
def test_an_answer_that_states_something_is_not_read_as_a_decline(text):
    assert declines_from_context(text) is False


def test_the_material_must_be_named_for_an_unlabelled_answer_to_count():
    """A negation with no word for the material is a remark, not a report.

    Dropping this requirement is what made an earlier form of the test fire on
    answers scored correct: a model answers from its own knowledge and adds a
    sentence about what it was given.
    """
    assert declines_from_context("This was not determined by the study.") is False
    assert declines_from_context("This was not determined by the sources.") is True


# --------------------------------------------------------------------------- #
# The preconditions, read directly
# --------------------------------------------------------------------------- #
def _call(name="web_search", result=SILENT, error=None) -> ToolCall:
    return ToolCall(name=name, arguments={"query": "q"}, result=result,
                    error=error, iteration=1)


def _should(**over):
    kwargs = {
        "answer": "Final Answer: unknown",
        "calls": [_call()],
        "is_retrieval": lambda name: name == "web_search",
        "tools_suppressed": False,
        "iterations_left": 3,
        "requery_spent": False,
    }
    kwargs.update(over)
    return should_requery(
        kwargs.pop("answer"), kwargs.pop("calls"), kwargs.pop("is_retrieval"),
        **kwargs,
    )


def test_the_conditions_that_have_to_hold_together():
    assert _should() is True
    assert _should(requery_spent=True) is False
    assert _should(tools_suppressed=True) is False
    assert _should(iterations_left=1) is False
    assert _should(calls=[]) is False
    assert _should(calls=[_call(name="calculator")]) is False
    assert _should(answer="Final Answer: 1922") is False


def test_a_search_that_failed_or_came_back_empty_counts_on_its_own():
    """The two cheap tests, which do not need the answer read at all."""
    assert _should(answer="Final Answer: 1922", calls=[_call(error="boom")]) is True
    assert _should(answer="Final Answer: 1922", calls=[_call(result="")]) is True
    assert _should(
        answer="Final Answer: 1922",
        calls=[_call(result="Error executing tool web_search: timed out")],
    ) is True


def test_the_last_call_decides_which_tool_the_answer_was_written_from():
    calls = [_call(), _call(name="calculator", result="4")]
    assert _should(calls=calls) is False


def test_the_allowance_is_spent_on_read():
    guards = NativeToolLoop({}, nudge_cap=6)
    assert guards.retrieval_requeries == 0
    assert guards.take_retrieval_requery() is True
    assert guards.retrieval_requeries == MAX_RETRIEVAL_REQUERIES
    assert guards.take_retrieval_requery() is False
    assert guards.retrieval_requeries == MAX_RETRIEVAL_REQUERIES


# --------------------------------------------------------------------------- #
# The blocking loop
# --------------------------------------------------------------------------- #
def test_a_declining_answer_is_sent_back_for_one_more_query():
    tool = _Search([SILENT, CARRIES_THE_FACT])
    model = _Scripted([SEARCH_CALL, DECLINING, SEARCH_AGAIN, ANSWERED])
    response = _agent(model, [tool]).run(QUESTION)
    assert len(tool.queries) == 2
    assert "1922" in str(response)
    assert response.success is True
    assert response.stop_reason == "final_answer"
    assert NUDGE_SEARCH_AGAIN not in str(response)


def test_an_answer_written_from_the_passages_is_taken_as_it_is():
    """Nothing fires when the first observation answered the question."""
    tool = _Search([CARRIES_THE_FACT])
    model = _Scripted([SEARCH_CALL, ANSWERED])
    response = _agent(model, [tool]).run(QUESTION)
    assert len(tool.queries) == 1
    assert str(response).strip().endswith("1922")


def test_a_run_whose_tool_computes_rather_than_retrieves_is_untouched():
    """The shape this was not written for: a tool that returns an answer."""
    tool = _Calculator()
    model = _Scripted([
        'Thought: running it.\nAction: python_exec\nAction Input: {"code": "print(1)"}',
        "Final Answer: The sources do not specify this.",
    ])
    response = _agent(model, [tool]).run("Run the program and report its output.")
    assert len(tool.calls) == 1
    assert "do not specify" in str(response)


def test_a_second_decline_is_believed():
    """Never more than one further query, whatever the second answer says."""
    tool = _Search([SILENT, SILENT])
    model = _Scripted([SEARCH_CALL, DECLINING, SEARCH_AGAIN, DECLINING])
    response = _agent(model, [tool]).run(QUESTION)
    assert len(tool.queries) == 2
    assert response.success is True
    assert response.stop_reason == "final_answer"
    assert "do not specify" in str(response)


def test_a_second_search_returning_the_same_passages_still_answers():
    """"Not found" is an outcome, not a failure.

    A second query that returns the passages the first one did is exactly what
    the repeat guard is built to notice, and left to itself it ends the run
    reporting that a tool reproduced its result. A run sent back to search must
    not be able to reach that: the answer it writes afterwards is the answer.
    """
    tool = _Search([SILENT])          # every query returns the same passages
    model = _Scripted([SEARCH_CALL, DECLINING, SEARCH_AGAIN, DECLINING])
    response = _agent(model, [tool]).run(QUESTION)
    assert response.stop_reason == "final_answer"
    assert response.success is True
    assert "do not specify" in str(response)


def test_the_turn_after_the_nudge_is_required_to_call():
    """Where the provider enforces it, the re-query turn is required to call.

    A nudge alone is a request, and a model that has decided the material is
    missing declines a request; the constraint covers that one turn and no
    other.
    """
    tool = _Search([SILENT, CARRIES_THE_FACT])
    model = _NativeScripted(NATIVE_REQUERY_TURNS)
    _agent(model, [tool], tool_calling_mode="native").run(QUESTION)
    forced = [i for i, kw in enumerate(model.seen)
              if kw.get("tool_choice") == "required"]
    assert forced == [2]
    assert len(tool.queries) == 2


def test_the_nudge_reaches_the_model_and_not_the_caller():
    tool = _Search([SILENT, CARRIES_THE_FACT])
    model = _Scripted([SEARCH_CALL, DECLINING, SEARCH_AGAIN, ANSWERED])
    response = _agent(model, [tool]).run(QUESTION)
    assert any(NUDGE_SEARCH_AGAIN in p for p in model.prompts)
    assert NUDGE_SEARCH_AGAIN not in str(response)


def test_a_run_with_no_room_left_keeps_its_answer():
    """Two iterations are needed — one to search, one to answer from it."""
    tool = _Search([SILENT, CARRIES_THE_FACT])
    model = _Scripted([SEARCH_CALL, DECLINING])
    response = _agent(model, [tool], max_iterations=2).run(QUESTION)
    assert len(tool.queries) == 1
    assert "do not specify" in str(response)


# --------------------------------------------------------------------------- #
# The streamed loops
# --------------------------------------------------------------------------- #
def test_the_text_stream_re_queries_and_yields_one_answer():
    tool = _Search([SILENT, CARRIES_THE_FACT])
    model = _Scripted([SEARCH_CALL, DECLINING, SEARCH_AGAIN, ANSWERED])
    agent = _agent(model, [tool])
    events = list(agent.stream(QUESTION, include_events=True))
    answers = [e.text for e in events if e.kind == "answer"]
    assert len(tool.queries) == 2
    assert "".join(answers).strip().endswith("1922")


def test_the_native_stream_re_queries_and_yields_one_answer():
    tool = _Search([SILENT, CARRIES_THE_FACT])
    model = _NativeScripted(NATIVE_REQUERY_TURNS)
    agent = _agent(model, [tool], tool_calling_mode="native")
    events = list(agent.stream(QUESTION, include_events=True))
    answers = [e.text for e in events if e.kind == "answer"]
    assert len(tool.queries) == 2
    streamed = "".join(answers)
    assert "1922" in streamed
    assert "do not specify" not in streamed
    assert streamed == agent.last_stream_response.output


def test_the_native_stream_holds_back_only_a_turn_it_might_re_query():
    """A turn that cannot be re-queried streams as it always did."""
    tool = _Search([CARRIES_THE_FACT])
    model = _NativeScripted([
        ("", [FIRST_QUERY]),
        ("The standard was approved in 1922.", []),
    ])
    agent = _agent(model, [tool], tool_calling_mode="native")
    answers = [e.text for e in agent.stream(QUESTION, include_events=True)
               if e.kind == "answer"]
    assert len(tool.queries) == 1
    assert len(answers) > 1        # delivered as deltas, not as one block
    assert "".join(answers) == agent.last_stream_response.output


def test_run_and_stream_reach_the_same_decision_on_the_same_turns():
    script = [SEARCH_CALL, DECLINING, SEARCH_AGAIN, ANSWERED]
    blocking_tool = _Search([SILENT, CARRIES_THE_FACT])
    blocking = _agent(_Scripted(script), [blocking_tool]).run(QUESTION)
    streamed_tool = _Search([SILENT, CARRIES_THE_FACT])
    agent = _agent(_Scripted(script), [streamed_tool])
    streamed = "".join(e.text for e in agent.stream(QUESTION, include_events=True)
                       if e.kind == "answer")
    assert len(blocking_tool.queries) == len(streamed_tool.queries) == 2
    assert str(blocking).strip() == streamed.strip()
