"""A run is judged by whether its turns bring anything new, not by a count.

Four things are pinned here, and they fail separately.

**A turn is read for what it did.** A turn made progress when a tool it called
returned something the conversation does not already show in full; a new error
counts, the same error again does not. A turn stalled when it only repeated, had
a call declined, or reasoned right after a turn that only reasoned. The count
starts at the run's first new result, and a turn a guard sent back, forced, or
lost to an unreadable call is neither.

**The ceiling the caller set is the ceiling.** A tool whose every result is new
is not withdrawn after a fixed number of calls, and a ``max_iterations`` passed
to one ``run()`` moves the loop's own thresholds as well as its cap.

**A call the model made is run, or asked for again.** A tagged call whose
arguments are JSON with raw line breaks, or quoted the way Python quotes them,
runs; a tag with nothing readable after it is sent back once with a call
required on the next turn where the request can require one; a reply that is a
call written out as text is sent back once and reported the second time.

**"Action: None" is not a tool.** What follows it is read again, and an answer
found there is the answer; a declared no-action after progress asks for the
answer at once.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import NUDGE_HAVE_ANSWER, NUDGE_HAVE_RESULTS
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    PromptCachePolicy,
    TokenCount,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

TASK = "Work out the value."


# ---------------------------------------------------------------------------
# A tool and a model that do what the script says
# ---------------------------------------------------------------------------


class _Tool(BaseTool):
    """A tool whose replies come from a list; an exception entry is raised."""

    def __init__(
        self,
        name: str = "calculator",
        category: ToolCategory = ToolCategory.COMPUTATION,
        replies: list[Any] | None = None,
        param: str = "expression",
    ) -> None:
        super().__init__(metadata=ToolMetadata(
            name=name, description=f"The {name} tool.", category=category,
            parameters=[ParameterSpec(
                name=param, type=ParameterType.STRING,
                description="Input.", required=True,
            )],
        ))
        self.replies = list(replies or [])
        self.inputs: list[dict[str, Any]] = []

    async def _execute(self, **kwargs: Any) -> Any:
        self.inputs.append(dict(kwargs))
        if not self.replies:
            return f"result {len(self.inputs)}"
        reply = self.replies[min(len(self.inputs) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"id": f"call-{name}-{json.dumps(arguments, sort_keys=True)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class _Script(BaseModel):
    """Replays a script and records every request it was sent.

    A script entry is a string (the turn's text) or a dict with ``text`` and
    ``calls`` (a provider-native tool call list). ``requires`` and ``forbids``
    are the two request-level capabilities an adapter may declare.
    """

    def __init__(
        self,
        turns: list[Any],
        *,
        native: bool = True,
        requires: bool = True,
        forbids: bool = False,
        caches: bool = False,
    ) -> None:
        super().__init__(model_name="script-model", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.turns = list(turns)
        self.native = native
        self.requires = requires
        self.forbids = forbids
        self.caches = caches
        self.index = 0
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return self.native

    def tool_call_support(self) -> str:
        return "api" if self.native else "none"

    def supports_forced_tool_call(self) -> bool:
        return self.requires

    def supports_suppressed_tool_call(self) -> bool:
        return self.forbids

    def prompt_cache_policy(self) -> Any:
        return PromptCachePolicy(style="automatic") if self.caches else None

    def supports_message_protocol(self) -> bool:
        return False

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(
            prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
        )
        self.kwargs.append(dict(kwargs))
        turn = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        if isinstance(turn, dict):
            text, calls = turn.get("text", ""), turn.get("calls")
        else:
            text, calls = turn, None
        return GenerationResult(
            text=text, tokens_used=5,
            finish_reason="tool_calls" if calls else "stop",
            model_name=self.model_name,
            metadata={"tool_calls": calls} if calls else {},
        )

    def generate_stream(self, prompt: Any, config: Any = None, **kwargs: Any):
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _agent(model: _Script, tools: list[BaseTool], **config: Any) -> Agent:
    """An agent with both settings opted in, unless a test says otherwise.

    Both ship off: ``max_turns_without_progress`` never asks for the answer,
    and a call the runtime could not run is not recovered. The mechanisms are
    exercised here with them on, and the shipped defaults have tests of their
    own.
    """
    options: dict[str, Any] = {
        "name": "probe", "model": model, "tools": tools,
        "max_iterations": 10, "raise_on_error": False,
        "enable_memory": False, "tool_calling_mode": "hybrid",
        "max_turns_without_progress": 2, "recover_lost_tool_calls": True,
    }
    options.update(config)
    return Agent(AgentConfig(**options))


def _lines(caplog: pytest.LogCaptureFixture, phrase: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if phrase in r.getMessage()]


def _required(model: _Script) -> list[bool]:
    return [k.get("tool_choice") == "required" for k in model.kwargs]


def _withdrawn(model: _Script) -> list[bool]:
    return [not k.get("tools") or k.get("tool_choice") == "none" for k in model.kwargs]


# ---------------------------------------------------------------------------
# What counts as new
# ---------------------------------------------------------------------------


def test_a_result_the_conversation_already_shows_is_not_new() -> None:
    loop = NativeToolLoop({"calculator": _Tool()})
    assert loop.observe_result("calculator", "42") is True
    assert loop.observe_result("calculator", " 42 ") is False
    assert loop.observe_result("calculator", "43") is True
    # The same text from another tool is another observation.
    loop.tools["search"] = _Tool("search")
    assert loop.observe_result("search", "42") is True


def test_a_new_error_is_progress_and_the_same_error_again_is_not() -> None:
    loop = NativeToolLoop({"python_exec": _Tool("python_exec")})
    first = "Error executing tool 'python_exec': NameError: x"
    second = "Error executing tool 'python_exec': TypeError: y"
    assert loop.observe_result("python_exec", first) is True
    assert loop.observe_result("python_exec", second) is True
    assert loop.observe_result("python_exec", first) is False
    assert loop.stalled_calls == {"python_exec": 1}


def test_a_shortened_observation_is_no_longer_shown_in_full() -> None:
    """Fetching a result the model can no longer read is new work again."""
    from effgen.core.thread import ActionStep, ObservationStep

    loop = NativeToolLoop({"search": _Tool("search")})
    assert loop.observe_result("search", "passage one") is True
    shortened = ObservationStep(text="passage", compacted="elided")
    loop.refresh_shown_results([ActionStep(tool="search", raw="q"), shortened])
    assert loop.observe_result("search", "passage one") is True

    kept = ObservationStep(text="passage one")
    loop.refresh_shown_results([ActionStep(tool="search", raw="q"), kept])
    assert loop.observe_result("search", "passage one") is False


def test_a_declined_reply_is_not_a_result_the_conversation_shows() -> None:
    from effgen.core.thread import ActionStep, ObservationStep

    loop = NativeToolLoop({"search": _Tool("search")})
    loop.refresh_shown_results([
        ActionStep(tool="search", raw="q"),
        ObservationStep(text="passage one", declined="loop_detected"),
    ])
    assert loop.observe_result("search", "passage one") is True


# ---------------------------------------------------------------------------
# Which turns are stalls
# ---------------------------------------------------------------------------


_FRESH = iter(range(10**6))


def _turn(loop: NativeToolLoop, **marks: Any) -> str | None:
    loop.begin_turn()
    for _ in range(marks.get("new", 0)):
        loop.observe_result("calculator", f"fresh {next(_FRESH)}")
    for _ in range(marks.get("seen", 0)):
        loop.observe_result("calculator", "old")
    if marks.get("declined"):
        loop.note_declined()
    if marks.get("neutral"):
        loop.note_neutral()
    if marks.get("reasoned"):
        loop.note_reasoning_only()
    if marks.get("declared"):
        loop.note_reasoning_only(declared=True)
    if marks.get("forced"):
        loop.force_tool_call = True
        loop.take_forced_tool_call()
    return loop.end_turn()


def _stalling_loop(limit: int | None = 2) -> NativeToolLoop:
    loop = NativeToolLoop({"calculator": _Tool()}, max_turns_without_progress=limit)
    loop.shown_results[NativeToolLoop._result_key("calculator", "old")] = 1
    return loop


def test_one_turn_of_reasoning_is_planning_and_two_are_a_stall() -> None:
    loop = _stalling_loop()
    assert _turn(loop, new=1) is None
    assert _turn(loop, reasoned=True) is None
    assert loop.turns_without_progress == 0
    assert _turn(loop, reasoned=True) is None
    assert loop.turns_without_progress == 1
    assert _turn(loop, seen=1) == "stalled"
    assert loop.force_text_answer is True


def test_a_new_result_resets_the_count() -> None:
    loop = _stalling_loop()
    assert _turn(loop, new=1) is None
    assert _turn(loop, seen=1) is None
    assert _turn(loop, new=1) is None
    assert _turn(loop, declined=True) is None
    assert loop.turns_without_progress == 1
    assert loop.force_text_answer is False


def test_turns_a_guard_sent_back_forced_or_lost_are_neither() -> None:
    loop = _stalling_loop()
    assert _turn(loop, new=1) is None
    assert _turn(loop, seen=1) is None
    for marks in ({"neutral": True}, {"forced": True, "seen": 1},
                  {"neutral": True, "declined": True}):
        assert _turn(loop, **marks) is None
    assert loop.turns_without_progress == 1
    # A forced turn that did bring something new is still progress.
    assert _turn(loop, forced=True, new=1) is None
    assert loop.turns_without_progress == 0


def test_nothing_is_counted_before_the_first_new_result() -> None:
    loop = _stalling_loop(limit=1)
    for marks in ({"seen": 1}, {"declined": True}, {"reasoned": True},
                  {"reasoned": True}, {"declared": True}):
        assert _turn(loop, **marks) is None
    assert loop.turns_without_progress == 0
    assert loop.force_text_answer is False


def test_a_declared_no_action_after_progress_asks_at_once() -> None:
    loop = _stalling_loop(limit=5)
    assert _turn(loop, new=1) is None
    assert _turn(loop, declared=True) == "declared"
    assert loop.force_text_answer is True


def test_with_no_limit_nothing_asks_for_the_answer() -> None:
    loop = _stalling_loop(limit=None)
    assert _turn(loop, new=1) is None
    for marks in ({"seen": 1}, {"declined": True}, {"declared": True},
                  {"seen": 1}, {"seen": 1}):
        assert _turn(loop, **marks) is None
    assert loop.force_text_answer is False


def test_a_run_already_asked_for_its_answer_is_not_asked_again() -> None:
    loop = _stalling_loop(limit=1)
    assert _turn(loop, new=1) is None
    loop.force_text_answer = True
    assert _turn(loop, seen=1) is None


# ---------------------------------------------------------------------------
# The ceiling the caller set is the ceiling
# ---------------------------------------------------------------------------


class _Progressing(_Script):
    """A new expression every turn; answers only when told to answer now."""

    def __init__(self) -> None:
        super().__init__([], native=False)

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
        self.prompts.append(text)
        self.kwargs.append(dict(kwargs))
        self.index += 1
        body = (f"Thought: step {self.index}\nAction: calculator\n"
                f'Action Input: {{"expression": "{self.index} * 7"}}')
        return GenerationResult(text=body, tokens_used=5, finish_reason="stop",
                                model_name=self.model_name)


def _progressing_run(config_cap: int | None, run_cap: int | None):
    model = _Progressing()
    extra = {} if config_cap is None else {"max_iterations": config_cap}
    agent = _agent(model, [_Tool(replies=None)], tool_calling_mode="react", **extra)
    response = agent.run(TASK, **({} if run_cap is None else {"max_iterations": run_cap}))
    reminder = next(
        (i + 1 for i, p in enumerate(model.prompts) if NUDGE_HAVE_ANSWER in p), None
    )
    return model, response, reminder


def test_a_tool_whose_every_result_is_new_is_not_withdrawn_by_count(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    model, response, _ = _progressing_run(30, None)

    assert not _lines(caplog, "[Loop detected]")
    assert int(response.tool_calls) >= 28
    assert response.stop_reason == "max_iterations_partial"
    assert not _lines(caplog, "[progress]")


def test_a_per_call_ceiling_moves_the_loops_thresholds_too() -> None:
    """``run(max_iterations=N)`` is the run's ceiling for every derived count."""
    model, response, reminder = _progressing_run(None, 30)
    assert int(response.tool_calls) >= 28
    assert model.index == 30
    assert reminder == 29

    model, response, reminder = _progressing_run(30, 4)
    assert model.index == 4
    assert reminder == 3


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, True, "2", 1.5])
def test_an_invalid_setting_is_a_construction_error(value: Any) -> None:
    with pytest.raises(ValueError, match="max_turns_without_progress"):
        AgentConfig(model=_Script([]), max_turns_without_progress=value)


def test_an_invalid_per_call_setting_is_refused_before_any_turn() -> None:
    model = _Script(["Final Answer: 1"])
    agent = _agent(model, [_Tool()])
    with pytest.raises(ValueError, match="None to never ask"):
        agent.run(TASK, max_turns_without_progress=0)
    assert model.index == 0


def test_the_default_never_asks_and_a_number_is_accepted() -> None:
    assert AgentConfig(model=_Script([])).max_turns_without_progress is None
    assert AgentConfig(
        model=_Script([]), max_turns_without_progress=2
    ).max_turns_without_progress == 2


def test_the_shipped_default_does_not_recover_a_lost_call(caplog) -> None:
    """Off by default: the call is not read, and no call is required back.

    The turn is sent back in words once and reported the second time, which is
    what the loop did before any of this existed.
    """
    caplog.set_level(logging.INFO, logger="effgen")
    tool = _executor(["[1, 3]"])
    model = _Script([PYTHON_LITERAL, PYTHON_LITERAL, "Final Answer: [1, 3]"])
    agent = Agent(AgentConfig(
        name="probe", model=model, tools=[tool], max_iterations=10,
        raise_on_error=False, enable_memory=False, tool_calling_mode="hybrid",
    ))
    response = agent.run(TASK)

    assert AgentConfig(model=_Script([])).recover_lost_tool_calls is False
    assert not _lines(caplog, "not strict JSON")
    assert tool.inputs == []
    assert _required(model) == [False, False]
    assert response.stop_reason == "written_tool_call"


def test_the_shipped_default_still_reaches_the_reader_that_reads_the_call() -> None:
    """Guards over-correction: the default loses no call it used to run.

    The turn writes the call twice — once in a tagged body only the lenient
    reader reads, and once in the format the ordinary reader reads. A reader
    that accepts more settles the turn as soon as it succeeds, so a default
    that asked for it and then dropped what it returned would leave the run
    with no call at all: worse than either setting. The call still runs.
    """
    text = (
        PYTHON_LITERAL.rsplit("\n", 1)[0]
        + "\nAction: python_exec\n"
        + 'Action Input: {"code": "values = [3, 1]; print(sorted(values))"}'
    )
    tool = _executor(["[1, 3]"])
    model = _Script([text, "Final Answer: [1, 3]"])
    agent = Agent(AgentConfig(
        name="probe", model=model, tools=[tool], max_iterations=10,
        raise_on_error=False, enable_memory=False, tool_calling_mode="hybrid",
    ))
    response = agent.run(TASK)

    assert tool.inputs and "sorted(values)" in tool.inputs[0]["code"]
    assert response.output == "[1, 3]"
    assert response.stop_reason == "final_answer"


def test_the_setting_can_be_turned_on_for_one_call(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    tool = _executor(["[1, 3]"])
    model = _Script([PYTHON_LITERAL, "Final Answer: [1, 3]"])
    agent = Agent(AgentConfig(
        name="probe", model=model, tools=[tool], max_iterations=10,
        raise_on_error=False, enable_memory=False, tool_calling_mode="hybrid",
    ))
    response = agent.run(TASK, recover_lost_tool_calls=True)

    assert _lines(caplog, "not strict JSON (python literal)")
    assert tool.inputs and "sorted(values)" in tool.inputs[0]["code"]
    assert response.output == "[1, 3]"


def test_an_invalid_recovery_setting_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="recover_lost_tool_calls"):
        AgentConfig(model=_Script([]), recover_lost_tool_calls="yes")


@pytest.mark.parametrize("recover", [False, True])
def test_a_readers_own_strategy_is_never_handed_an_argument_it_lacks(
    recover: bool,
) -> None:
    """A caller's strategy written before ``lenient`` keeps working.

    Turning the recovery setting on asks the run's reader for a lenient read.
    A strategy of the caller's own predates that argument, and naming it would
    fail the run outright, so it is named only on a reader that takes it.
    """
    from effgen.core.tool_calling import ToolCallingStrategy, ToolCallResult

    class Mine(ToolCallingStrategy):
        @property
        def name(self) -> str:
            return "mine"

        def format_tools_for_prompt(self, tools: Any) -> str:
            return ""

        def parse_response(self, text: str, tools: Any = None) -> ToolCallResult:
            result = ToolCallResult(raw_text=text)
            result.final_answer = "read by the caller's own strategy"
            return result

    agent = Agent(AgentConfig(
        name="probe", model=_Script(["anything at all"]), tools=[_executor([])],
        max_iterations=4, raise_on_error=False, enable_memory=False,
        tool_calling_mode="react", recover_lost_tool_calls=recover,
    ))
    agent._tool_calling_strategy = Mine()
    response = agent.run(TASK)

    assert response.stop_reason == "final_answer"
    assert response.output == "read by the caller's own strategy"


def test_an_agent_left_at_the_default_keeps_its_tools(caplog) -> None:
    """Guards over-correction: stalled turns ask for nothing unless the caller opts in."""
    caplog.set_level(logging.INFO, logger="effgen")
    error = RuntimeError("the input file is missing")
    model = _Script([_code_turn(1), _code_turn(2), _code_turn(3), "Final Answer: missing"])
    agent = Agent(AgentConfig(
        name="probe", model=model, tools=[_executor([error])], max_iterations=10,
        raise_on_error=False, enable_memory=False, tool_calling_mode="hybrid",
    ))
    agent.run(TASK)

    assert not _lines(caplog, "[progress]")
    assert _withdrawn(model) == [False] * 4


def test_a_delegated_child_is_built_with_the_parents_setting() -> None:
    """The child's configuration is assembled field by field from the parent's."""
    from effgen.core import sub_agent_manager

    tree = ast.parse(inspect.getsource(sub_agent_manager))
    built = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "child_cfg" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", "") == "AgentConfig"
    ]
    assert len(built) == 1
    keywords = {kw.arg: kw.value for kw in built[0].value.keywords}
    for field in ("max_turns_without_progress", "recover_lost_tool_calls"):
        value = keywords.get(field)
        assert isinstance(value, ast.Call) and getattr(value.func, "id", "") == "getattr"
        assert isinstance(value.args[0], ast.Name) and value.args[0].id == "parent_cfg"


# ---------------------------------------------------------------------------
# Stalled turns ask for the answer
# ---------------------------------------------------------------------------


def _executor(replies: list[Any]) -> _Tool:
    return _Tool("python_exec", ToolCategory.CODE_EXECUTION, replies=replies, param="code")


def _code_turn(n: int) -> dict[str, Any]:
    return {"text": "", "calls": [_call("python_exec", code=f"attempt({n})")]}


def test_the_same_error_on_two_further_attempts_asks_for_the_answer(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    error = RuntimeError("the input file is missing")
    model = _Script([_code_turn(1), _code_turn(2), _code_turn(3),
                     "Final Answer: the input file is missing"])
    agent = _agent(model, [_executor([error])])
    response = agent.run(TASK)

    assert len(_lines(caplog, "turns in a row brought no new result")) == 1
    assert response.output == "the input file is missing"
    assert _withdrawn(model) == [False, False, False, True]
    assert NUDGE_HAVE_RESULTS in model.prompts[-1]
    steps = response.metadata["thread"].steps
    assert any(getattr(s, "nudge_id", "") == "have_results" for s in steps)


def test_different_errors_then_a_success_are_all_progress(caplog) -> None:
    """Guards over-correction: a run that fails differently each time is working."""
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([_code_turn(1), _code_turn(2), _code_turn(3), "Final Answer: 7"])
    agent = _agent(model, [_executor([
        RuntimeError("NameError: x"), RuntimeError("TypeError: y"), "7",
    ])])
    response = agent.run(TASK)

    assert not _lines(caplog, "[progress]")
    assert response.output == "7"
    assert _withdrawn(model) == [False, False, False, False]


def test_with_the_setting_off_the_run_keeps_its_tools(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    error = RuntimeError("the input file is missing")
    script = [_code_turn(1), _code_turn(2), _code_turn(3), "Final Answer: missing"]

    model = _Script(script)
    _agent(model, [_executor([error])], max_turns_without_progress=None).run(TASK)
    assert not _lines(caplog, "[progress]")
    assert _withdrawn(model) == [False] * 4

    model = _Script(script)
    _agent(model, [_executor([error])]).run(TASK, max_turns_without_progress=None)
    assert not _lines(caplog, "[progress]")
    assert _withdrawn(model) == [False] * 4


def test_nothing_is_withdrawn_before_the_executor_has_run(caplog) -> None:
    """Two reasoning turns before the first call are not a stall.

    Guards over-correction: the refusal of an answer given without running the
    executor still fires, and the run is not asked for its answer early.
    """
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([
        "Final Answer: 7",
        "Thought: I should think about the program first.",
        "Thought: Still thinking about the program.",
        _code_turn(1),
        "Final Answer: 7",
    ])
    response = _agent(model, [_executor(["7"])]).run(TASK)

    assert _lines(caplog, "execution refusal:")
    assert not _lines(caplog, "[progress]")
    assert response.output == "7"


def test_a_session_run_is_judged_on_its_own_turns(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    error = RuntimeError("the input file is missing")
    model = _Script([_code_turn(1), "Final Answer: first",
                     _code_turn(2), _code_turn(3), _code_turn(4), "Final Answer: second"])
    agent = _agent(model, [_executor(["done", error])], enable_memory=True)

    agent.run(TASK)
    assert not _lines(caplog, "[progress]")
    second = agent.run("And now the second part.")

    assert len(_lines(caplog, "turns in a row brought no new result")) == 1
    assert second.output == "second"
    assert {type(p) for p in model.prompts} == {str}


# ---------------------------------------------------------------------------
# "Action: None"
# ---------------------------------------------------------------------------


def test_text_after_a_declared_no_action_is_the_answer(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([
        {"text": "", "calls": [_call("calculator", expression="6*7")]},
        ("Thought: I have the product.\nAction: None\nAction Input: None\n"
         "The product is 42."),
    ])
    response = _agent(model, [_Tool(replies=["42"])]).run(TASK)

    assert _lines(caplog, "declared no action and answered after it")
    assert response.output == "The product is 42."
    assert model.index == 2
    assert not any("No tool named 'None'" in p for p in model.prompts)


def test_a_declared_no_action_with_nothing_after_it_asks_for_the_answer(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([
        {"text": "", "calls": [_call("calculator", expression="6*7")]},
        "Thought: I have the product.\nAction: None\nAction Input: None",
        "Final Answer: 42",
    ])
    response = _agent(model, [_Tool(replies=["42"])]).run(TASK)

    assert _lines(caplog, "the turn declared no action; asking for the answer")
    assert response.output == "42"
    assert _withdrawn(model) == [False, False, True]
    assert not any("No tool named 'None'" in p for p in model.prompts)


@pytest.mark.parametrize("after", [
    "Observation: None\nThought: I should compute it again.",
    "",
])
def test_scaffolding_after_the_declaration_is_not_an_answer(after: str, caplog) -> None:
    """Guards over-correction: the scaffold's next labels are not an answer."""
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([
        {"text": "", "calls": [_call("calculator", expression="6*7")]},
        f"Thought: done.\nAction: None\n{after}",
        "Final Answer: 42",
    ])
    response = _agent(model, [_Tool(replies=["42"])]).run(TASK)

    assert not _lines(caplog, "answered after it")
    assert response.output == "42"
    assert model.index == 3


@pytest.mark.parametrize("words", ["None needed", "N/A", "(no tool)", "no action required"])
def test_each_declaration_is_read_as_no_action(words: str) -> None:
    from effgen.core.tool_calling import ReActStrategy

    result = ReActStrategy().parse_response(
        f"Thought: done.\nAction: {words}\nAction Input: none", {"calculator": _Tool()}
    )
    assert result.declared_no_action is True
    assert result.is_tool_call is False
    assert result.tool_name is None


def test_a_tool_actually_named_none_is_still_called() -> None:
    """Guards over-correction: a held tool is a tool, whatever its name."""
    from effgen.core.tool_calling import ReActStrategy

    result = ReActStrategy().parse_response(
        'Thought: use it.\nAction: none\nAction Input: {"expression": "1"}',
        {"none": _Tool("none")},
    )
    assert result.is_tool_call is True
    assert result.tool_name == "none"


# ---------------------------------------------------------------------------
# A call the model made is run, or asked for again
# ---------------------------------------------------------------------------

PYTHON_LITERAL = (
    '<tool_call>\n{"name": "python_exec", "arguments": {"code": '
    "'values = [3, 1]; print(sorted(values))'}}\n</tool_call>\nboxed{[1, 3]}"
)
RAW_LINE_BREAKS = (
    "Let me run it.\nAction: Call the python_exec function.\n<tool_call>\n"
    '{"name": "python_exec", "arguments": {"code": "values = [3, 1]\n'
    'print(sorted(values))\n"}}\n</tool_call>'
)
STRAY_BRACKET = (
    '<tool_call>\n{"name": "python_exec", "arguments": {"code": '
    '"print(sorted([3, 1]))")}}\n</tool_call>'
)
DANGLING = "I will sort the list with the tool.\n<tool_call>\n"


@pytest.mark.parametrize(("text", "how"), [
    (PYTHON_LITERAL, "python literal"),
    (RAW_LINE_BREAKS, "raw line breaks"),
])
def test_a_call_whose_arguments_are_not_strict_json_runs(text: str, how: str, caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    tool = _executor(["[1, 3]"])
    model = _Script([text, "Final Answer: [1, 3]"])
    response = _agent(model, [tool]).run(TASK)

    assert _lines(caplog, f"not strict JSON ({how})")
    assert tool.inputs and "sorted(values)" in tool.inputs[0]["code"]
    assert response.output == "[1, 3]"
    assert model.index == 2
    assert not _lines(caplog, "could not be read")


def test_a_call_with_a_stray_bracket_is_not_guessed_at(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    tool = _executor(["[1, 3]"])
    model = _Script([STRAY_BRACKET, _code_turn(1), "Final Answer: [1, 3]"])
    response = _agent(model, [tool]).run(TASK)

    assert not _lines(caplog, "not strict JSON")
    assert _lines(caplog, "could not be read (its arguments could not be read)")
    assert tool.inputs == [{"code": "attempt(1)"}]
    assert response.output == "[1, 3]"
    assert _required(model) == [False, True, False]


def test_a_dangling_call_tag_requires_a_call_where_the_request_can(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([DANGLING, _code_turn(1), "Final Answer: [1, 3]"])
    response = _agent(model, [_executor(["[1, 3]"])]).run(TASK)

    assert _lines(caplog, "(nothing followed the call tag); requiring a call")
    assert _required(model) == [False, True, False]
    assert "could not be run: nothing followed the call tag" in model.prompts[1]
    assert response.output == "[1, 3]"


def test_a_dangling_call_tag_is_asked_for_in_words_elsewhere(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([DANGLING, _code_turn(1), "Final Answer: [1, 3]"], requires=False)
    response = _agent(model, [_executor(["[1, 3]"])]).run(TASK)

    assert _lines(caplog, "asking again (no request-level constraint available here)")
    assert _required(model) == [False, False, False]
    assert "could not be run: nothing followed the call tag" in model.prompts[1]
    assert response.output == "[1, 3]"


def test_a_second_dangling_tag_names_no_tool_and_the_run_goes_on(caplog) -> None:
    """Guards over-correction: a tag that names nothing is not reported as a call."""
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([DANGLING, DANGLING, _code_turn(1), "Final Answer: [1, 3]"])
    response = _agent(model, [_executor(["[1, 3]"])]).run(TASK)

    assert len(_lines(caplog, "could not be read")) == 1
    assert response.stop_reason == "final_answer"
    assert response.output == "[1, 3]"


def test_a_written_out_call_as_the_answer_is_sent_back_once(caplog) -> None:
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script(['calculator("6*7")', {"text": "", "calls": [
        _call("calculator", expression="6*7")]}, "Final Answer: 42"])
    response = _agent(model, [_Tool(replies=["42"])]).run(TASK)

    assert _lines(caplog, "(it was written out as text instead of being made)")
    assert response.stop_reason == "final_answer"
    assert response.output == "42"
    assert _required(model) == [False, True, False]


def test_a_named_call_after_a_dangling_tag_is_warned_before_it_is_reported(caplog) -> None:
    """A tag that named nothing does not count toward reporting a written call.

    The run is asked for the call once; a later call written out for a held
    tool is warned the first time and reported only the second time.
    """
    caplog.set_level(logging.INFO, logger="effgen")
    written = '<tool_call>\n{"name": "python_exec", "arguments": print(1)"}\n</tool_call>'
    model = _Script([DANGLING, written, _code_turn(1), "Final Answer: 1"])
    response = _agent(model, [_executor(["1"])]).run(TASK)

    assert response.stop_reason == "final_answer"
    assert response.output == "1"
    assert model.index == 4

    model = _Script([DANGLING, written, written, "Final Answer: 1"])
    response = _agent(model, [_executor(["1"])]).run(TASK)
    assert response.stop_reason == "written_tool_call"
    assert model.index == 3


def test_a_written_out_call_twice_is_reported() -> None:
    model = _Script(['calculator("6*7")', 'calculator("6*7")', "Final Answer: 42"])
    response = _agent(model, [_Tool(replies=["42"])]).run(TASK)

    assert response.stop_reason == "written_tool_call"
    assert response.outcome == "failed"
    assert model.index == 2


def test_a_call_shown_in_a_code_block_is_documentation(caplog) -> None:
    """Guards over-correction: an answer that shows a call is still an answer."""
    caplog.set_level(logging.INFO, logger="effgen")
    answer = 'Run it like this: `calculator("6*7")`, which returns 42.'
    model = _Script([answer])
    response = _agent(model, [_Tool(replies=["42"])]).run(TASK)

    assert response.output == answer
    assert not _lines(caplog, "could not be read")
    assert model.index == 1


# ---------------------------------------------------------------------------
# One request shape
# ---------------------------------------------------------------------------


def test_a_turn_that_forbids_a_call_is_never_also_made_to_require_one(
    monkeypatch, caplog,
) -> None:
    """A withdrawn turn with the forced flag set goes out forbidding, not requiring."""
    caplog.set_level(logging.INFO, logger="effgen")
    monkeypatch.setattr(NativeToolLoop, "take_forced_tool_call", lambda self: True)
    model = _Script([
        {"text": "", "calls": [_call("calculator", expression="1+1")]},
        {"text": "", "calls": [_call("calculator", expression="2*1")]},
        "Final Answer: 2",
    ], forbids=True, caches=True)
    response = _agent(model, [_Tool(replies=["2"])]).run(TASK)

    assert response.output == "2"
    last = model.kwargs[-1]
    assert last.get("tools") and last.get("tool_choice") == "none"
    assert not any(
        k.get("tool_choice") == "required" and w
        for k, w in zip(model.kwargs, _withdrawn(model), strict=True)
    )
    assert _lines(caplog, "the run's tools are withdrawn on this turn")


def test_an_answer_request_after_a_stall_travels_in_the_runs_shape() -> None:
    """On a provider that keeps the prefix, the definitions stay and no call is allowed."""
    error = RuntimeError("the input file is missing")
    model = _Script([_code_turn(1), _code_turn(2), _code_turn(3), "Final Answer: missing"],
                    forbids=True, caches=True)
    _agent(model, [_executor([error])]).run(TASK)

    last = model.kwargs[-1]
    assert last.get("tools") and last.get("tool_choice") == "none"
    assert [k.get("tool_choice") for k in model.kwargs[:3]] == [None, None, None]


# ---------------------------------------------------------------------------
# The streamed path takes the same decisions
# ---------------------------------------------------------------------------

STREAM_SCRIPTS = {
    "lenient read": ([PYTHON_LITERAL, "Final Answer: [1, 3]"], ["[1, 3]"]),
    "dangling tag": ([DANGLING, _code_turn(1), "Final Answer: [1, 3]"], ["[1, 3]"]),
    "stalled": ([_code_turn(1), _code_turn(2), _code_turn(3), "Final Answer: missing"],
                [RuntimeError("the input file is missing")]),
    "written answer": (['python_exec("print(1)")', _code_turn(1), "Final Answer: 1"], ["1"]),
}


@pytest.mark.parametrize("name", sorted(STREAM_SCRIPTS))
def test_a_streamed_run_takes_the_same_turns(name: str) -> None:
    """``stream()`` reaches every decision ``run()`` reaches, on the same turns.

    Guards over-correction as well: both paths share one loop, so the parity
    held before these decisions existed and has to keep holding with them.
    """
    script, replies = STREAM_SCRIPTS[name]
    blocking_model = _Script(script)
    blocking = _agent(blocking_model, [_executor(replies)]).run(TASK)
    streamed_model = _Script(script)
    agent = _agent(streamed_model, [_executor(replies)])
    text = "".join(str(piece) for piece in agent.stream(TASK))
    record = agent.last_stream_response

    assert record is not None
    assert record.output == blocking.output
    assert record.stop_reason == blocking.stop_reason
    assert streamed_model.index == blocking_model.index
    assert _required(streamed_model) == _required(blocking_model)
    assert _withdrawn(streamed_model) == _withdrawn(blocking_model)
    # Nothing of a turn that was sent back reached the consumer.
    assert "<tool_call>" not in text and "python_exec(" not in text
    assert text == blocking.output


# ---------------------------------------------------------------------------
# A run with no tools is untouched
# ---------------------------------------------------------------------------


def test_an_agent_with_no_tools_reads_none_of_this(caplog) -> None:
    """Guards over-correction: no tools, nothing to lose or to declare."""
    caplog.set_level(logging.INFO, logger="effgen")
    model = _Script([DANGLING])
    response = _agent(model, []).run(TASK)

    assert not _lines(caplog, "[call]")
    assert not _lines(caplog, "[progress]")
    assert model.index == 1
    assert response.success
