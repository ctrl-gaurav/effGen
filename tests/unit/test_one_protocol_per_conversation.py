"""One run's conversation goes out in one protocol, start to finish.

The guards stop offering tool definitions for the rest of a run once the model
has spent its allowance of multi-call turns or repeated a call with nothing
usable to fall back on. The frame then falls back to the text scaffold, which
is the only one that can ask for an answer without offering a call. That must
not change how the conversation travels: a run that has sent a request as
messages sends every later request of that run as messages too, with the
scaffold as the current user message and no tool definitions beside it.

Everything here is offline. The provider is scripted, so what is asserted is
the request object the framework handed it and the keyword arguments that went
with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.messages import Message, Role
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

TASK = "Check the shop ledger total for me and then say it in a sentence."
FOLLOW_UP = "And what did I just ask you?"
ANSWER = "The ledger total is 36."


class Calculator(BaseTool):
    """A tool that returns one recorded result."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="The expression.", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        return "36"


class Recorder(BaseModel):
    """Asks for the same call every turn until it is asked for text.

    Records the request object and the keyword arguments of every turn, which
    is what the assertions read: a log says what the framework decided, the
    request says what the model was actually sent.
    """

    def __init__(self, *, calls: int = 6, carries_messages: bool = True) -> None:
        super().__init__(model_name="recorder", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self._calls = calls
        self._carries = carries_messages
        self.requests: list[Any] = []
        self.kwargs: list[dict[str, Any]] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return True

    def supports_message_protocol(self) -> bool:
        return self._carries

    def supports_conversation(self) -> bool:
        return self._carries

    def generate(self, prompt, config=None, **kwargs: Any) -> GenerationResult:
        self.requests.append(prompt)
        self.kwargs.append(dict(kwargs))
        self.index += 1
        if self.index <= self._calls:
            return GenerationResult(
                text="I will use the calculator.", tokens_used=12,
                finish_reason="tool_calls", model_name=self.model_name,
                metadata={"tool_calls": [{
                    "id": f"call_{self.index}", "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "12*3"}),
                    },
                }]},
            )
        return GenerationResult(
            text=ANSWER, tokens_used=9, finish_reason="stop",
            model_name=self.model_name,
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _agent(model: BaseModel, protocol: str, **extra: Any) -> Agent:
    settings: dict[str, Any] = {
        "name": "protocol", "model": model, "tools": [Calculator()],
        "max_iterations": 8, "raise_on_error": False, "enable_memory": True,
        "tool_calling_mode": "native", "prompt_protocol": protocol,
    }
    settings.update(extra)
    return Agent(AgentConfig(**settings))


def _kinds(model: Recorder) -> list[str]:
    return [
        "messages" if isinstance(r, list) else "flat" for r in model.requests
    ]


def _second_turn(
    agent: Agent, model: Recorder, *, second_calls: int = 6,
) -> list[str]:
    """Answer one turn so the session has a turn, then run a second one.

    The first turn is answered after a single call, which is what puts an
    exchange on the session. ``second_calls`` is how long the second turn keeps
    asking for the same call before the guards stop offering it any.
    """
    model._calls = 1
    agent.run(TASK)
    model.requests.clear()
    model.kwargs.clear()
    model.index = 0
    model._calls = second_calls
    agent.run(FOLLOW_UP)
    return _kinds(model)


# ---------------------------------------------------------------------------
# The defect: a run that changed protocol in the middle of itself
# ---------------------------------------------------------------------------


def test_a_session_run_sends_every_request_as_messages() -> None:
    """At the shipped default, one conversation goes out in one protocol."""
    model = Recorder()
    kinds = _second_turn(_agent(model, "auto"), model)

    assert len(kinds) > 1, kinds
    assert set(kinds) == {"messages"}, kinds


def test_the_suppressed_turn_is_the_one_that_used_to_change_protocol() -> None:
    """The guards' own turn travels as messages, not as the flat string."""
    model = Recorder()
    agent = _agent(model, "auto")
    kinds = _second_turn(agent, model)

    # The last request of the run is the one the guards pushed towards an
    # answer: it asked for text and got it.
    assert kinds[-1] == "messages", kinds
    assert model.requests[-1][-1].role is Role.USER


def test_no_tool_definitions_travel_with_a_suppressed_turn() -> None:
    """Asserted on the request, not on the prose."""
    model = Recorder()
    agent = _agent(model, "auto")
    _second_turn(agent, model)

    assert "tools" not in model.kwargs[-1], model.kwargs[-1]
    assert "tool_choice" not in model.kwargs[-1], model.kwargs[-1]


def test_a_suppressed_turn_asks_for_the_answer_without_offering_a_call() -> None:
    """The turn is a message list whose last user message asks for the answer."""
    model = Recorder()
    agent = _agent(model, "auto")
    _second_turn(agent, model)

    last = model.requests[-1]
    assert isinstance(last, list), type(last)
    assert last[-1].role is Role.USER
    assert "Final Answer" in last[-1].text
    assert not model.kwargs[-1].get("tools"), model.kwargs[-1]


def test_a_suppressed_turn_states_the_persona_and_earlier_turns_once() -> None:
    """What the roles carry is not repeated inside the scaffold's text."""
    persona = "You are Ledgerly, a terse bookkeeper."
    model = Recorder()
    agent = _agent(model, "auto", system_prompt=persona)
    _second_turn(agent, model)

    last = model.requests[-1]
    assert isinstance(last, list), type(last)
    text = "\n".join(m.text for m in last)
    assert text.count("Ledgerly") == 1, text
    assert "Earlier in this conversation" not in text, text


class _Echo(BaseTool):
    """A tool whose every result differs, so only the call allowance can stop it."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="echo", description="Repeat the expression back.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="The expression.", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        return str(kwargs.get("expression"))


class _TwoCallsPerTurn(Recorder):
    """Makes two distinct calls on every turn that offers tools, then answers."""

    def generate(self, prompt, config=None, **kwargs: Any) -> GenerationResult:
        if not kwargs.get("tools") or self._calls == 1:
            return super().generate(prompt, config, **kwargs)
        self.requests.append(prompt)
        self.kwargs.append(dict(kwargs))
        self.index += 1
        return GenerationResult(
            text="Two at once.", tokens_used=8, finish_reason="tool_calls",
            model_name=self.model_name,
            metadata={"tool_calls": [{
                "id": f"call_{self.index}_{n}", "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": json.dumps({"expression": f"{self.index}+{n}"}),
                },
            } for n in (1, 2)]},
        )


def test_a_spent_multi_call_allowance_keeps_the_conversation_on_messages() -> None:
    """The guards' other route to a tool-free turn travels the same way."""
    model = _TwoCallsPerTurn(calls=1)
    agent = _agent(model, "auto", tools=[_Echo()], max_iterations=12)
    agent.run(TASK)
    model.requests.clear()
    model.kwargs.clear()
    model.index = 0
    model._calls = 99
    agent.run(FOLLOW_UP)

    assert any(not k.get("tools") for k in model.kwargs), "no turn was suppressed"
    assert set(_kinds(model)) == {"messages"}, _kinds(model)


def test_the_session_turns_stay_the_turns_they_were() -> None:
    """The earlier exchange is not pasted into the question as prose."""
    model = Recorder()
    agent = _agent(model, "auto")
    _second_turn(agent, model)

    roles = [m.role for m in model.requests[-1]]
    assert Role.ASSISTANT in roles, roles
    assert roles.count(Role.USER) >= 2, roles


def test_the_opt_in_path_stops_falling_back_per_turn() -> None:
    """``"messages"`` asked for messages, including on a suppressed turn."""
    model = Recorder()
    agent = _agent(model, "messages", enable_memory=False)
    agent.run(TASK)

    assert set(_kinds(model)) == {"messages"}, _kinds(model)
    assert "tools" not in model.kwargs[-1], model.kwargs[-1]


def test_a_streamed_session_run_travels_the_same_way() -> None:
    """One loop, so a streamed run resolves exactly as a blocking one does."""
    model = Recorder(calls=1)
    agent = _agent(model, "auto")
    agent.run(TASK)
    model.requests.clear()
    model.kwargs.clear()
    model.index = 0
    model._calls = 6
    list(agent.stream(FOLLOW_UP))

    assert set(_kinds(model)) == {"messages"}, _kinds(model)


# ---------------------------------------------------------------------------
# Shapes this was not written for, which must not move
# ---------------------------------------------------------------------------


def test_a_run_that_continues_nothing_still_sends_the_flat_string() -> None:
    model = Recorder()
    agent = _agent(model, "auto")
    agent.run(TASK)

    assert set(_kinds(model)) == {"flat"}, _kinds(model)


def test_a_run_whose_tools_were_never_suppressed_is_unchanged() -> None:
    """One call, an answer, and no guard ever fires."""
    model = Recorder()
    kinds = _second_turn(_agent(model, "auto"), model, second_calls=1)

    assert set(kinds) == {"messages"}, kinds


def test_a_caller_template_keeps_the_flat_string() -> None:
    """A caller who supplies the frame owns it; nothing here moves it."""
    model = Recorder()
    agent = _agent(
        model, "auto",
        system_prompt_template=(
            "{tools_description}\n{conversation_history}\n{task}\n{scratchpad}"
        ),
    )
    kinds = _second_turn(agent, model)

    assert set(kinds) == {"flat"}, kinds


def test_a_model_that_carries_no_conversation_keeps_the_flat_string() -> None:
    model = Recorder(carries_messages=False)
    kinds = _second_turn(_agent(model, "auto"), model)

    assert set(kinds) == {"flat"}, kinds


def test_a_run_with_no_tools_is_untouched() -> None:
    """With no tools there is nothing for the guards to suppress."""
    model = Recorder(calls=0)
    agent = Agent(AgentConfig(
        name="protocol", model=model, tools=[], max_iterations=4,
        raise_on_error=False, enable_memory=True, prompt_protocol="auto",
    ))
    agent.run(TASK)
    model.requests.clear()
    agent.run(FOLLOW_UP)

    assert model.requests, "the run sent nothing"


# ---------------------------------------------------------------------------
# The resolver, read directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("declared", ["auto", "messages"])
def test_the_resolver_keeps_a_started_conversation_on_messages(
    declared: str,
) -> None:
    agent = _agent(Recorder(), declared)

    assert agent._resolve_prompt_protocol(
        tools_travel_as_parameter=False,
        conversation_carries_earlier_turns=True,
        conversation_already_on_messages=True,
    ) == "messages"


@pytest.mark.parametrize("declared", ["auto", "messages"])
def test_a_conversation_that_never_started_on_messages_is_flat(
    declared: str,
) -> None:
    agent = _agent(Recorder(), declared)

    assert agent._resolve_prompt_protocol(
        tools_travel_as_parameter=False,
        conversation_carries_earlier_turns=True,
        conversation_already_on_messages=False,
    ) == "flat"


def test_an_explicit_flat_is_still_flat() -> None:
    agent = _agent(Recorder(), "flat")

    assert agent._resolve_prompt_protocol(
        tools_travel_as_parameter=False,
        conversation_carries_earlier_turns=True,
        conversation_already_on_messages=True,
    ) == "flat"


def test_the_message_is_one_the_framework_can_convert() -> None:
    """What travels is a real message list, not a string in a list."""
    model = Recorder()
    agent = _agent(model, "auto")
    _second_turn(agent, model)

    assert all(isinstance(m, Message) for m in model.requests[-1])


# ---------------------------------------------------------------------------
# A frame that travels as messages stays there on the answer turn
# ---------------------------------------------------------------------------
#
# On an adapter that takes a conversation, a caller's persona and a session's
# earlier turns travel as their own messages on every tool turn, whatever the
# protocol resolved to. The turn the guards push towards an answer is one more
# turn of the same run, so it carries them the same way instead of folding them
# back into one string.

PERSONA = "You are Ledgerly, a terse bookkeeper."


def _roles(model: Recorder) -> list[list[Role]]:
    return [
        [m.role for m in r] if isinstance(r, list) else [] for r in model.requests
    ]


def test_a_persona_run_that_continues_nothing_keeps_one_request_shape() -> None:
    """At the default, with no session, the persona stays the system turn."""
    model = Recorder()
    agent = _agent(model, "auto", enable_memory=False, system_prompt=PERSONA)
    agent.run(TASK)

    assert any(not k.get("tools") for k in model.kwargs), "no turn was suppressed"
    assert set(_kinds(model)) == {"messages"}, _kinds(model)
    roles = _roles(model)
    assert all(r == roles[0] for r in roles), roles
    assert roles[0][0] is Role.SYSTEM, roles


def test_an_explicit_flat_session_run_keeps_one_request_shape() -> None:
    """``"flat"`` keeps the run's own steps in one string; earlier turns stay turns."""
    model = Recorder()
    kinds = _second_turn(_agent(model, "flat"), model)

    assert any(not k.get("tools") for k in model.kwargs), "no turn was suppressed"
    assert set(kinds) == {"messages"}, kinds
    roles = _roles(model)
    assert all(r == roles[0] for r in roles), roles
    assert Role.ASSISTANT in roles[-1], roles


def test_the_answer_turn_at_flat_states_the_persona_and_earlier_turns_once() -> None:
    """What the roles carry is not repeated inside the scaffold's text."""
    model = Recorder()
    agent = _agent(model, "flat", system_prompt=PERSONA)
    _second_turn(agent, model)

    last = model.requests[-1]
    assert isinstance(last, list), type(last)
    assert last[0].role is Role.SYSTEM
    text = "\n".join(m.text for m in last)
    assert text.count("Ledgerly") == 1, text
    assert "Earlier in this conversation" not in text, text
    assert "Final Answer" in last[-1].text
    assert "tools" not in model.kwargs[-1], model.kwargs[-1]


def test_carrying_the_frame_is_not_the_message_protocol() -> None:
    """The run still reports the protocol it resolved to.

    A guard: this reads the same before and after the answer turn kept the
    frame's roles, and pins that the two are not conflated.
    """
    model = Recorder()
    agent = _agent(model, "auto", enable_memory=False, system_prompt=PERSONA)
    response = agent.run(TASK)

    assert response.metadata["prompt_protocol"] == "flat"


class _RefusesAToolFreeList(Recorder):
    """Takes a message list that offers tools, refuses one that offers none."""

    def generate(self, prompt, config=None, **kwargs: Any) -> GenerationResult:
        if kwargs.get("tools"):
            return super().generate(prompt, config, **kwargs)
        self.requests.append(prompt)
        self.kwargs.append(dict(kwargs))
        if isinstance(prompt, list):
            from effgen.models.errors import InvalidRequestError

            raise InvalidRequestError(
                "openai", self.model_name,
                "a message list is not accepted without tool definitions",
            )
        return GenerationResult(
            text=f"Final Answer: {ANSWER}", tokens_used=9, finish_reason="stop",
            model_name=self.model_name,
        )


def test_a_provider_that_refuses_the_list_on_the_answer_turn_gets_the_string(
    caplog,
) -> None:
    """The turn is sent again as the string it would have been, and answered."""
    model = _RefusesAToolFreeList()
    agent = _agent(model, "flat", enable_memory=False, system_prompt=PERSONA)
    with caplog.at_level("INFO"):
        response = agent.run(TASK)

    assert "[frame] the provider refused the run's frame as messages" in caplog.text
    assert _kinds(model)[-2:] == ["messages", "flat"], _kinds(model)
    assert model.requests[-1].count("Ledgerly") == 1, model.requests[-1]
    assert "36" in str(response.output), response.output


def test_a_run_whose_tools_are_written_into_the_prompt_keeps_the_one_string() -> None:
    """A run that never put its frame on roles does not start on the answer turn.

    A guard: the ReAct-text frame is the only one this run ever sends.
    """
    model = Recorder()
    kinds = _second_turn(
        _agent(model, "flat", tool_calling_mode="react", system_prompt=PERSONA),
        model,
    )

    assert set(kinds) == {"flat"}, kinds
