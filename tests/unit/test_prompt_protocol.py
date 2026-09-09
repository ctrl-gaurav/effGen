"""The run's conversation reaches the model as a conversation.

``AgentConfig.prompt_protocol`` decides whether a turn goes out as one flat
string or as the message list the run actually was: a system turn carrying the
persona and the tool contract, the task, the model's own reasoning beside the
call it made, and each tool result answering the call it belongs to.

Everything here is offline. The provider is scripted, so what is asserted is
what the framework assembled and what the wire conversion made of it.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from collections.abc import Iterator
from typing import Any

import pytest

import effgen.models
from effgen.core import agent_runtime
from effgen.core.agent import Agent, AgentConfig
from effgen.core.messages import Message, Role, TextPart, ToolCallPart, ToolResultPart
from effgen.core.thread import ActionStep, AgentThread, ObservationStep, ThoughtStep
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.models.lazy import LazyModel
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

REASONING = "I need the product of 12 and 3, so I will use the calculator."
ANSWER = "The ledger total is 36."
CALL_ID = "call_provider_abc123"
TASK = "Check the shop ledger total for me and then say it in a sentence."


# ---------------------------------------------------------------------------
# Fixtures: a tool, and a provider that answers the way a real one does
# ---------------------------------------------------------------------------


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


class ScriptedNative(BaseModel):
    """Returns a native tool call with text beside it, then an answer.

    ``refusals`` makes the first *n* requests fail the way a template that will
    not take an assistant turn carrying both text and a call fails.
    """

    def __init__(
        self, *, carries_messages: bool = True, refusals: int = 0,
        model_name: str = "scripted-native",
    ) -> None:
        super().__init__(model_name=model_name, model_type=ModelType.OPENAI)
        self._is_loaded = True
        self._carries = carries_messages
        self._refusals = refusals
        self.prompts: list[Any] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return True

    def supports_message_protocol(self) -> bool:
        return self._carries

    def generate(self, prompt, config=None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt)
        if self._refusals > 0:
            self._refusals -= 1
            from effgen.models.errors import InvalidRequestError

            raise InvalidRequestError(
                "openai", self.model_name,
                "assistant message must not carry content and tool_calls",
            )
        self.index += 1
        if self.index == 1:
            return GenerationResult(
                text=REASONING, tokens_used=12, finish_reason="tool_calls",
                model_name=self.model_name,
                metadata={"tool_calls": [{
                    "id": CALL_ID, "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "12*3"}),
                    },
                }]},
            )
        return GenerationResult(
            text=ANSWER,
            tokens_used=9, finish_reason="stop", model_name=self.model_name,
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _agent(model: BaseModel, protocol: str, *, tools: bool = True) -> Agent:
    return Agent(AgentConfig(
        name="protocol", model=model, tools=[Calculator()] if tools else [],
        max_iterations=4, raise_on_error=False, enable_memory=False,
        tool_calling_mode="native", prompt_protocol=protocol,
    ))


def _wire(messages: list[Message]) -> list[dict[str, Any]]:
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model_name = "an-adapter-model"
    return [adapter._message_to_openai(m) for m in messages]


def _probe_cache() -> dict:
    """What this process has learned about each model's template.

    Read off the module rather than imported by name so this file still
    collects against a tree that has no message protocol, where every test in
    it is supposed to fail rather than the whole file to error.
    """
    return getattr(agent_runtime, "_MESSAGE_PROTOCOL_PROBE", {})


@pytest.fixture(autouse=True)
def _forget_what_a_model_taught_this_process():
    """Each test starts with nothing learned about any model's template."""
    _probe_cache().clear()
    yield
    _probe_cache().clear()


# ---------------------------------------------------------------------------
# The configuration switch
# ---------------------------------------------------------------------------


def test_a_run_sends_the_flat_transcript_unless_asked_otherwise() -> None:
    assert AgentConfig(model="x").prompt_protocol == "flat"


def test_a_protocol_nobody_named_is_a_construction_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        AgentConfig(model="x", prompt_protocol="mesages")
    message = str(excinfo.value)
    for named in ("flat", "messages", "auto"):
        assert named in message


@pytest.mark.parametrize("protocol", ["flat", "messages", "auto"])
def test_every_named_protocol_is_accepted(protocol: str) -> None:
    assert AgentConfig(model="x", prompt_protocol=protocol).prompt_protocol == protocol


# ---------------------------------------------------------------------------
# The declared capability
# ---------------------------------------------------------------------------


def test_an_adapter_carries_nothing_until_it_says_it_does() -> None:
    """The default is False, so a third-party adapter keeps the flat string."""

    class Bare(ScriptedNative):
        pass

    assert BaseModel.supports_message_protocol(Bare(carries_messages=False)) is False


def test_the_openai_converter_declares_what_it_carries() -> None:
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model_name = "an-adapter-model"
    assert adapter.supports_message_protocol() is adapter.supports_tool_calling()
    assert adapter.supports_message_protocol() is True


def test_a_lazily_loaded_model_answers_for_the_model_it_wraps() -> None:
    """A missed delegation makes a capable model silently incapable."""
    inner = ScriptedNative(carries_messages=True)
    assert LazyModel(inner).supports_message_protocol() is True
    assert LazyModel(ScriptedNative(carries_messages=False)).supports_message_protocol() is False


# ---------------------------------------------------------------------------
# The wire conversion
# ---------------------------------------------------------------------------


def test_an_assistant_turn_reaches_the_wire_with_its_call_and_its_words() -> None:
    message = Message(role=Role.ASSISTANT, content=[
        TextPart(text=REASONING),
        ToolCallPart(tool_call_id=CALL_ID, name="calculator",
                     arguments={"expression": "12*3"}),
    ])
    sent = _wire([message])[0]
    assert sent["role"] == "assistant"
    assert sent["content"] == REASONING
    assert sent["tool_calls"] == [{
        "id": CALL_ID, "type": "function",
        "function": {"name": "calculator",
                     "arguments": json.dumps({"expression": "12*3"})},
    }]


def test_a_call_with_no_words_beside_it_still_reaches_the_wire() -> None:
    message = Message(role=Role.ASSISTANT, content=[
        ToolCallPart(tool_call_id=CALL_ID, name="calculator", arguments={}),
    ])
    sent = _wire([message])[0]
    assert sent["content"] is None
    assert sent["tool_calls"][0]["id"] == CALL_ID


def test_a_tool_result_answers_the_call_it_belongs_to() -> None:
    message = Message(role=Role.TOOL, content=[
        ToolResultPart(tool_call_id=CALL_ID, result="36"),
    ])
    sent = _wire([message])[0]
    assert sent == {"role": "tool", "tool_call_id": CALL_ID, "content": "36"}


def test_a_result_that_is_not_a_string_is_sent_as_data() -> None:
    message = Message(role=Role.TOOL, content=[
        ToolResultPart(tool_call_id=CALL_ID, result={"total": 36}),
    ])
    assert json.loads(_wire([message])[0]["content"]) == {"total": 36}


def test_an_ordinary_text_turn_is_unchanged_by_the_new_branches() -> None:
    """A guard, not a proof: it passes on both trees, and it is supposed to.

    The two new branches in the converter must not change what a plain text
    message becomes.
    """
    message = Message(role=Role.USER, content=[TextPart(text="hello")])
    assert _wire([message])[0] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# What the loop keeps
# ---------------------------------------------------------------------------


def test_the_model_s_own_words_and_the_provider_s_call_id_survive() -> None:
    """Both were dropped before the parser was asked to carry them."""
    agent = _agent(ScriptedNative(), "flat")
    thread = agent.run(TASK).metadata["thread"]
    actions = thread.actions()
    assert [step.reasoning for step in actions] == [REASONING]
    assert [step.call_id for step in actions] == [CALL_ID]
    answered = [step.call_id for step in thread.observations()]
    assert answered == [CALL_ID]


def test_carrying_the_reasoning_does_not_move_the_flat_transcript() -> None:
    """``to_text()`` is what the flat prompt embeds; it must not move.

    A guard against over-correction, not a proof of the change: it passes
    against the tree before the protocol existed too, and it is supposed to.
    """
    thread = AgentThread()
    thread.append(ThoughtStep(text=""))
    thread.append(ActionStep(tool="calculator", raw="12*3",
                             reasoning=REASONING, call_id=CALL_ID))
    thread.append(ObservationStep(text="36", call_id=CALL_ID))
    assert thread.to_text() == (
        "\nThought: \nAction: calculator\nAction Input: 12*3\nObservation: 36"
    )
    assert REASONING not in thread.to_text()


def test_a_call_nothing_answered_is_named_by_its_id() -> None:
    thread = AgentThread()
    thread.append(ActionStep(tool="calculator", raw="1+1", call_id="a"))
    thread.append(ObservationStep(text="2", call_id="a"))
    thread.append(ActionStep(tool="calculator", raw="2+2", call_id="b"))
    assert thread.unanswered_call_ids() == ["b"]
    thread.append(ObservationStep(text="4", call_id="b"))
    assert thread.unanswered_call_ids() == []


def test_a_call_with_no_id_of_its_own_is_still_named() -> None:
    thread = AgentThread()
    thread.append(ActionStep(tool="calculator", raw="1+1"))
    unanswered = thread.unanswered_call_ids()
    assert len(unanswered) == 1
    assert unanswered[0].startswith("effgen-")


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def test_the_run_goes_out_as_the_conversation_it_was() -> None:
    model = ScriptedNative()
    response = _agent(model, "messages").run(TASK)

    assert response.metadata["prompt_protocol"] == "messages"
    assert all(isinstance(prompt, list) for prompt in model.prompts)

    wire = _wire(model.prompts[-1])
    roles = [message["role"] for message in wire]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "assistant" in roles and "tool" in roles

    assistant = next(m for m in wire if m["role"] == "assistant")
    assert assistant["content"] == REASONING
    assert assistant["tool_calls"][0]["id"] == CALL_ID
    tool = next(m for m in wire if m["role"] == "tool")
    assert tool["tool_call_id"] == CALL_ID
    assert tool["content"] == "36"


def test_the_system_turn_is_the_same_bytes_on_every_turn_of_a_run() -> None:
    """That identical prefix is the thing a provider can cache."""
    model = ScriptedNative()
    _agent(model, "messages").run(TASK)
    systems = {
        json.dumps(_wire(prompt)[0], sort_keys=True) for prompt in model.prompts
    }
    assert len(systems) == 1


def test_the_task_reaches_the_model_once_and_as_the_user() -> None:
    model = ScriptedNative()
    _agent(model, "messages").run(TASK)
    for prompt in model.prompts:
        wire = _wire(prompt)
        assert [m for m in wire if m.get("content") == TASK] == [
            {"role": "user", "content": TASK}
        ]


def test_a_caller_cannot_tell_which_protocol_ran_except_from_metadata() -> None:
    flat = _agent(ScriptedNative(), "flat").run(TASK)
    messages = _agent(ScriptedNative(), "messages").run(TASK)
    assert flat.output == messages.output
    assert flat.success == messages.success
    assert set(flat.metadata) == set(messages.metadata)
    assert flat.metadata["prompt_protocol"] == "flat"
    assert messages.metadata["prompt_protocol"] == "messages"


def test_every_run_says_which_protocol_it_used() -> None:
    assert _agent(ScriptedNative(), "flat").run(TASK).metadata["prompt_protocol"] == "flat"


def test_a_run_that_answered_without_tools_says_so_too() -> None:
    """A run with no tools never reaches the loop, and still carries the key.

    Without this a caller reading ``metadata["prompt_protocol"]`` gets a
    ``KeyError`` on exactly the runs where the answer is "flat, and it could
    not have been anything else".
    """
    for protocol in ("flat", "messages", "auto"):
        response = _agent(ScriptedNative(), protocol, tools=False).run(TASK)
        assert response.metadata["prompt_protocol"] == "flat", protocol


def test_a_run_with_no_tools_sends_the_same_bytes_on_both_protocols() -> None:
    """The shape this change was not written for: nothing may move.

    With no tools the run takes direct inference, so no protocol path can fire
    and the request is the same string whatever the caller asked for. This is
    the offline half of the no-tools control in the measured comparison, where
    a live server's own sampling makes byte-identity unobservable.
    """
    prompts = {}
    for protocol in ("flat", "messages"):
        model = ScriptedNative()
        _agent(model, protocol, tools=False).run(TASK)
        prompts[protocol] = list(model.prompts)
    assert all(isinstance(p, str) for p in prompts["flat"])
    assert prompts["flat"] == prompts["messages"]


# ---------------------------------------------------------------------------
# The fallbacks
# ---------------------------------------------------------------------------


def test_a_model_that_cannot_carry_the_shape_gets_the_flat_transcript(caplog) -> None:
    model = ScriptedNative(carries_messages=False)
    with caplog.at_level("INFO"):
        response = _agent(model, "messages").run(TASK)
    assert all(isinstance(prompt, str) for prompt in model.prompts)
    assert response.metadata["prompt_protocol"] == "flat"
    assert "[protocol] messages is not available on this model" in caplog.text


def test_a_turn_whose_tools_are_prose_gets_the_flat_transcript(caplog) -> None:
    """The ReAct-text path has no protocol call to express."""
    model = ScriptedNative()
    agent = Agent(AgentConfig(
        name="protocol", model=model, tools=[Calculator()], max_iterations=3,
        raise_on_error=False, enable_memory=False, tool_calling_mode="react",
        prompt_protocol="messages",
    ))
    with caplog.at_level("INFO"):
        agent.run(TASK)
    assert all(isinstance(prompt, str) for prompt in model.prompts)
    assert "do not travel as a request parameter" in caplog.text


def test_a_callers_own_template_still_receives_the_transcript(caplog) -> None:
    model = ScriptedNative()
    agent = Agent(AgentConfig(
        name="protocol", model=model, tools=[Calculator()], max_iterations=3,
        raise_on_error=False, enable_memory=False, tool_calling_mode="native",
        prompt_protocol="messages",
        system_prompt_template="{tools_description}|{conversation_history}|{task}|{scratchpad}",
    ))
    with caplog.at_level("INFO"):
        agent.run(TASK)
    assert all(isinstance(prompt, str) for prompt in model.prompts)
    assert all("|" in prompt for prompt in model.prompts)


def test_auto_says_it_at_information_and_an_explicit_ask_at_warning(caplog) -> None:
    with caplog.at_level("INFO"):
        _agent(ScriptedNative(carries_messages=False), "auto").run(TASK)
    levels = {r.levelname for r in caplog.records if "[protocol]" in r.message}
    assert levels == {"INFO"}
    caplog.clear()
    _probe_cache().clear()
    agent_runtime._message_protocol_unavailable_warned.clear()
    with caplog.at_level("INFO"):
        _agent(ScriptedNative(carries_messages=False), "messages").run(TASK)
    levels = {r.levelname for r in caplog.records if "[protocol]" in r.message}
    assert "WARNING" in levels


def test_neither_fallback_raises() -> None:
    """A caller who set the field once and then swapped models gets an answer."""
    response = _agent(ScriptedNative(carries_messages=False), "messages").run(TASK)
    assert response.output == ANSWER


# ---------------------------------------------------------------------------
# The refusal probe
# ---------------------------------------------------------------------------


def test_a_template_that_refuses_both_on_one_turn_is_retried_split(caplog) -> None:
    model = ScriptedNative(refusals=1, model_name="refuses-once")
    with caplog.at_level("INFO"):
        response = _agent(model, "messages").run(TASK)
    assert "retrying with the reasoning as its own turn" in caplog.text
    assert response.metadata["prompt_protocol"] == "messages"
    assert isinstance(model.prompts[1], list)


def test_a_model_that_refuses_the_shape_outright_falls_back_and_is_remembered(
    caplog,
) -> None:
    model = ScriptedNative(refusals=2, model_name="refuses-always")
    with caplog.at_level("INFO"):
        response = _agent(model, "messages").run(TASK)
    assert "[protocol] the model refused the message protocol" in caplog.text
    assert isinstance(model.prompts[2], str)
    assert response.metadata["prompt_protocol"] == "flat"
    assert list(_probe_cache().values()) == ["refused"]

    # The second run on the same model does not pay for the lesson again.
    again = ScriptedNative(refusals=0, model_name="refuses-always")
    _agent(again, "messages").run(TASK)
    assert all(isinstance(prompt, str) for prompt in again.prompts)


def test_a_failure_that_is_not_about_the_request_shape_is_not_retried() -> None:
    """Auth, a missing model and a rate limit are not fixed by another shape."""
    from effgen.core.agent_react import _is_request_shape_refusal

    assert _is_request_shape_refusal({"finish_reason": "stop"}) is False
    for category in ("auth", "not_found", "rate_limited", "transient"):
        assert _is_request_shape_refusal({
            "finish_reason": "error",
            "metadata": {"error_detail": {"category": category}},
        }) is False
    assert _is_request_shape_refusal({
        "finish_reason": "error",
        "metadata": {"error_detail": {"category": "invalid_request"}},
    }) is True


def test_the_split_rendering_keeps_the_reasoning_ahead_of_the_call() -> None:
    from effgen.core.agent_runtime import _count_tool_parts, _split_reasoning_from_calls

    message = Message(role=Role.ASSISTANT, content=[
        TextPart(text=REASONING),
        ToolCallPart(tool_call_id=CALL_ID, name="calculator", arguments={}),
    ])
    split = _split_reasoning_from_calls([message])
    assert len(split) == 2
    assert [type(part).__name__ for part in split[0].content] == ["TextPart"]
    assert [type(part).__name__ for part in split[1].content] == ["ToolCallPart"]
    assert _count_tool_parts(split) == (1, 0)


def test_the_split_rendering_leaves_an_ordinary_turn_alone() -> None:
    from effgen.core.agent_runtime import _split_reasoning_from_calls

    message = Message(role=Role.USER, content=[TextPart(text="hello")])
    assert _split_reasoning_from_calls([message]) == [message]


# ---------------------------------------------------------------------------
# The streamed loops are excluded by declaration, not by omission
# ---------------------------------------------------------------------------


def test_the_streamed_loops_say_they_do_not_send_messages() -> None:
    import inspect

    from effgen.core import agent_stream_native, agent_streaming

    for module in (agent_stream_native, agent_streaming):
        source = inspect.getsource(module)
        assert "carried_by_this_loop=False" in source, module.__name__


def test_a_loop_that_does_not_carry_the_shape_answers_flat(caplog) -> None:
    agent = _agent(ScriptedNative(), "messages")
    with caplog.at_level("INFO"):
        resolved = agent._resolve_prompt_protocol(
            tools_travel_as_parameter=True, carried_by_this_loop=False,
        )
    assert resolved == "flat"
    assert "this loop does not send the conversation as messages" in caplog.text


# ---------------------------------------------------------------------------
# The capability and the converter cannot drift apart
# ---------------------------------------------------------------------------


def _model_classes() -> list[type]:
    classes: list[type] = []
    for info in pkgutil.iter_modules(effgen.models.__path__):
        try:
            module = importlib.import_module(f"effgen.models.{info.name}")
        except Exception:
            continue
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == module.__name__
            ):
                classes.append(obj)
    return classes


def _declares(cls: type) -> bool:
    try:
        obj = cls.__new__(cls)
    except TypeError:
        return False  # an abstract base declares nothing
    for attribute, value in (
        ("model_name", "an-adapter-model"), ("base_url", None), ("_is_loaded", True),
    ):
        try:
            object.__setattr__(obj, attribute, value)
        except Exception:
            return False
    try:
        return bool(obj.supports_message_protocol())
    except Exception:
        return False


def test_an_adapter_that_declares_the_protocol_actually_carries_it() -> None:
    """Read from a real conversion, not from the source text.

    Six adapters already report ``tool_call_support() == "api"`` and drop both
    tool parts on the way to the provider. The declaration is about the
    conversion, so it is checked by converting.
    """
    conversation = [
        Message(role=Role.ASSISTANT, content=[
            TextPart(text=REASONING),
            ToolCallPart(tool_call_id=CALL_ID, name="calculator", arguments={"a": 1}),
        ]),
        Message(role=Role.TOOL, content=[
            ToolResultPart(tool_call_id=CALL_ID, result="36"),
        ]),
    ]
    declared = [cls for cls in _model_classes() if _declares(cls)]
    assert declared, "no adapter declares the message protocol"
    for cls in declared:
        obj = cls.__new__(cls)
        object.__setattr__(obj, "model_name", "an-adapter-model")
        convert = getattr(obj, "_create_messages", None)
        assert callable(convert), f"{cls.__name__} declares the protocol and cannot convert"
        sent = convert(conversation)
        assert any("tool_calls" in m for m in sent), cls.__name__
        assert any("tool_call_id" in m for m in sent), cls.__name__


def test_an_adapter_that_does_not_declare_it_is_never_offered_the_shape() -> None:
    """The fallback is the declaration, so an undeclared adapter stays flat."""
    model = ScriptedNative(carries_messages=False)
    agent = _agent(model, "auto")
    assert agent._resolve_prompt_protocol(tools_travel_as_parameter=True) == "flat"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_the_protocol_is_a_plain_string_in_the_run_document() -> None:
    document = _agent(ScriptedNative(), "messages").run(TASK).to_dict()
    assert document["metadata"]["prompt_protocol"] == "messages"
    json.dumps(document["metadata"]["prompt_protocol"])
