"""The persona, the session's earlier turns and a task's content parts.

All three belong to the run's conversation rather than to the string one turn
happens to be rendered as: the persona is a system step, an earlier exchange is
the turns it was, and a picture is a content part on the question. These cover
what a caller can observe of that — the roles a request carries, what a
multi-turn session remembers when a tool is attached, and that an agent holding
both an image and a tool runs the reasoning loop instead of answering in one
call.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.messages import Message, Role
from effgen.core.multimodal import image_from
from effgen.core.thread import AgentThread, SystemStep, TaskStep
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

#: A one-pixel PNG, so a test needs no fixture file on disk.
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
    b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class Calc(BaseTool):
    """A deterministic tool, so a turn's call has one right answer."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="the expression", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        return "36"


class Scripted(BaseModel):
    """Replays a script and keeps every prompt it was handed."""

    def __init__(
        self, script: list[dict], *, native: bool = True, conversation: bool = True,
    ) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.script = script
        self.native = native
        self.conversation = conversation
        self.prompts: list[Any] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return self.native

    def tool_call_support(self) -> str:
        return "api" if self.native else "none"

    def streams_tool_calls(self) -> bool:
        return False

    def supports_conversation(self) -> bool:
        return self.conversation

    def supports_vision(self) -> bool:
        return True

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt)
        turn = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        calls = turn.get("calls")
        return GenerationResult(
            text=turn.get("text", ""),
            tokens_used=5,
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


def _answer(text: str) -> dict:
    return {"text": f"Thought: done.\nFinal Answer: {text}"}


def _call(name: str, arguments: str) -> dict:
    return {"id": "call-1", "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _agent(model: Scripted, **config: Any) -> Agent:
    settings: dict[str, Any] = {
        "name": "framed", "model": model, "tools": [Calc()],
        "max_iterations": 4, "tool_calling_mode": "hybrid",
        "raise_on_error": False,
    }
    settings.update(config)
    return Agent(AgentConfig(**settings))


def _roles(prompt: Any) -> list[str]:
    if isinstance(prompt, str):
        return []
    return [
        getattr(getattr(m, "role", None), "value", str(getattr(m, "role", "?")))
        for m in prompt
    ]


def _text_of(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    chunks = []
    for message in prompt:
        for part in getattr(message, "content", []) or []:
            if getattr(part, "text", None):
                chunks.append(part.text)
    return "\n".join(chunks)


# --- the steps themselves ----------------------------------------------------


def test_an_earlier_turn_is_a_step_that_renders_in_the_role_it_was_spoken_in() -> None:
    """A prior exchange is two messages, not one block of text."""
    from effgen.core.thread import TurnStep

    thread = AgentThread(steps=[
        TurnStep(text="my name is Ada", role="user"),
        TurnStep(text="noted", role="assistant"),
        TaskStep(text="what is my name?"),
    ])
    assert thread.to_text() == ""
    assert [(m.role.value, m.text) for m in thread.to_messages()] == [
        ("user", "my name is Ada"),
        ("assistant", "noted"),
        ("user", "what is my name?"),
    ]


def test_an_earlier_turn_round_trips_through_the_serialised_form() -> None:
    """A stored thread reads its earlier turns back with their roles."""
    from effgen.core.thread import TurnStep

    thread = AgentThread(steps=[
        TurnStep(text="hello", role="user"),
        TurnStep(text="hi", role="assistant"),
    ])
    again = AgentThread.from_dict(thread.to_dict())
    assert again.to_dict() == thread.to_dict()
    assert [step.role for step in again.prior_turns()] == ["user", "assistant"]


def test_the_history_rendering_is_for_a_frame_that_takes_one_string() -> None:
    """A text frame gets the same turns rendered, not a separate formatter."""
    from effgen.core.thread import TurnStep

    thread = AgentThread(steps=[
        TurnStep(text="my name is Ada", role="user"),
        TurnStep(text="noted", role="assistant"),
    ])
    rendered = thread.history_text()
    assert "my name is Ada" in rendered
    assert "noted" in rendered
    assert AgentThread().history_text() == ""


def test_the_frame_is_on_the_thread_and_not_in_the_transcript() -> None:
    """The persona, the earlier turns and the question render as no bytes."""
    from effgen.core.thread import TurnStep

    thread = AgentThread(steps=[
        SystemStep(text="be brief", source="persona"),
        TurnStep(text="hello", role="user"),
        TaskStep(text="what is 12 times 3?"),
    ])
    assert thread.to_text() == ""
    assert thread.system_text() == "be brief"
    assert len(thread.prior_turns()) == 1
    task = thread.task()
    assert task is not None and task.text == "what is 12 times 3?"


# --- what a run sends --------------------------------------------------------


def test_a_run_that_carries_none_of_the_three_still_sends_one_string() -> None:
    """No persona, no earlier turns, no parts — the request does not move.

    A guard, not a proof: this holds on the tree before this change too, and
    that is the point of it.
    """
    model = Scripted([_answer("36")])
    _agent(model).run("What is 12 times 3?")
    assert isinstance(model.prompts[0], str)


def test_a_custom_persona_reaches_the_system_slot_on_the_native_path() -> None:
    """The persona is a system turn, not a prefix pasted on the question."""
    persona = "You are a laconic pirate."
    model = Scripted([_answer("Aye, 36.")])
    _agent(model, system_prompt=persona).run("What is 12 times 3?")
    prompt = model.prompts[0]
    assert _roles(prompt)[0] == "system"
    assert prompt[0].text.startswith(persona)
    assert not prompt[-1].text.startswith(persona)


def test_a_persona_still_reaches_a_model_that_takes_only_a_string() -> None:
    """An adapter that does not carry a conversation still gets the persona.

    A guard, not a proof: it held before this change and has to keep holding,
    because a string frame is still how such a model is reached.
    """
    persona = "You are a laconic pirate."
    model = Scripted([_answer("Aye, 36.")], conversation=False)
    _agent(model, system_prompt=persona).run("What is 12 times 3?")
    prompt = model.prompts[0]
    assert isinstance(prompt, str)
    assert prompt.startswith(persona)


def test_a_multi_turn_session_with_a_tool_remembers_the_earlier_turns() -> None:
    """The second turn of a session is told what the first one said."""
    model = Scripted([_answer("Noted."), _answer("Paris")])
    agent = _agent(model)
    agent.run("My favourite city is Paris. Remember it.")
    first_of_turn_two = len(model.prompts)
    agent.run("Which city did I name?")
    prompt = model.prompts[first_of_turn_two]
    # No persona on this agent, so no system turn: the earlier exchange and the
    # new question, in the roles they were spoken in.
    assert _roles(prompt) == ["user", "assistant", "user"]
    assert "Paris" in _text_of(prompt)


def test_a_multi_turn_session_reaches_a_string_model_as_text() -> None:
    """A model taking one string is still told the earlier turns.

    A guard, not a proof: it held before this change. What moved is the shape
    the turns are kept in, not whether such a model is told about them.
    """
    model = Scripted([_answer("Noted."), _answer("Paris")], conversation=False)
    agent = _agent(model)
    agent.run("My favourite city is Paris. Remember it.")
    first_of_turn_two = len(model.prompts)
    agent.run("Which city did I name?")
    prompt = model.prompts[first_of_turn_two]
    assert isinstance(prompt, str)
    assert "Paris" in prompt


def test_the_tool_contract_keeps_its_place_when_the_frame_carries_roles() -> None:
    """Moving to roles moves the persona, and nothing about the tool contract.

    The contract is stated on the opening turn either way. Only the persona
    becomes a system turn, so a run that gains roles gains exactly that.
    """
    persona = "You are a laconic pirate."
    model = Scripted([_answer("36")])
    agent = _agent(model, system_prompt=persona)
    agent.run("What is 12 times 3?")
    prompt = model.prompts[0]
    assert prompt[0].text == persona
    contract = agent._tool_contract()
    assert contract
    assert contract in prompt[-1].text
    assert contract not in prompt[0].text


# --- a picture and a tool in the same run ------------------------------------


def test_an_agent_given_an_image_and_tools_runs_the_loop() -> None:
    """The run reasons and calls a tool instead of answering in one call."""
    model = Scripted([
        {"calls": [_call("calculator", '{"expression": "12*3"}')]},
        _answer("36 apples"),
    ])
    response = _agent(model).run(
        "How many apples, times three?", inputs=[image_from(PNG, mime="image/png")],
    )
    assert len(model.prompts) == 2
    assert len(response.tool_calls) == 1
    assert response.output == "36 apples"


def test_the_image_travels_on_the_question_every_turn() -> None:
    """Each turn's request carries the part, so the model can still see it."""
    model = Scripted([
        {"calls": [_call("calculator", '{"expression": "12*3"}')]},
        _answer("36 apples"),
    ])
    _agent(model).run(
        "How many apples, times three?", inputs=[image_from(PNG, mime="image/png")],
    )
    # Two turns, or the claim is vacuous: a run that answered in one call
    # carries the picture on that call whatever the loop does.
    assert len(model.prompts) == 2
    for prompt in model.prompts:
        assert isinstance(prompt, list)
        parts = [p for m in prompt for p in (m.content or [])]
        assert any(getattr(p, "type", None) == "image" for p in parts)


def test_an_image_reaches_a_model_that_is_driven_by_the_text_scaffold() -> None:
    """A model with no native tool calling still gets the picture."""
    model = Scripted([
        {"text": "Thought: check.\nAction: calculator\nAction Input: "
                 '{"expression": "12*3"}'},
        _answer("36 apples"),
    ], native=False)
    response = _agent(model).run(
        "How many apples, times three?", inputs=[image_from(PNG, mime="image/png")],
    )
    assert len(response.tool_calls) == 1
    parts = [p for m in model.prompts[0] for p in (m.content or [])]
    assert any(getattr(p, "type", None) == "image" for p in parts)


def test_the_run_records_that_its_question_carried_a_part() -> None:
    """The thread's task step holds it, so a stored run still has it."""
    model = Scripted([_answer("one apple")])
    response = _agent(model).run(
        "How many apples?", inputs=[image_from(PNG, mime="image/png")],
    )
    thread = response.metadata["thread"]
    task = thread.task()
    assert task is not None
    assert [getattr(p, "type", None) for p in task.parts] == ["image"]
    assert json.loads(json.dumps(response.to_dict()["metadata"]["thread"]))


def test_a_tool_free_agent_with_an_image_still_answers_directly() -> None:
    """Nothing for a loop to do, so the run is one call, as it always was.

    A guard, not a proof: an agent with no tools answered in one call before
    this change and must keep doing so.
    """
    model = Scripted([{"text": "one apple"}])
    response = _agent(model, tools=[]).run(
        "How many apples?", inputs=[image_from(PNG, mime="image/png")],
    )
    assert len(model.prompts) == 1
    assert response.output == "one apple"


# --- what an adapter declares ------------------------------------------------


def test_every_adapter_that_converts_a_conversation_declares_it() -> None:
    """The declaration follows the seam, so a new adapter cannot forget it."""
    import importlib

    adapters = [
        ("openai_adapter", "OpenAIAdapter"),
        ("gemini_adapter", "GeminiAdapter"),
        ("anthropic_adapter", "AnthropicAdapter"),
        ("groq_adapter", "GroqAdapter"),
        ("together_adapter", "TogetherAdapter"),
        ("cerebras_adapter", "CerebrasAdapter"),
        ("fireworks_adapter", "FireworksAdapter"),
        ("hf_inference_adapter", "HFInferenceAdapter"),
        ("replicate_adapter", "ReplicateAdapter"),
    ]
    declared = []
    for module_name, class_name in adapters:
        module = importlib.import_module(f"effgen.models.{module_name}")
        cls = getattr(module, class_name)
        assert hasattr(cls, "_create_messages"), class_name
        assert BaseModel.supports_conversation(cls.__new__(cls)) is True, class_name
        declared.append(class_name)
    assert len(declared) == 9


def test_a_model_that_takes_only_text_declares_that_it_does_not() -> None:
    """The default is no, so a string-only engine is never sent a list."""
    model = Scripted([_answer("36")], conversation=False)
    assert model.supports_conversation() is False
    assert _agent(model)._model_carries_a_conversation() is False


# --- the configuration a run was built with ----------------------------------


def test_a_plain_list_of_guardrails_is_used_rather_than_dropped() -> None:
    """An agent configured with guardrails runs with them."""
    from effgen.guardrails import PIIGuardrail

    model = Scripted([_answer("36")])
    agent = _agent(model, guardrails=[PIIGuardrail()])
    assert agent._guardrail_chain is not None
    assert len(agent._guardrail_chain.guardrails) == 1


def test_a_guardrails_value_that_is_not_one_says_so() -> None:
    """Silently unguarded is the failure a caller cannot see."""
    model = Scripted([_answer("36")])
    with pytest.raises(TypeError, match="guardrails"):
        _agent(model, guardrails={"pii": True})


def test_a_message_carries_its_role_into_the_request(monkeypatch: Any) -> None:
    """A guard on the shape the frame builder produces."""
    from effgen.core.thread import TurnStep

    thread = AgentThread(steps=[
        SystemStep(text="be brief"),
        TurnStep(text="hello", role="user"),
        TaskStep(text="what?"),
    ])
    model = Scripted([_answer("36")])
    messages = _agent(model)._frame_as_messages("the frame", thread, carry_roles=True)
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER, Role.USER]
    assert isinstance(messages[0], Message)
    assert messages[-1].text == "the frame"


# --- a picture on a model somebody else is serving ---------------------------


def _served(**kwargs: Any):
    from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter

    return OpenAICompatibleAdapter(
        model_name="some-org/some-vision-model",
        base_url="http://127.0.0.1:9/v1",
        context_length=8192,
        **kwargs,
    )


def _image_prompt() -> list[Message]:
    from effgen.core.messages import TextPart

    return [Message(role=Role.USER,
                    content=[TextPart(text="what is this?"),
                             image_from(PNG, mime="image/png")])]


def test_a_served_model_is_not_refused_an_image_by_another_vendors_catalog() -> None:
    """No catalog here describes it, so the endpoint answers for itself."""
    adapter = _served()
    assert adapter._vision_support() is None
    adapter._validate_media_support(_image_prompt())


def test_a_served_model_is_not_refused_an_image_on_the_tool_calling_path() -> None:
    """The same question is asked at both entry points, and answered the same.

    A turn carrying tool definitions goes through a different method, and it
    had its own copy of the catalog rule.
    """
    from effgen.errors import CapabilityNotSupportedError

    adapter = _served()
    adapter._is_loaded = True
    try:
        adapter.generate_with_tools(_image_prompt(), tools=[])
    except CapabilityNotSupportedError:  # pragma: no cover - the defect
        pytest.fail("the tool-calling path refused an image on its own catalog rule")
    except Exception:  # noqa: BLE001 - the endpoint is not there; that is fine
        pass

    refusing = _served(supports_vision=False)
    refusing._is_loaded = True
    with pytest.raises(CapabilityNotSupportedError):
        refusing.generate_with_tools(_image_prompt(), tools=[])


def test_a_caller_who_knows_the_served_model_takes_no_images_is_told_so() -> None:
    """A declared no is a refusal before the call is billed.

    A guard, not a proof: the refusal also happens on the tree before this
    change, where it comes from a rule about another vendor's model ids rather
    than from what the caller declared. What must not happen is a declared
    ``False`` silently becoming a sent request.
    """
    from effgen.errors import CapabilityNotSupportedError

    adapter = _served(supports_vision=False)
    with pytest.raises(CapabilityNotSupportedError):
        adapter._validate_media_support(_image_prompt())


def test_a_text_only_model_of_this_vendor_is_still_refused_an_image() -> None:
    """The catalog's answer still holds for the models it describes."""
    from effgen.errors import CapabilityNotSupportedError
    from effgen.models.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model_name = "gpt-3.5-turbo"
    assert adapter._vision_support() is not None
    with pytest.raises(CapabilityNotSupportedError):
        adapter._validate_media_support(_image_prompt())
