"""A tool result is a tool message, and every call gets one.

A conversation held as messages answers each tool call with a ``tool`` turn
quoting the call's own id. Two things follow, and this file holds both:

* **no conversation leaves effGen with a call nothing answered.** The loop
  declines to dispatch on several paths — it has the result already, the same
  call keeps coming back, the tool is not one this agent holds — and a provider
  rejects a conversation carrying a call with no reply, so a declined call gets
  a reply saying so;
* **the call id survives the trip to the provider**, on every adapter whose
  protocol has somewhere to put it, and the adapters whose protocol does not
  say so rather than sending a shape the provider will not take.

Everything here is offline: the provider is scripted and the adapters are asked
only what they would have sent.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.messages import (
    Message,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from effgen.core.thread import ActionStep, AgentThread, ObservationStep, ThoughtStep
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

CALL_A = "call_provider_aaa"
CALL_B = "call_provider_bbb"
RESULT = "36"


# ---------------------------------------------------------------------------
# A tool, and a provider that answers the way a real one does
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
        return RESULT


class Scripted(BaseModel):
    """Answers a scripted list of provider turns, calls and ids included."""

    def __init__(
        self, turns: list[dict[str, Any]], *, carries_messages: bool = True
    ) -> None:
        super().__init__(model_name="scripted-native", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.turns = turns
        self.index = 0
        self._carries = carries_messages
        self.prompts: list[Any] = []

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
        turn = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        return GenerationResult(
            text=turn.get("text", ""),
            tokens_used=8,
            finish_reason=turn.get("finish_reason", "stop"),
            model_name=self.model_name,
            metadata={"tool_calls": turn["calls"]} if turn.get("calls") else {},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _agent(model: BaseModel, protocol: str = "flat", *, tools: bool = True) -> Agent:
    return Agent(AgentConfig(
        name="tool-messages", model=model,
        tools=[Calculator()] if tools else [],
        max_iterations=4, raise_on_error=False, enable_memory=False,
        tool_calling_mode="native", prompt_protocol=protocol,
    ))


def _run(turns: list[dict[str, Any]], task: str = "Check the ledger total.") -> Any:
    return _agent(Scripted(turns)).run(task)


def _tool_message_ids(thread: AgentThread) -> list[str]:
    """The call ids the conversation's tool turns answer."""
    return [
        part.tool_call_id
        for message in thread.to_messages()
        if message.role is Role.TOOL
        for part in message.content
        if isinstance(part, ToolResultPart)
    ]


def _called_ids(thread: AgentThread) -> list[str]:
    """The call ids the conversation's assistant turns asked for."""
    return [
        part.tool_call_id
        for message in thread.to_messages()
        for part in message.content
        if isinstance(part, ToolCallPart)
    ]


# ---------------------------------------------------------------------------
# The declining paths: every requested call is answered
# ---------------------------------------------------------------------------

REPEAT = _call(CALL_A, "calculator", {"expression": "12*3"})
UNKNOWN = _call(CALL_B, "ledger_lookup", {"account": "main"})

DECLINING_PATHS: dict[str, list[dict[str, Any]]] = {
    # the same call over and over: replayed from the record, then the loop
    # stops offering the tool, then the run ends without an answer
    "repeat_and_loop_breaker": [
        {"text": "again", "finish_reason": "tool_calls", "calls": [REPEAT]},
    ],
    # a tool this agent does not hold, asked for on its own
    "unknown_tool": [
        {"text": "look it up", "finish_reason": "tool_calls", "calls": [UNKNOWN]},
        {"text": "Final Answer: 36"},
    ],
    # a batch where one of the two names a tool this agent does not hold
    "unknown_tool_in_a_batch": [
        {"text": "both please", "finish_reason": "tool_calls",
         "calls": [REPEAT, UNKNOWN]},
        {"text": "Final Answer: 36"},
    ],
    # a batch of calls the agent does not hold at all
    "batch_of_unknown_tools": [
        {"text": "both please", "finish_reason": "tool_calls",
         "calls": [UNKNOWN, _call("call_c", "ledger_write", {"row": 1})]},
        {"text": "Final Answer: 36"},
    ],
    # a run that keeps calling and never writes an answer
    "loop_to_the_iteration_cap": [
        {"text": "one", "finish_reason": "tool_calls",
         "calls": [_call("c1", "calculator", {"expression": "1+1"})]},
        {"text": "two", "finish_reason": "tool_calls",
         "calls": [_call("c2", "calculator", {"expression": "2+2"})]},
        {"text": "three", "finish_reason": "tool_calls",
         "calls": [_call("c3", "calculator", {"expression": "3+3"})]},
        {"text": "four", "finish_reason": "tool_calls",
         "calls": [_call("c4", "calculator", {"expression": "4+4"})]},
    ],
}


@pytest.mark.parametrize("path", sorted(DECLINING_PATHS))
def test_no_run_ends_holding_a_call_nothing_answered(path: str) -> None:
    """Every call the model made has a reply, on every path that declines one."""
    thread = _run(DECLINING_PATHS[path]).metadata["thread"]
    assert thread.unanswered_calls() == []
    assert thread.unanswered_call_ids() == []
    assert sorted(set(_called_ids(thread))) == sorted(set(_tool_message_ids(thread)))


@pytest.mark.parametrize("path", sorted(DECLINING_PATHS))
def test_a_declined_call_says_it_was_declined(path: str) -> None:
    """A reply the tool did not produce is marked as one, not passed off as a result."""
    thread = _run(DECLINING_PATHS[path]).metadata["thread"]
    observations = thread.observations()
    assert observations, "no tool ever replied"
    for observation in observations:
        assert observation.declined is None or observation.declined in {
            "loop_detected", "already_computed", "unknown_tool",
        }


def test_a_call_for_a_tool_the_agent_does_not_hold_is_answered() -> None:
    """The reply names the tools that are callable rather than saying nothing."""
    thread = _run(DECLINING_PATHS["unknown_tool"]).metadata["thread"]
    declined = [s for s in thread.observations() if s.declined == "unknown_tool"]
    assert len(declined) == 1
    assert "calculator" in declined[0].text
    assert declined[0].call_id == CALL_B


def test_a_batch_call_for_a_missing_tool_reaches_the_run(caplog) -> None:
    """A batched call the agent cannot make is recorded and answered, not dropped."""
    with caplog.at_level("INFO"):
        response = _run(DECLINING_PATHS["unknown_tool_in_a_batch"])
    thread = response.metadata["thread"]
    assert [step.tool for step in thread.actions()] == ["calculator", "ledger_lookup"]
    declined = [s for s in thread.observations() if s.declined == "unknown_tool"]
    assert len(declined) == 1 and declined[0].call_id == CALL_B
    # the run says it happened, once per call, with a phrase a log search finds
    assert sum(
        "is not a tool this agent holds" in record.getMessage()
        for record in caplog.records
    ) == 1
    # and the model reads the refusal in the run it is shown
    assert "ledger_lookup" in thread.to_text()


def test_a_repeat_is_answered_from_the_record_with_the_tools_own_result() -> None:
    """The reply to a repeated call is the result, so it is not marked declined."""
    thread = _run(DECLINING_PATHS["repeat_and_loop_breaker"]).metadata["thread"]
    replies = [s for s in thread.observations() if s.text == RESULT]
    assert len(replies) > 1, "the repeat was never answered from the record"
    assert all(step.declined is None for step in replies)


def test_a_run_the_loop_breaker_stopped_still_answered_its_last_call() -> None:
    """The call that tripped the breaker is in the run, with the reply it got."""
    thread = _run(
        [{"text": "same", "finish_reason": "tool_calls", "calls": [REPEAT]}]
    ).metadata["thread"]
    assert thread.unanswered_call_ids() == []
    assert any(s.declined == "loop_detected" for s in thread.observations())


# ---------------------------------------------------------------------------
# The batch path records what each call did
# ---------------------------------------------------------------------------


BATCH = [
    {"text": "two sums", "finish_reason": "tool_calls", "calls": [
        _call(CALL_A, "calculator", {"expression": "12*3"}),
        _call(CALL_B, "calculator", {"expression": "6*6"}),
    ]},
    {"text": "Final Answer: 36"},
]
SINGLE = [
    {"text": "one sum", "finish_reason": "tool_calls",
     "calls": [_call(CALL_A, "calculator", {"expression": "12*3"})]},
    {"text": "Final Answer: 36"},
]


def test_the_batch_path_records_what_each_call_ran_with_and_returned() -> None:
    """A batched call is a call the run made, and its record says what it did."""
    response = _run(BATCH)
    assert response.tool_calls.total == 2
    assert len(response.tool_calls) == 2
    for record in response.tool_calls:
        assert record.name == "calculator"
        assert record.arguments, "a record with no arguments cannot say what ran"
        assert record.result == RESULT
        assert record.duration is not None and record.duration >= 0
        assert record.iteration == 1


def test_a_batched_call_records_the_same_fields_as_a_single_one() -> None:
    """The detail on the batch path is the detail on the one-call path."""
    batched = _run(BATCH).tool_calls[0].to_dict()
    single = _run(SINGLE).tool_calls[0].to_dict()
    assert sorted(batched) == sorted(single)
    for field in ("name", "result", "error", "iteration", "ok"):
        assert batched[field] == single[field]
    assert (batched["duration"] is None) == (single["duration"] is None)
    # The two paths agree on what was called with; they spell it differently.
    # A provider-native batch arrives parsed and is recorded as the mapping it
    # is, while a single call is recorded as the text the turn wrote.
    assert json.loads(single["arguments"]) == batched["arguments"]


def test_a_batched_call_quotes_the_id_the_provider_minted() -> None:
    """Each tool turn answers the provider's own id, not one effGen invented."""
    thread = _run(BATCH).metadata["thread"]
    assert [step.call_id for step in thread.actions()] == [CALL_A, CALL_B]
    assert _tool_message_ids(thread)[:2] == [CALL_A, CALL_B]


# ---------------------------------------------------------------------------
# A call with no id of its own is named the same way wherever it is rendered
# ---------------------------------------------------------------------------


def test_a_step_names_the_same_call_alone_as_it_does_in_its_thread() -> None:
    """A step rendered on its own and in its thread name one call, not two."""
    action = ActionStep(tool="calculator", raw='{"expression": "12*3"}')
    observation = ObservationStep(text=RESULT)
    thread = AgentThread(steps=[
        ThoughtStep(text="first"),
        ActionStep(tool="calculator", raw="{}"), ObservationStep(text="1"),
        action, observation,
    ])
    rendered = thread.to_messages()
    assert action.to_messages()[0].content[-1].tool_call_id == (
        rendered[-2].content[-1].tool_call_id
    )
    assert observation.to_messages()[0].content[0].tool_call_id == (
        rendered[-1].content[0].tool_call_id
    )


def test_a_call_that_carried_a_provider_id_keeps_it() -> None:
    """A minted id never replaces one the provider supplied."""
    thread = AgentThread()
    thread.append(ActionStep(tool="calculator", raw="{}", call_id=CALL_A))
    thread.append(ObservationStep(text=RESULT))
    assert thread.actions()[0].call_id == CALL_A
    assert thread.observations()[0].call_id == CALL_A


def test_a_thread_built_from_a_transcript_names_every_call_it_recovered() -> None:
    """A run read back from flat text still answers each call it recovered."""
    thread = AgentThread.from_scratchpad(
        "\nThought: add them\nAction: calculator\nAction Input: 12*3"
        "\nObservation: 36"
    )
    assert thread.unanswered_call_ids() == []
    assert thread.actions()[0].call_id
    assert thread.observations()[0].call_id == thread.actions()[0].call_id


# ---------------------------------------------------------------------------
# The call id reaches the provider — one case per adapter
# ---------------------------------------------------------------------------

CONVERSATION = [
    Message(role=Role.SYSTEM, content=[TextPart(text="You are careful.")]),
    Message(role=Role.USER, content=[TextPart(text="What is 12*3?")]),
    Message(role=Role.ASSISTANT, content=[
        TextPart(text="I will use the calculator."),
        ToolCallPart(tool_call_id=CALL_A, name="calculator",
                     arguments={"expression": "12*3"}),
    ]),
    Message(role=Role.TOOL, content=[
        ToolResultPart(tool_call_id=CALL_A, result=RESULT),
    ]),
]

#: Every adapter that can be offered tool definitions, with the catalog its
#: capability declarations are read from. The conversion has one name on all of
#: them — ``_create_messages`` — even though the payloads differ: an
#: OpenAI-protocol endpoint takes ``tool_calls`` and a ``tool`` message, the
#: Messages API takes ``tool_use``/``tool_result`` blocks, and Gemini takes
#: ``function_call``/``function_response`` parts.
ADAPTERS: list[tuple[str, str, str, str, str]] = [
    ("openai", "openai_adapter", "OpenAIAdapter",
     "openai_models", "OPENAI_MODELS"),
    ("openai_compatible", "openai_compatible_adapter", "OpenAICompatibleAdapter",
     "openai_models", "OPENAI_MODELS"),
    ("groq", "groq_adapter", "GroqAdapter", "groq_models", "GROQ_MODELS"),
    ("together", "together_adapter", "TogetherAdapter",
     "together_models", "TOGETHER_MODELS"),
    ("hf", "hf_inference_adapter", "HFInferenceAdapter",
     "hf_inference_models", "HF_MODELS"),
    ("cerebras", "cerebras_adapter", "CerebrasAdapter",
     "cerebras_models", "CEREBRAS_MODELS"),
    ("fireworks", "fireworks_adapter", "FireworksAdapter",
     "fireworks_models", "FIREWORKS_MODELS"),
    ("anthropic", "anthropic_adapter", "AnthropicAdapter",
     "anthropic_models", "ANTHROPIC_MODELS"),
    ("gemini", "gemini_adapter", "GeminiAdapter",
     "gemini_models", "GEMINI_MODELS"),
    ("replicate", "replicate_adapter", "ReplicateAdapter",
     "replicate_models", "REPLICATE_MODELS"),
]

_CATALOG = {row[0]: (row[3], row[4]) for row in ADAPTERS}


def _catalog(module: str, name: str) -> dict[str, Any]:
    return getattr(importlib.import_module(f"effgen.models.{module}"), name)


def _tool_capable_model(provider: str, module: str, name: str) -> str | None:
    """A model this catalog says takes tool definitions, or None."""
    for model, info in _catalog(module, name).items():
        if not isinstance(info, dict) or not info.get("supports_native_tools"):
            continue
        if provider == "replicate" and info.get("input_schema") != "messages":
            continue
        return str(model)
    return None


def _adapter(module: str, class_name: str, model: str, provider: str) -> Any:
    cls = getattr(importlib.import_module(f"effgen.models.{module}"), class_name)
    obj = cls.__new__(cls)
    object.__setattr__(obj, "model_name", model)
    object.__setattr__(obj, "base_url", None)
    object.__setattr__(obj, "api_key", "")
    object.__setattr__(obj, "_is_loaded", True)
    object.__setattr__(obj, "_info", _catalog(*_CATALOG[provider]).get(model, {}))
    return obj


def _for_provider(row: tuple[str, str, str, str, str]) -> Any:
    """The adapter under test, or a skip when the catalog offers no tool model."""
    provider, module, class_name, catalog_module, catalog_name = row
    model = _tool_capable_model(provider, catalog_module, catalog_name)
    if model is None:
        pytest.skip(f"{provider} lists no tool-capable model")
    return _adapter(module, class_name, model, provider)


@pytest.mark.parametrize("row", ADAPTERS, ids=[row[0] for row in ADAPTERS])
def test_an_adapter_that_takes_tools_says_whether_it_carries_a_call(
    row: tuple[str, str, str, str, str],
) -> None:
    """A declaration is about the conversion, so it is read off a real model."""
    adapter = _for_provider(row)
    assert adapter.supports_tool_calling() is True
    assert adapter.supports_message_protocol() is True


@pytest.mark.parametrize("row", ADAPTERS, ids=[row[0] for row in ADAPTERS])
def test_a_call_id_reaches_the_provider_and_so_does_its_answer(
    row: tuple[str, str, str, str, str],
) -> None:
    """The id is on the call and on the turn answering it, in what is sent."""
    adapter = _for_provider(row)
    sent = json.dumps(adapter._create_messages(CONVERSATION), default=str)
    # once for the call the assistant made, once for the turn answering it
    assert sent.count(CALL_A) == 2, sent
    assert "calculator" in sent


@pytest.mark.parametrize("row", ADAPTERS, ids=[row[0] for row in ADAPTERS])
def test_one_message_is_a_conversation_of_one(
    row: tuple[str, str, str, str, str],
) -> None:
    """A caller who passes a single message is not one the converter refuses."""
    adapter = _for_provider(row)
    sent = adapter._create_messages(
        Message(role=Role.USER, content=[TextPart(text="hello")])
    )
    assert sent is not None, row[0]
    assert "hello" in json.dumps(sent, default=str), row[0]


def test_a_request_assembled_from_one_message_still_carries_it() -> None:
    """Hoisting a conversation's instructions does not require a conversation.

    The provider whose instructions live outside the message array reads them
    off the turns it was given, and one turn is as many as it needs.
    """
    from effgen.models.base import GenerationConfig

    adapter = _adapter(
        "anthropic_adapter", "AnthropicAdapter",
        _tool_capable_model("anthropic", "anthropic_models", "ANTHROPIC_MODELS")
        or "claude-3-5-sonnet-latest",
        "anthropic",
    )
    for message, expected in (
        (Message(role=Role.USER, content=[TextPart(text="hello")]), "hello"),
        (Message(role=Role.SYSTEM, content=[TextPart(text="be terse")]), "be terse"),
    ):
        request = adapter._build_request(message, GenerationConfig(), None, None, {})
        assert request["messages"], "an empty message array is rejected outright"
        assert expected in json.dumps(request, default=str)


def test_an_adapter_with_no_tool_role_says_so_rather_than_sending_the_shape() -> None:
    """A local chat-template engine declares nothing it cannot carry."""
    from effgen.models.transformers_engine import TransformersEngine
    from effgen.models.vllm_engine import VLLMEngine

    for cls in (TransformersEngine, VLLMEngine):
        engine = cls.__new__(cls)
        assert engine.supports_message_protocol() is False


def test_a_provider_without_a_tool_role_degrades_and_says_so_once(caplog) -> None:
    """The run falls back to the flat rendering and the reason is stated once."""
    model = Scripted(SINGLE, carries_messages=False)
    with caplog.at_level("INFO"):
        response = _agent(model, "messages").run("Check the ledger total.")
    assert response.metadata["prompt_protocol"] == "flat"
    said = [
        record for record in caplog.records
        if "messages is not available on this model" in record.getMessage()
    ]
    assert len(said) == 1


def test_a_lazily_loaded_adapter_answers_for_the_adapter_it_wraps() -> None:
    """A wrapper does not get to declare a capability its model does not have."""
    from effgen.models.lazy import LazyModel

    wrapped = LazyModel(Scripted(SINGLE, carries_messages=False))
    assert wrapped.supports_message_protocol() is False
    assert LazyModel(Scripted(SINGLE)).supports_message_protocol() is True


# ---------------------------------------------------------------------------
# The run under the message protocol
# ---------------------------------------------------------------------------


def test_a_run_under_the_message_protocol_answers_every_call_it_made() -> None:
    """The conversation the provider is sent holds no call with no result."""
    response = _agent(Scripted(BATCH), "messages").run("Check the ledger total.")
    thread = response.metadata["thread"]
    assert response.metadata["prompt_protocol"] == "messages"
    assert thread.unanswered_call_ids() == []
    calls = _called_ids(thread)
    assert calls == [CALL_A, CALL_B]
    assert _tool_message_ids(thread) == calls


def test_the_wire_conversion_answers_every_call_the_conversation_holds() -> None:
    """Converted for a provider, each call still has the turn that answers it."""
    from effgen.models.openai_adapter import OpenAIAdapter

    thread = _agent(
        Scripted(DECLINING_PATHS["unknown_tool_in_a_batch"]), "messages"
    ).run("Check the ledger total.").metadata["thread"]
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    object.__setattr__(adapter, "model_name", "an-adapter-model")
    sent = [adapter._message_to_openai(m) for m in thread.to_messages()]
    asked = {
        call["id"] for message in sent for call in message.get("tool_calls") or []
    }
    answered = {
        message["tool_call_id"] for message in sent if message.get("role") == "tool"
    }
    assert asked and asked == answered
