"""A request built by an agent carries the cache breakpoints the run asked for.

``AgentConfig.cache_system_prompt`` and ``AgentConfig.cache_tools`` have been
documented defaults for a long time and reached no request: the two methods that
applied them were never called, so a marker never left the process. These assert
the whole path — the run's ask, the adapter's rendering, and the caller's ability
to turn it off.

Offline: the Anthropic adapter's request assembly is exercised directly and
through an agent whose client is scripted. No key is read and no call is made.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.anthropic_adapter import AnthropicAdapter
from effgen.models.anthropic_cache import (
    MAX_CACHE_BREAKPOINTS,
    count_cache_breakpoints,
)
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
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

FULL_ASK = {"system": True, "tools": True, "conversation": True}


def _adapter() -> AnthropicAdapter:
    stub: Any = AnthropicAdapter.__new__(AnthropicAdapter)
    stub.model_name = "claude-sonnet-4-6"
    return stub


def _request(messages: int = 4) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-6",
        "system": "You are a terse clerk.",
        "tools": [{"name": "ledger"}, {"name": "notes"}],
        "messages": [
            {"role": "user", "content": f"turn {i}"} for i in range(messages)
        ],
    }


def test_a_full_ask_places_exactly_the_declared_number():
    adapter = _adapter()
    request = _request()
    placed = adapter._apply_prompt_cache(request, FULL_ASK)
    assert placed == MAX_CACHE_BREAKPOINTS
    assert count_cache_breakpoints(
        request["system"], request["messages"], request["tools"]
    ) == MAX_CACHE_BREAKPOINTS


def test_the_moving_breakpoint_sits_on_the_last_completed_turn():
    """Never on the turn being written: that writes a cache nobody reads."""
    adapter = _adapter()
    request = _request(messages=4)
    adapter._apply_prompt_cache(request, FULL_ASK)
    marked = [
        i for i, m in enumerate(request["messages"])
        if isinstance(m["content"], list)
        and "cache_control" in m["content"][-1]
    ]
    assert marked == [0, len(request["messages"]) - 2]
    assert request["messages"][-1]["content"] == "turn 3"


def test_both_flags_off_places_nothing():
    adapter = _adapter()
    request = _request()
    placed = adapter._apply_prompt_cache(
        request, {"system": False, "tools": False, "conversation": False}
    )
    assert placed == 0
    assert count_cache_breakpoints(
        request["system"], request["messages"], request["tools"]
    ) == 0
    assert request["system"] == "You are a terse clerk."


def test_the_budget_is_never_exceeded_and_drops_from_the_bottom():
    """A policy with room for two keeps the two that change least."""
    adapter = _adapter()
    request = _request()
    adapter.prompt_cache_policy = lambda: PromptCachePolicy(  # type: ignore[method-assign]
        style="explicit", max_breakpoints=2, reports_cached_tokens=True
    )
    placed = adapter._apply_prompt_cache(request, FULL_ASK)
    assert placed == 2
    assert "cache_control" in request["tools"][-1]
    assert isinstance(request["system"], list)
    assert all(isinstance(m["content"], str) for m in request["messages"])


def test_markers_the_caller_placed_are_counted_against_the_budget():
    adapter = _adapter()
    request = _request()
    request["system"] = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    placed = adapter._apply_prompt_cache(request, FULL_ASK)
    assert placed == MAX_CACHE_BREAKPOINTS - 1
    assert count_cache_breakpoints(
        request["system"], request["messages"], request["tools"]
    ) == MAX_CACHE_BREAKPOINTS


def test_an_automatic_provider_is_never_marked():
    adapter = _adapter()
    adapter.prompt_cache_policy = lambda: PromptCachePolicy(  # type: ignore[method-assign]
        style="automatic", reports_cached_tokens=True
    )
    request = _request()
    assert adapter._apply_prompt_cache(request, FULL_ASK) == 0
    assert request["system"] == "You are a terse clerk."


def test_the_definitions_are_marked_however_they_were_handed_over():
    """A run passes the tools as a keyword argument, not in the positional slot.

    The tool list is the most stable and most expensive part of the prefix and
    the first thing this provider evaluates, so a request that carries the
    definitions has to carry the breakpoint that belongs on them, whichever way
    the caller handed them over.
    """
    from effgen.core.agent_generation import model_call_kwargs
    from effgen.core.messages import Message, Role
    from effgen.models.base import GenerationConfig

    adapter = _adapter()
    tools = [{"name": "ledger", "description": "a", "input_schema": {}},
             {"name": "notes", "description": "b", "input_schema": {}}]
    conversation = [
        Message(role=Role.USER, content="turn one"),
        Message(role=Role.ASSISTANT, content="turn two"),
        Message(role=Role.USER, content="turn three"),
    ]
    # Exactly the keyword arguments the run forwards to the adapter.
    forwarded = model_call_kwargs({"tools": tools, "prompt_cache": FULL_ASK})
    assert set(forwarded) == {"tools", "prompt_cache"}

    request = adapter._build_request(
        conversation, GenerationConfig(), "You are a terse clerk.", None, dict(forwarded),
    )
    assert count_cache_breakpoints(
        request.get("system"), request.get("messages"), request.get("tools")
    ) == MAX_CACHE_BREAKPOINTS
    assert "cache_control" in request["tools"][-1]
    assert "prompt_cache" not in request, "the ask must not reach the wire"


def test_the_caller_conversation_is_not_mutated():
    adapter = _adapter()
    messages = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"},
                {"role": "user", "content": "three"}]
    request = {"model": "m", "messages": messages, "system": "s", "tools": [{"name": "t"}]}
    adapter._apply_prompt_cache(request, FULL_ASK)
    assert messages[0]["content"] == "one"
    assert request["messages"] is not messages


# --------------------------------------------------------------------------
# Through an agent: the run's ask reaches the request, and never the wire.
# --------------------------------------------------------------------------

class Ledger(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="ledger", description="Look up a figure.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="What to look up.", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        return "24"


class Explicit(BaseModel):
    """An engine that places breakpoints, recording what it was asked for."""

    def __init__(self) -> None:
        super().__init__(model_name="explicit", model_type=ModelType.ANTHROPIC)
        self._is_loaded = True
        self.asks: list[Any] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return True

    def supports_message_protocol(self) -> bool:
        return True

    def supports_conversation(self) -> bool:
        return True

    def prompt_cache_policy(self) -> PromptCachePolicy:
        return PromptCachePolicy(
            style="explicit", max_breakpoints=MAX_CACHE_BREAKPOINTS,
            reports_cached_tokens=True, reports_cache_writes=True,
        )

    def generate(self, prompt, config: GenerationConfig | None = None, **kwargs: Any):
        self.asks.append(kwargs.get("prompt_cache"))
        self.index += 1
        if self.index == 1:
            return GenerationResult(
                text="", tokens_used=4, finish_reason="tool_calls",
                model_name=self.model_name,
                metadata={"tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "ledger",
                                 "arguments": json.dumps({"query": "q1"})},
                }]},
            )
        return GenerationResult(
            text="The figure is 24.", tokens_used=5, finish_reason="stop",
            model_name=self.model_name,
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _run(**config: Any) -> Explicit:
    model = Explicit()
    agent = Agent(AgentConfig(
        name="cache", model=model, tools=[Ledger()], max_iterations=4,
        raise_on_error=False, tool_calling_mode="native", **config,
    ))
    agent.run("Look up q1 and tell me the figure.")
    return model


@pytest.mark.parametrize("flags,expected", [
    ({}, {"system": True, "tools": True, "conversation": True}),
    ({"cache_system_prompt": False, "cache_tools": False},
     {"system": False, "tools": False, "conversation": True}),
])
def test_the_run_hands_the_adapter_its_ask(flags, expected):
    model = _run(**flags)
    assert model.asks, "no request was made"
    assert all(ask == expected for ask in model.asks), model.asks


def test_an_automatic_provider_is_asked_for_nothing():
    """The key never appears, so a provider that matches the prefix itself
    receives the request it received before any of this existed."""

    class Automatic(Explicit):
        def prompt_cache_policy(self) -> PromptCachePolicy:
            return PromptCachePolicy(style="automatic", reports_cached_tokens=True)

    model = Automatic()
    agent = Agent(AgentConfig(
        name="auto", model=model, tools=[Ledger()], max_iterations=4,
        raise_on_error=False, tool_calling_mode="native",
    ))
    agent.run("Look up q1 and tell me the figure.")
    assert model.asks and all(ask is None for ask in model.asks)
