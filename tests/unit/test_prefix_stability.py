"""A run keeps the prompt prefix a provider's cache can match.

Two things used to throw it away. The turn that asks for the answer after the
guards stop offering tools dropped the definitions *and* rebuilt the request
from a different frame, so the next request shared almost nothing with the last.
And a session's second run changed shape, so the question after the first one
started a new prefix instead of extending it.

Both now hold where the provider has a cache to keep and can express what the
turn needs. Where it declares neither, the request is the one effGen always
sent — which is what the byte-identity assertions here are for.

Offline: the provider is scripted, so what is asserted is the request object the
framework handed it and the keyword arguments that went with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from effgen.core.agent import Agent, AgentConfig
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

TASK = "Look up the shop ledger figure and then say it in a sentence."
FOLLOW_UP = "And what did I just ask you?"
ANSWER = "The ledger figure is 24."


class Ledger(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="ledger", description="Look up a figure in the shop ledger.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="What to look up.", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        return "24"


class Scripted(BaseModel):
    """Asks for a call every turn it is allowed to, and records every request.

    *cache* and *forbids* are the two declarations the run reads; *honours*
    decides whether a turn that forbade a call actually gets one anyway, which
    is how a provider that ignores the constraint is reproduced.
    """

    def __init__(
        self,
        *,
        cache: bool = True,
        forbids: bool = True,
        honours: bool = True,
        answer_after: int = 3,
    ) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self._cache = cache
        self._forbids = forbids
        self._honours = honours
        self._answer_after = answer_after
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
        return True

    def supports_conversation(self) -> bool:
        return True

    def supports_suppressed_tool_call(self) -> bool:
        return self._forbids

    def prompt_cache_policy(self) -> PromptCachePolicy | None:
        if not self._cache:
            return None
        return PromptCachePolicy(style="automatic", reports_cached_tokens=True)

    def generate(self, prompt, config=None, **kwargs: Any) -> GenerationResult:
        self.requests.append(prompt)
        self.kwargs.append(dict(kwargs))
        self.index += 1
        forbidden = kwargs.get("tool_choice") == "none"
        may_call = (
            bool(kwargs.get("tools"))
            and self.index <= self._answer_after
            and not (forbidden and self._honours)
        )
        if may_call:
            return GenerationResult(
                text="", tokens_used=6, finish_reason="tool_calls",
                model_name=self.model_name,
                metadata={"tool_calls": [{
                    "id": f"c{self.index}", "type": "function",
                    "function": {"name": "ledger",
                                 "arguments": json.dumps({"query": f"q{self.index}"})},
                }]},
            )
        return GenerationResult(
            text=ANSWER, tokens_used=7, finish_reason="stop",
            model_name=self.model_name,
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _agent(model: BaseModel, **extra: Any) -> Agent:
    settings: dict[str, Any] = {
        "name": "prefix", "model": model, "tools": [Ledger()],
        "max_iterations": 8, "raise_on_error": False, "enable_memory": True,
        "tool_calling_mode": "native",
    }
    settings.update(extra)
    return Agent(AgentConfig(**settings))


def _canon(request: Any, kwargs: dict[str, Any]) -> str:
    """One request as a string, tools first, so a prefix can be measured.

    Only what a provider would hash: the tool definitions and each turn's role
    and text. A message carries a timestamp of its own, which changes on every
    turn and is not part of any request.
    """
    parts = [json.dumps(kwargs.get("tools") or [], sort_keys=True, default=str)]
    if isinstance(request, list):
        for message in request:
            role = getattr(getattr(message, "role", None), "value", "user")
            text = getattr(message, "text", None)
            if text is None:
                text = str(getattr(message, "content", message))
            parts.append(f"{role}:{text}")
    else:
        parts.append(str(request))
    return "\n".join(parts)


def _shared_prefix(a: str, b: str) -> float:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / len(b) if b else 0.0


# --------------------------------------------------------------------------
# R1 — the turn that asks for the answer
# --------------------------------------------------------------------------

def _withdrawal_pair_prefix(model: Scripted) -> float:
    """The shared prefix across the pair where the tools stopped being offered."""
    for i in range(1, len(model.kwargs)):
        was_offered = bool(model.kwargs[i - 1].get("tools"))
        stops_here = (
            model.kwargs[i].get("tool_choice") == "none"
            or (was_offered and not model.kwargs[i].get("tools"))
        )
        if stops_here:
            return _shared_prefix(
                _canon(model.requests[i - 1], model.kwargs[i - 1]),
                _canon(model.requests[i], model.kwargs[i]),
            )
    raise AssertionError("the run never stopped offering tools")


def test_the_withdrawal_turn_keeps_the_tools_and_the_shape():
    """The same definitions, the same conversation, and no call permitted.

    Measured against the same run on a provider that cannot express the
    constraint, which is what effGen sent before: that one rebuilds the request
    from a different frame and shares almost nothing with the turn before it.
    """
    kept = Scripted(answer_after=99)
    _agent(kept, max_iterations=5).run(TASK)
    rebuilt = Scripted(forbids=False, answer_after=99)
    _agent(rebuilt, max_iterations=5).run(TASK)

    forbidden = [i for i, k in enumerate(kept.kwargs) if k.get("tool_choice") == "none"]
    assert forbidden, "no turn ever forbade a call"
    first = forbidden[0]
    assert first > 0
    assert kept.kwargs[first]["tools"] == kept.kwargs[first - 1]["tools"]

    kept_share = _withdrawal_pair_prefix(kept)
    rebuilt_share = _withdrawal_pair_prefix(rebuilt)
    assert kept_share >= 0.70, kept_share
    assert kept_share > rebuilt_share * 5, (kept_share, rebuilt_share)


def test_a_provider_that_cannot_forbid_a_call_is_sent_the_old_shape():
    model = Scripted(forbids=False, answer_after=99)
    _agent(model, max_iterations=5).run(TASK)
    assert all(k.get("tool_choice") != "none" for k in model.kwargs)
    assert any(not k.get("tools") for k in model.kwargs), (
        "the definitions were never withdrawn"
    )


def test_a_provider_with_no_cache_is_sent_the_old_shape():
    model = Scripted(cache=False, answer_after=99)
    _agent(model, max_iterations=5).run(TASK)
    assert all(k.get("tool_choice") != "none" for k in model.kwargs)
    assert all(not isinstance(r, list) for r in model.requests), (
        "a provider that caches nothing was moved onto messages"
    )


def test_a_model_that_ignores_the_constraint_falls_back_and_still_answers():
    """One turn is spent learning it; the turns after it drop the definitions."""
    model = Scripted(honours=False, answer_after=99)
    response = _agent(model, max_iterations=8).run(TASK)

    forbidden = [i for i, k in enumerate(model.kwargs) if k.get("tool_choice") == "none"]
    assert len(forbidden) == 1, f"the constraint was retried: {forbidden}"
    assert any(not k.get("tools") for k in model.kwargs[forbidden[0] + 1:]), (
        "the run never fell back to withdrawing the definitions"
    )
    assert ANSWER in (response.output or "")


# --------------------------------------------------------------------------
# R2 — a session keeps one shape
# --------------------------------------------------------------------------

def _kinds(model: Scripted) -> list[str]:
    return ["messages" if isinstance(r, list) else "flat" for r in model.requests]


def test_a_session_opens_in_the_shape_its_second_run_will_use():
    model = Scripted()
    agent = _agent(model)
    agent.run(TASK)
    first_run = len(model.requests)
    agent.run(FOLLOW_UP)

    kinds = _kinds(model)
    assert set(kinds) == {"messages"}, kinds
    boundary_before = _canon(
        model.requests[first_run - 1], model.kwargs[first_run - 1]
    )
    boundary_after = _canon(model.requests[first_run], model.kwargs[first_run])
    assert _shared_prefix(boundary_before, boundary_after) >= 0.70, (
        _shared_prefix(boundary_before, boundary_after)
    )


def test_an_agent_with_no_memory_keeps_the_shape_it_always_had():
    model = Scripted()
    _agent(model, enable_memory=False).run(TASK)
    assert _kinds(model) == ["flat"] * len(model.requests)


def test_a_caller_who_asked_for_flat_still_opens_flat():
    """``prompt_protocol="flat"`` is an answer, not a preference to be improved.

    A session's later run still puts the persona and the earlier turns on their
    own messages, as it always did — that is the frame, not the protocol — but
    nothing here moves the first run off the string the caller asked for.
    """
    model = Scripted()
    agent = _agent(model, prompt_protocol="flat")
    agent.run(TASK)
    assert _kinds(model) == ["flat"] * len(model.requests)


def test_a_provider_with_no_cache_keeps_the_boundary_it_always_had():
    """The second run still moves onto messages; the first one does not."""
    model = Scripted(cache=False)
    agent = _agent(model)
    agent.run(TASK)
    first_run = len(model.requests)
    agent.run(FOLLOW_UP)
    kinds = _kinds(model)
    assert set(kinds[:first_run]) == {"flat"}
    assert set(kinds[first_run:]) == {"messages"}
