"""A run that continues a conversation sends the whole of it in one protocol.

A session's earlier exchanges travel as the ``user`` and ``assistant`` messages
they were. Before this, the run's *own* steps still travelled as text inside the
current user message, so the model was shown its own tool call as something the
user had narrated — and called the tool again. The repeat guard then stopped
that turn, and with ``raise_on_error`` at its default the caller got an
exception instead of an answer. These cover the rule that closes it, and the two
asks it leaves exact: ``"flat"`` is still flat and ``"messages"`` is still
messages.
"""

from __future__ import annotations

import logging
from typing import Any

from effgen.core.agent import Agent, AgentConfig
from effgen.core.messages import ToolCallPart, ToolResultPart
from effgen.core.session import Session
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

CALL_ID = "call_provider_abc123"


class Calculator(BaseTool):
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
        return "21"


class Native(BaseModel):
    """Calls the tool once, then answers, and keeps every request it was sent."""

    def __init__(self, *, carries_messages: bool = True) -> None:
        super().__init__(model_name="native", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self._carries = carries_messages
        self.prompts: list[Any] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return True

    def supports_conversation(self) -> bool:
        return True

    def supports_message_protocol(self) -> bool:
        return self._carries

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt)
        self.index += 1
        if self.index == 1:
            return GenerationResult(
                text="I will multiply them.", tokens_used=8,
                finish_reason="tool_calls", model_name=self.model_name,
                metadata={"tool_calls": [{
                    "id": CALL_ID, "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "7 * 3"}',
                    },
                }]},
            )
        return GenerationResult(
            text="The team works 21 hours.", tokens_used=6,
            finish_reason="stop", model_name=self.model_name,
        )

    def generate_stream(self, prompt: Any, config: Any = None, **kwargs: Any):
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _agent(model: Native, **config: Any) -> Agent:
    return Agent(AgentConfig(
        name="ledger", model=model, tools=[Calculator()],
        max_iterations=4, raise_on_error=False, **config,
    ))


def _continued(model: Native, **config: Any) -> Any:
    """One run that continues a two-message session."""
    session = Session(session_id="prior")
    session.add_message("user", "My team has 7 people. Just acknowledge it.")
    session.add_message("assistant", "Acknowledged. 7 people on the team.")
    agent = _agent(model, **config)
    try:
        return agent.run(
            "Use the calculator to multiply my team size by 3.", session=session,
        )
    finally:
        agent.close()


def _fresh(model: Native, **config: Any) -> Any:
    agent = _agent(model, **config)
    try:
        return agent.run("Use the calculator to multiply 7 by 3.")
    finally:
        agent.close()


def _parts(prompt: Any) -> list[Any]:
    """Every content part of every message in a request."""
    out: list[Any] = []
    for message in prompt:
        content = message.content
        out.extend(content if isinstance(content, list) else [])
    return out


def _user_text(prompt: Any) -> str:
    """Everything the request says in a user turn, as one string."""
    if isinstance(prompt, str):
        return prompt
    said = []
    for message in prompt:
        if str(getattr(message.role, "value", message.role)) != "user":
            continue
        content = message.content
        for part in content if isinstance(content, list) else []:
            said.append(str(getattr(part, "text", "")))
    return "\n".join(said)


class TestWhatTheDefaultDecides:
    def test_a_run_that_continues_nothing_sends_the_flat_string(self):
        response = _fresh(Native())
        assert response.metadata["prompt_protocol"] == "flat"

    def test_a_run_that_continues_a_conversation_sends_turns(self):
        response = _continued(Native())
        assert response.metadata["prompt_protocol"] == "messages"

    def test_it_is_logged_when_it_fires(self, caplog):
        with caplog.at_level(logging.INFO, logger="effgen.core.agent_runtime"):
            _continued(Native())
        assert any(
            "the run continues a conversation" in r.message for r in caplog.records
        )


class TestTheModelIsNotShownItsOwnCallAsNarration:
    def test_no_user_turn_narrates_the_runs_own_call(self):
        model = Native()
        _continued(model)
        later = [p for p in model.prompts if not isinstance(p, str)][1:]
        assert later, "the run took only one turn"
        for prompt in later:
            said = _user_text(prompt)
            assert "Action: calculator" not in said
            assert "Previous steps:" not in said

    def test_the_call_and_its_result_travel_as_the_turns_they_were(self):
        model = Native()
        _continued(model)
        last = model.prompts[-1]
        assert not isinstance(last, str)
        calls = [p for p in _parts(last) if isinstance(p, ToolCallPart)]
        results = [p for p in _parts(last) if isinstance(p, ToolResultPart)]
        assert [c.tool_call_id for c in calls] == [CALL_ID]
        assert [r.tool_call_id for r in results] == [CALL_ID]

    def test_the_turn_answers_rather_than_repeating_the_call(self):
        response = _continued(Native())
        assert response.success
        assert response.tool_calls == 1
        assert response.stop_reason == "final_answer"


class TestAnExplicitAskIsExact:
    def test_flat_stays_flat_on_a_continued_conversation(self):
        response = _continued(Native(), prompt_protocol="flat")
        assert response.metadata["prompt_protocol"] == "flat"

    def test_messages_stays_messages_on_a_run_that_continues_nothing(self):
        response = _fresh(Native(), prompt_protocol="messages")
        assert response.metadata["prompt_protocol"] == "messages"


class TestTheWholeConversationIsOneProtocolEitherWay:
    def test_a_model_that_does_not_carry_messages_keeps_all_of_it_flat(self):
        """Falling back is still consistent: nothing travels as turns."""
        response = _continued(Native(carries_messages=False))
        assert response.metadata["prompt_protocol"] == "flat"

    def test_a_run_with_no_tools_is_unaffected(self):
        """No tool definitions travel as a parameter, so there is no call to express."""
        session = Session(session_id="prior")
        session.add_message("user", "My team has 7 people.")
        session.add_message("assistant", "Noted.")
        agent = Agent(AgentConfig(
            name="plain", model=Native(), tools=[], raise_on_error=False,
        ))
        try:
            response = agent.run("How many people?", session=session)
        finally:
            agent.close()
        assert response.metadata["prompt_protocol"] == "flat"
