"""What a request carries, and what it stops carrying twice.

Four things a prompt used to say more than once, or say on a guess:

* how fully a tool is described in prose was chosen from substrings of the
  model's name; it now comes from what the model's adapter declares;
* the framework's own generated system prompt and the tool list under it
  stated the same five rules, and said twice that the model may reason step by
  step and use tools;
* the line pointing at earlier conversation was stated on runs that carry none;
* a tool result the request already carried, byte for byte, was sent again in
  full every turn after the call that produced it.

Each test states the request a caller would actually send, so a change that
puts any of it back fails here.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.thread import (
    ActionStep,
    AgentThread,
    ObservationStep,
    TaskStep,
    ThoughtStep,
)
from effgen.models.base import BaseModel, GenerationResult, ModelType
from effgen.prompts.tool_prompt_generator import ToolPromptGenerator
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)


class _Ledger(BaseTool):
    """A tool with a parameter and no examples of its own."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="ledger_lookup",
            description="Look up a figure in the shop ledger.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="What to look up.", required=True,
            )],
        ))

    async def _execute(self, **kwargs):  # pragma: no cover - never called here
        return "ledger total: 36"


class _Silent(BaseModel):
    """An adapter that declares nothing about how a tool should be stated."""

    def __init__(self, name: str = "some-model") -> None:
        super().__init__(model_name=name, model_type=ModelType.OPENAI)

    def load(self) -> None:
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    def generate(self, prompt, config=None, **kwargs):  # pragma: no cover
        return GenerationResult(text="Final Answer: 36", model=self.model_name)

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield "Final Answer: 36"

    def count_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def get_context_length(self) -> int:
        return 8192

    def get_metadata(self):
        return {"model_name": self.model_name}


class _Compact(_Silent):
    """An adapter that declares its models read a tool schema natively."""

    def prompt_detail(self) -> str | None:
        return "compact"


@pytest.fixture
def tools():
    return [_Ledger()]


def _agent(model, tools, **config):
    return Agent(AgentConfig(name="a", model=model, tools=tools, **config))


# ---------------------------------------------------------------------------
# how fully a tool is stated
# ---------------------------------------------------------------------------


def test_the_model_declares_how_fully_its_tools_are_stated(tools):
    """A declared "compact" drops the parameter block; silence keeps it."""
    assert _agent(_Compact(), tools)._verbose_tools is False
    assert _agent(_Silent(), tools)._verbose_tools is True


def test_a_model_name_no_longer_decides_it(tools):
    """Two adapters that declare the same thing answer the same, whatever the
    model is called. The names here are the ones the old substring table
    matched — and the one it matched by accident."""
    for name in ("gpt-4o", "claude-3-5-sonnet", "gemini-2.5-flash",
                 "Qwen/Qwen2.5-1.5B-Instruct", "openai/gpt-oss-20b"):
        assert _agent(_Silent(name), tools)._verbose_tools is True
        assert _agent(_Compact(name), tools)._verbose_tools is False


def test_the_caller_still_wins(tools):
    assert _agent(_Compact(), tools, verbose_tools=True)._verbose_tools is True
    assert _agent(_Silent(), tools, verbose_tools=False)._verbose_tools is False


def test_an_adapter_that_raises_is_read_as_silence(tools):
    class _Broken(_Silent):
        def prompt_detail(self):
            raise RuntimeError("no")

    assert _agent(_Broken(), tools)._verbose_tools is True


# ---------------------------------------------------------------------------
# the rules are stated once
# ---------------------------------------------------------------------------


def test_the_generated_system_prompt_states_the_rules_and_the_list_does_not(tools):
    """An agent handed tools and no persona sends one copy of the rules."""
    agent = _agent(_Silent(), tools)
    assert agent._framework_system_prompt is True
    prompt = agent._tool_prompt_generator.generate_react_prompt(
        task="t", system_prompt=agent.config.system_prompt,
        rules_already_stated=True,
    )
    assert prompt.count("Common Mistakes to Avoid:") == 1
    assert "IMPORTANT RULES:" not in prompt
    assert prompt.count("You can reason step-by-step and use tools") == 1
    # the tools themselves are still there
    assert "ledger_lookup" in prompt


def test_a_callers_own_system_prompt_gets_every_rule(tools):
    """Nothing is left out of a prompt the framework did not write."""
    agent = _agent(_Silent(), tools, system_prompt="You are Ledger, a clerk.")
    assert agent._framework_system_prompt is False
    prompt = agent._tool_prompt_generator.generate_react_prompt(
        task="t", system_prompt=agent.config.system_prompt,
        rules_already_stated=False,
    )
    assert "IMPORTANT RULES:" in prompt
    assert prompt.count("You can reason step-by-step and use tools") == 1


def test_the_tools_section_on_its_own_still_carries_the_rules(tools):
    """The default is unchanged, so a caller's own template is unchanged."""
    generator = ToolPromptGenerator(tools=tools, model_name="m")
    assert "IMPORTANT RULES:" in generator.generate_tools_section()
    assert "IMPORTANT RULES:" not in generator.generate_tools_section(rules=False)


# ---------------------------------------------------------------------------
# the line about earlier conversation
# ---------------------------------------------------------------------------


def test_the_conversation_line_is_stated_only_with_a_conversation(tools):
    generator = ToolPromptGenerator(tools=tools, model_name="m")
    note = "IMPORTANT: If there is previous conversation context above"
    assert note not in generator.generate_react_prompt(task="t")
    with_history = generator.generate_react_prompt(
        task="t", conversation_history="Earlier in this conversation:\nUser: hi",
    )
    assert note in with_history


@pytest.mark.parametrize("model_name", ["qwen2.5", "meta-llama-3", "phi-3", "zzz"])
def test_every_prompt_shape_leaves_it_out_together(tools, model_name):
    generator = ToolPromptGenerator(tools=tools, model_name=model_name)
    note = "IMPORTANT: If there is previous conversation context above"
    assert note not in generator.generate_react_prompt(task="t")


def test_a_callers_template_still_receives_the_scratchpad(tools):
    """A caller's own template is filled exactly as it was."""
    template = (
        "{tools_description}\n--\n{conversation_history}\n--\n{task}\n--\n{scratchpad}"
    )
    agent = _agent(_Silent(), tools, system_prompt_template=template)
    filled = agent.config.system_prompt_template.format(
        tools_description=agent._get_tools_description(),
        conversation_history="",
        task="q",
        scratchpad="\nThought: t",
    )
    assert filled.endswith("--\n\nThought: t")
    assert "IMPORTANT RULES:" in filled


# ---------------------------------------------------------------------------
# a repeated result is sent once
# ---------------------------------------------------------------------------


_TRACEBACK = (
    "Error: Traceback (most recent call last):\n"
    '  File "<string>", line 3, in <module>\n'
    "SyntaxError: invalid syntax, and this line makes it long enough to matter"
)


def _looping_thread(result: str) -> AgentThread:
    steps = [TaskStep(text="q")]
    for word in ("a", "b", "c"):
        steps += [
            ThoughtStep(text=word),
            ActionStep(tool="python_exec", raw='{"code": "x"}'),
            ObservationStep(text=result),
        ]
    return AgentThread(steps=steps)


def test_an_identical_result_is_written_once_in_the_transcript():
    text = _looping_thread(_TRACEBACK).to_text()
    assert text.count("SyntaxError: invalid syntax") == 1
    assert text.count("(the same result as the identical call above)") == 2


def test_an_identical_result_is_sent_once_as_messages():
    messages = _looping_thread(_TRACEBACK).to_messages()
    results = [
        part.result
        for message in messages
        for part in message.content
        if getattr(part, "result", None) is not None
    ]
    assert len(results) == 3
    assert results[0] == _TRACEBACK
    assert results[1] == results[2] == "(the same result as the identical call above)"


def test_the_step_still_holds_the_whole_reply():
    """Only the rendering is shortened; what is stored and checkpointed is not."""
    thread = _looping_thread(_TRACEBACK)
    thread.to_text()
    assert [
        step.text for step in thread.steps if isinstance(step, ObservationStep)
    ] == [_TRACEBACK] * 3
    rebuilt = AgentThread.from_dict(thread.to_dict())
    assert [
        step.text for step in rebuilt.steps if isinstance(step, ObservationStep)
    ] == [_TRACEBACK] * 3


def test_a_short_repeat_is_left_alone():
    """Replacing it would make the request longer, not shorter."""
    text = _looping_thread("36").to_text()
    assert text.count("Observation: 36") == 3
    assert "the same result as" not in text


def test_two_different_results_are_both_sent():
    thread = AgentThread(steps=[
        TaskStep(text="q"),
        ActionStep(tool="t", raw="{}"),
        ObservationStep(text=_TRACEBACK),
        ActionStep(tool="t", raw="{}"),
        ObservationStep(text=_TRACEBACK.replace("syntax", "name")),
    ])
    text = thread.to_text()
    assert "invalid syntax" in text
    assert "invalid name" in text
    assert "the same result as" not in text


def test_two_different_calls_with_the_same_answer_are_both_sent():
    """The note says the call above was identical, so it has to have been.

    Two different questions can come back with the same sentence -- a tool that
    is misconfigured says so whatever it was asked, and a search that finds
    nothing returns the same empty answer. Writing the second one as "the same
    result as the identical call above" tells the model the two questions were
    the same question, which they were not.
    """
    thread = AgentThread(steps=[
        TaskStep(text="q"),
        ActionStep(tool="search", arguments={"query": "alpha"}),
        ObservationStep(text=_TRACEBACK),
        ActionStep(tool="search", arguments={"query": "beta"}),
        ObservationStep(text=_TRACEBACK),
    ])
    text = thread.to_text()
    assert "the same result as" not in text
    assert text.count("invalid syntax") == 2
    results = [
        part.result
        for message in thread.to_messages()
        for part in (message.content if isinstance(message.content, list) else [])
        if getattr(part, "result", None) is not None
    ]
    assert results == [_TRACEBACK, _TRACEBACK]


def test_the_same_call_twice_is_still_sent_once():
    """The saving itself: same tool, same arguments, same answer."""
    thread = AgentThread(steps=[
        TaskStep(text="q"),
        ActionStep(tool="search", arguments={"query": "alpha"}),
        ObservationStep(text=_TRACEBACK),
        ActionStep(tool="search", arguments={"query": "alpha"}),
        ObservationStep(text=_TRACEBACK),
    ])
    text = thread.to_text()
    assert text.count("invalid syntax") == 1
    assert "the same result as the identical call above" in text


def test_it_is_reported_once_per_thread(caplog):
    thread = _looping_thread(_TRACEBACK)
    with caplog.at_level("INFO", logger="effgen.core.thread"):
        thread.to_text()
        thread.to_messages()
        thread.to_text()
    fired = [
        record for record in caplog.records
        if "a repeated tool result is sent once" in record.getMessage()
    ]
    assert len(fired) == 1
    assert "2 repeat(s)" in fired[0].getMessage()


def test_a_growing_thread_still_extends_its_own_rendering():
    """Turn k's transcript stays a prefix of turn k+1's.

    Collapsing a repeat rewrites bytes a provider's cache has already matched
    if it ever changes something the earlier request carried. It cannot: a
    result is collapsed only from its second appearance, and the walk is in
    order, so what turn k sent turn k+1 sends again unchanged. That is what
    keeps the cacheable prefix this rendering sits inside intact.
    """
    thread = AgentThread(steps=[TaskStep(text="q")])
    renderings = [thread.to_text()]
    for word in ("a", "b", "c", "d"):
        thread.append(ThoughtStep(text=word))
        thread.append(ActionStep(tool="python_exec", raw='{"code": "x"}'))
        thread.append(ObservationStep(text=_TRACEBACK))
        renderings.append(thread.to_text())
    for earlier, later in zip(renderings, renderings[1:]):
        assert later.startswith(earlier)
    assert renderings[-1].count("SyntaxError: invalid syntax") == 1
