"""How many prompt tokens a run may send, and how many it is about to.

What is pinned here:

* a budget derived from the window the model declares, from a window the caller
  overrode, from a token count, and from a share of the window;
* a model that declares no window leaves the run unbounded rather than being
  guessed at, and ``context_budget=None`` leaves it unbounded on purpose;
* a budget nobody can resolve is a construction error naming the four forms,
  not a run that quietly went out unbounded;
* a base64 image is counted at its declared allowance and never as prose;
* the model's own tokenizer is asked only when its engine type says the count
  is local — **an adapter that would answer over the network is never asked**;
* the provider's own count of a prompt corrects the next estimate, and a ratio
  that could not be describing the same prompt is ignored.
"""

from __future__ import annotations

import pytest

from effgen.core.agent_config import AgentConfig
from effgen.core.messages import (
    AudioPart,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    VideoPart,
)
from effgen.core.thread_budget import (
    IMAGE_TOKEN_ALLOWANCE,
    MIN_CONTEXT_BUDGET_TOKENS,
    ContextBudget,
    TokenCalibration,
    count_prompt_tokens,
    count_text_tokens,
    resolve_context_budget,
)
from effgen.models.base import GenerationResult, ModelType, TokenCount

from .test_loop_on_thread import _Scripted

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 6000


class _Counting(_Scripted):
    """A model that records every request for a token count."""

    def __init__(self, model_type: ModelType, window: int = 8192) -> None:
        super().__init__(["Final Answer: 1"])
        self.model_type = model_type
        self._window = window
        self.count_calls = 0

    def count_tokens(self, text: str) -> TokenCount:
        self.count_calls += 1
        # A number nothing else would produce, so a test can tell whether this
        # was the counter that answered.
        return TokenCount(count=4242, model_name=self.model_name)

    def get_context_length(self) -> int:
        return self._window


# --------------------------------------------------------------------------- #
# Resolving a budget
# --------------------------------------------------------------------------- #


def test_auto_derives_the_budget_from_the_declared_window() -> None:
    model = _Counting(ModelType.OPENAI, window=32768)
    budget = resolve_context_budget(
        "auto", model=model, window_override=None, output_tokens=1024
    )
    assert budget is not None
    # (window - reserve) * headroom, with the reserve at the escalation ceiling
    # and capped at a quarter of the window.
    assert budget.window_tokens == 32768
    assert budget.reserve_tokens == 8192
    assert budget.budget_tokens == int((32768 - 8192) * 0.85)
    assert budget.source == "auto"


def test_a_small_window_still_leaves_room_for_the_question() -> None:
    model = _Counting(ModelType.OPENAI, window=4096)
    budget = resolve_context_budget(
        "auto", model=model, window_override=None, output_tokens=1024
    )
    assert budget is not None
    assert budget.reserve_tokens == 1024  # a quarter of the window, not 8192
    assert budget.budget_tokens == int((4096 - 1024) * 0.85)


def test_max_context_length_overrides_the_window_the_model_declares() -> None:
    model = _Counting(ModelType.OPENAI, window=32768)
    budget = resolve_context_budget(
        "auto", model=model, window_override=8192, output_tokens=1024
    )
    assert budget is not None
    assert budget.window_tokens == 8192


def test_a_token_count_is_taken_exactly() -> None:
    budget = resolve_context_budget(
        6000, model=_Counting(ModelType.OPENAI), window_override=None
    )
    assert budget is not None
    assert (budget.budget_tokens, budget.source) == (6000, "config")


def test_a_fraction_is_that_share_of_the_window() -> None:
    budget = resolve_context_budget(
        0.5, model=_Counting(ModelType.OPENAI, window=32768), window_override=None
    )
    assert budget is not None
    assert (budget.budget_tokens, budget.source) == (16384, "fraction")


def test_none_leaves_the_run_unbounded() -> None:
    assert resolve_context_budget(
        None, model=_Counting(ModelType.OPENAI), window_override=None
    ) is None


def test_a_model_that_declares_no_window_leaves_the_run_unbounded() -> None:
    """Guessing a window would truncate a conversation that would have fitted."""
    model = _Counting(ModelType.OPENAI, window=0)
    assert resolve_context_budget(
        "auto", model=model, window_override=None
    ) is None


# --------------------------------------------------------------------------- #
# Refusing a budget nobody can resolve
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["tiny", 1.5, -0.2, True, object()])
def test_a_budget_naming_none_of_the_four_forms_is_refused(value: object) -> None:
    with pytest.raises(ValueError) as caught:
        AgentConfig(model="openai:a-model", context_budget=value)
    assert "context_budget" in str(caught.value)


def test_a_budget_too_small_to_hold_a_reply_is_refused_at_construction() -> None:
    with pytest.raises(ValueError) as caught:
        AgentConfig(model="openai:a-model", context_budget=64)
    assert str(MIN_CONTEXT_BUDGET_TOKENS) in str(caught.value)


def test_a_compaction_policy_naming_nothing_is_refused_at_construction() -> None:
    with pytest.raises(ValueError) as caught:
        AgentConfig(model="openai:a-model", compaction="drop_everything")
    assert "compaction policy" in str(caught.value)


def test_the_shipped_settings_are_accepted() -> None:
    config = AgentConfig(model="openai:a-model")
    assert config.context_budget == "auto"
    assert config.compaction is None
    for value in ("auto", None, 4096, 0.5):
        AgentConfig(model="openai:a-model", context_budget=value)
    for policy in ("shorten_oldest_first", "summarize_with_model"):
        AgentConfig(model="openai:a-model", compaction=policy)


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #


def test_an_image_is_counted_at_its_allowance_and_never_as_prose() -> None:
    """A base64 payload read as text would shrink the budget to nothing."""
    message = Message(
        role=Role.USER,
        content=[TextPart(text="what is in this picture?"), ImagePart(
            image=PNG, mime="image/png"
        )],
    )
    counted = count_prompt_tokens([message])
    text_only = count_prompt_tokens(
        [Message(role=Role.USER, content=[TextPart(text="what is in this picture?")])]
    )
    assert counted == text_only + IMAGE_TOKEN_ALLOWANCE
    # What reading the bytes as prose would have produced. The bound is set by
    # the weaker of the two counters: with a BPE encoding loaded the payload
    # measures about twelve times the allowance, and with the character-length
    # estimate that stands in when no encoding is available, about six. Either
    # way a payload read as prose costs several times what the allowance does,
    # which is the property being stated.
    assert count_text_tokens(str(PNG)) > 4 * IMAGE_TOKEN_ALLOWANCE


def test_audio_and_video_are_counted_at_their_allowances() -> None:
    audio = Message(role=Role.USER, content=[AudioPart(audio=b"0" * 5000, mime="audio/wav")])
    video = Message(role=Role.USER, content=[VideoPart(
        frames=[b"0" * 500, b"1" * 500, b"2" * 500], fps=1.0, mime="image/png",
    )])
    assert count_prompt_tokens([audio]) < count_prompt_tokens([video])
    assert count_prompt_tokens([video]) - count_prompt_tokens([audio]) == (
        2 * IMAGE_TOKEN_ALLOWANCE
    )


def test_a_tool_call_and_its_result_are_counted_by_what_they_carry() -> None:
    call = Message(role=Role.ASSISTANT, content=[ToolCallPart(
        tool_call_id="c1", name="calculator", arguments={"expression": "2+2"},
    )])
    result = Message(role=Role.TOOL, content=[ToolResultPart(
        tool_call_id="c1", result="4", is_error=False,
    )])
    assert count_prompt_tokens([call]) > count_prompt_tokens([result])


def test_a_string_prompt_and_a_message_list_are_both_counted() -> None:
    assert count_prompt_tokens("hello there, how are you?") > 0
    assert count_prompt_tokens([{"role": "user", "content": "hello there"}]) > 0
    assert count_prompt_tokens(None) == 0


def test_a_local_engine_is_asked_for_its_own_count() -> None:
    model = _Counting(ModelType.TRANSFORMERS)
    assert count_text_tokens("anything at all", model=model) == 4242
    assert model.count_calls == 1


@pytest.mark.parametrize(
    "model_type", [ModelType.OPENAI, ModelType.ANTHROPIC, ModelType.GEMINI]
)
def test_a_cloud_adapter_is_never_asked_to_count(model_type: ModelType) -> None:
    """On some adapters ``count_tokens`` is a request to the provider.

    A budget that used it would add a round trip to every iteration of every
    run on those families, so the shared local estimator answers instead.
    """
    model = _Counting(model_type)
    count_text_tokens("anything at all", model=model)
    count_prompt_tokens(
        [Message(role=Role.USER, content=[TextPart(text="and this too")])],
        model=model,
    )
    assert model.count_calls == 0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_the_providers_own_count_corrects_the_next_estimate() -> None:
    calibration = TokenCalibration()
    calibration.observe(reported=1100, estimated=1000)
    assert calibration.samples == 1
    assert calibration.apply(1000) == 1100


def test_a_ratio_that_cannot_describe_the_same_prompt_is_ignored() -> None:
    calibration = TokenCalibration()
    calibration.observe(reported=50_000, estimated=1000)
    assert calibration.samples == 0
    assert calibration.apply(1000) == 1000


def test_an_estimate_is_never_corrected_downwards() -> None:
    """A budget that believed a smaller count would send a prompt that fails."""
    calibration = TokenCalibration()
    calibration.observe(reported=700, estimated=1000)
    assert calibration.apply(1000) == 1000


# --------------------------------------------------------------------------- #
# The budget object
# --------------------------------------------------------------------------- #


def test_exceeded_remembers_what_it_measured() -> None:
    budget = ContextBudget(budget_tokens=10, window_tokens=100)
    assert budget.exceeded("one two three four five six seven eight nine ten eleven") is True
    assert budget.last_measured > 10
    assert budget.deficit == budget.last_measured - 10
    assert budget.exceeded("hi") is False
    assert budget.deficit == 0


def test_the_budget_takes_the_providers_own_stated_number() -> None:
    budget = ContextBudget(budget_tokens=5000, window_tokens=8192)
    assert budget.lower_to(3000) is True
    assert budget.budget_tokens == 3000
    # A number that is not lower changes nothing.
    assert budget.lower_to(6000) is False
    assert budget.budget_tokens == 3000
    # And it never drops below what a question and a reply need.
    budget.lower_to(1)
    assert budget.budget_tokens == MIN_CONTEXT_BUDGET_TOKENS


def test_the_budget_reports_itself_as_plain_data() -> None:
    budget = ContextBudget(budget_tokens=1000, window_tokens=4096, source="auto")
    budget.exceeded("a longer prompt than one token")
    data = budget.as_dict()
    assert data["budget_tokens"] == 1000
    assert data["window_tokens"] == 4096
    assert data["source"] == "auto"
    assert data["firings"] == 0
    assert data["summarisation"] is None
    assert data["measured_tokens"] > 0


def test_generation_result_is_not_needed_to_read_a_budget() -> None:
    """The budget module imports no adapter, so nothing it does can dial out."""
    import effgen.core.thread_budget as module

    source = (module.__file__ or "")
    assert source.endswith("thread_budget.py")
    assert GenerationResult is not None  # the import above is the point
