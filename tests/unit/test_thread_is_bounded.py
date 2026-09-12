"""A run stays inside the tokens it may send, and says so when it cannot.

The loop used to grow its prompt until the provider refused it, and what the
caller was handed was the provider's sentence about token counts. What is
pinned here:

* the prompt the model is actually sent is under the budget, on every frame the
  loop assembles one in — the ReAct text scaffold, a caller's own template, a
  model that takes tool definitions, and the message protocol;
* a run that used to be refused now answers, and makes **no** refused request;
* a run whose frame alone will not fit raises a typed error naming the budget,
  before any request is sent;
* the error is an ``InvalidRequestError``, so a caller already catching a
  prompt that was too long keeps catching it; with ``raise_on_error=False`` the
  same run reports the failure instead of raising;
* the same over-size prompt is **never sent twice** — on a provider's wording
  and on the local engines' own;
* the lines the loop logs once per run still fire once per run when a turn is
  rebuilt eight times;
* ``metadata["context_budget"]`` is present on every outcome, including a run
  that was never bounded.
"""

from __future__ import annotations

import json
import logging

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.thread_budget import count_prompt_tokens
from effgen.errors import RunStoppedError
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.models.errors import (
    ContextBudgetExceededError,
    InvalidRequestError,
    classify_provider_error,
    context_overflow_hint,
)
from effgen.tools.builtin.calculator import Calculator

FIRING = "[context] compacted the thread"


class _Refusing(BaseModel):
    """Answers with a tool call until *answer_at*, and refuses an over-long prompt.

    Records every prompt it was handed, so a test can ask both what was sent
    and whether the same thing was ever sent twice.
    """

    def __init__(
        self,
        window: int = 2000,
        answer_at: int = 6,
        *,
        native: bool = False,
        wording: str = "provider",
        refuse_above: int | None = None,
    ) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self.window = window
        self.answer_at = answer_at
        self.native = native
        self.wording = wording
        self.refuse_above = refuse_above
        self.prompts: list[str] = []
        self.sizes: list[int] = []
        self.refusals = 0
        self._is_loaded = True

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        rendered = prompt if isinstance(prompt, str) else json.dumps(
            prompt, default=str
        )
        self.prompts.append(rendered)
        size = count_prompt_tokens(prompt)
        self.sizes.append(size)
        ceiling = self.refuse_above if self.refuse_above is not None else self.window
        if size > ceiling:
            self.refusals += 1
            if self.wording == "local":
                raise ValueError(
                    f"Prompt length ({size} tokens) exceeds model context "
                    f"length ({ceiling} tokens)"
                )
            raise RuntimeError(
                f"This model's maximum context length is {ceiling} tokens. "
                f"However, you requested {size} tokens."
            )
        turn = len(self.prompts)
        meta: dict = {"prompt_tokens": size}
        if turn >= self.answer_at:
            text = "Final Answer: 4"
        elif self.native:
            # A model that takes tool definitions writes its call the way its
            # adapter reports one, not as prose in the transcript.
            text = ""
            meta["tool_calls"] = [{
                "id": f"call-{turn}",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": f"{turn}+{turn}"}),
                },
            }]
        else:
            text = (
                f"Thought: step {turn}.\nAction: calculator\n"
                f"Action Input: {json.dumps({'expression': f'{turn}+{turn}'})}"
            )
        return GenerationResult(
            text=text, tokens_used=6, finish_reason="stop",
            model_name=self.model_name, metadata=meta,
        )

    def generate_stream(self, prompt, config=None, **kwargs):
        yield self.generate(prompt).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return self.window

    def supports_tool_calling(self) -> bool:
        return self.native


class _Bulky(Calculator):
    """A calculator whose answer is long enough to fill a small window."""

    async def _execute(self, **kwargs):
        result = await super()._execute(**kwargs)
        return f"{result} " + "padding words for the transcript " * 40


class _Lines(logging.Handler):
    """Collects the loop's own log lines, which is how a firing is counted."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: BLE001 - a log handler never fails a run
            pass

    def count(self, prefix: str) -> int:
        return sum(1 for line in self.lines if line.startswith(prefix))


@pytest.fixture()
def lines():
    handler = _Lines()
    log = logging.getLogger("effgen.core.agent_loop")
    log.addHandler(handler)
    previous = log.level
    log.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)


def _agent(model, **cfg) -> Agent:
    return Agent(config=AgentConfig(
        name="bounded-test",
        model=model,
        tools=cfg.pop("tools", [_Bulky()]),
        tool_calling_mode=cfg.pop("tool_calling_mode", "react"),
        max_iterations=cfg.pop("max_iterations", 12),
        enable_memory=False,
        **cfg,
    ))


# --------------------------------------------------------------------------- #
# A run that used to be refused
# --------------------------------------------------------------------------- #


def test_a_run_that_would_have_been_refused_now_answers(lines) -> None:
    model = _Refusing(window=2400, answer_at=7)
    response = _agent(model).run("what is two plus two?")
    assert response.success is True
    assert model.refusals == 0, "the provider was never sent a prompt it refuses"
    budget = response.metadata["context_budget"]
    assert budget["source"] == "auto"
    assert max(model.sizes) <= budget["budget_tokens"]
    assert lines.count(FIRING) == budget["firings"] > 0


def test_work_that_fits_never_reaches_the_budget(lines) -> None:
    model = _Refusing(window=32768, answer_at=3)
    response = _agent(model, tools=[Calculator()]).run("what is two plus two?")
    assert response.success is True
    assert lines.count(FIRING) == 0
    assert response.metadata["context_budget"]["firings"] == 0
    assert response.metadata["context_budget"]["steps_dropped"] == 0


def test_an_unbounded_run_reports_that_it_was_unbounded() -> None:
    model = _Refusing(window=32768, answer_at=3)
    response = _agent(model, context_budget=None).run("what is two plus two?")
    assert response.metadata["context_budget"]["source"] == "unbounded"
    assert response.metadata["context_budget"]["budget_tokens"] is None


# --------------------------------------------------------------------------- #
# Every frame the loop assembles a prompt in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "config"),
    [
        ("react text", {"tool_calling_mode": "react"}),
        ("native", {"tool_calling_mode": "native"}),
        ("custom template", {
            "tool_calling_mode": "react",
            "system_prompt_template": (
                "{tools_description}\n{conversation_history}\n{task}\n{scratchpad}"
            ),
        }),
    ],
)
def test_the_prompt_actually_sent_is_under_the_budget(label, config) -> None:
    # A budget named outright rather than derived, so the same number binds
    # every frame and the comparison between them is the frame, not the window.
    model = _Refusing(window=32768, answer_at=7, native=config.get(
        "tool_calling_mode"
    ) == "native")
    response = _agent(model, context_budget=1200, **config).run(
        "what is two plus two?"
    )
    budget = response.metadata["context_budget"]
    assert budget["budget_tokens"] == 1200
    assert budget["firings"] > 0, f"{label}: the budget never bound this frame"
    assert max(model.sizes) <= 1200, label
    assert model.refusals == 0, label


def test_the_message_protocol_is_measured_as_the_message_list_it_is() -> None:
    model = _Refusing(window=32768, answer_at=7, native=True)
    response = _agent(
        model, tool_calling_mode="native", prompt_protocol="messages",
        context_budget=1200,
    ).run("what is two plus two?")
    budget = response.metadata["context_budget"]
    assert budget["firings"] > 0
    assert max(model.sizes) <= 1200
    assert model.refusals == 0


# --------------------------------------------------------------------------- #
# The floor
# --------------------------------------------------------------------------- #


def _unfittable_task() -> str:
    return "please answer this: " + "material the run must keep in mind " * 200


def test_a_frame_that_will_not_fit_raises_before_anything_is_sent() -> None:
    model = _Refusing(window=32768, answer_at=2)
    with pytest.raises(ContextBudgetExceededError) as caught:
        _agent(model, context_budget=512).run(_unfittable_task())
    error = caught.value
    assert error.budget_tokens == 512
    assert error.measured_tokens > 512
    assert error.budget_source == "config"
    assert "512" in str(error)
    assert model.prompts == [], "no request was sent"


def test_the_typed_error_is_still_an_invalid_request() -> None:
    """A caller already catching a prompt that was too long keeps catching it."""
    model = _Refusing(window=32768, answer_at=2)
    with pytest.raises(InvalidRequestError):
        _agent(model, context_budget=512).run(_unfittable_task())


def test_without_raise_on_error_the_same_run_reports_the_failure() -> None:
    model = _Refusing(window=32768, answer_at=2)
    response = _agent(
        model, context_budget=512, raise_on_error=False
    ).run(_unfittable_task())
    assert response.success is False
    assert "does not fit" in response.output
    assert response.metadata["context_budget"]["budget_tokens"] == 512
    assert response.metadata["thread"] is not None
    assert model.prompts == []


# --------------------------------------------------------------------------- #
# The same over-size prompt is never sent twice
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("wording", ["provider", "local"])
def test_an_over_size_prompt_is_never_sent_twice(wording) -> None:
    """Our count and the provider's can disagree; paying twice for that is not on."""
    # The model declares a window the budget trusts, and refuses well below it.
    model = _Refusing(
        window=32768, answer_at=99, wording=wording, refuse_above=400,
    )
    # Either wording ends the run; which typed error it ends with is settled
    # by the tests above. What is pinned here is that nothing was paid twice.
    with pytest.raises((InvalidRequestError, RunStoppedError, RuntimeError)):
        _agent(model, max_iterations=6).run("what is two plus two?")
    assert len(model.prompts) == len(set(model.prompts)), (
        "the identical prompt was sent more than once"
    )


def test_the_local_window_check_is_not_retried_and_is_recognised() -> None:
    """effGen's own window check raises a bare ValueError carrying no status."""
    error = ValueError(
        "Prompt length (4212 tokens) exceeds model context length (4096 tokens)"
    )
    assert classify_provider_error(error).should_retry is False
    assert classify_provider_error(error).category == "invalid_request"
    assert context_overflow_hint(str(error)) is not None


def test_a_rate_limit_is_still_retryable() -> None:
    """The phrase list also holds rate limits, which a later moment can fix."""
    error = RuntimeError("Rate limit reached: 30000 tokens per minute")
    assert classify_provider_error(error).should_retry is True


# --------------------------------------------------------------------------- #
# A turn rebuilt eight times still reports itself once
# --------------------------------------------------------------------------- #


def test_the_lines_logged_once_per_run_still_fire_once(lines) -> None:
    model = _Refusing(window=32768, answer_at=7, native=True)
    agent = _agent(
        model, tool_calling_mode="native", prompt_protocol="messages",
        system_prompt="you are a careful assistant", context_budget=1200,
    )
    response = agent.run("what is two plus two?")
    assert response.metadata["context_budget"]["firings"] > 0, (
        "the turn was rebuilt, which is what this test is about"
    )
    assert lines.count("[protocol] the run sends the conversation as messages") <= 1
    assert lines.count("[frame] the run's frame travels as messages") <= 1
    assert lines.count("[context] the run is bounded at") == 1


# --------------------------------------------------------------------------- #
# The key is on every outcome
# --------------------------------------------------------------------------- #


def test_the_budget_is_reported_on_an_answer_a_stop_and_a_cap() -> None:
    answered = _agent(_Refusing(window=32768, answer_at=2), tools=[Calculator()]).run(
        "what is two plus two?"
    )
    assert answered.metadata["context_budget"]["source"] == "auto"

    capped = _agent(
        _Refusing(window=32768, answer_at=99), tools=[Calculator()],
        max_iterations=2, raise_on_error=False,
    ).run("what is two plus two?")
    assert capped.metadata["context_budget"]["source"] == "auto"

    # And the same mapping travels on the thread, so it survives a checkpoint
    # and a stored session turn.
    assert answered.metadata["thread"].metadata["context_budget"] == (
        answered.metadata["context_budget"]
    )


def test_the_run_store_records_how_many_times_a_run_compacted() -> None:
    from effgen.observability.run_log import _thread_shape

    shape = _thread_shape({
        "version": 1,
        "steps": [{"kind": "task"}, {"kind": "thought"}],
        "metadata": {"context_budget": {"firings": 3}},
    })
    assert shape["thread_steps"] == 2
    assert shape["compactions"] == 3
    # A thread with no budget on it records the shape it always did.
    assert "compactions" not in _thread_shape({"version": 1, "steps": []})


def test_the_budget_is_reported_on_a_generation_failure() -> None:
    """A run that failed at the provider still says what it was allowed to send."""
    class _Failing(_Refusing):
        def generate(self, prompt, config=None, **kwargs):
            raise RuntimeError("the provider is having a bad day")

    response = _agent(
        _Failing(window=32768), tools=[Calculator()], raise_on_error=False,
    ).run("what is two plus two?")
    assert response.success is False
    assert response.metadata["context_budget"]["source"] == "auto"
    assert response.metadata["context_budget"]["budget_tokens"] is not None


def test_the_budget_is_reported_when_a_run_writes_its_call_out() -> None:
    """The written-call outcome builds its own metadata, and carries the key."""
    class _Writing(_Refusing):
        def generate(self, prompt, config=None, **kwargs):
            self.prompts.append("x")
            return GenerationResult(
                text="Final Answer: I would call calculator with 2+2 to get 4.",
                tokens_used=6, finish_reason="stop", model_name=self.model_name,
                metadata={},
            )

    response = _agent(
        _Writing(window=32768), tools=[Calculator()], raise_on_error=False,
    ).run("what is two plus two?")
    assert "context_budget" in response.metadata
    assert response.metadata["context_budget"]["source"] in {"auto", "unbounded"}


# --------------------------------------------------------------------------- #
# A run with no tools takes one turn, and is bounded too
# --------------------------------------------------------------------------- #


def test_a_run_with_no_tools_reports_its_budget() -> None:
    """It never enters the loop, so it had carried no budget at all."""
    model = _Refusing(window=32768, answer_at=1)
    response = _agent(model, tools=[]).run("what is two plus two?")
    assert response.success is True
    assert response.metadata["context_budget"]["source"] == "auto"
    assert response.metadata["context_budget"]["firings"] == 0


def test_a_no_tool_question_that_will_not_fit_raises_before_sending() -> None:
    """There is no transcript to give up: the whole prompt is the question."""
    model = _Refusing(window=32768, answer_at=1)
    with pytest.raises(ContextBudgetExceededError) as caught:
        _agent(model, tools=[], context_budget=512).run(_unfittable_task())
    assert caught.value.budget_tokens == 512
    assert "the question and the conversation it is asked in" in str(caught.value)
    assert model.prompts == []


def test_a_no_tool_run_reports_the_failure_when_it_does_not_raise() -> None:
    model = _Refusing(window=32768, answer_at=1)
    response = _agent(
        model, tools=[], context_budget=512, raise_on_error=False,
    ).run(_unfittable_task())
    assert response.success is False
    assert response.metadata["context_budget"]["budget_tokens"] == 512
    assert model.prompts == []


def test_a_no_tool_run_left_unbounded_says_so() -> None:
    model = _Refusing(window=32768, answer_at=1)
    response = _agent(model, tools=[], context_budget=None).run("two plus two?")
    assert response.metadata["context_budget"]["source"] == "unbounded"
