"""How many prompt tokens a run may send, and how many it is about to.

A run's conversation grows with every turn. Nothing used to bound it, so a
long enough run assembled a prompt its model could not read and the provider
refused it — and what the caller was handed was the provider's sentence about
token counts, not a statement about the run.

This module is the measuring side of the bound. :class:`ContextBudget` says how
many prompt tokens one run may send, derived from the window the model declares
and the output budget the run reserves; :func:`count_prompt_tokens` says how
many a prompt actually carries, on every shape a prompt can take.

Three rules decide how the counting is done, and each of them is here because
the obvious alternative costs the caller something:

* **No network call, ever.** ``BaseModel.count_tokens`` is a request to the
  provider on some adapters, so a budget that used it would add a round trip to
  every iteration of every run on those families. The model's own tokenizer is
  asked only when its engine type declares the count is local.
* **The provider's own number is better than ours, one turn late.** After a
  turn the loop knows what the provider counted and what this module estimated
  for the same prompt, so the ratio between them corrects the next turn's
  estimate. See :class:`TokenCalibration`.
* **A picture is not prose.** Running a text estimator over a serialised
  message list counts a base64 payload as words and over-counts by orders of
  magnitude, which would shrink the budget to nothing on a run carrying an
  image. Non-text parts are counted at a declared flat allowance instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..models._adapter_utils import estimate_tokens

logger = logging.getLogger(__name__)

__all__ = [
    "AUDIO_TOKEN_ALLOWANCE",
    "DEFAULT_HEADROOM",
    "IMAGE_TOKEN_ALLOWANCE",
    "MIN_CONTEXT_BUDGET_TOKENS",
    "CompactionStats",
    "ContextBudget",
    "TokenCalibration",
    "count_prompt_tokens",
    "count_text_tokens",
    "resolve_context_budget",
]

#: What one image is counted as. An approximation: providers price an image by
#: its dimensions and their own tiling, which is not knowable from the bytes
#: without decoding them. Counting it as a flat allowance is wrong by a
#: bounded amount; counting its base64 payload as prose is wrong by a factor of
#: a thousand.
IMAGE_TOKEN_ALLOWANCE = 1024

#: What one audio clip is counted as, for the same reason.
AUDIO_TOKEN_ALLOWANCE = 1024

#: What one sampled video frame is counted as; a clip costs this per frame.
VIDEO_FRAME_TOKEN_ALLOWANCE = IMAGE_TOKEN_ALLOWANCE

#: Fraction of the usable window a run is allowed to fill. The gap absorbs the
#: difference between a local estimate and the provider's own count, which is
#: measured in single-figure percentages on real prompts.
DEFAULT_HEADROOM = 0.85

#: A budget smaller than this cannot hold a question and a reply, so a caller
#: who names one is told at construction rather than at the first turn.
MIN_CONTEXT_BUDGET_TOKENS = 512

#: The output budget a run reserves is raised to at least this, because a turn
#: that is cut off mid-answer is retried once at a larger budget.
_OUTPUT_RESERVE_CEILING = 8192

#: The reserve never takes more than this share of the window, so a small
#: window still leaves room for the question.
_RESERVE_SHARE_OF_WINDOW = 4

#: How far the provider's count may differ from ours before the ratio is
#: treated as measuring something other than the same prompt and ignored.
_CALIBRATION_FLOOR = 0.5
_CALIBRATION_CEILING = 2.0


def count_text_tokens(text: str, *, model: Any = None) -> int:
    """Count the tokens in one piece of text, without a network call.

    Args:
        text: The text to count.
        model: The model the prompt is going to, asked for its own count only
            when its engine type declares that count is local.

    Returns:
        The token count.
    """
    if not text:
        return 0
    if model is not None and _counts_locally(model):
        try:
            count = model.count_tokens(text)
            return int(getattr(count, "count", count))
        except Exception:  # noqa: BLE001 - a tokenizer that will not load is not a run failure
            logger.debug("the model's own tokenizer declined to count", exc_info=True)
    return estimate_tokens(text)


def _counts_locally(model: Any) -> bool:
    """Whether *model* counts tokens against a tokenizer it already holds.

    Read from the engine type the adapter declares, never from its name: the
    same declaration the streamed-usage estimate already reads, and the reason
    a cloud adapter is never asked.
    """
    try:
        from ..models.base import _LOCAL_ENGINE_TYPES

        return getattr(model, "model_type", None) in _LOCAL_ENGINE_TYPES
    except Exception:  # noqa: BLE001 - an adapter that declares nothing is not local
        return False


def _part_tokens(part: Any, *, model: Any = None) -> int:
    """Count one content part, at its declared allowance when it is not text."""
    kind = str(getattr(part, "type", "") or "")
    if kind == "text":
        return count_text_tokens(str(getattr(part, "text", "")), model=model)
    if kind == "image":
        return IMAGE_TOKEN_ALLOWANCE
    if kind == "audio":
        return AUDIO_TOKEN_ALLOWANCE
    if kind == "video_frames":
        frames = getattr(part, "frames", None) or []
        return VIDEO_FRAME_TOKEN_ALLOWANCE * max(1, len(frames))
    if kind == "tool_call":
        name = str(getattr(part, "name", ""))
        try:
            arguments = json.dumps(getattr(part, "arguments", {}) or {}, default=str)
        except (TypeError, ValueError):
            arguments = str(getattr(part, "arguments", ""))
        return count_text_tokens(f"{name} {arguments}", model=model)
    if kind == "tool_result":
        return count_text_tokens(str(getattr(part, "result", "")), model=model)
    # A part this release does not know: count what text it can show, so an
    # unknown shape is never counted as zero.
    return count_text_tokens(str(getattr(part, "text", "") or ""), model=model)


#: Tokens the provider's own message envelope costs, per message. Every chat
#: protocol wraps a message in role markers; four is the figure the common
#: tokenizers land on and it keeps a many-message prompt from reading smaller
#: than it is.
_MESSAGE_OVERHEAD_TOKENS = 4


def count_prompt_tokens(prompt: Any, *, model: Any = None) -> int:
    """Count the tokens in a prompt, whatever shape it arrived in.

    A prompt reaches a model as one string, as a list of
    :class:`~effgen.core.messages.Message`, or as a list of plain mappings in
    a provider's own wire shape. All three are counted here, so the number
    checked against a budget is the number of the prompt about to be sent
    rather than an estimate of something near it.

    Args:
        prompt: The prompt, as the loop assembled it.
        model: The model it is going to, for a local tokenizer where there is
            one.

    Returns:
        The token count.
    """
    if prompt is None:
        return 0
    if isinstance(prompt, str):
        return count_text_tokens(prompt, model=model)
    if isinstance(prompt, list):
        total = 0
        for message in prompt:
            total += _MESSAGE_OVERHEAD_TOKENS
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str):
                total += count_text_tokens(content, model=model)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += _part_tokens(_PartView(part), model=model)
                    else:
                        total += _part_tokens(part, model=model)
            elif content is not None:
                total += count_text_tokens(str(content), model=model)
        return total
    return count_text_tokens(str(prompt), model=model)


class _PartView:
    """Attribute access over a content part that arrived as a mapping."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)


@dataclass
class TokenCalibration:
    """The ratio between the provider's count of a prompt and ours.

    The loop reads the provider's own ``prompt_tokens`` off every turn. Paired
    with this module's estimate of the same prompt it gives a correction that
    is exact one turn later and costs nothing — which is what makes the budget
    right on a served model whose tokenizer is not the estimator's.

    A ratio outside :data:`_CALIBRATION_FLOOR`–:data:`_CALIBRATION_CEILING` is
    not a tokenizer difference; it is the two numbers describing different
    prompts, so it is logged and ignored.
    """

    ratio: float = 1.0
    samples: int = 0

    def observe(self, *, reported: int, estimated: int) -> None:
        """Take the provider's count of a prompt this module estimated."""
        if reported <= 0 or estimated <= 0:
            return
        ratio = reported / estimated
        if ratio < _CALIBRATION_FLOOR or ratio > _CALIBRATION_CEILING:
            logger.info(
                "[context] the provider counted %d tokens where we estimated %d",
                reported, estimated,
            )
            return
        self.ratio = ratio
        self.samples += 1

    def apply(self, estimated: int) -> int:
        """Correct an estimate by the ratio measured so far."""
        if self.samples == 0 or self.ratio <= 1.0:
            return estimated
        return int(estimated * self.ratio)


@dataclass
class CompactionStats:
    """What compaction did over one run, for the caller and the run document."""

    firings: int = 0
    observations_shortened: int = 0
    steps_dropped: int = 0
    tokens_dropped: int = 0
    summarisation: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The counters as plain data."""
        return {
            "firings": self.firings,
            "observations_shortened": self.observations_shortened,
            "steps_dropped": self.steps_dropped,
            "tokens_dropped": self.tokens_dropped,
            "summarisation": self.summarisation,
        }


@dataclass
class ContextBudget:
    """How many prompt tokens one run may send, and what it last measured.

    Attributes:
        budget_tokens: The ceiling, in prompt tokens.
        window_tokens: The window the ceiling was derived from, or ``None``
            when the caller named the ceiling outright.
        reserve_tokens: What was held back for the run's own output.
        source: ``"auto"`` when derived from the model's declared window,
            ``"config"`` when the caller named a token count, ``"fraction"``
            when they named a share of the window.
        model: The model the prompts go to, for a local tokenizer.
        calibration: The correction measured from the provider's own counts.
        stats: What compaction has done so far this run.
        last_measured: The size of the last prompt :meth:`exceeded` was asked
            about, so a policy can see how much it has to give up.
    """

    budget_tokens: int
    window_tokens: int | None = None
    reserve_tokens: int = 0
    source: str = "auto"
    model: Any = None
    calibration: TokenCalibration = field(default_factory=TokenCalibration)
    stats: CompactionStats = field(default_factory=CompactionStats)
    last_measured: int = 0

    def measure(self, prompt: Any) -> int:
        """The corrected token count of *prompt*."""
        return self.calibration.apply(count_prompt_tokens(prompt, model=self.model))

    def exceeded(self, prompt: Any) -> bool:
        """Whether *prompt* is over the budget, remembering what it measured."""
        self.last_measured = self.measure(prompt)
        return self.last_measured > self.budget_tokens

    @property
    def deficit(self) -> int:
        """How far the last measured prompt was over the budget."""
        return max(0, self.last_measured - self.budget_tokens)

    def observe_reported(self, *, reported: int, estimated: int) -> None:
        """Take the provider's own count of the prompt just sent."""
        self.calibration.observe(reported=reported, estimated=estimated)

    def lower_to(self, tokens: int) -> bool:
        """Lower the ceiling to the provider's own stated number.

        A provider that refuses a prompt usually says what it would have
        accepted. Believing it is better than believing our estimate, which
        has just been shown to be wrong in the direction that matters.

        Args:
            tokens: The ceiling to take, in prompt tokens.

        Returns:
            Whether the ceiling actually moved down.
        """
        lowered = max(MIN_CONTEXT_BUDGET_TOKENS, int(tokens))
        if lowered >= self.budget_tokens:
            return False
        self.budget_tokens = lowered
        return True

    def as_dict(self) -> dict[str, Any]:
        """What the run document and the caller are told about the budget."""
        data: dict[str, Any] = {
            "budget_tokens": self.budget_tokens,
            "window_tokens": self.window_tokens,
            "source": self.source,
            "measured_tokens": self.last_measured,
            "calibration": round(self.calibration.ratio, 4),
        }
        data.update(self.stats.as_dict())
        return data


#: What a run with no budget in force reports, so the key is always present.
UNBOUNDED_CONTEXT_BUDGET: dict[str, Any] = {
    "budget_tokens": None,
    "window_tokens": None,
    "source": "unbounded",
    "measured_tokens": 0,
    "calibration": 1.0,
    "firings": 0,
    "observations_shortened": 0,
    "steps_dropped": 0,
    "tokens_dropped": 0,
    "summarisation": None,
}


def validate_context_budget(value: Any) -> Any:
    """Check a ``context_budget`` setting and hand it back unchanged.

    Args:
        value: What the caller configured.

    Returns:
        *value*, when it names one of the four forms.

    Raises:
        ValueError: When it names none of them, or is a token count too small
            to hold a question and a reply.
    """
    if value is None or value == "auto":
        return value
    # A bool is an int to Python and a mistake to a reader, so it falls through
    # to the same sentence every other shape that is not a budget gets.
    if isinstance(value, int) and not isinstance(value, bool):
        if value < MIN_CONTEXT_BUDGET_TOKENS:
            raise ValueError(
                f"context_budget={value} is too small to hold a question and a "
                f"reply. Pass at least {MIN_CONTEXT_BUDGET_TOKENS} tokens, a "
                f"fraction of the model's window, or None to leave the run "
                f"unbounded."
            )
        return value
    if isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"context_budget={value} is not a share of the model's context "
                f"window. Pass a fraction above 0 and at most 1.0, a token "
                f"count as an int, 'auto', or None."
            )
        return value
    raise ValueError(
        f"context_budget={value!r} is not a budget. Pass 'auto' to derive one "
        f"from the window the model declares, an int for that many prompt "
        f"tokens, a float in (0, 1] for that share of the window, or None to "
        f"leave the run unbounded."
    )


def _declared_window(model: Any) -> int | None:
    """The context window *model* declares, or ``None`` when it declares none."""
    getter = getattr(model, "get_context_length", None)
    if not callable(getter):
        return None
    try:
        window = int(getter() or 0)
    except Exception:  # noqa: BLE001 - a model that cannot say is one that did not
        logger.debug("the model declined to state its context window", exc_info=True)
        return None
    return window if window > 0 else None


def resolve_context_budget(
    value: Any,
    *,
    model: Any,
    window_override: int | None = None,
    output_tokens: int | None = None,
) -> ContextBudget | None:
    """Decide the budget one run is bound by, once, before its first turn.

    Args:
        value: The caller's ``context_budget`` setting.
        model: The model the run's prompts go to.
        window_override: ``AgentConfig.max_context_length``, when the caller
            set one. It wins over the window the model declares, which is what
            that field has always been documented to mean.
        output_tokens: The output budget the run resolved, held back from the
            window so a turn has room to answer.

    Returns:
        The budget, or ``None`` when the run is unbounded — which is what a
        caller who passed ``None`` asked for, and what ``"auto"`` means on a
        model that declares no window it could be derived from.
    """
    if value is None:
        return None
    window = window_override or _declared_window(model)
    if isinstance(value, int) and not isinstance(value, bool):
        return ContextBudget(
            budget_tokens=int(value),
            window_tokens=window,
            source="config",
            model=model,
        )
    if window is None:
        # Nothing to derive from. Guessing a window would silently truncate a
        # conversation that would have fitted, so the run stays unbounded and
        # says so.
        logger.info(
            "[context] the model declares no context window, so the run is unbounded"
        )
        return None
    reserve = min(
        max(int(output_tokens or 0), _OUTPUT_RESERVE_CEILING),
        window // _RESERVE_SHARE_OF_WINDOW,
    )
    if isinstance(value, float):
        budget = int(window * value)
        source = "fraction"
    else:  # "auto"
        budget = int((window - reserve) * DEFAULT_HEADROOM)
        source = "auto"
    return ContextBudget(
        budget_tokens=max(1, budget),
        window_tokens=window,
        reserve_tokens=reserve,
        source=source,
        model=model,
    )
