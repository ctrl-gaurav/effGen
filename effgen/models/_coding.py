"""How well a model does the work a coding agent asks of it.

A coding turn is not a chat turn: the model has to answer with a *tool call*
that writes a file or runs a command, read the real output, and go again. Models
differ in whether they can do that at all. Three behaviours have been measured
on this framework's coding path, and each fails in a way that reads like
success:

- a chat template that renders no tool definitions — the model never receives
  the tools, so it describes the change instead of making it;
- a model that receives the tools but answers the question directly, so the run
  finishes with an answer and an empty ``files_written``;
- a model that writes the call out as prose instead of emitting it.

This module holds what has been measured, plus the rules that generalise past
the table, and answers one question: is this model a reasonable choice for a
coding run, and if not, what should be used instead. It never blocks a run —
the answer is a note, printed once, before the first call.

The data ages, so every measured entry carries the date it was confirmed on and
:data:`VERIFIED_ON` records the last review of the table as a whole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: The date the measured table below was last confirmed end to end.
VERIFIED_ON = "2026-08-09"

#: Verdicts, from most to least usable.
SUITABLE = "suitable"
LIMITED = "limited"
UNSUITABLE = "unsuitable"
UNKNOWN = "unknown"

#: Below this many billion parameters, a local model is reported as limited for
#: coding: the measured failures above all come from models under it.
SMALL_MODEL_B = 7.0

#: Model-id prefixes that name a local engine rather than a provider.
_LOCAL_ENGINES = frozenset({"transformers", "vllm", "gguf", "mlx"})

_PARAM_COUNT = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:[-_.]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class CodingSuitability:
    """What to expect from one model on a coding run.

    Attributes:
        model_id: The id asked about, as given.
        provider: The provider or local engine it resolves to, when known.
        verdict: ``suitable``, ``limited``, ``unsuitable`` or ``unknown``.
        reason: One sentence on what happens on a coding run.
        fix: What to use instead; empty when the model is suitable.
        evidence: Where the verdict comes from — ``measured`` (a run on this
            framework), ``capability`` (the loaded model reports no tool
            calling), ``catalog`` (the model catalog), or ``size``.
        measured_on: ISO date of the measurement, for a ``measured`` verdict.
    """

    model_id: str
    provider: str | None = None
    verdict: str = UNKNOWN
    reason: str = ""
    fix: str = ""
    evidence: str = ""
    measured_on: str | None = None

    @property
    def is_suitable(self) -> bool:
        """True when nothing needs to be said before the run."""
        return self.verdict == SUITABLE

    def note(self) -> str:
        """The one line shown before a coding run, or ``""`` when suitable."""
        if self.is_suitable:
            return ""
        parts = [f"{self.model_id} for coding: {self.reason}."]
        if self.fix:
            parts.append(self.fix)
        if self.measured_on:
            parts.append(f"(measured {self.measured_on})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return the note as a JSON-serializable dict."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "verdict": self.verdict,
            "reason": self.reason,
            "fix": self.fix,
            "evidence": self.evidence,
            "measured_on": self.measured_on,
        }


#: What a coding run does on these ids, measured on this framework. Keyed by the
#: bare model id, lowercased; a ``provider:id`` key pins one provider's serving
#: of a model that behaves differently elsewhere.
_MEASURED: dict[str, dict[str, Any]] = {
    "qwen/qwen2.5-1.5b-instruct": {
        "verdict": LIMITED,
        "reason": (
            "it receives the tools but often answers the task directly, so a "
            "run asked to write a file can finish with nothing written"
        ),
        "fix": (
            "Use a larger local model (Qwen/Qwen2.5-7B-Instruct) or a keyed "
            "cloud model for work that has to land on disk."
        ),
        "measured_on": "2026-08-09",
    },
    "google/gemma-2-2b-it": {
        "verdict": UNSUITABLE,
        "reason": (
            "its chat template renders no tool definitions, so it never "
            "receives the coding tools and can only describe a change"
        ),
        "fix": "Use Qwen/Qwen2.5-7B-Instruct locally, or a keyed cloud model.",
        "measured_on": "2026-08-06",
    },
    "microsoft/phi-3.5-mini-instruct": {
        "verdict": UNSUITABLE,
        "reason": (
            "its chat template renders no tool definitions, so it never "
            "receives the coding tools and can only describe a change"
        ),
        "fix": "Use Qwen/Qwen2.5-7B-Instruct locally, or a keyed cloud model.",
        "measured_on": "2026-08-06",
    },
    "qwen/qwen2.5-7b-instruct": {
        "verdict": SUITABLE,
        "reason": "it calls the coding tools and works from their real output",
        "measured_on": "2026-08-06",
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "verdict": SUITABLE,
        "reason": "it calls the coding tools and works from their real output",
        "measured_on": "2026-08-06",
    },
}


def split_model_id(model_id: str, provider: str | None = None) -> tuple[str, str | None]:
    """Return ``(bare_id, provider)`` for *model_id*, honoring a prefix.

    ``transformers:Qwen/Qwen2.5-7B-Instruct`` and ``openai:gpt-5-nano`` both
    carry their engine or provider in front of the id; an explicit *provider*
    argument is used when the id carries none.
    """
    text = (model_id or "").strip()
    if ":" in text:
        prefix, rest = text.split(":", 1)
        if rest:
            return rest, prefix
    return text, provider


def parameter_count_b(model_id: str) -> float | None:
    """Return the parameter count in billions read off *model_id*, or ``None``.

    Reads the ``7B``/``1.5b`` marker most published checkpoints carry in their
    name. A name without one yields ``None`` — never a guess.
    """
    match = _PARAM_COUNT.search(model_id.replace("/", "-"))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:  # pragma: no cover - the regex only matches numbers
        return None


def _from_table(bare: str, provider: str | None) -> dict[str, Any] | None:
    """Return the measured entry for this id, preferring a provider-pinned one."""
    keys = []
    if provider:
        keys.append(f"{provider.lower()}:{bare.lower()}")
    keys.append(bare.lower())
    for key in keys:
        entry = _MEASURED.get(key)
        if entry is not None:
            return entry
    return None


def coding_suitability(
    model_id: str, provider: str | None = None, model: Any = None
) -> CodingSuitability:
    """Report what to expect from *model_id* on a coding run.

    Resolution stops at the first answer: what has been measured, then what a
    loaded *model* says about its own tool calling, then what the catalog knows,
    then the model's size. An id nothing knows about is reported as
    ``unknown`` and the run goes ahead — the catalog is allowed to be behind.

    Args:
        model_id: The model id, with or without a ``provider:``/engine prefix.
        provider: Provider for a bare id.
        model: A loaded model, when there is one. Its ``tool_call_support()``
            is the check that generalises past the measured table.

    Returns:
        A :class:`CodingSuitability`. This function does not raise.
    """
    try:
        return _resolve(model_id, provider, model)
    except Exception:  # noqa: BLE001 - a note never breaks a run
        return CodingSuitability(model_id=model_id, provider=provider)


def _resolve(model_id: str, provider: str | None, model: Any) -> CodingSuitability:
    bare, resolved_provider = split_model_id(model_id, provider)

    def suitability(**fields: Any) -> CodingSuitability:
        """Build a result, filling in the two fields every branch shares."""
        return CodingSuitability(model_id=model_id, provider=resolved_provider, **fields)

    entry = _from_table(bare, resolved_provider)
    if entry is not None:
        return suitability(
            verdict=entry["verdict"],
            reason=entry["reason"],
            fix=entry.get("fix", ""),
            evidence="measured",
            measured_on=entry.get("measured_on"),
        )

    support = None
    if model is not None:
        probe = getattr(model, "tool_call_support", None)
        if callable(probe):
            support = probe()
    if support == "none":
        return suitability(
            verdict=UNSUITABLE,
            reason=(
                "it receives no tool definitions — its chat template renders "
                "none — so it can only describe a change, not make one"
            ),
            fix="Use Qwen/Qwen2.5-7B-Instruct locally, or a keyed cloud model.",
            evidence="capability",
        )

    record = _catalog_record(bare, resolved_provider)
    if record is not None and not record.supports_tools:
        return suitability(
            verdict=UNSUITABLE,
            reason=(
                "the catalog records no tool calling for it, so it cannot "
                "write a file or run a command"
            ),
            fix="Pick a model listed with tool support: effgen models list --tools.",
            evidence="catalog",
        )

    # The size rule is about the checkpoint, not about where it runs: a 3B model
    # served by a provider has the same trouble with a coding turn as the same
    # weights loaded locally.
    size = parameter_count_b(bare)
    if size is not None and size < SMALL_MODEL_B:
        return suitability(
            verdict=LIMITED,
            reason=(
                f"at about {size:g}B parameters it often answers a coding task "
                "instead of calling a tool, so a run can finish with nothing "
                "written"
            ),
            fix=(
                "Use a larger local model (Qwen/Qwen2.5-7B-Instruct) or a keyed "
                "cloud model for work that has to land on disk."
            ),
            evidence="size",
        )

    if record is not None and record.supports_tools:
        return suitability(
            verdict=SUITABLE,
            reason="the catalog records tool calling for it",
            evidence="catalog",
        )

    return suitability(
        verdict=UNKNOWN,
        reason=(
            "the catalog does not know this id, so how it handles the coding "
            "tools has not been measured"
        ),
        fix="Run it and check that files_written is not empty.",
        evidence="catalog",
    )


def _is_local_engine(provider: str | None) -> bool:
    """True when *provider* names a local engine rather than a served provider."""
    return (provider or "").lower() in _LOCAL_ENGINES


def _catalog_record(bare: str, provider: str | None) -> Any:
    """Return the catalog's record for this id, or ``None``.

    A checkpoint loaded by a local engine is resolved by download, not by the
    cloud catalog, so it is never looked up there: a provider that happens to
    serve a model of the same name says nothing about the local one. Cloud ids
    carrying a slash (``accounts/fireworks/models/…``, ``Qwen/Qwen3.5-9B``) are
    ordinary catalog entries and are looked up normally.
    """
    if _is_local_engine(provider):
        return None
    from effgen.models import _catalog

    return _catalog.lookup(bare, provider)


def measured_ids() -> tuple[str, ...]:
    """Return every id the measured table carries, as written in it."""
    return tuple(sorted(_MEASURED))
