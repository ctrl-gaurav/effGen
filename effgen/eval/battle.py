"""Head-to-head model runs over a single prompt.

Sends one prompt to several models at once, streaming each answer as it
arrives, and reports what each model said next to what it cost and how long it
took. Built on :meth:`effgen.core.agent.Agent.stream`, so a battle adds no new
model-execution path: the tokens, the cost and the timings all come from the
same machinery a single ``run`` uses.

Usage::

    from effgen.eval.battle import run_battle

    result = run_battle("Explain a hash map in two sentences.",
                        ["openai:gpt-5-nano", "groq:openai/gpt-oss-20b"])
    for c in result.contenders:
        print(c.model, c.latency_s, c.cost_usd)
        print(c.answer)
    print(result.verdict["fastest"], result.verdict["cheapest"])
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: How a contender is progressing. ``loading`` covers fetching or initializing
#: the model — real time for a local model, and reported separately from
#: generation so a column that is loading does not read as one that is stuck.
STATES = ("pending", "loading", "streaming", "done", "failed")


@dataclass
class Contender:
    """One model's side of a battle.

    Attributes:
        model: The model id as the caller wrote it.
        answer: The model's answer, accumulated as it streamed.
        error: The failure message when the model did not produce an answer.
        state: One of :data:`STATES`.
        load_s: Seconds spent loading the model before generation began.
        ttft_s: Seconds from the start of generation to the first answer token.
        latency_s: Seconds from the start of generation to the last token.
        prompt_tokens/completion_tokens/total_tokens: Token counts for the run.
        cost_usd: What the run cost, or ``None`` for a model with no published
            per-token price.
        estimated_tokens: ``True`` when the counts were counted locally because
            the backend reported none.
    """

    model: str
    answer: str = ""
    error: str | None = None
    state: str = "pending"
    load_s: float | None = None
    ttft_s: float | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    estimated_tokens: bool = False

    @property
    def ok(self) -> bool:
        """Whether this contender produced an answer."""
        return self.error is None and bool(self.answer)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "model": self.model,
            "answer": self.answer,
            "error": self.error,
            "state": self.state,
            "load_s": _round(self.load_s, 4),
            "ttft_s": _round(self.ttft_s, 4),
            "latency_s": _round(self.latency_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass
class BattleResult:
    """Every contender's answer and measurements for one prompt.

    Attributes:
        verdict: The measured outcomes — ``fastest`` (lowest end-to-end
            latency), ``cheapest`` (lowest cost among priced models),
            ``longest`` (most characters) — plus ``judge`` when a judge model
            was asked for a pick. Each entry names the model, or is ``None``
            when nothing qualified. Failed contenders are never eligible.
    """

    prompt: str
    contenders: list[Contender] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    generated_at: str | None = None

    @property
    def finishers(self) -> list[Contender]:
        """Contenders that produced an answer."""
        return [c for c in self.contenders if c.ok]

    def total_cost(self) -> float | None:
        """What the whole battle cost, or ``None`` if nothing was priced."""
        priced = [c.cost_usd for c in self.contenders if c.cost_usd is not None]
        return sum(priced) if priced else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "prompt": self.prompt,
            "contenders": [c.to_dict() for c in self.contenders],
            "verdict": self.verdict,
            "wall_s": _round(self.wall_s, 4),
            "total_cost_usd": self.total_cost(),
            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the JSON rendering of :meth:`to_dict`."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render the battle as Markdown: the tally, the verdict, the answers."""
        lines = ["# Model Battle", "", f"**Prompt:** {self.prompt}", ""]
        lines.extend([
            "| Model | Result | TTFT | Latency | Tokens (in/out) | Cost |",
            "|-------|--------|------|---------|-----------------|------|",
        ])
        for c in self.contenders:
            result = "answered" if c.ok else "failed"
            ttft = f"{c.ttft_s:.2f}s" if c.ttft_s is not None else "—"
            lat = f"{c.latency_s:.2f}s" if c.latency_s is not None else "—"
            tok = (
                f"{c.prompt_tokens or 0}/{c.completion_tokens or 0}"
                + ("*" if c.estimated_tokens else "")
                if c.total_tokens is not None
                else "—"
            )
            # A model that never answered has no price to report — an empty
            # cell, not "unpriced", which describes a model that did run.
            cost = fmt_cost(c.cost_usd) if c.ok else "—"
            lines.append(f"| {c.model} | {result} | {ttft} | {lat} | {tok} | {cost} |")
        if any(c.estimated_tokens for c in self.contenders):
            lines.extend(["", "\\* token counts counted locally, not reported by the provider"])

        lines.extend(["", "## Verdict", ""])
        if not self.finishers:
            lines.append("No model answered, so there is nothing to compare.")
        for key in ("fastest", "cheapest", "longest"):
            entry = self.verdict.get(key)
            if entry:
                lines.append(f"- **{key.capitalize()}**: {entry['model']} — {entry['detail']}")
        judge = self.verdict.get("judge")
        if judge:
            if judge.get("winner"):
                lines.append(
                    f"- **Judged pick**: {judge['winner']} — {judge.get('reasoning', '')} "
                    f"(judged by {judge['judge_model']})"
                )
            elif judge.get("error"):
                lines.append(f"- **Judged pick**: unavailable — {judge['error']}")

        for c in self.contenders:
            lines.extend(["", f"## {c.model}", ""])
            lines.append(c.answer if c.ok else f"_Did not answer — {c.error}_")

        total = self.total_cost()
        spent = fmt_cost(total) if self.finishers else "—"
        lines.extend([
            "",
            (f"_{len(self.finishers)}/{len(self.contenders)} answered in "
            f"{self.wall_s:.2f}s; total cost {spent}._"),
        ])
        return "\n".join(lines)


def fmt_cost(value: float | None) -> str:
    """Render a USD amount, or say the model publishes no price."""
    if value is None:
        return "unpriced"
    if value == 0:
        return "$0.00"
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


# ---------------------------------------------------------------------------
# Running a battle
# ---------------------------------------------------------------------------

def _run_contender(
    contender: Contender,
    prompt: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
    system_prompt: str | None,
    on_update: Callable[[Contender], None],
) -> None:
    """Stream one model's answer into *contender*, reporting progress.

    Loading and generating are timed separately: a local model can spend most
    of its wall clock being loaded, which is real but is not the latency of the
    model answering.
    """
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models import load_model

    agent = None
    load_started = time.perf_counter()
    contender.state = "loading"
    on_update(contender)
    try:
        model = load_model(contender.model)
        contender.load_s = time.perf_counter() - load_started
        config_kwargs: dict[str, Any] = {
            "name": f"battle-{contender.model}",
            "model": model,
            "require_model": True,
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_tokens"] = max_tokens
        if system_prompt:
            config_kwargs["system_prompt"] = system_prompt
        agent = Agent(AgentConfig(**config_kwargs))

        gen_started = time.perf_counter()
        contender.state = "streaming"
        on_update(contender)
        for delta in agent.stream(prompt):
            if not delta:
                continue
            if contender.ttft_s is None:
                contender.ttft_s = time.perf_counter() - gen_started
            contender.answer += delta
            on_update(contender)
        contender.latency_s = time.perf_counter() - gen_started

        usage = agent.last_stream_usage or {}
        contender.prompt_tokens = usage.get("prompt_tokens")
        contender.completion_tokens = usage.get("completion_tokens")
        contender.total_tokens = usage.get("total_tokens")
        contender.cost_usd = usage.get("cost_usd")
        contender.estimated_tokens = bool(usage.get("estimated"))
        contender.state = "done"
        if not contender.answer:
            contender.error = "the model returned no answer"
            contender.state = "failed"
    except Exception as exc:  # noqa: BLE001 - one contender failing must not end the battle
        logger.debug("Contender %s failed", contender.model, exc_info=True)
        contender.error = str(exc)
        contender.state = "failed"
        if contender.latency_s is None:
            contender.latency_s = time.perf_counter() - load_started
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                logger.debug("Closing battle agent failed", exc_info=True)
        on_update(contender)


def _measured_verdict(contenders: list[Contender]) -> dict[str, Any]:
    """Pick the fastest, cheapest and longest answers from what was measured.

    Only contenders that answered are eligible, so a model that failed can
    never win by virtue of having spent nothing and taken no time.
    """
    finishers = [c for c in contenders if c.ok]
    verdict: dict[str, Any] = {"fastest": None, "cheapest": None, "longest": None}
    if not finishers:
        return verdict

    timed = [c for c in finishers if c.latency_s is not None]
    if timed:
        fastest = min(timed, key=lambda c: c.latency_s or 0.0)
        verdict["fastest"] = {
            "model": fastest.model,
            "detail": f"answered in {fastest.latency_s:.2f}s",
        }
    priced = [c for c in finishers if c.cost_usd is not None]
    if priced:
        cheapest = min(priced, key=lambda c: c.cost_usd or 0.0)
        unpriced = len(finishers) - len(priced)
        detail = f"{fmt_cost(cheapest.cost_usd)} for this run"
        if unpriced:
            detail += f"; {unpriced} model(s) publish no price and were not ranked"
        verdict["cheapest"] = {"model": cheapest.model, "detail": detail}
    longest = max(finishers, key=lambda c: len(c.answer))
    verdict["longest"] = {
        "model": longest.model,
        "detail": f"{len(longest.answer)} characters",
    }
    return verdict


def judge_battle(judge_model: str, prompt: str, contenders: list[Contender]) -> dict[str, Any]:
    """Ask *judge_model* which answer is best. Returns the judge's pick.

    The judge is a model the caller names, not one of the contenders by
    default — a contender grading its own answer is the bias a head-to-head
    exists to avoid. The returned dict always names the judge so a reader can
    weigh the pick, and it is reported separately from the measured numbers.

    Args:
        judge_model: The model asked to pick a winner.
        prompt: The prompt the contenders answered.
        contenders: The answers to compare.
    """
    from effgen.core.agent import Agent, AgentConfig

    finishers = [c for c in contenders if c.ok]
    if len(finishers) < 2:
        return {
            "judge_model": judge_model,
            "winner": None,
            "error": "fewer than two contenders answered",
        }

    answers = "\n\n".join(
        f"Answer {idx}. (from {c.model}):\n{c.answer}" for idx, c in enumerate(finishers, 1)
    )
    judge_prompt = (
        "You are judging answers to one question. Pick the single best answer "
        "on accuracy, then clarity.\n"
        'Respond ONLY with JSON: {"winner": <answer number>, "reasoning": "<one sentence>"}\n\n'
        f"Question: {prompt}\n\n{answers}\n"
    )
    agent = None
    try:
        agent = Agent(AgentConfig(name=f"battle-judge-{judge_model}",
                                  model=judge_model, require_model=True))
        response = agent.run(judge_prompt)
        text = getattr(response, "output", "") or str(response)
        import re

        match = re.search(r'\{.*?"winner"\s*:\s*(\d+).*?\}', text, re.DOTALL)
        if not match:
            return {
                "judge_model": judge_model,
                "winner": None,
                "error": "the judge did not return a pick",
            }
        index = int(match.group(1))
        if not 1 <= index <= len(finishers):
            return {
                "judge_model": judge_model,
                "winner": None,
                "error": f"the judge named answer {index}, which was not in the field",
            }
        try:
            reasoning = json.loads(match.group(0)).get("reasoning", "")
        except (ValueError, TypeError):
            reasoning = ""
        return {
            "judge_model": judge_model,
            "winner": finishers[index - 1].model,
            "reasoning": reasoning,
        }
    except Exception as exc:  # noqa: BLE001 - a judge failure leaves the measurements intact
        logger.debug("Battle judge failed", exc_info=True)
        return {"judge_model": judge_model, "winner": None, "error": str(exc)}
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:  # noqa: BLE001
                logger.debug("Closing judge agent failed", exc_info=True)


def run_battle(
    prompt: str,
    models: list[str],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    judge: str | None = None,
    on_update: Callable[[Contender], None] | None = None,
) -> BattleResult:
    """Send *prompt* to every model in *models* at once and collect the results.

    Args:
        prompt: The prompt every contender answers.
        models: Model ids, e.g. ``["openai:gpt-5-nano", "groq:openai/gpt-oss-20b"]``.
        temperature: Sampling temperature applied to every contender.
        max_tokens: Output cap applied to every contender.
        system_prompt: System prompt applied to every contender.
        judge: Model id asked to pick a winner. Optional; the measured
            outcomes (fastest, cheapest, longest) need no judge and are always
            reported.
        on_update: Called with a :class:`Contender` whenever it changes, from
            that contender's own thread — used to drive a live display. It must
            be cheap and must not raise.

    Returns:
        A :class:`BattleResult`. A contender that fails is recorded with its
        error and does not end the battle or affect the others.

    Raises:
        ValueError: If no prompt or fewer than two models were given.
    """
    if not prompt or not prompt.strip():
        raise ValueError("a battle needs a prompt")
    if len(models) < 2:
        raise ValueError("a battle needs at least two models to compare")

    contenders = [Contender(model=m) for m in models]
    lock = threading.Lock()

    def _notify(contender: Contender) -> None:
        if on_update is None:
            return
        with lock:
            try:
                on_update(contender)
            except Exception:  # noqa: BLE001 - a display must never break the run
                logger.debug("Battle update callback failed", exc_info=True)

    started = time.perf_counter()
    threads = [
        threading.Thread(
            target=_run_contender,
            args=(c, prompt),
            kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "on_update": _notify,
            },
            name=f"battle-{c.model}",
            daemon=True,
        )
        for c in contenders
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    result = BattleResult(
        prompt=prompt,
        contenders=contenders,
        wall_s=time.perf_counter() - started,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    result.verdict = _measured_verdict(contenders)
    if judge:
        result.verdict["judge"] = judge_battle(judge, prompt, contenders)
    return result
