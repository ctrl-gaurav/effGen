"""
Live integration test: a run reports a price only when one is published.

Three real calls, one per pricing status:

* ``groq:openai/gpt-oss-20b`` — the catalog publishes a rate, so the call
  reports a real ``cost_usd``.
* ``cerebras:gpt-oss-120b`` — a genuine free tier, so ``cost_usd`` is ``0.0``.
* ``groq:allam-2-7b`` — the catalog carries no rate, so ``cost_usd`` is
  ``None``. It used to report ``0.0``, which a reader could not tell apart
  from the free tier above.

Both the non-streaming and the streaming path are checked, plus what
``Agent.run`` puts on the response metadata.

Skipped automatically when the provider key is absent.
Run with:
    pytest tests/integration/test_unpriced_cost_live.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parents[2] / ".env", override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

skip_no_groq = pytest.mark.skipif(
    not GROQ_API_KEY, reason="GROQ_API_KEY not set — live pricing test skipped"
)
skip_no_cerebras = pytest.mark.skipif(
    not CEREBRAS_API_KEY, reason="CEREBRAS_API_KEY not set — live pricing test skipped"
)

PRICED_MODEL = "groq:openai/gpt-oss-20b"
FREE_MODEL = "cerebras:gpt-oss-120b"
UNPRICED_MODEL = "groq:allam-2-7b"

PROMPT = "Reply with exactly: OK"


def _skip_on_transient(exc: Exception) -> None:
    msg = str(exc).lower()
    if any(
        key in msg
        for key in ("429", "503", "rate limit", "too_many_requests", "overload", "quota")
    ):
        pytest.skip(f"transient upstream condition: {exc}")


def _generate(spec: str):
    from effgen import load_model
    from effgen.models.base import GenerationConfig

    model = load_model(spec)
    try:
        return model, model.generate(PROMPT, GenerationConfig(max_tokens=256, temperature=0.0))
    except Exception as exc:
        _skip_on_transient(exc)
        raise


@skip_no_groq
def test_a_priced_model_reports_a_real_cost():
    _model, result = _generate(PRICED_MODEL)
    cost = result.metadata["cost_usd"]
    assert isinstance(cost, float) and cost > 0, result.metadata


@skip_no_cerebras
def test_a_free_tier_reports_zero_because_zero_is_the_truth():
    _model, result = _generate(FREE_MODEL)
    assert result.metadata["cost_usd"] == 0.0, result.metadata


@skip_no_groq
def test_an_unpriced_model_reports_no_cost_on_the_non_streaming_path():
    from effgen.models._cost import pricing_status

    assert pricing_status("groq", "allam-2-7b") == "unpriced"
    _model, result = _generate(UNPRICED_MODEL)
    assert result.metadata["cost_usd"] is None, result.metadata
    # The tokens are known even though the money is not.
    assert result.metadata["total_tokens"] > 0


@skip_no_groq
def test_an_unpriced_model_reports_no_cost_on_the_streaming_path():
    from effgen import load_model
    from effgen.models.base import GenerationConfig, get_stream_usage

    model = load_model(UNPRICED_MODEL)
    try:
        chunks = list(
            model.generate_stream(PROMPT, GenerationConfig(max_tokens=256, temperature=0.0))
        )
    except Exception as exc:
        _skip_on_transient(exc)
        raise
    assert "".join(chunks).strip()
    usage = get_stream_usage(model)
    assert usage is not None, "the stream reported no usage at all"
    assert usage["cost_usd"] is None, usage
    assert usage["total_tokens"] > 0


@skip_no_groq
def test_a_run_on_an_unpriced_model_carries_no_cost_and_says_how_many_calls():
    from effgen import Agent
    from effgen.core.agent import AgentConfig

    agent = Agent(config=AgentConfig(model=UNPRICED_MODEL, max_tokens=256, temperature=0.0))
    try:
        response = agent.run("What is 2+2? Answer with the number only.")
    except Exception as exc:
        _skip_on_transient(exc)
        raise
    assert response.success, response.metadata
    assert "cost_usd" not in response.metadata, response.metadata
    assert response.metadata["unpriced_calls"] >= 1, response.metadata


@skip_no_groq
def test_a_run_on_a_priced_model_still_carries_its_cost():
    from effgen import Agent
    from effgen.core.agent import AgentConfig

    agent = Agent(config=AgentConfig(model=PRICED_MODEL, max_tokens=256, temperature=0.0))
    try:
        response = agent.run("What is 2+2? Answer with the number only.")
    except Exception as exc:
        _skip_on_transient(exc)
        raise
    assert response.success, response.metadata
    assert response.metadata["cost_usd"] > 0, response.metadata
    assert "unpriced_calls" not in response.metadata, response.metadata
