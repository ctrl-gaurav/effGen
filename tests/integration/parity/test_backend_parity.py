"""
Backend parity tests — parametrized across all effGen providers.

The suite checks two independent guarantees so they never get conflated:

  * Answer correctness — every backend solves "(17 * 23) + sqrt(144) = 403",
    by whatever path it picks. A capable reasoning model may do the trivial
    arithmetic in its head; requiring a Calculator call here would falsely fail
    such a model, so this assertion does not look at tool counts.

  * Tool calling works — every backend that advertises native tool calling must
    answer a question whose value is unknowable without invoking the tool (an
    opaque inventory count). Reaching that value is direct proof the tool path
    is wired up, and it stays a fair test even for a model clever enough to
    shortcut arithmetic.

Tests make live API calls and are individually skipped when the provider's API
key is absent.

Run:
    pytest tests/integration/parity/test_backend_parity.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# Load env from both standard locations
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _has(env_var: str) -> bool:
    return bool(os.getenv(env_var))


LIVE_TEST_TIMEOUT_SECONDS = 20
LIVE_TEST_MAX_RETRIES = 1
TRANSIENT_PROVIDER_KEYWORDS = (
    "quota",
    "429",
    "402",
    "payment required",
    "resource_exhausted",
    "insufficient credits",
    "credits exhausted",
    "exceeded your monthly included credits",
    "depleted your monthly included credits",
    "rate_limit",
    "rate limit",
    "timeout",
    "timed out",
    "503",
    "service unavailable",
    "server error",
    "api failed",
)


def _is_transient_provider_error(*texts: str | None) -> bool:
    haystack = " ".join(text or "" for text in texts).lower()
    return any(keyword in haystack for keyword in TRANSIENT_PROVIDER_KEYWORDS)


# ---------------------------------------------------------------------------
# Provider × model matrix
# Prefer small/fast/free models; one per provider. Pick an instruct model rather
# than a reasoning model: the canonical task runs on the default token budget,
# and a reasoning model can spend all of it before emitting a visible token.
# ---------------------------------------------------------------------------

PARITY_PARAMS = [
    pytest.param(
        "cerebras", "gpt-oss-120b",
        marks=pytest.mark.skipif(not _has("CEREBRAS_API_KEY"), reason="SKIPPED: CEREBRAS_API_KEY not set"),
        id="cerebras/gpt-oss-120b",
    ),
    pytest.param(
        "groq", "qwen/qwen3.8-27b",
        marks=pytest.mark.skipif(not _has("GROQ_API_KEY"), reason="SKIPPED: GROQ_API_KEY not set"),
        id="groq/qwen3.8-27b",
    ),
    pytest.param(
        "together", "Qwen/Qwen2.5-7B-Instruct-Turbo",
        marks=pytest.mark.skipif(not _has("TOGETHER_API_KEY"), reason="SKIPPED: TOGETHER_API_KEY not set"),
        id="together/qwen2.5-7b-instruct-turbo",
    ),
    pytest.param(
        "fireworks", "accounts/fireworks/models/kimi-k2p6",
        marks=pytest.mark.skipif(not _has("FIREWORKS_API_KEY"), reason="SKIPPED: FIREWORKS_API_KEY not set"),
        id="fireworks/kimi-k2p6",
    ),
    pytest.param(
        "hf", "Qwen/Qwen2.5-72B-Instruct",
        marks=pytest.mark.skipif(not _has("HF_TOKEN"), reason="SKIPPED: HF_TOKEN not set"),
        id="hf/qwen2.5-72b",
    ),
    pytest.param(
        "gemini", "gemini-2.5-flash-lite",
        marks=pytest.mark.skipif(not _has("GOOGLE_API_KEY"), reason="SKIPPED: GOOGLE_API_KEY not set"),
        id="gemini/gemini-2.5-flash-lite",
    ),
    pytest.param(
        "openai", "gpt-4o-mini",
        marks=pytest.mark.skipif(not _has("OPENAI_API_KEY"), reason="SKIPPED: OPENAI_API_KEY not set"),
        id="openai/gpt-4o-mini",
    ),
    pytest.param(
        "anthropic", "claude-3-haiku-20240307",
        marks=pytest.mark.skipif(not _has("ANTHROPIC_API_KEY"), reason="SKIPPED: ANTHROPIC_API_KEY not set"),
        id="anthropic/claude-3-haiku",
    ),
    # Replicate: no billing credits — mark as xfail if key present but credits absent
    pytest.param(
        "replicate", "meta/meta-llama-3-8b-instruct",
        marks=[
            pytest.mark.skipif(not _has("REPLICATE_API_TOKEN"), reason="SKIPPED: REPLICATE_API_TOKEN not set"),
            pytest.mark.xfail(reason="Replicate requires billing credits; may fail without balance", strict=False),
        ],
        id="replicate/llama-3-8b",
    ),
]


def _load_adapter(provider: str, model_id: str):
    """Load a live adapter for (provider, model_id)."""
    a: Any
    if provider == "cerebras":
        from effgen.models.cerebras_adapter import CerebrasAdapter
        a = CerebrasAdapter(model_id)
    elif provider == "groq":
        from effgen.models.groq_adapter import GroqAdapter
        a = GroqAdapter(model_id)
    elif provider == "together":
        from effgen.models.together_adapter import TogetherAdapter
        a = TogetherAdapter(
            model_id,
            max_retries=LIVE_TEST_MAX_RETRIES,
            timeout=LIVE_TEST_TIMEOUT_SECONDS,
        )
    elif provider == "fireworks":
        from effgen.models.fireworks_adapter import FireworksAdapter
        a = FireworksAdapter(
            model_id,
            max_retries=LIVE_TEST_MAX_RETRIES,
            timeout=30,
            enable_rate_limiting=False,
        )
    elif provider == "hf":
        from effgen.models.hf_inference_adapter import HFInferenceAdapter
        a = HFInferenceAdapter(model_id)
    elif provider == "gemini":
        from effgen.models.gemini_adapter import GeminiAdapter
        a = GeminiAdapter(model_id)
    elif provider == "openai":
        from effgen.models.openai_adapter import OpenAIAdapter
        a = OpenAIAdapter(model_id)
    elif provider == "anthropic":
        from effgen.models.anthropic_adapter import AnthropicAdapter
        a = AnthropicAdapter(model_id)
    elif provider == "replicate":
        from effgen.models.replicate_adapter import ReplicateAdapter
        a = ReplicateAdapter(model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    a.load()
    return a


# ---------------------------------------------------------------------------
# Main parity test
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.live
@pytest.mark.parametrize("provider,model_id", PARITY_PARAMS)
def test_canonical_task_parity(provider, model_id):
    """Every backend reaches (17*23)+sqrt(144)=403, by whatever path it picks.

    This is the answer-correctness guarantee. It deliberately does not require a
    Calculator call: a capable model that does the arithmetic in its reasoning
    is just as correct as one that delegates to the tool, and demanding a tool
    call here would falsely fail the former. Tool calling is proved separately by
    ``test_tool_required_parity``.
    """
    from tests.integration.parity.canonical_task import run_canonical_task

    adapter = _load_adapter(provider, model_id)
    try:
        result = run_canonical_task(adapter, strategy="react")
    finally:
        try:
            adapter.unload()
        except Exception:
            pass

    # Skip on provider-side transient errors, not framework behavior.
    if _is_transient_provider_error(result["error"], result["answer_text"]):
        pytest.skip(f"{provider}/{model_id} transient provider error — re-run after cooldown")

    assert result["error"] is None, (
        f"{provider}/{model_id} raised an error: {result['error']}"
    )
    assert result["answer_correct"], (
        f"{provider}/{model_id} answer missing '403'. Got: {result['answer_text']!r}"
    )


# ---------------------------------------------------------------------------
# Native tool-calling strategy (providers that support it)
# ---------------------------------------------------------------------------

NATIVE_PARAMS = [
    pytest.param(
        "cerebras", "gpt-oss-120b",
        marks=pytest.mark.skipif(not _has("CEREBRAS_API_KEY"), reason="SKIPPED: CEREBRAS_API_KEY not set"),
        id="cerebras/gpt-oss-120b/native",
    ),
    pytest.param(
        "groq", "qwen/qwen3.8-27b",
        marks=pytest.mark.skipif(not _has("GROQ_API_KEY"), reason="SKIPPED: GROQ_API_KEY not set"),
        id="groq/qwen3.8-27b/native",
    ),
    pytest.param(
        "fireworks", "accounts/fireworks/models/kimi-k2p6",
        marks=pytest.mark.skipif(not _has("FIREWORKS_API_KEY"), reason="SKIPPED: FIREWORKS_API_KEY not set"),
        id="fireworks/kimi-k2p6/native",
    ),
    pytest.param(
        "together", "Qwen/Qwen2.5-7B-Instruct-Turbo",
        marks=pytest.mark.skipif(not _has("TOGETHER_API_KEY"), reason="SKIPPED: TOGETHER_API_KEY not set"),
        id="together/qwen2.5-7b-instruct-turbo/native",
    ),
    pytest.param(
        "gemini", "gemini-2.5-flash-lite",
        marks=pytest.mark.skipif(not _has("GOOGLE_API_KEY"), reason="SKIPPED: GOOGLE_API_KEY not set"),
        id="gemini/gemini-2.5-flash-lite/native",
    ),
    pytest.param(
        "openai", "gpt-4o-mini",
        marks=pytest.mark.skipif(not _has("OPENAI_API_KEY"), reason="SKIPPED: OPENAI_API_KEY not set"),
        id="openai/gpt-4o-mini/native",
    ),
]


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.live
@pytest.mark.parametrize("provider,model_id", NATIVE_PARAMS)
def test_native_strategy_parity(provider, model_id):
    """Native tool-calling providers must also solve the canonical task."""
    from tests.integration.parity.canonical_task import run_canonical_task

    adapter = _load_adapter(provider, model_id)
    try:
        result = run_canonical_task(adapter, strategy="native")
    finally:
        try:
            adapter.unload()
        except Exception:
            pass

    # Skip on provider-side transient errors, not framework behavior.
    if _is_transient_provider_error(result["error"], result["answer_text"]):
        pytest.skip(f"{provider}/{model_id} (native) transient provider error — re-run after cooldown")

    assert result["error"] is None, (
        f"{provider}/{model_id} (native) raised an error: {result['error']}"
    )
    assert result["answer_correct"], (
        f"{provider}/{model_id} (native) answer missing '403'. Got: {result['answer_text']!r}"
    )


# ---------------------------------------------------------------------------
# Tool-calling proof — a question only the tool can answer
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.live
@pytest.mark.parametrize("provider,model_id", NATIVE_PARAMS)
def test_tool_required_parity(provider, model_id):
    """Every native-tool backend must actually call a tool when it has to.

    The question asks for an opaque inventory count that exists only inside the
    tool, so the answer cannot be guessed or reasoned out — reaching it proves
    the backend's tool-calling path works. Unlike the canonical arithmetic task,
    this stays a fair tool-use proof even for a model capable enough to shortcut
    simple math. The backend passes if it calls the tool through any of its
    tool-calling modes (native first, then the scaffolded react loop).
    """
    from tests.integration.parity.canonical_task import (
        TOOL_REQUIRED_SENTINEL,
        run_tool_required_task,
    )

    adapter = _load_adapter(provider, model_id)
    try:
        result = run_tool_required_task(adapter)
    finally:
        try:
            adapter.unload()
        except Exception:
            pass

    # Skip on provider-side transient errors, not framework behavior.
    if _is_transient_provider_error(result["error"], result["answer_text"]):
        pytest.skip(f"{provider}/{model_id} (tool-required) transient provider error — re-run after cooldown")

    assert result["error"] is None, (
        f"{provider}/{model_id} (tool-required) raised an error: {result['error']}"
    )
    assert result["tool_calls"] >= 1, (
        f"{provider}/{model_id} answered without calling the tool in any mode "
        f"(last tried {result.get('strategy')!r}, got {result['tool_calls']} calls); "
        f"the value is unknowable otherwise. Answer: {result['answer_text']!r}"
    )
    assert result["answer_correct"], (
        f"{provider}/{model_id} called the tool but the answer is missing "
        f"{TOOL_REQUIRED_SENTINEL}. Got: {result['answer_text']!r}"
    )
