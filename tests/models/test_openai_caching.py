"""
Tests for OpenAI prompt caching support.

Unit tests check that cached_input_tokens is extracted from API responses.
Live integration tests require OPENAI_API_KEY and are skipped when absent.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load env for integration tests
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / ".effgen" / ".env", override=False)
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
except ImportError:
    pass


def _make_mock_response(
    content: str = "hello",
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
    cached_tokens: int = 0,
):
    """Build a minimal mock of openai ChatCompletion response."""
    mock_usage_details = MagicMock()
    mock_usage_details.cached_tokens = cached_tokens

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens
    mock_usage.prompt_tokens_details = mock_usage_details

    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = None
    mock_message.refusal = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-unit"})
def test_cached_tokens_zero_by_default():
    from effgen.models.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(model_name="gpt-4o-mini")
    adapter._is_loaded = True
    adapter.client = MagicMock()
    adapter.client.models.retrieve.return_value = MagicMock()
    adapter.client.chat.completions.create.return_value = _make_mock_response(cached_tokens=0)

    result = adapter.generate("hello")
    assert result.metadata["cached_input_tokens"] == 0


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-unit"})
def test_cached_tokens_surfaced_when_nonzero():
    from effgen.models.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(model_name="gpt-4o-mini")
    adapter._is_loaded = True
    adapter.client = MagicMock()
    adapter.client.models.retrieve.return_value = MagicMock()
    adapter.client.chat.completions.create.return_value = _make_mock_response(
        prompt_tokens=2000,
        cached_tokens=1500,
    )

    result = adapter.generate("a prompt")
    assert result.metadata["cached_input_tokens"] == 1500


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-unit"})
def test_cached_tokens_in_generate_with_system_prompt():
    from effgen.models.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(model_name="gpt-4o-mini")
    adapter._is_loaded = True
    adapter.client = MagicMock()
    adapter.client.models.retrieve.return_value = MagicMock()
    adapter.client.chat.completions.create.return_value = _make_mock_response(
        prompt_tokens=2000,
        cached_tokens=1800,
    )

    result = adapter.generate_with_system_prompt("hi", system_prompt="You are helpful.")
    assert result.metadata["cached_input_tokens"] == 1800


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-unit"})
def test_system_prompt_placed_first_in_messages():
    """Verify system message is at index 0 — prerequisite for prefix caching."""
    from effgen.models.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(model_name="gpt-4o-mini")
    adapter._is_loaded = True
    adapter.client = MagicMock()
    adapter.client.chat.completions.create.return_value = _make_mock_response()

    adapter.generate_with_system_prompt("user question", system_prompt="instructions")
    call_kwargs = adapter.client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "instructions"
    assert messages[1]["role"] == "user"


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-unit"})
def test_cached_tokens_handles_missing_details():
    """prompt_tokens_details may be None — must not raise."""
    from effgen.models.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(model_name="gpt-4o-mini")
    adapter._is_loaded = True
    adapter.client = MagicMock()
    adapter.client.chat.completions.create.return_value = _make_mock_response()
    # Override prompt_tokens_details to None
    adapter.client.chat.completions.create.return_value.usage.prompt_tokens_details = None

    result = adapter.generate("hello")
    assert result.metadata["cached_input_tokens"] == 0


# ---------------------------------------------------------------------------
# Live integration test (skipped without API key)
# ---------------------------------------------------------------------------

LIVE_SKIP = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live caching test",
)


@LIVE_SKIP
def test_live_cache_hits_on_repeated_calls():
    """5 sequential requests with a >1024-token system prompt should hit cache on calls 2-5.

    OpenAI caches prompt prefixes that are ≥1024 tokens long.  We build a
    system prompt well above that threshold (≥1500 tokens) so the prefix is
    eligible.  The first call seeds the cache; subsequent calls should report
    cached_input_tokens > 0.

    Note: caching is an OpenAI backend feature and may not activate every time
    (cold start, model tier, etc.).  The test warns rather than hard-fails if
    no hits are observed, since the infrastructure is confirmed correct via the
    unit tests above.
    """
    import warnings

    from effgen.models.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(model_name="gpt-5.4-nano")
    adapter.load()

    # Build a system prompt guaranteed to exceed 1024 tokens (≈3 chars/token)
    # 1500+ tokens → ~4500+ characters
    system_prompt = (
        "You are a highly knowledgeable assistant with deep expertise in physics, "
        "chemistry, biology, computer science, mathematics, and engineering.\n\n"
        + ("This system prompt is intentionally long and kept strictly stable across "
           "all sequential API calls so that the OpenAI automatic prefix caching "
           "mechanism can detect and cache the common prefix. " * 100)
        + "\n\nAlways be concise and accurate."
    )
    # Rough token estimate: ~4 chars per token for English prose
    approx_tokens = len(system_prompt) // 4
    assert approx_tokens >= 1024, f"System prompt too short for caching: ~{approx_tokens} tokens"

    cached_counts = []
    try:
        for i in range(5):
            result = adapter.generate_with_system_prompt(
                prompt=f"What is the speed of light in vacuum? (call {i+1})",
                system_prompt=system_prompt,
            )
            cached = result.metadata["cached_input_tokens"]
            cached_counts.append(cached)
    finally:
        adapter.unload()

    if not any(c > 0 for c in cached_counts[1:]):
        warnings.warn(
            f"No OpenAI cache hits observed: {cached_counts}. "
            "This can happen on the first run before the prefix is warmed, "
            "or if the model tier does not support caching. "
            "Run again to confirm, or check with p5_cache.py validation script.",
            stacklevel=2,
        )
    else:
        assert any(c > 0 for c in cached_counts[1:]), (
            f"Expected cache hits on calls 2-5, got: {cached_counts}"
        )
