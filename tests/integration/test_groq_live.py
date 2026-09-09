"""Integration tests for GroqAdapter — skipped if GROQ_API_KEY absent, real calls otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _skip_rate_limit(exc: Exception) -> None:
    from effgen.models._rate_limit import RateLimitExceeded

    if isinstance(exc, RateLimitExceeded):
        pytest.skip(f"Groq transient rate limit/quota exhausted: {exc}")
    raise exc


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: GROQ_API_KEY not set")
class TestGroqLive:
    def test_the_default_model_answers(self):
        """A GroqAdapter built with no arguments reaches a model Groq serves.

        This is the first call most users make, and the shipped default is the
        one id nobody passes explicitly — so nothing else catches it going
        stale. The budget is deliberately generous: the default is a reasoning
        family that spends output tokens on a hidden chain before it emits any
        answer, and a budget that only covers the chain returns empty text with
        ``finish_reason="length"`` rather than an error.
        """
        from effgen.models.groq_adapter import GroqAdapter
        from effgen.models.groq_models import GROQ_DEFAULT_MODEL

        adapter = GroqAdapter()
        assert adapter.model_name == GROQ_DEFAULT_MODEL
        adapter.load()
        try:
            try:
                result = adapter.generate(
                    "What is 2 + 2? Answer with just the number.",
                    max_tokens=512,
                )
            except Exception as exc:
                _skip_rate_limit(exc)
            assert result.text.strip(), (
                f"default model {adapter.model_name!r} returned no text"
            )
            assert "4" in result.text
            assert result.metadata["provider"] == "groq"
        finally:
            adapter.unload()

    def test_generate_small_chat_model(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("openai/gpt-oss-20b")
        adapter.load()
        try:
            try:
                result = adapter.generate("Respond with exactly: GROQ_OK", config=None)
            except Exception as exc:
                _skip_rate_limit(exc)
            assert result.text, "Expected non-empty response"
            assert result.tokens_used > 0
            assert result.metadata["provider"] == "groq"
        finally:
            adapter.unload()

    def test_generate_large_chat_model(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("openai/gpt-oss-120b")
        adapter.load()
        try:
            try:
                result = adapter.generate("What is 2 + 2? Answer with just the number.")
            except Exception as exc:
                _skip_rate_limit(exc)
            assert "4" in result.text
        finally:
            adapter.unload()

    def test_load_model_via_provider(self):
        from effgen.models import load_model

        model = load_model("openai/gpt-oss-20b", provider="groq")
        try:
            try:
                result = model.generate("Say hello in one word")
            except Exception as exc:
                _skip_rate_limit(exc)
            assert result.text
        finally:
            model.unload()

    def test_generate_stream_yields_chunks(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("openai/gpt-oss-20b")
        adapter.load()
        try:
            try:
                chunks = list(adapter.generate_stream("Count from 1 to 3 briefly."))
            except Exception as exc:
                _skip_rate_limit(exc)
            assert len(chunks) >= 1, "Expected at least one streaming chunk"
            full_text = "".join(chunks)
            assert len(full_text) > 0
        finally:
            adapter.unload()

    def test_native_tool_calling(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("openai/gpt-oss-120b")
        adapter.load()
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "Math expression"}},
                    "required": ["expression"],
                },
            },
        }]
        try:
            try:
                # Ask explicitly to use the tool: this proves the native
                # tool-call decode path, not whether the model bothers to
                # delegate trivial arithmetic it could do in its head.
                result = adapter.generate_with_tools(
                    "What is 17 * 23? Use the calculator tool.", tools
                )
            except Exception as exc:
                _skip_rate_limit(exc)
            assert len(result.metadata["tool_calls"]) >= 1
            tc = result.metadata["tool_calls"][0]
            assert tc["function"]["name"] == "calculator"
            # The reported arguments are the JSON string the model generated,
            # never a mapping the adapter parsed on the caller's behalf.
            assert isinstance(tc["function"]["arguments"], str)
        finally:
            adapter.unload()

    def test_usage_populated(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("openai/gpt-oss-20b")
        adapter.load()
        try:
            try:
                result = adapter.generate("Hi")
            except Exception as exc:
                _skip_rate_limit(exc)
            assert result.metadata["prompt_tokens"] > 0
            assert result.metadata["completion_tokens"] > 0
            assert result.metadata["total_tokens"] > 0
        finally:
            adapter.unload()
