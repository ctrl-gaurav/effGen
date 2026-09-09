"""Token counting on a machine that cannot reach tiktoken's data host.

tiktoken ships no BPE data. The first call for an encoding downloads it from a
blob host into a cache under the temporary directory, so an air-gapped
deployment, a runner behind a proxy that does not allow that host, or a
container started with an empty cache has no encoding to count with.

That download used to raise out of ``count_tokens``, and — through the
pre-flight prompt check every adapter runs — out of ``generate()`` itself: a
request the provider would have answered failed before it was sent, reporting a
name-resolution failure for a host unrelated to the provider.

A token count is an estimate. What is pinned here is that an encoding which
cannot be loaded degrades to a length-based estimate, that the estimate is a
usable number rather than zero, that a generate call still reaches the provider,
and that the machine is told once rather than on every call.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from effgen.models import _adapter_utils


class _NoDataHost(Exception):
    """Stands in for the connection error tiktoken raises with no route out."""


@pytest.fixture
def unreachable_bpe(monkeypatch):
    """Make every tiktoken encoding lookup fail the way an absent route does."""
    tiktoken = pytest.importorskip("tiktoken")
    monkeypatch.setattr(_adapter_utils, "_bpe_encodings", {})
    monkeypatch.setattr(_adapter_utils, "_bpe_unavailable_warned", set())

    calls: list[str] = []

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        raise _NoDataHost("could not resolve the host that serves the BPE data")

    monkeypatch.setattr(tiktoken, "get_encoding", refuse)
    monkeypatch.setattr(tiktoken, "encoding_for_model", refuse)
    return calls


# --------------------------------------------------------------------------- #
# The shared estimator
# --------------------------------------------------------------------------- #
class TestTheEstimatorDegrades:
    def test_an_unreachable_encoding_reads_as_absent(self, unreachable_bpe):
        assert _adapter_utils.get_bpe_encoding() is None

    def test_a_count_is_still_produced(self, unreachable_bpe):
        assert _adapter_utils.estimate_tokens("The capital of France is Paris.") > 0

    def test_empty_text_is_zero_tokens(self, unreachable_bpe):
        assert _adapter_utils.estimate_tokens("") == 0

    def test_text_shorter_than_one_token_still_counts_as_one(self, unreachable_bpe):
        assert _adapter_utils.estimate_tokens("hi") == 1

    def test_the_download_is_attempted_once_per_encoding(self, unreachable_bpe):
        for _ in range(5):
            _adapter_utils.estimate_tokens("some text to count")
        assert len(unreachable_bpe) == 1

    def test_the_reason_is_reported_once(self, unreachable_bpe, caplog):
        with caplog.at_level(logging.INFO, logger="effgen.models._adapter_utils"):
            for _ in range(5):
                _adapter_utils.estimate_tokens("some text to count")
        said = [r for r in caplog.records if "estimated from text length" in r.getMessage()]
        assert len(said) == 1

    def test_the_reason_names_what_to_do_about_it(self, unreachable_bpe, caplog):
        with caplog.at_level(logging.INFO, logger="effgen.models._adapter_utils"):
            _adapter_utils.estimate_tokens("some text to count")
        message = caplog.records[-1].getMessage()
        assert "cl100k_base" in message
        assert "tiktoken cache" in message

    def test_it_is_reported_below_warning(self, unreachable_bpe, caplog):
        """A working estimate is not a warning — it must not colour a normal run."""
        with caplog.at_level(logging.INFO, logger="effgen.models._adapter_utils"):
            _adapter_utils.estimate_tokens("some text to count")
        assert all(r.levelno < logging.WARNING for r in caplog.records)


# --------------------------------------------------------------------------- #
# Every adapter that counts with tiktoken
# --------------------------------------------------------------------------- #
def _openai(monkeypatch):
    from effgen.models.openai_adapter import OpenAIAdapter

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    return OpenAIAdapter(model_name="gpt-4o-mini")


def _cerebras(monkeypatch):
    from effgen.models.cerebras_adapter import CerebrasAdapter

    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key-not-used")
    return CerebrasAdapter()


def _groq(monkeypatch):
    from effgen.models.groq_adapter import GroqAdapter

    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    return GroqAdapter(model_name="openai/gpt-oss-20b")


def _together(monkeypatch):
    from effgen.models.together_adapter import TogetherAdapter

    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")
    return TogetherAdapter()


def _fireworks(monkeypatch):
    from effgen.models.fireworks_adapter import FireworksAdapter

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key-not-used")
    return FireworksAdapter()


ADAPTERS = {
    "openai": _openai,
    "cerebras": _cerebras,
    "groq": _groq,
    "together": _together,
    "fireworks": _fireworks,
}


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
def test_count_tokens_reports_a_number_without_the_bpe_data(
    provider, unreachable_bpe, monkeypatch
):
    adapter = ADAPTERS[provider](monkeypatch)
    result = adapter.count_tokens("The capital of France is Paris.")
    assert result.count > 0
    assert result.model_name == adapter.model_name


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
def test_count_tokens_never_raises_the_download_failure(
    provider, unreachable_bpe, monkeypatch
):
    adapter = ADAPTERS[provider](monkeypatch)
    adapter.count_tokens("x" * 4000)  # must not raise


def test_a_generate_call_still_reaches_the_provider(unreachable_bpe, monkeypatch):
    """The pre-flight prompt check counts tokens; it must not fail the call."""
    adapter = _openai(monkeypatch)

    message = MagicMock()
    message.content = "Paris."
    message.tool_calls = None
    message.reasoning = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(
        prompt_tokens=8, completion_tokens=2, total_tokens=10,
        prompt_tokens_details=None, completion_tokens_details=None,
    )
    response.model = "gpt-4o-mini"

    client = MagicMock()
    client.chat.completions.create.return_value = response
    adapter.client = client
    adapter._is_loaded = True

    result = adapter.generate("What is the capital of France?")
    assert result.text == "Paris."
    assert client.chat.completions.create.called
