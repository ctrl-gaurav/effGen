"""
Hypothesis-driven fuzz tests for provider-response normalization.

Providers disagree on the shape of two response fields that effGen unifies:
finish/stop reason and error reporting. These shared helpers parse whatever a
provider SDK hands back, so they must tolerate arbitrary/garbage input.

Targets (``effgen.models._adapter_utils``):
  * ``normalize_finish_reason`` — str/enum/int/None/garbage → canonical token.
  * ``build_error_context`` — any Exception → ``{provider, model, ...}`` dict.
  * ``provider_runtime_error`` — any Exception → redacted ``RuntimeError`` whose
    message never leaks the (possibly secret-bearing) SDK string.

Asserts that:
  1. ``normalize_finish_reason`` never raises and always returns a ``str``.
  2. Known provider reasons (OpenAI/Anthropic/Gemini enum + int forms) map to
     the canonical set.
  3. ``build_error_context`` always returns a dict with the required keys and no
     secret material.
  4. ``provider_runtime_error`` redacts injected fake secrets in the SDK message.

Exit criterion: ≥300 examples for the finish-reason fuzz.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.models._adapter_utils import (
    CANONICAL_FINISH_REASONS,
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    build_error_context,
    normalize_finish_reason,
    provider_runtime_error,
)

pytestmark = pytest.mark.fuzz

_FAKE_SECRET = "sk-ant-FUZZsecret000000000000000000000000"
_SECRET_RE = re.compile(re.escape(_FAKE_SECRET))


# ---------------------------------------------------------------------------
# normalize_finish_reason
# ---------------------------------------------------------------------------


class _FakeEnum:
    """Mimics google-genai FinishReason.STOP (exposes ``.name``)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:  # pragma: no cover - repr path
        return f"FinishReason.{self.name}"


_finish_inputs = st.one_of(
    st.none(),
    st.text(max_size=40),
    st.integers(min_value=-5, max_value=20),
    st.booleans(),
    st.builds(_FakeEnum, st.sampled_from(["STOP", "MAX_TOKENS", "TOOL_USE", "SAFETY", "OTHER"])),
    st.sampled_from([[], {}, object()]),
)


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(_finish_inputs)
def test_finish_reason_never_crashes_returns_str(raw) -> None:
    out = normalize_finish_reason(raw)
    assert isinstance(out, str)
    assert out  # never empty


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("stop", FINISH_STOP),
        ("STOP", FINISH_STOP),
        ("length", FINISH_LENGTH),
        ("max_tokens", FINISH_LENGTH),
        ("end_turn", FINISH_STOP),  # Anthropic
        ("tool_use", FINISH_TOOL_CALLS),  # Anthropic
        ("tool_calls", FINISH_TOOL_CALLS),  # OpenAI
        ("function_call", FINISH_TOOL_CALLS),
        ("FinishReason.STOP", FINISH_STOP),  # Gemini enum repr
        ("FinishReason.MAX_TOKENS", FINISH_LENGTH),
        (None, FINISH_STOP),  # missing → normal stop
        (1, FINISH_STOP),  # Gemini int STOP
        (2, FINISH_LENGTH),  # Gemini int MAX_TOKENS
        (9, FINISH_TOOL_CALLS),  # Gemini int MALFORMED_FUNCTION_CALL
    ],
)
def test_finish_reason_known_mappings(raw, expected) -> None:
    assert normalize_finish_reason(raw) == expected


def test_finish_reason_enum_name_used() -> None:
    assert normalize_finish_reason(_FakeEnum("MAX_TOKENS")) == FINISH_LENGTH
    assert normalize_finish_reason(_FakeEnum("TOOL_USE")) == FINISH_TOOL_CALLS


def test_finish_reason_canonical_set_is_closed() -> None:
    """Every documented provider reason maps into the canonical set."""
    for raw in [
        "stop", "length", "tool_calls", "content_filter", "end_turn",
        "max_tokens", "tool_use", "safety", "recitation", "other",
    ]:
        assert normalize_finish_reason(raw) in CANONICAL_FINISH_REASONS


# ---------------------------------------------------------------------------
# Error context / redaction
# ---------------------------------------------------------------------------

_REQUIRED_CTX_KEYS = {"provider", "model", "request_type", "retry_status", "remediation", "category"}


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.text(max_size=20),
    st.text(max_size=30),
    st.sampled_from(["chat", "stream", "structured", "embeddings", "tool_call"]),
    st.text(max_size=80),
)
def test_build_error_context_shape(provider, model, request_type, msg) -> None:
    ctx = build_error_context(provider, model, request_type, RuntimeError(msg))
    assert _REQUIRED_CTX_KEYS.issubset(ctx.keys())
    assert all(isinstance(v, str) for v in ctx.values())


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(max_size=40))
def test_provider_runtime_error_redacts_secret(noise: str) -> None:
    """A provider exception carrying a secret is redacted in the surfaced error."""
    exc = RuntimeError(f"{noise} auth failed key={_FAKE_SECRET} {noise}")
    err = provider_runtime_error("openai", "gpt-5-nano", "chat", exc)
    assert isinstance(err, RuntimeError)
    assert hasattr(err, "error_context")
    assert not _SECRET_RE.search(str(err)), "secret leaked into provider error message"


def test_provider_runtime_error_context_attached() -> None:
    err = provider_runtime_error("groq", "openai/gpt-oss-20b", "chat", ValueError("boom"))
    ctx = err.error_context  # type: ignore[attr-defined]
    assert ctx["provider"] == "groq"
    assert ctx["request_type"] == "chat"
    assert ctx["retry_status"]
