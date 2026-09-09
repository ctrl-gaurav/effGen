"""One client request must not become a dozen upstream requests.

A rate limit is the one failure where retrying harder makes the problem worse:
the quota is already spent, and every extra request spends it further. Three
layers each retried the same refusal and multiplied rather than shared a budget
— the provider SDK, the adapter's own loop, and the agent's generation loop. One
``POST /v1/chat/completions`` produced 12 requests at the upstream and held the
client for 20.5s at a *stated* 2s delay; at the delays providers really state
(groq 17s, Gemini 16s) the same shape holds a client for minutes.

The decision, made once: **the layer that knows the provider's stated delay owns
the retry, and no other layer retries a rate limit.** That is the adapter's own
backoff loop where it has one (groq, together, cerebras, fireworks) and the
SDK's where it does not (openai). The agent no longer re-retries a
``rate_limited`` classification; every other retryable class is unchanged.
"""
from __future__ import annotations

import os
import time

import pytest

from tests.unit.failure_injection import FaultServer

pytestmark = pytest.mark.timeout(120)


def test_a_rate_limited_call_is_retried_by_one_layer_only():
    """Measured end to end against a peer that refuses everything with 429."""
    server = FaultServer("rate_limit", "openai", retry_after_s=2).start()
    saved_key = os.environ.get("OPENAI_API_KEY")
    saved_base = os.environ.get("OPENAI_BASE_URL")
    try:
        os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
        os.environ["OPENAI_BASE_URL"] = server.url
        from effgen.core.agent import Agent, AgentConfig

        started = time.time()
        with Agent(AgentConfig(
            name="amp", model="openai:gpt-4o-mini", tools=[],
            raise_on_error=False, enable_memory=False, enable_sub_agents=False,
        )) as agent:
            response = agent.run("hi")
        held = time.time() - started
    finally:
        server.stop()
        os.environ.pop("OPENAI_BASE_URL", None)
        if saved_base is not None:
            os.environ["OPENAI_BASE_URL"] = saved_base
        if saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = saved_key

    assert response.success is False
    # One layer of retry: the SDK's, which honours the stated Retry-After.
    # Twelve was the measured amplification before the agent stopped retrying.
    assert server.call_count <= 5, (
        f"{server.call_count} upstream requests for one client request — "
        "more than one layer is retrying the rate limit"
    )
    assert held < 12, f"held the client {held:.1f}s at a stated 2s delay"


def test_the_agent_still_retries_a_transient_failure():
    """Only the rate-limit class stopped being retried at the agent layer."""
    from effgen.models.errors import classify_provider_error

    assert classify_provider_error(ConnectionError("connection reset")).should_retry
    assert classify_provider_error(TimeoutError("timed out")).should_retry


@pytest.mark.parametrize(
    "provider,module,cls,sdk_module,sdk_name,model",
    [
        ("groq", "effgen.models.groq_adapter", "GroqAdapter",
         "groq", "Groq", "openai/gpt-oss-20b"),
        ("together", "effgen.models.together_adapter", "TogetherAdapter",
         "together", "Together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
        ("cerebras", "effgen.models.cerebras_adapter", "CerebrasAdapter",
         "cerebras.cloud.sdk", "Cerebras", "gpt-oss-120b"),
    ],
)
def test_an_adapter_with_its_own_loop_switches_the_sdk_retry_off(
    provider, module, cls, sdk_module, sdk_name, model, monkeypatch
):
    """Otherwise the two layers multiply instead of sharing the budget."""
    import importlib
    from unittest.mock import MagicMock

    sdk = pytest.importorskip(sdk_module)
    adapter_cls = getattr(importlib.import_module(module), cls)
    seen: dict = {}

    def _fake_sdk(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(sdk, sdk_name, _fake_sdk)
    adapter = adapter_cls(model_name=model, api_key="k", enable_rate_limiting=False)
    adapter.load()
    assert seen.get("max_retries") == 0, (
        f"{provider} still lets its SDK retry beneath the adapter's own loop"
    )
