"""Every adapter that says it reads a cache hit actually reports one.

Table-driven on purpose: an adapter added later that declares
``reports_cached_tokens`` and forgets to read the field fails here rather than
reporting a permanent zero that looks exactly like a provider with no cache.

Offline: each adapter's own usage-reading seam is called with a scripted payload
in the shape its provider sends. No key is read and no request is made.
"""

from __future__ import annotations

from typing import Any

import pytest

from effgen.models._usage import cached_prompt_tokens
from effgen.models.base import PromptCachePolicy, get_stream_usage


class _Details:
    def __init__(self, cached: int) -> None:
        self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt: int, completion: int, cached: int | None) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.prompt_tokens_details = None if cached is None else _Details(cached)


def test_the_shared_reader_handles_every_shape_a_provider_sends():
    assert cached_prompt_tokens(_Usage(100, 10, 64)) == 64
    assert cached_prompt_tokens(_Usage(100, 10, None)) == 0
    assert cached_prompt_tokens(_Usage(100, 10, 0)) == 0
    assert cached_prompt_tokens({"prompt_tokens_details": {"cached_tokens": 32}}) == 32
    assert cached_prompt_tokens(object()) == 0
    assert cached_prompt_tokens(_Usage(100, 10, -5)) == 0


# --------------------------------------------------------------------------
# The OpenAI-protocol adapters, through the code each one really runs.
# --------------------------------------------------------------------------

#: Each provider that speaks the OpenAI protocol, with a model id its own
#: catalog carries — the adapters refuse an id they do not know, and this is
#: about usage accounting, not about catalogs.
OPENAI_PROTOCOL = (
    ("groq", "openai/gpt-oss-20b"),
    ("together", "openai/gpt-oss-20b"),
    ("fireworks", "gpt-oss-120b"),
    ("cerebras", "gpt-oss-120b"),
)
PROVIDERS = [name for name, _ in OPENAI_PROTOCOL]


def _adapter(provider: str) -> Any:
    import importlib

    model = dict(OPENAI_PROTOCOL)[provider]
    module = importlib.import_module(f"effgen.models.{provider}_adapter")
    cls = next(
        obj for name, obj in vars(module).items()
        if name.endswith("Adapter") and getattr(obj, "__module__", "") == module.__name__
    )
    adapter = cls(model_name=model, api_key="stub-not-a-key")
    adapter._enable_cost_tracking = False
    return adapter


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_adapter_declares_that_it_reads_a_hit(provider):
    policy = _adapter(provider).prompt_cache_policy()
    assert isinstance(policy, PromptCachePolicy)
    assert policy.reports_cached_tokens


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_blocking_call_carries_the_hit_into_its_metadata(provider, monkeypatch):
    """The provider reported 64 cached tokens; the result says 64."""
    adapter = _adapter(provider)
    captured: dict[str, Any] = {}

    class _Message:
        content = "the answer"
        tool_calls = None

    class _Choice:
        index = 0
        message = _Message()
        finish_reason = "stop"

    class _Response:
        choices = [_Choice()]
        usage = _Usage(1000, 10, 640)

    def _fake_create(**params: Any) -> Any:
        captured["params"] = params
        return _Response()

    from effgen.models.base import GenerationConfig

    adapter._client = type("C", (), {
        "chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(_fake_create)})()
        })()
    })()
    result = adapter._do_generate(  # noqa: SLF001 - the seam under test
        "hi", GenerationConfig(max_tokens=32),
    )
    assert captured["params"]["model"] == adapter.model_name
    assert result.metadata["cached_input_tokens"] == 640
    assert result.metadata["prompt_tokens"] == 1000


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_streamed_call_carries_the_hit_into_its_usage(provider):
    """A streamed turn reports the hit exactly as a blocking one does."""
    from effgen.models.base import accumulate_stream_cost

    adapter = _adapter(provider)
    accumulate_stream_cost(
        adapter, 0.001, 1010, prompt_tokens=1000, completion_tokens=10,
        cached_input_tokens=640,
    )
    usage = get_stream_usage(adapter)
    assert usage is not None
    assert usage["cached_input_tokens"] == 640


def test_gemini_reads_its_own_field_on_both_paths():
    from effgen.models.gemini_adapter import GeminiAdapter

    class _Meta:
        prompt_token_count = 7334
        candidates_token_count = 120
        total_token_count = 7454
        cached_content_token_count = 7146

    assert GeminiAdapter._cached_content_tokens(_Meta()) == 7146
    assert GeminiAdapter._cached_content_tokens(object()) == 0
    stub: Any = GeminiAdapter.__new__(GeminiAdapter)
    stub.model_name = "gemini-2.5-flash"
    assert GeminiAdapter.prompt_cache_policy(stub).reports_cached_tokens


def test_anthropic_reports_the_whole_input_and_both_cache_halves():
    """``input_tokens`` excludes the cache, so the prompt count adds it back."""
    from effgen.models.anthropic_adapter import AnthropicAdapter

    class _Usage2:
        input_tokens = 200
        output_tokens = 40
        cache_read_input_tokens = 7000
        cache_creation_input_tokens = 300

    class _Response2:
        usage = _Usage2()

    stub: Any = AnthropicAdapter.__new__(AnthropicAdapter)
    stub.model_name = "claude-sonnet-4-6"
    prompt, completion, cached, written = AnthropicAdapter._parse_usage(stub, _Response2())
    assert (prompt, completion, cached, written) == (7500, 40, 7000, 300)
    policy = AnthropicAdapter.prompt_cache_policy(stub)
    assert policy.reports_cached_tokens and policy.reports_cache_writes


def test_fireworks_keeps_the_cached_count_its_sdk_would_discard(monkeypatch):
    """The hit is on the wire; it has to survive being parsed.

    This provider states the cache hit in ``usage.prompt_tokens_details``, and a
    live session recorded 51,131 of 56,438 prompt tokens served from its cache
    while every call reported 0 — the SDK's usage model declares three fields
    and drops the rest before any adapter sees it. Loading the adapter widens
    that one model, so the number the provider sent is the number the run reads.
    """
    pytest.importorskip("fireworks.client")
    from fireworks.client.api import UsageInfo

    from effgen.models.fireworks_adapter import FireworksAdapter

    monkeypatch.setenv("FIREWORKS_API_KEY", "stub-not-a-key")
    monkeypatch.setattr(
        "fireworks.client.Fireworks",
        lambda **kwargs: object(),
        raising=False,
    )
    adapter = FireworksAdapter(model_name="gpt-oss-120b", api_key="stub-not-a-key")
    adapter.load()

    # The response the provider sends, parsed by the SDK exactly as a live call
    # parses it: a response model compiles its own validator from the usage
    # model, so the usage model alone being widened is not enough.
    from fireworks.client.api import ChatCompletionResponse, ChatCompletionStreamResponse

    usage = {"prompt_tokens": 4647, "completion_tokens": 30, "total_tokens": 4677,
             "prompt_tokens_details": {"cached_tokens": 4584}}
    assert cached_prompt_tokens(UsageInfo(**usage)) == 4584
    blocking = ChatCompletionResponse.model_validate({
        "id": "r", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hi"}}],
        "usage": usage,
    })
    assert cached_prompt_tokens(blocking.usage) == 4584
    streamed = ChatCompletionStreamResponse.model_validate({
        "id": "r", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "hi"}}],
        "usage": usage,
    })
    assert cached_prompt_tokens(streamed.usage) == 4584
