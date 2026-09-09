"""One multi-turn tool loop, written once, works on every adapter.

``metadata["tool_calls"]`` has one shape everywhere, so *reading* a call is
portable. Re-submitting the turn was not: the shipped example appended
``metadata["message"]`` — written by the OpenAI adapter and by no other, so the
same loop raised ``KeyError: 'message'`` on groq, cerebras, together, fireworks
and hf — and a ``{"role": "tool", ...}`` message, which is the OpenAI wire
format and neither Gemini's ``functionResponse`` parts nor Anthropic's
``tool_result`` blocks.

``build_assistant_message`` and ``build_tool_result_message`` give every adapter
a way to produce its *own* provider's shape, so the loop around the read is
portable too.
"""
from __future__ import annotations

import json

import pytest

from effgen.models.base import BaseModel, GenerationResult

CALL = {
    "id": "call_abc123",
    "type": "function",
    "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'},
}

ADAPTERS = {
    "openai": ("effgen.models.openai_adapter", "OpenAIAdapter", "gpt-4o-mini"),
    "groq": ("effgen.models.groq_adapter", "GroqAdapter", "openai/gpt-oss-20b"),
    "together": ("effgen.models.together_adapter", "TogetherAdapter",
                 "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    "fireworks": ("effgen.models.fireworks_adapter", "FireworksAdapter",
                  "accounts/fireworks/models/gpt-oss-120b"),
    "cerebras": ("effgen.models.cerebras_adapter", "CerebrasAdapter", "gpt-oss-120b"),
    "hf_inference": ("effgen.models.hf_inference_adapter", "HFInferenceAdapter",
                     "Qwen/Qwen2.5-7B-Instruct"),
    "gemini": ("effgen.models.gemini_adapter", "GeminiAdapter", "gemini-3.1-flash-lite"),
    "anthropic": ("effgen.models.anthropic_adapter", "AnthropicAdapter",
                  "claude-3-5-haiku-20241022"),
    "replicate": ("effgen.models.replicate_adapter", "ReplicateAdapter",
                  "meta/meta-llama-3-8b-instruct"),
}


def _adapter(provider):
    import importlib

    module, cls, model = ADAPTERS[provider]
    adapter_cls = getattr(importlib.import_module(module), cls)
    try:
        return adapter_cls(model_name=model, api_key="k", enable_rate_limiting=False)
    except TypeError:
        return adapter_cls(model_name=model, api_key="k")


def _result():
    return GenerationResult(
        text="", tokens_used=5, finish_reason="tool_calls", model_name="m",
        metadata={"tool_calls": [CALL]},
    )


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
def test_every_adapter_can_build_both_messages(provider):
    """The loop compiles on every provider — no ``KeyError: 'message'``."""
    adapter = _adapter(provider)
    assistant = adapter.build_assistant_message(_result())
    tool_result = adapter.build_tool_result_message("call_abc123", "calculator", "42")
    assert isinstance(assistant, dict) and assistant.get("role")
    assert isinstance(tool_result, dict) and tool_result.get("role")


@pytest.mark.parametrize(
    "provider",
    ["openai", "groq", "together", "fireworks", "cerebras", "hf_inference", "replicate"],
)
def test_the_openai_protocol_adapters_share_one_shape(provider):
    adapter = _adapter(provider)
    assistant = adapter.build_assistant_message(_result())
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"] == [CALL]

    tool_result = adapter.build_tool_result_message("call_abc123", "calculator", "42")
    assert tool_result == {
        "role": "tool", "tool_call_id": "call_abc123", "content": "42",
    }


def test_a_provider_native_message_is_preferred_when_the_adapter_kept_one():
    """``metadata["message"]`` round-trips fields the uniform shape drops."""
    adapter = _adapter("openai")
    native = {"role": "assistant", "content": None, "tool_calls": [CALL], "extra": 1}
    result = GenerationResult(
        text="", tokens_used=1, finish_reason="tool_calls", model_name="m",
        metadata={"tool_calls": [CALL], "message": native},
    )
    assert adapter.build_assistant_message(result) is native


def test_gemini_speaks_parts_not_messages():
    adapter = _adapter("gemini")
    assistant = adapter.build_assistant_message(_result())
    assert assistant["role"] == "model"
    call_part = assistant["parts"][-1]["functionCall"]
    assert call_part["name"] == "calculator"
    # Parsed back into an object: Gemini does not take the JSON string.
    assert call_part["args"] == {"expression": "6*7"}

    tool_result = adapter.build_tool_result_message("call_abc123", "calculator", "42")
    response = tool_result["parts"][0]["functionResponse"]
    assert tool_result["role"] == "user"
    assert response["name"] == "calculator"
    assert response["response"] == {"result": "42"}


def test_anthropic_speaks_tool_result_blocks():
    adapter = _adapter("anthropic")
    tool_result = adapter.build_tool_result_message("toolu_1", "calculator", "42")
    assert tool_result["role"] == "user"
    block = tool_result["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_1"
    assert block["content"] == "42"


def test_the_documented_loop_is_provider_agnostic():
    """The loop from the example, written once and run against every adapter."""
    for provider in sorted(ADAPTERS):
        adapter = _adapter(provider)
        result = _result()
        conversation: list = [{"role": "user", "content": "What is 6*7?"}]
        conversation.append(adapter.build_assistant_message(result))
        for call in result.metadata["tool_calls"]:
            function = call["function"]
            arguments = json.loads(function["arguments"])
            answer = str(eval(arguments["expression"]))  # noqa: S307 - fixed literal
            conversation.append(
                adapter.build_tool_result_message(call["id"], function["name"], answer)
            )
        assert len(conversation) == 3, provider
        assert all(isinstance(m, dict) and m.get("role") for m in conversation), provider


def test_the_methods_are_on_the_base_class_so_a_new_adapter_inherits_them():
    assert callable(BaseModel.build_assistant_message)
    assert callable(BaseModel.build_tool_result_message)
