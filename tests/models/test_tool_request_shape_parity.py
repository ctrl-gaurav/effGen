"""A streamed tool request must be shaped like a non-streamed one.

``generate()`` normalized the tool definitions and set ``tool_choice="auto"``
behind the catalog's ``supports_native_tools`` gate. ``generate_stream()`` had
neither: the caller's ``tools=`` keyword reached the provider through
``request_params.update(kwargs)`` exactly as passed. So the same agent, the same
turn and the same definitions produced a different request depending on whether
the turn streamed.

The gate and ``tool_choice`` are one decision — the gate decides which models
are offered definitions at all — so both live in one helper that both paths
call, which is the end state the report proposes.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

TOOLS = [{
    "name": "calculator",
    "description": "Evaluate an expression",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}]

CASES = [
    ("groq", "effgen.models.groq_adapter", "GroqAdapter", "openai/gpt-oss-20b"),
    ("together", "effgen.models.together_adapter", "TogetherAdapter",
     "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    ("fireworks", "effgen.models.fireworks_adapter", "FireworksAdapter",
     "accounts/fireworks/models/gpt-oss-120b"),
]


def _adapter(module, cls, model):
    import importlib

    adapter_cls = getattr(importlib.import_module(module), cls)
    adapter = adapter_cls(model_name=model, api_key="k", enable_rate_limiting=False)
    adapter._client = MagicMock()
    adapter._is_loaded = True
    return adapter


def _captured_request(adapter, streaming: bool):
    create = adapter._client.chat.completions.create
    if streaming:
        create.return_value = iter(())
        list(adapter.generate_stream("hi", tools=TOOLS))
    else:
        message = MagicMock()
        message.content = "ok"
        message.tool_calls = None
        message.reasoning = None
        message.reasoning_content = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        create.return_value = response
        adapter.generate_with_tools("hi", TOOLS)
    return create.call_args.kwargs


@pytest.mark.parametrize("provider,module,cls,model", CASES)
def test_both_paths_send_the_same_tool_shape(provider, module, cls, model):
    streamed = _captured_request(_adapter(module, cls, model), streaming=True)
    blocking = _captured_request(_adapter(module, cls, model), streaming=False)

    assert streamed.get("tool_choice") == "auto", (
        f"{provider} streams without tool_choice; the two paths disagree"
    )
    assert streamed.get("tools") == blocking.get("tools"), (
        f"{provider} normalizes the definitions on one path only"
    )
    assert blocking.get("tool_choice") == streamed.get("tool_choice")


def test_the_catalog_gate_applies_to_both_paths():
    """A model the catalog says has no native tools is offered none, either way."""
    from effgen.models._adapter_utils import apply_tool_request

    ungated: dict = {}
    apply_tool_request(ungated, TOOLS, {"supports_native_tools": False})
    assert ungated == {}

    unknown: dict = {}
    apply_tool_request(unknown, TOOLS, None)
    assert unknown == {}

    gated: dict = {}
    apply_tool_request(gated, TOOLS, {"supports_native_tools": True})
    assert gated["tool_choice"] == "auto"
    assert gated["tools"][0]["type"] == "function"
    assert gated["tools"][0]["function"]["name"] == "calculator"


def test_no_tools_leaves_the_request_untouched():
    from effgen.models._adapter_utils import apply_tool_request

    request: dict = {"model": "m"}
    apply_tool_request(request, None, {"supports_native_tools": True})
    assert request == {"model": "m"}
