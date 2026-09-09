"""
Unit tests for GroqAdapter (mocks OK for adapter plumbing).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.groq_adapter import GroqAdapter
from effgen.models.groq_models import (
    GROQ_DEFAULT_MODEL,
    GROQ_MODELS,
    available_models,
    chat_models,
    model_info,
    tool_capable_models,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestGroqModelsRegistry:
    def test_available_models_nonempty(self):
        assert len(available_models()) > 0

    def test_chat_models_subset(self):
        assert set(chat_models()).issubset(set(available_models()))

    def test_tool_capable_subset_of_chat(self):
        assert set(tool_capable_models()).issubset(set(chat_models()))

    def test_model_info_known(self):
        info = model_info("openai/gpt-oss-20b")
        assert info["context"] == 131_072
        assert info["supports_native_tools"] is True

    def test_model_info_unknown_raises(self):
        with pytest.raises(KeyError):
            model_info("nonexistent-model-xyz")

    def test_default_model_in_registry(self):
        assert GROQ_DEFAULT_MODEL in GROQ_MODELS

    def test_all_chat_models_have_required_fields(self):
        required = {"context", "max_output", "supports_native_tools", "supports_streaming"}
        for model_id, info in GROQ_MODELS.items():
            if info.get("modality") == "chat":
                missing = required - set(info.keys())
                assert not missing, f"{model_id} missing fields: {missing}"

    def test_known_tool_capable_models(self):
        capable = tool_capable_models()
        assert "openai/gpt-oss-120b" in capable
        assert "openai/gpt-oss-20b" in capable
        assert "qwen/qwen3.6-27b" in capable

    def test_guard_models_no_tools(self):
        for model_id in ["meta-llama/llama-prompt-guard-2-22m", "meta-llama/llama-prompt-guard-2-86m"]:
            assert GROQ_MODELS[model_id]["supports_native_tools"] is False

    def test_stt_models_not_in_chat(self):
        chat = chat_models()
        assert "whisper-large-v3" not in chat
        assert "whisper-large-v3-turbo" not in chat

    def test_reasoning_models_are_flagged(self):
        """Groq's reasoning families carry the flag that earns the larger budget."""
        for model_id in (
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
        ):
            assert GROQ_MODELS[model_id].get("reasoning") is True, model_id
        assert not GROQ_MODELS["groq/compound-mini"].get("reasoning")

    def test_only_one_vision_model(self):
        vision = [k for k, v in GROQ_MODELS.items() if v.get("supports_vision")]
        assert vision == ["qwen/qwen3.6-27b"], vision


# ---------------------------------------------------------------------------
# GroqAdapter unit tests (mocked)
# ---------------------------------------------------------------------------

class TestGroqAdapterInit:
    def test_unknown_model_raises(self):
        from effgen.models.errors import ModelNotFoundError
        with pytest.raises(ModelNotFoundError, match="Unknown Groq model"):
            GroqAdapter("does-not-exist-model")

    def test_stt_model_raises(self):
        with pytest.raises(ValueError, match="stt model"):
            GroqAdapter("whisper-large-v3")

    def test_default_model_accepted(self):
        adapter = GroqAdapter()
        assert adapter.model_name == GROQ_DEFAULT_MODEL

    def test_rate_limiter_wired_by_default(self):
        adapter = GroqAdapter("openai/gpt-oss-20b")
        assert adapter._rate_limiter is not None

    def test_rate_limiter_disabled(self):
        adapter = GroqAdapter("openai/gpt-oss-20b", enable_rate_limiting=False)
        assert adapter._rate_limiter is None

    def test_context_length_before_load(self):
        adapter = GroqAdapter("openai/gpt-oss-120b")
        assert adapter.get_context_length() == 131_072


class TestGroqAdapterLoad:
    def test_load_no_key_raises(self):
        adapter = GroqAdapter("openai/gpt-oss-20b", api_key=None)
        # Stub SDK so the no-key path is reached even when groq isn't installed
        stub = MagicMock()
        stub.Groq = MagicMock()
        with patch.dict("sys.modules", {"groq": stub}):
            with patch.dict("os.environ", {}, clear=True):
                import os
                os.environ.pop("GROQ_API_KEY", None)
                with pytest.raises(ValueError, match="GROQ_API_KEY"):
                    adapter.load()

    def test_load_sets_is_loaded(self):
        with patch("effgen.models.groq_adapter.os.getenv", return_value="fake-key"):
            with patch("effgen.models.groq_adapter.GroqAdapter.load") as mock_load:
                mock_load.return_value = None
                adapter = GroqAdapter("openai/gpt-oss-20b", api_key="fake-key")
                # Manually set state as load() is mocked
                adapter._is_loaded = True
                assert adapter._is_loaded

    def test_import_error_on_missing_groq(self):
        adapter = GroqAdapter("openai/gpt-oss-20b", api_key="fake-key")
        with patch("builtins.__import__", side_effect=ImportError("No module named 'groq'")):
            with pytest.raises((ImportError, RuntimeError)):
                adapter.load()

    def test_unload_clears_client(self):
        adapter = GroqAdapter("openai/gpt-oss-20b", api_key="fake-key")
        adapter._client = MagicMock()
        adapter._is_loaded = True
        adapter.unload()
        assert adapter._client is None
        assert not adapter._is_loaded


class TestGroqAdapterGenerate:
    def _make_mock_response(self, text="Hello!", tool_calls=None):
        """Build a mock Groq API response."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = text
        mock_message.tool_calls = tool_calls
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop" if not tool_calls else "tool_calls"
        mock_response.choices = [mock_choice]
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        return mock_response

    def _loaded_adapter(self, model="openai/gpt-oss-20b"):
        adapter = GroqAdapter(model, api_key="fake-key", enable_rate_limiting=False, enable_cost_tracking=False)
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_generate_not_loaded_raises(self):
        adapter = GroqAdapter("openai/gpt-oss-20b", api_key="fake-key")
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("Hello")

    def test_generate_returns_text(self):
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Bonjour!")
        result = adapter.generate("Say hello in French")
        assert result.text == "Bonjour!"
        assert result.model_name == "openai/gpt-oss-20b"

    def test_generate_usage_populated(self):
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Hi")
        result = adapter.generate("Hi")
        assert result.metadata["prompt_tokens"] == 10
        assert result.metadata["completion_tokens"] == 5
        assert result.metadata["total_tokens"] == 15
        assert result.metadata["estimated_usage"] is False

    def test_zero_usage_response_is_estimated_not_zero(self):
        # Groq returns an all-zero usage object on some tool-call responses
        # even though the call was billed. The adapter estimates token counts
        # from the request and output instead of reporting a misleading zero.
        adapter = self._loaded_adapter("openai/gpt-oss-120b")
        tc = MagicMock()
        tc.id = "tc1"
        tc.type = "function"
        tc.function.name = "retrieval"
        tc.function.arguments = '{"query": "database backup schedule"}'
        resp = self._make_mock_response("", tool_calls=[tc])
        resp.usage.prompt_tokens = 0
        resp.usage.completion_tokens = 0
        resp.usage.total_tokens = 0
        adapter._client.chat.completions.create.return_value = resp
        tools = [{"type": "function", "function": {
            "name": "retrieval", "description": "search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
        result = adapter.generate_with_tools(
            "Find the database backup schedule in the knowledge base.", tools,
        )
        assert result.metadata["estimated_usage"] is True
        assert result.metadata["prompt_tokens"] > 0
        assert result.metadata["completion_tokens"] > 0
        assert result.metadata["total_tokens"] == (
            result.metadata["prompt_tokens"] + result.metadata["completion_tokens"]
        )

    def test_estimate_prompt_tokens_counts_messages_and_tools(self):
        adapter = self._loaded_adapter()
        req = {
            "messages": [
                {"role": "system", "content": "You are a retrieval assistant."},
                {"role": "user", "content": "What is the VPN hostname?"},
            ],
            "tools": [{"type": "function", "function": {"name": "retrieval"}}],
        }
        n = adapter._estimate_prompt_tokens(req)
        assert n > 0

    def test_generate_forwards_frequency_and_presence_penalty(self):
        """A pinned penalty must reach the Groq request, matching seed/top_p."""
        from effgen.models.base import GenerationConfig

        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Hi")
        adapter.generate(
            "Hi",
            config=GenerationConfig(frequency_penalty=1.5, presence_penalty=0.5),
        )
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["frequency_penalty"] == 1.5
        assert call_kwargs["presence_penalty"] == 0.5

    def test_generate_omits_default_penalties(self):
        """The neutral 0.0 default is not sent, matching the seed/top_p convention."""
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Hi")
        adapter.generate("Hi")
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs

    def test_reasoning_model_asks_for_a_parsed_reasoning_chain(self):
        """A reasoning family must not return its chain inside the answer.

        Groq's default ``reasoning_format`` is ``"raw"``, which embeds the chain
        in ``message.content`` between ``<think>`` tags — so ``result.text`` for
        a one-word question is several hundred tokens of reasoning and no answer,
        and a tool-calling turn can spend its whole budget thinking. Asking for
        ``"parsed"`` puts the chain on ``message.reasoning`` instead.
        """
        adapter = self._loaded_adapter("qwen/qwen3.6-27b")
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Paris")
        adapter.generate("What is the capital of France?")
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_format"] == "parsed"

    def test_reasoning_format_is_sent_on_the_streaming_path_too(self):
        adapter = self._loaded_adapter("openai/gpt-oss-20b")
        adapter._client.chat.completions.create.return_value = iter([])
        list(adapter.generate_stream("Hi"))
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_format"] == "parsed"

    def test_non_reasoning_model_is_not_sent_reasoning_format(self):
        """Groq rejects the parameter on families that do not reason."""
        adapter = self._loaded_adapter("groq/compound-mini")
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Hi")
        adapter.generate("Hi")
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert "reasoning_format" not in call_kwargs

    def test_caller_can_override_the_reasoning_format(self):
        """A caller who wants the raw chain back keeps that option."""
        adapter = self._loaded_adapter("qwen/qwen3.6-27b")
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Paris")
        adapter.generate("Hi", reasoning_format="raw")
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_format"] == "raw"

    def test_generate_with_tools_calls_api(self):
        adapter = self._loaded_adapter("openai/gpt-oss-120b")
        tc = MagicMock()
        tc.id = "tc1"
        tc.type = "function"
        tc.function.name = "calculator"
        tc.function.arguments = '{"expression": "2+2"}'
        mock_resp = self._make_mock_response("", tool_calls=[tc])
        adapter._client.chat.completions.create.return_value = mock_resp
        tools = [{"type": "function", "function": {"name": "calculator", "description": "calc",
                   "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}}]
        result = adapter.generate_with_tools("What is 2+2?", tools)
        assert len(result.metadata["tool_calls"]) == 1
        assert result.metadata["tool_calls"][0]["function"]["name"] == "calculator"

    def test_generate_with_tools_recovers_failed_generation_tool_call(self):
        adapter = self._loaded_adapter("openai/gpt-oss-120b")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Failed to call a function.', "
            "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator{\"expression\": \"2+2\"}</function>'}}"
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }]
        result = adapter.generate_with_tools("What is 2+2?", tools)
        tool_call = result.metadata["tool_calls"][0]
        assert tool_call["function"]["name"] == "calculator"
        assert json.loads(tool_call["function"]["arguments"]) == {"expression": "2+2"}

    def test_recovered_tool_call_logs_at_info_not_warning(self, caplog):
        """The tool_use_failed recovery is not actionable — it must log at
        INFO (shown only with --verbose) so a successful turn's default
        output doesn't carry a stray WARNING line."""
        import logging

        adapter = self._loaded_adapter("openai/gpt-oss-120b")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Failed to call a function.', "
            "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator{\"expression\": \"2+2\"}</function>'}}"
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }]
        with caplog.at_level(logging.INFO, logger="effgen.models.groq_adapter"):
            adapter.generate_with_tools("What is 2+2?", tools)
        recovery_records = [
            r for r in caplog.records if "tool_use_failed but included a parseable" in r.message
        ]
        assert recovery_records, "expected the recovery note to be logged"
        assert all(r.levelno == logging.INFO for r in recovery_records)
        assert not any(r.levelno >= logging.WARNING for r in recovery_records)

    def test_generate_with_tools_recovers_failed_generation_with_closing_bracket(self):
        adapter = self._loaded_adapter("openai/gpt-oss-120b")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Failed to call a function.', "
            "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator>"
            "{\"expression\": \"(17 * 23) + sqrt(144)\"}</function>'}}"
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }]
        result = adapter.generate_with_tools("What is (17 * 23) + sqrt(144)?", tools)
        tool_call = result.metadata["tool_calls"][0]
        assert tool_call["function"]["name"] == "calculator"
        assert json.loads(tool_call["function"]["arguments"]) == {
            "expression": "(17 * 23) + sqrt(144)"
        }

    def test_generate_with_tools_recovers_failed_generation_missing_closing_tag(self):
        """A denied/retried tool call sometimes comes back with no closing
        `</function>` tag at all, or two calls run together with no
        separator — the first call is still recovered rather than the
        whole turn failing with a raw provider error."""
        adapter = self._loaded_adapter("openai/gpt-oss-20b")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': \"Failed to call a function. "
            "Please adjust your prompt.\", 'type': 'invalid_request_error', "
            "'code': 'tool_use_failed', 'failed_generation': "
            '\'<function=issue_refund>{"order_id": "ORD-1001"} \\n\\n'
            "<function=order_lookup>{\"order_id\": \"ORD-1001\"}'}}"
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "issue_refund",
                    "description": "Refund an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "order_lookup",
                    "description": "Look up an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                    },
                },
            },
        ]
        result = adapter.generate_with_tools("Refund ORD-1001", tools)
        tool_call = result.metadata["tool_calls"][0]
        assert tool_call["function"]["name"] == "issue_refund"
        assert json.loads(tool_call["function"]["arguments"]) == {"order_id": "ORD-1001"}
        # Only the first call is recovered; a second concatenated call is not
        # silently executed too.
        assert len(result.metadata["tool_calls"]) == 1

    def test_generate_with_tools_recovers_failed_generation_no_tag_no_trailer(self):
        """No closing tag and nothing trailing (end of string) also recovers."""
        adapter = self._loaded_adapter("openai/gpt-oss-20b")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Failed to call a function.', "
            "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator>{\"expression\": \"2+2\"}'}}"
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }]
        result = adapter.generate_with_tools("What is 2+2?", tools)
        tool_call = result.metadata["tool_calls"][0]
        assert tool_call["function"]["name"] == "calculator"
        assert json.loads(tool_call["function"]["arguments"]) == {"expression": "2+2"}

    def test_count_tokens_returns_positive(self):
        adapter = self._loaded_adapter()
        tc = adapter.count_tokens("Hello world")
        assert tc.count > 0

    def test_supports_native_tools_property(self):
        adapter = self._loaded_adapter("openai/gpt-oss-120b")
        assert adapter.supports_native_tools is True
        adapter2 = self._loaded_adapter("allam-2-7b")
        assert adapter2.supports_native_tools is False

    def test_rate_limit_status_disabled(self):
        adapter = GroqAdapter("openai/gpt-oss-20b", api_key="fake-key", enable_rate_limiting=False)
        status = adapter.rate_limit_status()
        assert status["enabled"] is False

    def test_rate_limit_status_enabled(self):
        adapter = GroqAdapter("openai/gpt-oss-20b", api_key="fake-key", enable_rate_limiting=True)
        status = adapter.rate_limit_status()
        assert status["enabled"] is True

    def test_supports_tool_calling_method(self):
        adapter = self._loaded_adapter("openai/gpt-oss-120b")
        assert adapter.supports_tool_calling() is True
        assert adapter.supports_function_calling() is True
        adapter2 = self._loaded_adapter("allam-2-7b")
        assert adapter2.supports_tool_calling() is False

    def test_bad_key_raises_model_auth_error(self):
        from effgen.models.errors import ModelAuthError
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 401 - {'error': {'message': 'Invalid API Key', "
            "'code': 'invalid_api_key'}}"
        )
        with pytest.raises(ModelAuthError) as exc:
            adapter.generate("hi")
        assert exc.value.provider == "groq"
        assert "401" in str(exc.value)

    def test_request_too_large_raises_invalid_request_not_rate_limit(self):
        # Groq returns 413 with a rate_limit_exceeded code for a single oversized
        # request. It must classify as a non-retryable invalid request (not a
        # rate limit routed through failover), and the org id must be redacted.
        from effgen.models._rate_limit import RateLimitExceeded
        from effgen.models.errors import InvalidRequestError
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 413 - {'error': {'message': 'Request too large for "
            "model `openai/gpt-oss-20b` in organization `org_secret123` on "
            "tokens per minute (TPM): Limit 6000, Requested 9288, please "
            "reduce your message size and try again.', 'type': 'tokens', "
            "'code': 'rate_limit_exceeded'}}"
        )
        with pytest.raises(InvalidRequestError) as exc:
            adapter.generate("hi")
        assert not isinstance(exc.value, RateLimitExceeded)
        msg = str(exc.value)
        assert "org_secret123" not in msg
        assert "reduce" in msg.lower() or "larger-context" in msg.lower()


def test_is_request_too_large_helper():
    from effgen.models.groq_adapter import _is_request_too_large
    msg = "Error code: 413 - Request too large for model ..."
    assert _is_request_too_large(msg, msg.lower()) is True
    rate = "Error code: 429 - Rate limit reached for requests"
    assert _is_request_too_large(rate, rate.lower()) is False


def test_redact_groq_org_helper():
    from effgen.models.groq_adapter import _redact_groq_org
    msg = "Request too large in organization `org_01abcXYZ` service tier"
    out = _redact_groq_org(msg)
    assert "org_01abcXYZ" not in out
    assert "organization `***`" in out


# ---------------------------------------------------------------------------
# GroqAdapter.generate_stream — cost/token accounting
# ---------------------------------------------------------------------------

class _StreamChunk:
    """Mimics a Groq/OpenAI-compatible SSE chunk."""
    def __init__(self, content=None, finish_reason=None, usage=None):
        choice = MagicMock()
        choice.delta.content = content
        choice.delta.tool_calls = None
        choice.finish_reason = finish_reason
        self.choices = [choice] if content is not None or finish_reason else []
        self.usage = usage


class TestGroqAdapterStream:
    def _loaded_adapter(self, model="openai/gpt-oss-20b", enable_cost_tracking=True):
        adapter = GroqAdapter(
            model, api_key="fake-key", enable_rate_limiting=False,
            enable_cost_tracking=enable_cost_tracking,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def _usage_chunk(self, prompt_tokens=10, completion_tokens=5):
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        return usage

    def test_stream_yields_text_and_accumulates_total_cost(self):
        """A terminal usage-only chunk with empty `choices` must still be read,
        and its cost must fold into `total_cost` so `effgen chat`'s per-turn
        footer and `/cost` reflect the real spend."""
        adapter = self._loaded_adapter()
        chunks = [
            _StreamChunk(content="Hel"),
            _StreamChunk(content="lo"),
            _StreamChunk(finish_reason="stop"),
            # Terminal usage chunk with NO choices at all.
            _StreamChunk(usage=self._usage_chunk(10, 5)),
        ]
        adapter._client.chat.completions.create.return_value = iter(chunks)
        assert getattr(adapter, "total_cost", 0.0) == 0.0
        out = "".join(adapter.generate_stream("hi"))
        assert out == "Hello"
        assert adapter.total_cost > 0.0
        # The streamed turn's token count must also reach total_tokens so the
        # per-turn footer shows tokens, not just cost.
        assert adapter.total_tokens == 15

    def test_stream_with_disabled_cost_tracking_leaves_total_cost_unset(self):
        adapter = self._loaded_adapter(enable_cost_tracking=False)
        chunks = [
            _StreamChunk(content="Hi"),
            _StreamChunk(finish_reason="stop"),
            _StreamChunk(usage=self._usage_chunk(10, 5)),
        ]
        adapter._client.chat.completions.create.return_value = iter(chunks)
        list(adapter.generate_stream("hi"))
        assert getattr(adapter, "total_cost", 0.0) == 0.0

    def test_stream_with_no_usage_data_leaves_total_cost_unset(self):
        """A stream that never carries usage (e.g. no final usage chunk at
        all) must not fabricate a cost — total_cost stays unset."""
        adapter = self._loaded_adapter()
        chunks = [_StreamChunk(content="Hi"), _StreamChunk(finish_reason="stop")]
        adapter._client.chat.completions.create.return_value = iter(chunks)
        list(adapter.generate_stream("hi"))
        assert getattr(adapter, "total_cost", 0.0) == 0.0


class TestReActPathToolUseFailedRecovery:
    """The recovery must reach the one path that needs it.

    The ReAct strategy describes its tools in the prompt and sends no ``tools``
    array, so Groq applies ``tool_choice: "none"``. A gpt-oss model calls a tool
    anyway and Groq rejects the whole completion with a 400 whose body quotes
    the call the model wrote. The recovery was gated on ``tools`` being present,
    which is exactly what this path does not send, so the run failed on a call
    that was right there in the error.
    """

    def _loaded_adapter(self, model="openai/gpt-oss-20b"):
        from unittest.mock import MagicMock

        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter(model_name=model, api_key="k", enable_rate_limiting=False)
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    _ERROR = (
        "Error code: 400 - {'error': {'message': 'Tool choice is none, but model "
        "called a tool', 'type': 'invalid_request_error', 'code': 'tool_use_failed', "
        "'failed_generation': '{\"name\": \"calculator\", \"arguments\": "
        "{\"expression\":\"234 * 567\"}}'}}"
    )

    def test_a_json_failed_generation_is_read_as_a_call(self):
        from effgen.models.groq_adapter import _parse_failed_generation_tool_call

        call = _parse_failed_generation_tool_call(self._ERROR)
        assert call is not None
        assert call["function"]["name"] == "calculator"
        assert json.loads(call["function"]["arguments"]) == {"expression": "234 * 567"}

    def test_the_recovery_fires_when_the_request_advertised_no_tools(self):
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(self._ERROR)

        result = adapter.generate("Thought: I should calculate.\nAction: calculator")

        assert result.finish_reason == "tool_calls"
        call = result.metadata["tool_calls"][0]
        assert call["function"]["name"] == "calculator"
        assert json.loads(call["function"]["arguments"]) == {"expression": "234 * 567"}

    def test_an_unreadable_failed_generation_names_the_mode_to_switch_to(self):
        """When the call cannot be recovered the error says what happened."""
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Tool choice is none, but model "
            "called a tool', 'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': 'not a call at all'}}"
        )
        with pytest.raises(RuntimeError) as excinfo:
            adapter.generate("Action: calculator")
        message = str(excinfo.value)
        assert "called a tool in a request that advertised none" in message
        assert "native" in message and "react" in message
