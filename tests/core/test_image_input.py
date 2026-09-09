"""
Unit tests for image input in provider adapters.

Tests adapter translation logic with fakes (no live API calls).
"""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock

import pytest

from effgen.core.messages import Message, Role, TextPart
from effgen.core.multimodal import image_from
from effgen.errors import CapabilityNotSupportedError
from effgen.models.capabilities import Capability

# ---------------------------------------------------------------------------
# Helper: create a minimal JPEG image
# ---------------------------------------------------------------------------

def _fake_jpeg(width: int = 50, height: int = 50) -> bytes:
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xe0" + b"\x00" * 200


def _image_message(text: str = "What's in this image?") -> Message:
    img_data = _fake_jpeg()
    img_part = image_from(img_data, mime="image/jpeg")
    return Message(
        role=Role.USER,
        content=[img_part, TextPart(text=text)],
    )


# ---------------------------------------------------------------------------
# OpenAI adapter translation
# ---------------------------------------------------------------------------

class TestOpenAIAdapterImageTranslation:
    def test_message_with_image_produces_image_url_content(self):
        from effgen.models.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.model_name = "gpt-4o-mini"

        msg = _image_message()
        result = adapter._message_to_openai(msg)

        assert result["role"] == "user"
        parts = result["content"]
        assert isinstance(parts, list)
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        url = image_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_text_only_message_simplifies_to_string(self):
        from effgen.models.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.model_name = "gpt-4o-mini"
        msg = Message(role=Role.USER, content=[TextPart(text="Hello")])
        result = adapter._message_to_openai(msg)
        assert result["content"] == "Hello"

    def test_create_messages_with_message_object(self):
        from effgen.models.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.model_name = "gpt-4o-mini"
        msg = _image_message()
        messages = adapter._create_messages(msg)
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_vision_unsupported_model_raises(self):
        from effgen.models.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.model_name = "gpt-3.5-turbo"
        adapter._is_loaded = True
        adapter._is_reasoning_model = False
        adapter._supports_sampling_params = True
        adapter.client = MagicMock()

        msg = _image_message()
        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            adapter.generate(msg)
        assert exc_info.value.capability == Capability.vision.value

    def test_stream_vision_unsupported_model_raises(self):
        from effgen.models.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.model_name = "gpt-3.5-turbo"
        adapter._is_loaded = True
        adapter._is_reasoning_model = False
        adapter._supports_sampling_params = True
        adapter.client = MagicMock()

        with pytest.raises(CapabilityNotSupportedError):
            next(adapter.generate_stream(_image_message()))


# ---------------------------------------------------------------------------
# Groq adapter translation
# ---------------------------------------------------------------------------

class TestGroqAdapterImageTranslation:
    def test_message_to_groq_produces_image_url(self):
        from effgen.models.groq_adapter import GroqAdapter

        msg = _image_message()
        result = GroqAdapter._message_to_groq(msg)

        assert result["role"] == "user"
        parts = result["content"]
        assert isinstance(parts, list)
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        url = image_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/")

    def test_text_only_simplifies(self):
        from effgen.models.groq_adapter import GroqAdapter

        msg = Message(role=Role.USER, content=[TextPart(text="Hello")])
        result = GroqAdapter._message_to_groq(msg)
        assert result["content"] == "Hello"

    def test_non_vision_model_raises_on_image(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter.__new__(GroqAdapter)
        adapter.model_name = "openai/gpt-oss-20b"
        adapter._is_loaded = True
        adapter._client = MagicMock()
        adapter._rate_limiter = None

        msg = _image_message()
        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            adapter.generate(msg)
        assert exc_info.value.capability == Capability.vision.value

    def test_stream_non_vision_model_raises_on_image(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter.__new__(GroqAdapter)
        adapter.model_name = "openai/gpt-oss-20b"
        adapter._is_loaded = True
        adapter._client = MagicMock()

        with pytest.raises(CapabilityNotSupportedError):
            next(adapter.generate_stream(_image_message()))


# ---------------------------------------------------------------------------
# Gemini adapter translation
# ---------------------------------------------------------------------------

class TestGeminiAdapterImageTranslation:
    def test_message_to_genai_parts_contains_image(self):
        pytest.importorskip("google.genai")
        from effgen.models.gemini_adapter import GeminiAdapter
        from effgen.multimodal.image_pre import prepare

        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter.model_name = "gemini-3.1-flash-lite-preview"

        msg = _image_message()
        parts = adapter._message_to_genai_parts(msg, prepare)

        # Should produce at least one Part with inline data
        assert len(parts) >= 1
        has_bytes_part = any(
            getattr(p, "inline_data", None) is not None
            for p in parts
        )
        assert has_bytes_part

    def test_prepare_content_with_message(self):
        pytest.importorskip("google.genai")
        from effgen.models.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter.model_name = "gemini-3.1-flash-lite-preview"

        msg = _image_message()
        content = adapter._prepare_content(msg)
        assert isinstance(content, list)
        assert len(content) >= 1


# ---------------------------------------------------------------------------
# Anthropic adapter translation (code only — no live test)
# ---------------------------------------------------------------------------

class TestAnthropicAdapterImageTranslation:
    def test_message_to_anthropic_contains_base64_image(self):
        from effgen.models.anthropic_adapter import AnthropicAdapter

        msg = _image_message()
        result = AnthropicAdapter._effgen_message_to_anthropic(msg)

        assert result["role"] == "user"
        parts = result["content"]
        assert isinstance(parts, list)
        image_blocks = [p for p in parts if p.get("type") == "image"]
        assert len(image_blocks) == 1
        source = image_blocks[0]["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "image/jpeg"
        # Validate it's real base64
        decoded = base64.b64decode(source["data"])
        assert len(decoded) > 0

    def test_build_content_with_message_object(self):
        from effgen.models.anthropic_adapter import AnthropicAdapter

        msg = _image_message()
        content = AnthropicAdapter._build_content(msg)
        assert isinstance(content, list)
        image_blocks = [p for p in content if p.get("type") == "image"]
        assert len(image_blocks) == 1

    def test_non_vision_model_raises_on_image(self):
        from effgen.models.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.model_name = "unknown-text-only-model"
        adapter._is_loaded = True

        with pytest.raises(CapabilityNotSupportedError):
            adapter.generate(_image_message())


# ---------------------------------------------------------------------------
# Together adapter translation
# ---------------------------------------------------------------------------

class TestTogetherAdapterImageTranslation:
    def test_message_to_together_produces_image_url(self):
        from effgen.models.together_adapter import TogetherAdapter

        msg = _image_message()
        result = TogetherAdapter._message_to_together(msg)

        parts = result["content"]
        assert isinstance(parts, list)
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1

    def test_non_vision_model_raises_on_image(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter.__new__(TogetherAdapter)
        adapter.model_name = "meta-llama/Llama-3.2-1B"
        adapter._is_loaded = True
        adapter._client = MagicMock()
        adapter._rate_limiter = None

        with pytest.raises(CapabilityNotSupportedError):
            adapter.generate(_image_message())


# ---------------------------------------------------------------------------
# HF adapter translation
# ---------------------------------------------------------------------------

class TestHFAdapterImageTranslation:
    def test_effgen_message_to_dict_produces_image_url(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter.__new__(HFInferenceAdapter)
        adapter.model_name = "test-model"
        adapter._is_loaded = False
        adapter._client = None
        adapter._info = {}
        adapter._rate_limiter = None
        msg = _image_message()
        result = adapter._effgen_message_to_dict(msg)

        parts = result["content"]
        assert isinstance(parts, list)
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1

    def test_non_vision_model_raises_on_image(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter.__new__(HFInferenceAdapter)
        adapter.model_name = "text-only-model"
        adapter._is_loaded = True
        adapter._client = MagicMock()
        adapter._info = {"supports_vision": False, "use_text_generation": False}
        adapter._rate_limiter = None

        with pytest.raises(CapabilityNotSupportedError):
            adapter.generate(_image_message())


# ---------------------------------------------------------------------------
# CapabilityNotSupportedError shape
# ---------------------------------------------------------------------------

class TestCapabilityError:
    def test_error_message_includes_capability(self):
        err = CapabilityNotSupportedError(Capability.vision, provider="test_provider")
        assert "vision" in str(err)
        assert "test_provider" in str(err)
        assert err.capability == "vision"

    def test_error_with_hint(self):
        err = CapabilityNotSupportedError(Capability.vision, hint="Use gpt-4o instead.")
        assert "gpt-4o" in str(err)
