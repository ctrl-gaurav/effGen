"""Unit tests for MLX-VLM engine implementation."""

from unittest.mock import MagicMock, patch

import pytest

from effgen.models.base import GenerationConfig, GenerationResult, ModelType
from effgen.models.mlx_vlm_engine import MLXVLMEngine


@pytest.fixture
def engine():
    """Create an MLXVLMEngine instance with default params."""
    return MLXVLMEngine(
        model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        max_tokens=4096,
        apply_chat_template=True,
        system_prompt="You are a helpful vision assistant.",
    )


@pytest.fixture
def loaded_engine():
    """Create an MLXVLMEngine that appears loaded with mocked internals."""
    eng = MLXVLMEngine(
        model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        max_tokens=4096,
        apply_chat_template=True,
    )
    eng.model = MagicMock()
    eng.processor = MagicMock()
    eng.tokenizer = MagicMock()
    eng.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    eng.tokenizer.apply_chat_template.return_value = "<|im_start|>user\nDescribe<|im_end|>"
    eng.vlm_config = MagicMock()
    eng._is_loaded = True
    eng._context_length = 4096
    return eng


class TestMLXVLMEngineInit:
    """Tests for MLXVLMEngine constructor."""

    def test_mlx_vlm_engine_init(self, engine):
        assert engine.model_name == "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        assert engine.max_tokens_limit == 4096
        assert engine.apply_chat_template is True
        assert engine.system_prompt == "You are a helpful vision assistant."
        assert engine.processor is None
        assert engine.vlm_config is None
        assert engine.model is None
        assert engine._is_loaded is False

    def test_mlx_vlm_engine_model_type(self, engine):
        assert engine.model_type == ModelType.MLX_VLM


class TestMLXVLMEngineLoad:
    """Tests for MLXVLMEngine.load()."""

    @patch("effgen.hardware.platform.is_apple_silicon", return_value=False)
    def test_mlx_vlm_engine_load_not_apple_silicon(self, mock_as, engine):
        from effgen.hardware.platform import is_apple_silicon
        is_apple_silicon.cache_clear()
        with pytest.raises(RuntimeError, match="MLX-VLM requires Apple Silicon"):
            engine.load()
        is_apple_silicon.cache_clear()

    @patch("effgen.hardware.platform.get_unified_memory_gb", return_value=32.0)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=True)
    def test_mlx_vlm_engine_load_success(self, mock_as, mock_mem, engine):
        from effgen.hardware.platform import is_apple_silicon
        is_apple_silicon.cache_clear()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.max_position_embeddings = 8192
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_vlm_config = MagicMock()

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (mock_model, mock_processor)

        mock_mlx_vlm_utils = MagicMock()
        mock_mlx_vlm_utils.load_config.return_value = mock_vlm_config

        with patch.dict("sys.modules", {
            "mlx_vlm": mock_mlx_vlm,
            "mlx_vlm.utils": mock_mlx_vlm_utils,
        }):
            engine.load()

        assert engine._is_loaded is True
        assert engine.model is mock_model
        assert engine.processor is mock_processor
        assert engine.tokenizer is mock_processor.tokenizer
        assert engine.vlm_config is mock_vlm_config

        is_apple_silicon.cache_clear()

    @patch("effgen.hardware.platform.is_apple_silicon", return_value=True)
    def test_mlx_vlm_engine_load_no_mlx_vlm_installed(self, mock_as, engine):
        from effgen.hardware.platform import is_apple_silicon
        is_apple_silicon.cache_clear()
        with patch.dict("sys.modules", {"mlx_vlm": None, "mlx_vlm.utils": None}):
            with pytest.raises(RuntimeError, match="mlx-vlm is not installed"):
                engine.load()
        is_apple_silicon.cache_clear()


class TestMLXVLMEngineGenerate:
    """Tests for MLXVLMEngine.generate()."""

    def test_mlx_vlm_generate_not_loaded(self, engine):
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            engine.generate("Describe this image", images=["img.png"])

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_vlm_generate_with_images(self, mock_validate, loaded_engine):
        """Generate with images should use vlm_generate pipeline."""
        mock_vlm_gen = MagicMock(return_value="A cat sitting on a table")

        mock_vlm_prompt_utils = MagicMock()
        mock_vlm_prompt_utils.apply_chat_template.return_value = "formatted_prompt_with_image_tokens"

        with patch.dict("sys.modules", {
            "mlx_vlm": MagicMock(generate=mock_vlm_gen),
            "mlx_vlm.prompt_utils": mock_vlm_prompt_utils,
        }):
            with patch("mlx_vlm.generate", mock_vlm_gen):
                result = loaded_engine.generate(
                    prompt="Describe this image",
                    images=["test_image.png"],
                )

        assert isinstance(result, GenerationResult)
        assert result.text == "A cat sitting on a table"
        assert result.metadata["num_images"] == 1
        assert result.metadata["engine"] == "mlx_vlm"

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_vlm_generate_text_only(self, mock_validate, loaded_engine):
        """No images should delegate to parent MLXEngine.generate."""
        mock_mlx_generate = MagicMock(return_value="Text-only response")
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate = mock_mlx_generate

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = loaded_engine.generate(prompt="Hello, no images here")

        assert isinstance(result, GenerationResult)
        assert result.text == "Text-only response"
        assert result.metadata["engine"] == "mlx"

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_vlm_generate_empty_images_delegates(self, mock_validate, loaded_engine):
        """Empty images list should also delegate to parent."""
        mock_mlx_generate = MagicMock(return_value="Fallback text")
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate = mock_mlx_generate

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = loaded_engine.generate(prompt="Hello", images=[])

        assert result.metadata["engine"] == "mlx"


class TestMLXVLMEngineStream:
    """Tests for MLXVLMEngine.generate_stream()."""

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_vlm_generate_stream_with_images(self, mock_validate, loaded_engine):
        """Streaming with images should yield the full response at once."""
        mock_vlm_gen = MagicMock(return_value="Image description output")

        mock_vlm_prompt_utils = MagicMock()
        mock_vlm_prompt_utils.apply_chat_template.return_value = "formatted"

        with patch.dict("sys.modules", {
            "mlx_vlm": MagicMock(generate=mock_vlm_gen),
            "mlx_vlm.prompt_utils": mock_vlm_prompt_utils,
        }):
            with patch("mlx_vlm.generate", mock_vlm_gen):
                chunks = list(loaded_engine.generate_stream(
                    prompt="Describe image",
                    images=["image.png"],
                ))

        # VLM streaming with images yields the complete result in one chunk
        assert len(chunks) == 1
        assert chunks[0] == "Image description output"

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_vlm_generate_stream_text_only(self, mock_validate, loaded_engine):
        """Text-only streaming should delegate to parent MLXEngine.generate_stream."""
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"

        mock_stream_fn = MagicMock(return_value=iter([chunk1, chunk2]))
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate = mock_stream_fn

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            chunks = list(loaded_engine.generate_stream(prompt="Hi"))

        assert chunks == ["Hello", " world"]


class TestMLXVLMEngineToolCalling:
    """Tests for tool calling support."""

    def test_mlx_vlm_supports_tool_calling(self, engine):
        """VLMs should always return False for tool calling."""
        assert engine.supports_tool_calling() is False

    def test_mlx_vlm_supports_tool_calling_loaded(self, loaded_engine):
        """Even when loaded, VLMs should not support tool calling."""
        assert loaded_engine.supports_tool_calling() is False


class TestMLXVLMEngineUnload:
    """Tests for MLXVLMEngine.unload()."""

    def test_mlx_vlm_unload(self, loaded_engine):
        assert loaded_engine._is_loaded is True
        assert loaded_engine.model is not None
        assert loaded_engine.processor is not None
        assert loaded_engine.vlm_config is not None

        loaded_engine.unload()

        assert loaded_engine._is_loaded is False
        assert loaded_engine.model is None
        assert loaded_engine.processor is None
        assert loaded_engine.tokenizer is None
        assert loaded_engine.vlm_config is None

    def test_mlx_vlm_unload_when_not_loaded(self, engine):
        """Unloading when not loaded should not raise."""
        engine.unload()
        assert engine._is_loaded is False
        assert engine.processor is None
        assert engine.vlm_config is None


class TestMLXVLMEngineBatch:
    """Tests for MLXVLMEngine.generate_batch() with images."""

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_vlm_batch_with_images_list(self, mock_validate, loaded_engine):
        """Batch with per-prompt images should process each independently."""
        mock_vlm_gen = MagicMock(side_effect=["Cat image", "Dog image"])

        mock_vlm_prompt_utils = MagicMock()
        mock_vlm_prompt_utils.apply_chat_template.return_value = "formatted"

        with patch.dict("sys.modules", {
            "mlx_vlm": MagicMock(generate=mock_vlm_gen),
            "mlx_vlm.prompt_utils": mock_vlm_prompt_utils,
        }):
            with patch("mlx_vlm.generate", mock_vlm_gen):
                results = loaded_engine.generate_batch(
                    prompts=["Describe cat", "Describe dog"],
                    images_list=[["cat.png"], ["dog.png"]],
                )

        assert len(results) == 2
        assert results[0].text == "Cat image"
        assert results[1].text == "Dog image"

    def test_mlx_vlm_batch_images_list_length_mismatch(self, loaded_engine):
        """Mismatched images_list length should raise ValueError."""
        with pytest.raises(ValueError, match="images_list length"):
            loaded_engine.generate_batch(
                prompts=["A", "B"],
                images_list=[["img.png"]],  # only 1, but 2 prompts
            )
