"""Unit tests for MLX engine implementation."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from effgen.models.base import GenerationConfig, GenerationResult, ModelType, TokenCount
from effgen.models.mlx_engine import MLXEngine


@pytest.fixture
def engine():
    """Create an MLXEngine instance with default params."""
    return MLXEngine(
        model_name="mlx-community/test-model-4bit",
        max_tokens=4096,
        apply_chat_template=True,
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def loaded_engine():
    """Create an MLXEngine that appears loaded with mocked model and tokenizer."""
    eng = MLXEngine(
        model_name="mlx-community/test-model-4bit",
        max_tokens=4096,
        apply_chat_template=True,
    )
    eng.model = MagicMock()
    eng.tokenizer = MagicMock()
    eng.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    eng.tokenizer.apply_chat_template.return_value = "<|im_start|>user\nHello<|im_end|>"
    eng._is_loaded = True
    eng._context_length = 4096
    return eng


class TestMLXEngineInit:
    """Tests for MLXEngine constructor."""

    def test_mlx_engine_init(self, engine):
        assert engine.model_name == "mlx-community/test-model-4bit"
        assert engine.max_tokens_limit == 4096
        assert engine.apply_chat_template is True
        assert engine.system_prompt == "You are a helpful assistant."
        assert engine.adapter_path is None
        assert engine.lazy_load is False
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine._is_loaded is False

    def test_mlx_engine_init_with_adapter(self):
        eng = MLXEngine(
            model_name="test-model",
            adapter_path="/path/to/adapter",
            lazy_load=True,
        )
        assert eng.adapter_path == "/path/to/adapter"
        assert eng.lazy_load is True

    def test_mlx_engine_model_type(self, engine):
        assert engine.model_type == ModelType.MLX


class TestMLXEngineLoad:
    """Tests for MLXEngine.load()."""

    @patch("effgen.hardware.platform.is_apple_silicon", return_value=False)
    def test_mlx_engine_load_not_apple_silicon(self, mock_as, engine):
        from effgen.hardware.platform import is_apple_silicon
        is_apple_silicon.cache_clear()
        with pytest.raises(RuntimeError, match="MLX requires Apple Silicon"):
            engine.load()
        is_apple_silicon.cache_clear()

    @patch("effgen.hardware.platform.is_apple_silicon", return_value=True)
    def test_mlx_engine_load_no_mlx_installed(self, mock_as, engine):
        from effgen.hardware.platform import is_apple_silicon
        is_apple_silicon.cache_clear()
        with patch.dict("sys.modules", {"mlx_lm": None}):
            with pytest.raises(RuntimeError, match="mlx-lm is not installed"):
                engine.load()
        is_apple_silicon.cache_clear()

    @patch("effgen.hardware.platform.get_unified_memory_gb", return_value=16.0)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=True)
    def test_mlx_engine_load_success(self, mock_as, mock_mem):
        from effgen.hardware.platform import is_apple_silicon
        is_apple_silicon.cache_clear()

        # Create engine without explicit max_tokens so _detect_context_length
        # reads from model config
        eng = MLXEngine(model_name="mlx-community/test-model-4bit")

        mock_config = MagicMock()
        mock_config.max_position_embeddings = 8192
        mock_model = MagicMock()
        mock_model.config = mock_config
        mock_model.args = None
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            eng.load()

        assert eng._is_loaded is True
        assert eng.model is mock_model
        assert eng.tokenizer is mock_tokenizer
        assert eng._context_length == 8192

        is_apple_silicon.cache_clear()


class TestMLXEngineGenerate:
    """Tests for MLXEngine.generate()."""

    def test_mlx_engine_generate_not_loaded(self, engine):
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            engine.generate("Hello")

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_engine_generate(self, mock_validate, loaded_engine):
        mock_mlx_generate = MagicMock(return_value="Generated response text")
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate = mock_mlx_generate

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            config = GenerationConfig(
                temperature=0.5,
                top_p=0.95,
                max_tokens=256,
            )
            result = loaded_engine.generate("Hello", config=config)

        assert isinstance(result, GenerationResult)
        assert result.text == "Generated response text"
        assert result.finish_reason == "stop"
        assert result.model_name == "mlx-community/test-model-4bit"
        assert result.metadata["engine"] == "mlx"

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_generation_config_mapping(self, mock_validate, loaded_engine):
        """Verify GenerationConfig fields map to MLX params correctly."""
        mock_mlx_generate = MagicMock(return_value="output")
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate = mock_mlx_generate

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            config = GenerationConfig(
                temperature=0.3,
                top_p=0.85,
                max_tokens=128,
                repetition_penalty=1.2,
                seed=42,
            )
            loaded_engine.generate("test", config=config)

        call_kwargs = mock_mlx_generate.call_args
        assert call_kwargs.kwargs["temp"] == 0.3
        assert call_kwargs.kwargs["top_p"] == 0.85
        assert call_kwargs.kwargs["max_tokens"] == 128
        assert call_kwargs.kwargs["repetition_penalty"] == 1.2
        assert call_kwargs.kwargs["seed"] == 42


class TestMLXEngineChatTemplate:
    """Tests for chat template application."""

    def test_chat_template_application(self, loaded_engine):
        """Verify tokenizer.apply_chat_template is called for unformatted prompts."""
        loaded_engine.tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        loaded_engine.system_prompt = "You are helpful"

        result = loaded_engine._format_prompt_with_chat_template("Hello")
        loaded_engine.tokenizer.apply_chat_template.assert_called_once()
        assert "<|im_start|>" in result

    def test_chat_template_skip_when_formatted(self, loaded_engine):
        """Prompts already containing template markers should be returned as-is."""
        formatted_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        result = loaded_engine._format_prompt_with_chat_template(formatted_prompt)
        assert result == formatted_prompt
        # apply_chat_template should NOT have been called
        loaded_engine.tokenizer.apply_chat_template.assert_not_called()

    def test_chat_template_skip_when_disabled(self):
        """When apply_chat_template=False, prompt should pass through unchanged."""
        eng = MLXEngine(model_name="test", apply_chat_template=False)
        eng.tokenizer = MagicMock()
        result = eng._format_prompt_with_chat_template("Hello")
        assert result == "Hello"

    def test_chat_template_llama_format_detected(self, loaded_engine):
        """Llama-style [INST] markers should be detected as already formatted."""
        formatted = "[INST] Hello [/INST]"
        result = loaded_engine._format_prompt_with_chat_template(formatted)
        assert result == formatted


class TestMLXEngineStream:
    """Tests for MLXEngine.generate_stream()."""

    def test_mlx_engine_generate_stream_not_loaded(self, engine):
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            list(engine.generate_stream("Hello"))

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_engine_generate_stream(self, mock_validate, loaded_engine):
        """Verify streaming yields text chunks."""
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"

        mock_stream_fn = MagicMock(return_value=iter([chunk1, chunk2]))
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate = mock_stream_fn

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            chunks = list(loaded_engine.generate_stream("test prompt"))

        assert chunks == ["Hello", " world"]


class TestMLXEngineBatch:
    """Tests for MLXEngine.generate_batch()."""

    def test_mlx_engine_generate_batch_not_loaded(self, engine):
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            engine.generate_batch(["Hello", "World"])

    @patch("effgen.models.mlx_engine.MLXEngine.validate_prompt", return_value=True)
    def test_mlx_engine_generate_batch(self, mock_validate, loaded_engine):
        """Verify batch processes prompts sequentially."""
        mock_mlx_generate = MagicMock(side_effect=["Response 1", "Response 2"])
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate = mock_mlx_generate

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            results = loaded_engine.generate_batch(["Prompt 1", "Prompt 2"])

        assert len(results) == 2
        assert all(isinstance(r, GenerationResult) for r in results)
        assert results[0].text == "Response 1"
        assert results[1].text == "Response 2"


class TestMLXEngineTokens:
    """Tests for MLXEngine.count_tokens()."""

    def test_mlx_engine_count_tokens(self, loaded_engine):
        loaded_engine.tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
        result = loaded_engine.count_tokens("Hello, world!")
        assert isinstance(result, TokenCount)
        assert result.count == 7
        assert result.model_name == "mlx-community/test-model-4bit"

    def test_mlx_engine_count_tokens_not_loaded(self, engine):
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            engine.count_tokens("Hello")


class TestMLXEngineUnload:
    """Tests for MLXEngine.unload()."""

    def test_mlx_engine_unload(self, loaded_engine):
        assert loaded_engine._is_loaded is True
        assert loaded_engine.model is not None
        assert loaded_engine.tokenizer is not None

        loaded_engine.unload()

        assert loaded_engine._is_loaded is False
        assert loaded_engine.model is None
        assert loaded_engine.tokenizer is None

    def test_mlx_engine_unload_when_not_loaded(self, engine):
        """Unloading when not loaded should not raise."""
        engine.unload()
        assert engine._is_loaded is False


class TestMLXEngineContextManager:
    """Tests for __enter__ / __exit__."""

    @patch.object(MLXEngine, "load")
    @patch.object(MLXEngine, "unload")
    def test_mlx_engine_context_manager(self, mock_unload, mock_load):
        eng = MLXEngine(model_name="test-model")
        with eng as ctx:
            assert ctx is eng
            mock_load.assert_called_once()
        mock_unload.assert_called_once()

    @patch.object(MLXEngine, "unload")
    def test_context_manager_already_loaded(self, mock_unload):
        """If already loaded, __enter__ should not call load() again."""
        eng = MLXEngine(model_name="test-model")
        eng._is_loaded = True

        with patch.object(MLXEngine, "load") as mock_load:
            with eng:
                mock_load.assert_not_called()
        mock_unload.assert_called_once()


class TestMLXEngineMisc:
    """Miscellaneous MLXEngine tests."""

    def test_get_context_length_default(self, engine):
        assert engine.get_context_length() == 4096

    def test_get_max_batch_size(self, engine):
        assert engine.get_max_batch_size() == 1

    def test_supports_tool_calling_not_loaded(self, engine):
        assert engine.supports_tool_calling() is False

    def test_repr(self, engine):
        r = repr(engine)
        assert "MLXEngine" in r
        assert "mlx-community/test-model-4bit" in r
