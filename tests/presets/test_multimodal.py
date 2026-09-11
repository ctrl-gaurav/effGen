"""
Tests for the multimodal preset.

Unit: registration, tool list, system prompt, create_agent factory.
Live: skipped unless API keys are present (marked pytest.mark.live).
"""

from __future__ import annotations

import os

import pytest

from effgen.models.base import BaseModel, ModelType, TokenCount
from tests._harness.provider_unavailable import skip_if_provider_refused

# ---------------------------------------------------------------------------
# Unit — preset registration
# ---------------------------------------------------------------------------

def test_multimodal_preset_registered():
    from effgen.presets import list_presets
    presets = list_presets()
    assert "multimodal" in presets


def test_multimodal_preset_config():
    from effgen.presets.registry import get_preset
    cfg = get_preset("multimodal")
    assert cfg.name == "multimodal"
    assert cfg.temperature <= 0.5
    assert cfg.max_iterations >= 8


def test_multimodal_preset_tools():
    from effgen.presets.registry import get_preset
    cfg = get_preset("multimodal")
    tool_names = set(cfg.tool_names)
    # Must have the auto-dispatch hub and core modality tools
    assert "multimodal_describe" in tool_names
    assert "image_caption" in tool_names
    assert "audio_transcribe" in tool_names
    assert "ocr" in tool_names
    assert "image_info" in tool_names


def test_multimodal_preset_system_prompt_modality_aware():
    from effgen.presets.registry import get_preset
    cfg = get_preset("multimodal")
    sp = cfg.system_prompt.lower()
    assert "image" in sp
    assert "audio" in sp
    assert "video" in sp


def test_multimodal_preset_tags():
    from effgen.presets.registry import get_preset
    cfg = get_preset("multimodal")
    assert "multimodal" in cfg.tags
    assert "vision" in cfg.tags


class _FakeModel(BaseModel):
    """Minimal model stub that satisfies the Agent.__init__ checks.

    It answers the way a model answers an agent that holds tools: with a final
    answer rather than a bare number. An agent with tools runs the reasoning
    loop whether or not the task carries a picture, and a reply that names
    neither an action nor an answer is a reply the loop has to ask about again —
    so a stub that never finishes would spend the whole iteration budget saying
    the same thing.
    """

    def __init__(self):
        super().__init__(model_name="fake-model", model_type=ModelType.OPENAI)
        self.config = type("C", (), {"temperature": 0.7})()
        self.last_prompt = None

    def load(self):
        self._is_loaded = True

    def generate(self, messages, config=None, **kwargs):
        from effgen.models.base import GenerationResult
        self.last_prompt = messages
        return GenerationResult(
            text="Final Answer: 42",
            tokens_used=5,
            finish_reason="stop",
            model_name=self.model_name,
            metadata={},
        )

    def generate_stream(self, messages, config=None, **kwargs):
        yield "Final Answer: 42"

    @property
    def loaded(self):
        return True

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 8192

    def unload(self) -> None:
        self._is_loaded = False


def test_create_agent_multimodal_returns_agent():
    from effgen.core.agent import Agent
    from effgen.presets import create_agent

    agent = create_agent("multimodal", _FakeModel())
    assert isinstance(agent, Agent)
    assert "multimodal" in agent.name.lower() or agent.name.endswith("agent")


def test_create_agent_multimodal_has_tools():
    from effgen.presets import create_agent

    agent = create_agent("multimodal", _FakeModel())
    # agent.tools is a dict {name: tool_instance}
    tool_names = set(agent.tools.keys())
    assert "multimodal_describe" in tool_names
    assert "image_caption" in tool_names
    assert "audio_transcribe" in tool_names


def test_create_agent_multimodal_system_prompt_override():
    from effgen.presets import create_agent

    custom = "Custom system prompt."
    agent = create_agent("multimodal", _FakeModel(), system_prompt=custom)
    assert agent.config.system_prompt == custom


def test_create_agent_multimodal_extra_tools():
    from effgen.presets import create_agent
    from effgen.tools.builtin.calculator import Calculator

    extra = [Calculator()]
    agent = create_agent("multimodal", _FakeModel(), extra_tools=extra)
    tool_names = set(agent.tools.keys())
    assert "calculator" in tool_names
    assert "multimodal_describe" in tool_names


def test_multimodal_agent_run_accepts_structured_inputs_with_tools():
    from effgen.core.messages import ImagePart, Message
    from effgen.presets import create_agent

    model = _FakeModel()
    agent = create_agent("multimodal", model)
    image = ImagePart(
        image=b"\x89PNG\r\n\x1a\n" + b"0" * 16,
        mime="image/png",
    )

    result = agent.run("Count visible people.", inputs=[image])

    skip_if_provider_refused(result)
    assert result.success
    assert result.output == "42"
    assert result.metadata["multimodal_inputs"] is True
    assert isinstance(model.last_prompt, list)
    assert all(isinstance(message, Message) for message in model.last_prompt)
    assert model.last_prompt[-1].has_image


def test_multimodal_agent_with_tools_runs_the_loop_with_or_without_a_picture():
    """An agent holding tools reasons the same way whichever the task carries.

    A task carrying a picture used to skip the reasoning loop and answer from
    the model's first reply; it does not any more. This pins the two against
    each other, so the picture path cannot drift away from the plain one again.
    """
    from effgen.core.messages import ImagePart
    from effgen.presets import create_agent

    image = ImagePart(image=b"\x89PNG\r\n\x1a\n" + b"0" * 16, mime="image/png")

    with_picture = create_agent("multimodal", _FakeModel())
    without = create_agent("multimodal", _FakeModel())

    framed = with_picture.run("Count visible people.", inputs=[image])
    plain = without.run("Count visible people.")

    assert framed.success and plain.success
    assert framed.output == plain.output == "42"
    assert framed.iterations == plain.iterations


def test_a_run_says_it_carried_media_whichever_path_answered_it():
    """`multimodal_inputs` is the caller's key, not one path's key.

    It used to be stamped only by the tool-free path, so an agent holding tools
    answered a picture without saying it had been given one. A caller reading
    the key should not have to know which path answered.
    """
    from effgen.core.agent import Agent, AgentConfig
    from effgen.core.messages import ImagePart
    from effgen.presets import create_agent

    image = ImagePart(image=b"\x89PNG\r\n\x1a\n" + b"0" * 16, mime="image/png")

    with_tools = create_agent("multimodal", _FakeModel())
    without_tools = Agent(AgentConfig(name="mm-plain", model=_FakeModel()))

    assert with_tools.run("Count them.", inputs=[image]).metadata["multimodal_inputs"] is True
    assert without_tools.run("Count them.", inputs=[image]).metadata["multimodal_inputs"] is True
    assert with_tools.run("Count them.").metadata["multimodal_inputs"] is False


def test_gemini_3_flash_lite_alias_resolves():
    from effgen.models.gemini_models import model_info

    info = model_info("gemini-3-flash-lite")
    assert info["canonical_id"] == "gemini-3.1-flash-lite"
    assert info["supports_vision"] is True
    assert info["supports_audio"] is True
    assert info["supports_video"] is True


def test_model_loader_detects_gemini_3_flash_lite():
    from effgen.models.base import ModelType
    from effgen.models.model_loader import ModelLoader

    assert ModelLoader()._detect_model_type("gemini-3-flash-lite") == ModelType.GEMINI


# ---------------------------------------------------------------------------
# Live — image query (skipped unless GOOGLE_API_KEY or OPENAI_API_KEY present)
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="No vision API key available",
)
def test_multimodal_preset_image_live(tmp_path):
    """Create a simple image and ask the agent to describe it."""
    from PIL import Image

    # Create a bright red square image
    img = Image.new("RGB", (128, 128), color=(255, 0, 0))
    img_path = tmp_path / "red_square.png"
    img.save(str(img_path))

    from effgen.presets import create_agent

    # Prefer OpenAI for this smoke to avoid Gemini free-tier daily quota.
    if os.getenv("OPENAI_API_KEY"):
        from effgen.models.openai_adapter import OpenAIAdapter
        model = OpenAIAdapter(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        model.load()
    else:
        from effgen.models.gemini_adapter import GeminiAdapter
        model = GeminiAdapter(model_name="gemini-3-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
        model.load()

    agent = create_agent("multimodal", model)
    result = agent.run(
        f"Describe what you see in this image: {img_path}. "
        "Use the multimodal_describe or image_caption tool."
    )
    skip_if_provider_refused(result)
    assert result.success, f"Agent failed: {result.output}"
    out_lower = result.output.lower()
    assert any(w in out_lower for w in ("red", "square", "image", "color", "colour")), (
        f"Expected color/shape description, got: {result.output[:200]}"
    )
