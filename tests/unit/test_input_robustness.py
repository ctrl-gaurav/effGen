"""
Input-type robustness for Agent.run()/stream() and multimodal ergonomics.

Covers:
- run()/stream()/run_async() accept str | Message | list[ContentPart]
- a clear TypeError for unsupported task types
- the safe task-preview helper never assumes subscriptability
- a bare str inside a Message content list is auto-wrapped in a TextPart
- image_from()'s URL fetch sends a real User-Agent
"""

from __future__ import annotations

import inspect
import urllib.request

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.messages import ImagePart, Message, Role, TextPart
from effgen.core.multimodal import _FETCH_USER_AGENT, _fetch_url, image_from

# A minimal, valid PNG payload (magic bytes + filler) for ImagePart construction.
_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def _img() -> ImagePart:
    return image_from(_PNG, mime="image/png")


# ---------------------------------------------------------------------------
# inputs is an explicit keyword parameter (discoverable via inspect)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["run", "stream", "run_async"])
def test_inputs_is_explicit_param(method):
    sig = inspect.signature(getattr(Agent, method))
    assert "inputs" in sig.parameters
    assert sig.parameters["inputs"].default is None


# ---------------------------------------------------------------------------
# _coerce_task_input accepts str | Message | list[ContentPart]
# ---------------------------------------------------------------------------

def test_coerce_plain_str_passthrough():
    assert Agent._coerce_task_input("hello") == ("hello", None)


def test_coerce_message_text_only():
    msg = Message(role=Role.USER, content=[TextPart(text="hi"), "there"])
    task, inputs = Agent._coerce_task_input(msg)
    assert task == "hi\nthere"
    assert inputs is None


def test_coerce_message_with_media_routes_to_inputs():
    msg = Message(role=Role.USER, content=[_img(), "what is this"])
    task, inputs = Agent._coerce_task_input(msg)
    assert task == "what is this"
    assert inputs is not None
    assert [type(p).__name__ for p in inputs] == ["ImagePart"]


def test_coerce_list_of_content_parts():
    task, inputs = Agent._coerce_task_input([_img(), "caption"])
    assert task == "caption"
    assert [type(p).__name__ for p in inputs] == ["ImagePart"]


def test_coerce_merges_explicit_inputs():
    msg = Message(role=Role.USER, content=[_img(), "describe"])
    task, inputs = Agent._coerce_task_input(msg, inputs=[_img()])
    assert task == "describe"
    assert len(inputs) == 2


def test_coerce_bad_task_type_raises_typeerror():
    with pytest.raises(TypeError, match="str, a Message, or a"):
        Agent._coerce_task_input(12345)


def test_coerce_bad_content_item_raises_typeerror():
    with pytest.raises(TypeError, match="content item"):
        Agent._coerce_task_input([object()])


# ---------------------------------------------------------------------------
# _extract_task_preview is safe for every accepted task type
# ---------------------------------------------------------------------------

def test_preview_str():
    assert Agent._extract_task_preview("abcdef", 3) == "abc"


def test_preview_message_does_not_crash():
    msg = Message(role=Role.USER, content=[_img(), "hello"])
    assert Agent._extract_task_preview(msg) == "hello"


def test_preview_list_without_text():
    assert "content part" in Agent._extract_task_preview([_img()])


# ---------------------------------------------------------------------------
# A bare str in a Message content list is auto-wrapped in a TextPart
# ---------------------------------------------------------------------------

def test_message_bare_str_in_content_wrapped():
    msg = Message(role=Role.USER, content=[_img(), "describe this"])
    assert [type(p).__name__ for p in msg.content] == ["ImagePart", "TextPart"]
    assert msg.text == "describe this"


def test_message_invalid_content_still_rejected():
    from effgen.errors import InvalidMultimodalContent

    with pytest.raises(InvalidMultimodalContent):
        Message(role=Role.USER, content=[object()])


# ---------------------------------------------------------------------------
# ``image_from`` URL fetch sends a real User-Agent
# ---------------------------------------------------------------------------

def test_fetch_url_sends_user_agent(monkeypatch):
    captured = {}

    class _Resp:
        def read(self):
            return b"data"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        captured["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = _fetch_url("https://example.com/x.png")
    assert out == b"data"
    assert captured["ua"] == _FETCH_USER_AGENT
    assert "urllib" not in (captured["ua"] or "").lower()


# ---------------------------------------------------------------------------
# The default agent's image inputs= carry the same grounding guidance the
# multimodal preset already has, without touching a caller's own persona.
# ---------------------------------------------------------------------------

def _mock_agent(system_prompt: str | None = None) -> Agent:
    from tests.fixtures.mock_models import MockModel

    kwargs = {"model": MockModel(responses=["ok"])}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return Agent(config=AgentConfig(**kwargs))


def test_default_persona_gains_image_grounding_guidance():
    agent = _mock_agent()
    messages = agent._build_multimodal_prompt("describe this", [_img()])
    system_text = messages[0].content[0].text
    assert system_text.startswith("You are a helpful AI assistant.")
    assert "state only what is visible" in system_text


def test_default_persona_unchanged_without_an_image():
    """Audio-only inputs don't need the image-grounding line appended."""
    from effgen.core.multimodal import audio_from

    agent = _mock_agent()
    messages = agent._build_multimodal_prompt(
        "transcribe this", [audio_from(b"RIFF" + b"0" * 32, mime="audio/wav")]
    )
    system_text = messages[0].content[0].text
    assert system_text == "You are a helpful AI assistant."


def test_custom_persona_not_modified_by_image_input():
    """A caller's own system_prompt is never rewritten, even with an image."""
    custom = "You are a pirate. Speak in pirate slang."
    agent = _mock_agent(system_prompt=custom)
    messages = agent._build_multimodal_prompt("describe this", [_img()])
    assert messages[0].content[0].text == custom
