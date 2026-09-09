"""Carrying a conversation's tool call and its result to a provider.

A run held as messages puts the model's own words and the call it made on one
assistant turn, and answers that call with a turn quoting the call's id. Every
provider expresses that exchange, but each expresses it differently and none of
them accepts a call with no answer: a conversation carrying a tool call that
nothing replies to is rejected by the request schema.

This module holds the parts of that translation that do not belong to any one
adapter:

* :func:`split_tool_parts` separates a message's tool call and tool result from
  the content an adapter already knows how to convert, so an adapter gains the
  two parts without changing how it renders text, images or video;
* :func:`openai_message` assembles the OpenAI-protocol message dict from what
  the adapter converted plus those parts — the shape Groq, Together, Cerebras,
  Fireworks, Hugging Face inference endpoints and every OpenAI-compatible
  server accept;
* :func:`message_to_openai` is the whole conversion for an adapter that has no
  converter of its own;
* :func:`call_names_by_id` reads the name each call id belongs to, for a
  provider whose result turn is keyed by the tool's name rather than by the id.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "call_names_by_id",
    "message_to_openai",
    "messages_to_openai",
    "messages_via",
    "openai_message",
    "result_text",
    "split_tool_parts",
]


def result_text(result: Any) -> str:
    """A tool result as the text a provider's result turn carries."""
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


def split_tool_parts(message: Any) -> tuple[list[Any], list[dict[str, Any]], Any]:
    """Separate a message's tool parts from the rest of its content.

    Args:
        message: A :class:`~effgen.core.messages.Message`.

    Returns:
        The content parts an adapter converts itself, the tool calls in
        OpenAI's ``tool_calls`` shape, and the message's tool result part or
        ``None``. A turn carrying more than one result keeps the first: a
        result turn answers one call.
    """
    from effgen.core.messages import ToolCallPart, ToolResultPart

    other: list[Any] = []
    calls: list[dict[str, Any]] = []
    result: Any = None
    for part in getattr(message, "content", []) or []:
        if isinstance(part, ToolCallPart):
            calls.append({
                "id": part.tool_call_id,
                "type": "function",
                "function": {
                    "name": part.name,
                    "arguments": json.dumps(part.arguments),
                },
            })
        elif isinstance(part, ToolResultPart):
            if result is None:
                result = part
        else:
            other.append(part)
    return other, calls, result


def openai_message(
    role: str,
    content_parts: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    result: Any,
) -> dict[str, Any]:
    """One OpenAI-protocol message from converted content and tool parts.

    Args:
        role: The message's role.
        content_parts: The content the adapter converted, in OpenAI's part
            shape.
        calls: The turn's tool calls, as :func:`split_tool_parts` returns them.
        result: The turn's tool result part, or ``None``.

    Returns:
        The message dict. A turn carrying a result is a ``tool`` message
        quoting the call id it answers, whatever role the message declared —
        the id is what the provider matches on. An assistant turn that only
        made a call carries ``content: None``, which is the key present and
        empty rather than absent.
    """
    if result is not None:
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result_text(result.result),
        }

    if len(content_parts) == 1 and content_parts[0].get("type") == "text":
        message_dict: dict[str, Any] = {"role": role, "content": content_parts[0]["text"]}
    elif content_parts:
        message_dict = {"role": role, "content": content_parts}
    else:
        message_dict = {"role": role, "content": None}
    if calls:
        message_dict["tool_calls"] = calls
    return message_dict


def message_to_openai(
    message: Any, *, provider: str, model_name: str = ""
) -> dict[str, Any]:
    """Convert one message to the OpenAI protocol, tool parts included.

    For an adapter with no conversion of its own. Text, images and video
    frames travel as OpenAI content parts; a tool call and a tool result
    travel as :func:`openai_message` puts them.

    Args:
        message: A :class:`~effgen.core.messages.Message`.
        provider: The provider name image pre-processing is keyed by.
        model_name: The model, where pre-processing varies by model.

    Returns:
        The message dict.
    """
    from effgen.core.messages import ImagePart, TextPart, VideoPart
    from effgen.multimodal.image_pre import prepare as _preprocess_image

    other, calls, result = split_tool_parts(message)
    role = message.role.value
    content_parts: list[dict[str, Any]] = []
    for part in other:
        if isinstance(part, TextPart):
            content_parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            processed = _preprocess_image(part, provider, model_name)
            b64 = base64.b64encode(processed.image).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{processed.mime};base64,{b64}"},
            })
        elif isinstance(part, VideoPart):
            for frame in part.frames:
                b64 = base64.b64encode(frame).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{part.mime};base64,{b64}"},
                })
    return openai_message(role, content_parts, calls, result)


def messages_to_openai(
    prompt: Any, *, provider: str, model_name: str = ""
) -> list[dict[str, Any]] | None:
    """Convert a prompt that is a conversation, or report that it is not.

    Args:
        prompt: What a caller passed as the prompt.
        provider: The provider name image pre-processing is keyed by.
        model_name: The model, where pre-processing varies by model.

    Returns:
        The message list when *prompt* is one or more
        :class:`~effgen.core.messages.Message`, else ``None`` so the caller
        keeps whatever handling it had for a string.
    """
    try:
        from effgen.core.messages import Message
    except ImportError:  # pragma: no cover - effgen.core is always importable
        return None
    if isinstance(prompt, Message):
        return [message_to_openai(prompt, provider=provider, model_name=model_name)]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
        return [
            message_to_openai(m, provider=provider, model_name=model_name)
            for m in prompt
        ]
    return None


def messages_via(convert: Any, prompt: Any) -> list[dict[str, Any]] | None:
    """Apply an adapter's own per-message conversion to a whole prompt.

    Args:
        convert: The adapter's single-message conversion.
        prompt: What a caller passed as the prompt.

    Returns:
        The message list when *prompt* is one or more
        :class:`~effgen.core.messages.Message`, else ``None`` so the caller
        keeps whatever handling it had for a string.
    """
    try:
        from effgen.core.messages import Message
    except ImportError:  # pragma: no cover - effgen.core is always importable
        return None
    if isinstance(prompt, Message):
        return [convert(prompt)]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
        return [convert(message) for message in prompt]
    return None


def call_names_by_id(messages: Any) -> dict[str, str]:
    """The tool each call id names, read from the calls in a conversation.

    A result turn carries the id it answers but not the tool's name, and some
    providers key a result on the name. Reading the name off the call that
    earned the id keeps the two in step without the result having to repeat it.

    Args:
        messages: The conversation, as
            :class:`~effgen.core.messages.Message` objects.

    Returns:
        Call id to tool name, for every call in the conversation.
    """
    from effgen.core.messages import ToolCallPart

    names: dict[str, str] = {}
    for message in messages or ():
        for part in getattr(message, "content", []) or []:
            if isinstance(part, ToolCallPart):
                names[part.tool_call_id] = part.name
    return names
