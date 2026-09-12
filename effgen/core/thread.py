"""The typed conversation state of one agent run.

A run's conversation is a sequence of typed *steps* — what the caller asked,
what the model thought, which tool it called with which arguments, what came
back, and how the run ended. :class:`AgentThread` holds that sequence and can
render it two ways:

* :meth:`AgentThread.to_text` — the flat ReAct transcript, the string the loop
  has always assembled by appending to a local accumulator. It is defined as
  plain concatenation of each step's own text, so every byte of the transcript
  comes from exactly one step.
* :meth:`AgentThread.to_messages` — a list of
  :class:`~effgen.core.messages.Message` (exported publicly as
  ``MultimodalMessage``), with the assistant's reasoning and its tool call on
  the *same* message, and each tool result answering a real call id.

The two renderings are views over one state, so a caller can move between the
text protocol and the message protocol without the run being replayed.

The *frame* around a transcript — the system preamble, the tool list, the
format specification, the question — is assembled elsewhere and is not part of
a thread's flat rendering: :class:`SystemStep` and :class:`TaskStep` render as
the empty string in :meth:`to_text` and as messages in :meth:`to_messages`.

Every call in a thread carries an id — the provider's own where the turn made a
provider-native call, otherwise one minted from the step's position — and every
result carries the id of the call it answers, so a conversation leaving effGen
never holds a call nothing replied to.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .messages import (
    ContentPart,
    Message,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

logger = logging.getLogger(__name__)

#: Serialisation version written by :meth:`AgentThread.to_dict` and checked by
#: :meth:`AgentThread.from_dict`. Bump it when the on-disk shape changes.
THREAD_SCHEMA_VERSION = 1

#: Prefix of a call id minted locally when a turn carried none. The suffix is
#: the step's index in the thread, so a replayed run mints the same ids.
_LOCAL_CALL_ID_PREFIX = "effgen-"

#: Key a tool's arguments land under when the turn wrote free text rather than
#: a JSON object. The same key the response parser uses, so a caller reading
#: arguments back sees one convention, not two.
_RAW_INPUT_KEY = "__raw_input__"

#: Observation length kept by ``to_messages(summary=True)``.
_SUMMARY_OBSERVATION_CHARS = 200

#: A thought shorter than this is bookkeeping, not an answer worth returning.
_SUBSTANTIVE_THOUGHT_CHARS = 20

#: Line :meth:`AgentThread.history_text` opens the earlier turns with, for a
#: frame that can only take one string.
_HISTORY_HEADER = "Earlier in this conversation:"

#: How much of an earlier reply that flat rendering keeps. The message
#: rendering keeps all of it; a text frame restates the whole history on every
#: turn, so a long earlier answer would crowd out the run itself.
_HISTORY_REPLY_CHARS = 300

__all__ = [
    "THREAD_SCHEMA_VERSION",
    "ActionStep",
    "AgentThread",
    "AnswerStep",
    "NudgeStep",
    "ObservationStep",
    "Step",
    "SystemStep",
    "TaskStep",
    "ThoughtStep",
    "TurnStep",
]


@runtime_checkable
class Step(Protocol):
    """One entry in a run's conversation.

    A step knows its own two renderings and its own serialised form. The
    ``kind`` string is stable, is what :meth:`AgentThread.from_dict` dispatches
    on, and is what a log line or a grep over a stored run matches.
    """

    kind: str

    def to_text(self) -> str:
        """The step's fragment of the flat transcript, newline included."""
        ...

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """The step as provider messages; empty when it carries none of its own."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data, carrying its ``kind``."""
        ...


def _coerce_arguments(raw: str) -> dict[str, Any]:
    """Read a rendered ``Action Input`` back into an arguments mapping."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {_RAW_INPUT_KEY: raw}
    if isinstance(parsed, dict):
        return parsed
    return {_RAW_INPUT_KEY: raw}


def _text_parts(text: str) -> list[ContentPart]:
    """A one-part content list, or an empty one when there is no text."""
    return [TextPart(text=text)] if text else []


@dataclass
class SystemStep:
    """Instruction the run is framed by: a persona, a contract, a format spec.

    Renders as nothing in the flat transcript — the frame is assembled around
    the transcript, not inside it — and as a ``system`` message otherwise.
    """

    text: str
    source: Literal["persona", "contract", "format"] = "persona"
    kind: str = field(default="system", init=False)

    def to_text(self) -> str:
        """The empty string: framing is not transcript."""
        return ""

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One ``system`` message carrying the instruction."""
        return [Message(role=Role.SYSTEM, content=_text_parts(self.text))]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data."""
        return {"kind": self.kind, "text": self.text, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemStep:
        """Rebuild the step from :meth:`to_dict` output."""
        return cls(text=data.get("text", ""), source=data.get("source", "persona"))


@dataclass
class TaskStep:
    """What the caller asked, with any non-text parts it arrived with.

    Renders as nothing in the flat transcript — the frame carries the question
    — and as one ``user`` message otherwise, images and audio included.
    """

    text: str
    parts: list[ContentPart] = field(default_factory=list)
    kind: str = field(default="task", init=False)

    def to_text(self) -> str:
        """The empty string: the question belongs to the frame."""
        return ""

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One ``user`` message: the text, then any other content parts."""
        return [Message(role=Role.USER, content=_text_parts(self.text) + list(self.parts))]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data. Non-text parts are kept by type name."""
        return {
            "kind": self.kind,
            "text": self.text,
            "parts": [_part_to_dict(part) for part in self.parts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskStep:
        """Rebuild the step from :meth:`to_dict` output."""
        return cls(
            text=data.get("text", ""),
            parts=[_part_from_dict(part) for part in data.get("parts", []) or []],
        )


@dataclass
class TurnStep:
    """One message from earlier in the session, before this run started.

    Renders as nothing in the flat transcript — an earlier turn is not a step
    of *this* run — and as the ``user`` or ``assistant`` message it was
    otherwise, so a conversation reaches the model as turns instead of as a
    block of text pasted into the current question.
    :meth:`AgentThread.history_text` is the rendering for a frame that can only
    take one string.
    """

    text: str
    role: Literal["user", "assistant"] = "user"
    parts: list[ContentPart] = field(default_factory=list)
    kind: str = field(default="turn", init=False)

    def to_text(self) -> str:
        """The empty string: earlier turns are framing, not this run's steps."""
        return ""

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One message in the role the turn was spoken in."""
        role = Role.ASSISTANT if self.role == "assistant" else Role.USER
        return [Message(role=role, content=_text_parts(self.text) + list(self.parts))]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data. Non-text parts are kept by type name."""
        return {
            "kind": self.kind,
            "text": self.text,
            "role": self.role,
            "parts": [_part_to_dict(part) for part in self.parts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnStep:
        """Rebuild the step from :meth:`to_dict` output."""
        role: Literal["user", "assistant"] = (
            "assistant" if data.get("role") == "assistant" else "user"
        )
        return cls(
            text=data.get("text", ""),
            role=role,
            parts=[_part_from_dict(part) for part in data.get("parts", []) or []],
        )


@dataclass
class ThoughtStep:
    """The model's own reasoning for a turn.

    A turn that made a provider-native tool call reports no thought, and the
    transcript is text the model reads back — so an absent thought renders as
    the bare label and an empty line, never as the word ``None``.
    """

    text: str = ""
    kind: str = field(default="thought", init=False)

    def to_text(self) -> str:
        """``"\\nThought: {text}"``, with the label kept when the text is empty."""
        return f"\nThought: {self.text}"

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One ``assistant`` message, or none when there is no reasoning.

        A thought immediately followed by an action is folded into that
        action's message by :meth:`AgentThread.to_messages`, so this rendering
        applies to a thought that stands on its own.
        """
        if summary or not self.text:
            return []
        return [Message(role=Role.ASSISTANT, content=_text_parts(self.text))]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data."""
        return {"kind": self.kind, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThoughtStep:
        """Rebuild the step from :meth:`to_dict` output."""
        return cls(text=data.get("text", ""))


@dataclass
class ActionStep:
    """A tool the turn asked for, with the arguments it asked for.

    ``raw`` is the ``Action Input`` text exactly as the turn wrote it, and it
    is what :meth:`to_text` renders, so a transcript rebuilt from steps is the
    transcript the turn produced. ``arguments`` is the same input as a mapping,
    which is what a provider's tool-call protocol needs.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    reasoning: str = ""
    raw: str = ""
    kind: str = field(default="action", init=False)

    def __post_init__(self) -> None:
        if self.raw and not self.arguments:
            self.arguments = _coerce_arguments(self.raw)
        elif self.arguments and not self.raw:
            self.raw = json.dumps(self.arguments)

    def to_text(self) -> str:
        """``"\\nAction: {tool}\\nAction Input: {raw}"``."""
        return f"\nAction: {self.tool}\nAction Input: {self.raw}"

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One ``assistant`` message carrying the reasoning **and** the call.

        The reasoning travels as a :class:`~effgen.core.messages.TextPart` on
        the same message as the :class:`~effgen.core.messages.ToolCallPart`.
        Dropping it when the call is present loses the model's own account of
        why it called what it called.

        A step in a thread carries its own :attr:`call_id`, so this renders the
        same id :meth:`AgentThread.to_messages` gives it. A step that has never
        joined a thread has no position to mint from and renders under the id a
        thread would give a step at its start.
        """
        call_id = self.call_id or f"{_LOCAL_CALL_ID_PREFIX}0"
        content: list[ContentPart] = _text_parts(self.reasoning)
        content.append(
            ToolCallPart(tool_call_id=call_id, name=self.tool, arguments=dict(self.arguments))
        )
        return [Message(role=Role.ASSISTANT, content=content)]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data."""
        return {
            "kind": self.kind,
            "tool": self.tool,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "reasoning": self.reasoning,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionStep:
        """Rebuild the step from :meth:`to_dict` output."""
        return cls(
            tool=data.get("tool", ""),
            arguments=dict(data.get("arguments") or {}),
            call_id=data.get("call_id"),
            reasoning=data.get("reasoning", ""),
            raw=data.get("raw", ""),
        )


@dataclass
class ObservationStep:
    """What a tool returned for the call before it.

    Attributes:
        text: The reply the model reads.
        call_id: The call this answers.
        is_error: Whether the reply reports a failure.
        declined: Why the framework answered in its own words rather than a
            tool's — ``"loop_detected"`` and ``"already_computed"`` for a call
            the run stopped making, ``"unknown_tool"`` for a tool the agent
            does not hold. ``None`` when the reply is a tool's own result,
            including a repeat answered from the record.
        compacted: How ``text`` came to be shorter than what the tool returned
            — ``"elided"`` when its opening was kept, ``"summarized"`` when it
            was rewritten. ``None`` when the step holds the whole reply.
        original_chars: What the reply measured before it was shortened; 0
            when it was not.

    A shortened result is a flag on this step rather than a step kind of its
    own, so a reader that does not know the flag still reads the step as the
    observation it is and still sees the text the model was sent.
    """

    text: str
    call_id: str | None = None
    is_error: bool = False
    declined: str | None = None
    compacted: str | None = None
    original_chars: int = 0
    kind: str = field(default="observation", init=False)

    def to_text(self) -> str:
        """``"\\nObservation: {text}"``."""
        return f"\nObservation: {self.text}"

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One ``tool`` message answering the call id it carries.

        A step in a thread carries the id of the call before it, so this
        renders the same id :meth:`AgentThread.to_messages` gives it. A step
        that has never joined a thread renders under the id a thread would give
        a result at its start.
        """
        call_id = self.call_id or f"{_LOCAL_CALL_ID_PREFIX}0"
        return [
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultPart(
                        tool_call_id=call_id, result=self.text, is_error=self.is_error
                    )
                ],
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data."""
        return {
            "kind": self.kind,
            "text": self.text,
            "call_id": self.call_id,
            "is_error": self.is_error,
            "declined": self.declined,
            "compacted": self.compacted,
            "original_chars": self.original_chars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationStep:
        """Rebuild the step from :meth:`to_dict` output."""
        return cls(
            text=data.get("text", ""),
            call_id=data.get("call_id"),
            is_error=bool(data.get("is_error", False)),
            declined=data.get("declined"),
            compacted=data.get("compacted"),
            original_chars=int(data.get("original_chars") or 0),
        )


@dataclass
class NudgeStep:
    """A line the framework injected, not something the model or a tool said.

    The loop steers a stalling run by writing into the transcript. Recording
    those lines as their own step keeps them attributable: a reader of the
    thread can tell what the framework said from what the run produced, which
    a flat transcript cannot.

    ``render_as`` reproduces where the injection site put the line: ``"raw"``
    writes it on its own line, ``"observation"`` writes it as an observation.
    """

    text: str
    render_as: Literal["raw", "observation"] = "raw"
    nudge_id: str = ""
    kind: str = field(default="nudge", init=False)

    def to_text(self) -> str:
        """``"\\n{text}"``, or ``"\\nObservation: {text}"`` when so rendered."""
        if self.render_as == "observation":
            return f"\nObservation: {self.text}"
        return f"\n{self.text}"

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One ``user`` turn, marked as the framework's own words."""
        if summary:
            return []
        return [
            Message(
                role=Role.USER,
                content=_text_parts(self.text),
                metadata={"effgen_nudge": self.nudge_id or True},
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data."""
        return {
            "kind": self.kind,
            "text": self.text,
            "render_as": self.render_as,
            "nudge_id": self.nudge_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NudgeStep:
        """Rebuild the step from :meth:`to_dict` output."""
        return cls(
            text=data.get("text", ""),
            render_as=data.get("render_as", "raw"),
            nudge_id=data.get("nudge_id", ""),
        )


@dataclass
class AnswerStep:
    """How the run ended: the answer it reached, or why it stopped without one.

    Renders as nothing in the flat transcript — the answer is what the run
    returns, not text the model is shown again — and as one ``assistant``
    message otherwise.
    """

    text: str
    stop_reason: str | None = None
    kind: str = field(default="answer", init=False)

    def to_text(self) -> str:
        """The empty string: an answer ends a run rather than extending it."""
        return ""

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """One ``assistant`` message carrying the answer."""
        return [Message(role=Role.ASSISTANT, content=_text_parts(self.text))]

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data."""
        return {"kind": self.kind, "text": self.text, "stop_reason": self.stop_reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerStep:
        """Rebuild the step from :meth:`to_dict` output."""
        return cls(text=data.get("text", ""), stop_reason=data.get("stop_reason"))


class _StepReader(Protocol):
    """What a step type has to offer for a stored step to be read back."""

    def from_dict(self, data: dict[str, Any]) -> Step:
        """Rebuild one step of this kind."""
        ...


#: ``kind`` to the type that reads it back. One registry, so a stored thread
#: names its steps in the same vocabulary a grep or a log line uses.
_STEP_TYPES: dict[str, _StepReader] = {
    "system": SystemStep,
    "task": TaskStep,
    "turn": TurnStep,
    "thought": ThoughtStep,
    "action": ActionStep,
    "observation": ObservationStep,
    "nudge": NudgeStep,
    "answer": AnswerStep,
}


#: Wrapper the serialised form uses for a field that holds raw bytes, so a
#: thread carrying an image is still plain data a JSON writer accepts.
_BYTES_KEY = "__bytes_b64__"


def _encode(value: Any) -> Any:
    """Make one field value JSON-writable, keeping raw bytes recoverable."""
    if isinstance(value, bytes | bytearray):
        return {_BYTES_KEY: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any) -> Any:
    """Undo :func:`_encode`."""
    if isinstance(value, dict):
        if set(value) == {_BYTES_KEY}:
            return base64.b64decode(value[_BYTES_KEY])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _part_to_dict(part: ContentPart) -> dict[str, Any]:
    """Serialise a content part by its own fields."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(part) and not isinstance(part, type):
        return {key: _encode(value) for key, value in asdict(part).items()}
    return {"type": "text", "text": str(part)}


def _part_from_dict(data: dict[str, Any]) -> ContentPart:
    """Read a content part back, falling back to text for unknown types."""
    from . import messages as _messages

    part_type = str(data.get("type", "text"))
    class_name = {
        "text": "TextPart",
        "image": "ImagePart",
        "audio": "AudioPart",
        "video": "VideoPart",
        "tool_call": "ToolCallPart",
        "tool_result": "ToolResultPart",
    }.get(part_type)
    cls: type[ContentPart] | None = getattr(_messages, class_name, None) if class_name else None
    if cls is None:
        return TextPart(text=str(data.get("text", "")))
    fields = {k: _decode(v) for k, v in data.items() if k != "type"}
    try:
        part: ContentPart = cls(**fields)
    except TypeError:
        return TextPart(text=str(data.get("text", "")))
    return part


def step_from_dict(data: dict[str, Any]) -> Step:
    """Rebuild one step from :meth:`Step.to_dict` output.

    Args:
        data: A mapping carrying a known ``kind``.

    Returns:
        The step that ``kind`` names.

    Raises:
        ValueError: If ``kind`` is missing or is not a known step kind.
    """
    kind = data.get("kind")
    cls = _STEP_TYPES.get(str(kind))
    if cls is None:
        raise ValueError(
            f"step kind {kind!r} is not one this release knows. "
            f"Use one of {sorted(_STEP_TYPES)}, or upgrade effGen to the release "
            f"that wrote this thread."
        )
    step: Step = cls.from_dict(data)
    return step


# The markers a flat transcript is written with. Used only by
# :meth:`AgentThread.from_scratchpad`, which recovers what it can from text
# written before a run's steps were kept.
_FLAT_MARKER_RE = re.compile(
    r"^(Thought|Action Input|Action|Observation):[ ]?(.*)$",
)

#: Label :meth:`AgentThread.from_scratchpad` gives a line carrying no marker.
_RECOVERED_RAW = "__raw__"


@dataclass
class AgentThread:
    """The conversation of one run, as typed steps.

    Args:
        steps: The run's steps, in the order they happened.
        version: The serialisation version this thread was built at.
        metadata: Anything a caller wants to travel with the thread.
    """

    steps: list[Step] = field(default_factory=list)
    version: int = THREAD_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._settle_call_ids(0)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def append(self, step: Step) -> None:
        """Add one step to the end of the thread.

        A call that carried no id of its own is given one here, from its
        position, so the step names the same call however it is rendered.
        """
        start = len(self.steps)
        self.steps.append(step)
        self._settle_call_ids(start)

    def extend(self, steps: list[Step]) -> None:
        """Add several steps, in order, settling any call id they lack."""
        start = len(self.steps)
        self.steps.extend(steps)
        self._settle_call_ids(start)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """The flat transcript: every step's text, concatenated and nothing else.

        Returns:
            The transcript string, which a prompt template receives as its
            ``{scratchpad}`` field.
        """
        return "".join(step.to_text() for step in self.steps)

    def to_messages(self, *, summary: bool = False) -> list[Message]:
        """The conversation as provider messages.

        Every :class:`SystemStep` is merged into a single leading ``system``
        message, so the list carries exactly one however many instructions the
        run was framed by. A :class:`ThoughtStep` immediately followed by an
        :class:`ActionStep` is folded into that action's assistant message, so
        the reasoning and the tool call travel together. Every
        :class:`ObservationStep` answers the call id of the action before it,
        minting one when the turn carried none.

        Args:
            summary: Render a shortened form — nudges and standalone thoughts
                are dropped and observations are abbreviated. Useful when an
                older part of a long run has to be shown compactly.

        Returns:
            The messages, in order, with the system message first.
        """
        call_ids = self._call_ids()
        system_text = self.system_text()
        messages: list[Message] = []
        if system_text:
            messages.append(
                Message(role=Role.SYSTEM, content=[TextPart(text=system_text)])
            )

        pending_thought: str | None = None
        for index, step in enumerate(self.steps):
            if isinstance(step, SystemStep):
                continue
            if isinstance(step, ThoughtStep):
                nxt = self.steps[index + 1] if index + 1 < len(self.steps) else None
                if isinstance(nxt, ActionStep):
                    pending_thought = step.text
                    continue
                messages.extend(step.to_messages(summary=summary))
                continue
            if isinstance(step, ActionStep):
                reasoning = step.reasoning or pending_thought or ""
                pending_thought = None
                content: list[ContentPart] = _text_parts(reasoning)
                content.append(
                    ToolCallPart(
                        tool_call_id=call_ids[index],
                        name=step.tool,
                        arguments=dict(step.arguments),
                    )
                )
                messages.append(Message(role=Role.ASSISTANT, content=content))
                continue
            if isinstance(step, ObservationStep):
                text = step.text
                if summary and len(text) > _SUMMARY_OBSERVATION_CHARS:
                    text = text[:_SUMMARY_OBSERVATION_CHARS] + "…"
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultPart(
                                tool_call_id=call_ids[index],
                                result=text,
                                is_error=step.is_error,
                            )
                        ],
                    )
                )
                continue
            messages.extend(step.to_messages(summary=summary))
        return messages

    def render(self, protocol: Literal["flat", "messages"]) -> str | list[Message]:
        """Render the thread in one of the two protocols.

        Args:
            protocol: ``"flat"`` for the transcript string, ``"messages"`` for
                the message list.

        Returns:
            The rendering that ``protocol`` names.

        Raises:
            ValueError: If ``protocol`` is neither of the two.
        """
        if protocol == "flat":
            return self.to_text()
        if protocol == "messages":
            return self.to_messages()
        raise ValueError(
            f"{protocol!r} is not a rendering this thread has. "
            f"Pass 'flat' for the transcript string, or 'messages' for the "
            f"provider message list."
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The thread as plain data, carrying its schema version."""
        return {
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentThread:
        """Rebuild a thread from :meth:`to_dict` output.

        A thread written by a newer release is read as far as its steps are
        recognised; an unknown step kind raises rather than being dropped
        silently, because a transcript missing a step is not the transcript.

        Args:
            data: A mapping as :meth:`to_dict` produces.

        Returns:
            The thread.
        """
        version = int(data.get("version", THREAD_SCHEMA_VERSION))
        if version > THREAD_SCHEMA_VERSION:
            logger.warning(
                "[thread] reading schema version %s with a reader built for %s",
                version,
                THREAD_SCHEMA_VERSION,
            )
        steps = [step_from_dict(entry) for entry in data.get("steps", []) or []]
        return cls(steps=steps, version=version, metadata=dict(data.get("metadata") or {}))

    @classmethod
    def from_scratchpad(cls, text: str) -> AgentThread:
        """Recover what a flat transcript still carries, as steps.

        This is a lossy reading, used for a transcript stored before a run's
        steps were kept: a tool call id, the type of a tool's arguments and
        whether a line was injected by the framework are not in the text and do
        not come back. What does come back is the order of thoughts, calls,
        rendered inputs and observations, and — as an injected line, which is
        what such a line usually is — every line carrying no marker at all. No
        line is dropped, so re-rendering the recovered thread gives back the
        text it was read from.

        Args:
            text: A transcript as :meth:`to_text` renders one.

        Returns:
            The recovered thread.
        """
        steps: list[Step] = []
        pending_action: str | None = None
        current: tuple[str, list[str]] | None = None

        def close_pending_action() -> None:
            """An ``Action`` no ``Action Input`` followed is a line, not a call."""
            nonlocal pending_action
            if pending_action is not None:
                steps.append(NudgeStep(text=f"Action: {pending_action}", render_as="raw"))
                pending_action = None

        def flush() -> None:
            nonlocal current, pending_action
            if current is None:
                return
            label, lines = current
            body = "\n".join(lines)
            current = None
            if label == "Action Input":
                if pending_action is None:
                    steps.append(NudgeStep(text=f"Action Input: {body}", render_as="raw"))
                else:
                    steps.append(ActionStep(tool=pending_action, raw=body))
                    pending_action = None
                return
            close_pending_action()
            if label == "Thought":
                steps.append(ThoughtStep(text=body))
            elif label == "Action":
                pending_action = body
            elif label == "Observation":
                steps.append(ObservationStep(text=body))
            else:
                steps.append(NudgeStep(text=body, render_as="raw"))

        lines = (text or "").split("\n")
        if lines and lines[0] == "":
            # Every step's text opens with the newline that separates it from
            # the one before, so the transcript's leading newline is not a line.
            lines = lines[1:]
        for line in lines:
            match = _FLAT_MARKER_RE.match(line)
            if match:
                flush()
                current = (match.group(1), [match.group(2)])
            elif current is not None:
                current[1].append(line)
            else:
                # Not a marker and nothing to attach it to: the framework's own
                # words. Keeping it is what makes the reading render the text it
                # was given rather than a shorter one.
                current = (_RECOVERED_RAW, [line])
        flush()
        close_pending_action()
        logger.info("[thread] recovered %d steps from a flat transcript", len(steps))
        return cls(steps=steps)

    # ------------------------------------------------------------------
    # The frame the run is asked in
    # ------------------------------------------------------------------

    def system_text(self) -> str:
        """Every :class:`SystemStep`\'s text, in order, as one instruction.

        The persona, the tool contract and any format specification a run was
        framed by. Empty when the run was framed by none of them, which is what
        an agent left on the default persona with no tools reports.

        Returns:
            The instruction, or ``""``.
        """
        return "\n\n".join(
            step.text for step in self.steps
            if isinstance(step, SystemStep) and step.text
        )

    def persona_text(self) -> str:
        """The caller's own persona, or ``""``.

        A subset of :meth:`system_text`: only the steps a caller's
        ``system_prompt`` put there, not the tool contract or a format
        specification the framework added. A run carrying one has something to
        say in a system turn that a text frame cannot express.

        Returns:
            The persona, or ``""``.
        """
        return "\n\n".join(
            step.text for step in self.steps
            if isinstance(step, SystemStep) and step.source == "persona" and step.text
        )

    def prior_turns(self) -> list[TurnStep]:
        """The turns that happened before this run, in order.

        Returns:
            The session's earlier messages, or an empty list for a first turn.
        """
        return [step for step in self.steps if isinstance(step, TurnStep)]

    def task(self) -> TaskStep | None:
        """The question this run is answering, with any parts it arrived with.

        Returns:
            The task step, or ``None`` for a thread that carries no frame —
            one recovered from a flat transcript, for instance.
        """
        for step in self.steps:
            if isinstance(step, TaskStep):
                return step
        return None

    def history_text(self) -> str:
        """The earlier turns as text, for a frame that can only take a string.

        The message rendering sends each turn in the role it was spoken in;
        this is what a prompt template's conversation-history field receives
        instead, when the model is reached with one string. An earlier reply is
        shortened to :data:`_HISTORY_REPLY_CHARS`, because a text frame
        restates the whole history on every turn.

        Returns:
            The rendering, or ``""`` when the run has no earlier turns.
        """
        turns = self.prior_turns()
        if not turns:
            return ""
        lines = [_HISTORY_HEADER, ""]
        for turn in turns:
            text = turn.text
            if turn.role == "assistant" and len(text) > _HISTORY_REPLY_CHARS:
                text = text[:_HISTORY_REPLY_CHARS] + "..."
            speaker = "Assistant" if turn.role == "assistant" else "User"
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Reading the run back — what a regex over the transcript used to do
    # ------------------------------------------------------------------

    def observations(self) -> list[ObservationStep]:
        """Every observation in the run, in order."""
        return [step for step in self.steps if isinstance(step, ObservationStep)]

    def last_observation(self) -> ObservationStep | None:
        """The most recent observation, or ``None`` if no tool ever returned."""
        observations = self.observations()
        return observations[-1] if observations else None

    def thoughts(self) -> list[ThoughtStep]:
        """Every thought in the run, in order."""
        return [step for step in self.steps if isinstance(step, ThoughtStep)]

    def last_thought(self) -> ThoughtStep | None:
        """The most recent non-empty thought, or ``None`` if there was none."""
        for step in reversed(self.steps):
            if isinstance(step, ThoughtStep) and step.text.strip():
                return step
        return None

    def actions(self) -> list[ActionStep]:
        """Every tool call the run asked for, in order."""
        return [step for step in self.steps if isinstance(step, ActionStep)]

    def unanswered_calls(self) -> list[ActionStep]:
        """Calls with no observation answering them.

        Returns:
            The actions whose call id no observation carries, in order. A
            provider rejects a conversation that holds one, so an empty list is
            the invariant a message-protocol run has to keep.
        """
        call_ids = self._call_ids()
        answered = {
            call_ids[index]
            for index, step in enumerate(self.steps)
            if isinstance(step, ObservationStep)
        }
        return [
            step
            for index, step in enumerate(self.steps)
            if isinstance(step, ActionStep) and call_ids[index] not in answered
        ]

    def unanswered_call_ids(self) -> list[str]:
        """The call ids no observation in this thread answers.

        The ids themselves rather than the steps, because a call the turn
        carried no id for is answered by one minted from its position, which a
        step on its own cannot say.

        Returns:
            The unanswered ids, in order, each appearing once.
        """
        ids = self._call_ids()
        answered = {
            ids[index]
            for index, step in enumerate(self.steps)
            if isinstance(step, ObservationStep)
        }
        seen: set[str] = set()
        unanswered: list[str] = []
        for index, step in enumerate(self.steps):
            if not isinstance(step, ActionStep):
                continue
            call_id = ids[index]
            if call_id in answered or call_id in seen:
                continue
            seen.add(call_id)
            unanswered.append(call_id)
        return unanswered

    def partial_answer(self) -> str | None:
        """What the run had reached when it stopped without answering.

        Framework-injected lines are not candidates — they are the framework's
        words, not the run's — and neither are errors. Repeated results are
        collapsed, because a run that loops on one tool produces the same
        passage repeatedly and joining the copies answers nothing.

        Returns:
            The observations the run collected, joined with ``" | "`` when
            there is more than one; failing that the last substantive thought;
            failing that ``None``.
        """
        seen: set[str] = set()
        collected: list[str] = []
        for step in self.observations():
            if step.is_error:
                continue
            text = step.text.strip()
            if not text or text.lower().startswith("error"):
                continue
            if text in seen:
                continue
            seen.add(text)
            collected.append(text)
        if collected:
            return " | ".join(collected)
        thought = self.last_thought()
        if thought is not None and len(thought.text.strip()) > _SUBSTANTIVE_THOUGHT_CHARS:
            return thought.text.strip()
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _settle_call_ids(self, start: int) -> None:
        """Give every call from *start* on an id, and every result the id it answers.

        A turn that made a provider-native call arrives carrying the provider's
        own id. A turn that wrote its call as text carries none, so the thread
        mints one from the step's position — the same id a replay of the same
        run mints, and the same id whether the step is rendered on its own or
        as part of the thread.

        Args:
            start: The first index to settle; the steps before it already are.
        """
        current = f"{_LOCAL_CALL_ID_PREFIX}0"
        for step in reversed(self.steps[:start]):
            if isinstance(step, ActionStep | ObservationStep) and step.call_id:
                current = step.call_id
                break
        for index in range(start, len(self.steps)):
            step = self.steps[index]
            if isinstance(step, ActionStep):
                if not step.call_id:
                    step.call_id = f"{_LOCAL_CALL_ID_PREFIX}{index}"
                current = step.call_id
            elif isinstance(step, ObservationStep):
                if not step.call_id:
                    step.call_id = current
                else:
                    current = step.call_id

    def _call_ids(self) -> list[str]:
        """The call id in force at each step index.

        Every call and every result carries its own id by the time it is in a
        thread (:meth:`_settle_call_ids`); this reads the id in force at each
        index so a step between a call and its result can be attributed too.
        """
        ids: list[str] = []
        current = f"{_LOCAL_CALL_ID_PREFIX}0"
        for index, step in enumerate(self.steps):
            if isinstance(step, ActionStep):
                current = step.call_id or f"{_LOCAL_CALL_ID_PREFIX}{index}"
            elif isinstance(step, ObservationStep) and step.call_id:
                current = step.call_id
            ids.append(current)
        return ids

