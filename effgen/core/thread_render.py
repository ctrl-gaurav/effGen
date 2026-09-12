"""Render a run's conversation as steps, stably enough to diff two runs.

A run's conversation is a list of typed steps
(:class:`~effgen.core.thread.AgentThread`). Reading it back means seeing what
the model was framed by, what it was asked, what it reasoned, which tools it
called with which arguments and what came back — in the order it happened,
rather than as one flattened transcript string.

Two properties make this rendering worth having:

*Nothing varies between two runs that took the same path.* No timestamp, no
run id and no provider-minted tool-call id appears unless the caller asks for
ids with ``include_ids=True``, so ``diff`` on two renderings shows what the two
runs actually did differently.

*Secrets do not travel.* Every rendered string goes through the shared
:mod:`effgen.observability.redact` scrubber by default, so a provider key that
reached a tool's output or an instruction does not reach a report, a terminal
or a shared HTML card.

Usage::

    from effgen.core.thread_render import render_thread, thread_as_text

    response = agent.run("what is 6 * 7?")
    print(thread_as_text(response.thread))

    for step in render_thread(response.thread):
        print(step.position, step.kind, step.label)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .thread import AgentThread

__all__ = ["RenderedStep", "render_thread", "thread_as_text"]

#: Indent one nesting level — a delegated child's steps sit under the parent's.
_INDENT = "    "

#: What a body line is prefixed with, so a step's text cannot be mistaken for
#: the next step's heading in a diff.
_BODY = "| "


@dataclass(frozen=True)
class RenderedStep:
    """One step of a conversation, ready to print.

    Attributes:
        position: The step's 1-based place in the thread it came from. A
            delegated child's steps are numbered within the child's own thread.
        kind: The step kind, as :meth:`effgen.core.thread.Step.to_dict` reports
            it — ``"system"``, ``"task"``, ``"turn"``, ``"thought"``,
            ``"action"``, ``"observation"``, ``"nudge"``, ``"answer"`` or
            ``"delegation"``.
        label: A short heading for the step, naming the tool a call asked for
            or the role a turn was spoken in.
        body: The step's own text, redacted unless the caller turned that off,
            and truncated when a limit was given.
        detail: Anything else worth showing that is stable across runs — a
            failed observation's flag, a compacted observation's original
            length, a delegation's outcome. Values are strings so a renderer
            can print them without knowing what they are.
        depth: How far the step is nested. ``0`` for the run's own steps, ``1``
            for the steps of a child it delegated to, and so on.
    """

    position: int
    kind: str
    label: str
    body: str = ""
    detail: dict[str, str] = field(default_factory=dict)
    depth: int = 0

    def to_text(self) -> str:
        """The step as printable lines: a heading, then its body, indented."""
        pad = _INDENT * self.depth
        head = f"{pad}{self.position:>3}. {self.label}"
        if self.detail:
            head += "  " + " ".join(f"{k}={v}" for k, v in self.detail.items())
        if not self.body:
            return head
        body = "\n".join(
            f"{pad}     {_BODY}{line}" for line in self.body.splitlines() or [""]
        )
        return f"{head}\n{body}"

    def to_dict(self) -> dict[str, Any]:
        """The step as plain data, for a JSON document or a template."""
        return {
            "position": self.position,
            "kind": self.kind,
            "label": self.label,
            "body": self.body,
            "detail": dict(self.detail),
            "depth": self.depth,
        }


def _thread_of(source: Any) -> AgentThread | None:
    """Read *source* as a thread, whether it arrived as one or as its data."""
    if source is None:
        return None
    if isinstance(source, AgentThread):
        return source
    if isinstance(source, dict):
        try:
            return AgentThread.from_dict(source)
        except (ValueError, TypeError, KeyError):
            return None
    if hasattr(source, "steps"):
        thread: AgentThread = source
        return thread
    return None


def _clip(text: str, limit: int | None) -> str:
    """*text*, shortened to *limit* characters with an ellipsis when it is longer."""
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _attachments(parts: list[Any]) -> str:
    """The non-text content a question arrived with, named by type."""
    kinds = [str(getattr(part, "type", "")) for part in parts]
    return ", ".join(k for k in kinds if k and k != "text")


def _arguments_text(step: Any) -> str:
    """A call's input exactly as the turn wrote it, or its arguments as data.

    The raw text is preferred because it is what the model produced; a
    provider-native call carries no raw text, and its arguments are rendered
    with their keys sorted so two runs that passed the same input render the
    same bytes.
    """
    raw = str(getattr(step, "raw", "") or "")
    if raw:
        return raw
    arguments = dict(getattr(step, "arguments", None) or {})
    if not arguments:
        return ""
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(sorted(arguments.items()))


def _describe(step: Any, *, include_ids: bool) -> tuple[str, str, dict[str, str]]:
    """The label, body and stable detail for one step."""
    kind = str(getattr(step, "kind", "") or "step")
    detail: dict[str, str] = {}

    if kind == "system":
        return f"system ({getattr(step, 'source', 'persona')})", str(step.text), detail
    if kind == "task":
        attached = _attachments(list(getattr(step, "parts", None) or []))
        if attached:
            detail["attachments"] = attached
        return "task", str(step.text), detail
    if kind == "turn":
        return f"turn ({getattr(step, 'role', 'user')})", str(step.text), detail
    if kind == "thought":
        return "thought", str(step.text), detail
    if kind == "action":
        if include_ids and getattr(step, "call_id", None):
            detail["call_id"] = str(step.call_id)
        return f"action ({getattr(step, 'tool', '')})", _arguments_text(step), detail
    if kind == "observation":
        label = "observation (error)" if getattr(step, "is_error", False) else "observation"
        if getattr(step, "declined", None):
            detail["declined"] = str(step.declined)
        if getattr(step, "compacted", None):
            detail["compacted"] = str(step.compacted)
            detail["original_chars"] = str(getattr(step, "original_chars", 0))
        if include_ids and getattr(step, "call_id", None):
            detail["call_id"] = str(step.call_id)
        return label, str(step.text), detail
    if kind == "nudge":
        return f"nudge ({getattr(step, 'render_as', 'raw')})", str(step.text), detail
    if kind == "answer":
        if getattr(step, "stop_reason", None):
            detail["stop_reason"] = str(step.stop_reason)
        return "answer", str(step.text), detail
    if kind == "delegation":
        detail["outcome"] = "ok" if getattr(step, "success", True) else "failed"
        if getattr(step, "error", None):
            detail["error"] = str(step.error)
        label = f"delegation ({getattr(step, 'role', '') or 'child'}:{step.child_id})"
        return label, str(getattr(step, "task", "") or ""), detail

    return kind, str(getattr(step, "text", "") or ""), detail


def render_thread(
    source: Any,
    *,
    redact: bool = True,
    include_ids: bool = False,
    max_chars: int | None = None,
    depth: int = 0,
) -> list[RenderedStep]:
    """Render a conversation as steps.

    Args:
        source: The conversation — an :class:`~effgen.core.thread.AgentThread`,
            or the mapping :meth:`~effgen.core.thread.AgentThread.to_dict`
            produces. Anything else renders as no steps.
        redact: Pass every rendered string through the shared secret scrubber.
            On by default: a rendering is meant to be read, saved and shared.
        include_ids: Carry tool-call ids in ``detail``. Off by default, because
            a provider mints a fresh id per call and two runs that did the same
            work would otherwise not compare equal.
        max_chars: Shorten each step's body to this many characters. ``None``
            keeps the whole text.
        depth: The nesting level to render at. Callers leave this at ``0``; a
            delegated child's steps are rendered one level deeper.

    Returns:
        The steps, in order, with a delegated child's own steps following the
        delegation that produced them.
    """
    thread = _thread_of(source)
    if thread is None:
        return []

    scrub = None
    if redact:
        from ..observability.redact import get_redactor

        scrub = get_redactor().scrub

    rendered: list[RenderedStep] = []
    for position, step in enumerate(thread.steps, start=1):
        label, body, detail = _describe(step, include_ids=include_ids)
        if scrub is not None:
            label = scrub(label)
            body = scrub(body)
            detail = {key: scrub(value) for key, value in detail.items()}
        rendered.append(RenderedStep(
            position=position,
            kind=str(getattr(step, "kind", "") or "step"),
            label=label,
            body=_clip(body, max_chars),
            detail=detail,
            depth=depth,
        ))
        child = getattr(step, "thread", None)
        if str(getattr(step, "kind", "")) == "delegation" and child is not None:
            rendered.extend(render_thread(
                child,
                redact=redact,
                include_ids=include_ids,
                max_chars=max_chars,
                depth=depth + 1,
            ))
    return rendered


def thread_as_text(
    source: Any,
    *,
    redact: bool = True,
    include_ids: bool = False,
    max_chars: int | None = None,
) -> str:
    """The conversation as one block of text, one heading and body per step.

    Two runs that took the same path render the same bytes, so ``diff`` on two
    of these shows what the runs did differently rather than when they ran.

    Args:
        source: The conversation, as :func:`render_thread` accepts it.
        redact: Pass every rendered string through the secret scrubber.
        include_ids: Carry tool-call ids, which differ between two runs.
        max_chars: Shorten each step's body to this many characters.

    Returns:
        The rendering, with no trailing newline. The empty string when there is
        no conversation to render.
    """
    steps = render_thread(
        source, redact=redact, include_ids=include_ids, max_chars=max_chars
    )
    return "\n".join(step.to_text() for step in steps)
