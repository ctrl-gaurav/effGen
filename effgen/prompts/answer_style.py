"""What the framework tells a model about the *form* of the answer.

Attaching tools, pasting an observation block and restating a format are all
things the framework does to a prompt, and each of them ends up saying
something about how long the answer should be. Said in four places they
disagree; said once, last, in a sentence that names no form, they do not.

That sentence is what this module holds. It is the only text the framework puts
after the caller's own task, so the caller's last word stays the model's last
word on *what* to answer while the framework states only *how much*. Everything
else the framework adds — the tool list, the contract for those tools, the
generated persona — is read before the task.

Two named styles are shorthand, and any other string is stated verbatim:

``"brief"``
    "Answer in the form the question asks for, and nothing else." It names no
    form — not a letter, not a number, not a sentence — and reads nothing from
    the task. The form is the question's to state and the caller's to pin; this
    only says not to add to it.
``"full"``
    "Explain your reasoning in the answer." The way back for a caller who wants
    the working shown in the answer rather than thrown away.

The style is resolved per call, then from ``AgentConfig.answer_style``, then
from :data:`DEFAULT_ANSWER_STYLE` — the same order every other generation
setting uses — and a child run inherits it from its parent's configuration like
any other field.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Ask for the answer the question asked for and nothing besides it.
ANSWER_STYLE_BRIEF = "brief"
#: Ask for the reasoning to be part of the answer.
ANSWER_STYLE_FULL = "full"

#: The text each named style states. A style that is not a name here is a
#: caller's own sentence and is stated exactly as written.
ANSWER_STYLE_TEXTS: dict[str, str] = {
    ANSWER_STYLE_BRIEF: (
        "Answer in the form the question asks for, and nothing else."
    ),
    ANSWER_STYLE_FULL: "Explain your reasoning in the answer.",
}

#: The style a run states when neither the call nor the agent named one.
#:
#: ``None`` states nothing, which is the prompt every run sent before this
#: existed. It is the shipped default, and it is measured rather than assumed:
#: stating the line by default was run over ten sets at two model sizes against
#: the same samples, beside the same tree with it off. It does buy output — on a
#: multiple-choice set it takes a larger model's answer from 141 tokens to 105 —
#: but it also takes away the model's reason to use a tool, and where searching
#: is what answers the question that costs accuracy. On the set whose tool
#: reaches the network, at the smaller size, searches per sample fell to 0.000
#: and the answer was right once in fifty against eight in fifty; that set's two
#: arms cannot be paired, so it is a direction rather than a result, but one
#: arithmetic set read 5.5 points below its own 3.9-point band as well, which
#: is a result and is the clause that decides this.
#:
#: So the control ships and the default does not. A caller who wants shorter
#: answers asks for them — ``AgentConfig(answer_style="brief")`` or
#: ``run(task, answer_style="brief")`` — and gets the saving on the shapes where
#: it is free, without every run being quietly told to stop using the tools it
#: was given.
DEFAULT_ANSWER_STYLE: str | None = None


def resolve_answer_style(per_call: Any, configured: Any) -> str | None:
    """Return the style one run states, or ``None`` for none.

    The same order as every other generation setting: a value pinned on the
    call, then one configured on the agent, then :data:`DEFAULT_ANSWER_STYLE`.
    ``""`` is a statement rather than an absence — it is how a caller asks the
    framework to say nothing about the answer's form — so it stops the search
    and answers ``None``.

    Args:
        per_call: ``answer_style`` passed to this call, if any.
        configured: ``AgentConfig.answer_style``, if set.

    Returns:
        The style to state, or ``None``.
    """
    for candidate in (per_call, configured):
        if candidate is None:
            continue
        text = str(candidate).strip()
        return text or None
    return DEFAULT_ANSWER_STYLE


def answer_style_text(style: Any) -> str:
    """Return the sentence *style* states, or ``""`` when it states none.

    A named style is looked up; anything else is the caller's own sentence and
    is returned as written. Logs ``[answer] answer style stated: <name>`` once
    per prompt that carries one, naming the style rather than reprinting it.
    """
    if style is None:
        return ""
    name = str(style).strip()
    if not name:
        return ""
    text = ANSWER_STYLE_TEXTS.get(name.lower())
    logger.info(
        "[answer] answer style stated: %s", name.lower() if text else "caller"
    )
    return text or name


__all__ = [
    "ANSWER_STYLE_BRIEF",
    "ANSWER_STYLE_FULL",
    "ANSWER_STYLE_TEXTS",
    "DEFAULT_ANSWER_STYLE",
    "answer_style_text",
    "resolve_answer_style",
]
