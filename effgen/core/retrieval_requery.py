"""One more search, when the run's own answer says the first one found nothing.

A tool prompt ends with the last observation, so after a search the last thing
the model reads is a block of source passages and a line telling it what to do
with them. That line ends "if they do not answer it, say so and name what is
missing" — an instruction to conclude. Nothing anywhere in the loops offers the
other move, searching again with different words, so a run whose one query came
back on topic and silent has exactly two things it can do: invent an answer, or
report that it cannot. A run that searched once badly can never search again.

The rule here is the missing move, bounded at one use:

    When a run holding a context-retrieval tool has just searched, and its own
    answer reports that what came back does not answer the question, the run
    spends one — and only one — further search with a different query before
    that answer is accepted.

**Reading the answer, not the observation.** Three signals could say a retrieval
observation was unhelpful: the tool reported an error, the tool returned
nothing, and the model's own turn saying the material does not answer the
question. The first two are one cheap test each and are checked first; the third
is what actually fires. It is a general signal rather than a guess about prose
because the framework asked for it: the closing line of a retrieval prompt and
the lookup contract a retrieval tool is given both end with the same sentence,
and both are selected from the tool's declared category. Detecting that sentence
being obeyed is recognising compliance with a contract the framework states.

**The claim, not the prose around it.** :func:`declines_from_context` reads what
the answer commits to — the text after its last answer label, or the whole text
when there is no label — and fires only when *that* is a statement that the
material is silent. An earlier form of this test keyed on a negation near a
"says" verb anywhere in the answer, and fired on answers scored correct: a model
that answers a multiple-choice question from its own knowledge and then remarks
that the passages were not useful. The remark is not the answer. Requiring a
declining sentence to also name the material — sources, passages, results,
context — is what separates the two.

**Bounded at one.** The point is not to search until something turns up. A
capable model already re-queries several times unprompted, to the iteration cap;
what the framework owes a run is one more search, not a search policy. So
:data:`MAX_RETRIEVAL_REQUERIES` is one, it is spent before the second search
runs, and the answer that follows it is accepted whatever it says — including
"not found", which stays an answer rather than becoming a failure.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

from .tool_call_record import ToolCall

logger = logging.getLogger(__name__)

#: Further searches one run may be sent back for. One: a run that searched once
#: badly gets a second query, and a run that has already had its second query
#: keeps whatever it says next.
MAX_RETRIEVAL_REQUERIES = 1

#: Iterations a run must still have for a re-query to be worth spending: one
#: turn for the search and one for the answer written from it. Below that the
#: nudge would cost the run its answer.
REQUERY_MIN_ITERATIONS_LEFT = 2

#: Whether the turn that follows the nudge is also required to call a tool where
#: the provider can enforce it. A nudge alone is a request; models that already
#: decided the material was missing tend to decline it again. The loops read
#: this, so the two halves of the ask can be measured apart.
REQUERY_FORCES_TOOL_CALL = True

# The last "Answer:" / "Final Answer:" label, anywhere on a line.
_LABEL_RE = re.compile(r"(?:final[ \t]*)?answer[ \t]*[:\-][ \t]*", re.I)

# A claim that asserts the material is silent rather than asserting a value.
_DECLINE_RE = re.compile(
    r"^\W*(?:the\s+)?"
    r"(?:(?:exact|precise|specific)\s+)?"
    r"(?:answer|value|year|date|number|name|amount|figure|information|detail)?"
    r"\s*(?:is|was|are|were|:)?\s*"
    r"(?:"
    r"unknown\b"
    r"|unclear\b"
    r"|undetermined\b"
    r"|indeterminate\b"
    r"|n/?a\b"
    r"|none\s+(?:given|provided|found|stated)\b"
    r"|no\s+(?:information|data|answer|year|date|value|mention|record)\b"
    r"|insufficient\s+(?:information|data|evidence)\b"
    r"|not\s+(?:specified|available|found|known|stated|mentioned|given|provided|"
    r"determined|disclosed|listed|indicated|explicit\w*)\b"
    r"|cannot\s+be\s+(?:determined|found|established|answered)\b"
    r"|(?:could|can)\s*(?:not|n't)\s+(?:be\s+)?(?:determined|found|established|answered)\b"
    r"|unable\s+to\s+(?:determine|find|answer|establish)\b"
    r")",
    re.I,
)

# What the model calls the material it was given. A sentence that says something
# is missing without naming the material is a remark about the question, not a
# report that the search came back silent: "not the most precise answer among
# the options provided" is an answer, and a detector without this fired on it.
_MATERIAL_RE = re.compile(
    r"\b(?:sources?|passages?|results?|searches|search|documents?|context|"
    r"information|data|texts?|articles?|records?|references?|snippets?|"
    r"materials?|observations?|excerpts?)\b",
    re.I,
)

# A sentence reporting that the material is silent. Used only where the answer
# carries no label and the whole text has to be read as the claim.
_SILENT_SENTENCE_RE = re.compile(
    r"\b(?:do(?:es)?\s*n[o']t|did\s*n[o']t|is\s+not|are\s+not|was\s+not|were\s+not|"
    r"cannot|can\s*not|can't|could\s+not|couldn't|unable|no|lack\w*|without|"
    r"insufficient|fail\w*\s+to)\b[^.\n]{0,60}?\b"
    r"(?:specif\w+|mention\w*|state[sd]?|provide[sd]?|contain\w*|include[sd]?|give[sn]?|"
    r"list\w*|indicat\w*|report\w*|say[s]?|found|find|determin\w*|answer\w*|address\w*|"
    r"available|explicit\w*|information|detail\w*|data)\b",
    re.I,
)

# A sentence that only says more work would be needed. Neither an answer nor a
# claim about the material, so it neither fires the test nor blocks it.
_MORE_WORK_SENTENCE_RE = re.compile(
    r"\b(?:further|additional|more)\s+(?:research|search\w*|information|sources|data)\b"
    r"|\bwould\s+be\s+(?:needed|required)\b|\bneed\w*\s+(?:more|further|additional)\b",
    re.I,
)

#: How a dispatch that produced no usable result opens, in the text the loop
#: puts in front of the model — the same prefixes a computed result is tested
#: against before it is handed back as an answer.
_FAILED_OBSERVATION_PREFIXES = (
    "error executing tool",
    "tool execution failed:",
    "error:",
)


def _claim_of(text: str) -> tuple[str, bool]:
    """What *text* commits to, and whether an answer label selected it.

    Returns the text after the last ``Answer:`` / ``Final Answer:`` label, or
    the whole trimmed text when there is no label with anything after it.
    """
    stripped = (text or "").strip()
    labels = list(_LABEL_RE.finditer(stripped))
    if labels:
        tail = stripped[labels[-1].end():].strip()
        if tail:
            return tail, True
    return stripped, False


def declines_from_context(text: str) -> bool:
    """True when *text* answers by reporting that the material is silent.

    Two branches, chosen by whether the answer labelled its claim.

    **Labelled.** The claim is what follows the last answer label, and the test
    is whether that claim is itself a statement of absence — "unknown", "not
    specified", "cannot be determined" and the short list around them. A label
    whose claim is a value is an answer, whatever the prose above it says.

    **Unlabelled.** Every sentence must be either a report that the material is
    silent — a negation *and* a word for the material — or a remark that more
    research would be needed. One sentence asserting anything else means the
    model answered, and the test does not fire.

    Args:
        text: The answer as it would be returned, already sanitized.

    Returns:
        True when the answer's claim is that the material does not answer the
        question.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    claim, labelled = _claim_of(stripped)
    if _DECLINE_RE.match(claim):
        return True
    if labelled:
        return False
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", stripped) if s.strip()]
    silent = 0
    for sentence in sentences:
        if _SILENT_SENTENCE_RE.search(sentence) and _MATERIAL_RE.search(sentence):
            silent += 1
        elif _MORE_WORK_SENTENCE_RE.search(sentence):
            continue
        else:
            return False
    return silent > 0


def _unhelpful_reason(answer: str, call: ToolCall) -> str | None:
    """Why the search *call* left the run without what it asked for, or ``None``.

    The two cheap tests come first: a dispatch that failed, and a dispatch that
    came back with nothing. Neither has ever been observed on a recorded run —
    a search that finds nothing returns passages about something else rather
    than an empty result — so the reason that fires in practice is the third,
    which is a fact about the answer rather than about the observation.
    """
    if call.error:
        return "the search reported an error"
    result = (call.result or "").strip()
    if not result or result.lower().startswith(_FAILED_OBSERVATION_PREFIXES):
        return "the search returned nothing"
    if declines_from_context(answer):
        return "the answer reports the material does not answer the question"
    return None


def should_requery(
    answer: str,
    calls: Sequence[ToolCall],
    is_retrieval: Callable[[str], bool],
    *,
    tools_suppressed: bool,
    iterations_left: int,
    requery_spent: bool,
) -> bool:
    """True when this answer should be preceded by one more, different search.

    Every condition below has to hold, which is what keeps the rule inert on
    runs it has nothing to offer:

    * the run dispatched at least one call and the last of them was a
      context-retrieval tool — the tool whose observation selected the closing
      line the model was answering;
    * tools are still being offered, so the run has not already been pushed into
      writing an answer from what it has. A run past that point cannot be sent
      back to search, and this is what keeps a second search from turning a
      "not found" into a stopped run;
    * two iterations remain, one to search and one to answer;
    * the re-query has not been spent;
    * the search left the run without what it asked for, per
      :func:`_unhelpful_reason`.

    Args:
        answer: The answer as it would be returned, already sanitized.
        calls: The run's tool-call records, oldest first.
        is_retrieval: Tells a tool name that returns source material from one
            that computes an answer — the same predicate that picks the closing
            instruction after a retrieval observation.
        tools_suppressed: Whether the run has stopped offering tool definitions.
        iterations_left: Iterations the run has after this turn.
        requery_spent: Whether this run has already been sent back to search.

    Returns:
        True when the run should append the re-query nudge and continue instead
        of returning *answer*.
    """
    if requery_spent or tools_suppressed:
        return False
    if iterations_left < REQUERY_MIN_ITERATIONS_LEFT:
        return False
    if not calls or not is_retrieval(calls[-1].name):
        return False
    reason = _unhelpful_reason(answer, calls[-1])
    if reason is None:
        return False
    logger.info(
        "retrieval re-query: %s; searching once more with a different query "
        "after '%s'",
        reason,
        calls[-1].name,
    )
    return True


__all__ = [
    "MAX_RETRIEVAL_REQUERIES",
    "REQUERY_FORCES_TOOL_CALL",
    "REQUERY_MIN_ITERATIONS_LEFT",
    "declines_from_context",
    "should_requery",
]
