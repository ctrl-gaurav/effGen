"""Matching keyword lists against text that may not be written in English.

The routing heuristics — complexity scoring, task decomposition, sub-agent
triggering — decide from keyword lists tested with ``in`` against a lowercased
task. That is cheap and predictable, and it stays that way for a second
language; what it is not is robust to the two ways the same Spanish word
reaches us.

**Accents are optional in practice.** People type ``codigo`` for ``código`` and
``ecuacion`` for ``ecuación``, especially on keyboards without dead keys. A
list holding only the accented spelling silently misses those, and the miss is
invisible: the task simply scores lower. So both the lists and the text are
folded to unaccented lowercase before they meet, which leaves ASCII — every
English keyword — untouched.

**Substring matching has a direction.** ``"analiza" in "analizar los datos"``
is true; ``"analizar" in "analiza los datos"`` is false. Spanish verbs reach a
task in whichever form the sentence needs, so a list that stores the infinitive
misses the imperative that most instructions are actually written in. Storing
the shorter shared stem — ``analiza``, ``escrib``, ``genera`` — matches both,
which is why the lists here are not simply dictionary forms.
"""

from __future__ import annotations

import unicodedata

__all__ = ["count_and_clauses", "fold", "fold_keywords"]

#: Function words that appear in ordinary Spanish and effectively never as
#: standalone words in English. They are the cheap test for "is this sentence
#: Spanish at all", used only to decide whether an ambiguous one-letter token
#: should be read as a conjunction — never to pick a language for routing,
#: which short queries are far too small to support reliably.
SPANISH_FUNCTION_WORDS = (
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "para", "con", "por", "que", "en", "sobre", "desde", "hasta",
)


def fold(text: str) -> str:
    """Return *text* lowercased with its accents removed.

    ASCII is returned unchanged apart from the lowercasing, so folding an
    English task is exactly ``str.lower``.

    Args:
        text: Any text, in any language.

    Returns:
        The text in lowercase with combining marks stripped, ready to be
        tested against a folded keyword list.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold_keywords(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return *mapping* with every keyword folded, keys untouched.

    Lets a keyword list stay readable in the source — written the way the
    language spells it, accents and all — while being compared in the folded
    form the text arrives in.

    Args:
        mapping: Category name to the keywords that indicate it.

    Returns:
        The same mapping with each keyword passed through :func:`fold`,
        duplicates that folding collapses removed, and order preserved.
    """
    folded: dict[str, list[str]] = {}
    for category, keywords in mapping.items():
        seen: dict[str, None] = {}
        for keyword in keywords:
            seen.setdefault(fold(keyword), None)
        folded[category] = list(seen)
    return folded


def count_and_clauses(text: str) -> int:
    """How many times *text* joins two things with "and".

    Counts English ``" and "`` always, and Spanish ``" y "`` only where a
    Spanish function word shows the sentence is Spanish. That guard is what
    keeps the Spanish conjunction from inflating an English task's requirement
    count: ``"the y axis and the x axis"`` names an axis, and read without the
    guard it describes a second requirement it does not have.

    Args:
        text: The task, in any language and any case.

    Returns:
        int: The number of "and" joins found.
    """
    folded = fold(text)
    count = folded.count(" and ")
    padded = f" {folded} "
    if any(f" {word} " in padded for word in SPANISH_FUNCTION_WORDS):
        count += folded.count(" y ")
    return count
