# Matching non-English tasks

effGen decides three things from keyword lists before any model is called: how
complex a task is (`ComplexityAnalyzer`), how it should be broken up
(`DecompositionEngine`), and whether it warrants sub-agents (`SubAgentRouter`).
Those lists were English-only, so an equivalent Spanish request scored lower
and was routed differently. They are now bilingual EN/ES. This note records the
decisions behind that, because each one is easy to undo by accident.

## There is no language-detection step

The obvious design is to detect the language and then pick a keyword list. It
was tried and rejected: general-purpose detectors are unreliable on text this
short. `langdetect` classified `"qué es Python"` as French with 99.99%
confidence — and a router that mis-detects the language silently applies the
wrong list to every heuristic downstream.

Merging both languages into one list has none of that failure mode. A Spanish
keyword cannot fire on an English task unless the word is genuinely shared, and
the cost of carrying both is a slightly longer list.

Adding a third language means extending the same lists. It does not mean adding
a detection step.

## Text and keywords are folded before they meet

People type `codigo` for `código` and `ecuacion` for `ecuación`, particularly
on keyboards without dead keys. A list holding only the accented spelling
misses those, and the miss is invisible — the task simply scores lower.

So both sides are passed through `effgen.core._i18n.fold`, which lowercases and
strips combining marks. Keyword lists stay spelled correctly in the source and
are folded once at import by `fold_keywords`. English is ASCII, so folding it
is exactly `str.lower` and nothing about English behaviour changes.

**If you add a keyword, write it properly, with its accents.** Do not
pre-fold it by hand.

## Verb forms depend on how the module matches

The two matchers in play behave differently, and a keyword that works in one
can be dead in the other:

| module | matcher | consequence |
|---|---|---|
| `complexity_analyzer` | whole word (`_mentions`) | `analiza` does **not** match `analizar` |
| `decomposition_engine`, `router` | substring (`in`) | `analiza` **does** match `analizar` |

Whole-word matching is deliberate — substring matching once read `api` out of
`capital` and `graph` out of `paragraph`. The consequence for Spanish is that
`complexity_analyzer` needs **both** conjugations listed (`analizar` *and*
`analiza`), because neither contains the other and no stem covering both is a
word. The substring modules need only the shorter form.

Spanish instructions are usually imperative (`analiza los datos`) while
descriptions are usually infinitive (`analizar los datos`). Both reach the
router, so both belong in the list.

## `" y "` is only counted as a conjunction in Spanish text

`" and "` is a reliable signal that a task states more than one requirement.
Its Spanish equivalent `" y "` is one letter, and English text contains it
without meaning a conjunction: `"the y axis and the x axis"` names an axis.
Counting it unconditionally inflated the requirement count of English tasks.

`count_and_clauses` therefore counts `" y "` only when the text also contains a
Spanish function word (`el`, `la`, `de`, `para`, `que`, …). That is a guard on
one ambiguous token, not the language-detection step rejected above: getting it
wrong costs at most one requirement of one score, and it cannot mis-route a
task on its own.

## What to check when extending this

- Equivalent EN/ES phrasings should produce the same domains, tools and score.
- The accented and unaccented spellings of the same request should agree.
- Both conjugations of a verb should agree.
- An English task's score must not move at all.

`tests/unit/test_bilingual_keywords.py` asserts each of those.
