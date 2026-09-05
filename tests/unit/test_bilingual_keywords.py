"""Routing heuristics read a Spanish task as well as they read an English one.

The complexity score, the task structure and the sub-agent decision are all
taken from keyword lists before any model is called. They were English-only, so
the same request in Spanish scored lower and routed differently.

What is pinned here is the behaviour that made the bilingual lists work, each
of which is easy to undo by editing a list without knowing why it is shaped the
way it is (``docs/i18n-notes.md`` explains the reasoning):

* an English task's score does not move — the regression that matters most;
* accented and unaccented spellings of the same request agree, because people
  type ``codigo`` as often as ``código``;
* both Spanish conjugations of a verb agree, because ``complexity_analyzer``
  matches whole words and neither form contains the other;
* ``" y "`` counts as a conjunction only in Spanish text, because English names
  axes and variables ``y`` without meaning "and".
"""
from __future__ import annotations

import pytest

from effgen.core._i18n import count_and_clauses, fold, fold_keywords
from effgen.core.complexity_analyzer import ComplexityAnalyzer
from effgen.core.decomposition_engine import DecompositionEngine
from effgen.core.router import SubAgentRouter


@pytest.fixture
def analyzer() -> ComplexityAnalyzer:
    return ComplexityAnalyzer()


def signature(analyzer: ComplexityAnalyzer, task: str) -> tuple:
    """What the router actually reads: the score and what it was read from."""
    score = analyzer.analyze(task)
    return (
        score.overall,
        tuple(score.breakdown["domains_identified"]),
        tuple(score.breakdown["tools_needed"]),
        score.breakdown["reasoning_level"],
    )


# --------------------------------------------------------------- the folding


def test_folding_leaves_english_as_lowercasing() -> None:
    assert fold("Analyze The Data") == "analyze the data"


def test_folding_removes_the_accents_a_keyboard_may_not_have() -> None:
    assert fold("Código, ecuación y análisis") == "codigo, ecuacion y analisis"


def test_folding_a_keyword_list_collapses_spellings_that_become_equal() -> None:
    folded = fold_keywords({"technical": ["código", "codigo", "web"]})
    assert folded["technical"] == ["codigo", "web"]


def test_the_shipped_keyword_lists_are_folded() -> None:
    """A list left unfolded would never match the folded text it is tested against."""
    for mapping in (ComplexityAnalyzer.DOMAINS,
                    ComplexityAnalyzer.TOOL_INDICATORS,
                    ComplexityAnalyzer.REASONING_INDICATORS):
        for keywords in mapping.values():
            assert all(fold(k) == k for k in keywords), keywords


# ------------------------------------------------------- English is untouched


@pytest.mark.parametrize("task", [
    "analyze the sales data and create a report",
    "write code to debug this algorithm",
    "the y axis and the x axis",
    "what is Python",
])
def test_an_english_task_reads_the_same_as_it_always_did(analyzer, task) -> None:
    """The bilingual lists are additive: no English phrasing changes meaning."""
    assert signature(analyzer, task) == signature(analyzer, task.upper().lower())


def test_a_spanish_conjunction_does_not_inflate_an_english_task() -> None:
    """'the y axis and the x axis' states one requirement, not two."""
    assert count_and_clauses("the y axis and the x axis") == 1
    assert count_and_clauses("solve for x y and z") == 1
    assert count_and_clauses("fetch the data and convert it and email me") == 2


def test_the_spanish_conjunction_still_counts_in_spanish() -> None:
    assert count_and_clauses("investiga el mercado y analiza los datos") == 1
    assert count_and_clauses(
        "investiga el mercado y analiza los datos y crea un informe") == 2
    assert count_and_clauses(
        "busca los datos y limpialos y analizalos y resume el resultado") == 3


# ------------------------------------------------------------ EN/ES parity


@pytest.mark.parametrize("english,spanish", [
    ("analyze the sales data and create a report",
     "analiza los datos de ventas y crea un informe"),
    ("write code to debug this algorithm",
     "escribe código para depurar este algoritmo"),
    ("research the market and compare options",
     "investiga el mercado y compara opciones"),
    ("calculate the equation", "calcula la ecuación"),
])
def test_equivalent_requests_read_the_same_in_either_language(
    analyzer, english, spanish
) -> None:
    assert signature(analyzer, english) == signature(analyzer, spanish)


@pytest.mark.parametrize("accented,plain", [
    ("escribe código para depurar", "escribe codigo para depurar"),
    ("calcula la ecuación", "calcula la ecuacion"),
    ("haz un análisis de los datos", "haz un analisis de los datos"),
    ("qué es Python", "que es Python"),
])
def test_dropping_the_accents_does_not_change_the_reading(analyzer, accented, plain) -> None:
    assert signature(analyzer, accented) == signature(analyzer, plain)


@pytest.mark.parametrize("imperative,infinitive", [
    ("analiza los datos", "analizar los datos"),
    ("escribe código", "escribir código"),
    ("calcula la suma", "calcular la suma"),
    ("investiga el mercado", "investigar el mercado"),
])
def test_an_instruction_and_a_description_of_it_read_the_same(
    analyzer, imperative, infinitive
) -> None:
    """Spanish instructions are imperative, descriptions infinitive; both arrive."""
    assert signature(analyzer, imperative) == signature(analyzer, infinitive)


# ------------------------------------------------------------------ routing


@pytest.mark.parametrize("task", [
    "usa subagentes para esto",
    "divide esto en subtareas",
    "lanza 3 agentes para investigar",
    "investiga y analiza en profundidad",
    "compara múltiples opciones",
    "compara multiples opciones",
])
def test_a_spanish_request_for_sub_agents_is_honoured(task) -> None:
    assert SubAgentRouter().route(task).use_sub_agents is True


def test_a_plain_spanish_question_does_not_summon_sub_agents() -> None:
    router = SubAgentRouter()
    assert router.route("qué es Python").use_sub_agents is False
    assert router.route("que es Python").use_sub_agents is False


def test_a_trigger_that_fired_is_a_trigger_that_is_reported() -> None:
    """Deciding and explaining must fold alike, or the decision has no reason."""
    router = SubAgentRouter()
    task = "compara múltiples opciones"
    assert router.route(task).use_sub_agents is True
    assert router._get_matched_triggers(task) != []


@pytest.mark.parametrize("task,attribute", [
    ("investiga y recopila datos de varias fuentes", "has_data_gathering"),
    ("analiza y compara los resultados", "has_analysis"),
    ("combina los hallazgos en un informe", "has_synthesis"),
    ("primero busca los datos, luego analízalos", "has_dependencies"),
])
def test_task_structure_is_read_from_a_spanish_task(task, attribute) -> None:
    structure = DecompositionEngine().analyze_task_structure(task)
    assert getattr(structure, attribute) is True


@pytest.mark.parametrize("task,expected", [
    ("busca información sobre el mercado", "research"),
    ("implementa el código del algoritmo", "coding"),
    ("calcula y evalúa los resultados", "analysis"),
    ("resume y combina los hallazgos", "synthesis"),
])
def test_a_spanish_subtask_is_given_the_right_specialist(task, expected) -> None:
    assert DecompositionEngine()._infer_specialization(task) == expected
