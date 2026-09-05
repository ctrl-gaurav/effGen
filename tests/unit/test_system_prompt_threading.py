"""A root agent's system_prompt reaches the work its sub-agents are given.

Decomposition prompts and specialist personas are fixed English text. An agent
configured to answer in another language kept that language in its own final
answer and lost it everywhere in between: the subtasks a model was asked to
generate, and the persona each spawned specialist was given, were English
regardless.

The parent's ``system_prompt`` is now threaded down to both. What is pinned
here is that it arrives, that it is added rather than substituted, and — the
part worth guarding — that a caller who set no ``system_prompt`` sends exactly
the bytes it sent before.
"""
from __future__ import annotations

import json

import pytest

from effgen.core.decomposition_engine import DecompositionEngine


class RecordingClient:
    """An llm_client that keeps the prompt and returns a usable decomposition."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps({"subtasks": [
            {"description": "Reunir los datos", "expected_output": "Los datos",
             "specialization": "research"},
            {"description": "Analizar los datos", "expected_output": "El análisis",
             "specialization": "analysis"},
        ]})


@pytest.fixture
def client() -> RecordingClient:
    return RecordingClient()


@pytest.fixture
def engine(client) -> DecompositionEngine:
    return DecompositionEngine(llm_client=client)


STRATEGIES = ["parallel_sub_agents", "sequential_sub_agents", "hybrid", "anything_else"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_prompt_is_unchanged_when_no_system_prompt_was_set(engine, client, strategy) -> None:
    """The common path has to be byte-identical, not merely similar."""
    engine.decompose("Investiga el mercado", strategy)
    without = client.prompts[-1]
    engine.decompose("Investiga el mercado", strategy, context={})
    assert client.prompts[-1] == without
    engine.decompose("Investiga el mercado", strategy, context={"other": "value"})
    assert client.prompts[-1] == without
    assert "same language as this instruction" not in without


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_system_prompt_reaches_the_decomposition_prompt(engine, client, strategy) -> None:
    engine.decompose("Investiga el mercado", strategy,
                     context={"system_prompt": "Responde siempre en español"})
    prompt = client.prompts[-1]
    assert "Responde siempre en español" in prompt
    assert "same language as this instruction" in prompt


def test_the_task_still_reaches_the_prompt_beside_the_instruction(engine, client) -> None:
    engine.decompose("Investiga el mercado", "parallel_sub_agents",
                     context={"system_prompt": "Responde siempre en español"})
    assert "Investiga el mercado" in client.prompts[-1]


def test_a_long_system_prompt_is_truncated_rather_than_pasted_whole(engine, client) -> None:
    """A whole persona in the decomposition prompt would crowd out the task."""
    engine.decompose("Investiga el mercado", "parallel_sub_agents",
                     context={"system_prompt": "x" * 5000})
    prompt = client.prompts[-1]
    assert "x" * 200 in prompt
    assert "x" * 400 not in prompt


def test_an_empty_system_prompt_is_treated_as_unset(engine, client) -> None:
    engine.decompose("Investiga el mercado", "parallel_sub_agents", context={"system_prompt": ""})
    assert "same language as this instruction" not in client.prompts[-1]


def test_decomposition_still_returns_subtasks_with_the_note_added(engine) -> None:
    subtasks = engine.decompose("Investiga el mercado", "parallel_sub_agents",
                                context={"system_prompt": "Responde siempre en español"})
    assert [s.description for s in subtasks] == ["Reunir los datos", "Analizar los datos"]
