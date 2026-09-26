"""Inline citation markers are requested, never assumed.

A model answering from retrieved passages will write ``"C [1]"`` when it is told
to cite them. That marker is not metadata — it is inside ``response.output``, so
it breaks an exact-match comparison, a structured-output schema, and any program
that reads the answer. These tests pin both halves of the contract:

* by default nothing asks for markers, and a trailing marker run a model
  produced anyway is removed from the answer — but only for a run that actually
  retrieved context, and only for an index that run's observations could have
  offered;
* with ``cite_sources=True`` the passages are shown to the model as a numbered
  list, and marker ``n`` resolves to ``response.citations[n - 1]``.

The models here are scripted in-process, which is deterministic plumbing rather
than a stand-in for live API behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import (
    CONTEXT_ANSWER_INSTRUCTION,
    CONTEXT_CITATION_INSTRUCTION,
    CONTINUE_INSTRUCTION,
    _strip_run_citation_markers,
)
from effgen.models._usage import tool_call_entry
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
    clear_stream_tool_calls,
    record_stream_tool_calls,
)
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

CITE_SENTENCE = "Cite each passage"

# Rows a retrieval tool returns when its results are structured: the citation
# miner reads these and can number them.
ROWS_A = {"results": [
    {"id": "a1", "content": "Alpha claim.", "metadata": {"source": "docs/alpha.md"}},
    {"id": "a2", "content": "Beta claim.", "metadata": {"source": "docs/beta.md"}},
    {"id": "a3", "content": "Gamma claim.", "metadata": {"source": "docs/gamma.md"}},
]}
ROWS_B = {"results": [
    {"id": "b1", "content": "Delta claim.", "metadata": {"source": "docs/delta.md"}},
    {"id": "b2", "content": "Epsilon claim.", "metadata": {"source": "docs/epsilon.md"}},
]}
URL_ROWS = [
    {"title": "Alpha", "url": "https://example.org/alpha", "snippet": "Alpha claim."},
    {"title": "Beta", "url": "https://example.org/beta", "snippet": "Beta claim."},
]
# The other shape a retrieval tool returns: one string, nothing to mine.
STRING_RESULT = "[Result 1] Alpha claim.\n\n[Result 2] Beta claim.\n\n[Result 3] Gamma claim."

CHOICE_TASK = (
    "Which statement is best supported?\nA. Sunlight drives most ecosystems.\n"
    "B. Most ecosystems are on land.\nC. Carbon dioxide is scarce.\n"
    "D. All producers are plants.\nAnswer with the letter only."
)


class _Retrieval(BaseTool):
    """A knowledge-base tool that replays a fixed sequence of results."""

    def __init__(self, payloads, name: str = "knowledge_search") -> None:
        self._payloads = list(payloads)
        self._calls = 0
        super().__init__(metadata=ToolMetadata(
            name=name,
            description="Look a topic up in the reference knowledge base.",
            category=ToolCategory.INFORMATION_RETRIEVAL,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="A short query.", required=True,
            )],
        ))

    async def _execute(self, **kwargs):
        payload = self._payloads[min(self._calls, len(self._payloads) - 1)]
        self._calls += 1
        return payload


class _Calculator(BaseTool):
    """A computation tool, so a run can hold a tool that retrieves nothing."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="The expression.", required=True,
            )],
        ))

    async def _execute(self, **kwargs) -> str:
        return "42"


class _ReActModel(BaseModel):
    """Text-only scripted model: drives the ReAct-text branch of the loop."""

    def __init__(self, turns: list[str]) -> None:
        super().__init__(model_name="scripted-react", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.turns = list(turns)
        self.prompts: list[str] = []
        self._index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return False

    def supports_function_calling(self) -> bool:
        return False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        text = self.turns[min(self._index, len(self.turns) - 1)]
        self._index += 1
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop", model_name=self.model_name,
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=max(1, len(text) // 4), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 8192


class _NativeModel(BaseModel):
    """Scripted model that declares native tool calls, the way an adapter does.

    Drives the native/hybrid prompt in the blocking loop and, through
    ``generate_stream``, the streamed native loop.
    """

    def __init__(self, turns: list[tuple[str, list[tuple[str, dict]]]]) -> None:
        super().__init__(model_name="scripted-native", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.turns = list(turns)
        self.prompts: list[str] = []
        self._index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return True

    def streams_tool_calls(self) -> bool:
        return True

    def _next(self, prompt) -> tuple[str, list[tuple[str, dict]]]:
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        turn = self.turns[min(self._index, len(self.turns) - 1)]
        self._index += 1
        return turn

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        text, calls = self._next(prompt)
        entries = [tool_call_entry(n, json.dumps(a)) for n, a in calls]
        return GenerationResult(
            text=text,
            tokens_used=7,
            finish_reason="tool_calls" if entries else "stop",
            model_name=self.model_name,
            metadata={"tool_calls": entries},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        text, calls = self._next(prompt)
        clear_stream_tool_calls(self)
        if calls:
            record_stream_tool_calls(
                self, [tool_call_entry(n, json.dumps(a)) for n, a in calls]
            )
        for i in range(0, len(text), 4):
            yield text[i:i + 4]

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=max(1, len(text) // 4), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 8192


def _act(query: str, tool: str = "knowledge_search") -> str:
    return (f"Thought: look it up.\nAction: {tool}\n"
            f"Action Input: {json.dumps({'query': query})}")


def _react_run(results, final: str, task: str = CHOICE_TASK, queries=("alpha",), **cfg):
    """Run the ReAct-text loop over a retrieval tool and return (model, response).

    ``results`` is what the tool returns, one entry per call.
    """
    model = _ReActModel([_act(q) for q in queries] + [f"Thought: done.\nFinal Answer: {final}"])
    run_kwargs = cfg.pop("run_kwargs", {})
    agent = Agent(config=AgentConfig(
        name="cite-test", model=model, tools=[_Retrieval(results)],
        tool_calling_mode="react", max_iterations=8, **cfg,
    ))
    return model, agent.run(task, **run_kwargs)


def _native_run(results, final: str, task: str = CHOICE_TASK, **cfg):
    """Run the native/hybrid loop over a retrieval tool, one result per call."""
    model = _NativeModel([
        ("", [("knowledge_search", {"query": "alpha"})]),
        (f"Final Answer: {final}", []),
    ])
    run_kwargs = cfg.pop("run_kwargs", {})
    agent = Agent(config=AgentConfig(
        name="cite-test", model=model, tools=[_Retrieval(results)],
        tool_calling_mode="native", max_iterations=8, **cfg,
    ))
    return model, agent.run(task, **run_kwargs)


def _markers(text: str) -> list[int]:
    import re
    return [int(m) for m in re.findall(r"\[(\d+)\]", text or "")]


# --------------------------------------------------------------------------- #
# The default answer
# --------------------------------------------------------------------------- #
class TestDefaultAnswerCarriesNoMarkers:
    def test_react_loop_returns_exactly_the_choice(self):
        model, resp = _react_run([ROWS_A], "A [1]")
        assert resp.output == "A"
        assert resp.metadata["citation_markers_stripped"] == 1

    def test_native_loop_returns_exactly_the_choice(self):
        model, resp = _native_run([ROWS_A], "A [1], [2]")
        assert resp.output == "A"
        assert resp.metadata["citation_markers_stripped"] == 2

    def test_no_prompt_asks_for_markers_on_the_react_path(self):
        model, _ = _react_run([ROWS_A], "A")
        assert sum(CITE_SENTENCE in p for p in model.prompts) == 0

    def test_no_prompt_asks_for_markers_on_the_native_path(self):
        model, _ = _native_run([ROWS_A], "A")
        assert sum(CITE_SENTENCE in p for p in model.prompts) == 0

    def test_no_prompt_asks_for_markers_on_the_streamed_path(self):
        model = _NativeModel([
            ("", [("knowledge_search", {"query": "alpha"})]),
            ("A", []),
        ])
        agent = Agent(config=AgentConfig(
            name="cite-test", model=model, tools=[_Retrieval([ROWS_A])],
            tool_calling_mode="native", max_iterations=8,
        ))
        list(agent.stream(CHOICE_TASK))
        assert model.prompts, "model was never called"
        assert sum(CITE_SENTENCE in p for p in model.prompts) == 0

    @pytest.mark.parametrize(
        ("answer", "expected", "stripped"),
        [
            ("C [1]", "C", 1),
            ("D [1], [2]", "D", 2),
            # No separator at all — a shape seen from a live model.
            ("D [1][2][3]", "D", 3),
            ("C [1] , [2] ", "C", 2),
        ],
    )
    def test_every_marker_shape_is_removed(self, answer, expected, stripped):
        _, resp = _react_run([ROWS_A], answer)
        assert resp.output == expected
        assert resp.metadata["citation_markers_stripped"] == stripped


class TestTheStripIsGated:
    """The cases where the answer must be left exactly as the model wrote it.

    They pass whether or not the strip exists, since leaving the answer alone
    is also what no strip at all does. They are here to fail if a later change
    widens the strip, which is a failure mode no positive test can catch.
    """

    def test_an_answer_from_a_run_that_never_retrieved_is_untouched(self):
        """A bracketed number is part of the answer when nothing was retrieved."""
        model = _ReActModel([
            'Thought: compute.\nAction: calculator\nAction Input: {"expression": "6*7"}',
            "Thought: done.\nFinal Answer: The isotope is carbon-14 [2]",
        ])
        agent = Agent(config=AgentConfig(
            name="cite-test", model=model, tools=[_Calculator()],
            tool_calling_mode="react", max_iterations=8,
        ))
        resp = agent.run("Which isotope dates organic material? Cite the footnote number.")
        assert resp.output == "The isotope is carbon-14 [2]"
        assert "citation_markers_stripped" not in resp.metadata
        assert sum(CITE_SENTENCE in p for p in model.prompts) == 0

    def test_an_index_past_the_passage_count_is_part_of_the_answer(self):
        """Three passages were offered, so a trailing [9] is not a reference."""
        _, resp = _react_run([ROWS_A], "Isotope [9]")
        assert resp.output == "Isotope [9]"
        assert "citation_markers_stripped" not in resp.metadata

    def test_an_answer_that_is_only_a_marker_is_kept(self):
        _, resp = _react_run([ROWS_A], "[2]")
        assert resp.output == "[2]"
        assert "citation_markers_stripped" not in resp.metadata

    def test_a_marker_inside_a_correct_answer_survives(self):
        answer = (
            "B\n\nExplanation: seeds that wait for the right conditions "
            "are called dormant [1]."
        )
        _, resp = _react_run([ROWS_A], answer)
        assert resp.output == answer
        assert "citation_markers_stripped" not in resp.metadata


class TestStripFunction:
    """The pure function, apart from the run state that gates it.

    ``test_removes_a_trailing_marker_run`` and the two bound cases pin what the
    function removes. ``test_leaves_everything_else_alone`` and
    ``test_none_is_returned_unchanged`` are the negative half: they hold for a
    build that never strips as well as for this one, and exist to fail if the
    pattern is ever widened.
    """

    @pytest.mark.parametrize(
        ("text", "expected", "count"),
        [
            ("C [1]", "C", 1),
            ("D [1], [2]", "D", 2),
            ("D [1][2][3]", "D", 3),
            ("C [1] , [2] ", "C", 2),
            ("A [1]\n", "A", 1),
        ],
    )
    def test_removes_a_trailing_marker_run(self, text, expected, count):
        assert _strip_run_citation_markers(text) == (expected, count)

    @pytest.mark.parametrize(
        "text",
        [
            "The isotope is carbon-14 [2]. It decays slowly.",
            "See [1] above for the derivation.",
            "[2]",
            "  [1], [2]  ",
            "",
            "no markers at all",
        ],
    )
    def test_leaves_everything_else_alone(self, text):
        assert _strip_run_citation_markers(text) == (text, 0)

    def test_none_is_returned_unchanged(self):
        assert _strip_run_citation_markers(None) == (None, 0)

    def test_the_index_bound_is_applied_when_passages_are_known(self):
        assert _strip_run_citation_markers("D [9]", passages=3) == ("D [9]", 0)
        assert _strip_run_citation_markers("D [3]", passages=3) == ("D", 1)
        assert _strip_run_citation_markers("D [2], [4]", passages=3) == ("D [2], [4]", 0)

    def test_the_bound_is_not_applied_when_nothing_countable_was_offered(self):
        assert _strip_run_citation_markers("D [9]", passages=0) == ("D", 1)


# --------------------------------------------------------------------------- #
# Markers when they are asked for
# --------------------------------------------------------------------------- #
class TestRequestedMarkersResolve:
    def test_one_call_every_marker_resolves(self):
        model, resp = _react_run(
            [ROWS_A], "Alpha is best supported [1]", cite_sources=True,
        )
        assert sum(CITE_SENTENCE in p for p in model.prompts) == 1
        assert resp.output == "Alpha is best supported [1]"
        assert [c.source for c in resp.citations] == [
            "docs/alpha.md", "docs/beta.md", "docs/gamma.md",
        ]
        assert resp.citations[0].source == "docs/alpha.md"
        assert _unresolvable(resp) == []

    def test_two_calls_a_marker_points_at_the_passage_the_model_saw(self):
        model, resp = _react_run(
            [ROWS_A, ROWS_B], "Delta is best supported [4]",
            queries=("alpha", "delta"), cite_sources=True,
        )
        # The passages are numbered across calls, so [4] is the first row of
        # the second observation.
        assert resp.citations[3].source == "docs/delta.md"
        assert [c.index for c in resp.citations] == [1, 2, 3, 4, 5]
        assert _unresolvable(resp) == []
        # The model was shown that numbering, not a raw result dict.
        assert "[4] Source: docs/delta.md" in model.prompts[-1]

    def test_a_passage_retrieved_twice_keeps_its_number(self):
        _, resp = _react_run(
            [ROWS_A, ROWS_A], "Alpha again [1]",
            queries=("alpha", "alpha"), cite_sources=True,
        )
        assert [c.index for c in resp.citations] == [1, 2, 3]
        assert [c.source for c in resp.citations] == [
            "docs/alpha.md", "docs/beta.md", "docs/gamma.md",
        ]

    def test_url_rows_are_resolvable(self):
        model = _ReActModel([
            _act("alpha", "web_search"),
            "Thought: done.\nFinal Answer: Alpha is best supported [1]",
        ])
        agent = Agent(config=AgentConfig(
            name="cite-test", model=model,
            tools=[_Retrieval([URL_ROWS], name="web_search")],
            tool_calling_mode="react", max_iterations=8, cite_sources=True,
        ))
        resp = agent.run("Which claim is best supported?")
        assert resp.citations, "a URL source the model was shown must resolve"
        assert resp.citations[0].source == "https://example.org/alpha"
        assert _unresolvable(resp) == []

    def test_a_string_result_is_never_asked_to_be_cited(self):
        """Nothing to number means nothing to cite, so no marker is requested."""
        model, resp = _react_run(
            [STRING_RESULT], "Alpha is best supported", cite_sources=True,
        )
        assert sum(CITE_SENTENCE in p for p in model.prompts) == 0
        assert _markers(resp.output) == []
        assert _unresolvable(resp) == []

    def test_the_run_keyword_and_the_config_field_produce_the_same_prompt(self):
        by_config, _ = _react_run([ROWS_A], "Alpha [1]", cite_sources=True)
        by_call, _ = _react_run([ROWS_A], "Alpha [1]", run_kwargs={"cite_sources": True})
        assert by_call.prompts == by_config.prompts

    def test_the_run_keyword_can_switch_it_back_off(self):
        model, resp = _react_run(
            [ROWS_A], "Alpha [1]", cite_sources=True,
            run_kwargs={"cite_sources": False},
        )
        assert sum(CITE_SENTENCE in p for p in model.prompts) == 0
        assert resp.output == "Alpha"


def _unresolvable(resp) -> list[int]:
    """Markers in the answer with no citation entry behind them."""
    cits = resp.citations or []
    return [
        n for n in _markers(resp.output)
        if not (1 <= n <= len(cits) and cits[n - 1].index == n)
    ]


# --------------------------------------------------------------------------- #
# Nothing else moves
# --------------------------------------------------------------------------- #
class TestNothingElseMoves:
    """Invariants: what a caller saw before, they still see.

    Like :class:`TestTheStripIsGated`, these held before markers became opt-in
    too, by design — an invariant that only held after the change would not be
    one.
    """

    def test_default_mode_citations_and_sources_are_unchanged(self):
        """The evidence fields are what they were before markers became opt-in."""
        _, resp = _react_run([ROWS_A], "Alpha is best supported")
        assert [(c.index, c.source) for c in resp.citations] == [
            (1, "docs/alpha.md"), (2, "docs/beta.md"), (3, "docs/gamma.md"),
        ]
        assert resp.sources == ["docs/alpha.md", "docs/beta.md", "docs/gamma.md"]

    def test_default_mode_two_calls_keep_the_run_wide_source_list(self):
        _, resp = _react_run(
            [ROWS_A, ROWS_B], "Delta is best supported", queries=("alpha", "delta"),
        )
        assert resp.sources == [
            "docs/alpha.md", "docs/beta.md", "docs/gamma.md",
            "docs/delta.md", "docs/epsilon.md",
        ]

    def test_the_non_retrieval_close_is_unchanged(self):
        assert CONTINUE_INSTRUCTION == (
            "Continue solving the task. If you have the final answer, "
            "state it clearly."
        )

    def test_the_retrieval_close_no_longer_asks_for_markers(self):
        assert CITE_SENTENCE not in CONTEXT_ANSWER_INSTRUCTION
        assert CITE_SENTENCE in CONTEXT_CITATION_INSTRUCTION
        assert "source material" in CONTEXT_ANSWER_INSTRUCTION
