"""
Durability tests for the RAG / memory stack:

- Retrieved evidence is surfaced on ``AgentResponse.sources`` / ``.citations``
  (the knowledge-base / search path actually exposes its grounding).
- Memory stores (filesystem JSON, SQLite, vector) round-trip.
- Corrupt persistence files raise a clear, file-naming error instead of a bare
  ``JSONDecodeError`` / ``DatabaseError``.

Model behaviour is scripted with the shared mock so the *wiring* is what is
under test; the same behaviour is proven against real models in the integration suite.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.retrieval import Retrieval
from tests.fixtures.mock_models import MockToolCallingModel

# ---------------------------------------------------------------------------
# RAG sources / citations are surfaced
# ---------------------------------------------------------------------------

def _rag_agent(final_answer: str = "The capital of Atlantis is Marisol.") -> Agent:
    rt = Retrieval()
    rt.add_documents(
        [{
            "content": "The capital of Atlantis is Marisol, a coastal city.",
            "id": "d1",
            "metadata": {"source": "atlantis.txt"},
        }],
        chunk=False,
    )
    model = MockToolCallingModel([
        {"thought": "search the kb", "action": "retrieval",
         "action_input": '{"query": "capital of Atlantis"}'},
        {"thought": "I have the answer", "action": "Final Answer",
         "action_input": final_answer},
    ])
    cfg = AgentConfig(name="rag-test", model=model, tools=[rt], max_iterations=4)
    return Agent(cfg)


def test_rag_run_populates_sources_and_citations():
    agent = _rag_agent()
    res = agent.run("What is the capital of Atlantis?")

    assert res.success
    assert res.tool_calls >= 1
    # The headline F44 fix: sources + citations are no longer empty.
    assert res.sources == ["atlantis.txt"]
    assert len(res.citations) == 1
    cite = res.citations[0]
    assert cite.index == 1
    assert cite.source == "atlantis.txt"
    assert "Marisol" in cite.quote
    assert cite.relevance_score > 0


def test_rag_citations_serialize_in_to_dict():
    agent = _rag_agent()
    res = agent.run("What is the capital of Atlantis?")
    data = res.to_dict()
    assert data["sources"] == ["atlantis.txt"]
    assert data["citations"][0]["source"] == "atlantis.txt"
    assert data["citations"][0]["index"] == 1


def test_no_citations_without_retrieval_tool():
    """A non-retrieval tool must not fabricate sources/citations."""
    from effgen.tools.builtin.calculator import Calculator

    model = MockToolCallingModel([
        {"thought": "calc", "action": "calculator",
         "action_input": '{"expression": "15 * 15"}'},
        {"thought": "done", "action": "Final Answer", "action_input": "225"},
    ])
    cfg = AgentConfig(name="calc-test", model=model, tools=[Calculator()], max_iterations=4)
    res = Agent(cfg).run("What is 15 squared?")
    assert res.sources == []
    assert res.citations == []


def test_collect_citations_dedup_and_indexing():
    """Repeated retrieval of the same source is de-duplicated; indices are 1-based."""
    agent = _rag_agent()
    agent._collected_citations = []
    result = {
        "results": [
            {"content": "Alpha passage", "score": 0.9, "id": "a",
             "metadata": {"source": "doc_a.txt"}},
            {"content": "Beta passage", "score": 0.8, "id": "b",
             "metadata": {"source": "doc_b.txt"}},
            # duplicate of the first → must be deduped
            {"content": "Alpha passage", "score": 0.9, "id": "a",
             "metadata": {"source": "doc_a.txt"}},
        ],
    }
    tool = agent.tools["retrieval"]
    agent._collect_citations(tool, "retrieval", result)

    from effgen.core.agent import AgentResponse
    resp = AgentResponse(output="x")
    agent._attach_citations(resp)

    assert resp.sources == ["doc_a.txt", "doc_b.txt"]
    assert [c.index for c in resp.citations] == [1, 2]
    assert {c.source for c in resp.citations} == {"doc_a.txt", "doc_b.txt"}


def test_collect_citations_supports_url_sources():
    """Search-shaped results expose a URL as the source."""
    agent = _rag_agent()
    agent._collected_citations = []
    result = {"results": [
        {"snippet": "Result text", "url": "https://example.com/page", "score": 0.5},
    ]}
    agent._collect_citations(agent.tools["retrieval"], "retrieval", result)
    from effgen.core.agent import AgentResponse
    resp = AgentResponse(output="x")
    agent._attach_citations(resp)
    assert resp.sources == ["https://example.com/page"]


def test_humanize_retrieval_observation():
    """A raw ``{'results': [...]}`` observation renders as passage text."""
    agent = _rag_agent()
    raw = "{'results': [{'content': 'Marisol is the capital.', 'score': 1.0}], 'total_found': 1}"
    rendered = agent._humanize_observation(raw)
    assert rendered == "Marisol is the capital."
    # Non-retrieval observations pass through untouched.
    assert agent._humanize_observation("just a plain answer") == "just a plain answer"


# ---------------------------------------------------------------------------
# Memory round-trips
# ---------------------------------------------------------------------------

def _sample_memory():
    from effgen.memory.long_term import ImportanceLevel, MemoryEntry, MemoryType
    return MemoryEntry(
        id="m1",
        content="The sky is blue.",
        memory_type=MemoryType.FACT,
        importance=ImportanceLevel.HIGH,
        metadata={"k": "v"},
        tags=["weather"],
    )


def test_json_memory_roundtrip(tmp_path):
    from effgen.memory.long_term import JSONStorageBackend
    path = tmp_path / "mem.json"
    backend = JSONStorageBackend(path)
    backend.save_memory(_sample_memory())

    reloaded = JSONStorageBackend(path)
    got = reloaded.get_memory("m1")
    assert got is not None
    assert got.content == "The sky is blue."
    assert got.tags == ["weather"]


def test_sqlite_memory_roundtrip(tmp_path):
    from effgen.memory.long_term import SQLiteStorageBackend
    path = tmp_path / "mem.db"
    backend = SQLiteStorageBackend(path)
    backend.save_memory(_sample_memory())

    reloaded = SQLiteStorageBackend(path)
    got = reloaded.get_memory("m1")
    assert got is not None
    assert got.content == "The sky is blue."
    assert got.metadata == {"k": "v"}


def test_vector_store_roundtrip(tmp_path):
    pytest.importorskip("faiss")
    from effgen.memory.vector_store import VectorMemoryStore
    vd = tmp_path / "vec"
    store = VectorMemoryStore(backend_type="faiss", persist_directory=str(vd))
    store.add(content="Atlantis sits in the western sea.", metadata={"src": "kb"})
    store.save()

    store2 = VectorMemoryStore(backend_type="faiss", persist_directory=str(vd))
    store2.load()
    results = store2.search("Where is Atlantis?", k=1)
    assert results
    assert "Atlantis" in results[0].entry.content


# ---------------------------------------------------------------------------
# Corrupt persistence files → clear, file-naming errors
# ---------------------------------------------------------------------------

def test_corrupt_json_memory_raises_clear_error(tmp_path):
    from effgen.memory.long_term import JSONStorageBackend
    path = tmp_path / "mem.json"
    path.write_text("{ this is not valid json !!!")
    with pytest.raises(ValueError) as exc:
        JSONStorageBackend(path)
    assert "mem.json" in str(exc.value)


def test_corrupt_sqlite_memory_raises_clear_error(tmp_path):
    from effgen.memory.long_term import SQLiteStorageBackend
    path = tmp_path / "mem.db"
    path.write_bytes(b"definitely not a sqlite database, just junk bytes here")
    with pytest.raises(ValueError) as exc:
        SQLiteStorageBackend(path)
    assert "mem.db" in str(exc.value)


def test_corrupt_vector_entries_raises_clear_error(tmp_path):
    pytest.importorskip("faiss")
    from effgen.memory.vector_store import VectorMemoryStore
    vd = tmp_path / "vec"
    vd.mkdir()
    (vd / "entries.json").write_text("{ broken json")
    # The store auto-loads its persist directory on construction.
    with pytest.raises(ValueError) as exc:
        VectorMemoryStore(backend_type="faiss", persist_directory=str(vd))
    assert "entries.json" in str(exc.value)


# ---------------------------------------------------------------------------
# Reranker base placeholder is actionable
# ---------------------------------------------------------------------------

def test_reranker_base_actionable_error():
    from effgen.rag.reranker import Reranker
    with pytest.raises(NotImplementedError) as exc:
        Reranker().rerank("q", [])
    assert "rerank()" in str(exc.value)
