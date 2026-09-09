"""Live integration tests for multi-agent orchestration & workflows.

Real model calls (no mocks). Cloud patterns run on Groq's cheap
``openai/gpt-oss-20b`` when ``GROQ_API_KEY`` is set; otherwise skipped.

Proves the machinery end-to-end:
- sequential / parallel / hierarchical team patterns succeed,
- a DAG (fan-out/fan-in) runs and routes a bare-string task to its entry node,
- a deliberately-bad model id inside a team and inside a DAG node yields an
  a clear ``success=False`` with a per-agent/node error (never a silent success).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)

CLOUD_MODEL = "groq:openai/gpt-oss-20b"


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.api,
    pytest.mark.skipif(not _has_groq(), reason="SKIPPED: GROQ_API_KEY not set"),
]


def _mk_agent(name: str):
    from effgen import Agent, AgentConfig, load_model
    return Agent(AgentConfig(name=name, model=load_model(CLOUD_MODEL),
                             enable_sub_agents=False, enable_memory=False))


def _skip_if_rate_limited(exc: Exception) -> None:
    from effgen.models._rate_limit import RateLimitExceeded
    if isinstance(exc, RateLimitExceeded):
        pytest.skip(f"Groq transient rate limit: {exc}")
    raise exc


def _skip_if_refused(res: object) -> None:
    """A team or DAG reports a refused call in its result, not by raising.

    ``_skip_if_rate_limited`` only sees exceptions, and neither orchestrator
    lets one out: a node that the provider would not serve comes back as
    ``success=False`` with the 429 recorded per node. Without this the run reads
    as ``assert False is True`` — an effGen defect — when the account is simply
    out of tokens for the day.
    """
    from tests._harness.provider_unavailable import skip_if_provider_refused
    skip_if_provider_refused(res, what="the orchestrated call")


def test_sequential_team_live():
    from effgen import MultiAgentOrchestrator, OrchestrationPattern
    a, b = _mk_agent("drafter"), _mk_agent("editor")
    try:
        orch = MultiAgentOrchestrator()
        orch.create_team("seq", [a, b], pattern=OrchestrationPattern.SEQUENTIAL)
        res = orch.assign_task("Write one short sentence about the sea.", "seq")
        _skip_if_refused(res)
        assert res.success is True
        assert len(res.agent_responses) == 2
        assert res.output.strip()
    except Exception as e:  # transient quota
        _skip_if_rate_limited(e)
    finally:
        a.close()
        b.close()


def test_parallel_team_live():
    from effgen import MultiAgentOrchestrator, OrchestrationPattern
    a, b = _mk_agent("p1"), _mk_agent("p2")
    try:
        orch = MultiAgentOrchestrator()
        orch.create_team("par", [a, b], pattern=OrchestrationPattern.PARALLEL)
        res = orch.assign_task("Name one primary color.", "par")
        _skip_if_refused(res)
        assert res.success is True
        assert len(res.agent_responses) == 2
    except Exception as e:
        _skip_if_rate_limited(e)
    finally:
        a.close()
        b.close()


def test_hierarchical_team_live():
    from effgen import MultiAgentOrchestrator, OrchestrationPattern
    mgr, w1, w2 = _mk_agent("manager"), _mk_agent("w1"), _mk_agent("w2")
    try:
        orch = MultiAgentOrchestrator()
        orch.create_team("hier", [w1, w2],
                         pattern=OrchestrationPattern.HIERARCHICAL, manager_agent=mgr)
        res = orch.assign_task("List two benefits of sleep.", "hier")
        _skip_if_refused(res)
        assert res.success is True
        assert res.output.strip()
        # The manager's initial decomposition call is billed too (3 manager/
        # worker calls total: decomposition + 2 workers, then synthesis) — the
        # team total must exceed the sum of just the two worker responses.
        worker_cost = sum(r["cost_usd"] for r in res.agent_responses)
        assert res.metadata["cost_usd"] > worker_cost
        assert res.metadata["tokens_used"] > sum(r["tokens_used"] for r in res.agent_responses)
    except Exception as e:
        _skip_if_rate_limited(e)
    finally:
        mgr.close()
        w1.close()
        w2.close()


def test_dag_fanout_fanin_live():
    from effgen import WorkflowDAG, WorkflowNode
    src, b1, b2, join = (_mk_agent("src"), _mk_agent("b1"),
                         _mk_agent("b2"), _mk_agent("join"))
    try:
        dag = WorkflowDAG("fan")
        for nid, ag in [("src", src), ("b1", b1), ("b2", b2), ("join", join)]:
            dag.add_node(WorkflowNode(id=nid, agent=ag))
        dag.connect("src", "b1")
        dag.connect("src", "b2")
        dag.connect("b1", "join")
        dag.connect("b2", "join")
        assert dag.topological_order()[0] == "src"
        # bare string routed to the entry node:
        res = dag.run("Reply with the single word: ping.")
        _skip_if_refused(res)
        assert res.success is True
        statuses = {n["id"]: n["status"] for n in res.node_results}
        assert all(s == "completed" for s in statuses.values())
    except Exception as e:
        _skip_if_rate_limited(e)
    finally:
        for ag in (src, b1, b2, join):
            ag.close()


def test_team_with_bad_model_reports_the_failure_live():
    from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern
    good = _mk_agent("good")
    bad = Agent(AgentConfig(name="bad", model="totally-not-a-real-model-xyz",
                            provider="groq", require_model=False,
                            enable_sub_agents=False, enable_memory=False))
    try:
        orch = MultiAgentOrchestrator()
        orch.create_team("ft", [good, bad], pattern=OrchestrationPattern.SEQUENTIAL)
        res = orch.assign_task("Say hi.", "ft")
        assert res.success is False
        assert res.metadata.get("reason") in ("sub_agent_failed", "team_error")
        assert res.metadata.get("error")
    except Exception as e:
        _skip_if_rate_limited(e)
    finally:
        good.close()
        bad.close()


def test_dag_with_bad_node_reports_the_failure_live():
    from effgen import Agent, AgentConfig, WorkflowDAG, WorkflowNode
    ok = _mk_agent("ok")
    bad = Agent(AgentConfig(name="bad", model="totally-not-a-real-model-xyz",
                            provider="groq", require_model=False,
                            enable_sub_agents=False, enable_memory=False))
    try:
        dag = WorkflowDAG("bad")
        dag.add_node(WorkflowNode(id="ok", agent=ok))
        dag.add_node(WorkflowNode(id="bad", agent=bad))
        dag.connect("ok", "bad")
        res = dag.run("hi")
        _skip_if_refused(res)
        statuses = {n["id"]: n["status"] for n in res.node_results}
        assert statuses["ok"] == "completed"
        assert statuses["bad"] == "failed"
        assert res.success is False
    except Exception as e:
        _skip_if_rate_limited(e)
    finally:
        ok.close()
        bad.close()
