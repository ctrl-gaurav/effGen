"""The four things a caller can do once a child's work is a conversation.

Every import here is a name that existed before delegations were kept as
steps, so this module collects against the previous release and fails on its
assertions rather than on its imports — the claims are about behaviour, and
they are checked as behaviour.

One claim each:

* a sub-agent's thread is reachable from the parent's response, and survives
  the document the response writes;
* a workflow checkpoint round-trips with every node's thread;
* a DAG with a failing node says which node failed and what its thread held;
* a run with no tools hands back a conversation like every other run.
"""

from __future__ import annotations

import json

from effgen.core.agent import Agent, AgentConfig
from effgen.core.router import RoutingDecision, RoutingStrategy
from effgen.core.task import SubTask
from effgen.core.workflow import WorkflowDAG, WorkflowNode
from effgen.core.workflow_checkpoint import InMemoryCheckpointStore
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount


class _Answers(BaseModel):
    """Answers whatever it is asked, in one turn."""

    def __init__(self) -> None:
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        return GenerationResult(
            text="Final Answer: done", tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs):
        yield self.generate(prompt).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def supports_tool_calling(self) -> bool:
        return False


class _NodeAgent:
    """A node agent that answers and hands back the conversation it had."""

    def __init__(self, name: str, *, succeed: bool = True) -> None:
        self.name = name
        self._succeed = succeed

    def run(self, task: str, context=None, **kw):
        from effgen.core.thread import AgentThread, AnswerStep, TaskStep, ThoughtStep

        thread = AgentThread(steps=[
            TaskStep(text=task), ThoughtStep(text=f"{self.name} thinking"),
        ])

        class _R:
            pass

        r = _R()
        r.tokens_used = 3
        if self._succeed:
            thread.append(AnswerStep(text=f"{self.name}-ok", stop_reason="final_answer"))
            r.output, r.success, r.metadata = f"{self.name}-ok", True, {"thread": thread}
        else:
            thread.append(AnswerStep(text="", stop_reason="tool_failed"))
            r.output, r.success = "the node broke", False
            r.metadata = {"thread": thread,
                          "error": {"type": "ToolError", "message": "the node broke"}}
        return r


def _dag(*, second_ok: bool = True) -> WorkflowDAG:
    dag = WorkflowDAG("pipeline")
    dag.add_node(WorkflowNode(id="first", agent=_NodeAgent("first")))
    dag.add_node(WorkflowNode(id="second", agent=_NodeAgent("second", succeed=second_ok)))
    dag.connect("first", "second")
    return dag


def test_a_parent_response_reaches_the_thread_of_every_child_it_spawned():
    agent = Agent(AgentConfig(
        name="parent", model=_Answers(), require_model=False,
        enable_sub_agents=True, enable_memory=False, raise_on_error=False,
    ))
    # The router only decomposes a task it judges complex enough, and which
    # tasks those are is not what this pins; the decision is handed in so the
    # claim is about what a decomposed run gives back.
    decision = RoutingDecision(
        use_sub_agents=True,
        strategy=RoutingStrategy.SEQUENTIAL_SUB_AGENTS,
        num_sub_agents=2,
        decomposition=[
            SubTask(id="st_1", description="research the market", expected_output="x"),
            SubTask(id="st_2", description="draft the summary", expected_output="x"),
        ],
    )
    try:
        response = agent._run_with_sub_agents(
            "research the market, then draft a summary", decision, {},
        )
    finally:
        agent.close()

    threads = response.sub_agent_threads()
    assert threads, "a decomposed run must hand back its children's conversations"
    for child_id, thread in threads.items():
        assert thread.task() is not None, f"{child_id} kept no question"
        assert thread.steps[-1].kind == "answer"

    # ...and the same data is in the document the response writes.
    written = json.loads(json.dumps(response.to_dict()))
    steps = written["metadata"]["thread"]["steps"]
    nested = [s for s in steps if s.get("kind") == "delegation" and s.get("thread")]
    assert len(nested) == len(threads)


def test_a_workflow_checkpoint_round_trips_with_every_node_thread():
    store = InMemoryCheckpointStore()
    result = _dag().run("go", checkpoint=store, run_id="r1")
    assert result.success

    saved = store.load("r1")
    reread = type(saved).from_dict(json.loads(json.dumps(saved.to_dict())))

    for node_id in ("first", "second"):
        recovered = reread.thread_for(node_id)
        assert recovered is not None, f"{node_id} was stored without its conversation"
        assert [s.kind for s in recovered.steps] == ["task", "thought", "answer"]
        assert recovered.to_dict() == result.node_thread(node_id).to_dict()


def test_a_dag_with_a_failing_node_says_which_node_and_what_its_thread_held():
    result = _dag(second_ok=False).run("go")

    assert not result.success
    failed = result.failed_nodes()
    assert [f["node_id"] for f in failed] == ["second"]
    assert "the node broke" in failed[0]["error"]
    held = failed[0]["thread"]
    assert held is not None, "a failed node must say how far it got"
    assert held.steps[1].text == "second thinking"
    assert held.steps[-1].stop_reason == "tool_failed"


def test_a_run_with_no_tools_hands_back_its_conversation():
    agent = Agent(AgentConfig(
        name="a", model=_Answers(), require_model=False,
        enable_memory=False, raise_on_error=False,
    ))
    try:
        response = agent.run("what is 6 times 7?")
    finally:
        agent.close()

    thread = response.metadata.get("thread")
    assert thread is not None, "a tool-free run built no conversation"
    assert thread.task().text == "what is 6 times 7?"
    assert thread.to_text() == ""
