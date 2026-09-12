"""Work handed to another agent is kept as a conversation, not as a string.

A sub-agent, a team stage and a workflow node all used to hand their answer
back as text and throw everything else away: the child's own reasoning, the
calls it made, what it was actually shown. The parent then passed the next
child its context by pasting the earlier answers in front of its question.

What is pinned here:

* a child runs on its own :class:`~effgen.core.thread.AgentThread`, and that
  thread is reachable from the parent's response, survives ``to_dict()`` and is
  written into a checkpoint;
* the parent decides which of its own steps a child starts with — a projection
  of the parent's thread, selected, not concatenated;
* a workflow checkpoint round-trips with every node's thread;
* a DAG with a failing node says which node failed and what its thread held;
* a child's prompt-token budget is the parent's, and the projected steps count
  against it.

The shapes the change was not written for are here too: a child that calls a
tool, a child that fails, children nested two deep, and a workflow node that is
not an agent at all.
"""

from __future__ import annotations

import json

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.checkpoint import Checkpoint
from effgen.core.orchestrator import (
    MultiAgentOrchestrator,
    OrchestrationPattern,
    TeamConfig,
)
from effgen.core.task import SubTask
from effgen.core.thread import (
    ActionStep,
    AgentThread,
    AnswerStep,
    DelegationStep,
    ObservationStep,
    TaskStep,
    ThoughtStep,
    TurnStep,
    step_from_dict,
)
from effgen.core.thread_projection import (
    LastCycles,
    NoParentContext,
    ParentAnswers,
    ParentTask,
    ThreadProjection,
    resolve_projection,
)
from effgen.core.workflow import WorkflowDAG, WorkflowNode
from effgen.core.workflow_checkpoint import (
    InMemoryCheckpointStore,
    WorkflowCheckpoint,
)
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.builtin.calculator import Calculator


class _Scripted(BaseModel):
    """Says what it was told to say, in order, and records every prompt."""

    def __init__(self, turns: list[str], *, window: int = 4096) -> None:
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self._turns = turns
        self._window = window
        self.prompts: list[str] = []
        self.calls = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        self.prompts.append(prompt)
        text = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs):
        yield self.generate(prompt).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return self._window

    def supports_tool_calling(self) -> bool:
        return False


def _parent(model: BaseModel, **kw) -> Agent:
    return Agent(AgentConfig(
        name="parent", model=model, require_model=False,
        enable_sub_agents=True, enable_memory=False, raise_on_error=False, **kw
    ))


def _subtasks(*descriptions: str) -> list[SubTask]:
    return [
        SubTask(id=f"st_{i}", description=text, expected_output="x")
        for i, text in enumerate(descriptions, start=1)
    ]


# --------------------------------------------------------------------------- #
# The step itself
# --------------------------------------------------------------------------- #
def test_a_delegation_carries_the_child_thread_through_serialisation():
    child = AgentThread(steps=[
        TaskStep(text="add them"),
        ThoughtStep(text="use the calculator"),
        ActionStep(tool="calculator", arguments={"expression": "1+1"}),
        ObservationStep(text="2"),
        AnswerStep(text="2", stop_reason="final_answer"),
    ])
    parent = AgentThread(steps=[
        TaskStep(text="the whole job"),
        DelegationStep(child_id="st_1", role="sub-agent", task="add them",
                       thread=child, output="2"),
    ])

    data = parent.to_dict()
    json.dumps(data)                      # a saved document stays a document
    back = AgentThread.from_dict(data)

    assert back.child_threads()["st_1"].to_dict() == child.to_dict()
    assert [s.kind for s in back.child_threads()["st_1"].steps] == [
        "task", "thought", "action", "observation", "answer",
    ]


def test_a_delegation_renders_as_nothing_in_the_transcript():
    # Every existing prompt is a rendering of a thread. A step that rendered
    # text would change every prompt a delegating run has ever sent.
    thread = AgentThread(steps=[
        ThoughtStep(text="t"),
        DelegationStep(child_id="c", output="whatever the child said"),
    ])
    assert thread.to_text() == ThoughtStep(text="t").to_text()


def test_a_delegation_without_a_child_thread_is_still_recorded():
    thread = AgentThread(steps=[
        DelegationStep(child_id="n1", role="node", output="done", thread=None),
    ])
    assert [d.child_id for d in thread.delegations()] == ["n1"]
    assert thread.child_threads() == {}      # nothing to inspect, and it says so
    assert AgentThread.from_dict(thread.to_dict()).delegations()[0].output == "done"


def test_an_unknown_step_kind_still_raises():
    with pytest.raises(ValueError, match="not one this release knows"):
        step_from_dict({"kind": "handoff"})


# --------------------------------------------------------------------------- #
# The projection: selected, not concatenated
# --------------------------------------------------------------------------- #
def _worked_parent() -> AgentThread:
    return AgentThread(steps=[
        TaskStep(text="plan the launch"),
        ThoughtStep(text="check the date"),
        ActionStep(tool="calendar", arguments={"q": "date"}),
        ObservationStep(text="14 March"),
        ThoughtStep(text="check the venue"),
        ActionStep(tool="search", arguments={"q": "venue"}),
        ObservationStep(text="the hall"),
        DelegationStep(child_id="budget", role="sub-agent", output="12k"),
        DelegationStep(child_id="catering", role="sub-agent",
                       success=False, error="no supplier"),
    ])


def test_the_default_projection_shows_a_child_nothing():
    assert NoParentContext().project(_worked_parent(), task="x") == []


def test_a_projection_carries_the_parent_task_as_a_turn_not_as_the_question():
    carried = ParentTask().project(_worked_parent(), task="book the hall")
    assert [s.kind for s in carried] == ["turn"]
    assert isinstance(carried[0], TurnStep)
    assert carried[0].role == "user" and carried[0].text == "plan the launch"


def test_a_projection_carries_what_the_finished_children_answered():
    carried = ParentAnswers().project(_worked_parent(), task="book the hall")
    texts = [s.text for s in carried]
    assert texts[0] == "plan the launch"
    assert "budget: 12k" in texts
    assert any("catering" in t and "no supplier" in t for t in texts)


def test_a_projection_can_be_limited_to_the_most_recent_children():
    carried = ParentAnswers(include_task=False, limit=1).project(
        _worked_parent(), task="x",
    )
    assert len(carried) == 1 and "catering" in carried[0].text


def test_last_cycles_carries_whole_cycles_and_no_unanswered_call():
    carried = LastCycles(1).project(_worked_parent(), task="x")
    assert [s.kind for s in carried] == ["turn", "thought", "action", "observation"]
    assert carried[2].tool == "search"


def test_a_projection_never_hands_over_a_call_nothing_answered():
    parent = AgentThread(steps=[
        TaskStep(text="job"),
        ThoughtStep(text="t"),
        ActionStep(tool="search", arguments={}),   # no observation follows
    ])

    class _Everything(ThreadProjection):
        name = "everything"

        def select(self, parent, *, task):
            return list(parent.steps)

    carried = _Everything().project(parent, task="x")
    assert not any(isinstance(s, ActionStep) for s in carried)


def test_a_projection_drops_a_result_whose_call_was_not_selected():
    parent = AgentThread(steps=[
        ActionStep(tool="search", arguments={}, call_id="c1"),
        ObservationStep(text="hit", call_id="c1"),
    ])

    class _ResultsOnly(ThreadProjection):
        name = "results-only"

        def select(self, parent, *, task):
            return [s for s in parent.steps if isinstance(s, ObservationStep)]

    assert _ResultsOnly().project(parent, task="x") == []


def test_a_projection_of_no_parent_is_empty():
    assert ParentAnswers().project(None, task="x") == []


@pytest.mark.parametrize(
    "value,expected",
    [(None, "none"), ("parent-task", "parent-task"), (ParentAnswers, "parent-answers")],
)
def test_a_projection_is_resolved_from_whatever_the_caller_passed(value, expected):
    assert resolve_projection(value).name == expected


def test_an_unknown_projection_name_says_what_exists():
    with pytest.raises(ValueError, match="last-cycles"):
        resolve_projection("everything")


def test_a_projection_cannot_be_built_from_a_number():
    with pytest.raises(TypeError, match="ThreadProjection"):
        resolve_projection(7)


def test_a_negative_cycle_count_is_refused():
    with pytest.raises(ValueError, match="n=0"):
        LastCycles(-1)


# --------------------------------------------------------------------------- #
# Sub-agents: the child's thread reaches the parent's response
# --------------------------------------------------------------------------- #
def test_a_sub_agent_run_hands_back_every_child_thread():
    agent = _parent(_Scripted(["Final Answer: done"]))
    results = agent.sub_agent_manager.execute_sequential(_subtasks("one", "two"))

    assert [r.success for r in results] == [True, True]
    assert all(isinstance(r.thread, AgentThread) for r in results)
    assert all(r.thread.task().text in ("one", "two") for r in results)


def test_a_decomposed_response_reaches_its_children_threads():
    agent = _parent(_Scripted(["Final Answer: done"]))
    thread = AgentThread(steps=[TaskStep(text="the whole job")])
    agent.sub_agent_manager.parent_thread = thread
    agent.sub_agent_manager.execute_sequential(_subtasks("one", "two"))

    assert sorted(thread.child_threads()) == ["st_1", "st_2"]
    assert thread.child_threads()["st_1"].task().text == "one"


def test_the_documented_accessor_is_agent_response_sub_agent_threads():
    from effgen.core.agent_response import AgentResponse

    child = AgentThread(steps=[TaskStep(text="one")])
    parent = AgentThread(steps=[
        TaskStep(text="job"),
        DelegationStep(child_id="st_1", thread=child),
    ])
    response = AgentResponse(output="done", metadata={"thread": parent})

    assert list(response.sub_agent_threads()) == ["st_1"]
    assert response.sub_agent_threads()["st_1"] is child
    # ...and the same data is in the document the response writes.
    written = response.to_dict()
    json.dumps(written)
    steps = written["metadata"]["thread"]["steps"]
    assert steps[1]["thread"]["steps"][0]["text"] == "one"


def test_a_child_thread_survives_a_checkpoint():
    child = AgentThread(steps=[TaskStep(text="one"), AnswerStep(text="1")])
    parent = AgentThread(steps=[
        TaskStep(text="job"), DelegationStep(child_id="st_1", thread=child),
    ])
    saved = Checkpoint(
        checkpoint_id="cp-1", agent_name="parent", task="job", iteration=1,
        thread=parent.to_dict(),
    )

    back = Checkpoint.from_dict(json.loads(json.dumps(saved.to_dict()))).to_thread()

    assert back.child_threads()["st_1"].to_dict() == child.to_dict()


def test_a_child_starts_with_the_steps_its_parent_chose():
    model = _Scripted(["Final Answer: done"])
    agent = _parent(model)
    agent.sub_agent_manager.projection = ParentAnswers()
    agent.sub_agent_manager.parent_thread = AgentThread(steps=[
        TaskStep(text="plan the launch"),
        DelegationStep(child_id="budget", output="12k"),
    ])

    agent.sub_agent_manager.execute_sequential(_subtasks("book the hall"))

    prompt = model.prompts[-1]
    assert "plan the launch" in prompt and "budget: 12k" in prompt
    # The child's question is still its own question, not a paste-up of both.
    result = agent.sub_agent_manager.sub_agent_results["st_1"]
    assert result.thread.task().text == "book the hall"


def test_a_second_child_sees_what_the_first_answered():
    model = _Scripted(["Final Answer: 12k"])
    agent = _parent(model)
    agent.sub_agent_manager.projection = ParentAnswers()
    agent.sub_agent_manager.parent_thread = AgentThread(
        steps=[TaskStep(text="plan the launch")],
    )

    agent.sub_agent_manager.execute_sequential(_subtasks("the budget", "the venue"))

    assert "st_1: 12k" in model.prompts[-1]


# --------------------------------------------------------------------------- #
# Shapes this was not written for
# --------------------------------------------------------------------------- #
def test_a_child_that_calls_a_tool_keeps_the_call_on_its_thread():
    model = _Scripted([
        "Thought: compute\nAction: calculator\nAction Input: 2+2",
        "Final Answer: 4",
    ])
    agent = Agent(AgentConfig(
        name="parent", model=model, require_model=False, tools=[Calculator()],
        enable_sub_agents=True, enable_memory=False, raise_on_error=False,
    ))

    results = agent.sub_agent_manager.execute_sequential(_subtasks("add 2 and 2"))

    thread = results[0].thread
    assert [s.tool for s in thread.actions()] == ["calculator"]
    assert thread.observations() and thread.unanswered_call_ids() == []


def test_a_child_that_fails_is_recorded_with_what_went_wrong():
    agent = _parent(_Scripted(["Final Answer: done"]))
    agent.sub_agent_manager.parent_agent = None      # no model: the child raises
    thread = AgentThread(steps=[TaskStep(text="job")])
    agent.sub_agent_manager.parent_thread = thread

    results = agent.sub_agent_manager.execute_sequential(_subtasks("one"))

    assert results[0].success is False and results[0].thread is None
    step = thread.delegations()[0]
    assert step.success is False and "RuntimeError" in (step.error or "")


def test_children_nested_two_deep_each_keep_their_own_thread():
    inner = AgentThread(steps=[TaskStep(text="innermost")])
    middle = AgentThread(steps=[
        TaskStep(text="middle"), DelegationStep(child_id="inner", thread=inner),
    ])
    outer = AgentThread(steps=[
        TaskStep(text="outer"), DelegationStep(child_id="middle", thread=middle),
    ])

    back = AgentThread.from_dict(json.loads(json.dumps(outer.to_dict())))
    deep = back.child_threads()["middle"].child_threads()["inner"]
    assert deep.task().text == "innermost"


def test_a_workflow_node_that_is_not_an_agent_records_no_thread():
    class _Plain:
        name = "plain"

        def run(self, task, **kw):
            class _R:
                output = "computed"
                success = True
                tokens_used = 0
                metadata: dict = {}
            return _R()

    dag = WorkflowDAG("plain")
    dag.add_node(WorkflowNode(id="n1", agent=_Plain()))
    result = dag.run("go")

    assert result.success and result.outputs["n1"] == "computed"
    assert result.node_thread("n1") is None
    assert result.thread.delegations()[0].output == "computed"
    json.dumps(result.to_dict())


# --------------------------------------------------------------------------- #
# Workflows: checkpoints round-trip, and a failure says where it stopped
# --------------------------------------------------------------------------- #
class _ThreadedAgent:
    """A node agent that answers and hands back a conversation, like a real one."""

    def __init__(self, name: str, *, succeed: bool = True) -> None:
        self.name = name
        self._succeed = succeed
        self.prior: list = []

    def run(self, task: str, context=None, **kw):
        self.prior.append(list(kw.get("_prior_steps") or []))
        thread = AgentThread(steps=[
            TaskStep(text=task),
            ThoughtStep(text=f"{self.name} thinking"),
        ])

        class _R:
            pass

        r = _R()
        if self._succeed:
            thread.append(AnswerStep(text=f"{self.name}-ok", stop_reason="final_answer"))
            r.output, r.success = f"{self.name}-ok", True
            r.metadata = {"thread": thread}
        else:
            thread.append(AnswerStep(text="", stop_reason="tool_failed"))
            r.output, r.success = "the node broke", False
            r.metadata = {"thread": thread,
                          "error": {"type": "ToolError", "message": "the node broke"}}
        r.tokens_used = 3
        return r


def _two_node_dag(*, second_ok: bool = True, projection=None) -> WorkflowDAG:
    dag = WorkflowDAG("pipeline", projection=projection)
    dag.add_node(WorkflowNode(id="first", agent=_ThreadedAgent("first")))
    dag.add_node(WorkflowNode(id="second",
                              agent=_ThreadedAgent("second", succeed=second_ok)))
    dag.connect("first", "second")
    return dag


def test_every_node_hands_back_its_own_thread():
    result = _two_node_dag().run("go")

    assert result.success
    assert sorted(result.threads) == ["first", "second"]
    assert result.node_thread("first").task().text == "go"
    assert [d.child_id for d in result.thread.delegations()] == ["first", "second"]


def test_a_workflow_checkpoint_round_trips_with_every_node_thread():
    store = InMemoryCheckpointStore()
    result = _two_node_dag().run("go", checkpoint=store, run_id="r1")
    assert result.success

    saved = store.load("r1")
    written = json.loads(json.dumps(saved.to_dict()))
    back = WorkflowCheckpoint.from_dict(written)

    for nid in ("first", "second"):
        assert back.thread_for(nid).to_dict() == result.node_thread(nid).to_dict()
    assert back.tasks["first"] == "go"


def test_a_resumed_workflow_still_has_the_threads_of_the_nodes_it_skipped():
    store = InMemoryCheckpointStore()
    first_run = _two_node_dag(second_ok=False).run("go", checkpoint=store, run_id="r2")
    assert not first_run.success

    fixed = _two_node_dag()
    resumed = fixed.run("go", checkpoint=store, run_id="r2")

    assert resumed.success
    # "first" was not run again, and its conversation came back from the store.
    assert fixed.get_node("first").agent.prior == []
    assert resumed.node_thread("first").to_dict() == \
        first_run.node_thread("first").to_dict()


def test_a_failing_node_says_which_node_and_what_its_thread_held():
    result = _two_node_dag(second_ok=False).run("go")

    assert not result.success
    failed = result.failed_nodes()
    assert [f["node_id"] for f in failed] == ["second"]
    assert "the node broke" in failed[0]["error"]
    thread = failed[0]["thread"]
    assert [s.kind for s in thread.steps] == ["task", "thought", "answer"]
    assert thread.steps[-1].stop_reason == "tool_failed"
    assert thread.steps[1].text == "second thinking"


def test_a_workflow_node_can_be_shown_the_run_as_a_conversation():
    dag = _two_node_dag(projection=ParentAnswers())
    dag.run("go")

    carried = dag.get_node("second").agent.prior[0]
    assert [s.kind for s in carried] == ["turn", "turn"]
    assert "first-ok" in carried[-1].text


def test_a_workflow_shows_a_node_nothing_by_default():
    dag = _two_node_dag()
    dag.run("go")
    assert dag.get_node("second").agent.prior == [[], []][:1] or \
        dag.get_node("second").agent.prior[0] == []


# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #
def test_a_team_hands_back_a_thread_per_stage():
    from tests.unit.test_orchestration import FakeAgent

    orch = MultiAgentOrchestrator()
    team = TeamConfig(name="team", pattern=OrchestrationPattern.SEQUENTIAL,
                      agents=[FakeAgent("a"), FakeAgent("b")])
    res = orch.assign_task("go", team)

    assert res.success
    assert [d.child_id for d in res.thread.delegations()] == ["a", "b"]
    assert res.thread.steps[-1].kind == "answer"
    json.dumps(res.to_dict())


def test_a_team_agent_speaking_twice_keeps_a_thread_per_round():
    from effgen.core.orchestrator import _TeamRun
    from tests.unit.test_orchestration import _FakeResponse

    run = _TeamRun("go", None)
    run.record("a", "round 1", _FakeResponse(output="first"), role="collaborator")
    run.record("a", "round 2", _FakeResponse(output="second"), role="collaborator")
    run.finish("second", success=True)

    assert [d.child_id for d in run.thread.delegations()] == ["a", "a#1"]
    assert [d.output for d in run.thread.delegations()] == ["first", "second"]


def test_a_team_result_metadata_still_carries_no_raw_model_text():
    from tests.unit.test_orchestration import FakeAgent

    orch = MultiAgentOrchestrator()
    team = TeamConfig(name="team", pattern=OrchestrationPattern.SEQUENTIAL,
                      agents=[FakeAgent("a", succeed=False)])
    res = orch.assign_task("go", team)

    assert "sk-secret-key-12345" not in str(res.metadata)


# --------------------------------------------------------------------------- #
# The budget a run may send applies to a child too
# --------------------------------------------------------------------------- #
def test_a_child_inherits_the_budget_its_parent_was_given():
    model = _Scripted(["Final Answer: done"])
    agent = _parent(model, context_budget=1500, max_context_length=8000)

    agent.sub_agent_manager.execute_sequential(_subtasks("one"))
    child_thread = agent.sub_agent_manager.sub_agent_results["st_1"].thread

    assert child_thread.metadata["context_budget"]["budget_tokens"] == 1500


def test_a_child_of_an_unbounded_parent_is_unbounded_too():
    model = _Scripted(["Final Answer: done"])
    agent = _parent(model, context_budget=None)

    agent.sub_agent_manager.execute_sequential(_subtasks("one"))
    child_thread = agent.sub_agent_manager.sub_agent_results["st_1"].thread

    assert child_thread.metadata["context_budget"]["budget_tokens"] is None


def test_the_projected_steps_count_against_the_child_budget():
    model = _Scripted(["Final Answer: done"])
    lean = _parent(model)
    lean.sub_agent_manager.execute_sequential(_subtasks("one"))
    without = len(model.prompts[-1])

    model2 = _Scripted(["Final Answer: done"])
    rich = _parent(model2)
    rich.sub_agent_manager.projection = ParentAnswers()
    rich.sub_agent_manager.parent_thread = AgentThread(steps=[
        TaskStep(text="plan the launch"),
        DelegationStep(child_id="budget", output="12k " * 50),
    ])
    rich.sub_agent_manager.execute_sequential(_subtasks("one"))
    with_projection = len(model2.prompts[-1])

    assert with_projection > without


# --------------------------------------------------------------------------- #
# A tool-free run: the one shape of run that handed back no conversation
# --------------------------------------------------------------------------- #
def test_a_tool_free_run_hands_back_its_conversation():
    model = _Scripted(["42"])
    agent = Agent(AgentConfig(name="a", model=model, require_model=False,
                              enable_memory=False, raise_on_error=False))
    response = agent.run("what is 6 times 7?")
    agent.close()

    thread = response.metadata["thread"]
    assert [s.kind for s in thread.steps] == ["task", "answer"]
    assert thread.task().text == "what is 6 times 7?"
    assert thread.steps[-1].text == "42"
    assert thread.to_text() == ""            # it renders no prompt bytes
    assert thread.metadata["context_budget"]["budget_tokens"] is not None


def test_a_tool_free_run_carries_its_persona_and_earlier_turns():
    model = _Scripted(["42", "42"])
    agent = Agent(AgentConfig(name="a", model=model, require_model=False,
                              system_prompt="Answer only in French.",
                              enable_memory=True, raise_on_error=False))
    agent.run("what is 6 times 7?")
    response = agent.run("and again?")
    agent.close()

    kinds = [s.kind for s in response.metadata["thread"].steps]
    assert kinds[0] == "system" and "turn" in kinds and kinds[-1] == "answer"


def test_a_tool_free_run_that_fails_still_hands_back_where_it_got_to():
    class _Broken(_Scripted):
        def generate(self, prompt, config=None, **kwargs):
            raise RuntimeError("the provider refused")

    agent = Agent(AgentConfig(name="a", model=_Broken(["x"]), require_model=False,
                              enable_memory=False, raise_on_error=False))
    response = agent.run("what is 6 times 7?")
    agent.close()

    assert response.success is False
    thread = response.metadata["thread"]
    assert thread.task().text == "what is 6 times 7?"
    assert thread.steps[-1].stop_reason == "generation_failed"
