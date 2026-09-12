# Multi-Agent Orchestration

effGen supports task decomposition via sub-agents.

## How Sub-Agents Work

When `enable_sub_agents=True` (default), the agent can decompose complex tasks:

1. The router analyzes the task complexity
2. If sub-agents are needed, the task is split into subtasks
3. Each subtask runs with its own agent configuration
4. Results are aggregated into a final response

## Example

```python
from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator, PythonREPL, WebSearch

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = Agent(AgentConfig(
    name="orchestrator",
    model=model,
    tools=[Calculator(), PythonREPL(), WebSearch()],
    enable_sub_agents=True,
    max_iterations=15,
))

# Complex task that benefits from decomposition
result = agent.run(
    "Research the GDP of the top 5 economies, then calculate "
    "the average and standard deviation"
)
```

## Execution Modes

```python
from effgen.core.agent import AgentMode

# Let the agent decide
result = agent.run(task, mode=AgentMode.AUTO)

# Force single-agent execution
result = agent.run(task, mode=AgentMode.SINGLE)

# Force sub-agent decomposition
result = agent.run(task, mode=AgentMode.SUB_AGENTS)
```

## Reading what a child actually did

A child runs on its own conversation, and that conversation comes back with the
parent's response. `response.sub_agent_threads()` is the accessor: it returns
`{child_id: AgentThread}` for every child that reached a model, in the order the
work was handed out.

```python
result = agent.run(task, mode=AgentMode.SUB_AGENTS)

for child_id, thread in result.sub_agent_threads().items():
    print(child_id, thread.task().text)
    for step in thread.actions():
        print("   called", step.tool, step.arguments)
```

Every delegation is on the parent's own thread, whether or not the child
produced a conversation — a child that failed before reaching a model has no
thread but still has a record:

```python
for step in result.metadata["thread"].delegations():
    print(step.child_id, step.success, step.error or step.output[:60])
```

Both survive `result.to_dict()`, so a saved run document carries them, and both
travel into a checkpoint with the rest of the parent's steps.

A workflow keeps the same record. `WorkflowResult.node_thread(node_id)` returns
one node's conversation, and `failed_nodes()` says which nodes failed, why, and
what each one's thread held when it stopped:

```python
result = dag.run("draft the memo")

for failure in result.failed_nodes():
    print(failure["node_id"], failure["error"])
    print("   got as far as", [s.kind for s in failure["thread"].steps])
```

A team result carries `TeamResponse.thread` and `TeamResponse.agent_threads()`
in the same shape.

## Choosing what a child is shown

By default a child sees only the question it was asked. To give it more, pass a
projection — a rule that *selects* steps from the parent's conversation, which
the child then starts its own with. The child's question stays its own question,
and what it was shown is on its thread where the context budget counts it.

```python
from effgen import ParentAnswers

agent.sub_agent_manager.projection = ParentAnswers()   # or "parent-answers"
```

`NoParentContext` (the default) carries nothing, `ParentTask` carries the job
the parent was given, `ParentAnswers` adds what each finished child answered,
and `LastCycles(n)` carries the parent's own last `n` complete cycles of work.
Subclass `ThreadProjection` for a rule of your own. A workflow and a team take
the same argument: `WorkflowDAG(projection=...)` and
`TeamConfig(projection=...)`.

A child inherits its parent's `context_budget`, `compaction` and
`max_context_length`, so a decomposition that is bounded stays bounded all the
way down, and the projected steps are given up before the child's own question
is.

## Tips

- Sub-agents work best with larger models (3B+)
- Simple tasks run faster in `SINGLE` mode
- Use `--verbose` in CLI to see the routing decision
