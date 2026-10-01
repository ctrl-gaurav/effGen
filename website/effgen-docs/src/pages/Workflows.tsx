import { GitBranch } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function Workflows() {
  return (
    <DocPage
      subtitle="Declaring steps as a graph, running them in dependency order, and drawing the result."
      icon={<GitBranch size={48} />}
    >
      <p>
        A <code>WorkflowDAG</code> is a set of named nodes and the edges between them. You declare
        what depends on what; the graph works out the order, runs independent nodes together, and
        reports every node's outcome separately. Where a{' '}
        <Link to="/multi-agent">team</Link> is a list with a pattern, a workflow is a shape.
      </p>

      <h2>Three lines and a run</h2>

      <CodeBlock filename="brief.py" code={`from effgen import Agent, AgentConfig, WorkflowDAG, WorkflowNode


def agent(name, prompt):
    return Agent(AgentConfig(name=name, model="gpt-5-nano", provider="openai",
                             system_prompt=prompt))


dag = WorkflowDAG("brief")
dag.add_node(WorkflowNode(id="draft", agent=agent(
    "draft", "Reply with exactly one sentence.")))
dag.add_node(WorkflowNode(id="shorten", agent=agent(
    "shorten", "Rewrite the text you are given in at most eight words. Reply with the rewrite alone.")))
dag.connect("draft", "shorten")

print("order:", dag.topological_order())
result = dag.run("Explain what a tide is.")
print("success:", result.success)
for node, output in result.outputs.items():
    print(f"  {node}: {output}")`} />

      <Terminal
        command="python brief.py"
        output={`order: ['draft', 'shorten']
success: True
  draft: A tide is the regular rise and fall of sea level caused primarily by the gravitational forces of the Moon and Sun acting on the Earth's oceans.
  shorten: Tides: regular rise and fall of sea level.`}
        caption={`Run against effGen 1.0.0. A single string is routed to the entry nodes; a node downstream is given its upstream's output.`}
      />

      <ApiTable
        headers={['You pass to run()', 'What happens']}
        rows={[
          [
            'A string',
            'It becomes the task for every entry node — the nodes with no incoming edge.',
          ],
          [
            <code>{'{node_id: task}'}</code>,
            'Each named node starts from that task. Useful when two entry nodes need different inputs.',
          ],
          [
            <code>None</code>,
            'Entry nodes run with no task. Fine when the node’s own agent already knows what to do.',
          ],
        ]}
      />

      <h2>Branching on an answer</h2>
      <p>
        An edge can carry a condition. It receives the <strong>source node's output</strong> and, when
        it returns false, the target is marked <code>skipped</code> with the reason rather than run.
      </p>

      <CodeBlock filename="triage.py" code={`from effgen import WorkflowDAG, WorkflowNode


class Say:
    """Stands in for an agent: returns a fixed answer for any task."""

    def __init__(self, answer):
        self.answer = answer

    def run(self, task, **kwargs):
        return self.answer


dag = WorkflowDAG("triage")
dag.add_node(WorkflowNode(id="classify", agent=Say("urgent")))
dag.add_node(WorkflowNode(id="page_oncall", agent=Say("paged the on-call engineer")))
dag.add_node(WorkflowNode(id="queue_ticket", agent=Say("queued for the morning")))
dag.connect("classify", "page_oncall", condition=lambda out: "urgent" in str(out).lower())
dag.connect("classify", "queue_ticket", condition=lambda out: "urgent" not in str(out).lower())

result = dag.run("The checkout page is returning 500s.")
for node in result.node_results:
    detail = result.outputs.get(node["id"]) or node["metadata"].get("skip_reason", "")
    print(f"{node['id']:13} {node['status']:9} {detail}")
print("success:", result.success)`} />

      <Terminal
        command="python triage.py"
        output={`classify      completed urgent
page_oncall   completed paged the on-call engineer
queue_ticket  skipped   condition on edge from 'classify' was not met
success: True`}
        caption="A node’s `agent` is anything with a `run(task)` method, which is what makes a graph testable without a model. A skipped branch does not make the run a failure — it was not supposed to happen."
      />

      <h2>When a node fails</h2>
      <p>
        A failed node is recorded as failed, and everything that depended on it is marked{' '}
        <code>skipped</code> — not run on the upstream's error text. That distinction is the point:
        without it, an internal failure gets rewritten by the next model into a fluent,
        customer-facing sentence that is not true.
      </p>

      <CodeBlock filename="failure.py" code={`from effgen import WorkflowDAG, WorkflowNode


class Boom:
    """Stands in for an agent whose run fails."""

    def run(self, task, **kwargs):
        raise RuntimeError("the upstream service returned 500")


dag = WorkflowDAG("pipeline")
dag.add_node(WorkflowNode(id="fetch", agent=Boom()))
dag.add_node(WorkflowNode(id="summarize"))
dag.connect("fetch", "summarize")

result = dag.run("Fetch and summarise today's numbers.")
print("success:", result.success)
for node in result.node_results:
    print(f"  {node['id']:10} {node['status']:9} "
          f"{node.get('error') or node['metadata'].get('skip_reason', '')}")`} />

      <Terminal
        command="python failure.py"
        output={`success: False
  fetch      failed    RuntimeError: the upstream service returned 500
  summarize  skipped   upstream 'fetch' did not complete (failed)`}
      />

      <ApiTable
        headers={['Status', 'Means']}
        rows={[
          [<code>pending</code>, 'Declared, not reached.'],
          [<code>running</code>, 'In flight.'],
          [<code>completed</code>, 'Ran and produced an output.'],
          [
            <code>skipped</code>,
            <>
              Not run, on purpose. <code>metadata["skip_reason"]</code> says whether an edge
              condition was not met or an upstream did not complete.
            </>,
          ],
          [
            <code>failed</code>,
            <>
              Raised. <code>error</code> carries the exception type and message.
            </>,
          ],
        ]}
        caption={<><code>NodeStatus</code>, from <code>effgen.core.workflow</code>.</>}
      />

      <h2>Two graphs that will not run</h2>
      <p>
        Both are caught before anything is called, so neither costs a model call.
      </p>

      <CodeBlock filename="empty.py" code={`from effgen import WorkflowDAG

result = WorkflowDAG("nothing").run("anything")
print("success:", result.success)
print("metadata:", result.metadata)`} />

      <Terminal
        command="python empty.py"
        output={`success: False
metadata: {'name': 'nothing', 'node_count': 0, 'reason': 'empty_workflow', 'error': 'Workflow has no nodes. Add nodes with add_node() before running.'}`}
        caption="An empty workflow is a configuration mistake, and it is reported as a failure rather than as a vacuous success."
      />

      <CodeBlock filename="cycle.py" code={`from effgen import WorkflowDAG, WorkflowNode

dag = WorkflowDAG("loop")
for node_id in ("a", "b"):
    dag.add_node(WorkflowNode(id=node_id))
dag.connect("a", "b")
try:
    dag.connect("b", "a")
except ValueError as exc:
    print(type(exc).__name__, "->", exc)`} />

      <Terminal
        command="python cycle.py"
        output={`ValueError -> Cycle detected in workflow DAG — cannot topologically sort`}
        caption={
          <>
            <code>connect</code> validates as it goes, so the edge that closes the loop is the one
            that raises — you learn where the cycle is while you are writing it.
          </>
        }
      />

      <h2>From a YAML file</h2>
      <p>
        A workflow that is configuration rather than code goes in a file, and the command line runs
        it. <code>depends_on</code> is the edge list, and each node names a{' '}
        <Link to="/presets">preset</Link> or a model.
      </p>

      <CodeBlock
        language="yaml"
        filename="workflow.yaml"
        code={`nodes:
  - id: draft
    preset: general
    task: "Explain what a tide is. Reply with exactly one sentence."
  - id: shorten
    preset: general
    task: "Rewrite the text above in at most eight words."
    depends_on: [draft]`}
      />

      <CodeBlock
        language="bash"
        code={`effgen workflow validate workflow.yaml --diagram`}
      />

      <Terminal
        command="effgen workflow validate workflow.yaml --diagram"
        output={`Workflow 'workflow' is valid.
  Nodes: 2
  Edges: 1
  Execution order: draft -> shorten

Workflow: workflow
2 node(s), 1 edge(s), 2 level(s)
Level 0
  ○ draft   pending
      └─▶ shorten
Level 1
  ○ shorten   pending`}
        caption="Validation reads the file and builds the graph without running a node, so a typo in depends_on costs nothing to find."
      />

      <Terminal
        command="effgen workflow run workflow.yaml -m gpt-5-nano --diagram"
        output={`Running workflow 'workflow' (2 nodes)...

Workflow succeeded in 10.72s

Workflow: workflow
2 node(s), 1 edge(s), 2 level(s)
Level 0
  ● draft   completed  4.82s  $0.0005  6,546 tok
      └─▶ shorten
Level 1
  ● shorten   completed  5.90s  $0.0004  6,883 tok

Outputs:
  draft: A tide is the regular rise and fall of sea level caused mainly by the
gravitational forces of the Moon and Sun on the Earth's oceans, producing
alternating high and low tides about twice each day.
  shorten: Tides: regular sea level changes from celestial gravity.`}
        caption="The same diagram after a run carries each node's status, duration, cost and tokens."
      />

      <ParamTable
        nameLabel="Flag"
        params={[
          {
            name: 'file',
            type: 'path',
            required: true,
            description: 'The workflow YAML file. Positional.',
          },
          {
            name: '-m, --model',
            type: 'str',
            description: 'Default model for all agents.',
          },
          {
            name: '-i, --input NODE TASK',
            type: 'str str',
            description: 'Input for a specific node. Repeatable.',
          },
          {
            name: '--task',
            type: 'str',
            description:
              'A single task string routed to the workflow entry node(s) — the alternative to --input.',
          },
          {
            name: '--json',
            type: 'flag',
            description: 'Emit the workflow result as JSON to stdout, for CI gating.',
          },
          {
            name: '--diagram',
            type: 'flag',
            description:
              'Draw the workflow as a dependency graph: nodes by level, edges, per-node status, duration and cost.',
          },
          {
            name: '-q, --quiet',
            type: 'flag',
            description: 'Quiet output (errors only); --json still emits to stdout.',
          },
        ]}
        caption={
          <>
            <code>effgen workflow run --help</code>. <code>effgen workflow validate</code> takes the
            file plus <code>--json</code>, <code>--diagram</code> and <code>-q</code>.
          </>
        }
      />

      <h2>The API</h2>

      <ParamTable
        nameLabel="Method"
        params={[
          {
            name: 'add_node(node)',
            type: 'None',
            description: (
              <>
                Adds a <code>WorkflowNode</code>. Ids are unique.
              </>
            ),
          },
          {
            name: 'connect(source, target, key=None, condition=None)',
            type: 'WorkflowEdge',
            description:
              'Adds an edge and validates against cycles. condition receives the source’s output.',
          },
          {
            name: 'add_edge(edge)',
            type: 'None',
            description: (
              <>
                The same, given a <code>WorkflowEdge</code> you built.
              </>
            ),
          },
          {
            name: 'run(initial_inputs=None, context=None, *, checkpoint=None, run_id=None)',
            type: 'WorkflowResult',
            description: (
              <>
                Runs the graph. <code>execute()</code> is the same call under its other name.
              </>
            ),
          },
          {
            name: 'run_async(…)',
            type: 'WorkflowResult',
            description: (
              <>
                Awaited, same arguments and result. <code>execute_async()</code> is its alias.
              </>
            ),
          },
          {
            name: 'topological_order()',
            type: 'list[str]',
            description: 'The order the nodes will run in. Raises on a cycle.',
          },
          { name: 'entry_nodes()', type: 'list[str]', description: 'Nodes with no incoming edge.' },
          { name: 'get_node(node_id)', type: 'WorkflowNode | None', description: 'Look one up.' },
          {
            name: 'to_dict()',
            type: 'dict',
            description: 'The graph as data — nodes, edges, levels. Nothing runs.',
          },
        ]}
        caption={
          <>
            <code>WorkflowDAG(name="workflow")</code>, from <code>effgen</code>.{' '}
            <code>checkpoint</code> and <code>run_id</code> are covered in{' '}
            <Link to="/checkpointing">Checkpointing</Link>.
          </>
        }
      />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'id', type: 'str', required: true, description: 'Unique within the graph.' },
          {
            name: 'agent',
            type: 'Any',
            default: 'None',
            description:
              'Anything with run(task). A node with no agent cannot run and is reported failed.',
          },
          {
            name: 'tools',
            type: 'list[str]',
            default: '[]',
            description: 'Tool names for a node built from configuration rather than an instance.',
          },
          {
            name: 'input_keys',
            type: 'list[str]',
            default: '[]',
            description: 'Which upstream outputs this node reads.',
          },
          {
            name: 'output_key',
            type: 'str',
            default: "''",
            description: 'The key its output is stored under. Defaults to the node id.',
          },
          {
            name: 'metadata',
            type: 'dict',
            default: '{}',
            description: 'Free space. skip_reason is written here by the runner.',
          },
          {
            name: 'status',
            type: 'NodeStatus',
            default: 'PENDING',
            description: 'Updated as the run proceeds.',
          },
          { name: 'output', type: 'Any', default: 'None', description: 'What the node produced.' },
          {
            name: 'error',
            type: 'str | None',
            default: 'None',
            description: 'The exception type and message, when it failed.',
          },
          {
            name: 'execution_time',
            type: 'float',
            default: '0.0',
            description: 'Seconds this node took.',
          },
        ]}
        caption={<><code>WorkflowNode</code>, from <code>effgen</code>.</>}
      />

      <ApiTable
        headers={['WorkflowResult field', 'Carries']}
        rows={[
          [<code>success</code>, 'False if any node failed, or if the graph was empty.'],
          [<code>outputs</code>, <>{'{node_id: output}'} for every node that completed.</>],
          [
            <code>node_results</code>,
            <>
              One dict per node: <code>id</code>, <code>status</code>, <code>error</code>,{' '}
              <code>execution_time</code>, <code>input_keys</code>, <code>output_key</code>,{' '}
              <code>tools</code> and <code>metadata</code>.
            </>,
          ],
          [<code>execution_time</code>, 'Wall clock for the whole graph.'],
          [
            <code>metadata</code>,
            <>
              <code>name</code>, <code>node_count</code>, <code>levels</code>,{' '}
              <code>topological_order</code>, <code>edges</code>, <code>execution_id</code>,{' '}
              <code>cost_usd</code>, <code>tokens_used</code>, and <code>reason</code> on a failure.
            </>,
          ],
        ]}
        caption={
          <>
            The node output itself is in <code>outputs</code>, not in the{' '}
            <code>node_results</code> entry — reading <code>node["output"]</code> off that dict finds
            nothing.
          </>
        }
      />

      <Callout type="note" title="Independent nodes run together">
        <p>
          The runner works in topological levels: everything at one level runs before anything at the
          next, and nodes within a level run in parallel. <code>metadata["levels"]</code> is that
          grouping, and it is what <code>--diagram</code> draws.
        </p>
      </Callout>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>ValueError: Cycle detected in workflow DAG</code>,
            'An edge closes a loop.',
            <>
              The <code>connect</code> call that raised is the one to remove. A DAG cannot express a
              retry loop; put the loop inside a node.
            </>,
          ],
          [
            <>
              <code>success=False</code> with <code>reason: empty_workflow</code>
            </>,
            'No nodes were added.',
            <>
              <code>add_node</code> before <code>run</code>. Building the graph in a loop that never
              executed is the usual cause.
            </>,
          ],
          [
            'A node is failed with no exception of your own',
            <>
              It has no <code>agent</code>, so there is nothing to call.
            </>,
            <>
              Give it one — anything with <code>run(task)</code> will do.
            </>,
          ],
          [
            'Everything downstream is skipped',
            'Working as intended: a node whose required upstream failed or was skipped is not run on the upstream’s error text.',
            <>
              Fix the failed node. <Link to="/checkpointing">Resuming</Link> retries it without
              re-running what already succeeded.
            </>,
          ],
          [
            'Both branches of a condition ran',
            'Both conditions returned true. The condition sees the source’s output, not the task.',
            'Make them mutually exclusive, or accept that a fan-out is what you asked for.',
          ],
          [
            <><code>KeyError</code> reading a node’s output</>,
            <>
              A skipped or failed node has no entry in <code>outputs</code>.
            </>,
            <>
              <code>result.outputs.get(node_id)</code>, and check <code>node_results</code> for why.
            </>,
          ],
          [
            'The YAML runs but every node uses the wrong model',
            <>
              A node with only a <code>preset</code> takes the preset’s model unless{' '}
              <code>-m</code> overrides it.
            </>,
            <>
              <code>effgen workflow run file.yaml -m gpt-5-nano</code> sets the default for every
              agent.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/checkpointing', '/multi-agent', '/cli/batch']} />
    </DocPage>
  );
}
