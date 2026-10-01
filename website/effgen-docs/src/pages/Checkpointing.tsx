import { Save } from 'lucide-react';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function Checkpointing() {
  return (
    <DocPage
      subtitle="Saving a workflow as it goes, and picking it up from the last step that finished."
      icon={<Save size={48} />}
    >
      <p>
        Long work fails part of the way through. Checkpointing writes down what has already been done
        so the next attempt starts from there rather than from the top. There are two of them,
        because there are two kinds of long work: one agent's iterations, and a workflow's nodes.
      </p>

      <ApiTable
        headers={['What is saved', 'Written by', 'Resumed by']}
        rows={[
          [
            'One agent’s run — its steps, iteration, memory, the model id',
            <>
              <code>run(checkpoint_dir=…, checkpoint_interval=…)</code> or{' '}
              <code>effgen run --checkpoint-dir</code>
            </>,
            <>
              <code>agent.resume(...)</code> or <code>effgen resume --checkpoint</code>
            </>,
          ],
          [
            'A workflow’s node states and outputs',
            <>
              <code>dag.run(checkpoint=store, run_id=…)</code>
            </>,
            'Running the same line again with the same run_id. There is no separate resume call.',
          ],
        ]}
      />

      <h2>Resuming a workflow</h2>
      <p>
        Pass a store and a run id. A run id the store has never seen starts from the beginning; one
        it knows continues from where it stopped.
      </p>

      <CodeBlock filename="resume.py" code={`from effgen import InMemoryCheckpointStore, WorkflowDAG, WorkflowNode


class Step:
    """Stands in for an agent. \`fail\` makes the first attempt fail."""

    def __init__(self, answer, fail=False):
        self.answer, self.fail, self.calls = answer, fail, 0

    def run(self, task, **kwargs):
        self.calls += 1
        if self.fail:
            self.fail = False
            raise RuntimeError("the upstream service returned 500")
        return self.answer


research = Step("three papers found")
draft = Step("a paragraph about them", fail=True)
store = InMemoryCheckpointStore()


def build():
    dag = WorkflowDAG("report")
    dag.add_node(WorkflowNode(id="research", agent=research))
    dag.add_node(WorkflowNode(id="draft", agent=draft))
    dag.connect("research", "draft")
    return dag


first = build().run("Write the Q3 summary.", checkpoint=store, run_id="q3")
print("first run :", first.success, {n["id"]: n["status"] for n in first.node_results})

second = build().run("Write the Q3 summary.", checkpoint=store, run_id="q3")
print("second run:", second.success, {n["id"]: n["status"] for n in second.node_results})
print("research ran", research.calls, "time(s); draft ran", draft.calls, "time(s)")`} />

      <Terminal
        command="python resume.py"
        output={`first run : False {'research': 'completed', 'draft': 'failed'}
second run: True {'research': 'completed', 'draft': 'completed'}
research ran 1 time(s); draft ran 2 time(s)`}
        caption={`Run against effGen 1.0.0. The second attempt did not call \`research\` again — its output was restored from the checkpoint and flowed downstream. Only the node that failed was retried.`}
      />

      <ApiTable
        headers={['Node state when the run stopped', 'On resume']}
        rows={[
          [<code>completed</code>, 'Not run again. Its output is restored and flows downstream.'],
          [<code>skipped</code>, 'Stays skipped, with the reason it was skipped for.'],
          [<code>failed</code>, <strong>Retried — that is usually why you are resuming.</strong>],
          [<code>pending</code>, 'Run normally.'],
        ]}
      />

      <Callout type="note" title="When progress is written">
        <p>
          After each topological level — the point at which every node in that level has finished and
          the next has not started. Writes are atomic, so a crash during a save leaves the previous
          checkpoint intact rather than a half-written file.
        </p>
      </Callout>

      <h2>Re-running a finished run is cheap</h2>
      <p>
        A run id whose every node completed replays its stored outputs without calling a model, which
        makes a workflow idempotent under a job runner that retries. To genuinely start over, delete
        the run id.
      </p>

      <CodeBlock filename="replay.py" code={`from effgen import InMemoryCheckpointStore, WorkflowDAG, WorkflowNode


class Counted:
    def __init__(self):
        self.calls = 0

    def run(self, task, **kwargs):
        self.calls += 1
        return "done"


step = Counted()
store = InMemoryCheckpointStore()


def build():
    dag = WorkflowDAG("nightly")
    dag.add_node(WorkflowNode(id="only", agent=step))
    return dag


build().run("go", checkpoint=store, run_id="nightly-2026-08-23")
build().run("go", checkpoint=store, run_id="nightly-2026-08-23")
print("model calls after two runs:", step.calls)

store.delete("nightly-2026-08-23")
build().run("go", checkpoint=store, run_id="nightly-2026-08-23")
print("after store.delete():", step.calls)`} />

      <Terminal
        command="python replay.py"
        output={`model calls after two runs: 1
after store.delete(): 2`}
        caption="Two runs, one model call. Pick a run id that names the work — the date, the ticket, the batch — rather than a fresh uuid each time, or nothing is ever resumed."
      />

      <h2>Crossing a process boundary</h2>
      <p>
        <code>FileCheckpointStore</code> writes to disk, so a pipeline that died with the process
        picks up in the next one. The store holds run <em>state</em>, never the graph — agents own
        sockets, model handles and credentials, none of which survive a process. Rebuild the same{' '}
        <code>WorkflowDAG</code> in the new process and hand it the same <code>run_id</code>.
      </p>

      <CodeBlock filename="file_store.py" code={`import tempfile

from effgen import FileCheckpointStore, WorkflowDAG, WorkflowNode


class Say:
    def run(self, task, **kwargs):
        return "the summary"


directory = tempfile.mkdtemp(prefix="effgen-workflows-")
store = FileCheckpointStore(directory)

dag = WorkflowDAG("report")
dag.add_node(WorkflowNode(id="summarize", agent=Say()))
dag.run("Summarise the quarter.", checkpoint=store, run_id="q3-summary")

print("runs on disk:", store.list_runs())
saved = store.load("q3-summary")
print("workflow    :", saved.workflow)
print("node ids    :", saved.node_ids)
print("completed   :", list(saved.completed))
print("outputs     :", saved.outputs)`} />

      <Terminal
        command="python file_store.py"
        output={`runs on disk: ['q3-summary']
workflow    : report
node ids    : ['summarize']
completed   : ['summarize']
outputs     : {'summarize': 'the summary'}`}
        caption={
          <>
            <code>FileCheckpointStore()</code> with no argument writes to{' '}
            <code>~/.effgen/workflows</code>. <code>EFFGEN_WORKFLOW_DIR</code> moves it, which is what
            containers and test suites set so a run cannot write into a developer's home directory.
          </>
        }
      />

      <h2>The graph has to match</h2>
      <p>
        Resuming into a graph with different node ids raises rather than mixing outputs from two
        different workflows. The message names what was added and what was removed.
      </p>

      <CodeBlock filename="mismatch.py" code={`from effgen import InMemoryCheckpointStore, WorkflowDAG, WorkflowNode


class Say:
    def run(self, task, **kwargs):
        return "ok"


store = InMemoryCheckpointStore()

first = WorkflowDAG("report")
first.add_node(WorkflowNode(id="research", agent=Say()))
first.run("go", checkpoint=store, run_id="r1")

changed = WorkflowDAG("report")
changed.add_node(WorkflowNode(id="research", agent=Say()))
changed.add_node(WorkflowNode(id="review", agent=Say()))
try:
    changed.run("go", checkpoint=store, run_id="r1")
except ValueError as exc:
    print(type(exc).__name__, "->", exc)`} />

      <Terminal
        command="python mismatch.py"
        output={`ValueError -> Checkpoint 'r1' was saved for a different graph (added: ['review'], removed: none). Resume with the same nodes, or delete the checkpoint to start over.`}
        caption="Adding a step to a pipeline that has checkpoints in flight means starting those runs over. The alternative is a report half-written by an older version of the graph."
      />

      <h2>The three stores</h2>

      <ApiTable
        headers={['Store', 'Lives', 'Use when']}
        rows={[
          [
            <code>InMemoryCheckpointStore()</code>,
            'In the process.',
            'The thing you need to survive is a failed node, not a failed process — a retry inside one long-lived worker.',
          ],
          [
            <code>FileCheckpointStore(directory=None)</code>,
            <>
              On disk. <code>~/.effgen/workflows</code> by default.
            </>,
            'The process can die. This is the usual choice.',
          ],
          [
            'Your own',
            'Wherever you put it.',
            <>
              A database or an object store. Implement three methods —{' '}
              <code>save(checkpoint)</code>, <code>load(run_id)</code>,{' '}
              <code>delete(run_id)</code> — and pass it as <code>checkpoint=</code>.
            </>,
          ],
        ]}
        caption={
          <>
            Both shipped stores also have <code>list_runs()</code>.{' '}
            <code>CheckpointStore</code> is the protocol they satisfy.
          </>
        }
      />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'run_id', type: 'str', required: true, description: 'The id it was saved under.' },
          { name: 'workflow', type: 'str', default: "''", description: 'The graph’s name.' },
          {
            name: 'node_ids',
            type: 'list[str]',
            default: '[]',
            description: 'Every node in the graph, which is what the mismatch check compares.',
          },
          {
            name: 'completed',
            type: 'dict',
            default: '{}',
            description: 'Nodes that finished, keyed by id.',
          },
          {
            name: 'skipped',
            type: 'dict[str, str]',
            default: '{}',
            description: 'Nodes that were skipped, and why.',
          },
          {
            name: 'failed',
            type: 'dict[str, str]',
            default: '{}',
            description: 'Nodes that failed, and their error. These are the ones retried.',
          },
          {
            name: 'outputs',
            type: 'dict',
            default: '{}',
            description: 'What each completed node produced. Replayed on resume.',
          },
          {
            name: 'updated_at',
            type: 'float',
            default: '0.0',
            description: 'Unix time of the last save.',
          },
          { name: 'metadata', type: 'dict', default: '{}', description: 'Free space.' },
        ]}
        caption={
          <>
            <code>WorkflowCheckpoint</code>, from <code>effgen.core.workflow</code>. JSON only — no
            pickle.
          </>
        }
      />

      <h2>Checkpointing one agent's run</h2>
      <p>
        The other kind. A single agent working through iterations writes a snapshot every{' '}
        <code>checkpoint_interval</code> iterations, and <code>effgen resume</code> picks up the
        latest one in the directory.
      </p>

      <CodeBlock
        language="bash"
        filename="checkpointed_run.sh"
        code={`effgen run --checkpoint-dir ./cli-checkpoints --checkpoint-interval 1 \\
  -m gpt-5-nano --provider openai -t calculator --no-animation -q \\
  "Work out 81234 * 9317 with the calculator, then explain the result in one sentence."
echo "--- files written ---"
ls ./cli-checkpoints
echo "--- resume ---"
effgen --no-animation resume --checkpoint ./cli-checkpoints -m gpt-5-nano`}
      />

      <Terminal
        command="bash checkpointed_run.sh"
        output={`Response
╭─────────────────────────────── Agent Response ───────────────────────────────╮
│ 756857178                                                                    │
│                                                                              │
│ One-sentence explanation: 81234 multiplied by 9317 equals 756857178.         │
╰──────────────────────────────────────────────────────────────────────────────╯
--- files written ---
cli-agent-1787530213769-097932ef.json
cli-agent-1787530220399-4853e15e.json
latest.json
--- resume ---
Resuming 'Work out 81234 * 9317 with the calculator, then explain the result in
one senten' from iteration 2
756857178

One-sentence explanation: 81234 multiplied by 9317 equals 756857178.`}
        caption={
          <>
            One file per checkpointed iteration, plus <code>latest.json</code>. Resume without a
            specific file uses the newest. <code>--no-animation</code> and <code>-q</code> are global
            flags, so they go before the sub-command on <code>resume</code>.
          </>
        }
      />

      <ParamTable
        nameLabel="Flag"
        params={[
          {
            name: '--checkpoint-dir',
            type: 'path',
            description: 'Directory to write agent checkpoints. On effgen run.',
          },
          {
            name: '--checkpoint-interval',
            type: 'int',
            description: 'Checkpoint every N iterations. Requires --checkpoint-dir.',
          },
          {
            name: '--checkpoint',
            type: 'str',
            required: true,
            description: 'On effgen resume: a checkpoint id, a JSON path, or a directory (uses the latest).',
          },
          {
            name: '-m, --model',
            type: 'str',
            description:
              'Model to resume under. Without it, the model the checkpoint was created with is reused.',
          },
          {
            name: '--preset',
            type: 'str',
            description:
              'Preset to resume the run under: coding, general, math, media, minimal, multimodal, notify, rag, research.',
          },
        ]}
        caption={
          <>
            <code>effgen run --help</code> and <code>effgen resume --help</code>.
          </>
        }
      />

      <p>
        In Python the pair is <code>run(..., checkpoint_dir=, checkpoint_interval=)</code> and{' '}
        <code>agent.resume(checkpoint_id=None, checkpoint_dir="./checkpoints")</code>. Build the
        resuming agent with the same tools it had.
      </p>

      <Callout type="note" title="Since 1.1.0, a resumed run continues its conversation">
        <p>
          A checkpoint stores the run’s steps in its <code>thread</code> field, and still writes the
          flat transcript beside it under <code>scratchpad</code>, so a file 1.1.0 writes also
          resumes on a 1.0.x build. Resuming picks up the conversation the run had instead of
          restarting the task, and does not redo the turns the checkpoint holds.
        </p>
        <p>
          A checkpoint written by 1.0.x has no steps. <code>Checkpoint.to_thread()</code> rebuilds
          them from the transcript and logs{' '}
          <code>[compat] rebuilt a thread from a flat transcript</code>. The rebuild cannot recover
          the provider’s call id (a recovered call is named from its position), a tool’s arguments as
          values (they come back as the string they were printed as), whether a line was the
          framework’s own, or the run’s frame — the persona, the tool contract, earlier turns and the
          task.
        </p>
      </Callout>

      <CodeBlock filename="agent_checkpoint.py" code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
agent.run("Long task…", checkpoint_dir="./checkpoints", checkpoint_interval=1)

# Later, on a fresh agent built with the same tools:
resumed = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
result = resumed.resume(checkpoint_dir="./checkpoints")`} />

      <h2>A checkpoint that will not load</h2>
      <p>
        A truncated or malformed file raises <code>CorruptStateError</code>, which names the file
        rather than handing you a <code>JSONDecodeError</code> from somewhere inside a parser. A
        missing one raises <code>FileNotFoundError</code>.
      </p>

      <CodeBlock filename="corrupt.py" code={`from pathlib import Path

from effgen.core.checkpoint import CheckpointManager
from effgen.errors import CorruptStateError

Path("/tmp/checkpoints").mkdir(exist_ok=True)
Path("/tmp/checkpoints/run-7.json").write_text("{ this is not json")

try:
    CheckpointManager("/tmp/checkpoints").load("run-7")
except CorruptStateError as exc:
    print(type(exc).__name__)
    print(exc)`} />

      <Terminal
        command="python corrupt.py"
        output={`CorruptStateError
Cannot read checkpoint file '/tmp/checkpoints/run-7.json' - it is corrupt, truncated, or not valid JSON.
Detail: Expecting property name enclosed in double quotes: line 1 column 3 (char 2)

Fix: inspect the file, restore a backup, or delete it to start fresh.`}
        caption={
          <>
            <code>CheckpointManager(dir, backend="sqlite")</code> is the other storage option for
            agent checkpoints.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Nothing is ever resumed — every run starts from the top',
            'A fresh run_id each time. The store has never seen it before, so there is nothing to continue.',
            'Derive the run id from the work: the date, the ticket, the input file.',
          ],
          [
            <code>ValueError: Checkpoint '…' was saved for a different graph</code>,
            'Node ids were added or removed since the checkpoint was written.',
            <>
              Resume with the same nodes, or <code>store.delete(run_id)</code> and start over. The
              message names both differences.
            </>,
          ],
          [
            'A run finishes instantly and costs nothing',
            'Every node was already complete, so the stored outputs were replayed.',
            <>
              That is the idempotency guarantee. <code>store.delete(run_id)</code> forces real work.
            </>,
          ],
          [
            <code>CorruptStateError</code>,
            'The file is truncated or not JSON.',
            <>
              The path is in the message. Delete it — an unreadable checkpoint costs one re-run, and
              a partial one costs a wrong answer.
            </>,
          ],
          [
            <><code>FileNotFoundError</code> on resume</>,
            'No checkpoint at that id or path.',
            <>
              <code>store.list_runs()</code>, or check the directory. A run that failed before its
              first level completed has nothing saved.
            </>,
          ],
          [
            'Resume used a different model than the run did',
            <>
              <code>-m/--model</code> overrode the model the checkpoint was created with.
            </>,
            'Leave it off to reuse the saved model. A mismatch is warned about, not silently accepted.',
          ],
          [
            <code>Checkpoint: saved data held the wrong type for …</code>,
            'A field in the saved record did not match its declared type and was replaced so the rest of the record still loads.',
            'A warning, not a failure — the run resumed. It usually means the checkpoint was written by a different version.',
          ],
          [
            'Checkpoints filling a disk',
            'Nothing prunes them for you.',
            <>
              <code>store.delete(run_id)</code> when a run is genuinely finished with, and put the
              directory somewhere with a retention policy via{' '}
              <code>EFFGEN_WORKFLOW_DIR</code>.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/workflows', '/sessions', '/errors']} />
    </DocPage>
  );
}
