import { Bug } from 'lucide-react';
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

export default function Debug() {
  return (
    <DocPage
      subtitle="Watching a run from the inside — the prompt, the tool calls, the timings and the decision at each step."
      icon={<Bug size={48} />}
    >
      <p>
        When a run gives the wrong answer, the question is nearly always <em>which step went
        wrong</em>. Three surfaces answer it at different depths: <code>--explain</code> and{' '}
        <code>--trace</code> on any run, <code>effgen debug</code> for a stepped inspection, and{' '}
        <code>DebugAgent</code> when you want the trace as data.
      </p>

      <h2>Start here: the trace on a normal run</h2>

      <p>
        Nothing needs to change about the command. <code>--explain</code> prints the tool the agent
        chose at each iteration with its arguments, its result and its duration.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "What is 7*6? Use the calculator." -m openai:gpt-5-nano -t calculator --explain`}
      />

      <Terminal
        command={`effgen run "What is 7*6? Use the calculator." -m openai:gpt-5-nano -t calculator --explain -q`}
        output={`
Response
╭───────────────────────────────────────── Agent Response ─────────────────────────────────────────╮
│ 42                                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

Execution Trace
💭 Iteration 1: Reasoning...
🔧 calculator(expression="7*6", operation="calculate")  ⏱ 1.9s
   ✓ 42

Execution Statistics
                   Execution Statistics
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric         ┃ Value                                 ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Mode           │ single                                │
│ Success        │ Yes                                   │
│ Iterations     │ 1                                     │
│ Tool Calls     │ ToolCallList(['calculator'], total=1) │
│ Tokens Used    │ 312                                   │
│ Execution Time │ 1.86s                                 │
└────────────────┴───────────────────────────────────────┘`}
        maxLines={24}
        caption={
          <>
            <code>--trace</code> adds a timeline with per-step durations above the same statistics
            block. Both flags work on every model and every tool-calling path, which is what makes
            them the first thing to reach for.
          </>
        }
      />

      <ApiTable
        headers={['Reach for', 'When']}
        rows={[
          [
            <code>--explain</code>,
            'You want to know which tool was chosen, with what arguments, and what came back.',
          ],
          [
            <code>--trace</code>,
            'You want the same steps ordered on a timeline with their durations — which step is slow.',
          ],
          [
            <code>effgen debug</code>,
            <>
              You want to stop between iterations and look at the run’s steps, or you want the run
              summary framed on its own.
            </>,
          ],
          [
            <code>DebugAgent</code>,
            'You want the trace as an object, in a test or a script.',
          ],
          [
            <code>-v/--verbose</code>,
            <>
              You want the framework's own DEBUG and INFO logs — the adapter, the retries, the
              router. <Link to="/observability">Observability</Link> covers the log surface.
            </>,
          ],
          [
            <code>--card out.html</code>,
            <>
              You want to send the whole trace to someone else.{' '}
              <Link to="/cli/reports">Reports and run cards</Link>.
            </>,
          ],
        ]}
      />

      <h2>
        <code>effgen debug</code>
      </h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen debug "What is 137 * 19? Use the calculator." -m openai:gpt-5-nano
effgen debug "Plan a 3-step research task" --preset research --step`}
      />

      <Terminal
        command={`effgen debug "What is 137 * 19? Use the calculator." -m openai:gpt-5-nano`}
        output={`╭──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ effGen Debug Mode                                                                                │
│ Task: What is 137 * 19? Use the calculator.                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
          Run Summary
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric        ┃ Value       ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Agent         │ debug_agent │
├───────────────┼─────────────┤
│ Iterations    │ 0           │
├───────────────┼─────────────┤
│ Total Tokens  │ 222         │
├───────────────┼─────────────┤
│ Total Latency │ 2.262s      │
├───────────────┼─────────────┤
│ Success       │ True        │
├───────────────┼─────────────┤
│ Output        │ 2603        │
└───────────────┴─────────────┘`}
        maxLines={22}
        caption={
          <>
            Captured on a real pseudo-terminal. <code>Iterations: 0</code> is not a bug in the run —
            see <em>What the per-iteration trace records</em> below.
          </>
        }
      />

      <ParamTable
        nameLabel="Argument"
        params={[
          { name: 'task', required: true, description: 'Task to execute' },
          { name: '-m MODEL, --model MODEL', description: 'Model to use' },
          {
            name: '--provider PROVIDER',
            description: (
              <>
                Provider for a bare model id (e.g. <code>groq</code>). Equivalent to the{' '}
                <code>provider:model</code> prefix.
              </>
            ),
          },
          {
            name: '--preset {coding,general,math,media,minimal,multimodal,notify,rag,research}',
            description: 'Use a preset agent configuration',
          },
          { name: '--step', type: 'flag', description: 'Step through each iteration' },
        ]}
        caption={
          <>
            Every argument <code>effgen debug --help</code> declares. One of <code>-m/--model</code>{' '}
            or <code>--preset</code> is required.
          </>
        }
      />

      <Terminal
        command={`effgen debug "hi"`}
        output={`Could not create agent config. Provide -m/--model (e.g. \`-m gpt-5-nano\`) or --preset. Run \`effgen
models list\` for options or \`effgen doctor\` to check usable providers.`}
        caption="Exit 2 — a usage error, distinct from exit 1, which means the run itself failed."
      />

      <ApiTable
        headers={['Exit code', 'Means']}
        rows={[
          [<code>0</code>, 'The run succeeded.'],
          [<code>1</code>, 'The run failed.'],
          [<code>2</code>, 'A configuration or usage error — no model, an unknown preset.'],
        ]}
      />

      <Callout type="tip" title="It needs a terminal to draw on">
        <p>
          The panels and tables are drawn with the same live renderer the rest of the command line
          uses. Redirected to a file the frames arrive without their content, so capture it under a
          pseudo-terminal (<code>script -qc "effgen debug …" out.txt</code>) — or use{' '}
          <code>effgen run --explain --json</code>, which was designed for a pipe.
        </p>
      </Callout>

      <h2>The trace as data</h2>

      <p>
        <code>DebugAgent</code> wraps an agent and runs it with <code>debug=True</code>, attaching a{' '}
        <code>DebugTrace</code> to the response's metadata. It is the form to use in a test.
      </p>

      <CodeBlock
        filename="debug_trace.py"
        code={`from effgen import AgentConfig
from effgen.tools import get_registry
from effgen.debug import DebugAgent

calculator = get_registry().get_tool_sync("calculator")
config = AgentConfig(model="openai:gpt-5-nano", tools=[calculator])

with DebugAgent(config) as agent:
    result = agent.run("What is 137 * 19? Use the calculator.")

trace = result.metadata["debug_trace"]
print("answer:  ", result.output)
print("iterations:", len(trace.iterations))
for it in trace.iterations:
    print(f"  [{it.iteration}] action={it.action} input={it.action_input}")
    print(f"       observation={str(it.observation)[:60]}")
    print(f"       tokens={it.tokens_used} latency={it.latency:.2f}s")
print(trace.summary())`}
      />

      <Terminal
        command="python debug_trace.py"
        output={`answer:   2603
iterations: 0
DebugTrace(openai:gpt-5-nano, 0 iters, 275 tokens, 2.32s, success=True)`}
      />

      <Callout type="warning" title="AgentConfig(tools=…) takes Tool instances, not names">
        <p>
          Passing <code>tools=["calculator"]</code> raises{' '}
          <code>TypeError: AgentConfig(tools=…) expects Tool instances, not names — got a str
          'calculator'</code>, and the message names the fix:{' '}
          <code>get_registry().get_tool_sync("calculator")</code>. The <em>command line</em> takes
          names, because it does that lookup for you.
        </p>
      </Callout>

      <h3>What the objects carry</h3>

      <ApiTable
        headers={['DebugTrace field', 'Carries']}
        rows={[
          [<code>task</code>, 'The task string the run was given.'],
          [<code>agent_name</code>, 'The agent that ran it.'],
          [<code>run_id</code>, <>The run id, the same one <Link to="/cli/history">history</Link> stores.</>],
          [
            <code>iterations</code>,
            <>
              A list of <code>DebugIteration</code>. See below for when it is populated.
            </>,
          ],
          [<code>total_tokens</code>, "The run's token total."],
          [<code>total_latency</code>, 'Wall-clock seconds.'],
          [<code>final_answer</code>, <>The answer, or <code>None</code> on a failure.</>],
          [<code>success</code>, 'Whether the run reported success.'],
          [<code>metadata</code>, 'Anything else the run attached.'],
          [
            <code>summary()</code>,
            <>
              One line: <code>DebugTrace(model, N iters, T tokens, Ss, success=…)</code>.
            </>,
          ],
          [<code>to_dict()</code>, 'The whole trace as plain data.'],
          [<code>print_rich()</code>, 'The framed rendering `effgen debug` prints.'],
        ]}
      />

      <ApiTable
        headers={['DebugIteration field', 'Carries']}
        rows={[
          [<code>iteration</code>, 'Its number in the loop.'],
          [
            <>
              <code>raw_prompt</code>, <code>raw_response</code>
            </>,
            'What was sent and what came back, each capped at 2,000 characters.',
          ],
          [<code>thought</code>, 'The reasoning the model wrote before choosing.'],
          [
            <>
              <code>action</code>, <code>action_input</code>
            </>,
            'The tool it chose and the arguments it passed.',
          ],
          [<code>observation</code>, 'What the tool returned.'],
          [<code>final_answer</code>, 'Set on the iteration that produced the answer.'],
          [
            <>
              <code>tokens_used</code>, <code>latency</code>
            </>,
            "That iteration's own cost in tokens and seconds.",
          ],
          [
            <>
              <code>scratchpad_snapshot</code>, <code>memory_snapshot</code>
            </>,
            'The working context as it stood at that step.',
          ],
          [
            <code>thread_snapshot</code>,
            <>
              The run’s steps as they stood at that iteration, new in 1.1.0.{' '}
              <code>DebugIteration.to_dict()</code> writes it under a <code>thread</code> key, and the
              inspector panel renders every iteration from it, not only behind <code>--step</code>.
            </>,
          ],
        ]}
      />

      <h3>What the per-iteration trace records</h3>

      <p>
        <code>iterations</code> is filled in on the <strong>text ReAct path</strong> — the loop that
        reads the model's tool calls out of its own text, keeping a scratchpad between steps. On the{' '}
        <strong>native and hybrid tool-calling paths</strong>, where the tool definitions go to the
        provider's own tool-calling API, the loop returns as soon as the provider hands back a final
        answer, and the trace carries the totals — tokens, latency, the answer, success — with an
        empty <code>iterations</code> list. That is what the <code>Iterations: 0</code> above is.
      </p>

      <p>
        Every current cloud model takes the native path, so in practice{' '}
        <code>trace.iterations</code> is empty for them. For step-by-step detail on those models,
        use <code>--explain</code> or <code>--trace</code>, which read the execution trace the run
        records regardless of path, or the{' '}
        <Link to="/tracing">span stream</Link>, which times every model and tool call.{' '}
        <Link to="/tool-calling">Tool calling</Link> explains which path a given model takes and how
        to pin one.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>trace.iterations</code> is empty
            </>,
            'The run took the native or hybrid tool-calling path, which does not keep a per-iteration scratchpad.',
            <>
              Use <code>--explain</code>/<code>--trace</code>, or the span stream. The totals on the
              trace are still correct.
            </>,
          ],
          [
            <>
              <code>No debug trace captured.</code>
            </>,
            'The run finished without the debug machinery attaching anything — a preset with no tools has no loop to trace.',
            <>
              Give the agent a tool, or read the answer, which is printed under the message.
            </>,
          ],
          [
            <>
              <code>Could not create agent config. Provide -m/--model … or --preset</code>, exit{' '}
              <code>2</code>
            </>,
            <>
              <code>effgen debug</code> requires one of the two. It will not choose a model for you.
            </>,
            <>
              Add <code>-m openai:gpt-5-nano</code> or <code>--preset math</code>.
            </>,
          ],
          [
            'Empty boxes and headers with nothing in them',
            'The output was redirected, so the live renderer had no terminal to draw on.',
            <>
              Capture under a pseudo-terminal, or use <code>effgen run --explain</code>.
            </>,
          ],
          [
            <>
              A warning that an agent was garbage-collected without <code>close()</code>
            </>,
            'A `DebugAgent` was created and never closed — common in a notebook or a REPL.',
            <>
              <code>with DebugAgent(config) as agent:</code>, which closes it on the way out.
            </>,
          ],
          [
            'The answer is right but a tool never ran',
            'The model answered from its own knowledge instead of calling the tool.',
            <>
              The trace shows <code>Tool Calls: total=0</code>. A stricter system prompt, or a model
              that advertises tool calling — <code>effgen models list</code> marks them.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>--trace</code> on <code>effgen run</code> and the shareable{' '}
          <code>--card</code> are new, and the run's execution trace and execution tree are now part
          of the saved result. <code>effgen debug</code> and <code>DebugAgent</code> carry over from
          earlier releases with the same surface.
        </p>
      </Callout>

      <SeeAlso paths={['/tracing', '/observability', '/errors']} />
    </DocPage>
  );
}
