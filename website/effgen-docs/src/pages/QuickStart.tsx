import { Zap } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  CodeTabs,
  DocPage,
  SeeAlso,
  Steps,
  Step,
  Terminal,
} from '../components/docs';
import { presetCount, version } from '../siteData';

export default function QuickStart() {
  return (
    <DocPage
      subtitle="An agent that answers a question, from an empty shell to a printed result."
      icon={<Zap size={48} />}
    >
      <p>
        Three things get you a working agent: the package, a model to run against, and one call.
        This page does all three, twice — once from Python and once from the command line — and
        then says what came back.
      </p>

      <h2>The whole thing</h2>

      <CodeBlock
        language="bash"
        code={`pip install effgen
export OPENAI_API_KEY=...`}
      />

      <CodeTabs
        tabs={[
          {
            label: 'Python',
            filename: 'hello.py',
            code: `from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
result = agent.run("What is 17% of 250?")

print(result)                  # printing the response prints the answer
print(result.success, result.tool_call_count)`,
          },
          {
            label: 'Command line',
            language: 'bash',
            code: `effgen run "What is 25 * 17?" -m openai:gpt-5-nano`,
          },
        ]}
        caption="The same agent either way. The command line builds the same AgentConfig the Python call does."
      />

      <Terminal
        command="python hello.py"
        output={`42.5
True 1`}
        caption={`Run against effGen 1.0.0.`}
      />

      <Callout type="note" title="Any of these model ids works">
        <p>
          <code>openai:gpt-5-nano</code> and <code>gemini:gemini-3.1-flash-lite</code> are cheap
          cloud models. <code>Qwen/Qwen2.5-1.5B-Instruct</code> is a bare HuggingFace repo id and
          runs on your own machine with no key at all — it downloads once, then works offline.{' '}
          <Link to="/models">Models and loading</Link> covers the three id forms.
        </p>
      </Callout>

      <h2>Step by step</h2>

      <Steps>
        <Step title="Install the package">
          <p>
            The base install carries the agent loop, the tools, the command line and every
            provider adapter. Extras add the heavy optional stacks — see{' '}
            <Link to="/installation">Installation</Link>.
          </p>
          <CodeBlock language="bash" showLineNumbers={false} code={`pip install effgen`} />
        </Step>

        <Step title="Give it a model">
          <p>
            A cloud model needs its provider's key in the environment. effGen also reads a{' '}
            <code>.env</code> file — <code>~/.effgen/.env</code>, then the nearest one above your
            working directory. Confirm what it can see with <code>effgen doctor</code>, which
            prints which keys are present and never a key value.
          </p>
          <CodeBlock
            language="bash"
            showLineNumbers={false}
            code={`export OPENAI_API_KEY=...
effgen doctor`}
          />
        </Step>

        <Step title="Run a task">
          <p>
            <code>create_agent</code> takes a preset name and a model id. The preset decides the
            tools and the system prompt; there are {presetCount} of them.
          </p>
          <CodeBlock
            showLineNumbers={false}
            code={`from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
print(agent.run("What is 17% of 250?"))`}
          />
        </Step>

        <Step title="Read what came back">
          <p>
            <code>agent.run()</code> returns an <code>AgentResponse</code>, not a string. Printing
            it prints the answer; the fields carry everything else about the run.
          </p>
        </Step>
      </Steps>

      <h2>What the response carries</h2>

      <CodeBlock
        filename="response.py"
        code={`from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
r = agent.run("What is 6 * 7?")

print(r.text)                        # the answer; r.output and r.content are the same string
print(r.success, r.iterations)
print(r.tool_calls.total, r.tool_call_count)
print(r.model, r.provider)
print(r.metadata["cost_usd"], r.metadata["latency_ms"])`}
      />

      <Terminal
        command="python response.py"
        output={`42

Verification:
- Calculator result: 6 * 7 = 42
- Python result: print(6 * 7) -> 42

Conclusion: Indeed, 6 multiplied by 7 equals 42.
True 2
2 2
openai:gpt-5-nano openai
0.00065765 12403.7`}
        caption="Two tool calls over two loop iterations, for two thirds of a tenth of a cent. The math preset asked its calculator and then checked itself with the Python REPL."
      />

      <p>
        <Link to="/agents">Agents</Link> documents every field of both{' '}
        <code>AgentConfig</code> and <code>AgentResponse</code>.
      </p>

      <h2>From the command line</h2>

      <CodeBlock
        language="bash"
        code={`effgen run "What is 25 * 17?" -m openai:gpt-5-nano
effgen run --preset math -m openai:gpt-5-nano "What is the square root of 144?"
effgen chat -m openai:gpt-5-nano
effgen presets`}
      />

      <Terminal
        command={'effgen run "What is 25 * 17?" -m openai:gpt-5-nano'}
        output={`effGen v1.0.1 - Running Task

Initializing agent: cli-agent
Model: openai:gpt-5-nano
Tools: 1 available
Sub-agents: enabled

Task: What is 25 * 17?

Thinking...

Response
╭─────────────────────────────── Agent Response ───────────────────────────────╮
│ 425                                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
✓ Done in 4.6s · 546 tokens · $0.0001`}
        caption="The summary line is what every run ends with: how long it took, how many tools it used, the tokens and the cost."
      />

      <h2>Starting a project instead of a script</h2>
      <p>
        For anything you will come back to, <code>effgen quickstart --init</code> writes a
        configuration, an <code>.env</code> template, a runnable example and a daily spend cap
        into a directory. It calls no model and asks no questions.
      </p>
      <CodeBlock
        language="bash"
        code={`effgen quickstart --init my-agent
cd my-agent
cp .env.example .env      # then paste one key into it
effgen run "What is 25 * 17?" -c effgen.yaml`}
      />
      <p>
        <Link to="/first-project">Your first project</Link> shows every file it writes.
      </p>

      <h2>When it does not work</h2>

      <ApiTable
        headers={['What you see', 'What happened', 'What to do']}
        rows={[
          [
            <>A message naming the missing environment variable</>,
            'No key for the provider you asked for.',
            <>
              Export it, or put it in <code>~/.effgen/.env</code>, then check with{' '}
              <code>effgen doctor</code>.
            </>,
          ],
          [
            <code>ModelNotFoundError</code>,
            'The provider does not serve that model id.',
            <>
              <code>effgen models list --provider &lt;name&gt;</code> shows the ids it does serve;{' '}
              <code>effgen models refresh</code> updates the catalog.
            </>,
          ],
          [
            <code>AmbiguousModelError</code>,
            'The bare id exists on more than one provider.',
            <>
              Prefix it — <code>groq:openai/gpt-oss-120b</code> — or pass{' '}
              <code>provider=</code>.
            </>,
          ],
          [
            <code>BackendUnreachableError</code>,
            'Nothing answered at the endpoint: a refused connection, an unresolvable host or a missing route.',
            <>
              Check the server is up. This raises whatever <code>raise_on_error</code> says — see{' '}
              <Link to="/migration">Migrating to {version}</Link>.
            </>,
          ],
          [
            'The answer is effGen reporting that the run stopped',
            <>
              The agent hit <code>max_iterations</code>, which small models do on multi-step
              tasks.
            </>,
            <>
              Raise it, or set <code>raise_on_error=False</code> and read{' '}
              <code>metadata["partial_output"]</code> for the model's own text.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/first-project', '/agents', '/presets']} />
    </DocPage>
  );
}
