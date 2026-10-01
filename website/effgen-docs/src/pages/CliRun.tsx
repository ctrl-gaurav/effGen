import { Play } from 'lucide-react';
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
import { siteData } from '../siteData';

const chatSlashCommands = siteData.code.chat_slash_commands;

export default function CliRun() {
  return (
    <DocPage
      subtitle="One task and an answer with effgen run; a conversation that remembers with effgen chat. Same agent, same flags, two shapes."
      icon={<Play size={48} />}
    >
      <p>
        <code>effgen run</code> takes a task, runs an agent, prints the answer and exits.{' '}
        <code>effgen chat</code> keeps the conversation open. They share a model, a preset, a tool
        list, a persona, a guardrail preset and a session store, so what you learn on one applies to
        the other.
      </p>

      <h2>The shortest useful run</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "What is 7*6? Use the calculator." -m openai:gpt-5-nano -t calculator`}
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
            Without <code>--explain</code> the run prints the answer panel and a one-line summary.{' '}
            <Link to="/debug">Debugging</Link> covers the trace flags.
          </>
        }
      />

      <h2>Options on <code>effgen run</code></h2>

      <ParamTable
        nameLabel="Flag"
        params={[
          {
            name: 'task',
            description: 'Task description (launches the interactive wizard if not provided)',
          },
          { name: '-m MODEL, --model MODEL', description: 'Model to use' },
          {
            name: '--provider PROVIDER',
            description: (
              <>
                Provider for a bare model id (openai, groq, cerebras, gemini, together, fireworks,
                replicate, anthropic, hf). Equivalent to the <code>provider:model</code> prefix.
              </>
            ),
          },
          { name: '-n NAME, --name NAME', description: 'Agent name' },
          { name: '-t TOOLS …, --tools TOOLS …', description: 'Tools to enable' },
          { name: '-c CONFIG, --config CONFIG', description: 'Configuration file' },
          {
            name: '--system-prompt TEXT, --persona TEXT',
            description:
              'Custom persona / system prompt for this run — e.g. "You are a patient Socratic tutor who never gives the answer."',
          },
          { name: '--temperature TEMPERATURE', type: 'float', description: 'Temperature' },
          {
            name: '--max-tokens MAX_TOKENS',
            type: 'int',
            description:
              'Max output tokens. Raise for token-heavy or reasoning models, which spend part of the budget on hidden reasoning before any visible text.',
          },
          { name: '--max-iterations MAX_ITERATIONS', type: 'int', description: 'Max iterations' },
          { name: '--mode {auto,single,sub_agents}', description: 'Execution mode' },
          { name: '--no-sub-agents', type: 'flag', description: 'Disable sub-agents' },
          { name: '--stream', type: 'flag', description: 'Stream output' },
          {
            name: '-o OUTPUT, --output OUTPUT',
            description:
              'Write the full result as a JSON document to this file (output, success, tool_calls, tokens, cost, trace, citations, metadata)',
          },
          {
            name: '--card PATH.html',
            description: (
              <>
                Write a shareable HTML card for this run — the task, the answer, the tool trace with
                per-step durations, sources and citations, and tokens/cost/latency. Self-contained
                and opens with no network access.{' '}
                <Link to="/cli/reports">Reports and run cards</Link>.
              </>
            ),
          },
          {
            name: '--json',
            type: 'flag',
            description:
              'Emit that same JSON result object to stdout (for piping to jq). Human output goes to stderr; combine with -q for clean stdout.',
          },
          {
            name: '--preset {coding,general,math,media,minimal,multimodal,notify,rag,research}',
            description: 'Use a preset agent configuration',
          },
          {
            name: '--guardrails NAME',
            description: (
              <>
                Redact or block PII and screen for prompt injection before the task reaches the
                model: <code>strict</code>, <code>standard</code> (alias{' '}
                <code>default</code>/<code>balanced</code>), <code>phi</code> (alias{' '}
                <code>hipaa</code>/<code>deidentify</code>), <code>minimal</code> or{' '}
                <code>none</code>. Also honoured from a <code>-c/--config</code> file's{' '}
                <code>guardrails</code> key.
              </>
            ),
          },
          { name: '--explain', type: 'flag', description: 'Show why the agent chose each tool' },
          {
            name: '--trace',
            type: 'flag',
            description: 'Show a step-by-step timeline with per-step durations',
          },
          {
            name: '--checkpoint-dir CHECKPOINT_DIR',
            description: 'Directory to write agent checkpoints',
          },
          {
            name: '--checkpoint-interval CHECKPOINT_INTERVAL',
            type: 'int',
            description: 'Checkpoint every N iterations (requires --checkpoint-dir)',
          },
          {
            name: '--session-id ID',
            description: (
              <>
                Persistent conversation session id, shared with <code>effgen chat --session-id</code>{' '}
                and <code>effgen sessions</code>. Recalls prior turns and saves new ones. Distinct
                from <code>effgen resume --checkpoint</code>, which restores a mid-run snapshot.
              </>
            ),
          },
          {
            name: '-f PATH, --file PATH, --input PATH',
            description: (
              <>
                Attach a file. An image is passed as multimodal input; a document or a source file
                is read and prepended to the task as context; any other file that decodes as UTF-8
                is read as plain text. Repeatable.
              </>
            ),
          },
          { name: '-v, --verbose', type: 'flag', description: 'Verbose output (DEBUG/INFO logs)' },
          { name: '-q, --quiet', type: 'flag', description: 'Quiet output (errors only)' },
          {
            name: '--no-animation',
            type: 'flag',
            description: 'Disable live spinners and progress animation',
          },
        ]}
        caption={
          <>
            Every flag <code>effgen run --help</code> declares. <code>-c</code> here is{' '}
            <code>--config</code>, not <code>--concurrency</code> as it is on{' '}
            <code>effgen batch</code>.
          </>
        }
      />

      <h2>Attaching a file</h2>

      <p>
        <code>-f/--file</code> decides what to do from the extension, and it is repeatable — a
        question about three files is three flags.
      </p>

      <ApiTable
        headers={['File', 'What happens to it']}
        rows={[
          [
            <>
              <code>.png</code> <code>.jpg</code> <code>.gif</code> <code>.webp</code> …
            </>,
            <>
              Passed as multimodal input, so the model sees the image.{' '}
              <Link to="/multimodal">Multimodal</Link>.
            </>,
          ],
          [
            <>
              <code>.pdf</code> <code>.docx</code> <code>.xlsx</code> <code>.txt</code>{' '}
              <code>.md</code> <code>.csv</code> …
            </>,
            'Read and prepended to the task as context.',
          ],
          [
            <>
              <code>.py</code> <code>.js</code> <code>.ts</code> <code>.go</code> <code>.rs</code>{' '}
              <code>.java</code> <code>.sql</code> …
            </>,
            'The same — read and prepended.',
          ],
          ['Anything else that decodes as UTF-8', 'Read as plain text.'],
        ]}
      />

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "Summarize this" --file report.pdf -m openai:gpt-5-nano
effgen run "Draft a reply" --persona "terse, formal" --json | jq .output`}
      />

      <h2>The JSON contract</h2>

      <p>
        <code>--json</code> puts one document on stdout and everything a person reads on stderr.{' '}
        <code>-q</code> silences the human half entirely.
      </p>

      <Terminal
        command={`effgen run "Say the word ok and nothing else." -m gemini:gemini-3.1-flash-lite -q --json | jq "{success, output, cost: .metadata.cost_usd}"`}
        output={`{
  "success": true,
  "output": "Ok",
  "cost": 3.95e-05
}`}
      />

      <Callout type="note" title="Fixed in 1.1.0: a run that called a tool serializes">
        <p>
          In 1.0.0 and 1.0.1, <code>--json</code>, <code>-o FILE</code> and <code>--card</code> failed
          with <code>Object of type ToolCall is not JSON serializable</code> whenever the run made a
          tool call, because the saved document's <code>execution_tree</code> carried tool-call
          objects. From 1.1.0 the tree is written as data, so those runs serialize like any other.
          The three documents, and <code>--show-thread</code>, go through one scrubber; the terminal
          answer panel still prints the run's own words unredacted.
        </p>
      </Callout>

      <h3>Reading the run back step by step</h3>
      <p>
        <code>--show-thread</code> prints the run's conversation after the answer: the frame it
        started from, every thought, every tool call with its input, every result, and the answer.
        The same steps are in the <code>--json</code> document under{' '}
        <code>metadata.thread</code>.
      </p>

      <CodeBlock
        language="bash"
        code={`effgen run "What is 7*6? Use the calculator." -m openai:gpt-5-nano -t calculator --show-thread`}
      />

      <h3>What a saved result carries</h3>

      <Terminal
        command={`effgen run "Name one prime number under 10." -m openai:gpt-5-nano -o saved-run.json -q && python -c 'import json; print(sorted(json.load(open("saved-run.json"))))'`}
        output={`['citations', 'execution_time', 'execution_trace', 'execution_tree', 'iterations',
 'metadata', 'mode', 'model', 'outcome', 'output', 'partial', 'provider',
 'routing_decision', 'sources', 'started_at', 'stop_reason', 'success', 'task',
 'tokens_used', 'tool_call_details', 'tool_calls']`}
        caption={
          <>
            <code>tool_calls</code> is the count and <code>tool_call_details</code> is the list.{' '}
            <code>outcome</code>, <code>stop_reason</code> and <code>partial</code> say how the
            run ended. <code>metadata</code> carries <code>cost_usd</code>, the token breakdown, latency,{' '}
            <code>partial_output</code> and <code>input_redaction</code>.{' '}
            <code>effgen report saved-run.json</code> turns it into HTML.
          </>
        }
      />

      <h2><code>effgen chat</code></h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen chat -m openai:gpt-5-nano
effgen chat --preset research -t calculator wikipedia
effgen chat --session-id support-42       # resume a saved session`}
      />

      <Terminal
        command="effgen chat -m openai:gpt-5-nano"
        output={`
effGen v1.0.1 · chat
Model: gpt-5-nano
Type your message and press Enter.  End a line with \\ for multi-line input.
Slash commands (type / for the menu): /help  /model  /tools  /status  /cost  /trace  /reset  /save
/session  /load  /doctor  /exit
gpt-5-nano › What is 2+2?
assistant

4
· 1.2s · 226 tok · $0.000082
gpt-5-nano › /cost

Session cost
Turns: 1
Tokens: 226
Cost: $0.000082
(process total across all models: $0.000082)
gpt-5-nano › /exit
Goodbye!`}
        maxLines={22}
        caption={
          <>
            Captured on a real pseudo-terminal. The per-turn footer carries time, tokens and cost;{' '}
            <code>/cost</code> gives the session total and the process total side by side.
          </>
        }
      />

      <h3>Slash commands in a chat session</h3>

      <ApiTable
        headers={['Command', 'Does']}
        rows={chatSlashCommands.map((command) => [<code>{command.name}</code>, command.summary])}
        caption={
          <>
            All <code>{siteData.code.chat_slash_command_count}</code>, read off the session's own
            command table. <Link to="/cli/code"><code>effgen code</code></Link> has a larger set of{' '}
            <code>{siteData.code.slash_command_count}</code>, including the ones that write files.
          </>
        }
      />

      <h3>Options on <code>effgen chat</code></h3>

      <p>
        The same model, preset, tool, persona, guardrail, temperature and max-token flags as{' '}
        <code>run</code>, plus:
      </p>

      <ParamTable
        nameLabel="Flag"
        params={[
          {
            name: '--session-id ID, --resume ID',
            description: (
              <>
                Continue a persistent session by id — the same store as{' '}
                <code>effgen run --session-id</code> and <code>effgen sessions list</code>. Prior
                turns are recalled and new turns saved; a new id starts a fresh session.
              </>
            ),
          },
          {
            name: '--preset {…}',
            description: (
              <>
                Attaches the preset's tools and system prompt, the same as{' '}
                <code>effgen run --preset</code>.
              </>
            ),
          },
          {
            name: '-t TOOL …, --tools TOOL …',
            description: (
              <>
                Tools for the session. Also addable mid-session with <code>/tools</code>.
              </>
            ),
          },
          {
            name: '--system-prompt TEXT, --persona TEXT',
            description: (
              <>
                Steers every reply — unlike <code>--preset</code>, which only labels the session.
              </>
            ),
          },
          {
            name: '--guardrails NAME',
            description: (
              <>
                Applied on every turn, and carried across a <code>/model</code> or{' '}
                <code>/tools</code> rebuild.
              </>
            ),
          },
        ]}
        caption={
          <>
            Every flag <code>effgen chat --help</code> declares that is not shared with{' '}
            <code>run</code>. It has no <code>--json</code>, no <code>-o</code> and no{' '}
            <code>--card</code>: a conversation is read back with{' '}
            <Link to="/cli/history"><code>effgen sessions show</code></Link>.
          </>
        }
      />

      <h2>Which to reach for</h2>

      <ApiTable
        headers={['You want', 'Use']}
        rows={[
          ['One answer, in a script', <><code>effgen run … --json</code></>],
          [
            'One answer, and to keep the context for later',
            <>
              <code>effgen run … --session-id ID</code>
            </>,
          ],
          ['To go back and forth', <><code>effgen chat</code></>],
          [
            'Many prompts from a file',
            <>
              <Link to="/cli/batch"><code>effgen batch</code></Link>
            </>,
          ],
          [
            'To change files in a repository',
            <>
              <Link to="/cli/code"><code>effgen code</code></Link>
            </>,
          ],
          [
            'To see every step',
            <>
              <code>--explain</code> / <code>--trace</code>, or{' '}
              <Link to="/debug"><code>effgen debug</code></Link>
            </>,
          ],
        ]}
      />

      <h2>Sessions</h2>

      <p>
        <code>--session-id</code> makes two separate processes one conversation. The id is yours to
        choose, and the same store is shared by <code>run</code>, <code>chat</code> and{' '}
        <code>code</code>.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "My favourite number is 41. Remember it." -m openai:gpt-5-nano --session-id demo-42 -q
effgen run "What is my favourite number plus one?" -m openai:gpt-5-nano --session-id demo-42 -q`}
      />

      <Terminal
        command={`effgen run "What is my favourite number plus one?" -m openai:gpt-5-nano --session-id demo-42 -q`}
        output={`
Response
╭───────────────────────────────────────── Agent Response ─────────────────────────────────────────╮
│ 42                                                                                               │
│                                                                                                  │
│ Since your favourite number is 41, 41 + 1 = 42.                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯`}
        caption={
          <>
            <Link to="/cli/history">Runs and sessions history</Link> covers reading them back;{' '}
            <Link to="/sessions">Sessions</Link> covers the library API behind them.
          </>
        }
      />

      <Callout type="note" title="A session is not a checkpoint">
        <p>
          <code>--session-id</code> saves a <em>conversation</em>. <code>--checkpoint-dir</code>{' '}
          with <code>--checkpoint-interval</code> saves a <em>mid-run snapshot</em>, which{' '}
          <code>effgen resume --checkpoint</code> restarts. Different stores, different commands —{' '}
          <Link to="/checkpointing">Checkpointing</Link> covers the second.
        </p>
      </Callout>

      <h2>Where files land</h2>

      <p>
        <code>EFFGEN_WORKSPACE</code> names the directory the file and shell tools read and write by
        default, and the only directory sandboxed code may write to. Set it to keep files an agent
        generates out of the current directory; it is created if missing. Unset, it is the current
        directory.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>Object of type ToolCall is not JSON serializable</code>, exit <code>1</code>
            </>,
            'The serialization defect in 1.0.0 and 1.0.1, fixed in 1.1.0. The run succeeded; writing it out did not.',
            <>
              <code>pip install -U effgen</code>. On an older release, <code>--card out.html</code>{' '}
              or <code>effgen runs show &lt;id&gt; --json</code>.
            </>,
          ],
          [
            <>
              <code>The model `…` does not exist or you do not have access to it</code>
            </>,
            'The catalog knows the id — it carries the date it was verified — but the provider has retired it, or this account is not entitled to it.',
            <>
              <code>effgen models refresh --provider &lt;name&gt;</code>, then pick a current id
              from <code>effgen models list</code>.
            </>,
          ],
          [
            'The answer is empty and the log mentions a max_tokens cap',
            'A reasoning model spent its whole output budget on internal reasoning before writing anything. The warning names the finish reason and the cap.',
            <>
              Raise <code>--max-tokens</code> — 8192 is a reasonable starting point — or use a model
              that answers without an extended reasoning chain.
            </>,
          ],
          [
            <>
              A panel saying the run stopped at its iteration cap, then{' '}
              <em>Partial progress</em>
            </>,
            'No final answer was written. What follows the heading is tool output and reasoning, deliberately not presented as a result.',
            <>
              Raise <code>--max-iterations</code>, or use a model that needs fewer steps.
            </>,
          ],
          [
            <>
              <code>Rate limited … 429 RESOURCE_EXHAUSTED</code>
            </>,
            'The provider quota for that key and model is spent for the window. The client already honoured the stated retry delay and did not retry again at the agent layer.',
            <>
              Wait out the window, lower concurrency, or route to another provider —{' '}
              <Link to="/routing">Model routing and fallback</Link>.
            </>,
          ],
          [
            'The command opens a wizard instead of running',
            <>
              No task argument was given. <code>effgen run</code> with no task launches the
              interactive setup wizard.
            </>,
            'Quote the task as the first argument.',
          ],
          [
            'A tool never fires even though it is attached',
            'The model answered from its own knowledge, or it does not advertise tool calling.',
            <>
              <code>--explain</code> shows <code>total=0</code> when nothing ran.{' '}
              <code>effgen models list</code> marks the models that support tools;{' '}
              <Link to="/tool-calling">Tool calling</Link> covers the paths.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>--session-id</code>, <code>--card</code>, <code>--trace</code>,{' '}
          <code>--guardrails</code>, <code>--persona</code> and repeatable <code>--file</code> are
          new on <code>run</code>; <code>chat</code> gained <code>--session-id</code>,{' '}
          <code>--preset</code>, <code>--guardrails</code> and the shared answer presentation. Two
          behaviours changed: <code>raise_on_error</code> now defaults to <code>True</code>, and an
          unreachable backend raises <code>BackendUnreachableError</code> regardless of that flag —{' '}
          <Link to="/migration">Migrating to 1.2.0</Link>. 1.1.0 added <code>--show-thread</code>.
        </p>
      </Callout>

      <SeeAlso paths={['/cli', '/cli/code', '/cli/history']} />
    </DocPage>
  );
}
