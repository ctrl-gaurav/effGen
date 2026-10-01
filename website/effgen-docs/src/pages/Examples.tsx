import { useMemo, useState } from 'react';
import { PlayCircle, Search } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData } from '../siteData';
import { siteHref } from '../siteLinks';
import './Examples.css';

/**
 * The six examples the main site writes up in full, keyed by the script each
 * one is built from.
 *
 * The rows on this page and the cards on `/examples` therefore name the same
 * scripts, and a reader following either one arrives at the same file in the
 * framework repository.
 */
const WRITTEN_UP: Record<string, { id: string; title: string }> = {
  'tools/coding_agent': { id: 'code-assistant', title: 'Code assistant' },
  'web_retrieval/web_agent': { id: 'research-agent', title: 'Research agent' },
  'advanced/data_processing_agent': { id: 'data-analysis', title: 'Data analysis' },
  'advanced/multi_agent_pipeline': { id: 'multi-agent', title: 'Multi-agent pipeline' },
  'web_retrieval/weather_agent': { id: 'weather-json-pipeline', title: 'Weather, with no API key' },
  'web_retrieval/retrieval_agent': { id: 'rag-knowledge-base', title: 'Retrieval over your own documents' },
};

const REPO = 'https://github.com/ctrl-gaurav/effGen/blob/main';

export default function Examples() {
  const [query, setQuery] = useState('');
  const catalogue = siteData.examples;
  const needle = query.trim().toLowerCase();

  const visible = useMemo(
    () =>
      catalogue.items.filter(
        (item) =>
          !needle ||
          item.name.toLowerCase().includes(needle) ||
          item.summary.toLowerCase().includes(needle),
      ),
    [catalogue.items, needle],
  );

  const byGroup = useMemo(() => {
    const map = new Map<string, typeof visible>();
    for (const item of visible) {
      const list = map.get(item.group);
      if (list) list.push(item);
      else map.set(item.group, [item]);
    }
    return map;
  }, [visible]);

  return (
    <DocPage
      subtitle={`The ${catalogue.count} runnable programs in the framework repository, what each one shows, and how to start it.`}
      icon={<PlayCircle size={48} />}
      toc={false}
    >
      <p>
        These are whole programs, not fragments: each one loads a model, builds an agent and prints
        something. Six of them are written up with their output on{' '}
        <a href={siteHref('/examples')}>the examples page</a>; this page is the whole directory,
        including the ones that need a GPU and the ones that need a key.
      </p>

      <Callout type="warning" title="They ship with the repository, not with the wheel">
        <p>
          <code>pip install effgen</code> does not put <code>examples/</code> on your disk — the
          package build excludes it. Clone the repository, or download the directory, and{' '}
          <code>effgen examples</code> will find it: it looks beside the installed package, then in
          the working directory, then wherever <code>EFFGEN_EXAMPLES_DIR</code> points. With none
          of those present the command says so rather than listing nothing.
        </p>
      </Callout>

      <h2>Listing and running one</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen examples list                       # every script, with the command that runs it
effgen examples run openai/basic_chat      # run one by name

python examples/openai/basic_chat.py       # or start it yourself, from a checkout`}
      />

      <Terminal
        command="effgen examples list"
        output={`Available Examples
                              Example Scripts (52)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name                                 ┃ Command                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ advanced/advanced_streaming_agent    │ effgen examples run                   │
│                                      │ advanced/advanced_streaming_agent     │
│ advanced/agent_communication         │ effgen examples run                   │
│                                      │ advanced/agent_communication          │
│ advanced/async_concurrent_agent      │ effgen examples run                   │
│                                      │ advanced/async_concurrent_agent       │
│ advanced/conversational_agent        │ effgen examples run                   │
│                                      │ advanced/conversational_agent         │
│ basic/basic_agent                    │ effgen examples run basic/basic_agent │
│ basic/qa_agent                       │ effgen examples run basic/qa_agent    │
└──────────────────────────────────────┴───────────────────────────────────────┘`}
        maxLines={18}
        caption={`Trimmed to the first rows. ${catalogue.count} scripts, in ${catalogue.groups.length} directories.`}
      />

      <p>
        A name that does not exist is matched by basename across the tree first, so{' '}
        <code>effgen examples run qa_agent</code> finds <code>basic/qa_agent</code>. When nothing
        matches, the command says so and exits 1.
      </p>

      <Terminal command="effgen examples run nope/not_here" output={`✗ Example not found: nope/not_here`} />

      <Callout
        type="danger"
        title={`${catalogue.parses_arguments} of the ${catalogue.count} cannot be started by the run sub-command`}
      >
        <p>
          A script that builds its own <code>argparse</code> parser sees effGen's command line
          rather than its own, because <code>examples run</code> executes the file in the running
          process without resetting <code>sys.argv</code>. Those{' '}
          {catalogue.parses_arguments} scripts exit 2 with an{' '}
          <em>unrecognized arguments</em> message naming the effGen command you typed. Start them
          with <code>python</code> instead — the last column below says which of the two to use for
          every script.
        </p>
      </Callout>

      <Terminal
        command="effgen examples run tools/coding_agent"
        output={`Running Example: tools/coding_agent

usage: effgen [-h] [--model MODEL] [--interactive] [--regression]
              [--no-cleanup]
effgen: error: unrecognized arguments: examples run tools/coding_agent`}
        caption="The script's own options are parsed — against effGen's argv. Exit code 2."
      />

      <Terminal
        command="python examples/tools/coding_agent.py --help"
        output={`usage: coding_agent.py [-h] [--model MODEL] [--interactive] [--regression]
                       [--no-cleanup]

effGen Code Execution Agent Example

options:
  -h, --help     show this help message and exit
  --model MODEL  Model to use (default: Qwen/Qwen2.5-3B-Instruct)
  --interactive  Interactive chat mode
  --regression   Run regression tests only
  --no-cleanup   Skip cleanup of generated files`}
        caption="The same script, started directly. Its flags work, and --model is how you point it at something other than the local default."
      />

      <h2>What each one needs</h2>

      <ApiTable
        headers={['Directory', 'What it shows', 'Scripts', 'What it needs']}
        rows={[
          [
            <code>basic/</code>,
            'One agent, one question — the shortest path through the framework.',
            String(catalogue.groups.find((g) => g.id === 'basic')?.count ?? ''),
            'A local model; the MLX and GUI ones need Apple Silicon or a display',
          ],
          [
            <code>tools/</code>,
            'Agents that call tools: files, code execution, several at once.',
            String(catalogue.groups.find((g) => g.id === 'tools')?.count ?? ''),
            'A local model',
          ],
          [
            <code>advanced/</code>,
            'Streaming, memory across turns, concurrency, agent-to-agent pipelines.',
            String(catalogue.groups.find((g) => g.id === 'advanced')?.count ?? ''),
            'A local model',
          ],
          [
            <code>web_retrieval/</code>,
            'Web search, the weather, and retrieval over your own documents.',
            String(catalogue.groups.find((g) => g.id === 'web_retrieval')?.count ?? ''),
            'A local model; network access, no key',
          ],
          [
            <code>plugins_presets/</code>,
            'Building from a preset, and registering a tool you wrote.',
            String(catalogue.groups.find((g) => g.id === 'plugins_presets')?.count ?? ''),
            'A local model',
          ],
          [
            <code>openai/</code>,
            'Chat, tool calling, structured output, caching, reasoning effort, and the four native provider tools.',
            String(catalogue.groups.find((g) => g.id === 'openai')?.count ?? ''),
            <code>OPENAI_API_KEY</code>,
          ],
          [
            <code>cerebras/</code>,
            'The same ground on Cerebras: streaming, multi-turn, rate limits, cost tracking.',
            String(catalogue.groups.find((g) => g.id === 'cerebras')?.count ?? ''),
            <code>CEREBRAS_API_KEY</code>,
          ],
          [
            <code>data/</code>,
            'Downloading the ARC set the retrieval examples read.',
            String(catalogue.groups.find((g) => g.id === 'data')?.count ?? ''),
            'Network access',
          ],
          [
            <code>utils/</code>,
            'Sweeping one model across settings to compare them.',
            String(catalogue.groups.find((g) => g.id === 'utils')?.count ?? ''),
            'A local model',
          ],
        ]}
      />

      <Callout type="note" title="The default model in most of them">
        <p>
          Everything outside <code>openai/</code> and <code>cerebras/</code> defaults to{' '}
          <code>Qwen/Qwen2.5-3B-Instruct</code> loaded locally in 4-bit, which downloads about 6 GB
          on first run and wants a GPU. Most take <code>--model</code>, and every one of them can
          be pointed at a cloud model or at your own endpoint by editing the two lines that build
          the config — <Link to="/local-models">Local models</Link> and{' '}
          <Link to="/openai-compatible">Any OpenAI-compatible server</Link> cover both.
        </p>
      </Callout>

      <h2>Every script</h2>

      <div className="examples-controls">
        <label className="examples-search">
          <Search size={16} aria-hidden="true" />
          <span className="sr-only">Search the examples</span>
          <input
            type="search"
            value={query}
            placeholder={`Search ${catalogue.count} scripts by name or description`}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <p className="examples-count" aria-live="polite">
          {visible.length === catalogue.count
            ? `Showing all ${catalogue.count}.`
            : `Showing ${visible.length} of ${catalogue.count}.`}
        </p>
      </div>

      {visible.length === 0 && (
        <p className="examples-empty">Nothing matches that. Clear the search to see all of them.</p>
      )}

      {catalogue.groups.map((group) => {
        const items = byGroup.get(group.id);
        if (!items || items.length === 0) return null;
        return (
          <div key={group.id} className="examples-group">
            <h3>
              <code>{group.id}/</code>
            </h3>
            <ApiTable
              headers={['Script', 'What it does', 'Start it with']}
              rows={items.map((item) => {
                const written = WRITTEN_UP[item.name];
                return [
                  <>
                    <a href={`${REPO}/${item.file}`}>{item.name.split('/')[1]}</a>
                    {written && (
                      <>
                        {' '}
                        <a className="examples-writeup" href={siteHref(`/examples/${written.id}`)}>
                          written up
                        </a>
                      </>
                    )}
                  </>,
                  item.summary || '—',
                  item.parses_arguments ? (
                    <code>python {item.file}</code>
                  ) : (
                    <code>effgen examples run {item.name}</code>
                  ),
                ];
              })}
            />
          </div>
        );
      })}

      <h2>Two that fail here, and why</h2>

      <p>
        Both of these were run against effGen 1.0.1 while this page was written, and neither is
        something you have done wrong.
      </p>

      <Terminal
        command="effgen examples run openai/basic_chat"
        output={`Running Example: openai/basic_chat

Loaded: gpt-5.4-nano (context=1,047,576 tokens)

--- Basic generation ---
Response: A large language model is a machine-learning system trained on vast text data to predict and generate human-like language, often by learning statistical patterns and relationships in the text.
Tokens used: 36
✗ Error running example: 'cost'`}
        maxLines={14}
        caption="The generation succeeds; the script then reads a metadata key that is not there."
      />

      <p>
        <code>examples/openai/basic_chat.py</code> prints{' '}
        <code>result.metadata['cost']</code>. The OpenAI adapter reports the spend under{' '}
        <code>cost_usd</code> and <code>total_cost</code>, so that line raises{' '}
        <code>KeyError: 'cost'</code> after the answer has already been printed. Use{' '}
        <code>result.metadata['cost_usd']</code>, or <code>model.get_total_cost()</code> for the
        running total — <Link to="/cost">Cost &amp; budgets</Link> has the ledger behind both.
      </p>

      <CodeBlock
        filename="metadata.py"
        code={`from effgen import load_model

model = load_model("gpt-5-nano", provider="openai")
result = model.generate("Say the word ready.")
print("text     :", result.text.strip())
print("metadata :", sorted(result.metadata))
print("get_total_cost():", model.get_total_cost())`}
      />

      <Terminal
        command="python metadata.py"
        output={`text     : ready
metadata : ['cached_input_tokens', 'completion_tokens', 'cost_usd', 'duration_s', 'latency_ms', 'prompt_tokens', 'reasoning_tokens', 'tool_calls', 'total_cost', 'total_tokens', 'truncated']
get_total_cost(): 5.575e-05`}
        maxLines={10}
      />

      <Terminal
        command="effgen examples run cerebras/basic_cerebras"
        output={`Running Example: cerebras/basic_cerebras

Loading Cerebras adapter (gpt-oss-120b) ...
Context length : 65,536 tokens
✗ Error running example: Cerebras generation failed : Error code: 402 -
{'message': 'Payment required to access this resource. Visit your billing tab.',
'type': 'payment_required_error', 'param': 'quota', 'code': 'payment_required'}.`}
        maxLines={12}
        caption="A 402 from the provider, not a fault in the example: the account this was run under has no Cerebras credit. The adapter reports the provider's own message rather than a generic failure."
      />

      <SeeAlso paths={['/tutorials', '/cookbook', '/quickstart']} />
    </DocPage>
  );
}
