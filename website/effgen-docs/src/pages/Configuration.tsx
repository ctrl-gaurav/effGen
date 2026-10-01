import { Settings } from 'lucide-react';
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
import { siteData, version } from '../siteData';

export default function Configuration() {
  return (
    <DocPage
      subtitle="Every setting an agent takes, where it can be set, and which source wins."
      icon={<Settings size={48} />}
    >
      <p>
        The same agent can be described in a Python call, in a YAML file, in environment variables
        or in command-line flags. They are the same settings — the fields of{' '}
        <code>AgentConfig</code> — reached four ways, and this page says which one wins when two
        of them disagree.
      </p>

      <h2>The shortest configured run</h2>

      <CodeBlock
        language="yaml"
        filename="effgen.yaml"
        code={`model: openai:gpt-5-nano
system_prompt: You are a helpful assistant. Answer concisely and say when you are unsure.
temperature: 0.2
max_tokens: 512
max_iterations: 5`}
      />

      <CodeBlock
        language="bash"
        code={`effgen run "What is 25 * 17?" -c effgen.yaml`}
      />

      <p>
        The same file from Python, through <code>ConfigLoader</code>:
      </p>

      <CodeBlock
        filename="load.py"
        code={`from effgen import ConfigLoader

config = ConfigLoader().load_config("effgen.yaml")
print(type(config).__name__)
print(config)`}
      />

      <Terminal command="python load.py" output={`Config
Config(data={'model': 'openai:gpt-5-nano', 'system_prompt': 'You are a helpful assistant. Answer concisely.', 'temperature': 0.2, 'max_tokens': 512, 'max_iterations': 5}, _source_file=None, _loaded_at=datetime.datetime(2026, 8, 22, 20, 25, 55, 249004))`} />

      <h2>Which source wins</h2>

      <ApiTable
        headers={['Source', 'Beats', 'Example']}
        rows={[
          [
            '1. A command-line flag',
            'everything below it',
            <code>effgen run "…" -c effgen.yaml -m gemini:gemini-3.1-flash-lite</code>,
          ],
          [
            '2. A keyword on the call',
            'the config it is passed with',
            <code>agent.run("…", temperature=0)</code>,
          ],
          [
            '3. A field on AgentConfig',
            'the YAML file that built it',
            <code>AgentConfig(model="openai:gpt-5-nano", temperature=0.2)</code>,
          ],
          [
            '4. A configuration file',
            'the environment',
            <code>model: openai:gpt-5-nano</code>,
          ],
          [
            '5. A real environment variable',
            'a .env file',
            <code>export OPENAI_API_KEY=…</code>,
          ],
          ['6. A .env file', 'nothing — it is the floor', <code>~/.effgen/.env</code>],
        ]}
        caption="A value already exported in your shell is never overwritten by a file effGen loads."
      />

      <h2>Where keys come from</h2>
      <p>
        effGen loads <code>.env</code> files before running any command, in a fixed order. Earlier
        entries win, and a real environment variable always beats a file.
      </p>

      <ApiTable
        headers={['Order', 'Path', 'What it is for']}
        rows={[
          [
            '1',
            <code>$EFFGEN_DOTENV</code>,
            'An explicit path you set. Nothing else is searched ahead of it.',
          ],
          [
            '2',
            <code>~/.effgen/.env</code>,
            'Your per-user keys, shared by every project on the machine.',
          ],
          [
            '3',
            <>
              <code>./.env</code> and each parent directory
            </>,
            "The nearest project .env above your working directory — a checkout's repository root, usually.",
          ],
        ]}
      />

      <CodeBlock
        language="bash"
        code={`export EFFGEN_DOTENV=/secure/keys/effgen.env   # look here and nowhere else
effgen doctor

export EFFGEN_NO_DOTENV=1                     # or EFFGEN_DOTENV=none
effgen serve                                  # sees only what the orchestrator injected`}
      />

      <Callout type="tip" title="Turn the search off in production">
        <p>
          <code>EFFGEN_NO_DOTENV=1</code> stops the filesystem search entirely, so a server process
          sees only the environment variables it was given and never a stray <code>.env</code> left
          in a deploy image or a working directory.
        </p>
      </Callout>

      <p>
        The same discovery is available to a script or a notebook through{' '}
        <code>effgen.load_env()</code>, which returns the paths it loaded and never overwrites a
        value already in the environment.
      </p>

      <h3>Provider keys</h3>

      <ApiTable
        headers={['Provider', 'Environment variable']}
        rows={siteData.models.providers.map((provider) => [
          <code>{provider.name}</code>,
          provider.env_keys.map((key, i) => (
            <span key={key}>
              {i > 0 ? ' or ' : ''}
              <code>{key}</code>
            </span>
          )),
        ])}
        caption={
          <>
            Read from the installed provider registry. <code>openai_compatible</code> reads an
            endpoint rather than a credential — see{' '}
            <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
          </>
        }
      />

      <Terminal
        command="effgen doctor"
        output={`               effgen doctor — Provider Status                
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Provider          ┃ Key     ┃ Env Var             ┃ Models ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ anthropic         │ missing │ —                   │     17 │
│ cerebras          │ present │ CEREBRAS_API_KEY    │      2 │
│ fireworks         │ present │ FIREWORKS_API_KEY   │     16 │
│ gemini            │ present │ GOOGLE_API_KEY      │      8 │
│ groq              │ present │ GROQ_API_KEY        │     14 │
│ hf                │ present │ HF_TOKEN            │    124 │
│ openai            │ present │ OPENAI_API_KEY      │     30 │
│ openai_compatible │ missing │ —                   │      0 │
│ replicate         │ present │ REPLICATE_API_TOKEN │     37 │
│ together          │ present │ TOGETHER_API_KEY    │    168 │
└───────────────────┴─────────┴─────────────────────┴────────┘

System
┌───────────────────────────┬─────────────────────┐
│ Physical GPUs (NVML)      │ 8                   │
│ Driver CUDA               │ 13.3                │
│ torch CUDA build          │ 13.0                │
│ torch.cuda.is_available() │ True                │
│ torch                     │ 2.11.0+cu130        │
│ vLLM                      │ importable (0.26.0) │
└───────────────────────────┴─────────────────────┘

Coding (effgen code)
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check     ┃ Status        ┃ Detail                                           ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ workspace │ ready         │ /tmp/effgen-project (from current directory)     │
│ sandbox   │ limited       │ subprocess — network isolated, writes confined   │
│           │               │ to the workspace                                 │
│ git       │ no repository │ git version 2.34.1; /tmp/effgen-project is not   │
│           │               │ inside a repository                              │
└───────────┴───────────────┴──────────────────────────────────────────────────┘
  sandbox: Install Docker to confine reads as well; the subprocess sandbox is 
used until then.
  git: Run effgen code from a repository (or git init the workspace) for 
branch/status context.
  Try it: effgen code "write fib.py and run it"

Missing keys — set in ~/.effgen/.env or export:
  export ANTHROPIC_API_KEY=<your-key>
  export EFFGEN_BASE_URL or OPENAI_BASE_URL or OPENAI_API_BASE=<your-key>`}
        caption="doctor reports which keys effGen can see. It never prints a key value."
      />

      <h2>The configuration file</h2>
      <p>
        A configuration file carries only the keys a run applies:{' '}
        <code>model</code>, <code>provider</code>, <code>system_prompt</code>,{' '}
        <code>temperature</code>, <code>max_tokens</code>, <code>max_iterations</code> and{' '}
        <code>guardrails</code>. Any other key that names an <code>AgentConfig</code> field is
        reported when the file loads, so a setting can never be a silent no-op.
      </p>

      <ParamTable
        nameLabel="Key"
        params={[
          {
            name: 'model',
            type: 'str',
            required: true,
            description:
              'The model id, written provider-prefixed, because a file travels without a --provider flag beside it.',
          },
          {
            name: 'provider',
            type: 'str',
            description: 'Provider for a bare model id, when the id is not prefixed.',
          },
          {
            name: 'system_prompt',
            type: 'str',
            description: 'System-level instructions for every run that loads this file.',
          },
          {
            name: 'temperature',
            type: 'float',
            default: '0.7',
            description: 'Generation temperature.',
          },
          {
            name: 'max_tokens',
            type: 'int',
            description:
              'Output-token budget for one answer. A reasoning model spends part of it on hidden reasoning before it emits a visible token, so it needs a larger one.',
          },
          {
            name: 'max_iterations',
            type: 'int',
            default: '10',
            description: 'How many steps the agent may take on one task.',
          },
          {
            name: 'guardrails',
            type: 'str | list',
            description: (
              <>
                A guardrail preset name or chain — see <Link to="/guardrails">Guardrails</Link>.
              </>
            ),
          },
        ]}
        caption="The keys effgen.yaml applies. Everything else on AgentConfig is set in Python."
      />

      <Callout type="warning" title="max_tokens follows the model">
        <p>
          A reasoning model spends part of its output budget on hidden reasoning before the first
          visible token. A budget that is too small can run out before a word is emitted — and
          that empty answer is still billed. <code>effgen quickstart --init</code> writes the
          larger budget when it detects such a model, and says why in the file.
        </p>
      </Callout>

      <h2>The config command</h2>

      <ParamTable
        nameLabel="Command"
        params={[
          {
            name: 'effgen config show',
            type: '-f, --file FILE',
            description: 'Print the configuration effGen would use.',
          },
          {
            name: 'effgen config validate',
            type: '-f, --file FILE',
            required: true,
            description: 'Check a file and report what is wrong with it. The file is required.',
          },
          {
            name: 'effgen config init',
            type: '-o, --output FILE · --force',
            description:
              'Write a starter configuration. Defaults to config.yaml; --force overwrites an existing one.',
          },
          {
            name: 'effgen config set',
            type: 'key value',
            required: true,
            description: (
              <>
                Set one value, such as <code>budget.daily 1.0</code>.
              </>
            ),
          },
        ]}
        caption={<><code>effgen config --help</code>, {version}.</>}
      />

      <Terminal command="effgen config init" output={`✓ Configuration initialized: config.yaml
Run it with: effgen run "What is 25 * 17?" -c config.yaml
A whole project — this file, a .env template and a runnable example — comes 
from: effgen quickstart --init`} />

      <h2>Output and diagnostics</h2>

      <ParamTable
        nameLabel="Flag"
        params={siteData.cli.global_options.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={<><code>effgen --help</code> — the options every command accepts.</>}
      />

      <p>
        By default the command line is quiet: tables and answers print with no library log noise.
        The live status line while a task runs is TTY-aware and disappears before the answer
        prints, so redirected output is never corrupted. It turns itself off when output is piped,
        when a CI environment is detected, and when any of <code>--no-animation</code>,{' '}
        <code>--quiet</code>, <code>NO_COLOR</code> or <code>EFFGEN_NO_ANIM=1</code> applies.
      </p>

      <h2>Environment variables</h2>
      <p>
        effGen reads its settings from <code>EFFGEN_*</code> variables. The ones that configure
        the command line and the library are below; the server's, the sandbox's and the coding
        agent's are documented on their own pages.
      </p>

      <ApiTable
        headers={['Variable', 'What it does']}
        rows={[
          [<code>EFFGEN_DOTENV</code>, <>An explicit <code>.env</code> path, searched ahead of everything else. <code>none</code> disables the search.</>],
          [<code>EFFGEN_NO_DOTENV</code>, <>Set to <code>1</code> to disable the <code>.env</code> search entirely.</>],
          [<code>EFFGEN_HOME</code>, <>Where effGen keeps its own state. Defaults to <code>~/.effgen</code>.</>],
          [<code>EFFGEN_DEFAULT_MODEL</code>, 'The model used when a call names none. Without it, effGen asks rather than picking a paid model for you.'],
          [<code>EFFGEN_BASE_URL</code>, <>An OpenAI-protocol endpoint, consulted before <code>OPENAI_BASE_URL</code> and <code>OPENAI_API_BASE</code>.</>],
          [<code>EFFGEN_THEME</code>, <>The command line's colour theme: {siteData.cli.themes.map((theme, i) => (<span key={theme}>{i > 0 ? ', ' : ''}<code>{theme}</code></span>))}.</>],
          [<code>EFFGEN_NO_ANIM</code>, 'Turn off the live status line and the progress bars.'],
          [<code>EFFGEN_TIPS</code>, 'Turn the occasional usage tip off.'],
          [<code>EFFGEN_NO_GPU_WARN</code>, 'Silence the warning that torch cannot use the NVIDIA GPUs it can see.'],
          [<code>EFFGEN_PLUGINS_DIR</code>, 'Where to look for tool plugins.'],
          [<code>EFFGEN_DISABLE_PLUGINS</code>, 'Load no plugins at all.'],
          [<code>EFFGEN_PROMPTS_DIR</code>, 'Where to look for prompt templates of your own.'],
          [<code>EFFGEN_EXAMPLES_DIR</code>, <>Where the example programs live, for <code>effgen examples</code> outside a checkout.</>],
          [<code>EFFGEN_SESSIONS_DIR</code>, 'Where stored conversations are kept.'],
          [<code>EFFGEN_RUN_HISTORY_DIR</code>, <>Where run history is kept. <code>EFFGEN_RUN_HISTORY=0</code> turns recording off, and <code>EFFGEN_RUN_HISTORY_MAX_DAYS</code> ages it out.</>],
          [<code>EFFGEN_WORKFLOW_DIR</code>, 'Where workflow checkpoints are kept.'],
          [<code>EFFGEN_HEALTH_REMOTE</code>, <>Allow <code>effgen health</code> to make its network checks, which are otherwise opt-in.</>],
          [<code>CUDA_VISIBLE_DEVICES</code>, 'Which GPUs a local engine may use. A standard NVIDIA variable, honoured as usual.'],
        ]}
        caption={
          <>
            Server settings (<code>EFFGEN_API_KEY</code>, <code>EFFGEN_RATE_LIMIT</code>,{' '}
            <code>EFFGEN_CORS_ORIGINS</code>, the OIDC pair) are on{' '}
            <Link to="/api-server">API server</Link>; the sandbox's are on{' '}
            <Link to="/execution">Code execution</Link>.
          </>
        }
      />

      <h2>Memory settings</h2>
      <p>
        <code>AgentConfig.memory_config</code> is a dict, and these are its keys. Anything not
        given keeps the default.
      </p>

      <ParamTable
        nameLabel="Key"
        params={[
          { name: 'short_term_max_tokens', type: 'int', default: '4096', description: 'Token budget for the working conversation.' },
          { name: 'short_term_max_messages', type: 'int', default: '100', description: 'How many messages are kept before the oldest are folded away.' },
          { name: 'summarization_threshold', type: 'float', default: '0.8', description: 'The share of the token budget that triggers summarisation.' },
          { name: 'keep_recent_messages', type: 'int', default: '4', description: 'Recent messages that are never summarised.' },
          { name: 'summary_budget_ratio', type: 'float', default: '0.4', description: 'The share of the budget retained summaries may occupy; older summaries are folded together to stay inside it.' },
          { name: 'long_term_backend', type: 'str', default: '"sqlite"', description: 'Where long-term memory is stored.' },
          { name: 'long_term_persist_path', type: 'str | None', default: 'None', description: 'Where that store lives on disk.' },
          { name: 'auto_summarize', type: 'bool', default: 'True', description: 'Whether old context is summarised automatically.' },
        ]}
        caption={
          <>
            What is <em>dropped</em> when the window fills is a separate choice — see{' '}
            <Link to="/compaction">Context compaction</Link>.
          </>
        }
      />

      <h2>When configuration goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What happened', 'What to do']}
        rows={[
          [
            'A key in the file is reported when it loads',
            <>
              The key names an <code>AgentConfig</code> field the file format does not apply.
            </>,
            'Set it in Python, or remove it. It is reported rather than ignored so it is never a silent no-op.',
          ],
          [
            'The key you exported is not seen',
            <>
              A different <code>.env</code> won, or the search is disabled.
            </>,
            <>
              <code>effgen doctor</code> reports what is visible;{' '}
              <code>EFFGEN_DOTENV</code> pins one file.
            </>,
          ],
          [
            'An answer arrives empty but billed',
            <>
              <code>max_tokens</code> ran out during a reasoning model's hidden reasoning.
            </>,
            'Raise it. 4096 is a workable floor for a reasoning model.',
          ],
          [
            <code>effgen config validate</code>,
            'exits non-zero',
            'The file has a problem, and the message names it. Fix and re-run — it makes no model call.',
          ],
        ]}
      />

      <SeeAlso paths={['/first-project', '/agents', '/generation']} />
    </DocPage>
  );
}
