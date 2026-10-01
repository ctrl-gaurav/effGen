import { TerminalSquare } from 'lucide-react';
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
import { siteHref } from '../siteLinks';

const commands = siteData.cli.commands;
const subcommands = siteData.cli.subcommands;
const globalOptions = siteData.cli.global_options;

/**
 * Where each command is documented in full.
 *
 * A command with no entry is described here and nowhere else — every command
 * is in the table below either way, so the list on this page is the command
 * surface and not a selection from it.
 */
const DOCUMENTED_AT: Record<string, string> = {
  run: '/cli/run',
  chat: '/cli/run',
  code: '/cli/code',
  top: '/cli/top',
  monitor: '/cli/top',
  report: '/cli/reports',
  runs: '/cli/history',
  sessions: '/cli/history',
  resume: '/checkpointing',
  batch: '/cli/batch',
  workflow: '/workflows',
  serve: '/api-server',
  debug: '/debug',
  cost: '/cost',
  eval: '/evaluation',
  compare: '/compare',
  battle: '/compare',
  loadtest: '/loadtest',
  models: '/models',
  tools: '/tools',
  presets: '/presets',
  prompts: '/prompts',
  config: '/configuration',
  quickstart: '/quickstart',
  tutorial: '/quickstart',
  examples: '/examples',
  'create-plugin': '/custom-tools',
};

export default function Cli() {
  return (
    <DocPage
      subtitle={`All ${siteData.cli.command_count} effgen commands, the options every one of them accepts, and where each is documented in full.`}
      icon={<TerminalSquare size={48} />}
    >
      <p>
        Everything the framework does from Python it also does from a shell. One binary,{' '}
        <code>{siteData.cli.command_count}</code> commands and{' '}
        <code>{siteData.cli.subcommand_count}</code> sub-commands, a consistent set of global
        options, <code>--json</code> on the commands that produce data, and generated shell
        completion that cannot fall behind the parser.
      </p>

      <p className="doc-crosslink">
        This page is the reference: every command, every flag, every exit code. For what the
        command line feels like to use, see <a href={siteHref('/cli')}>the command line page</a> on
        the main site.
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
        caption={
          <>
            <code>--explain</code> adds the trace under the answer; without it the run prints the
            answer and a one-line summary. <Link to="/cli/run">run and chat</Link> covers both.
          </>
        }
      />

      <h2>Naming a model</h2>

      <p>
        Three spellings reach the same place, and the provider-prefixed one is the portable form —
        it carries the provider with it, so a command can be pasted into a script without also
        carrying a <code>--provider</code> flag.
      </p>

      <ApiTable
        headers={['Form', 'Example', 'When']}
        rows={[
          [
            'Provider-prefixed',
            <code>-m openai:gpt-5-nano</code>,
            'Anything you will paste somewhere else. One argument, unambiguous.',
          ],
          [
            'Bare id plus --provider',
            <code>-m gpt-5-nano --provider openai</code>,
            'The same thing in two arguments, when the provider is a variable in your script.',
          ],
          [
            'A local repository id',
            <code>-m Qwen/Qwen2.5-1.5B-Instruct</code>,
            <>
              No provider means local weights. <Link to="/local-models">Local models</Link> covers
              the engines.
            </>,
          ],
        ]}
        caption={
          <>
            The nine adapters that ship a catalog are <code>{siteData.models.with_catalog.join(', ')}</code>.{' '}
            <code>effgen models list</code> prints them with their models.
          </>
        }
      />

      <h2>Options every command accepts</h2>

      <ParamTable
        nameLabel="Global option"
        params={globalOptions.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          <>
            From <code>effgen --help</code>. These sit before the sub-command —{' '}
            <code>effgen --theme light models list</code> — though <code>--theme</code>,{' '}
            <code>-v</code>, <code>-q</code> and <code>--no-animation</code> are also accepted after
            it on the commands that take them.
          </>
        }
      />

      <Callout type="note" title="Two short flags mean different things in different commands">
        <p>
          <code>-p</code> is <code>--port</code> in <code>serve</code>, <code>top</code> and{' '}
          <code>monitor</code>, and <code>--print</code> in <code>code</code>. <code>-c</code> is{' '}
          <code>--concurrency</code> in <code>batch</code> and <code>loadtest</code>, and{' '}
          <code>--config</code> in <code>run</code>. The bindings are fixed and tested; in a script,
          write the long name.
        </p>
      </Callout>

      <h2>Every command</h2>

      <ApiTable
        headers={['Command', 'What it does', 'Documented in']}
        rows={commands.map((command) => {
          const subs = subcommands[command.name];
          const path = DOCUMENTED_AT[command.name];
          return [
            <>
              <code>effgen {command.name}</code>
              {subs && (
                <>
                  <br />
                  <span className="param-mono" style={{ fontSize: '0.85em' }}>
                    {subs.join(' · ')}
                  </span>
                </>
              )}
            </>,
            command.summary,
            path ? <Link to={path}>{path.replace(/^\//, '')}</Link> : 'this page',
          ];
        })}
        caption={
          <>
            From <code>effgen --help</code> and each sub-command's own <code>--help</code>, at{' '}
            {siteData.version}.
          </>
        }
      />

      <h3>The five with no page of their own</h3>

      <ApiTable
        headers={['Command', 'What it prints']}
        rows={[
          [
            <code>effgen health</code>,
            'Reaches the provider endpoints and reports which answer. It makes network calls, so it is the slow check.',
          ],
          [
            <code>effgen doctor</code>,
            <>
              The local picture: which provider keys are present, the Python and platform version,
              the sandbox backend, whether git is available, and what{' '}
              <code>effgen code</code> would use. <code>--live --cheap</code> adds one small call
              per keyed provider to confirm the keys work. It also lists what each served or local
              model was measured to do with a tool, from <code>~/.effgen/capabilities.json</code>.{' '}
              <code>--probe MODEL</code> measures one model now: a local model, or one served at{' '}
              <code>--base-url URL</code>, with <code>--api-key-env VAR</code> naming the variable
              that holds the endpoint's key. <code>--refresh</code> measures it again and replaces
              the stored result, and <code>--json</code> prints the report as JSON.
            </>,
          ],
          [
            <code>effgen monitor</code>,
            <>
              An alias for <code>effgen top</code> — same parser, same flags, same output.
            </>,
          ],
          [
            <code>effgen tutorial</code>,
            <>
              An alias for <code>effgen quickstart</code>.
            </>,
          ],
          [
            <code>effgen resume --checkpoint ID</code>,
            <>
              Restarts an interrupted run from a checkpoint snapshot.{' '}
              <Link to="/checkpointing">Checkpointing</Link> covers it, and note that it is a
              different store from <code>--session-id</code>: a checkpoint is a mid-run snapshot, a
              session is a conversation.
            </>,
          ],
        ]}
      />

      <h2>Machine-readable output</h2>

      <p>
        Every command that produces data takes <code>--json</code>. The rule is the same
        everywhere: <strong>stdout carries the JSON document and nothing else</strong>, and
        everything a person reads goes to stderr. Adding <code>-q</code> silences the human half
        entirely.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "Say the word ok and nothing else." -m gemini:gemini-3.1-flash-lite -q --json \\
  | jq "{success, output, cost: .metadata.cost_usd}"`}
      />

      <Terminal
        command={`effgen run "Say the word ok and nothing else." -m gemini:gemini-3.1-flash-lite -q --json | jq "{success, output, cost: .metadata.cost_usd}"`}
        output={`{
  "success": true,
  "output": "Ok",
  "cost": 3.95e-05
}`}
      />

      <h2>Shell completion</h2>

      <p>
        The completion scripts are generated by introspecting the live parser and the tool and
        preset registries, so the names they offer are the names the binary accepts.
      </p>

      <CodeBlock
        language="bash"
        filename="~/.bashrc"
        code={`# Bash
eval "$(effgen --completion bash)"

# Zsh — in ~/.zshrc
eval "$(effgen --completion zsh)"

# Fish
effgen --completion fish | source`}
      />

      <Terminal
        command="effgen --completion bash | head -12"
        output={`
_effgen_completion() {
    local cur prev commands presets tools
    COMPREPLY=()
    cur="\${COMP_WORDS[COMP_CWORD]}"
    prev="\${COMP_WORDS[COMP_CWORD-1]}"

    commands="batch battle bench chat code compare config cost create-plugin debug doctor eval examples health loadtest models monitor presets prompts quickstart report resume run runs serve sessions tools top tutorial workflow"
    presets="coding general math media minimal multimodal notify rag research"
    tools="agentic_search anthropic_bash anthropic_computer anthropic_text_editor arxiv audio_transcribe bash calculator code_execution code_executor crypto_price currency_converter dataframe datetime discord_webhook docker docx email_draft email_imap email_smtp excel file_operations geocode git github google_search hackernews http image_caption image_info json_tool language_detect maps multimodal_describe news notification ocr openai_code_interpreter openai_file_search openai_web_search pdf plot pubmed python_repl qr_generate qr_read reddit retrieval rss_feed semantic_scholar slack_draft slack_webhook stackoverflow stats stock_price system_info text_processing translate url_context url_fetch weather web_search wikipedia wolfram_alpha youtube_metadata youtube_transcript"`}
        maxLines={14}
        caption={
          <>
            The command list here is the parser's, and the tool list is the registry's — including
            the aliases and the always-on tools that <code>effgen tools list</code> groups
            differently. Nothing in it is hand-maintained.
          </>
        }
      />

      <h2>Where state lives</h2>

      <ApiTable
        headers={['Path', 'What it holds']}
        rows={[
          [
            <code>$EFFGEN_HOME</code>,
            <>
              Everything below, default <code>~/.effgen</code>. Point it somewhere else to keep a
              project's history separate, or to give a container a mounted volume.
            </>,
          ],
          [
            <code>$EFFGEN_HOME/runs</code>,
            <>
              Run history, one JSONL file per day. <Link to="/cli/history">Runs and sessions</Link>.
            </>,
          ],
          [<code>$EFFGEN_HOME/sessions</code>, 'Saved conversations, by session id.'],
          [
            <code>$EFFGEN_HOME/costs.sqlite</code>,
            <>
              The spend ledger every adapter writes to. <Link to="/cost">Cost and budgets</Link>.
            </>,
          ],
          [
            <code>$EFFGEN_HOME/budget.json</code>,
            <>
              The daily and monthly caps <code>effgen cost set-budget</code> writes.
            </>,
          ],
          [
            <code>$EFFGEN_WORKSPACE</code>,
            <>
              Not under <code>EFFGEN_HOME</code>: the directory the file and shell tools read and
              write, and the only one sandboxed code may write to. Unset, it is the current
              directory.
            </>,
          ],
        ]}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>Could not create agent config. Provide -m/--model … or --preset.</code>, exit{' '}
              <code>2</code>
            </>,
            'The command needs a model and neither a flag nor a preset named one.',
            <>
              Pass <code>-m</code> or <code>--preset</code>. For a shell-wide default, export{' '}
              <code>EFFGEN_DEFAULT_MODEL</code> — <code>effgen config set</code> writes only the two
              budget keys.
            </>,
          ],
          [
            <>
              <code>does not exist or you do not have access to it</code> on an id{' '}
              <code>effgen models list</code> shows
            </>,
            'The catalog knows the id; the provider has retired it or your account is not entitled to it. The catalog carries the date it was verified.',
            <>
              <code>effgen models refresh --provider &lt;name&gt;</code>, then pick a current id.
            </>,
          ],
          [
            <>
              <code>effgen: command not found</code>
            </>,
            'Installed into an environment that is not on PATH — common after a `pip install --user`.',
            <>
              <code>python -m effgen.cli</code> runs the same entry point. Then put the
              environment's <code>bin</code> on PATH.
            </>,
          ],
          [
            'A command prints a JSON document mixed into human text',
            <>
              Reading stdout and stderr together. <code>--json</code> keeps the document on stdout
              and the prose on stderr.
            </>,
            <>
              Add <code>-q</code>, or redirect: <code>… --json 2&gt;/dev/null | jq .</code>
            </>,
          ],
          [
            'Colour escapes in a log file',
            'Something forced colour on a non-terminal.',
            <>
              <code>NO_COLOR=1</code> or <code>--no-animation</code>.{' '}
              <Link to="/cli/appearance">Appearance and themes</Link> lists every switch.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>code</code>, <code>top</code>, <code>monitor</code>, <code>battle</code>,{' '}
          <code>report</code>, <code>runs</code>, <code>sessions</code> and <code>resume</code> are
          new commands in this release, and <code>--json</code> now covers every command that
          produces data. The Python floor is {siteData.python_requires}.
        </p>
      </Callout>

      <SeeAlso paths={['/cli/run', '/cli/code', '/cli/appearance']} />
    </DocPage>
  );
}
