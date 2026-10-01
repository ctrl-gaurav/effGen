import { FolderPlus } from 'lucide-react';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { version } from '../siteData';

export default function FirstProject() {
  return (
    <DocPage
      subtitle="What effgen quickstart --init writes into an empty directory, and how to run it."
      icon={<FolderPlus size={48} />}
    >
      <p>
        A script is fine for one question. For anything you will come back to,{' '}
        <code>effgen quickstart --init</code> turns an empty directory into a project that runs: a
        configuration <code>effgen run</code> reads, an <code>.env</code> template naming every
        provider variable, one runnable example, and a daily spend cap. It makes no model call and
        asks no questions, so it works before any key is set, in a pipe, and in CI.
      </p>

      <h2>Scaffolding one</h2>

      <CodeBlock
        language="bash"
        code={`effgen quickstart --init my-agent -m openai:gpt-5-nano`}
      />

      <Terminal
        command="effgen quickstart --init my-agent -m openai:gpt-5-nano"
        output={`effGen project
Directory: my-agent

  wrote    effgen.yaml   model, prompt and per-run caps
  wrote    .env.example  key names, no values
  wrote    example.py    a runnable agent script
  wrote    .gitignore    keeps .env out of git

Model:      openai:gpt-5-nano
            given with -m/--model
            change it on the 'model:' line of effgen.yaml

Spend cap:  $5.00 a day across all runs (already configured)
            'effgen cost set-budget N' changes it
            'effgen cost clear-budget' removes it

Next three commands
  1. cp .env.example .env  # then paste one key into it
  2. effgen doctor  # confirm effGen sees it
  3. effgen run "What is 25 * 17?" -c effgen.yaml`}
        caption={`Run against effGen 1.0.0. Without -m it detects a model from the keys it can see, and says which and why.`}
      />

      <p>
        <code>--init</code> takes an optional directory, created if it does not exist. With no
        argument it scaffolds into the current directory.
      </p>

      <h2>What it writes</h2>

      <ApiTable
        headers={['File', 'What it is']}
        rows={[
          [
            <code>effgen.yaml</code>,
            <>
              The agent configuration. <code>effgen run -c effgen.yaml</code> reads it.
            </>,
          ],
          [
            <code>.env.example</code>,
            'Every provider variable, named, with no value. Copy it to .env and fill one in.',
          ],
          [
            <code>example.py</code>,
            <>
              The same agent as a Python script — <code>python example.py</code>.
            </>,
          ],
          [
            <code>.gitignore</code>,
            <>
              Excludes <code>.env</code>, <code>.effgen/</code> and the usual Python artefacts.
            </>,
          ],
        ]}
        caption="Nothing else is written inside the directory. The one file written outside it is the daily spend cap, in the same ~/.effgen/budget.json that effgen cost set-budget writes."
      />

      <h3>effgen.yaml</h3>

      <CodeBlock
        language="yaml"
        filename="my-agent/effgen.yaml"
        code={`# effGen project configuration.
#
# Used by:  effgen run "your task" -c effgen.yaml
#
# Next three commands:
#   1. cp .env.example .env  # then paste one key into it
#   2. effgen doctor  # confirm effGen sees it
#   3. effgen run "What is 25 * 17?" -c effgen.yaml
#
# max_tokens caps what one answer may cost; max_iterations caps how many steps
# the agent may take on one task. Both apply to every run that loads this file.
# A daily spend cap across all runs is separate: effgen cost set-budget 1.00
#
# openai:gpt-5-nano spends part of its output budget on hidden reasoning,
# so max_tokens is 4096 here rather than the usual 512. A smaller
# budget can run out before the first visible word, and that empty
# answer is still billed.

model: openai:gpt-5-nano
system_prompt: You are a helpful assistant. Answer concisely and say when you are unsure.
temperature: 0.2
max_tokens: 4096
max_iterations: 5`}
        caption="Written verbatim by the command above. The reasoning-model paragraph appears only when the model it picked is one."
      />

      <p>
        The file carries only the keys a run applies — <code>model</code>, <code>provider</code>,{' '}
        <code>system_prompt</code>, <code>temperature</code>, <code>max_tokens</code>,{' '}
        <code>max_iterations</code> and <code>guardrails</code>. Any other key that names an{' '}
        <code>AgentConfig</code> field is reported when the file loads, so a setting can never be a
        silent no-op. The model id is written provider-prefixed, because a file travels without a{' '}
        <code>--provider</code> flag beside it.
      </p>

      <Callout type="tip" title="max_tokens follows the model">
        <p>
          A reasoning model spends part of its output budget on hidden reasoning before the first
          visible token, so the scaffold writes 4096 rather than the usual 512 — and says so in the
          file. A smaller budget can run out before a word is emitted, and that empty answer is
          still billed.
        </p>
      </Callout>

      <h3>.env.example</h3>

      <CodeBlock
        language="bash"
        filename="my-agent/.env.example"
        code={`# effGen provider keys.
#
#   cp .env.example .env
#   $EDITOR .env          # paste your key after the '=' of one line
#   effgen doctor         # confirms which keys effGen can see
#
# effGen loads the nearest .env automatically, walking up from the
# directory you run in. A variable already set in the environment wins
# over this file, and an empty value counts as no key at all. Set
# EFFGEN_NO_DOTENV=1 to skip the file search entirely.
#
# .env is listed in .gitignore — keep it that way.

# anthropic
ANTHROPIC_API_KEY=

# cerebras
CEREBRAS_API_KEY=

# fireworks
FIREWORKS_API_KEY=`}
        caption="The head of the file. Every registered provider's variable is named and left empty; no value is invented."
      />

      <h3>example.py</h3>

      <CodeBlock
        filename="my-agent/example.py"
        code={`"""One effGen agent, one question, one answer — with what it cost.

Run it from this directory:

    python example.py
    python example.py "your own question"

The model and the caps match effgen.yaml, so this script and
\`effgen run -c effgen.yaml\` do the same thing.
"""

from __future__ import annotations

import sys

from effgen import Agent, AgentConfig, load_env

# Keep in step with the \`model:\` line in effgen.yaml.
MODEL = "openai:gpt-5-nano"


def main() -> int:
    # Reads the .env next to this file (and any parent), so a key pasted there
    # is picked up without exporting anything.
    load_env()

    task = " ".join(sys.argv[1:]) or "What is 25 * 17?"
    agent = Agent(AgentConfig(
        name="example",
        model=MODEL,
        system_prompt="You are a helpful assistant. Answer concisely.",
        temperature=0.2,
        max_tokens=4096,
        max_iterations=5,
    ))
    try:
        response = agent.run(task)
    finally:
        agent.close()

    print(response.text)
    if not response.success:
        # A failure carries its own message in the text above; the exit code is
        # what a script downstream can act on.
        return 1

    metadata = response.metadata or {}
    cost = metadata.get("cost_usd")
    tokens = metadata.get("total_tokens")
    if cost is not None:
        print(f"\\ncost: \${cost:.6f}  tokens: {tokens}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())`}
        caption="Written verbatim by the scaffold. The cost line goes to stderr, so piping the script gives you the answer alone."
      />

      <Terminal
        command={'python example.py "What is 25 * 17?"'}
        output={`cost: $0.000031  tokens: 99
425`}
        caption="stdout and stderr interleaved, which is what a terminal shows."
      />

      <h2>The three commands it hands you</h2>

      <CodeBlock
        language="bash"
        code={`cp .env.example .env      # then paste one key into it
effgen doctor             # confirm effGen sees it
effgen run "What is 25 * 17?" -c effgen.yaml`}
      />

      <p>
        <code>-m/--model</code> on the command line still wins over the <code>model:</code> line,
        so one file can be run against several models without editing it:
      </p>

      <CodeBlock
        language="bash"
        code={`effgen run "What is 25 * 17?" -c effgen.yaml
effgen run "What is 25 * 17?" -c effgen.yaml -m gemini:gemini-3.1-flash-lite`}
      />

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={[
          {
            name: '--init [DIR]',
            type: 'path',
            description:
              'Scaffold in DIR, which is created if it does not exist. Defaults to the current directory. Prompts for nothing, calls no model, and skips every other quickstart step.',
          },
          {
            name: '--force',
            type: 'flag',
            description: 'Overwrite a scaffolded file that is already there. Without it, existing files are kept.',
          },
          {
            name: '--budget USD',
            type: 'float',
            default: '1.00',
            description: 'The daily spend cap to set when none is configured. 0 sets none. An existing cap is never changed.',
          },
          {
            name: '-m MODEL, --model MODEL',
            type: 'str',
            description: 'Write this model id instead of the one detected from your keys.',
          },
          {
            name: '--provider PROVIDER',
            type: 'str',
            description: 'The provider for a bare model id.',
          },
        ]}
        caption={<><code>effgen quickstart --help</code>, {version}. The remaining flags belong to the guided first run rather than to <code>--init</code>.</>}
      />

      <h2>Running it again</h2>
      <p>An existing file is kept, not replaced.</p>

      <Terminal
        command="effgen quickstart --init reasoning -m gemini:gemini-3.1-flash-lite"
        output={`effGen project
Directory: reasoning

  kept     effgen.yaml   already there
  kept     .env.example  already there
  kept     example.py    already there
  kept     .gitignore    already there

Existing files were left as they are; --force replaces them.

Model:      gemini:gemini-3.1-flash-lite
            given with -m/--model
            change it on the 'model:' line of effgen.yaml

Spend cap:  $5.00 a day across all runs (already configured)
            'effgen cost set-budget N' changes it
            'effgen cost clear-budget' removes it

Next three commands
  1. cp .env.example .env  # then paste one key into it
  2. effgen doctor  # confirm effGen sees it
  3. effgen run "What is 25 * 17?" -c effgen.yaml`}
        caption="Every file was already there, so every one was kept — and the command says so rather than silently doing nothing."
      />

      <CodeBlock
        language="bash"
        code={`effgen quickstart --init my-agent --force   # rewrite them instead`}
      />

      <h2>Two independent caps</h2>

      <ApiTable
        headers={['Cap', 'Where it lives', 'What it bounds']}
        rows={[
          [
            'Per run',
            <code>effgen.yaml</code>,
            <>
              <code>max_tokens</code> bounds one answer; <code>max_iterations</code> bounds how
              many steps the agent may take on one task.
            </>,
          ],
          [
            'Per day, across every run',
            <code>~/.effgen/budget.json</code>,
            'A spend cap the scaffold sets at $1.00 a day when you have none. It never changes one you already have.',
          ],
        ]}
      />

      <CodeBlock
        language="bash"
        code={`effgen quickstart --init my-agent --budget 5   # a different cap, when none is set
effgen quickstart --init my-agent --budget 0   # set no cap
effgen cost set-budget 2.50                    # change it later
effgen cost clear-budget                       # remove it
effgen cost today                              # what has been spent against it`}
      />

      <h2>Just the configuration</h2>
      <p>
        <code>effgen config init</code> writes the same document on its own, without the{' '}
        <code>.env</code> template, the example or the spend cap.
      </p>

      <CodeBlock
        language="bash"
        code={`effgen config init                 # config.yaml
effgen config init -o agent.yaml   # somewhere else
effgen config validate --file agent.yaml`}
      />

      <h2>If something is missing</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'The model it chose is not the one you want',
            'With no -m it picks from the keys it can see, and with no keys at all it picks a local model.',
            <>
              Pass <code>-m</code>, or edit the <code>model:</code> line. The file says which it
              chose and why.
            </>,
          ],
          [
            'Nothing was written',
            'The files were already there.',
            <>
              It reports each one as kept. <code>--force</code> replaces them.
            </>,
          ],
          [
            <>
              <code>effgen doctor</code> still reports no key
            </>,
            <>
              <code>.env.example</code> was copied but not filled in, or an empty assignment was
              left in place.
            </>,
            <>
              An empty value counts as no key. Paste a key after the <code>=</code>, and remember a
              variable already exported in your shell wins over the file.
            </>,
          ],
          [
            'A spend cap you did not expect',
            'One was already configured, and the scaffold left it alone.',
            <>
              <code>effgen cost set-budget</code> changes it, <code>clear-budget</code> removes it.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/quickstart', '/configuration', '/cost']} />
    </DocPage>
  );
}
