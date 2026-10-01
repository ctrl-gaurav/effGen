import { PenLine } from 'lucide-react';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function PromptAuthoring() {
  return (
    <DocPage
      subtitle="Writing your own template, and trying one against a model before you ship it."
      icon={<PenLine size={48} />}
    >
      <p>
        Your own templates do not go in the installed package. Point{' '}
        <code>EFFGEN_PROMPTS_DIR</code> at a directory of Python files and every{' '}
        <code>effgen prompts</code> command finds them beside the built-in library — and the
        playground gives you somewhere to set the variables and see the answer before you commit
        to one.
      </p>

      <h2>A template in one file</h2>

      <CodeBlock filename="external.sh" language="bash" code={`DIR="$(mktemp -d)"
cat > "$DIR/marketing.py" <<'PYFILE'
from effgen.prompts.library import LibraryPrompt


def _render(topic, audience, tone="plain", **_):
    return f"Explain {topic} to {audience} in a {tone} tone. Use one analogy."


PROMPTS = [
    LibraryPrompt(
        name="marketing.explainer.v1",
        domain="marketing",
        variant="zero_shot",
        description="Explain a topic to an audience in a chosen tone.",
        template=_render,
        input_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "minLength": 2},
                "audience": {"type": "string", "minLength": 2},
                "tone": {"type": "string"},
            },
            "required": ["topic", "audience"],
        },
        fixture={"topic": "vector databases", "audience": "product managers"},
        expected_shape=None,
        tags=["marketing"],
    )
]
PYFILE

export EFFGEN_PROMPTS_DIR="$DIR"
effgen prompts list --domain marketing
effgen prompts render marketing.explainer.v1`} />

      <Terminal
        command="bash external.sh"
        output={`                                 Prompt Library                                 
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name                   ┃ Domain    ┃ Variant   ┃ Description                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ marketing.explainer.v1 │ marketing │ zero_shot │ Explain a topic to an       │
│                        │           │           │ audience in a chosen tone.  │
└────────────────────────┴───────────┴───────────┴─────────────────────────────┘

Total: 1 prompt(s)
╭────────────────────────────── Rendered Prompt ───────────────────────────────╮
│ Explain vector databases to product managers in a plain tone. Use one        │
│ analogy.                                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯`}
        caption={`Run against effGen 1.0.0. The template was discovered from the directory, listed and rendered, with nothing installed.`}
      />

      <p>
        Each <code>*.py</code> file in the directory is loaded; files whose name starts with{' '}
        <code>_</code> are skipped. Separate several directories with your platform's path
        separator — <code>:</code> on Linux and macOS, <code>;</code> on Windows.
      </p>

      <ApiTable
        headers={['How a file offers its templates', 'What it looks like']}
        rows={[
          [
            'A module-level list',
            <>
              <code>PROMPTS = [LibraryPrompt(...), ...]</code> — the form above.
            </>,
          ],
          [
            'Registering directly',
            <>
              <code>registry.register(LibraryPrompt(...))</code> at import time.
            </>,
          ],
        ]}
      />

      <h3>In one process, without a file</h3>

      <CodeBlock filename="register.py" code={`from effgen.prompts.library import LibraryPrompt, registry


def _render(topic, audience, tone="plain", **_):
    return f"Explain {topic} to {audience} in a {tone} tone. Use one analogy."


registry.register(LibraryPrompt(
    name="marketing.explainer.v1",
    domain="marketing",
    variant="zero_shot",
    description="Explain a topic to an audience in a chosen tone.",
    template=_render,
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "minLength": 2},
            "audience": {"type": "string", "minLength": 2},
            "tone": {"type": "string"},
        },
        "required": ["topic", "audience"],
    },
    fixture={"topic": "vector databases", "audience": "product managers"},
    expected_shape=None,
    tags=["marketing"],
))

print(registry.get("marketing.explainer.v1").render(**registry.get("marketing.explainer.v1").fixture))`} />

      <Terminal
        command="python register.py"
        output={`Explain vector databases to product managers in a plain tone. Use one analogy.`}
        caption="Visible to this process only. For a template the CLI and the playground can see too, use a directory."
      />

      <h2>What a template declares</h2>

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'name',
            type: 'str',
            required: true,
            description: 'The dotted name — domain.purpose.version. Put the variant on the end when one template has several.',
          },
          { name: 'domain', type: 'str', required: true, description: 'Its domain. A new name creates a new domain; nothing has to be registered first.' },
          {
            name: 'variant',
            type: 'str',
            required: true,
            description: 'zero_shot, cot, few_shot, tool or structured.',
          },
          { name: 'description', type: 'str', required: true, description: 'What it does and what it takes. This is what `prompts list` prints.' },
          {
            name: 'template',
            type: 'Callable[..., str]',
            required: true,
            description: 'The render function. Give it **_ so an unexpected input does not raise.',
          },
          {
            name: 'input_schema',
            type: 'dict',
            required: true,
            description: 'A JSON schema for the inputs. `validate_input` and `prompts render --input` both check against it.',
          },
          {
            name: 'fixture',
            type: 'dict',
            required: true,
            description: 'A worked set of inputs. `prompts render` with no input file uses it, and the eval harness renders it.',
          },
          {
            name: 'expected_shape',
            type: 'dict | None',
            default: 'None',
            description: 'How the output is checked in a live eval. None means it is not checked.',
          },
          { name: 'tags', type: 'list[str]', default: '[]', description: 'Labels for search.' },
        ]}
        caption={<><code>effgen.prompts.library.LibraryPrompt</code></>}
      />

      <h3>Checking the output</h3>

      <CodeBlock
        filename="shapes.py"
        code={`# A JSON schema — parsed, then checked for the keys it requires.
expected_shape = {
    "type": "json",
    "schema": {"required": ["summary", "key_points"]},
}

# A pattern the output has to match.
expected_shape = {
    "type": "regex",
    "pattern": r"(?i)introduction|abstract",
}

# A function returning True, or a string saying what was wrong.
expected_shape = {
    "type": "callable",
    "fn": lambda output: len(output.split()) >= 50 or "output too short",
}`}
        caption="Three fragments, one field. Only a live eval reads it; a golden eval compares the rendering instead."
      />

      <Callout type="tip" title="Give the render function **_">
        <p>
          The playground seeds variables from the fixture and lets you set more, so a render
          function that raises on an unexpected keyword is awkward to work with. Every bundled
          template takes <code>**_</code> for that reason.
        </p>
      </Callout>

      <h2>The playground</h2>
      <p>
        An interactive session: pick a template, set variables, render, run it against a model, and
        save what you did.
      </p>

      <CodeBlock language="bash" code={`effgen prompts playground`} />

      <ApiTable
        headers={['Command', 'What it does']}
        rows={[
          [<code>select &lt;name&gt;</code>, 'Pick a template. Its variables are seeded from the fixture.'],
          [
            <code>set &lt;key&gt; &lt;value&gt;</code>,
            'Bind a variable. The value is JSON-decoded when it parses, and treated as a plain string when it does not.',
          ],
          [<code>unset &lt;key&gt;</code>, 'Remove a binding.'],
          [<code>render</code>, 'Print the rendered prompt with the current bindings.'],
          [<code>run [--model &lt;id&gt;]</code>, 'Render, send to a model, and keep the output in the session.'],
          [
            <code>save [&lt;path&gt;]</code>,
            <>
              Write the session to JSON. Without a path it names itself under{' '}
              <code>~/.effgen/playground/</code>.
            </>,
          ],
          [<code>load &lt;path&gt;</code>, 'Restore a saved session.'],
          [<code>reload</code>, "Re-import the selected template's module, so an edit shows up without a restart."],
          [<code>list [--domain &lt;d&gt;]</code>, 'List registered templates.'],
          [<code>show &lt;name&gt;</code>, 'Its schema, fixture and a rendered preview.'],
          [<code>help</code>, 'The command reference.'],
          [<>
            <code>exit</code>, <code>quit</code>, Ctrl-D
          </>, 'Leave.'],
        ]}
      />

      <Callout type="note" title="reload is the reason to use it">
        <p>
          Editing a template file and typing <code>reload</code> re-imports the module and
          refreshes the registry, so the loop from a change to seeing its effect on a real model is
          two keystrokes rather than a restart.
        </p>
      </Callout>

      <h3>Without the REPL</h3>

      <ApiTable
        headers={['Command', 'What it does']}
        rows={[
          [
            <code>effgen prompts render &lt;name&gt;</code>,
            <>
              Print the rendered prompt. <code>-i/--input FILE</code> supplies variables as JSON,
              merged over the fixture and validated against the schema.
            </>,
          ],
          [
            <code>effgen prompts run &lt;name&gt; -m &lt;model&gt;</code>,
            <>
              Render and send in one step. <code>--max-tokens</code> and{' '}
              <code>--temperature</code> apply to that run.
            </>,
          ],
        ]}
      />

      <Terminal command="effgen prompts render coding.docstring_fill.v1" output={`╭────────────────────────────── Rendered Prompt ───────────────────────────────╮
│ You are an expert Python technical writer.                                   │
│                                                                              │
│ Add Google-style docstrings to every function and class in the code below    │
│ that is currently missing one.                                               │
│                                                                              │
│ Use this docstring format:                                                   │
│ """Short one-line summary.                                                   │
│                                                                              │
│ Args:                                                                        │
│     param_name (type): Description.                                          │
│                                                                              │
│ Returns:                                                                     │
│     type: Description.                                                       │
│                                                                              │
│ Raises:                                                                      │
│     ExceptionType: When and why it is raised.                                │
│ """                                                                          │
│                                                                              │
│ Rules:                                                                       │
│   - Infer parameter types and return types from usage patterns and names.    │
│   - Include an 'Args' (or equivalent) section only if the function has       │
│ parameters.                                                                  │
│   - Include a 'Returns' section only if the function returns a non-None      │
│ value.                                                                       │
│   - Include a 'Raises' section only if exceptions are explicitly raised.     │
│   - Preserve all existing code exactly — only add docstrings, change nothing │
│ else.                                                                        │
│   - Output the full updated code, no markdown fences, no extra commentary.   │
│                                                                              │
│ Code:                                                                        │
│ \`\`\`python                                                                    │
│ def merge_sorted(a, b):                                                      │
│     result = []                                                              │
│     i = j = 0                                                                │
│     while i < len(a) and j < len(b):                                         │
│         if a[i] <= b[j]:                                                     │
│             result.append(a[i])                                              │
│             i += 1                                                           │
│         else:                                                                │
│             result.append(b[j])                                              │
│             j += 1                                                           │
│     result.extend(a[i:])                                                     │
│     result.extend(b[j:])                                                     │
│     return result                                                            │
│                                                                              │
│ \`\`\`                                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯`} maxLines={20} />

      <Terminal
        command="effgen prompts run coding.docstring_fill.v1 -m openai:gpt-5-nano --max-tokens 4000"
        output={`╭────────────────────────────── Rendered Prompt ───────────────────────────────╮
│ You are an expert Python technical writer.                                   │
│                                                                              │
│ Add Google-style docstrings to every function and class in the code below    │
│ that is currently missing one.                                               │
│                                                                              │
│ Use this docstring format:                                                   │
│ """Short one-line summary.                                                   │
│                                                                              │
│ Args:                                                                        │
│     param_name (type): Description.                                          │
│                                                                              │
│ Returns:                                                                     │
│     type: Description.                                                       │
│                                                                              │
│ Raises:                                                                      │
│     ExceptionType: When and why it is raised.                                │
│ """                                                                          │
│                                                                              │
│ Rules:                                                                       │
│   - Infer parameter types and return types from usage patterns and names.    │
│   - Include an 'Args' (or equivalent) section only if the function has       │
│ parameters.                                                                  │
│   - Include a 'Returns' section only if the function returns a non-None      │
│ value.                                                                       │
│   - Include a 'Raises' section only if exceptions are explicitly raised.     │
│   - Preserve all existing code exactly — only add docstrings, change nothing │
│ else.                                                                        │
│   - Output the full updated code, no markdown fences, no extra commentary.   │
│                                                                              │
│ Code:                                                                        │
│ \`\`\`python                                                                    │
│ def merge_sorted(a, b):                                                      │
│     result = []                                                              │
│     i = j = 0                                                                │
│     while i < len(a) and j < len(b):                                         │
│         if a[i] <= b[j]:                                                     │
│             result.append(a[i])                                              │
│             i += 1                                                           │
│         else:                                                                │
│             result.append(b[j])                                              │
│             j += 1                                                           │
│     result.extend(a[i:])                                                     │
│     result.extend(b[j:])                                                     │
│     return result                                                            │
│                                                                              │
│ \`\`\`                                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── Model Output (openai:gpt-5-nano) ──────────────────────╮
│ def merge_sorted(a, b):                                                      │
│     """Merge two sorted lists into a single sorted list.                     │
│                                                                              │
│     This function assumes that input lists a and b are sorted in             │
│ non-decreasing order                                                         │
│     and returns a new list containing all elements from both lists in sorted │
│ order.                                                                       │
│                                                                              │
│     Args:                                                                    │
│         a (list): First sorted input list.                                   │
│         b (list): Second sorted input list.                                  │
│                                                                              │
│     Returns:                                                                 │
│         list: Merged list containing all elements from a and b in            │
│ non-decreasing order.                                                        │
│     """                                                                      │
│                                                                              │
│     result = []                                                              │
│     i = j = 0                                                                │
│     while i < len(a) and j < len(b):                                         │
│         if a[i] <= b[j]:                                                     │
│             result.append(a[i])                                              │
│             i += 1                                                           │
│         else:                                                                │
│             result.append(b[j])                                              │
│             j += 1                                                           │
│     result.extend(a[i:])                                                     │
│     result.extend(b[j:])                                                     │
│     return result                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯
tokens: 1542 (269 in / 1273 out)  ·  cost: $0.000523  ·  latency: 7287 ms
✓ output matches expected_shape`}
        maxLines={26}
      />

      <h2>Session files</h2>
      <p>
        A playground session is plain JSON — the template, the variables, every render and every
        run with its model and its output — so it can be read back in a script or checked into a
        repository beside the template it was used to develop.
      </p>

      <CodeBlock filename="session.py" code={`from pathlib import Path

from effgen.prompts.library.session import PlaygroundSession

session = PlaygroundSession(prompt_name="coding.docstring_fill.v1")
session.variables["style"] = "google"
saved = session.save(Path("/tmp/effgen-playground.json"))

restored = PlaygroundSession.load(saved)
print(saved)
print(restored.prompt_name, restored.variables)`} />

      <Terminal command="python session.py" output={`/tmp/effgen-playground.json
coding.docstring_fill.v1 {'style': 'google'}`} />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'prompt_name', type: 'str', default: "''", description: 'The template the session is about.' },
          { name: 'variables', type: 'dict[str, Any]', default: '{}', description: 'The current bindings.' },
          { name: 'render_history', type: 'list[str]', default: '[]', description: 'Every rendering, in order.' },
          {
            name: 'run_history',
            type: 'list[RunEntry]',
            default: '[]',
            description: 'Every run — the model, the rendered prompt, the output and a timestamp.',
          },
          { name: 'created_at', type: 'str', description: 'When the session started.' },
          { name: 'updated_at', type: 'str', description: 'When it was last written.' },
        ]}
        caption={
          <>
            <code>effgen.prompts.library.session.PlaygroundSession</code>.{' '}
            <code>save(path)</code> takes a <code>Path</code> and returns where it wrote;{' '}
            <code>load(path)</code> is a class method.
          </>
        }
      />

      <h2>Contributing a domain to the library</h2>
      <p>
        For a template that should ship with the framework rather than sit in your own directory:
      </p>

      <ApiTable
        headers={['Step', 'What to do']}
        rows={[
          ['1', <>Create <code>effgen/prompts/library/domains/&lt;domain&gt;/__init__.py</code>.</>],
          ['2', <>One file per template, e.g. <code>my_prompt_v1.py</code>.</>],
          ['3', <>In each, build a <code>LibraryPrompt</code> and call <code>registry.register(prompt)</code>.</>],
          ['4', <>Add a fixture under <code>tests/prompts/fixtures/&lt;domain&gt;/</code>.</>],
          ['5', <>Run <code>effgen prompts eval</code> to generate the golden file.</>],
          ['6', <>Add tests in <code>tests/prompts/test_&lt;domain&gt;.py</code>.</>],
        ]}
        caption="The registry auto-discovers packages under domains/ at startup, so nothing has to be listed anywhere."
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Your template is not in `effgen prompts list`',
            <>
              <code>EFFGEN_PROMPTS_DIR</code> is unset in that shell, names the wrong directory, or
              the file starts with <code>_</code>.
            </>,
            <>
              It is read at command time. Export it in the shell you run{' '}
              <code>effgen</code> from.
            </>,
          ],
          [
            'A file in the directory is silently ignored',
            'It raised while being imported, or it exposes neither PROMPTS nor a register call.',
            <>
              Import it yourself with <code>python -c "import my_file"</code> to see the error.
            </>,
          ],
          [
            <><code>TypeError</code> from the render function</>,
            'The playground passed a variable the function does not take.',
            <>
              End the signature with <code>**_</code>, as every bundled template does.
            </>,
          ],
          [
            <><code>AttributeError: 'str' object has no attribute 'parent'</code></>,
            <>
              <code>save()</code> was given a string.
            </>,
            <>
              It takes a <code>Path</code>, or nothing at all — with no argument it names the file
              itself.
            </>,
          ],
          [
            '`render --input` rejects your JSON',
            'It did not validate against the template’s input_schema.',
            <>
              <code>effgen prompts show &lt;name&gt;</code> prints the schema. The message names
              the property that failed.
            </>,
          ],
          [
            'An empty answer from `prompts run`',
            'A reasoning model spent its whole completion budget before writing anything.',
            <>
              Raise <code>--max-tokens</code>. The message says how many reasoning tokens were
              produced and that no answer was.
            </>,
          ],
          [
            '`reload` does not pick up your edit',
            'The template was registered from a module the reload cannot re-import.',
            <>
              Reload works on the selected template's own module. Restart the playground if it was
              registered in-process.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/prompts', '/prompts/gallery', '/playground']} />
    </DocPage>
  );
}
