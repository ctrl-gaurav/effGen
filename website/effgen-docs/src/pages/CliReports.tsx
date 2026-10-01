import { FileText } from 'lucide-react';
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

export default function CliReports() {
  return (
    <DocPage
      subtitle="Turning a run, a bake-off or a month of spend into one HTML file that opens with the network off."
      icon={<FileText size={48} />}
    >
      <p>
        Two flags and one command. <code>--report out.html</code> renders what a command just
        measured; <code>--card out.html</code> renders one <code>effgen run</code> in full; and{' '}
        <code>effgen report</code> renders a JSON result you saved earlier, without running any
        model again. Every file is self-contained — style, script and charts inline — so it opens
        from disk or from an email attachment with no network access.
      </p>

      <h2>One run, one file</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "What is 18723 * 4409? Use the calculator tool." \\
  -m openai:gpt-5-nano -t calculator --card run.html`}
      />

      <Terminal
        command={`effgen run "What is 18723 * 4409? Use the calculator tool." -m openai:gpt-5-nano -t calculator --card run.html -q`}
        output={`
Response
╭───────────────────────────────────────── Agent Response ─────────────────────────────────────────╮
│ 82549707                                                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
✓ Run card written to run.html`}
      />

      <Terminal
        command={`grep -oE "https?://[^\\" ]+" run.html | sort -u | wc -l; grep -c "<script" run.html; grep -c "<style" run.html`}
        output={`0
1
1`}
        caption="Zero references to another host, one inline script, one inline style block — the property that makes the file safe to attach and readable on a laptop with the wifi off."
      />

      <h3>What the card carries</h3>

      <ApiTable
        headers={['Section', 'Contents']}
        rows={[
          ['Header', 'The task, the model and provider that answered it, and a succeeded/failed badge.'],
          ['Answer', 'The full answer, rendered as markdown.'],
          [
            'Tool trace',
            'Every step, with its input, its result or its typed failure, and its own duration.',
          ],
          ['Sources and citations', "The run's sources, and the quoted passages behind them."],
          ['Measurements', 'Tokens, cost and latency.'],
          [
            'A failed run',
            <>
              The typed error in place of an answer — type, category, provider, model, message, and
              whether it was retryable. <Link to="/errors">Errors and exceptions</Link>.
            </>,
          ],
          ['Buttons', <>Copy the task, or copy the equivalent <code>effgen run</code> command.</>],
        ]}
        caption={
          <>
            Links to a run's own sources are limited to <code>http</code> and <code>https</code>; a
            source with any other scheme is rendered as inert text rather than as a link.
          </>
        }
      />

      <p>
        A run on local hardware reads <code>unpriced (local)</code>, and a hosted model the catalog
        has no rate for reads <code>unpriced (no published rate)</code> — never <code>$0.00</code>,
        which means a genuine free tier. <code>--card</code> is additive: the terminal output and{' '}
        <code>--json</code> are unchanged, and it composes with <code>-o</code>.
      </p>

      <h3>A card for a run already in history</h3>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen runs show 34fa69a237d8 --card summary.html`}
      />

      <Terminal
        command="effgen runs show 34fa69a237d8 --card summary.html"
        output={`Summary card written to summary.html
Run:      34fa69a237d8
When:     2026-08-24T18:59:25-04:00
Status:   ok
Model:    openai:gpt-5-nano (openai)
Agent:    cli-agent
Tokens:   219 in / 153 out
Cost:     $0.000072
Duration: 1.36s`}
        caption={
          <>
            <Link to="/cli/history">History</Link> keeps a truncated answer and no step trace, so
            that card says on its face that it is a summary. Use <code>effgen run --card</code> at
            run time for the full answer, the trace and the sources.
          </>
        }
      />

      <h2>Reports from the commands that measure things</h2>

      <p>
        <code>compare</code>, <code>battle</code>, <code>eval</code>, <code>cost</code> and{' '}
        <code>loadtest</code> each take <code>--report out.html</code>: a headline verdict, the
        tables the terminal printed, and inline charts.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen compare --models "openai:gpt-5-nano,gemini:gemini-3.1-flash-lite" \\
  --suite math --optimize cost --report bakeoff.html

effgen eval --suite math -m openai:gpt-5-nano --max-cases 3 --report eval-report.html
effgen cost today --report spend.html
effgen loadtest --duration 30 --concurrency 10 --report capacity.html`}
      />

      <Terminal
        command="effgen eval --suite math -m openai:gpt-5-nano --max-cases 3 --report eval-report.html"
        output={`
HTML report written to eval-report.html

  Exit gate: PASS — accuracy 100.0% >= --fail-under 50%`}
        caption={
          <>
            <code>--report</code> is additive: the terminal output, <code>--json</code> and{' '}
            <code>-o</code> are all unchanged, and the exit gate still decides the exit code.
          </>
        }
      />

      <p>
        The page follows the reader's light/dark system preference and carries a toggle. Its header
        stamps the generation time, the effGen version and the command that produced the result, so
        a file someone forwards can be traced back and re-run.
      </p>

      <h2>Rendering a result you saved earlier</h2>

      <p>
        <code>effgen report</code> turns a JSON result captured at some point in the past into the
        same HTML file, without calling a model. The shape is inferred from the document.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen eval --suite math -m openai:gpt-5-nano --json > eval.json
effgen report eval.json                 # writes eval.html
effgen report eval.json -o for-team.html
effgen run "..." -o run.json && effgen report run.json`}
      />

      <Terminal
        command={`effgen run "Name one prime number under 10. Answer with the number only." -m openai:gpt-5-nano -o saved-run.json -q && effgen report saved-run.json`}
        output={`
Response
╭───────────────────────────────────────── Agent Response ─────────────────────────────────────────╮
│ 7                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
✓ Response saved to saved-run.json
HTML report written to saved-run.html (run)`}
        caption={
          <>
            The kind it chose is named in the last word — <code>(run)</code>. With no{' '}
            <code>-o</code> the output path is the result path with an <code>.html</code> extension.
          </>
        }
      />

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'result',
            required: true,
            description: 'Path to a JSON result saved from run/compare/eval/cost/loadtest',
          },
          {
            name: '-o PATH.html, --output PATH.html',
            description: 'Where to write the HTML report (default: the result path with an .html extension)',
          },
          {
            name: '--kind {run,comparison,eval,cost,loadtest,battle}',
            description: 'Report shape to render, when the JSON cannot be identified',
          },
        ]}
        caption={
          <>
            Every argument <code>effgen report --help</code> declares.
          </>
        }
      />

      <h3>It refuses rather than writing a blank page</h3>

      <Terminal
        command={`echo '{"hello":"world"}' > notaresult.json && effgen report notaresult.json`}
        output={`✗ Could not tell which report this result is. Expected the JSON emitted by \`effgen run --json\`,
\`effgen compare --json\`, \`effgen eval --json\`, \`effgen cost --json\`, or \`effgen loadtest\`. Pass
--kind to say explicitly (run, comparison, eval, cost, loadtest, battle).`}
        caption="Exit 2, and no file is written. A missing path is the same exit code with a different message."
      />

      <Terminal
        command="effgen report /nonexistent.json"
        output={`✗ No such result file: /nonexistent.json. Check the path, or produce one with \`effgen eval --suite
math --json > results.json\`.`}
        caption="Exit 2."
      />

      <h2>Which artefact for which job</h2>

      <ApiTable
        headers={['You want', 'Use']}
        rows={[
          [
            'To show one run — the answer, every tool step, the sources',
            <>
              <code>effgen run … --card run.html</code>
            </>,
          ],
          [
            'To show a comparison, an evaluation, spend or a load test',
            <>
              <code>--report out.html</code> on that command
            </>,
          ],
          [
            'To render something measured yesterday, or on another machine',
            <>
              <code>effgen report result.json</code>
            </>,
          ],
          [
            'A record of one stored run, after the fact',
            <>
              <code>effgen runs show &lt;id&gt; --card summary.html</code>
            </>,
          ],
          [
            'The data itself, for a script',
            <>
              <code>--json</code> (stdout) or <code>-o result.json</code> (a file) —{' '}
              <Link to="/cli">the CLI overview</Link> covers the split
            </>,
          ],
          [
            'A live view rather than a file',
            <>
              <Link to="/cli/top"><code>effgen top</code></Link>, or the{' '}
              <Link to="/dashboard">dashboard</Link>
            </>,
          ],
        ]}
      />

      <h2>In continuous integration</h2>

      <CodeBlock
        language="yaml"
        filename=".github/workflows/eval.yml"
        code={`- name: Evaluate
  run: |
    effgen eval --suite math -m openai:gpt-5-nano \\
      --fail-under 80 --report eval-report.html --json > eval.json
  env:
    OPENAI_API_KEY: \${{ secrets.OPENAI_API_KEY }}

- name: Publish the report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: eval-report
    path: eval-report.html`}
        caption={
          <>
            <code>if: always()</code> because the report is most useful on the run that failed the
            gate. <Link to="/evaluation">Evaluation and CI gates</Link> covers the gate itself.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>Object of type ToolCall is not JSON serializable</code> from{' '}
              <code>effgen run -o …</code> or <code>--json</code>, exit <code>1</code>
            </>,
            <>
              A defect in 1.0.0 and 1.0.1, fixed in 1.1.0: the saved document's{' '}
              <code>execution_tree</code> carried tool-call objects that the JSON writer could not
              encode, so any run that called a tool failed to serialize.
            </>,
            <>
              Upgrade with <code>pip install -U effgen</code>. On an older release, read the run back
              from history with <code>effgen runs show &lt;id&gt; --json</code>.
            </>,
          ],
          [
            <>
              <code>Could not tell which report this result is</code>, exit <code>2</code>
            </>,
            'The document is not one of the recognised result shapes.',
            <>
              Name the shape with <code>--kind</code>, or re-produce the result with the{' '}
              <code>--json</code> of the command that measures it.
            </>,
          ],
          [
            <>
              <code>… is not valid JSON</code>, exit <code>2</code>
            </>,
            <>
              The file is truncated or is not JSON — a redirect that captured human output as well,
              or the serialization failure above.
            </>,
            <>
              Check the file parses: <code>jq . result.json</code>. Remember that{' '}
              <code>--json</code> puts the document on stdout and prose on stderr, so{' '}
              <code>… --json 2&gt;/dev/null &gt; out.json</code> keeps them apart.
            </>,
          ],
          [
            'The report opens but a chart is empty',
            'The result carries no data for that chart — a comparison of one model, an evaluation with one case.',
            'Expected. The tables carry the same figures.',
          ],
          [
            <>
              Costs read <code>unpriced</code> rather than a number
            </>,
            'The catalog has no per-token rate for that model. effGen does not invent one.',
            <>
              <code>effgen models refresh --provider &lt;name&gt;</code>.{' '}
              <Link to="/cost">Cost and budgets</Link> explains why zero and unpriced are kept
              apart.
            </>,
          ],
          [
            'The file will not render for a colleague',
            'Something re-saved it, or a mail client rewrote it.',
            'Attach it rather than pasting it inline. The original is one file with everything inlined and no external request.',
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>effgen report</code>, <code>--report</code> and <code>--card</code> are all new in
          this release, along with the run-card export on <code>effgen runs show</code>. The
          self-containment of the generated files is enforced by a test in the framework, not by
          convention.
        </p>
      </Callout>

      <SeeAlso paths={['/cli/history', '/compare', '/evaluation']} />
    </DocPage>
  );
}
