import { FlaskConical } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  Figure,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData } from '../siteData';
import { figureOf, webCapture } from '../webCaptures';

const { snippet_kinds, modes } = siteData.web.playground;

export default function Playground() {
  return (
    <DocPage
      subtitle="Trying a model, a prompt and a set of tools in the browser, and copying the result back out as code."
      icon={<FlaskConical size={48} />}
    >
      <p>
        The server serves a second page at <code>/playground</code>: pick a model from the real
        catalog, pick a preset or individual tools, write a prompt, and watch the answer stream in
        with its tokens, cost and tool trace. Then copy the same run out as{' '}
        {snippet_kinds.map((kind, i) => (
          <span key={kind}>
            {i > 0 && (i === snippet_kinds.length - 1 ? ' or ' : ', ')}
            <code>{kind}</code>
          </span>
        ))}
        . It adds no new way to execute a model — every run is one{' '}
        <code>POST /v1/chat/completions</code>, so it inherits that endpoint's auth, rate limits and
        spend controls.
      </p>

      <h2>Opening it</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen serve --port 8246`} />

      <p>
        Then <code>http://127.0.0.1:8246/playground</code>. The start-up banner prints the URL, and
        the same server serves the <Link to="/dashboard">dashboard</Link> beside it.
      </p>

      <Figure
        {...figureOf(webCapture('playground-run', 'dark'))}
        caption={webCapture('playground-run', 'dark').produced_by}
      />

      <h2>What the page loads first</h2>

      <p>
        Before you can type anything the page fetches two documents, both same-origin. Nothing on
        it is hard-coded: the presets are the server's presets and the models are the server's
        catalog.
      </p>

      <Terminal
        command={`curl -s http://127.0.0.1:8246/playground/bootstrap | python -c "import json,sys; print(sorted(json.load(sys.stdin)))"`}
        output={`['catalog_url', 'default_model', 'defaults', 'dev_mode', 'presets', 'session_key',
 'spend_authorized', 'tools', 'version']`}
      />

      <ApiTable
        headers={['Key', 'Carries']}
        rows={[
          [
            <code>presets</code>,
            <>
              All {siteData.presets.count} presets, each with its description, its system prompt,
              its tool list and its temperature. <Link to="/presets">Presets</Link>.
            </>,
          ],
          [<code>tools</code>, 'The tools offered as individual checkboxes alongside a preset.'],
          [
            <code>defaults</code>,
            <>
              The form's starting values — <code>{'{'}"max_tokens": 512, "temperature": 0.7,
              "stream": true{'}'}</code>.
            </>,
          ],
          [<code>default_model</code>, 'The model the picker opens on.'],
          [
            <code>catalog_url</code>,
            <>
              Where to fetch the model list — <code>/v1/models/catalog</code>.
            </>,
          ],
          [
            <code>session_key</code>,
            'A key minted for local viewing, so a loopback server does not have to be given one by hand.',
          ],
          [
            <>
              <code>dev_mode</code>, <code>spend_authorized</code>
            </>,
            'What posture the server is in, so the page can show the right banner.',
          ],
          [<code>version</code>, 'The effGen version serving the page.'],
        ]}
      />

      <Terminal
        command={`curl -s http://127.0.0.1:8246/v1/models/catalog | python -c "import json,sys; d=json.load(sys.stdin); print(sorted(d)); print('data:', len(d['data']), 'local:', len(d['local']))"`}
        output={`['counts', 'data', 'local', 'object', 'providers']
data: 416 local: 50`}
        caption={
          <>
            The same {siteData.models.models} catalogued models the CLI browses, with pricing and
            capability flags, plus the local models this host can serve.{' '}
            <code>providers</code> carries each provider's model count, the date its catalog was
            verified and its default model.{' '}
            <Link to="/catalog">The model catalog</Link> covers the data.
          </>
        }
      />

      <h2>Two modes</h2>

      <ApiTable
        headers={['Mode', 'What it does']}
        rows={[
          [
            <code>{modes[0]}</code>,
            'One model answers one prompt. The answer streams into a result panel with tokens, cost and — when tools ran — the step trace.',
          ],
          [
            <code>{modes[1]}</code>,
            <>
              Several models answer the same prompt in parallel, one column each, filling in as
              they stream, closed by a verdict panel. The same race{' '}
              <Link to="/compare"><code>effgen battle</code></Link> runs in the terminal.
            </>,
          ],
        ]}
      />

      <Figure
        {...figureOf(webCapture('playground-battle', 'dark'))}
        caption={webCapture('playground-battle', 'dark').produced_by}
      />

      <Figure
        {...figureOf(webCapture('playground-trace', 'dark'))}
        caption={webCapture('playground-trace', 'dark').produced_by}
      />

      <h2>Copying a run back out</h2>

      <p>
        Every finished run can be copied as {snippet_kinds.length} kinds of snippet, generated from
        the form state rather than from a template — the model, the prompt, the tools, the
        temperature, the token cap and the system prompt you actually used. That is the point of the
        page: it is a place to find the settings, not a place to keep the work.
      </p>

      <ApiTable
        headers={['Snippet', 'Runs it']}
        rows={[
          [
            <code>{snippet_kinds[0]}</code>,
            <>
              Against the same endpoint from a shell.{' '}
              <Link to="/openai-api">The OpenAI-compatible API</Link>.
            </>,
          ],
          [
            <code>{snippet_kinds[1]}</code>,
            <>
              As an <code>effgen run</code> command. <Link to="/cli/run">run and chat</Link>.
            </>,
          ],
          [
            <code>{snippet_kinds[2]}</code>,
            <>
              As an <code>Agent</code> script. <Link to="/agents">Agents</Link>.
            </>,
          ],
        ]}
      />

      <h2>The request behind every run</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`curl -s -X POST http://127.0.0.1:8246/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"openai:gpt-5-nano","messages":[{"role":"user","content":"Say ok and nothing else."}]}'`}
      />

      <Terminal
        command={`curl -s -X POST http://127.0.0.1:8246/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"openai:gpt-5-nano","messages":[{"role":"user","content":"Say ok and nothing else."}]}' | python -m json.tool`}
        output={`{
    "id": "chatcmpl-6299830d3ca548baa6abb2a7",
    "object": "chat.completion",
    "created": 1787614181,
    "model": "openai:gpt-5-nano",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "ok"
            },
            "finish_reason": "stop",
            "logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 266,
        "total_tokens": 291
    },
    "effgen": {
        "requested_model": "openai:gpt-5-nano",
        "resolved_model": "openai:gpt-5-nano",
        "alias_applied": false,
        "cost_usd": 0.00010765,
        "run_id": "fde29596fcc8"
    }
}`}
        maxLines={24}
        caption={
          <>
            A standard chat-completion body plus an <code>effgen</code> block carrying the resolved
            model, whether an alias was applied, the cost and the <code>run_id</code> — the same id{' '}
            <Link to="/cli/history"><code>effgen runs show</code></Link> takes.
          </>
        }
      />

      <h2>Keyboard</h2>

      <p>
        The playground and the <Link to="/dashboard">dashboard</Link> share one keyboard layer —
        the same <code>webui.js</code>, served from both pages — so a shortcut learned on one works
        on the other.
      </p>

      <ApiTable
        headers={['Key', 'Action']}
        rows={[
          [<kbd>Cmd/Ctrl-K</kbd>, 'Open the command palette'],
          [<kbd>?</kbd>, 'Show the shortcut reference'],
          [
            <>
              <kbd>↑</kbd> <kbd>↓</kbd>
            </>,
            'Move through palette results',
          ],
          [<kbd>Enter</kbd>, 'Run the highlighted command'],
          [<kbd>Esc</kbd>, 'Close the palette, the shortcut list, or an open detail pane'],
          [<kbd>Tab</kbd>, 'Move through the page; the first stop is a "Skip to content" link'],
        ]}
        caption={
          <>
            The colour theme is stored under one key, <code>effgen-theme</code>, shared by every
            effGen web surface.
          </>
        }
      />

      <h2>Keys and spend</h2>

      <Callout type="warning" title="A key you paste stays in the tab">
        <p>
          The API key is held in memory for that tab only and is never written to disk. Closing the
          tab loses it, which is deliberate. On a loopback server the bootstrap document supplies a
          session key so there is nothing to paste at all.
        </p>
      </Callout>

      <p>
        Because a run is an ordinary call to <code>/v1/chat/completions</code>, everything that
        governs that endpoint governs the page: authentication, per-principal rate limits,{' '}
        <Link to="/cost">daily budgets</Link> and the audit log. A prompt run here appears in{' '}
        <Link to="/cli/history">run history</Link>, in the dashboard's traffic and per-model panels,
        and in <Link to="/cli/top"><code>effgen top</code></Link>, exactly like any other request.
      </p>

      <h2>Self-contained</h2>

      <p>
        Its three static files — {siteData.web.static_files
          .filter((file) => file.file.startsWith('playground/'))
          .map((file) => file.file.split('/').pop())
          .join(', ')}{' '}
        — plus the two shared with the dashboard, ship inside the <code>effgen</code> package. The
        shipped files carry {siteData.web.external_references} references to another host, so the
        page renders the same in an air-gapped deployment. The one loopback address in them is text
        the page <em>displays</em> — the example endpoint inside the copy-as-curl snippet — not
        something the browser requests.
      </p>

      <h2>Playground or terminal?</h2>

      <ApiTable
        headers={['You want', 'Use']}
        rows={[
          [
            'To find the settings — which model, which tools, what temperature',
            'The playground, then copy the snippet.',
          ],
          [
            'To run the same thing repeatedly',
            <>
              <Link to="/cli/run"><code>effgen run</code></Link>, which the copied CLI snippet gives
              you.
            </>,
          ],
          [
            'To race models on one prompt',
            <>
              Either: <strong>Battle</strong> here, or{' '}
              <Link to="/compare"><code>effgen battle</code></Link> with{' '}
              <code>--report out.html</code> for a file to share.
            </>,
          ],
          [
            'To score models against expected answers',
            <>
              <Link to="/compare"><code>effgen compare</code></Link> — the playground has no scoring
              suite.
            </>,
          ],
          [
            'To give a colleague something to try',
            <>
              The playground on a server they can reach. Give them a key rather than{' '}
              <code>EFFGEN_DEV_MODE=1</code>.
            </>,
          ],
        ]}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'An authentication banner across the top and no model list',
            <>
              The server needs a key and the page has none — the catalog and bootstrap calls are
              authenticated like any other data route.
            </>,
            <>
              Paste the key the server printed at start-up, or set{' '}
              <code>EFFGEN_API_KEY</code> before starting it.
            </>,
          ],
          [
            <>
              A <code>404</code> naming the model
            </>,
            'The model id is not one this server can serve — a provider whose key is missing, or an id the catalog no longer has.',
            <>
              Pick from the picker rather than typing an id.{' '}
              <code>effgen doctor</code> lists which providers are usable.
            </>,
          ],
          [
            'The answer arrives in one block instead of streaming',
            <>
              Streaming is off in the form, or something between the browser and the server buffers
              the response.
            </>,
            <>
              Check the stream toggle. Behind a reverse proxy, disable response buffering for{' '}
              <code>/v1/chat/completions</code>.
            </>,
          ],
          [
            'The cost reads unpriced',
            'The catalog has no published rate for that model.',
            <>
              <code>effgen models refresh --provider &lt;name&gt;</code>. The tokens are still
              counted — <Link to="/cost">Cost and budgets</Link>.
            </>,
          ],
          [
            <>
              A <code>429</code>
            </>,
            'A rate limit or a spent budget — the endpoint\'s, not the page\'s.',
            <>
              <Link to="/api-server">The API server</Link> covers the per-principal limits;{' '}
              <code>effgen cost today</code> shows the day's spend.
            </>,
          ],
          [
            'A battle column reports a failure while the others answer',
            'One model failed. It does not end the race and cannot win the verdict.',
            <>
              Expected. The column carries the typed error —{' '}
              <Link to="/errors">Errors and exceptions</Link>.
            </>,
          ],
          [
            <>
              <code>/playground</code> 404s while <code>/dashboard</code> works
            </>,
            'Almost always a different process on that port than the one you started.',
            <>
              Compare the port with the start-up banner. Both pages come from the same server;
              neither exists without it.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          The playground is new in this release, along with Battle mode, the copy-as-snippet buttons
          and the command palette it shares with the dashboard. It adds no model-execution path:
          every run goes through the same <code>/v1/chat/completions</code> endpoint an external
          client would use.
        </p>
      </Callout>

      <SeeAlso paths={['/dashboard', '/openai-api', '/compare']} />
    </DocPage>
  );
}
