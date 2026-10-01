import { LayoutDashboard } from 'lucide-react';
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
import { siteHref } from '../siteLinks';

const { panels, summary_cards, data_endpoints } = siteData.web.dashboard;

export default function Dashboard() {
  return (
    <DocPage
      subtitle="The real-time page the server serves at /dashboard — what each panel reads, and the endpoints behind them."
      icon={<LayoutDashboard size={48} />}
    >
      <p>
        A single-page app the API server serves at <code>/dashboard</code>. It polls the server every
        five seconds and streams trace spans over Server-Sent Events. Everything it needs ships
        inside the <code>effgen</code> package — {siteData.web.external_references} references to
        another host in{' '}
        {(siteData.web.static_bytes / 1024).toFixed(0)}&nbsp;KB of static files — so it renders
        identically in an air-gapped deployment.
      </p>

      <p className="doc-crosslink">
        This page is the reference: every panel, every endpoint, every payload. For what it looks
        like with traffic running through it, see <a href={siteHref('/dashboard')}>the dashboard
        page</a> on the main site.
      </p>

      <h2>Opening it</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen serve --port 8244`}
      />

      <Terminal
        command="effgen serve --port 8244"
        output={`
effGen v1.0.1 - API Server
Auth: DISABLED (EFFGEN_DEV_MODE=1) — do not use in production
Starting server on 127.0.0.1:8244
  OpenAI-compatible API : http://127.0.0.1:8244/v1
  Interactive docs      : http://127.0.0.1:8244/docs
  Dashboard             : http://127.0.0.1:8244/dashboard
  Playground            : http://127.0.0.1:8244/playground
  Both pages: Cmd/Ctrl-K opens the command palette, ? lists shortcuts.`}
        caption={
          <>
            The <code>Auth: DISABLED</code> line is there because this server was started with{' '}
            <code>EFFGEN_DEV_MODE=1</code> — see <em>Authentication</em> below.{' '}
            <Link to="/api-server">The API server</Link> covers the rest of{' '}
            <code>effgen serve</code>.
          </>
        }
      />

      <Figure
        {...figureOf(webCapture('dashboard-full', 'dark'))}
        caption={webCapture('dashboard-full', 'dark').produced_by}
      />

      <h2>The panels</h2>

      <p>
        A jump row under the header links to each of the {panels.length}, and selecting one scrolls
        to it <em>and moves focus into it</em>, so the next <kbd>Tab</kbd> continues from there.
      </p>

      <ApiTable
        headers={['Panel', 'What it shows']}
        rows={[
          [
            <strong>Summary cards</strong>,
            <>{summary_cards.join(', ')} — across the top of the page.</>,
          ],
          [
            <strong>{panels[0].title}</strong>,
            'Progress bars showing how close p99 latency, error rate and availability are to their SLO thresholds.',
          ],
          [
            <strong>{panels[1].title}</strong>,
            'A rolling line chart of average latency over recent polling intervals, drawn on a canvas.',
          ],
          [
            <strong>{panels[2].title}</strong>,
            <>
              One row per model <em>and provider</em>: calls, error rate, p95 latency, the dominant
              failure class with its remediation, tokens and cost. Six columns sort on click. Spend
              that could not be matched to a row is stated below the table rather than spread across
              it.
            </>,
          ],
          [
            <strong>{panels[3].title}</strong>,
            'One chip per status code. Each states the code, the status class and the count as text, so it does not depend on colour.',
          ],
          [
            <strong>{panels[4].title}</strong>,
            <>
              Requests, failures and error rate per route and method, worst first, with a per-class
              breakdown. This is the panel with a denominator — it separates the routes behind a
              shared status code. Traffic outside the recorded route list, the dashboard's own
              polling included, is labelled <code>other</code>.
            </>,
          ],
          [<strong>{panels[5].title}</strong>, 'The last 50 agent runs with model, tokens, cost, duration and a success/error badge.'],
          [
            <strong>{panels[6].title}</strong>,
            'Stored runs and saved sessions, filterable by text and status, each run opening a detail pane.',
          ],
          [
            <strong>{panels[7].title}</strong>,
            'A real-time feed of trace spans over SSE, with a pause toggle and a clear button.',
          ],
          [
            <strong>{panels[8].title}</strong>,
            'Spans grouped by run, positioned by start offset and sized by duration.',
          ],
          [
            <strong>{panels[9].title}</strong>,
            <>
              Team and workflow executions as a node-link graph: agents (the manager marked apart)
              and the tools they reached as nodes; delegation, handoff and tool use as edges. Status
              is carried by a glyph and a text label as well as colour, and nodes are
              keyboard-focusable.
            </>,
          ],
          [
            <strong>{panels[10].title}</strong>,
            'Every model the catalog knows, with context window, output limit, price and capabilities — filterable and paged.',
          ],
          [
            <strong>{panels[11].title}</strong>,
            <>
              Every registered Prometheus metric name and its current value.{' '}
              <Link to="/metrics">Metrics</Link> documents the families.
            </>,
          ],
        ]}
        caption={
          <>
            Read off the shipped page's own panel ids and titles, so this table cannot describe a
            panel the release does not have.
          </>
        }
      />

      <Figure
        {...figureOf(webCapture('dashboard-by-model', 'dark'))}
        caption={webCapture('dashboard-by-model', 'dark').produced_by}
      />

      <Figure
        {...figureOf(webCapture('dashboard-topology', 'dark'))}
        caption={webCapture('dashboard-topology', 'dark').produced_by}
      />

      <h2>Keyboard</h2>

      <p>
        The dashboard and the <Link to="/playground">playground</Link> share one keyboard layer, so a
        shortcut learned on one works on the other.
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
          [
            <kbd>Tab</kbd>,
            'Move through the page; the first stop is a "Skip to content" link',
          ],
        ]}
      />

      <p>
        The palette searches four groups, all built from data the page has already loaded:{' '}
        <strong>Navigate</strong> (every panel, plus the other surface), <strong>Actions</strong>{' '}
        (switch theme, refresh, clear or pause the span stream, focus a search box),{' '}
        <strong>Runs</strong> (stored runs matched on task text, model, status or run id — selecting
        one opens its detail) and <strong>Models</strong> (the catalog matched on id, provider,
        family or capability — selecting one filters the catalog table). The commands used most
        recently lead the list when the palette opens empty.
      </p>

      <Figure
        {...figureOf(webCapture('dashboard-palette', 'dark'))}
        caption={webCapture('dashboard-palette', 'dark').produced_by}
      />

      <p>
        The theme choice is stored under one key, <code>effgen-theme</code>, shared by every effGen
        web surface, so a theme picked on one applies to the other.
      </p>

      <h2>The endpoints behind it</h2>

      <ApiTable
        headers={['Endpoint', 'Serves']}
        rows={[
          [
            <code>{data_endpoints[0]}</code>,
            'The whole polling payload — metrics, SLOs, per-model, per-status, per-route, recent runs and buffered spans. Polled every five seconds.',
          ],
          [<code>{data_endpoints[1]}</code>, 'The SSE span stream.'],
          [<code>{data_endpoints[2]}</code>, 'The model catalog the catalog panel pages through.'],
          [<code>{data_endpoints[3]}</code>, 'Stored runs and saved sessions for the history panel.'],
          [
            <code>{data_endpoints[4]}</code>,
            <>
              Recent team and workflow executions as node-link graphs. Takes{' '}
              <code>?limit=</code>.
            </>,
          ],
        ]}
        caption="Every one is same-origin. The page makes no request to any other host."
      />

      <h3>
        <code>/dashboard/data.json</code>
      </h3>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`curl -s http://127.0.0.1:8244/dashboard/data.json | python -m json.tool`}
      />

      <Terminal
        command="curl -s http://127.0.0.1:8244/dashboard/data.json | python -m json.tool | head -60"
        output={`{
    "ts": "2026-08-24T23:26:43Z",
    "version": "1.0.1",
    "metrics": {
        "total_requests": 4,
        "total_errors": 1,
        "avg_latency_s": 1.6825,
        "total_tokens": 1056,
        "cost_usd": 0.000399,
        "daily_cost_usd": 0.000399,
        "priced_runs": 3,
        "unpriced_runs": 1,
        "http_client_errors": 3,
        "http_server_errors": 0
    },
    "slo": {
        "p99_latency_burn": 2.45,
        "error_rate_burn": 25.0,
        "availability": 0.75,
        "latency_threshold_s": 2.0,
        "p50_latency_s": 1.75,
        "p95_latency_s": 4.5,
        "p99_latency_s": 4.9
    },
    "by_model": [
        {
            "model": "gpt-5-nano",
            "provider": "openai",
            "calls": 3,
            "errors": 0,
            "input_tokens": 66,
            "output_tokens": 990,
            "outcomes": { "ok": 3 },
            "top_error": null,
            "top_error_hint": null,
            "error_rate": 0.0,
            "p95_latency_s": 4.625,
            "cost_usd": 0.000399
        },
        {
            "model": "gpt-9-does-not-exist",
            "provider": "openai",
            "calls": 1,
            "errors": 1,
            "outcomes": { "not_found": 1 },
            "top_error": "not_found",
            "top_error_hint": "Model id not found — run \`effgen models list\` to see ids, …",
            "error_rate": 1.0,
            "p95_latency_s": 0.2425,
            "cost_usd": null
        }
    ]
}`}
        maxLines={26}
        caption={
          <>
            A real payload, after three completions on <code>gpt-5-nano</code> and one on a model id
            that does not exist. <code>cost_usd: null</code> on the failed row, not{' '}
            <code>0</code> — the request never billed anything, so there is nothing to report.
          </>
        }
      />

      <Terminal
        command={`curl -s http://127.0.0.1:8244/dashboard/data.json | python -c "import json,sys; print(sorted(json.load(sys.stdin)))"`}
        output={`['by_model', 'by_route', 'by_status', 'by_status_detail', 'metrics', 'prompt_templates',
 'raw_metrics', 'recent_runs', 'recent_spans', 'slo', 'slos', 'spans', 'ts',
 'unattributed_cost_usd', 'version']`}
      />

      <Callout type="note" title="Three counters that look like each other and are not">
        <p>
          <code>metrics.total_requests</code> counts model calls.{' '}
          <code>by_status</code> counts HTTP responses on <em>every</em> route — health checks and
          the dashboard's own polling included — which is why it is normally the larger number, and{' '}
          <code>by_status_detail</code> and <code>by_route</code> carry the route and method that
          separate a bad model id on <code>/v1/chat/completions</code> from a probe of an unknown
          path. <code>unattributed_cost_usd</code> holds spend that could not be matched to a{' '}
          <code>by_model</code> row, so the cost column always sums to money actually attributed.
        </p>
      </Callout>

      <h3>The span stream</h3>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`curl -sN http://127.0.0.1:8244/dashboard/spans`}
      />

      <Terminal
        command="curl -sN http://127.0.0.1:8244/dashboard/spans | head -2"
        output={`data: {"ts": "23:26:43", "name": "effgen.model.call openai:gpt-9-does-not-exist", "kind": "model",
"agent": null, "tool": null, "model": "openai:gpt-9-does-not-exist", "duration_ms": 134.6,
"status": "error", "error": "Generation failed: The model \`gpt-9-does-not-exist\` does not exist or
you do not have access to it. Did you mean: gpt-5-mini, gpt-5-nano, gpt-4o-mini? …", "note": null,
"run_id": "5d0809d3722c", "offset_ms": 0.1, "execution_id": null, "execution_kind": null,
"execution_name": null, "parent_agent": null, "role": null}`}
        maxLines={10}
        caption="One JSON object per SSE event, line-wrapped here for the page."
      />

      <ApiTable
        headers={['Field', 'Means']}
        rows={[
          [
            <code>kind</code>,
            <>
              <code>agent</code>, <code>model</code>, <code>tool</code> or <code>router</code>. The
              matching <code>agent</code>/<code>model</code>/<code>tool</code> field names what the
              span timed — read those rather than parsing <code>name</code>, which is the display
              label.
            </>,
          ],
          [
            <code>status</code>,
            <>
              <code>ok</code>, <code>error</code> or <code>skipped</code>. A run that reports a
              failure without raising still records it here.
            </>,
          ],
          [<code>duration_ms</code>, 'How long the span took.'],
          [
            <>
              <code>run_id</code>, <code>offset_ms</code>
            </>,
            'Which run it belongs to, and where it sits on that run’s timeline.',
          ],
          [
            <>
              <code>execution_id</code>, <code>execution_kind</code>, <code>execution_name</code>,{' '}
              <code>parent_agent</code>, <code>role</code>
            </>,
            'Group the spans of one team or workflow run — the fields the topology graph is built from.',
          ],
        ]}
        caption={
          <>
            <Link to="/tracing">Tracing and spans</Link> covers the span model itself.
          </>
        }
      />

      <h3>The topology endpoint</h3>

      <Terminal
        command="curl -s 'http://127.0.0.1:8244/dashboard/topology.json?limit=6' | python -m json.tool"
        output={`{
    "executions": [],
    "count": 0
}`}
        caption={
          <>
            Empty because nothing multi-agent had run on that server yet. It is built from the
            durable run store <em>plus</em> the buffered spans, so a team run started from a script
            or the CLI appears here too — not only work done inside the server process.
          </>
        }
      />

      <CodeBlock
        language="json"
        filename="a populated execution"
        code={`{
  "executions": [
    {
      "id": "b25fed1b57d3",
      "kind": "team",
      "name": "newsroom",
      "status": "ok",
      "cost_usd": 0.000042,
      "tokens": 1234,
      "nodes": [
        {"id": "lead", "type": "manager", "status": "ok", "model": "openai/gpt-oss-20b",
         "runs": 2, "cost_usd": 0.00002, "tokens": 800, "duration_s": 1.2}
      ],
      "edges": [{"source": "lead", "target": "researcher", "kind": "delegation", "count": 1}]
    }
  ],
  "count": 1
}`}
        caption={
          <>
            The shape a team run produces. <Link to="/multi-agent">Multi-agent teams</Link> covers
            running one.
          </>
        }
      />

      <h2>Authentication</h2>

      <p>
        The static shell — HTML, JS and CSS — is public, so the page can load and prompt for a key.
        The five data endpoints require authentication by default and answer with a typed{' '}
        <code>invalid_api_key</code> envelope without one.
      </p>

      <Terminal
        command="curl -s -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:8245/dashboard && curl -s http://127.0.0.1:8245/dashboard/data.json"
        output={`200
{"error": {"message": "Missing API key (send 'Authorization: Bearer <key>' or 'X-API-Key: <key>')", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}`}
        caption="The same server with no EFFGEN_DEV_MODE and no EFFGEN_PUBLIC_DASHBOARD: the page loads, its data does not."
      />

      <Figure
        {...figureOf(webCapture('dashboard-auth', 'dark'))}
        caption={webCapture('dashboard-auth', 'dark').produced_by}
      />

      <ApiTable
        headers={['Posture', 'Set', 'Result']}
        rows={[
          [
            'Default',
            'nothing',
            <>
              An ephemeral key is minted and printed once at start-up. The page loads; its data calls
              answer <code>invalid_api_key</code> until a key is supplied.
            </>,
          ],
          [
            'A fixed key',
            <code>EFFGEN_API_KEY=…</code>,
            <>
              The page prompts for it, and <code>effgen top --api-key</code> uses the same one.
            </>,
          ],
          [
            'Local viewing',
            <code>EFFGEN_PUBLIC_DASHBOARD=1</code>,
            'The five data endpoints answer without a key. For a loopback-bound server you are watching yourself.',
          ],
          [
            'Local development',
            <code>EFFGEN_DEV_MODE=1</code>,
            'All authentication off, with a loud warning at start-up and a CRITICAL log line. Never on anything another machine can reach.',
          ],
        ]}
        caption={
          <>
            In a shared deployment, restrict access at the network or ingress layer as well —{' '}
            <Link to="/api-server">The API server</Link> and <Link to="/security">Security</Link>.
          </>
        }
      />

      <h2>Getting runs into it</h2>

      <p>
        The <code>Agent</code> class records every run automatically, best-effort — it never breaks a
        run when the dashboard machinery is unavailable. A custom integration that bypasses{' '}
        <code>Agent</code> can record one itself:
      </p>

      <CodeBlock
        filename="record.py"
        code={`from effgen.observability.run_log import record_run

record_run(
    model="openai:gpt-5-nano",
    input_tokens=250,
    output_tokens=80,
    duration_s=1.12,
    cost_usd=0.000032,
)`}
      />

      <h2>Accessibility</h2>

      <p>
        Four guarantees the framework's own tests drive the real pages to hold:
      </p>

      <ApiTable
        headers={['Guarantee', 'What it means in practice']}
        rows={[
          [
            'Focus survives a refresh',
            "Focus on a run's disclosure button stays there when the five-second poll rebuilds the History table, and the topology graph behaves the same way. If the run is no longer listed, focus moves to the History panel rather than to the top of the document.",
          ],
          [
            'Nothing is announced when nothing changed',
            'Every value the page writes goes through a write-if-changed rule, so an idle dashboard is silent rather than re-reading five cards, the SLO line and the connection status every poll.',
          ],
          [
            'Contrast',
            'Every control boundary clears 3:1 against the surface behind it in both themes (WCAG 1.4.11), and every text pair clears its AA threshold.',
          ],
          [
            'Sorting is announced',
            <>
              A sortable header is a button inside the <code>&lt;th&gt;</code>, exactly one header
              carries <code>aria-sort</code>, and the new order is spoken.
            </>,
          ],
        ]}
      />

      <p>
        Smooth scrolling is skipped for a viewer who prefers reduced motion, and the charts are
        drawn on a canvas with the topology graph built as inline SVG — no chart or graph library is
        involved, which is part of why the page needs no network.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              The page loads but every panel is empty and the data calls return{' '}
              <code>invalid_api_key</code>
            </>,
            'The default posture: the shell is public, the data is not.',
            <>
              Enter the key the server printed at start-up, set <code>EFFGEN_API_KEY</code>, or for
              local viewing start it with <code>EFFGEN_PUBLIC_DASHBOARD=1</code>.
            </>,
          ],
          [
            'Traffic figures are far lower than the spend ledger says',
            'They measure different things. These counters are this server process since it started; the ledger is durable and machine-wide.',
            <>
              Working as intended. <Link to="/cli/top">effgen top</Link> labels the same distinction
              panel by panel.
            </>,
          ],
          [
            'The topology panel is empty',
            'Nothing multi-agent has run yet.',
            <>
              Run a <Link to="/multi-agent">team</Link> or a{' '}
              <Link to="/workflows">workflow</Link>. Single-agent runs produce no graph.
            </>,
          ],
          [
            'A cost column reads — for a model that clearly ran',
            'The catalog has no published rate for it, or the call failed before it billed anything.',
            <>
              <code>effgen models refresh --provider &lt;name&gt;</code>.{' '}
              <Link to="/cost">Cost and budgets</Link> explains why zero and unpriced are kept
              apart.
            </>,
          ],
          [
            'The latency chart grows taller on every redraw',
            <>
              A defect in this release on displays with a device pixel ratio above 1: the canvas
              doubles in height each time it is drawn.
            </>,
            'Reload the page to reset it. It does not affect the figures, only the drawing.',
          ],
          [
            'The span stream stops updating',
            'The SSE connection dropped, or the stream is paused.',
            'The panel has a pause toggle — check it first, then reload. The buffered spans are also in the polling payload.',
          ],
          [
            <>
              <code>/dashboard</code> 404s
            </>,
            'Something else is listening on that port, or the server did not bind.',
            <>
              Check the start-up banner — it prints the URL it bound. <code>--port</code> picks
              another.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          The history, span-stream, run-timeline, topology, catalog and per-route panels are all new
          in this release, along with the command palette and the shared keyboard layer. The
          per-model table now scopes every figure to one <code>(model, provider)</code> pair rather
          than to the model name alone.
        </p>
      </Callout>

      <SeeAlso paths={['/playground', '/cli/top', '/observability']} />
    </DocPage>
  );
}
