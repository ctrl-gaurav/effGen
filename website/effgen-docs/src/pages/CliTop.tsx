import { Gauge } from 'lucide-react';
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

export default function CliTop() {
  return (
    <DocPage
      subtitle="One screen for what ran, what the server is serving, what it cost and what the GPUs are doing — refreshing in place, or as one JSON snapshot."
      icon={<Gauge size={48} />}
    >
      <p>
        <code>effgen top</code> (and its alias <code>effgen monitor</code>) puts everything effGen
        already records side by side. On a terminal it redraws in place; piped, or with{' '}
        <code>--json</code>, it prints one snapshot and exits — so the same command works in a
        window you are watching and in a cron job you are not.
      </p>

      <CodeBlock language="bash" filename="terminal" code={`effgen top --once`} />

      <Terminal
        command="effgen top --once"
        output={`effGen 1.0.1 — wangserv — 2026-08-24T22:57:10Z
server http://127.0.0.1:8000 unavailable

Activity — completed runs, local run history (all processes)
  no runs recorded yet

Traffic — unavailable: no server reachable at http://127.0.0.1:8000/dashboard/data.json ([Errno 111] Connection refused)

Per-model — unavailable: no server reachable at http://127.0.0.1:8000/dashboard/data.json ([Errno 111] Connection refused)

Spend — local cost ledger, last 24 hours
Metric        Value
------------  -----------------------
Total         $0.000000
Requests      0
Daily budget  $1.00 (0.0% used)
Burn rate     $0.000000/h (last hour)

GPU — physical devices, all processes
GPU  Name        Memory            Mem %  Util
---  ----------  ----------------  -----  ----
0    NVIDIA A40  1.85 / 44.42 GB   4%     0%
1    NVIDIA A40  0.0 / 44.42 GB    0%     100%
2    NVIDIA A40  0.0 / 44.42 GB    0%     100%
3    NVIDIA A40  44.35 / 44.42 GB  100%   100%`}
        maxLines={24}
        caption={
          <>
            A real snapshot, on a host with eight GPUs and no server running. Note what the two
            server-backed panels say: <em>unavailable</em>, with the URL that was tried — not zeros
            that would read as "no traffic, no errors".
          </>
        }
      />

      <h2>Five panels, five different measurements</h2>

      <p>
        This is the thing to understand before reading any number on the screen: the panels come
        from different sources, over different windows, about different processes. A server's
        in-memory request count and the durable spend ledger routinely differ by orders of
        magnitude. Each panel therefore states its own scope, and the figures are never added
        together.
      </p>

      <ApiTable
        headers={['Panel', 'Source', 'Window', 'Needs a server']}
        rows={[
          [
            'Activity',
            <>
              The run history, <code>$EFFGEN_HOME/runs</code>
            </>,
            'Completed runs, every process on the host',
            'No',
          ],
          [
            'Traffic',
            <>
              The server's own <code>/dashboard/data.json</code>
            </>,
            'That server process, since it started',
            'Yes',
          ],
          [
            'Per-model',
            <>
              The same endpoint's <code>by_model</code> block
            </>,
            'That server process, since it started',
            'Yes',
          ],
          [
            'Spend',
            'The local cost ledger',
            'Last 24 hours, plus a trailing-hour burn rate',
            'No',
          ],
          [
            'GPU',
            <>
              The GPU layer — the same one <code>effgen models status</code> reads
            </>,
            'Physical devices, every process',
            'No',
          ],
        ]}
      />

      <Callout type="warning" title="Two counters in the Traffic panel that are not the same counter">
        <p>
          <strong>Generations</strong> counts model calls. The <strong>HTTP status
          histogram</strong> counts responses on <em>every</em> route — health checks and the
          dashboard's own polling included — so it is normally the larger of the two. Both are
          labelled, and the histogram's scope travels in the JSON as{' '}
          <code>by_status_scope</code>.
        </p>
      </Callout>

      <p>
        Every Per-model row is scoped to one <code>(model, provider)</code> pair. A model name
        served by two providers is two rows, and neither borrows the other's latency tail or spend.
        Activity lists <strong>completed</strong> runs only — a run still executing is not in the
        history yet. A model with no published price shows <code>—</code>, never <code>$0</code>,
        and is left out of the spend total, which reports how many such models it excluded.
      </p>

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={[
          { name: '--json', type: 'flag', description: 'Print one snapshot as JSON and exit' },
          {
            name: '--once',
            type: 'flag',
            description: 'Print one static snapshot and exit (no refresh loop)',
          },
          {
            name: '--interval SECONDS',
            type: 'float',
            default: '2',
            description: 'Seconds between refreshes. Accepted range 0.5–300.',
          },
          {
            name: '--count N',
            type: 'int',
            description: 'Stop after N refreshes (default: run until you quit)',
          },
          {
            name: '--limit N',
            type: 'int',
            default: '12',
            description: 'Runs to show in the activity panel',
          },
          {
            name: '--url URL',
            description: (
              <>
                Server base URL (default: <code>EFFGEN_SERVER_URL</code> or{' '}
                <code>http://127.0.0.1:8000</code>)
              </>
            ),
          },
          {
            name: '-p PORT, --port PORT',
            type: 'int',
            description: (
              <>
                Server port on <code>127.0.0.1</code> — shorthand for <code>--url</code>
              </>
            ),
          },
          {
            name: '--api-key KEY',
            description: (
              <>
                API key for the server (default: <code>EFFGEN_API_KEY</code>)
              </>
            ),
          },
          {
            name: '--no-animation',
            type: 'flag',
            description: 'Print one static snapshot instead of the live view',
          },
        ]}
        caption={
          <>
            Every flag <code>effgen top --help</code> declares. <code>effgen monitor</code> is the
            same parser under a second name.
          </>
        }
      />

      <h3>On a terminal</h3>

      <p>
        <code>q</code>, <code>Escape</code> or <code>Ctrl-C</code> quits and restores the terminal.
        GPUs are sampled every 5 seconds whatever <code>--interval</code> says, because that is the
        slowest source. <code>--theme</code> and <code>NO_COLOR</code> are honoured here as
        everywhere else in the command line.
      </p>

      <p>
        The full-screen view is used only on an interactive terminal that has not opted out. Piped
        output, <code>--json</code>, <code>--once</code>, <code>--no-animation</code>,{' '}
        <code>NO_COLOR</code>, <code>CI=1</code> and <code>EFFGEN_NO_ANIM=1</code> each print one
        static snapshot and exit — no cursor control ever reaches a pipe, and no invocation loops
        forever inside a script.
      </p>

      <h2>Finding the server</h2>

      <p>
        The base URL is taken from <code>--url</code>, then <code>--port</code> (on{' '}
        <code>127.0.0.1</code>), then <code>EFFGEN_SERVER_URL</code>, then{' '}
        <code>http://127.0.0.1:8000</code> — the port <code>effgen serve</code> binds by default.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`export EFFGEN_SERVER_URL=http://10.0.0.4:8000
export EFFGEN_API_KEY=…            # the key the server was started with
effgen top`}
      />

      <p>
        Reading a server's data needs its API key unless it was started with{' '}
        <code>EFFGEN_PUBLIC_DASHBOARD=1</code>. A rejected key is reported in the panel with the
        reason, so a missing key never reads as an idle server.
      </p>

      <h2>The JSON snapshot</h2>

      <p>
        <code>--json</code> writes one document: a <code>header</code>, then the{' '}
        <code>activity</code>, <code>traffic</code>, <code>by_model</code>, <code>spend</code> and{' '}
        <code>gpu</code> panels. Every panel carries <code>scope</code>, <code>available</code> and{' '}
        <code>unavailable_reason</code>, which is what lets an automated consumer tell "no data"
        from "zero".
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen top --json | jq "{header: .header.server_reachable, spend: .spend, gpu_devices: (.gpu.devices|length)}"`}
      />

      <Terminal
        command={`effgen top --json | jq "{header: .header.server_reachable, spend: .spend, gpu_devices: (.gpu.devices|length)}"`}
        output={`{
  "header": false,
  "spend": {
    "scope": "local cost ledger, last 24 hours",
    "available": true,
    "unavailable_reason": null,
    "total_cost_usd": 0,
    "requests": 0,
    "daily_budget_usd": 1,
    "budget_used_pct": 0,
    "burn_rate_usd_per_hour": 0,
    "burn_window_s": 3600,
    "unpriced_models": [],
    "rows": []
  },
  "gpu_devices": 8
}`}
        maxLines={18}
        caption={
          <>
            <code>server_reachable: false</code> and <code>spend.available: true</code> in one
            document: the server panels are dark and the local ones are not.
          </>
        }
      />

      <h3>Watching without a terminal</h3>

      <CodeBlock
        language="bash"
        filename="check-burn.sh"
        code={`#!/usr/bin/env bash
# Alert when spend velocity crosses a threshold. No terminal needed.
effgen top --json | jq -e '.spend.burn_rate_usd_per_hour < 0.5' >/dev/null \\
  || echo "spend velocity above threshold"`}
        caption={
          <>
            <code>jq -e</code> sets its exit status from the result, so the check is the pipeline's
            own exit code. <Link to="/slos">SLOs and alerting</Link> covers the server-side
            equivalent.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              Traffic and Per-model read <code>unavailable: … Connection refused</code>
            </>,
            'No server is answering at the URL. The local panels are still correct.',
            <>
              Start one with <code>effgen serve</code>, or point at the right host with{' '}
              <code>--url</code> / <code>EFFGEN_SERVER_URL</code>. If you only run local agents,
              this is the normal state.
            </>,
          ],
          [
            'The server panels report an authentication failure',
            <>
              The server needs a key and none was supplied, or the one supplied was rejected.
            </>,
            <>
              <code>--api-key</code>, or export <code>EFFGEN_API_KEY</code>. For local viewing,
              start the server with <code>EFFGEN_PUBLIC_DASHBOARD=1</code>.
            </>,
          ],
          [
            'Activity is empty after runs that clearly happened',
            <>
              A different <code>EFFGEN_HOME</code>, or <code>EFFGEN_RUN_HISTORY=0</code>.
            </>,
            <>
              <code>effgen runs list</code> prints the directory it read.{' '}
              <Link to="/cli/history">Runs and sessions</Link> covers the store.
            </>,
          ],
          [
            'Traffic and Spend disagree by orders of magnitude',
            'Working as intended. One is a process counter since start-up, the other is a durable 24-hour ledger.',
            'Read each panel against its own scope line. They are never meant to reconcile.',
          ],
          [
            'The GPU panel shows utilization from work that is not yours',
            'It reads physical devices across every process on the host, not just this one.',
            'On a shared machine, treat it as machine-level context rather than as your run\'s usage.',
          ],
          [
            'It takes over the screen inside a script',
            'A terminal was allocated, so the live view engaged.',
            <>
              <code>--once</code>, <code>--json</code> or <code>--no-animation</code>. Any of the
              three prints one snapshot and exits.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>effgen top</code> is new in this release, along with the alias{' '}
          <code>effgen monitor</code>. The per-panel <code>scope</code> and{' '}
          <code>unavailable_reason</code> fields exist so that a panel with nothing in it can say
          why, rather than reporting zero.
        </p>
      </Callout>

      <SeeAlso paths={['/dashboard', '/cost', '/observability']} />
    </DocPage>
  );
}
