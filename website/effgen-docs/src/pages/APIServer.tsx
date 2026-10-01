import { Server } from 'lucide-react';
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

const serveOptions = siteData.cli.command_options['serve'] ?? [];
const serveEnv = siteData.cli.serve_env ?? [];
const roles = siteData.production.rbac_roles;

/** What each `serve` flag and environment setting is for, in one sentence. */
const SERVE_TYPES: Record<string, string> = {
  '--host HOST': 'str',
  '-p PORT, --port PORT': 'int',
  '--rate-limit N': 'int',
  '--trust-proxy': 'flag',
};

const SERVE_DEFAULTS: Record<string, string> = {
  '--host HOST': '127.0.0.1',
  '-p PORT, --port PORT': '8000',
  '--rate-limit N': 'EFFGEN_RATE_LIMIT',
  '--trust-proxy': 'off',
};

export default function APIServer() {
  return (
    <DocPage
      subtitle="Running `effgen serve`: authentication, roles, audit logging and rate limits."
      icon={<Server size={48} />}
    >
      <p>
        <code>effgen serve</code> runs the same application the{' '}
        <code>effgen.server.app:create_app</code> factory builds — the{' '}
        <Link to="/openai-api">OpenAI-compatible <code>/v1</code> routes</Link>, authentication,
        roles and budgets, the audit log, <Link to="/metrics">metrics</Link>, the{' '}
        <Link to="/dashboard">dashboard</Link> and the playground, from one place. It is
        authenticated by default: with no key and no issuer configured it mints an ephemeral key at
        startup and prints it once, so there is no configuration in which the server is open.
      </p>

      <h2>Start it</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen serve --port 8000`}
        caption={
          <>
            Binds <code>127.0.0.1</code>. Nothing outside the machine can reach it until you pass{' '}
            <code>--host 0.0.0.0</code>.
          </>
        }
      />

      <Terminal command="curl -s http://127.0.0.1:8000/health" output={`{"status":"ok","version":"1.0.1"}`} />

      <p>
        That is one of the handful of public routes. Everything else needs a credential, and says so
        in the same error envelope the rest of the API uses:
      </p>

      <Terminal
        command={`curl -s -o /dev/null -w 'HTTP %{http_code}\\n' http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/models`}
        output={`HTTP 401
{"error": {"message": "Missing API key (send 'Authorization: Bearer <key>' or 'X-API-Key: <key>')", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}`}
        title="curl"
      />

      <Terminal
        command={`curl -s -H "Authorization: Bearer $EFFGEN_API_KEY" http://127.0.0.1:8000/v1/models \\
  | python -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"`}
        output={`gpt-4
gpt-4-turbo
gpt-4o
gpt-4o-mini
gpt-3.5-turbo
gpt-3.5-turbo-instruct
default
effgen-default
Qwen/Qwen2.5-3B-Instruct
openai:gpt-5-nano`}
        title="curl"
        caption={
          <>
            Captured against effGen 1.0.1. <code>/v1/models</code> lists the aliases plus every
            id this process has already served — it is not the catalogue. Any{' '}
            <code>provider:model</code> id the server can reach is callable whether or not it
            appears there; <Link to="/catalog">the model catalog</Link> is the full list.
          </>
        }
      />

      <h2>What it serves</h2>

      <ApiTable
        headers={['Method', 'Path', 'What it is']}
        rows={[
          [
            'POST',
            <>
              <code>/v1/chat/completions</code>, <code>/v1/completions</code>,{' '}
              <code>/v1/embeddings</code>
            </>,
            <>
              The OpenAI protocol. <Link to="/openai-api">The OpenAI-compatible API</Link> is the
              page for these.
            </>,
          ],
          [
            'GET',
            <>
              <code>/v1/models</code>, <code>/v1/models/catalog</code>
            </>,
            'What this process has served, and the bundled catalogue.',
          ],
          [
            'POST',
            <code>/run</code>,
            <>
              effGen's own shape: a task in, an <code>output</code> / <code>success</code> /{' '}
              <code>metadata</code> object out — the same fields an{' '}
              <Link to="/agents"><code>AgentResponse</code></Link> carries.
            </>,
          ],
          [
            'GET',
            <code>/tools</code>,
            <>
              Every tool the server can run, with its parameters. The{' '}
              <Link to="/tools/gallery">gallery</Link> is the same list documented.
            </>,
          ],
          [
            'GET',
            <code>/whoami</code>,
            'The principal the current credential resolves to, and its roles.',
          ],
          [
            'GET',
            <>
              <code>/rbac/policy</code>, <code>/rbac/roles</code>
            </>,
            'The effective policy for the caller, and every defined role.',
          ],
          [
            'GET',
            <>
              <code>/metrics</code>, <code>/slo</code>
            </>,
            <>
              Prometheus text and error-budget status. <Link to="/metrics">Metrics</Link> and{' '}
              <Link to="/slos">SLOs and alerting</Link>.
            </>,
          ],
          [
            'GET',
            <>
              <code>/health</code>, <code>/healthz</code>, <code>/livez</code>,{' '}
              <code>/ready</code>, <code>/readyz</code>
            </>,
            'Liveness and readiness. Public, and exempt from the rate limit.',
          ],
          [
            'GET',
            <>
              <code>/dashboard</code>, <code>/playground</code>
            </>,
            <>
              The two web surfaces and their data endpoints. See{' '}
              <Link to="/dashboard">Dashboard</Link>.
            </>,
          ],
          [
            'GET',
            <>
              <code>/openapi.json</code>, <code>/docs</code>, <code>/redoc</code>
            </>,
            'The schema and the two schema viewers. Public — schema only, no data.',
          ],
        ]}
        caption="Read from the running server's own /openapi.json."
      />

      <Terminal
        command={`curl -s -H "Authorization: Bearer $EFFGEN_API_KEY" http://127.0.0.1:8000/whoami
curl -s -H "Authorization: Bearer $EFFGEN_API_KEY" http://127.0.0.1:8000/tools | …
curl -s -H "Authorization: Bearer $EFFGEN_API_KEY" http://127.0.0.1:8000/run \\
  -H 'Content-Type: application/json' \\
  -d '{"task":"Reply with the single word ok.","model":"openai:gpt-5-nano"}'`}
        output={`{"sub":"api-key","iss":"static-api-key","roles":["admin"],"email":""}
66 tools: agentic_search, anthropic_bash, anthropic_computer, anthropic_text_editor, arxiv, audio_transcribe ...
{"output":"ok","success":true,"metadata":{"mode":"single","iterations":1,"tool_calls":[],"execution_time":0.7195723056793213}}`}
        title="curl"
      />

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={serveOptions.map((option) => ({
          name: option.name,
          type: SERVE_TYPES[option.name],
          default: SERVE_DEFAULTS[option.name],
          description: option.description,
        }))}
        caption={
          <>
            Every flag <code>effgen serve --help</code> declares, read from the binary. Four flags is
            the whole command-line surface — everything else is an environment variable, below.
          </>
        }
      />

      <Callout type="warning" title="One worker">
        <p>
          <code>effgen serve</code> runs a single worker. For more, run the app factory under your
          own server — <code>uvicorn effgen.server.app:create_app --factory --workers 4 --port 8000</code>{' '}
          — which builds the identical application, auth and all. <Link to="/deployment">Deployment</Link>{' '}
          covers the container and cluster shapes.
        </p>
      </Callout>

      <h2>Environment</h2>

      <ParamTable
        nameLabel="Variable"
        params={serveEnv.map((setting) => ({
          name: setting.name,
          description: setting.description,
        }))}
        caption={
          <>
            The operational settings <code>effgen serve --help</code> documents in its own epilog.
          </>
        }
      />

      <h2>Authentication</h2>

      <p>Pick one posture. There is no fourth option in which requests arrive unauthenticated.</p>

      <ApiTable
        headers={['Posture', 'How', 'What the client sends']}
        rows={[
          [
            'Static key',
            <code>EFFGEN_API_KEY=&lt;key&gt; effgen serve</code>,
            <>
              <code>Authorization: Bearer &lt;key&gt;</code> or <code>X-API-Key: &lt;key&gt;</code>.
              Compared in constant time, and mapped to the roles in{' '}
              <code>EFFGEN_API_KEY_ROLES</code> (default <code>admin</code>).
            </>,
          ],
          [
            'OIDC / JWT',
            <code>EFFGEN_OIDC_ISSUER=… EFFGEN_OIDC_CLIENT_ID=…</code>,
            <>
              A bearer JWT, validated against the issuer's JWKS. The JWKS is discovered from{' '}
              <code>/.well-known/openid-configuration</code> unless{' '}
              <code>EFFGEN_OIDC_JWKS_URI</code> names it.
            </>,
          ],
          [
            'Nothing configured',
            <code>effgen serve</code>,
            'An ephemeral key is minted and printed once at startup. Restarting the server mints a new one.',
          ],
          [
            'Dev mode',
            <code>EFFGEN_DEV_MODE=1 effgen serve</code>,
            <>
              No credential. Every request is the <code>dev-user</code> principal with the{' '}
              <code>admin</code> role, and a warning is printed to stderr. Local only.
            </>,
          ],
        ]}
      />

      <p>
        A JWT is read for <code>sub</code>, <code>iss</code>, <code>aud</code>, <code>exp</code> and{' '}
        <code>roles</code>; when <code>roles</code> is absent the space-separated <code>scope</code>{' '}
        claim is used instead. Expired tokens and audience or issuer mismatches are always rejected,
        the JWKS is cached for an hour, and the <code>Authorization</code> header value is never
        written to a log or an audit record.
      </p>

      <CodeBlock
        continues
        filename="verify.py"
        code={`from effgen.server.auth import TokenPayload, verify_jwt

payload: TokenPayload = verify_jwt(
    raw_token,
    issuer="https://your-issuer/",
    client_id="your-client-id",
    jwks_uri="https://your-issuer/.well-known/jwks.json",   # optional; discovered if omitted
)
print(payload.sub, payload.roles)`}
        caption={
          <>
            The same check <code>AuthMiddleware</code> runs. After it, a handler reads{' '}
            <code>request.state.user</code>.
          </>
        }
      />

      <h3>Public routes</h3>

      <p>
        These need no credential, and a trailing slash names the same route:{' '}
        <code>/health</code>, <code>/healthz</code>, <code>/livez</code>, <code>/ready</code>,{' '}
        <code>/readyz</code>, <code>/slo</code>, <code>/openapi.json</code>, <code>/docs</code>,{' '}
        <code>/redoc</code>, and the page shells for <code>/dashboard</code> and{' '}
        <code>/playground</code> so they can load and ask for a key. Everything else — every{' '}
        <code>/v1</code> route, and the data endpoints behind those two pages — needs credentials.
      </p>

      <ApiTable
        headers={['Setting', 'What it opens']}
        rows={[
          [
            <code>EFFGEN_PUBLIC_METRICS=1</code>,
            <>
              Serves <code>/metrics</code> without auth. It is protected by default;{' '}
              <code>EFFGEN_METRICS_AUTH=1</code> forces auth back on.
            </>,
          ],
          [
            <code>EFFGEN_PUBLIC_DASHBOARD=1</code>,
            <>
              Serves <code>/dashboard/data.json</code> and <code>/dashboard/spans</code> without
              auth, for local viewing.
            </>,
          ],
          [
            <code>EFFGEN_PUBLIC_PLAYGROUND=1</code>,
            <>
              The same for <code>/playground/bootstrap</code>.
            </>,
          ],
          [
            <code>EFFGEN_CORS_ORIGINS</code>,
            'Comma-separated allowed origins. Empty by default — cross-origin is fail-closed for a backend API.',
          ],
        ]}
      />

      <h2>Roles and budgets</h2>

      <p>
        Every authenticated request carries role names, and the server resolves an effective policy
        from them. Five roles ship. A role restricts by listing what is <em>permitted</em>, so an
        empty <code>allowed_tools</code> means all tools; a role that is meant to run nothing at all
        sets <code>deny_tools</code> instead.
      </p>

      <ApiTable
        headers={['Role', 'Tools', 'Models', 'Cost cap per day']}
        rows={roles.map((role) => [
          <code>{role.name}</code>,
          role.tools === 'none' ? 'none' : 'all',
          role.models === 'all' ? 'all' : (role.models as string[]).join(', '),
          role.max_cost_per_day === 0 ? 'unlimited' : `$${role.max_cost_per_day}`,
        ])}
        caption={
          <>
            Read from <code>effgen.server.rbac.list_roles()</code> in the installed package.
          </>
        }
      />

      <CodeBlock filename="roles.py" code={`from effgen.server.rbac import PolicyDenied, list_roles, resolve_policy

for role in list_roles():
    cap = "unlimited" if role.max_cost_per_day == 0.0 else f"\${role.max_cost_per_day:g}/day"
    print(f"{role.name:14} tools={'none' if role.deny_tools else 'all':5} cap={cap}")

print()
print("reader alone       ->", resolve_policy(["reader"]).allows_tool("web_search"))
print("reader+researcher  ->", resolve_policy(["reader", "researcher"]).allows_tool("web_search"))
try:
    resolve_policy(["nobody"])
except PolicyDenied as exc:
    print("an unknown role    ->", exc)`} />

      <Terminal command="python roles.py" output={`admin          tools=all   cap=unlimited
researcher     tools=all   cap=$50/day
limited_user   tools=all   cap=$5/day
viewer         tools=none  cap=$5/day
reader         tools=none  cap=$1/day

reader alone       -> False
reader+researcher  -> True
an unknown role    -> unknown role 'nobody'; known roles: ['admin', 'limited_user', 'reader', 'researcher', 'viewer']. Request only what the principal's roles allow, or grant the role the missing permission in the policy configuration.`} />

      <p>
        Several roles resolve to the most permissive of them: a tool is allowed if any granted role
        permits it, a model likewise, and the cost cap is the maximum across the roles — where a cap
        of <code>0.0</code> means unlimited, so any unlimited role makes the effective cap unlimited.
        An <em>unknown</em> role is refused rather than ignored.
      </p>

      <CodeBlock filename="denied.py" code={`from effgen.server.rbac import PolicyDenied, resolve_policy

policy = resolve_policy(["reader"])
try:
    policy.check_tool("python_repl")
except PolicyDenied as exc:
    print(type(exc).__name__, "->", exc)`} />

      <Terminal command="python denied.py" output={`PolicyDenied -> role reader does not permit tool python_repl (roles=['reader']). Request only what the principal's roles allow, or grant the role the missing permission in the policy configuration.`} />

      <p>
        Two routes report the same thing over HTTP: <code>GET /rbac/policy</code> for the current
        principal's effective policy, and <code>GET /rbac/roles</code> for every defined role.
      </p>

      <Terminal
        command={`curl -s -H "Authorization: Bearer $EFFGEN_API_KEY" http://127.0.0.1:8000/rbac/policy`}
        output={`{"roles":["admin"],"allowed_tools":["*"],"allowed_models":["*"],"max_cost_per_day":0.0}`}
      />

      <h3>Cost caps</h3>

      <p>
        <code>max_cost_per_day</code> is USD, tracked per principal for the current UTC day. The
        model-invoking routes charge a per-call estimate — <code>EFFGEN_PER_CALL_COST_USD</code>,
        default <code>$0.01</code> — against it. Once the accrued spend meets the cap, further calls
        are answered <strong>HTTP 429</strong> with a <code>BudgetExceeded</code> detail.{' '}
        <code>EFFGEN_BUDGET_PERSIST=0</code> keeps the tally in memory only and{' '}
        <code>EFFGEN_BUDGET_DIR</code> moves the snapshot. This is the server's per-principal cap;{' '}
        <Link to="/cost">cost and budgets</Link> covers the process-wide spend gate, which is a
        different thing.
      </p>

      <h3>A policy file of your own</h3>

      <CodeBlock
        filename="policy.json"
        language="json"
        code={`[
  {
    "name": "ops",
    "allowed_tools": ["bash", "docker"],
    "allowed_models": [],
    "max_cost_per_day": 100.0
  },
  {
    "name": "readonly",
    "allowed_tools": [],
    "allowed_models": ["gpt-5-nano"],
    "max_cost_per_day": 1.0,
    "deny_tools": true
  }
]`}
        caption={
          <>
            Point <code>EFFGEN_RBAC_POLICY_FILE</code> at it to replace the built-in roles. In
            process, <code>reset_registry([Role(...), ...])</code> does the same.
          </>
        }
      />

      <h2>Rate limiting</h2>

      <p>
        <code>--rate-limit N</code> (or <code>EFFGEN_RATE_LIMIT</code>) caps requests per minute per
        client IP; <code>0</code> disables it. Health probes are always exempt. The client IP is the
        raw socket peer unless <code>--trust-proxy</code> is set, because any caller can put whatever
        it likes in <code>X-Forwarded-For</code> — turn it on only behind a proxy that overwrites
        that header.
      </p>

      <Callout type="warning" title="A 429 that is a rate limit says so">
        <p>
          A limited request is answered <code>429</code> with{' '}
          <code>code: "rate_limit_exceeded"</code> and a <code>Retry-After</code> header. A{' '}
          <code>429</code> from a budget cap carries <code>BudgetExceeded</code> instead, so the two
          are distinguishable without reading the message text.
        </p>
      </Callout>

      <h2>The audit log</h2>

      <p>
        Every request and response pair is appended as one JSON line to a daily file under{' '}
        <code>~/.effgen/audit/&lt;YYYY-MM-DD&gt;.jsonl</code>, or{' '}
        <code>EFFGEN_AUDIT_DIR</code> if you move it.
      </p>

      <ApiTable
        headers={['Field', 'Type', 'What it holds']}
        rows={[
          [<code>ts</code>, 'str', 'ISO-8601 UTC timestamp.'],
          [<code>principal</code>, 'str', <>The JWT <code>sub</code> claim, or <code>"anonymous"</code>.</>],
          [<code>roles</code>, 'list[str]', 'The roles at the time of the request.'],
          [<code>endpoint</code>, 'str', <><code>"METHOD /path"</code>.</>],
          [<code>request_summary</code>, 'str', 'Method, path and scrubbed query string.'],
          [<code>response_summary</code>, 'str', <><code>"HTTP &lt;status&gt; (&lt;content-type&gt;)"</code>.</>],
          [
            <code>outcome</code>,
            'str',
            <>
              <code>ok</code> for 1xx–3xx, <code>denied</code> for 401, 403 and a 429 budget cap,{' '}
              <code>error</code> for any other 4xx or 5xx.
            </>,
          ],
          [<code>request_id</code>, 'str', <>The <code>X-Request-ID</code> header value.</>],
          [<code>duration_ms</code>, 'float', 'Handler wall-clock time.'],
          [<code>extra</code>, 'object', 'Reserved.'],
        ]}
        caption={
          <>
            What is <em>not</em> in a record matters as much: no request or response body, no{' '}
            <code>Authorization</code> value, and query parameters that look like secrets —{' '}
            <code>key</code>, <code>token</code>, <code>secret</code>, <code>password</code>,{' '}
            <code>auth</code>, <code>api_key</code>, <code>api-key</code>, <code>bearer</code> —
            scrubbed to <code>[REDACTED]</code>.
          </>
        }
      />

      <CodeBlock
        filename="read_audit.py"
        code={`from effgen.server.audit import AuditRecord, read_audit_records

records: list[AuditRecord] = read_audit_records()          # today
yesterday = read_audit_records("2026-08-22")               # any day

for record in records[-3:]:
    print(record.outcome, record.endpoint, f"{record.duration_ms:.1f}ms")`}
        caption="The files are JSONL, so anything that tails a log file also works."
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>401</code> with <code>code: "invalid_api_key"</code>
            </>,
            'No credential, or one the server does not accept.',
            <>
              Send <code>Authorization: Bearer &lt;key&gt;</code> or <code>X-API-Key</code>. With
              nothing configured, the key you want is the ephemeral one printed at startup — and it
              changes on every restart, so set <code>EFFGEN_API_KEY</code> for anything that runs
              more than once.
            </>,
          ],
          [
            <>
              <code>403</code> from a route that worked yesterday
            </>,
            "The principal's roles no longer permit that tool or model.",
            <>
              Read <code>GET /rbac/policy</code> as that principal. A role granting no tools cannot
              be fixed by adding a second read-only role; it needs one that grants tools.
            </>,
          ],
          [
            <>
              <code>429</code> with a <code>BudgetExceeded</code> detail
            </>,
            "The principal's daily cost cap is spent.",
            <>
              Raise <code>max_cost_per_day</code> for that role, or wait for the UTC day to roll.
              This is per principal, not per process.
            </>,
          ],
          [
            'Every client shares one rate-limit bucket',
            <>
              The server is behind a proxy and <code>--trust-proxy</code> is off, so every request
              looks like it came from the proxy's IP.
            </>,
            <>
              Turn it on — but only if the proxy overwrites <code>X-Forwarded-For</code>, or callers
              can spoof their way past the limit.
            </>,
          ],
          [
            <>
              <code>/metrics</code> answers <code>401</code> to Prometheus
            </>,
            'It is protected by default.',
            <>
              Give the scrape job a bearer token, or set <code>EFFGEN_PUBLIC_METRICS=1</code>{' '}
              deliberately. <Link to="/metrics">Metrics</Link> shows both.
            </>,
          ],
          [
            'A loud warning about dev mode in the logs',
            <>
              <code>EFFGEN_DEV_MODE=1</code> is set, so auth is off entirely.
            </>,
            'Unset it. Every request is currently an admin.',
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          Two of these changed. <code>X-Forwarded-For</code> is now trusted only when you enable it,
          so a rate limit cannot be defeated by a header a caller sets — and <code>effgen serve</code>{' '}
          no longer lets uvicorn rewrite the client address behind that setting. The server also
          stays responsive during a long generation: a non-streaming completion used to block the
          event loop, so <code>/health</code> timed out for the length of the call.
        </p>
      </Callout>

      <SeeAlso paths={['/openai-api', '/deployment', '/metrics']} />
    </DocPage>
  );
}
