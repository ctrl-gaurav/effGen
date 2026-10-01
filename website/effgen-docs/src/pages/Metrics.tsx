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
import { siteData } from '../siteData';

const instruments = siteData.production.metrics;

export default function Metrics() {
  return (
    <DocPage
      subtitle="The Prometheus instruments exposed at `/metrics`, their labels, and the structured log stream."
      icon={<Gauge size={48} />}
    >
      <p>
        effGen keeps {instruments.length} Prometheus instruments in process and exposes them as
        Prometheus text at <code>GET /metrics</code> on a running{' '}
        <Link to="/api-server">server</Link>, or through <code>export_metrics()</code> from any
        script. The same runs also emit a structured JSON log line per event, redacted at the
        encoder. Neither can block inference: a failed export is never a failed call.
      </p>

      <h2>Read them</h2>

      <CodeBlock filename="record.py" code={`from effgen.observability import export_metrics, record_model_call, record_tokens, record_tool_call

record_model_call(provider="openai", model="gpt-5-nano", outcome="ok", latency=0.42)
record_tool_call(tool="calculator", outcome="ok", latency=0.012)
record_tokens(provider="openai", model="gpt-5-nano", input_tokens=128, output_tokens=64, cached_tokens=32)

for line in export_metrics().splitlines():
    if line.startswith("effgen_tokens_total") or line.startswith("effgen_tool_call_latency_seconds_count"):
        print(line)`} />

      <Terminal
        command="python record.py"
        output={`effgen_tool_call_latency_seconds_count{outcome="ok",tool="calculator"} 1
effgen_tokens_total{kind="input",model="gpt-5-nano",provider="openai"} 128.0
effgen_tokens_total{kind="output",model="gpt-5-nano",provider="openai"} 64.0
effgen_tokens_total{kind="cached",model="gpt-5-nano",provider="openai"} 32.0`}
        caption={
          <>
            The instruments record themselves on every agent, model and tool call — the{' '}
            <code>record_*</code> helpers are for work effGen did not make, such as your own HTTP
            client or a model you call directly.
          </>
        }
      />

      <h2>The instruments</h2>

      <ApiTable
        headers={['Name', 'Kind', 'What it measures']}
        rows={instruments.map((instrument) => [
          <code>{instrument.name}</code>,
          instrument.kind,
          instrument.help,
        ])}
        caption={
          <>
            Read out of <code>effgen.observability.metrics</code> in the installed package, so an
            instrument the framework adds or drops cannot disagree with this table.
          </>
        }
      />

      <CodeBlock filename="instruments.py" code={`from effgen.observability import metrics

for attribute in sorted(dir(metrics)):
    instrument = getattr(metrics, attribute)
    name = getattr(instrument, "name", None)
    if isinstance(name, str) and name.startswith("effgen_"):
        declared = getattr(instrument, "label_names", None)
        labels = ", ".join(declared) if declared else "(per call)"
        print(f"{name:44} {type(instrument).__name__:16} {labels}")`} />

      <Terminal
        command="python instruments.py"
        output={`effgen_agent_iteration_latency_seconds       LabeledHistogram preset
effgen_bulkhead_active                       LabeledGauge     (per call)
effgen_bulkhead_queued                       LabeledGauge     (per call)
effgen_bulkhead_utilization_pct              LabeledGauge     (per call)
effgen_circuit_breaker_state                 LabeledGauge     (per call)
effgen_http_requests_total                   LabeledCounter   (per call)
effgen_model_call_latency_seconds            LabeledHistogram provider, model, outcome
effgen_tokens_total                          LabeledCounter   (per call)
effgen_tool_call_latency_seconds             LabeledHistogram tool, outcome`}
        caption={
          <>
            The three histograms declare their label names up front. The counters and gauges take
            their labels per observation — <code>effgen_tokens_total</code> carries{' '}
            <code>provider</code>, <code>model</code> and <code>kind</code>, the bulkhead and
            breaker gauges carry the provider, and <code>effgen_http_requests_total</code> carries
            route, method and status.
          </>
        }
      />

      <Callout type="note" title="Changed in 1.2.0">
        <p>
          <code>effgen_model_call_latency_seconds</code> observes each model call: its count is the
          number of calls and its value each call’s wait. Earlier releases observed each run’s whole
          wall time once. <code>effgen_tokens_used_total</code> takes only a run’s own calls, so a
          decomposed run’s sub-agents are not counted twice. The server’s registry also gains{' '}
          <code>effgen_run_framework_seconds</code> (a run’s time spent in effGen itself),{' '}
          <code>effgen_model_cost_usd_total</code> and <code>effgen_model_unpriced_calls_total</code>.
          Streamed runs record no Prometheus series yet.
        </p>
      </Callout>

      <h3>Histogram buckets</h3>

      <p>
        All three latency histograms share one set of bucket boundaries, in seconds:{' '}
        <code>0.05</code>, <code>0.1</code>, <code>0.25</code>, <code>0.5</code>, <code>1.0</code>,{' '}
        <code>2.5</code>, <code>5.0</code>, <code>10.0</code>, <code>20.0</code>,{' '}
        <code>30.0</code>, <code>60.0</code>, <code>+Inf</code>. They are chosen for model calls,
        which is why the top of the range is a minute rather than a second.
      </p>

      <h3>Recording your own observations</h3>

      <ParamTable
        nameLabel="Helper"
        params={[
          {
            name: 'record_model_call',
            type: '(provider, model, outcome, latency)',
            description: (
              <>
                <code>outcome</code> is <code>ok</code>, <code>error</code> or{' '}
                <code>timeout</code>.
              </>
            ),
          },
          {
            name: 'record_tool_call',
            type: '(tool, outcome, latency)',
            description: (
              <>
                <code>outcome</code> is <code>ok</code> or <code>error</code>.
              </>
            ),
          },
          {
            name: 'record_agent_iteration',
            type: '(preset, latency)',
            description: 'One complete iteration — model call, tool calls, next state.',
          },
          {
            name: 'record_tokens',
            type: '(provider, model, input_tokens, output_tokens, cached_tokens)',
            description: (
              <>
                <code>cached</code> is cache-hit prompt tokens, counted separately from{' '}
                <code>input</code>.
              </>
            ),
          },
          {
            name: 'export_metrics',
            type: '() -> str',
            description: 'The whole registry as Prometheus text, at any moment.',
          },
        ]}
        caption={
          <>
            All exported from <code>effgen.observability</code>.
          </>
        }
      />

      <h2>Scraping</h2>

      <Terminal
        command={`curl -s http://127.0.0.1:8000/metrics | grep -E '^effgen_' | head -8`}
        output={`effgen_http_requests_total{method="GET",route="/health",status="200"} 2.0
effgen_http_requests_total{method="GET",route="/v1/models",status="401"} 2.0
effgen_http_requests_total{method="GET",route="/rbac/policy",status="200"} 1.0
effgen_http_requests_total{method="GET",route="/rbac/roles",status="200"} 1.0
effgen_http_requests_total{method="GET",route="/v1/models",status="200"} 1.0`}
        title="curl"
      />

      <CodeBlock
        language="yaml"
        filename="prometheus.yml"
        code={`scrape_configs:
  - job_name: effgen
    static_configs:
      - targets: ['127.0.0.1:8000']
    metrics_path: /metrics
    # /metrics needs auth by default. Either give the job the key:
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/effgen-api-key
    # …or start the server with EFFGEN_PUBLIC_METRICS=1 and drop this block.`}
        caption={
          <>
            <code>/metrics</code> is protected by default and answers <code>401</code> to an
            anonymous scrape. <code>EFFGEN_PUBLIC_METRICS=1</code> opens it;{' '}
            <code>EFFGEN_METRICS_AUTH=1</code> forces auth back on when something else opened it.
          </>
        }
      />

      <h3>Queries worth having</h3>

      <CodeBlock
        language="promql"
        filename="queries.promql"
        code={`# p95 model-call latency
histogram_quantile(0.95, rate(effgen_model_call_latency_seconds_bucket[5m]))

# error rate per provider
sum by (provider) (rate(effgen_model_call_latency_seconds_count{outcome="error"}[5m]))
  / sum by (provider) (rate(effgen_model_call_latency_seconds_count[5m]))

# input tokens in the last hour
increase(effgen_tokens_total{kind="input"}[1h])

# a breaker that is open (2 = open, 1 = half-open, 0 = closed)
max by (provider) (effgen_circuit_breaker_state) > 0

# bulkhead saturation
effgen_bulkhead_utilization_pct > 80`}
        caption={
          <>
            The breaker and bulkhead gauges are what make{' '}
            <Link to="/reliability">the reliability primitives</Link> visible from outside the
            process.
          </>
        }
      />

      <h2>The log stream</h2>

      <p>
        Metrics say how much and how long. The log says what happened. Every critical path — agent
        runs, model calls, tool executions, router decisions — emits one JSON object per line.
      </p>

      <CodeBlock filename="logging_setup.py" code={`from effgen.observability import configure_logging, get_logger

configure_logging(level="INFO", include_src=False)

# configure_logging configures the \`effgen\` logger namespace, so a logger named
# under it inherits the handler. A bare "myapp" would propagate to the root
# logger instead and print nothing.
log = get_logger("effgen.myapp")

log.event("model.call.started", provider="openai", model="gpt-5-nano", cached_tokens=0)
log.event("checkout.failed", customer="acme", api_key="sk-EXAMPLE-not-a-real-key-000000")`} />

      <Terminal command="python logging_setup.py" output={``} />

      <ApiTable
        headers={['Field', 'Present', 'What it holds']}
        rows={[
          [<code>ts</code>, 'always', 'ISO-8601 UTC, microsecond precision.'],
          [<code>level</code>, 'always', <><code>DEBUG</code> … <code>CRITICAL</code>.</>],
          [<code>module</code>, 'always', <>The logger name — usually <code>__name__</code>.</>],
          [<code>event</code>, 'always', <>A short dotted label, e.g. <code>model.call.done</code>.</>],
          [<code>attributes</code>, 'when non-empty', 'Your keyword arguments, after redaction.'],
          [<code>trace_id</code> , 'with an active span', <>32-char hex. See <Link to="/tracing">Tracing</Link>.</>],
          [<code>span_id</code>, 'with an active span', '16-char hex.'],
          [<code>run_id</code>, 'when set', <>One <code>Agent.run()</code> invocation.</>],
          [<code>workflow_id</code>, 'when set', 'Shared across a multi-agent workflow.'],
          [<code>agent_name</code>, 'when set', 'The active agent.'],
          [<code>session_id</code>, 'when set', <>The <Link to="/sessions">session</Link>.</>],
          [<code>iteration</code>, 'when set', 'The loop iteration number.'],
          [<code>exception</code>, 'on errors', <><code>{'{type, message, file, line}'}</code>.</>],
          [<code>src</code>, 'optional', <><code>{'{file, line, func}'}</code> — <code>include_src</code>.</>],
        ]}
      />

      <ParamTable
        nameLabel="configure_logging"
        params={[
          { name: 'level', type: 'str | int', default: '"INFO"', description: 'Level for the effgen.* logger namespace.' },
          { name: 'json', type: 'bool', default: 'True', description: 'False prints plain text, which is easier to read while developing.' },
          { name: 'stream', type: 'IO | None', default: 'None', description: 'Defaults to sys.stderr.' },
          { name: 'include_src', type: 'bool', default: 'True', description: 'Add the file, line and function that emitted the line.' },
          { name: 'redact', type: 'bool', default: 'True', description: 'Apply secret redaction. Leave it on.' },
        ]}
        caption="Call it once at startup."
      />

      <h3>Helpers</h3>

      <CodeBlock
        continues
        filename="events.py"
        code={`log.event("model.call.started", model="gpt-5-nano", cached_tokens=0)

log.debug("detail", key="value")
log.info("something happened", key="value")
log.warning("soft fault", key="value")
log.error("hard fault", key="value")

log.model_event("call.done", provider="openai", model="gpt-5-nano")
log.tool_event("executed", tool="web_search", latency_ms=123)
log.agent_event("run.started", agent="my_agent", task="hello")
log.router_event("decision", policy="cost", selected_provider="openai")`}
        caption={
          <>
            The four domain helpers prefix the event name for you, so{' '}
            <code>log.model_event("call.done", …)</code> emits{' '}
            <code>model.call.done</code>.
          </>
        }
      />

      <h2>Redaction</h2>

      <p>
        Redaction happens at the encoder, on every attribute value, before anything is serialised.
        That is the reason it holds: there is no path that writes an attribute without passing
        through it, so a secret cannot escape by being logged from somewhere new.
      </p>

      <CodeBlock filename="redaction.py" code={`from effgen.observability import get_redactor

redactor = get_redactor()
print(redactor.pattern_names())
print(redactor.scrub_value("Authorization: Bearer abcdefghijklmnop"))
print(redactor.scrub_dict({"model": "gpt-5-nano", "key": "sk-EXAMPLE-not-a-real-key-000000"}))`} />

      <Terminal
        command="python redaction.py"
        output={`['anthropic_key', 'cerebras_key', 'google_key', 'hf_key', 'groq_key', 'replicate_key', 'fireworks_key', 'github_token', 'slack_token', 'aws_access_key', 'openai_key', 'bearer_token', 'slack_webhook', 'discord_webhook', 'env_secret']
Authorization: <REDACTED:bearer_token>
{'model': 'gpt-5-nano', 'key': '<REDACTED:openai_key>'}`}
        caption={
          <>
            Fifteen patterns ship, covering the provider key formats, bearer tokens, GitHub and
            Slack tokens, AWS access keys, Slack and Discord webhook URLs, and{' '}
            <code>NAME=secret</code> pairs in free text. Add your own with{' '}
            <code>get_redactor().add_pattern("custom_token", r"tok-[A-Za-z0-9]{'{32}'}")</code>.
          </>
        }
      />

      <Callout type="warning" title="Redaction is not a reason to log a secret">
        <p>
          It is a backstop for the ones that arrive by accident — inside an error message, an echoed
          request, a provider response. A credential you pass deliberately as a log attribute is
          matched only if it looks like one of the fifteen patterns.
        </p>
      </Callout>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              Prometheus reports the target down with a <code>401</code>
            </>,
            <>
              <code>/metrics</code> is protected by default.
            </>,
            <>
              Give the scrape job a bearer token, or start the server with{' '}
              <code>EFFGEN_PUBLIC_METRICS=1</code>.
            </>,
          ],
          [
            'The scrape is empty apart from HELP and TYPE lines',
            'Nothing has been recorded in this process yet. The instruments are per process, not shared.',
            'Send some traffic. Multiple workers each expose their own numbers — scrape each one.',
          ],
          [
            'Counters reset to zero',
            'The process restarted. These are in-memory instruments.',
            <>
              Query with <code>rate()</code> and <code>increase()</code>, which handle resets, rather
              than reading the raw counter.
            </>,
          ],
          [
            'Logs are plain text and unparseable by the collector',
            <>
              <code>configure_logging(json=False)</code>, or it was never called.
            </>,
            <>
              Call <code>configure_logging(level="INFO")</code> once at startup, before anything
              logs.
            </>,
          ],
          [
            <>
              No <code>trace_id</code> on any line
            </>,
            'No OpenTelemetry span is active.',
            <>
              Call <code>setup_tracing()</code> before <code>configure_logging()</code>, then run
              the work inside a span — <Link to="/tracing">Tracing and spans</Link>.
            </>,
          ],
          [
            'A cost query returns nothing for one model',
            'The model publishes no price, so its calls are counted in tokens but not in dollars.',
            <>
              That is deliberate. <Link to="/cost">Cost and budgets</Link> explains what{' '}
              <code>unpriced</code> means and how to change it.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/observability', '/tracing', '/slos']} />
    </DocPage>
  );
}
