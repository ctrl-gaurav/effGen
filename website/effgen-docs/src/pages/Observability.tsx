import { Eye } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  QuickLinks,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData } from '../siteData';

const instrumentCount = siteData.production.metrics.length;

export default function Observability() {
  return (
    <DocPage
      subtitle="What a run records, where it goes, and how to look at it."
      icon={<Eye size={48} />}
    >
      <p>
        A run leaves five kinds of trace behind: the response's own metadata, a durable record in the
        run history, a structured log line per event, {instrumentCount} Prometheus instruments, and
        an OpenTelemetry span tree. None of it blocks the work — a failed export is never a failed
        call — and none of it needs a server: the history and the log are on disk from the first{' '}
        <code>agent.run()</code>.
      </p>

      <h2>What one run already tells you</h2>

      <CodeBlock filename="metadata.py" code={`from effgen import Agent, AgentConfig
from effgen.tools import get_registry

calculator = get_registry().get_tool_sync("calculator")

with Agent(AgentConfig(model="gpt-5-nano", provider="openai", tools=[calculator])) as agent:
    response = agent.run("What is 127 * 43? Use the calculator.")

print("success   :", response.success)
print("tool calls:", response.tool_calls.total)
for call in response.tool_calls:
    print(f"  {call.name}({call.arguments}) -> {call.result} in {call.duration:.2f}s")
print("cost      :", response.metadata.get("cost_usd"))
print("tokens    :", response.metadata.get("total_tokens"))
print("latency   :", response.metadata.get("latency_ms"))`} />

      <Terminal
        command="python metadata.py"
        output={`success   : True
tool calls: 1
  calculator({"expression": "127 * 43"}) -> 5461 in 0.00s
cost      : 7.26e-05
tokens    : 374
latency   : 2952.1`}
        caption={
          <>
            No configuration was needed for any of that. <code>tool_calls</code> carries the calls
            themselves — name, arguments, result, duration, and an error where one failed — so a run
            that recovered from a failing tool still says so. <Link to="/agents">Agents</Link> has
            the whole response surface.
          </>
        }
      />

      <h2>The five signals</h2>

      <ApiTable
        headers={['Signal', 'Where it lives', 'Answers']}
        rows={[
          [
            'Response metadata',
            <>
              <code>response.metadata</code>, in memory
            </>,
            'What did this call cost, how long did it take, what did it do?',
          ],
          [
            'Run history',
            <>
              <code>$EFFGEN_HOME/runs</code>, JSONL per day
            </>,
            'What has this machine run, and how did it go? Readable from another process.',
          ],
          [
            'Structured logs',
            'stderr, JSON per line',
            'What happened inside, in order, with secrets redacted.',
          ],
          [
            'Metrics',
            <>
              In process, and <code>GET /metrics</code>
            </>,
            'How much, how fast, how often — aggregated across calls.',
          ],
          [
            'Traces',
            'An OTLP collector',
            'Where the time went in one specific run, as a tree.',
          ],
        ]}
      />

      <CodeBlock filename="signals.py" code={`from effgen.observability import export_metrics, get_logger, get_slo_tracker
from effgen.observability.slo import SLO

log = get_logger("myapp")
tracker = get_slo_tracker()
tracker.register(SLO("model_call_success", 99.0, 3600))

log.event("checkout.started", customer="acme")
tracker.record("model_call_success", ok=True)

print("registered SLOs:", tracker.list_slos())
print("metrics text is", len(export_metrics()), "bytes")`} />

      <Terminal command="python signals.py" output={`registered SLOs: ['model_call_success']
metrics text is 2453 bytes`} />

      <QuickLinks
        links={[
          {
            icon: '📊',
            title: 'Metrics',
            description: 'The Prometheus instruments and the log line schema.',
            path: '/metrics',
          },
          {
            icon: '🌲',
            title: 'Tracing and spans',
            description: 'Span hierarchy, samplers, attributes and exporters.',
            path: '/tracing',
          },
          {
            icon: '🎯',
            title: 'SLOs and alerting',
            description: 'Error budgets, burn rates and the alert rule pack.',
            path: '/slos',
          },
          {
            icon: '💵',
            title: 'Cost and budgets',
            description: 'What a run cost and the cap that stops the day.',
            path: '/cost',
          },
          {
            icon: '🌊',
            title: 'Load testing and chaos',
            description: 'Throughput under load, and deliberate failure.',
            path: '/loadtest',
          },
          {
            icon: '🛡️',
            title: 'Reliability',
            description: 'Timeouts, retries, circuit breakers and bulkheads.',
            path: '/reliability',
          },
        ]}
      />

      <h2>The run history</h2>

      <p>
        Every run appends a record to <code>$EFFGEN_HOME/runs</code> — one JSONL file per day,{' '}
        <code>~/.effgen/runs</code> by default. It is durable and cross-process: a run from a script
        is visible to the command line, the dashboard and another Python process.
      </p>

      <CodeBlock filename="runs.py" code={`from effgen import Agent, AgentConfig
from effgen.observability.run_log import read_runs

with Agent(AgentConfig(model="gpt-5-nano", provider="openai")) as agent:
    agent.run("Reply with the single word ok.")

for run in read_runs(limit=1):
    print(run["run_id"], run["model"], run["status"], f"{run['duration_s']:.2f}s")
    print(run["task"], "->", run["output"])`} />

      <Terminal command="python runs.py" output={`b01c1e734b7b gpt-5-nano ok 3.21s
Reply with the single word ok. -> ok`} />

      <ApiTable
        headers={['Field', 'What it holds']}
        rows={[
          [<code>ts</code>, 'When the run finished.'],
          [<code>run_id</code>, 'The short id, which also appears in the log lines and the trace.'],
          [<code>status</code>, <>How it ended — <code>ok</code> or a failure.</>],
          [
            <>
              <code>model</code>, <code>provider</code>, <code>agent</code>
            </>,
            'What ran it.',
          ],
          [<code>session_id</code>, <>The <Link to="/sessions">session</Link>, when the run had one.</>],
          [
            <>
              <code>execution_id</code>, <code>execution_kind</code>, <code>execution_name</code>
            </>,
            'The team or workflow execution this run belonged to.',
          ],
          [
            <>
              <code>parent_agent</code>, <code>role</code>
            </>,
            'Who delegated the work, and what part this agent played.',
          ],
          [
            <>
              <code>task</code>, <code>output</code>
            </>,
            'What was asked and what came back, truncated for the record.',
          ],
          [
            <>
              <code>input_tokens</code>, <code>output_tokens</code>, <code>duration_s</code>,{' '}
              <code>cost_usd</code>
            </>,
            <>
              The measurements. <code>cost_usd</code> is <code>null</code> for an{' '}
              <Link to="/cost">unpriced model</Link>.
            </>,
          ],
          [<code>error</code>, 'The structured error, when the run failed.'],
        ]}
        caption={
          <>
            <code>read_runs()</code> filters on <code>status</code>, <code>model</code>,{' '}
            <code>search</code>, <code>since</code>, <code>until</code>, <code>session_id</code> and{' '}
            <code>execution_id</code>, and takes a <code>limit</code>. <code>get_run(id)</code> reads
            one, <code>cleanup_runs()</code> deletes old ones.
          </>
        }
      />

      <ApiTable
        headers={['Variable', 'Default', 'What it does']}
        rows={[
          [<code>EFFGEN_HOME</code>, <code>~/.effgen</code>, 'Where runs, sessions, costs and budgets live.'],
          [
            <code>EFFGEN_RUN_HISTORY</code>,
            <code>1</code>,
            <>
              <code>0</code> keeps history in memory only — nothing is written to disk.
            </>,
          ],
          [<code>EFFGEN_RUN_HISTORY_MAX_DAYS</code>, <code>30</code>, 'Retention, in days.'],
        ]}
      />

      <h3>One execution's runs</h3>

      <p>
        A team run issues an execution id, and every member's record carries it along with the agent
        that delegated the work and the role it played. That is what turns a scatter of run records
        back into one team run — in the history, in traces and in the dashboard's topology panel.
      </p>

      <CodeBlock filename="execution.py" code={`from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern
from effgen.observability import run_log

manager = Agent(AgentConfig(name="manager", model="gpt-5-nano", provider="openai"))
team = [
    Agent(AgentConfig(name="billing", model="gpt-5-nano", provider="openai",
                      enable_sub_agents=False,
                      system_prompt="You handle refunds and billing only. Answer in one sentence.")),
    Agent(AgentConfig(name="tech", model="gpt-5-nano", provider="openai",
                      enable_sub_agents=False,
                      system_prompt="You handle login and app bugs only. Answer in one sentence.")),
]

orchestrator = MultiAgentOrchestrator()
orchestrator.create_team("support", team, pattern=OrchestrationPattern.HIERARCHICAL,
                         manager_agent=manager)
result = orchestrator.assign_task("I was charged twice for order ORD-7788.", "support")

execution_id = result.metadata["execution_id"]
print("execution:", execution_id)
for row in run_log.read_runs(execution_id=execution_id):
    print(f"  {row['agent']:9} {row.get('role', ''):8} <- {row.get('parent_agent', '')!s:9} {row['status']}")`} />

      <Terminal
        command="python execution.py"
        output={`execution: 3f4a1f6cf8d0
  manager   manager  <- None      ok
  tech      worker   <- manager   ok
  billing   worker   <- manager   ok
  tech      worker   <- manager   ok
  tech      worker   <- manager   ok
  tech      worker   <- manager   ok
  billing   worker   <- manager   ok
  billing   worker   <- manager   ok
  tech      worker   <- manager   ok
  billing   worker   <- manager   ok
  manager   manager  <- None      ok`}
        caption={
          <>
            One hierarchical team, eleven runs: the manager's own two, and nine subtasks dispatched
            by name to the two workers. <Link to="/multi-agent">Multi-agent teams</Link> is where the
            team itself is built.
          </>
        }
      />

      <CodeBlock filename="executions.py" code={`from effgen.observability import run_log

for execution in run_log.read_executions(limit=2):
    print(execution["id"], execution["kind"], execution["name"], len(execution["runs"]), "runs")`} />

      <Terminal
        command="python executions.py"
        output={`3f4a1f6cf8d0 team support 11 runs
806087c8bd04 team desk 2 runs`}
        caption={
          <>
            <code>read_executions()</code> reconstructs executions from the stored runs, so a team or{' '}
            <Link to="/workflows">workflow</Link> run from a script or the command line is visible to
            a reader in another process.
          </>
        }
      />

      <h2>Logs</h2>

      <p>
        <code>configure_logging()</code> once at startup turns the whole <code>effgen</code> logger
        namespace into structured JSON on stderr. Nothing else has to change: the agent loop, the
        adapters, the tools and the router already emit events.
      </p>

      <CodeBlock filename="logging_run.py" code={`from effgen import Agent, AgentConfig
from effgen.observability import configure_logging
from effgen.tools import get_registry

configure_logging(level="INFO", include_src=False)

calculator = get_registry().get_tool_sync("calculator")
with Agent(AgentConfig(model="gpt-5-nano", provider="openai", tools=[calculator])) as agent:
    print(agent.run("What is 127 * 43? Use the calculator.").text)`} />

      <Terminal
        command="python logging_run.py"
        output={`{"ts": "2026-08-24T03:51:08.995847+00:00", "level": "INFO", "module": "effgen.models.model_loader", "event": "Loading model: gpt-5-nano"}
{"ts": "2026-08-24T03:51:08.995979+00:00", "level": "INFO", "module": "effgen.models.model_loader", "event": "Loading OpenAI model: gpt-5-nano"}
{"ts": "2026-08-24T03:51:09.308029+00:00", "level": "INFO", "module": "effgen.models.openai_adapter", "event": "Initializing OpenAI client for model 'gpt-5-nano' against the OpenAI API..."}
{"ts": "2026-08-24T03:51:10.137692+00:00", "level": "INFO", "module": "effgen.models.openai_adapter", "event": "OpenAI client initialized for 'gpt-5-nano' against the OpenAI API"}
{"ts": "2026-08-24T03:51:10.137887+00:00", "level": "INFO", "module": "effgen.models.model_loader", "event": "Validating model..."}
{"ts": "2026-08-24T03:51:10.359659+00:00", "level": "INFO", "module": "effgen.models.model_loader", "event": "Token counting works: 'Hello, world!' = 4 tokens"}
{"ts": "2026-08-24T03:51:10.359820+00:00", "level": "INFO", "module": "effgen.models.model_loader", "event": "Context length: 1047576 tokens"}
{"ts": "2026-08-24T03:51:10.359870+00:00", "level": "INFO", "module": "effgen.models.model_loader", "event": "Model validation passed"}
{"ts": "2026-08-24T03:51:10.360044+00:00", "level": "INFO", "module": "effgen.core.tool_calling", "event": "Auto-detected native tool calling support, using hybrid strategy"}
{"ts": "2026-08-24T03:51:10.360098+00:00", "level": "INFO", "module": "effgen.core.agent", "event": "Tool calling strategy: hybrid"}
{"ts": "2026-08-24T03:51:10.360776+00:00", "level": "INFO", "module": "effgen.core.agent", "event": "[gpt-5-nano] task_start", "run_id": "438eff4fde2f", "agent_name": "gpt-5-nano", "attributes": {"event_type": "agent", "event": "task_start", "task": "What is 127 * 43? Use the calculator.", "mode": "single"}}
{"ts": "2026-08-24T03:51:10.360933+00:00", "level": "INFO", "module": "effgen.core.agent", "event": "agent.run.started", "attributes": {"agent": "gpt-5-nano", "task": "What is 127 * 43? Use the calculator.", "mode": "single", "run_id": "438eff4fde2f"}}
{"ts": "2026-08-24T03:51:12.924759+00:00", "level": "INFO", "module": "effgen.models.openai_adapter.usage", "event": "[gpt-5-nano] input=220tok (cached=0) output=158tok | call=$0.000074 session=$0.000074"}
{"ts": "2026-08-24T03:51:13.848973+00:00", "level": "INFO", "module": "effgen.core.agent_react", "event": "agent.iteration.generate", "attributes": {"iteration": 1, "tokens": 158, "model": "gpt-5-nano"}}
{"ts": "2026-08-24T03:51:13.849178+00:00", "level": "INFO", "module": "effgen.core.agent_react", "event": "[Iteration 1] Raw model output: ..."}
{"ts": "2026-08-24T03:51:13.849265+00:00", "level": "INFO", "module": "effgen.core.agent_react", "event": "[Iteration 1] Parsed - Action: calculator, Input: {\\"expression\\": \\"127 * 43\\", \\"operation\\": \\"calculate\\"}, Final: None"}
{"ts": "2026-08-24T03:51:13.852195+00:00", "level": "INFO", "module": "effgen.core.agent_tool_execution", "event": "Tool 'calculator' executed successfully"}
{"ts": "2026-08-24T03:51:13.852320+00:00", "level": "INFO", "module": "effgen.core.agent_react", "event": "tool.executed", "attributes": {"tool": "calculator", "latency_ms": 2.9}}
{"ts": "2026-08-24T03:51:13.852403+00:00", "level": "INFO", "module": "effgen.core.agent_react", "event": "Tool result added to scratchpad: 5461..."}
{"ts": "2026-08-24T03:51:13.852571+00:00", "level": "INFO", "module": "effgen.core.agent_react", "event": "Returning direct calculator result for simple arithmetic task"}
{"ts": "2026-08-24T03:51:13.852976+00:00", "level": "INFO", "module": "effgen.core.agent", "event": "[gpt-5-nano] task_complete", "run_id": "438eff4fde2f", "agent_name": "gpt-5-nano", "attributes": {"event_type": "agent", "event": "task_complete", "latency": 3.4923181533813477, "tokens": 378, "tool_calls": "ToolCallList(['calculator'], total=1)", "success": true}}
{"ts": "2026-08-24T03:51:13.853093+00:00", "level": "INFO", "module": "effgen.core.agent", "event": "agent.run.completed", "attributes": {"agent": "gpt-5-nano", "run_id": "438eff4fde2f", "latency_ms": 3492.3, "tokens": 378, "tool_calls": ["calculator('{\\"expression\\": \\"127 * 43\\", \\"operation\\": \\"calculate\\"}') -> ok"], "success": true}}`}
        maxLines={16}
        title="stderr"
        caption={
          <>
            The lines carry the <code>run_id</code> that <code>read_runs()</code> reports, so a log
            line, a history record and a trace all name the same run. The schema and the redaction
            rules are on <Link to="/metrics">the metrics page</Link>.
          </>
        }
      />

      <Callout type="warning" title="Logs go to stderr, not stdout">
        <p>
          <code>python script.py &gt; out.txt</code> captures the answer and not the log.
          Redirect both — <code>&gt; out.txt 2&gt; log.jsonl</code> — or pass{' '}
          <code>stream=</code> to <code>configure_logging</code>.
        </p>
      </Callout>

      <h2>On a server</h2>

      <p>
        A running <Link to="/api-server">server</Link> exposes the same signals over HTTP:{' '}
        <code>GET /metrics</code> for the Prometheus text, <code>GET /slo</code> for registered
        error budgets, <code>GET /dashboard/data.json</code> for measured percentiles and spend, and{' '}
        <code>GET /dashboard/spans</code> as a live span stream. Every request also appends an{' '}
        <Link to="/api-server">audit record</Link>, which is a different thing from a run record —
        it is about the HTTP call, not the agent.
      </p>

      <p>
        <code>effgen top</code> reads a running server's dashboard data and puts run activity,
        traffic, per-model latency, spend and GPU load on one screen.{' '}
        <Link to="/dashboard">The dashboard</Link> is the same data in a browser.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Nothing is logged at all',
            <>
              <code>configure_logging()</code> was never called, or your logger sits outside the{' '}
              <code>effgen</code> namespace.
            </>,
            <>
              Call it once at startup, and name your logger{' '}
              <code>get_logger("effgen.myapp")</code> — a bare <code>"myapp"</code> propagates to the
              root logger, which has no handler, and prints nothing.
            </>,
          ],
          [
            <>
              <code>read_runs()</code> is empty after a run
            </>,
            <>
              <code>EFFGEN_RUN_HISTORY=0</code>, or a different <code>EFFGEN_HOME</code> was in
              effect.
            </>,
            <>
              Check both. A container needs <code>$EFFGEN_HOME</code> on a mounted volume for the
              history to outlive it.
            </>,
          ],
          [
            <>
              <code>cost_usd</code> is <code>null</code> on a run that certainly cost something
            </>,
            'That model publishes no per-token rate.',
            <>
              Deliberate — <Link to="/cost">Cost and budgets</Link> explains why an absent cost beats
              an invented zero. The token counts are still there.
            </>,
          ],
          [
            <>
              <code>execution_id</code> is <code>None</code> on every record
            </>,
            'The runs were single-agent. Only a team or a workflow issues one.',
            <>
              <Link to="/multi-agent">Multi-agent teams</Link> and{' '}
              <Link to="/workflows">workflows</Link> are what group runs.
            </>,
          ],
          [
            'Metrics are empty on one worker and full on another',
            'The instruments are per process. Each worker counts its own.',
            'Scrape every worker and aggregate in Prometheus. The run history, being on disk, is already shared.',
          ],
          [
            'A run succeeded but a tool inside it failed',
            'The agent recovered. Success is about the run, not about every step.',
            <>
              <code>response.tool_calls.failed</code> lists them, and the log line and the tool span
              both carry the error.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          Durable run and session history is new. Runs are keyed by the same id their trace spans
          carry, and a run from the command line, a script and the server share one history that
          survives a restart — which is what makes <code>read_runs()</code> useful from a process
          that did not make the call.
        </p>
      </Callout>

      <SeeAlso paths={['/metrics', '/tracing', '/slos']} />
    </DocPage>
  );
}
