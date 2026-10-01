import { DollarSign } from 'lucide-react';
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

const costOptions = siteData.cli.command_options['cost'] ?? [];

export default function Cost() {
  return (
    <DocPage
      subtitle="What a run cost, what the day has cost, and the budget that stops it going further."
      icon={<DollarSign size={48} />}
    >
      <p>
        Every call through an effGen adapter writes one row to a local SQLite store: provider, model,
        prompt and completion tokens, cost and timestamp. <code>effgen cost</code> reads it back,{' '}
        <code>set-budget</code> puts a cap on the day, and every response carries its own{' '}
        <code>cost_usd</code>. The token counts are the provider's own, so the totals reconcile
        against an invoice rather than approximating one.
      </p>

      <h2>What has today cost?</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen cost today`} />

      <Terminal
        command="effgen cost today"
        output={`
effGen Cost Summary — Last 24 hours
--------------------------------------------------------------------------------
Provider     Model                                             Reqs   Cost (USD)
--------------------------------------------------------------------------------
openai       gpt-5-nano                                         137    $0.050975
gemini       gemini-3.1-flash-lite                               20    $0.001749
--------------------------------------------------------------------------------
TOTAL                                                           157 $   0.052724

Daily budget: $0.0527 / $1.0000 (5%)`}
        caption="Captured on the machine this page was written on, so the rows are that machine's real traffic."
      />

      <ApiTable
        headers={['Sub-command', 'Window']}
        rows={[
          [<code>effgen cost today</code>, 'The last 24 hours, per provider and model.'],
          [<code>effgen cost week</code>, 'A rolling seven days.'],
          [<code>effgen cost by-provider</code>, 'Lifetime totals, grouped by provider.'],
          [<code>effgen cost prune</code>, 'Delete old events from the local ledger.'],
          [<code>effgen cost set-budget &lt;amount&gt;</code>, 'Set the daily cap, in USD.'],
          [<code>effgen cost clear-budget</code>, 'Remove the configured limits.'],
        ]}
      />

      <Terminal command="effgen cost week" output={`
effGen Cost Summary — Last 7 days
--------------------------------------------------------------------------------
Provider     Model                                             Reqs   Cost (USD)
--------------------------------------------------------------------------------
openai       gpt-5-nano                                         317    $0.101922
openai       gpt-5.4                                             11    $0.064095
gemini       gemini-3.1-flash-lite                              120    $0.017042
openai       gpt-5-mini                                          13    $0.013606
openai       gpt-5.4-nano                                         7    $0.003396
openai       o4-mini                                              2    $0.000882
gemini       gemini-2.5-flash                                     2    $0.000706
openai       openai:gpt-5-nano                                    6    $0.000675
fireworks    accounts/fireworks/models/gpt-oss-120b               1    $0.000033
together     Qwen/Qwen3.5-9B                                      1    $0.000031
hf_inference Qwen/Qwen2.5-7B-Instruct                             1    $0.000008
gemini       gemini-2.5-flash-lite                                1    $0.000001
groq         meta-llama/llama-prompt-guard-2-22m                  1    $0.000000
openai       Qwen/Qwen2.5-14B-Instruct                        45056     unpriced
openai       Qwen/Qwen2.5-3B-Instruct                         69154     unpriced
openai       Qwen/Qwen2.5-7B-Instruct                         53705     unpriced
openai       Qwen/Qwen2.5-32B-Instruct                        60115     unpriced
openai       Qwen/Qwen2.5-1.5B-Instruct                       35735     unpriced
--------------------------------------------------------------------------------
TOTAL                                                         264248 $   0.202398`} maxLines={18} />

      <Terminal
        command="effgen cost by-provider"
        output={`
effGen Cost Summary — Lifetime
--------------------------------------------------------------------------------
Provider     Model                                             Reqs   Cost (USD)
--------------------------------------------------------------------------------
openai       all models                                       397037    $2.501868
gemini       all models                                        2271    $0.404521
groq         all models                                       11263    $0.301670
together     all models                                         984    $0.237754
fireworks    all models                                         615    $0.072131
hf_inference all models                                         172    $0.008884
anthropic    all models                                           4    $0.000620
cerebras     all models                                        6240    $0.000000
--------------------------------------------------------------------------------
TOTAL                                                         418586 $   3.527447`}
        maxLines={16}
        caption={
          <>
            Lifetime, which on a machine that has been developing against effGen for a while is a
            large number. A provider showing <code>$0.000000</code> across thousands of requests is a
            genuine free tier — it is not an unpriced model, which reads differently. See below.
          </>
        }
      />

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={costOptions.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          <>
            Every flag <code>effgen cost --help</code> declares. They apply to each sub-command, so{' '}
            <code>effgen cost week --report spend.html</code> writes the shareable report for the
            week.
          </>
        }
      />

      <Terminal command="effgen cost today --json" output={`{
  "period": "Last 24 hours",
  "period_days": 1,
  "total_requests": 157,
  "total_cost_usd": 0.05272385,
  "daily_budget_usd": 1.0,
  "rows": [
    {
      "provider": "openai",
      "model": "gpt-5-nano",
      "requests": 137,
      "prompt_tokens": 78678,
      "completion_tokens": 117603,
      "cost_usd": 0.0509751,
      "cost_label": "$0.050975"
    },
    {
      "provider": "gemini",
      "model": "gemini-3.1-flash-lite",
      "requests": 20,
      "prompt_tokens": 5387,
      "completion_tokens": 268,
      "cost_usd": 0.00174875,
      "cost_label": "$0.001749"
    }
  ]
}`} maxLines={18} />

      <h2>Budgets</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen cost set-budget 1.0
effgen cost today`} />

      <Terminal command="effgen cost set-budget 1.0 && effgen cost today" output={`✓ Daily budget set to $1.0000 USD

effGen Cost Summary — Last 24 hours
--------------------------------------------------------------------------------
Provider     Model                                             Reqs   Cost (USD)
--------------------------------------------------------------------------------
openai       gpt-5-nano                                         137    $0.050975
gemini       gemini-3.1-flash-lite                               20    $0.001749
--------------------------------------------------------------------------------
TOTAL                                                           157 $   0.052724

Daily budget: $0.0527 / $1.0000 (5%)`} />

      <p>
        The budget is stored in <code>~/.effgen/budget.json</code>, and{' '}
        <code>effgen config set budget.daily 1.0</code> and{' '}
        <code>effgen config set budget.monthly 20.0</code> write the same file. A rolling 30-day cap
        is optional; the daily one is what <code>set-budget</code> sets.
      </p>

      <ApiTable
        headers={['Spend against the budget', 'What happens']}
        rows={[
          [
            'At or above 80%',
            <>
              A <code>UserWarning</code> is emitted, once, when the threshold is crossed.
            </>,
          ],
          [
            'At or above 100%',
            <>
              A paid call raises <code>BudgetExceededError</code>. A router with a fallback chain
              treats it as retriable and fails over.
            </>,
          ],
          [
            'Calls that cost nothing, past 100%',
            <>
              Still allowed: a model running in this process, a model the catalog flags as a free
              tier, and a provider that publishes no prices, such as a server reached with{' '}
              <code>base_url=</code>. A model a pricing provider bills but the catalog does not
              list yet is still refused. Failover onto a free-tier provider is possible after the
              budget is spent.
            </>,
          ],
        ]}
      />

      <CodeBlock filename="budget_error.py" code={`from effgen.models._cost import CostTracker
from effgen.models.errors import BudgetExceededError

tracker = CostTracker(storage=None)          # in-memory: does not touch ~/.effgen
tracker.record("openai", "gpt-5-nano", prompt_tokens=1000, completion_tokens=500)
print("recorded cost:", tracker.total_cost())

try:
    raise BudgetExceededError(budget_usd=1.0, actual_usd=1.24, period="daily",
                              provider="openai", model="gpt-5-nano")
except BudgetExceededError as exc:
    print(type(exc).__name__, "->", exc)`} />

      <Terminal command="python budget_error.py" output={`recorded cost: 0.00025
BudgetExceededError -> Daily budget $1.0000 exceeded: actual=$1.2400 (provider='openai', model='gpt-5-nano'). Raised to the caller; a router configured with a fallback_chain can attempt failover to a free-tier provider instead of raising.`} />

      <p>
        The check happens twice, and the difference matters. Before a call,{' '}
        <code>check_preflight</code> refuses to start when existing spend is already at the cap — so
        that call is never billed. After a call, <code>record</code> catches the one that pushed
        spend over for the first time, which no pre-flight check could have foreseen.
      </p>

      <p>
        Since 1.2.0 a call refused before it is sent raises <code>BudgetExceededError</code> out
        of <code>run()</code> whatever <code>raise_on_error</code> says, and no request reaches the
        provider. It is not a <code>RuntimeError</code>, so catch it by name.{' '}
        <code>stream()</code> raises it, <code>run_batch()</code> raises and stops, the server
        answers HTTP 429 with <code>budget_exceeded</code>, and <code>effgen run --json</code>{' '}
        prints the error document.
      </p>

      <Callout type="note" title="Two budgets, two scopes">
        <p>
          This is the process-wide gate, configured on the machine making the calls. A running{' '}
          <Link to="/api-server">server</Link> also enforces a <em>per-principal</em>{' '}
          <code>max_cost_per_day</code> from the role policy, and answers <code>429</code> when that
          one is spent. They are independent.
        </p>
      </Callout>

      <h2>Models with no published price</h2>

      <p>
        The catalogue does not carry a per-token rate for every model. effGen refuses to invent one.
        A call on an unpriced model is <strong>allowed at any budget level</strong> — the gate
        refuses spend it can measure, and blocking on a price nobody published would block a model
        that may well be free. Its tokens are still recorded; only its cost is unknown, so it is
        missing from the budget total.
      </p>

      <CodeBlock filename="unpriced.py" code={`from effgen.models._cost import CostTracker

tracker = CostTracker(storage=None)
priced = tracker.record("openai", "gpt-5-nano", prompt_tokens=1000, completion_tokens=500)
unknown = tracker.record("openai", "not-in-the-catalog", prompt_tokens=1000, completion_tokens=500)

print("catalogued model ->", priced)
print("uncatalogued     ->", unknown)
for row in tracker.summary():
    print(row)`} />

      <Terminal
        command="python unpriced.py"
        output={`catalogued model -> 0.00025
uncatalogued     -> None
{'provider': 'openai', 'model': 'gpt-5-nano', 'requests': 1, 'prompt_tokens': 1000, 'completion_tokens': 500, 'total_tokens': 1500, 'cost_usd': 0.00025, 'unpriced_requests': 0, 'pricing': 'priced'}
{'provider': 'openai', 'model': 'not-in-the-catalog', 'requests': 1, 'prompt_tokens': 1000, 'completion_tokens': 500, 'total_tokens': 1500, 'cost_usd': None, 'unpriced_requests': 1, 'pricing': 'unpriced'}`}
        caption={
          <>
            <code>None</code>, not <code>0.0</code>. The summary row carries{' '}
            <code>unpriced_requests</code> and a <code>pricing</code> label so the distinction
            survives into anything reading the data.
          </>
        }
      />

      <p>
        With a budget configured, the first such call in a process emits a{' '}
        <code>UserWarning</code> naming the model — once per model per process, not once per call:
      </p>

      <CodeBlock
        language="text"
        filename="warning"
        code={`effGen budget: no published price for 'groq:allam-2-7b', so this call's spend is
not counted toward the configured budget. Run \`effgen models refresh --provider
groq\` to pick up a published rate.`}
        caption={
          <>
            Everywhere a cost is reported, such a call reads <code>unpriced</code> — or{' '}
            <code>null</code> in JSON — rather than <code>$0.000000</code>. A real{' '}
            <code>$0.00</code> means a genuine free tier.{' '}
            <Link to="/catalog">The model catalog</Link> covers refreshing prices.
          </>
        }
      />

      <h2>Failover on a budget</h2>

      <p>
        <code>BudgetExceededError</code> is classified retriable by the router's retry policy. When
        the router catches one it excludes the provider that raised it, re-routes to the next
        candidate, and emits a <code>RouterEvent</code> with{' '}
        <code>reason="budget_exceeded_daily"</code>. Nothing in the calling code changes.
      </p>

      <CodeBlock
        filename="failover.py"
        code={`from effgen.models.capabilities import Capability
from effgen.models.router import PolicyBasedRouter, RoutingContext
from effgen.models.routing.first_available import FirstAvailablePolicy

router = PolicyBasedRouter(policies=[FirstAvailablePolicy()], failover_hops=3)

events = []
router.subscribe(events.append)


def call_model(pair):
    from effgen import load_model
    return load_model(f"{pair.provider}:{pair.model_id}").generate("Hello")


result = router.route_and_execute(
    RoutingContext(required_capabilities={Capability.chat}),
    call_model,
)`}
        continues
        caption={
          <>
            <Link to="/routing">Model routing and fallback</Link> is the page for the policies and
            the hop limit; this is the budget's part in it.
          </>
        }
      />

      <h2>Reading the store yourself</h2>

      <CodeBlock filename="store.py" code={`from effgen.models._cost import CostTracker
from effgen.models._cost_store import SQLiteCostStore

tracker = CostTracker.get()          # the SQLite-backed singleton
store = SQLiteCostStore()            # ~/.effgen/costs.sqlite

events = store.query_today()
print(f"{len(events)} events in the last 24 hours")
for event in events[:5]:
    price = "unpriced" if event.cost_usd is None else f"\${event.cost_usd:.6f}"
    print(f"  {event.provider}/{event.model}: {price}")`} />

      <Terminal command="python store.py" output={`157 events in the last 24 hours
  gemini/gemini-3.1-flash-lite: $0.000289
  gemini/gemini-3.1-flash-lite: $0.000360
  openai/gpt-5-nano: $0.000095
  openai/gpt-5-nano: $0.000148
  openai/gpt-5-nano: $0.000407`} />

      <ApiTable
        headers={['Call', 'What it gives you']}
        rows={[
          [<code>CostTracker.get()</code>, 'The process-wide tracker, SQLite-backed. This is the one adapters write to.'],
          [
            <code>CostTracker(storage=None)</code>,
            'An in-memory tracker that persists nothing — for a test, or for measuring one piece of work in isolation.',
          ],
          [
            <code>tracker.record(provider, model, prompt_tokens, completion_tokens)</code>,
            <>
              Records one call and returns its cost, or <code>None</code> when the model is
              unpriced.
            </>,
          ],
          [<code>tracker.summary()</code>, 'One row per provider and model, with tokens, cost, and the unpriced count.'],
          [<code>tracker.total_cost()</code>, 'The total for this tracker.'],
          [
            <code>SQLiteCostStore()</code>,
            <>
              The store at <code>~/.effgen/costs.sqlite</code>. <code>query_today()</code>,{' '}
              <code>query_week()</code>, <code>query_month()</code>, <code>query_all()</code>,{' '}
              <code>query_since(ts)</code> and <code>cleanup()</code> return events;{' '}
              <code>spend_today()</code>, <code>spend_week()</code>, <code>spend_month()</code>,{' '}
              <code>spend_since(ts)</code>, <code>count()</code> and <code>count_since(ts)</code>{' '}
              return totals without reading the rows behind them; and{' '}
              <code>prune(max_age_days=..., keep_rows=...)</code> deletes old events and returns how
              many went.
            </>,
          ],
        ]}
        caption={
          <>
            A <code>CostEvent</code> carries <code>provider</code>, <code>model</code>,{' '}
            <code>prompt_tokens</code>, <code>completion_tokens</code>, <code>cost_usd</code> and{' '}
            <code>timestamp</code>.
          </>
        }
      />

      <h2>Keeping the ledger bounded</h2>

      <p>
        The ledger gains one row per model call and normal operation removes none, so it grows for
        as long as you use effGen. The budget check before each call reads a total summed in SQLite
        against an index on the timestamp, so its cost follows the window it asks about rather than
        the size of the file — but the file itself keeps growing. Once it passes 250,000 events
        effGen logs one line naming <code>effgen cost prune</code>. Nothing is deleted for you:
        these are your own spend records, and <code>effgen cost by-provider</code> reports them over
        the ledger’s whole lifetime.
      </p>

      <p>
        Since 1.2.0 the file also bounds itself. At 250,000 rows it folds its oldest rows into
        per-model totals, so every total stays exact while the row count stops growing.{' '}
        <code>EFFGEN_COST_MAX_ROWS</code> sets the ceiling, and <code>0</code> keeps every row.
        Opening a file written by an earlier release adds two columns, <code>calls</code> and{' '}
        <code>unpriced_calls</code>, and rows in <code>effgen cost --json</code> gain{' '}
        <code>unpriced_requests</code>. A write that triggers a fold waits for it. Many agents in
        one process share the ledger without queueing on it; <code>SQLiteCostStore.flush()</code>{' '}
        writes whatever is buffered.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen cost prune --dry-run           # what would go, keeping the last 90 days
effgen cost prune                     # keep the last 90 days
effgen cost prune --older-than-days 30
effgen cost prune --keep-rows 100000  # keep the newest 100,000 events`}
      />

      <p>
        <code>--dry-run</code> and <code>--json</code> work together, so a scheduled job can report
        before it deletes.
      </p>

      <h2>Per-run cost</h2>

      <p>
        A single run's cost is on the response, not only in the aggregate:{' '}
        <code>response.metadata["cost_usd"]</code>, alongside token counts and latency. Since
        1.2.0 the run also carries <code>response.ledger</code>, a <code>RunLedger</code>: its model
        and tool calls; its prompt, completion, cached and cache-write tokens; its cost; and its
        wall time split into model, tool, caller, child and framework time, per run and per
        iteration. Sub-agents, workflow nodes, team members and an agent run inside a tool attach
        to their parent once, and <code>total()</code> adds them up. A model you serve yourself
        reads <code>cost_usd=None</code> — unpriced, not free — and records its provider as{' '}
        <code>openai_compatible</code>.
      </p>

      <CodeBlock
        filename="ledger.py"
        code={`response = agent.run("What is 17 * 23?")
ledger = response.ledger

print(ledger.llm_calls, ledger.tool_calls, ledger.prompt_tokens, ledger.completion_tokens)
print(f"model {ledger.model_wait_s:.2f} s, tools {ledger.tool_wait_s:.3f} s, "
      f"framework {ledger.framework_s * 1000:.1f} ms")
print(ledger.cost_usd)`}
      />

      <p>
        Cached tokens are part of <code>prompt_tokens</code>, not added to it. They are read on
        OpenAI, Anthropic, Groq, Together, Fireworks, Cerebras and Gemini, and priced at the cached
        rate where the model catalog carries one; elsewhere they are billed in the estimate at the
        full input rate.{' '}
        <Link to="/observability">Observability</Link> shows the whole metadata block, and{' '}
        <code>effgen top</code> puts 24-hour spend, the daily budget and a dollar-per-hour burn rate
        next to live activity.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              A model reads <code>unpriced</code> rather than a number
            </>,
            'The catalogue has no per-token rate for it.',
            <>
              <code>effgen models refresh --provider &lt;name&gt;</code> picks up a published rate.
              Until then its tokens are counted and its dollars are not.
            </>,
          ],
          [
            <>
              <code>BudgetExceededError</code> on the first call of the day
            </>,
            'Yesterday\'s spend is still inside the 24-hour window — "daily" is rolling, not midnight-to-midnight.',
            <>
              Raise the cap, or wait for the window to move. <code>effgen cost today</code> shows
              exactly what is in the window.
            </>,
          ],
          [
            'The budget is set but never triggers',
            'The traffic is on free-tier or unpriced models, which do not accumulate measurable spend.',
            'Working as intended. Cap concurrency or request count if what you want to limit is volume.',
          ],
          [
            <>
              <code>effgen cost today</code> is empty after a run
            </>,
            <>
              A different <code>EFFGEN_HOME</code> was in effect, or the run used an in-memory
              tracker.
            </>,
            <>
              The store is <code>$EFFGEN_HOME/costs.sqlite</code>, default{' '}
              <code>~/.effgen</code>. A container needs that path mounted to keep history.
            </>,
          ],
          [
            'A provider shows $0.000000 across thousands of requests',
            'A genuine free tier. Zero is a measured price, not a missing one.',
            <>
              Compare with an unpriced model, which reads <code>unpriced</code>. The two are
              deliberately different.
            </>,
          ],
          [
            'Costs differ from the provider invoice',
            <>
              Some calls are unpriced, or the catalogue rate is behind the provider's current
              pricing.
            </>,
            <>
              Refresh the catalogue. The token counts are the provider's own and are the reliable
              half of the comparison.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          Reporting no cost for a model that publishes no rate is new. Earlier releases printed{' '}
          <code>$0.000000</code>, which was indistinguishable from a free tier and quietly made
          budget totals wrong. Streamed calls now report their cost and tokens on every provider,
          and per-model spend adds up to the total.
        </p>
      </Callout>

      <SeeAlso paths={['/catalog', '/routing', '/observability']} />
    </DocPage>
  );
}
