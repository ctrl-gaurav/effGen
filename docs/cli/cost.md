# effgen cost - Spend Dashboard & Budget Management

The `effgen cost` subcommand gives you a live view of your API spend and lets you set budget guardrails that automatically trigger provider failover.

## Subcommands

| Subcommand | Description |
|---|---|
| `effgen cost today` | Per-provider/model summary for the last 24 hours |
| `effgen cost week` | Rolling 7-day spend summary |
| `effgen cost by-provider` | Lifetime totals grouped by provider |
| `effgen cost prune` | Delete old events from the local ledger |
| `effgen cost set-budget <amount>` | Set a daily budget in USD |
| `effgen cost clear-budget` | Remove configured budget limits |

Every spend subcommand takes `--report out.html` for a self-contained spend
report (total against the daily budget, a per-provider/model table, and
cost-share charts) and `-o PATH`, whose extension chooses the format. See
[CLI developer-experience surfaces](../dx/cli.md).

For a live view that puts 24-hour spend, the daily budget and a $/hour burn
rate next to run activity, server traffic and GPU load, see
[`effgen top`](top.md).

## Viewing spend

```bash
# Today's spend (last 24 hours)
effgen cost today

# Last 7 days
effgen cost week

# All-time totals
effgen cost by-provider
```

Example output:

```
                  effGen Cost Summary - Last 24 hours
+----------+-----------------------+----------+---------------+---------------+------------+
| Provider | Model                 | Requests | Prompt Tokens |  Compl Tokens | Cost (USD) |
+----------+-----------------------+----------+---------------+---------------+------------+
| openai   | gpt-4o-mini           |       12 |        48,300 |         9,210 |  $0.008265 |
| groq     | openai/gpt-oss-20b  |       35 |       105,000 |        17,500 |  $0.000000 |
| cerebras | gpt-oss-120b           |       28 |        84,000 |        14,000 |  $0.000000 |
+----------+-----------------------+----------+---------------+---------------+------------+

Total: 75 requests  $0.008265 USD
Daily budget: [########............] $0.0083 / $1.0000 (1%)
```

## Setting budgets

```bash
# Set a $1 daily budget
effgen cost set-budget 1.0

# Or via the config subcommand
effgen config set budget.daily 1.0

# Optional rolling 30-day budget
effgen config set budget.monthly 20.0
```

Budget state is stored in `~/.effgen/budget.json`:

```json
{
  "daily": 1.0,
  "monthly": 20.0
}
```

## Budget alerts

When a budget is configured, every paid API call checks cumulative daily and monthly spend. Zero-cost calls are still allowed after a budget is exhausted so router failover can use free-tier providers.

| Threshold | Behaviour |
|---|---|
| >= 80% of budget | `UserWarning` emitted when the threshold is crossed |
| >= 100% of budget | `BudgetExceededError` raised for paid calls; router fails over |

### What a spent budget refuses

Once a budget is spent, a call is refused only when its spend is something the
budget counts. The call goes ahead when:

- the model runs in this process on this machine (a Transformers, vLLM, MLX or
  GGUF engine) — nobody bills it;
- the catalog flags the model as a free tier;
- the provider publishes no prices at all — a server you run yourself, reached
  with `base_url=` or `provider="openai_compatible"`. Its calls are recorded
  under `openai_compatible`, never priced at the rate a cloud provider charges
  for a model with the same name.

A model a pricing provider bills but the catalog does not list yet (a new
release, a fine-tuned id) is still refused.

A refusal reaches the caller as `BudgetExceededError` itself — from
`Agent.run()` whatever `raise_on_error` says, from `Agent.stream()`, and from
`Agent.run_batch()`, which stops at the first refused query rather than
finishing as a column of failed rows. The server answers it with HTTP 429 and
the error code `budget_exceeded`. The refused call never reaches the provider.

### Models with no published price

The catalog does not publish a per-token rate for every model. A call on one of
those is **allowed at any budget level** — the gate refuses spend it can
measure, and refusing on a price nobody published would block a model that may
well be free. The call's token counts are still recorded; only its cost is
unknown, so it is missing from the budget total. When a budget is configured,
the first such call in a process emits a `UserWarning` naming the model:

```
effGen budget: no published price for 'groq:allam-2-7b', so this call's spend is
not counted toward the configured budget. Run `effgen models refresh --provider
groq` to pick up a published rate.
```

The heads-up fires once per model per process, not once per call, however many
agents record calls on that model at the same moment. For a server you run
yourself the heads-up names `openai_compatible` and suggests no refresh, since
no catalog prices such a server. Everywhere a
cost is reported, such a call reads `unpriced` (or `None` in JSON) rather than
`$0.000000` — a real `$0.00` means a genuine free tier.

### Failover on budget exceed

`BudgetExceededError` is classified as *retriable* by the router's `RetryPolicy`. When the router catches it, it:

1. Excludes the provider that triggered the error.
2. Re-routes to the next best candidate (e.g., a free-tier provider like Cerebras or Groq).
3. Emits a `RouterEvent` with `reason="budget_exceeded_daily"`.

```python
from effgen.models.router import PolicyBasedRouter, RoutingContext
from effgen.models.routing.first_available import FirstAvailablePolicy
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(
    policies=[FirstAvailablePolicy()],
    failover_hops=3,
)

events = []
router.subscribe(lambda ev: events.append(ev))

def call_model(pair):
    from effgen import load_model
    m = load_model(f"{pair.provider}:{pair.model_id}")
    return m.generate("Hello")

result = router.route_and_execute(
    RoutingContext(required_capabilities={Capability.chat}),
    call_model,
)
```

If the daily budget is hit during `call_model`, the router automatically falls over to the next available provider without any changes to your calling code.

## Removing a budget

```bash
effgen cost clear-budget
```

## Where data is stored

Cost events are persisted in `~/.effgen/costs.sqlite`.  Each API call through effGen's adapters writes one row:

```
cost_events(provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp,
            calls, unpriced_calls)
```

`calls` is 1 for a single call; `unpriced_calls` is 1 when the call's model has
no published price, in which case the row adds its tokens and nothing to
`cost_usd`, and reads back as `cost_usd=None`. A ledger written by an earlier
version gains the two columns when it is opened, every existing row reading as
one priced call.

The file is written automatically when you use `CostTracker.get()` (the default singleton).

### Keeping the ledger bounded

The ledger gains one row per model call. Budget checks read a total summed in
SQLite against an index on `timestamp`, so their cost follows the window they
ask about rather than the size of the file, and a reading is reused for up to a
second and updated in place with the spend this process records, so a burst of
calls pays for one read.

The ledger holds itself to **250,000 rows**. When it passes that, the oldest
rows are folded into one row per provider, model and hour until the table is
back to 80% of the ceiling (by day, then by 30 days, only if hourly buckets do
not get it there). Folding keeps every total — spend, tokens, calls and unpriced
calls per provider and model — so `effgen cost by-provider` reports the same
lifetime numbers before and after; what it gives up is the per-call detail of
the rows it folded. A folded row carries the time of its latest call, so a
budget window counts that spend for at most one bucket longer, never shorter.
The first fold in a process logs a warning (`cost ledger: folded ...`); later
ones log at INFO.

`EFFGEN_COST_MAX_ROWS` moves the ceiling; `EFFGEN_COST_MAX_ROWS=0` turns folding
off, in which case the ledger grows without bound and logs one line naming
`effgen cost prune` once it passes 250,000 events. Pruning deletes spend
records, so it only happens when you ask:

```bash
effgen cost prune --dry-run          # what would go, keeping the last 90 days
effgen cost prune                    # keep the last 90 days
effgen cost prune --older-than-days 30
effgen cost prune --keep-rows 100000 # keep the newest 100,000 events
```

`--dry-run` and `--json` work together, so a scheduled job can report before it
deletes.

## Programmatic access

```python
from effgen.models._cost import CostTracker
from effgen.models._cost_store import SQLiteCostStore

# Global tracker (SQLite-backed)
tracker = CostTracker.get()

# Query today's events directly
store = SQLiteCostStore()          # ~/.effgen/costs.sqlite
events = store.query_today()       # last 24 hours
events = store.query_week()        # last 7 days
events = store.query_all()         # lifetime

for ev in events:
    print(f"{ev.provider}/{ev.model}: ${ev.cost_usd:.6f}")

# Totals without materialising the rows behind them — this is what the budget
# check uses, and its cost follows the window rather than the whole ledger.
store.spend_today()                # USD over the last 24 hours
store.spend_week()                 # USD over the last 7 days
store.spend_month()                # USD over the last 30 days
store.count()                      # events stored
store.prune(max_age_days=90)       # returns how many rows were deleted

# In-memory tracker (no persistence, back-compat)
mem_tracker = CostTracker(storage=None)
```
