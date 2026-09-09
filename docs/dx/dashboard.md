# effGen Local Dashboard

The effGen local dashboard is a lightweight single-page web app served by the API server at `/dashboard`.  It gives a real-time view into a running effGen deployment.

## Panels

| Panel | Description |
|-------|-------------|
| **Summary cards** | Total requests, model-call errors, average latency, session cost, and total tokens used. |
| **SLO burn rates** | Visual progress bars showing how close p99 latency, error rate, and availability are to their SLO thresholds. |
| **Request latency chart** | Rolling line chart of average latency over recent polling intervals, drawn on a canvas. |
| **By model** | One row per model *and provider*: calls, error rate, p95 latency, the dominant failure class with its remediation, tokens and cost. Model, Provider, Calls, Error rate, p95 and Cost sort on click. Spend that could not be matched to a row is stated below the table rather than spread across it. |
| **HTTP responses by status** | One chip per status code. Each chip states the code, the status class and the count as text, so it does not depend on color. |
| **Responses by route** | Requests, failures and error rate per route and method, worst-first, with a per-class breakdown. This is the panel with a denominator: it separates the routes behind a shared status code. Traffic outside the recorded route list — including the dashboard's own polling — is labelled `other`. |
| **Recent agent runs** | Table of the last 50 agent runs with model, token counts, cost, duration, and success/error badge. |
| **History** | Stored runs and saved sessions, filterable by text and status, each run opening a detail pane. |
| **Live span stream** | Real-time feed of trace spans via Server-Sent Events (SSE).  Includes a pause toggle and clear button. |
| **Run timeline** | Spans grouped by run, bars positioned by start offset and sized by duration. |
| **Agent topology** | Team and workflow executions as a node-link graph: agents (the manager marked apart) and the tools they reached as nodes, delegation, handoff and tool use as edges. Status is carried by a glyph and a text label as well as color; nodes are keyboard-focusable and open a detail panel. |
| **Model catalog** | Every model the catalog knows, with context window, output limit, price and capabilities, filterable and paged. |
| **Prometheus metrics (raw)** | Table of all registered Prometheus metric names and their current values. |

## Keyboard navigation

Both web surfaces — the dashboard and the in-browser playground at
`/playground` — share one keyboard layer.

| Key | Action |
|-----|--------|
| `Cmd/Ctrl-K` | Open the command palette |
| `?` | Show the shortcut reference |
| `↑` `↓` | Move through palette results |
| `Enter` | Run the highlighted command |
| `Esc` | Close the palette, the shortcut list, or an open detail pane |
| `Tab` | Move through the page; the first stop is a "Skip to content" link |

The palette searches four groups of commands, built from data the page has
already loaded: **Navigate** (every panel, plus the other surface), **Actions**
(switch theme, refresh, clear or pause the span stream, focus a search box),
**Runs** (stored runs, matched on task text, model, status or run id — selecting
one opens its detail), and **Models** (the catalog, matched on id, provider,
family or capability — selecting one filters the catalog table). The commands
invoked most recently lead the list when the palette opens with an empty query.

A section jump row under the header links to every panel. Selecting an entry —
from the row or the palette — scrolls to the panel and moves focus to it, so the
next `Tab` continues from there; the traversal cost does not grow with the number
of stored runs. Smooth scrolling is skipped when the viewer prefers reduced
motion.

The colour-theme choice is stored under one key, `effgen-theme`, shared by every
effGen web surface, so a theme picked on one applies on the other.

## Starting the server

```bash
# Development mode — auth disabled
EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory --host 0.0.0.0 --port 8080
```

Then open **[http://localhost:8080/dashboard](http://localhost:8080/dashboard)** in a browser.

## Data endpoint

The dashboard polls `/dashboard/data.json` every 5 seconds.  You can also query it directly:

```bash
curl http://localhost:8080/dashboard/data.json | python -m json.tool
```

Response structure:

```json
{
  "ts": "2026-05-26T12:00:00Z",
  "metrics": {
    "total_requests": 142,
    "total_errors": 3,
    "avg_latency_s": 0.7213,
    "total_tokens": 18500,
    "daily_cost_usd": 0.014200
  },
  "slo": {
    "p99_latency_burn": 0.36,
    "error_rate_burn": 0.21,
    "availability": 0.979
  },
  "recent_runs": [
    {
      "ts": "12:00:01",
      "model": "cerebras:gpt-oss-120b",
      "input_tokens": 120,
      "output_tokens": 340,
      "duration_s": 0.543,
      "cost_usd": 0.00001,
      "error": null
    }
  ],
  "by_model": [
    {
      "model": "gpt-oss-120b",
      "provider": "cerebras",
      "calls": 4,
      "errors": 1,
      "error_rate": 0.25,
      "p95_latency_s": 8.0,
      "input_tokens": 100,
      "output_tokens": 50,
      "cost_usd": 0.75,
      "outcomes": {"ok": 3, "not_found": 1},
      "top_error": "not_found",
      "top_error_hint": "Model id not found — run `effgen models list` to see ids, …"
    }
  ],
  "unattributed_cost_usd": 0.0,
  "by_status": {"200": 7, "400": 1, "404": 3},
  "by_status_detail": [
    {"status": "404", "class": "4xx", "route": "/v1/chat/completions", "method": "POST", "count": 2},
    {"status": "404", "class": "4xx", "route": "other", "method": "GET", "count": 1}
  ],
  "by_route": [
    {"route": "/v1/chat/completions", "method": "POST", "requests": 8, "errors": 4,
     "error_rate": 0.5, "by_status": {"200": 4, "400": 1, "401": 1, "404": 2}}
  ],
  "recent_spans": [...],
  "raw_metrics": {...}
}
```

Every figure in a `by_model` row is scoped to that row's `(model, provider)`
pair, so one model name served by two providers reports each provider's own
latency tail and each provider's own spend. `outcomes` tallies the recorded
outcome label verbatim; `top_error` names the most frequent failure and
`top_error_hint` is the same remediation sentence the CLI prints for that class.
`unattributed_cost_usd` holds spend from a run that could not be matched to any
row — reported apart so the cost column always sums to money actually attributed.

`by_status` keeps its `{status: count}` shape. `by_status_detail` and `by_route`
sit beside it and carry the `route` and `method` the request counter already
records, which is what separates a bad model id on `/v1/chat/completions` from a
probe of an unknown path.

## Accessibility

The two web surfaces make these guarantees, and the tests in
`tests/dx/test_web_a11y.py` drive the real pages to hold them:

- **Focus survives a refresh.** Focus on a run's disclosure button stays there
  when the five-second poll rebuilds the History table; the topology graph
  behaves the same way. If the run is no longer listed, focus moves to the
  History panel rather than to the top of the document.
- **Nothing is announced when nothing changed.** Every value the page writes
  goes through a write-if-changed rule, so an idle dashboard is silent instead
  of re-reading five cards, the SLO line and the connection status every poll.
- **A streamed answer announces once.** The playground's answer box is
  `aria-atomic="false"` and marks itself `aria-busy` for the duration of the
  stream, with deltas appended rather than the whole answer rewritten. A battle
  grid is not a live region at all — the verdict is, and it is stated once.
- **Every control boundary clears 3:1** against the surface behind it in both
  themes (WCAG 1.4.11), and every text pair clears its AA threshold.
- **Sorting is announced.** A sortable header is a button inside the `<th>`,
  exactly one header carries `aria-sort`, and the new order is spoken.

## SSE span stream

Trace spans are pushed over SSE at `/dashboard/spans`.  Each event is a JSON object:

```json
{
  "ts": "12:00:01",
  "name": "effgen.model.call cerebras:gpt-oss-120b",
  "kind": "model",
  "agent": null,
  "tool": null,
  "model": "cerebras:gpt-oss-120b",
  "duration_ms": 543.2,
  "status": "ok",
  "error": null,
  "run_id": "9f2c1d40ab77",
  "offset_ms": 12.4,
  "execution_id": "b25fed1b57d3",
  "execution_kind": "team",
  "execution_name": "newsroom",
  "parent_agent": "lead",
  "role": "worker"
}
```

`kind` is one of `agent`, `model`, `tool` or `router`, and the matching
`agent`/`tool`/`model` field names what the span timed — read those rather than
parsing `name`, which is the display label. `status` is `ok`, `error` or
`skipped`; a run that reports a failure without raising still records it here.
The `execution_*` fields group the spans of one team or workflow run.

## Topology endpoint

`GET /dashboard/topology.json?limit=6` returns recent team and workflow
executions as node-link graphs:

```json
{
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
}
```

It is built from the durable run store plus the buffered spans, so a team or
workflow run from a script or the CLI appears here too, not only work done
inside the server process. `executions` is empty when nothing multi-agent has
run yet.

## Authentication

The static SPA shell (HTML/JS/CSS) is public so the page can load and prompt for
a key. The data endpoints — `/dashboard/data.json`, `/dashboard/spans`,
`/dashboard/catalog.json`, `/dashboard/history.json` and
`/dashboard/topology.json` — require authentication by default and return a
typed `invalid_api_key` envelope without one. Set `EFFGEN_PUBLIC_DASHBOARD=1` to
open them for local viewing, and restrict access at the network/ingress level in
a shared deployment.

## Static files

The SPA consists of these files shipped with the `effgen` package:

| File | Purpose |
|------|---------|
| `effgen/dashboard/static/index.html` | Dashboard HTML shell |
| `effgen/dashboard/static/app.js` | All dashboard JavaScript (polling, charts, SSE) |
| `effgen/dashboard/static/style.css` | Dark and light theme styles |
| `effgen/webui/static/webui.js` | Command palette, shortcuts, focus handling — shared with the playground |
| `effgen/webui/static/webui.css` | Styling for the shared keyboard layer |

The two shared files live outside the dashboard's own static directory and are
served by both surfaces (`/dashboard/webui.js` and `/playground/webui.js`), each
under the access rule of the page that loads them.

Every asset is served from the package: the page references no external host, so
it renders the same in an air-gapped deployment. The charts are drawn on a
canvas and the topology graph is inline SVG built in `app.js` — no chart or
graph library is involved.

## Adding run records

To populate the "Recent Agent Runs" panel, call `record_run` from your code:

```python
from effgen.observability.run_log import record_run

record_run(
    model="cerebras:zai-glm-4.7",
    input_tokens=250,
    output_tokens=80,
    duration_s=1.12,
    cost_usd=0.000032,
)
```

The `Agent` class calls this automatically on every run (best-effort — it never
breaks a run if the dashboard machinery is unavailable), so manual calls are only
needed for custom integrations that bypass `Agent`.
