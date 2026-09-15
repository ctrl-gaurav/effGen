# effGen Metrics Reference

effGen exposes Prometheus-compatible metrics via the `/metrics` HTTP endpoint and the `effgen.observability.metrics` module.

---

## Histograms

All latency histograms use logarithmic-ish bucket boundaries:

```
0.05s  0.1s  0.25s  0.5s  1.0s  2.5s  5.0s  10.0s  20.0s  30.0s  60.0s  +Inf
```

### `effgen_model_call_latency_seconds`

**Type:** Histogram  
**Labels:** `provider`, `model`, `outcome`

Measures one model call: the time inside the provider request, retries and the adapter's back-off between them included. An agent run records one observation per model call it made, labelled with that call's outcome, so the series' count is the number of model calls, not the number of runs. The same figures are on each run's ledger (`response.ledger`, `metadata["ledger"]["calls"]`).

| Label | Values |
|---|---|
| `provider` | `cerebras`, `openai`, `gemini`, `groq`, … |
| `model` | model identifier, e.g. `gpt-oss-120b`, `gpt-4o-mini` |
| `outcome` | `ok` (success), `error` (exception), `timeout` |

**Recording:**
```python
from effgen.observability.metrics import record_model_call
import time

t0 = time.perf_counter()
try:
    response = model.call(...)
    record_model_call(provider="cerebras", model="gpt-oss-120b",
                      outcome="ok", latency=time.perf_counter() - t0)
except TimeoutError:
    record_model_call(provider="cerebras", model="gpt-oss-120b",
                      outcome="timeout", latency=time.perf_counter() - t0)
    raise
except Exception:
    record_model_call(provider="cerebras", model="gpt-oss-120b",
                      outcome="error", latency=time.perf_counter() - t0)
    raise
```

---

### `effgen_run_framework_seconds`

**Type:** Histogram  
**Labels:** `agent`

Time per agent run spent in effGen itself: the run's wall time less the time it waited on model calls, on tool executions and on the caller (a stream's consumer, a human approval). One observation per run. Buckets run from 0.5 ms to 5 s. The per-run and per-iteration figures are on the run's ledger (`response.ledger.framework_s`, `response.ledger.steps`).

```python
from effgen.observability.metrics import record_run_framework
record_run_framework(agent="support-bot", seconds=0.004)
```

Cached prompt tokens a provider reports are counted in `effgen_tokens_total{kind="cached"}`.

---

### `effgen_tool_call_latency_seconds`

**Type:** Histogram  
**Labels:** `tool`, `outcome`

Measures tool execution duration, from the agent dispatcher handing off to the tool to the tool returning a result.

| Label | Values |
|---|---|
| `tool` | tool name, e.g. `calculator`, `web_search`, `python_repl` |
| `outcome` | `ok`, `error` |

**Recording:**
```python
from effgen.observability.metrics import record_tool_call
record_tool_call(tool="calculator", outcome="ok", latency=0.012)
```

---

### `effgen_agent_iteration_latency_seconds`

**Type:** Histogram  
**Labels:** `preset`

Measures one complete agent iteration: receiving the user message, calling the model, executing any tool calls, and returning the next agent state.

| Label | Values |
|---|---|
| `preset` | preset/agent profile name, e.g. `default`, `code`, `research` |

**Recording:**
```python
from effgen.observability.metrics import record_agent_iteration
record_agent_iteration(preset="default", latency=2.5)
```

---

## Counters

### `effgen_tokens_total`

**Type:** Counter  
**Labels:** `provider`, `model`, `kind`

Cumulative token count since process start.

| Label | Values |
|---|---|
| `provider` | provider identifier |
| `model` | model identifier |
| `kind` | `input` (prompt tokens), `output` (completion tokens), `cached` (cache-hit prompt tokens) |

**Recording:**
```python
from effgen.observability.metrics import record_tokens
record_tokens(
    provider="cerebras",
    model="gpt-oss-120b",
    input_tokens=128,
    output_tokens=64,
    cached_tokens=32,
)
```

---

## HTTP Endpoint

The `/metrics` endpoint (requires `EFFGEN_API_KEY` auth if set) returns Prometheus text format:

```
GET /metrics HTTP/1.1
Host: localhost:8000
Accept: text/plain

HTTP/1.1 200 OK
Content-Type: text/plain; version=0.0.4; charset=utf-8

# HELP effgen_model_call_latency_seconds Latency of model (LLM) calls in seconds
# TYPE effgen_model_call_latency_seconds histogram
effgen_model_call_latency_seconds_bucket{model="gpt-oss-120b",outcome="ok",provider="cerebras",le="0.05"} 0
...
```

### Example Prometheus scrape config

```yaml
scrape_configs:
  - job_name: effgen
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    # If EFFGEN_API_KEY is set, add bearer auth:
    # authorization:
    #   credentials: <your-api-key>
```

---

## Programmatic Usage

```python
from effgen.observability import (
    record_model_call, record_tool_call,
    record_agent_iteration, record_tokens,
    export_metrics,
)

# Record observations inline:
record_model_call(provider="cerebras", model="gpt-oss-120b",
                  outcome="ok", latency=0.42)

# Export Prometheus text format at any time:
prometheus_text = export_metrics()
```

---

## Grafana Quick-Start Queries

```promql
# p95 model call latency (1 min rate):
histogram_quantile(0.95,
  rate(effgen_model_call_latency_seconds_bucket[1m])
)

# Error rate per provider:
sum by (provider) (
  rate(effgen_model_call_latency_seconds_count{outcome="error"}[5m])
)
/
sum by (provider) (
  rate(effgen_model_call_latency_seconds_count[5m])
)

# Total input tokens consumed in last hour:
increase(effgen_tokens_total{kind="input"}[1h])
```
