# Reliability Primitives

effGen ships four reliability primitives under
`effgen.reliability`:

| Primitive | Module | Purpose |
|-----------|--------|---------|
| Timeouts | `timeouts.py` | Explicit wall-clock limits on every I/O call |
| Retries | `retry.py` | Jittered exponential backoff with span events |
| Circuit Breaker | `circuit.py` | Fast-fail / recovery state machine per provider |
| Bulkhead | `bulkhead.py` | Concurrency cap + bounded queue per provider |

---

## Configuration

```python
from effgen.reliability import ReliabilityConfig, TimeoutConfig

cfg = ReliabilityConfig(
    timeouts=TimeoutConfig(
        model_call=60,   # seconds
        tool_call=30,
        http=20,
        agent_loop=600,
        queue=5,
    )
)
print(cfg.to_dict())
```

`ReliabilityConfig.defaults()` returns the defaults shown above.

---

## Timeouts

Every I/O boundary in effGen **must** carry an explicit timeout.
`timeout=None` is detected at test time by `audit_no_none_timeouts()`.

### Context manager

```python
from effgen.reliability import with_timeout, EffGenTimeoutError

# Sync
with with_timeout(30.0, "tool_call"):
    result = tool.run(...)

# Async
from effgen.reliability.timeouts import async_timeout
async with async_timeout(60.0, "model_call"):
    result = await adapter.generate(...)
```

### Wrapping a callable

```python
from effgen.reliability import apply_timeout

wrapped = apply_timeout(my_fn, seconds=30.0, operation="my_op")
result = wrapped(arg1, arg2)         # raises EffGenTimeoutError on overrun

# Works with async too
async_wrapped = apply_timeout(my_async_fn, seconds=60.0)
result = await async_wrapped()
```

### httpx timeout helper

```python
from effgen.reliability.timeouts import make_httpx_timeout

timeout = make_httpx_timeout(http=20.0, connect=5.0)
# Returns httpx.Timeout(timeout=20.0, connect=5.0)
```

### Error

`effgen.reliability.EffGenTimeoutError` is a subclass of the built-in
`TimeoutError`.  It carries `.operation` and `.limit` attributes.

---

## Retries

```python
from effgen.reliability import Retry, retryable, is_transient_error

policy = Retry(
    max_attempts=4,
    base_delay=0.5,     # seconds
    max_delay=30.0,
    jitter=True,
    retryable=is_transient_error,   # default
    honour_retry_after=True,        # respect HTTP 429 Retry-After header
)

@retryable(policy)
def call_api() -> str:
    ...

@retryable(policy)
async def call_api_async() -> str:
    ...
```

`is_transient_error` returns `True` for:
- `ConnectionError`, `ConnectionResetError`, `OSError`
- `TimeoutError`, `asyncio.TimeoutError`, `EffGenTimeoutError`
- Any error carrying a structured `.error_context` (every provider adapter
  wraps a real SDK failure this way, so a live 429/5xx from any provider
  classifies correctly even after wrapping — see `effgen.models.errors`)
- A raw, never-wrapped SDK exception with HTTP 429 or 5xx via `.status_code`
  or `.response.status_code`
- `httpx` network/timeout errors

A plain application exception with none of the above signals is *not*
retried — retrying a bug in your own code will not fix it.

### Delay computation

```
delay = base_delay × 2^attempt, capped at max_delay
if jitter: delay += uniform(0, base_delay)
if 429 with Retry-After header: delay = min(Retry-After, max_delay)
```

### OTel span events

Each retry attempt adds an `effgen.retry.attempt` event to the current OTel
span with attributes `effgen.retry.attempt`, `effgen.retry.reason`, and
`effgen.retry.delay_s`.

### Errors

- `RetryExhausted(attempts, last_error)` — all attempts consumed.
- Original exceptions that are not retryable are re-raised immediately.

---

## Circuit Breaker

```python
from effgen.reliability import CircuitBreaker, CircuitBreakerOpen, CircuitState

cb = CircuitBreaker(
    name="openai",
    failure_threshold=5,       # consecutive failures → OPEN
    recovery_timeout=30.0,     # seconds OPEN → HALF_OPEN
    half_open_probes=1,        # successes HALF_OPEN → CLOSED
)

if cb.is_call_permitted():
    try:
        result = call_model()
        cb.on_success()
    except Exception as exc:
        cb.on_failure(exc)
        raise
else:
    raise CircuitBreakerOpen("openai")
```

### State machine

```
CLOSED ──(threshold failures)──► OPEN
  ▲                                 │
  │                          (recovery_timeout elapsed)
  │                                 ▼
  └──(half_open_probes successes)─ HALF_OPEN
```

- **CLOSED** — all calls flow through.
- **OPEN** — `is_call_permitted()` returns `False`; stats track `.total_rejected`.
- **HALF_OPEN** — one probe call allowed; success → CLOSED, failure → OPEN.

### ProviderRegistry integration

```python
from effgen.models.registry import ProviderRegistry

cb = ProviderRegistry.get_circuit_breaker(
    "openai",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_probes=1,
)

bh = ProviderRegistry.get_bulkhead("openai", max_concurrency=10)

stats = ProviderRegistry.reliability_stats()
# {"openai": {"circuit_breaker": {...}, "bulkhead": {...}}}
```

### Per-provider registry

```python
from effgen.reliability.circuit import get_circuit_breaker

cb = get_circuit_breaker("cerebras", failure_threshold=3, recovery_timeout=15.0)
```

### An agent's tools

Every agent also keeps one breaker per tool it holds (three failures in a row,
a 60-second cooldown). It counts only failures on the tool's own side — a
connection or timeout error, an HTTP 5xx or 429, missing credentials —
recognised from the exception's class. A call whose input the tool could not
use does not count, so a model that sends bad input a few times does not get
the tool refused to it for the next minute. While a tool's breaker is open a
call to it is answered with "tool temporarily disabled due to repeated
failures", and a run whose every tool is in that state ends with
`stop_reason="tool_failed"` (see [conventions](../api/conventions.md)).

---

## Bulkhead

Prevents one misbehaving provider from starving others by capping
concurrency and queue depth.

```python
from effgen.reliability import Bulkhead, BulkheadFull

bh = Bulkhead(
    name="cerebras",
    max_concurrency=10,     # max simultaneous active calls
    queue_size=50,          # max callers waiting for a permit
    queue_timeout=5.0,      # seconds to wait before BulkheadFull
)

# Sync context manager
with bh.acquire():
    result = call_cerebras(...)

# Async context manager
async with bh.async_acquire():
    result = await call_cerebras_async(...)

# Decorators
@bh.guard()
def call_cerebras(): ...

@bh.async_guard()
async def async_call_cerebras(): ...
```

### Stats

```python
s = bh.stats()
# {
#   "name": "cerebras",
#   "max_concurrency": 10,
#   "active": 3,
#   "queued": 1,
#   "total_accepted": 120,
#   "total_rejected": 2,
#   "total_timeout": 0,
#   "utilization_pct": 30.0,
# }
```

### Per-provider registry

```python
from effgen.reliability.bulkhead import get_bulkhead

bh = get_bulkhead("groq", max_concurrency=20, queue_size=100, queue_timeout=5.0)
```

---

## Enforcement

`tests/reliability/test_timeouts.py::TestNoNoneTimeouts::test_no_none_timeouts_in_source`
scans the `effgen/` source tree for `timeout=None` keyword arguments using the
AST and **fails the build** if any are found.

Run the full reliability suite:

```bash
pytest tests/reliability/ -m reliability -v
```

---

## Error hierarchy

```
TimeoutError (built-in)
  └── effgen.reliability.EffGenTimeoutError

Exception
  ├── effgen.reliability.RetryExhausted
  ├── effgen.reliability.CircuitBreakerOpen
  └── effgen.reliability.BulkheadFull
```
