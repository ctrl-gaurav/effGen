import { Activity } from 'lucide-react';
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

export default function Reliability() {
  return (
    <DocPage
      subtitle="Retries, timeouts, circuit breakers and what the framework does when a provider is down."
      icon={<Activity size={48} />}
    >
      <p>
        Four primitives in <code>effgen.reliability</code> stand between your agent and a provider
        having a bad day: a timeout on every I/O call, a retry policy that knows which failures are
        worth retrying, a circuit breaker that stops hammering a dead endpoint, and a bulkhead so one
        slow provider cannot consume every worker. They are on by default and configurable, and each
        one is usable on its own.
      </p>

      <h2>The defaults</h2>

      <CodeBlock filename="defaults.py" code={`from effgen.reliability import ReliabilityConfig

print(ReliabilityConfig.defaults().to_dict())`} />

      <Terminal
        command="python defaults.py"
        output={`{'timeouts': {'model_call': 60.0, 'tool_call': 30.0, 'http': 20.0, 'agent_loop': 600.0, 'queue': 5.0}, 'retry': {'max_attempts': 3, 'base_delay': 0.5, 'max_delay': 30.0, 'jitter': True, 'retryable_status_codes': [429, 500, 502, 503, 504]}, 'circuit_breaker': {'failure_threshold': 5, 'recovery_timeout': 30.0, 'half_open_probes': 1}, 'bulkhead': {'max_concurrency': 20, 'queue_size': 100}}`}
        caption={`Run against effGen 1.0.0. Everything below is one of these numbers, said longer.`}
      />

      <ApiTable
        headers={['Primitive', 'Import', 'Stops']}
        rows={[
          [
            'Timeouts',
            <code>with_timeout, apply_timeout</code>,
            'A call that will never come back from holding a worker forever.',
          ],
          [
            'Retries',
            <code>Retry, retryable</code>,
            'A transient blip from being reported to the caller as a failure.',
          ],
          [
            'Circuit breaker',
            <code>CircuitBreaker</code>,
            'A queue of calls piling up against a provider that is down.',
          ],
          [
            'Bulkhead',
            <code>Bulkhead</code>,
            'One slow provider from starving every other request in the process.',
          ],
        ]}
        caption={<>All four are exported from <code>effgen.reliability</code>.</>}
      />

      <h2>Timeouts</h2>
      <p>
        Every I/O boundary in effGen carries an explicit limit — <code>timeout=None</code> is treated
        as a defect and a test fails the build on one. The context manager form wraps a block; the
        wrapper form wraps a callable, and both work for sync and async code.
      </p>

      <CodeBlock filename="timeouts.py" code={`import time

from effgen.reliability import EffGenTimeoutError, apply_timeout, with_timeout


def slow():
    time.sleep(2)


try:
    with with_timeout(0.2, "tool_call"):
        slow()
except EffGenTimeoutError as exc:
    print(type(exc).__name__, "->", exc)
    print("operation:", exc.operation, "limit:", exc.limit)
    print("is a TimeoutError:", isinstance(exc, TimeoutError))

wrapped = apply_timeout(slow, seconds=0.2, operation="my_op")
try:
    wrapped()
except EffGenTimeoutError as exc:
    print("wrapped   ->", exc)`} />

      <Terminal
        command="python timeouts.py"
        output={`TimeoutError -> effGen operation 'tool_call' exceeded 0.2s timeout. Raise the timeout for this operation, or reduce what the call asks for (a smaller prompt, fewer tools, a faster model).
operation: tool_call limit: 0.2
is a TimeoutError: True
wrapped   -> effGen operation 'my_op' exceeded 0.2s timeout. Raise the timeout for this operation, or reduce what the call asks for (a smaller prompt, fewer tools, a faster model).`}
        caption={
          <>
            <code>EffGenTimeoutError</code> is an alias for a subclass of the built-in{' '}
            <code>TimeoutError</code>, which is why <code>type(exc).__name__</code> prints{' '}
            <code>TimeoutError</code>. Existing <code>except TimeoutError</code> catches it.
          </>
        }
      />

      <ParamTable
        nameLabel="Timeout"
        params={[
          {
            name: 'model_call',
            type: 'float',
            default: '60.0',
            description: 'One generation, including the provider’s own queueing.',
          },
          {
            name: 'tool_call',
            type: 'float',
            default: '30.0',
            description: 'One tool dispatch.',
          },
          {
            name: 'http',
            type: 'float',
            default: '20.0',
            description: 'A single HTTP request made by a tool or an adapter.',
          },
          {
            name: 'agent_loop',
            type: 'float',
            default: '600.0',
            description: 'A whole run, across every iteration it takes.',
          },
          {
            name: 'queue',
            type: 'float',
            default: '5.0',
            description: 'How long a caller waits for a bulkhead permit.',
          },
        ]}
        caption={
          <>
            <code>TimeoutConfig</code>, passed as <code>ReliabilityConfig(timeouts=…)</code>. Seconds.
          </>
        }
      />

      <p>
        For an HTTP client of your own, <code>make_httpx_timeout(http=20.0, connect=5.0)</code>{' '}
        returns the matching <code>httpx.Timeout</code>.
      </p>

      <h2>Retries</h2>
      <p>
        <code>Retry</code> is the policy and <code>@retryable(policy)</code> applies it. The delay is
        exponential — <code>base_delay × 2^attempt</code>, capped at <code>max_delay</code> — with
        jitter added so a fleet of clients does not retry in lockstep. A <code>429</code> carrying{' '}
        <code>Retry-After</code> uses that value instead.
      </p>

      <CodeBlock filename="retry.py" code={`from effgen.reliability import Retry, retryable

policy = Retry(max_attempts=3, base_delay=0.05, max_delay=1.0, jitter=False)
attempts = 0


@retryable(policy)
def flaky():
    global attempts
    attempts += 1
    if attempts < 3:
        raise ConnectionError("connection reset by peer")
    return "ok"


print(flaky(), "after", attempts, "attempts")


@retryable(policy)
def broken():
    raise ValueError("a bug in your own code")


try:
    broken()
except ValueError as exc:
    print("not retried:", exc)`} />

      <Terminal
        command="python retry.py"
        output={`ok after 3 attempts
not retried: a bug in your own code`}
        caption="The decorator works on an async function too, with the same policy object."
      />

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'max_attempts',
            type: 'int',
            default: '3',
            description: 'Total attempts, not retries after the first.',
          },
          {
            name: 'base_delay',
            type: 'float',
            default: '0.5',
            description: 'Seconds. The first backoff, doubled each attempt.',
          },
          {
            name: 'max_delay',
            type: 'float',
            default: '30.0',
            description: 'The cap the doubling stops at.',
          },
          {
            name: 'jitter',
            type: 'bool',
            default: 'True',
            description: (
              <>
                Adds <code>uniform(0, base_delay)</code> to each wait.
              </>
            ),
          },
          {
            name: 'retryable',
            type: 'Callable[[BaseException], bool]',
            default: 'is_transient_error',
            description: 'The test for whether a failure is worth another attempt.',
          },
          {
            name: 'honour_retry_after',
            type: 'bool',
            default: 'True',
            description: (
              <>
                Respect a <code>Retry-After</code> header on a 429.
              </>
            ),
          },
        ]}
      />

      <h3>What counts as transient</h3>
      <p>
        Retrying a bug in your own code does not fix it, so the default test is narrow. It returns
        true for connection and OS errors, timeouts, anything carrying effGen's structured{' '}
        <code>error_context</code> — which every provider adapter attaches, so a live 429 or 5xx is
        still classified as transient after being wrapped — a raw SDK exception with a 429 or 5xx status,
        and <code>httpx</code> network errors. Nothing else.
      </p>

      <CodeBlock filename="transient.py" code={`from effgen.models.errors import ModelAuthError, ProviderTransientError
from effgen.reliability import is_transient_error

for exc in (
    ConnectionError("connection reset"),
    TimeoutError("read timed out"),
    ProviderTransientError("openai", "gpt-5-nano", status_code=503),
    ModelAuthError("openai", "gpt-5-nano"),
    ValueError("bad argument"),
):
    print(f"{type(exc).__name__:24} retried: {is_transient_error(exc)}")`} />

      <Terminal
        command="python transient.py"
        output={`ConnectionError          retried: True
TimeoutError             retried: True
ProviderTransientError   retried: True
ModelAuthError           retried: False
ValueError               retried: False`}
        caption={
          <>
            A bad key is not going to become a good key. <Link to="/errors">Errors and exceptions</Link>{' '}
            lists which typed errors are retryable.
          </>
        }
      />

      <p>
        When every attempt is used up, <code>RetryExhausted(attempts, last_error)</code> is raised. A
        failure the policy does not consider retryable is re-raised immediately, unchanged. Each
        attempt also adds an <code>effgen.retry.attempt</code> event to the current span, carrying
        the attempt number, the reason and the delay — see <Link to="/tracing">Tracing</Link>.
      </p>

      <h2>Circuit breaker</h2>
      <p>
        After enough consecutive failures the breaker opens and stops permitting calls, so a provider
        that is down costs one failure rather than one per request. After{' '}
        <code>recovery_timeout</code> it lets a probe through; a success closes it, a failure opens
        it again.
      </p>

      <CodeBlock filename="circuit.py" code={`from effgen.reliability import CircuitBreaker

breaker = CircuitBreaker(name="openai", failure_threshold=3, recovery_timeout=30.0)

for i in range(3):
    breaker.on_failure(ConnectionError("provider down"))
    print(f"failure {i + 1}: state={breaker.state.value} permitted={breaker.is_call_permitted()}")

print("rejected so far:", breaker.stats()["total_rejected"])
print("call permitted :", breaker.is_call_permitted())
print("rejected now   :", breaker.stats()["total_rejected"])`} />

      <Terminal
        command="python circuit.py"
        output={`failure 1: state=closed permitted=True
failure 2: state=closed permitted=True
failure 3: state=open permitted=False
rejected so far: 1
call permitted : False
rejected now   : 2`}
        caption="Asking whether a call is permitted while the breaker is open is itself counted as a rejection, which is what makes the rejected count the size of the outage."
      />

      <ApiTable
        headers={['State', 'What happens', 'Leaves when']}
        rows={[
          [<code>CLOSED</code>, 'Every call goes through.', 'failure_threshold consecutive failures.'],
          [
            <code>OPEN</code>,
            <>
              <code>is_call_permitted()</code> returns <code>False</code> and callers raise{' '}
              <code>CircuitBreakerOpen</code>.
            </>,
            'recovery_timeout seconds elapse.',
          ],
          [
            <code>HALF_OPEN</code>,
            'A limited number of probe calls are let through.',
            'half_open_probes successes close it; one failure opens it again.',
          ],
        ]}
      />

      <p>
        The usual shape is around a call, not by hand — and there is one breaker per provider name,
        shared, so every caller in the process sees the same state.
      </p>

      <CodeBlock continues filename="guarded.py" code={`from effgen.reliability import CircuitBreakerOpen
from effgen.reliability.circuit import get_circuit_breaker

breaker = get_circuit_breaker("cerebras", failure_threshold=3, recovery_timeout=15.0)

if not breaker.is_call_permitted():
    raise CircuitBreakerOpen("cerebras")
try:
    result = call_model()
    breaker.on_success()
except Exception:
    breaker.on_failure()
    raise`} />

      <p>
        <code>ProviderRegistry</code> exposes the same registry alongside the bulkheads, and{' '}
        <code>ProviderRegistry.reliability_stats()</code> returns both for every provider at once —
        which is what the dashboard's provider panel reads.
      </p>

      <CodeBlock filename="registry.py" code={`from effgen.models.registry import ProviderRegistry

breaker = ProviderRegistry.get_circuit_breaker("openai", failure_threshold=5, recovery_timeout=30.0)
bulkhead = ProviderRegistry.get_bulkhead("openai", max_concurrency=10)

stats = ProviderRegistry.reliability_stats()
# {"openai": {"circuit_breaker": {...}, "bulkhead": {...}}}`} />

      <h2>Bulkhead</h2>
      <p>
        A bulkhead caps how many calls to one provider can be in flight, and how many callers may
        queue for a permit. Past both, a caller is rejected rather than left waiting — a fast
        rejection you can fall back from beats a slow one that has already cost you a worker.
      </p>

      <CodeBlock filename="bulkhead.py" code={`from effgen.reliability import Bulkhead, BulkheadFull

bulkhead = Bulkhead(name="groq", max_concurrency=2, queue_size=0, queue_timeout=0.1)

with bulkhead.acquire():
    with bulkhead.acquire():
        print("in flight:", bulkhead.stats()["active"], "of", bulkhead.stats()["max_concurrency"])
        try:
            with bulkhead.acquire():
                print("this never runs")
        except BulkheadFull as exc:
            print("third caller rejected:", exc)

print("after release:", bulkhead.stats())`} />

      <Terminal
        command="python bulkhead.py"
        output={`in flight: 2 of 2
third caller rejected: Bulkhead 'groq' is at capacity — 2 active calls and 0 queued. Consider increasing max_concurrency / queue_size or reducing call rate.
after release: {'name': 'groq', 'max_concurrency': 2, 'queue_size': 0, 'active': 0, 'queued': 0, 'total_accepted': 2, 'total_rejected': 1, 'total_timeout': 0, 'utilization_pct': 0.0}`}
      />

      <ParamTable
        nameLabel="Argument"
        params={[
          { name: 'name', type: 'str', required: true, description: 'Usually the provider name.' },
          {
            name: 'max_concurrency',
            type: 'int',
            default: '20',
            description: 'Calls allowed in flight at once.',
          },
          {
            name: 'queue_size',
            type: 'int',
            default: '100',
            description: 'Callers allowed to wait for a permit. Past this, BulkheadFull.',
          },
          {
            name: 'queue_timeout',
            type: 'float',
            default: '5.0',
            description: 'How long a queued caller waits before BulkheadFull.',
          },
        ]}
        caption={
          <>
            Also usable as <code>async with bh.async_acquire()</code>, or as the decorators{' '}
            <code>@bh.guard()</code> and <code>@bh.async_guard()</code>.{' '}
            <code>get_bulkhead(name, …)</code> returns the shared one for a provider.
          </>
        }
      />

      <h2>Failing over to another model</h2>
      <p>
        Retries and breakers cover one provider having a bad minute. When a provider is out entirely, the answer is
        a different model — that is <Link to="/routing">Model routing and fallback</Link>, and its{' '}
        <code>AllCandidatesExhaustedError</code> names every hop that was tried and why each one
        failed.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>RetryExhausted</code>,
            'Every attempt failed with something the policy considered retryable.',
            <>
              <code>exc.last_error</code> is the real failure — read that, not the wrapper. Raising{' '}
              <code>max_attempts</code> only helps if the outage is shorter than the backoff.
            </>,
          ],
          [
            'A failure was not retried at all',
            <>
              <code>is_transient_error</code> returned <code>False</code> — an auth error, a bad
              model id, an invalid request.
            </>,
            <>
              These do not become true on a second attempt. Pass your own{' '}
              <code>retryable=</code> if your call really is different.
            </>,
          ],
          [
            <code>CircuitBreakerOpen</code>,
            'The breaker for that provider is open after consecutive failures.',
            <>
              Wait out <code>recovery_timeout</code>, or fall back to another provider. Restarting
              the process resets it, which is not a fix.
            </>,
          ],
          [
            <code>BulkheadFull</code>,
            'Concurrency and queue are both at capacity.',
            <>
              Raise <code>max_concurrency</code> if the provider can take it, or lower your own
              request rate. Rejection is the design.
            </>,
          ],
          [
            <><code>EffGenTimeoutError</code> on a long prompt</>,
            <>
              The call exceeded <code>model_call</code>, which is 60 seconds by default.
            </>,
            <>
              A reasoning model on a long prompt needs more. Raise it through{' '}
              <code>TimeoutConfig</code>, and check <code>agent_loop</code> too on a tool-using run.
            </>,
          ],
          [
            'Every client retries at the same instant',
            <>
              <code>jitter=False</code>.
            </>,
            'Leave jitter on outside a test.',
          ],
          [
            'A 429 keeps coming back',
            'The retry is honouring Retry-After and the window has not passed.',
            <>
              Lower concurrency with a bulkhead rather than retrying harder. <Link to="/cost">Cost
              and budgets</Link> covers rate limits.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          An unreachable backend now raises <code>BackendUnreachableError</code> whatever{' '}
          <code>raise_on_error</code> is set to. It is classified as retryable, so a retry policy
          still covers a server that is coming back up — but it is never silently turned into an
          empty answer. <Link to="/errors">Errors and exceptions</Link> has the reasoning.
        </p>
      </Callout>

      <SeeAlso paths={['/errors', '/routing', '/observability']} />
    </DocPage>
  );
}
