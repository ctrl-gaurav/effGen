import { AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { version } from '../siteData';

export default function Errors() {
  return (
    <DocPage
      subtitle="Every typed error effGen raises, what causes it, and what to do about it."
      icon={<AlertTriangle size={48} />}
    >
      <p>
        effGen does not raise bare <code>KeyError</code>s and does not hand you a provider SDK's
        stack trace. Every failure it produces is a named class carrying the facts about the failure
        — which provider, which model, whether another attempt could work — and a message that ends
        with what to do next. This page is the list.
      </p>

      <h2>Two ways a failure reaches you</h2>
      <p>
        <code>raise_on_error</code> decides whether a run raises or returns. It defaults to{' '}
        <code>True</code>, which was a change in 1.0.0.
      </p>

      <CodeBlock filename="both_ways.py" code={`from effgen import Agent, AgentConfig, load_model
from effgen.models.errors import ModelAuthError

model = load_model("gpt-5-nano", provider="openai", api_key="sk-not-a-real-key")

# raise_on_error defaults to True since 1.0.0.
agent = Agent(AgentConfig(model=model))
try:
    agent.run("What is 2 + 2?")
except ModelAuthError as exc:
    print("raised:", type(exc).__name__)

# Opt back into the old behaviour and read the outcome off the response.
quiet = Agent(AgentConfig(model=model, raise_on_error=False))
response = quiet.run("What is 2 + 2?")
print("returned:", response.success, response.metadata["error"]["type"])`} />

      <Terminal
        command="python both_ways.py"
        output={`raised: ModelAuthError
returned: False ModelAuthError`}
        caption={`Run against effGen 1.0.0.`}
      />

      <h3>The structured error on a response</h3>
      <p>
        With <code>raise_on_error=False</code>, <code>response.metadata["error"]</code> is a dict with
        the same shape whether the failure came from a direct call or from inside a tool loop.
      </p>

      <CodeBlock filename="structured.py" code={`from effgen import Agent, AgentConfig, load_model

model = load_model("gpt-5-nano", provider="openai", api_key="sk-not-a-real-key")
agent = Agent(AgentConfig(model=model, raise_on_error=False))

response = agent.run("What is 2 + 2?")
print("success:", response.success)
print("error  :", response.metadata["error"])`} />

      <Terminal
        command="python structured.py"
        output={`success: False
error  : {'type': 'ModelAuthError', 'category': 'auth', 'provider': 'openai', 'model': 'gpt-5-nano', 'message': 'Incorrect API key provided: sk-not-a*****-key. You can find your API key at https://platform.openai.com/account/api-keys.', 'remediation': 'Check the provider API key (present, correct, not expired, and has access to this model).', 'retryable': False}`}
        caption="Seven keys, always the same seven. `retryable` is what a retry policy branches on; `remediation` is the sentence to show a human."
      />

      <Callout type="warning" title="One failure ignores raise_on_error">
        <p>
          <code>BackendUnreachableError</code> is raised whatever <code>raise_on_error</code> says.
          Nothing answered at the address the run was sent to, so there is no answer to return and no
          degraded result to hand back — a returned <code>success=False</code> here would look
          exactly like a model that declined, and the two need different fixes. See{' '}
          <a href="#nothing-answered">below</a>.
        </p>
      </Callout>

      <h3 id="stopped">Answered, stopped, failed</h3>
      <p>
        A run ends in one of three states, and <code>response.outcome</code> names which without you
        having to read <code>output</code>. <code>response.stop_reason</code> names the exit the run
        took and is present on every response — an answered run reports{' '}
        <code>"final_answer"</code> — and it always equals <code>metadata["reason"]</code>.
      </p>

      <ApiTable
        headers={['outcome', 'What it means', 'output', 'partial']}
        rows={[
          [<code>answered</code>, 'The model wrote an answer.', 'The answer.', <code>None</code>],
          [
            <code>stopped</code>,
            'The loop ended the run before the model wrote one: max_iterations_partial, max_iterations_exhausted, loop_detected, repeated_tool_result, null_final_from_model or, since 1.3.0, tool_failed.',
            'What happened, and what to do.',
            'The progress, when there was any.',
          ],
          [<code>failed</code>, 'The run could not be carried out.', 'The classified failure.', <code>None</code>],
        ]}
      />

      <p>
        A stopped run has tool results and reasoning but no answer, so they travel in{' '}
        <code>response.partial</code>, a <code>PartialResult</code> with <code>observations</code>,{' '}
        <code>last_observation</code>, <code>last_thought</code>, <code>text</code>,{' '}
        <code>iterations</code> and <code>tool_calls</code>. <code>metadata["partial_output"]</code>{' '}
        carries <code>partial.text</code> under the key earlier releases used. Under the default{' '}
        <code>raise_on_error=True</code> a stopped run raises <code>RunStoppedError</code>, which
        subclasses <code>RuntimeError</code> — so code already catching the iteration cap keeps
        working — and carries <code>.response</code>, <code>.stop_reason</code> and{' '}
        <code>.partial</code>. With the flag off, the same run comes back with{' '}
        <code>success=False</code> and <code>outcome == "stopped"</code>. Since 1.3.0 a stopped
        run’s <code>partial</code> never carries a tool’s error message: a run whose only
        observations were errors has <code>partial=None</code>.
      </p>

      <CodeBlock filename="stopped.py" code={`from effgen import Agent, AgentConfig, RunStoppedError
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    tools=[Calculator()],
    max_iterations=1,
))
try:
    agent.run("With the calculator: work out 24344 * 334, then multiply that by 7, "
              "then subtract 19, then divide by 3.")
except RunStoppedError as exc:
    print("raised:", type(exc).__name__, "is a RuntimeError:", isinstance(exc, RuntimeError))
    print("stop_reason:", exc.stop_reason)
    print("outcome:", exc.response.outcome)
    print("partial:", exc.partial.text if exc.partial else None)`} />

      <Terminal
        command="python stopped.py"
        output={`raised: RunStoppedError is a RuntimeError: True
stop_reason: max_iterations_partial
outcome: stopped
partial: 8130896`}
        caption={`Run against effGen 1.0.1. One iteration is enough for one calculator call and not for an answer, so the run stops holding the result it had reached.`}
      />

      <p>
        Since 1.3.0 a run that is about to stop while holding tool results first gets one closing
        request — its own calls and results, with no tools — and when the reply is an answer the
        run succeeds with <code>metadata["answer_source"] == "closing_request"</code>. A run like
        the one above can therefore return an answer on 1.3.0 where 1.0.1 raised.
      </p>

      <h3 id="termination">Done, not possible, stuck, tool failed</h3>
      <p>
        <code>response.termination</code>, new in 1.3.0, says how a run ended in the four ways a
        tool loop can end, plus the runs that could not be carried out. It is derived from{' '}
        <code>success</code>, <code>stop_reason</code> and <code>metadata["tool_results"]</code>,
        and <code>to_dict()</code> carries it.
      </p>

      <ApiTable
        headers={['termination', 'What it means', 'stop_reason']}
        rows={[
          [<code>done</code>, 'The model wrote an answer.', <code>final_answer</code>],
          [<code>not_possible</code>, 'The model wrote an answer, but every call was declined, failed on the tool’s own side or returned nothing — usually an answer saying the task cannot be done with these tools.', <code>final_answer</code>],
          [<code>stuck</code>, 'The run kept proposing work that brought nothing new, was asked for its answer, and wrote none.', 'loop_detected, repeated_tool_result, max_iterations_*, null_final_from_model'],
          [<code>tool_failed</code>, 'The tools the run needed failed, and the run has no answer.', <code>tool_failed</code>],
          [<code>error</code>, 'The run could not be carried out.', 'the other failed reasons'],
        ]}
      />

      <p>
        A tool that fails on its own side — a connection or timeout error, an HTTP 5xx or 429,
        missing credentials, recognised from the exception’s class — three times in a row is not
        called again in the run, and a run left with no usable tool ends <code>tool_failed</code>{' '}
        with <code>metadata["error"]["kind"] == "tool"</code> and the tools named in{' '}
        <code>metadata["unavailable_tools"]</code>. Such a run gets no closing request. Four
        failures in a row on a tool’s input withdraw that tool for the run; a run left with no tool
        is asked for its answer and, if it writes none, ends <code>tool_failed</code> with{' '}
        <code>kind == "input"</code>. Under the default <code>raise_on_error=True</code> both raise{' '}
        <code>RunStoppedError</code>. A tool that rejects the same input twice, or every input in
        the same words, can still be stopped by the loop guard first, so that run ends{' '}
        <code>stuck</code> rather than <code>tool_failed</code>; this is a known issue in 1.3.0.
      </p>

      <h2>Model and provider errors</h2>
      <p>
        These are in <code>effgen.models.errors</code>. They are what a provider call fails with,
        after the raw SDK exception has been classified and its message redacted and bounded.
      </p>

      <ApiTable
        headers={['Error', 'Raised when', 'Retryable', 'What to do']}
        rows={[
          [
            <code>ModelAuthError</code>,
            'The provider rejected the credentials.',
            'No',
            'Check the key is present, current, and entitled to that model.',
          ],
          [
            <code>ModelNotFoundError</code>,
            'The model id does not exist on that provider.',
            'No',
            <>
              <code>effgen models list</code> for the ids, <code>effgen models refresh</code> to
              update the catalog. Check the provider prefix.
            </>,
          ],
          [
            <code>AmbiguousModelError</code>,
            'The id exists on more than one provider and none was named.',
            'No',
            <>
              Prefix it: <code>groq:llama-3.3-70b</code>.
            </>,
          ],
          [
            <code>ModelUnavailableError</code>,
            'The model exists but the serverless tier is not serving it right now.',
            'Yes',
            'Retry, or take one of the suggestions it carries.',
          ],
          [
            <code>ProviderTransientError</code>,
            'The provider returned a 5xx.',
            'Yes',
            'The retry policy already covers it. Check the status page if it persists.',
          ],
          [
            <code>ModelTimeoutError</code>,
            'A prediction did not finish inside the adapter timeout.',
            'Yes',
            <>
              Raise the timeout, or ask for less. See <Link to="/reliability">Reliability</Link>.
            </>,
          ],
          [
            <code>InvalidRequestError</code>,
            'The request itself is malformed — a bad parameter, an oversized prompt, an invalid schema.',
            'No',
            'Fix the request. A second attempt sends the same bad request.',
          ],
          [
            <code>ModelRefusalError</code>,
            'The model declined a structured-output request.',
            'No',
            'Rephrase, or relax the schema.',
          ],
          [
            <code>ToolIncompatibleError</code>,
            'A tool cannot be used with the configured model.',
            'No',
            <>
              Use a model that declares tool calling — see <Link to="/tool-calling">Tool calling</Link>.
            </>,
          ],
          [
            <code>BackendUnreachableError</code>,
            'Nothing answered at the endpoint.',
            'Yes',
            'Check the server is up and the base_url, host and port are right.',
          ],
          [
            <code>BudgetExceededError</code>,
            'Cumulative spend crossed the configured daily or monthly budget. Since 1.2.0 run() raises it whatever raise_on_error says, before any request is sent, and only for a call that costs money. It is not a RuntimeError.',
            'No',
            <>
              Raise the budget or wait for the period to roll — see <Link to="/cost">Cost and
              budgets</Link>.
            </>,
          ],
          [
            <code>AllCandidatesExhaustedError</code>,
            'Every failover hop failed.',
            'No',
            <>
              It carries each hop and its failure. Read those, not the wrapper —{' '}
              <Link to="/routing">Routing</Link>.
            </>,
          ],
          [
            <code>NoCandidateWithinBudgetError</code>,
            'No model in the catalog fits the cost budget given.',
            'No',
            'It names the cheapest candidate and what it costs. Raise the budget to that.',
          ],
          [
            <code>NoCandidateWithinLatencyError</code>,
            'No model fits the latency budget given.',
            'No',
            'It names the fastest candidate and its latency.',
          ],
        ]}
        caption={
          <>
            Derived from <code>effgen.models.errors</code> and{' '}
            <code>classify_provider_error</code>. All fourteen subclass{' '}
            <code>Exception</code> directly.
          </>
        }
      />

      <h2>Framework errors</h2>
      <p>
        These are in <code>effgen.errors</code> and all subclass <code>EffGenError</code>, so{' '}
        <code>except EffGenError</code> catches every one. They are about the environment a tool or a
        run needs, not about a provider.
      </p>

      <ApiTable
        headers={['Error', 'Raised when', 'Carries']}
        rows={[
          [
            <code>MissingSystemDependency</code>,
            'A required binary is not on PATH.',
            <>
              <code>dependency</code>, <code>install_instructions</code>. The message lists the
              apt/brew/choco/conda lines.
            </>,
          ],
          [
            <code>MissingCredentialsError</code>,
            'A tool needs environment variables that are not set.',
            <>
              <code>tool_name</code>, <code>missing_vars</code>
            </>,
          ],
          [
            <code>CorruptStateError</code>,
            'A session or checkpoint file will not parse.',
            <>
              <code>kind</code>, <code>path</code>, <code>detail</code> — the file is named so you
              can find it.
            </>,
          ],
          [
            <code>CorruptDocumentError</code>,
            'A PDF, DOCX or XLSX cannot be read.',
            <>
              <code>doc_type</code>, <code>detail</code>
            </>,
          ],
          [
            <code>CapabilityNotSupportedError</code>,
            'An adapter was asked for something it does not do — vision, audio.',
            <>
              <code>capability</code>, <code>provider</code>
            </>,
          ],
          [
            <code>InvalidMultimodalContent</code>,
            'An image, audio or video part failed validation.',
            <>
              <code>part_type</code>, <code>reason</code>
            </>,
          ],
          [
            <code>OCRBackendUnavailable</code>,
            'Neither Tesseract nor the cloud fallback is available.',
            <code>tried_backends</code>,
          ],
          [
            <code>AudioBackendUnavailable</code>,
            'No transcription backend is available.',
            <code>tried_backends</code>,
          ],
          [
            <code>NoVisionProviderAvailable</code>,
            'Image captioning found no vision-capable provider.',
            <code>tried_providers</code>,
          ],
        ]}
      />

      <CodeBlock filename="hierarchy.py" code={`import effgen.errors as core
import effgen.models.errors as models

print("effgen.errors — EffGenError and its subclasses")
for name in sorted(dir(core)):
    obj = getattr(core, name)
    if isinstance(obj, type) and issubclass(obj, core.EffGenError):
        print(f"  {name}")

print()
print("effgen.models.errors")
for name in sorted(dir(models)):
    obj = getattr(models, name)
    if isinstance(obj, type) and issubclass(obj, Exception) and obj.__module__ == models.__name__:
        print(f"  {name}")`} />

      <Terminal
        command="python hierarchy.py"
        output={`effgen.errors — EffGenError and its subclasses
  AudioBackendUnavailable
  CapabilityNotSupportedError
  CorruptDocumentError
  CorruptStateError
  EffGenError
  InvalidMultimodalContent
  MissingCredentialsError
  MissingSystemDependency
  NoVisionProviderAvailable
  OCRBackendUnavailable

effgen.models.errors
  AllCandidatesExhaustedError
  AmbiguousModelError
  BackendUnreachableError
  BudgetExceededError
  InvalidRequestError
  ModelAuthError
  ModelNotFoundError
  ModelRefusalError
  ModelTimeoutError
  ModelUnavailableError
  NoCandidateWithinBudgetError
  NoCandidateWithinLatencyError
  ProviderTransientError
  ToolIncompatibleError`}
        caption="The list, from the installed package. The reliability primitives add four more — see the table at the foot of this page."
      />

      <h2>What a message looks like</h2>
      <p>
        The shape is the same every time: what happened, then what to do about it. Text quoted from
        somewhere else — a provider body, a parser complaint — is redacted and cut to 240 characters
        before it goes in.
      </p>

      <CodeBlock filename="messages.py" code={`from effgen import load_model
from effgen.errors import CorruptStateError, MissingCredentialsError
from effgen.models.errors import ModelNotFoundError

try:
    load_model("llama-9000-turbo", provider="groq")
except ModelNotFoundError as exc:
    print("---", type(exc).__name__, "---")
    print(exc)
    print()

for exc in (
    MissingCredentialsError("SlackWebhookTool", ["SLACK_WEBHOOK_URL"]),
    CorruptStateError("checkpoint", "./checkpoints/run-7.json", "Expecting value: line 1 column 1"),
):
    print(f"--- {type(exc).__name__} ---")
    print(exc)
    print()`} />

      <Terminal
        command="python messages.py"
        output={`--- ModelNotFoundError ---
groq error (model='llama-9000-turbo'): Unknown Groq model 'llama-9000-turbo'. Did you mean: allam-2-7b, whisper-large-v3-turbo, meta-llama/llama-prompt-guard-2-86m? Available groq models: qwen/qwen3.6-27b, qwen/qwen3.8-27b, openai/gpt-oss-120b, openai/gpt-oss-20b, openai/gpt-oss… (280 characters). Model id not found — run \`effgen models list\` to see ids, \`effgen models refresh\` to update the catalog, and verify the id/provider prefix.

--- MissingCredentialsError ---
SlackWebhookTool requires credentials that are not configured: SLACK_WEBHOOK_URL.

Set SLACK_WEBHOOK_URL in the environment or in the project .env file, then re-run.

--- CorruptStateError ---
Cannot read checkpoint file './checkpoints/run-7.json' - it is corrupt, truncated, or not valid JSON.
Detail: Expecting value: line 1 column 1

Fix: inspect the file, restore a backup, or delete it to start fresh.`}
      />

      <h2>Classifying a failure yourself</h2>
      <p>
        <code>classify_provider_error(exc)</code> maps any exception — effGen's own, or a raw provider
        SDK exception — onto an <code>ErrorClass</code>. It recognises effGen's types first, then
        falls back to SDK class names, HTTP status codes and message text.
      </p>

      <CodeBlock filename="classify.py" code={`from effgen.models.errors import (
    BackendUnreachableError, InvalidRequestError, ModelAuthError,
    ModelNotFoundError, ModelRefusalError, ProviderTransientError,
    classify_provider_error,
)

for exc in (
    ModelAuthError("openai", "gpt-5-nano"),
    ModelNotFoundError("groq", "llama-9000-turbo"),
    ProviderTransientError("gemini", "gemini-3.1-flash-lite", status_code=503),
    InvalidRequestError("openai", "gpt-5-nano", "max_tokens too large"),
    ModelRefusalError("cannot comply", "gpt-5-nano"),
    BackendUnreachableError("openai_compatible", "local-model", endpoint="http://127.0.0.1:9/v1"),
):
    kind = classify_provider_error(exc)
    print(f"{type(exc).__name__:26} {kind.category:16} should_retry={kind.should_retry}")`} />

      <Terminal
        command="python classify.py"
        output={`ModelAuthError             auth             should_retry=False
ModelNotFoundError         not_found        should_retry=False
ProviderTransientError     transient        should_retry=True
InvalidRequestError        invalid_request  should_retry=False
ModelRefusalError          refusal          should_retry=False
BackendUnreachableError    unreachable      should_retry=True`}
      />

      <p>
        The twelve categories are <code>auth</code>, <code>not_found</code>,{' '}
        <code>rate_limited</code>, <code>transient</code>, <code>timeout</code>,{' '}
        <code>refusal</code>, <code>invalid_request</code>, <code>not_loaded</code>,{' '}
        <code>resource_exhausted</code>, <code>unreachable</code>, <code>fatal</code> and{' '}
        <code>unknown</code>. <code>should_retry</code> is true for the retryable and rate-limited
        ones; the rest fail fast because a second attempt sends the same request into the same wall.
      </p>

      <Callout type="note" title="Unknown defaults to retryable">
        <p>
          A failure effGen has never seen is classified <code>unknown</code> and marked retryable, so
          a genuine transient blip is not turned into a hard failure by a gap in the classifier. Every
          recognised auth, not-found, refusal and invalid-request class fails fast.
        </p>
      </Callout>

      <h2 id="nothing-answered">Nothing answered at that endpoint</h2>
      <p>
        Point a run at an address where no server is listening and you get{' '}
        <code>BackendUnreachableError</code>, carrying the endpoint it tried. This is the one failure
        that ignores <code>raise_on_error=False</code>.
      </p>

      <CodeBlock filename="unreachable.py" code={`from effgen import Agent, AgentConfig
from effgen.models.errors import BackendUnreachableError

agent = Agent(AgentConfig(
    model="local-model",
    provider="openai_compatible",
    base_url="http://127.0.0.1:9/v1",   # nothing is listening here
    raise_on_error=False,                # does not apply to this one
))
try:
    agent.run("Say hello.")
except BackendUnreachableError as exc:
    print(type(exc).__name__)
    print(exc)`} />

      <Terminal
        command="python unreachable.py"
        output={`BackendUnreachableError
openai did not answer (model='local-model'): OpenAI generation failed [will_retry]: Connection error.. Nothing answered at that endpoint — check the server is running and the base_url, host and port are right. The call was sent to http://127.0.0.1:9/v1. Nothing answered at that endpoint — check the server is running and the base_url, host and port are right.`}
        caption={
          <>
            The endpoint is named, because a wrong <code>base_url</code> is the usual cause and a
            message that does not say which address was tried cannot tell you that. See{' '}
            <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
          </>
        }
      />

      <Callout type="warning" title="New in 1.0.0">
        <p>
          Two of the three breaking changes in 1.0.0 are on this page:{' '}
          <code>raise_on_error</code> now defaults to <code>True</code>, and an unreachable backend
          raises regardless of it. Code that read <code>response.success</code> and never expected an
          exception should pass <code>raise_on_error=False</code> explicitly.{' '}
          <Link to="/migration">Migrating to 1.2.0</Link> has the one-line change for each.
        </p>
      </Callout>

      <h2>Reliability errors</h2>
      <p>
        The four in <code>effgen.reliability</code> are about the machinery around a call rather than
        the call itself. <Link to="/reliability">Reliability</Link> covers each one.
      </p>

      <ApiTable
        headers={['Error', 'Base', 'Raised when']}
        rows={[
          [
            <code>EffGenTimeoutError</code>,
            <>built-in <code>TimeoutError</code></>,
            <>
              A guarded call ran past its limit. Carries <code>operation</code> and{' '}
              <code>limit</code>.
            </>,
          ],
          [
            <code>RetryExhausted</code>,
            <code>Exception</code>,
            <>
              Every attempt failed. Carries <code>attempts</code> and <code>last_error</code>.
            </>,
          ],
          [
            <code>CircuitBreakerOpen</code>,
            <code>Exception</code>,
            'The breaker for that provider is open and not permitting calls.',
          ],
          [
            <code>BulkheadFull</code>,
            <code>Exception</code>,
            'Concurrency and queue are both at capacity.',
          ],
        ]}
      />

      <h2>Catching them</h2>

      <CodeBlock filename="catching.py" code={`from effgen.errors import EffGenError
from effgen.models.errors import ModelAuthError, classify_provider_error
from effgen.reliability import EffGenTimeoutError

try:
    response = agent.run(task)
except ModelAuthError:
    ...                       # a specific failure you have a specific answer for
except EffGenTimeoutError:
    ...                       # too slow, not broken
except EffGenError:
    ...                       # every framework error: a missing binary, a corrupt file
except Exception as exc:
    kind = classify_provider_error(exc)
    if kind.should_retry:
        ...                   # worth another attempt
    raise`} />

      <p>
        Provider errors do not share a base class, so the general case is{' '}
        <code>classify_provider_error</code> rather than <code>except SomeProviderError</code>. That
        is deliberate: it is the same call whether the exception came from effGen or straight out of
        a provider SDK.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Code that used to return now raises',
            <>
              <code>raise_on_error</code> defaults to <code>True</code> in {version}.
            </>,
            <>
              Pass <code>raise_on_error=False</code> to keep the old shape, or catch the typed error.
            </>,
          ],
          [
            <><code>KeyError: 'error'</code> on a response</>,
            <>
              <code>metadata["error"]</code> only exists on a failed run.
            </>,
            <>
              Check <code>response.success</code> first, or use{' '}
              <code>metadata.get("error")</code>.
            </>,
          ],
          [
            'A message ends mid-word with a character count',
            'It quoted a long upstream body and was cut at 240 characters.',
            <>
              The full body is in the logs at <code>EFFGEN_LOG_LEVEL=DEBUG</code>.
            </>,
          ],
          [
            <><code>&lt;REDACTED:openai_key&gt;</code> in an error</>,
            'The provider echoed your credential back and it was replaced before the message was built.',
            <>
              Working as intended. <Link to="/security">Security</Link> covers the redactor.
            </>,
          ],
          [
            'A retry loop that never gives up on a bad key',
            <>
              Your own retry, not effGen's. <code>ModelAuthError</code> classifies as{' '}
              <code>auth</code>, which is not retryable.
            </>,
            <>
              Branch on <code>classify_provider_error(exc).should_retry</code>.
            </>,
          ],
          [
            <><code>ReasoningOnlyResponse</code> in <code>metadata["error"]["type"]</code></>,
            'A reasoning model spent its whole budget on the internal chain and returned no visible text.',
            <>
              Raise <code>max_tokens</code>. <Link to="/generation">Generation controls</Link>{' '}
              explains the reporting.
            </>,
          ],
          [
            'An exception from a middleware hook',
            'Middleware exceptions are not caught — that is what lets an approval gate stop a run.',
            <>
              Handle it in the hook if you did not mean it to escape. See{' '}
              <Link to="/middleware">Middleware</Link>.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/reliability', '/migration', '/routing']} />
    </DocPage>
  );
}
