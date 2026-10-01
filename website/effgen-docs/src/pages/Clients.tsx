import { Plug } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  CodeTabs,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function Clients() {
  return (
    <DocPage
      subtitle="The Python and TypeScript clients: calling a running server from your own code."
      icon={<Plug size={48} />}
    >
      <p>
        Two thin clients talk to a running <Link to="/api-server"><code>effgen serve</code></Link>:{' '}
        <code>effgen.client.EffGenClient</code>, which ships inside the Python package, and{' '}
        <code>effgen-client</code>, a dependency-free TypeScript client in the framework repository
        under <code>clients/typescript/</code>. Both wrap the same{' '}
        <Link to="/openai-api">OpenAI-compatible endpoints</Link>, add retries with backoff, and
        raise a typed error instead of returning a status code.
      </p>

      <h2>The shortest call</h2>

      <CodeBlock filename="first.py" code={`import os

from effgen.client import EffGenClient

client = EffGenClient(base_url="http://127.0.0.1:8000", api_key=os.environ["EFFGEN_API_KEY"])

print(client.health())

answer = client.chat("Reply with the single word ok.", model="openai:gpt-5-nano")
print(answer.content)
print(answer.model, answer.usage)`} />

      <Terminal
        command="python first.py"
        output={`HealthStatus(status='ok', details={'status': 'ok', 'version': '1.0.0'})
ok
openai:gpt-5-nano {'prompt_tokens': 26, 'completion_tokens': 138, 'total_tokens': 164}`}
        caption={`Against effGen 1.0.0 serving on 127.0.0.1:8000.`}
      />

      <Callout type="tip" title="Which client do I want?">
        <p>
          Use <code>EffGenClient</code> when you are calling a server from another process.
          If your code can import effGen, an <Link to="/agents"><code>Agent</code></Link> in the same
          process is simpler and has no HTTP hop. If you already have OpenAI SDK code, point its{' '}
          <code>base_url</code> at the server and change nothing else —{' '}
          <Link to="/openai-api">the OpenAI-compatible API</Link> covers that.
        </p>
      </Callout>

      <h2>Constructing the client</h2>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'base_url',
            type: 'str',
            default: "'http://localhost:8000'",
            description: 'Base URL of the server. No trailing path — the client appends /v1/… itself.',
          },
          {
            name: 'api_key',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                Sent as <code>Authorization: Bearer &lt;key&gt;</code>. Required unless the server
                is in dev mode.
              </>
            ),
          },
          { name: 'timeout', type: 'float', default: '60.0', description: 'Per-request timeout in seconds.' },
          {
            name: 'max_retries',
            type: 'int',
            default: '3',
            description: 'Attempts for a connection error, a timeout, a 429 or a 5xx. Other statuses are not retried.',
          },
          {
            name: 'backoff_base',
            type: 'float',
            default: '0.5',
            description: (
              <>
                Exponential backoff base in seconds: the delay is{' '}
                <code>backoff_base * 2**attempt</code>, with jitter.
              </>
            ),
          },
        ]}
        caption={
          <>
            From <code>EffGenClient.__init__</code> in the installed package.
          </>
        }
      />

      <h2>What it can do</h2>

      <ApiTable
        headers={['Method', 'Signature', 'Returns']}
        rows={[
          [
            <code>chat</code>,
            <code>chat(message, tools=None, model=None, **kwargs)</code>,
            <>
              A <code>ChatResponse</code>. <code>tools</code> is a list of tool names or tool
              definitions; the server runs them and returns a resolved answer.
            </>,
          ],
          [
            <code>achat</code>,
            <code>achat(message, tools=None, model=None, **kwargs)</code>,
            <>The same, awaited.</>,
          ],
          [
            <code>chat_stream</code>,
            <code>chat_stream(message, model=None)</code>,
            <>
              An <code>AsyncIterator[str]</code> of content deltas.
            </>,
          ],
          [
            <code>chat_stream_sync</code>,
            <code>chat_stream_sync(message, model=None)</code>,
            <>
              An <code>Iterator[str]</code> — the same stream from synchronous code.
            </>,
          ],
          [
            <code>embed</code>,
            <code>embed(texts, model="text-embedding-3-small")</code>,
            <code>list[list[float]]</code>,
          ],
          [<code>aembed</code>, <code>aembed(texts, model=…)</code>, 'The same, awaited.'],
          [
            <code>health</code>,
            <code>health()</code>,
            <>
              A <code>HealthStatus</code> with <code>status</code> and <code>details</code>. Public,
              so it answers without a key.
            </>,
          ],
          [<code>ahealth</code>, <code>ahealth()</code>, 'The same, awaited.'],
        ]}
        caption={
          <>
            Read from the class in the installed package. Anything in <code>**kwargs</code> is
            passed through to the request body, so a parameter the server accepts and this table
            does not name still works.
          </>
        }
      />

      <h3>A response</h3>

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'content', type: 'str', description: 'The answer text.' },
          { name: 'model', type: 'str | None', description: 'The model that actually ran, which is not always the one you asked for.' },
          { name: 'tool_calls', type: 'list[Any]', description: 'Tool calls the server reported, when the model made any.' },
          { name: 'usage', type: 'dict', description: 'Prompt, completion and total tokens, as the provider reported them.' },
          { name: 'raw', type: 'dict', description: 'The whole decoded response body, for anything this dataclass does not name.' },
        ]}
        caption={
          <>
            <code>effgen.client.ChatResponse</code>. <code>HealthStatus</code> carries{' '}
            <code>status</code> and <code>details</code>.
          </>
        }
      />

      <h2>Streaming</h2>

      <CodeBlock filename="stream.py" code={`import os

from effgen.client import EffGenClient

client = EffGenClient(base_url="http://127.0.0.1:8000", api_key=os.environ["EFFGEN_API_KEY"])

for chunk in client.chat_stream_sync("Count from 1 to 5, digits only.", model="openai:gpt-5-nano"):
    print(chunk, end="", flush=True)
print()`} />

      <Terminal command="python stream.py" output={`1 2 3 4 5`} />

      <p>
        <code>chat_stream</code> is the async form and is used with <code>async for</code>. Both
        yield content deltas only — a stream that fails mid-way ends with a terminal event rather
        than a truncated answer, and the client surfaces that as an exception.
      </p>

      <h2>Async</h2>

      <CodeBlock filename="async_client.py" code={`import asyncio
import os

from effgen.client import EffGenClient


async def main() -> None:
    client = EffGenClient(base_url="http://127.0.0.1:8000", api_key=os.environ["EFFGEN_API_KEY"])
    health = await client.ahealth()
    answer = await client.achat("Reply with the single word ok.", model="openai:gpt-5-nano")
    print(health.status, "->", answer.content)


asyncio.run(main())`} />

      <Terminal command="python async_client.py" output={`ok -> ok`} />

      <h2>Tools</h2>

      <p>
        Tools run <em>on the server</em>. Naming one lets the agent use it and hands you the finished
        answer; the client never receives a tool call to execute itself. The names are resolved
        against the built-in <Link to="/tools/gallery">tool registry</Link> and are subject to the
        caller's <Link to="/api-server">role policy</Link>.
      </p>

      <CodeBlock filename="tools.py" code={`import os

from effgen.client import EffGenClient

client = EffGenClient(base_url="http://127.0.0.1:8000", api_key=os.environ["EFFGEN_API_KEY"])

answer = client.chat(
    "What is 127 * 43? Use the calculator.",
    tools=["calculator"],
    model="openai:gpt-5-nano",
)
print(answer.content)`} />

      <Terminal command="python tools.py" output={`5461`} />

      <h2>Errors</h2>

      <ApiTable
        headers={['Exception', 'Raised on', 'Retried automatically']}
        rows={[
          [<code>EffGenConnectionError</code>, 'The server could not be reached at all.', 'Yes'],
          [<code>EffGenTimeoutError</code>, 'The request outran the client timeout.', 'Yes'],
          [
            <code>EffGenAPIError</code>,
            'Any non-2xx the more specific classes below do not cover.',
            'Only when the status is a 5xx',
          ],
          [<code>EffGenAuthError</code>, '401 or 403.', 'No — a second try sends the same key'],
          [<code>EffGenRateLimitError</code>, '429.', 'Yes'],
          [<code>EffGenServerError</code>, '5xx.', 'Yes'],
        ]}
        caption={
          <>
            All six derive from <code>EffGenClientError</code>, so one <code>except</code> catches
            everything the client raises. The three that carry a status also carry{' '}
            <code>status_code</code> and the decoded <code>payload</code>.
          </>
        }
      />

      <CodeBlock filename="errors.py" code={`from effgen.client import EffGenAuthError, EffGenClient, EffGenConnectionError

wrong_key = EffGenClient(base_url="http://127.0.0.1:8000", api_key="not-the-key", max_retries=0)
try:
    wrong_key.chat("hello", model="openai:gpt-5-nano")
except EffGenAuthError as exc:
    print(type(exc).__name__, "status", exc.status_code, "->", exc)

nothing_there = EffGenClient(base_url="http://127.0.0.1:9", max_retries=0)
try:
    nothing_there.health()
except EffGenConnectionError as exc:
    print(type(exc).__name__, "->", exc)`} />

      <Terminal
        command="python errors.py"
        output={`EffGenAuthError status 401 -> Invalid API key. Check the key the client sends against the one this server was started with (EFFGEN_API_KEY). Check the API key the client was built with.
EffGenConnectionError -> HTTPConnectionPool(host='127.0.0.1', port=9): Max retries exceeded with url: /health (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=9): Failed to establish a new connection: [Errno 111] Connection refused"))`}
        caption={
          <>
            The message is the server's own, redacted and bounded, followed by what to do about that
            status. <code>max_retries=0</code> is what makes the failure immediate here; the default
            of 3 would back off and try again first.
          </>
        }
      />

      <h2>TypeScript</h2>

      <p>
        The TypeScript client is in the framework repository at{' '}
        <code>clients/typescript/</code>. It uses the platform's own <code>fetch</code> and{' '}
        <code>ReadableStream</code> and has no runtime dependencies, so it runs on Node 18 or newer,
        Deno, Bun and in a browser. Build it with <code>npm install &amp;&amp; npm run build</code>{' '}
        in that directory.
      </p>

      <CodeTabs
        tabs={[
          {
            label: 'chat',
            language: 'typescript',
            code: `import { EffGenClient } from "effgen-client";

const client = new EffGenClient({
  baseUrl: "http://127.0.0.1:8000",
  apiKey: process.env.EFFGEN_API_KEY,
});

const res = await client.chat("What is 127 * 43?", { tools: ["calculator"] });
console.log(res.content);
console.log(res.model, res.usage);`,
          },
          {
            label: 'stream',
            language: 'typescript',
            code: `for await (const chunk of client.chatStream("Count from 1 to 5.")) {
  process.stdout.write(chunk);
}`,
          },
          {
            label: 'embed & health',
            language: 'typescript',
            code: `const vectors = await client.embed(["Hello", "World"]);
console.log(vectors.length, vectors[0].length);

const health = await client.health();
console.log(health.status, health.ok);`,
          },
          {
            label: 'errors',
            language: 'typescript',
            code: `import {
  EffGenAuthError,
  EffGenClientError,
  EffGenRateLimitError,
} from "effgen-client";

try {
  await client.chat("hello");
} catch (err) {
  if (err instanceof EffGenAuthError) console.error("key rejected", err.statusCode);
  else if (err instanceof EffGenRateLimitError) console.error("slow down");
  else if (err instanceof EffGenClientError) console.error("client error", err.message);
  else throw err;
}`,
          },
        ]}
      />

      <ApiTable
        headers={['Option', 'Type', 'Default']}
        rows={[
          [<code>baseUrl</code>, <code>string</code>, <code>"http://localhost:8000"</code>],
          [<code>apiKey</code>, <code>string</code>, '— (omit in dev mode)'],
          [<code>timeoutMs</code>, <code>number</code>, <code>60000</code>],
          [<code>maxRetries</code>, <code>number</code>, <code>3</code>],
          [<code>backoffBaseMs</code>, <code>number</code>, <code>500</code>],
          [
            <code>fetchImpl</code>,
            <code>typeof fetch</code>,
            <>
              the platform's <code>fetch</code> — override it in a test
            </>,
          ],
        ]}
        caption={
          <>
            <code>EffGenClientOptions</code> from <code>clients/typescript/src/index.ts</code>. The
            error classes mirror the Python ones one for one, and{' '}
            <code>EffGenAPIError</code> carries <code>statusCode</code> and <code>payload</code>.
          </>
        }
      />

      <Callout type="note" title="Two small differences from the Python client">
        <p>
          <code>ChatResponse.toolCalls</code> is camel-cased in TypeScript where Python has{' '}
          <code>tool_calls</code>, and <code>HealthStatus</code> adds a boolean{' '}
          <code>ok</code> alongside <code>status</code>. The default embedding model also differs —
          pass the model explicitly to <code>embed()</code> rather than relying on either default.
        </p>
      </Callout>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>EffGenAuthError</code> with status 401
            </>,
            'The key the client sends is not the one the server was started with.',
            <>
              With nothing configured, the server mints an ephemeral key at startup and a restart
              replaces it. Set <code>EFFGEN_API_KEY</code> on the server and give the client the same
              value.
            </>,
          ],
          [
            <>
              <code>EffGenConnectionError</code> naming the host and port
            </>,
            'Nothing is listening there.',
            <>
              Check the server is up and that <code>base_url</code> has no path on it —{' '}
              <code>http://host:8000</code>, not <code>http://host:8000/v1</code>.
            </>,
          ],
          [
            'A call takes far longer than the timeout',
            'The retry ran its course first: three attempts, each with its own timeout, plus backoff.',
            <>
              Lower <code>max_retries</code> where a fast failure matters more than a successful
              retry.
            </>,
          ],
          [
            <>
              <code>EffGenAPIError</code> with status 403
            </>,
            'The key is valid; the principal is not allowed that tool or model.',
            <>
              Read <code>GET /rbac/policy</code> as that principal —{' '}
              <Link to="/api-server">roles and budgets</Link>.
            </>,
          ],
          [
            'A named tool has no effect',
            'The server resolves tool names against its own registry, and an unknown name is not a tool.',
            <>
              Check the spelling against <code>effgen tools list</code>; the{' '}
              <Link to="/tools/gallery">gallery</Link> has all of them.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/openai-api', '/api-server', '/agents']} />
    </DocPage>
  );
}
