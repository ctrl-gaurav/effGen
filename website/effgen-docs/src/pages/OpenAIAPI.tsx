import { Plug2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  CodeTabs,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function OpenAIAPI() {
  return (
    <DocPage
      subtitle="The endpoints `effgen serve` exposes, and talking to them with the official OpenAI SDK."
      icon={<Plug2 size={48} />}
    >
      <p>
        <Link to="/api-server"><code>effgen serve</code></Link> speaks the OpenAI protocol, so the
        official <code>openai</code> client — or anything else that talks to an OpenAI-compatible
        endpoint — points at it unchanged. Any model id effGen can reach is callable through it,
        including ids from providers that are not OpenAI, and a handful of OpenAI names are aliased
        so an OpenAI-only client works without knowing that.
      </p>

      <Callout type="note" title="Two directions, one protocol">
        <p>
          This page is effGen <em>serving</em> the OpenAI protocol.{' '}
          <Link to="/openai-compatible">Any OpenAI-compatible server</Link> is the other direction —
          effGen as the client of vLLM, SGLang, Ollama, LM Studio or a gateway. The two compose:
          effGen can serve the protocol in front of a model it reaches by speaking the protocol.
        </p>
      </Callout>

      <h2>The shortest call</h2>

      <CodeBlock filename="first.py" code={`import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ["EFFGEN_API_KEY"])

answer = client.chat.completions.create(
    model="openai:gpt-5-nano",
    messages=[{"role": "user", "content": "Reply with the single word ok."}],
)
print(answer.choices[0].message.content)
print(answer.model, answer.usage)`} />

      <Terminal
        command="python first.py"
        output={`ok
openai:gpt-5-nano CompletionUsage(completion_tokens=266, prompt_tokens=26, total_tokens=292, completion_tokens_details=None, prompt_tokens_details=None)`}
        caption={`Against effGen 1.0.0 serving on 127.0.0.1:8000. The usage numbers are the provider's own.`}
      />

      <h2>Endpoints</h2>

      <ApiTable
        headers={['Method', 'Path', 'What it does']}
        rows={[
          [
            'POST',
            <code>/v1/chat/completions</code>,
            'Chat completions, streaming and not. Accepts tools; the server runs them.',
          ],
          ['POST', <code>/v1/completions</code>, 'Legacy text completions, streaming and not.'],
          [
            'GET',
            <code>/v1/models</code>,
            'The aliases plus every id this process has already served. Not the catalogue.',
          ],
          [
            'GET',
            <code>/v1/models/catalog</code>,
            <>
              The bundled catalogue, filterable by <code>provider</code>. This is the exhaustive
              list.
            </>,
          ],
          ['POST', <code>/v1/embeddings</code>, 'Text embeddings, computed locally.'],
        ]}
        caption={
          <>
            The <code>/v1</code> surface, from the running server's own{' '}
            <code>/openapi.json</code>. <Link to="/api-server">The API server page</Link> covers the
            non-<code>/v1</code> routes — <code>/health</code>, <code>/whoami</code>,{' '}
            <code>/rbac/*</code>, <code>/metrics</code>, <code>/slo</code> and effGen's own{' '}
            <code>/run</code> and <code>/tools</code>.
          </>
        }
      />

      <h2>Authentication</h2>

      <p>
        The server is fail-closed. Send the key as either header, and the SDK's{' '}
        <code>api_key</code> becomes the bearer token:
      </p>

      <CodeTabs
        tabs={[
          {
            label: 'openai SDK',
            language: 'python',
            code: `import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ["EFFGEN_API_KEY"])`,
          },
          {
            label: 'curl',
            language: 'bash',
            code: `curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $EFFGEN_API_KEY"
curl -s http://127.0.0.1:8000/v1/models -H "X-API-Key: $EFFGEN_API_KEY"`,
          },
        ]}
        caption={
          <>
            <Link to="/api-server">Roles and budgets</Link> apply to whoever the key or token
            resolves to — a 403 from <code>/v1/chat/completions</code> is a policy denial, not a bad
            request.
          </>
        }
      />

      <h2>Model ids</h2>

      <p>
        Ids are effGen ids. <code>provider:model</code> is canonical, <code>provider/model</code> is
        accepted and normalised, and a bare id with no provider is loaded locally.
      </p>

      <CodeBlock filename="models.py" code={`import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ["EFFGEN_API_KEY"])

for model in client.models.list().data:
    print(model.id)`} />

      <Terminal
        command="python models.py"
        output={`gpt-4
gpt-4-turbo
gpt-4o
gpt-4o-mini
gpt-3.5-turbo
gpt-3.5-turbo-instruct
default
effgen-default
Qwen/Qwen2.5-3B-Instruct
openai:gpt-5-nano`}
        caption={
          <>
            Six aliases, two default names, and the ids this server had served when the list was
            asked for. Any other <code>provider:model</code> id the server can reach is callable
            whether or not it appears here.
          </>
        }
      />

      <h3>Aliases</h3>

      <ApiTable
        headers={['Alias', 'Resolves to']}
        rows={[
          [
            <>
              <code>gpt-4</code>, <code>gpt-4-turbo</code>, <code>gpt-4o</code>
            </>,
            <code>Qwen/Qwen2.5-7B-Instruct</code>,
          ],
          [
            <>
              <code>gpt-4o-mini</code>, <code>gpt-3.5-turbo</code>
            </>,
            <code>Qwen/Qwen2.5-3B-Instruct</code>,
          ],
          [
            <>
              <code>effgen-default</code>, <code>default</code>
            </>,
            <>
              <code>EFFGEN_DEFAULT_MODEL</code>, and <code>Qwen/Qwen2.5-3B-Instruct</code> when that
              is unset.
            </>,
          ],
        ]}
        caption="So a client that only knows OpenAI names gets an answer instead of a 404."
      />

      <p>
        Aliasing is never silent. The response <code>model</code> field names the model that actually
        ran, and a non-standard <code>effgen</code> object documents the mapping — OpenAI clients
        ignore unknown top-level keys, so it costs them nothing:
      </p>

      <Terminal
        command={`curl -s http://127.0.0.1:8000/v1/chat/completions \\
  -H "Authorization: Bearer $EFFGEN_API_KEY" -H 'Content-Type: application/json' \\
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'`}
        output={`{"id":"chatcmpl-ec21e240bfb543ea98314514","object":"chat.completion","created":1787543168,"model":"Qwen/Qwen2.5-3B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I assist you today?"},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":44,"completion_tokens":10,"total_tokens":54},"effgen":{"requested_model":"gpt-4o-mini","resolved_model":"Qwen/Qwen2.5-3B-Instruct","alias_applied":true,"run_id":"5368a18882ee"}}`}
        title="curl"
        caption={
          <>
            <code>requested_model</code> is what was asked for, <code>resolved_model</code> is what
            ran, and <code>alias_applied</code> says whether a substitution happened. This call took
            40 seconds because the alias resolves to a local model that had to be loaded first —
            name a hosted id to avoid that.
          </>
        }
      />

      <h3>The whole catalogue</h3>

      <Terminal
        command={`curl -s -H "Authorization: Bearer $EFFGEN_API_KEY" \\
  'http://127.0.0.1:8000/v1/models/catalog?provider=openai'`}
        output={`30 models
  gpt-5.4
  gpt-5.4-mini
  gpt-5.4-nano
  gpt-5.4-pro
  gpt-5`}
        title="curl"
        caption={
          <>
            The bundled catalogue, filtered. <Link to="/catalog">The model catalog and pricing</Link>{' '}
            is the same data offline, and <code>effgen models browse</code> is the interactive view.
          </>
        }
      />

      <h2>Streaming</h2>

      <p>
        Streaming is real server-sent events, incremental as the model produces text. Matching
        OpenAI, no usage chunk is sent unless you ask for one with{' '}
        <code>stream_options.include_usage</code>.
      </p>

      <CodeBlock filename="stream.py" code={`import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ["EFFGEN_API_KEY"])

stream = client.chat.completions.create(
    model="openai:gpt-5-nano",
    messages=[{"role": "user", "content": "Count from 1 to 5, digits only."}],
    stream=True,
    stream_options={"include_usage": True},
)
for event in stream:
    if event.choices and event.choices[0].delta.content:
        print(event.choices[0].delta.content, end="")
    if event.usage:
        print("\\nusage:", event.usage)`} />

      <Terminal command="python stream.py" output={`1 2 3 4 5
usage: CompletionUsage(completion_tokens=210, prompt_tokens=30, total_tokens=240, completion_tokens_details=None, prompt_tokens_details=None)`} />

      <Callout type="note" title="A stream that fails does not simply stop">
        <p>
          A mid-stream failure is emitted as a terminal SSE event carrying an <code>error</code>{' '}
          object and <code>finish_reason: "error"</code>, followed by <code>[DONE]</code> — rather
          than a truncated answer a client would read as a complete one. Both{' '}
          <code>/v1/chat/completions</code> and <code>/v1/completions</code> behave this way.
        </p>
      </Callout>

      <h2>Tools</h2>

      <p>
        effGen runs tools <em>server-side</em>, through its own agent loop. Passing{' '}
        <code>tools</code> lets the agent use them and return an answer that is already resolved; the
        server does not stream client-side <code>tool_calls</code> deltas for you to execute and send
        back. That is a deliberate difference from the OpenAI API, and it is the whole compatibility
        story for tools.
      </p>

      <CodeBlock
        continues
        filename="tools.py"
        code={`r = client.chat.completions.create(
    model="openai:gpt-5-nano",
    messages=[{"role": "user", "content": "Use the calculator to compute 127 * 43."}],
    tools=[{"type": "function",
            "function": {"name": "calculator",
                         "parameters": {"type": "object",
                                        "properties": {"expression": {"type": "string"}}}}}],
)
print(r.choices[0].message.content)`}
        caption={
          <>
            Tool names resolve against effGen's built-in{' '}
            <Link to="/tools/gallery">tool registry</Link> and are subject to the caller's role
            policy. <Link to="/clients">The Python client</Link> takes the same thing as a list of
            names.
          </>
        }
      />

      <h2>Embeddings</h2>

      <CodeBlock filename="embeddings.py" code={`import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ["EFFGEN_API_KEY"])

vectors = client.embeddings.create(model="text-embedding-3-small", input=["hello", "world"])
print(len(vectors.data), "vectors of", len(vectors.data[0].embedding), "dimensions")`} />

      <Terminal
        command="python embeddings.py"
        output={`2 vectors of 384 dimensions`}
        caption="Computed locally with a SentenceTransformer, so the dimension is the local model's, not OpenAI's."
      />

      <h2>Usage accounting</h2>

      <p>
        <code>usage</code> is real: the provider's own counts where the upstream API returns them,
        and tokenizer counts otherwise. It is never a character-count estimate. That is what makes{' '}
        <Link to="/cost">the cost figures</Link> add up against a provider invoice.
      </p>

      <h2>Errors</h2>

      <p>
        Every error the server returns uses the OpenAI envelope with an accurate status and a
        redacted message — the model routes, embeddings, the ops and RBAC routes, the dashboard and
        playground endpoints, failures raised before a route runs (auth, rate limit, RBAC, body-size
        cap, request validation) and failures raised outside any route (unknown URL, wrong method,
        an unhandled error). <strong>Branch on <code>type</code> and <code>code</code></strong>, not
        on the message text.
      </p>

      <Terminal
        command={`# empty messages, an unknown model, and no credential
curl -s -w '\\nHTTP %{http_code}\\n' http://127.0.0.1:8000/v1/chat/completions …`}
        output={`{"error":{"message":"messages must not be empty","type":"invalid_request_error","param":null,"code":"empty_messages"}}
HTTP 400
{"error":{"message":"openai error (model='no-such-model'): The model \`no-such-model\` does not exist or you do not have access to it. Did you mean: o1-pro, o3-pro, o3-mini? Available openai models: gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.4-pro, gpt-5, gpt-5-mini (+24 more). Model id not found — run \`effgen models list\` to see ids, \`effgen models refresh\` to update the catalog, and verify the id/provider prefix.","type":"model_not_found","param":null,"code":"model_not_found"}}
HTTP 404
{"error": {"message": "Missing API key (send 'Authorization: Bearer <key>' or 'X-API-Key: <key>')", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}
HTTP 401`}
        title="curl"
        maxLines={16}
      />

      <ApiTable
        headers={['Status', 'Means', 'Typical code']}
        rows={[
          [<code>400</code>, 'Invalid request.', <><code>empty_messages</code>, <code>empty_content</code>, <code>empty_prompt</code>, <code>empty_input</code></>],
          [<code>401</code>, 'No credential, or one the server does not accept.', <code>invalid_api_key</code>],
          [<code>403</code>, 'Authenticated, but the role policy refuses the tool or model.', '—'],
          [<code>404</code>, 'The model id or the URL does not exist.', <code>model_not_found</code>],
          [<code>405</code>, 'Wrong method for that path.', '—'],
          [<code>413</code>, 'Request body over the cap.', '—'],
          [<code>422</code>, 'The body did not validate.', '—'],
          [<code>429</code>, 'Rate limit, or a role budget cap.', <code>rate_limit_exceeded</code>],
          [
            <code>502</code>,
            'A key is configured for that provider and the provider rejected it.',
            <code>upstream_auth_failed</code>,
          ],
          [
            <code>503</code>,
            'No key is configured for that provider — a configuration gap, not a bad key.',
            <code>upstream_key_missing</code>,
          ],
          [<code>504</code>, 'The upstream call timed out.', '—'],
          [<code>500</code>, 'Anything else.', '—'],
        ]}
        caption={
          <>
            The two upstream statuses say different things about the credential, consistently across
            every provider, and neither ever echoes a key. A request with nothing for the model to
            act on is refused with <code>400</code> before any billed call.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'The SDK reports a connection error',
            <>
              <code>base_url</code> is missing the <code>/v1</code> suffix, or the server is not up.
            </>,
            <>
              The OpenAI SDK wants <code>http://host:8000/v1</code>. Note that{' '}
              <Link to="/clients"><code>EffGenClient</code></Link> wants the opposite — the bare
              origin.
            </>,
          ],
          [
            <>
              <code>404 model_not_found</code> on an id you can see in the provider's own console
            </>,
            "The id is not in effGen's catalogue for that provider.",
            <>
              The message names near misses and what is available. <code>effgen models refresh</code>{' '}
              picks up new ids; <Link to="/catalog">the catalog page</Link> explains the refresh.
            </>,
          ],
          [
            <>
              <code>503 upstream_key_missing</code>
            </>,
            'The server has no API key for that provider.',
            <>
              Set the provider's variable in the server's environment — not the client's. The server
              makes the call.
            </>,
          ],
          [
            'A first call takes tens of seconds, then later ones are fast',
            'The id resolved to a local model, which had to be loaded. Loaded models are pooled and reused after that.',
            <>
              <code>EFFGEN_MODEL_POOL_SIZE</code> sets how many stay warm (default 4). Name a hosted
              id if you do not want a local load at all.
            </>,
          ],
          [
            'No usage on a streamed response',
            'You did not ask for one.',
            <>
              Pass <code>stream_options={'{'}"include_usage": true{'}'}</code>. This matches OpenAI's
              own behaviour.
            </>,
          ],
          [
            <>
              <code>tool_calls</code> never arrive for the client to execute
            </>,
            'By design — the server executes tools and returns the resolved answer.',
            <>
              If your code must run the tool itself, run the loop on your side with an{' '}
              <Link to="/agents">Agent</Link> rather than through this endpoint.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          One envelope now covers every failure — unknown URLs, wrong methods, missing static
          assets, unhandled route errors, RBAC denials and the shutdown drain included. A
          content-free request is refused before it is billed, an absent provider key is a{' '}
          <code>503</code> on every provider, an upstream <code>429</code> passes its delay on as{' '}
          <code>Retry-After</code>, a mid-stream failure emits a terminal error event instead of
          truncating, and the body-size cap now covers <code>/v1/embeddings</code>.
        </p>
      </Callout>

      <SeeAlso paths={['/api-server', '/clients', '/openai-compatible']} />
    </DocPage>
  );
}
