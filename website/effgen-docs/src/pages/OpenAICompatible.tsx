import { Server } from 'lucide-react';
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

export default function OpenAICompatible() {
  return (
    <DocPage
      subtitle="Pointing an agent at vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio or a gateway with base_url."
      icon={<Server size={48} />}
    >
      <p>
        vLLM, SGLang, TGI, llama.cpp's server, Ollama, LM Studio, LiteLLM and most corporate
        gateways all expose the OpenAI chat-completions API. Give effGen a{' '}
        <code>base_url</code> and it drives the model you are already serving, instead of loading a
        second copy of the weights inside the agent's process.
      </p>

      <h2>Pointing at one</h2>

      <CodeTabs
        tabs={[
          {
            label: 'A model',
            filename: 'endpoint.py',
            code: `from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)
print(model.generate("What is 6 times 7?").text)`,
          },
          {
            label: 'An agent',
            filename: 'endpoint_agent.py',
            code: `from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
))
print(agent.run("What is 6 times 7?").output)`,
          },
        ]}
        caption="base_url is the whole instruction: give one and the call goes to that server, whatever provider says."
      />

      <Callout type="note" title="What was actually run here">
        <p>
          The captures on this page were taken against effGen's own server —{' '}
          <code>effgen serve</code> speaks the same protocol — because it is the endpoint that
          exists on any machine with effGen installed. Point the same code at vLLM or Ollama and
          nothing in it changes but the port and the model id.
        </p>
      </Callout>

      <CodeBlock
        language="bash"
        code={`# in one shell
export EFFGEN_API_KEY=local-dev-key
effgen serve -p 8077`}
      />

      <CodeBlock
        filename="served.py"
        code={`from effgen import load_model

model = load_model(
    "openai:gpt-5-nano",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8077/v1",
    api_key="local-dev-key",
)
print(model.list_served_models()[:4])
print(model.generate("What is 6 times 7? Answer with the number only.").text.strip())
print("cost:", model.generate("Say OK.").metadata.get("cost"))`}
      />

      <Terminal command="python served.py" output={`['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
42
cost: None`} caption={`Run against effGen 1.0.0.`} />

      <h2>Why serve the model separately</h2>
      <p>
        Loading in-process means one copy of the weights per agent process, no sharing between
        agents, no continuous batching across callers, and a GPU tied to the agent's lifetime. A
        shared server fixes all four: the weights load once, every caller's requests batch
        together, and the GPU outlives any individual run. It is also the only way to have several
        frameworks — or several versions of your own service — generate under identical settings,
        which is what a fair comparison needs.
      </p>

      <h2>Where the endpoint comes from</h2>

      <ApiTable
        headers={['Order', 'Source']}
        rows={[
          [
            '1',
            <>
              <code>base_url=</code> passed to <code>load_model()</code>, <code>AgentConfig</code>{' '}
              or the adapter
            </>,
          ],
          ['2', <code>EFFGEN_BASE_URL</code>],
          ['3', <code>OPENAI_BASE_URL</code>],
          ['4', <code>OPENAI_API_BASE</code>],
        ]}
        caption="effGen's own variable is consulted first, so you can point effGen at a server without redirecting every other OpenAI client on the machine."
      />

      <Callout type="warning" title="A machine-wide OPENAI_BASE_URL cannot silently reroute an OpenAI call">
        <p>
          <code>provider="openai"</code> <em>with</em> a <code>base_url</code> routes here, because
          a URL of your own means the model ids, the context window and the pricing are the
          server's rather than OpenAI's. <em>Without</em> one it stays on OpenAI. Ask for{' '}
          <code>provider="openai_compatible"</code> to pick the environment variable up.
        </p>
      </Callout>

      <p>These all reach the same adapter:</p>

      <CodeBlock
        filename="forms.py"
        code={`from effgen import load_model

URL = "http://127.0.0.1:9/v1"          # nothing is listening; no call is made here
for kwargs in (
    dict(provider="openai_compatible", base_url=URL),
    dict(base_url=URL),
    dict(provider="openai", base_url=URL, api_key="EMPTY"),
    dict(provider="openai-compatible", base_url=URL),
    dict(provider="vllm_server", base_url=URL),
):
    model = load_model("my-model", **kwargs)
    print(f"{type(model).__name__:26s} {kwargs}")`}
      />

      <Terminal command="python forms.py" output={`OpenAICompatibleAdapter    {'provider': 'openai_compatible', 'base_url': 'http://127.0.0.1:9/v1'}
OpenAICompatibleAdapter    {'base_url': 'http://127.0.0.1:9/v1'}
OpenAICompatibleAdapter    {'provider': 'openai', 'base_url': 'http://127.0.0.1:9/v1', 'api_key': 'EMPTY'}
OpenAICompatibleAdapter    {'provider': 'openai-compatible', 'base_url': 'http://127.0.0.1:9/v1'}
OpenAICompatibleAdapter    {'provider': 'vllm_server', 'base_url': 'http://127.0.0.1:9/v1'}`} />

      <p>
        <code>"openai-compatible"</code>, <code>"openai_compat"</code>,{' '}
        <code>"compatible"</code>, <code>"server"</code>, <code>"vllm_server"</code> and{' '}
        <code>"local_server"</code> are all accepted spellings of the provider, and{' '}
        <code>load_model(f"openai_compatible:{'{model_id}'}", base_url=URL)</code> works too.
      </p>

      <h2>Credentials</h2>
      <p>
        A local server that authenticates nothing needs none — effGen sends a placeholder, which
        vLLM, SGLang, TGI, llama.cpp and Ollama all accept. Pass a real one for a gateway that
        checks it:
      </p>

      <CodeBlock
        code={`import os

from effgen import load_model

model = load_model(
    "my-model",
    provider="openai_compatible",
    base_url="https://gateway.internal/v1",
    api_key=os.environ["GATEWAY_TOKEN"],
)`}
      />

      <h2>What effGen does not assume</h2>
      <p>The server serves its own model ids, so no OpenAI catalog is consulted.</p>

      <ParamTable
        nameLabel="Setting"
        params={[
          {
            name: 'context_length',
            type: 'int',
            default: '32768',
            description:
              'The window effGen plans compaction against. Pass the real one when your server’s differs — effGen warns when it is assuming, naming the value and the flag that sets it, rather than failing later at a size nobody chose.',
          },
          {
            name: 'supports_reasoning',
            type: 'bool',
            default: 'False',
            description: 'Set it when the model you serve emits a reasoning stream.',
          },
          {
            name: 'sampling',
            type: '—',
            description: (
              <>
                The full surface — <code>top_p</code>, <code>top_k</code>, <code>seed</code> and
                the penalties — is offered, because every implementation of the protocol accepts
                it.
              </>
            ),
          },
          {
            name: 'cost',
            type: '—',
            description:
              'Calls report no price. What your own server costs cannot be derived from a token count, so effGen states nothing rather than a fabricated $0.',
          },
        ]}
        caption="Passed alongside base_url on load_model or the adapter."
      />

      <CodeBlock
        filename="window.py"
        code={`from effgen import load_model

model = load_model(
    "openai:gpt-5-nano",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8078/v1",
    api_key="local-dev-key",
)
print("assumed window:", model.get_context_length())

sized = load_model(
    "openai:gpt-5-nano",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8078/v1",
    api_key="local-dev-key",
    context_length=8192,
)
print("stated window: ", sized.get_context_length())`}
      />

      <Terminal command="python window.py" output={`assumed window: 32768
stated window:  8192`} />

      <h2>Asking the endpoint what it serves</h2>

      <CodeBlock
        continues
        code={`served = load_model(
    "whatever",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8078/v1",
    api_key="local-dev-key",
)
print(served.list_served_models())`}
      />

      <p>
        An endpoint that does not implement <code>/models</code> returns an empty list rather than
        failing — some minimal servers have nothing to say about themselves.
      </p>

      <h2>When nothing is listening</h2>
      <p>
        A refused connection, a host that does not resolve and a route that does not exist are all
        classified <strong>unreachable</strong> — separately from a server that answered badly,
        which stays transient and is still retried — and they raise{' '}
        <code>BackendUnreachableError</code>.
      </p>

      <CodeBlock
        filename="unreachable.py"
        code={`from effgen import Agent, AgentConfig
from effgen.models.errors import BackendUnreachableError

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:9/v1",     # nothing is listening here
))

try:
    agent.run("What is 6 times 7?")
except BackendUnreachableError as e:
    print(type(e).__name__)
    print(str(e)[:220])`}
      />

      <Terminal command="python unreachable.py" output={`BackendUnreachableError
openai did not answer (model='Qwen/Qwen2.5-7B-Instruct'): OpenAI generation failed [will_retry]: Connection error.. Nothing answered at that endpoint — check the server is running and the base_url, host and port are righ`} />

      <Callout type="warning" title="There is no opt-out, by design">
        <p>
          This raises whatever <code>AgentConfig.raise_on_error</code> says. A task that ran and
          failed is a result you can inspect; a backend that was never reached is not, and
          returning one silently is how a whole batch completes against nothing and still looks
          healthy in the summary. Classification reads the exception chain, because provider SDKs
          shorten a refused port to "Connection error." and keep the real cause on{' '}
          <code>__cause__</code>.
        </p>
      </Callout>

      <h2>Serving a model to point at</h2>

      <CodeTabs
        tabs={[
          {
            label: 'vLLM',
            language: 'bash',
            code: `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
# then base_url="http://127.0.0.1:8000/v1" and whatever --served-model-name was
# given, which defaults to the repo id.`,
          },
          {
            label: 'Ollama',
            language: 'bash',
            code: `ollama serve
# then base_url="http://127.0.0.1:11434/v1" and the model id you pulled.`,
          },
          {
            label: 'effGen',
            language: 'bash',
            code: `export EFFGEN_API_KEY=local-dev-key
effgen serve -p 8077
# then base_url="http://127.0.0.1:8077/v1" and a provider:model id it can route.`,
          },
        ]}
      />

      <h2>Common failures</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>BackendUnreachableError</code>,
            'Nothing answered at that address.',
            <>
              Check the server is up, and that the port and the <code>/v1</code> suffix are right.
            </>,
          ],
          [
            'A 404 naming the model id',
            'The server does not serve that id.',
            <>
              <code>list_served_models()</code> asks it what it does serve. On vLLM the id is{' '}
              <code>--served-model-name</code>, which defaults to the repo id.
            </>,
          ],
          [
            'A 401 from a gateway',
            'The endpoint checks credentials and the placeholder was rejected.',
            <>
              Pass <code>api_key=</code>.
            </>,
          ],
          [
            'A warning naming an assumed context window',
            'The server did not say how large its window is, so effGen assumed 32,768.',
            <>
              Pass <code>context_length=</code> with the real number.
            </>,
          ],
          [
            'Cost is None',
            'Not a failure.',
            'A server of your own has no published token price, so effGen reports nothing rather than a made-up zero.',
          ],
          [
            'A call went to OpenAI instead of your server',
            <>
              <code>provider="openai"</code> was used without a <code>base_url</code>, so the
              environment variable was ignored on purpose.
            </>,
            <>
              Pass <code>base_url=</code>, or ask for{' '}
              <code>provider="openai_compatible"</code>.
            </>,
          ],
        ]}
      />

      <h2>The other direction</h2>
      <p>
        This page is about effGen <em>calling</em> a server that speaks the protocol.{' '}
        <Link to="/openai-api">OpenAI-compatible API</Link> is about effGen{' '}
        <em>serving</em> your agents over it, so that someone else's OpenAI client can call them.
      </p>

      <SeeAlso paths={['/local-models', '/models', '/openai-api']} />
    </DocPage>
  );
}
