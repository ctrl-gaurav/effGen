import { Cpu } from 'lucide-react';
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
import { modelCount, providerCount, providersWithCatalog, siteData } from '../siteData';
import { siteHref } from '../siteLinks';

export default function Models() {
  return (
    <DocPage
      subtitle="Naming a model, loading it, and what load_model gives you back."
      icon={<Cpu size={48} />}
    >
      <p>
        Everything that generates text in effGen is a <code>BaseModel</code>, and{' '}
        <code>load_model</code> is how you get one. The same call reaches a cloud provider, a set
        of weights on your own GPU, or a server you already run — the difference is in the id you
        give it.
      </p>

      <p className="doc-crosslink">
        This page is the reference. For the catalogue itself — every provider, what each one
        serves and what it costs — see <a href={siteHref('/models')}>the models page</a> on the
        main site.
      </p>

      <h2>Loading one</h2>

      <CodeBlock
        filename="load.py"
        code={`from effgen import load_model

model = load_model("openai:gpt-5-nano")          # provider prefix
result = model.generate("Name the largest ocean in one word.")

print(result.text.strip())
print(result.tokens_used, result.finish_reason, result.model_name)`}
      />

      <Terminal command="python load.py" output={`Pacific
202 stop gpt-5-nano`} caption={`Run against effGen 1.0.0.`} />

      <p>
        An <code>Agent</code> takes either the loaded model or the id itself —{' '}
        <code>AgentConfig(model="openai:gpt-5-nano")</code> loads it for you, and fails at
        construction if the id is wrong or the key is missing rather than at the first run.
      </p>

      <h2>The three ways to name a model</h2>

      <ApiTable
        headers={['Form', 'Example', 'What it means']}
        rows={[
          [
            <code>provider:model</code>,
            <code>openai:gpt-5-nano</code>,
            'A model in a provider catalog, reached over that provider’s API. The prefix removes any ambiguity.',
          ],
          [
            <>a bare id plus <code>provider=</code></>,
            <code>load_model("gpt-5-nano", provider="openai")</code>,
            'The same thing, when the id comes from somewhere that cannot carry a prefix.',
          ],
          [
            <>a bare HuggingFace repo id</>,
            <code>Qwen/Qwen2.5-1.5B-Instruct</code>,
            'Weights run on this machine, through a local engine. No key, no network after the first download.',
          ],
          [
            <code>engine:model</code>,
            <code>transformers:Qwen/Qwen2.5-1.5B-Instruct</code>,
            <>
              The same weights, with the engine named explicitly:{' '}
              {siteData.models.local_engines.map((engine, i) => (
                <span key={engine}>
                  {i > 0 ? ', ' : ''}
                  <code>{engine}</code>
                </span>
              ))}
              .
            </>,
          ],
        ]}
        caption={
          <>
            <code>hf:&lt;repo&gt;</code> is the <em>remote</em> HuggingFace Inference API. To run
            the same repo locally, drop the prefix and pass an engine — see{' '}
            <Link to="/local-models">Local models and engines</Link>.
          </>
        }
      />

      <Callout type="warning" title="A bare id that exists on several providers is refused">
        <p>
          <code>openai/gpt-oss-120b</code> is in more than one catalog, so effGen raises{' '}
          <code>AmbiguousModelError</code> naming the providers rather than picking one. Prefix it.
        </p>
      </Callout>

      <h2>load_model</h2>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'model_name',
            type: 'str',
            required: true,
            description: 'The model identifier, in any of the four forms above.',
          },
          {
            name: 'engine',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                <code>"vllm"</code>, <code>"transformers"</code>, <code>"auto-fast"</code>, or{' '}
                <code>None</code> for automatic. <code>auto-fast</code> prefers vLLM when it
                imports and the GPU is usable, and falls back to transformers otherwise;{' '}
                <code>None</code> defaults to transformers.
              </>
            ),
          },
          {
            name: 'engine_config',
            type: 'dict | None',
            default: 'None',
            description: 'Engine options passed straight through.',
          },
          {
            name: 'tensor_parallel_size',
            type: 'int | None',
            default: 'None',
            description:
              'How many GPUs to shard across (vLLM only). Auto-detected from model size when not given.',
          },
          {
            name: 'gpu_memory_utilization',
            type: 'float | None',
            default: 'None',
            description:
              'Fraction of GPU memory to use, 0.0–1.0 (vLLM only). The default is 0.90; lower it on CUDA out-of-memory.',
          },
          {
            name: 'apply_chat_template',
            type: 'bool',
            default: 'True',
            description:
              "Apply the model's chat template automatically for instruction-tuned models (vLLM only).",
          },
          {
            name: 'provider',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                Route to this remote provider — the same choice the prefix makes. Use{' '}
                <code>"openai_compatible"</code> for a server of your own.
              </>
            ),
          },
          {
            name: 'base_url',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                An endpoint speaking the OpenAI protocol. Giving one routes the call to the
                OpenAI-compatible adapter whatever <code>provider</code> says, because the ids, the
                window and the pricing are then the server's. Falls back to{' '}
                <code>EFFGEN_BASE_URL</code>, <code>OPENAI_BASE_URL</code> or{' '}
                <code>OPENAI_API_BASE</code>.
              </>
            ),
          },
          {
            name: 'api_key',
            type: 'str | None',
            default: 'None',
            description:
              'Credential for that endpoint. A local server that checks nothing needs none; effGen sends a placeholder.',
          },
          {
            name: '**kwargs',
            type: 'Any',
            description: (
              <>
                Additional parameters, such as <code>quantization="4bit"</code>,{' '}
                <code>trust_remote_code=True</code> or <code>require_gpu=True</code>.
              </>
            ),
          },
        ]}
        caption={
          <>
            <code>
              load_model(model_name, engine=None, engine_config=None, tensor_parallel_size=None,
              gpu_memory_utilization=None, apply_chat_template=True, provider=None, base_url=None,
              api_key=None, **kwargs) -&gt; BaseModel
            </code>
          </>
        }
      />

      <h2>What you get back</h2>
      <p>
        A <code>BaseModel</code>. Every adapter and every local engine answers to the same surface,
        which is what makes swapping one for another a one-line change.
      </p>

      <ParamTable
        nameLabel="Method"
        params={[
          {
            name: 'generate(prompt, config=None, **kwargs)',
            type: 'GenerationResult',
            description: 'One completion.',
          },
          {
            name: 'generate_stream(prompt, config=None, **kwargs)',
            type: 'Iterator[str]',
            description: 'The same completion, yielded as it arrives.',
          },
          {
            name: 'chat(messages, config=None, tools=None, **kwargs)',
            type: 'GenerationResult',
            description: 'A completion from a message list, which is what a multi-turn loop needs.',
          },
          {
            name: 'generate_with_tools(prompt, tools, config=None, messages=None, **kwargs)',
            type: 'GenerationResult',
            description: (
              <>
                A completion the model may answer with a tool call. The calls arrive in{' '}
                <code>metadata["tool_calls"]</code> — see <Link to="/tool-calling">Tool calling</Link>.
              </>
            ),
          },
          {
            name: 'build_assistant_message(result)',
            type: 'dict',
            description: "The provider's own shape for replaying that turn back to it.",
          },
          {
            name: 'build_tool_result_message(call_id, name, content)',
            type: 'dict',
            description: "The provider's own shape for a tool result. Together these make one tool loop portable across providers.",
          },
          {
            name: 'count_tokens(text)',
            type: 'TokenCount',
            description: 'How long a piece of text is in this model’s units.',
          },
          {
            name: 'get_context_length()',
            type: 'int',
            description: 'The window effGen plans compaction against.',
          },
          {
            name: 'supports_tool_calling()',
            type: 'bool',
            description: (
              <>
                Whether the model offers native tool calling.{' '}
                <code>tool_call_support()</code> says how, and{' '}
                <code>streams_tool_calls()</code> whether it does so while streaming.
              </>
            ),
          },
          {
            name: 'get_total_cost()',
            type: 'float',
            description: (
              <>
                What this model has cost since it was loaded. <code>reset_cost()</code> zeroes it.
                An unpriced model reports no cost rather than <code>0</code>.
              </>
            ),
          },
          {
            name: 'load() / unload() / is_loaded()',
            type: 'None / None / bool',
            description:
              'Explicit lifecycle. load_model does the loading; unload releases GPU memory when you are done with a local model.',
          },
          {
            name: 'get_metadata()',
            type: 'dict',
            description: 'The catalog record behind this model — context, pricing, capabilities.',
          },
        ]}
        caption="Adapters add their own methods on top — generate_structured, list_served_models and the provider-native tool paths among them."
      />

      <h3>GenerationResult</h3>

      <ApiTable
        headers={['Field', 'Type', 'What it is']}
        rows={[
          [<code>text</code>, <code>str</code>, 'The generated text.'],
          [<code>tokens_used</code>, <code>int</code>, 'Total tokens for the call.'],
          [
            <code>finish_reason</code>,
            <code>str</code>,
            'Why generation stopped — the model finished, hit the token budget, or hit a stop sequence.',
          ],
          [<code>model_name</code>, <code>str</code>, 'The model that answered.'],
          [
            <code>metadata</code>,
            <code>dict</code>,
            <>
              Cost, prompt and completion token counts, <code>cached_input_tokens</code>, and{' '}
              <code>tool_calls</code> — which is always present and always a list.
            </>,
          ],
        ]}
      />

      <h2>The registry</h2>
      <p>
        Every adapter registers itself the first time its module is imported, which{' '}
        <code>load_model</code> does for you. {providerCount} adapters are registered;{' '}
        {providersWithCatalog} ship a bundled catalog, together describing {modelCount} models.{' '}
        <code>openai_compatible</code> ships none, because it serves whatever the endpoint you
        point it at serves.
      </p>

      <CodeBlock
        filename="registry.py"
        code={`from effgen.models.registry import lookup

provider, adapter_cls, info = lookup("openai:gpt-5-nano")
print(provider, adapter_cls.__name__)
for key in ("context", "max_output", "supports_native_tools", "input_price_per_1m", "output_price_per_1m"):
    print(f"  {key} = {info.get(key)}")`}
      />

      <Terminal command="python registry.py" output={`openai OpenAIAdapter
  context = 1047576
  max_output = 32768
  supports_native_tools = True
  input_price_per_1m = 0.05
  output_price_per_1m = 0.4`} />

      <ParamTable
        nameLabel="Function"
        params={[
          {
            name: 'list_providers()',
            type: 'list[str]',
            description: 'Every registered provider name, sorted.',
          },
          {
            name: 'list_models(provider)',
            type: 'list[dict]',
            description: (
              <>
                Every model that provider's catalog carries, each dict carrying{' '}
                <code>model_id</code> plus its metadata. Raises <code>KeyError</code> for a
                provider that is not registered.
              </>
            ),
          },
          {
            name: 'lookup(model_id, provider=None)',
            type: 'tuple[str, type, dict]',
            description: (
              <>
                Resolve an id to its provider, adapter class and catalog record. Understands the{' '}
                <code>provider:model</code> prefix.
              </>
            ),
          },
          {
            name: 'check_keys(providers=None)',
            type: 'dict',
            description: (
              <>
                Which credentials are present in this environment, as{' '}
                <code>{'{provider: {available, env_key, env_keys_checked}}'}</code>. It reads no
                value.
              </>
            ),
          },
        ]}
        caption={
          <>
            From <code>effgen.models.registry</code> and <code>effgen.models.auth</code>; all four
            are also exported from the top-level package.
          </>
        }
      />

      <h2>From the command line</h2>

      <CodeBlock
        language="bash"
        code={`effgen models list                     # every provider, and the local cache
effgen models list --provider gemini   # one provider, in full detail
effgen models info openai:gpt-5-nano   # one model's record
effgen models browse --tools --free    # search and filter across every provider
effgen models refresh                  # update the catalog from each keyed provider`}
      />

      <Terminal command="effgen models info openai:gpt-5-nano" output={`Model: openai:gpt-5-nano
┌───────────────────────────┬─────────────────┐
│ Provider                  │ openai          │
│ Display name              │ gpt-5-nano      │
│ Family                    │ chat            │
│ Context window            │ 1,047,576       │
│ Max output                │ 32,768          │
│ Price ($/1M in / out)     │ $0.05/$0.4      │
│ Tool calling              │ yes             │
│ Coding                    │ suitable        │
│ Vision                    │ yes             │
│ Audio                     │ no              │
│ Free tier                 │ no              │
│ Rate limits (rpm/tpm/rpd) │ — / — / —       │
│ Deprecated                │ no              │
│ Price source              │ bundled-catalog │
│ Verified on               │ 2026-06-17      │
│ Auth ready                │ yes             │
└───────────────────────────┴─────────────────┘

Use: effgen run --provider openai -m gpt-5-nano "..."`} />

      <p>
        <Link to="/catalog">The model catalog and pricing</Link> documents what the catalog records
        and every flag <code>browse</code> takes.
      </p>

      <h2>When loading fails</h2>

      <ApiTable
        headers={['Error', 'When', 'What to do']}
        rows={[
          [
            <code>AmbiguousModelError</code>,
            'The bare id exists in more than one catalog.',
            <>
              Prefix it, or pass <code>provider=</code>. The error lists the providers on{' '}
              <code>e.providers</code>.
            </>,
          ],
          [
            <code>ModelNotFoundError</code>,
            'The provider does not serve that id.',
            <>
              The message suggests near matches.{' '}
              <code>effgen models list --provider &lt;name&gt;</code> shows the ids, and{' '}
              <code>effgen models refresh</code> updates a stale catalog.
            </>,
          ],
          [
            <code>KeyError</code>,
            'No catalog anywhere carries the id.',
            'Check the spelling, or name the provider so effGen knows whose catalog to consult.',
          ],
          [
            <code>ModelAuthError</code>,
            'The credential was missing or rejected.',
            <>
              <code>effgen doctor</code> says which keys are visible.
            </>,
          ],
          [
            <code>BackendUnreachableError</code>,
            'Nothing answered at the endpoint.',
            <>
              Raised regardless of <code>raise_on_error</code>. See{' '}
              <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
            </>,
          ],
          [
            <code>CapabilityNotSupportedError</code>,
            'The model cannot do what was asked of it — vision on a text-only model, for instance.',
            <>
              <code>effgen models browse --vision</code> and its siblings filter the catalog by
              capability.
            </>,
          ],
        ]}
      />

      <CodeBlock
        filename="ambiguous.py"
        code={`from effgen.models.errors import AmbiguousModelError
from effgen.models.registry import lookup

try:
    lookup("openai/gpt-oss-120b")
except AmbiguousModelError as e:
    print(type(e).__name__, "->", e.model_id)
    print("  on:", e.providers)

provider, adapter_cls, info = lookup("groq:openai/gpt-oss-120b")
print("resolved:", provider, info["context"])`}
      />

      <Terminal command="python ambiguous.py" output={`AmbiguousModelError -> openai/gpt-oss-120b
  on: ['groq', 'hf', 'replicate', 'together']
resolved: groq 131072`} />

      <SeeAlso paths={['/providers', '/catalog', '/local-models']} />
    </DocPage>
  );
}
