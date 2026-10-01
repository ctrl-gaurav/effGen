import { Library } from 'lucide-react';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { modelCount, providersWithCatalog, siteData, version } from '../siteData';

const BROWSE = siteData.cli.command_options['models browse'] ?? [];
const LIST = siteData.cli.command_options['models list'] ?? [];

export default function Catalog() {
  return (
    <DocPage
      subtitle="The bundled catalog: what it records about each model, and how to browse it."
      icon={<Library size={48} />}
    >
      <p>
        effGen ships a catalog of {modelCount} models across {providersWithCatalog} providers, and
        it is not just a list of ids: each record carries the context window, the maximum output,
        the published prices, whether the model can call tools or see images, and when the record
        was last checked against the provider's live API. Routing, cost estimates and every "can
        this model do that" check read it.
      </p>

      <h2>Searching it</h2>

      <CodeBlock
        language="bash"
        code={`effgen models browse --provider openai --tools --sort price-in --limit 6`}
      />

      <Terminal
        command="effgen models browse --provider openai --tools --sort price-in --limit 6"
        output={`Model Catalog
PROVIDER  MODEL ID        CONTEXT   MAXOUT    $/1M IN   $/1M OUT  TOOLS  VIS  FREE
openai  gpt-5-nano    1,047,576   32,768      $0.05       $0.4    yes  yes     -
openai  gpt-4.1-nano  1,047,576   32,768       $0.1       $0.4    yes  yes     -
openai  gpt-4o-mini     128,000   16,384      $0.15       $0.6    yes  yes     -
openai  gpt-5.4-nano  1,047,576   32,768       $0.2      $1.25    yes  yes     -
openai  gpt-5-mini    1,047,576   32,768      $0.25         $2    yes  yes     -
openai  gpt-4.1-mini  1,047,576   32,768       $0.4       $1.6    yes  yes     -

showing 6 of 30  ·  pricing from catalog snapshot`}
        caption={`Run against effGen 1.0.0.`}
      />

      <h2>What a record holds</h2>

      <CodeBlock
        filename="record.py"
        code={`import json

from effgen.models.registry import list_models

record = next(m for m in list_models("gemini") if m["model_id"] == "gemini-3.1-flash-lite")
print(json.dumps(record, indent=2, sort_keys=True))`}
      />

      <Terminal command="python record.py" output={`{
  "context": 1000000,
  "family": "flash-lite",
  "free_tier": true,
  "max_output": 32768,
  "model_id": "gemini-3.1-flash-lite",
  "notes": "Cheapest Gemini 3.x text model. Best free-tier choice. Grounding not available on free tier.",
  "pricing_per_1m_input": 0.25,
  "pricing_per_1m_output": 1.5,
  "rpd": 500,
  "rpm": 15,
  "supports_audio": true,
  "supports_grounding": false,
  "supports_native_tools": true,
  "supports_thinking": true,
  "supports_video": true,
  "supports_vision": true,
  "tier": "free",
  "tpm": 250000
}`} />

      <ApiTable
        headers={['Key', 'What it is']}
        rows={[
          [<code>model_id</code>, 'The id the provider serves it under — what goes after the prefix.'],
          [<code>family</code>, 'The model family, used for search and for grouping.'],
          [<code>context</code>, 'The context window in tokens. effGen plans compaction against it.'],
          [<code>max_output</code>, 'The largest completion the provider will return.'],
          [<code>supports_native_tools</code>, 'Whether tools can be offered as function definitions rather than as text.'],
        ]}
        caption="The five keys every record in every catalog carries."
      />

      <p>
        On top of those, each provider records what it publishes. The prices are the ones to know
        about, because there are two spellings: OpenAI's records use{' '}
        <code>input_price_per_1m</code> and <code>output_price_per_1m</code> (and{' '}
        <code>cached_input_price_per_1m</code>, which is the only place a cached-prefix price is
        recorded), while the other catalogs use <code>pricing_per_1m_input</code> and{' '}
        <code>pricing_per_1m_output</code>. Read both, or read the price through{' '}
        <code>effgen models info</code>, which normalises them.
      </p>

      <ApiTable
        headers={['Key', 'Recorded by', 'What it is']}
        rows={[
          [<code>supports_vision</code>, 'anthropic, fireworks, gemini, groq, hf, together', 'Image input.'],
          [<code>supports_audio</code> , 'gemini, hf', 'Audio input.'],
          [<code>supports_video</code>, 'gemini', 'Video input.'],
          [<code>supports_streaming</code>, 'fireworks, groq, replicate, together', 'Token streaming.'],
          [<code>supports_thinking</code>, 'anthropic, gemini', 'Extended reasoning.'],
          [<code>supports_reasoning</code>, 'openai', 'The o-series reasoning tier.'],
          [<code>supports_grounding</code>, 'gemini', 'A web search the provider runs itself.'],
          [<code>supports_prompt_caching</code>, 'anthropic, openai', 'Whether a prefix can be cached.'],
          [<code>supports_structured_output</code>, 'hf', 'Schema-constrained output.'],
          [<code>free_tier</code> , 'cerebras, gemini', 'Whether the model has a free quota.'],
          [<code>rpm</code>, 'cerebras, fireworks, gemini, groq, together', 'Requests per minute, with rpd, rph, tpm, tpd and tph beside it where published.'],
          [<code>cost_per_second_usd</code>, 'replicate', 'Compute-time billing, for models that are not priced by token at all.'],
          [<code>serverless</code>, 'together', 'Whether the model runs without a dedicated endpoint.'],
          [<code>requires_endpoint</code>, 'hf', 'Whether a dedicated endpoint has to be started first.'],
          [<code>deprecated</code>, 'cerebras', 'Whether the provider has announced its retirement.'],
          [<code>notes</code>, 'gemini, groq', 'A sentence about when to pick it.'],
        ]}
        caption="A key that is absent is absent, rather than filled in with a guess — which is why a filter reads the key it needs rather than assuming every record has it."
      />

      <Callout type="warning" title="Unpriced is not free">
        <p>
          A model whose provider publishes no token price shows <code>unpriced</code>, and a call
          to it reports no cost rather than <code>$0</code>. <code>--free</code> filters to models
          that are actually on a free tier, which is a different thing. Several Replicate models
          bill by compute second rather than by token, so a token count cannot give you a cost at
          all.
        </p>
      </Callout>

      <h2>effgen models browse</h2>

      <ParamTable
        nameLabel="Flag"
        params={BROWSE.map((option) => ({ name: option.name, description: option.description }))}
        caption={<><code>effgen models browse --help</code>, {version}.</>}
      />

      <CodeBlock
        language="bash"
        code={`effgen models browse --vision --min-context 500000 --sort context --desc --limit 6
effgen models browse --search qwen --tools --json
effgen models browse --free --max-price-out 0 --include-local`}
      />

      <Terminal
        command="effgen models browse --vision --min-context 500000 --sort context --desc --limit 6"
        output={`Model Catalog
PROVIDER   MODEL ID                                             CONTEXT   MAXOUT    $/1M IN   $/1M OUT  TOOLS  VIS  FREE
gemini     gemini-3.1-pro-preview                             2,000,000   65,536         $2        $12    yes  yes     -
gemini     gemini-2.5-pro                                     2,000,000   32,768      $1.25        $10    yes  yes     -
hf         meta-llama/Llama-Guard-4-12B                       1,048,576   16,384       $0.2       $0.2      -  yes     -
hf         meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8  1,048,576   16,384      $0.27      $0.85      -  yes     -
fireworks  accounts/fireworks/models/kimi-k3                  1,048,576   16,384   unpriced   unpriced    yes  yes     -
fireworks  accounts/fireworks/models/inkling                  1,048,576   16,384   unpriced   unpriced    yes  yes     -

showing 6 of 27  ·  pricing from catalog snapshot`}
      />

      <h2>effgen models list</h2>
      <p>
        <code>browse</code> is the cross-provider search. <code>list</code> is the per-provider
        view: with no flags it prints the registry overview and, below it, every model already in
        your HuggingFace cache with its size and whether the download completed.
      </p>

      <ParamTable
        nameLabel="Flag"
        params={LIST.map((option) => ({ name: option.name, description: option.description }))}
        caption={<><code>effgen models list --help</code>, {version}.</>}
      />

      <Terminal command="effgen models list --provider gemini" output={`Available Models
gemini — 8 models (auth: ready, verified: 2026-08-13)
  gemini-3.1-flash-lite   ctx=  1000000  in=    $0.25  out=     $1.5  tools vision *
  gemini-3-flash-preview  ctx=  1000000  in=     $0.5  out=       $3  tools vision
  gemini-3.1-pro-preview  ctx=  2000000  in=       $2  out=      $12  tools vision
  gemini-2.5-flash-lite   ctx=  1000000  in=     $0.1  out=     $0.4  tools vision
  gemini-2.5-flash        ctx=  1000000  in=     $0.3  out=     $2.5  tools vision
  gemini-2.5-pro          ctx=  2000000  in=    $1.25  out=      $10  tools vision
  gemma-4-26b-a4b-it      ctx=    65536  in=     free  out=     free        
  gemma-4-31b-it          ctx=    65536  in=     free  out=     free        `} />

      <h2>One model's record</h2>

      <CodeBlock language="bash" code={`effgen models info openai:gpt-5-nano
effgen models info openai:gpt-5-nano --json`} />

      <p>
        <code>info</code> takes a bare id too, and raises <code>AmbiguousModelError</code> when the
        id is in more than one catalog — prefix it to say which you mean.
      </p>

      <h2>Keeping it current</h2>
      <p>
        The bundled catalog is a snapshot, and every record carries the date it was checked. Where
        a provider has a live models API, <code>effgen models refresh</code> pulls the current list
        and writes a new snapshot; <code>--dry-run</code> shows what would change without writing
        anything.
      </p>

      <CodeBlock
        language="bash"
        code={`effgen models refresh --provider gemini --dry-run
effgen models refresh                      # every provider you hold a key for`}
      />

      <Terminal command="effgen models refresh --provider gemini --dry-run" output={`Refresh model catalog (dry run)
✓ gemini: 19 live models (+11 / -0 / ~0 changed) — would update snapshot
    + gemini-3.1-flash-lite-preview
    + gemini-3.1-flash-live-preview
    + gemini-3.1-pro-preview-customtools
    + gemini-3.5-flash
    + gemini-3.5-flash-lite
    + gemini-3.6-flash
    + gemini-3.7-flash
    + gemini-flash-latest
    + gemini-flash-lite-latest
    + gemini-omni-flash-preview`} />

      <ParamTable
        nameLabel="Flag"
        params={(siteData.cli.command_options['models refresh'] ?? []).map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          (siteData.cli.command_options['models refresh'] ?? []).length > 0 ? (
            <>
              <code>effgen models refresh --help</code>, {version}.
            </>
          ) : (
            <>
              <code>--provider PROVIDER</code> refreshes one provider,{' '}
              <code>--dry-run</code> shows what would change without writing the snapshot.
            </>
          )
        }
      />

      <p>
        A snapshot drifts in both directions: it can miss a model the provider has added, and it
        can keep one the provider has retired — a call to which comes back{' '}
        <code>ModelNotFoundError</code> rather than being caught at lookup, because the catalog
        still knows the id. The dry run above shows both sides for one provider.
      </p>

      <p>
        Several adapters also expose drift detection from Python — a check that compares the
        bundled snapshot against the live API and warns when models have appeared, disappeared or
        been repriced, so a stale catalog is visible rather than quietly wrong.
      </p>

      <h2>Reading the catalog from Python</h2>

      <CodeBlock
        filename="filter.py"
        code={`from effgen.models.registry import list_models

rows = [
    m for m in list_models("openai")
    if m.get("supports_native_tools") and (m.get("input_price_per_1m") or 0) <= 0.10
]
for m in sorted(rows, key=lambda m: m["input_price_per_1m"]):
    print(f"{m['model_id']:24s} {m['context']:>8,} ctx  \${m['input_price_per_1m']}/1M in")`}
      />

      <Terminal command="python filter.py" output={`gpt-5-nano               1,047,576 ctx  $0.05/1M in
gpt-4.1-nano             1,047,576 ctx  $0.1/1M in`} />

      <h2>What is in the catalog today</h2>

      <ApiTable
        headers={['Provider', 'Models', 'Tool-capable', 'Vision', 'Free tier', 'Checked']}
        rows={siteData.models.providers
          .filter((provider) => provider.models > 0)
          .map((provider) => [
            <code>{provider.name}</code>,
            String(provider.models),
            String(provider.supports_tools),
            String(provider.supports_vision),
            String(provider.free_tier),
            provider.verified_on ?? '—',
          ])}
        caption={
          <>
            Read from the installed catalog. <code>openai_compatible</code> is not listed because
            it carries none — it serves whatever your endpoint serves.
          </>
        }
      />

      <h2>When a lookup goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>AmbiguousModelError</code>,
            'The id is in more than one catalog.',
            <>
              Prefix it. <code>e.providers</code> lists which ones.
            </>,
          ],
          [
            <code>KeyError</code>,
            'No catalog carries that id.',
            <>
              <code>effgen models browse --search &lt;text&gt;</code> finds it if it exists.
            </>,
          ],
          [
            'The catalog knows an id the provider 404s on',
            'The snapshot is older than the provider’s current line-up.',
            <>
              <code>effgen models refresh</code>, then check with <code>--dry-run</code> first if
              you want to see the change before it is written.
            </>,
          ],
          [
            <><code>unpriced</code> in the price column</>,
            'The provider publishes no token price for that model.',
            'Expect no cost figure from a run against it. That is reported, not estimated.',
          ],
        ]}
      />

      <SeeAlso paths={['/models', '/providers', '/routing']} />
    </DocPage>
  );
}
