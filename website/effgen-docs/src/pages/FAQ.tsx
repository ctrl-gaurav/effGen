import { HelpCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';
import {
  modelCount,
  presetCount,
  providerCount,
  providersWithCatalog,
  pythonVersions,
  siteData,
  toolCount,
  version,
} from '../siteData';

export default function FAQ() {
  return (
    <DocPage
      subtitle="The questions that come up first: hardware, keys, cost, offline use and model choice."
      icon={<HelpCircle size={48} />}
    >
      <h2>Do I need a GPU?</h2>
      <p>
        No. A cloud model needs only a key, and a small local model runs on a CPU — slowly, but it
        runs. A GPU matters when you want to run larger weights yourself, and{' '}
        <code>quantization="4bit"</code> is what fits a model onto a smaller card.
      </p>

      <CodeBlock
        code={`from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")                # a key, no GPU
agent = create_agent("math", "Qwen/Qwen2.5-1.5B-Instruct")       # a GPU or a CPU, no key`}
      />

      <p>
        <code>effgen doctor</code> reports what the machine has: the GPUs it can see, the CUDA the
        driver supports, the CUDA torch was built for, and whether the two agree.{' '}
        <Link to="/local-models">Local models and engines</Link> covers the four engines.
      </p>

      <h2>Do I need a paid API?</h2>
      <p>
        No. Several providers have free tiers, a local model needs no account at all, and the
        built-in search and reference tools use free endpoints. What a paid key buys is the larger
        cloud models.
      </p>
      <p>
        Of the {modelCount} catalogued models,{' '}
        {siteData.models.capability_totals.free_tier} are marked free-tier and{' '}
        {siteData.models.capability_totals.priced} are priced.{' '}
        <code>effgen models browse --free</code> lists the free ones.
      </p>

      <h2>Which model should I start with?</h2>

      <ApiTable
        headers={['If you want', 'Try', 'Why']}
        rows={[
          [
            'A cheap cloud model with tools',
            <code>openai:gpt-5-nano</code>,
            'Tool-capable, a large window, and among the cheapest per token in the catalog.',
          ],
          [
            'A free-tier cloud model',
            <code>gemini:gemini-3.1-flash-lite</code>,
            'Free-tier eligible, tool-capable, and the one provider that can ground an answer in a web search it runs itself.',
          ],
          [
            'Something on your own machine',
            <code>Qwen/Qwen2.5-1.5B-Instruct</code>,
            'Small enough to run on a CPU, instruction-tuned, and it downloads once.',
          ],
          [
            'A model you already serve',
            <>
              anything, with <code>base_url</code>
            </>,
            <>
              effGen drives your vLLM, Ollama or gateway endpoint instead of loading a second copy
              — see <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
            </>,
          ],
        ]}
        caption={
          <>
            <code>effgen models browse --tools --free</code> answers this against the live catalog
            rather than against a table someone wrote down.
          </>
        }
      />

      <h2>Can it work offline?</h2>
      <p>
        Yes, with a local model. The catalog, the presets and the tool registry all ship inside the
        package, so nothing needs the network to start; after a model's weights are downloaded
        once, a run touches nothing outside the machine — as long as the tools you give it do not
        themselves call out.
      </p>

      <CodeBlock
        filename="offline.py"
        code={`from effgen import get_tool_registry, list_presets
from effgen.models.registry import list_models, list_providers

# Nothing here touches the network: the catalog, the presets and the tool
# registry all ship inside the package.
print(len(list_providers()), "adapters,", sum(len(list_models(p)) for p in list_providers()), "catalogued models")
print(len(list_presets()), "presets")
print(len(get_tool_registry().list_tools()), "tools")`}
      />

      <Terminal command="python offline.py" output={`10 adapters, 416 catalogued models
9 presets
66 tools`} caption={`Run against effGen 1.0.0 with no network calls. The counts are the same in ${version}.`} />

      <h2>What does a run cost?</h2>
      <p>
        Every response carries it. <code>metadata["cost_usd"]</code> is what that run cost, from
        the catalog's published prices and the tokens actually used — and a model with no published
        price reports no cost rather than a fabricated zero.
      </p>

      <CodeBlock
        filename="cost.py"
        code={`from effgen import create_agent

agent = create_agent("minimal", "openai:gpt-5-nano")
r = agent.run("Say OK.")

print(r.metadata["cost_usd"], r.metadata["total_tokens"], r.metadata["latency_ms"])
print(r.model, r.provider)`}
      />

      <Terminal command="python cost.py" output={`5.62e-05 158 3936.5
openai:gpt-5-nano openai`} />

      <p>
        A daily cap across every run lives in <code>~/.effgen/budget.json</code>:{' '}
        <code>effgen cost set-budget 1.00</code> sets one, <code>effgen cost today</code> reports
        what has been spent, and <code>effgen quickstart --init</code> sets $1.00 a day when you
        have none. <Link to="/cost">Cost and budgets</Link> has the reports.
      </p>

      <h2>Where do my keys go?</h2>
      <p>
        In the environment, or in a <code>.env</code> file effGen finds on its own —{' '}
        <code>$EFFGEN_DOTENV</code> first, then <code>~/.effgen/.env</code>, then the nearest{' '}
        <code>.env</code> above your working directory. A variable already exported wins over a
        file, and <code>EFFGEN_NO_DOTENV=1</code> turns the file search off entirely for a
        production process. <code>effgen doctor</code> says which keys effGen can see and never
        prints one. <Link to="/configuration">Configuration</Link> has the full order and the
        per-provider variable names.
      </p>

      <h2>My agent does not call any tool</h2>
      <p>
        Whether to call a tool is the model's decision. Three things move it: a system prompt that
        says the tool exists and when to use it; a model large enough to follow the format — below
        about 3B this gets unreliable; and <code>tool_calling_mode</code>. In{' '}
        <code>"react"</code> the tools are described in the prompt and the model is asked for a
        step, which small models follow more reliably than a native function-calling API.
      </p>
      <p>
        If a particular tool <em>must</em> have run, check it —{' '}
        <code>response.tool_calls.by_name("calculator").total</code> — rather than assuming.{' '}
        <Link to="/tool-calling">Tool calling</Link> covers the strategies.
      </p>

      <h2>My agent loops without answering</h2>
      <p>
        It reached <code>max_iterations</code>. Raise the cap, simplify the task, or use a more
        capable model. To see what it did get to, set <code>raise_on_error=False</code> and read{' '}
        <code>metadata["partial_output"]</code> — with the default <code>True</code> the run raises
        instead.
      </p>

      <h2>CUDA out of memory</h2>
      <p>
        The weights do not fit. In order of least effort: load in 4-bit with{' '}
        <code>quantization="4bit"</code>; pick a smaller model; lower{' '}
        <code>gpu_memory_utilization</code> on vLLM; select one card with{' '}
        <code>CUDA_VISIBLE_DEVICES=0</code>. If <code>torch.cuda.is_available()</code> is{' '}
        <code>False</code> while <code>nvidia-smi</code> lists your GPUs, the torch wheel is built
        for a CUDA runtime the driver does not support — <Link to="/installation">Installation</Link>{' '}
        has the table.
      </p>

      <h2>Which Python versions work?</h2>
      <p>
        {pythonVersions.join(', ')}. 3.10 was dropped for 1.0.0 — see{' '}
        <Link to="/migration">Migrating to {version}</Link>.
      </p>

      <h2>How do I add my own tool?</h2>
      <p>
        Decorate a function with <code>@tool</code>, or build one from a function with{' '}
        <code>Tool.from_function</code>. To share tools across projects, package them as a plugin.{' '}
        <Link to="/custom-tools">Writing tools and plugins</Link> covers both, and{' '}
        <Link to="/tools/gallery">the tool gallery</Link> lists the {toolCount} that ship.
      </p>

      <h2>Which provider does effGen support?</h2>
      <p>
        {providerCount} adapters, {providersWithCatalog} of them with a bundled catalog:{' '}
        {siteData.models.adapters.map((name, i) => (
          <span key={name}>
            {i > 0 ? ', ' : ''}
            <code>{name}</code>
          </span>
        ))}
        . The last of those is not a hosted service — it is how effGen talks to a server of your
        own. <Link to="/providers">Providers</Link> has what each one supports.
      </p>

      <h2>Do I have to configure an agent from scratch?</h2>
      <p>
        No — there are {presetCount} presets, each a set of tools and a system prompt someone has
        already worked out. <code>create_agent("math", "openai:gpt-5-nano")</code> is the whole
        configuration. <Link to="/presets">Presets</Link> lists them.
      </p>

      <h2>Something imports and then fails at run time</h2>
      <p>
        An optional dependency is missing. effGen keeps the heavy stacks — vLLM, vector databases,
        OCR, OpenCV, the speech models — out of the base install, and the error names the extra to
        install. <Link to="/installation">Installation</Link> lists what each extra carries.
      </p>

      <Callout type="tip" title="When a message is not enough">
        <p>
          <code>effgen doctor</code> is the first thing to run for anything environmental — keys,
          GPUs, torch, the sandbox and git are all in one report.{' '}
          <code>EFFGEN_LOG_LEVEL=DEBUG</code> and <code>-v</code> turn on the detail behind a
          failure, and <Link to="/errors">Errors and exceptions</Link> lists every typed error with
          what raises it.
        </p>
      </Callout>

      <SeeAlso paths={['/quickstart', '/installation', '/errors']} />
    </DocPage>
  );
}
