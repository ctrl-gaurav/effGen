import { Target } from 'lucide-react';
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
import { presetCount, siteData } from '../siteData';

const PRESETS = siteData.presets.items;

export default function Presets() {
  return (
    <DocPage
      subtitle="Ready-made agent configurations, what each one turns on, and how to start from one."
      icon={<Target size={48} />}
    >
      <p>
        A preset is an <code>AgentConfig</code> somebody has already worked out: a set of tools, a
        system prompt written for them, and the temperature and iteration cap that suit the work.{' '}
        <code>create_agent</code> takes a preset name and a model id and gives you the agent.
        There are {presetCount}.
      </p>

      <h2>Starting from one</h2>

      <CodeTabs
        tabs={[
          {
            label: 'Python',
            filename: 'preset.py',
            code: `from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
print(agent.run("What is 17% of 250?"))`,
          },
          {
            label: 'Command line',
            language: 'bash',
            code: `effgen run --preset math -m openai:gpt-5-nano "What is 17% of 250?"`,
          },
        ]}
      />

      <Terminal command="python preset.py" output={`42.5`} caption={`Run against effGen 1.0.0.`} />

      <Callout type="tip" title="New here? Start with math or minimal">
        <p>
          Both are small and fast, so a small model has few tool schemas to reason over.{' '}
          <code>general</code> is the broad one — every category of built-in tool at once, which
          makes it heavy: its tool schemas alone are{' '}
          {PRESETS.find((preset) => preset.name === 'general')?.approx_tokens_per_call.toLocaleString()}{' '}
          tokens on every request. The unsandboxed shell is in <code>coding</code>, not in{' '}
          <code>general</code>.
        </p>
      </Callout>

      <h2>The {presetCount} presets</h2>

      <ApiTable
        headers={['Preset', 'Tools', 'Tokens per call', 'What it is for']}
        rows={PRESETS.map((preset) => [
          <code>{preset.name}</code>,
          preset.tool_count === 0 ? 'none' : String(preset.tool_count),
          preset.approx_tokens_per_call === 0
            ? '—'
            : `~${preset.approx_tokens_per_call.toLocaleString()}`,
          preset.description,
        ])}
        caption={
          <>
            Read from the installed preset registry. <em>Tokens per call</em> is the size of the
            tool schemas sent on every request — it is what decides whether a preset fits a
            small-context or rate-limited model.
          </>
        }
      />

      <h2>What each one sets</h2>
      <p>
        Beyond the tools, a preset pins the settings that suit its work. Anything not listed keeps
        the <Link to="/agents">
          <code>AgentConfig</code>
        </Link>{' '}
        default.
      </p>

      <ApiTable
        headers={['Preset', 'temperature', 'max_iterations', 'memory', 'sub-agents']}
        rows={PRESETS.map((preset) => [
          <code>{preset.name}</code>,
          <code>{preset.temperature}</code>,
          <code>{preset.max_iterations}</code>,
          preset.enable_memory ? 'on' : 'off',
          preset.enable_sub_agents ? 'on' : 'off',
        ])}
      />

      <h2>Which tools each one carries</h2>

      <ApiTable
        headers={['Preset', 'Tools']}
        rows={PRESETS.map((preset) => [
          <code>{preset.name}</code>,
          preset.tools.length === 0 ? (
            <em>none — direct model inference only</em>
          ) : (
            preset.tools.map((tool, i) => (
              <span key={tool}>
                {i > 0 ? ' · ' : ''}
                <code>{tool}</code>
              </span>
            ))
          ),
        ])}
        caption={
          <>
            Every one of these names is a registry name, so it can also be passed to{' '}
            <code>extra_tools</code> or to the command line's <code>-t/--tools</code>. The{' '}
            <Link to="/tools/gallery">tool gallery</Link> documents each of them.
          </>
        }
      />

      <h2>create_agent</h2>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'preset',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                A preset name. Pass this <strong>or</strong> <code>domain</code>, not both.
              </>
            ),
          },
          {
            name: 'model',
            type: 'BaseModel | str | None',
            default: 'None',
            description: (
              <>
                A loaded model or an id. A cloud model may use a <code>provider:model</code>{' '}
                prefix, a local one an <code>engine:model</code> prefix. When omitted,{' '}
                <code>EFFGEN_DEFAULT_MODEL</code> is used if set — otherwise an error says how to
                pick one, because effGen never silently picks a paid model.
              </>
            ),
          },
          {
            name: 'domain',
            type: 'Domain | None',
            default: 'None',
            description: (
              <>
                Build the agent from a knowledge domain instead of a preset — its system prompt,
                tools and guardrails. See <Link to="/domains">Domains</Link>.
              </>
            ),
          },
          {
            name: 'agent_name',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                Override the agent's name. <code>name</code> is accepted as an alias.
              </>
            ),
          },
          {
            name: 'extra_tools',
            type: 'list | None',
            default: 'None',
            description: (
              <>
                Tools added on top of the preset's. Each entry is a tool instance or a registry
                name; an unknown name raises <code>ValueError</code> with suggestions.{' '}
                <code>tools</code> is accepted as an alias.
              </>
            ),
          },
          {
            name: 'knowledge_base',
            type: 'Any',
            default: 'None',
            description: (
              <>
                Only used by the <code>rag</code> preset, where it is required: a file, a
                directory, raw text, a list of those, or a prebuilt{' '}
                <code>VectorMemoryStore</code>, indexed at creation.
              </>
            ),
          },
          {
            name: 'system_prompt',
            type: 'str | None',
            default: 'None',
            description: "Replace the preset's system prompt.",
          },
          {
            name: 'max_iterations',
            type: 'int | None',
            default: 'None',
            description: "Override the preset's iteration cap.",
          },
          {
            name: 'temperature',
            type: 'float | None',
            default: 'None',
            description: "Override the preset's temperature.",
          },
          {
            name: 'enable_memory',
            type: 'bool | None',
            default: 'None',
            description: "Override the preset's memory setting.",
          },
          {
            name: 'guardrails',
            type: 'Any',
            default: 'None',
            description: (
              <>
                A guardrail chain or a preset name such as <code>"standard"</code>, applied to the
                agent's input and output.
              </>
            ),
          },
          {
            name: 'session_id',
            type: 'str | None',
            default: 'None',
            description:
              'A persistent session id, so multi-turn context carries across processes — the same id the command line exposes.',
          },
          {
            name: '**config_overrides',
            type: 'Any',
            description: (
              <>
                Model-loading options (<code>engine</code>, <code>quantization</code>,{' '}
                <code>tensor_parallel_size</code>, <code>gpu_memory_utilization</code>,{' '}
                <code>apply_chat_template</code>, <code>trust_remote_code</code>) go to{' '}
                <code>load_model</code>; everything else goes to <code>AgentConfig</code>. An
                unrecognised keyword raises <code>TypeError</code> listing what is accepted.
              </>
            ),
          },
        ]}
        caption={
          <>
            <code>
              create_agent(preset=None, model=None, *, domain=None, agent_name=None,
              extra_tools=None, knowledge_base=None, system_prompt=None, max_iterations=None,
              temperature=None, enable_memory=None, guardrails=None, session_id=None,
              **config_overrides)
            </code>
          </>
        }
      />

      <h2>Changing what a preset gives you</h2>
      <p>
        Every override is a keyword. Nothing has to be rebuilt from{' '}
        <code>AgentConfig</code> to change one setting.
      </p>

      <CodeBlock
        filename="override.py"
        code={`from effgen import create_agent

agent = create_agent(
    "math",
    "openai:gpt-5-nano",
    extra_tools=["datetime"],          # a registry name, or a tool instance
    temperature=0.0,
)
print([t.metadata.name for t in agent.config.tools])
print(agent.run("What is 144 / 12?").text)`}
      />

      <Terminal command="python override.py" output={`['calculator', 'python_repl', 'datetime']
12.0`} />

      <h2>Reading a preset without building an agent</h2>

      <CodeBlock
        filename="inspect.py"
        code={`from effgen import list_presets
from effgen.presets import get_preset

for name, description in list_presets().items():
    preset = get_preset(name)
    print(f"{name:11s} {len(preset.tool_names):2d} tools  {description[:48]}")

math = get_preset("math")
print(math.tool_names, math.temperature, math.max_iterations, math.enable_sub_agents)`}
      />

      <Terminal command="python inspect.py" output={`math         2 tools  Mathematical reasoning agent with Calculator and
research    15 tools  Research agent with WebSearch, URLFetch, Wikiped
coding       4 tools  Coding agent with CodeExecutor, PythonREPL, File
general     31 tools  General-purpose agent with a broad set of built-
rag          1 tools  Retrieval-Augmented Generation agent with hybrid
minimal      0 tools  Minimal agent with no tools — direct model infer
multimodal   7 tools  Multimodal agent for image, audio, and video und
notify       4 tools  Notification agent that can send emails (SMTP), 
media        2 tools  Media processing agent with AudioTranscribeTool 
['calculator', 'python_repl'] 0.3 8 False`} />

      <p>
        The same list from the command line, with <code>--json</code> when a script wants it:
      </p>

      <CodeBlock language="bash" code={`effgen presets
effgen presets --json`} />

      <Terminal command="effgen presets" output={`Available Agent Presets
  math          2 tools · ~338 tok/call
               Mathematical reasoning agent with Calculator and PythonREPL.
  research      15 tools · ~3374 tok/call
               Research agent with WebSearch, URLFetch, Wikipedia, academic 
search (arXiv, PubMed, Semantic Scholar), RSS feeds, news, YouTube 
transcript/metadata, Reddit, Hacker News, and document parsing (PDF, DOCX, 
Excel) tools.
  coding        4 tools · ~1047 tok/call
               Coding agent with CodeExecutor, PythonREPL, FileOperations, and 
BashTool.
  general       31 tools · ~7944 tok/call
               General-purpose agent with a broad set of built-in tools, 
including QR, OCR, audio transcription, image analysis, document parsing (PDF, 
DOCX, Excel), weather/geo, email (SMTP/IMAP), Slack, and Discord webhooks. The 
unsandboxed shell (bash) is opt-in via the 'coding' preset, not bundled here.
  rag           1 tool · ~270 tok/call
               Retrieval-Augmented Generation agent with hybrid search over a 
knowledge base.
  minimal       no tools
               Minimal agent with no tools — direct model inference only.
  multimodal    7 tools · ~2201 tok/call
               Multimodal agent for image, audio, and video understanding. Uses 
Gemini Flash (primary) with OpenAI gpt-4o-mini and HF fallbacks. Equipped with 
ImageInfo, ImageCaption, OCR, AudioTranscribe, PDF, Weather, and 
MultimodalDescribe (auto-dispatch for any media type).
  notify        4 tools · ~1069 tok/call
               Notification agent that can send emails (SMTP), read email 
(IMAP), and post Slack or Discord messages. Configure credentials via env vars: 
SMTP_HOST/SMTP_USER/SMTP_PASSWORD, IMAP_HOST/IMAP_USER/IMAP_PASSWORD, 
SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL.
  media         2 tools · ~684 tok/call
               Media processing agent with AudioTranscribeTool (speech-to-text) 
and ImageCaptionTool (vision captioning via OpenAI/Gemini). Handles audio files 
(MP3, WAV, OGG, FLAC) and images (PNG, JPEG, WEBP).

'~N tok/call' is the approximate tool-schema size sent on every request — a 
tool-heavy preset costs more per call and can exceed a small-context or 
rate-limited model.

Usage: effgen run --preset <name> "your task"`} maxLines={26} />

      <h2>The rag preset needs a knowledge base</h2>
      <p>
        <code>rag</code> is the one preset that will not build without an argument. Omitting{' '}
        <code>knowledge_base</code> raises <code>ValueError</code>, as does passing sources that
        yield no documents — a typo'd path fails loudly rather than producing an agent with an
        empty index. If some sources ingest and others do not, a <code>RuntimeWarning</code> names
        each skipped source and why, so the agent is never queried against a partial corpus.
      </p>

      <CodeBlock
        code={`from effgen import create_agent

agent = create_agent("rag", "openai:gpt-5-nano", knowledge_base="./docs")
r = agent.run("What does the deployment guide say about rollbacks?")
print(r.text)
print(r.sources)`}
      />

      <p>
        <Link to="/rag">RAG</Link> covers indexing, retrieval and citations in full.
      </p>

      <h2>When a preset name is wrong</h2>

      <CodeBlock
        filename="unknown.py"
        code={`from effgen import create_agent
from effgen.presets import UnknownPresetError

try:
    create_agent("maths", "openai:gpt-5-nano")
except UnknownPresetError as e:
    print(e)`}
      />

      <Terminal command="python unknown.py" output={`Unknown preset 'maths'. Did you mean 'math'? Available presets: coding, general, math, media, minimal, multimodal, notify, rag, research.`} />

      <p>
        <code>UnknownPresetError</code> subclasses both <code>ValueError</code> and{' '}
        <code>KeyError</code>, so an existing <code>except</code> for either still catches it.
      </p>

      <SeeAlso paths={['/agents', '/tools', '/domains']} />
    </DocPage>
  );
}
