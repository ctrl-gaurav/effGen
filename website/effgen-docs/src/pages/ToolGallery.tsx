import { useMemo, useState } from 'react';
import { KeyRound, LayoutGrid, Search, ShieldAlert, Timer } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Callout, CodeBlock, DocPage, ParamTable, SeeAlso } from '../components/docs';
import type { Param } from '../components/docs';
import { siteData, toolCount, version } from '../siteData';
import gallery from '../data/toolGallery.json';
import './ToolGallery.css';

interface GalleryEntry {
  name: string;
  code?: string;
  output?: string;
  error?: string;
  note?: string;
  native?: string;
}

const RUNS = new Map<string, GalleryEntry>(
  (gallery.items as GalleryEntry[]).map((entry) => [entry.name, entry]),
);

const CATEGORY_LABEL: Record<string, string> = {
  information_retrieval: 'Information retrieval',
  data_processing: 'Data processing',
  external_api: 'External APIs',
  communication: 'Communication',
  code_execution: 'Code execution',
  system: 'System',
  computation: 'Computation',
  file_operations: 'File operations',
};

const NATIVE_LABEL: Record<string, string> = {
  openai: 'OpenAI-hosted',
  gemini: 'Gemini-hosted',
  anthropic: 'Anthropic-hosted',
};

function formatDefault(value: unknown): string | undefined {
  if (value === null || value === undefined) return undefined;
  if (typeof value === 'string') return `'${value}'`;
  return JSON.stringify(value);
}

export default function ToolGallery() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');

  const tools = siteData.tools.items;

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return tools.filter((tool) => {
      if (category !== 'all' && tool.category !== category) return false;
      if (!needle) return true;
      return (
        tool.name.includes(needle) ||
        tool.class_name.toLowerCase().includes(needle) ||
        tool.description.toLowerCase().includes(needle) ||
        tool.tags.some((tag) => tag.toLowerCase().includes(needle))
      );
    });
  }, [tools, query, category]);

  const categories = Object.keys(siteData.tools.category_counts);

  return (
    <DocPage
      subtitle="Every built-in tool, with a runnable snippet for each."
      icon={<LayoutGrid size={48} />}
      toc={false}
    >
      <p>
        All {toolCount} tools effGen {version} ships, with the class to import, the parameters the
        tool declares, and a snippet that was run against this release. Filter by category, or
        search by name, class or tag.
      </p>

      <Callout type="info" title="How to read a snippet">
        <p>
          <code>execute()</code> is a coroutine that takes keyword arguments and returns a{' '}
          <code>ToolResult</code> with <code>.success</code>, <code>.output</code> and{' '}
          <code>.error</code>. On failure <code>.output</code> is <code>None</code>, so every
          snippet whose call can fail for a reason outside itself checks <code>.success</code>{' '}
          first. <Link to="/tools">Tools</Link> explains the contract; any of these can be handed
          to an agent as <code>tools=[…]</code>.
        </p>
      </Callout>

      <div className="tool-gallery-controls">
        <label className="tool-gallery-search">
          <Search size={16} aria-hidden="true" />
          <span className="sr-only">Search the tool gallery</span>
          <input
            type="search"
            value={query}
            placeholder={`Search ${toolCount} tools by name, class or tag`}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <div className="tool-gallery-filters" role="group" aria-label="Filter by category">
          <button
            type="button"
            className={category === 'all' ? 'is-active' : undefined}
            aria-pressed={category === 'all'}
            onClick={() => setCategory('all')}
          >
            All <span>{toolCount}</span>
          </button>
          {categories.map((name) => (
            <button
              key={name}
              type="button"
              className={category === name ? 'is-active' : undefined}
              aria-pressed={category === name}
              onClick={() => setCategory(name)}
            >
              {CATEGORY_LABEL[name] ?? name} <span>{siteData.tools.category_counts[name]}</span>
            </button>
          ))}
        </div>
      </div>

      <p className="tool-gallery-count" aria-live="polite">
        {visible.length === toolCount
          ? `Showing all ${toolCount} tools.`
          : `Showing ${visible.length} of ${toolCount} tools.`}
      </p>

      {visible.length === 0 && (
        <p className="tool-gallery-empty">
          No tool matches that. Clear the search, or pick a different category.
        </p>
      )}

      {visible.map((tool) => {
        const run = RUNS.get(tool.name);
        const params: Param[] = tool.params.map((param) => ({
          name: param.name,
          type: param.type,
          required: param.required,
          default: formatDefault(param.default),
          description: param.enum
            ? `${param.description} One of: ${param.enum.join(', ')}.`
            : param.description,
        }));

        return (
          <section key={tool.name} className="tool-card" id={`tool-${tool.name}`}>
            <header className="tool-card-head">
              <h2>
                <code>{tool.name}</code>
              </h2>
              <div className="tool-card-badges">
                <span className="tool-badge">{CATEGORY_LABEL[tool.category] ?? tool.category}</span>
                {run?.native && (
                  <span className="tool-badge tool-badge-native">
                    {NATIVE_LABEL[run.native] ?? run.native}
                  </span>
                )}
                {tool.requires_api_key && (
                  <span className="tool-badge tool-badge-key">
                    <KeyRound size={12} aria-hidden="true" /> needs a key
                  </span>
                )}
                {tool.requires_approval && (
                  <span className="tool-badge tool-badge-approval">
                    <ShieldAlert size={12} aria-hidden="true" /> needs approval
                  </span>
                )}
                <span className="tool-badge tool-badge-quiet">
                  <Timer size={12} aria-hidden="true" /> {tool.timeout_seconds}s
                </span>
              </div>
            </header>

            <p className="tool-card-class">
              <code>
                from {tool.module} import {tool.class_name}
              </code>
            </p>

            <p className="tool-card-description">{tool.description}</p>

            {params.length > 0 && (
              <ParamTable
                nameLabel="Parameter"
                params={params}
                caption={
                  <>
                    {params.length} parameter{params.length === 1 ? '' : 's'}, from{' '}
                    <code>{tool.class_name}().metadata</code>.
                  </>
                }
              />
            )}

            {run?.native && (
              <p className="tool-card-note">
                Executed by the provider, not on your machine. It has to be paired with a{' '}
                {NATIVE_LABEL[run.native] ?? run.native} model —{' '}
                <Link to="/native-provider-tools">Provider-native tools</Link> has the setup, the
                cost and the error you get if you pair it with anything else.
              </p>
            )}

            {run?.code && <CodeBlock filename={`${tool.name}.py`} code={run.code} />}

            {run?.output && (
              <CodeBlock
                language="text"
                filename="output"
                code={run.output}
                caption={`Captured from that script on ${gallery.run_at}, against effGen 1.0.0.`}
              />
            )}

            {run?.note && <p className="tool-card-note">{run.note}</p>}

            {!run?.output && run?.error && (
              <CodeBlock
                language="text"
                filename="what it prints here"
                code={run.error}
                caption="The tool's own message on this host, not a stack trace."
              />
            )}
          </section>
        );
      })}

      <h2>Using several at once</h2>
      <p>
        Any tool above can be given to an agent. A <Link to="/presets">preset</Link> is a tool set
        someone has already chosen, with a system prompt to match and a stated token cost per call.
      </p>

      <CodeBlock
        filename="researcher.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin.arxiv import ArXivTool
from effgen.tools.builtin.hackernews import HackerNewsTool
from effgen.tools.builtin.translate import TranslateTool

agent = Agent(AgentConfig(
    name="researcher",
    model="gpt-5-nano",
    provider="openai",
    tools=[ArXivTool(), TranslateTool(), HackerNewsTool()],
    system_prompt="You are a research assistant.",
))

response = agent.run("Find the top Hacker News post right now and summarise it in one French sentence.")
print(response.text)`}
      />

      <CodeBlock
        language="text"
        filename="output"
        code={`Le post le plus populaire de Hacker News en ce moment raconte que l’auteur a dépensé 266 dollars et utilisé quatre modèles d’IA pour posséder sa tablette, et que GLM-5.3 l’a terminé en un jour. [1]`}
        caption="Three tools, one run. The citation marker is the agent's, not the page's."
      />

      <SeeAlso paths={['/tools', '/custom-tools', '/presets']} />
    </DocPage>
  );
}
