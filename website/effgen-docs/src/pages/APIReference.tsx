import { useMemo, useState } from 'react';
import { Braces, Search } from 'lucide-react';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import type { Param } from '../components/docs';
import { anchorFor, apiNames, apiReference } from '../apiReference';
import type { ApiName } from '../apiReference';
import { version } from '../siteData';
import './APIReference.css';

const KIND_LABEL: Record<string, string> = {
  class: 'class',
  dataclass: 'dataclass',
  enum: 'enum',
  exception: 'exception',
  function: 'function',
  alias: 'type alias',
  module: 'module',
  value: 'value',
};

/** The kinds the filter row offers, in the order it offers them. */
const KIND_FILTERS = ['class', 'dataclass', 'enum', 'function', 'exception'] as const;

function matches(entry: ApiName, needle: string): boolean {
  if (!needle) return true;
  return (
    entry.name.toLowerCase().includes(needle) ||
    entry.module.toLowerCase().includes(needle) ||
    entry.summary.toLowerCase().includes(needle) ||
    entry.members.some((member) => member.name.toLowerCase().includes(needle))
  );
}

function paramsOf(entry: ApiName): Param[] {
  return entry.params.map((param) => ({
    name: param.name,
    type: param.type ?? undefined,
    default: param.default ?? undefined,
    required: param.required,
    description: param.description || '—',
  }));
}

function Entry({ entry }: { entry: ApiName }) {
  const params = paramsOf(entry);

  return (
    <section className="api-entry" id={anchorFor(entry.name)}>
      <div className="api-entry-head">
        <h3>
          <code>{entry.name}</code>
        </h3>
        <span className={`api-kind api-kind-${entry.kind}`}>{KIND_LABEL[entry.kind]}</span>
        <code className="api-module">{entry.module}</code>
      </div>

      {entry.signature && (
        <p
          className="api-signature"
          tabIndex={0}
          role="group"
          aria-label={`Signature of ${entry.name}`}
        >
          <code>
            {entry.name}
            {entry.signature}
          </code>
        </p>
      )}

      {entry.summary && <p className="api-summary">{entry.summary}</p>}

      {entry.bases.length > 0 && (
        <p className="api-meta">
          Inherits{' '}
          {entry.bases.map((base, index) => (
            <span key={base}>
              {index > 0 && ', '}
              <code>{base}</code>
            </span>
          ))}
        </p>
      )}

      {entry.values.length > 0 && (
        <p className="api-meta">
          {entry.kind === 'alias' ? 'One of' : 'Members'}:{' '}
          {entry.values.map((value, index) => (
            <span key={value.name}>
              {index > 0 && ', '}
              <code>{value.name}</code>
              {value.value && <span className="api-enum-value"> = {value.value}</span>}
            </span>
          ))}
        </p>
      )}

      {params.length > 0 && (
        <ParamTable
          nameLabel="Parameter"
          params={params}
          caption={
            <>
              {params.length} parameter{params.length === 1 ? '' : 's'}, from{' '}
              <code>inspect.signature({entry.name})</code>.
            </>
          }
        />
      )}

      {entry.returns && (
        <p className="api-meta">
          Returns <code>{entry.returns.type ?? 'None'}</code>
          {entry.returns.description ? ` — ${entry.returns.description}` : ''}
        </p>
      )}

      {entry.raises.length > 0 && (
        <p className="api-meta">
          Raises{' '}
          {entry.raises.map((raise, index) => (
            <span key={raise.name}>
              {index > 0 && ', '}
              <code>{raise.name}</code>
            </span>
          ))}
        </p>
      )}

      {entry.members.length > 0 && (
        <details className="api-members">
          <summary>
            {entry.members.length} public member{entry.members.length === 1 ? '' : 's'}
          </summary>
          <ul>
            {entry.members.map((member) => (
              <li key={member.name}>
                <code>
                  {member.is_async ? 'await ' : ''}
                  {member.name}
                  {member.signature}
                </code>
                {member.kind !== 'method' && (
                  <span className="api-member-kind"> {member.kind}</span>
                )}
                {member.summary && <span className="api-member-what"> — {member.summary}</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

export default function APIReference() {
  const [query, setQuery] = useState('');
  const [kind, setKind] = useState('all');

  const needle = query.trim().toLowerCase();

  const visible = useMemo(
    () =>
      apiNames.filter(
        (entry) => (kind === 'all' || entry.kind === kind) && matches(entry, needle),
      ),
    [needle, kind],
  );

  const byArea = useMemo(() => {
    const map = new Map<string, ApiName[]>();
    for (const entry of visible) {
      const list = map.get(entry.area);
      if (list) list.push(entry);
      else map.set(entry.area, [entry]);
    }
    return map;
  }, [visible]);

  const total = apiReference.public_names;

  return (
    <DocPage
      subtitle={`Every one of the ${total} names the effgen package exports, with its signature, its arguments and the module it comes from.`}
      icon={<Braces size={48} />}
      toc={false}
    >
      <p>
        <code>import effgen</code> puts {total} names in one namespace. Everything on this page is
        importable as <code>from effgen import …</code> — there is no second path to learn, and
        nothing below sits behind a submodule import. The <em>module</em> beside each name is where
        the object is defined, which is what a traceback will show you; it is not where you import
        it from.
      </p>

      <CodeBlock
        filename="surface.py"
        code={`import effgen

print("effgen", effgen.__version__)
print("public names:", len(effgen.__all__))
print("first ten   :", sorted(effgen.__all__)[:10])`}
      />

      <Terminal
        command="python surface.py"
        output={`effgen 1.1.0
public names: 250
first ten   : ['ActionStep', 'Agent', 'AgentConfig', 'AgentEvaluator', 'AgentMiddleware', 'AgentState', 'AgentSystemPromptBuilder', 'AgentThread', 'Alert', 'AlertSeverity']`}
        caption={`Run against effGen 1.1.0, which exported 250 names; 1.2.0 adds RunLedger, and 1.3.0 adds ToolCallingProbe and probe_tool_calling. The reference below is generated from the ${version} list itself.`}
      />

      <h2>The four lines most programs start with</h2>

      <CodeBlock
        filename="ask.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai", tools=[Calculator()]))
response = agent.run("Use the calculator to work out 4280 * 0.17 + 1993 ** 2, and give me the number.")

print(response.output)
print("success:", response.success, "| calls:", response.tool_calls.total,
      "| tools:", response.tool_calls.names)`}
      />

      <Terminal
        command="python ask.py"
        output={`3972776.6
success: True | calls: 1 | tools: ['calculator']`}
      />

      <p>
        <code>Agent</code>, <code>AgentConfig</code> and every built-in tool class are on this page.{' '}
        <code>AgentResponse</code> — the thing <code>run()</code> hands back — is not, because you
        never construct one; it and the other names in that position are in{' '}
        <a href="#not-exported">the second table</a>.
      </p>

      <h2>Reading a signature yourself</h2>

      <p>
        This page is a copy of what the package already knows about itself. When you want the
        authority rather than the copy, ask the package:
      </p>

      <CodeBlock
        filename="signature.py"
        code={`import inspect

from effgen import Agent, AgentConfig, load_model

print("load_model", inspect.signature(load_model))
print()
print("Agent.run", inspect.signature(Agent.run))
print()
print("AgentConfig fields:", len(inspect.signature(AgentConfig).parameters))
print(inspect.getdoc(load_model).splitlines()[0])`}
      />

      <Terminal
        command="python signature.py"
        output={`load_model (model_name: 'str', engine: 'str | None' = None, engine_config: 'dict[str, Any] | None' = None, tensor_parallel_size: 'int | None' = None, gpu_memory_utilization: 'float | None' = None, apply_chat_template: 'bool' = True, provider: 'str | None' = None, base_url: 'str | None' = None, api_key: 'str | None' = None, **kwargs: 'Any') -> 'BaseModel'

Agent.run (self, task: "'str | Message | list[Any]'", mode: 'AgentMode | None' = None, context: 'dict[str, Any] | None' = None, output_schema: 'dict[str, Any] | None' = None, output_model: 'Any' = None, inputs: 'list[Any] | None' = None, **kwargs: 'Any') -> 'AgentResponse'

AgentConfig fields: 49
Convenience function to quickly load a model.`}
        maxLines={14}
        caption="The quotes around each annotation are how Python stores a postponed annotation; the signatures below have them removed."
      />

      <h2>What a tool returns</h2>

      <p>
        Worth stating once, because it is the shape most often got wrong:{' '}
        <code>execute()</code> is a coroutine, it takes keyword arguments, and it returns a{' '}
        <code>ToolResult</code>. A <code>ToolResult</code> has six fields, none of them called{' '}
        <code>data</code>, and it is not subscriptable.
      </p>

      <CodeBlock
        filename="tool_result.py"
        code={`import asyncio

from effgen.tools.builtin import Calculator


async def main() -> None:
    result = await Calculator().execute(expression="2 ** 10")
    print("type   :", type(result).__name__)
    print("fields :", sorted(vars(result)))
    print("success:", result.success)
    print("output :", result.output)
    try:
        result["output"]
    except TypeError as error:
        print("subscripting it:", type(error).__name__, "—", error)


asyncio.run(main())`}
      />

      <Terminal
        command="python tool_result.py"
        output={`type   : ToolResult
fields : ['error', 'execution_time', 'metadata', 'output', 'success', 'timestamp']
success: True
output : {'result': 1024, 'formatted': '1024', 'expression': '2 ** 10'}
subscripting it: TypeError — 'ToolResult' object is not subscriptable`}
      />

      <h2>Every name</h2>

      <p>
        {total} names in {apiReference.areas.length} areas. Search by name, by module or by what a
        thing does; every entry has an address of its own, so a link can point at one name.
      </p>

      <div className="api-controls">
        <label className="api-search">
          <Search size={16} aria-hidden="true" />
          <span className="sr-only">Search the API reference</span>
          <input
            type="search"
            value={query}
            placeholder={`Search ${total} names by name, module or description`}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <div className="api-filters" role="group" aria-label="Filter by kind">
          <button
            type="button"
            className={kind === 'all' ? 'is-active' : undefined}
            aria-pressed={kind === 'all'}
            onClick={() => setKind('all')}
          >
            All <span>{total}</span>
          </button>
          {KIND_FILTERS.map((name) => (
            <button
              key={name}
              type="button"
              className={kind === name ? 'is-active' : undefined}
              aria-pressed={kind === name}
              onClick={() => setKind(name)}
            >
              {KIND_LABEL[name]} <span>{apiReference.kind_counts[name] ?? 0}</span>
            </button>
          ))}
        </div>
      </div>

      <p className="api-count" aria-live="polite">
        {visible.length === total
          ? `Showing all ${total} names.`
          : `Showing ${visible.length} of ${total} names.`}
      </p>

      {visible.length === 0 && (
        <p className="api-empty">
          Nothing matches that. Clear the search, or pick a different kind.
        </p>
      )}

      {apiReference.areas.map((area) => {
        const entries = byArea.get(area.id);
        if (!entries || entries.length === 0) return null;
        return (
          <div key={area.id} className="api-area">
            <h2 id={`area-${area.id}`}>{area.title}</h2>
            <p className="api-area-blurb">{area.blurb}</p>
            {entries.map((entry) => (
              <Entry key={entry.name} entry={entry} />
            ))}
          </div>
        );
      })}

      <h2 id="not-exported">Names that are not on the top level</h2>

      <p>
        These come up constantly and none of them is in <code>effgen.__all__</code>, so the import
        is the longer one. Each path below is checked every time this page is generated.
      </p>

      <ApiTable
        headers={['Name', 'Import', 'What it is']}
        rows={apiReference.not_exported.map((row) => [
          <code>{row.name}</code>,
          <code>
            from {row.module} import {row.name}
          </code>,
          row.what,
        ])}
      />

      <Callout type="note" title="Where this page comes from">
        <p>
          Every row above is read out of the installed package —{' '}
          <code>effgen.__all__</code>, <code>inspect.signature</code> and each object's own
          docstring — by a script in this site's repository, and checked in as data. A release that
          adds, moves or renames a name changes this page by being regenerated; nothing here is
          maintained by hand, and a name cannot quietly fall off it. Generated against effGen{' '}
          {apiReference.version} on {apiReference.derived_at}.
        </p>
      </Callout>

      <SeeAlso paths={['/agents', '/tools/gallery', '/errors']} />
    </DocPage>
  );
}
