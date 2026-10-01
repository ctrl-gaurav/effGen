import { Wrench } from 'lucide-react';
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
import { siteData, toolCount } from '../siteData';

const categories = Object.entries(siteData.tools.category_counts).sort(
  (a, b) => b[1] - a[1],
);

const CATEGORY_BLURB: Record<string, string> = {
  information_retrieval: 'Search and fetch — the web, Wikipedia, arXiv, PubMed, GitHub, RSS, YouTube.',
  data_processing: 'Turn a file or a blob into text or rows — PDF, DOCX, Excel, OCR, audio, images.',
  external_api: 'A third-party service behind one call — weather, geocoding, prices, HTTP.',
  communication: 'Draft and send — email, Slack, Discord, desktop notifications.',
  code_execution: 'Run code, locally in a sandbox or on the provider’s machines.',
  system: 'Read the machine the agent is running on — shell, git, Docker, resource usage.',
  computation: 'Arithmetic, dates and statistics, without a model guessing at them.',
  file_operations: 'Read and write files inside a directory the tool is allowed to touch.',
};

export default function Tools() {
  return (
    <DocPage
      subtitle="What a tool is, how an agent is given one, and what calling one returns."
      icon={<Wrench size={48} />}
    >
      <p>
        A tool is an async callable with a described set of parameters. An agent is handed a list
        of them, the model asks for one by name, effGen validates the arguments, runs it, and
        hands the result back into the conversation. effGen ships {toolCount} of them across{' '}
        {categories.length} categories, and a function of your own becomes one with a decorator.
      </p>

      <h2>Calling a tool</h2>
      <p>
        Every tool is called the same way: <code>await tool.execute(**kwargs)</code>, which
        returns a <code>ToolResult</code>.
      </p>

      <CodeBlock
        filename="first_tool.py"
        code={`import asyncio

from effgen.tools.builtin.calculator import Calculator

result = asyncio.run(Calculator().execute(expression="2 ** 10 + sqrt(144)"))
print(result.success, result.output["result"])`}
      />

      <Terminal
        command="python first_tool.py"
        output={`True 1036.0`}
        caption={`Run against effGen 1.0.0.`}
      />

      <Callout type="warning" title="Keyword arguments, and a typed result">
        <p>
          <code>execute()</code> takes <strong>keyword arguments</strong>, not a dictionary, and{' '}
          <code>ToolResult</code> is a dataclass, not a mapping. There is no{' '}
          <code>.data</code> field and the result is not subscriptable, so{' '}
          <code>result["data"]["…"]</code> raises <code>TypeError</code>. Read{' '}
          <code>result.output</code>.
        </p>
      </Callout>

      <p>
        In a plain script, drive the coroutine with <code>asyncio.run(...)</code>; inside an{' '}
        <code>async</code> function, <code>await</code> it directly.
      </p>

      <h2>The ToolResult contract</h2>
      <p>
        Six fields, the same six for every tool in the framework and every tool you write.
      </p>

      <CodeBlock
        filename="result_shape.py"
        code={`import asyncio

from effgen.tools.builtin.calculator import Calculator

result = asyncio.run(Calculator().execute(expression="6 * 7"))

print("success        ", result.success)
print("output         ", result.output)
print("error          ", result.error)
print("execution_time ", round(result.execution_time, 6), "seconds")
print("metadata       ", result.metadata)
print("timestamp      ", result.timestamp)`}
      />

      <Terminal
        command="python result_shape.py"
        output={`success         True
output          {'result': 42, 'formatted': '42', 'expression': '6 * 7'}
error           None
execution_time  0.001874 seconds
metadata        {'tool_name': 'calculator', 'tool_version': '1.0.0'}
timestamp       2026-08-23T14:37:56.852805`}
      />

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'success',
            type: 'bool',
            description: 'Whether the call did what it was asked. Check this before reading output.',
          },
          {
            name: 'output',
            type: 'Any',
            description: (
              <>
                What the tool produced — usually a <code>dict</code>, sometimes a list.{' '}
                <code>None</code> when the call failed.
              </>
            ),
          },
          {
            name: 'error',
            type: 'str | None',
            default: 'None',
            description:
              'The reason the call failed, in a sentence meant to be shown to whoever is reading the run.',
          },
          {
            name: 'execution_time',
            type: 'float',
            default: '0.0',
            description: 'Wall-clock seconds the call took, filled in by the base class.',
          },
          {
            name: 'metadata',
            type: 'dict[str, Any]',
            default: '{}',
            description:
              'Whatever the tool wants to record about the call — the operation it ran, an upstream request id, a cache hit.',
          },
          {
            name: 'timestamp',
            type: 'str',
            description: 'ISO 8601, set when the result was constructed.',
          },
        ]}
        caption={
          <>
            <code>effgen.tools.base_tool.ToolResult</code>. <code>to_dict()</code> and{' '}
            <code>to_json()</code> serialise the whole record.
          </>
        }
      />

      <h2>When a call fails</h2>
      <p>
        A tool that cannot do its job returns a result with <code>success=False</code> rather than
        raising. That keeps one failing tool from ending an agent run, and it keeps the reason
        readable.
      </p>

      <CodeBlock
        filename="failure.py"
        code={`import asyncio

from effgen.tools.builtin.calculator import Calculator

result = asyncio.run(Calculator().execute(expression="1/0"))

print(result.success)
print(result.error)
print(result.output)`}
      />

      <Terminal
        command="python failure.py"
        output={`False
Tool execution failed: Calculation failed: division by zero. Check the expression for balanced brackets, a supported function and no stray characters.
None`}
      />

      <p>
        So any call that can fail for a reason outside your snippet — it reaches a network
        service, it names a file you have to supply, it needs a credential — gets two lines in
        front of it:
      </p>

      <CodeBlock
        code={`if not result.success:
    raise SystemExit(result.error)`}
      />

      <p>
        Without them, <code>result.output["…"]</code> raises <code>TypeError</code> on{' '}
        <code>None</code> and the actual reason — a 5xx, a rate limit, a missing key, a path that
        does not exist — never reaches you. Every sample in the{' '}
        <Link to="/tools/gallery">tool gallery</Link> whose call touches the outside world carries
        that check.
      </p>

      <h3>Arguments are validated before the tool runs</h3>
      <p>
        A missing required parameter, a value outside a declared range, or a string that is not in
        a declared <code>enum</code> is caught by the base class. The tool's own code never sees
        it, and the message names what to pass.
      </p>

      <CodeBlock
        filename="validation.py"
        code={`import asyncio

from effgen.tools.builtin.arxiv import ArXivTool

result = asyncio.run(ArXivTool().execute(operation="fetch"))
print(result.success)
print(result.error)`}
      />

      <Terminal
        command="python validation.py"
        output={`False
Tool execution failed: operation='fetch' requires 'arxiv_id'`}
      />

      <h2>What a tool declares about itself</h2>
      <p>
        Every tool carries a <code>ToolMetadata</code> record. It is what the model is shown, what
        the argument validator checks against, and what <code>effgen tools info</code> prints.
      </p>

      <CodeBlock
        filename="metadata.py"
        code={`from effgen.tools.builtin.weather import WeatherTool

meta = WeatherTool().metadata
print(meta.name, "|", meta.category.value, "| timeout", meta.timeout_seconds, "s")
for spec in meta.parameters:
    flag = "required" if spec.required else f"default={spec.default!r}"
    print(f"  {spec.name:12} {spec.type.value:8} {flag}")`}
      />

      <Terminal
        command="python metadata.py"
        output={`weather | external_api | timeout 20 s
  operation    string   default='current'
  lat          float    default=None
  lon          float    default=None
  location     string   default=None
  days         integer  default=7
  start_date   string   default=None
  end_date     string   default=None
  units        string   default='metric'`}
      />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'name', type: 'str', description: 'The registry name, and the name the model calls.' },
          {
            name: 'description',
            type: 'str',
            description: 'What the tool is for. This text goes into the prompt, so it is written for the model.',
          },
          {
            name: 'category',
            type: 'ToolCategory',
            description: `One of the ${categories.length} categories below.`,
          },
          {
            name: 'parameters',
            type: 'list[ParameterSpec]',
            default: '[]',
            description: 'Name, type, description, required, default, enum, min_value, max_value.',
          },
          { name: 'returns', type: 'dict[str, Any]', default: '{}', description: 'A JSON schema for what output holds.' },
          { name: 'version', type: 'str', default: "'1.0.0'", description: 'The tool’s own version.' },
          { name: 'author', type: 'str | None', default: 'None', description: 'Who wrote it. Set by plugins more than by built-ins.' },
          {
            name: 'requires_auth',
            type: 'bool',
            default: 'False',
            description: 'Whether the tool needs a signed-in session of some kind.',
          },
          {
            name: 'requires_api_key',
            type: 'bool',
            default: 'False',
            description: 'Whether an API key has to be in the environment for the call to work.',
          },
          {
            name: 'requires_approval',
            type: 'bool',
            default: 'False',
            description: (
              <>
                Whether a call has to be confirmed before it runs — see{' '}
                <Link to="/human-loop">Human in the loop</Link>.
              </>
            ),
          },
          { name: 'cost_estimate', type: 'str', default: "'low'", description: 'A rough cost band: low, medium or high.' },
          { name: 'timeout_seconds', type: 'int', default: '30', description: 'How long one call is given before it is abandoned.' },
          { name: 'tags', type: 'list[str]', default: '[]', description: 'Free-form labels, used for search and filtering.' },
          { name: 'examples', type: 'list[dict]', default: '[]', description: 'Worked argument sets the tool ships with.' },
        ]}
        caption={<><code>effgen.tools.base_tool.ToolMetadata</code></>}
      />

      <h2>Giving an agent tools</h2>
      <p>
        Pass instances. The agent turns the metadata into the tool schema its provider expects,
        dispatches calls, feeds results back, and records every call on the response.
      </p>

      <CodeBlock
        filename="in_agent.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.datetime_tool import DateTimeTool

agent = Agent(AgentConfig(
    name="helper",
    model="gpt-5-nano",
    provider="openai",
    tools=[Calculator(), DateTimeTool()],
))

response = agent.run("What is 4817 * 236? Use the calculator.")
print(response.text)
print("tools called:", [call.name for call in response.tool_calls])`}
      />

      <Terminal
        command="python in_agent.py"
        output={`1136812
tools called: ['calculator']`}
      />

      <p>
        <code>response.tool_calls</code> holds one record per call — <code>name</code>,{' '}
        <code>arguments</code>, <code>result</code>, <code>duration</code>, <code>error</code> and
        the <code>iteration</code> it happened on. <Link to="/agents">Agents</Link> covers the
        rest of the response.
      </p>

      <Callout type="tip" title="A preset is a tool list that already fits">
        <p>
          The {siteData.presets.count} presets are named bundles — research, coding, media, notify
          and the rest — each with a tool set and a system prompt chosen together, and a stated
          token cost per call. <Link to="/presets">Presets</Link> lists them.
        </p>
      </Callout>

      <h2>The registry</h2>
      <p>
        Tools are also addressable by name. The registry is what <code>effgen run -t</code>, the
        MCP server and the plugin loader all read.
      </p>

      <CodeBlock
        filename="registry.py"
        code={`from effgen.tools import get_registry
from effgen.tools.base_tool import ToolCategory

registry = get_registry()
registry.discover_builtin_tools()

print(len(registry.list_tools()), "tools registered")
print(sorted(registry.get_tools_by_category(ToolCategory.COMPUTATION)))

weather = registry.get_tool_sync("weather")
print(type(weather).__name__, "from", type(weather).__module__)`}
      />

      <Terminal
        command="python registry.py"
        output={`66 tools registered
['calculator', 'datetime', 'stats']
WeatherTool from effgen.tools.builtin.weather`}
      />

      <ApiTable
        headers={['Method', 'What it does']}
        rows={[
          [<code>list_tools()</code>, 'Every registered name.'],
          [<code>get_tool_sync(name)</code>, 'The instance, initialising it if it has not been used yet.'],
          [<code>get_tool(name)</code>, 'The same, awaitable.'],
          [<code>get_metadata(name)</code>, 'Its ToolMetadata without constructing it twice.'],
          [<code>get_all_metadata()</code>, 'A mapping of every name to its metadata.'],
          [<code>get_tools_by_category(category)</code>, 'The names filed under one category.'],
          [
            <code>register_tool(tool)</code>,
            <>
              Add one — an instance from <code>@tool</code>, or a <code>BaseTool</code> subclass.
            </>,
          ],
          [<code>unregister_tool(name)</code>, 'Remove one.'],
          [<code>export_schemas()</code>, 'Every tool as a provider-shaped function schema.'],
          [<code>register_plugin(plugin)</code>, 'Add every tool a ToolPlugin carries.'],
        ]}
        caption={<><code>effgen.tools.registry.ToolRegistry</code>, reached with <code>get_registry()</code>.</>}
      />

      <h2>From the command line</h2>

      <Terminal
        command="effgen tools list --category computation"
        output={`Available Tools
                              Registered Tools (3)                              
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name       ┃ Category    ┃ Description                                       ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ calculator │ computation │ Perform mathematical calculations, evaluate       │
│            │             │ expres...                                         │
│ datetime   │ computation │ Get current date/time, convert timezones, perform │
│            │             │ ...                                               │
│ stats      │ computation │ Compute statistics from a list of numbers: mean,  │
│            │             │ m...                                              │
└────────────┴─────────────┴───────────────────────────────────────────────────┘`}
      />

      <Terminal
        command="effgen tools info calculator"
        output={`Tool: calculator

Description: Perform mathematical calculations, evaluate expressions, and 
convert units
Category: computation
Version: 1.0.0
Tags: math, calculator, computation, conversion
Operation aliases: arithmetic -> calculate, compute -> calculate, conversion -> 
convert_units, convert -> convert_units, eval -> calculate, evaluate -> 
calculate, math -> calculate, statistic -> statistics, stats -> statistics, 
units -> convert_units

Parameters:
{                                                                               
  "name": "calculator",                                                         
  "description": "Perform mathematical calculations, evaluate expressions, and c
  "parameters": {                                                               
    "type": "object",                                                           
    "properties": {                                                             
      "expression": {                                                           
        "description": "Mathematical expression to evaluate",                   
        "type": "string",                                                       
        "minLength": 1                                                          
      },                                                                        
      "operation": {                                                            
        "description": "Type of operation",                                     
        "type": "string",                                                       
        "enum": [                                                               
          "calculate",                                                          
          "convert_units",                                                      
          "statistics"                                                          
        ],                                                                      
        "default": "calculate"                                                  
      },                                                                        
      "from_unit": {                                                            
        "description": "Source unit for conversion",                            
        "type": "string"                                                        
      },                                                                        
      "to_unit": {                                                              
        "description": "Target unit for conversion",                            
        "type": "string"                                                        
      },                                                                        
      "precision": {                                                            
        "description": "Number of decimal places for result",                   
        "type": "integer",                                                      
        "minimum": 0,                                                           
        "maximum": 15                                                           
      }                                                                         
    },                                                                          
    "required": [                                                               
      "expression"                                                              
    ]                                                                           
  }                                                                             
}                                                                               

Example:
  effgen tools test calculator -i '{"expression": "2 + 2 * 3"}'`}
      />

      <p>
        <code>effgen tools list --json</code> prints the same inventory as machine-readable JSON,
        and <code>effgen tools test &lt;name&gt;</code> runs a tool's own worked example.
      </p>

      <h2>The categories</h2>

      <ApiTable
        headers={['Category', 'Tools', 'What is in it']}
        rows={categories.map(([name, count]) => [
          <code>{name}</code>,
          String(count),
          CATEGORY_BLURB[name] ?? '',
        ])}
        caption={
          <>
            Derived from the installed package — {toolCount} tools in {categories.length}{' '}
            categories. <Link to="/tools/gallery">The gallery</Link> has every one, with a snippet.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <><code>TypeError: 'ToolResult' object is not subscriptable</code></>,
            'The result was read as a dictionary.',
            <>
              Read <code>result.output</code>. <code>ToolResult</code> is a dataclass and has no{' '}
              <code>.data</code> field.
            </>,
          ],
          [
            <><code>TypeError: execute() takes 1 positional argument but 2 were given</code></>,
            'A dictionary was passed positionally.',
            <>
              Spread it: <code>execute(operation="search", query="…")</code>, or{' '}
              <code>execute(**arguments)</code>.
            </>,
          ],
          [
            <><code>result.success is False</code>, <code>Parameter validation failed: …</code></>,
            'An argument was missing, out of range, or not one of the declared choices.',
            'The message names the parameters the tool takes. It never reached the tool’s own code.',
          ],
          [
            <><code>result.success is False</code>, <code>… requires credentials that are not configured</code></>,
            'The tool needs an API key or a service address that is not in the environment.',
            'The error names the variables to set. Nothing is sent and no partial call is made.',
          ],
          [
            <><code>TypeError: 'NoneType' object is not subscriptable</code></>,
            <>
              <code>result.output</code> was read after a failed call.
            </>,
            <>
              Check <code>result.success</code> first — that is what the two-line guard above is
              for.
            </>,
          ],
          [
            <code>ToolIncompatibleError</code>,
            'A provider-native tool was given to a model whose provider does not have it.',
            <>
              Raised at agent construction, before any API call. See{' '}
              <Link to="/native-provider-tools">Provider-native tools</Link>.
            </>,
          ],
          [
            'A call that never returns',
            <>
              The tool passed its <code>timeout_seconds</code>.
            </>,
            'The result reports the timeout. Raise the tool’s timeout, or pass a per-call one where the tool accepts it.',
          ],
        ]}
      />

      <Callout type="note" title="What changed in 1.0.0">
        <p>
          An unreachable backend now raises <code>BackendUnreachableError</code> from an agent run
          regardless of <code>raise_on_error</code>, and <code>raise_on_error</code> itself
          defaults to <code>True</code>. A tool's own failure is still reported on the{' '}
          <code>ToolResult</code> rather than raised. <Link to="/migration">Migrating to 1.2.0</Link>{' '}
          has the three breaking changes.
        </p>
      </Callout>

      <SeeAlso paths={['/tools/gallery', '/custom-tools', '/tool-calling']} />
    </DocPage>
  );
}
