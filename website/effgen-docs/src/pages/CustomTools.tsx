import { Puzzle } from 'lucide-react';
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

export default function CustomTools() {
  return (
    <DocPage
      subtitle="Turning a function into a tool, and packaging tools as an installable plugin."
      icon={<Puzzle size={48} />}
    >
      <p>
        A typed function with a docstring is already everything a tool needs, so most tools are one
        decorator. The full <code>BaseTool</code> interface is there when you want range checks,
        an <code>enum</code>, initialisation that has to happen once, or a lifecycle hook.
      </p>

      <h2>The short way</h2>

      <CodeBlock filename="counter.py" code={`from effgen import Agent, AgentConfig, tool


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count words in.
    """
    return len(text.split())


agent = Agent(AgentConfig(
    name="counter",
    model="gpt-5-nano",
    provider="openai",
    tools=[word_count],
))

print(agent.run("How many words are in 'the quick brown fox'? Use the tool.").text)`} />

      <Terminal
        command="python counter.py"
        output={`4`}
        caption={`Run against effGen 1.0.0.`}
      />

      <p>
        <code>@tool</code> reads the function's name, its type hints and its docstring and builds
        the metadata from them. Nothing is repeated.
      </p>

      <CodeBlock filename="derived_schema.py" code={`from effgen import tool


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count words in.
    """
    return len(text.split())


meta = word_count.metadata
print(meta.name, "|", meta.category.value)
print(meta.description)
for spec in meta.parameters:
    print(f"  {spec.name}: {spec.type.value} required={spec.required} — {spec.description}")`} />

      <Terminal command="python derived_schema.py" output={`word_count | system
Count the words in a piece of text.
  text: string required=True — The text to count words in.`} />

      <ApiTable
        headers={['What the schema needs', 'Where it comes from']}
        rows={[
          [<code>name</code>, 'The function name, unless you pass one.'],
          [<code>description</code>, "The docstring's summary line."],
          [<code>parameters</code>, 'The signature — one ParameterSpec per argument.'],
          ['each parameter type', 'Its annotation, mapped onto the JSON type.'],
          ['each parameter description', <>Its <code>Args:</code> entry in the docstring.</>],
          [
            'required vs optional',
            'Whether the argument has a default. A default becomes the parameter’s default.',
          ],
          [<code>category</code>, <>Defaults to <code>system</code>; pass one to change it.</>],
        ]}
        caption={
          <>
            So the docstring is not decoration — the <code>Args:</code> lines are what the model
            reads when it decides whether to call the tool.
          </>
        }
      />

      <h3>Overriding what it derived</h3>

      <CodeBlock filename="multiply.py" code={`import asyncio

from effgen import tool


@tool(name="multiply", category="computation")
def mul(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


result = asyncio.run(mul.execute(a=6, b=7))
print(result.success, result.output)`} />

      <Terminal command="python multiply.py" output={`True 42`} />

      <p>
        The decorated object is a real tool instance, so it can be called directly as well as
        handed to an agent.
      </p>

      <ParamTable
        nameLabel="Argument"
        params={[
          { name: 'name', type: 'str', default: 'the function name', description: 'The name the model calls and the registry files it under.' },
          {
            name: 'description',
            type: 'str',
            default: 'the docstring summary',
            description: 'What the tool is for, as the model sees it.',
          },
          {
            name: 'category',
            type: 'str | ToolCategory',
            default: "'system'",
            description: 'One of the eight tool categories.',
          },
          { name: 'tags', type: 'list[str]', default: '[]', description: 'Labels for search and filtering.' },
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
          { name: 'timeout_seconds', type: 'int', default: '30', description: 'How long one call is given.' },
        ]}
        caption={
          <>
            <code>@tool</code> with no arguments uses every default; <code>@tool(...)</code> takes
            these.
          </>
        }
      />

      <h3>Without the decorator</h3>

      <CodeBlock filename="from_function.py" code={`import asyncio

from effgen import Tool


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


add_tool = Tool.from_function(add)
print(add_tool.name, "->", asyncio.run(add_tool.execute(a=2, b=3)).output)`} />

      <Terminal command="python from_function.py" output={`add -> 5`} />

      <h3>Async functions</h3>
      <p>
        An <code>async def</code> is wrapped without a thread. A synchronous function is run in a
        worker thread, so a slow one does not block the event loop either way.
      </p>

      <CodeBlock filename="fetch_length.py" code={`import asyncio

from effgen import tool


@tool
async def fetch_length(url: str) -> int:
    """Report how many bytes a URL returns.

    Args:
        url: The address to fetch.
    """
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return len(await response.read())


print(asyncio.run(fetch_length.execute(url="https://example.com")).output)`} />

      <Terminal command="python fetch_length.py" output={`559`} />

      <h2>The full interface</h2>
      <p>
        Subclass <code>BaseTool</code>, hand <code>ToolMetadata</code> to{' '}
        <code>super().__init__()</code>, and implement <code>_execute</code>. Note the underscore:
        the public <code>execute()</code> is the base class's, and it is what validates arguments,
        times the call and wraps the return value in a <code>ToolResult</code>.
      </p>

      <CodeBlock filename="uppercase.py" code={`import asyncio

from effgen.tools.base_tool import (
    BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata,
)


class UpperCaseTool(BaseTool):
    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="uppercase",
                description="Convert text to uppercase.",
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="text",
                        type=ParameterType.STRING,
                        description="Text to convert.",
                        required=True,
                    ),
                ],
                returns={"type": "object", "properties": {"result": {"type": "string"}}},
            )
        )

    async def _execute(self, **kwargs):
        return {"result": kwargs["text"].upper()}


result = asyncio.run(UpperCaseTool().execute(text="hello world"))
print(result.success, result.output)`} />

      <Terminal command="python uppercase.py" output={`True {'result': 'HELLO WORLD'}`} />

      <Callout type="warning" title="The constructor takes no arguments">
        <p>
          <code>register_tool</code>, the plugin loader and <code>effgen run -t</code> all
          construct a tool by calling the class with nothing, so the metadata has to be built
          inside <code>__init__</code>. Overriding the <code>metadata</code> property instead
          leaves those three paths without one.
        </p>
      </Callout>

      <h3>Parameter validation</h3>
      <p>
        <code>ParameterSpec</code> is where a range, a set of choices and a default live. The base
        class enforces them before <code>_execute</code> is called, so the tool's own code never
        sees a bad argument.
      </p>

      <CodeBlock filename="report.py" code={`import asyncio

from effgen.tools.base_tool import (
    BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata,
)


class ReportTool(BaseTool):
    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="report",
                description="Render rows in a chosen format.",
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="count", type=ParameterType.INTEGER,
                        description="Number of rows.", required=True,
                        min_value=1, max_value=100,
                    ),
                    ParameterSpec(
                        name="format", type=ParameterType.STRING,
                        description="Output format.",
                        enum=["json", "csv", "text"], default="json",
                    ),
                ],
            )
        )

    async def _execute(self, **kwargs):
        return {"rows": kwargs["count"], "format": kwargs["format"]}


print(asyncio.run(ReportTool().execute(count=5)).output)
print(asyncio.run(ReportTool().execute(count=500)).error)
print(asyncio.run(ReportTool().execute(count=5, format="xml")).error)`} />

      <Terminal command="python report.py" output={`None
Parameter validation failed: Parameter 'count' must be <= 100. Pass the parameters 'report' requires: count (integer).
Parameter validation failed: Invalid format 'xml'. Allowed: json, csv, text. Pass the parameters 'report' requires: count (integer).`} />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'name', type: 'str', required: true, description: 'The argument name, as the model will spell it.' },
          {
            name: 'type',
            type: 'ParameterType',
            required: true,
            description: 'STRING, INTEGER, NUMBER, BOOLEAN, ARRAY or OBJECT.',
          },
          {
            name: 'description',
            type: 'str',
            required: true,
            description: 'What to pass. This reaches the model, so write it for the model.',
          },
          { name: 'required', type: 'bool', default: 'False', description: 'Whether the call fails without it.' },
          { name: 'default', type: 'Any', default: 'None', description: 'What is used when it is not passed.' },
          { name: 'enum', type: 'list | None', default: 'None', description: 'The only accepted values.' },
          { name: 'min_value', type: 'float | None', default: 'None', description: 'Lower bound, for a number or integer.' },
          { name: 'max_value', type: 'float | None', default: 'None', description: 'Upper bound, for a number or integer.' },
        ]}
        caption={<><code>effgen.tools.base_tool.ParameterSpec</code></>}
      />

      <h2>Making a tool addressable by name</h2>
      <p>
        Passing an instance to <code>AgentConfig(tools=[...])</code> is enough to use it. Register
        it as well and it turns up in <code>effgen tools list</code>, in{' '}
        <code>effgen run -t &lt;name&gt;</code>, and in an{' '}
        <Link to="/protocols">MCP server</Link>.
      </p>

      <CodeBlock filename="register.py" code={`from effgen import tool
from effgen.tools import get_registry


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count words in.
    """
    return len(text.split())


get_registry().register_tool(word_count)
print("word_count" in get_registry().list_tools())`} />

      <Terminal command="python register.py" output={`True`} />

      <p>
        <code>register_tool</code> takes either form — an instance from <code>@tool</code> or a{' '}
        <code>BaseTool</code> subclass — and a plugin's <code>tools</code> list may mix the two.
      </p>

      <h2>Packaging tools as a plugin</h2>
      <p>
        A plugin is an ordinary Python package that effGen discovers through an entry point. The
        scaffold command writes one.
      </p>

      <Terminal command="effgen create-plugin my_tools" output={`Created plugin scaffold at effgen-plugin-my_tools/
  effgen-plugin-my_tools/my_tools/tools.py       — add your custom tools here
  effgen-plugin-my_tools/my_tools/plugin.py     — register tools in the plugin class
  effgen-plugin-my_tools/pyproject.toml — package metadata & entry point

Next: cd into it and \`pip install -e .\` to register the plugin.
effgen-plugin-my_tools/my_tools/__init__.py
effgen-plugin-my_tools/my_tools/plugin.py
effgen-plugin-my_tools/my_tools/tools.py
effgen-plugin-my_tools/pyproject.toml
effgen-plugin-my_tools/README.md`} />

      <p>Wrap the tools in a <code>ToolPlugin</code>:</p>

      <CodeBlock
        filename="my_tools/plugin.py"
        code={`from effgen.tools.plugin import ToolPlugin

from my_tools.tools import MyTool


class MyPlugin(ToolPlugin):
    name = "my_tools"
    version = "1.0.0"
    tools = [MyTool]`}
      />

      <p>and declare the entry point so effGen finds it after an install:</p>

      <CodeBlock
        filename="pyproject.toml"
        language="toml"
        code={`[project.entry-points."effgen.plugins"]
my_tools = "my_tools.plugin:MyPlugin"`}
      />

      <ApiTable
        headers={['Where a plugin can come from', 'How']}
        rows={[
          [
            'An installed package',
            <>
              The <code>effgen.plugins</code> entry point above. Discovered on every start, no
              configuration.
            </>,
          ],
          [
            'A directory of .py files',
            <>
              <code>~/.effgen/plugins/</code>, or a directory named by{' '}
              <code>EFFGEN_PLUGINS_DIR</code>. Any file holding a <code>ToolPlugin</code> subclass
              is loaded.
            </>,
          ],
          [
            'Explicitly, in code',
            <>
              <code>PluginManager().discover_all()</code>, or{' '}
              <code>get_registry().register_plugin(plugin)</code>.
            </>,
          ],
        ]}
      />

      <CodeBlock filename="plugins.py" code={`from effgen.tools.plugin import PluginManager

manager = PluginManager()
manager.discover_all()
print(manager.loaded_plugins)`} />

      <Terminal
        command="python plugins.py"
        output={`{}`}
        caption="An empty list on a machine with no plugins installed — the discovery itself succeeded."
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <><code>TypeError: __init__() missing 1 required positional argument: 'metadata'</code></>,
            <>
              A <code>BaseTool</code> subclass was constructed with no arguments, and its{' '}
              <code>__init__</code> does not build its own metadata.
            </>,
            <>
              Build <code>ToolMetadata</code> inside <code>__init__</code> and pass it to{' '}
              <code>super().__init__(metadata=…)</code>.
            </>,
          ],
          [
            'The model never calls your tool',
            'The description does not say what it is for, or the parameter descriptions do not say what to pass.',
            'Both strings go into the prompt. Write them for the model, name the units, and say when the tool applies.',
          ],
          [
            <><code>result.success is False</code>, <code>Parameter validation failed: …</code></>,
            'The call did not match the ParameterSpec list.',
            <>
              Expected — that is the validation working. <code>_execute</code> was never reached.
            </>,
          ],
          [
            'A tool that raises rather than reporting',
            <>
              An exception escaped <code>_execute</code>.
            </>,
            <>
              The base class catches it and returns <code>success=False</code> with the message.
              Raise deliberately only for a programming error you want to surface.
            </>,
          ],
          [
            'A registered tool is not in `effgen tools list`',
            'The registration ran in a different process from the CLI.',
            <>
              <code>register_tool</code> is per-process. Ship it as a plugin, or drop the file into{' '}
              <code>~/.effgen/plugins/</code>, so every process picks it up.
            </>,
          ],
          [
            <code>ToolRegistrationError</code>,
            'Two tools claim the same name.',
            <>
              Pass <code>@tool(name="…")</code> to rename yours, or{' '}
              <code>unregister_tool</code> the one already there.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/tools', '/tools/gallery', '/protocols']} />
    </DocPage>
  );
}
