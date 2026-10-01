import { Cloud } from 'lucide-react';
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

export default function NativeProviderTools() {
  return (
    <DocPage
      subtitle="Web search, code execution and file search run by the provider rather than by effGen."
      icon={<Cloud size={48} />}
    >
      <p>
        Some providers run tools on their own machines: the model calls them without a round trip
        back to you. effGen wraps nine of them as ordinary tool objects, so they go into{' '}
        <code>tools=[…]</code> beside local ones — and refuses at construction if the model they
        are paired with cannot run them.
      </p>

      <h2>One of them, working</h2>

      <CodeBlock filename="search_agent.py" code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

agent = Agent(AgentConfig(
    name="search-agent",
    model="gpt-5-nano",
    provider="openai",
    tools=[OpenAIWebSearchTool()],
    tool_calling_mode="native",
))

response = agent.run("What is the current version of the Python language? Cite a source.")
print(response.text)`} />

      <Terminal
        command="python search_agent.py"
        output={`Python 3.14.7.`}
        caption={`Run against effGen 1.0.0 on 2026-08-23.`}
      />

      <h2>What ships</h2>

      <ApiTable
        headers={['Tool', 'Class', 'Provider type', 'What it does']}
        rows={[
          [
            <code>openai_web_search</code>,
            <code>OpenAIWebSearchTool</code>,
            <code>web_search_preview</code>,
            'Live web queries with citations.',
          ],
          [
            <code>openai_code_interpreter</code>,
            <code>OpenAICodeInterpreterTool</code>,
            <code>code_interpreter</code>,
            'A Python runtime on OpenAI’s machines.',
          ],
          [
            <code>openai_file_search</code>,
            <code>OpenAIFileSearchTool</code>,
            <code>file_search</code>,
            'Vector search over documents you uploaded to OpenAI.',
          ],
          [
            <code>google_search</code>,
            <code>GoogleSearchTool</code>,
            <code>google_search</code>,
            'Google Search grounding for a Gemini answer.',
          ],
          [
            <code>url_context</code>,
            <code>GeminiUrlContextTool</code>,
            <code>url_context</code>,
            'Lets Gemini read the pages named in the prompt.',
          ],
          [
            <code>code_execution</code>,
            <code>GeminiCodeExecutionTool</code>,
            <code>code_execution</code>,
            'Gemini writes and runs Python on Google’s machines.',
          ],
          [
            <code>anthropic_bash</code>,
            <code>AnthropicBashTool</code>,
            <code>bash</code>,
            'Shell commands in an Anthropic-hosted sandbox. Experimental.',
          ],
          [
            <code>anthropic_text_editor</code>,
            <code>AnthropicTextEditorTool</code>,
            <code>text_editor</code>,
            'View and edit files in an Anthropic-hosted environment. Experimental.',
          ],
          [
            <code>anthropic_computer</code>,
            <code>AnthropicComputerTool</code>,
            <code>computer</code>,
            'Mouse, keyboard and screenshots on a virtual machine. Experimental.',
          ],
        ]}
        caption="Nine tools across three providers. All of them run on the provider’s infrastructure, never on yours."
      />

      <h3>The spec each one sends</h3>
      <p>
        Every wrapper knows the exact object its provider expects, which is worth seeing because it
        is the whole of what leaves your process.
      </p>

      <CodeBlock filename="specs.py" code={`from effgen.tools.builtin.anthropic_native import (
    AnthropicBashTool, AnthropicComputerTool, AnthropicTextEditorTool,
)
from effgen.tools.builtin.gemini_native import (
    GeminiCodeExecutionTool, GeminiUrlContextTool, GoogleSearchTool,
)
from effgen.tools.builtin.openai_native import (
    OpenAICodeInterpreterTool, OpenAIFileSearchTool, OpenAIWebSearchTool,
)

for tool in [OpenAIWebSearchTool(), OpenAICodeInterpreterTool(), OpenAIFileSearchTool()]:
    print(f"{tool.name:24} {tool.to_openai_tool_spec()}")

for tool in [GoogleSearchTool(), GeminiUrlContextTool(), GeminiCodeExecutionTool()]:
    print(f"{tool.name:24} {tool.to_gemini_tool()}")

for tool in [AnthropicBashTool(), AnthropicTextEditorTool(), AnthropicComputerTool()]:
    print(f"{tool.name:24} {tool.to_anthropic_tool_spec()}")`} />

      <Terminal command="python specs.py" output={`openai_web_search        {'type': 'web_search_preview', 'search_context_size': 'medium'}
openai_code_interpreter  {'type': 'code_interpreter', 'container': {'type': 'auto'}}
openai_file_search       {'type': 'file_search', 'max_num_results': 10}
google_search            retrieval=None computer_use=None file_search=None google_search=GoogleSearch() google_maps=None code_execution=None enterprise_web_search=None function_declarations=None google_search_retrieval=None parallel_ai_search=None url_context=None mcp_servers=None
url_context              retrieval=None computer_use=None file_search=None google_search=None google_maps=None code_execution=None enterprise_web_search=None function_declarations=None google_search_retrieval=None parallel_ai_search=None url_context=UrlContext() mcp_servers=None
code_execution           retrieval=None computer_use=None file_search=None google_search=None google_maps=None code_execution=ToolCodeExecution() enterprise_web_search=None function_declarations=None google_search_retrieval=None parallel_ai_search=None url_context=None mcp_servers=None
anthropic_bash           {'type': 'bash_20250124', 'name': 'bash'}
anthropic_text_editor    {'type': 'text_editor_20250728', 'name': 'str_replace_based_edit_tool'}
anthropic_computer       {'type': 'computer_20251124', 'name': 'computer', 'display_width_px': 1024, 'display_height_px': 768, 'display_number': 1}`} />

      <Callout type="warning" title="OpenAI web search carries a per-call surcharge">
        <p>
          <code>OpenAIWebSearchTool</code> is billed on top of normal token cost — roughly{' '}
          <strong>$30 per 1,000 calls</strong> for GPT-4o-class models as of 2026-04-24, and lower
          tiers differ. <code>code_interpreter</code> and <code>file_search</code> have no search
          surcharge, though running code can add compute charges on OpenAI's side. Check the
          provider's own pricing before you put one in a loop, and watch{' '}
          <Link to="/cost">Cost &amp; budgets</Link>.
        </p>
      </Callout>

      <h2>Pairing one with the wrong model</h2>
      <p>
        A native tool needs its own provider's adapter. Rather than letting the request fail
        somewhere inside the provider's API, effGen raises at <code>Agent</code> construction,
        before anything is sent.
      </p>

      <CodeBlock filename="incompatible.py" code={`from effgen import Agent, AgentConfig
from effgen.models.errors import ToolIncompatibleError
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

try:
    Agent(AgentConfig(
        name="bad-agent",
        model="gemini-3.1-flash-lite",
        provider="gemini",
        tools=[OpenAIWebSearchTool()],
    ))
except ToolIncompatibleError as exc:
    print(type(exc).__name__)
    print(exc)`} />

      <Terminal command="python incompatible.py" output={`ToolIncompatibleError
Tool 'openai_web_search' is incompatible with model 'gemini-3.1-flash-lite'. OpenAI native tools (web_search, code_interpreter, file_search) are executed server-side by OpenAI and require an OpenAIAdapter. Current model: 'gemini-3.1-flash-lite'. Switch to an OpenAI model or remove the native tool. Remove the tool from the agent, or choose a model that supports tool calling.`} />

      <h2>Mixing native and local tools</h2>
      <p>
        One agent can hold both. The adapter routes native tools through the provider's own path
        and dispatches the local ones here, in the same run.
      </p>

      <CodeBlock filename="hybrid.py" code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

agent = Agent(AgentConfig(
    name="hybrid-agent",
    model="gpt-5-nano",
    provider="openai",
    tools=[
        OpenAIWebSearchTool(),   # runs on OpenAI's side
        Calculator(),            # runs here
    ],
    tool_calling_mode="auto",
))

response = agent.run("What is 4817 * 236? Use the calculator.")
print(response.text)
print("tools called:", [call.name for call in response.tool_calls])`} />

      <Terminal command="python hybrid.py" output={`4817 × 236 = 1,136,812.
tools called: ['calculator']`} />

      <h2>The OpenAI tools in detail</h2>

      <h3>Web search</h3>

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'search_context_size',
            type: 'str',
            default: "'medium'",
            description: 'low, medium or high — how much context each result contributes. Higher costs more tokens.',
          },
          {
            name: 'user_location',
            type: 'dict | None',
            default: 'None',
            description: 'Geographic context, e.g. {"type": "approximate", "country": "US"}.',
          },
        ]}
        caption={<><code>OpenAIWebSearchTool(...)</code></>}
      />

      <h3>Code interpreter</h3>

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'container',
            type: 'dict',
            default: '{"type": "auto"}',
            description: 'Container configuration for the sandbox. auto lets OpenAI choose the runtime.',
          },
        ]}
        caption={<><code>OpenAICodeInterpreterTool(...)</code>. Results — stdout, stderr and files — come back in the response.</>}
      />

      <h3>File search</h3>

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'vector_store_ids',
            type: 'list[str]',
            default: '[]',
            description: 'The OpenAI vector stores to search. Created through the Files API.',
          },
          { name: 'max_num_results', type: 'int', default: '10', description: 'How many chunks to retrieve, 1–50.' },
          {
            name: 'ranking_options',
            type: 'dict | None',
            default: 'None',
            description: 'Override the ranker or the score threshold.',
          },
          {
            name: 'filters',
            type: 'dict | None',
            default: 'None',
            description: 'A metadata filter, for retrieval scoped to part of a store.',
          },
        ]}
        caption={
          <>
            <code>OpenAIFileSearchTool(...)</code>. <code>add_vector_store(id)</code> and{' '}
            <code>remove_vector_store(id)</code> change the list at runtime.
          </>
        }
      />

      <CodeBlock
        filename="vector_store.py"
        code={`from openai import OpenAI

from effgen.tools.builtin.openai_native import OpenAIFileSearchTool

client = OpenAI()

with open("my_doc.txt", "rb") as handle:
    uploaded = client.files.create(file=handle, purpose="assistants")

store = client.vector_stores.create(name="my-docs")
client.vector_stores.files.create(vector_store_id=store.id, file_id=uploaded.id)

tool = OpenAIFileSearchTool(vector_store_ids=[store.id])`}
        caption="Not run on this site: it uploads a document to an OpenAI account and leaves a vector store behind. It is the openai client’s own API, unchanged."
      />

      <h2>The Gemini tools</h2>
      <p>
        Gemini's three are activated the same way, against a Gemini model. Grounded search returns
        an answer with the sources Gemini used; <code>url_context</code> lets it read pages named
        in the prompt; <code>code_execution</code> lets it write and run Python on Google's
        machines.
      </p>

      <CodeBlock
        filename="gemini_search.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin.gemini_native import GoogleSearchTool

agent = Agent(AgentConfig(
    name="search-agent",
    model="gemini-3.1-flash-lite",
    provider="gemini",
    tools=[GoogleSearchTool()],
    tool_calling_mode="native",
))

response = agent.run("Who won the 2026 Nobel Prize in Physics?")
print(response.text)`}
        caption="Written and run, but the run was refused: this account's Gemini quota returned 429 RESOURCE_EXHAUSTED, so no output is claimed for it. The spec it sends is in the specs.py output above, which did run."
      />

      <h2>The Anthropic tools</h2>
      <p>
        The three computer-use tools are marked <strong>experimental</strong> by the framework.
        They need an Anthropic model, the{' '}
        <code>computer-use-2025-11-24</code> beta header, and{' '}
        <code>anthropic_computer</code> additionally only runs on Opus 4.7, Opus 4.6, Sonnet 4.6
        and Opus 4.5. The specs above were read from the installed package; no live Anthropic call
        was made from this site.
      </p>

      <h2>Native or local?</h2>

      <ApiTable
        headers={['What you want', 'Use']}
        rows={[
          ['The latest live web data, with citations', <code>OpenAIWebSearchTool</code>],
          [
            'Web search you control the sources of',
            <>
              <code>WebSearch</code> — see <Link to="/tools/gallery">the gallery</Link>
            </>,
          ],
          ['Sandboxed code execution on the provider’s infrastructure', <code>OpenAICodeInterpreterTool</code>],
          [
            'Code execution on your machine, in a sandbox you configure',
            <>
              <code>CodeExecutor</code> — see <Link to="/execution">Code execution</Link>
            </>,
          ],
          ['Retrieval over files already in an OpenAI vector store', <code>OpenAIFileSearchTool</code>],
          [
            'Retrieval over local documents with your own embeddings',
            <>
              <code>effgen.rag</code> and the <code>retrieval</code> tool — see{' '}
              <Link to="/rag">RAG</Link>
            </>,
          ],
        ]}
        caption="Native tools move the work to the provider: less to set up, an extra bill, and one provider. Local tools run on your machine, cost nothing per call, and work with any model."
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>ToolIncompatibleError</code>,
            'A native tool was paired with a model from another provider.',
            <>
              Raised at <code>Agent</code> construction, before any request. Switch the model, or
              drop the tool.
            </>,
          ],
          [
            'The tool is never used',
            <>
              The definitions went out on a path that cannot carry them.
            </>,
            <>
              Set <code>tool_calling_mode="native"</code> so the adapter routes them the way the
              provider expects.
            </>,
          ],
          [
            'A bill much larger than the token cost',
            <>
              <code>openai_web_search</code> is billed per call on top of tokens.
            </>,
            <>
              Lower <code>search_context_size</code>, call it less, and set a budget —{' '}
              <Link to="/cost">Cost &amp; budgets</Link>.
            </>,
          ],
          [
            <>A <code>429 RESOURCE_EXHAUSTED</code> from Gemini</>,
            'The account’s quota for that model is spent. Grounded search costs more quota than a plain call.',
            'Wait for the quota window, or use a key on a paid tier. Nothing about the code is wrong.',
          ],
          [
            'File search returns nothing',
            'The vector store is empty, or the files are still being indexed.',
            <>
              Check the store through the OpenAI Files API. <code>vector_store_ids</code> has to
              name a store that has finished indexing.
            </>,
          ],
          [
            'An Anthropic tool is refused',
            <>
              The beta header is missing, or the model is not one of the four{' '}
              <code>anthropic_computer</code> supports.
            </>,
            <>
              Use a supported model and send{' '}
              <code>computer-use-2025-11-24</code>. These three are experimental.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/tool-calling', '/tools/gallery', '/providers']} />
    </DocPage>
  );
}
