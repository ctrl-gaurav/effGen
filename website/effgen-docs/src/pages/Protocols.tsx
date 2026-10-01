import { Network } from 'lucide-react';
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
import { toolCount, version } from '../siteData';

export default function Protocols() {
  return (
    <DocPage
      subtitle="Connecting to tools and agents that speak Model Context Protocol, A2A or ACP."
      icon={<Network size={48} />}
    >
      <p>
        effGen can be either end of an interop protocol: it serves its own tools over MCP for any
        MCP client to call, and it consumes tools and agents from MCP, A2A and ACP servers as if
        they were local.
      </p>

      <ApiTable
        headers={['Protocol', 'Status', 'What ships', 'Import from']}
        rows={[
          [
            <strong>MCP</strong>,
            'Stable',
            'Server and client',
            <code>effgen.tools.protocols.mcp_official</code>,
          ],
          [
            <strong>A2A</strong>,
            'Experimental',
            'Client, plus the agent-card and task models',
            <code>effgen.tools.protocols.a2a</code>,
          ],
          [
            <strong>ACP</strong>,
            'Experimental',
            'Server and client',
            <code>effgen.tools.protocols.acp</code>,
          ],
        ]}
        caption={`Model Context Protocol, Agent-to-Agent and Agent Communication Protocol, in effGen ${version}.`}
      />

      <Callout type="warning" title="Every HTTP transport binds to loopback">
        <p>
          All three servers listen on <code>127.0.0.1</code> by default and warn when bound to a
          public address. Put authentication in front of one before it is reachable from anywhere
          else — <Link to="/api-server">API server</Link> covers the auth and RBAC effGen ships.
        </p>
      </Callout>

      <h2>Serving effGen tools over MCP</h2>

      <CodeBlock filename="server.py" code={`from effgen.tools.protocols.mcp_official import create_server

# Fail-closed: the shell, REPL and code-execution tools are not exposed
# unless you ask for them by name.
server = create_server(name="effgen-tools")
print(type(server).__name__)

# server.run_stdio()                 # for a stdio client
# server.run_http(port=8000)         # streamable HTTP at http://127.0.0.1:8000/mcp`} />

      <Terminal
        command="python server.py"
        output={`EffGenMCPServer`}
        caption={`Run against effGen 1.0.0.`}
      />

      <Callout type="danger" title="The dangerous tools are not exposed unless you say so">
        <p>
          <code>create_server()</code> is fail-closed: the shell, the REPL and the code-execution
          tools are held back, because an MCP client is often something you did not write. Opt in
          by name rather than wholesale where you can.
        </p>
      </Callout>

      <ParamTable
        nameLabel="Argument"
        params={[
          { name: 'name', type: 'str', default: "'effgen-tools'", description: 'The server name MCP clients see.' },
          {
            name: 'expose_unsafe_tools',
            type: 'bool',
            default: 'False',
            description: 'Expose the code-execution, shell and filesystem tools as well. Off unless you set it.',
          },
          {
            name: 'allowed_tools',
            type: 'list[str] | None',
            default: 'None',
            description: 'Expose only these tools, by name. The narrowest option, and the one to prefer.',
          },
          {
            name: 'blocked_tools',
            type: 'list[str] | None',
            default: 'None',
            description: 'Expose everything else but these.',
          },
        ]}
        caption={<><code>effgen.tools.protocols.mcp_official.create_server</code></>}
      />

      <ApiTable
        headers={['Transport', 'How to run it', 'Who connects']}
        rows={[
          [
            'stdio',
            <code>server.run_stdio()</code>,
            'A desktop MCP client that launches the server as a subprocess.',
          ],
          [
            'streamable HTTP',
            <code>server.run_http(port=8000)</code>,
            <>
              Anything that can reach <code>http://127.0.0.1:8000/mcp</code>.
            </>,
          ],
        ]}
      />

      <h3>Two MCP stacks</h3>

      <ApiTable
        headers={['Module', 'When to use it']}
        rows={[
          [
            <code>mcp_official</code>,
            <>
              Built on the official MCP Python SDK (FastMCP). <strong>Use this one</strong>{' '}
              whenever <code>pip install "mcp[cli]"</code> is possible.
            </>,
          ],
          [
            <code>mcp</code>,
            'A self-contained implementation with no external SDK dependency, for an environment where the official package cannot be installed.',
          ],
        ]}
      />

      <h2>Calling any MCP server</h2>

      <CodeBlock filename="client.py" code={`import asyncio
import sys

from effgen.tools.protocols.mcp_official import EffGenMCPClient, MCPServerConfig


async def main():
    config = MCPServerConfig(
        name="effgen",
        transport="stdio",
        command=sys.executable,
        args=["-m", "effgen.tools.protocols.mcp_official"],
    )
    async with EffGenMCPClient(config) as client:
        names = sorted(tool.name for tool in client.get_tools())
        print(len(names), "tools offered; first five:", names[:5])

        result = await client.call_tool("calculator", {"expression": "15*15"})
        print(result.content[0].text)


asyncio.run(main())`} />

      <Terminal command="python client.py" output={`54 tools offered; first five: ['agentic_search', 'arxiv', 'audio_transcribe', 'calculator', 'crypto_price']
{
  "success": true,
  "output": {
    "result": 225,
    "formatted": "225",
    "expression": "15*15"
  },
  "metadata": {
    "tool_name": "calculator",
    "tool_version": "1.0.0"
  }
}`} />

      <p>
        Fifty-four of the {toolCount} tools are offered, not all of them — the fail-closed
        default is what holds the rest back. The tool result comes back as the serialised{' '}
        <code>ToolResult</code> — the same{' '}
        <code>success</code> / <code>output</code> / <code>error</code> record{' '}
        <Link to="/tools">a local call</Link> returns, over the wire.
      </p>

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'name', type: 'str', required: true, description: 'A label for this server, used in logs and errors.' },
          {
            name: 'transport',
            type: 'str',
            required: true,
            description: 'stdio or streamable-http.',
          },
          {
            name: 'command',
            type: 'str | None',
            default: 'None',
            description: 'stdio only — the executable to launch.',
          },
          {
            name: 'args',
            type: 'list[str]',
            default: '[]',
            description: 'stdio only — its arguments.',
          },
          {
            name: 'url',
            type: 'str | None',
            default: 'None',
            description: 'streamable-http only — the endpoint, e.g. http://127.0.0.1:8000/mcp.',
          },
        ]}
        caption={<><code>MCPServerConfig</code>, the one object both transports are described with.</>}
      />

      <h2>Giving an agent an MCP server's tools</h2>
      <p>
        <code>get_effgen_tools()</code> wraps what the client discovered as ordinary effGen tools,
        so an <code>Agent</code> can call them. The session has to stay open for the whole run,
        which is why this uses <code>run_async</code> inside the <code>async with</code>.
      </p>

      <CodeBlock filename="mcp_agent.py" code={`import asyncio
import sys

from effgen import Agent, AgentConfig
from effgen.tools.protocols.mcp_official import EffGenMCPClient, MCPServerConfig


async def main():
    config = MCPServerConfig(
        name="effgen",
        transport="stdio",
        command=sys.executable,
        args=["-m", "effgen.tools.protocols.mcp_official"],
    )
    async with EffGenMCPClient(config) as client:
        tools = [t for t in client.get_effgen_tools() if t.name == "calculator"]
        agent = Agent(AgentConfig(
            name="mcp-consumer",
            model="gpt-5-nano",
            provider="openai",
            tools=tools,
        ))
        response = await agent.run_async("Use the calculator tool to compute 23 + 19.")
        print(response.text)


asyncio.run(main())`} />

      <Terminal command="python mcp_agent.py" output={`42`} />

      <h2>ACP — Agent Communication Protocol</h2>
      <p>
        Experimental, and both ends ship. A capability is a named handler with an input schema; the
        server can be called in-process or served over HTTP.
      </p>

      <CodeBlock filename="acp.py" code={`import asyncio

from effgen.tools.protocols.acp import ACPRequest, ACPServer


async def echo(input_data, context):
    return {"echo": input_data.get("text", "")}


async def main():
    server = ACPServer(agent_id="echo", name="Echo", version="1.0.0", description="echoes text")
    server.register_capability(
        name="echo",
        description="echo text back",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=echo,
    )

    response = await server.handle_request(ACPRequest(capability="echo", input={"text": "hi"}))
    print(response.status.value, response.output)


asyncio.run(main())`} />

      <Terminal command="python acp.py" output={`completed {'echo': 'hi'}`} />

      <p>
        Require authentication with{' '}
        <code>ACPServer(..., config=ACPServerConfig(require_auth=True))</code> and issue capability
        tokens with <code>server.add_capability_token(...)</code>. A request without a valid token
        is rejected with an <code>UNAUTHORIZED</code> error rather than being served.
      </p>

      <h2>A2A — Agent-to-Agent</h2>
      <p>
        Experimental, and client-side: effGen ships the agent-card model, the wire and task model,
        the authentication handlers and <code>A2AClient</code>, and you point it at an external
        A2A-compatible agent.
      </p>

      <CodeBlock filename="a2a_card.py" code={`from effgen.tools.protocols.a2a import (
    AgentCard, AuthScheme, Capability, CapabilityType, EndpointConfig,
)

card = AgentCard(
    name="remote-agent",
    description="a remote A2A agent",
    version="1.0.0",
    capabilities=[Capability(
        name="add",
        type=CapabilityType.TASK_EXECUTION,
        description="add numbers",
        inputSchema={"type": "object"},
    )],
    endpoint=EndpointConfig(url="https://example.com/agent", protocol="https"),
    authSchemes=[AuthScheme.BEARER],
)

print(card.name, "|", [c.name for c in card.capabilities], "|", card.endpoint.url)`} />

      <Terminal command="python a2a_card.py" output={`remote-agent | ['add'] | https://example.com/agent`} />

      <p>Then connect, execute a task against a capability, and disconnect:</p>

      <CodeBlock
        continues
        filename="a2a_client.py"
        code={`import asyncio
import os

from effgen.tools.protocols.a2a import A2AClient, A2AClientConfig, BearerAuthHandler

# \`card\` is the AgentCard built above.


async def main():
    client = A2AClient(card, A2AClientConfig(), BearerAuthHandler(os.environ["A2A_TOKEN"]))
    await client.connect()
    try:
        task = await client.execute_task(capability="add", input_data={"a": 2, "b": 3})
        print(task)
    finally:
        await client.disconnect()


asyncio.run(main())`}
        caption="Not run on this site: it needs an external A2A agent to connect to, and there is none here. The card above was built and read back."
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <><code>ModuleNotFoundError: No module named 'mcp'</code></>,
            <>
              <code>mcp_official</code> was imported without the official SDK installed.
            </>,
            <>
              <code>pip install "mcp[cli]"</code>, or import{' '}
              <code>effgen.tools.protocols.mcp</code> instead.
            </>,
          ],
          [
            'A tool you expected is missing from `get_tools()`',
            'It is one of the shell, REPL or code-execution tools, which are held back by default.',
            <>
              Pass <code>allowed_tools=[…]</code> naming it, or{' '}
              <code>expose_unsafe_tools=True</code> if you really mean all of them.
            </>,
          ],
          [
            'The agent run fails after the client closes',
            <>
              The <code>async with</code> block ended before the run did.
            </>,
            <>
              Keep the run inside the block and use <code>await agent.run_async(...)</code> — the
              synchronous <code>run()</code> cannot hold the session open.
            </>,
          ],
          [
            'A connection refused on the HTTP transport',
            'Nothing is listening, or the server bound to loopback and the client is on another host.',
            <>
              Check the URL ends in <code>/mcp</code>, and remember the default bind is{' '}
              <code>127.0.0.1</code>.
            </>,
          ],
          [
            <>An ACP response with status <code>failed</code></>,
            'The capability handler raised, or the input did not match its schema.',
            <>
              Read <code>response.error</code>. The status is on{' '}
              <code>response.status.value</code>, so a failure is never a silent empty output.
            </>,
          ],
          [
            <>An ACP <code>UNAUTHORIZED</code> error</>,
            <>
              The server was built with <code>require_auth=True</code> and the request carried no
              valid capability token.
            </>,
            <>
              Issue one with <code>server.add_capability_token(...)</code> and send it with the
              request.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/tools', '/custom-tools', '/api-server']} />
    </DocPage>
  );
}
