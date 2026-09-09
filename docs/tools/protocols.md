# Agent Interop Protocols (MCP, A2A, ACP)

effGen can speak to other agent ecosystems through standard protocols. The
implementations live under `effgen.tools.protocols`.

| Protocol | Status | What ships | Recommended entry point |
|----------|--------|------------|-------------------------|
| **MCP** (Model Context Protocol) | Stable | Server **and** client | `effgen.tools.protocols.mcp_official` |
| **A2A** (Agent-to-Agent) | Experimental | Client + protocol model | `effgen.tools.protocols.a2a` |
| **ACP** (Agent Communication Protocol) | Experimental | Server + client | `effgen.tools.protocols.acp` |

> **Security.** Every HTTP transport binds to `127.0.0.1` by default and warns
> when bound to a public address. Add authentication before exposing any of
> these servers to the network.

---

## MCP — Model Context Protocol

There are two MCP stacks:

- **`mcp_official`** — built on the official MCP Python SDK (FastMCP).
  **Use this** whenever `pip install "mcp[cli]"` is available.
- **`mcp`** — a self-contained implementation with no external SDK dependency,
  for environments where the official package cannot be installed.

### Expose effGen tools over MCP (server)

```python
import asyncio
from effgen.tools.protocols.mcp_official import create_server

# Fail-closed by default: code-execution / shell / filesystem tools
# (bash, python_repl, code_executor, …) are NOT exposed unless you opt in.
server = create_server(name="effgen-tools")

# Opt in explicitly when you really want them:
#   create_server(expose_unsafe_tools=True)        # expose everything
#   create_server(allowed_tools=["calculator"])    # expose only these
#   create_server(blocked_tools=["wikipedia"])     # hide specific tools

asyncio.run(server.run_stdio())          # for Claude Desktop / stdio clients
# asyncio.run(server.run_http(port=8000))  # streamable-HTTP at http://127.0.0.1:8000/mcp
```

### Call any MCP server (client)

```python
import asyncio, sys
from effgen.tools.protocols.mcp_official import EffGenMCPClient, MCPServerConfig

async def main():
    config = MCPServerConfig(
        name="effgen",
        transport="stdio",
        command=sys.executable,
        args=["-m", "effgen.tools.protocols.mcp_official"],  # stdio server
    )
    async with EffGenMCPClient(config) as client:
        for tool in client.get_tools():
            print(tool.name)
        result = await client.call_tool("calculator", {"expression": "15*15"})
        print(result.content[0].text)   # {"success": true, "output": {"result": 225}, ...}

asyncio.run(main())
```

For an HTTP server, use `MCPServerConfig(transport="streamable-http",
url="http://127.0.0.1:8000/mcp")`.

### Use an external MCP server's tools inside an effGen Agent

`get_effgen_tools()` wraps the discovered MCP tools as effGen tools, so an
`Agent` can call them. Keep the client connected for the agent run, and use the
async `run_async` entry point so the MCP session stays live:

```python
import asyncio
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.protocols.mcp_official import EffGenMCPClient, MCPServerConfig

async def main():
    config = MCPServerConfig(
        name="math", transport="stdio",
        command="python", args=["my_mcp_server.py"],
    )
    async with EffGenMCPClient(config) as client:
        tools = client.get_effgen_tools()          # -> list[BaseTool]
        agent = Agent(config=AgentConfig(
            name="mcp-consumer",
            model="openai/gpt-oss-20b", provider="groq",
            tools=tools,
        ))
        result = await agent.run_async("Use the add tool to compute 23 + 19.")
        print(result.output)

asyncio.run(main())
```

---

## A2A — Agent-to-Agent (experimental)

effGen ships the **client** side of A2A plus the agent-card model, the wire
protocol/task model, and authentication handlers. Point the client at an
external A2A-compatible agent.

```python
import asyncio
from effgen.tools.protocols.a2a import (
    AgentCard, Capability, CapabilityType, EndpointConfig, AuthScheme,
    A2AClient, A2AClientConfig, BearerAuthHandler,
)

card = AgentCard(
    name="remote-agent", description="a remote A2A agent", version="1.0.0",
    capabilities=[Capability(
        name="add", type=CapabilityType.TASK_EXECUTION,
        description="add numbers", inputSchema={"type": "object"},
    )],
    endpoint=EndpointConfig(url="https://example.com/agent", protocol="https"),
    authSchemes=[AuthScheme.BEARER],
)

async def main():
    client = A2AClient(card, A2AClientConfig(), BearerAuthHandler("YOUR_TOKEN"))
    await client.connect()
    try:
        task = await client.execute_task(capability="add", input_data={"a": 2, "b": 3})
        print(task)
    finally:
        await client.disconnect()

asyncio.run(main())
```

---

## ACP — Agent Communication Protocol (experimental)

effGen ships both an ACP **server** and **client**.

```python
import asyncio
from effgen.tools.protocols.acp import ACPServer, ACPClient, ACPRequest

async def echo(input_data, context):
    return {"echo": input_data.get("text", "")}

async def main():
    server = ACPServer(agent_id="echo", name="Echo", version="1.0.0",
                       description="echoes text")
    server.register_capability(
        name="echo", description="echo text back",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=echo,
    )

    # In-process call:
    resp = await server.handle_request(ACPRequest(capability="echo", input={"text": "hi"}))
    print(resp.status.value, resp.output)   # completed {'echo': 'hi'}

    # Or serve over HTTP (binds 127.0.0.1 by default) and call with ACPClient:
    #   server.run(port=8080)               # in another process
    #   async with ACPClient("http://127.0.0.1:8080") as client:
    #       out = await client.execute_sync("echo", {"text": "hi"})

asyncio.run(main())
```

Require authentication with `ACPServer(..., config=ACPServerConfig(require_auth=True))`
and issue capability tokens with `server.add_capability_token(...)`. Requests
without a valid token are rejected with an `UNAUTHORIZED` error.
