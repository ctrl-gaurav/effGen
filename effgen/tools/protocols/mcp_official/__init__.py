"""
Official Model Context Protocol (MCP) implementation for effGen.

This module provides standards-compliant MCP server and client implementations
using the official MCP Python SDK (https://github.com/modelcontextprotocol/python-sdk).

The MCP protocol enables applications to provide context to LLMs in a standardized
way, separating the concerns of providing context from the actual LLM interaction.

## Key Features

- **Standards-Compliant**: Implements the official MCP specification
- **FastMCP Integration**: Uses the high-level FastMCP server API
- **Multiple Transports**: Supports STDIO, HTTP, and SSE transports
- **Tool Exposure**: Automatically exposes effGen tools as MCP tools
- **Resource Management**: Support for exposing resources to LLMs
- **Structured Output**: Supports structured tool outputs with validation
- **Progress Reporting**: Real-time progress updates for long-running operations

## Installation

The MCP SDK is required for this module:
    pip install mcp[cli]>=1.0.0

## Server Usage

```python
from effgen.tools.registry import ToolRegistry
from effgen.tools.protocols.mcp_official import EffGenMCPServer

# Create registry and register tools
registry = ToolRegistry()
registry.discover_builtin_tools()

# Create and run server
server = EffGenMCPServer(tools_registry=registry)
await server.run_stdio()  # For Claude Desktop
# OR
await server.run_http()   # For HTTP clients
```

## Client Usage

```python
from effgen.tools.protocols.mcp_official import (
    EffGenMCPClient,
    MCPServerConfig
)

# Configure server connection
config = MCPServerConfig(
    name="my-server",
    transport="stdio",
    command="python",
    args=["server.py"]
)

# Use client
async with EffGenMCPClient(config) as client:
    # List available tools
    tools = client.get_tools()

    # Call a tool
    result = await client.call_tool("calculator", {"operation": "add", "a": 5, "b": 3})

    # Access result
    if result.isError:
        print(f"Error: {result.content[0].text}")
    else:
        print(f"Result: {result.structuredContent}")
```

## See Also

- Official MCP Documentation: https://modelcontextprotocol.io
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP Specification: https://spec.modelcontextprotocol.io
"""

import warnings

__version__ = "0.0.1"

# Check if MCP SDK is installed
try:
    import mcp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    warnings.warn(
        "MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0\n"
        "MCP features will not be available until the package is installed.",
        ImportWarning
    )

# Only import if MCP is available
if MCP_AVAILABLE:
    from .server import (
        EffGenMCPServer,
        EffGenMCPServerConfig,
        create_server,
        main_stdio,
        main_http,
    )

    from .client import (
        EffGenMCPClient,
        MCPServerConfig,
        create_client,
        create_stdio_client,
        create_http_client,
    )

    __all__ = [
        # Server
        "EffGenMCPServer",
        "EffGenMCPServerConfig",
        "create_server",
        "main_stdio",
        "main_http",
        # Client
        "EffGenMCPClient",
        "MCPServerConfig",
        "create_client",
        "create_stdio_client",
        "create_http_client",
        # Availability flag
        "MCP_AVAILABLE",
    ]
else:
    # Define placeholder classes that raise helpful errors
    class _MCPNotAvailable:
        """Placeholder class when MCP is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0"
            )

    EffGenMCPServer = _MCPNotAvailable
    EffGenMCPServerConfig = _MCPNotAvailable
    EffGenMCPClient = _MCPNotAvailable
    MCPServerConfig = _MCPNotAvailable

    def create_server(*args, **kwargs):
        raise ImportError("MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0")

    def create_client(*args, **kwargs):
        raise ImportError("MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0")

    def create_stdio_client(*args, **kwargs):
        raise ImportError("MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0")

    def create_http_client(*args, **kwargs):
        raise ImportError("MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0")

    def main_stdio(*args, **kwargs):
        raise ImportError("MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0")

    def main_http(*args, **kwargs):
        raise ImportError("MCP SDK not installed. Please install with: pip install mcp[cli]>=1.0.0")

    __all__ = [
        "EffGenMCPServer",
        "EffGenMCPServerConfig",
        "EffGenMCPClient",
        "MCPServerConfig",
        "create_server",
        "create_client",
        "create_stdio_client",
        "create_http_client",
        "main_stdio",
        "main_http",
        "MCP_AVAILABLE",
    ]

