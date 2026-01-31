"""
Model Context Protocol (MCP) implementation for effGen.

This module provides MCP client and server implementations for tool
and resource communication.
"""

from .protocol import (
    MCPProtocolHandler,
    MCPTool,
    MCPResource,
    MCPCapabilities,
    MCPRequest,
    MCPResponse,
    MCPNotification,
    MCPError,
    ErrorCode,
    TransportType,
    MessageType,
    MCPVersion,
)

from .client import (
    MCPClient,
    MCPServerConfig,
    MCPTransport,
    StdioTransport,
    HTTPTransport,
    SSETransport,
)

from .server import (
    MCPServer,
    create_server,
    main_stdio,
    main_http,
)


__all__ = [
    # Protocol
    "MCPProtocolHandler",
    "MCPTool",
    "MCPResource",
    "MCPCapabilities",
    "MCPRequest",
    "MCPResponse",
    "MCPNotification",
    "MCPError",
    "ErrorCode",
    "TransportType",
    "MessageType",
    "MCPVersion",
    # Client
    "MCPClient",
    "MCPServerConfig",
    "MCPTransport",
    "StdioTransport",
    "HTTPTransport",
    "SSETransport",
    # Server
    "MCPServer",
    "create_server",
    "main_stdio",
    "main_http",
]
