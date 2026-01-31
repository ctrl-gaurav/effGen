"""
MCP client implementation for connecting to MCP servers.

This module provides a client for connecting to Model Context Protocol (MCP)
servers, discovering tools, and executing tool calls.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import subprocess
from dataclasses import dataclass
import httpx
from enum import Enum

from .protocol import (
    MCPProtocolHandler,
    MCPTool,
    MCPResource,
    MCPCapabilities,
    MCPRequest,
    MCPResponse,
    MCPError,
    ErrorCode,
    TransportType,
)
from ...base_tool import BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType


logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: TransportType
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    timeout: int = 30


class MCPTransport:
    """Base class for MCP transports."""

    async def send(self, message: MCPRequest) -> None:
        """Send a message."""
        raise NotImplementedError

    async def receive(self) -> MCPResponse:
        """Receive a message."""
        raise NotImplementedError

    async def connect(self) -> None:
        """Connect to the server."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        raise NotImplementedError


class StdioTransport(MCPTransport):
    """STDIO transport for MCP (subprocess communication)."""

    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize STDIO transport.

        Args:
            command: Command to execute
            args: Command arguments
            env: Environment variables
        """
        self.command = command
        self.args = args
        self.env = env
        self.process: Optional[asyncio.subprocess.Process] = None
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Start the subprocess."""
        full_env = dict(asyncio.subprocess.os.environ)
        if self.env:
            full_env.update(self.env)

        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        logger.info(f"Started MCP server: {self.command} {' '.join(self.args)}")

    async def disconnect(self) -> None:
        """Stop the subprocess."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            logger.info(f"Stopped MCP server: {self.command}")

    async def send(self, message: MCPRequest) -> None:
        """Send message to subprocess stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Transport not connected")

        async with self._write_lock:
            data = json.dumps(message.to_dict()) + "\n"
            self.process.stdin.write(data.encode())
            await self.process.stdin.drain()

    async def receive(self) -> MCPResponse:
        """Receive message from subprocess stdout."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Transport not connected")

        async with self._read_lock:
            line = await self.process.stdout.readline()
            if not line:
                raise ConnectionError("Server closed connection")

            data = json.loads(line.decode())
            handler = MCPProtocolHandler()
            message = handler.parse_message(data)

            if isinstance(message, MCPResponse):
                return message
            else:
                raise ValueError(f"Expected response, got {type(message)}")


class HTTPTransport(MCPTransport):
    """HTTP transport for MCP."""

    def __init__(self, url: str, timeout: int = 30):
        """
        Initialize HTTP transport.

        Args:
            url: Server URL
            timeout: Request timeout
        """
        self.url = url
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        """Create HTTP client."""
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()

    async def send(self, message: MCPRequest) -> None:
        """Send HTTP request."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        # Store for receive
        self._last_request = message

    async def receive(self) -> MCPResponse:
        """Send request and receive response."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        response = await self.client.post(
            self.url,
            json=self._last_request.to_dict(),
        )
        response.raise_for_status()

        data = response.json()
        handler = MCPProtocolHandler()
        message = handler.parse_message(data)

        if isinstance(message, MCPResponse):
            return message
        else:
            raise ValueError(f"Expected response, got {type(message)}")


class SSETransport(MCPTransport):
    """Server-Sent Events transport for MCP."""

    def __init__(self, url: str, timeout: int = 30):
        """
        Initialize SSE transport.

        Args:
            url: Server URL
            timeout: Request timeout
        """
        self.url = url
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        """Connect to SSE endpoint."""
        self.client = httpx.AsyncClient(timeout=self.timeout)
        # Start event listener
        asyncio.create_task(self._listen_events())

    async def disconnect(self) -> None:
        """Close SSE connection."""
        if self.client:
            await self.client.aclose()

    async def _listen_events(self) -> None:
        """Listen for SSE events."""
        if not self.client:
            return

        async with self.client.stream("GET", self.url) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    await self._event_queue.put(data)

    async def send(self, message: MCPRequest) -> None:
        """Send message via POST."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        await self.client.post(
            self.url,
            json=message.to_dict(),
        )

    async def receive(self) -> MCPResponse:
        """Receive message from event queue."""
        data = await self._event_queue.get()
        handler = MCPProtocolHandler()
        message = handler.parse_message(data)

        if isinstance(message, MCPResponse):
            return message
        else:
            raise ValueError(f"Expected response, got {type(message)}")


class MCPClient:
    """
    Client for connecting to MCP servers.

    Features:
    - Multiple transport support (STDIO, HTTP, SSE)
    - Server discovery and capability negotiation
    - Tool listing and execution
    - Resource management
    - Automatic reconnection
    - Error handling and retry logic
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self.transport: Optional[MCPTransport] = None
        self.protocol = MCPProtocolHandler()
        self.capabilities: Optional[MCPCapabilities] = None
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self._connected = False
        self._pending_requests: Dict[Union[str, int], asyncio.Future] = {}

    async def connect(self) -> None:
        """Connect to MCP server and perform initialization."""
        # Create transport
        if self.config.transport == TransportType.STDIO:
            if not self.config.command:
                raise ValueError("Command required for STDIO transport")
            self.transport = StdioTransport(
                self.config.command,
                self.config.args or [],
                self.config.env,
            )
        elif self.config.transport == TransportType.HTTP:
            if not self.config.url:
                raise ValueError("URL required for HTTP transport")
            self.transport = HTTPTransport(self.config.url, self.config.timeout)
        elif self.config.transport == TransportType.SSE:
            if not self.config.url:
                raise ValueError("URL required for SSE transport")
            self.transport = SSETransport(self.config.url, self.config.timeout)
        else:
            raise ValueError(f"Unsupported transport: {self.config.transport}")

        # Connect transport
        await self.transport.connect()

        # Initialize protocol
        await self._initialize()

        self._connected = True
        logger.info(f"Connected to MCP server: {self.config.name}")

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if self.transport:
            await self.transport.disconnect()
        self._connected = False
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _initialize(self) -> None:
        """Perform MCP initialization handshake."""
        # Send initialize request
        request = self.protocol.create_initialize_request(
            protocol_version="1.0",
            capabilities=MCPCapabilities(
                tools=True,
                resources=True,
            ),
            client_info={
                "name": "effGen",
                "version": "1.0.0",
            },
        )

        response = await self._send_request(request)

        if response.error:
            raise RuntimeError(f"Initialization failed: {response.error.message}")

        # Parse server capabilities
        result = response.result or {}
        self.capabilities = MCPCapabilities.from_dict(
            result.get("capabilities", {})
        )

        logger.info(f"Server capabilities: {self.capabilities.to_dict()}")

        # List tools if supported
        if self.capabilities.tools:
            await self._list_tools()

        # List resources if supported
        if self.capabilities.resources:
            await self._list_resources()

    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """
        Send request and wait for response.

        Args:
            request: Request to send

        Returns:
            Response from server
        """
        if not self.transport:
            raise RuntimeError("Not connected")

        # Send request
        await self.transport.send(request)

        # Wait for response
        response = await self.transport.receive()

        return response

    async def _list_tools(self) -> None:
        """List available tools from server."""
        request = self.protocol.create_tools_list_request()
        response = await self._send_request(request)

        if response.error:
            logger.error(f"Failed to list tools: {response.error.message}")
            return

        tools_data = response.result or {}
        tools_list = tools_data.get("tools", [])

        for tool_data in tools_list:
            tool = MCPTool.from_dict(tool_data)
            self.tools[tool.name] = tool

        logger.info(f"Discovered {len(self.tools)} tools")

    async def _list_resources(self) -> None:
        """List available resources from server."""
        request = self.protocol.create_resources_list_request()
        response = await self._send_request(request)

        if response.error:
            logger.error(f"Failed to list resources: {response.error.message}")
            return

        resources_data = response.result or {}
        resources_list = resources_data.get("resources", [])

        for resource_data in resources_list:
            resource = MCPResource.from_dict(resource_data)
            self.resources[resource.uri] = resource

        logger.info(f"Discovered {len(self.resources)} resources")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        request = self.protocol.create_tool_call_request(tool_name, arguments)
        response = await self._send_request(request)

        if response.error:
            raise RuntimeError(f"Tool call failed: {response.error.message}")

        return response.result

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")

        request = self.protocol.create_resource_read_request(uri)
        response = await self._send_request(request)

        if response.error:
            raise RuntimeError(f"Resource read failed: {response.error.message}")

        return response.result

    def get_tools(self) -> List[MCPTool]:
        """Get list of available tools."""
        return list(self.tools.values())

    def get_resources(self) -> List[MCPResource]:
        """Get list of available resources."""
        return list(self.resources.values())

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
