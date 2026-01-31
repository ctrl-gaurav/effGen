"""
MCP server implementation to expose effGen tools.

This module provides a server that exposes effGen tools as MCP tools,
allowing them to be used by MCP clients like Claude Desktop.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .protocol import (
    MCPProtocolHandler,
    MCPTool,
    MCPResource,
    MCPCapabilities,
    MCPRequest,
    MCPResponse,
    MCPNotification,
    ErrorCode,
)
from ...base_tool import BaseTool, ToolMetadata
from ...registry import ToolRegistry


logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for MCP server."""
    name: str
    version: str
    capabilities: MCPCapabilities
    tools_registry: Optional[ToolRegistry] = None


class MCPServer:
    """
    MCP server to expose effGen tools.

    Features:
    - Expose effGen tools as MCP tools
    - Handle MCP protocol messages
    - Support STDIO transport for Claude Desktop
    - Tool execution with error handling
    - Resource exposure
    - Sampling capability for nested LLM calls

    Example:
        registry = ToolRegistry()
        registry.register_tool(MyTool)

        server = MCPServer(
            name="effgen-tools",
            version="1.0.0",
            tools_registry=registry
        )

        await server.run_stdio()
    """

    def __init__(
        self,
        name: str,
        version: str,
        tools_registry: Optional[ToolRegistry] = None,
        enable_sampling: bool = False,
    ):
        """
        Initialize MCP server.

        Args:
            name: Server name
            version: Server version
            tools_registry: Tool registry to expose
            enable_sampling: Whether to enable sampling capability
        """
        self.name = name
        self.version = version
        self.tools_registry = tools_registry or ToolRegistry()
        self.protocol = MCPProtocolHandler()
        self.capabilities = MCPCapabilities(
            tools=True,
            resources=False,
            prompts=False,
            sampling=enable_sampling,
        )
        self._initialized = False
        self._methods: Dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tool_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resource_read,
        }

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle initialize request.

        Args:
            params: Request parameters

        Returns:
            Initialization response
        """
        protocol_version = params.get("protocolVersion")
        client_capabilities = MCPCapabilities.from_dict(
            params.get("capabilities", {})
        )
        client_info = params.get("clientInfo", {})

        logger.info(
            f"Initializing with client: {client_info.get('name')} v{client_info.get('version')}"
        )

        # Initialize tools registry
        if not self._initialized:
            self.tools_registry.discover_builtin_tools()
            await self.tools_registry.initialize_all()
            self._initialized = True

        return {
            "protocolVersion": "1.0",
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }

    async def _handle_tools_list(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle tools list request.

        Args:
            params: Request parameters

        Returns:
            List of available tools
        """
        tools = []

        # Get all tools from registry
        for tool_name in self.tools_registry.list_tools():
            metadata = self.tools_registry.get_metadata(tool_name)
            mcp_tool = self._convert_tool_to_mcp(metadata)
            tools.append(mcp_tool.to_dict())

        return {"tools": tools}

    async def _handle_tool_call(self, params: Dict[str, Any]) -> Any:
        """
        Handle tool call request.

        Args:
            params: Request parameters with tool name and arguments

        Returns:
            Tool execution result
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        # Get tool from registry
        try:
            tool = await self.tools_registry.get_tool(tool_name)
        except KeyError:
            raise ValueError(f"Tool not found: {tool_name}")

        # Execute tool
        result = await tool.execute(**arguments)

        # Format response
        if result.success:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result.output, indent=2) if result.output else "",
                    }
                ],
                "isError": False,
            }
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result.error or "Tool execution failed",
                    }
                ],
                "isError": True,
            }

    async def _handle_resources_list(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle resources list request.

        Args:
            params: Request parameters

        Returns:
            List of available resources
        """
        # For now, we don't expose any resources
        # This can be extended to expose agent state, logs, etc.
        return {"resources": []}

    async def _handle_resource_read(self, params: Dict[str, Any]) -> Any:
        """
        Handle resource read request.

        Args:
            params: Request parameters with resource URI

        Returns:
            Resource content
        """
        uri = params.get("uri")
        if not uri:
            raise ValueError("Resource URI is required")

        # Not implemented yet
        raise ValueError(f"Resource not found: {uri}")

    def _convert_tool_to_mcp(self, metadata: ToolMetadata) -> MCPTool:
        """
        Convert effGen tool metadata to MCP tool format.

        Args:
            metadata: Tool metadata

        Returns:
            MCP tool specification
        """
        # Build JSON Schema for input
        properties = {}
        required = []

        for param in metadata.parameters:
            prop = {
                "type": param.type.value,
                "description": param.description,
            }

            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        input_schema = {
            "type": "object",
            "properties": properties,
        }

        if required:
            input_schema["required"] = required

        return MCPTool(
            name=metadata.name,
            description=metadata.description,
            inputSchema=input_schema,
            returnSchema=metadata.returns if metadata.returns else None,
        )

    async def _process_request(self, request: MCPRequest) -> MCPResponse:
        """
        Process an MCP request.

        Args:
            request: Request to process

        Returns:
            Response
        """
        method = request.method
        params = request.params

        try:
            # Find handler for method
            handler = self._methods.get(method)
            if not handler:
                return self.protocol.create_response(
                    request.id,
                    error=self.protocol.create_error(
                        ErrorCode.METHOD_NOT_FOUND,
                        f"Method not found: {method}",
                    ),
                )

            # Execute handler
            result = await handler(params)

            return self.protocol.create_response(
                request.id,
                result=result,
            )

        except Exception as e:
            logger.error(f"Error handling {method}: {e}", exc_info=True)
            return self.protocol.create_response(
                request.id,
                error=self.protocol.create_error(
                    ErrorCode.INTERNAL_ERROR,
                    str(e),
                ),
            )

    async def run_stdio(self) -> None:
        """
        Run server with STDIO transport.

        This is the main entry point for Claude Desktop integration.
        Reads JSON-RPC messages from stdin and writes responses to stdout.
        """
        logger.info(f"Starting MCP server: {self.name} v{self.version}")

        try:
            # Read from stdin line by line
            while True:
                try:
                    # Read line from stdin
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )

                    if not line:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    # Parse request
                    try:
                        message = self.protocol.parse_message(line)
                    except Exception as e:
                        logger.error(f"Failed to parse message: {e}")
                        error_response = self.protocol.create_response(
                            None,
                            error=self.protocol.create_error(
                                ErrorCode.PARSE_ERROR,
                                f"Parse error: {e}",
                            ),
                        )
                        print(self.protocol.serialize_message(error_response), flush=True)
                        continue

                    # Handle notification (no response needed)
                    if isinstance(message, MCPNotification):
                        logger.debug(f"Received notification: {message.method}")
                        continue

                    # Process request
                    if isinstance(message, MCPRequest):
                        response = await self._process_request(message)

                        # Write response to stdout
                        output = self.protocol.serialize_message(response)
                        print(output, flush=True)

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            # Cleanup
            await self.tools_registry.cleanup_all()
            logger.info("Server shutdown complete")

    async def run_http(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """
        Run server with HTTP transport.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        from aiohttp import web

        async def handle_request(request: web.Request) -> web.Response:
            try:
                data = await request.json()
                message = self.protocol.parse_message(data)

                if isinstance(message, MCPRequest):
                    response = await self._process_request(message)
                    return web.json_response(response.to_dict())
                else:
                    return web.json_response(
                        {"error": "Expected request message"},
                        status=400,
                    )

            except Exception as e:
                logger.error(f"Error handling HTTP request: {e}")
                return web.json_response(
                    {"error": str(e)},
                    status=500,
                )

        app = web.Application()
        app.router.add_post("/mcp", handle_request)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)

        logger.info(f"Starting HTTP server on {host}:{port}")
        await site.start()

        # Keep server running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            await self.tools_registry.cleanup_all()
            await runner.cleanup()


def create_server(
    name: str = "effgen-tools",
    version: str = "1.0.0",
    tools_registry: Optional[ToolRegistry] = None,
) -> MCPServer:
    """
    Create an MCP server instance.

    Args:
        name: Server name
        version: Server version
        tools_registry: Tool registry to expose

    Returns:
        MCP server instance
    """
    return MCPServer(
        name=name,
        version=version,
        tools_registry=tools_registry,
    )


async def main_stdio():
    """Main entry point for STDIO server."""
    server = create_server()
    await server.run_stdio()


async def main_http(host: str = "127.0.0.1", port: int = 8000):
    """Main entry point for HTTP server."""
    server = create_server()
    await server.run_http(host, port)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "http":
        asyncio.run(main_http())
    else:
        asyncio.run(main_stdio())
