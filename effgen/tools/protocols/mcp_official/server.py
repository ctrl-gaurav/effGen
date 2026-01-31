"""
Official MCP Server implementation using the mcp library.

This module provides a standards-compliant MCP server that exposes effGen
tools using the official Model Context Protocol Python SDK.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.types import CallToolResult, TextContent

from ...base_tool import BaseTool, ToolResult
from ...registry import ToolRegistry


logger = logging.getLogger(__name__)


@dataclass
class EffGenMCPServerConfig:
    """Configuration for effGen MCP server."""
    name: str = "effgen-tools"
    version: str = "1.0.0"
    instructions: str = "MCP server exposing effGen tools for AI agents"
    tools_registry: Optional[ToolRegistry] = None


class EffGenMCPServer:
    """
    Official MCP server implementation for effGen.
    
    This server uses the official MCP Python SDK (FastMCP) to expose
    effGen tools in a standards-compliant way.
    
    Features:
    - Standards-compliant MCP protocol
    - Automatic tool registration and discovery
    - Structured output support
    - Progress reporting
    - Error handling
    - STDIO and HTTP transports
    
    Example:
        ```python
        from effgen.tools.registry import ToolRegistry
        from effgen.tools.protocols.mcp_official import EffGenMCPServer
        
        # Create registry and register tools
        registry = ToolRegistry()
        registry.discover_builtin_tools()
        
        # Create and run server
        server = EffGenMCPServer(tools_registry=registry)
        await server.run_stdio()
        ```
    """
    
    def __init__(
        self,
        name: str = "effgen-tools",
        version: str = "1.0.0",
        instructions: Optional[str] = None,
        tools_registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize EffGen MCP server.
        
        Args:
            name: Server name
            version: Server version
            instructions: Server instructions for clients
            tools_registry: Tool registry to expose (creates new if None)
        """
        self.name = name
        self.version = version
        self.instructions = instructions or (
            "MCP server exposing effGen tools for AI agents. "
            "Provides access to various tools for code execution, "
            "web search, data processing, and more."
        )
        self.tools_registry = tools_registry or ToolRegistry()
        self._initialized = False
        
        # Create FastMCP instance
        self.mcp = FastMCP(
            name=self.name,
            instructions=self.instructions,
        )
        
        logger.info(f"Initialized EffGen MCP Server: {self.name} v{self.version}")
    
    async def initialize(self) -> None:
        """Initialize the server and register tools."""
        if self._initialized:
            return
        
        # Discover and initialize builtin tools
        self.tools_registry.discover_builtin_tools()
        await self.tools_registry.initialize_all()
        
        # Register all tools with MCP
        await self._register_all_tools()
        
        self._initialized = True
        logger.info(f"Server initialized with {len(self.tools_registry.list_tools())} tools")
    
    async def _register_all_tools(self) -> None:
        """Register all tools from the registry with MCP."""
        for tool_name in self.tools_registry.list_tools():
            await self._register_tool(tool_name)
    
    async def _register_tool(self, tool_name: str) -> None:
        """
        Register a single tool with MCP.
        
        Args:
            tool_name: Name of the tool to register
        """
        try:
            metadata = self.tools_registry.get_metadata(tool_name)
            
            # Build input schema from tool parameters
            properties = {}
            required = []
            
            for param in metadata.parameters:
                prop = {
                    "type": param.type.value,
                    "description": param.description or "",
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
            
            # Create tool function wrapper
            async def tool_wrapper(
                ctx: Context[ServerSession, None], 
                **kwargs
            ) -> Dict[str, Any]:
                """Wrapper function for tool execution."""
                # Log tool execution
                await ctx.info(f"Executing tool: {tool_name}")
                
                try:
                    # Get tool instance
                    tool = await self.tools_registry.get_tool(tool_name)
                    
                    # Execute tool
                    result = await tool.execute(**kwargs)
                    
                    # Check if execution was successful
                    if result.success:
                        await ctx.debug(f"Tool {tool_name} executed successfully")
                        
                        # Return structured output
                        return {
                            "success": True,
                            "output": result.output,
                            "metadata": result.metadata or {},
                        }
                    else:
                        error_msg = result.error or "Tool execution failed"
                        await ctx.error(f"Tool {tool_name} failed: {error_msg}")
                        
                        return {
                            "success": False,
                            "error": error_msg,
                            "metadata": result.metadata or {},
                        }
                
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    await ctx.error(error_msg)
                    logger.error(error_msg, exc_info=True)
                    
                    return {
                        "success": False,
                        "error": error_msg,
                    }
            
            # Set function metadata
            tool_wrapper.__name__ = tool_name
            tool_wrapper.__doc__ = metadata.description or f"Tool: {tool_name}"
            
            # Add annotations for parameter types
            annotations = {}
            for param in metadata.parameters:
                # Map parameter types to Python types
                if param.type.value == "string":
                    annotations[param.name] = str
                elif param.type.value == "integer":
                    annotations[param.name] = int
                elif param.type.value == "number":
                    annotations[param.name] = float
                elif param.type.value == "boolean":
                    annotations[param.name] = bool
                elif param.type.value == "array":
                    annotations[param.name] = list
                elif param.type.value == "object":
                    annotations[param.name] = dict
                else:
                    annotations[param.name] = Any
            
            annotations["return"] = Dict[str, Any]
            tool_wrapper.__annotations__ = annotations
            
            # Register with FastMCP
            self.mcp.tool()(tool_wrapper)
            
            logger.debug(f"Registered tool: {tool_name}")
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_name}: {e}", exc_info=True)
    
    async def run_stdio(self) -> None:
        """
        Run server with STDIO transport.
        
        This is the main entry point for Claude Desktop and other
        STDIO-based MCP clients.
        """
        try:
            # Initialize if not already done
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Starting MCP server (STDIO): {self.name}")
            
            # Run FastMCP with STDIO transport
            self.mcp.run(transport="stdio")
            
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()
    
    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        transport: str = "streamable-http"
    ) -> None:
        """
        Run server with HTTP transport.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            transport: Transport type ("streamable-http" or "sse")
        """
        try:
            # Initialize if not already done
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Starting MCP server ({transport}): {host}:{port}")
            
            # Run FastMCP with HTTP transport
            self.mcp.run(
                transport=transport,
                host=host,
                port=port
            )
            
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.tools_registry.cleanup_all()
            logger.info("Server cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def get_mcp_instance(self) -> FastMCP:
        """
        Get the underlying FastMCP instance.
        
        This allows for advanced customization and direct access
        to the MCP server.
        
        Returns:
            FastMCP instance
        """
        return self.mcp


def create_server(
    name: str = "effgen-tools",
    version: str = "1.0.0",
    tools_registry: Optional[ToolRegistry] = None,
) -> EffGenMCPServer:
    """
    Create an EffGen MCP server instance.
    
    Args:
        name: Server name
        version: Server version
        tools_registry: Tool registry to expose
    
    Returns:
        EffGen MCP server instance
    """
    return EffGenMCPServer(
        name=name,
        version=version,
        tools_registry=tools_registry,
    )


async def main_stdio():
    """Main entry point for STDIO server."""
    server = create_server()
    await server.run_stdio()


async def main_http(
    host: str = "127.0.0.1",
    port: int = 8000,
    transport: str = "streamable-http"
):
    """Main entry point for HTTP server."""
    server = create_server()
    await server.run_http(host, port, transport)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "http":
            host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
            port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
            asyncio.run(main_http(host, port))
        elif sys.argv[1] == "sse":
            host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
            port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
            asyncio.run(main_http(host, port, transport="sse"))
        else:
            print("Usage: python server.py [stdio|http|sse] [host] [port]")
            sys.exit(1)
    else:
        # Default to STDIO
        asyncio.run(main_stdio())

