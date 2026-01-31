"""
Official MCP Client implementation using the mcp library.

This module provides a standards-compliant MCP client for connecting to
MCP servers and using their tools and resources.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    Tool,
    Resource,
    Prompt,
    CallToolResult,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import AnyUrl


logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""
    name: str
    transport: str = "stdio"  # "stdio", "http", or "sse"
    
    # For STDIO transport
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    
    # For HTTP/SSE transports
    url: Optional[str] = None
    
    # Common settings
    timeout: int = 30


class EffGenMCPClient:
    """
    Official MCP client implementation for effGen.
    
    This client uses the official MCP Python SDK to connect to MCP servers,
    discover tools and resources, and execute operations.
    
    Features:
    - Standards-compliant MCP protocol
    - Multiple transport support (STDIO, HTTP, SSE)
    - Tool discovery and execution
    - Resource access
    - Prompt management
    - Async context manager support
    
    Example:
        ```python
        from effgen.tools.protocols.mcp_official import EffGenMCPClient, MCPServerConfig
        
        # Configure server
        config = MCPServerConfig(
            name="my-server",
            transport="stdio",
            command="python",
            args=["server.py"]
        )
        
        # Use client
        async with EffGenMCPClient(config) as client:
            # List tools
            tools = client.get_tools()
            
            # Call a tool
            result = await client.call_tool("my_tool", {"arg": "value"})
        ```
    """
    
    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.session: Optional[ClientSession] = None
        self._context_manager = None
        self._connected = False
        
        # Tool and resource caches
        self._tools: Dict[str, Tool] = {}
        self._resources: Dict[str, Resource] = {}
        self._prompts: Dict[str, Prompt] = {}
        
        logger.info(f"Initialized MCP client for server: {config.name}")
    
    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return
        
        try:
            # Create appropriate transport context
            if self.config.transport == "stdio":
                if not self.config.command:
                    raise ValueError("Command required for STDIO transport")
                
                server_params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args or [],
                    env=self.config.env,
                )
                
                self._context_manager = stdio_client(server_params)
                
            elif self.config.transport in ("http", "streamable-http"):
                if not self.config.url:
                    raise ValueError("URL required for HTTP transport")
                
                self._context_manager = streamablehttp_client(self.config.url)
                
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")
            
            # Enter transport context
            transport_context = await self._context_manager.__aenter__()
            
            # For STDIO, unpack read/write streams
            if self.config.transport == "stdio":
                read_stream, write_stream = transport_context
            else:
                # For HTTP, unpack read/write streams and session
                read_stream, write_stream, _ = transport_context
            
            # Create session
            self.session = ClientSession(read_stream, write_stream)
            
            # Initialize session
            await self.session.initialize()
            
            # Discover capabilities
            await self._discover()
            
            self._connected = True
            logger.info(f"Connected to MCP server: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}", exc_info=True)
            await self.disconnect()
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self._connected:
            return
        
        try:
            # Exit context manager
            if self._context_manager:
                await self._context_manager.__aexit__(None, None, None)
            
            self._connected = False
            self.session = None
            logger.info(f"Disconnected from MCP server: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}", exc_info=True)
    
    async def _discover(self) -> None:
        """Discover server capabilities, tools, and resources."""
        if not self.session:
            raise RuntimeError("Not connected")
        
        try:
            # List tools
            tools_result = await self.session.list_tools()
            for tool in tools_result.tools:
                self._tools[tool.name] = tool
            
            logger.info(f"Discovered {len(self._tools)} tools")
            
            # List resources
            try:
                resources_result = await self.session.list_resources()
                for resource in resources_result.resources:
                    self._resources[str(resource.uri)] = resource
                
                logger.info(f"Discovered {len(self._resources)} resources")
            except Exception as e:
                logger.debug(f"No resources available: {e}")
            
            # List prompts
            try:
                prompts_result = await self.session.list_prompts()
                for prompt in prompts_result.prompts:
                    self._prompts[prompt.name] = prompt
                
                logger.info(f"Discovered {len(self._prompts)} prompts")
            except Exception as e:
                logger.debug(f"No prompts available: {e}")
                
        except Exception as e:
            logger.error(f"Error discovering capabilities: {e}", exc_info=True)
            raise
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> CallToolResult:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool not found
            RuntimeError: If not connected or call fails
        """
        if not self._connected or not self.session:
            raise RuntimeError("Not connected")
        
        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        try:
            result = await self.session.call_tool(
                tool_name,
                arguments=arguments or {}
            )
            
            logger.debug(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Tool call failed: {e}", exc_info=True)
            raise RuntimeError(f"Tool call failed: {e}")
    
    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource from the MCP server.
        
        Args:
            uri: Resource URI
        
        Returns:
            Resource content
        
        Raises:
            ValueError: If resource not found
            RuntimeError: If not connected or read fails
        """
        if not self._connected or not self.session:
            raise RuntimeError("Not connected")
        
        try:
            result = await self.session.read_resource(AnyUrl(uri))
            logger.debug(f"Resource {uri} read successfully")
            return result
            
        except Exception as e:
            logger.error(f"Resource read failed: {e}", exc_info=True)
            raise RuntimeError(f"Resource read failed: {e}")
    
    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Get a prompt from the MCP server.
        
        Args:
            prompt_name: Name of the prompt
            arguments: Prompt arguments
        
        Returns:
            Prompt result
        
        Raises:
            ValueError: If prompt not found
            RuntimeError: If not connected or get fails
        """
        if not self._connected or not self.session:
            raise RuntimeError("Not connected")
        
        if prompt_name not in self._prompts:
            raise ValueError(f"Prompt not found: {prompt_name}")
        
        try:
            result = await self.session.get_prompt(
                prompt_name,
                arguments=arguments or {}
            )
            
            logger.debug(f"Prompt {prompt_name} retrieved successfully")
            return result
            
        except Exception as e:
            logger.error(f"Prompt get failed: {e}", exc_info=True)
            raise RuntimeError(f"Prompt get failed: {e}")
    
    def get_tools(self) -> List[Tool]:
        """
        Get list of available tools.
        
        Returns:
            List of tool definitions
        """
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(name)
    
    def get_resources(self) -> List[Resource]:
        """
        Get list of available resources.
        
        Returns:
            List of resource definitions
        """
        return list(self._resources.values())
    
    def get_resource(self, uri: str) -> Optional[Resource]:
        """
        Get a specific resource by URI.
        
        Args:
            uri: Resource URI
        
        Returns:
            Resource definition or None if not found
        """
        return self._resources.get(uri)
    
    def get_prompts(self) -> List[Prompt]:
        """
        Get list of available prompts.
        
        Returns:
            List of prompt definitions
        """
        return list(self._prompts.values())
    
    def get_prompt_info(self, name: str) -> Optional[Prompt]:
        """
        Get a specific prompt info by name.
        
        Args:
            name: Prompt name
        
        Returns:
            Prompt definition or None if not found
        """
        return self._prompts.get(name)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


def create_client(config: MCPServerConfig) -> EffGenMCPClient:
    """
    Create an MCP client instance.
    
    Args:
        config: Server configuration
    
    Returns:
        MCP client instance
    """
    return EffGenMCPClient(config)


async def create_stdio_client(
    name: str,
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> EffGenMCPClient:
    """
    Create and connect to a STDIO MCP client.
    
    Args:
        name: Server name
        command: Command to execute
        args: Command arguments
        env: Environment variables
    
    Returns:
        Connected MCP client
    """
    config = MCPServerConfig(
        name=name,
        transport="stdio",
        command=command,
        args=args,
        env=env,
    )
    
    client = create_client(config)
    await client.connect()
    return client


async def create_http_client(
    name: str,
    url: str,
) -> EffGenMCPClient:
    """
    Create and connect to an HTTP MCP client.
    
    Args:
        name: Server name
        url: Server URL
    
    Returns:
        Connected MCP client
    """
    config = MCPServerConfig(
        name=name,
        transport="http",
        url=url,
    )
    
    client = create_client(config)
    await client.connect()
    return client

