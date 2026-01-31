"""
Tool registry for the effGen framework.

This module provides a central registry for tool registration, discovery,
lazy loading, and dependency management.
"""

import importlib
import inspect
from typing import Dict, List, Optional, Type, Any, Set
from pathlib import Path
import logging
from collections import defaultdict
import asyncio

from .base_tool import BaseTool, ToolCategory, ToolMetadata


logger = logging.getLogger(__name__)


class ToolDependencyError(Exception):
    """Raised when tool dependencies cannot be satisfied."""
    pass


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""
    pass


class ToolRegistry:
    """
    Central registry for managing tools in the effGen framework.

    Features:
    - Tool registration with metadata validation
    - Lazy loading of tools
    - Dependency management and resolution
    - Tool discovery and filtering
    - Version management
    - Plugin system for external tools

    Example:
        registry = ToolRegistry()
        registry.register_tool(MyTool)
        tool = await registry.get_tool("my_tool")
        result = await tool.execute(param="value")
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instances: Dict[str, BaseTool] = {}
        self._metadata_cache: Dict[str, ToolMetadata] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._categories: Dict[ToolCategory, Set[str]] = defaultdict(set)
        self._plugins: Dict[str, Path] = {}
        self._initialized_tools: Set[str] = set()
        self._lock = asyncio.Lock()

    def register_tool(
        self,
        tool_class: Type[BaseTool],
        override: bool = False
    ) -> None:
        """
        Register a tool class with the registry.

        Args:
            tool_class: The tool class to register (must inherit from BaseTool)
            override: Whether to override existing tool with same name

        Raises:
            ToolRegistrationError: If registration fails
        """
        # Validate tool class
        if not inspect.isclass(tool_class):
            raise ToolRegistrationError(f"Expected class, got {type(tool_class)}")

        if not issubclass(tool_class, BaseTool):
            raise ToolRegistrationError(
                f"Tool class {tool_class.__name__} must inherit from BaseTool"
            )

        # Create temporary instance to get metadata
        try:
            temp_instance = tool_class()
            metadata = temp_instance.metadata
            name = metadata.name
        except Exception as e:
            raise ToolRegistrationError(
                f"Failed to instantiate tool {tool_class.__name__}: {e}"
            )

        # Check for name collision
        if name in self._tools and not override:
            # Silently skip re-registration instead of throwing error/warning
            logger.debug(f"Tool '{name}' already registered, skipping")
            return

        # Register the tool
        self._tools[name] = tool_class
        self._metadata_cache[name] = metadata
        self._categories[metadata.category].add(name)

        # Store dependencies
        if temp_instance.dependencies:
            self._dependencies[name] = set(temp_instance.dependencies)

        logger.info(f"Registered tool: {name} (v{metadata.version})")

    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool from the registry.

        Args:
            name: Name of the tool to unregister

        Raises:
            KeyError: If tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        # Clean up instance if exists
        if name in self._instances:
            asyncio.create_task(self._instances[name].cleanup())
            del self._instances[name]

        # Remove from all tracking structures
        metadata = self._metadata_cache[name]
        self._categories[metadata.category].discard(name)
        del self._tools[name]
        del self._metadata_cache[name]
        if name in self._dependencies:
            del self._dependencies[name]
        self._initialized_tools.discard(name)

        logger.info(f"Unregistered tool: {name}")

    async def get_tool(self, name: str, initialize: bool = True) -> BaseTool:
        """
        Get a tool instance by name.

        Implements lazy loading - tools are only instantiated when first requested.

        Args:
            name: Name of the tool to get
            initialize: Whether to initialize the tool if not already initialized

        Returns:
            BaseTool: The tool instance

        Raises:
            KeyError: If tool is not registered
            ToolDependencyError: If tool dependencies cannot be satisfied
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        async with self._lock:
            # Return existing instance if available
            if name in self._instances:
                tool = self._instances[name]
                if initialize and name not in self._initialized_tools:
                    await tool.initialize()
                    self._initialized_tools.add(name)
                return tool

            # Check and resolve dependencies
            if name in self._dependencies:
                await self._resolve_dependencies(name)

            # Create new instance
            tool_class = self._tools[name]
            tool = tool_class()
            self._instances[name] = tool

            # Initialize if requested
            if initialize:
                await tool.initialize()
                self._initialized_tools.add(name)

            logger.debug(f"Loaded tool: {name}")
            return tool

    async def _resolve_dependencies(self, tool_name: str) -> None:
        """
        Resolve and load tool dependencies.

        Args:
            tool_name: Name of the tool whose dependencies to resolve

        Raises:
            ToolDependencyError: If dependencies cannot be satisfied
        """
        dependencies = self._dependencies.get(tool_name, set())
        if not dependencies:
            return

        # Check all dependencies are registered
        missing = dependencies - set(self._tools.keys())
        if missing:
            raise ToolDependencyError(
                f"Tool '{tool_name}' has missing dependencies: {missing}"
            )

        # Detect circular dependencies
        visited = set()
        path = []

        def check_circular(name: str) -> None:
            if name in path:
                cycle = " -> ".join(path + [name])
                raise ToolDependencyError(f"Circular dependency detected: {cycle}")
            if name in visited:
                return

            path.append(name)
            visited.add(name)

            for dep in self._dependencies.get(name, []):
                check_circular(dep)

            path.pop()

        check_circular(tool_name)

        # Load dependencies
        for dep_name in dependencies:
            if dep_name not in self._instances:
                await self.get_tool(dep_name, initialize=True)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None,
        name_filter: Optional[str] = None
    ) -> List[str]:
        """
        List available tools with optional filtering.

        Args:
            category: Filter by tool category
            tags: Filter by tags (returns tools with ANY of these tags)
            name_filter: Filter by name substring (case-insensitive)

        Returns:
            List[str]: List of tool names matching the filters
        """
        tools = set(self._tools.keys())

        # Filter by category
        if category:
            tools &= self._categories.get(category, set())

        # Filter by tags
        if tags:
            matching = set()
            for name in tools:
                metadata = self._metadata_cache[name]
                if any(tag in metadata.tags for tag in tags):
                    matching.add(name)
            tools = matching

        # Filter by name
        if name_filter:
            filter_lower = name_filter.lower()
            tools = {name for name in tools if filter_lower in name.lower()}

        return sorted(tools)

    def get_metadata(self, name: str) -> ToolMetadata:
        """
        Get metadata for a registered tool.

        Args:
            name: Name of the tool

        Returns:
            ToolMetadata: Tool metadata

        Raises:
            KeyError: If tool is not registered
        """
        if name not in self._metadata_cache:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._metadata_cache[name]

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """
        Get metadata for all registered tools.

        Returns:
            Dict[str, ToolMetadata]: Mapping of tool names to metadata
        """
        return self._metadata_cache.copy()

    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """
        Get all tools in a specific category.

        Args:
            category: The tool category

        Returns:
            List[str]: List of tool names in the category
        """
        return sorted(self._categories.get(category, set()))

    def is_registered(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: Name of the tool

        Returns:
            bool: True if registered, False otherwise
        """
        return name in self._tools

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph for all tools.

        Returns:
            Dict[str, List[str]]: Mapping of tool names to their dependencies
        """
        return {name: sorted(deps) for name, deps in self._dependencies.items()}

    async def initialize_all(self) -> None:
        """Initialize all registered tools."""
        for name in self._tools.keys():
            if name not in self._initialized_tools:
                await self.get_tool(name, initialize=True)

    async def cleanup_all(self) -> None:
        """Clean up all initialized tools."""
        cleanup_tasks = []
        for name, tool in self._instances.items():
            if name in self._initialized_tools:
                cleanup_tasks.append(tool.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._instances.clear()
        self._initialized_tools.clear()

    def register_plugin(self, plugin_path: Path) -> None:
        """
        Register a plugin directory for external tools.

        Args:
            plugin_path: Path to the plugin directory

        Raises:
            ToolRegistrationError: If plugin loading fails
        """
        if not plugin_path.is_dir():
            raise ToolRegistrationError(f"Plugin path {plugin_path} is not a directory")

        # Try to import the plugin module
        try:
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_path.name}",
                plugin_path / "__init__.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all BaseTool subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseTool) and obj is not BaseTool:
                        self.register_tool(obj)

                self._plugins[plugin_path.name] = plugin_path
                logger.info(f"Registered plugin: {plugin_path.name}")

        except Exception as e:
            raise ToolRegistrationError(f"Failed to load plugin {plugin_path}: {e}")

    def discover_builtin_tools(self) -> None:
        """
        Automatically discover and register built-in tools.

        Scans the effgen.tools.builtin package for tool classes.
        """
        try:
            from . import builtin
            builtin_path = Path(builtin.__file__).parent

            # Import all Python files in builtin directory
            for file_path in builtin_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue

                module_name = f"effgen.tools.builtin.{file_path.stem}"
                try:
                    module = importlib.import_module(module_name)

                    # Find all BaseTool subclasses
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseTool) and
                            obj is not BaseTool and
                            obj.__module__ == module_name):
                            try:
                                self.register_tool(obj)
                            except ToolRegistrationError as e:
                                # Tool registration errors are already handled, just debug log
                                logger.debug(f"Skipping {name}: {e}")

                except ImportError as e:
                    logger.warning(f"Failed to import {module_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover builtin tools: {e}")

    def export_schemas(self, format: str = "json") -> Dict[str, Any]:
        """
        Export all tool schemas in the specified format.

        Args:
            format: Output format ("json" or "openai")

        Returns:
            Dict[str, Any]: Tool schemas
        """
        schemas = {}

        for name, metadata in self._metadata_cache.items():
            if format == "openai":
                schemas[name] = metadata.to_json_schema()
            else:
                schemas[name] = metadata.to_dict()

        return schemas

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry(tools={len(self._tools)}, initialized={len(self._initialized_tools)})>"


# Global tool registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        ToolRegistry: The global registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global tool registry (mainly for testing)."""
    global _global_registry
    if _global_registry:
        asyncio.create_task(_global_registry.cleanup_all())
    _global_registry = None
