"""
Tools module for the effGen framework.

This module provides the tool integration system including base classes,
registry, built-in tools, and protocol implementations.
"""

from .base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ToolResult,
    ParameterSpec,
    ParameterType,
)

from .registry import (
    ToolRegistry,
    ToolDependencyError,
    ToolRegistrationError,
    get_registry,
    reset_registry,
)

# Import protocol submodules
from . import protocols


__all__ = [
    # Base classes
    "BaseTool",
    "ToolMetadata",
    "ToolCategory",
    "ToolResult",
    "ParameterSpec",
    "ParameterType",
    # Registry
    "ToolRegistry",
    "ToolDependencyError",
    "ToolRegistrationError",
    "get_registry",
    "reset_registry",
    # Protocols
    "protocols",
]
