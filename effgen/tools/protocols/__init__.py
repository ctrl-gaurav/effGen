"""
Protocol integrations for effGen.

This package provides implementations for various agent communication protocols:
- MCP (Model Context Protocol): Anthropic's protocol for context and tool sharing
  - mcp: Custom implementation (legacy)
  - mcp_official: Official MCP SDK implementation (recommended)
- A2A (Agent-to-Agent): Google's protocol for agent communication
- ACP (Agent Communication Protocol): IBM's protocol for agent interoperability
"""

from . import mcp
from . import mcp_official
from . import a2a
from . import acp


__all__ = [
    "mcp",
    "mcp_official",
    "a2a",
    "acp",
]
