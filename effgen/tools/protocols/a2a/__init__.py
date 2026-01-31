"""
Agent-to-Agent (A2A) Protocol Integration.

This package implements Google's A2A protocol for agent-to-agent communication,
providing message protocol handling, task lifecycle management, and context passing.
"""

from .protocol import (
    A2AProtocolHandler,
    Task,
    TaskRequest,
    TaskUpdate,
    TaskState,
    A2AMessage,
    A2AError,
    ErrorCode,
    Artifact,
    MessagePart,
    MessagePartType,
    A2AVersion,
)

from .agent_card import (
    AgentCard,
    Capability,
    CapabilityType,
    EndpointConfig,
    AuthScheme,
)

from .client import (
    A2AClient,
    A2AClientConfig,
    AuthHandler,
    BearerAuthHandler,
    OAuth2AuthHandler,
    APIKeyAuthHandler,
    discover_agents,
)


__all__ = [
    # Protocol
    "A2AProtocolHandler",
    "Task",
    "TaskRequest",
    "TaskUpdate",
    "TaskState",
    "A2AMessage",
    "A2AError",
    "ErrorCode",
    "Artifact",
    "MessagePart",
    "MessagePartType",
    "A2AVersion",
    # Agent Card
    "AgentCard",
    "Capability",
    "CapabilityType",
    "EndpointConfig",
    "AuthScheme",
    # Client
    "A2AClient",
    "A2AClientConfig",
    "AuthHandler",
    "BearerAuthHandler",
    "OAuth2AuthHandler",
    "APIKeyAuthHandler",
    "discover_agents",
]
