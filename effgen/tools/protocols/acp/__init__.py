"""
Agent Communication Protocol (ACP) Integration.

This package implements IBM's ACP protocol for agent-to-agent communication,
providing agent manifests, synchronous/asynchronous requests, task tracking,
and OpenTelemetry instrumentation.
"""

from .protocol import (
    ACPProtocolHandler,
    AgentManifest,
    ACPRequest,
    ACPResponse,
    ACPError,
    TaskInfo,
    TaskStatus,
    RequestType,
    ErrorSeverity,
    SchemaDefinition,
    CapabilityDefinition,
    CapabilityToken,
    ACPVersion,
)

from .client import (
    ACPClient,
    ACPClientConfig,
    ACPAuthHandler,
    TokenAuthHandler,
    APIKeyAuthHandler,
    BearerAuthHandler,
    ACPDiscoveryClient,
    create_capability_token,
)

from .server import (
    ACPServer,
    ACPServerConfig,
    ACPCapabilityRegistry,
    capability,
)


__all__ = [
    # Protocol
    "ACPProtocolHandler",
    "AgentManifest",
    "ACPRequest",
    "ACPResponse",
    "ACPError",
    "TaskInfo",
    "TaskStatus",
    "RequestType",
    "ErrorSeverity",
    "SchemaDefinition",
    "CapabilityDefinition",
    "CapabilityToken",
    "ACPVersion",
    # Client
    "ACPClient",
    "ACPClientConfig",
    "ACPAuthHandler",
    "TokenAuthHandler",
    "APIKeyAuthHandler",
    "BearerAuthHandler",
    "ACPDiscoveryClient",
    "create_capability_token",
    # Server
    "ACPServer",
    "ACPServerConfig",
    "ACPCapabilityRegistry",
    "capability",
]
