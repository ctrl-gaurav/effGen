"""
Agent Card implementation for A2A protocol.

This module implements the Agent Card specification for capability advertisement
in the Agent-to-Agent (A2A) protocol developed by Google.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime


class AuthScheme(Enum):
    """Supported authentication schemes for A2A."""
    NONE = "none"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    MUTUAL_TLS = "mutual_tls"


class CapabilityType(Enum):
    """Types of capabilities an agent can advertise."""
    TASK_EXECUTION = "task_execution"
    INFORMATION_RETRIEVAL = "information_retrieval"
    DATA_PROCESSING = "data_processing"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    ORCHESTRATION = "orchestration"


@dataclass
class Capability:
    """
    Represents a specific capability of an agent.

    Attributes:
        name: Capability identifier
        type: Type of capability
        description: Human-readable description
        inputSchema: JSON schema for input parameters
        outputSchema: JSON schema for output format
        examples: Example usage scenarios
        metadata: Additional capability metadata
    """
    name: str
    type: CapabilityType
    description: str
    inputSchema: Dict[str, Any]
    outputSchema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary format."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }
        if self.outputSchema:
            result["outputSchema"] = self.outputSchema
        if self.examples:
            result["examples"] = self.examples
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Capability":
        """Create capability from dictionary."""
        return cls(
            name=data["name"],
            type=CapabilityType(data["type"]),
            description=data["description"],
            inputSchema=data["inputSchema"],
            outputSchema=data.get("outputSchema"),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EndpointConfig:
    """
    Configuration for agent communication endpoint.

    Attributes:
        url: Base URL for the agent endpoint
        protocol: Communication protocol (http, https, websocket)
        methods: Supported HTTP methods
        contentTypes: Supported content types
        maxPayloadSize: Maximum payload size in bytes
        streaming: Whether streaming responses are supported
    """
    url: str
    protocol: str = "https"
    methods: List[str] = field(default_factory=lambda: ["POST"])
    contentTypes: List[str] = field(default_factory=lambda: ["application/json"])
    maxPayloadSize: int = 10485760  # 10MB
    streaming: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert endpoint config to dictionary format."""
        return {
            "url": self.url,
            "protocol": self.protocol,
            "methods": self.methods,
            "contentTypes": self.contentTypes,
            "maxPayloadSize": self.maxPayloadSize,
            "streaming": self.streaming,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndpointConfig":
        """Create endpoint config from dictionary."""
        return cls(
            url=data["url"],
            protocol=data.get("protocol", "https"),
            methods=data.get("methods", ["POST"]),
            contentTypes=data.get("contentTypes", ["application/json"]),
            maxPayloadSize=data.get("maxPayloadSize", 10485760),
            streaming=data.get("streaming", False),
        )


@dataclass
class AgentCard:
    """
    Agent Card for capability advertisement in A2A protocol.

    The Agent Card is the core mechanism for agents to advertise their
    capabilities, endpoints, and requirements to other agents in the network.
    It follows Google's A2A protocol specification.

    Attributes:
        name: Unique agent identifier
        description: Human-readable agent description
        version: Agent version (semantic versioning)
        capabilities: List of agent capabilities
        endpoint: Communication endpoint configuration
        authSchemes: Supported authentication schemes
        metadata: Additional agent metadata
        created: Timestamp when card was created
        updated: Timestamp when card was last updated
        deprecated: Whether this agent version is deprecated
        tags: Tags for agent discovery and categorization
    """
    name: str
    description: str
    version: str
    capabilities: List[Capability]
    endpoint: EndpointConfig
    authSchemes: List[AuthScheme] = field(default_factory=lambda: [AuthScheme.NONE])
    metadata: Dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    deprecated: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent card to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "endpoint": self.endpoint.to_dict(),
            "authSchemes": [scheme.value for scheme in self.authSchemes],
            "metadata": self.metadata,
            "created": self.created,
            "updated": self.updated,
            "deprecated": self.deprecated,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """
        Create agent card from dictionary.

        Args:
            data: Dictionary containing agent card data

        Returns:
            AgentCard instance
        """
        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            capabilities=[
                Capability.from_dict(cap) for cap in data.get("capabilities", [])
            ],
            endpoint=EndpointConfig.from_dict(data["endpoint"]),
            authSchemes=[
                AuthScheme(scheme) for scheme in data.get("authSchemes", ["none"])
            ],
            metadata=data.get("metadata", {}),
            created=data.get("created", datetime.utcnow().isoformat()),
            updated=data.get("updated", datetime.utcnow().isoformat()),
            deprecated=data.get("deprecated", False),
            tags=data.get("tags", []),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Convert agent card to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCard":
        """
        Create agent card from JSON string.

        Args:
            json_str: JSON string containing agent card data

        Returns:
            AgentCard instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate agent card data.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate required fields
        if not self.name:
            return False, "Agent name is required"
        if not self.description:
            return False, "Agent description is required"
        if not self.version:
            return False, "Agent version is required"

        # Validate version format (semantic versioning)
        version_parts = self.version.split(".")
        if len(version_parts) != 3:
            return False, "Version must follow semantic versioning (major.minor.patch)"

        try:
            for part in version_parts:
                int(part)
        except ValueError:
            return False, "Version parts must be integers"

        # Validate capabilities
        if not self.capabilities:
            return False, "At least one capability is required"

        for cap in self.capabilities:
            if not cap.name or not cap.description:
                return False, f"Capability {cap.name} missing required fields"
            if not cap.inputSchema:
                return False, f"Capability {cap.name} missing input schema"

        # Validate endpoint
        if not self.endpoint.url:
            return False, "Endpoint URL is required"

        if not self.endpoint.url.startswith(("http://", "https://", "ws://", "wss://")):
            return False, "Endpoint URL must be a valid HTTP/WS URL"

        return True, None

    def get_capability(self, name: str) -> Optional[Capability]:
        """
        Get a capability by name.

        Args:
            name: Capability name

        Returns:
            Capability if found, None otherwise
        """
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    def has_capability(self, capability_type: CapabilityType) -> bool:
        """
        Check if agent has a specific capability type.

        Args:
            capability_type: Type of capability to check

        Returns:
            True if agent has the capability
        """
        return any(cap.type == capability_type for cap in self.capabilities)

    def supports_auth_scheme(self, scheme: AuthScheme) -> bool:
        """
        Check if agent supports a specific authentication scheme.

        Args:
            scheme: Authentication scheme to check

        Returns:
            True if agent supports the scheme
        """
        return scheme in self.authSchemes

    def update_timestamp(self) -> None:
        """Update the 'updated' timestamp to current UTC time."""
        self.updated = datetime.utcnow().isoformat()

    def add_capability(self, capability: Capability) -> None:
        """
        Add a new capability to the agent card.

        Args:
            capability: Capability to add
        """
        if not self.get_capability(capability.name):
            self.capabilities.append(capability)
            self.update_timestamp()

    def remove_capability(self, name: str) -> bool:
        """
        Remove a capability by name.

        Args:
            name: Capability name to remove

        Returns:
            True if capability was removed, False if not found
        """
        for i, cap in enumerate(self.capabilities):
            if cap.name == name:
                self.capabilities.pop(i)
                self.update_timestamp()
                return True
        return False

    def __repr__(self) -> str:
        """String representation of agent card."""
        return f"<AgentCard(name='{self.name}', version='{self.version}', capabilities={len(self.capabilities)})>"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} v{self.version}: {self.description}"
