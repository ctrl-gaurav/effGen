"""
Model Context Protocol (MCP) protocol handler.

This module implements the MCP protocol specification for tool and resource
communication between AI models and external servers.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime
import json


logger = logging.getLogger(__name__)


class MCPVersion(Enum):
    """Supported MCP protocol versions."""
    V1_0 = "1.0"


class MessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class TransportType(Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    WEBSOCKET = "websocket"


class ErrorCode(Enum):
    """MCP error codes based on JSON-RPC 2.0."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000


@dataclass
class MCPError:
    """MCP error representation."""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "code": self.code,
            "message": self.message,
        }
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class MCPToolParameter:
    """MCP tool parameter specification."""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        return result


@dataclass
class MCPTool:
    """MCP tool specification."""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    returnSchema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }
        if self.returnSchema:
            result["returnSchema"] = self.returnSchema
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            inputSchema=data["inputSchema"],
            returnSchema=data.get("returnSchema"),
        )


@dataclass
class MCPResource:
    """MCP resource specification."""
    uri: str
    name: str
    description: str
    mimeType: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
        }
        if self.mimeType:
            result["mimeType"] = self.mimeType
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResource":
        """Create from dictionary."""
        return cls(
            uri=data["uri"],
            name=data["name"],
            description=data["description"],
            mimeType=data.get("mimeType"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MCPCapabilities:
    """MCP server/client capabilities."""
    tools: bool = False
    resources: bool = False
    prompts: bool = False
    sampling: bool = False
    experimental: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts,
            "sampling": self.sampling,
            "experimental": self.experimental,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPCapabilities":
        """Create from dictionary."""
        return cls(
            tools=data.get("tools", False),
            resources=data.get("resources", False),
            prompts=data.get("prompts", False),
            sampling=data.get("sampling", False),
            experimental=data.get("experimental", {}),
        )


@dataclass
class MCPMessage:
    """Base MCP message following JSON-RPC 2.0."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MCPRequest(MCPMessage):
    """MCP request message."""
    method: str = ""
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.id is not None:
            result["id"] = self.id
        if self.params:
            result["params"] = self.params
        return result


@dataclass
class MCPResponse(MCPMessage):
    """MCP response message."""
    result: Optional[Any] = None
    error: Optional[MCPError] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error:
            result["error"] = self.error.to_dict()
        else:
            result["result"] = self.result
        return result


@dataclass
class MCPNotification(MCPMessage):
    """MCP notification message (no response expected)."""
    method: str = ""
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params:
            result["params"] = self.params
        return result


class MCPProtocolHandler:
    """
    Handler for MCP protocol messages.

    Implements JSON-RPC 2.0 message handling for MCP communication.
    Supports requests, responses, notifications, and errors.
    """

    def __init__(self, version: str = "1.0"):
        """
        Initialize protocol handler.

        Args:
            version: MCP protocol version
        """
        self.version = version
        self._request_id = 0

    def create_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[Union[str, int]] = None,
    ) -> MCPRequest:
        """
        Create an MCP request message.

        Args:
            method: Method name
            params: Method parameters
            request_id: Request ID (auto-generated if None)

        Returns:
            MCPRequest message
        """
        if request_id is None:
            self._request_id += 1
            request_id = self._request_id

        return MCPRequest(
            method=method,
            params=params,
            id=request_id,
        )

    def create_response(
        self,
        request_id: Union[str, int],
        result: Optional[Any] = None,
        error: Optional[MCPError] = None,
    ) -> MCPResponse:
        """
        Create an MCP response message.

        Args:
            request_id: ID of the request being responded to
            result: Result data (if successful)
            error: Error data (if failed)

        Returns:
            MCPResponse message
        """
        return MCPResponse(
            id=request_id,
            result=result,
            error=error,
        )

    def create_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> MCPNotification:
        """
        Create an MCP notification message.

        Args:
            method: Method name
            params: Method parameters

        Returns:
            MCPNotification message
        """
        return MCPNotification(
            method=method,
            params=params,
        )

    def create_error(
        self,
        code: ErrorCode,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> MCPError:
        """
        Create an MCP error.

        Args:
            code: Error code
            message: Error message
            data: Additional error data

        Returns:
            MCPError
        """
        return MCPError(
            code=code.value,
            message=message,
            data=data,
        )

    def parse_message(self, data: Union[str, bytes, Dict[str, Any]]) -> Union[MCPRequest, MCPResponse, MCPNotification]:
        """
        Parse MCP message from JSON data.

        Args:
            data: JSON string, bytes, or dict

        Returns:
            Parsed MCP message

        Raises:
            ValueError: If message is invalid
        """
        # Parse JSON if needed
        if isinstance(data, (str, bytes)):
            try:
                message_dict = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        else:
            message_dict = data

        # Validate JSON-RPC version
        if message_dict.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC version")

        # Determine message type
        if "method" in message_dict:
            if "id" in message_dict:
                # Request
                return MCPRequest(
                    method=message_dict["method"],
                    params=message_dict.get("params"),
                    id=message_dict["id"],
                )
            else:
                # Notification
                return MCPNotification(
                    method=message_dict["method"],
                    params=message_dict.get("params"),
                )
        elif "result" in message_dict or "error" in message_dict:
            # Response
            error = None
            if "error" in message_dict:
                error_data = message_dict["error"]
                error = MCPError(
                    code=error_data["code"],
                    message=error_data["message"],
                    data=error_data.get("data"),
                )

            return MCPResponse(
                id=message_dict.get("id"),
                result=message_dict.get("result"),
                error=error,
            )
        else:
            raise ValueError("Invalid MCP message format")

    def serialize_message(self, message: Union[MCPRequest, MCPResponse, MCPNotification]) -> str:
        """
        Serialize MCP message to JSON string.

        Args:
            message: MCP message

        Returns:
            JSON string
        """
        return json.dumps(message.to_dict())

    def validate_tool_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Validate MCP tool schema.

        Args:
            schema: Tool schema to validate

        Returns:
            True if valid

        Raises:
            ValueError: If schema is invalid
        """
        required_fields = ["name", "description", "inputSchema"]
        for field in required_fields:
            if field not in schema:
                raise ValueError(f"Missing required field: {field}")

        # Validate input schema is valid JSON Schema
        input_schema = schema["inputSchema"]
        if not isinstance(input_schema, dict):
            raise ValueError("inputSchema must be a dictionary")

        if "type" not in input_schema:
            raise ValueError("inputSchema must have a 'type' field")

        return True

    def create_initialize_request(
        self,
        protocol_version: str,
        capabilities: MCPCapabilities,
        client_info: Dict[str, str],
    ) -> MCPRequest:
        """
        Create MCP initialize request.

        Args:
            protocol_version: MCP protocol version
            capabilities: Client capabilities
            client_info: Client information (name, version)

        Returns:
            Initialize request
        """
        return self.create_request(
            method="initialize",
            params={
                "protocolVersion": protocol_version,
                "capabilities": capabilities.to_dict(),
                "clientInfo": client_info,
            },
        )

    def create_tools_list_request(self) -> MCPRequest:
        """Create request to list available tools."""
        return self.create_request(method="tools/list")

    def create_tool_call_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MCPRequest:
        """
        Create request to call a tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool call request
        """
        return self.create_request(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments,
            },
        )

    def create_resources_list_request(self) -> MCPRequest:
        """Create request to list available resources."""
        return self.create_request(method="resources/list")

    def create_resource_read_request(self, uri: str) -> MCPRequest:
        """
        Create request to read a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource read request
        """
        return self.create_request(
            method="resources/read",
            params={"uri": uri},
        )
