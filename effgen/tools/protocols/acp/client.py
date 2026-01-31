"""
ACP client implementation for connecting to ACP-compatible agents.

This module provides a client for connecting to agents using IBM's Agent
Communication Protocol (ACP), including discovery service integration,
capability tokens, and BeeAI platform support.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Callable
from dataclasses import dataclass
import httpx
from datetime import datetime, timedelta

from .protocol import (
    ACPProtocolHandler,
    AgentManifest,
    ACPRequest,
    ACPResponse,
    ACPError,
    TaskInfo,
    TaskStatus,
    RequestType,
    CapabilityToken,
    ErrorSeverity,
    SchemaDefinition,
    CapabilityDefinition,
)


logger = logging.getLogger(__name__)


@dataclass
class ACPClientConfig:
    """
    Configuration for ACP client.

    Attributes:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        verify_ssl: Whether to verify SSL certificates
        headers: Additional HTTP headers
        discovery_url: URL for agent discovery service
        enable_telemetry: Enable OpenTelemetry instrumentation
    """
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    headers: Dict[str, str] = None
    discovery_url: Optional[str] = None
    enable_telemetry: bool = True

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class ACPAuthHandler:
    """Base class for ACP authentication handlers."""

    async def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply authentication to request headers.

        Args:
            headers: Request headers

        Returns:
            Updated headers with authentication
        """
        return headers


class TokenAuthHandler(ACPAuthHandler):
    """Capability token authentication handler."""

    def __init__(self, token: CapabilityToken):
        """
        Initialize token auth handler.

        Args:
            token: Capability token
        """
        self.token = token

    async def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply token authentication."""
        if not self.token.is_valid():
            raise RuntimeError("Token has expired")
        headers["X-Capability-Token"] = self.token.tokenId
        return headers


class APIKeyAuthHandler(ACPAuthHandler):
    """API key authentication handler."""

    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        """
        Initialize API key auth handler.

        Args:
            api_key: API key
            header_name: Header name for API key
        """
        self.api_key = api_key
        self.header_name = header_name

    async def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply API key authentication."""
        headers[self.header_name] = self.api_key
        return headers


class BearerAuthHandler(ACPAuthHandler):
    """Bearer token authentication handler."""

    def __init__(self, token: str):
        """
        Initialize bearer auth handler.

        Args:
            token: Bearer token
        """
        self.token = token

    async def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply bearer authentication."""
        headers["Authorization"] = f"Bearer {self.token}"
        return headers


class ACPClient:
    """
    Client for connecting to ACP-compatible agents.

    Features:
    - Synchronous and asynchronous request handling
    - Streaming response support
    - Capability token management
    - Agent discovery integration
    - BeeAI platform support
    - OpenTelemetry instrumentation
    - Automatic retries with exponential backoff
    """

    def __init__(
        self,
        agent_url: str,
        config: Optional[ACPClientConfig] = None,
        auth_handler: Optional[ACPAuthHandler] = None,
        manifest: Optional[AgentManifest] = None,
    ):
        """
        Initialize ACP client.

        Args:
            agent_url: Agent base URL
            config: Client configuration
            auth_handler: Authentication handler
            manifest: Agent manifest (will be fetched if not provided)
        """
        self.agent_url = agent_url.rstrip("/")
        self.config = config or ACPClientConfig()
        self.auth_handler = auth_handler
        self.manifest = manifest
        self.http_client: Optional[httpx.AsyncClient] = None
        self._connected = False
        self._tracer = None

        # Initialize OpenTelemetry if enabled
        if self.config.enable_telemetry:
            try:
                from opentelemetry import trace
                self._tracer = trace.get_tracer(__name__)
            except ImportError:
                logger.warning("OpenTelemetry not available, telemetry disabled")
                self._tracer = None

    async def connect(self) -> None:
        """
        Connect to the ACP agent.

        Fetches agent manifest and validates connectivity.
        """
        # Create HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )

        # Fetch manifest if not provided
        if not self.manifest:
            self.manifest = await self.fetch_manifest()
            logger.info(f"Fetched manifest for agent: {self.manifest.name}")

        # Verify agent is reachable
        try:
            response = await self.http_client.get(f"{self.agent_url}/health")
            if response.status_code != 200:
                logger.warning(f"Agent health check returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Agent health check failed: {e}")

        self._connected = True
        logger.info(f"Connected to ACP agent: {self.manifest.name} at {self.agent_url}")

    async def disconnect(self) -> None:
        """Disconnect from the ACP agent."""
        if self.http_client:
            await self.http_client.aclose()
        self._connected = False
        logger.info(f"Disconnected from ACP agent: {self.manifest.name}")

    async def fetch_manifest(self) -> AgentManifest:
        """
        Fetch agent manifest from the agent.

        Returns:
            Agent manifest

        Raises:
            RuntimeError: If manifest fetch fails
        """
        try:
            headers = dict(self.config.headers)
            if self.auth_handler:
                headers = await self.auth_handler.apply_auth(headers)

            response = await self.http_client.get(
                f"{self.agent_url}/manifest",
                headers=headers,
            )
            response.raise_for_status()

            manifest_data = response.json()
            return AgentManifest.from_dict(manifest_data)

        except Exception as e:
            raise RuntimeError(f"Failed to fetch agent manifest: {e}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """
        Make an HTTP request with authentication and retries.

        Args:
            method: HTTP method
            endpoint: Endpoint path
            data: Request body data
            params: Query parameters

        Returns:
            HTTP response

        Raises:
            RuntimeError: If client is not connected or request fails
        """
        if not self.http_client or not self._connected:
            raise RuntimeError("Client not connected")

        url = f"{self.agent_url}{endpoint}"
        headers = dict(self.config.headers)
        headers["Content-Type"] = "application/json"

        # Apply authentication
        if self.auth_handler:
            headers = await self.auth_handler.apply_auth(headers)

        # Add OpenTelemetry context if enabled
        if self._tracer:
            try:
                from opentelemetry.propagate import inject
                inject(headers)
            except ImportError:
                pass

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                # Start span if telemetry enabled
                span_context = None
                if self._tracer:
                    span_context = self._tracer.start_span(
                        f"acp.request.{method.lower()}",
                        attributes={
                            "http.method": method,
                            "http.url": url,
                            "acp.agent": self.manifest.name if self.manifest else "unknown",
                        }
                    )

                try:
                    response = await self.http_client.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers=headers,
                    )
                    response.raise_for_status()

                    if span_context:
                        span_context.set_attribute("http.status_code", response.status_code)
                        span_context.end()

                    return response

                except Exception as e:
                    if span_context:
                        span_context.set_attribute("error", True)
                        span_context.record_exception(e)
                        span_context.end()
                    raise

            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise RuntimeError(f"Request failed: {e}")

                # Retry server errors (5xx) with backoff
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)

        raise RuntimeError(f"Request failed after {self.config.max_retries} attempts: {last_error}")

    async def execute_sync(
        self,
        capability: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ACPResponse:
        """
        Execute a synchronous request.

        Args:
            capability: Requested capability
            input_data: Request input
            context: Request context

        Returns:
            ACP response

        Raises:
            ValueError: If capability is not supported
            RuntimeError: If request fails
        """
        # Validate capability
        if not self.manifest.get_capability(capability):
            raise ValueError(
                f"Agent {self.manifest.name} does not support capability: {capability}"
            )

        # Create request
        request = ACPRequest(
            capability=capability,
            input=input_data,
            agentId=self.manifest.agentId,
            requestType=RequestType.SYNCHRONOUS,
            context=context or {},
        )

        # Send request
        response = await self._make_request(
            method="POST",
            endpoint="/execute",
            data=request.to_dict(),
        )

        # Parse response
        response_data = response.json()
        acp_response = ACPResponse.from_dict(response_data)

        logger.info(
            f"Executed sync request {request.requestId}: status={acp_response.status.value}"
        )
        return acp_response

    async def execute_async(
        self,
        capability: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskInfo:
        """
        Execute an asynchronous request.

        Args:
            capability: Requested capability
            input_data: Request input
            context: Request context

        Returns:
            Task information

        Raises:
            ValueError: If capability is not supported
            RuntimeError: If request fails
        """
        # Validate capability
        if not self.manifest.get_capability(capability):
            raise ValueError(
                f"Agent {self.manifest.name} does not support capability: {capability}"
            )

        # Create request
        request = ACPRequest(
            capability=capability,
            input=input_data,
            agentId=self.manifest.agentId,
            requestType=RequestType.ASYNCHRONOUS,
            context=context or {},
        )

        # Send request
        response = await self._make_request(
            method="POST",
            endpoint="/execute",
            data=request.to_dict(),
        )

        # Parse response
        task_data = response.json()
        task = TaskInfo.from_dict(task_data)

        logger.info(f"Created async task {task.taskId} for request {request.requestId}")
        return task

    async def get_task_status(self, task_id: str) -> TaskInfo:
        """
        Get task status.

        Args:
            task_id: Task identifier

        Returns:
            Task information

        Raises:
            RuntimeError: If task not found or request fails
        """
        response = await self._make_request(
            method="GET",
            endpoint=f"/tasks/{task_id}",
        )

        task_data = response.json()
        return TaskInfo.from_dict(task_data)

    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> TaskInfo:
        """
        Wait for task to complete.

        Args:
            task_id: Task identifier
            poll_interval: Polling interval in seconds
            timeout: Timeout in seconds

        Returns:
            Completed task

        Raises:
            TimeoutError: If task doesn't complete within timeout
            RuntimeError: If task fails
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            task = await self.get_task_status(task_id)

            # Check if task is complete
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.status == TaskStatus.COMPLETED:
                    logger.info(f"Task {task_id} completed successfully")
                    return task
                else:
                    error_msg = task.error.message if task.error else "Unknown error"
                    raise RuntimeError(f"Task {task_id} failed: {error_msg}")

            # Check timeout
            if timeout:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

            # Wait before polling again
            await asyncio.sleep(poll_interval)

    async def stream_task_updates(
        self, task_id: str
    ) -> AsyncIterator[TaskInfo]:
        """
        Stream task updates using Server-Sent Events.

        Args:
            task_id: Task identifier

        Yields:
            Task updates

        Raises:
            RuntimeError: If streaming is not supported or fails
        """
        if not self.http_client:
            raise RuntimeError("Client not connected")

        url = f"{self.agent_url}/tasks/{task_id}/stream"
        headers = dict(self.config.headers)

        # Apply authentication
        if self.auth_handler:
            headers = await self.auth_handler.apply_auth(headers)

        async with self.http_client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield TaskInfo.from_dict(data)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled

        Raises:
            RuntimeError: If cancellation fails
        """
        try:
            await self._make_request(
                method="DELETE",
                endpoint=f"/tasks/{task_id}",
            )
            logger.info(f"Cancelled task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    async def list_capabilities(self) -> List[CapabilityDefinition]:
        """
        List available capabilities.

        Returns:
            List of capability definitions
        """
        return self.manifest.capabilities

    async def get_capability(self, name: str) -> Optional[CapabilityDefinition]:
        """
        Get a capability by name.

        Args:
            name: Capability name

        Returns:
            Capability definition if found
        """
        return self.manifest.get_capability(name)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class ACPDiscoveryClient:
    """
    Client for ACP agent discovery service.

    Supports discovering agents from local registries, BeeAI platform,
    and custom discovery services.
    """

    def __init__(
        self,
        discovery_url: str,
        auth_handler: Optional[ACPAuthHandler] = None,
        timeout: int = 10,
    ):
        """
        Initialize discovery client.

        Args:
            discovery_url: Discovery service URL
            auth_handler: Authentication handler
            timeout: Request timeout
        """
        self.discovery_url = discovery_url.rstrip("/")
        self.auth_handler = auth_handler
        self.timeout = timeout

    async def discover_agents(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AgentManifest]:
        """
        Discover agents from the discovery service.

        Args:
            filters: Optional filters (capabilities, tags, etc.)

        Returns:
            List of discovered agent manifests

        Raises:
            RuntimeError: If discovery fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                headers = {}
                if self.auth_handler:
                    headers = await self.auth_handler.apply_auth(headers)

                response = await client.get(
                    f"{self.discovery_url}/agents",
                    params=filters or {},
                    headers=headers,
                )
                response.raise_for_status()

                agents_data = response.json()
                return [
                    AgentManifest.from_dict(agent)
                    for agent in agents_data.get("agents", [])
                ]

            except Exception as e:
                raise RuntimeError(f"Agent discovery failed: {e}")

    async def register_agent(
        self,
        manifest: AgentManifest,
        endpoint: str,
    ) -> bool:
        """
        Register an agent with the discovery service.

        Args:
            manifest: Agent manifest
            endpoint: Agent endpoint URL

        Returns:
            True if registration successful

        Raises:
            RuntimeError: If registration fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                headers = {"Content-Type": "application/json"}
                if self.auth_handler:
                    headers = await self.auth_handler.apply_auth(headers)

                data = manifest.to_dict()
                data["endpoint"] = endpoint

                response = await client.post(
                    f"{self.discovery_url}/agents",
                    json=data,
                    headers=headers,
                )
                response.raise_for_status()

                logger.info(f"Registered agent {manifest.name} with discovery service")
                return True

            except Exception as e:
                raise RuntimeError(f"Agent registration failed: {e}")

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the discovery service.

        Args:
            agent_id: Agent identifier

        Returns:
            True if unregistration successful

        Raises:
            RuntimeError: If unregistration fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                headers = {}
                if self.auth_handler:
                    headers = await self.auth_handler.apply_auth(headers)

                response = await client.delete(
                    f"{self.discovery_url}/agents/{agent_id}",
                    headers=headers,
                )
                response.raise_for_status()

                logger.info(f"Unregistered agent {agent_id} from discovery service")
                return True

            except Exception as e:
                raise RuntimeError(f"Agent unregistration failed: {e}")

    async def find_by_capability(
        self, capability: str
    ) -> List[AgentManifest]:
        """
        Find agents that support a specific capability.

        Args:
            capability: Capability name

        Returns:
            List of agent manifests
        """
        return await self.discover_agents(filters={"capability": capability})

    async def find_by_tags(
        self, tags: List[str]
    ) -> List[AgentManifest]:
        """
        Find agents by tags.

        Args:
            tags: List of tags

        Returns:
            List of agent manifests
        """
        return await self.discover_agents(filters={"tags": ",".join(tags)})


async def create_capability_token(
    agent_id: str,
    capabilities: List[str],
    expires_in: Optional[int] = None,
    permissions: Optional[Dict[str, List[str]]] = None,
) -> CapabilityToken:
    """
    Create a capability token for an agent.

    Args:
        agent_id: Agent identifier
        capabilities: List of allowed capabilities
        expires_in: Expiration time in seconds
        permissions: Permission levels per capability

    Returns:
        Capability token
    """
    import uuid

    token_id = str(uuid.uuid4())
    expires = None
    if expires_in:
        expires = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()

    return CapabilityToken(
        tokenId=token_id,
        agentId=agent_id,
        capabilities=capabilities,
        permissions=permissions or {},
        expires=expires,
    )
