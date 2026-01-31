"""
A2A client implementation for connecting to remote agents.

This module provides a client for initiating requests to remote agents,
task delegation, and managing agent-to-agent communication using the A2A protocol.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Callable
from dataclasses import dataclass
import httpx
from datetime import datetime

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
)
from .agent_card import AgentCard, AuthScheme


logger = logging.getLogger(__name__)


@dataclass
class A2AClientConfig:
    """
    Configuration for A2A client.

    Attributes:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay between retries in seconds
        verify_ssl: Whether to verify SSL certificates
        headers: Additional HTTP headers
    """
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class AuthHandler:
    """Base class for authentication handlers."""

    async def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply authentication to request headers.

        Args:
            headers: Request headers

        Returns:
            Updated headers with authentication
        """
        return headers


class BearerAuthHandler(AuthHandler):
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


class OAuth2AuthHandler(AuthHandler):
    """OAuth 2.1 authentication handler."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scopes: Optional[List[str]] = None,
    ):
        """
        Initialize OAuth2 auth handler.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            token_url: Token endpoint URL
            scopes: OAuth scopes
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scopes = scopes or []
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def _fetch_token(self) -> str:
        """Fetch a new OAuth token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": " ".join(self.scopes),
                },
            )
            response.raise_for_status()
            data = response.json()
            self._token = data["access_token"]
            # Store expiry if provided
            if "expires_in" in data:
                from datetime import timedelta
                self._token_expiry = datetime.utcnow() + timedelta(seconds=data["expires_in"])
            return self._token

    async def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply OAuth2 authentication."""
        # Check if token needs refresh
        if not self._token or (
            self._token_expiry and datetime.utcnow() >= self._token_expiry
        ):
            await self._fetch_token()

        headers["Authorization"] = f"Bearer {self._token}"
        return headers


class APIKeyAuthHandler(AuthHandler):
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


class A2AClient:
    """
    Client for connecting to A2A-compatible agents.

    Features:
    - Task creation and delegation
    - Progress monitoring with streaming
    - Authentication support (Bearer, OAuth 2.1, API Key)
    - Automatic retries with exponential backoff
    - Error handling and recovery
    - Agent discovery and capability matching
    """

    def __init__(
        self,
        agent_card: AgentCard,
        config: Optional[A2AClientConfig] = None,
        auth_handler: Optional[AuthHandler] = None,
    ):
        """
        Initialize A2A client.

        Args:
            agent_card: Remote agent's card
            config: Client configuration
            auth_handler: Authentication handler
        """
        self.agent_card = agent_card
        self.config = config or A2AClientConfig()
        self.auth_handler = auth_handler
        self.protocol = A2AProtocolHandler()
        self.http_client: Optional[httpx.AsyncClient] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to the remote agent."""
        # Validate agent card
        is_valid, error = self.agent_card.validate()
        if not is_valid:
            raise ValueError(f"Invalid agent card: {error}")

        # Create HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )

        # Verify agent is reachable
        try:
            response = await self.http_client.get(
                f"{self.agent_card.endpoint.url}/health"
            )
            if response.status_code != 200:
                logger.warning(f"Agent health check returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Agent health check failed: {e}")

        self._connected = True
        logger.info(f"Connected to agent: {self.agent_card.name}")

    async def disconnect(self) -> None:
        """Disconnect from the remote agent."""
        if self.http_client:
            await self.http_client.aclose()
        self._connected = False
        logger.info(f"Disconnected from agent: {self.agent_card.name}")

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

        url = f"{self.agent_card.endpoint.url}{endpoint}"
        headers = dict(self.config.headers)

        # Apply authentication
        if self.auth_handler:
            headers = await self.auth_handler.apply_auth(headers)

        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.http_client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                return response

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

    async def create_task(
        self,
        instruction: A2AMessage,
        capability: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Create a new task on the remote agent.

        Args:
            instruction: Task instruction message
            capability: Required capability
            context: Shared context
            metadata: Additional metadata

        Returns:
            Created task

        Raises:
            ValueError: If capability is not supported
            RuntimeError: If task creation fails
        """
        # Check if agent supports the capability
        if not self.agent_card.get_capability(capability):
            raise ValueError(
                f"Agent {self.agent_card.name} does not support capability: {capability}"
            )

        # Create task request
        task_request = TaskRequest(
            instruction=instruction,
            capability=capability,
            context=context or {},
            metadata=metadata or {},
        )

        # Send request
        response = await self._make_request(
            method="POST",
            endpoint="/tasks",
            data=task_request.to_dict(),
        )

        # Parse response
        task_data = response.json()
        task = Task.from_dict(task_data)

        logger.info(f"Created task {task.id} on agent {self.agent_card.name}")
        return task

    async def get_task(self, task_id: str) -> Task:
        """
        Get task status.

        Args:
            task_id: Task identifier

        Returns:
            Task

        Raises:
            RuntimeError: If task not found or request fails
        """
        response = await self._make_request(
            method="GET",
            endpoint=f"/tasks/{task_id}",
        )

        task_data = response.json()
        return Task.from_dict(task_data)

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

    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> Task:
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
            task = await self.get_task(task_id)

            # Check if task is complete
            if task.is_terminal():
                if task.is_successful():
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
    ) -> AsyncIterator[TaskUpdate]:
        """
        Stream task updates using Server-Sent Events.

        Args:
            task_id: Task identifier

        Yields:
            Task updates

        Raises:
            RuntimeError: If streaming is not supported or fails
        """
        if not self.agent_card.endpoint.streaming:
            raise RuntimeError(
                f"Agent {self.agent_card.name} does not support streaming"
            )

        if not self.http_client:
            raise RuntimeError("Client not connected")

        url = f"{self.agent_card.endpoint.url}/tasks/{task_id}/stream"
        headers = dict(self.config.headers)

        # Apply authentication
        if self.auth_handler:
            headers = await self.auth_handler.apply_auth(headers)

        async with self.http_client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield TaskUpdate.from_dict(data)

    async def execute_task(
        self,
        instruction: A2AMessage,
        capability: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> Task:
        """
        Execute a task and wait for completion.

        This is a convenience method that combines create_task and wait_for_task.

        Args:
            instruction: Task instruction
            capability: Required capability
            context: Shared context
            timeout: Timeout in seconds
            on_progress: Progress callback

        Returns:
            Completed task

        Raises:
            TimeoutError: If task times out
            RuntimeError: If task fails
        """
        # Create task
        task = await self.create_task(
            instruction=instruction,
            capability=capability,
            context=context,
        )

        # Stream updates if callback provided and streaming supported
        if on_progress and self.agent_card.endpoint.streaming:
            try:
                async for update in self.stream_task_updates(task.id):
                    if update.progress is not None:
                        on_progress(update.progress)
                    if update.state and update.state in (
                        TaskState.COMPLETED,
                        TaskState.FAILED,
                        TaskState.CANCELLED,
                    ):
                        break
            except Exception as e:
                logger.warning(f"Streaming failed, falling back to polling: {e}")

        # Wait for completion
        return await self.wait_for_task(task.id, timeout=timeout)

    async def discover_capabilities(self) -> List[str]:
        """
        Discover available capabilities from the agent.

        Returns:
            List of capability names
        """
        return [cap.name for cap in self.agent_card.capabilities]

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


async def discover_agents(
    discovery_url: str,
    filters: Optional[Dict[str, Any]] = None,
) -> List[AgentCard]:
    """
    Discover agents from a discovery service.

    Args:
        discovery_url: URL of the discovery service
        filters: Optional filters (tags, capabilities, etc.)

    Returns:
        List of discovered agent cards

    Raises:
        RuntimeError: If discovery fails
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                discovery_url,
                params=filters or {},
            )
            response.raise_for_status()

            agents_data = response.json()
            return [AgentCard.from_dict(agent) for agent in agents_data.get("agents", [])]

        except Exception as e:
            raise RuntimeError(f"Agent discovery failed: {e}")
