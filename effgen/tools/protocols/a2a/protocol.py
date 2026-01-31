"""
Agent-to-Agent (A2A) Protocol handler.

This module implements Google's A2A protocol specification for agent-to-agent
communication, including task lifecycle management, message protocols, and
context passing between agents.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime
import json
import uuid


logger = logging.getLogger(__name__)


class A2AVersion(Enum):
    """Supported A2A protocol versions."""
    V1_0 = "1.0"


class TaskState(Enum):
    """A2A task lifecycle states."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING = "pending"


class MessagePartType(Enum):
    """Types of message parts in A2A protocol."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    FORM = "form"
    STRUCTURED = "structured"


class ErrorCode(Enum):
    """A2A error codes."""
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    TASK_FAILED = "task_failed"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


@dataclass
class A2AError:
    """
    A2A error representation.

    Attributes:
        code: Error code
        message: Human-readable error message
        details: Additional error details
        retryable: Whether the error is retryable
    """
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    retryable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            result["details"] = self.details
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AError":
        """Create from dictionary."""
        return cls(
            code=ErrorCode(data["code"]),
            message=data["message"],
            details=data.get("details"),
            retryable=data.get("retryable", False),
        )


@dataclass
class MessagePart:
    """
    A part of an A2A message.

    Attributes:
        type: Type of message part
        content: Content of the message part
        mimeType: MIME type for binary content
        metadata: Additional metadata
    """
    type: MessagePartType
    content: Any
    mimeType: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type.value,
            "content": self.content,
        }
        if self.mimeType:
            result["mimeType"] = self.mimeType
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessagePart":
        """Create from dictionary."""
        return cls(
            type=MessagePartType(data["type"]),
            content=data["content"],
            mimeType=data.get("mimeType"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class A2AMessage:
    """
    A2A message containing instructions and context.

    Attributes:
        id: Unique message identifier
        parts: List of message parts
        context: Shared context between agents
        metadata: Additional message metadata
        timestamp: Message creation timestamp
    """
    parts: List[MessagePart]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "parts": [part.to_dict() for part in self.parts],
            "context": self.context,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            parts=[MessagePart.from_dict(part) for part in data.get("parts", [])],
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
        )

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a text part to the message.

        Args:
            text: Text content
            metadata: Optional metadata
        """
        self.parts.append(
            MessagePart(
                type=MessagePartType.TEXT,
                content=text,
                metadata=metadata or {},
            )
        )

    def add_image(
        self, content: str, mime_type: str = "image/png", metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an image part to the message.

        Args:
            content: Base64-encoded image or URL
            mime_type: MIME type of the image
            metadata: Optional metadata
        """
        self.parts.append(
            MessagePart(
                type=MessagePartType.IMAGE,
                content=content,
                mimeType=mime_type,
                metadata=metadata or {},
            )
        )

    def add_structured(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add structured data to the message.

        Args:
            data: Structured data
            metadata: Optional metadata
        """
        self.parts.append(
            MessagePart(
                type=MessagePartType.STRUCTURED,
                content=data,
                metadata=metadata or {},
            )
        )

    def get_text_parts(self) -> List[str]:
        """Get all text parts from the message."""
        return [
            part.content
            for part in self.parts
            if part.type == MessagePartType.TEXT
        ]


@dataclass
class Artifact:
    """
    Task artifact (output/result).

    Attributes:
        id: Unique artifact identifier
        name: Artifact name
        type: Artifact type
        content: Artifact content
        mimeType: MIME type
        metadata: Additional metadata
        created: Creation timestamp
    """
    name: str
    type: str
    content: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mimeType: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "content": self.content,
            "created": self.created,
        }
        if self.mimeType:
            result["mimeType"] = self.mimeType
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            type=data["type"],
            content=data["content"],
            mimeType=data.get("mimeType"),
            metadata=data.get("metadata", {}),
            created=data.get("created", datetime.utcnow().isoformat()),
        )


@dataclass
class Task:
    """
    A2A task representation.

    Attributes:
        id: Unique task identifier
        state: Current task state
        instruction: Task instruction message
        artifacts: Task output artifacts
        progress: Task progress (0.0 to 1.0)
        error: Error information if task failed
        metadata: Additional task metadata
        created: Task creation timestamp
        updated: Last update timestamp
        completed: Task completion timestamp
    """
    instruction: A2AMessage
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = TaskState.CREATED
    artifacts: List[Artifact] = field(default_factory=list)
    progress: float = 0.0
    error: Optional[A2AError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "state": self.state.value,
            "instruction": self.instruction.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "progress": self.progress,
            "metadata": self.metadata,
            "created": self.created,
            "updated": self.updated,
        }
        if self.error:
            result["error"] = self.error.to_dict()
        if self.completed:
            result["completed"] = self.completed
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        error = None
        if "error" in data:
            error = A2AError.from_dict(data["error"])

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            state=TaskState(data.get("state", "created")),
            instruction=A2AMessage.from_dict(data["instruction"]),
            artifacts=[Artifact.from_dict(a) for a in data.get("artifacts", [])],
            progress=data.get("progress", 0.0),
            error=error,
            metadata=data.get("metadata", {}),
            created=data.get("created", datetime.utcnow().isoformat()),
            updated=data.get("updated", datetime.utcnow().isoformat()),
            completed=data.get("completed"),
        )

    def update_state(self, state: TaskState) -> None:
        """
        Update task state.

        Args:
            state: New task state
        """
        self.state = state
        self.updated = datetime.utcnow().isoformat()
        if state == TaskState.COMPLETED:
            self.completed = self.updated
            self.progress = 1.0

    def update_progress(self, progress: float) -> None:
        """
        Update task progress.

        Args:
            progress: Progress value (0.0 to 1.0)
        """
        self.progress = max(0.0, min(1.0, progress))
        self.updated = datetime.utcnow().isoformat()

    def add_artifact(self, artifact: Artifact) -> None:
        """
        Add an artifact to the task.

        Args:
            artifact: Artifact to add
        """
        self.artifacts.append(artifact)
        self.updated = datetime.utcnow().isoformat()

    def set_error(self, error: A2AError) -> None:
        """
        Set task error and update state to FAILED.

        Args:
            error: Error information
        """
        self.error = error
        self.update_state(TaskState.FAILED)

    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)

    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.state == TaskState.COMPLETED and self.error is None


@dataclass
class TaskRequest:
    """
    Request to create a new task.

    Attributes:
        instruction: Task instruction
        capability: Required capability name
        context: Shared context
        metadata: Additional metadata
        streaming: Whether to stream progress updates
    """
    instruction: A2AMessage
    capability: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    streaming: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instruction": self.instruction.to_dict(),
            "capability": self.capability,
            "context": self.context,
            "metadata": self.metadata,
            "streaming": self.streaming,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRequest":
        """Create from dictionary."""
        return cls(
            instruction=A2AMessage.from_dict(data["instruction"]),
            capability=data["capability"],
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            streaming=data.get("streaming", False),
        )


@dataclass
class TaskUpdate:
    """
    Update for an existing task.

    Attributes:
        taskId: Task identifier
        state: Updated state
        progress: Updated progress
        artifacts: New artifacts
        error: Error information
    """
    taskId: str
    state: Optional[TaskState] = None
    progress: Optional[float] = None
    artifacts: List[Artifact] = field(default_factory=list)
    error: Optional[A2AError] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"taskId": self.taskId}
        if self.state:
            result["state"] = self.state.value
        if self.progress is not None:
            result["progress"] = self.progress
        if self.artifacts:
            result["artifacts"] = [a.to_dict() for a in self.artifacts]
        if self.error:
            result["error"] = self.error.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskUpdate":
        """Create from dictionary."""
        state = TaskState(data["state"]) if "state" in data else None
        error = A2AError.from_dict(data["error"]) if "error" in data else None

        return cls(
            taskId=data["taskId"],
            state=state,
            progress=data.get("progress"),
            artifacts=[Artifact.from_dict(a) for a in data.get("artifacts", [])],
            error=error,
        )


class A2AProtocolHandler:
    """
    Handler for A2A protocol operations.

    Implements Google's Agent-to-Agent protocol for task creation,
    lifecycle management, and message handling.
    """

    def __init__(self, version: str = "1.0"):
        """
        Initialize A2A protocol handler.

        Args:
            version: A2A protocol version
        """
        self.version = version
        self._tasks: Dict[str, Task] = {}

    def create_task(
        self,
        instruction: A2AMessage,
        capability: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Create a new task.

        Args:
            instruction: Task instruction message
            capability: Required capability
            context: Shared context
            metadata: Additional metadata

        Returns:
            Created task
        """
        task = Task(
            instruction=instruction,
            metadata=metadata or {},
        )
        task.metadata["capability"] = capability
        if context:
            task.metadata["context"] = context

        self._tasks[task.id] = task
        logger.info(f"Created task {task.id} for capability {capability}")
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task if found, None otherwise
        """
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, update: TaskUpdate) -> bool:
        """
        Update a task.

        Args:
            task_id: Task identifier
            update: Task update

        Returns:
            True if task was updated, False if not found
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if update.state:
            task.update_state(update.state)
        if update.progress is not None:
            task.update_progress(update.progress)
        if update.artifacts:
            for artifact in update.artifacts:
                task.add_artifact(artifact)
        if update.error:
            task.set_error(update.error)

        logger.info(f"Updated task {task_id}: state={task.state.value}, progress={task.progress}")
        return True

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was deleted, False if not found
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.info(f"Deleted task {task_id}")
            return True
        return False

    def list_tasks(
        self,
        state: Optional[TaskState] = None,
        capability: Optional[str] = None,
    ) -> List[Task]:
        """
        List tasks with optional filtering.

        Args:
            state: Filter by state
            capability: Filter by capability

        Returns:
            List of matching tasks
        """
        tasks = list(self._tasks.values())

        if state:
            tasks = [t for t in tasks if t.state == state]
        if capability:
            tasks = [
                t for t in tasks if t.metadata.get("capability") == capability
            ]

        return tasks

    def create_message(
        self,
        text: Optional[str] = None,
        parts: Optional[List[MessagePart]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> A2AMessage:
        """
        Create an A2A message.

        Args:
            text: Text content (convenience parameter)
            parts: Message parts
            context: Shared context

        Returns:
            A2A message
        """
        message_parts = parts or []
        if text:
            message_parts.insert(
                0,
                MessagePart(type=MessagePartType.TEXT, content=text),
            )

        return A2AMessage(
            parts=message_parts,
            context=context or {},
        )

    def create_artifact(
        self,
        name: str,
        content: Any,
        artifact_type: str = "result",
        mime_type: Optional[str] = None,
    ) -> Artifact:
        """
        Create a task artifact.

        Args:
            name: Artifact name
            content: Artifact content
            artifact_type: Type of artifact
            mime_type: MIME type

        Returns:
            Artifact
        """
        return Artifact(
            name=name,
            type=artifact_type,
            content=content,
            mimeType=mime_type,
        )

    def create_error(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ) -> A2AError:
        """
        Create an A2A error.

        Args:
            code: Error code
            message: Error message
            details: Additional details
            retryable: Whether error is retryable

        Returns:
            A2A error
        """
        return A2AError(
            code=code,
            message=message,
            details=details,
            retryable=retryable,
        )

    def serialize_task(self, task: Task) -> str:
        """
        Serialize task to JSON.

        Args:
            task: Task to serialize

        Returns:
            JSON string
        """
        return json.dumps(task.to_dict(), indent=2)

    def deserialize_task(self, data: Union[str, Dict[str, Any]]) -> Task:
        """
        Deserialize task from JSON.

        Args:
            data: JSON string or dictionary

        Returns:
            Task
        """
        if isinstance(data, str):
            data = json.loads(data)
        return Task.from_dict(data)
