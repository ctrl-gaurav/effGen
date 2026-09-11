"""Core agent system: the Agent, routing, sub-agents, sessions and workflows.

Components:
- Agent: Main agent class with ReAct loop and sub-agent support
- Router: Intelligent routing for sub-agent decisions
- SubAgentManager: Sub-agent lifecycle management
- ExecutionTracker: Transparent execution tracking
- Orchestrator: Multi-agent coordination
- Task and State management
"""

# ruff: noqa: I001

# Multimodal message schema
from .messages import (
    AudioPart,
    ContentPart,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    VideoPart,
)
from .multimodal import audio_from, image_from, video_from

# A run's conversation as typed steps
from .thread import (
    ActionStep,
    AgentThread,
    AnswerStep,
    NudgeStep,
    ObservationStep,
    Step,
    SystemStep,
    TaskStep,
    TurnStep,
    ThoughtStep,
)

# Agent
from .agent import Agent, AgentConfig, AgentMode, AgentResponse
from .agent_response import PartialResult

# Result Aggregation
from .aggregation import AggregatedResult, MergeStrategy, ResultAggregator, ToolResultCache

# Batch Execution
from .batch import BatchConfig, BatchResult, BatchRunner

# Clarification
from .clarification import ClarificationDetector, ClarificationRequest

# Complexity Analyzer
from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore

# Decomposition Engine
from .decomposition_engine import DecompositionEngine, TaskStructure

# Execution Tracker
from .execution_tracker import (
    EventType,
    ExecutionEvent,
    ExecutionNode,
    ExecutionStatus,
    ExecutionTracker,
)

# Feedback
from .feedback import FeedbackCollector, FeedbackEntry, FeedbackType

# Human-in-the-Loop
from .human_loop import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalMode,
    HumanApproval,
    HumanChoice,
    HumanInput,
)

# Lifecycle Management
from .lifecycle import AgentEntry, AgentLifecycleState, AgentPool, AgentRegistry

# Message Bus
from .message_bus import AgentMessage, MessageBus, MessageType

# Orchestrator
from .orchestrator import MultiAgentOrchestrator, OrchestrationPattern, TeamConfig, TeamResponse

# Router
from .router import RoutingDecision, RoutingStrategy, SubAgentRouter

# Shared State
from .shared_state import SharedState, StateMutation

# State
from .state import AgentState

# Structured Output
from .structured_output import (
    StructuredOutcome,
    StructuredOutputConfig,
    constrain_output,
    structured_generate,
    validate_json_schema,
)

# Sub-Agent Manager
from .sub_agent_manager import (
    SubAgentConfig,
    SubAgentManager,
    SubAgentResult,
    SubAgentSpecialization,
)

# Task
from .task import SubTask, Task, TaskPriority, TaskStatus

# Tool Calling Strategy
from .tool_calling import (
    HybridStrategy,
    NativeFunctionCallingStrategy,
    ReActStrategy,
    ToolCallingStrategy,
    ToolCallResult,
    ToolDefinition,
    get_strategy,
    tools_to_definitions,
)

# Workflow
from .workflow import WorkflowDAG, WorkflowEdge, WorkflowNode, WorkflowResult
from .workflow_checkpoint import (
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    WorkflowCheckpoint,
)

__all__ = [
    # A run's conversation as typed steps
    "AgentThread",
    "Step",
    "SystemStep",
    "TaskStep",
    "TurnStep",
    "ThoughtStep",
    "ActionStep",
    "ObservationStep",
    "NudgeStep",
    "AnswerStep",

    # Multimodal message schema
    "Role",
    "TextPart",
    "ImagePart",
    "AudioPart",
    "VideoPart",
    "ToolCallPart",
    "ToolResultPart",
    "ContentPart",
    "Message",
    "image_from",
    "audio_from",
    "video_from",

    # Agent
    "Agent",
    "AgentConfig",
    "AgentResponse",
    "PartialResult",
    "AgentMode",

    # Router
    "SubAgentRouter",
    "RoutingDecision",
    "RoutingStrategy",

    # Sub-Agent Manager
    "SubAgentManager",
    "SubAgentConfig",
    "SubAgentResult",
    "SubAgentSpecialization",

    # Execution Tracker
    "ExecutionTracker",
    "ExecutionEvent",
    "ExecutionStatus",
    "ExecutionNode",
    "EventType",

    # Orchestrator
    "MultiAgentOrchestrator",
    "TeamConfig",
    "TeamResponse",
    "OrchestrationPattern",

    # Complexity Analyzer
    "ComplexityAnalyzer",
    "ComplexityScore",

    # Decomposition Engine
    "DecompositionEngine",
    "TaskStructure",

    # Task
    "Task",
    "SubTask",
    "TaskStatus",
    "TaskPriority",

    # Lifecycle Management
    "AgentLifecycleState",
    "AgentEntry",
    "AgentPool",
    "AgentRegistry",

    # Message Bus
    "MessageBus",
    "AgentMessage",
    "MessageType",

    # Shared State
    "SharedState",
    "StateMutation",

    # Workflow
    "WorkflowDAG",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowCheckpoint",
    "CheckpointStore",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "WorkflowResult",

    # Batch Execution
    "BatchRunner",
    "BatchConfig",
    "BatchResult",

    # Result Aggregation
    "ResultAggregator",
    "AggregatedResult",
    "MergeStrategy",
    "ToolResultCache",

    # State
    "AgentState",

    # Structured Output
    "StructuredOutputConfig",
    "StructuredOutcome",
    "constrain_output",
    "structured_generate",
    "validate_json_schema",

    # Human-in-the-Loop
    "ApprovalMode",
    "ApprovalDecision",
    "ApprovalManager",
    "HumanApproval",
    "HumanInput",
    "HumanChoice",

    # Clarification
    "ClarificationDetector",
    "ClarificationRequest",

    # Feedback
    "FeedbackCollector",
    "FeedbackEntry",
    "FeedbackType",

    # Tool Calling Strategy
    "ToolCallingStrategy",
    "ReActStrategy",
    "NativeFunctionCallingStrategy",
    "HybridStrategy",
    "ToolCallResult",
    "ToolDefinition",
    "get_strategy",
    "tools_to_definitions",
]
