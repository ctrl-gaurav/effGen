"""
EffGen Core Module

This module contains the core agent system components:
- Agent: Main agent class with ReAct loop and sub-agent support
- Router: Intelligent routing for sub-agent decisions
- SubAgentManager: Sub-agent lifecycle management
- ExecutionTracker: Transparent execution tracking
- Orchestrator: Multi-agent coordination
- Task and State management
"""

# Agent
from .agent import (
    Agent,
    AgentConfig,
    AgentResponse,
    AgentMode
)

# Router
from .router import (
    SubAgentRouter,
    RoutingDecision,
    RoutingStrategy
)

# Sub-Agent Manager
from .sub_agent_manager import (
    SubAgentManager,
    SubAgentConfig,
    SubAgentResult,
    SubAgentSpecialization
)

# Execution Tracker
from .execution_tracker import (
    ExecutionTracker,
    ExecutionEvent,
    ExecutionStatus,
    ExecutionNode,
    EventType
)

# Orchestrator
from .orchestrator import (
    MultiAgentOrchestrator,
    TeamConfig,
    TeamResponse,
    OrchestrationPattern
)

# Complexity Analyzer
from .complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityScore
)

# Decomposition Engine
from .decomposition_engine import (
    DecompositionEngine,
    TaskStructure
)

# Task
from .task import (
    Task,
    SubTask,
    TaskStatus,
    TaskPriority
)

# State
from .state import (
    AgentState
)

__all__ = [
    # Agent
    "Agent",
    "AgentConfig",
    "AgentResponse",
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

    # State
    "AgentState",
]
