"""
effGen: A comprehensive framework for building agents with Small Language Models.

This framework enables SLMs to function as powerful agentic systems through:
- Tool integration (built-in tools + MCP/A2A/ACP protocols)
- Advanced prompt engineering optimized for SLMs
- Smart sub-agent decomposition for complex tasks
- Multi-GPU support with vLLM and Transformers
- Comprehensive configuration management
"""

__version__ = "0.0.1"
__author__ = "effGen Team"
__license__ = "MIT"

# Core imports
from effgen.core.agent import Agent, AgentConfig
from effgen.core.task import Task, SubTask, TaskStatus, TaskPriority
from effgen.core.state import AgentState

# Model imports
from effgen.models import (
    load_model,
    BaseModel,
    VLLMEngine,
    TransformersEngine,
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    ModelLoader,
    GenerationConfig,
    GenerationResult
)

# Tool imports
from effgen.tools import (
    BaseTool,
    ToolRegistry,
    get_registry as get_tool_registry
)

# Configuration imports
from effgen.config import (
    ConfigLoader,
    Config,
    ConfigValidator
)

# Prompt imports
from effgen.prompts import (
    TemplateManager,
    ChainManager,
    PromptOptimizer
)

# GPU imports
from effgen.gpu import (
    GPUAllocator,
    GPUMonitor,
    gpu_utils
)

# Memory imports
from effgen.memory import (
    ShortTermMemory,
    LongTermMemory,
    VectorMemoryStore,
    Message,
    MessageRole,
    MemoryEntry,
    MemoryType,
    ImportanceLevel,
    JSONStorageBackend,
    SQLiteStorageBackend
)

# Execution imports
from effgen.execution import (
    CodeExecutor,
    ExecutionResult,
    ExecutionStatus,
    CodeValidator,
    ValidationResult,
    ValidationSeverity,
    SandboxConfig
)

__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "Task",
    "SubTask",
    "TaskStatus",
    "TaskPriority",
    "AgentState",

    # Models
    "load_model",
    "BaseModel",
    "VLLMEngine",
    "TransformersEngine",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "ModelLoader",
    "GenerationConfig",
    "GenerationResult",

    # Tools
    "BaseTool",
    "ToolRegistry",
    "get_tool_registry",

    # Configuration
    "ConfigLoader",
    "Config",
    "ConfigValidator",

    # Prompts
    "TemplateManager",
    "ChainManager",
    "PromptOptimizer",

    # GPU
    "GPUAllocator",
    "GPUMonitor",
    "gpu_utils",

    # Memory
    "ShortTermMemory",
    "LongTermMemory",
    "VectorMemoryStore",
    "Message",
    "MessageRole",
    "MemoryEntry",
    "MemoryType",
    "ImportanceLevel",
    "JSONStorageBackend",
    "SQLiteStorageBackend",

    # Execution
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "CodeValidator",
    "ValidationResult",
    "ValidationSeverity",
    "SandboxConfig",
]
