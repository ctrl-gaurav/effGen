"""
effGen: A comprehensive framework for building agents with Small Language Models.

This framework enables SLMs to function as powerful agentic systems through:
- Tool integration (built-in tools + MCP/A2A/ACP protocols)
- Advanced prompt engineering optimized for SLMs
- Smart sub-agent decomposition for complex tasks
- Multi-GPU support with vLLM and Transformers
- Comprehensive configuration management
"""

__version__ = "0.1.3"
__author__ = "effGen Team"
__license__ = "Apache-2.0"

# Core imports
# Configuration imports
from effgen.config import Config, ConfigLoader, ConfigValidator
from effgen.core.agent import Agent, AgentConfig
from effgen.core.state import AgentState
from effgen.core.task import SubTask, Task, TaskPriority, TaskStatus

# GPU imports
from effgen.gpu import GPUAllocator, GPUMonitor, gpu_utils

# Memory imports
from effgen.memory import (
    ImportanceLevel,
    JSONStorageBackend,
    LongTermMemory,
    MemoryEntry,
    MemoryType,
    Message,
    MessageRole,
    ShortTermMemory,
    SQLiteStorageBackend,
    VectorMemoryStore,
)

# Model imports
from effgen.models import (
    AnthropicAdapter,
    BaseModel,
    GeminiAdapter,
    GenerationConfig,
    GenerationResult,
    ModelLoader,
    OpenAIAdapter,
    TransformersEngine,
    VLLMEngine,
    load_model,
)

# Batch & Aggregation imports
from effgen.core.batch import BatchConfig, BatchResult, BatchRunner
from effgen.core.aggregation import ResultAggregator

# Domain imports
from effgen.domains import (
    Domain,
    FinanceDomain,
    HealthDomain,
    KeywordExpander,
    LegalDomain,
    ScienceDomain,
    TechDomain,
)

# Preset imports
from effgen.presets import create_agent, list_presets

# Prompt imports
from effgen.prompts import ChainManager, PromptOptimizer, TemplateManager

# Guardrails imports
from effgen.guardrails import (
    Guardrail,
    GuardrailChain,
    GuardrailPosition,
    GuardrailResult,
    LengthGuardrail,
    PIIGuardrail,
    PromptInjectionGuardrail,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolPermissionGuardrail,
    TopicGuardrail,
    ToxicityGuardrail,
    get_guardrail_preset,
)

# Tool imports
from effgen.tools import BaseTool, ToolRegistry
from effgen.tools import get_registry as get_tool_registry

# MLX engine imports (Apple Silicon only)
try:
    from effgen.models.mlx_engine import MLXEngine
    from effgen.models.mlx_vlm_engine import MLXVLMEngine
except ImportError:
    pass

# Additional convenience imports
try:
    from effgen.tools.fallback import ToolFallbackChain
except ImportError:
    pass

try:
    from effgen.utils.circuit_breaker import CircuitBreaker
except ImportError:
    pass

try:
    from effgen.prompts.tool_prompt_generator import ToolPromptGenerator
except ImportError:
    pass

try:
    from effgen.prompts.agent_system_prompt import AgentSystemPromptBuilder
except ImportError:
    pass

# Execution imports
from effgen.execution import (
    CodeExecutor,
    CodeValidator,
    ExecutionResult,
    ExecutionStatus,
    SandboxConfig,
    ValidationResult,
    ValidationSeverity,
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

    # Batch & Aggregation
    "BatchRunner",
    "BatchConfig",
    "BatchResult",
    "ResultAggregator",

    # Domains
    "Domain",
    "KeywordExpander",
    "TechDomain",
    "ScienceDomain",
    "FinanceDomain",
    "HealthDomain",
    "LegalDomain",

    # Presets
    "create_agent",
    "list_presets",

    # Additional convenience exports
    "ToolFallbackChain",
    "CircuitBreaker",
    "ToolPromptGenerator",
    "AgentSystemPromptBuilder",

    # Execution
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "CodeValidator",
    "ValidationResult",
    "ValidationSeverity",
    "SandboxConfig",
]
