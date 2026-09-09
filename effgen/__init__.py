"""
effGen: A comprehensive framework for building agents with Small Language Models.

This framework runs SLMs as agentic systems through:
- Tool integration (built-in tools + MCP/A2A/ACP protocols)
- Advanced prompt engineering optimized for SLMs
- Smart sub-agent decomposition for complex tasks
- Multi-GPU support with vLLM and Transformers
- Comprehensive configuration management

The command-line tools load a nearby ``.env`` automatically. In a script or
notebook, call :func:`effgen.load_env` once to get the same key discovery::

    from effgen import load_env, create_agent
    load_env()  # reads OPENAI_API_KEY / GROQ_API_KEY / ... from a nearby .env
    agent = create_agent("minimal", "groq:openai/gpt-oss-20b")

The public surface (``effgen.__all__``) is resolved **lazily**: importing
``effgen`` only sets up version metadata and logging hygiene, and the first
access of a name (``effgen.Agent``, ``from effgen import load_model``) imports
just the submodule that defines it. This keeps ``import effgen`` and a bare
version check cheap, and defers heavy/optional dependencies (torch,
transformers, vLLM, provider SDKs) until something that needs them is used.
"""

# ruff: noqa: I001

__version__ = "1.0.1"
__author__ = "effGen Team"
__license__ = "Apache-2.0"

# Fail early and clearly on an unsupported interpreter. pip already refuses to
# install on < 3.11 (requires-python in pyproject.toml), but a source checkout
# run via PYTHONPATH skips that gate — and effGen's submodules use 3.11+ syntax,
# so without this guard the user would hit a cryptic SyntaxError instead. Keep
# this block free of any 3.11-only syntax so it runs on the wrong interpreter.
import os as _os
import sys as _sys
from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if _sys.version_info < (3, 11):  # noqa: UP036 — intentional guard for source-checkout runs on < 3.11
    raise RuntimeError(
        "effGen requires Python 3.11 or newer, but you are running "
        "{}.{}.{}. Please upgrade your interpreter (e.g. create a Python 3.12 "
        "environment) and reinstall effGen.".format(*_sys.version_info[:3])
    )

# Library logging hygiene (NullHandler on the package logger; quiet faiss). This
# is an import-only module whose side effects run first, before any submodule is
# imported, so importing effGen never emits log records on its own. It is
# intentionally cheap (stdlib logging only).
from effgen import _logging_setup as _logging_setup  # noqa: F401

# ---------------------------------------------------------------------------
# Lazy public surface.
#
# Each entry maps a public name to ``(module, attribute)``. ``__getattr__``
# imports the module on first access, binds the resolved object into this
# module's globals (so subsequent accesses are a plain dict lookup), and returns
# it. An attribute of ``None`` means the name *is* the submodule itself
# (e.g. ``gpu_utils``).
#
# This mapping is the single source of truth for both ``__getattr__`` and
# ``__dir__``. ``__all__`` (the documented 170-name surface) is derived from it
# below; the extra entries (provider-native tool classes, MLX engines) are
# historically importable but were never in ``__all__``, so they stay out of it.
# ---------------------------------------------------------------------------
_LAZY: dict[str, tuple[str, str | None]] = {
    "BackendUnreachableError": ("effgen.models.errors", "BackendUnreachableError"),
    "OpenAICompatibleAdapter": ("effgen.models.openai_compatible_adapter", "OpenAICompatibleAdapter"),
    "ToolCallList": ("effgen.core.tool_call_record", "ToolCallList"),
    "ToolCall": ("effgen.core.tool_call_record", "ToolCall"),
    "ToolApprovalMiddleware": ("effgen.core.middleware", "ToolApprovalMiddleware"),
    "MiddlewareChain": ("effgen.core.middleware", "MiddlewareChain"),
    "LoggingMiddleware": ("effgen.core.middleware", "LoggingMiddleware"),
    "SummarizeOldest": ("effgen.memory.compaction", "SummarizeOldest"),
    "KeepToolResults": ("effgen.memory.compaction", "KeepToolResults"),
    "KeepFirstAndLast": ("effgen.memory.compaction", "KeepFirstAndLast"),
    "DropOldest": ("effgen.memory.compaction", "DropOldest"),
    "CompactionStrategy": ("effgen.memory.compaction", "CompactionStrategy"),
    "Agent": ("effgen.core.agent", "Agent"),
    "PartialResult": ("effgen.core.agent_response", "PartialResult"),
    "RunStoppedError": ("effgen.errors", "RunStoppedError"),
    "AgentConfig": ("effgen.core.agent", "AgentConfig"),
    "AgentMiddleware": ("effgen.core.middleware", "AgentMiddleware"),
    "AgentEvaluator": ("effgen.eval.evaluator", "AgentEvaluator"),
    "AgentState": ("effgen.core.state", "AgentState"),
    "AgentSystemPromptBuilder": ("effgen.prompts.agent_system_prompt", "AgentSystemPromptBuilder"),
    "Alert": ("effgen.observability.alerting", "Alert"),
    "AlertSeverity": ("effgen.observability.alerting", "AlertSeverity"),
    "AlertWebhook": ("effgen.observability.alerting", "AlertWebhook"),
    "AllCandidatesExhaustedError": ("effgen.models.errors", "AllCandidatesExhaustedError"),
    "AmbiguousModelError": ("effgen.models.errors", "AmbiguousModelError"),
    "AnthropicAdapter": ("effgen.models.anthropic_adapter", "AnthropicAdapter"),
    "AudioPart": ("effgen.core.messages", "AudioPart"),
    "BaseModel": ("effgen.models.base", "BaseModel"),
    "BaseTool": ("effgen.tools.base_tool", "BaseTool"),
    "BatchConfig": ("effgen.core.batch", "BatchConfig"),
    "BatchResult": ("effgen.core.batch", "BatchResult"),
    "BatchRunner": ("effgen.core.batch", "BatchRunner"),
    "BudgetExceededError": ("effgen.models.errors", "BudgetExceededError"),
    "CapabilityNotSupportedError": ("effgen.errors", "CapabilityNotSupportedError"),
    "CerebrasAdapter": ("effgen.models.cerebras_adapter", "CerebrasAdapter"),
    "ChainManager": ("effgen.prompts.chain_manager", "ChainManager"),
    "CircuitBreaker": ("effgen.utils.circuit_breaker", "CircuitBreaker"),
    # CodeExecutor here is the convenience wrapper that runs code through a
    # local subprocess with a static AST/regex validator (or Docker). It is not
    # OS-level isolation on the local backend; for a model-facing tool with the
    # project's hardened isolation, use the registered `code_executor` tool
    # (effgen.tools.builtin.code_executor). See effgen/execution/sandbox.py.
    "CodeExecutor": ("effgen.execution.sandbox", "CodeExecutor"),
    "CodeValidator": ("effgen.execution.validators", "CodeValidator"),
    "Config": ("effgen.config.loader", "Config"),
    "ConfigLoader": ("effgen.config.loader", "ConfigLoader"),
    "ConfigValidator": ("effgen.config.validator", "ConfigValidator"),
    "ContentPart": ("effgen.core.messages", "ContentPart"),
    "CostBasedPolicy": ("effgen.models.routing.cost", "CostBasedPolicy"),
    "CostTracker": ("effgen.models._cost", "CostTracker"),
    "Domain": ("effgen.domains.base", "Domain"),
    "EvalResult": ("effgen.eval.evaluator", "EvalResult"),
    "ExecutionResult": ("effgen.execution.sandbox", "ExecutionResult"),
    "ExecutionStatus": ("effgen.execution.sandbox", "ExecutionStatus"),
    "FinanceDomain": ("effgen.domains.presets", "FinanceDomain"),
    "FireworksAdapter": ("effgen.models.fireworks_adapter", "FireworksAdapter"),
    "FirstAvailablePolicy": ("effgen.models.routing.first_available", "FirstAvailablePolicy"),
    "FunctionTool": ("effgen.tools.function_tool", "FunctionTool"),
    "GPUAllocator": ("effgen.gpu.allocator", "GPUAllocator"),
    "GPUMonitor": ("effgen.gpu.monitor", "GPUMonitor"),
    "GeminiAdapter": ("effgen.models.gemini_adapter", "GeminiAdapter"),
    "GeminiCodeExecutionTool": ("effgen.tools.builtin.gemini_native", "GeminiCodeExecutionTool"),
    "GeminiNativeTool": ("effgen.tools.builtin.gemini_native", "GeminiNativeTool"),
    "GeminiUrlContextTool": ("effgen.tools.builtin.gemini_native", "GeminiUrlContextTool"),
    "GenerationConfig": ("effgen.models.base", "GenerationConfig"),
    "GenerationResult": ("effgen.models.base", "GenerationResult"),
    "GoogleSearchTool": ("effgen.tools.builtin.gemini_native", "GoogleSearchTool"),
    "GroqAdapter": ("effgen.models.groq_adapter", "GroqAdapter"),
    "HFInferenceAdapter": ("effgen.models.hf_inference_adapter", "HFInferenceAdapter"),
    "HashDriftWarning": ("effgen.security.supply_chain", "HashDriftWarning"),
    "HealthDomain": ("effgen.domains.presets", "HealthDomain"),
    "ImagePart": ("effgen.core.messages", "ImagePart"),
    "ImportanceLevel": ("effgen.memory.long_term", "ImportanceLevel"),
    "InvalidMultimodalContent": ("effgen.errors", "InvalidMultimodalContent"),
    "InvalidRequestError": ("effgen.models.errors", "InvalidRequestError"),
    "JSONStorageBackend": ("effgen.memory.long_term", "JSONStorageBackend"),
    "KeywordExpander": ("effgen.domains.expander", "KeywordExpander"),
    "LatencyBasedPolicy": ("effgen.models.routing.latency", "LatencyBasedPolicy"),
    "LatencyTracker": ("effgen.models.latency_tracker", "LatencyTracker"),
    "LegalDomain": ("effgen.domains.presets", "LegalDomain"),
    "LongTermMemory": ("effgen.memory.long_term", "LongTermMemory"),
    "MLXEngine": ("effgen.models.mlx_engine", "MLXEngine"),
    "MLXVLMEngine": ("effgen.models.mlx_vlm_engine", "MLXVLMEngine"),
    "MemoryEntry": ("effgen.memory.long_term", "MemoryEntry"),
    "MemoryType": ("effgen.memory.long_term", "MemoryType"),
    "MultiAgentOrchestrator": ("effgen.core.orchestrator", "MultiAgentOrchestrator"),
    "Message": ("effgen.memory.short_term", "Message"),
    "MessageRole": ("effgen.memory.short_term", "MessageRole"),
    "ModelAuthError": ("effgen.models.errors", "ModelAuthError"),
    "ModelComparison": ("effgen.eval.comparison", "ModelComparison"),
    "ModelLoader": ("effgen.models.model_loader", "ModelLoader"),
    "ModelNotFoundError": ("effgen.models.errors", "ModelNotFoundError"),
    "ModelRefusalError": ("effgen.models.errors", "ModelRefusalError"),
    "ModelTimeoutError": ("effgen.models.errors", "ModelTimeoutError"),
    "ModelUnavailableError": ("effgen.models.errors", "ModelUnavailableError"),
    "MultimodalMessage": ("effgen.core.messages", "Message"),
    "NoCandidateWithinBudgetError": ("effgen.models.errors", "NoCandidateWithinBudgetError"),
    "OrchestrationPattern": ("effgen.core.orchestrator", "OrchestrationPattern"),
    "OpenAIAdapter": ("effgen.models.openai_adapter", "OpenAIAdapter"),
    "OpenAICodeInterpreterTool": ("effgen.tools.builtin.openai_native", "OpenAICodeInterpreterTool"),
    "OpenAIFileSearchTool": ("effgen.tools.builtin.openai_native", "OpenAIFileSearchTool"),
    "OpenAINativeTool": ("effgen.tools.builtin.openai_native", "OpenAINativeTool"),
    "OpenAIWebSearchTool": ("effgen.tools.builtin.openai_native", "OpenAIWebSearchTool"),
    "PolicyBasedRouter": ("effgen.models.router", "PolicyBasedRouter"),
    "PromptOptimizer": ("effgen.prompts.optimizer", "PromptOptimizer"),
    "ProviderModelPair": ("effgen.models.router", "ProviderModelPair"),
    "ProviderRegistry": ("effgen.models.registry", "ProviderRegistry"),
    "ProviderTransientError": ("effgen.models.errors", "ProviderTransientError"),
    "RateLimitCoordinator": ("effgen.models._rate_limit", "RateLimitCoordinator"),
    "RateLimitExceeded": ("effgen.models._rate_limit", "RateLimitExceeded"),
    "RegressionTracker": ("effgen.eval.regression", "RegressionTracker"),
    "ReplicateAdapter": ("effgen.models.replicate_adapter", "ReplicateAdapter"),
    "ResultAggregator": ("effgen.core.aggregation", "ResultAggregator"),
    "RetryPolicy": ("effgen.models.routing.retry", "RetryPolicy"),
    "Role": ("effgen.core.messages", "Role"),
    "RouterDecision": ("effgen.models.router", "RouterDecision"),
    "RouterEvent": ("effgen.models.router", "RouterEvent"),
    "RoutingContext": ("effgen.models.router", "RoutingContext"),
    "RoutingDecision": ("effgen.core.router", "RoutingDecision"),
    "RoutingPolicy": ("effgen.models.router", "RoutingPolicy"),
    "RoutingStrategy": ("effgen.core.router", "RoutingStrategy"),
    "SLO": ("effgen.observability.slo", "SLO"),
    "SLOTracker": ("effgen.observability.slo", "SLOTracker"),
    "SQLiteCostStore": ("effgen.models._cost_store", "SQLiteCostStore"),
    "SQLiteRateLimitStore": ("effgen.models._rate_limit_store", "SQLiteRateLimitStore"),
    "SQLiteStorageBackend": ("effgen.memory.long_term", "SQLiteStorageBackend"),
    "SandboxConfig": ("effgen.execution.sandbox", "SandboxConfig"),
    "ScienceDomain": ("effgen.domains.presets", "ScienceDomain"),
    "ShortTermMemory": ("effgen.memory.short_term", "ShortTermMemory"),
    "StreamChunk": ("effgen.models.anthropic_adapter", "StreamChunk"),
    "SubAgentRouter": ("effgen.core.router", "SubAgentRouter"),
    "SubTask": ("effgen.core.task", "SubTask"),
    "SuiteResults": ("effgen.eval.evaluator", "SuiteResults"),
    "Task": ("effgen.core.task", "Task"),
    "TaskPriority": ("effgen.core.task", "TaskPriority"),
    "TaskStatus": ("effgen.core.task", "TaskStatus"),
    "TeamConfig": ("effgen.core.orchestrator", "TeamConfig"),
    "TeamResponse": ("effgen.core.orchestrator", "TeamResponse"),
    "TechDomain": ("effgen.domains.presets", "TechDomain"),
    "TemplateManager": ("effgen.prompts.template_manager", "TemplateManager"),
    "TestCase": ("effgen.eval.evaluator", "TestCase"),
    "TestSuite": ("effgen.eval.suites", "TestSuite"),
    "TextPart": ("effgen.core.messages", "TextPart"),
    "TogetherAdapter": ("effgen.models.together_adapter", "TogetherAdapter"),
    "Tool": ("effgen.tools.function_tool", "Tool"),
    "ToolCallPart": ("effgen.core.messages", "ToolCallPart"),
    "ToolFallbackChain": ("effgen.tools.fallback", "ToolFallbackChain"),
    "ToolIncompatibleError": ("effgen.models.errors", "ToolIncompatibleError"),
    "ToolPromptGenerator": ("effgen.prompts.tool_prompt_generator", "ToolPromptGenerator"),
    "ToolRegistry": ("effgen.tools.registry", "ToolRegistry"),
    "ToolResultPart": ("effgen.core.messages", "ToolResultPart"),
    "TransformersEngine": ("effgen.models.transformers_engine", "TransformersEngine"),
    "VLLMEngine": ("effgen.models.vllm_engine", "VLLMEngine"),
    "ValidationResult": ("effgen.execution.validators", "ValidationResult"),
    "ValidationSeverity": ("effgen.execution.validators", "ValidationSeverity"),
    "VectorMemoryStore": ("effgen.memory.vector_store", "VectorMemoryStore"),
    "VerificationResult": ("effgen.security.supply_chain", "VerificationResult"),
    "VideoPart": ("effgen.core.messages", "VideoPart"),
    "CheckpointStore": ("effgen.core.workflow_checkpoint", "CheckpointStore"),
    "FileCheckpointStore": ("effgen.core.workflow_checkpoint", "FileCheckpointStore"),
    "InMemoryCheckpointStore": (
        "effgen.core.workflow_checkpoint", "InMemoryCheckpointStore",
    ),
    "WorkflowCheckpoint": ("effgen.core.workflow_checkpoint", "WorkflowCheckpoint"),
    "WorkflowDAG": ("effgen.core.workflow", "WorkflowDAG"),
    "WorkflowEdge": ("effgen.core.workflow", "WorkflowEdge"),
    "WorkflowNode": ("effgen.core.workflow", "WorkflowNode"),
    "WorkflowResult": ("effgen.core.workflow", "WorkflowResult"),
    "audio_from": ("effgen.core.multimodal", "audio_from"),
    "cerebras_available_models": ("effgen.models.cerebras_models", "available_models"),
    "cerebras_free_tier_models": ("effgen.models.cerebras_models", "free_tier_models"),
    "cerebras_model_info": ("effgen.models.cerebras_models", "model_info"),
    "check_keys": ("effgen.models.auth", "check_keys"),
    "check_slo_and_alert": ("effgen.observability.alerting", "check_slo_and_alert"),
    "create_agent": ("effgen.presets.registry", "create_agent"),
    "fireworks_available_models": ("effgen.models.fireworks_models", "available_models"),
    "fireworks_chat_models": ("effgen.models.fireworks_models", "chat_models"),
    "fireworks_pricing_table": ("effgen.models.fireworks_models", "pricing_table"),
    "fireworks_refresh_models": ("effgen.models.fireworks_models", "refresh_models"),
    "fireworks_tool_capable_models": ("effgen.models.fireworks_models", "tool_capable_models"),
    "gemini_available_models": ("effgen.models.gemini_models", "available_models"),
    "gemini_free_tier_models": ("effgen.models.gemini_models", "free_tier_models"),
    "gemini_model_info": ("effgen.models.gemini_models", "model_info"),
    "gemini_recommended_models": ("effgen.models.gemini_models", "recommended_models"),
    "get_guardrail_preset": ("effgen.guardrails", "get_guardrail_preset"),
    "get_tool_registry": ("effgen.tools.registry", "get_registry"),
    "gpu_utils": ("effgen.gpu.utils", None),
    "groq_available_models": ("effgen.models.groq_models", "available_models"),
    "groq_chat_models": ("effgen.models.groq_models", "chat_models"),
    "groq_tool_capable_models": ("effgen.models.groq_models", "tool_capable_models"),
    "hf_available_models": ("effgen.models.hf_inference_models", "available_models"),
    "hf_catalog_summary": ("effgen.models.hf_inference_models", "catalog_summary"),
    "hf_chat_models": ("effgen.models.hf_inference_models", "chat_models"),
    "hf_cheapest_provider": ("effgen.models.hf_inference_models", "cheapest_provider"),
    "hf_check_drift": ("effgen.models.hf_inference_models", "check_drift"),
    "hf_get_model_info": ("effgen.models.hf_inference_models", "get_model_info"),
    "hf_list_providers_for": ("effgen.models.hf_inference_models", "list_providers_for"),
    "hf_refresh_models": ("effgen.models.hf_inference_models", "refresh_models"),
    "hf_serverless_models": ("effgen.models.hf_inference_models", "serverless_models"),
    "hf_suggest_alternatives": ("effgen.models.hf_inference_models", "suggest_alternatives"),
    "hf_tool_capable_models": ("effgen.models.hf_inference_models", "tool_capable_models"),
    "image_from": ("effgen.core.multimodal", "image_from"),
    "list_models": ("effgen.models.registry", "list_models"),
    "list_presets": ("effgen.presets.registry", "list_presets"),
    "list_providers": ("effgen.models.registry", "list_providers"),
    "load_env": ("effgen._env", "load_env"),
    "load_model": ("effgen.models.model_loader", "load_model"),
    "lookup": ("effgen.models.registry", "lookup"),
    "openai_available_models": ("effgen.models.openai_models", "available_models"),
    "openai_chat_models": ("effgen.models.openai_models", "chat_models"),
    "openai_model_info": ("effgen.models.openai_models", "model_info"),
    "openai_reasoning_models": ("effgen.models.openai_models", "reasoning_models"),
    "replicate_available_models": ("effgen.models.replicate_models", "available_models"),
    "replicate_get_model_info": ("effgen.models.replicate_models", "get_model_info"),
    "replicate_refresh_models": ("effgen.models.replicate_models", "refresh_models"),
    "replicate_streaming_models": ("effgen.models.replicate_models", "streaming_models"),
    "replicate_tool_capable_models": ("effgen.models.replicate_models", "tool_capable_models"),
    "to_openai_schema": ("effgen.models.openai_schema", "to_openai_schema"),
    "tool": ("effgen.tools.function_tool", "tool"),
    "together_available_models": ("effgen.models.together_models", "available_models"),
    "together_chat_models": ("effgen.models.together_models", "chat_models"),
    "together_pricing_table": ("effgen.models.together_models", "pricing_table"),
    "together_refresh_models": ("effgen.models.together_models", "refresh_models"),
    "together_serverless_models": ("effgen.models.together_models", "serverless_models"),
    "together_tool_capable_models": ("effgen.models.together_models", "tool_capable_models"),
    "validate_alert_rules_yaml": ("effgen.observability.alerting", "validate_alert_rules_yaml"),
    "verify_installed_hashes": ("effgen.security.supply_chain", "verify_installed_hashes"),
    "verify_on_startup": ("effgen.security.supply_chain", "verify_on_startup"),
    "video_from": ("effgen.core.multimodal", "video_from"),
}

# Guardrail classes share one home; group them so the mapping above stays a flat
# name -> (module, attr) table.
for _name in (
    "Guardrail",
    "GuardrailChain",
    "GuardrailPosition",
    "GuardrailResult",
    "LengthGuardrail",
    "PIIGuardrail",
    "PromptInjectionGuardrail",
    "SystemPromptLeakGuardrail",
    "ToolInputGuardrail",
    "ToolOutputGuardrail",
    "ToolPermissionGuardrail",
    "TopicGuardrail",
    "ToxicityGuardrail",
):
    _LAZY[_name] = ("effgen.guardrails", _name)
del _name


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562 lazy import)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'effgen' has no attribute {name!r}")
    module_path, attr = target
    try:
        module = _import_module(module_path)
    except ImportError as exc:
        # Optional extras (provider-native tools, MLX, eval) may be missing in a
        # lean install — surface as a normal missing attribute, not an import
        # crash, mirroring the historical try/except import behaviour.
        raise AttributeError(
            f"module 'effgen' has no attribute {name!r} "
            f"(optional component '{module_path}' is not installed: {exc})"
        ) from exc
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value  # cache: subsequent access is a plain dict lookup
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY) | set(globals()))


# The documented public surface. Kept as an explicit literal (not derived from
# ``_LAZY``) so it is greppable and stable; the provider-native/MLX extras in
# ``_LAZY`` are intentionally excluded, matching prior releases.
__all__ = [
    # Pointing effGen at a server, hooks around the loop, and what a run did
    "OpenAICompatibleAdapter",
    "BackendUnreachableError",
    "AgentMiddleware",
    "MiddlewareChain",
    "LoggingMiddleware",
    "ToolApprovalMiddleware",
    "ToolCall",
    "ToolCallList",
    "PartialResult",
    "RunStoppedError",
    "CompactionStrategy",
    "SummarizeOldest",
    "DropOldest",
    "KeepFirstAndLast",
    "KeepToolResults",
    # Multimodal message schema
    "Role",
    "TextPart",
    "ImagePart",
    "AudioPart",
    "VideoPart",
    "ToolCallPart",
    "ToolResultPart",
    "ContentPart",
    "MultimodalMessage",
    "image_from",
    "audio_from",
    "video_from",
    "CapabilityNotSupportedError",
    "InvalidMultimodalContent",

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
    "StreamChunk",
    "GeminiAdapter",
    "CerebrasAdapter",
    "GroqAdapter",
    "TogetherAdapter",
    "FireworksAdapter",
    "ReplicateAdapter",
    "HFInferenceAdapter",
    "ModelLoader",
    "GenerationConfig",
    "GenerationResult",
    # Router (v0.2.4+)
    "PolicyBasedRouter",
    "RoutingPolicy",
    "RoutingContext",
    "RouterDecision",
    "RouterEvent",
    "ProviderModelPair",
    "FirstAvailablePolicy",
    "CostBasedPolicy",
    "LatencyBasedPolicy",
    "RetryPolicy",
    # Tracking (v0.2.4+)
    "LatencyTracker",
    "CostTracker",
    "SQLiteCostStore",
    "RateLimitCoordinator",
    "RateLimitExceeded",
    "SQLiteRateLimitStore",
    # Observability — alerting & SLO (v0.3.2+)
    "Alert",
    "AlertSeverity",
    "AlertWebhook",
    "check_slo_and_alert",
    "validate_alert_rules_yaml",
    "SLO",
    "SLOTracker",
    # Errors
    "ModelRefusalError",
    "ModelAuthError",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "ModelNotFoundError",
    "AmbiguousModelError",
    "NoCandidateWithinBudgetError",
    "ToolIncompatibleError",
    "AllCandidatesExhaustedError",
    "BudgetExceededError",
    "ProviderTransientError",
    "InvalidRequestError",
    "to_openai_schema",
    # Provider registry + auth
    "ProviderRegistry",
    "list_providers",
    "list_models",
    "lookup",
    "check_keys",
    # Cerebras helpers
    "cerebras_available_models",
    "cerebras_free_tier_models",
    "cerebras_model_info",
    # Groq helpers
    "groq_available_models",
    "groq_chat_models",
    "groq_tool_capable_models",
    # OpenAI helpers
    "openai_available_models",
    "openai_chat_models",
    "openai_reasoning_models",
    "openai_model_info",
    # Gemini helpers
    "gemini_available_models",
    "gemini_free_tier_models",
    "gemini_model_info",
    "gemini_recommended_models",
    # Together helpers
    "together_available_models",
    "together_chat_models",
    "together_tool_capable_models",
    "together_pricing_table",
    "together_refresh_models",
    "together_serverless_models",
    # Fireworks helpers
    "fireworks_available_models",
    "fireworks_chat_models",
    "fireworks_tool_capable_models",
    "fireworks_pricing_table",
    "fireworks_refresh_models",
    # Replicate helpers
    "replicate_available_models",
    "replicate_streaming_models",
    "replicate_tool_capable_models",
    "replicate_refresh_models",
    "replicate_get_model_info",
    # HF Inference helpers
    "hf_available_models",
    "hf_chat_models",
    "hf_tool_capable_models",
    "hf_serverless_models",
    "hf_suggest_alternatives",
    "hf_get_model_info",
    "hf_refresh_models",
    "hf_check_drift",
    "hf_catalog_summary",
    "hf_list_providers_for",
    "hf_cheapest_provider",

    # Tools
    "BaseTool",
    "tool",
    "Tool",
    "FunctionTool",
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

    # Multi-agent orchestration & workflows
    "MultiAgentOrchestrator",
    "TeamConfig",
    "TeamResponse",
    "OrchestrationPattern",
    "WorkflowDAG",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowResult",
    "WorkflowCheckpoint",
    "CheckpointStore",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "SubAgentRouter",
    "RoutingStrategy",
    "RoutingDecision",

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

    # Environment / keys
    "load_env",

    # Guardrails (also importable from effgen.guardrails). Listed here so the
    # everyday classes are as discoverable at the top level as the domains above.
    "Guardrail",
    "GuardrailChain",
    "GuardrailPosition",
    "GuardrailResult",
    "PIIGuardrail",
    "PromptInjectionGuardrail",
    "SystemPromptLeakGuardrail",
    "ToxicityGuardrail",
    "LengthGuardrail",
    "TopicGuardrail",
    "ToolInputGuardrail",
    "ToolOutputGuardrail",
    "ToolPermissionGuardrail",
    "get_guardrail_preset",

    # Additional convenience exports
    "ToolFallbackChain",
    "CircuitBreaker",
    "ToolPromptGenerator",
    "AgentSystemPromptBuilder",

    # Eval
    "AgentEvaluator",
    "EvalResult",
    "SuiteResults",
    "TestCase",
    "TestSuite",
    "ModelComparison",
    "RegressionTracker",

    # Execution
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "CodeValidator",
    "ValidationResult",
    "ValidationSeverity",
    "SandboxConfig",

    # Security
    "HashDriftWarning",
    "VerificationResult",
    "verify_installed_hashes",
    "verify_on_startup",
]

if TYPE_CHECKING:
    # Make the lazy names visible to type checkers and IDEs without paying the
    # import cost at runtime.
    from effgen.config import Config, ConfigLoader, ConfigValidator
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models import BaseModel, load_model
    from effgen.tools import BaseTool

# Supply-chain integrity check — runs only when EFFGEN_VERIFY_HASHES=1. The
# env-var gate is replicated here so a normal `import effgen` never imports the
# security module (keeping startup cheap); behaviour is unchanged when opted in.
if _os.environ.get("EFFGEN_VERIFY_HASHES", "0") == "1":
    from effgen.security.supply_chain import verify_on_startup as _verify_on_startup

    _verify_on_startup()
