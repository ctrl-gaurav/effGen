"""
Execution and sandbox systems for effGen.

This package provides secure code execution with Docker isolation,
multi-language support, resource limits, and security validation.
"""

from .sandbox import (
    CodeExecutor,
    BaseSandbox,
    LocalSandbox,
    SandboxConfig,
    ExecutionResult,
    ExecutionStatus,
    ExecutionPool,
    ExecutionHistory,
    CodeExecutorWithHistory
)

from .docker_sandbox import (
    DockerSandbox,
    DockerManager,
    DOCKER_AVAILABLE
)

from .validators import (
    CodeValidator,
    PythonValidator,
    JavaScriptValidator,
    BashValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity
)

__all__ = [
    # Sandbox
    "CodeExecutor",
    "BaseSandbox",
    "LocalSandbox",
    "SandboxConfig",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionPool",
    "ExecutionHistory",
    "CodeExecutorWithHistory",

    # Docker sandbox
    "DockerSandbox",
    "DockerManager",
    "DOCKER_AVAILABLE",

    # Validators
    "CodeValidator",
    "PythonValidator",
    "JavaScriptValidator",
    "BashValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
]
