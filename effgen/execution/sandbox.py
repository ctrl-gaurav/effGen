"""
Secure code execution sandbox with multi-language support.

This module provides a secure sandbox environment for executing code
with resource limits, security validation, and output capture.
"""

import os
import time
import signal
import tempfile
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from pathlib import Path

from .validators import CodeValidator, ValidationResult


class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time: float = 0.0
    memory_used: Optional[int] = None
    exit_code: int = 0
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "execution_time": self.execution_time,
            "memory_used": self.memory_used,
            "exit_code": self.exit_code,
            "metadata": self.metadata
        }


class SandboxConfig:
    """Configuration for sandbox execution."""

    def __init__(self,
                 timeout: int = 30,
                 memory_limit: str = "512M",
                 cpu_limit: Optional[float] = None,
                 allow_network: bool = False,
                 allow_file_ops: bool = False,
                 max_output_size: int = 1024 * 1024,  # 1MB
                 working_dir: Optional[str] = None,
                 env_vars: Optional[Dict[str, str]] = None,
                 custom_allow_imports: Optional[set] = None):
        """
        Initialize sandbox configuration.

        Args:
            timeout: Maximum execution time in seconds
            memory_limit: Maximum memory (e.g., "512M", "1G")
            cpu_limit: CPU limit as fraction (e.g., 0.5 = 50% of one CPU)
            allow_network: Whether to allow network access
            allow_file_ops: Whether to allow file operations
            max_output_size: Maximum output size in bytes
            working_dir: Working directory for execution
            env_vars: Environment variables to set
            custom_allow_imports: Custom set of allowed Python imports
        """
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.allow_network = allow_network
        self.allow_file_ops = allow_file_ops
        self.max_output_size = max_output_size
        self.working_dir = working_dir or tempfile.gettempdir()
        self.env_vars = env_vars or {}
        self.custom_allow_imports = custom_allow_imports

    def parse_memory_limit(self) -> int:
        """
        Parse memory limit string to bytes.

        Returns:
            Memory limit in bytes
        """
        limit = self.memory_limit.upper()
        if limit.endswith('K'):
            return int(limit[:-1]) * 1024
        elif limit.endswith('M'):
            return int(limit[:-1]) * 1024 * 1024
        elif limit.endswith('G'):
            return int(limit[:-1]) * 1024 * 1024 * 1024
        else:
            return int(limit)


class BaseSandbox(ABC):
    """
    Abstract base class for code execution sandboxes.

    All sandbox implementations must inherit from this class
    and implement the execute method.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize sandbox.

        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()
        self.validator = CodeValidator(
            allow_network=self.config.allow_network,
            allow_file_ops=self.config.allow_file_ops,
            custom_allow_imports=self.config.custom_allow_imports
        )

    @abstractmethod
    def execute(self, code: str, language: str) -> ExecutionResult:
        """
        Execute code in the sandbox.

        Args:
            code: Code to execute
            language: Programming language

        Returns:
            ExecutionResult with output and status
        """
        pass

    def validate_code(self, code: str, language: str) -> ValidationResult:
        """
        Validate code before execution.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            ValidationResult with any issues found
        """
        return self.validator.validate(code, language)

    def _prepare_environment(self) -> Dict[str, str]:
        """
        Prepare environment variables for execution.

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        env.update(self.config.env_vars)

        # Add security-related environment variables
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONUNBUFFERED'] = '1'

        return env

    def _truncate_output(self, output: str) -> str:
        """
        Truncate output if it exceeds maximum size.

        Args:
            output: Output string

        Returns:
            Truncated output if necessary
        """
        if len(output) > self.config.max_output_size:
            truncated_size = self.config.max_output_size - 100
            return (output[:truncated_size] +
                    f"\n... [Output truncated at {self.config.max_output_size} bytes]")
        return output


class LocalSandbox(BaseSandbox):
    """
    Local process-based sandbox for code execution.

    This sandbox executes code in a subprocess with resource limits.
    Less secure than Docker-based sandbox but simpler to set up.
    """

    def execute(self, code: str, language: str) -> ExecutionResult:
        """
        Execute code in local subprocess.

        Args:
            code: Code to execute
            language: Programming language

        Returns:
            ExecutionResult with output and status
        """
        start_time = time.time()

        # Validate code first
        validation_result = self.validate_code(code, language)
        if not validation_result.is_safe:
            return ExecutionResult(
                status=ExecutionStatus.VALIDATION_FAILED,
                error=f"Code validation failed: {validation_result.issues}",
                validation_result=validation_result,
                execution_time=time.time() - start_time
            )

        # Execute based on language
        try:
            if language.lower() in {'python', 'py'}:
                result = self._execute_python(code)
            elif language.lower() in {'javascript', 'js', 'node'}:
                result = self._execute_javascript(code)
            elif language.lower() in {'bash', 'sh', 'shell'}:
                result = self._execute_bash(code)
            else:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=f"Unsupported language: {language}",
                    execution_time=time.time() - start_time
                )

            result.execution_time = time.time() - start_time
            result.validation_result = validation_result
            return result

        except TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {self.config.timeout} seconds",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Execution error: {str(e)}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time
            )

    def _execute_python(self, code: str) -> ExecutionResult:
        """
        Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult
        """
        import subprocess
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute with subprocess
            env = self._prepare_environment()
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir
            )

            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout)
                output = stdout.decode('utf-8', errors='replace')
                error = stderr.decode('utf-8', errors='replace')

                output = self._truncate_output(output)
                error = self._truncate_output(error)

                status = (ExecutionStatus.SUCCESS if process.returncode == 0
                         else ExecutionStatus.ERROR)

                return ExecutionResult(
                    status=status,
                    output=output,
                    error=error,
                    exit_code=process.returncode
                )

            except subprocess.TimeoutExpired:
                process.kill()
                raise TimeoutError()

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _execute_javascript(self, code: str) -> ExecutionResult:
        """
        Execute JavaScript code using Node.js.

        Args:
            code: JavaScript code to execute

        Returns:
            ExecutionResult
        """
        import subprocess
        import shutil

        # Check if Node.js is available
        if not shutil.which('node'):
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error="Node.js is not installed or not in PATH"
            )

        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            env = self._prepare_environment()
            process = subprocess.Popen(
                ['node', temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir
            )

            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout)
                output = stdout.decode('utf-8', errors='replace')
                error = stderr.decode('utf-8', errors='replace')

                output = self._truncate_output(output)
                error = self._truncate_output(error)

                status = (ExecutionStatus.SUCCESS if process.returncode == 0
                         else ExecutionStatus.ERROR)

                return ExecutionResult(
                    status=status,
                    output=output,
                    error=error,
                    exit_code=process.returncode
                )

            except subprocess.TimeoutExpired:
                process.kill()
                raise TimeoutError()

        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _execute_bash(self, code: str) -> ExecutionResult:
        """
        Execute Bash script.

        Args:
            code: Bash script to execute

        Returns:
            ExecutionResult
        """
        import subprocess

        try:
            env = self._prepare_environment()
            process = subprocess.Popen(
                ['bash', '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir
            )

            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout)
                output = stdout.decode('utf-8', errors='replace')
                error = stderr.decode('utf-8', errors='replace')

                output = self._truncate_output(output)
                error = self._truncate_output(error)

                status = (ExecutionStatus.SUCCESS if process.returncode == 0
                         else ExecutionStatus.ERROR)

                return ExecutionResult(
                    status=status,
                    output=output,
                    error=error,
                    exit_code=process.returncode
                )

            except subprocess.TimeoutExpired:
                process.kill()
                raise TimeoutError()

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Bash execution error: {str(e)}"
            )


class CodeExecutor:
    """
    High-level code executor that manages sandbox selection and execution.

    This is the main interface for executing code in effGen.
    """

    def __init__(self,
                 sandbox_type: str = "local",
                 config: Optional[SandboxConfig] = None):
        """
        Initialize code executor.

        Args:
            sandbox_type: Type of sandbox ("local" or "docker")
            config: Sandbox configuration

        Raises:
            ValueError: If sandbox type is not supported
        """
        self.sandbox_type = sandbox_type
        self.config = config or SandboxConfig()

        # Create sandbox instance
        if sandbox_type == "local":
            self.sandbox = LocalSandbox(self.config)
        elif sandbox_type == "docker":
            # Import here to avoid circular dependency
            from .docker_sandbox import DockerSandbox
            self.sandbox = DockerSandbox(self.config)
        else:
            raise ValueError(f"Unsupported sandbox type: {sandbox_type}")

    def execute(self,
                code: str,
                language: str = "python",
                timeout: Optional[int] = None,
                memory_limit: Optional[str] = None,
                allow_network: Optional[bool] = None) -> ExecutionResult:
        """
        Execute code in sandbox.

        Args:
            code: Code to execute
            language: Programming language
            timeout: Override timeout for this execution
            memory_limit: Override memory limit for this execution
            allow_network: Override network allowance for this execution

        Returns:
            ExecutionResult with output and status
        """
        # Create temporary config if overrides are provided
        if any(x is not None for x in [timeout, memory_limit, allow_network]):
            config = SandboxConfig(
                timeout=timeout or self.config.timeout,
                memory_limit=memory_limit or self.config.memory_limit,
                allow_network=allow_network if allow_network is not None else self.config.allow_network,
                allow_file_ops=self.config.allow_file_ops,
                max_output_size=self.config.max_output_size,
                working_dir=self.config.working_dir,
                env_vars=self.config.env_vars
            )
            # Temporarily update sandbox config
            original_config = self.sandbox.config
            self.sandbox.config = config
            try:
                result = self.sandbox.execute(code, language)
            finally:
                self.sandbox.config = original_config
        else:
            result = self.sandbox.execute(code, language)

        return result

    def validate(self, code: str, language: str) -> ValidationResult:
        """
        Validate code without executing.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            ValidationResult with any issues found
        """
        return self.sandbox.validate_code(code, language)

    def is_safe(self, code: str, language: str) -> bool:
        """
        Quick check if code is safe to execute.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            True if code is safe, False otherwise
        """
        validation = self.validate(code, language)
        return validation.is_safe

    def execute_with_retry(self,
                          code: str,
                          language: str = "python",
                          max_retries: int = 3,
                          **kwargs) -> ExecutionResult:
        """
        Execute code with automatic retry on transient failures.

        Args:
            code: Code to execute
            language: Programming language
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments passed to execute()

        Returns:
            ExecutionResult from successful execution or last failure
        """
        last_result = None

        for attempt in range(max_retries):
            result = self.execute(code, language, **kwargs)

            if result.success:
                return result

            # Check if error is retryable
            if result.status == ExecutionStatus.TIMEOUT:
                # Timeout errors are not retryable
                return result

            if result.status == ExecutionStatus.VALIDATION_FAILED:
                # Validation errors are not retryable
                return result

            last_result = result

            # Add retry metadata
            if last_result.metadata is None:
                last_result.metadata = {}
            last_result.metadata['retry_attempt'] = attempt + 1

            # Brief delay before retry
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))

        return last_result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "sandbox_type": self.sandbox_type,
            "config": {
                "timeout": self.config.timeout,
                "memory_limit": self.config.memory_limit,
                "allow_network": self.config.allow_network,
                "allow_file_ops": self.config.allow_file_ops,
            }
        }


class ExecutionPool:
    """
    Pool for managing multiple concurrent code executions.

    Useful for batch processing or parallel execution of multiple code snippets.
    """

    def __init__(self,
                 sandbox_type: str = "local",
                 config: Optional[SandboxConfig] = None,
                 pool_size: int = 4):
        """
        Initialize execution pool.

        Args:
            sandbox_type: Type of sandbox to use
            config: Sandbox configuration
            pool_size: Maximum number of concurrent executions
        """
        self.sandbox_type = sandbox_type
        self.config = config or SandboxConfig()
        self.pool_size = pool_size

        # Create executor pool
        self.executors = [
            CodeExecutor(sandbox_type, config)
            for _ in range(pool_size)
        ]
        self.available_executors = list(range(pool_size))
        self.busy_executors = set()

    def execute(self,
               code: str,
               language: str = "python",
               **kwargs) -> ExecutionResult:
        """
        Execute code using an available executor from the pool.

        Args:
            code: Code to execute
            language: Programming language
            **kwargs: Additional execution arguments

        Returns:
            ExecutionResult
        """
        # Get available executor
        if not self.available_executors:
            # Wait for an executor to become available
            time.sleep(0.1)
            if not self.available_executors:
                raise RuntimeError("No executors available in pool")

        executor_idx = self.available_executors.pop(0)
        self.busy_executors.add(executor_idx)

        try:
            executor = self.executors[executor_idx]
            result = executor.execute(code, language, **kwargs)
            return result
        finally:
            # Return executor to pool
            self.busy_executors.remove(executor_idx)
            self.available_executors.append(executor_idx)

    def execute_batch(self,
                     code_snippets: List[Tuple[str, str]],
                     **kwargs) -> List[ExecutionResult]:
        """
        Execute multiple code snippets.

        Args:
            code_snippets: List of (code, language) tuples
            **kwargs: Additional execution arguments

        Returns:
            List of ExecutionResult objects
        """
        results = []

        for code, language in code_snippets:
            result = self.execute(code, language, **kwargs)
            results.append(result)

        return results

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get pool status.

        Returns:
            Dictionary with pool status information
        """
        return {
            "pool_size": self.pool_size,
            "available": len(self.available_executors),
            "busy": len(self.busy_executors),
            "utilization": len(self.busy_executors) / self.pool_size
        }


class ExecutionHistory:
    """
    Track execution history for debugging and analysis.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize execution history.

        Args:
            max_history: Maximum number of executions to track
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0

    def record(self,
              code: str,
              language: str,
              result: ExecutionResult) -> None:
        """
        Record an execution.

        Args:
            code: Executed code
            language: Programming language
            result: Execution result
        """
        entry = {
            "id": self.execution_count,
            "timestamp": time.time(),
            "code": code,
            "language": language,
            "status": result.status.value,
            "execution_time": result.execution_time,
            "success": result.success,
            "output_length": len(result.output),
            "error_length": len(result.error)
        }

        self.history.append(entry)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Update statistics
        self.execution_count += 1
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_execution_time += result.execution_time

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_executions": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0,
            "average_execution_time": self.total_execution_time / self.execution_count if self.execution_count > 0 else 0,
            "history_size": len(self.history)
        }

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent executions.

        Args:
            n: Number of recent executions to return

        Returns:
            List of execution records
        """
        return self.history[-n:]

    def clear(self) -> None:
        """Clear execution history."""
        self.history.clear()
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0


class CodeExecutorWithHistory(CodeExecutor):
    """
    Code executor that automatically tracks execution history.
    """

    def __init__(self,
                 sandbox_type: str = "local",
                 config: Optional[SandboxConfig] = None,
                 max_history: int = 1000):
        """
        Initialize executor with history tracking.

        Args:
            sandbox_type: Type of sandbox
            config: Sandbox configuration
            max_history: Maximum history entries to keep
        """
        super().__init__(sandbox_type, config)
        self.history = ExecutionHistory(max_history)

    def execute(self,
               code: str,
               language: str = "python",
               **kwargs) -> ExecutionResult:
        """
        Execute code and record to history.

        Args:
            code: Code to execute
            language: Programming language
            **kwargs: Additional execution arguments

        Returns:
            ExecutionResult
        """
        result = super().execute(code, language, **kwargs)
        self.history.record(code, language, result)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.

        Returns:
            Dictionary with executor and history statistics
        """
        stats = super().get_statistics()
        stats.update(self.history.get_statistics())
        return stats
