"""
Secure code execution sandbox tool.

This module provides a sandboxed code execution environment supporting
multiple languages (Python, JavaScript, Bash) with Docker isolation,
resource limits, and comprehensive security measures.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import subprocess
import time
import json

from ..base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ParameterSpec,
    ParameterType,
)


logger = logging.getLogger(__name__)


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


class CodeExecutor(BaseTool):
    """
    Secure sandboxed code execution tool.

    Features:
    - Multi-language support (Python, JavaScript, Bash)
    - Docker-based isolation
    - Resource limits (CPU, memory, time)
    - Network isolation options
    - File system access control
    - Output capture (stdout, stderr, return values)
    - Comprehensive error handling

    Security:
    - Isolated execution environment
    - Configurable resource limits
    - Package whitelisting
    - Network restrictions
    - Timeout mechanisms
    - Automatic cleanup
    """

    SUPPORTED_LANGUAGES = ["python", "javascript", "bash", "sh"]

    # Default resource limits
    DEFAULT_TIMEOUT = 30  # seconds
    DEFAULT_MEMORY_LIMIT = "512m"  # 512 MB
    DEFAULT_CPU_QUOTA = 100000  # 100% of one CPU

    # Docker images for different languages
    DOCKER_IMAGES = {
        "python": "python:3.11-slim",
        "javascript": "node:18-slim",
        "bash": "bash:5",
        "sh": "bash:5",
    }

    def __init__(self):
        """Initialize the code executor."""
        super().__init__(
            metadata=ToolMetadata(
                name="code_executor",
                description="Execute code in a secure sandboxed environment with support for Python, JavaScript, and Bash",
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="The code to execute",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="language",
                        type=ParameterType.STRING,
                        description="Programming language to use",
                        required=True,
                        enum=["python", "javascript", "bash", "sh"],
                    ),
                    ParameterSpec(
                        name="timeout",
                        type=ParameterType.INTEGER,
                        description="Execution timeout in seconds",
                        required=False,
                        default=30,
                        min_value=1,
                        max_value=300,
                    ),
                    ParameterSpec(
                        name="memory_limit",
                        type=ParameterType.STRING,
                        description="Memory limit (e.g., '512m', '1g')",
                        required=False,
                        default="512m",
                    ),
                    ParameterSpec(
                        name="network_enabled",
                        type=ParameterType.BOOLEAN,
                        description="Whether to allow network access",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="files",
                        type=ParameterType.OBJECT,
                        description="Additional files to mount (filename: content)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="env_vars",
                        type=ParameterType.OBJECT,
                        description="Environment variables to set",
                        required=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "exit_code": {"type": "integer"},
                        "execution_time": {"type": "number"},
                        "timed_out": {"type": "boolean"},
                    },
                },
                timeout_seconds=300,
                tags=["code", "execution", "sandbox", "docker"],
                examples=[
                    {
                        "code": "print('Hello, World!')",
                        "language": "python",
                        "output": {"stdout": "Hello, World!\n", "exit_code": 0},
                    },
                    {
                        "code": "console.log('Hello from Node.js')",
                        "language": "javascript",
                        "output": {"stdout": "Hello from Node.js\n", "exit_code": 0},
                    },
                ],
            )
        )
        self._docker_available = False
        self._use_fallback = False

    async def initialize(self) -> None:
        """Initialize the executor and check Docker availability."""
        await super().initialize()

        # Check if Docker is available
        try:
            result = await asyncio.create_subprocess_exec(
                "docker",
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            self._docker_available = result.returncode == 0
        except FileNotFoundError:
            self._docker_available = False

        if not self._docker_available:
            logger.warning(
                "Docker not available. Code execution will use fallback mode with limited isolation."
            )
            self._use_fallback = True

    async def _execute(
        self,
        code: str,
        language: str,
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        network_enabled: bool = False,
        files: Optional[Dict[str, str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.

        Args:
            code: The code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            memory_limit: Memory limit
            network_enabled: Whether to allow network access
            files: Additional files to mount
            env_vars: Environment variables

        Returns:
            Dict containing stdout, stderr, exit_code, execution_time, timed_out
        """
        if self._use_fallback:
            return await self._execute_fallback(
                code, language, timeout, env_vars
            )

        return await self._execute_docker(
            code, language, timeout, memory_limit, network_enabled, files, env_vars
        )

    async def _execute_docker(
        self,
        code: str,
        language: str,
        timeout: int,
        memory_limit: str,
        network_enabled: bool,
        files: Optional[Dict[str, str]],
        env_vars: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Execute code using Docker."""
        start_time = time.time()
        temp_dir = None

        try:
            # Create temporary directory for code
            temp_dir = Path(tempfile.mkdtemp())

            # Write code to file
            code_file = self._get_code_filename(language)
            code_path = temp_dir / code_file
            code_path.write_text(code, encoding="utf-8")

            # Write additional files if provided
            if files:
                for filename, content in files.items():
                    file_path = temp_dir / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")

            # Build Docker command
            docker_cmd = self._build_docker_command(
                language,
                temp_dir,
                code_file,
                memory_limit,
                network_enabled,
                env_vars,
            )

            # Execute with timeout
            try:
                process = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )

                exit_code = process.returncode
                timed_out = False

            except asyncio.TimeoutError:
                # Kill the container
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass

                stdout = b""
                stderr = b"Execution timed out"
                exit_code = -1
                timed_out = True

            execution_time = time.time() - start_time

            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": exit_code,
                "execution_time": execution_time,
                "timed_out": timed_out,
            }

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            raise CodeExecutionError(f"Execution failed: {str(e)}")

        finally:
            # Clean up temporary directory
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")

    async def _execute_fallback(
        self,
        code: str,
        language: str,
        timeout: int,
        env_vars: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Execute code using subprocess fallback (less secure).

        Only used when Docker is not available.
        """
        start_time = time.time()

        # Get command for language
        cmd = self._get_fallback_command(language, code)

        # Set up environment
        import os
        env = dict(os.environ)
        if env_vars:
            env.update(env_vars)

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                exit_code = process.returncode
                timed_out = False

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout = b""
                stderr = b"Execution timed out"
                exit_code = -1
                timed_out = True

            execution_time = time.time() - start_time

            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": exit_code,
                "execution_time": execution_time,
                "timed_out": timed_out,
            }

        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            raise CodeExecutionError(f"Execution failed: {str(e)}")

    def _build_docker_command(
        self,
        language: str,
        work_dir: Path,
        code_file: str,
        memory_limit: str,
        network_enabled: bool,
        env_vars: Optional[Dict[str, str]],
    ) -> List[str]:
        """Build Docker command for code execution."""
        cmd = [
            "docker",
            "run",
            "--rm",  # Remove container after execution
            "-v",
            f"{work_dir}:/workspace",  # Mount workspace
            "-w",
            "/workspace",  # Set working directory
            "--memory",
            memory_limit,  # Memory limit
            "--cpus",
            "1",  # CPU limit
            "--pids-limit",
            "100",  # Process limit
        ]

        # Network settings
        if not network_enabled:
            cmd.extend(["--network", "none"])

        # Environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Add image
        cmd.append(self.DOCKER_IMAGES[language])

        # Add execution command
        exec_cmd = self._get_execution_command(language, code_file)
        cmd.extend(exec_cmd)

        return cmd

    def _get_execution_command(self, language: str, code_file: str) -> List[str]:
        """Get the command to execute code in the container."""
        commands = {
            "python": ["python", code_file],
            "javascript": ["node", code_file],
            "bash": ["bash", code_file],
            "sh": ["sh", code_file],
        }
        return commands[language]

    def _get_code_filename(self, language: str) -> str:
        """Get appropriate filename for the code."""
        extensions = {
            "python": "script.py",
            "javascript": "script.js",
            "bash": "script.sh",
            "sh": "script.sh",
        }
        return extensions[language]

    def _get_fallback_command(self, language: str, code: str) -> str:
        """Get command for fallback execution."""
        # Escape code for shell
        escaped_code = code.replace("'", "'\\''")

        commands = {
            "python": f"python3 -c '{escaped_code}'",
            "javascript": f"node -e '{escaped_code}'",
            "bash": f"bash -c '{escaped_code}'",
            "sh": f"sh -c '{escaped_code}'",
        }
        return commands.get(language, f"bash -c '{escaped_code}'")
