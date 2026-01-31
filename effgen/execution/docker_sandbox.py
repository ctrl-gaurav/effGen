"""
Docker-based sandbox for secure code execution.

This module provides Docker-based isolation for executing code with
strong security guarantees and resource limits.
"""

import os
import time
import tempfile
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import docker
    from docker.errors import DockerException, ContainerError, ImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

from .sandbox import BaseSandbox, ExecutionResult, ExecutionStatus, SandboxConfig


class DockerSandbox(BaseSandbox):
    """
    Docker-based sandbox for secure code execution.

    Provides strong isolation using Docker containers with:
    - Network isolation
    - File system isolation
    - Resource limits (CPU, memory)
    - Automatic cleanup
    """

    # Default Docker images for each language
    DEFAULT_IMAGES = {
        'python': 'python:3.11-slim',
        'javascript': 'node:20-slim',
        'bash': 'bash:5.2',
    }

    def __init__(self,
                 config: Optional[SandboxConfig] = None,
                 docker_client: Optional[Any] = None,
                 custom_images: Optional[Dict[str, str]] = None):
        """
        Initialize Docker sandbox.

        Args:
            config: Sandbox configuration
            docker_client: Pre-configured Docker client (optional)
            custom_images: Custom Docker images for languages

        Raises:
            RuntimeError: If Docker is not available
        """
        super().__init__(config)

        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "Docker is not available. Please install docker library: "
                "pip install docker"
            )

        self.docker_client = docker_client or self._create_docker_client()
        self.images = {**self.DEFAULT_IMAGES, **(custom_images or {})}
        self._ensure_images()

    def _create_docker_client(self) -> Any:
        """
        Create Docker client.

        Returns:
            Docker client instance

        Raises:
            RuntimeError: If Docker daemon is not accessible
        """
        try:
            client = docker.from_env()
            # Test connection
            client.ping()
            return client
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Docker daemon: {e}\n"
                "Please ensure Docker is installed and running."
            )

    def _ensure_images(self) -> None:
        """
        Ensure required Docker images are available.

        Pulls images if they don't exist locally.
        """
        for language, image in self.images.items():
            try:
                self.docker_client.images.get(image)
            except ImageNotFound:
                # Image not found, try to pull it
                try:
                    print(f"Pulling Docker image: {image}")
                    self.docker_client.images.pull(image)
                except Exception as e:
                    print(f"Warning: Could not pull image {image}: {e}")

    def execute(self, code: str, language: str) -> ExecutionResult:
        """
        Execute code in Docker container.

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
                error=f"Code validation failed: {[i.message for i in validation_result.issues]}",
                validation_result=validation_result,
                execution_time=time.time() - start_time
            )

        # Normalize language
        language = self._normalize_language(language)
        if language not in self.images:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Unsupported language: {language}",
                execution_time=time.time() - start_time
            )

        # Execute in Docker
        try:
            result = self._execute_in_docker(code, language)
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

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language name.

        Args:
            language: Language name

        Returns:
            Normalized language name
        """
        language = language.lower()
        if language in {'py', 'python3'}:
            return 'python'
        elif language in {'js', 'node', 'nodejs'}:
            return 'javascript'
        elif language in {'sh', 'shell'}:
            return 'bash'
        return language

    def _execute_in_docker(self, code: str, language: str) -> ExecutionResult:
        """
        Execute code in Docker container.

        Args:
            code: Code to execute
            language: Programming language

        Returns:
            ExecutionResult
        """
        image = self.images[language]
        container = None

        try:
            # Prepare code file
            with tempfile.TemporaryDirectory() as temp_dir:
                code_file = self._create_code_file(code, language, temp_dir)

                # Prepare container configuration
                container_config = self._prepare_container_config(
                    image, code_file, language, temp_dir
                )

                # Run container
                container = self.docker_client.containers.run(
                    **container_config,
                    detach=True
                )

                # Wait for completion with timeout
                try:
                    exit_code = container.wait(timeout=self.config.timeout)
                    if isinstance(exit_code, dict):
                        exit_code = exit_code.get('StatusCode', 0)
                except Exception:
                    # Timeout occurred
                    container.kill()
                    raise TimeoutError()

                # Get logs
                logs = container.logs(stdout=True, stderr=True)
                output = logs.decode('utf-8', errors='replace')

                # Separate stdout and stderr if possible
                stdout, stderr = self._parse_output(output)

                # Truncate if needed
                stdout = self._truncate_output(stdout)
                stderr = self._truncate_output(stderr)

                # Get memory stats
                stats = container.stats(stream=False)
                memory_used = stats.get('memory_stats', {}).get('usage', None)

                status = (ExecutionStatus.SUCCESS if exit_code == 0
                         else ExecutionStatus.ERROR)

                return ExecutionResult(
                    status=status,
                    output=stdout,
                    error=stderr,
                    exit_code=exit_code,
                    memory_used=memory_used
                )

        finally:
            # Clean up container
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _create_code_file(self, code: str, language: str, temp_dir: str) -> str:
        """
        Create code file in temporary directory.

        Args:
            code: Code content
            language: Programming language
            temp_dir: Temporary directory path

        Returns:
            Path to code file
        """
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'bash': '.sh'
        }
        extension = extensions.get(language, '.txt')
        code_file = os.path.join(temp_dir, f'code{extension}')

        with open(code_file, 'w') as f:
            f.write(code)

        return code_file

    def _prepare_container_config(self,
                                   image: str,
                                   code_file: str,
                                   language: str,
                                   temp_dir: str) -> Dict[str, Any]:
        """
        Prepare Docker container configuration.

        Args:
            image: Docker image name
            code_file: Path to code file
            language: Programming language
            temp_dir: Temporary directory path

        Returns:
            Container configuration dictionary
        """
        # Prepare command
        commands = {
            'python': ['python', '/code/code.py'],
            'javascript': ['node', '/code/code.js'],
            'bash': ['bash', '/code/code.sh']
        }
        command = commands.get(language, ['cat', '/code/code.txt'])

        # Prepare volumes
        volumes = {
            temp_dir: {'bind': '/code', 'mode': 'ro'}
        }

        # Prepare resource limits
        mem_limit = self.config.parse_memory_limit()
        nano_cpus = None
        if self.config.cpu_limit:
            nano_cpus = int(self.config.cpu_limit * 1e9)

        # Prepare environment
        environment = self._prepare_environment()

        # Build configuration
        config = {
            'image': image,
            'command': command,
            'volumes': volumes,
            'mem_limit': mem_limit,
            'network_mode': 'none' if not self.config.allow_network else 'bridge',
            'network_disabled': not self.config.allow_network,
            'environment': environment,
            'read_only': not self.config.allow_file_ops,
            'security_opt': ['no-new-privileges'],
            'cap_drop': ['ALL'],
            'remove': False,  # We'll remove manually after getting stats
        }

        if nano_cpus:
            config['nano_cpus'] = nano_cpus

        return config

    def _parse_output(self, output: str) -> tuple[str, str]:
        """
        Parse combined output into stdout and stderr.

        Args:
            output: Combined output string

        Returns:
            Tuple of (stdout, stderr)
        """
        # Docker combines stdout and stderr
        # We can't perfectly separate them, so we'll use heuristics
        lines = output.split('\n')
        stdout_lines = []
        stderr_lines = []

        for line in lines:
            # Common error indicators
            if any(indicator in line.lower() for indicator in
                   ['error', 'exception', 'traceback', 'warning', 'failed']):
                stderr_lines.append(line)
            else:
                stdout_lines.append(line)

        return '\n'.join(stdout_lines), '\n'.join(stderr_lines)

    def cleanup(self) -> None:
        """
        Clean up Docker resources.

        Removes stopped containers and unused volumes.
        """
        try:
            # Remove stopped containers
            containers = self.docker_client.containers.list(
                all=True,
                filters={'status': 'exited'}
            )
            for container in containers:
                try:
                    container.remove()
                except Exception:
                    pass

            # Prune volumes
            self.docker_client.volumes.prune()

        except Exception as e:
            print(f"Warning: Docker cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


class DockerManager:
    """
    Manager for Docker sandbox operations.

    Handles Docker client lifecycle, image management, and cleanup.
    """

    def __init__(self):
        """Initialize Docker manager."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker library is not available")

        self.client = None
        self._ensure_docker()

    def _ensure_docker(self) -> None:
        """
        Ensure Docker is available and running.

        Raises:
            RuntimeError: If Docker is not accessible
        """
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Docker is not available: {e}\n"
                "Please ensure Docker is installed and running."
            )

    def create_sandbox(self,
                      config: Optional[SandboxConfig] = None,
                      custom_images: Optional[Dict[str, str]] = None) -> DockerSandbox:
        """
        Create a new Docker sandbox.

        Args:
            config: Sandbox configuration
            custom_images: Custom Docker images

        Returns:
            DockerSandbox instance
        """
        return DockerSandbox(
            config=config,
            docker_client=self.client,
            custom_images=custom_images
        )

    def pull_image(self, image: str) -> None:
        """
        Pull Docker image.

        Args:
            image: Image name to pull
        """
        try:
            print(f"Pulling Docker image: {image}")
            self.client.images.pull(image)
            print(f"Successfully pulled {image}")
        except Exception as e:
            raise RuntimeError(f"Failed to pull image {image}: {e}")

    def list_images(self) -> List[str]:
        """
        List available Docker images.

        Returns:
            List of image names
        """
        try:
            images = self.client.images.list()
            return [tag for img in images for tag in img.tags]
        except Exception as e:
            print(f"Warning: Could not list images: {e}")
            return []

    def remove_image(self, image: str, force: bool = False) -> None:
        """
        Remove Docker image.

        Args:
            image: Image name to remove
            force: Force removal even if containers are using it
        """
        try:
            self.client.images.remove(image, force=force)
            print(f"Removed image: {image}")
        except Exception as e:
            raise RuntimeError(f"Failed to remove image {image}: {e}")

    def cleanup_containers(self) -> int:
        """
        Clean up stopped containers.

        Returns:
            Number of containers removed
        """
        try:
            containers = self.client.containers.list(
                all=True,
                filters={'status': 'exited'}
            )
            count = 0
            for container in containers:
                try:
                    container.remove()
                    count += 1
                except Exception:
                    pass
            return count
        except Exception as e:
            print(f"Warning: Container cleanup failed: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Docker system statistics.

        Returns:
            Dictionary with system stats
        """
        try:
            info = self.client.info()
            return {
                'containers': info.get('Containers', 0),
                'containers_running': info.get('ContainersRunning', 0),
                'containers_stopped': info.get('ContainersStopped', 0),
                'images': info.get('Images', 0),
                'memory_total': info.get('MemTotal', 0),
                'cpus': info.get('NCPU', 0),
            }
        except Exception as e:
            print(f"Warning: Could not get Docker stats: {e}")
            return {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
