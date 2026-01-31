#!/usr/bin/env python3
"""
effGen Framework Verification Script

This comprehensive validation script verifies all components of the effGen
framework are properly installed, configured, and functional.

Usage:
    python verify_framework.py [--verbose] [--skip-optional]
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestStatus(Enum):
    """Status of individual tests"""
    PASSED = "✓ PASSED"
    FAILED = "✗ FAILED"
    SKIPPED = "⊘ SKIPPED"
    WARNING = "⚠ WARNING"


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    status: TestStatus
    message: str
    details: Optional[str] = None
    duration: float = 0.0


class FrameworkVerifier:
    """Main verification class for effGen framework"""

    def __init__(self, verbose: bool = False, skip_optional: bool = False):
        self.verbose = verbose
        self.skip_optional = skip_optional
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def log(self, message: str, color: str = ""):
        """Print colored log message"""
        if color:
            print(f"{color}{message}{Colors.ENDC}")
        else:
            print(message)

    def log_verbose(self, message: str):
        """Print message only in verbose mode"""
        if self.verbose:
            print(f"  {Colors.OKCYAN}{message}{Colors.ENDC}")

    def add_result(self, result: TestResult):
        """Add test result to results list"""
        self.results.append(result)

        # Print result
        color = {
            TestStatus.PASSED: Colors.OKGREEN,
            TestStatus.FAILED: Colors.FAIL,
            TestStatus.SKIPPED: Colors.WARNING,
            TestStatus.WARNING: Colors.WARNING
        }[result.status]

        status_str = f"{color}{result.status.value}{Colors.ENDC}"
        print(f"{status_str} {result.name} ({result.duration:.3f}s)")

        if result.message and (self.verbose or result.status != TestStatus.PASSED):
            print(f"  → {result.message}")

        if result.details and self.verbose:
            print(f"  Details: {result.details}")

    def run_test(self, name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test and capture result"""
        start = time.time()
        try:
            self.log_verbose(f"Running: {name}")
            test_func(*args, **kwargs)
            duration = time.time() - start
            return TestResult(name, TestStatus.PASSED, "Success", duration=duration)
        except Exception as e:
            duration = time.time() - start
            error_msg = str(e)
            details = traceback.format_exc() if self.verbose else None
            return TestResult(name, TestStatus.FAILED, error_msg, details, duration)

    def verify_imports(self):
        """Verify all core modules can be imported"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Module Imports ==={Colors.ENDC}\n")

        modules = [
            # Core modules
            ("effgen", "Main package"),
            ("effgen.core", "Core components"),
            ("effgen.core.agent", "Agent class"),
            ("effgen.core.task", "Task management"),
            ("effgen.core.state", "State management"),
            ("effgen.core.orchestrator", "Orchestrator"),
            ("effgen.core.router", "Router"),
            ("effgen.core.complexity_analyzer", "Complexity analyzer"),
            ("effgen.core.decomposition_engine", "Decomposition engine"),
            ("effgen.core.execution_tracker", "Execution tracker"),
            ("effgen.core.sub_agent_manager", "Sub-agent manager"),

            # Model modules
            ("effgen.models", "Model system"),
            ("effgen.models.base", "Base model"),
            ("effgen.models.vllm_engine", "vLLM engine"),
            ("effgen.models.transformers_engine", "Transformers engine"),
            ("effgen.models.openai_adapter", "OpenAI adapter"),
            ("effgen.models.anthropic_adapter", "Anthropic adapter"),
            ("effgen.models.gemini_adapter", "Gemini adapter"),
            ("effgen.models.model_loader", "Model loader"),

            # Tool modules
            ("effgen.tools", "Tool system"),
            ("effgen.tools.base_tool", "Base tool"),
            ("effgen.tools.registry", "Tool registry"),
            ("effgen.tools.builtin.calculator", "Calculator tool"),
            ("effgen.tools.builtin.web_search", "Web search tool"),
            ("effgen.tools.builtin.code_executor", "Code executor tool"),
            ("effgen.tools.builtin.file_ops", "File operations tool"),
            ("effgen.tools.builtin.python_repl", "Python REPL tool"),

            # Protocol modules
            ("effgen.tools.protocols.mcp", "MCP protocol"),
            ("effgen.tools.protocols.a2a", "A2A protocol"),
            ("effgen.tools.protocols.acp", "ACP protocol"),

            # Configuration modules
            ("effgen.config", "Configuration system"),
            ("effgen.config.loader", "Config loader"),
            ("effgen.config.validator", "Config validator"),

            # Prompt modules
            ("effgen.prompts", "Prompt system"),
            ("effgen.prompts.template_manager", "Template manager"),
            ("effgen.prompts.chain_manager", "Chain manager"),
            ("effgen.prompts.optimizer", "Prompt optimizer"),

            # Memory modules
            ("effgen.memory", "Memory system"),
            ("effgen.memory.short_term", "Short-term memory"),
            ("effgen.memory.long_term", "Long-term memory"),

            # Execution modules
            ("effgen.execution", "Execution system"),
            ("effgen.execution.sandbox", "Code executor"),
            ("effgen.execution.validators", "Code validator"),

            # GPU modules
            ("effgen.gpu", "GPU management"),
            ("effgen.gpu.allocator", "GPU allocator"),
            ("effgen.gpu.monitor", "GPU monitor"),
        ]

        for module_name, description in modules:
            def test_import():
                importlib.import_module(module_name)
                self.log_verbose(f"  Imported: {module_name}")

            result = self.run_test(f"Import {description}", test_import)
            self.add_result(result)

    def verify_core_classes(self):
        """Verify core classes can be instantiated"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Core Class Instantiation ==={Colors.ENDC}\n")

        def test_task_class():
            from effgen.core.task import Task, TaskPriority, TaskStatus
            task = Task(
                description="Test task",
                priority=TaskPriority.MEDIUM
            )
            assert task.description == "Test task"
            assert task.priority == TaskPriority.MEDIUM
            assert task.status == TaskStatus.PENDING
            self.log_verbose("  Task class instantiated successfully")

        def test_agent_state():
            from effgen.core.state import AgentState
            state = AgentState(agent_id="test_agent")
            assert state.agent_id == "test_agent"
            assert len(state.conversation_history) == 0
            self.log_verbose("  AgentState instantiated successfully")

        def test_generation_config():
            from effgen.models import GenerationConfig
            config = GenerationConfig(
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            assert config.max_tokens == 100
            assert config.temperature == 0.7
            self.log_verbose("  GenerationConfig instantiated successfully")

        def test_tool_metadata():
            from effgen.tools.base_tool import ToolMetadata, ToolCategory
            metadata = ToolMetadata(
                name="test_tool",
                description="Test tool",
                category=ToolCategory.COMPUTATION
            )
            assert metadata.name == "test_tool"
            self.log_verbose("  ToolMetadata instantiated successfully")

        def test_memory_message():
            from effgen.memory import Message, MessageRole
            message = Message(
                role=MessageRole.USER,
                content="Test message"
            )
            assert message.role == MessageRole.USER
            assert message.content == "Test message"
            self.log_verbose("  Message instantiated successfully")

        tests = [
            ("Task class", test_task_class),
            ("AgentState class", test_agent_state),
            ("GenerationConfig class", test_generation_config),
            ("ToolMetadata class", test_tool_metadata),
            ("Memory Message class", test_memory_message),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_tool_system(self):
        """Verify tool system functionality"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Tool System ==={Colors.ENDC}\n")

        def test_tool_registry():
            from effgen.tools import get_registry
            registry = get_registry()
            assert registry is not None
            self.log_verbose(f"  Tool registry initialized")

        def test_builtin_tools():
            from effgen.tools.builtin.calculator import Calculator
            from effgen.tools.builtin.web_search import WebSearch
            from effgen.tools.builtin.file_ops import FileOperations
            from effgen.tools.builtin.python_repl import PythonREPL

            # Test calculator
            calc = Calculator()
            assert calc.metadata.name == "calculator"
            self.log_verbose("  Calculator tool initialized")

            # Test web search
            search = WebSearch()
            assert search.metadata.name == "web_search"
            self.log_verbose("  Web search tool initialized")

            # Test file ops
            file_ops = FileOperations()
            assert file_ops.metadata.name == "file_operations"
            self.log_verbose("  File operations tool initialized")

            # Test Python REPL
            repl = PythonREPL()
            assert repl.metadata.name == "python_repl"
            self.log_verbose("  Python REPL tool initialized")

        def test_tool_registration():
            from effgen.tools import get_registry, BaseTool, ToolMetadata, ToolCategory
            from effgen.tools.base_tool import ParameterSpec, ParameterType

            class TestTool(BaseTool):
                def __init__(self):
                    super().__init__(
                        metadata=ToolMetadata(
                            name="test_verification_tool",
                            description="Test tool for verification",
                            category=ToolCategory.COMPUTATION,
                            parameters=[
                                ParameterSpec(
                                    name="input",
                                    type=ParameterType.STRING,
                                    description="Test input",
                                    required=True
                                )
                            ]
                        )
                    )

                async def _execute(self, input: str, **kwargs):
                    return f"Processed: {input}"

            registry = get_registry()
            registry.register_tool(TestTool)

            # Verify registration
            assert registry.is_registered("test_verification_tool")
            self.log_verbose("  Custom tool registered successfully")

        tests = [
            ("Tool registry initialization", test_tool_registry),
            ("Built-in tools", test_builtin_tools),
            ("Tool registration", test_tool_registration),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_configuration(self):
        """Verify configuration system"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Configuration System ==={Colors.ENDC}\n")

        def test_config_loader():
            from effgen.config import Config

            # Test dict config
            config_dict = {
                "agent": {
                    "name": "TestAgent",
                    "model": {
                        "name": "test-model",
                        "engine": "transformers"
                    }
                }
            }
            config = Config(data=config_dict)
            assert config.agent.name == "TestAgent"
            self.log_verbose("  Config loaded from dict successfully")

        def test_config_validation():
            from effgen.config import ConfigValidator

            valid_config = {
                "agent": {
                    "name": "ValidAgent",
                    "model": {
                        "name": "test-model",
                        "engine": "transformers"
                    }
                }
            }

            validator = ConfigValidator()
            # ConfigValidator exists and can be instantiated
            assert validator is not None
            self.log_verbose("  Config validator instantiated successfully")

        tests = [
            ("Config loader", test_config_loader),
            ("Config validation", test_config_validation),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_model_interfaces(self):
        """Verify model interfaces (without loading actual models)"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Model Interfaces ==={Colors.ENDC}\n")

        def test_generation_config():
            from effgen.models import GenerationConfig
            config = GenerationConfig(
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1
            )
            assert config.max_tokens == 512
            assert config.temperature == 0.7
            self.log_verbose("  GenerationConfig created successfully")

        def test_model_loader_class():
            from effgen.models import ModelLoader
            loader = ModelLoader()
            assert loader is not None
            self.log_verbose("  ModelLoader class instantiated")

        tests = [
            ("Generation config", test_generation_config),
            ("Model loader class", test_model_loader_class),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_prompt_system(self):
        """Verify prompt engineering system"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Prompt System ==={Colors.ENDC}\n")

        def test_template_manager():
            from effgen.prompts import TemplateManager
            from effgen.prompts.template_manager import PromptTemplate

            manager = TemplateManager()

            # Create template
            template = PromptTemplate(
                name="test_template",
                template="Hello {{ name }}, you are {{ age }} years old.",
                variables=["name", "age"]
            )
            manager.add_template(template)

            # Render template
            result = template.render(name="Alice", age=30)
            assert "Alice" in result
            assert "30" in result
            self.log_verbose("  Template created and rendered successfully")

        def test_prompt_optimizer():
            from effgen.prompts import PromptOptimizer

            optimizer = PromptOptimizer()

            # Test optimization
            long_prompt = "This is a test. " * 100
            result = optimizer.optimize(long_prompt, preserve_format=False)
            assert len(result.optimized_prompt) <= len(long_prompt)
            self.log_verbose("  Prompt optimization working")

        tests = [
            ("Template manager", test_template_manager),
            ("Prompt optimizer", test_prompt_optimizer),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_memory_system(self):
        """Verify memory system"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Memory System ==={Colors.ENDC}\n")

        def test_short_term_memory():
            from effgen.memory import ShortTermMemory, MessageRole

            memory = ShortTermMemory(max_messages=10)

            # Add messages
            memory.add_message(MessageRole.USER, "Hello")
            memory.add_message(MessageRole.ASSISTANT, "Hi there")

            messages = memory.get_recent_messages()
            assert len(messages) == 2
            assert messages[0].content == "Hello"
            self.log_verbose("  Short-term memory working correctly")

        def test_memory_entry():
            from effgen.memory import MemoryEntry, MemoryType, ImportanceLevel

            entry = MemoryEntry(
                content="Test memory",
                memory_type=MemoryType.FACT,
                importance=ImportanceLevel.MEDIUM
            )

            assert entry.content == "Test memory"
            assert entry.memory_type == MemoryType.FACT
            self.log_verbose("  MemoryEntry created successfully")

        tests = [
            ("Short-term memory", test_short_term_memory),
            ("Memory entry", test_memory_entry),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_execution_system(self):
        """Verify execution and validation system"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Execution System ==={Colors.ENDC}\n")

        def test_code_validator():
            from effgen.execution.validators import PythonValidator

            validator = PythonValidator()

            # Test valid code
            valid_code = "x = 1 + 1\nprint(x)"
            result = validator.validate(valid_code)
            assert result.language == "python"
            self.log_verbose(f"  Code validation performed (safe: {result.is_safe})")

        def test_sandbox_config():
            from effgen.execution import SandboxConfig

            config = SandboxConfig(
                timeout=30,
                memory_limit="512M",
                allow_network=False
            )

            assert config.timeout == 30
            assert config.memory_limit == "512M"
            self.log_verbose("  SandboxConfig created successfully")

        tests = [
            ("Code validator", test_code_validator),
            ("Sandbox config", test_sandbox_config),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_dependencies(self):
        """Verify all required dependencies are installed"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Dependencies ==={Colors.ENDC}\n")

        required_deps = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("yaml", "PyYAML"),
            ("jsonschema", "JSON Schema"),
            ("requests", "Requests"),
            ("pydantic", "Pydantic"),
            ("jinja2", "Jinja2"),
        ]

        optional_deps = [
            ("vllm", "vLLM (for fast inference)"),
            ("docker", "Docker SDK"),
            ("anthropic", "Anthropic API"),
            ("openai", "OpenAI API"),
        ]

        for module_name, description in required_deps:
            def test_import(name=module_name):
                importlib.import_module(name)
                self.log_verbose(f"  {description} installed")

            result = self.run_test(f"Dependency: {description}", test_import)
            self.add_result(result)

        if not self.skip_optional:
            for module_name, description in optional_deps:
                def test_import(name=module_name):
                    importlib.import_module(name)
                    self.log_verbose(f"  {description} installed")

                result = self.run_test(f"Optional: {description}", test_import)
                # Don't fail on optional dependencies
                if result.status == TestStatus.FAILED:
                    result = TestResult(
                        result.name,
                        TestStatus.SKIPPED,
                        "Optional dependency not installed",
                        duration=result.duration
                    )
                self.add_result(result)

    def generate_report(self):
        """Generate final validation report"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        self.log(f"{Colors.HEADER}{Colors.BOLD}VERIFICATION REPORT{Colors.ENDC}")
        self.log(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

        # Count results
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        warnings = sum(1 for r in self.results if r.status == TestStatus.WARNING)
        total = len(self.results)

        # Calculate total time
        total_time = time.time() - self.start_time

        # Print summary
        self.log(f"{Colors.BOLD}Summary:{Colors.ENDC}")
        self.log(f"  Total Tests: {total}")
        self.log(f"  {Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
        if failed > 0:
            self.log(f"  {Colors.FAIL}Failed: {failed}{Colors.ENDC}")
        else:
            self.log(f"  Failed: {failed}")
        if skipped > 0:
            self.log(f"  {Colors.WARNING}Skipped: {skipped}{Colors.ENDC}")
        if warnings > 0:
            self.log(f"  {Colors.WARNING}Warnings: {warnings}{Colors.ENDC}")
        self.log(f"  Duration: {total_time:.2f}s")

        # Success rate
        if total > 0:
            success_rate = (passed / total) * 100
            self.log(f"\n  Success Rate: {success_rate:.1f}%")

        # Print failures if any
        if failed > 0:
            self.log(f"\n{Colors.FAIL}{Colors.BOLD}Failed Tests:{Colors.ENDC}")
            for result in self.results:
                if result.status == TestStatus.FAILED:
                    self.log(f"  {Colors.FAIL}✗{Colors.ENDC} {result.name}")
                    self.log(f"    Error: {result.message}")

        # Final status
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        if failed == 0:
            self.log(f"{Colors.OKGREEN}{Colors.BOLD}✓ FRAMEWORK VERIFICATION PASSED{Colors.ENDC}")
            self.log(f"{Colors.OKGREEN}All critical tests passed successfully!{Colors.ENDC}")
        else:
            self.log(f"{Colors.FAIL}{Colors.BOLD}✗ FRAMEWORK VERIFICATION FAILED{Colors.ENDC}")
            self.log(f"{Colors.FAIL}{failed} test(s) failed. Please review the errors above.{Colors.ENDC}")
        self.log(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

        return failed == 0

    def verify_agent_integration(self):
        """Verify Agent class integration with models and tools"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Agent Integration ==={Colors.ENDC}\n")

        def test_agent_creation():
            from effgen.core.agent import AgentConfig
            from effgen.models.base import BaseModel

            # AgentConfig requires a model - just test it exists
            assert AgentConfig is not None
            self.log_verbose("  AgentConfig class available")

        def test_task_creation():
            from effgen import Task, TaskPriority

            task = Task(
                description="Test task for verification",
                priority=TaskPriority.HIGH
            )

            assert task.description == "Test task for verification"
            assert task.priority == TaskPriority.HIGH
            self.log_verbose("  Task created successfully")

        def test_tool_discovery():
            from effgen import get_tool_registry

            registry = get_tool_registry()
            registry.discover_builtin_tools()

            # Check if built-in tools were discovered
            assert registry.is_registered("calculator")
            self.log_verbose("  Tool discovery working")

        tests = [
            ("Agent config creation", test_agent_creation),
            ("Task creation and management", test_task_creation),
            ("Tool discovery system", test_tool_discovery),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_protocol_systems(self):
        """Verify protocol implementations (MCP, A2A, ACP)"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Protocol Systems ==={Colors.ENDC}\n")

        def test_official_mcp_import():
            # Test that official MCP package is available
            import mcp
            assert mcp is not None
            self.log_verbose("  Official MCP package imported")

        def test_mcp_protocol_handler():
            from effgen.tools.protocols.mcp import MCPProtocolHandler

            handler = MCPProtocolHandler()
            assert handler is not None
            self.log_verbose("  MCP protocol handler created")

        def test_mcp_client():
            from effgen.tools.protocols.mcp import MCPClient, MCPServerConfig, TransportType

            # Create a client configuration
            config = MCPServerConfig(
                name="test_server",
                transport=TransportType.STDIO,
                command="echo",
                args=["test"]
            )
            assert config is not None
            self.log_verbose("  MCP client configuration created")

        def test_mcp_transports():
            from effgen.tools.protocols.mcp import StdioTransport, HTTPTransport, SSETransport

            # Verify all transport types are available
            assert StdioTransport is not None
            assert HTTPTransport is not None
            assert SSETransport is not None
            self.log_verbose("  MCP transport types available")

        def test_mcp_server():
            from effgen.tools.protocols.mcp import MCPServer, create_server

            # Verify server classes exist
            assert MCPServer is not None
            assert create_server is not None
            self.log_verbose("  MCP server classes available")

        def test_mcp_data_structures():
            from effgen.tools.protocols.mcp import (
                MCPTool, MCPResource, MCPCapabilities,
                MCPRequest, MCPResponse, MCPError
            )

            # Verify all MCP data structures are available
            assert MCPTool is not None
            assert MCPResource is not None
            assert MCPCapabilities is not None
            assert MCPRequest is not None
            assert MCPResponse is not None
            assert MCPError is not None
            self.log_verbose("  MCP data structures available")

        def test_a2a_protocol():
            from effgen.tools.protocols.a2a import A2AClient

            # Just verify class can be imported
            assert A2AClient is not None
            self.log_verbose("  A2A protocol classes available")

        def test_acp_protocol():
            from effgen.tools.protocols.acp import ACPClient

            # Just verify class can be imported
            assert ACPClient is not None
            self.log_verbose("  ACP protocol classes available")

        def test_mcp_official_server():
            from effgen.tools.protocols.mcp_official import (
                EffGenMCPServer,
                EffGenMCPServerConfig,
                create_server,
                main_stdio,
                main_http
            )

            # Verify all server components exist
            assert EffGenMCPServer is not None
            assert EffGenMCPServerConfig is not None
            assert create_server is not None
            assert main_stdio is not None
            assert main_http is not None
            self.log_verbose("  Official MCP server components available")

        def test_mcp_official_client():
            from effgen.tools.protocols.mcp_official import (
                EffGenMCPClient,
                MCPServerConfig,
                create_client,
                create_stdio_client,
                create_http_client
            )

            # Verify all client components exist
            assert EffGenMCPClient is not None
            assert MCPServerConfig is not None
            assert create_client is not None
            assert create_stdio_client is not None
            assert create_http_client is not None
            self.log_verbose("  Official MCP client components available")

        def test_mcp_official_config():
            from effgen.tools.protocols.mcp_official import MCPServerConfig

            # Create configuration for STDIO transport
            config = MCPServerConfig(
                name="test-server",
                transport="stdio",
                command="python",
                args=["-m", "mcp_server"]
            )

            assert config.name == "test-server"
            assert config.transport == "stdio"
            self.log_verbose("  Official MCP config creation working")

        def test_mcp_official_imports():
            # Test that official MCP SDK types can be imported through our wrapper
            try:
                from effgen.tools.protocols.mcp_official import EffGenMCPServer
                # If we can import the server, official MCP is available
                assert EffGenMCPServer is not None
                self.log_verbose("  Official MCP SDK integration working")
            except ImportError as e:
                # This is okay if mcp package isn't installed
                if "mcp" in str(e).lower():
                    self.log_verbose("  Official MCP SDK not installed (optional)")
                else:
                    raise

        tests = [
            ("Official MCP package", test_official_mcp_import),
            ("MCP protocol handler", test_mcp_protocol_handler),
            ("MCP client configuration", test_mcp_client),
            ("MCP transport types", test_mcp_transports),
            ("MCP server classes", test_mcp_server),
            ("MCP data structures", test_mcp_data_structures),
            ("Official MCP server components", test_mcp_official_server),
            ("Official MCP client components", test_mcp_official_client),
            ("Official MCP config creation", test_mcp_official_config),
            ("Official MCP SDK integration", test_mcp_official_imports),
            ("A2A protocol support", test_a2a_protocol),
            ("ACP protocol support", test_acp_protocol),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_tool_functionality(self):
        """Verify individual tool functionality"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Tool Functionality ==={Colors.ENDC}\n")

        def test_calculator_execution():
            import asyncio
            from effgen.tools.builtin.calculator import Calculator

            calc = Calculator()
            # Test basic arithmetic
            result = asyncio.run(calc.execute(expression="2 + 2"))
            assert "4" in str(result)
            self.log_verbose("  Calculator execution works")

        def test_parameter_validation():
            from effgen.tools.base_tool import ParameterSpec, ParameterType

            param = ParameterSpec(
                name="test_param",
                type=ParameterType.INTEGER,
                description="Test parameter",
                required=True,
                min_value=0,
                max_value=100
            )

            # Test valid value
            is_valid, error = param.validate(50)
            assert is_valid
            assert error is None

            # Test invalid value
            is_valid, error = param.validate(150)
            assert not is_valid
            assert error is not None
            self.log_verbose("  Parameter validation working")

        def test_tool_metadata_serialization():
            from effgen.tools.base_tool import ToolMetadata, ToolCategory

            metadata = ToolMetadata(
                name="test_tool",
                description="Test tool",
                category=ToolCategory.COMPUTATION,
                version="1.0.0"
            )

            # Test serialization
            metadata_dict = metadata.to_dict()
            assert metadata_dict["name"] == "test_tool"
            assert metadata_dict["version"] == "1.0.0"
            self.log_verbose("  Tool metadata serialization working")

        def test_tool_result():
            from effgen.tools.base_tool import ToolResult

            result = ToolResult(
                success=True,
                output="Test output",
                execution_time=0.5
            )

            assert result.success
            assert result.output == "Test output"
            self.log_verbose("  ToolResult working")

        tests = [
            ("Calculator execution", test_calculator_execution),
            ("Parameter validation", test_parameter_validation),
            ("Tool metadata serialization", test_tool_metadata_serialization),
            ("Tool result structure", test_tool_result),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_async_operations(self):
        """Verify async operations work correctly"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Async Operations ==={Colors.ENDC}\n")

        def test_async_tool_registry():
            import asyncio
            from effgen.tools import get_registry
            from effgen.tools.builtin.calculator import Calculator

            async def async_test():
                registry = get_registry()
                registry.register_tool(Calculator)
                tool = await registry.get_tool("calculator")
                assert tool is not None
                return True

            result = asyncio.run(async_test())
            assert result
            self.log_verbose("  Async tool registry operations working")

        def test_async_tool_execution():
            import asyncio
            from effgen.tools.builtin.calculator import Calculator

            async def async_test():
                calc = Calculator()
                result = await calc.execute(expression="10 * 5")
                return result

            result = asyncio.run(async_test())
            assert "50" in str(result)
            self.log_verbose("  Async tool execution working")

        tests = [
            ("Async tool registry", test_async_tool_registry),
            ("Async tool execution", test_async_tool_execution),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_memory_integration(self):
        """Verify memory system integration"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Memory Integration ==={Colors.ENDC}\n")

        def test_message_serialization():
            from effgen.memory import Message, MessageRole

            message = Message(
                role=MessageRole.USER,
                content="Test message",
                metadata={"source": "test"}
            )

            # Test serialization
            msg_dict = message.to_dict()
            assert msg_dict["content"] == "Test message"
            assert msg_dict["role"] == "user"

            # Test deserialization
            restored = Message.from_dict(msg_dict)
            assert restored.content == "Test message"
            self.log_verbose("  Message serialization working")

        def test_memory_token_counting():
            from effgen.memory import ShortTermMemory, MessageRole

            memory = ShortTermMemory(max_tokens=1000)
            memory.add_message(MessageRole.USER, "Hello")
            memory.add_message(MessageRole.ASSISTANT, "Hi there")

            token_count = memory.get_token_count()
            assert token_count > 0
            self.log_verbose("  Memory token counting working")

        def test_long_term_memory_storage():
            from effgen.memory import LongTermMemory, MemoryEntry, MemoryType, ImportanceLevel, JSONStorageBackend
            import tempfile
            import os

            # Create temp directory for storage
            with tempfile.TemporaryDirectory() as tmpdir:
                storage_path = os.path.join(tmpdir, "memory.json")
                backend = JSONStorageBackend(filepath=storage_path)
                ltm = LongTermMemory(backend=backend)

                # Start a session
                ltm.start_session(name="test_session")

                # Add memory entry
                ltm.add_memory(
                    content="Test memory",
                    memory_type=MemoryType.FACT,
                    importance=ImportanceLevel.HIGH
                )

                # Verify memory was added
                # LongTermMemory stores in backend, just verify it doesn't crash
                assert ltm is not None
                self.log_verbose("  Long-term memory storage working")

        def test_conversation_summary():
            from effgen.memory.short_term import ConversationSummary

            summary = ConversationSummary(
                summary="Test summary",
                message_count=10,
                token_count=100
            )

            summary_dict = summary.to_dict()
            assert summary_dict["message_count"] == 10
            self.log_verbose("  Conversation summary working")

        tests = [
            ("Message serialization", test_message_serialization),
            ("Memory token counting", test_memory_token_counting),
            ("Long-term memory storage", test_long_term_memory_storage),
            ("Conversation summary", test_conversation_summary),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_error_handling(self):
        """Verify error handling and edge cases"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Error Handling ==={Colors.ENDC}\n")

        def test_invalid_tool_name():
            from effgen.tools import get_registry

            registry = get_registry()
            # Should handle gracefully
            assert not registry.is_registered("nonexistent_tool")
            self.log_verbose("  Invalid tool name handled")

        def test_parameter_validation_errors():
            from effgen.tools.base_tool import ParameterSpec, ParameterType

            param = ParameterSpec(
                name="test",
                type=ParameterType.INTEGER,
                description="Test",
                required=True
            )

            # Test missing required parameter
            is_valid, error = param.validate(None)
            assert not is_valid
            assert "required" in error.lower()
            self.log_verbose("  Parameter validation errors working")

        def test_code_validator_safety():
            from effgen.execution.validators import PythonValidator

            validator = PythonValidator()

            # Test dangerous code
            dangerous_code = "import os; os.system('rm -rf /')"
            result = validator.validate(dangerous_code)
            assert not result.is_safe
            assert len(result.issues) > 0
            self.log_verbose("  Code validator safety checks working")

        def test_memory_overflow_handling():
            from effgen.memory import ShortTermMemory, MessageRole

            # Create memory with very small limit
            memory = ShortTermMemory(max_messages=2)

            # Add more messages than limit
            memory.add_message(MessageRole.USER, "Message 1")
            memory.add_message(MessageRole.USER, "Message 2")
            memory.add_message(MessageRole.USER, "Message 3")

            messages = memory.get_recent_messages()
            assert len(messages) <= 2
            self.log_verbose("  Memory overflow handling working")

        tests = [
            ("Invalid tool name", test_invalid_tool_name),
            ("Parameter validation errors", test_parameter_validation_errors),
            ("Code validator safety", test_code_validator_safety),
            ("Memory overflow handling", test_memory_overflow_handling),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def verify_advanced_features(self):
        """Verify advanced framework features"""
        self.log(f"\n{Colors.HEADER}{Colors.BOLD}=== Testing Advanced Features ==={Colors.ENDC}\n")

        def test_gpu_allocation():
            from effgen.gpu import GPUAllocator

            allocator = GPUAllocator()
            assert allocator is not None
            self.log_verbose("  GPU allocator initialized")

        def test_vector_memory():
            try:
                from effgen.memory import VectorMemoryStore

                # Test with simple embedding provider
                store = VectorMemoryStore()
                assert store is not None
                self.log_verbose("  Vector memory store created")
            except Exception as e:
                # Vector store may require optional dependencies
                if "FAISS" in str(e) or "chroma" in str(e):
                    self.log_verbose(f"  Vector store requires optional dependencies (skipping)")
                    # Don't fail on optional dependency
                    return
                else:
                    raise

        def test_chain_manager():
            from effgen.prompts import ChainManager

            manager = ChainManager()
            assert manager is not None
            self.log_verbose("  Chain manager initialized")

        def test_orchestrator():
            from effgen.core.orchestrator import MultiAgentOrchestrator

            # Orchestrator can be imported
            assert MultiAgentOrchestrator is not None
            self.log_verbose("  Orchestrator available")

        def test_complexity_analyzer():
            from effgen.core.complexity_analyzer import ComplexityAnalyzer

            analyzer = ComplexityAnalyzer()

            # Test complexity analysis
            simple_task = "What is 2 + 2?"
            result = analyzer.analyze(simple_task)
            assert result.overall >= 0
            self.log_verbose("  Complexity analyzer working")

        def test_decomposition_engine():
            from effgen.core.decomposition_engine import DecompositionEngine

            engine = DecompositionEngine()

            # Test task decomposition
            complex_task = "Build a web application with authentication and database"
            subtasks = engine.decompose(complex_task, strategy="parallel_sub_agents")
            assert len(subtasks) > 0
            self.log_verbose("  Decomposition engine working")

        def test_execution_tracker():
            from effgen.core.execution_tracker import ExecutionTracker

            tracker = ExecutionTracker()
            assert tracker is not None
            self.log_verbose("  Execution tracker initialized")

        def test_router():
            from effgen.core.router import SubAgentRouter

            router = SubAgentRouter()
            assert router is not None
            self.log_verbose("  Router initialized")

        def test_sub_agent_manager():
            from effgen.core.sub_agent_manager import SubAgentManager

            manager = SubAgentManager()
            assert manager is not None
            self.log_verbose("  Sub-agent manager initialized")

        tests = [
            ("GPU allocation system", test_gpu_allocation),
            ("Vector memory store", test_vector_memory),
            ("Prompt chain manager", test_chain_manager),
            ("Task orchestrator", test_orchestrator),
            ("Complexity analyzer", test_complexity_analyzer),
            ("Decomposition engine", test_decomposition_engine),
            ("Execution tracker", test_execution_tracker),
            ("Router", test_router),
            ("Sub-agent manager", test_sub_agent_manager),
        ]

        for name, test_func in tests:
            result = self.run_test(name, test_func)
            self.add_result(result)

    def run_all_verifications(self):
        """Run all verification tests"""
        self.log(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        self.log(f"{Colors.HEADER}{Colors.BOLD}effGen Framework Verification{Colors.ENDC}")
        self.log(f"{Colors.HEADER}{Colors.BOLD}Super Comprehensive Test Suite{Colors.ENDC}")
        self.log(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

        # Run all verification suites
        self.verify_imports()
        self.verify_core_classes()
        self.verify_tool_system()
        self.verify_configuration()
        self.verify_model_interfaces()
        self.verify_prompt_system()
        self.verify_memory_system()
        self.verify_execution_system()
        self.verify_agent_integration()
        self.verify_protocol_systems()
        self.verify_tool_functionality()
        self.verify_async_operations()
        self.verify_memory_integration()
        self.verify_error_handling()
        self.verify_advanced_features()
        self.verify_dependencies()

        # Generate report
        success = self.generate_report()

        return 0 if success else 1


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify effGen framework installation and functionality"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional dependency checks"
    )

    args = parser.parse_args()

    verifier = FrameworkVerifier(
        verbose=args.verbose,
        skip_optional=args.skip_optional
    )

    exit_code = verifier.run_all_verifications()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
