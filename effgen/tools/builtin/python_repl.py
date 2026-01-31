"""
Python REPL tool with persistent sessions.

This module provides a Python REPL (Read-Eval-Print Loop) tool with
persistent session state, safe evaluation, and comprehensive error handling.
"""

import sys
import io
import traceback
from typing import Dict, Any, Optional
import logging
from contextlib import redirect_stdout, redirect_stderr
import ast
import builtins

from ..base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ParameterSpec,
    ParameterType,
)


logger = logging.getLogger(__name__)


class PythonREPL(BaseTool):
    """
    Python REPL tool with persistent session state.

    Features:
    - Persistent variable state across executions
    - Safe evaluation with restricted builtins
    - Import management
    - Comprehensive error handling
    - Output capture (stdout/stderr)
    - Expression vs statement handling
    - Session management (reset, save, restore)

    Security:
    - Restricted builtins (no file operations, eval, exec by default)
    - Configurable allowed imports
    - Timeout protection (inherited from base)
    - Memory limits (via base class)
    """

    # Dangerous builtins to restrict
    RESTRICTED_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
    }

    # Default allowed imports
    DEFAULT_ALLOWED_IMPORTS = {
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "decimal",
        "fractions",
    }

    def __init__(self):
        """Initialize the Python REPL."""
        super().__init__(
            metadata=ToolMetadata(
                name="python_repl",
                description="Execute Python code in a persistent REPL session with safe evaluation",
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="Python code to execute",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="session_id",
                        type=ParameterType.STRING,
                        description="Session identifier for persistent state (default: 'default')",
                        required=False,
                        default="default",
                    ),
                    ParameterSpec(
                        name="reset_session",
                        type=ParameterType.BOOLEAN,
                        description="Whether to reset the session before execution",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="return_variables",
                        type=ParameterType.BOOLEAN,
                        description="Whether to return all session variables",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="restricted_mode",
                        type=ParameterType.BOOLEAN,
                        description="Whether to use restricted builtins (safer but limited)",
                        required=False,
                        default=True,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "result": {"type": "any"},
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "error": {"type": "string"},
                        "variables": {"type": "object"},
                    },
                },
                timeout_seconds=30,
                tags=["python", "repl", "code", "execution"],
                examples=[
                    {
                        "code": "x = 10\ny = 20\nx + y",
                        "session_id": "default",
                        "output": {
                            "result": 30,
                            "stdout": "",
                            "error": None,
                        },
                    },
                    {
                        "code": "import math\nmath.sqrt(16)",
                        "output": {
                            "result": 4.0,
                            "stdout": "",
                        },
                    },
                ],
            )
        )
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._allowed_imports = self.DEFAULT_ALLOWED_IMPORTS.copy()

    async def initialize(self) -> None:
        """Initialize the REPL."""
        await super().initialize()
        # Create default session
        self._create_session("default")

    def _create_session(self, session_id: str) -> None:
        """Create a new session with clean namespace.

        Note: We use the same dict for both globals and locals to ensure
        that functions defined in exec() can access themselves for recursion.
        """
        namespace = {}
        self._sessions[session_id] = {
            "globals": namespace,
            "locals": namespace,  # Same dict for proper function recursion
        }

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create a session."""
        if session_id not in self._sessions:
            self._create_session(session_id)
        return self._sessions[session_id]

    def _get_restricted_builtins(self) -> Dict[str, Any]:
        """Get restricted builtins dictionary."""
        safe_builtins = {}
        for name in dir(builtins):
            if name not in self.RESTRICTED_BUILTINS:
                safe_builtins[name] = getattr(builtins, name)

        # Add a safe __import__ that only allows whitelisted modules
        original_import = builtins.__import__

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Safe import that only allows whitelisted modules."""
            if not self._is_import_allowed(name):
                raise ImportError(f"Import of '{name}' is not allowed")
            return original_import(name, globals, locals, fromlist, level)

        safe_builtins['__import__'] = safe_import
        return safe_builtins

    def _is_import_allowed(self, module_name: str) -> bool:
        """Check if an import is allowed."""
        # Get base module name (before any dots)
        base_module = module_name.split(".")[0]
        return base_module in self._allowed_imports

    def _check_imports(self, code: str) -> Optional[str]:
        """
        Check if code contains only allowed imports.

        Returns:
            Optional[str]: Error message if disallowed import found, None otherwise
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_import_allowed(alias.name):
                            return f"Import of '{alias.name}' is not allowed"
                elif isinstance(node, ast.ImportFrom):
                    if node.module and not self._is_import_allowed(node.module):
                        return f"Import from '{node.module}' is not allowed"
        except SyntaxError:
            # Let the actual execution handle syntax errors
            pass

        return None

    async def _execute(
        self,
        code: str,
        session_id: str = "default",
        reset_session: bool = False,
        return_variables: bool = False,
        restricted_mode: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute Python code in a REPL session.

        Args:
            code: Python code to execute
            session_id: Session identifier for state persistence
            reset_session: Whether to reset session state before execution
            return_variables: Whether to include all variables in response
            restricted_mode: Whether to use restricted builtins

        Returns:
            Dict containing result, stdout, stderr, error, and optionally variables
        """
        # Reset session if requested
        if reset_session:
            self._create_session(session_id)

        # Get session
        session = self._get_session(session_id)

        # Check imports in restricted mode
        if restricted_mode:
            import_error = self._check_imports(code)
            if import_error:
                return {
                    "result": None,
                    "stdout": "",
                    "stderr": "",
                    "error": import_error,
                    "variables": {} if return_variables else None,
                }

        # Set up restricted builtins if needed
        if restricted_mode:
            session["globals"]["__builtins__"] = self._get_restricted_builtins()

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to parse as expression first
                try:
                    tree = ast.parse(code, mode="eval")
                    compiled = compile(tree, "<repl>", mode="eval")
                    result = eval(
                        compiled,
                        session["globals"],
                        session["locals"],
                    )
                except SyntaxError:
                    # Not an expression, try as statements
                    tree = ast.parse(code, mode="exec")
                    compiled = compile(tree, "<repl>", mode="exec")
                    exec(
                        compiled,
                        session["globals"],
                        session["locals"],
                    )

                    # Check if last statement was an expression
                    if (
                        tree.body
                        and isinstance(tree.body[-1], ast.Expr)
                    ):
                        last_expr = tree.body[-1].value
                        result = eval(
                            compile(
                                ast.Expression(body=last_expr),
                                "<repl>",
                                mode="eval",
                            ),
                            session["globals"],
                            session["locals"],
                        )

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.debug(f"REPL execution error: {error}")

        # Get output
        stdout_value = stdout_capture.getvalue()
        stderr_value = stderr_capture.getvalue()

        # Prepare response
        response = {
            "result": result,
            "stdout": stdout_value,
            "stderr": stderr_value,
            "error": error,
        }

        # Include variables if requested
        if return_variables:
            # Filter out private variables and builtins
            variables = {
                k: repr(v)
                for k, v in {**session["globals"], **session["locals"]}.items()
                if not k.startswith("_") and k != "__builtins__"
            }
            response["variables"] = variables

        return response

    def add_allowed_import(self, module_name: str) -> None:
        """
        Add a module to the allowed imports list.

        Args:
            module_name: Name of the module to allow
        """
        self._allowed_imports.add(module_name)

    def remove_allowed_import(self, module_name: str) -> None:
        """
        Remove a module from the allowed imports list.

        Args:
            module_name: Name of the module to disallow
        """
        self._allowed_imports.discard(module_name)

    def reset_session(self, session_id: str) -> None:
        """
        Reset a session to clean state.

        Args:
            session_id: Session to reset
        """
        if session_id in self._sessions:
            self._create_session(session_id)

    def get_session_variables(self, session_id: str) -> Dict[str, str]:
        """
        Get all variables in a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict mapping variable names to their string representations
        """
        if session_id not in self._sessions:
            return {}

        session = self._sessions[session_id]
        return {
            k: repr(v)
            for k, v in {**session["globals"], **session["locals"]}.items()
            if not k.startswith("_") and k != "__builtins__"
        }

    async def cleanup(self) -> None:
        """Clean up all sessions."""
        self._sessions.clear()
        await super().cleanup()
