"""
Code validators for security and safety checks.

This module provides validators to analyze code before execution,
detecting dangerous patterns and ensuring safe execution practices.
"""

import ast
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a security or safety issue in code."""
    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    rule: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_safe: bool
    issues: List[ValidationIssue]
    language: str

    @property
    def has_critical(self) -> bool:
        """Check if there are critical issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)


class PythonValidator:
    """Validator for Python code security."""

    # Dangerous imports that should be blocked
    DANGEROUS_IMPORTS = {
        'os.system', 'subprocess', 'eval', 'exec', 'compile',
        '__import__', 'importlib', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'telnetlib', 'smtplib', 'pickle',
        'shelve', 'marshal', 'ctypes', 'cffi', 'multiprocessing'
    }

    # Dangerous function calls
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open',
        'input', 'raw_input', 'execfile', 'reload'
    }

    # Dangerous attributes
    DANGEROUS_ATTRIBUTES = {
        '__globals__', '__builtins__', '__dict__', '__class__',
        '__bases__', '__subclasses__', '__code__', '__closure__'
    }

    def __init__(self,
                 allow_imports: Optional[Set[str]] = None,
                 allow_file_ops: bool = False,
                 allow_network: bool = False):
        """
        Initialize Python validator.

        Args:
            allow_imports: Set of allowed import modules
            allow_file_ops: Whether to allow file operations
            allow_network: Whether to allow network operations
        """
        self.allow_imports = allow_imports or {'math', 'random', 'datetime', 'json', 're'}
        self.allow_file_ops = allow_file_ops
        self.allow_network = allow_network

    def validate(self, code: str) -> ValidationResult:
        """
        Validate Python code for security issues.

        Args:
            code: Python code to validate

        Returns:
            ValidationResult with detected issues
        """
        issues: List[ValidationIssue] = []

        # Check for syntax errors
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                rule="syntax_error"
            ))
            return ValidationResult(is_safe=False, issues=issues, language="python")

        # Perform AST-based validation
        issues.extend(self._check_imports(tree))
        issues.extend(self._check_function_calls(tree))
        issues.extend(self._check_attributes(tree))
        issues.extend(self._check_file_operations(tree))
        issues.extend(self._check_dangerous_patterns(code))

        # Determine if code is safe
        is_safe = not any(
            issue.severity in {ValidationSeverity.CRITICAL, ValidationSeverity.ERROR}
            for issue in issues
        )

        return ValidationResult(is_safe=is_safe, issues=issues, language="python")

    def _check_imports(self, tree: ast.AST) -> List[ValidationIssue]:
        """Check for dangerous imports."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allow_imports:
                        if alias.name in self.DANGEROUS_IMPORTS:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.CRITICAL,
                                message=f"Dangerous import blocked: {alias.name}",
                                line_number=node.lineno,
                                rule="dangerous_import"
                            ))
                        else:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Import not in whitelist: {alias.name}",
                                line_number=node.lineno,
                                rule="unapproved_import"
                            ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module not in self.allow_imports:
                    if any(danger in module for danger in self.DANGEROUS_IMPORTS):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Dangerous import blocked: from {module}",
                            line_number=node.lineno,
                            rule="dangerous_import"
                        ))
                    else:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Import not in whitelist: from {module}",
                            line_number=node.lineno,
                            rule="unapproved_import"
                        ))

        return issues

    def _check_function_calls(self, tree: ast.AST) -> List[ValidationIssue]:
        """Check for dangerous function calls."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in self.DANGEROUS_FUNCTIONS:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Dangerous function call blocked: {func_name}()",
                        line_number=node.lineno,
                        rule="dangerous_function"
                    ))

        return issues

    def _check_attributes(self, tree: ast.AST) -> List[ValidationIssue]:
        """Check for dangerous attribute access."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in self.DANGEROUS_ATTRIBUTES:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Dangerous attribute access blocked: {node.attr}",
                        line_number=node.lineno,
                        rule="dangerous_attribute"
                    ))

        return issues

    def _check_file_operations(self, tree: ast.AST) -> List[ValidationIssue]:
        """Check for file operations."""
        issues = []

        if not self.allow_file_ops:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'open':
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message="File operations not allowed",
                            line_number=node.lineno,
                            rule="file_operation_blocked"
                        ))

        return issues

    def _check_dangerous_patterns(self, code: str) -> List[ValidationIssue]:
        """Check for dangerous patterns using regex."""
        issues = []

        # Check for shell commands
        if re.search(r'os\.system|subprocess\.', code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="Shell command execution detected",
                rule="shell_command"
            ))

        # Check for code injection
        if re.search(r'eval\s*\(|exec\s*\(', code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="Code injection pattern detected",
                rule="code_injection"
            ))

        # Check for infinite loops (basic check)
        if re.search(r'while\s+True\s*:', code) and 'break' not in code:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Potential infinite loop detected",
                rule="infinite_loop"
            ))

        return issues


class JavaScriptValidator:
    """Validator for JavaScript code security."""

    DANGEROUS_PATTERNS = [
        r'eval\s*\(',
        r'Function\s*\(',
        r'require\s*\([\'"]child_process',
        r'require\s*\([\'"]fs',
        r'require\s*\([\'"]net',
        r'require\s*\([\'"]http',
        r'process\.exit',
        r'process\.kill',
        r'__dirname',
        r'__filename',
    ]

    def __init__(self, allow_network: bool = False, allow_file_ops: bool = False):
        """
        Initialize JavaScript validator.

        Args:
            allow_network: Whether to allow network operations
            allow_file_ops: Whether to allow file operations
        """
        self.allow_network = allow_network
        self.allow_file_ops = allow_file_ops

    def validate(self, code: str) -> ValidationResult:
        """
        Validate JavaScript code for security issues.

        Args:
            code: JavaScript code to validate

        Returns:
            ValidationResult with detected issues
        """
        issues: List[ValidationIssue] = []

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Dangerous pattern detected: {pattern}",
                    rule="dangerous_pattern"
                ))

        # Check for network operations
        if not self.allow_network:
            if re.search(r'fetch\s*\(|axios\.|http\.|net\.', code):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Network operations not allowed",
                    rule="network_blocked"
                ))

        # Check for file operations
        if not self.allow_file_ops:
            if re.search(r'fs\.|readFile|writeFile', code):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="File operations not allowed",
                    rule="file_operation_blocked"
                ))

        is_safe = not any(
            issue.severity in {ValidationSeverity.CRITICAL, ValidationSeverity.ERROR}
            for issue in issues
        )

        return ValidationResult(is_safe=is_safe, issues=issues, language="javascript")


class BashValidator:
    """Validator for Bash/Shell script security."""

    DANGEROUS_COMMANDS = {
        'rm', 'dd', 'mkfs', 'fdisk', 'format',
        'shutdown', 'reboot', 'halt', 'poweroff',
        'sudo', 'su', 'chmod', 'chown', 'chgrp',
        'curl', 'wget', 'nc', 'netcat', 'telnet',
        'ssh', 'scp', 'ftp', 'rsync'
    }

    def __init__(self, allow_network: bool = False):
        """
        Initialize Bash validator.

        Args:
            allow_network: Whether to allow network operations
        """
        self.allow_network = allow_network

    def validate(self, code: str) -> ValidationResult:
        """
        Validate Bash code for security issues.

        Args:
            code: Bash script to validate

        Returns:
            ValidationResult with detected issues
        """
        issues: List[ValidationIssue] = []

        # Split into lines and check each command
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Extract command (first word)
            command = line.split()[0] if line.split() else ""

            # Check for dangerous commands
            if command in self.DANGEROUS_COMMANDS:
                if command in {'curl', 'wget', 'nc', 'netcat', 'telnet', 'ssh', 'scp', 'ftp'}:
                    if not self.allow_network:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Network command not allowed: {command}",
                            line_number=line_num,
                            rule="network_blocked"
                        ))
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Dangerous command blocked: {command}",
                        line_number=line_num,
                        rule="dangerous_command"
                    ))

            # Check for command substitution with dangerous patterns
            if re.search(r'\$\(.*rm.*\)|\$\(.*dd.*\)', line):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    message="Dangerous command substitution detected",
                    line_number=line_num,
                    rule="dangerous_substitution"
                ))

        is_safe = not any(
            issue.severity in {ValidationSeverity.CRITICAL, ValidationSeverity.ERROR}
            for issue in issues
        )

        return ValidationResult(is_safe=is_safe, issues=issues, language="bash")


class CodeValidator:
    """
    Main code validator that dispatches to language-specific validators.
    """

    def __init__(self,
                 allow_network: bool = False,
                 allow_file_ops: bool = False,
                 custom_allow_imports: Optional[Set[str]] = None):
        """
        Initialize code validator.

        Args:
            allow_network: Whether to allow network operations
            allow_file_ops: Whether to allow file operations
            custom_allow_imports: Custom set of allowed Python imports
        """
        self.python_validator = PythonValidator(
            allow_imports=custom_allow_imports,
            allow_file_ops=allow_file_ops,
            allow_network=allow_network
        )
        self.javascript_validator = JavaScriptValidator(
            allow_network=allow_network,
            allow_file_ops=allow_file_ops
        )
        self.bash_validator = BashValidator(allow_network=allow_network)

    def validate(self, code: str, language: str) -> ValidationResult:
        """
        Validate code based on language.

        Args:
            code: Code to validate
            language: Programming language (python, javascript, bash)

        Returns:
            ValidationResult with detected issues

        Raises:
            ValueError: If language is not supported
        """
        language = language.lower()

        if language in {'python', 'py'}:
            return self.python_validator.validate(code)
        elif language in {'javascript', 'js', 'node'}:
            return self.javascript_validator.validate(code)
        elif language in {'bash', 'sh', 'shell'}:
            return self.bash_validator.validate(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def is_safe(self, code: str, language: str) -> bool:
        """
        Quick check if code is safe to execute.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            True if code is safe, False otherwise
        """
        try:
            result = self.validate(code, language)
            return result.is_safe
        except Exception:
            return False
