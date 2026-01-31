"""
Built-in tools for the effGen framework.

This module contains the standard set of tools that ship with effGen.
"""

# Import built-in tools (lazy loading handled by registry)
__all__ = [
    "CodeExecutor",
    "PythonREPL",
    "WebSearch",
    "FileOperations",
    "Calculator",
]


def __getattr__(name):
    """Lazy import of tools."""
    if name == "CodeExecutor":
        from .code_executor import CodeExecutor
        return CodeExecutor
    elif name == "PythonREPL":
        from .python_repl import PythonREPL
        return PythonREPL
    elif name == "WebSearch":
        from .web_search import WebSearch
        return WebSearch
    elif name == "FileOperations":
        from .file_ops import FileOperations
        return FileOperations
    elif name == "Calculator":
        from .calculator import Calculator
        return Calculator
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
