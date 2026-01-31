"""
Logging utilities for effGen framework.

This module provides comprehensive logging capabilities with rich formatting,
multiple log levels, file and console handlers, structured logging, and log rotation.

Features:
    - Rich console formatting with colors and emojis
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - File and console handlers with independent configurations
    - Structured logging with JSON output option
    - Automatic log rotation based on size and time
    - Context management for scoped logging
    - Performance tracking integration
    - Thread-safe logging operations

Example:
    Basic usage:
        >>> from effgen.utils.logging import setup_logger, get_logger
        >>> setup_logger(level="INFO", log_dir="./logs")
        >>> logger = get_logger(__name__)
        >>> logger.info("Agent started successfully")

    Structured logging:
        >>> logger.info("Task completed", extra={
        ...     "task_id": "123",
        ...     "duration": 5.2,
        ...     "tokens_used": 1500
        ... })

    Context-based logging:
        >>> with LogContext(agent_id="agent-001"):
        ...     logger.info("Processing task")
"""

import logging
import logging.handlers
import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from contextlib import contextmanager
import threading

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install as install_rich_traceback
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


# Thread-local storage for logging context
_context_storage = threading.local()


class LogLevel:
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format for structured logging.

    This formatter converts log records to JSON format, making them easier to
    parse and analyze with log aggregation tools.
    """

    def __init__(self, include_extra: bool = True):
        """
        Initialize the structured formatter.

        Args:
            include_extra: Whether to include extra fields from log records
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        if self.include_extra:
            # Get context from thread-local storage
            context = getattr(_context_storage, 'context', {})
            if context:
                log_data["context"] = context

            # Add any extra fields from the log record
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                              'levelname', 'levelno', 'lineno', 'module', 'msecs',
                              'pathname', 'process', 'processName', 'relativeCreated',
                              'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                              'message', 'asctime']:
                    log_data[key] = value

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color support for console output.

    Adds colors to log levels for better readability in console output.
    """

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True):
        """
        Initialize the colored formatter.

        Args:
            fmt: Format string for log messages
            use_colors: Whether to use colors in output
        """
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.

        Args:
            record: The log record to format

        Returns:
            Formatted log string with colors
        """
        if self.use_colors:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )

        return super().format(record)


class LoggerManager:
    """
    Centralized logger management for the effGen framework.

    Manages logger instances, handlers, and configurations across the application.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one logger manager exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the logger manager."""
        if self._initialized:
            return

        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: List[logging.Handler] = []
        self.log_dir: Optional[Path] = None
        self.level: int = logging.INFO
        self.structured: bool = False
        self._initialized = True

    def setup(
        self,
        level: Union[str, int] = "INFO",
        log_dir: Optional[Union[str, Path]] = None,
        log_file: str = "effgen.log",
        console_output: bool = True,
        file_output: bool = True,
        structured: bool = False,
        use_rich: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        format_string: Optional[str] = None,
    ) -> None:
        """
        Setup logging configuration for the application.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            log_file: Name of the log file
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file
            structured: Whether to use structured JSON logging
            use_rich: Whether to use rich formatting (if available)
            max_file_size: Maximum size of log file before rotation (bytes)
            backup_count: Number of backup files to keep
            format_string: Custom format string for log messages
        """
        # Convert string level to int
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.level = level
        self.structured = structured

        # Setup log directory
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing handlers
        self._clear_handlers()

        # Setup format string
        if format_string is None:
            if structured:
                format_string = None  # Will use JSON formatter
            else:
                format_string = (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(funcName)s:%(lineno)d - %(message)s"
                )

        # Setup console handler
        if console_output:
            if use_rich and RICH_AVAILABLE:
                console_handler = RichHandler(
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,
                    markup=True,
                )
                console_handler.setLevel(level)
                # Rich handler doesn't need a formatter
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)
                if structured:
                    formatter = StructuredFormatter()
                else:
                    formatter = ColoredFormatter(format_string)
                console_handler.setFormatter(formatter)

            self.handlers.append(console_handler)

        # Setup file handler with rotation
        if file_output and self.log_dir:
            log_path = self.log_dir / log_file
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)

            if structured:
                formatter = StructuredFormatter()
            else:
                # File logs don't need colors
                formatter = logging.Formatter(format_string)

            file_handler.setFormatter(formatter)
            self.handlers.append(file_handler)

            # Also add a time-based rotating handler for archival
            if not structured:  # Only add for non-structured logs
                time_handler = logging.handlers.TimedRotatingFileHandler(
                    self.log_dir / f"archive_{log_file}",
                    when='midnight',
                    interval=1,
                    backupCount=30,
                    encoding='utf-8'
                )
                time_handler.setLevel(level)
                time_handler.setFormatter(logging.Formatter(format_string))
                self.handlers.append(time_handler)

        # Install rich traceback if available
        if use_rich and RICH_AVAILABLE:
            install_rich_traceback(show_locals=True)

    def _clear_handlers(self) -> None:
        """Remove all existing handlers."""
        for handler in self.handlers:
            handler.close()
        self.handlers.clear()

        # Also clear handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.

        Args:
            name: Name for the logger (typically __name__)

        Returns:
            Configured logger instance
        """
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(self.level)
        logger.handlers.clear()
        logger.propagate = False

        # Add all configured handlers
        for handler in self.handlers:
            logger.addHandler(handler)

        self.loggers[name] = logger
        return logger

    def set_level(self, level: Union[str, int]) -> None:
        """
        Change the logging level for all loggers.

        Args:
            level: New logging level
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.level = level

        for logger in self.loggers.values():
            logger.setLevel(level)

        for handler in self.handlers:
            handler.setLevel(level)


# Global logger manager instance
_logger_manager = LoggerManager()


def setup_logger(
    level: Union[str, int] = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    log_file: str = "effgen.log",
    console_output: bool = True,
    file_output: bool = True,
    structured: bool = False,
    use_rich: bool = True,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> None:
    """
    Setup logging configuration for the application.

    This is the main entry point for configuring logging in effGen.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_file: Name of the log file
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        structured: Whether to use structured JSON logging
        use_rich: Whether to use rich formatting (if available)
        max_file_size: Maximum size of log file before rotation (bytes)
        backup_count: Number of backup files to keep
        format_string: Custom format string for log messages

    Example:
        >>> setup_logger(
        ...     level="DEBUG",
        ...     log_dir="./logs",
        ...     structured=True,
        ...     use_rich=True
        ... )
    """
    _logger_manager.setup(
        level=level,
        log_dir=log_dir,
        log_file=log_file,
        console_output=console_output,
        file_output=file_output,
        structured=structured,
        use_rich=use_rich,
        max_file_size=max_file_size,
        backup_count=backup_count,
        format_string=format_string,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting agent")
    """
    return _logger_manager.get_logger(name)


def set_log_level(level: Union[str, int]) -> None:
    """
    Change the logging level for all loggers.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> set_log_level("DEBUG")
    """
    _logger_manager.set_level(level)


@contextmanager
def LogContext(**kwargs):
    """
    Context manager for adding structured logging context.

    Context data will be included in all log messages within the context.
    This is useful for tracking request IDs, agent IDs, session IDs, etc.

    Args:
        **kwargs: Key-value pairs to add to the logging context

    Example:
        >>> with LogContext(agent_id="agent-001", task_id="task-123"):
        ...     logger.info("Processing task")
        ...     # Logs will include agent_id and task_id
    """
    # Get or create context
    if not hasattr(_context_storage, 'context'):
        _context_storage.context = {}

    # Save previous context
    previous_context = _context_storage.context.copy()

    # Add new context
    _context_storage.context.update(kwargs)

    try:
        yield
    finally:
        # Restore previous context
        _context_storage.context = previous_context


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    message: str = "An error occurred",
    level: int = logging.ERROR,
    **extra_fields
) -> None:
    """
    Log an exception with full traceback and additional context.

    Args:
        logger: Logger instance to use
        exception: Exception to log
        message: Custom message to include
        level: Log level to use
        **extra_fields: Additional fields to include in the log

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(logger, e, "Operation failed", task_id="123")
    """
    extra_fields['exception_type'] = type(exception).__name__
    extra_fields['exception_message'] = str(exception)

    logger.log(
        level,
        f"{message}: {exception}",
        exc_info=True,
        extra=extra_fields
    )


class PerformanceLogger:
    """
    Logger decorator for tracking function execution time.

    Can be used as a decorator or context manager to automatically log
    function execution times.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        """
        Initialize the performance logger.

        Args:
            logger: Logger instance to use (creates one if not provided)
            level: Log level for performance logs
        """
        self.logger = logger or get_logger(__name__)
        self.level = level
        self.start_time: Optional[float] = None

    def __call__(self, func):
        """
        Decorator usage.

        Example:
            >>> @PerformanceLogger(logger)
            ... def slow_function():
            ...     time.sleep(1)
        """
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                self.logger.log(
                    self.level,
                    f"Function '{func.__name__}' took {duration:.4f}s",
                    extra={'function': func.__name__, 'duration': duration}
                )
        return wrapper

    def __enter__(self):
        """
        Context manager entry.

        Example:
            >>> with PerformanceLogger(logger) as perf:
            ...     time.sleep(1)
        """
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        import time
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.log(
                self.level,
                f"Code block took {duration:.4f}s",
                extra={'duration': duration}
            )


# Convenience functions for common logging patterns
def log_agent_start(logger: logging.Logger, agent_id: str, **metadata) -> None:
    """Log agent startup with metadata."""
    logger.info(
        f"Agent started: {agent_id}",
        extra={'event': 'agent_start', 'agent_id': agent_id, **metadata}
    )


def log_agent_stop(logger: logging.Logger, agent_id: str, **metadata) -> None:
    """Log agent shutdown with metadata."""
    logger.info(
        f"Agent stopped: {agent_id}",
        extra={'event': 'agent_stop', 'agent_id': agent_id, **metadata}
    )


def log_task_start(logger: logging.Logger, task_id: str, **metadata) -> None:
    """Log task start with metadata."""
    logger.info(
        f"Task started: {task_id}",
        extra={'event': 'task_start', 'task_id': task_id, **metadata}
    )


def log_task_complete(logger: logging.Logger, task_id: str, **metadata) -> None:
    """Log task completion with metadata."""
    logger.info(
        f"Task completed: {task_id}",
        extra={'event': 'task_complete', 'task_id': task_id, **metadata}
    )


def log_task_error(logger: logging.Logger, task_id: str, error: Exception, **metadata) -> None:
    """Log task error with metadata."""
    log_exception(
        logger,
        error,
        f"Task failed: {task_id}",
        event='task_error',
        task_id=task_id,
        **metadata
    )


def log_tool_call(logger: logging.Logger, tool_name: str, **metadata) -> None:
    """Log tool invocation with metadata."""
    logger.debug(
        f"Tool called: {tool_name}",
        extra={'event': 'tool_call', 'tool_name': tool_name, **metadata}
    )


def log_model_inference(logger: logging.Logger, model_name: str, **metadata) -> None:
    """Log model inference with metadata."""
    logger.debug(
        f"Model inference: {model_name}",
        extra={'event': 'model_inference', 'model_name': model_name, **metadata}
    )


__all__ = [
    'LogLevel',
    'StructuredFormatter',
    'ColoredFormatter',
    'LoggerManager',
    'setup_logger',
    'get_logger',
    'set_log_level',
    'LogContext',
    'log_exception',
    'PerformanceLogger',
    'log_agent_start',
    'log_agent_stop',
    'log_task_start',
    'log_task_complete',
    'log_task_error',
    'log_tool_call',
    'log_model_inference',
]
