"""
Utility modules for effGen framework.

This package provides comprehensive utility functions including logging,
metrics collection, validation, and other helper functions used throughout
the effGen framework.

Modules:
    - logging: Advanced logging with rich formatting and rotation
    - metrics: Performance metrics and cost tracking
    - validators: Input validation and schema checking

Example:
    Basic usage:
        >>> from effgen.utils import setup_logger, get_logger
        >>> from effgen.utils import MetricsCollector
        >>> from effgen.utils import validate_model_name

    Setup logging:
        >>> setup_logger(level="INFO", log_dir="./logs")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")

    Track metrics:
        >>> metrics = MetricsCollector()
        >>> with metrics.track_execution("task"):
        ...     perform_task()
        >>> print(metrics.get_summary())

    Validate inputs:
        >>> validate_model_name("microsoft/phi-2")
        >>> validate_path("/path/to/file", must_exist=True)
"""

# Logging utilities
from effgen.utils.logging import (
    LogLevel,
    StructuredFormatter,
    ColoredFormatter,
    LoggerManager,
    setup_logger,
    get_logger,
    set_log_level,
    LogContext,
    log_exception,
    PerformanceLogger,
    log_agent_start,
    log_agent_stop,
    log_task_start,
    log_task_complete,
    log_task_error,
    log_tool_call,
    log_model_inference,
)

# Metrics utilities
from effgen.utils.metrics import (
    MODEL_PRICING,
    ExecutionMetric,
    TokenMetric,
    ResourceSnapshot,
    MetricsCollector,
    get_global_metrics,
)

# Validation utilities
from effgen.utils.validators import (
    ValidationError,
    validate_type,
    validate_string,
    validate_number,
    validate_list,
    validate_dict,
    validate_path,
    validate_file_extension,
    validate_url,
    validate_email,
    validate_model_name,
    validate_config_dict,
    validate_json_schema,
    load_and_validate_json,
    validate_pydantic_model,
    sanitize_string,
    sanitize_filename,
    require_type,
    validate_input,
    validate_output,
)


__all__ = [
    # Logging
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

    # Metrics
    'MODEL_PRICING',
    'ExecutionMetric',
    'TokenMetric',
    'ResourceSnapshot',
    'MetricsCollector',
    'get_global_metrics',

    # Validators
    'ValidationError',
    'validate_type',
    'validate_string',
    'validate_number',
    'validate_list',
    'validate_dict',
    'validate_path',
    'validate_file_extension',
    'validate_url',
    'validate_email',
    'validate_model_name',
    'validate_config_dict',
    'validate_json_schema',
    'load_and_validate_json',
    'validate_pydantic_model',
    'sanitize_string',
    'sanitize_filename',
    'require_type',
    'validate_input',
    'validate_output',
]


# Version information
__version__ = '0.0.1'
