"""
Validation utilities for effGen framework.

This module provides comprehensive validation capabilities including input validation,
schema validation, type checking, and data sanitization for the effGen framework.

Features:
    - Input validation helpers for common data types
    - JSON Schema validation
    - Pydantic model validation
    - Type checking utilities
    - Data sanitization and normalization
    - Custom validator creation
    - Validation error handling and reporting

Example:
    Basic validation:
        >>> from effgen.utils.validators import validate_model_name, validate_path
        >>> validate_model_name("gpt-4")  # Returns True
        >>> validate_path("/path/to/file")  # Validates path exists

    Schema validation:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> validate_json_schema({"name": "test"}, schema)

    Custom validators:
        >>> @require_type(str, int)
        ... def process_data(text, count):
        ...     pass
"""

import re
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Type, get_type_hints
from functools import wraps
from urllib.parse import urlparse

try:
    from jsonschema import validate as json_validate
    from jsonschema import ValidationError as JSONSchemaValidationError
    from jsonschema import Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class ValidationError(Exception):
    """
    Custom exception for validation errors.

    Attributes:
        message: Error message
        field: Field that failed validation
        value: Value that failed validation
        errors: List of detailed error messages
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        errors: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value
        self.errors = errors or []

    def __str__(self) -> str:
        """Return detailed error message."""
        parts = [self.message]
        if self.field:
            parts.append(f"Field: {self.field}")
        if self.value is not None:
            parts.append(f"Value: {self.value}")
        if self.errors:
            parts.append(f"Errors: {', '.join(self.errors)}")
        return " | ".join(parts)


# ============================================================================
# Type Validators
# ============================================================================

def validate_type(value: Any, expected_type: Type, allow_none: bool = False) -> bool:
    """
    Validate that a value is of the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type
        allow_none: Whether to allow None values

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if value is None and allow_none:
        return True

    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Expected type {expected_type.__name__}, got {type(value).__name__}",
            value=value
        )

    return True


def validate_string(
    value: Any,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    allow_empty: bool = True,
) -> bool:
    """
    Validate a string value.

    Args:
        value: Value to validate
        min_length: Minimum length
        max_length: Maximum length
        pattern: Regex pattern to match
        allow_empty: Whether to allow empty strings

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value).__name__}", value=value)

    if not allow_empty and len(value) == 0:
        raise ValidationError("String cannot be empty", value=value)

    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"String length {len(value)} is less than minimum {min_length}",
            value=value
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"String length {len(value)} exceeds maximum {max_length}",
            value=value
        )

    if pattern is not None and not re.match(pattern, value):
        raise ValidationError(
            f"String does not match pattern {pattern}",
            value=value
        )

    return True


def validate_number(
    value: Any,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allow_negative: bool = True,
) -> bool:
    """
    Validate a numeric value.

    Args:
        value: Value to validate
        min_value: Minimum value
        max_value: Maximum value
        allow_negative: Whether to allow negative numbers

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"Expected number, got {type(value).__name__}", value=value)

    if not allow_negative and value < 0:
        raise ValidationError("Negative numbers not allowed", value=value)

    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Value {value} is less than minimum {min_value}",
            value=value
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Value {value} exceeds maximum {max_value}",
            value=value
        )

    return True


def validate_list(
    value: Any,
    item_type: Optional[Type] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: bool = True,
) -> bool:
    """
    Validate a list value.

    Args:
        value: Value to validate
        item_type: Expected type of list items
        min_length: Minimum length
        max_length: Maximum length
        allow_empty: Whether to allow empty lists

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationError(f"Expected list, got {type(value).__name__}", value=value)

    if not allow_empty and len(value) == 0:
        raise ValidationError("List cannot be empty", value=value)

    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"List length {len(value)} is less than minimum {min_length}",
            value=value
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"List length {len(value)} exceeds maximum {max_length}",
            value=value
        )

    if item_type is not None:
        for i, item in enumerate(value):
            if not isinstance(item, item_type):
                raise ValidationError(
                    f"List item at index {i} is not of type {item_type.__name__}",
                    value=item
                )

    return True


def validate_dict(
    value: Any,
    required_keys: Optional[List[str]] = None,
    allowed_keys: Optional[List[str]] = None,
    key_type: Optional[Type] = None,
    value_type: Optional[Type] = None,
) -> bool:
    """
    Validate a dictionary value.

    Args:
        value: Value to validate
        required_keys: Keys that must be present
        allowed_keys: Only these keys are allowed
        key_type: Expected type of keys
        value_type: Expected type of values

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError(f"Expected dict, got {type(value).__name__}", value=value)

    if required_keys:
        missing_keys = set(required_keys) - set(value.keys())
        if missing_keys:
            raise ValidationError(
                f"Missing required keys: {', '.join(missing_keys)}",
                value=value
            )

    if allowed_keys:
        extra_keys = set(value.keys()) - set(allowed_keys)
        if extra_keys:
            raise ValidationError(
                f"Extra keys not allowed: {', '.join(extra_keys)}",
                value=value
            )

    if key_type is not None:
        for key in value.keys():
            if not isinstance(key, key_type):
                raise ValidationError(
                    f"Key '{key}' is not of type {key_type.__name__}",
                    value=key
                )

    if value_type is not None:
        for key, val in value.items():
            if not isinstance(val, value_type):
                raise ValidationError(
                    f"Value for key '{key}' is not of type {value_type.__name__}",
                    value=val
                )

    return True


# ============================================================================
# Path and File Validators
# ============================================================================

def validate_path(
    path: Union[str, Path],
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_if_missing: bool = False,
) -> bool:
    """
    Validate a file system path.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        must_be_file: Whether the path must be a file
        must_be_dir: Whether the path must be a directory
        create_if_missing: Create directory if it doesn't exist

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError(f"Expected path, got {type(path).__name__}", value=path)

    path = Path(path)

    if must_exist and not path.exists():
        if create_if_missing and must_be_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValidationError(f"Path does not exist: {path}", value=str(path))

    if path.exists():
        if must_be_file and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}", value=str(path))

        if must_be_dir and not path.is_dir():
            raise ValidationError(f"Path is not a directory: {path}", value=str(path))

    return True


def validate_file_extension(
    path: Union[str, Path],
    allowed_extensions: List[str],
) -> bool:
    """
    Validate file extension.

    Args:
        path: File path
        allowed_extensions: List of allowed extensions (e.g., ['.txt', '.json'])

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    path = Path(path)
    ext = path.suffix.lower()

    allowed_extensions = [e.lower() if e.startswith('.') else f'.{e.lower()}'
                         for e in allowed_extensions]

    if ext not in allowed_extensions:
        raise ValidationError(
            f"File extension '{ext}' not in allowed extensions: {allowed_extensions}",
            value=str(path)
        )

    return True


# ============================================================================
# URL and Network Validators
# ============================================================================

def validate_url(
    url: str,
    require_scheme: bool = True,
    allowed_schemes: Optional[List[str]] = None,
) -> bool:
    """
    Validate a URL.

    Args:
        url: URL to validate
        require_scheme: Whether to require a scheme (http://, https://, etc.)
        allowed_schemes: List of allowed schemes

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(url, str):
        raise ValidationError(f"Expected string, got {type(url).__name__}", value=url)

    try:
        result = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL: {e}", value=url)

    if require_scheme and not result.scheme:
        raise ValidationError("URL missing scheme (http://, https://, etc.)", value=url)

    if allowed_schemes and result.scheme not in allowed_schemes:
        raise ValidationError(
            f"URL scheme '{result.scheme}' not in allowed schemes: {allowed_schemes}",
            value=url
        )

    return True


def validate_email(email: str) -> bool:
    """
    Validate an email address.

    Args:
        email: Email address to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError("Invalid email address", value=email)
    return True


# ============================================================================
# Model and Configuration Validators
# ============================================================================

def validate_model_name(model_name: str) -> bool:
    """
    Validate a model name.

    Args:
        model_name: Model name to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValidationError("Model name must be a non-empty string", value=model_name)

    # Allow alphanumeric, hyphens, underscores, dots, and slashes (for HF models)
    pattern = r'^[a-zA-Z0-9._/-]+$'
    if not re.match(pattern, model_name):
        raise ValidationError(
            "Model name contains invalid characters. "
            "Only alphanumeric, hyphens, underscores, dots, and slashes are allowed.",
            value=model_name
        )

    return True


def validate_config_dict(
    config: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
    field_types: Optional[Dict[str, Type]] = None,
) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names
        field_types: Dictionary mapping field names to expected types

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Expected dict, got {type(config).__name__}", value=config)

    errors = []

    # Check required fields
    if required_fields:
        missing = [field for field in required_fields if field not in config]
        if missing:
            errors.append(f"Missing required fields: {', '.join(missing)}")

    # Check field types
    if field_types:
        for field, expected_type in field_types.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field '{field}': expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

    if errors:
        raise ValidationError("Configuration validation failed", errors=errors)

    return True


# ============================================================================
# JSON Schema Validators
# ============================================================================

def validate_json_schema(data: Any, schema: Dict[str, Any]) -> bool:
    """
    Validate data against a JSON schema.

    Args:
        data: Data to validate
        schema: JSON schema

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails or jsonschema not available
    """
    if not JSONSCHEMA_AVAILABLE:
        raise ValidationError(
            "jsonschema package not available. Install with: pip install jsonschema"
        )

    try:
        json_validate(instance=data, schema=schema)
        return True
    except JSONSchemaValidationError as e:
        raise ValidationError(
            f"JSON schema validation failed: {e.message}",
            field=".".join(str(p) for p in e.path),
            value=e.instance,
            errors=[e.message]
        )


def load_and_validate_json(
    filepath: Union[str, Path],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load and validate a JSON file.

    Args:
        filepath: Path to JSON file
        schema: Optional JSON schema to validate against

    Returns:
        Loaded and validated JSON data

    Raises:
        ValidationError: If validation fails
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise ValidationError(f"File does not exist: {filepath}", value=str(filepath))

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}", value=str(filepath))

    if schema:
        validate_json_schema(data, schema)

    return data


# ============================================================================
# Pydantic Validators
# ============================================================================

def validate_pydantic_model(data: Any, model_class: Type[BaseModel]) -> BaseModel:
    """
    Validate data against a Pydantic model.

    Args:
        data: Data to validate
        model_class: Pydantic model class

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If validation fails or Pydantic not available
    """
    if not PYDANTIC_AVAILABLE:
        raise ValidationError(
            "Pydantic package not available. Install with: pip install pydantic"
        )

    try:
        return model_class(**data) if isinstance(data, dict) else model_class(data)
    except PydanticValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        raise ValidationError(
            "Pydantic validation failed",
            errors=errors
        )


# ============================================================================
# Data Sanitization
# ============================================================================

def sanitize_string(
    value: str,
    remove_whitespace: bool = False,
    remove_special_chars: bool = False,
    lowercase: bool = False,
    max_length: Optional[int] = None,
) -> str:
    """
    Sanitize a string value.

    Args:
        value: String to sanitize
        remove_whitespace: Remove all whitespace
        remove_special_chars: Remove special characters
        lowercase: Convert to lowercase
        max_length: Maximum length (truncate if longer)

    Returns:
        Sanitized string
    """
    result = str(value)

    if remove_whitespace:
        result = re.sub(r'\s+', '', result)
    else:
        result = result.strip()
        result = re.sub(r'\s+', ' ', result)

    if remove_special_chars:
        result = re.sub(r'[^a-zA-Z0-9\s]', '', result)

    if lowercase:
        result = result.lower()

    if max_length and len(result) > max_length:
        result = result[:max_length]

    return result


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to be safe for file systems.

    Args:
        filename: Filename to sanitize
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Remove path separators and other unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Ensure not empty
    if not filename:
        filename = 'unnamed'

    # Truncate if too long
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    return filename


# ============================================================================
# Decorator Validators
# ============================================================================

def require_type(*expected_types: Type):
    """
    Decorator to validate function argument types.

    Args:
        *expected_types: Expected types for each argument

    Example:
        >>> @require_type(str, int)
        ... def process(name, count):
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate positional arguments
            for i, (arg, expected_type) in enumerate(zip(args, expected_types)):
                if not isinstance(arg, expected_type):
                    raise ValidationError(
                        f"Argument {i} must be of type {expected_type.__name__}, "
                        f"got {type(arg).__name__}",
                        value=arg
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(**validators: Callable):
    """
    Decorator to validate function arguments using custom validators.

    Args:
        **validators: Mapping of argument names to validator functions

    Example:
        >>> @validate_input(name=lambda x: len(x) > 0, count=lambda x: x > 0)
        ... def process(name, count):
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = func.__code__.co_varnames[:func.__code__.co_argcount]

            # Combine positional and keyword arguments
            all_args = dict(zip(sig, args))
            all_args.update(kwargs)

            # Validate each argument
            for arg_name, validator in validators.items():
                if arg_name in all_args:
                    value = all_args[arg_name]
                    try:
                        if not validator(value):
                            raise ValidationError(
                                f"Validation failed for argument '{arg_name}'",
                                field=arg_name,
                                value=value
                            )
                    except Exception as e:
                        if isinstance(e, ValidationError):
                            raise
                        raise ValidationError(
                            f"Validation error for argument '{arg_name}': {e}",
                            field=arg_name,
                            value=value
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_output(validator: Callable):
    """
    Decorator to validate function return value.

    Args:
        validator: Function to validate the return value

    Example:
        >>> @validate_output(lambda x: isinstance(x, int) and x > 0)
        ... def get_count():
        ...     return 42
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                if not validator(result):
                    raise ValidationError(
                        f"Return value validation failed for {func.__name__}",
                        value=result
                    )
            except Exception as e:
                if isinstance(e, ValidationError):
                    raise
                raise ValidationError(
                    f"Return value validation error: {e}",
                    value=result
                )
            return result
        return wrapper
    return decorator


__all__ = [
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
