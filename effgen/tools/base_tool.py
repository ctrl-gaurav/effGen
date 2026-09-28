"""
Base tool interface for the effGen framework.

This module provides the abstract base class that all tools must inherit from,
ensuring consistent interfaces for tool metadata, parameter validation, and execution.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from effgen.errors import with_next_step


def _redact_error(text: str) -> str:
    """Scrub secrets from a tool error message before it is surfaced.

    Tool inputs (and the exceptions raised while processing them) can contain
    API keys, bearer tokens, or webhook URLs. A ``ToolResult.error`` flows into
    logs, traces, and the model's context, so it must be redacted on every
    path. Falls back to the raw text only if the redactor cannot be imported —
    redaction must never itself break tool execution.
    """
    try:
        from effgen.observability.redact import get_redactor

        return get_redactor().scrub(text)
    except Exception:  # noqa: BLE001 - redaction is best-effort, never fatal
        return text


def _http_status_of(exc: BaseException) -> int | None:
    """The HTTP status an HTTP client's error carries, or ``None``.

    Read from the error's ``response`` (``httpx``, ``requests``) or from the
    error itself (``urllib``'s ``HTTPError`` carries ``code`` and ``status``).
    """
    response = getattr(exc, "response", None)
    for holder in (response, exc):
        for name in ("status_code", "status", "code"):
            value = getattr(holder, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _inner_status(output: Any) -> tuple[bool | None, str | None]:
    """Detect a self-reported failure inside a tool's returned value.

    Builtin tools follow a ``{"success": bool, "error"/"message": str}``
    convention for outcomes that aren't exceptions (missing files, blocked
    paths, runtime errors in sandboxed code). Returns ``(False, message)`` when
    the payload explicitly reports failure, otherwise ``(None, None)`` so the
    caller leaves the outer status untouched.

    Only an explicit ``success is False`` is treated as a failure. A bare
    ``error``/``valid`` key is intentionally ignored, because some tools report
    domain results that way (e.g. JSON validation returning ``valid: False`` is
    a *successful* check, not a tool failure).
    """
    if isinstance(output, dict) and output.get("success") is False:
        msg = output.get("error") or output.get("message")
        return False, str(msg) if msg else None
    return None, None


class ToolCategory(Enum):
    """Categories for organizing tools."""
    INFORMATION_RETRIEVAL = "information_retrieval"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATIONS = "file_operations"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"


class ParameterType(Enum):
    """Supported parameter types for tool inputs."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    items_type: ParameterType | None = None  # For arrays
    # When False, the parameter is hidden from the model-facing tool schema
    # (``to_json_schema`` / ReAct prompt listings) so a model can never emit it,
    # while developer code may still pass it explicitly to ``execute()``. Use for
    # safety toggles that must stay under developer control (never LLM control).
    model_facing: bool = True

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """
        Validate a value against this parameter specification.

        Args:
            value: The value to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check required
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, None

        # Type validation
        type_checks = {
            ParameterType.STRING: lambda v: isinstance(v, str),
            ParameterType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            ParameterType.FLOAT: lambda v: isinstance(v, int | float) and not isinstance(v, bool),
            ParameterType.BOOLEAN: lambda v: isinstance(v, bool),
            ParameterType.ARRAY: lambda v: isinstance(v, list | tuple),
            ParameterType.OBJECT: lambda v: isinstance(v, dict),
            ParameterType.ANY: lambda v: True,
        }

        if self.type in type_checks and not type_checks[self.type](value):
            return False, f"Parameter '{self.name}' must be of type {self.type.value}"

        # Enum validation
        if self.enum is not None and value not in self.enum:
            return False, f"Parameter '{self.name}' must be one of {self.enum}"

        # Numeric range validation
        if self.type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"

        # String length validation
        if self.type == ParameterType.STRING:
            if self.min_length is not None and len(value) < self.min_length:
                return False, f"Parameter '{self.name}' must have length >= {self.min_length}"
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"Parameter '{self.name}' must have length <= {self.max_length}"

        # Array items validation
        if self.type == ParameterType.ARRAY and self.items_type:
            type_check = type_checks.get(self.items_type)
            if type_check:
                for i, item in enumerate(value):
                    if not type_check(item):
                        return False, f"Parameter '{self.name}[{i}]' must be of type {self.items_type.value}"

        return True, None


@dataclass
class ToolMetadata:
    """Metadata describing a tool's capabilities and requirements."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ParameterSpec] = field(default_factory=list)
    returns: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    author: str | None = None
    requires_auth: bool = False
    requires_api_key: bool = False
    cost_estimate: str = "low"  # low, medium, high
    timeout_seconds: int = 30
    tags: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    requires_approval: bool = False

    @property
    def model_facing_parameters(self) -> list[ParameterSpec]:
        """Parameters exposed to the model (those with ``model_facing=True``).

        Safety toggles marked ``model_facing=False`` are excluded so a model can
        never set them via tool-calling; developer code may still pass them to
        ``execute()`` directly.
        """
        return [p for p in self.parameters if getattr(p, "model_facing", True)]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "version": self.version,
            "author": self.author,
            "requires_auth": self.requires_auth,
            "requires_api_key": self.requires_api_key,
            "cost_estimate": self.cost_estimate,
            "timeout_seconds": self.timeout_seconds,
            "tags": self.tags,
            "examples": self.examples,
        }

    def to_json_schema(self) -> dict[str, Any]:
        """
        Convert metadata to JSON Schema format for LLM function calling.

        Returns:
            Dict: JSON Schema representation
        """
        # Map effGen ParameterType.value → JSON Schema type.
        # Note: "any" is not a valid JSON Schema type — we drop the
        # "type" key entirely and let OpenAI/Anthropic accept any value.
        # "float" is not valid either — JSON Schema uses "number".
        _TYPE_MAP = {
            "string": "string",
            "integer": "integer",
            "float": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }

        properties = {}
        required = []

        # Only advertise model-facing parameters: safety toggles flagged
        # model_facing=False must never appear in the schema a model sees.
        for param in self.model_facing_parameters:
            json_type = _TYPE_MAP.get(param.type.value)
            prop: dict[str, Any] = {"description": param.description}
            if json_type is not None:
                prop["type"] = json_type
            # ParameterType.ANY → omit type; some strict-schema providers
            # require an array of allowed types or schema-by-example.

            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            if param.min_value is not None:
                prop["minimum"] = param.min_value
            if param.max_value is not None:
                prop["maximum"] = param.max_value
            if param.min_length is not None:
                prop["minLength"] = param.min_length
            if param.max_length is not None:
                prop["maxLength"] = param.max_length
            if param.items_type:
                items_json_type = _TYPE_MAP.get(param.items_type.value)
                if items_json_type is not None:
                    prop["items"] = {"type": items_json_type}

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            }
        }

        if required:
            schema["parameters"]["required"] = required

        return schema


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the effGen framework.

    All tools must inherit from this class and implement the required methods.
    This ensures a consistent interface across all tools for metadata, validation,
    and execution.

    Example:
        class MyTool(BaseTool):
            def __init__(self):
                super().__init__(
                    metadata=ToolMetadata(
                        name="my_tool",
                        description="Does something useful",
                        category=ToolCategory.COMPUTATION,
                        parameters=[
                            ParameterSpec(
                                name="input",
                                type=ParameterType.STRING,
                                description="Input value",
                                required=True
                            )
                        ]
                    )
                )

            async def _execute(self, input: str, **kwargs) -> Any:
                # Tool implementation
                return f"Processed: {input}"
    """

    # Candidate names for the single "which action" selector parameter. Tools
    # name it inconsistently (``operation``/``action``/``op``/``mode``); we treat
    # them as interchangeable so a model that guesses any of them succeeds.
    _SELECTOR_PARAM_NAMES = ("operation", "action", "op", "mode")

    # Optional per-tool map of friendly/natural selector values to the canonical
    # enum value (e.g. ``{"current_time": "now"}``). Subclasses override this so
    # callers can use the obvious verb instead of memorizing the exact enum.
    operation_aliases: dict[str, str] = {}

    #: Whether this tool's output is retrieved source material rather than a
    #: computed answer. The agent loop reads it when the same call repeats: a
    #: computed result that repeats is a confident answer and is returned, while
    #: retrieved context is not — the model is given one tool-free turn to write
    #: the answer from what it already has. Tools in the
    #: ``INFORMATION_RETRIEVAL`` category are treated this way without setting
    #: it; set it on a tool whose category says otherwise (a file tool narrowed
    #: to reading, for instance).
    is_context_retrieval: bool = False

    def __init__(self, metadata: ToolMetadata) -> None:
        """
        Initialize the tool with metadata.

        Args:
            metadata: Tool metadata specification
        """
        self._metadata = metadata
        self._initialized = False
        self._dependencies: list[str] = []

    def _selector_param(self) -> str | None:
        """Return this tool's canonical action-selector parameter name, if any."""
        names = {p.name for p in self._metadata.parameters}
        for cand in self._SELECTOR_PARAM_NAMES:
            if cand in names:
                return cand
        return None

    def _normalize_selector(self, kwargs: dict) -> dict:
        """Accept selector synonyms (param name and value) before validation.

        Two ergonomics fixes happen here:

        * **Parameter name** — if the tool's selector is ``operation`` but the
          caller passed ``action`` (or vice-versa), move the value onto the
          canonical name so it is not stripped as "unknown".
        * **Selector value** — map natural verbs (``current_time`` → ``now``,
          ``parse`` → ``query`` …) declared in :attr:`operation_aliases` onto the
          real enum value.
        """
        selector = self._selector_param()
        if selector is None:
            return kwargs

        # Param-name aliasing: only when the canonical name is absent, and never
        # consume a synonym that is itself a *distinct* declared parameter of this
        # tool (e.g. data_analysis has both ``operation`` and ``op`` — ``op`` is a
        # real filter-operator value, not a misnamed selector).
        param_names = {p.name for p in self._metadata.parameters}
        if selector not in kwargs or kwargs.get(selector) is None:
            for alias in self._SELECTOR_PARAM_NAMES:
                if alias == selector or alias in param_names:
                    continue
                if kwargs.get(alias) is not None:
                    kwargs[selector] = kwargs.pop(alias)
                    break

        # Value aliasing.
        value = kwargs.get(selector)
        if isinstance(value, str) and self.operation_aliases:
            mapped = self.operation_aliases.get(value)
            if mapped is None:
                mapped = self.operation_aliases.get(value.lower())
            if mapped is not None:
                kwargs[selector] = mapped
        return kwargs

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return self._metadata

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._metadata.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self._metadata.description

    @property
    def category(self) -> ToolCategory:
        """Get tool category."""
        return self._metadata.category

    @property
    def dependencies(self) -> list[str]:
        """Get tool dependencies (names of other required tools)."""
        return self._dependencies

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        """
        Validate input parameters against the tool's parameter specifications.

        Args:
            **kwargs: Parameters to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check for unknown parameters — warn and strip rather than reject,
        # since SLMs often hallucinate extra parameters
        known_params = {p.name for p in self._metadata.parameters}
        unknown = set(kwargs.keys()) - known_params
        if unknown:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                f"Tool '{self._metadata.name}': ignoring unknown parameters {unknown}"
            )
            for k in unknown:
                del kwargs[k]

        # Validate each parameter
        for param_spec in self._metadata.parameters:
            value = kwargs.get(param_spec.name)
            is_valid, error = param_spec.validate(value)
            if not is_valid:
                # Friendlier guidance for selector/enum mistakes: list the
                # allowed values (and any natural-name aliases) instead of a
                # bare repr, so a model or human can self-correct.
                if param_spec.enum is not None and value is not None:
                    allowed = ", ".join(str(v) for v in param_spec.enum)
                    msg = (
                        f"Invalid {param_spec.name} '{value}'. "
                        f"Allowed: {allowed}."
                    )
                    if self.operation_aliases:
                        alias_str = ", ".join(
                            f"{a}->{c}" for a, c in sorted(self.operation_aliases.items())
                        )
                        msg += f" Aliases: {alias_str}."
                    return False, msg
                return False, error

        return True, None

    async def initialize(self) -> None:
        """
        Initialize the tool (load resources, connect to services, etc.).
        Called once before first use. Override in subclasses if needed.
        """
        self._initialized = True

    async def cleanup(self) -> None:
        """
        Clean up tool resources (close connections, release memory, etc.).
        Called when tool is no longer needed. Override in subclasses if needed.
        """
        self._initialized = False

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        Execute the tool's main functionality.

        This method must be implemented by all tool subclasses.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Any: Tool execution result

        Raises:
            Exception: Any errors during execution
        """
        pass

    def _validation_next_step(self, error: str) -> str:
        """Say what the call should have passed, read off this tool's own schema.

        A rejected call is the most common failure a caller reaches, and the
        rejection on its own ("Parameter 'operation' is required") names the
        symptom without saying what would have been accepted. The guidance is
        derived from the declared parameters rather than written per tool, so it
        stays accurate as a tool's schema changes.

        Args:
            error: The validation message, whose quoted parameter name selects
                the specification the guidance is built from.

        Returns:
            One sentence naming the accepted values for the rejected parameter
            when it declares an enum, and otherwise the parameters this tool
            requires with their types. Empty when the tool declares neither.
        """
        specs = {p.name: p for p in self._metadata.parameters if p.model_facing}
        match = re.search(r"'([^']+)'", error or "")
        offender = specs.get((match.group(1) if match else "").split("[")[0])

        if offender is not None and offender.enum:
            values = ", ".join(str(v) for v in offender.enum[:12])
            if len(offender.enum) > 12:
                values += ", …"
            return f"Accepted values for '{offender.name}': {values}."

        required = [p for p in specs.values() if p.required]
        if required:
            named = ", ".join(f"{p.name} ({p.type.value})" for p in required[:6])
            if len(required) > 6:
                named += ", …"
            return f"Pass the parameters '{self.name}' requires: {named}."

        if offender is not None:
            return f"Pass '{offender.name}' as {offender.type.value}."
        return ""

    def _coerce_parameters(self, kwargs: dict) -> dict:
        """Coerce LLM-supplied parameter values to their declared types.

        LLMs occasionally send a value as a string: an integer as ``"0"``, or a
        whole object as ``"{}"`` (Llama 3.2 fills optional object parameters
        that way). This method converts those to the declared type so downstream
        validation succeeds. An object/array string is only accepted when it
        parses as JSON *and* yields the declared type; anything else is left as
        it is for validation to reject.
        """
        coerced = dict(kwargs)
        param_map = {p.name: p for p in self._metadata.parameters}
        for name, value in list(coerced.items()):
            spec = param_map.get(name)
            if spec is None or value is None:
                continue
            if spec.type == ParameterType.INTEGER and isinstance(value, str):
                try:
                    coerced[name] = int(value)
                except (ValueError, TypeError):  # leave the raw value for schema validation to reject
                    pass
            elif spec.type == ParameterType.FLOAT and isinstance(value, str):
                try:
                    coerced[name] = float(value)
                except (ValueError, TypeError):  # leave the raw value for schema validation to reject
                    pass
            elif spec.type == ParameterType.BOOLEAN and isinstance(value, str):
                coerced[name] = value.lower() in ("true", "1", "yes")
            elif spec.type in (ParameterType.OBJECT, ParameterType.ARRAY) and isinstance(
                value, str
            ):
                try:
                    parsed = json.loads(value)
                except (ValueError, TypeError):  # leave the raw value for validation to reject
                    continue
                expected = dict if spec.type == ParameterType.OBJECT else list
                if isinstance(parsed, expected):
                    coerced[name] = parsed
        return coerced

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with parameter validation and error handling.

        This is the main entry point for tool execution. It handles:
        - Parameter validation
        - Tool initialization (if needed)
        - Execution timing
        - Error handling
        - Result formatting

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult: Execution result with metadata
        """
        start_time = time.time()

        try:
            # Accept selector synonyms (operation/action + natural verb values)
            kwargs = self._normalize_selector(kwargs)

            # Coerce string-typed numerics from LLM responses before validation
            kwargs = self._coerce_parameters(kwargs)

            # Validate parameters
            is_valid, error = self.validate_parameters(**kwargs)
            if not is_valid:
                return ToolResult(
                    success=False,
                    output=None,
                    error=_redact_error(
                        with_next_step(
                            f"Parameter validation failed: {error}",
                            self._validation_next_step(error or ""),
                        )
                    ),
                    execution_time=time.time() - start_time
                )

            # Initialize if needed
            if not self._initialized:
                await self.initialize()

            # Execute the tool
            output = await self._execute(**kwargs)

            execution_time = time.time() - start_time

            # Result envelope: many tools report their own outcome inside the
            # returned dict (``{"success": False, ...}``). Never bury a failure
            # inside a "successful" ToolResult — reflect the real outcome so the
            # outer and inner status agree.
            inner_ok, inner_err = _inner_status(output)
            if inner_ok is False:
                return ToolResult(
                    success=False,
                    output=output,
                    error=_redact_error(inner_err or "Tool reported a failure"),
                    execution_time=execution_time,
                    metadata={
                        "tool_name": self.name,
                        "tool_version": self._metadata.version,
                    },
                )

            return ToolResult(
                success=True,
                output=output,
                execution_time=execution_time,
                metadata={
                    "tool_name": self.name,
                    "tool_version": self._metadata.version,
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                output=None,
                error=_redact_error(f"Tool execution failed: {str(e)}"),
                execution_time=execution_time,
                metadata={
                    "tool_name": self.name,
                    "error_type": type(e).__name__,
                    # The whole class chain, so a caller can tell a
                    # connection or timeout failure from bad input without
                    # importing the library that raised it.
                    "error_classes": [
                        cls.__name__ for cls in type(e).__mro__ if cls is not object
                    ],
                    "error_status": _http_status_of(e),
                }
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', category='{self.category.value}')>"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
