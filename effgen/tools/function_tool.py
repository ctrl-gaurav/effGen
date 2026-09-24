"""Low-boilerplate tool authoring: ``@tool`` and ``Tool.from_function``.

The full :class:`~effgen.tools.base_tool.BaseTool` interface is powerful but
verbose — to add one tool you subclass it, hand-build a :class:`ToolMetadata`
with :class:`ParameterSpec` objects, and implement the async ``_execute`` hook.
Most users just want to turn a plain typed Python function into a tool.

This module is an **ergonomic wrapper over the existing BaseTool machinery**,
not a new tool subsystem. ``@tool`` (and the equivalent
``Tool.from_function(fn)``) derive the tool's name, description and JSON schema
from the function's name, signature, type hints and docstring, then wrap it in a
real ``BaseTool`` so it behaves identically wherever tool *instances* are used —
``AgentConfig(tools=[...])``, parameter validation, the async ``execute``
contract, and provider-native function-calling.

Example::

    from effgen import tool, Agent, AgentConfig

    @tool
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.\"\"\"
        return a + b

    agent = Agent(AgentConfig(name="calc", model="gpt-5-nano", tools=[add]))

The same function authored explicitly::

    from effgen import Tool

    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.\"\"\"
        return a + b

    add_tool = Tool.from_function(add)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import types
import typing
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

from .base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

__all__ = ["FunctionTool", "Tool", "tool"]

_logger = logging.getLogger(__name__)


# Python annotation -> effGen ParameterType. Anything unrecognised maps to ANY
# so the function still works; the model just gets a looser schema.
_TYPE_MAP: dict[type, ParameterType] = {
    str: ParameterType.STRING,
    int: ParameterType.INTEGER,
    float: ParameterType.FLOAT,
    bool: ParameterType.BOOLEAN,
    list: ParameterType.ARRAY,
    tuple: ParameterType.ARRAY,
    dict: ParameterType.OBJECT,
}


def _unwrap_annotation(annotation: Any) -> tuple[Any, list[Any] | None]:
    """Resolve Optional/Union/Literal/Annotated down to a concrete base type.

    Returns ``(base_type, enum_values)`` where *enum_values* is non-None only
    for ``typing.Literal[...]`` annotations.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return Any, None

    origin = get_origin(annotation)

    # Annotated[T, ...] -> T
    if origin is typing.Annotated:  # pragma: no cover - exercised via get_type_hints
        return _unwrap_annotation(get_args(annotation)[0])

    # Literal["a", "b"] -> enum of values, base type inferred from first value.
    if origin is typing.Literal:
        values = list(get_args(annotation))
        base = type(values[0]) if values else str
        return base, values

    # Optional[T] / Union[T, None] / PEP-604 ``T | None`` -> first non-None member.
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _unwrap_annotation(non_none[0])
        return Any, None

    return annotation, None


def _array_item_type(annotation: Any) -> ParameterType | None:
    """Resolve the element type of a ``list[T]``/``tuple[T, ...]`` annotation.

    Returns ``None`` when *annotation* isn't a parametrised sequence, or its
    element type isn't one ``ParameterType`` can represent — the caller then
    emits an array schema with no ``items``, same as before this existed.
    """
    args = get_args(annotation)
    if not args:
        return None
    elem, _ = _unwrap_annotation(args[0])
    elem = get_origin(elem) or elem
    if isinstance(elem, type):
        return _TYPE_MAP.get(elem)
    return None


def _param_type_of(annotation: Any) -> tuple[ParameterType, list[Any] | None, ParameterType | None]:
    base, enum = _unwrap_annotation(annotation)
    items_type = None
    origin = get_origin(base) or base
    if isinstance(origin, type) and origin in (list, tuple):
        items_type = _array_item_type(base)
    if isinstance(origin, type):
        return _TYPE_MAP.get(origin, ParameterType.ANY), enum, items_type
    return ParameterType.ANY, enum, items_type


def _parse_docstring(
    doc: str | None, known_params: set[str] | None = None
) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {param_name: description}).

    Understands the common ``Args:``/``Arguments:``/``Parameters:`` section with
    ``name: description`` or ``name (type): description`` lines (Google/NumPy
    style). *known_params* (the function's real parameter names) is used to tell
    a new param entry from a wrapped continuation line, rather than guessing
    from indentation. Best-effort — anything it can't parse simply
    yields no per-param description.
    """
    if not doc:
        return "", {}
    known = known_params or set()
    lines = inspect.cleandoc(doc).splitlines()

    summary_parts: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    current: str | None = None
    _arg_headers = ("args:", "arguments:", "parameters:", "params:")
    _other_headers = ("returns:", "return:", "raises:", "yields:", "examples:", "example:", "note:", "notes:")

    for raw in lines:
        line = raw.rstrip()
        low = line.strip().lower()
        if low in _arg_headers:
            in_args = True
            current = None
            continue
        if in_args and low in _other_headers:
            in_args = False
            current = None
            continue
        if in_args:
            stripped = line.strip()
            if not stripped:
                continue
            # "name: desc" or "name (type): desc" — only start a new entry when
            # the key is an actual parameter (avoids mis-splitting a wrapped
            # description that happens to contain a colon).
            started = False
            if ":" in stripped:
                key, _, desc = stripped.partition(":")
                key = key.split("(", 1)[0].strip()
                if key and " " not in key and (not known or key in known):
                    params[key] = desc.strip()
                    current = key
                    started = True
            # Continuation of the previous param description.
            if not started and current is not None:
                params[current] = (params[current] + " " + stripped).strip()
            continue
        summary_parts.append(line)

    summary = " ".join(p.strip() for p in summary_parts if p.strip()).strip()
    return summary, params


class FunctionTool(BaseTool):
    """A :class:`BaseTool` that delegates execution to a plain Python function.

    Built by :func:`tool` / :meth:`from_function`; you rarely construct it
    directly. Supports both synchronous and ``async def`` functions. The wrapped
    function's return value is passed straight through ``BaseTool.execute`` (so
    its success/failure envelope still applies).
    """

    def __init__(self, func: Callable[..., Any], metadata: ToolMetadata) -> None:
        super().__init__(metadata=metadata)
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func)
        # Expose the original callable so power users can still reach it.
        self.func = func

    async def _execute(self, **kwargs: Any) -> Any:
        if self._is_async:
            return await self._func(**kwargs)
        # Run sync functions off the event loop so a slow tool doesn't block
        # other concurrent tool/agent work. The worker thread runs in a copy of
        # the caller's context, as ``asyncio.to_thread`` would: an agent the
        # function starts is then counted as a child of the run that called the
        # tool — its calls, tokens and cost in that run's ledger total — and
        # its spans nest under the tool's.
        import contextvars
        from functools import partial

        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        return await loop.run_in_executor(
            None, partial(context.run, self._func, **kwargs),
        )

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        category: ToolCategory | str | None = None,
        requires_approval: bool = False,
        cost_estimate: str | None = None,
        timeout_seconds: int | None = None,
    ) -> "FunctionTool":
        """Build a ``FunctionTool`` from *func*'s signature, hints and docstring.

        Args:
            func: A plain (sync or async) Python function with type hints.
            name: Override the tool name (defaults to ``func.__name__``).
            description: Override the description (defaults to the docstring
                summary).
            category: Optional :class:`ToolCategory` (or its string value).
            requires_approval: Mark this tool as needing human approval before
                each call. Reaches ``AgentConfig(approval_mode="dangerous_only",
                approval_callback=...)``, which gates any tool with
                ``requires_approval=True`` (independent of name) alongside the
                tools matched by :data:`effgen.core.human_loop.
                DANGEROUS_TOOL_KEYWORDS`. Set this on any tool with a real-world
                side effect a user did not explicitly ask for on this call — a
                refund, a delete, a send, a purchase::

                    @tool(requires_approval=True)
                    def issue_refund(order_id: str, amount: float) -> str:
                        \"\"\"Refund an order. Requires approval before running.\"\"\"
                        ...
            cost_estimate: Optional cost hint surfaced in the tool's metadata
                (``"low"``/``"medium"``/``"high"``); defaults to ``"low"``.
            timeout_seconds: Optional per-call timeout hint surfaced in the
                tool's metadata; defaults to 30.

        Returns:
            A ready-to-use ``FunctionTool``.
        """
        if not callable(func):
            raise TypeError(
                f"@tool/Tool.from_function expects a callable, got {type(func).__name__}."
            )

        tool_name = name or getattr(func, "__name__", None)
        if not tool_name or tool_name == "<lambda>":
            raise ValueError(
                "Cannot derive a tool name from this function; pass name= explicitly."
            )

        try:
            hints = get_type_hints(func, include_extras=True)
        except Exception:
            # Forward refs / odd annotations shouldn't break authoring.
            hints = getattr(func, "__annotations__", {}) or {}

        sig = inspect.signature(func)
        _names = {
            p for p, pa in sig.parameters.items()
            if p not in ("self", "cls")
            and pa.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        summary, param_docs = _parse_docstring(getattr(func, "__doc__", None), _names)
        tool_desc = description or summary or f"Call the {tool_name} function."

        parameters: list[ParameterSpec] = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                # *args/**kwargs can't be described as named schema params.
                continue
            annotation = hints.get(pname, param.annotation)
            ptype, enum, items_type = _param_type_of(annotation)
            required = param.default is inspect.Parameter.empty
            default = None if param.default is inspect.Parameter.empty else param.default
            parameters.append(
                ParameterSpec(
                    name=pname,
                    type=ptype,
                    description=param_docs.get(pname, f"The {pname} parameter."),
                    required=required,
                    default=default,
                    enum=enum,
                    items_type=items_type,
                )
            )

        if isinstance(category, str):
            try:
                category = ToolCategory(category)
            except ValueError:
                valid = ", ".join(c.value for c in ToolCategory)
                _logger.warning(
                    "@tool: unknown category %r for '%s' — falling back to %s. "
                    "Valid values: %s.",
                    category, tool_name, ToolCategory.SYSTEM.value, valid,
                )
                category = None
        tool_category = category or ToolCategory.SYSTEM

        metadata_kwargs: dict[str, Any] = {
            "name": tool_name,
            "description": tool_desc,
            "category": tool_category,
            "parameters": parameters,
            "author": getattr(func, "__module__", None),
            "requires_approval": requires_approval,
        }
        if cost_estimate is not None:
            metadata_kwargs["cost_estimate"] = cost_estimate
        if timeout_seconds is not None:
            metadata_kwargs["timeout_seconds"] = timeout_seconds
        metadata = ToolMetadata(**metadata_kwargs)
        return cls(func, metadata)


# ``Tool`` is the documented facade so ``Tool.from_function(fn)`` reads well; it
# is the same class as ``FunctionTool``.
Tool = FunctionTool


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory | str | None = None,
    requires_approval: bool = False,
    cost_estimate: str | None = None,
    timeout_seconds: int | None = None,
) -> Any:
    """Decorator turning a plain function into a ready-to-use effGen tool.

    Works bare or parametrised::

        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b

        @tool(name="multiply", category="computation")
        def mul(a: int, b: int) -> int:
            \"\"\"Multiply two integers.\"\"\"
            return a * b

        @tool(requires_approval=True)
        def issue_refund(order_id: str, amount: float) -> str:
            \"\"\"Refund an order. Requires approval before running.\"\"\"
            ...

    The decorated name becomes a :class:`FunctionTool` instance you can drop
    into ``AgentConfig(tools=[...])`` or register with ``get_registry()``.
    Mark a tool with a real-world side effect (a refund, a delete, a send)
    ``requires_approval=True`` so it is gated when the agent runs with
    ``AgentConfig(approval_mode="dangerous_only", approval_callback=...)`` —
    see :func:`FunctionTool.from_function` for the full explanation.

    Args:
        func: The function being decorated, when used bare.
        name: Tool name, defaulting to the function's name.
        description: Tool description, defaulting to the function's docstring.
        category: The catalog category the tool is listed under.
        requires_approval: Gate the tool behind the agent's approval callback.
        cost_estimate: A short note on what one call costs.
        timeout_seconds: Seconds one call may run before it is stopped.

    Returns:
        The ``FunctionTool`` built from the decorated function when used bare,
        and otherwise the decorator that builds it.
    """

    def wrap(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool.from_function(
            f,
            name=name,
            description=description,
            category=category,
            requires_approval=requires_approval,
            cost_estimate=cost_estimate,
            timeout_seconds=timeout_seconds,
        )

    if func is not None:
        return wrap(func)
    return wrap
