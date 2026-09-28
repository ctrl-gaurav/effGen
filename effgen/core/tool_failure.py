"""Whether a tool call failed on the tool's side, and what the loop learns of it.

A tool call can fail for two different reasons, and they deserve different
treatment. The model can hand the tool input it cannot use — an expression that
does not parse, a parameter of the wrong type — and a corrected call succeeds.
Or the tool itself cannot do its job right now — its service refuses the
connection, times out, answers with a server error, or has no credentials — and
no change to the input helps. The second is what this module recognises.

The decision is made from the exception's class only, never from its message:
the class is known exactly where the failure is caught, and a message is
wording that changes with the library and the locale. Classes from optional
HTTP libraries are matched by name along the exception's method resolution
order, so recognising them imports nothing.

The dispatch layer reports each failure to whoever is collecting for the call
in progress (:func:`collecting_failures`). The collector is a context variable
set around one dispatch, so one agent serving several runs at once keeps each
run's failures apart.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = [
    "TOOL_SIDE",
    "INPUT_SIDE",
    "ToolFailure",
    "collecting_failures",
    "error_class_names",
    "failure_side",
    "record_failure",
]

#: The tool could not do its job: the service, the network or the setup failed.
TOOL_SIDE = "tool"
#: The call's input was the problem, or the cause is not known to be the tool's.
INPUT_SIDE = "input"

#: Exception class names that mean the tool's side failed, wherever they appear
#: in the exception's method resolution order. Builtins cover ``socket`` and
#: ``asyncio``/``concurrent.futures`` timeouts (both are ``TimeoutError``);
#: the rest are the connection and timeout classes of the common HTTP clients,
#: and effGen's own "not configured on this machine" errors.
_TOOL_SIDE_CLASS_NAMES = frozenset({
    "ConnectionError",          # builtin, and requests.exceptions.ConnectionError
    "TimeoutError",             # builtin, asyncio, concurrent.futures
    "gaierror",                 # socket name resolution
    "herror",
    "TransportError",           # httpx
    "TimeoutException",         # httpx
    "NetworkError",             # httpx
    "RemoteProtocolError",      # httpx
    "Timeout",                  # requests
    "ClientConnectionError",    # aiohttp
    "ServerTimeoutError",       # aiohttp
    "URLError",                 # urllib
    "MissingCredentialsError",  # effgen.errors
    "MissingSystemDependency",  # effgen.errors
    "CircuitOpen",              # the dispatch layer's own refusal
})

#: HTTP error classes whose meaning depends on the status they carry.
_HTTP_STATUS_CLASS_NAMES = frozenset({"HTTPStatusError", "HTTPError"})


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status an HTTP client's error carries, when it carries one."""
    response = getattr(exc, "response", None)
    for holder in (response, exc):
        for name in ("status_code", "status", "code"):
            value = getattr(holder, name, None)
            if isinstance(value, int):
                return value
    return None


def failure_side(
    exc: BaseException | None = None,
    *,
    class_names: tuple[str, ...] | list[str] = (),
    status: int | None = None,
) -> str:
    """Say whose side a failure is on: :data:`TOOL_SIDE` or :data:`INPUT_SIDE`.

    Args:
        exc: The exception, when the caller holds it.
        class_names: The names along the exception's method resolution order,
            when the exception itself was already turned into a result (a
            tool's own error envelope records them).
        status: The HTTP status the error carried, when known.

    Returns:
        :data:`TOOL_SIDE` for a connection, timeout, service (HTTP 5xx or 429)
        or configuration failure; :data:`INPUT_SIDE` for everything else.
    """
    names = list(class_names)
    if exc is not None:
        names.extend(cls.__name__ for cls in type(exc).__mro__)
        if status is None:
            status = _status_of(exc)
    # An HTTP error that carries its status is judged by the status first:
    # ``urllib``'s ``HTTPError`` is also a ``URLError``, and a 404 for a bad
    # address is the input's problem, not the service's.
    if any(name in _HTTP_STATUS_CLASS_NAMES for name in names) and status is not None:
        return TOOL_SIDE if status >= 500 or status == 429 else INPUT_SIDE
    if any(name in _TOOL_SIDE_CLASS_NAMES for name in names):
        return TOOL_SIDE
    return INPUT_SIDE


def error_class_names(exc: BaseException) -> list[str]:
    """The names along *exc*'s method resolution order, most specific first."""
    return [cls.__name__ for cls in type(exc).__mro__ if cls is not object]


@dataclass(frozen=True)
class ToolFailure:
    """One failed tool call, as the dispatch layer saw it.

    Attributes:
        tool: The tool's registered name.
        error_type: The exception's class name.
        message: The exception's message, as the observation carries it.
        side: :data:`TOOL_SIDE` or :data:`INPUT_SIDE`.
    """

    tool: str
    error_type: str
    message: str
    side: str

    @property
    def tool_side(self) -> bool:
        """Whether the tool's own side failed."""
        return self.side == TOOL_SIDE


_SINK: contextvars.ContextVar[list[ToolFailure] | None] = contextvars.ContextVar(
    "effgen_tool_failures", default=None,
)


def record_failure(failure: ToolFailure) -> None:
    """Hand *failure* to whoever is collecting for the dispatch in progress."""
    sink = _SINK.get()
    if sink is not None:
        sink.append(failure)


@contextlib.contextmanager
def collecting_failures() -> Iterator[list[ToolFailure]]:
    """Collect the failures reported while the block runs.

    Yields:
        The list the failures are appended to, in the order they happened.
    """
    sink: list[ToolFailure] = []
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)
        outer = _SINK.get()
        if outer is not None:
            outer.extend(sink)
