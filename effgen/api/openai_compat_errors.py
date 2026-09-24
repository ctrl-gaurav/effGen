"""Error classification, redaction and the OpenAI error envelope.

Maps runner/provider exceptions to HTTP statuses, redacts key/secret-looking
substrings from error text, and builds the ``{"error": {...}}`` envelope every
route and middleware layer emits. Every name is re-exported from
``effgen.api.openai_compat``; import from there.
"""
from __future__ import annotations

import re
from typing import Any

# A credential-prefixed token too short for the shared redactor's per-provider
# patterns, which require the 20-plus characters a real key carries. This is
# the last point text leaves the server, so a short one is blanked here rather
# than trusted to be a placeholder.
_SHORT_PREFIXED_SECRET_RE = re.compile(r"\b(?:sk|csk|gsk|xai)-[A-Za-z0-9_\-]{8,}")


def _redact(text: str) -> str:
    """Strip secret material from an error message before it leaves the server.

    Matching is on the shape of a key, never on the words around one. A
    keyword-and-anything-after rule reads "Check the provider API key (present,
    correct, ...)" as a secret and blanks the sentence that tells the caller
    what to do, while missing key shapes that carry no keyword at all.

    Redaction never masks the error itself: if the redactor cannot be loaded,
    the shorter sweep still runs and the rest of the text is returned as is.
    """
    if not text:
        return ""
    try:
        from effgen.observability.redact import get_redactor

        text = get_redactor().scrub(text)
    except Exception:  # noqa: BLE001 - redaction must not replace the error
        pass
    return _SHORT_PREFIXED_SECRET_RE.sub("[REDACTED]", text)


class UnknownToolError(ValueError):
    """A client requested a tool the server does not host.

    The OpenAI-compatible endpoint executes only effGen's *own* registered
    tools server-side; it does not forward arbitrary client-defined function
    specs to the model for the client to execute. Rather than silently dropping
    an unhosted tool (and returning prose), the endpoint raises this so the
    caller gets a clear ``400`` naming the offending tool(s).
    """

    def __init__(self, tool_names: list[str]) -> None:
        self.tool_names = list(tool_names)
        names = ", ".join(repr(n) for n in self.tool_names)
        super().__init__(
            f"tool(s) {names} are not available on this server. This endpoint "
            "executes only its own registered tools server-side (run "
            "`effgen tools list` to see them); pass one of those by name, or "
            "omit `tools` and let the model answer directly. This server does "
            "not forward client-defined function tools for the client to execute."
        )


def _classify_http(exc: Exception) -> tuple[int, str, str | None]:
    """Map an exception to (http_status, openai_error_type, code).

    Reuses effGen's provider-error taxonomy so the OpenAI-compatible surface
    fails with the right status (404 for a bad model, 502/503 for an upstream
    provider-auth/config failure, 429 for rate limits, …) instead of a blanket
    500. The server's own client-auth rejection (a forged/absent caller token)
    is handled upstream of this runner and stays 401.
    """
    if isinstance(exc, UnknownToolError):
        return 400, "invalid_request_error", "unknown_tool"
    from effgen.models.errors import BudgetExceededError

    if isinstance(exc, BudgetExceededError):
        # The process's own spend cap refused the call: the same answer the
        # server's per-principal cap gives, and the one OpenAI clients read as
        # a spent quota rather than a malformed request.
        return 429, "insufficient_quota", "budget_exceeded"
    category = "unknown"
    try:
        from effgen.models.errors import classify_provider_error

        category = classify_provider_error(exc).category
    except Exception:  # noqa: BLE001
        pass
    low = str(exc).lower()
    if category == "auth":
        # The server's own client-auth layer rejects bad caller credentials with
        # 401 *before* the runner is ever invoked. An auth failure that surfaces
        # HERE therefore comes from the upstream provider — the server is missing
        # or holding a rejected provider key — so it is the server's problem, not
        # the caller's. Returning 401 would wrongly blame the client's
        # credentials; report a gateway error instead. A missing key is a
        # configuration gap (503 Service Unavailable); a present-but-rejected key
        # is an upstream failure (502 Bad Gateway).
        # Adapters word an absent credential two ways — "API key not found"
        # (cerebras, fireworks, groq, together) and "API key not provided"
        # (anthropic, gemini, openai) — and both mean the same thing: nothing
        # was configured. Match both so one condition gets one status.
        if any(k in low for k in (
            "not found", "not set", "not provided", "no api key", "set the",
            "missing", "not configured",
        )):
            return 503, "upstream_unavailable", "upstream_key_missing"
        return 502, "upstream_error", "upstream_auth_failed"
    if category == "not_found":
        return 404, "model_not_found", "model_not_found"
    if category == "rate_limited":
        return 429, "rate_limit_exceeded", "rate_limit_exceeded"
    if category in ("invalid_request", "refusal", "fatal"):
        return 400, "invalid_request_error", None
    if category == "timeout":
        return 504, "timeout", None
    # A failed model load is, from the client's view, a bad model id.
    if any(k in low for k in ("load model", "not a valid model", "model_not_found", "could not find")):
        return 404, "model_not_found", "model_not_found"
    return 500, "server_error", None


def _error_payload(message: str, err_type: str, code: str | None) -> dict[str, Any]:
    """Build an OpenAI-style error envelope (redacted)."""
    return {
        "error": {
            "message": _redact(message),
            "type": err_type,
            "param": None,
            "code": code,
        }
    }


# Map an HTTP status to an OpenAI error ``type``/``code`` so that *every* error
# the server emits — including the ones raised by middleware before a route runs
# (auth 401, rate-limit 429, RBAC 403, validation 422) — uses the same
# ``{"error": {message, type, param, code}}`` envelope a model error uses. An
# OpenAI client can then branch on ``err.type`` / ``err.code`` uniformly.
_STATUS_ERROR_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "invalid_request_error",
    403: "permission_error",
    404: "model_not_found",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_exceeded",
    500: "server_error",
    502: "upstream_error",
    503: "upstream_unavailable",
    504: "timeout",
}
_STATUS_ERROR_CODE: dict[int, str] = {
    401: "invalid_api_key",
    403: "permission_denied",
    413: "request_too_large",
    429: "rate_limit_exceeded",
}


def error_envelope(
    status: int,
    message: str,
    *,
    code: str | None = None,
    error_type: str | None = None,
    redact: bool = True,
) -> dict[str, Any]:
    """Build the standard OpenAI error envelope for an HTTP *status*.

    Shared by the OpenAI-compat routes **and** the ASGI middleware (auth,
    rate-limit, RBAC/budget) so the whole server speaks one error shape. Set
    ``redact=False`` for server-authored, secret-free messages (e.g. the auth
    hint that legitimately spells out ``Authorization: Bearer <key>``) so the
    key/secret scrubber does not mangle the guidance; provider/upstream error
    text is always redacted (``redact=True``).

    ``error_type`` overrides the type derived from *status*. Pass it when the
    status alone would mislabel the failure — a 404 for an unknown URL path is
    an ``invalid_request_error``, not the ``model_not_found`` a 404 from the
    model routes means.

    Args:
        status: The HTTP status the envelope describes.
        message: The message shown to the client.
        code: Machine-readable code placed in the envelope.
        error_type: Overrides the type derived from *status*.
        redact: Run *message* through the secret scrubber first.

    Returns:
        The error body, ready to serialize.
    """
    err_type = error_type or _STATUS_ERROR_TYPE.get(
        status, "server_error" if status >= 500 else "invalid_request_error"
    )
    err_code = code if code is not None else _STATUS_ERROR_CODE.get(status)
    if not redact:
        return {
            "error": {"message": message, "type": err_type, "param": None, "code": err_code}
        }
    return _error_payload(message, err_type, err_code)
