"""
effGen model error types.

Defines exceptions raised by model adapters beyond the standard
RuntimeError/ValueError hierarchy, plus :func:`classify_provider_error`, a
single classification layer the agent/retry logic uses to decide whether an
error is worth retrying and how to report it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from effgen.errors import quote_for_message, redact_for_message, with_next_step

# ---------------------------------------------------------------------------
# Structured error context (shared by typed errors below and by adapters via
# effgen.models._adapter_utils). One source of truth for the per-category
# retry status + remediation text so every provider failure reports the same
# machine-readable shape: {provider, model, request_type, retry_status,
# remediation, category}.
# ---------------------------------------------------------------------------

# retry_status values (stable, machine-readable).
RETRY_WILL_RETRY = "will_retry"
RETRY_RATE_LIMITED = "rate_limited"
RETRY_NON_RETRYABLE = "non_retryable"

# category -> retry_status
_RETRY_STATUS_BY_CATEGORY: dict[str, str] = {
    "auth": RETRY_NON_RETRYABLE,
    "not_found": RETRY_NON_RETRYABLE,
    "refusal": RETRY_NON_RETRYABLE,
    "invalid_request": RETRY_NON_RETRYABLE,
    "not_loaded": RETRY_NON_RETRYABLE,
    "resource_exhausted": RETRY_NON_RETRYABLE,
    "fatal": RETRY_NON_RETRYABLE,
    "rate_limited": RETRY_RATE_LIMITED,
    "transient": RETRY_WILL_RETRY,
    "unreachable": RETRY_WILL_RETRY,
    "timeout": RETRY_WILL_RETRY,
    "unknown": RETRY_WILL_RETRY,
}

# category -> remediation hint (no secrets; safe to log/surface).
REMEDIATION_BY_CATEGORY: dict[str, str] = {
    "auth": "Check the provider API key (present, correct, not expired, and has access to this model).",
    "not_found": "Model id not found — run `effgen models list` to see ids, `effgen models refresh` to update the catalog, and verify the id/provider prefix.",
    "rate_limited": "Rate limited — lower concurrency or configure a rate limit; the client backs off and retries.",
    "timeout": "Request timed out — increase the adapter timeout or retry.",
    "transient": "Transient provider error — retry shortly; check the provider status page if it persists.",
    "unreachable": "Nothing answered at that endpoint — check the server is running and the base_url, host and port are right.",
    "invalid_request": "Request rejected as invalid — check parameters, prompt size, and any JSON schema.",
    "not_loaded": "The model is not loaded — call load() on the adapter first, or build it with load_model(), which loads it for you.",
    "resource_exhausted": "The device ran out of memory — the same request will fail again until less is asked of it or more memory is free.",
    "refusal": "The model refused the request — rephrase or adjust the content.",
    "fatal": "Unrecoverable error — check configuration and provider status.",
    "unknown": "Unexpected provider error — check the provider status and effGen logs (set EFFGEN_LOG_LEVEL=DEBUG).",
}

# Phrases providers use to report that a request — the prompt, or often just
# a large preset's tool-schema payload — exceeded a model's context window or
# a per-request/per-minute token limit. Providers word this differently:
# OpenAI/Anthropic-style context errors, Groq/OpenAI-compatible per-minute
# rate errors, and generic "too large" request rejections.
_CONTEXT_OVERFLOW_SIGNALS = (
    "context_length_exceeded",
    "maximum context length",
    "reduce the length",
    "reduce your message size",
    "reduce the request",
    "reduce the size",
    "tokens per minute",
    "request too large",
    "too large for model",
    "context window",
    "exceeds model context length",
)

# The subset of the above that says the prompt is larger than the window,
# rather than that a rate limit was reached. A window is a property of the
# model and does not change while a request is in flight, so sending the same
# prompt again spends the budget on the same refusal — which is what the local
# engines' own check used to cause, its wording matching none of the phrases
# above and being classified as an unknown, retryable failure.
_CONTEXT_WINDOW_SIGNALS = (
    "context_length_exceeded",
    "maximum context length",
    "exceeds model context length",
    "too large for model",
)


def generation_failure_text(detail: dict[str, Any]) -> str:
    """Render a failed generation's error record as the text a caller reads.

    The record keeps the cause and the guidance apart so a typed error can be
    rebuilt from it without stacking two copies of the guidance. This is where
    they are put back together for display.

    Args:
        detail: An error record carrying at least ``message``, and usually the
            ``category`` the guidance is looked up from.

    Returns:
        What failed, then what to do about it.
    """
    message = detail.get("message") or "generation failed"
    remediation = detail.get("remediation") or REMEDIATION_BY_CATEGORY.get(
        detail.get("category", ""), REMEDIATION_BY_CATEGORY["unknown"]
    )
    return with_next_step(f"Generation failed: {message}", remediation)


def context_overflow_hint(message: str) -> str | None:
    """Return an actionable hint when *message* signals a context/token-rate overflow.

    Fires when the request (the prompt, or a large preset's tool-schema
    payload) exceeded a model's context window or a provider's token-rate
    limit.

    Returns ``None`` when *message* is not such a signal, so callers can
    leave their base remediation text untouched.
    """
    lowered = (message or "").lower()
    if not any(s in lowered for s in _CONTEXT_OVERFLOW_SIGNALS):
        return None
    return (
        " The request may be too large for this model — try a smaller preset "
        "(e.g. `general` or `math` instead of `research`) or a model with a "
        "larger context window / higher rate limit."
    )


def scrub_provider_message(message: str) -> str:
    """Return *message* with any secret material replaced by a redaction marker.

    Provider SDKs echo parts of the request back in an error body, and some
    include the submitted credential. Every typed error below routes its
    ``message`` through this helper so an adapter cannot put a key into an
    exception string that later reaches a log, a CLI panel, or an API
    envelope. Redaction never masks the error itself: if the redactor is
    unavailable, the original text is returned.
    """
    return redact_for_message(message) if message else message


#: How much of one failed candidate's error text the failover summary quotes.
#: A failover run reports several failures at once, so each gets a smaller
#: share than a single error message would.
_PER_FAILURE_ECHO_LIMIT = 160

#: Total room the failover summary gets, whatever the hop count. Bounds the
#: message when a long chain each contributes its share.
_FAILURE_SUMMARY_LIMIT = 800


# Attributes provider SDKs keep a copy of the error text in when the value is
# not in the instance dictionary (a property, or a slot). Instance attributes
# are found by inspection, so this list only has to cover what inspection
# cannot reach.
_SCRUBBED_ATTRIBUTES = ("message", "body", "detail", "text", "reason", "error")

# How far down a ``__cause__``/``__context__`` chain to scrub. Deep enough for
# any real SDK wrapping depth, bounded so a pathological chain cannot stall.
_MAX_CHAIN_DEPTH = 20

# How deep to descend into a structured attribute (a parsed JSON error body).
_MAX_VALUE_DEPTH = 6


def _scrub_value(value: Any, depth: int = 0) -> Any:
    """Return *value* with every string inside it redacted.

    Handles the shapes an SDK stores a parsed error body in — a string, a dict,
    a list or a tuple — and leaves anything else untouched.
    """
    if depth > _MAX_VALUE_DEPTH:
        return value
    if isinstance(value, str):
        return scrub_provider_message(value)
    if isinstance(value, dict):
        return {k: _scrub_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v, depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(v, depth + 1) for v in value)
    return value


def scrub_exception(exc: BaseException) -> BaseException:
    """Redact secret material from *exc* and every exception it chains to.

    A typed effGen error routes its own ``message`` through
    :func:`scrub_provider_message`, but ``raise ... from exc`` keeps the
    provider SDK's exception as ``__cause__`` — and providers echo the
    submitted credential in a 401 body. Printing the traceback, logging with
    ``exc_info``, or letting the process die then shows the key even though the
    message the caller reads was redacted.

    Each exception in the chain has its ``args`` and every text-bearing
    attribute it carries rewritten in place — the SDKs keep the parsed error
    body under a different name in each library (``body``, ``details``,
    ``server_message``), so the attributes are found by inspection rather than
    from a list that would go stale on the next SDK release. Values that hold
    no text are left alone, and an exception that refuses the assignment is
    skipped rather than replacing the failure with a new one.

    Returns *exc* so it can be used inline in a ``raise`` statement.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    depth = 0
    while node is not None and depth < _MAX_CHAIN_DEPTH and id(node) not in seen:
        seen.add(id(node))
        depth += 1
        try:
            args = node.args
            if any(isinstance(a, str) for a in args):
                node.args = tuple(
                    scrub_provider_message(a) if isinstance(a, str) else a
                    for a in args
                )
        except Exception:  # noqa: BLE001 - redaction must not replace the error
            pass
        try:
            names = set(vars(node)) | set(_SCRUBBED_ATTRIBUTES)
        except TypeError:  # no instance dictionary
            names = set(_SCRUBBED_ATTRIBUTES)
        for name in names:
            if name.startswith("__"):
                continue
            value = getattr(node, name, None)
            if isinstance(value, str | dict | list | tuple) and value:
                try:
                    setattr(node, name, _scrub_value(value))
                except Exception:  # noqa: BLE001 - read-only attribute; skip it
                    pass
        node = node.__cause__ or node.__context__
    return exc


# How a provider states the delay it wants before the next attempt. The header
# is authoritative when present; the text forms are what the body carries when
# the SDK has already discarded the response.
_RETRY_DELAY_PATTERNS = (
    # Gemini: "details": [{"@type": ".../RetryInfo", "retryDelay": "16s"}]
    re.compile(r"retry[_\s]?delay\W{0,4}(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE),
    # Groq / OpenAI-compatible: "Please try again in 17.3s" / "in 1m12s"
    re.compile(
        r"(?:try|retry)\s+again\s+in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:try|retry)\s+again\s+in\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b",
        re.IGNORECASE,
    ),
)

#: Longest delay a provider hint is believed. A larger stated value is clamped,
#: so a malformed or hostile body cannot park a caller for hours.
MAX_RETRY_AFTER_SECONDS = 300.0


def _retry_after_from_text(text: str) -> float | None:
    """Parse the delay a provider stated in prose, in seconds."""
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        try:
            if len(groups) == 2:
                minutes = float(groups[0]) if groups[0] else 0.0
                return minutes * 60.0 + float(groups[1])
            return float(groups[0])
        except (TypeError, ValueError):
            continue
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long the provider asked the caller to wait, in seconds.

    Reads, in order of authority, the ``retry_after`` attribute an adapter may
    have set, the ``Retry-After`` response header, and the delay stated in the
    error text — Gemini reports it as a ``retryDelay`` field and the
    OpenAI-compatible providers as "please try again in 17.3s". The whole
    ``__cause__``/``__context__`` chain is walked, because by the time a caller
    sees the error the response that carried the header is usually two wrappers
    down.

    Returns ``None`` when the provider said nothing, so a caller can fall back
    to its own backoff rather than inventing a delay. A stated delay above
    :data:`MAX_RETRY_AFTER_SECONDS` is clamped to it.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    depth = 0
    while node is not None and depth < _MAX_CHAIN_DEPTH and id(node) not in seen:
        seen.add(id(node))
        depth += 1
        value = getattr(node, "retry_after", None)
        if isinstance(value, int | float) and value > 0:
            return min(float(value), MAX_RETRY_AFTER_SECONDS)
        response = getattr(node, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                raw = headers.get("retry-after") or headers.get("Retry-After")
            except Exception:  # noqa: BLE001 - a header map that will not read
                raw = None
            if raw:
                try:
                    return min(float(str(raw).strip()), MAX_RETRY_AFTER_SECONDS)
                except ValueError:
                    pass
        stated = _retry_after_from_text(str(node))
        if stated is not None and stated > 0:
            return min(stated, MAX_RETRY_AFTER_SECONDS)
        node = node.__cause__ or node.__context__
    return None


def error_context_dict(
    provider: str,
    model: str,
    request_type: str,
    category: str,
) -> dict[str, str]:
    """Build the canonical structured error-context dict for *category*.

    The returned dict is always free of secret material (the fields are
    static/derived), so it is safe to attach to exceptions, log, and surface.

    Args:
        provider: The provider the call went to.
        model: The model id the call used.
        request_type: What was attempted, such as ``generate`` or ``stream``.
        category: The classified failure kind.

    Returns:
        The context dict attached to the raised error.
    """
    return {
        "provider": provider or "",
        "model": model or "",
        "request_type": request_type or "request",
        "retry_status": _RETRY_STATUS_BY_CATEGORY.get(category, RETRY_WILL_RETRY),
        "remediation": REMEDIATION_BY_CATEGORY.get(category, REMEDIATION_BY_CATEGORY["unknown"]),
        "category": category,
    }


class ModelRefusalError(Exception):
    """Raised when a model refuses to answer a structured-output request.

    OpenAI structured outputs may return a ``refusal`` field instead of
    valid JSON content.  The adapter raises this exception so callers can
    distinguish a genuine refusal (policy / safety) from a malformed
    response or a network error.

    Attributes:
        refusal_message: The raw refusal string returned by the model.
        model_name: The model that issued the refusal.
    """

    def __init__(self, refusal_message: str, model_name: str = "") -> None:
        self.refusal_message = scrub_provider_message(refusal_message)
        self.model_name = model_name
        self.error_context = error_context_dict("", model_name, "request", "refusal")
        suffix = f" (model={model_name!r})" if model_name else ""
        super().__init__(
            with_next_step(
                f"Model refused to generate structured output{suffix}: "
                f"{quote_for_message(self.refusal_message)}",
                self.error_context["remediation"],
            )
        )


class ModelAuthError(Exception):
    """Raised when a model adapter fails to authenticate with its provider.

    Distinguishes auth failures (bad/missing API key, revoked credentials)
    from other transport or rate-limit errors so callers get a clear
    "fix your key" signal instead of a generic 500.

    Attributes:
        provider: Provider name (e.g. ``"groq"``).
        model_name: The model that was requested.
        message: Human-readable cause.
    """

    def __init__(self, provider: str, model_name: str = "", message: str = "") -> None:
        self.provider = provider
        self.model_name = model_name
        self.message = scrub_provider_message(message)
        self.error_context = error_context_dict(provider, model_name, "request", "auth")
        suffix = f" (model={model_name!r})" if model_name else ""
        body = quote_for_message(self.message) if self.message else "authentication failed"
        super().__init__(
            with_next_step(
                f"{provider} auth error{suffix}: {body}",
                self.error_context["remediation"],
            )
        )


class ModelTimeoutError(Exception):
    """Raised when a model prediction does not complete within the timeout.

    Replicate and other async-polling providers may take variable time to
    complete.  This error is raised when the configurable timeout is exceeded
    so callers can distinguish a slow/stuck prediction from a network error.

    Attributes:
        provider: Provider name (e.g. ``"replicate"``).
        model_name: The model that was requested.
        timeout_seconds: The timeout that was exceeded.
        prediction_id: The prediction ID (if available) so it can be cancelled.
    """

    def __init__(
        self,
        provider: str,
        model_name: str = "",
        timeout_seconds: float = 300,
        prediction_id: str = "",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.prediction_id = prediction_id
        self.error_context = error_context_dict(provider, model_name, "request", "timeout")
        suffix = f" (model={model_name!r})" if model_name else ""
        pid = f", prediction_id={prediction_id!r}" if prediction_id else ""
        super().__init__(
            f"{provider} prediction timed out after {timeout_seconds}s{suffix}{pid}. "
            f"Increase timeout= on the adapter or check provider status."
        )


class ModelUnavailableError(Exception):
    """Raised when a model is temporarily unavailable on the provider's serverless tier.

    HuggingFace Serverless Inference rotates model availability.  This error
    surfaces the 503/404 with actionable suggestions so callers know what to try
    instead of getting a raw HTTP error.

    Attributes:
        provider: Provider name (e.g. ``"hf"``).
        model_name: The model that is unavailable.
        suggestions: List of alternative model IDs to try.
        message: Human-readable cause.
    """

    def __init__(
        self,
        provider: str,
        model_name: str = "",
        suggestions: list[str] | None = None,
        message: str = "",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.suggestions = suggestions or []
        self.message = scrub_provider_message(message)
        self.error_context = error_context_dict(provider, model_name, "request", "not_found")
        suffix = f" (model={model_name!r})" if model_name else ""
        body = (
            quote_for_message(self.message)
            if self.message
            else "model is not available on the serverless tier"
        )
        follow_on = (
            "Try one of: " + ", ".join(self.suggestions) + "."
            if self.suggestions
            else self.error_context["remediation"]
        )
        super().__init__(
            with_next_step(f"{provider} unavailable{suffix}: {body}", follow_on)
        )


class ModelNotFoundError(Exception):
    """Raised when a requested model ID does not exist on the provider.

    Attributes:
        provider: Provider name.
        model_name: The model that was not found.
    """

    def __init__(self, provider: str, model_name: str = "", message: str = "") -> None:
        self.provider = provider
        self.model_name = model_name
        self.message = scrub_provider_message(message)
        self.error_context = error_context_dict(provider, model_name, "request", "not_found")
        suffix = f" (model={model_name!r})" if model_name else ""
        body = quote_for_message(self.message) if self.message else "model not found"
        super().__init__(
            with_next_step(
                f"{provider} error{suffix}: {body}",
                self.error_context["remediation"],
            )
        )


class AmbiguousModelError(Exception):
    """Raised when a model ID exists in multiple providers and no provider prefix is given.

    Example: ``"llama-3.3-70b"`` could be Groq, Together, or Fireworks.
    Callers should disambiguate with ``"groq:openai/gpt-oss-120b"`` syntax.

    Attributes:
        model_id: The ambiguous model identifier.
        providers: List of providers that each claim this model ID.
    """

    def __init__(self, model_id: str, providers: list[str]) -> None:
        self.model_id = model_id
        self.providers = list(providers)
        plist = ", ".join(f'"{p}"' for p in providers)
        super().__init__(
            f"Model {model_id!r} is available on multiple providers: [{plist}]. "
            f"Disambiguate with 'provider:model_id' syntax, e.g. "
            f'"{providers[0]}:{model_id}".'
        )


class NoCandidateWithinBudgetError(Exception):
    """Raised when no provider/model fits within the user's budget.

    Attributes:
        user_budget_usd:    The budget that was not satisfiable.
        cheapest_cost_usd:  The lowest estimated cost among available candidates.
        cheapest_pair:      ``(provider, model_id)`` for the cheapest option.
    """

    def __init__(
        self,
        user_budget_usd: float,
        cheapest_cost_usd: float,
        cheapest_pair: "tuple[str, str] | None" = None,
    ) -> None:
        self.user_budget_usd = user_budget_usd
        self.cheapest_cost_usd = cheapest_cost_usd
        self.cheapest_pair = cheapest_pair
        pair_str = f" ({cheapest_pair[0]}/{cheapest_pair[1]})" if cheapest_pair else ""
        super().__init__(
            f"No candidate fits budget ${user_budget_usd:.6f} USD. "
            f"Cheapest available: ${cheapest_cost_usd:.6f}{pair_str}. "
            f"Increase user_budget_usd or use a free-tier provider."
        )


class NoCandidateWithinLatencyError(Exception):
    """Raised when no provider/model fits within the latency budget.

    Attributes:
        latency_budget_ms:    The SLA budget that was not satisfiable.
        fastest_latency_ms:   The lowest p50 latency among available candidates.
        fastest_pair:         ``(provider, model_id)`` for the fastest option.
    """

    def __init__(
        self,
        latency_budget_ms: float,
        fastest_latency_ms: float,
        fastest_pair: "tuple[str, str] | None" = None,
    ) -> None:
        self.latency_budget_ms = latency_budget_ms
        self.fastest_latency_ms = fastest_latency_ms
        self.fastest_pair = fastest_pair
        pair_str = f" ({fastest_pair[0]}/{fastest_pair[1]})" if fastest_pair else ""
        super().__init__(
            f"No candidate fits latency budget {latency_budget_ms:.0f}ms. "
            f"Fastest available p50: {fastest_latency_ms:.0f}ms{pair_str}. "
            "Increase latency_budget_ms or seed the tracker with faster providers."
        )


class ProviderTransientError(Exception):
    """Raised when a provider returns a transient server-side error (5xx).

    Attributes:
        provider:   Provider name (e.g. ``"groq"``).
        model_name: The model that was requested.
        status_code: HTTP status code (e.g. 500, 503).
        message:    Human-readable cause.
    """

    def __init__(
        self,
        provider: str,
        model_name: str = "",
        status_code: int = 500,
        message: str = "",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.status_code = status_code
        self.message = scrub_provider_message(message)
        self.error_context = error_context_dict(provider, model_name, "request", "transient")
        suffix = f" (model={model_name!r})" if model_name else ""
        body = quote_for_message(self.message) if self.message else "transient server error"
        super().__init__(
            with_next_step(
                f"{provider} {status_code}{suffix}: {body}",
                self.error_context["remediation"],
            )
        )


class BackendUnreachableError(Exception):
    """Raised when nothing answered at the endpoint the run was sent to.

    A task that ran and failed is a result the caller can read; a backend that
    was never reached produced no result at all, so a run that ends this way
    raises whatever ``AgentConfig.raise_on_error`` says. A dead server would
    otherwise be indistinguishable from a model that could not solve the task,
    which lets a whole batch complete "successfully" against nothing.

    Attributes:
        provider:   Provider or adapter name (e.g. ``"openai_compatible"``).
        model_name: The model that was requested.
        endpoint:   The URL that did not answer, when one is known.
        message:    Human-readable cause.
    """

    def __init__(
        self,
        provider: str,
        model_name: str = "",
        message: str = "",
        endpoint: str = "",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.endpoint = endpoint
        self.message = scrub_provider_message(message)
        self.error_context = error_context_dict(
            provider, model_name, "request", "unreachable"
        )
        where = f" at {endpoint}" if endpoint else ""
        suffix = f" (model={model_name!r})" if model_name else ""
        body = quote_for_message(self.message) if self.message else "no response"
        super().__init__(
            with_next_step(
                f"{provider}{where} did not answer{suffix}: {body}",
                self.error_context["remediation"],
            )
        )


class AllCandidatesExhaustedError(Exception):
    """Raised when all failover hops have been exhausted without a successful response.

    Every candidate in the router's ordered list has either failed with a
    retriable error or been skipped.  The caller should surface this to the
    user with the list of failures so they can diagnose the root cause.

    Attributes:
        failures: List of ``(provider, model_id, exception)`` triples for every
                  attempted hop.
        hop_limit: The maximum number of hops that was configured.
    """

    def __init__(
        self,
        failures: "list[tuple[str, str, Exception]]",
        hop_limit: int = 3,
    ) -> None:
        self.failures = failures
        self.hop_limit = hop_limit
        self.attempts = len(failures)
        summary = "; ".join(
            f"{prov}/{model}: {type(exc).__name__}"
            f"({quote_for_message(exc, _PER_FAILURE_ECHO_LIMIT)})"
            for prov, model, exc in failures
        )
        super().__init__(
            with_next_step(
                f"All {len(failures)} candidate attempts exhausted after "
                f"{hop_limit} failover hops. "
                f"Failures: [{quote_for_message(summary, _FAILURE_SUMMARY_LIMIT)}]",
                "Check the first failure above — the later hops usually repeat "
                "it; widen the fallback chain or raise the hop limit if every "
                "candidate is genuinely unavailable.",
            )
        )


class InvalidRequestError(Exception):
    """Raised when the request itself is malformed and cannot be retried.

    Examples: invalid JSON schema, unsupported parameter, prompt too long.

    Attributes:
        provider:   Provider name.
        model_name: The model that rejected the request.
        message:    Human-readable cause.
    """

    def __init__(self, provider: str, model_name: str = "", message: str = "") -> None:
        self.provider = provider
        self.model_name = model_name
        self.message = scrub_provider_message(message)
        self.error_context = error_context_dict(provider, model_name, "request", "invalid_request")
        suffix = f" (model={model_name!r})" if model_name else ""
        body = quote_for_message(self.message) if self.message else "invalid request"
        super().__init__(
            with_next_step(
                f"{provider} invalid request{suffix}: {body}",
                self.error_context["remediation"],
            )
        )


class ContextBudgetExceededError(InvalidRequestError):
    """Raised when a run's conversation will not fit the tokens it may send.

    The run has already given up everything its compaction policy is allowed to
    give up — old tool results, old reasoning, whole answered cycles — and the
    smallest prompt it can still build is over the budget. What is left is the
    question, the instructions the run is framed by and its most recent work,
    none of which can go without changing what was asked.

    Raised **before the request is sent**, so a run whose frame alone is too
    large costs nothing at the provider. It is an
    :class:`InvalidRequestError`, so a caller already catching a prompt that
    was refused for being too long keeps catching this.

    Attributes:
        budget_tokens: What the run was allowed to send.
        measured_tokens: What the smallest prompt it could build measured.
        window_tokens: The context window the budget came from, when it was
            derived from one.
        budget_source: ``"auto"``, ``"config"`` or ``"fraction"`` — how the
            budget was arrived at.
        response: The run so far, when the caller asked for it to be kept.
        partial: The progress the run had made, when there was any.
    """

    def __init__(
        self,
        *,
        budget_tokens: int,
        measured_tokens: int,
        window_tokens: int | None = None,
        budget_source: str = "auto",
        provider: str = "effgen",
        model_name: str = "",
        protected: str = "",
        response: Any = None,
        partial: Any = None,
    ) -> None:
        self.budget_tokens = int(budget_tokens)
        self.measured_tokens = int(measured_tokens)
        self.window_tokens = window_tokens
        self.budget_source = budget_source
        self.response = response
        self.partial = partial
        held = protected or (
            "the question, the instructions the run is framed by and its most "
            "recent work"
        )
        window = (
            f" of a {window_tokens}-token context window" if window_tokens else ""
        )
        super().__init__(
            provider,
            model_name,
            (
                f"the conversation does not fit: this run may send "
                f"{self.budget_tokens} prompt tokens{window} and the smallest "
                f"prompt it can build measures {self.measured_tokens} — {held} "
                f"cannot be given up. Raise max_context_length or "
                f"context_budget, shorten the question, or run it on a model "
                f"with a larger context window."
            ),
        )


class BudgetExceededError(Exception):
    """Raised when cumulative spend crosses the configured daily/monthly budget.

    The router treats this as a retriable error and attempts failover to a
    free-tier provider when one is available.

    Attributes:
        budget_usd:   The budget limit that was crossed.
        actual_usd:   Current cumulative spend.
        period:       ``"daily"`` or ``"monthly"``.
        provider:     Provider that triggered the alert (if known).
        model:        Model that triggered the alert (if known).
    """

    def __init__(
        self,
        budget_usd: float,
        actual_usd: float,
        period: str = "daily",
        provider: str = "",
        model: str = "",
    ) -> None:
        self.budget_usd = budget_usd
        self.actual_usd = actual_usd
        self.period = period
        self.provider = provider
        self.model = model
        ctx = f" (provider={provider!r}, model={model!r})" if provider else ""
        from effgen.models._cost import format_usd
        super().__init__(
            f"{period.capitalize()} budget {format_usd(budget_usd)} exceeded: "
            f"actual={format_usd(actual_usd)}{ctx}. Raised to the caller; a "
            "router configured with a fallback_chain can attempt failover to "
            "a free-tier provider instead of raising."
        )


class ToolIncompatibleError(Exception):
    """Raised when a tool cannot be used with the configured model.

    Detected at Agent init time (not mid-run) so users get a clear error
    before any API calls are made.

    Attributes:
        tool_name: The tool that triggered the error.
        model_name: The model that does not support the tool.
        reason: Human-readable explanation.
    """

    def __init__(self, tool_name: str, model_name: str, reason: str = "") -> None:
        self.tool_name = tool_name
        self.model_name = model_name
        self.reason = reason
        parts = [f"Tool '{tool_name}' is incompatible with model '{model_name}'."]
        if reason:
            parts.append(quote_for_message(reason))
        super().__init__(
            with_next_step(
                " ".join(parts),
                "Remove the tool from the agent, or choose a model that "
                "supports tool calling.",
            )
        )


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorClass:
    """Classification of a provider/model error for retry + reporting.

    ``category`` is the single human-readable label (``"auth"``,
    ``"not_found"``, ``"rate_limited"``, ``"transient"``, ``"timeout"``,
    ``"refusal"``, ``"invalid_request"``, ``"not_loaded"``,
    ``"resource_exhausted"``, ``"fatal"``, or ``"unknown"``).
    The boolean flags let callers branch without string-matching ``category``.

    Only ``retryable`` errors (transient/timeout/unknown) and ``rate_limited``
    errors should be retried. ``auth``, ``not_found``, ``refusal``,
    ``invalid_request``, ``not_loaded``, ``resource_exhausted`` and ``fatal``
    errors will not succeed on retry and must fail fast.
    """

    category: str
    retryable: bool = False
    rate_limited: bool = False
    auth: bool = False
    not_found: bool = False
    refusal: bool = False
    fatal: bool = False

    @property
    def should_retry(self) -> bool:
        """True when retrying the call could plausibly succeed."""
        return self.retryable or self.rate_limited


# Reusable instances (frozen → safe to share).
_AUTH = ErrorClass("auth", auth=True, fatal=True)
_NOT_FOUND = ErrorClass("not_found", not_found=True)
_RATE_LIMITED = ErrorClass("rate_limited", rate_limited=True, retryable=True)
_TRANSIENT = ErrorClass("transient", retryable=True)
# Nothing answered at all: the connection was refused, the name did not
# resolve, or there was no route. Distinct from "transient", where the server
# did answer and answered badly — a run that never reached a backend produced
# no result, and the agent surface raises rather than reporting one.
_UNREACHABLE = ErrorClass("unreachable", retryable=True)
_TIMEOUT = ErrorClass("timeout", retryable=True)
_REFUSAL = ErrorClass("refusal", refusal=True)
_INVALID = ErrorClass("invalid_request", fatal=True)
_NOT_LOADED = ErrorClass("not_loaded", fatal=True)
_RESOURCE_EXHAUSTED = ErrorClass("resource_exhausted", fatal=True)
_FATAL = ErrorClass("fatal", fatal=True)
_UNKNOWN = ErrorClass("unknown", retryable=True)

# category -> the matching reusable ErrorClass, so an already-classified
# ``.error_context`` (attached by provider_runtime_error()/attach_error_context())
# can be turned back into an ErrorClass without re-deriving it.
_ERROR_CLASS_BY_CATEGORY: dict[str, ErrorClass] = {
    "auth": _AUTH,
    "not_found": _NOT_FOUND,
    "rate_limited": _RATE_LIMITED,
    "transient": _TRANSIENT,
    "unreachable": _UNREACHABLE,
    "timeout": _TIMEOUT,
    "refusal": _REFUSAL,
    "invalid_request": _INVALID,
    "not_loaded": _NOT_LOADED,
    "resource_exhausted": _RESOURCE_EXHAUSTED,
    "fatal": _FATAL,
    "unknown": _UNKNOWN,
}


def chained_message(exc: BaseException, limit: int = 5) -> str:
    """Return *exc*'s message joined with those of the errors that caused it.

    Provider SDKs wrap a socket failure in their own class and shorten the
    message to something generic — the OpenAI client reports "Connection
    error." for a refused port, a DNS failure and a dead route alike. The
    reason survives on ``__cause__``, so classification reads the chain rather
    than only the outermost text.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(parts) < limit and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " | ".join(parts).lower()


# Wording for a backend that never answered: the port was closed, the host
# name did not resolve, or there was no route to it. A socket that opened and
# then dropped ("connection reset") is not here — that server was reachable and
# the call is worth retrying against it as a transient fault.
UNREACHABLE_SIGNALS = (
    "connection refused",
    "failed to establish a new connection",
    "max retries exceeded with url",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "no route to host",
    "network is unreachable",
    "cannot connect to host",
    "all connection attempts failed",
)


# Wording the local runtimes (torch, vLLM, llama.cpp, MLX) use when the device
# runs out of memory. A cloud provider's failure carries an HTTP status and is
# classified before these are consulted.
DEVICE_MEMORY_SIGNALS = (
    "out of memory",
    "outofmemoryerror",
    "not enough memory",
    "insufficient device memory",
    "failed to allocate",
    # A quantized load that does not fit reports the shortfall as an offload
    # instruction rather than an allocation failure.
    "enough gpu ram",
)


# Phrases that state outright that the submitted credential was rejected.
# Kept narrow on purpose: each one names the key itself, so it cannot match a
# request-validation error that merely mentions an argument.
#: Phrases that mean "you have spent a per-minute budget", carried on a status
#: code that would otherwise read as a permanent property of the request. Groq
#: answers a spent token-per-minute allowance with 413 rather than 429, and the
#: payload names it outright. Kept to statements about a *rate*, so a genuinely
#: oversized body stays an invalid request.
_EXPLICIT_RATE_LIMIT_SIGNALS = (
    "rate_limit_exceeded",
    "rate limit exceeded",
    "tokens per minute",
    "requests per minute",
    "tokens per day",
    "requests per day",
)

_EXPLICIT_INVALID_KEY_SIGNALS = (
    "api key not valid",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "api_key_invalid",
    "wrong api key",
    "wrong_api_key",
)


def _status_code_of(exc: Exception) -> int | None:
    """Best-effort HTTP status extraction across SDK exception shapes."""
    for attr in ("status_code", "status", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def classify_provider_error(exc: Exception) -> ErrorClass:
    """Classify a provider/model exception into an :class:`ErrorClass`.

    Recognises effGen's own typed errors first, then falls back to inspecting
    SDK exception class names, HTTP status codes, and message text so raw
    provider-SDK errors (openai/groq/gemini/…) are still classified correctly.

    Args:
        exc: The exception raised by an adapter or provider SDK.

    Returns:
        An :class:`ErrorClass`. Unknown errors default to ``retryable`` so a
        genuine transient blip is not turned into a hard failure, while the
        recognised auth/not-found/refusal/invalid classes fail fast.
    """
    # 1. effGen typed errors (adapters map most provider errors to these).
    if isinstance(exc, ModelAuthError):
        return _AUTH
    if isinstance(exc, ModelNotFoundError | ModelUnavailableError):
        return _NOT_FOUND
    if isinstance(exc, ModelRefusalError):
        return _REFUSAL
    if isinstance(exc, ModelTimeoutError):
        return _TIMEOUT
    if isinstance(exc, ProviderTransientError):
        return _TRANSIENT
    if isinstance(exc, InvalidRequestError):
        return _INVALID
    if isinstance(
        exc,
        BudgetExceededError
        | NoCandidateWithinBudgetError
        | NoCandidateWithinLatencyError
        | AllCandidatesExhaustedError
        | AmbiguousModelError,
    ):
        return _FATAL

    # 1.5. An already-classified ``.error_context`` (attached by
    # provider_runtime_error()/attach_error_context() when the original SDK
    # exception was wrapped) is authoritative — it was computed from the
    # *original* exception, before wrapping lost the status code/class name
    # the heuristics below rely on. Re-deriving from the wrapped text alone
    # can silently downgrade a non-retryable error (e.g. a 400) to "unknown"
    # (retryable) if the wrapped message doesn't happen to repeat a keyword.
    ctx = getattr(exc, "error_context", None)
    if isinstance(ctx, dict):
        cached = _ERROR_CLASS_BY_CATEGORY.get(ctx.get("category", ""))
        if cached is not None:
            return cached

    # 1.8. An explicit "this credential is not valid" statement outranks the
    # status code. Gemini reports a rejected API key as HTTP 400
    # INVALID_ARGUMENT, so classifying by status alone labels the same failure
    # ``invalid_request`` on one code path and ``auth`` on another. Only these
    # unambiguous key phrases qualify — a generic INVALID_ARGUMENT stays an
    # invalid request.
    if any(k in str(exc).lower() for k in _EXPLICIT_INVALID_KEY_SIGNALS):
        return _AUTH

    # 1.9. A prompt larger than the model's window. The window does not change
    # while the request is in flight, so the same prompt sent again is refused
    # again — it is the request that is wrong, not the moment. Classified from
    # the message because the check that states it most plainly is effGen's own
    # and raises a bare ValueError carrying no status code.
    #
    # A payload that states a per-minute or per-day limit is excluded: a
    # provider that answers a spent token budget with "request too large for
    # model X on tokens per minute (TPM)" is reporting a throttle, and the same
    # prompt does fit a minute later. Reading it as an oversized request loses
    # the backoff and reports a transient limit as permanent.
    _text = str(exc).lower()
    if any(k in _text for k in _CONTEXT_WINDOW_SIGNALS) and not any(
        k in _text for k in _EXPLICIT_RATE_LIMIT_SIGNALS
    ):
        return _INVALID

    # 2. HTTP status code (raw SDK errors carry one).
    status = _status_code_of(exc)
    if status is not None:
        if status in (401, 403):
            return _AUTH
        if status == 404:
            return _NOT_FOUND
        if status == 429:
            return _RATE_LIMITED
        # 402 payment-required is an account/billing state (no credit, no
        # payment method): the same request will keep failing until billing
        # is fixed, so it must not be retried.
        # 413 payload-too-large is normally a property of the request, not a
        # transient rate limit — retrying the same oversized request will not
        # succeed. The exception is a provider that answers a spent per-minute
        # budget with 413: Groq does, and says so in the payload. Classifying
        # that as invalid_request loses the backoff and reports a throttle as a
        # permanent failure.
        if status == 413 and any(
            k in str(exc).lower() for k in _EXPLICIT_RATE_LIMIT_SIGNALS
        ):
            return _RATE_LIMITED
        if status in (400, 402, 413, 422):
            return _INVALID
        if status == 408 or status >= 500:
            return _TRANSIENT

    # 2.5. A request that could not be built. A ``TypeError`` with no HTTP
    # status behind it was raised while assembling the call — a wrong argument
    # type, a value the SDK cannot serialize — so the request never left the
    # process. Retrying re-sends the identical malformed call, spending the
    # whole budget on an outcome that is fixed by construction.
    if isinstance(exc, TypeError) and status is None:
        return _INVALID

    # 3. Exception class name heuristics.
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "toomanyrequests" in name:
        return _RATE_LIMITED
    if any(k in name for k in ("authentication", "permission", "unauthorized", "forbidden")):
        return _AUTH
    if "notfound" in name:
        return _NOT_FOUND
    if "timeout" in name or "timedout" in name:
        return _TIMEOUT
    if "connectionrefused" in name:
        return _UNREACHABLE
    if any(k in name for k in ("connection", "serviceunavailable", "internalserver", "apistatus", "overloaded")):
        # An SDK's generic connection error still names the underlying cause in
        # its message, so a refused port or an unresolvable host is reported as
        # unreachable rather than as a transient fault worth chasing.
        if any(k in chained_message(exc) for k in UNREACHABLE_SIGNALS):
            return _UNREACHABLE
        return _TRANSIENT
    if any(k in name for k in ("badrequest", "invalidrequest", "unprocessable", "validation")):
        return _INVALID

    # 4. Message-text heuristics (last resort).
    msg = str(exc).lower()
    # A local runtime that ran out of device memory. Reissuing the identical
    # request against the same device cannot succeed, so it must not be
    # retried; the caller has to ask for less or free memory. Cloud failures
    # never reach here — they carry an HTTP status and are classified above.
    if any(k in msg or k in name for k in DEVICE_MEMORY_SIGNALS):
        return _RESOURCE_EXHAUSTED
    # A value the SDK could not put on the wire. Same reasoning as the
    # ``TypeError`` rule above, for the SDKs that re-raise it as their own
    # exception class and keep only the sentence.
    if "not json serializable" in msg:
        return _INVALID
    # The same per-minute signals the status-code branch reads, for an error
    # that reaches here as text only — a contender's recorded error string,
    # rebuilt without the status code the original carried.
    if any(
        k in msg
        for k in (
            "rate limit", "rate-limit", "too many requests", "quota exceeded", "429",
            *_EXPLICIT_RATE_LIMIT_SIGNALS,
        )
    ):
        return _RATE_LIMITED
    # A credential that is missing or rejected. Providers call it a key or a
    # token depending on the house style — Hugging Face and Replicate say
    # "API token not found" where OpenAI says "API key not found" — and both
    # mean the caller has to set a credential, not that a model id is wrong.
    # A credential of the wrong shape is still a credential problem. Hugging
    # Face answers 400 "cannot select auto-router when using non-Hugging Face
    # API key" when the token is not one of its own — the mistake someone
    # makes pasting another provider's key into HF_TOKEN. Retrying it can
    # never succeed, so it must not read as transient.
    if "non-hugging face api key" in msg or "not a hugging face" in msg:
        return _AUTH
    if any(k in msg for k in (
        "invalid api key", "incorrect api key", "invalid_api_key", "no api key",
        "authentication", "unauthorized", "permission denied", "api key not",
        "api token not", "access token not", "api_token", "no api token",
    )):
        return _AUTH
    if "unknown provider" in msg:
        return _INVALID
    # Billing states (no credit, no payment method, spend cap) keep failing
    # until the account is topped up, so they fail fast rather than retry.
    if any(k in msg for k in (
        "insufficient credit", "insufficient funds", "payment required",
    )):
        return _INVALID
    if any(k in msg for k in ("model_not_found", "does not exist", "not found", "no such model", "unknown model")):
        return _NOT_FOUND
    # Hugging Face phrases a missing/inaccessible repo id without the words
    # "not found", so match its wording too — retrying such a load never
    # succeeds and the caller needs the model-id remediation, not a retry.
    if "is not a local folder" in msg or "not a valid model identifier" in msg:
        return _NOT_FOUND
    if "timed out" in msg or "timeout" in msg:
        return _TIMEOUT
    # Nothing was listening, or the host could not be found. Retrying may still
    # help once the server is up, but the caller's problem is the endpoint, not
    # the provider's health.
    if any(k in chained_message(exc) for k in UNREACHABLE_SIGNALS):
        return _UNREACHABLE
    # A socket that went away mid-call. Only the adapters whose SDK raises a
    # class named after the connection reach this by class name; the rest wrap
    # the operating system's error in their own message, and "connection reset
    # by peer" is not the phrase "connection error". Both are the same failure
    # and both are worth another attempt, so both say transient rather than
    # asking the caller to go and check the provider's status page.
    if any(
        k in msg
        for k in (
            "connection error",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection closed",
            "broken pipe",
            "server disconnected",
            "remote end closed connection",
            "service unavailable",
            "temporarily unavailable",
            "overloaded",
            "503",
            "502",
            "500",
        )
    ):
        return _TRANSIENT

    return _UNKNOWN


_SDK_ERROR_BODY_RE = re.compile(r"Error code: \d+ - \{")


def _balanced_brace_end(text: str, open_index: int) -> int | None:
    """Return the index just past the ``{...}`` starting at *open_index*
    (which must be ``{``), respecting quoted strings. ``None`` if unbalanced."""
    depth = 0
    in_str = False
    str_char = ""
    i = open_index
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == str_char:
                in_str = False
        elif ch in ("'", '"'):
            in_str = True
            str_char = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def simplify_embedded_provider_error(message: str) -> str:
    """Collapse an embedded raw SDK error body down to its inner message.

    OpenAI-compatible SDKs (openai, groq, together, fireworks, ...) format an
    HTTP error's ``str()`` as ``Error code: N - {<python-dict-repr of the JSON
    body>}``. An adapter that embeds that raw text inside its own actionable
    message (e.g. "request too large for X: <raw body> — reduce the request
    ...") ends up surfacing a dumped data structure instead of prose. This
    extracts the body's ``error.message`` field and substitutes it in place of
    the raw dict, leaving any surrounding text untouched. Returns *message*
    unchanged if no embedded body is found or it can't be parsed.
    """
    match = _SDK_ERROR_BODY_RE.search(message)
    if not match:
        return message
    brace_start = match.end() - 1
    end = _balanced_brace_end(message, brace_start)
    if end is None:
        return message
    try:
        parsed = ast.literal_eval(message[brace_start:end])
    except (ValueError, SyntaxError, RecursionError):
        return message
    inner = None
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            inner = err["message"]
        elif isinstance(parsed.get("message"), str):
            inner = parsed["message"]
    if not inner:
        return message
    return message[: match.start()] + inner + message[end:]


def error_has_status(exc: Any, code: int) -> bool:
    """Return whether *exc* reports the HTTP status *code*.

    The status an SDK recorded on the exception is used when it carries one.
    Otherwise the code is read out of the message as a standalone number, so a
    port, a request id, a byte count or a documentation link that merely
    contains those digits is not mistaken for the status — a 413 answered on
    port 40312 is a 413, not a 403.
    """
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value == code
        if isinstance(value, str) and value.isdigit():
            return int(value) == code
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == code
    return re.search(rf"(?<![\d.:/]){code}(?!\d)", str(exc)) is not None
