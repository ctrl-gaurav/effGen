"""Shared usage-accounting plumbing for provider adapters.

Cloud adapters repeat the same bookkeeping around every call: read token
usage off the SDK response (including cached prompt tokens), record the call
in the process-global :class:`~effgen.models._cost.CostTracker`, and build the
per-call ``GenerationResult.metadata`` block. These helpers keep that plumbing
in one place so every adapter reports the same shapes.

Internal module — no public API surface.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from effgen.models.errors import BudgetExceededError


def extract_openai_usage(usage: Any) -> tuple[int, int, int, int]:
    """Return ``(prompt, completion, total, cached)`` token counts from *usage*.

    *usage* is an OpenAI-style chat-completions usage object. Cached prompt
    tokens live under ``usage.prompt_tokens_details.cached_tokens`` and default
    to 0 when the details block is absent or ``None``.
    """
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    cached_tokens = 0
    if getattr(usage, "prompt_tokens_details", None):
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    return prompt_tokens, completion_tokens, total_tokens, cached_tokens


def cached_prompt_tokens(usage: Any) -> int:
    """Prompt tokens *usage* says the provider served from its cache.

    The OpenAI-shaped usage payload reports a cache hit under
    ``prompt_tokens_details.cached_tokens``, which every provider speaking that
    protocol either fills in or leaves absent. Absent, ``None`` and a details
    block that is a plain mapping all read as 0, so an adapter never has to
    guard the shape itself — and 0 means "this provider reported no hit", not
    "there was nothing to hit".

    Args:
        usage: The provider's usage object or mapping.

    Returns:
        The cached prompt-token count, or 0 when the payload reports none.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return 0
    value = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", 0)
    )
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def log_cache_hit(
    log: logging.Logger,
    provider: str,
    model: str,
    cached_tokens: int,
    prompt_tokens: int,
    *,
    stream: bool = False,
    cache_write_tokens: int = 0,
) -> None:
    """Say, once per call, how much of a prompt the provider served from cache.

    A run that asks for a cache and never gets one looks exactly like a run that
    never asked, so the hit is stated where it happens rather than only summed
    at the end. Nothing is logged for a call that reported neither a read nor a
    write, which is every call to a provider that does not cache.

    Args:
        log: The adapter's logger.
        provider: The provider that served the call.
        model: The model id the call used.
        cached_tokens: Prompt tokens served from the cache.
        prompt_tokens: The call's whole prompt-token count.
        stream: Whether the call was streamed.
        cache_write_tokens: Prompt tokens written into the cache.
    """
    if cached_tokens <= 0 and cache_write_tokens <= 0:
        return
    kind = "streamed call" if stream else "call"
    if cached_tokens > 0:
        share = (cached_tokens / prompt_tokens * 100.0) if prompt_tokens else 0.0
        log.info(
            "[cache] provider served %d of %d prompt tokens from its cache "
            "(%.1f%%) on a %s to %s/%s",
            cached_tokens, prompt_tokens, share, kind, provider, model,
        )
    if cache_write_tokens > 0:
        log.info(
            "[cache] provider wrote %d prompt tokens to its cache on a %s to %s/%s",
            cache_write_tokens, kind, provider, model,
        )


def cost_label(cost: float | None) -> str:
    """Render a per-call cost for a log line, or say the model has no price."""
    return "unpriced" if cost is None else f"${cost:.6f}"


def usage_metadata(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    cost: float | None,
    total_cost: float,
    cache_write_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the canonical per-call token/cost metadata block.

    ``cost_usd`` is this call's cost, or ``None`` when the model publishes no
    per-token price — never a fabricated ``0.0``, which a reader cannot tell
    from a genuine free tier. ``total_cost`` is a different number, not an
    alias: the cumulative cost across every call made on the adapter instance
    so far (including this one).

    Args:
        prompt_tokens: Input tokens the provider reported.
        completion_tokens: Output tokens the provider reported.
        total_tokens: Prompt plus completion tokens.
        cached_tokens: Prompt tokens served from the provider's cache.
        cost: This call's cost in US dollars, or ``None`` when unpriced.
        total_cost: Cumulative cost across the adapter instance, this call included.
        cache_write_tokens: Prompt tokens written to the provider's cache, on a
            provider that reports writes separately and bills them above the
            input rate. The key is added only when a number was given, so a
            provider that reports no writes carries no key rather than a ``0``
            a reader cannot tell from "it wrote nothing".

    Returns:
        The per-call token and cost block, in the shape an adapter stamps onto
        its result. Local engines report tokens without a cost and leave
        ``cost_usd`` off entirely.
    """
    block: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_tokens,
        "cost_usd": cost,
        "total_cost": total_cost,
    }
    if cache_write_tokens is not None:
        block["cache_write_tokens"] = int(cache_write_tokens)
    return block


def stringify_tool_arguments(arguments: Any) -> str:
    """Render one tool call's arguments as the JSON string the shape requires.

    A string passes through byte-for-byte, including one the model wrote
    malformed — the caller needs to see what was actually generated rather than
    a substituted empty object. Anything else (most often a mapping an SDK has
    already parsed) is serialized; ``None`` becomes ``"{}"``.
    """
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    try:
        return json.dumps(arguments, default=str)
    except (TypeError, ValueError):
        return "{}"


def tool_call_entry(
    name: Any,
    arguments: Any,
    *,
    call_id: Any = "",
    call_type: Any = "function",
) -> dict[str, Any]:
    """Build one element of ``metadata["tool_calls"]`` in the documented shape.

    Args:
        name: The tool the model asked for.
        arguments: The call arguments, as the provider sent them.
        call_id: The provider's id for the call, used to match the result back.
        call_type: The call kind the provider reported.

    Returns:
        One entry in the documented ``metadata["tool_calls"]`` shape.
    """
    return {
        "id": str(call_id) if call_id else "",
        "type": str(call_type) if call_type else "function",
        "function": {
            "name": str(name) if name else "",
            "arguments": stringify_tool_arguments(arguments),
        },
    }


def accumulate_stream_tool_call_deltas(
    buffer: dict[int, dict[str, Any]], deltas: Any
) -> None:
    """Fold one streamed chunk's tool-call fragments into *buffer*.

    Providers with an OpenAI-shaped stream send a call in pieces: the index and
    id first, the function name once, and the arguments as a run of string
    fragments. *buffer* is keyed by the call's index so parallel calls in one
    turn stay separate, and the fragments are concatenated in arrival order.

    Args:
        buffer: The per-index accumulator, mutated in place.
        deltas: The chunk's ``delta.tool_calls`` list.
    """
    for delta in deltas or []:
        index = delta.index if getattr(delta, "index", None) is not None else 0
        entry = buffer.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if getattr(delta, "id", None):
            entry["id"] = delta.id
        function = getattr(delta, "function", None)
        if function is None:
            continue
        if getattr(function, "name", None):
            entry["function"]["name"] = function.name
        if getattr(function, "arguments", None):
            entry["function"]["arguments"] += function.arguments


def stream_tool_call_entries(
    buffer: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return *buffer* as ``metadata["tool_calls"]`` entries, in index order.

    The accumulated ``arguments`` stay the JSON string the model streamed,
    matching what a non-streamed call reports.
    """
    return [
        tool_call_entry(
            entry["function"]["name"],
            entry["function"]["arguments"],
            call_id=entry["id"],
            call_type=entry["type"],
        )
        for _index, entry in sorted(buffer.items())
    ]


def normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Coerce a provider's tool-call list into the documented shape.

    Accepts the nested OpenAI form and the flat ``{"name", "arguments"}`` form,
    stringifying an already-parsed ``arguments`` in either. An element that
    matches neither is passed through unchanged rather than dropped, so a
    provider shape that is not modelled here still reaches the caller. A
    non-list *raw* yields ``[]``.
    """
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        function = item.get("function")
        if isinstance(function, dict) and function.get("name"):
            entry = dict(item)
            entry["id"] = str(item["id"]) if item.get("id") else ""
            entry["type"] = str(item.get("type") or "function")
            entry["function"] = {
                **function,
                "name": str(function["name"]),
                "arguments": stringify_tool_arguments(function.get("arguments")),
            }
            normalized.append(entry)
        elif item.get("name"):
            normalized.append(tool_call_entry(
                item["name"],
                item.get("arguments"),
                call_id=item.get("id", ""),
                call_type=item.get("type", "function"),
            ))
        else:
            normalized.append(item)
    return normalized


def tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    """Convert OpenAI-style ``message.tool_calls`` into a list of plain dicts.

    This is the definition of the tool-call shape every adapter reports in
    ``GenerationResult.metadata["tool_calls"]``:

    1. The key is always present and always a list — ``[]`` for a turn that
       called nothing, so a reader never has to guard against a missing key.
    2. Every element carries ``id`` (the provider's call id, ``""`` when the
       provider sends none), ``type`` (``"function"`` for a function call) and
       ``function`` with ``name`` and ``arguments``.
    3. ``arguments`` is a JSON **string**, exactly as the model generated it.
       Adapters never parse it: OpenAI-compatible wire formats require the
       string, and a model that emits malformed JSON stays visible instead of
       arriving as an empty argument set.
    4. Elements keep the provider's order, and parallel calls are separate
       elements.

    The Gemini adapter carries top-level ``name``/``arguments`` keys beside the
    nested block for callers written against its previous flat shape, and the
    OpenAI Responses API entries keep their ``type: "function_call"``
    discriminator and flat keys beside it.
    """
    tool_calls: list[dict[str, Any]] = []
    raw_calls = getattr(message, "tool_calls", None)
    if raw_calls:
        for tc in raw_calls:
            tool_calls.append(tool_call_entry(
                tc.function.name,
                tc.function.arguments,
                call_id=getattr(tc, "id", ""),
                call_type=getattr(tc, "type", "function"),
            ))
    return tool_calls


def record_tracker_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    log: logging.Logger,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Record one call in the process-global :class:`CostTracker`.

    Returns the catalog-priced USD cost (so it matches the model catalog and
    ``effgen cost``), or ``None`` when the model has no published price or
    tracking is unavailable. :class:`BudgetExceededError` propagates so budget
    limits stop the run; any other tracker failure is confined to a debug log —
    accounting must never break a successful generation.

    Args:
        provider: The provider that served the call.
        model: The model id the call used.
        prompt_tokens: Input tokens the call consumed.
        completion_tokens: Output tokens the call produced.
        log: Logger a tracker failure is confined to.
        cached_tokens: How many of *prompt_tokens* the provider served from its
            cache. Priced at the catalog's cached-input rate where the provider
            publishes one, and at the ordinary input rate where it does not —
            a saving is never assumed on the caller's behalf.
        cache_write_tokens: Prompt tokens written to the cache, billed above the
            input rate on the providers that report them.
    """
    try:
        from effgen.models._cost import CostTracker

        return CostTracker.get().record(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    except BudgetExceededError:
        raise
    except Exception:
        log.debug("CostTracker recording failed for %s", provider, exc_info=True)
        return None


__all__ = [
    "cached_prompt_tokens",
    "extract_openai_usage",
    "log_cache_hit",
    "cost_label",
    "usage_metadata",
    "stringify_tool_arguments",
    "tool_call_entry",
    "normalize_tool_calls",
    "tool_calls_from_message",
    "record_tracker_cost",
]
