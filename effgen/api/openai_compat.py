"""OpenAI-compatible API endpoints.

Provides ``/v1/chat/completions`` and ``/v1/completions`` endpoints matching the
OpenAI REST API spec, so the official ``openai`` Python client (and any
OpenAI-compatible client) can be pointed at an effGen server unchanged.

Compatibility level
-------------------
* **Usage** is real: provider-reported counts when the upstream API returns
  them, otherwise tokenizer counts — never a ``len(text) // 4`` estimate.
* **Model aliases** (``gpt-4`` → a local model, see :data:`MODEL_ALIASES`) are a
  documented compatibility *shim*. The response's ``model`` field reports the
  model that actually ran, and a non-standard ``effgen`` object records the
  requested alias vs. the resolved model (OpenAI clients ignore unknown keys).
* **Streaming** (SSE) is truly incremental. When the client sets
  ``stream_options.include_usage`` a final usage-only chunk is emitted; without
  it, no usage chunk is sent (matching OpenAI's behaviour).
* **Tools**: effGen runs tools **server-side** (its ReAct loop) and returns the
  final assistant message. It does **not** stream client-side ``tool_calls``
  deltas for the client to execute — passing ``tools`` lets effGen *use* its own
  registered tools; the answer comes back already resolved. A non-streaming
  response may carry a ``tool_calls`` array when a runner surfaces one. A tool
  the server does not host is **rejected with a clear 400** (it is never
  silently ignored), so a client expecting OpenAI function-calling is told
  plainly rather than getting prose back.
* **Errors** use the OpenAI error envelope (``{"error": {...}}``) with a sane
  status (404 model-not-found, 401 client-auth, 502/503 upstream-provider
  failure, 429 rate-limit, …) and are redacted.
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Model catalog (read-only)
# ---------------------------------------------------------------------------
from effgen.api.openai_compat_catalog import (  # noqa: E402,F401  re-exported for import/patch parity
    _LOCAL_WEIGHT_SUFFIXES,
    _local_cached_models,
    build_model_catalog,
)

# The endpoint layer is assembled from sibling modules — request schemas and
# model aliases, response/usage builders, error classification, the model
# catalog, and the SSE streaming generators. Every name they define is
# re-exported here so ``from effgen.api.openai_compat import X`` and patches
# against this module resolve unchanged.
from effgen.api.openai_compat_models import (  # noqa: F401  re-exported for import/patch parity
    _FALLBACK_DEFAULT_MODEL,
    DEFAULT_MODEL_ALIASES,
    MODEL_ALIASES,
    BaseModel,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    Field,
    RunnerResult,
    default_model_id,
    resolve_model_alias,
)

# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------
from effgen.api.openai_compat_responses import (  # noqa: E402,F401  re-exported for import/patch parity
    _approx_tokens,
    _chat_id,
    _cmpl_id,
    _get_tiktoken_encoder,
    _now,
    build_chat_chunk,
    build_chat_completion,
    build_text_completion,
    build_usage_chunk,
    count_tokens,
)

# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

# Type alias for the runner callable injected by the API server.
#
#   runner(prompt: str, *, model: str, tools: list, stream: bool)
#       -> str | iterator | RunnerResult
#
# Either shape is accepted: a plain function, which is dispatched to a worker
# thread, or an ``async def`` function, which is awaited on the event loop.
Runner = Callable[..., Any]


async def _call_runner(runner: Runner, prompt: str, **kwargs: Any) -> Any:
    """Invoke the runner, whether it is synchronous or asynchronous.

    A synchronous runner blocks for the duration of its work — model loading,
    the provider HTTP call, tool execution — so calling it directly from a
    coroutine holds the event loop for the whole generation and makes
    concurrent requests to ``/health``, ``/metrics`` and the dashboard
    endpoints wait behind it. Dispatching it to a thread keeps those endpoints
    answering while a generation is in flight.

    An ``async def`` runner is the shape an integrator whose model call is
    already asynchronous reaches for first. It is awaited directly: threading it
    would return an un-awaited coroutine object, which the caller then fails to
    read as a result, and the client would see an unexplained 500. Anything else
    that returns an awaitable — a ``functools.partial`` around a coroutine
    function, a class whose ``__call__`` is ``async`` — is awaited after the
    thread hands it back, so both spellings work.

    For a streaming request this covers the setup that produces the iterator;
    the iterator itself is consumed by :class:`StreamingResponse`, which
    already iterates a synchronous generator in a worker thread.
    """
    if asyncio.iscoroutinefunction(runner):
        return await runner(prompt, **kwargs)
    result = await asyncio.to_thread(runner, prompt, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


# Error classification and the shared OpenAI error envelope live in a sibling
# module (they are also used by the server middleware and auth layers).
from effgen.api.openai_compat_errors import (  # noqa: E402,F401  re-exported for import/patch parity
    _STATUS_ERROR_CODE,
    _STATUS_ERROR_TYPE,
    UnknownToolError,
    _classify_http,
    _error_payload,
    _redact,
    error_envelope,
)

# The SSE generator bodies for the two streaming endpoints.
from effgen.api.openai_compat_streaming import (  # noqa: E402
    chat_sse_iter,
    completion_sse_iter,
)


def _effgen_meta(requested: str, resolved: str) -> dict[str, Any]:
    """Build the non-standard ``effgen`` response object documenting aliasing.

    OpenAI-style aliases (e.g. ``gpt-4``) are mapped to concrete effGen models
    via :data:`MODEL_ALIASES`. Rather than silently swapping the model, every
    response carries an ``effgen`` object naming the requested alias and the
    model that actually ran so clients can tell when a compatibility shim was
    applied. OpenAI clients ignore unknown top-level keys.
    """
    return {
        "requested_model": requested,
        "resolved_model": resolved,
        # True when a compatibility alias (gpt-4 → local model) or the
        # effgen-default/default name was applied — not for internal
        # provider/model routing normalization.
        "alias_applied": requested in MODEL_ALIASES or requested in DEFAULT_MODEL_ALIASES,
    }


def _messages_to_prompt(messages: list[Any]) -> str:
    """Flatten a chat message list into a single prompt string.

    Used as a fallback path when an effGen agent is invoked directly rather
    than a chat-native model. The format mirrors ChatML loosely.
    """
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None) if not isinstance(msg, dict) else msg.get("role")
        content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        if isinstance(content, list):
            # Multimodal content: extract text parts only.
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        parts.append(f"{role}: {content or ''}")
    return "\n".join(parts)


def _has_actionable_content(messages: list[Any]) -> bool:
    """Return True if there is anything for the model to act on.

    A request whose every message has empty/whitespace/``None``/absent text
    content — and no image parts, no ``tool_calls``, and no tool result — gives
    the model nothing to answer. Such a request is rejected with a 400 before a
    billed model call, matching the Agent layer's empty-task guard and OpenAI's
    own handling. A message counts as actionable when it has non-blank text, a
    non-empty multimodal content list, ``tool_calls``, or is a ``tool`` result.
    """
    for msg in messages:
        is_dict = isinstance(msg, dict)
        role = msg.get("role") if is_dict else getattr(msg, "role", None)
        content = msg.get("content") if is_dict else getattr(msg, "content", None)
        tool_calls = msg.get("tool_calls") if is_dict else getattr(msg, "tool_calls", None)
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list) and content:
            return True
        if tool_calls:
            return True
        if role == "tool" and content is not None:
            return True
    return False


def create_openai_router(
    runner: Runner, *, extra_models: Callable[[], list[str]] | None = None
) -> Any:
    """Create a FastAPI router exposing OpenAI-compatible endpoints.

    Parameters
    ----------
    runner:
        Callable invoked with (prompt, *, model, tools, stream). Should return
        a string (non-stream) or an iterable of string chunks (stream).
    extra_models:
        Optional callable returning model ids the server has actually served
        this run (e.g. the pooled-model cache). ``GET /v1/models`` lists these
        alongside the drop-in :data:`MODEL_ALIASES` so a client can discover
        real, currently-servable ids instead of only the 6 legacy aliases.

    Returns
    -------
    fastapi.APIRouter
        Router to mount on the main FastAPI app.
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "fastapi is required for the OpenAI-compatible API. "
            "Install with `pip install fastapi`."
        ) from e

    def _error_response(exc: Exception) -> Any:
        """Return an OpenAI-style, redacted error JSONResponse for *exc*."""
        status, err_type, code = _classify_http(exc)
        message = str(exc)
        # A bare/unknown model id falls through to the local loader and fails
        # with a Transformers "not a valid model identifier" message. Add a
        # one-line hint pointing at the id shapes the server accepts.
        low = message.lower()
        if code == "model_not_found" and (
            "not a valid model identifier" in low or "is not a local folder" in low
        ):
            message += (
                " Pass a provider-prefixed model id (e.g. 'openai:gpt-5-nano', "
                "'groq:openai/gpt-oss-20b'), a valid local model id, or "
                "'effgen-default'."
            )
        headers: dict[str, str] = {}
        # A 429 the upstream provider produced carries whatever delay that
        # provider stated. The draft `RateLimit-*` headers describe *this*
        # server's window, so reporting the upstream's under those names would
        # mislead; `Retry-After` states a delay and nothing more, so it is the
        # one to pass on. Without it the client has nothing to pace against and
        # either retries immediately or invents a number.
        if status == 429:
            from effgen.models.errors import retry_after_seconds

            delay = retry_after_seconds(exc)
            if delay is not None:
                headers["Retry-After"] = str(max(1, int(round(delay))))
        return JSONResponse(
            status_code=status,
            content=_error_payload(message, err_type, code),
            headers=headers or None,
        )

    router = APIRouter(prefix="/v1", tags=["openai-compat"])

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        # The list is the drop-in aliases plus the ids this process has actually
        # served a successful response for this run. It is not exhaustive: any
        # `provider:model` id the server can reach (e.g. "openai:gpt-5-nano",
        # "groq:openai/gpt-oss-20b") is callable whether or not it appears here.
        now = _now()
        # The legacy OpenAI-flagship names (gpt-4, gpt-3.5-turbo, ...) are
        # drop-in compatibility aliases, each mapped to a concrete local model
        # (see ``root``). Mark them so a client does not read the list as a claim
        # that the server serves GPT-4 itself.
        data = [
            {
                "id": alias,
                "object": "model",
                "created": now,
                "owned_by": "effgen",
                "root": target,
                "effgen": {"compatibility_alias": True, "mapped_to": target},
            }
            for alias, target in MODEL_ALIASES.items()
        ]
        # The names that route to the server's configured default model.
        for name in sorted(DEFAULT_MODEL_ALIASES):
            data.append({
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "effgen",
                "root": default_model_id(),
                "effgen": {"compatibility_alias": True, "mapped_to": default_model_id()},
            })
        # Alongside the drop-in legacy aliases, list the ids the server has
        # actually served this run (e.g. "openai:gpt-5-nano"), so a client
        # discovers real, currently-servable models instead of only the
        # hardcoded aliases.
        if extra_models is not None:
            try:
                served = extra_models()
            except Exception:  # noqa: BLE001 - discoverability is best-effort
                served = []
            for model_id in served:
                if model_id in MODEL_ALIASES:
                    continue
                data.append({
                    "id": model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "effgen",
                    "root": model_id,
                })
        # OpenAI's schema is ``{object, data}``; the extra ``effgen`` key is an
        # additive hint (standard clients read ``data`` and ignore it) telling a
        # caller the list is not exhaustive — any reachable ``provider:model`` id
        # is callable whether or not it is listed here.
        return {
            "object": "list",
            "data": data,
            "effgen": {
                "note": (
                    "Listed ids are drop-in compatibility aliases plus models "
                    "this server has served this run. Any reachable "
                    "'provider:model' id (e.g. 'openai:gpt-5-nano', "
                    "'groq:openai/gpt-oss-20b') is also accepted as the "
                    "request 'model', whether or not it appears in this list."
                ),
            },
        }

    @router.get("/models/catalog")
    async def models_catalog(provider: str | None = None) -> dict[str, Any]:
        """Return the model catalog with pricing, capabilities and provenance.

        Unlike ``GET /v1/models`` (drop-in aliases plus ids served this run),
        this reports the same catalog the ``effgen models`` CLI reads: every
        known provider model with its context window, per-1M pricing, tool/vision
        support, free-tier flag, and the ``verified_on`` date and ``price_source``
        the price came from — plus the models present in the local cache. Prices
        are ``None`` when unpublished (never a fabricated ``$0``), so a picker can
        label priced/free/unpriced accurately. Pass ``?provider=<name>`` to scope
        to one provider. Degrades to an empty list if the catalog is unreadable
        rather than failing the request.
        """
        return build_model_catalog(provider)

    @router.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        # OpenAI rejects an empty `messages` array with a 400; match that instead
        # of inventing a question and replying 200 to a content-free request.
        if not request.messages:
            return JSONResponse(
                status_code=400,
                content=error_envelope(
                    400, "messages must not be empty", code="empty_messages"
                ),
            )
        # A request whose messages carry no text, no image, no tool_calls, and no
        # tool result gives the model nothing to answer. Reject it with a 400
        # before any billed model call rather than returning a paid-for
        # non-answer at 200.
        if not _has_actionable_content(request.messages):
            return JSONResponse(
                status_code=400,
                content=error_envelope(
                    400, "message content must not be empty", code="empty_content"
                ),
            )
        resolved = resolve_model_alias(request.model)
        model = resolved  # echoed in the response 'model' field (what actually ran)
        prompt = _messages_to_prompt(request.messages)
        prompt_tokens = _approx_tokens(prompt)
        effgen_meta = _effgen_meta(request.model, resolved)

        try:
            result = await _call_runner(
                runner,
                prompt,
                model=resolved,
                tools=request.tools or [],
                stream=bool(request.stream),
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            return _error_response(e)

        if request.stream:
            chat_id = _chat_id()
            include_usage = bool(
                (request.stream_options or {}).get("include_usage")
            )
            return StreamingResponse(
                chat_sse_iter(
                    result,
                    model=model,
                    chat_id=chat_id,
                    prompt_tokens=prompt_tokens,
                    effgen_meta=effgen_meta,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: prefer real usage + resolved model from a RunnerResult.
        if isinstance(result, RunnerResult):
            content = result.text
            if result.prompt_tokens is not None:
                prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            tool_calls = result.tool_calls
            finish_reason = result.finish_reason
            # Surface per-call cost next to token usage so operators can bill
            # and track. Lives in the non-standard `effgen` extension (OpenAI
            # clients ignore unknown keys); `None` when the model has no pricing.
            if result.cost_usd is not None:
                effgen_meta = {**effgen_meta, "cost_usd": result.cost_usd}
            # Surface the tool/step trace and run id a runner recorded (also in
            # the `effgen` extension). ``trace`` is a compact list of the tools
            # the server ran — ``{tool, args, result_summary, ok, duration_ms}``
            # — so a caller can show what the agent did; ``tool_calls`` is the
            # count. Absent when the run used no tools.
            # ``stop_reason``/``outcome``/``partial`` say whether the model
            # finished. A run the loop stopped is served as a completed request
            # whose generation was cut short (``finish_reason="length"``), and
            # these three are how a caller tells that apart from an answer
            # without parsing the content.
            extra = result.metadata or {}
            for key in (
                "trace", "tool_calls", "run_id", "stop_reason", "outcome", "partial",
            ):
                value = extra.get(key)
                if value:
                    effgen_meta = {**effgen_meta, key: value}
        else:
            content = result if isinstance(result, str) else "".join(str(c) for c in result)
            completion_tokens = None
            tool_calls = None
            finish_reason = "stop"

        return build_chat_completion(
            model,
            content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            effgen_meta=effgen_meta,
        )

    @router.post("/completions")
    async def completions(request: CompletionRequest) -> Any:
        model = resolve_model_alias(request.model)
        prompt = (
            request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
        )
        # An empty or whitespace-only prompt gives the model nothing to complete.
        # Reject it with a 400 before any billed call, the same way
        # /v1/chat/completions rejects content-free messages.
        if not prompt.strip():
            return JSONResponse(
                status_code=400,
                content=error_envelope(
                    400, "prompt must not be empty", code="empty_prompt"
                ),
            )
        prompt_tokens = _approx_tokens(prompt)

        try:
            result = await _call_runner(
                runner,
                prompt,
                model=model,
                tools=[],
                stream=bool(request.stream),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            return _error_response(e)

        if request.stream:
            cmpl_id = _cmpl_id()
            return StreamingResponse(
                completion_sse_iter(result, model=model, cmpl_id=cmpl_id),
                media_type="text/event-stream",
            )

        if isinstance(result, RunnerResult):
            text = result.text
            if result.prompt_tokens is not None:
                prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
        else:
            text = result if isinstance(result, str) else "".join(str(c) for c in result)
            completion_tokens = None
        return build_text_completion(
            model, text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

    return router
