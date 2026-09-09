"""Failure injection for provider adapters: a local server that answers badly.

The matrix in ``test_failure_matrix.py`` needs every provider adapter to meet
every way a provider call can go wrong — a rejected credential, a rate limit, a
request that never gets an answer, a body that is not what the SDK expects, a
stream that stops halfway, a connection that is torn down. Those are properties
of the *transport*, not of any model, so they are produced here by a real HTTP
server bound to loopback that the provider SDK talks to over a real socket. The
adapter, the SDK, the HTTP parser and effGen's classification layer all run
unmodified; only the peer is local.

Two tables drive everything:

* :data:`PROVIDERS` — one row per adapter: how to build it, how to point its
  SDK at the fault server, and which wire dialect that SDK speaks.
* :data:`FAILURES` — one row per failure class: what the server does, and what
  the adapter has to report when it happens.

A new provider is a row in the first table; a new failure class is a row in the
second, plus one handler per dialect. Neither needs a new module.

Every request carries :data:`SENTINEL_CREDENTIAL` as its key, and the fault
server echoes that value back inside its error bodies the way a real provider
sometimes does, so each cell can check that the credential does not survive into
the exception the caller sees.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FAILURES",
    "PROVIDERS",
    "SENTINEL_CREDENTIAL",
    "CellOutcome",
    "FailureSpec",
    "FaultServer",
    "MatrixReport",
    "ProviderSpec",
    "run_cell",
]

#: The credential every injected call is made with. Distinctive enough that a
#: substring search for it in an error message, a log line or a repr is
#: conclusive, and shaped like a real key so redaction has something to match.
SENTINEL_CREDENTIAL = "sk-effgen-faultmatrix-4a7c9e1b0d2f6835a1c4e7b9d0f2a4c6"

#: Seconds an adapter is given before a cell calls the call hung. Every adapter
#: is built with a client timeout of :data:`CLIENT_TIMEOUT_S`, so anything past
#: this is the adapter waiting on something it was told not to wait for.
HANG_LIMIT_S = 25.0

#: The per-call timeout every adapter in the matrix is built with.
CLIENT_TIMEOUT_S = 4.0


# ---------------------------------------------------------------------------
# Wire dialects — what a body has to look like for each SDK
# ---------------------------------------------------------------------------


def _openai_ok(text: str, finish: str) -> bytes:
    return json.dumps({
        "id": "chatcmpl-fault", "object": "chat.completion", "created": 1,
        "model": "fault-matrix",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }).encode()


def _anthropic_ok(text: str, finish: str) -> bytes:
    return json.dumps({
        "id": "msg_fault", "type": "message", "role": "assistant",
        "model": "fault-matrix",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "max_tokens" if finish == "length" else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }).encode()


def _gemini_ok(text: str, finish: str) -> bytes:
    return json.dumps({
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "finishReason": "MAX_TOKENS" if finish == "length" else "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": 11, "candidatesTokenCount": 4, "totalTokenCount": 15,
        },
        "modelVersion": "fault-matrix",
    }).encode()


def _replicate_ok(text: str, finish: str) -> bytes:
    return json.dumps({
        "id": "pred_fault", "model": "fault/matrix", "version": "v1",
        "status": "succeeded", "input": {"prompt": "hi"},
        "output": [text],
        "metrics": {"predict_time": 0.1, "total_time": 0.2,
                    "input_token_count": 11, "output_token_count": 4},
        "urls": {},
    }).encode()


def _openai_error(status: int, message: str, code: str) -> bytes:
    return json.dumps({
        "error": {"message": message, "type": "invalid_request_error", "code": code}
    }).encode()


def _gemini_error(status: int, message: str, code: str) -> bytes:
    return json.dumps({
        "error": {"code": status, "message": message, "status": code.upper()}
    }).encode()


def _replicate_error(status: int, message: str, code: str) -> bytes:
    return json.dumps({"detail": message, "status": status, "title": code}).encode()


def _openai_chunk(text: str, finish: str | None) -> bytes:
    return ("data: " + json.dumps({
        "id": "chatcmpl-fault", "object": "chat.completion.chunk", "created": 1,
        "model": "fault-matrix",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    }) + "\n\n").encode()


def _gemini_chunk(text: str, finish: str | None) -> bytes:
    return ("data: " + json.dumps({
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "finishReason": finish, "index": 0,
        }],
        "modelVersion": "fault-matrix",
    }) + "\n\n").encode()


def _anthropic_stream_prologue() -> list[bytes]:
    start = {"type": "message_start", "message": {
        "id": "msg_fault", "type": "message", "role": "assistant",
        "model": "fault-matrix", "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 0},
    }}
    block = {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}}
    return [
        f"event: message_start\ndata: {json.dumps(start)}\n\n".encode(),
        f"event: content_block_start\ndata: {json.dumps(block)}\n\n".encode(),
    ]


def _anthropic_chunk(text: str, finish: str | None) -> bytes:
    delta = {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": text}}
    return f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode()


@dataclass(frozen=True)
class Dialect:
    """How one SDK family expects a body to be shaped.

    Args:
        name: Dialect key used by :class:`ProviderSpec`.
        ok: Builds a complete non-streaming answer.
        error: Builds an error body for a status code.
        chunk: Builds one streaming frame.
        prologue: Frames a stream must open with before any content frame.
        cap_keys: The names this wire protocol carries the caller's output-token
            cap under. The truncation cell reads the request the adapter
            actually sent and checks the cap arrived, because a cap that is
            dropped on the way out costs money and produces no truncation
            signal to detect.
    """

    name: str
    ok: Callable[[str, str], bytes]
    error: Callable[[int, str, str], bytes]
    chunk: Callable[[str, str | None], bytes]
    prologue: Callable[[], list[bytes]] = list
    cap_keys: tuple[str, ...] = ("max_tokens",)


DIALECTS: dict[str, Dialect] = {
    # A reasoning model takes the cap under ``max_completion_tokens``; the
    # chat-completions models take ``max_tokens``. Either name is the cap
    # having arrived.
    "openai": Dialect(
        "openai", _openai_ok, _openai_error, _openai_chunk,
        cap_keys=("max_tokens", "max_completion_tokens"),
    ),
    "anthropic": Dialect(
        "anthropic", _anthropic_ok, _openai_error, _anthropic_chunk,
        _anthropic_stream_prologue,
    ),
    "gemini": Dialect(
        "gemini", _gemini_ok, _gemini_error, _gemini_chunk,
        cap_keys=("maxOutputTokens",),
    ),
    "replicate": Dialect(
        "replicate", _replicate_ok, _replicate_error, _openai_chunk,
        cap_keys=("max_tokens", "max_new_tokens"),
    ),
}


# ---------------------------------------------------------------------------
# The fault server
# ---------------------------------------------------------------------------


class FaultServer:
    """A loopback HTTP peer that answers one way, badly, on purpose.

    Raw sockets rather than :mod:`http.server` because two of the failure
    classes are not expressible as a response: a connection reset needs the
    socket torn down with ``SO_LINGER`` at zero, and a mid-stream abort needs
    the connection dropped between chunk frames with no terminating chunk.

    Requests that are not the generation call — the token counts and model
    lookups an adapter makes while loading — are answered normally, so the
    injected failure lands on the call under test rather than on ``load()``.

    Args:
        mode: Which failure to inject; one handler per name below.
        dialect: Which wire shape to answer in.
        delay_s: How long the ``timeout`` mode holds the connection open.
        retry_after_s: The delay the ``rate_limit`` mode states, in the header
            and in the body.
    """

    def __init__(
        self,
        mode: str,
        dialect: str = "openai",
        delay_s: float = HANG_LIMIT_S + 2.0,
        retry_after_s: int = 17,
    ) -> None:
        self.mode = mode
        self.dialect = DIALECTS[dialect]
        self.delay_s = delay_s
        self.retry_after_s = retry_after_s
        self.requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(128)
        self.port = int(self._sock.getsockname()[1])
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)

    @property
    def url(self) -> str:
        """Base URL to point an SDK at."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def call_count(self) -> int:
        """How many generation requests reached the server."""
        return len(self.requests)

    def start(self) -> "FaultServer":
        """Begin accepting connections."""
        self._accept_thread.start()
        return self

    def stop(self) -> None:
        """Stop accepting and close the listening socket."""
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    # -- connection handling ------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(HANG_LIMIT_S)
            head, body = _read_request(conn)
            if not head:
                return
            request_line = head.split("\r\n", 1)[0]
            method, path = (request_line.split(" ") + ["", ""])[:2]
            if method.upper() != "POST" or not _is_generation_path(path):
                self._respond(conn, 200, "OK", self._preflight_body(path))
                return
            with self._lock:
                self.requests.append((path, body))
            handler = getattr(self, f"_inject_{self.mode}")
            handler(conn)
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _preflight_body(self, path: str) -> bytes:
        """Answer the lookups an adapter makes on the way to the call."""
        if "counttokens" in path.lower():
            return json.dumps({"totalTokens": 11}).encode()
        if "count_tokens" in path.lower():
            return json.dumps({"input_tokens": 11}).encode()
        return json.dumps({"id": "fault-matrix", "object": "model"}).encode()

    def _respond(
        self,
        conn: socket.socket,
        status: int,
        reason: str,
        body: bytes,
        extra_headers: tuple[str, ...] = (),
    ) -> None:
        head = [
            f"HTTP/1.1 {status} {reason}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
            *extra_headers,
            "Connection: close",
            "",
            "",
        ]
        conn.sendall("\r\n".join(head).encode() + body)

    def _abort(self, conn: socket.socket) -> None:
        """Tear the connection down so the peer sees a reset, not a close."""
        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            conn.close()
        except OSError:
            pass

    # -- one method per failure class --------------------------------------

    def _inject_auth_rejected(self, conn: socket.socket) -> None:
        body = self.dialect.error(
            401,
            f"Incorrect API key provided: {SENTINEL_CREDENTIAL}. "
            "You can find your API key at the provider console.",
            "unauthenticated",
        )
        self._respond(conn, 401, "Unauthorized", body)

    def _inject_rate_limit(self, conn: socket.socket) -> None:
        body = self.dialect.error(
            429,
            f"Rate limit reached for requests. Limit 30 per minute. "
            f"Please try again in {self.retry_after_s}s.",
            "resource_exhausted",
        )
        self._respond(
            conn, 429, "Too Many Requests", body,
            (f"Retry-After: {self.retry_after_s}",
             "x-ratelimit-limit-requests: 30",
             "x-ratelimit-remaining-requests: 0"),
        )

    def _inject_timeout(self, conn: socket.socket) -> None:
        # Hold the request open with no answer. Past the hang limit the cell has
        # already failed, so the connection is torn down rather than held for
        # the rest of the run.
        deadline = time.monotonic() + self.delay_s
        while time.monotonic() < deadline and not self._stop.is_set():
            time.sleep(0.05)
        self._abort(conn)

    def _inject_connection_reset(self, conn: socket.socket) -> None:
        self._abort(conn)

    def _inject_malformed_body(self, conn: socket.socket) -> None:
        # HTTP 200 with a body that is not the JSON the SDK is going to parse:
        # what a proxy, a captive portal or a load balancer returns when it
        # answers on the provider's behalf.
        self._respond(
            conn, 200, "OK",
            b"<html><head><title>502 Bad Gateway</title></head>"
            b"<body>upstream connect error</body></html>",
        )

    def _inject_oversized_request(self, conn: socket.socket) -> None:
        body = self.dialect.error(
            413,
            "Request too large for model fault-matrix: the prompt exceeds the "
            "maximum context length. Reduce the length of the messages.",
            "invalid_argument",
        )
        self._respond(conn, 413, "Payload Too Large", body)

    def _inject_truncation(self, conn: socket.socket) -> None:
        self._respond(conn, 200, "OK", self.dialect.ok("The answer beg", "length"))

    def _inject_mid_stream_abort(self, conn: socket.socket) -> None:
        head = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/event-stream",
            "Cache-Control: no-cache",
            "Transfer-Encoding: chunked",
            "Connection: close",
            "",
            "",
        ]
        conn.sendall("\r\n".join(head).encode())
        frames = [*self.dialect.prologue(),
                  self.dialect.chunk("The ", None),
                  self.dialect.chunk("answer", None)]
        for frame in frames:
            conn.sendall(f"{len(frame):x}\r\n".encode() + frame + b"\r\n")
            time.sleep(0.01)
        # No terminating zero-length chunk and no `data: [DONE]`: the body is
        # cut off exactly where a dropped upstream cuts one off.
        self._abort(conn)


def _read_request(conn: socket.socket) -> tuple[str, str]:
    """Read one HTTP request; return ``(headers, body)`` as text."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return "", ""
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1])
            except ValueError:
                length = 0
    while len(rest) < length:
        chunk = conn.recv(65536)
        if not chunk:
            break
        rest += chunk
    return head.decode("latin-1"), rest.decode("utf-8", "replace")


_GENERATION_PATH_MARKERS = (
    "chat/completions",
    "generatecontent",
    "/v1/messages",
    "predictions",
    "/models/",
)


def _is_generation_path(path: str) -> bool:
    """Whether *path* is the call under test rather than a load-time lookup."""
    low = path.lower()
    if "counttokens" in low or "count_tokens" in low:
        return False
    return any(marker in low for marker in _GENERATION_PATH_MARKERS)


# ---------------------------------------------------------------------------
# Provider table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    """One adapter and everything needed to drive it at a fault server.

    Args:
        name: Provider key, as used by ``load_model``.
        dialect: Which wire shape the SDK expects.
        key_envs: Every environment variable the adapter reads a credential
            from. All of them are set to the sentinel for an injected call and
            all of them are cleared for the credential-absent cell.
        build: ``build(base_url) -> adapter``, already loaded and pointed at the
            fault server.
        build_without_credential: ``() -> adapter`` with no credential anywhere;
            the call it raises on is the credential-absent cell.
        missing_credential_hint: Text the credential-absent error has to carry
            so the caller is told what to set.
        sdk_module: The provider SDK the adapter imports. Most providers are an
            optional extra, so on an install without that extra every cell for
            the provider is recorded not applicable with that as the reason
            rather than failing — the matrix covers what is installed, and says
            what it did not cover.
        streaming: Whether ``generate_stream`` is exercised for this provider.
        skips: ``{failure name: reason}`` for cells this provider cannot
            express, printed in the report so nothing is dropped silently.
    """

    name: str
    dialect: str
    key_envs: tuple[str, ...]
    build: Callable[[str], Any]
    build_without_credential: Callable[[], Any]
    missing_credential_hint: str
    sdk_module: str
    streaming: bool = True
    skips: dict[str, str] = field(default_factory=dict)

    def sdk_missing(self) -> str:
        """The reason to record when this provider's SDK is absent, else ``""``."""
        import importlib.util

        try:
            if importlib.util.find_spec(self.sdk_module) is not None:
                return ""
        except (ImportError, ValueError):
            pass
        return f"the {self.sdk_module} SDK is not installed in this environment"


def _openai_provider(url: str) -> Any:
    from effgen.models.openai_adapter import OpenAIAdapter

    os.environ["OPENAI_BASE_URL"] = f"{url}/v1"
    adapter = OpenAIAdapter(
        model_name="gpt-5-nano", api_key=SENTINEL_CREDENTIAL,
        timeout=int(CLIENT_TIMEOUT_S), max_retries=0,
    )
    adapter.load()
    return adapter


def _groq_provider(url: str) -> Any:
    from effgen.models.groq_adapter import GroqAdapter

    os.environ["GROQ_BASE_URL"] = url
    adapter = GroqAdapter(
        model_name="openai/gpt-oss-20b", api_key=SENTINEL_CREDENTIAL,
        timeout=int(CLIENT_TIMEOUT_S), max_retries=0, enable_rate_limiting=False,
    )
    adapter.load()
    return adapter


def _together_provider(url: str) -> Any:
    from effgen.models.together_adapter import TogetherAdapter

    os.environ["TOGETHER_BASE_URL"] = f"{url}/v1"
    adapter = TogetherAdapter(
        model_name="Qwen/Qwen3.5-9B", api_key=SENTINEL_CREDENTIAL,
        timeout=int(CLIENT_TIMEOUT_S), max_retries=0, enable_rate_limiting=False,
    )
    adapter.load()
    return adapter


def _cerebras_provider(url: str) -> Any:
    from effgen.models.cerebras_adapter import CerebrasAdapter

    os.environ["CEREBRAS_BASE_URL"] = url
    adapter = CerebrasAdapter(
        model_name="gpt-oss-120b", api_key=SENTINEL_CREDENTIAL,
        timeout=int(CLIENT_TIMEOUT_S), max_retries=0, enable_rate_limiting=False,
    )
    adapter.load()
    return adapter


def _anthropic_provider(url: str) -> Any:
    from effgen.models.anthropic_adapter import AnthropicAdapter

    os.environ["ANTHROPIC_BASE_URL"] = url
    adapter = AnthropicAdapter(
        model_name="claude-opus-4-7", api_key=SENTINEL_CREDENTIAL,
        timeout=int(CLIENT_TIMEOUT_S), max_retries=0,
    )
    adapter.load()
    return adapter


def _gemini_provider(url: str) -> Any:
    from effgen.models.gemini_adapter import GeminiAdapter

    os.environ["GOOGLE_GEMINI_BASE_URL"] = url
    adapter = GeminiAdapter(
        model_name="gemini-3.1-flash-lite", api_key=SENTINEL_CREDENTIAL,
        timeout=CLIENT_TIMEOUT_S, max_retries=0,
    )
    adapter.load()
    return adapter


def _fireworks_provider(url: str) -> Any:
    from effgen.models.fireworks_adapter import FireworksAdapter

    adapter = FireworksAdapter(
        model_name="accounts/fireworks/models/gpt-oss-120b",
        api_key=SENTINEL_CREDENTIAL, timeout=int(CLIENT_TIMEOUT_S), max_retries=1,
        enable_rate_limiting=False, warn_unknown_model=False,
    )
    adapter.load()
    # The Fireworks SDK builds its request URL from the client's base_url at
    # call time, so re-pointing the loaded client is enough.
    adapter._client._client_v1.base_url = f"{url}/inference/v1"
    return adapter


def _hf_provider(url: str) -> Any:
    from effgen.models.hf_inference_adapter import HFInferenceAdapter

    adapter = HFInferenceAdapter(
        model_name="Qwen/Qwen2.5-7B-Instruct", api_token=SENTINEL_CREDENTIAL,
        endpoint_url=url, timeout=CLIENT_TIMEOUT_S, max_retries=1,
        enable_rate_limiting=False, warn_unknown_model=False,
    )
    adapter.load()
    return adapter


def _replicate_provider(url: str) -> Any:
    from effgen.models.replicate_adapter import ReplicateAdapter

    adapter = ReplicateAdapter(
        model_name="meta/meta-llama-3-8b-instruct", api_token=SENTINEL_CREDENTIAL,
        timeout=CLIENT_TIMEOUT_S, poll_interval=0.05, max_retries=1,
        enable_rate_limiting=False, warn_unknown_model=False,
    )
    adapter.load()
    adapter._client._base_url = url
    return adapter


def _plain(cls_path: str, **kwargs: Any) -> Callable[[], Any]:
    """Build an adapter with no credential and load it."""

    def build() -> Any:
        module_name, _, cls_name = cls_path.rpartition(".")
        import importlib

        cls = getattr(importlib.import_module(module_name), cls_name)
        adapter = cls(**kwargs)
        adapter.load()
        return adapter

    return build


PROVIDERS: list[ProviderSpec] = [
    ProviderSpec(
        "openai", "openai", ("OPENAI_API_KEY",), _openai_provider,
        _plain("effgen.models.openai_adapter.OpenAIAdapter",
               model_name="gpt-5-nano", timeout=int(CLIENT_TIMEOUT_S), max_retries=0),
        "OPENAI_API_KEY", "openai",
    ),
    ProviderSpec(
        "groq", "openai", ("GROQ_API_KEY",), _groq_provider,
        _plain("effgen.models.groq_adapter.GroqAdapter",
               model_name="openai/gpt-oss-20b", enable_rate_limiting=False),
        "GROQ_API_KEY", "groq",
    ),
    ProviderSpec(
        "together", "openai", ("TOGETHER_API_KEY",), _together_provider,
        _plain("effgen.models.together_adapter.TogetherAdapter",
               model_name="Qwen/Qwen3.5-9B", enable_rate_limiting=False),
        "TOGETHER_API_KEY", "together",
    ),
    ProviderSpec(
        "cerebras", "openai", ("CEREBRAS_API_KEY",), _cerebras_provider,
        _plain("effgen.models.cerebras_adapter.CerebrasAdapter",
               model_name="gpt-oss-120b", enable_rate_limiting=False),
        "CEREBRAS_API_KEY", "cerebras",
    ),
    ProviderSpec(
        "fireworks", "openai", ("FIREWORKS_API_KEY",), _fireworks_provider,
        _plain("effgen.models.fireworks_adapter.FireworksAdapter",
               model_name="accounts/fireworks/models/gpt-oss-120b",
               enable_rate_limiting=False, warn_unknown_model=False),
        "FIREWORKS_API_KEY", "fireworks",
    ),
    ProviderSpec(
        "hf", "openai", ("HF_TOKEN", "HUGGINGFACE_API_KEY"), _hf_provider,
        _plain("effgen.models.hf_inference_adapter.HFInferenceAdapter",
               model_name="Qwen/Qwen2.5-7B-Instruct", enable_rate_limiting=False,
               warn_unknown_model=False),
        "HF_TOKEN", "huggingface_hub",
    ),
    ProviderSpec(
        "anthropic", "anthropic", ("ANTHROPIC_API_KEY",), _anthropic_provider,
        _plain("effgen.models.anthropic_adapter.AnthropicAdapter",
               model_name="claude-opus-4-7", timeout=int(CLIENT_TIMEOUT_S), max_retries=0),
        "ANTHROPIC_API_KEY", "anthropic",
    ),
    ProviderSpec(
        "gemini", "gemini", ("GOOGLE_API_KEY", "GEMINI_API_KEY"), _gemini_provider,
        _plain("effgen.models.gemini_adapter.GeminiAdapter",
               model_name="gemini-3.1-flash-lite"),
        "GOOGLE_API_KEY", "google.genai",
    ),
    ProviderSpec(
        "replicate", "replicate", ("REPLICATE_API_TOKEN",), _replicate_provider,
        _plain("effgen.models.replicate_adapter.ReplicateAdapter",
               model_name="meta/meta-llama-3-8b-instruct",
               enable_rate_limiting=False, warn_unknown_model=False),
        "REPLICATE_API_TOKEN", "replicate",
        streaming=False,
        skips={
            "truncation": (
                "the predictions API returns no finish reason, so a prediction "
                "that stopped at the token cap is indistinguishable from one "
                "that finished"
            ),
            "mid_stream_abort": (
                "streaming reads a server-sent event source off a prediction "
                "that has to be created and polled first; covered live instead"
            ),
        },
    ),
]

PROVIDERS_BY_NAME: dict[str, ProviderSpec] = {p.name: p for p in PROVIDERS}


# ---------------------------------------------------------------------------
# Failure table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureSpec:
    """One way a provider call goes wrong, and what has to be reported.

    Args:
        name: Failure class, as it appears in the report.
        mode: The :class:`FaultServer` mode that produces it, or ``""`` when the
            failure needs no server (an absent credential).
        category: The ``classify_provider_error`` category the adapter's error
            must land in. A tuple accepts more than one, for classes providers
            genuinely word differently.
        retryable: What ``ErrorClass.retryable`` must be.
        should_retry: What ``ErrorClass.should_retry`` must be — retryable *or*
            rate limited, i.e. whether the caller may try again at all.
        streaming: Drive ``generate_stream`` instead of ``generate``.
        returns: The cell expects a value rather than an exception (truncation
            is a successful call that has to declare itself incomplete).
        hint: Text the reported error must contain, so the caller is told what
            to do rather than only what happened.
    """

    name: str
    mode: str
    category: tuple[str, ...]
    retryable: bool
    should_retry: bool
    streaming: bool = False
    returns: bool = False
    hint: str = ""


FAILURES: list[FailureSpec] = [
    FailureSpec("auth_rejected", "auth_rejected", ("auth",), False, False),
    FailureSpec("credential_absent", "", ("auth",), False, False),
    FailureSpec("rate_limit", "rate_limit", ("rate_limited",), True, True),
    FailureSpec("timeout", "timeout", ("timeout", "transient"), True, True),
    FailureSpec("truncation", "truncation", (), False, False, returns=True),
    FailureSpec("malformed_body", "malformed_body", ("transient", "unknown", "invalid_request"), True, True),
    FailureSpec("mid_stream_abort", "mid_stream_abort", ("transient", "unknown", "timeout"), True, True, streaming=True),
    FailureSpec("connection_reset", "connection_reset", ("transient", "unknown", "timeout"), True, True),
    FailureSpec("oversized_request", "oversized_request", ("invalid_request",), False, False),
]

FAILURES_BY_NAME: dict[str, FailureSpec] = {f.name: f for f in FAILURES}


# ---------------------------------------------------------------------------
# Running one cell
# ---------------------------------------------------------------------------


@dataclass
class CellOutcome:
    """What one provider × failure cell actually produced."""

    provider: str
    failure: str
    raised: str = ""
    category: str = ""
    retryable: bool | None = None
    should_retry: bool | None = None
    elapsed_s: float = 0.0
    credential_echoed: bool = False
    returned_value: bool = False
    cap_reached_the_wire: bool | None = None
    truncated: bool | None = None
    finish_reason: str = ""
    text: str = ""
    message: str = ""
    skipped: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every assertion for this cell held."""
        return not self.errors

    def row(self) -> str:
        """One line of the matrix table."""
        if self.skipped:
            mark, detail = "n/a ", f"not applicable: {self.skipped}"
        else:
            mark = "ok  " if self.ok else "FAIL"
            if self.returned_value:
                detail = (
                    f"returned finish={self.finish_reason!r} "
                    f"truncated={self.truncated} text={self.text[:24]!r}"
                )
            else:
                detail = (
                    f"{self.raised} cat={self.category} retryable={self.retryable} "
                    f"should_retry={self.should_retry}"
                )
        line = (
            f"  {mark} {self.provider:<10} {self.failure:<18} "
            f"{self.elapsed_s:6.2f}s  {detail}"
        )
        if self.errors:
            line += "\n" + "\n".join(f"        -> {e}" for e in self.errors)
        return line


@contextmanager
def _credential_env(spec: ProviderSpec, value: str | None) -> Iterator[None]:
    """Set (or clear) every credential variable *spec* reads, then restore.

    Also clears the base-URL overrides the provider rows set, so a run cannot
    leak a dead fault-server address into the next one.
    """
    touched = [
        *spec.key_envs,
        "OPENAI_BASE_URL", "GROQ_BASE_URL", "TOGETHER_BASE_URL",
        "CEREBRAS_BASE_URL", "ANTHROPIC_BASE_URL", "GOOGLE_GEMINI_BASE_URL",
    ]
    previous = {name: os.environ.get(name) for name in touched}
    try:
        for name in touched:
            os.environ.pop(name, None)
        if value is not None:
            for name in spec.key_envs:
                os.environ[name] = value
        yield
    finally:
        for name, was in previous.items():
            if was is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = was


#: The per-call output-token cap every cell asks for. Distinctive enough that
#: finding it in the request the adapter sent is conclusive.
REQUESTED_MAX_TOKENS = 37


def _drive(adapter: Any, failure: FailureSpec) -> Any:
    """Make the call the cell is about and return whatever it produces."""
    if failure.streaming:
        return "".join(
            adapter.generate_stream("Say hello.", max_tokens=REQUESTED_MAX_TOKENS)
        )
    return adapter.generate("Say hello.", max_tokens=REQUESTED_MAX_TOKENS)


def _cap_in_request(spec: ProviderSpec, body: str) -> bool:
    """Whether the caller's output-token cap reached the wire."""
    import json as _json

    dialect = DIALECTS[spec.dialect]
    try:
        payload = _json.loads(body)
    except ValueError:
        return False

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in dialect.cap_keys and value == REQUESTED_MAX_TOKENS:
                    return True
                if walk(value):
                    return True
        elif isinstance(node, list):
            return any(walk(item) for item in node)
        return False

    return walk(payload)


def run_cell(spec: ProviderSpec, failure: FailureSpec) -> CellOutcome:
    """Drive one provider through one failure class and check every rule.

    Every cell asserts the same five things: the call reported a typed,
    classified error (or, for truncation, a result that declares itself
    incomplete); the retry verdict matches the failure class; the call returned
    within the hang limit; the credential never appears in what the caller sees;
    and the call never quietly succeeded.
    """
    from effgen.models.errors import classify_provider_error

    outcome = CellOutcome(spec.name, failure.name)
    if failure.name in spec.skips:
        outcome.skipped = spec.skips[failure.name]
        return outcome
    absent = spec.sdk_missing()
    if absent:
        outcome.skipped = absent
        return outcome

    server: FaultServer | None = None
    adapter: Any = None
    started = time.monotonic()
    try:
        if failure.mode:
            server = FaultServer(failure.mode, spec.dialect).start()
            with _credential_env(spec, SENTINEL_CREDENTIAL):
                adapter = spec.build(server.url)
                started = time.monotonic()
                try:
                    value = _drive(adapter, failure)
                except Exception as exc:  # noqa: BLE001 - the cell is about it
                    _record_exception(outcome, exc, classify_provider_error)
                else:
                    _record_value(outcome, value)
                if server.requests:
                    outcome.cap_reached_the_wire = _cap_in_request(
                        spec, server.requests[-1][1]
                    )
        else:
            with _credential_env(spec, None):
                started = time.monotonic()
                try:
                    adapter = spec.build_without_credential()
                    value = _drive(adapter, failure)
                except Exception as exc:  # noqa: BLE001 - the cell is about it
                    _record_exception(outcome, exc, classify_provider_error)
                else:
                    _record_value(outcome, value)
    finally:
        outcome.elapsed_s = time.monotonic() - started
        if adapter is not None:
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001 - teardown must not mask the cell
                pass
        if server is not None:
            server.stop()

    _check(outcome, spec, failure)
    return outcome


def _record_exception(outcome: CellOutcome, exc: Exception, classify: Any) -> None:
    cls = classify(exc)
    outcome.raised = type(exc).__name__
    outcome.category = cls.category
    outcome.retryable = cls.retryable
    outcome.should_retry = cls.should_retry
    outcome.message = str(exc)
    outcome.credential_echoed = _credential_visible(exc)


def _record_value(outcome: CellOutcome, value: Any) -> None:
    outcome.returned_value = True
    text = value if isinstance(value, str) else getattr(value, "text", "")
    outcome.text = text or ""
    outcome.finish_reason = getattr(value, "finish_reason", "") or ""
    metadata = getattr(value, "metadata", None) or {}
    outcome.truncated = metadata.get("truncated")
    outcome.credential_echoed = SENTINEL_CREDENTIAL in repr(value)


def _credential_visible(exc: BaseException) -> bool:
    """Whether the sentinel survives anywhere the caller can reach it.

    Checks the exception text, its repr, every string attribute it carries, and
    the whole ``__cause__``/``__context__`` chain — a redacted message with a
    raw SDK error underneath still hands the key to anyone printing a traceback.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if SENTINEL_CREDENTIAL in str(node) or SENTINEL_CREDENTIAL in repr(node):
            return True
        for value in vars(node).values():
            if isinstance(value, str) and SENTINEL_CREDENTIAL in value:
                return True
            if isinstance(value, dict) and SENTINEL_CREDENTIAL in repr(value):
                return True
        node = node.__cause__ or node.__context__
    return False


def _check(outcome: CellOutcome, spec: ProviderSpec, failure: FailureSpec) -> None:
    """Apply every rule the cell has to satisfy, collecting each breach."""
    if failure.returns:
        if not outcome.returned_value:
            outcome.errors.append(
                f"a truncated answer must be returned and flagged, not raised "
                f"({outcome.raised}: {outcome.message[:160]})"
            )
        else:
            if outcome.finish_reason != "length":
                outcome.errors.append(
                    f"finish_reason is {outcome.finish_reason!r}, expected 'length'"
                )
            if outcome.truncated is not True:
                outcome.errors.append(
                    f"metadata['truncated'] is {outcome.truncated!r}, expected True"
                )
            if outcome.cap_reached_the_wire is False:
                outcome.errors.append(
                    f"the per-call max_tokens={REQUESTED_MAX_TOKENS} never reached "
                    f"the request, so the caller's cap was dropped"
                )
    else:
        if outcome.returned_value:
            outcome.errors.append(
                f"the call succeeded silently and returned {outcome.text[:60]!r}"
            )
        else:
            if outcome.category not in failure.category:
                outcome.errors.append(
                    f"category {outcome.category!r} is not one of {failure.category}"
                )
            if outcome.retryable is not failure.retryable:
                outcome.errors.append(
                    f"retryable is {outcome.retryable!r}, expected {failure.retryable!r}"
                )
            if outcome.should_retry is not failure.should_retry:
                outcome.errors.append(
                    f"should_retry is {outcome.should_retry!r}, "
                    f"expected {failure.should_retry!r}"
                )
            if failure.name == "credential_absent" and (
                spec.missing_credential_hint not in outcome.message
            ):
                outcome.errors.append(
                    f"the error does not name {spec.missing_credential_hint}: "
                    f"{outcome.message[:200]}"
                )
    if outcome.credential_echoed:
        outcome.errors.append("the submitted credential is visible to the caller")
    if outcome.elapsed_s > HANG_LIMIT_S:
        outcome.errors.append(
            f"took {outcome.elapsed_s:.1f}s, over the {HANG_LIMIT_S:.0f}s hang limit"
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class MatrixReport:
    """Every cell that was run, printable as one table."""

    cells: list[CellOutcome] = field(default_factory=list)

    def add(self, outcome: CellOutcome) -> None:
        """Record one cell."""
        self.cells.append(outcome)

    @property
    def failures(self) -> list[CellOutcome]:
        """Cells that broke a rule."""
        return [c for c in self.cells if not c.skipped and not c.ok]

    @property
    def skipped(self) -> list[CellOutcome]:
        """Cells a provider cannot express, each with its reason."""
        return [c for c in self.cells if c.skipped]

    def table(self, title: str = "failure matrix") -> str:
        """The whole matrix, one line per cell."""
        asserted = len(self.cells) - len(self.skipped)
        head = (
            f"[{title}] {asserted - len(self.failures)}/{asserted} cells green, "
            f"{len(self.skipped)} not applicable"
        )
        return "\n".join([head, *(c.row() for c in self.cells)])

    def assert_all_green(self, title: str = "failure matrix") -> None:
        """Fail with the whole table when any cell broke a rule."""
        if self.failures:
            raise AssertionError(
                f"{len(self.failures)} cells failed:\n{self.table(title)}"
            )
