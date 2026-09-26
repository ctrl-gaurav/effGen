"""Two streams on one model object each keep what they recorded.

A loaded model is routinely shared — one model, many agents on it — and a
stream records its tool calls and its usage for the consumer to read once the
text is done. Those records belong to the stream, not to the model: a second
stream running on the same object at the same time must not replace what the
first one's consumer reads.

The in-process cases interleave two streams by hand, on one thread and on two,
so the order is fixed and the result does not depend on timing. The last case
drives real agents through ``stream()`` against a scripted OpenAI-protocol
endpoint that sends each tool call as many small argument fragments, which is
how a long argument arrives from a served model.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from effgen.models._usage import tool_call_entry
from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
    clear_stream_tool_calls,
    clear_stream_usage,
    get_stream_tool_calls,
    get_stream_usage,
    record_stream_tool_calls,
    record_stream_usage,
)


class _Recorder(BaseModel):
    """Streams two pieces per prompt and records a call and usage named after it."""

    def __init__(self) -> None:
        super().__init__(model_name="recorder", model_type=ModelType.OPENAI)
        self._is_loaded = True

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        return GenerationResult(text=prompt, tokens_used=1, finish_reason="stop",
                                model_name=self.model_name)

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        # The call is recorded from the first piece, as an adapter records it
        # from its first delta; the second piece records nothing.
        clear_stream_tool_calls(self)
        record_stream_tool_calls(self, [tool_call_entry("echo", json.dumps({"v": prompt}))])
        yield prompt[:1]
        yield prompt[1:]
        record_stream_usage(self, len(prompt), 1)

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False


def _argument(model) -> str:
    calls = get_stream_tool_calls(model)
    return json.loads(calls[0]["function"]["arguments"])["v"] if calls else ""


def test_two_streams_pulled_in_turn_on_one_thread_read_their_own_calls() -> None:
    model = _Recorder()
    first, second = model.generate_stream("alpha"), model.generate_stream("bravo")
    next(first)
    next(second)
    # The second stream recorded last; the first stream's consumer, reading
    # straight after pulling its own next piece, reads the first stream's call.
    assert next(first) == "lpha"
    assert _argument(model) == "alpha"
    assert list(second) == ["ravo"]
    assert _argument(model) == "bravo"
    assert list(first) == []
    assert _argument(model) == "alpha"
    assert get_stream_usage(model)["prompt_tokens"] == len("alpha")


def test_streams_on_two_threads_read_their_own_calls_and_usage() -> None:
    model = _Recorder()
    seen: dict[str, tuple[str, int]] = {}
    first_started, second_done = threading.Event(), threading.Event()

    def consume(prompt: str, before_finishing: threading.Event | None,
                after_starting: threading.Event | None) -> None:
        clear_stream_tool_calls(model)
        clear_stream_usage(model)
        stream = model.generate_stream(prompt)
        next(stream)
        if after_starting is not None:
            after_starting.set()
        if before_finishing is not None:
            before_finishing.wait(5)
        list(stream)
        seen[prompt] = (_argument(model), get_stream_usage(model)["prompt_tokens"])

    a = threading.Thread(target=consume, args=("alpha", second_done, first_started))
    a.start()
    first_started.wait(5)
    consume("charlie-long", None, None)
    second_done.set()
    a.join(5)
    assert seen["alpha"] == ("alpha", len("alpha"))
    assert seen["charlie-long"] == ("charlie-long", len("charlie-long"))


def test_a_record_made_outside_a_stream_is_still_read_back() -> None:
    """A caller that records on a model directly still reads it."""
    model = _Recorder()
    record_stream_tool_calls(model, [tool_call_entry("echo", json.dumps({"v": "direct"}))])
    assert _argument(model) == "direct"
    clear_stream_tool_calls(model)
    assert get_stream_tool_calls(model) == []


# --------------------------------------------------------------------------- #
# Real agents on one shared adapter, against a scripted streaming endpoint
# --------------------------------------------------------------------------- #


def _expression(item: int) -> str:
    parts, n = [str(item)], 1
    while len(" + ".join(parts)) < 300:
        parts.append(str(n))
        n += 1
    return " + ".join(parts)


class _Fragmenting(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send(self, obj) -> None:
        data = f"data: {json.dumps(obj)}\n\n".encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["content-length"])))
        messages = request.get("messages") or []
        match = re.search(r"item-(\d+)", json.dumps(messages))
        item = int(match.group(1)) if match else -1
        base = {"id": f"c{item}", "object": "chat.completion.chunk", "model": "stub"}
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        if any(m.get("role") == "tool" for m in messages):
            self._send({**base, "choices": [{"index": 0, "delta": {"content": f"done-{item}"}}]})
            self._send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        else:
            argument = json.dumps({"expression": _expression(item)})
            self._send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": f"call_{item}", "type": "function",
                "function": {"name": "echo_expression", "arguments": ""}}]}}]})
            for i in range(0, len(argument), 7):
                self._send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0, "function": {"arguments": argument[i:i + 7]}}]}}]})
                time.sleep(0.002)
            self._send({**base, "choices": [{"index": 0, "delta": {},
                                             "finish_reason": "tool_calls"}]})
        self._send({**base, "choices": [], "usage": {
            "prompt_tokens": 10 + item, "completion_tokens": 5, "total_tokens": 15 + item}})
        done = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
        self.wfile.flush()


@pytest.fixture()
def fragmenting_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fragmenting)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_concurrent_streamed_agents_on_one_model_each_get_their_own_argument(
    fragmenting_endpoint,
) -> None:
    from effgen import Agent, tool
    from effgen.core.agent_config import AgentConfig
    from effgen.models import load_model

    received: dict[int, list[str]] = {}
    lock = threading.Lock()

    @tool
    def echo_expression(expression: str) -> str:
        """Return the length of the expression it was given.

        Args:
            expression: Any arithmetic expression.
        """
        key = int(re.match(r"\s*(-?\d+)", expression).group(1)) if re.match(
            r"\s*(-?\d+)", expression) else -1
        with lock:
            received.setdefault(key, []).append(expression)
        return f"ok {len(expression)}"

    model = load_model("stub", provider="openai_compatible", base_url=fragmenting_endpoint,
                       api_key="x", context_length=32768)
    usage: dict[int, int] = {}

    def one(item: int) -> str:
        agent = Agent(AgentConfig(name=f"a{item}", model=model, tools=[echo_expression],
                                  max_iterations=3, temperature=0.0))
        text = "".join(str(x) for x in agent.stream(f"Compute item-{item} with the tool."))
        usage[item] = int((agent.last_stream_usage or {}).get("prompt_tokens") or 0)
        return text

    items = list(range(1, 13))
    with ThreadPoolExecutor(max_workers=6) as pool:
        answers = list(pool.map(one, items))

    assert {k: v for k, v in received.items() if k not in items} == {}
    for item in items:
        assert received.get(item) == [_expression(item)], f"item {item}"
    assert all(f"done-{item}" in answer for item, answer in zip(items, answers))
    # Each item's endpoint reports 10 + item prompt tokens on each of its two turns.
    assert usage == {item: 2 * (10 + item) for item in items}
