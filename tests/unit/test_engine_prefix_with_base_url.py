"""The id an endpoint is asked for, when the caller names a base_url.

``openai:<id>`` with a ``base_url`` selects the OpenAI-compatible adapter — the
OpenAI adapter pointed at the caller's server — so the ``openai:`` prefix names
the adapter that makes the call, and the server is asked for ``<id>``. Which
adapter a prefix names is read from the prefix and the endpoint, never from the
rest of the id: a cloud id carries slashes of its own and keeps them.

Every case reads the ``model`` field of the request the endpoint received, not
just the adapter's attribute, because the request is what the server answers.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SERVED = "org-name/served-model-7b"


class _Endpoint(BaseHTTPRequestHandler):
    """Answers only the ids it serves, the way a vLLM server does."""

    serves = {SERVED, "openai/gpt-oss-20b"}
    seen: list[str] = []

    def log_message(self, *args) -> None:
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._json(200, {"object": "list", "data": [{"id": m} for m in self.serves]})

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append(request.get("model"))
        if request.get("model") not in self.serves:
            self._json(404, {"object": "error", "type": "NotFoundError", "code": 404,
                             "message": f"The model `{request.get('model')}` does not exist."})
            return
        if request.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            chunk = {"id": "x", "object": "chat.completion.chunk", "model": request["model"],
                     "choices": [{"index": 0, "delta": {"content": "Final Answer: 4"}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            self.close_connection = True
            return
        self._json(200, {"id": "x", "object": "chat.completion", "model": request["model"],
                         "choices": [{"index": 0, "finish_reason": "stop", "message": {
                             "role": "assistant", "content": "Final Answer: 4"}}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                                   "total_tokens": 8}})


@pytest.fixture()
def endpoint():
    _Endpoint.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _agent(model_id: str, base_url: str):
    from effgen import Agent
    from effgen.core.agent_config import AgentConfig

    return Agent(AgentConfig(model=model_id, base_url=base_url, api_key="x",
                             max_iterations=1, max_tokens=16, temperature=0.0))


def test_an_openai_prefixed_id_with_a_base_url_asks_the_server_for_the_id(endpoint) -> None:
    agent = _agent(f"openai:{SERVED}", endpoint)
    answer = agent.run("What is 2 + 2?")
    assert _Endpoint.seen and set(_Endpoint.seen) == {SERVED}
    assert "4" in str(answer)
    assert agent.model.model_name == SERVED


def test_a_streamed_run_asks_the_server_for_the_same_id(endpoint) -> None:
    agent = _agent(f"openai:{SERVED}", endpoint)
    text = "".join(str(piece) for piece in agent.stream("What is 2 + 2?"))
    assert set(_Endpoint.seen) == {SERVED}
    assert "4" in text


def test_load_model_with_a_base_url_asks_the_server_for_the_id(endpoint) -> None:
    from effgen.models import load_model

    model = load_model(f"openai:{SERVED}", base_url=endpoint, api_key="x")
    model.generate("What is 2 + 2?")
    assert _Endpoint.seen == [SERVED]


def test_a_bare_id_with_a_base_url_is_sent_unchanged(endpoint) -> None:
    """The documented form already worked and still does."""
    _agent(SERVED, endpoint).run("What is 2 + 2?")
    assert set(_Endpoint.seen) == {SERVED}


def test_a_cloud_id_with_a_slash_behind_a_base_url_keeps_its_slash(endpoint) -> None:
    _agent("openai:openai/gpt-oss-20b", endpoint).run("What is 2 + 2?")
    assert set(_Endpoint.seen) == {"openai/gpt-oss-20b"}


def test_a_prefix_naming_another_provider_is_left_for_the_server_to_refuse(endpoint) -> None:
    """Only the prefix that names this adapter is dropped."""
    from effgen.models.errors import ModelNotFoundError

    agent = _agent("together:meta-llama/Llama-3.3-70B-Instruct-Turbo", endpoint)
    with pytest.raises(ModelNotFoundError):
        agent.run("What is 2 + 2?")
    assert set(_Endpoint.seen) == {"together:meta-llama/Llama-3.3-70B-Instruct-Turbo"}


@pytest.mark.parametrize(
    ("model_id", "adapter", "wire_id", "sdk"),
    [
        ("groq:openai/gpt-oss-20b", "GroqAdapter", "openai/gpt-oss-20b", None),
        ("together:meta-llama/Llama-3.3-70B-Instruct-Turbo", "TogetherAdapter",
         "meta-llama/Llama-3.3-70B-Instruct-Turbo", None),
        # The Fireworks adapter builds its SDK client when it loads.
        ("fireworks:accounts/fireworks/models/llama-v3p1-8b-instruct", "FireworksAdapter",
         "accounts/fireworks/models/llama-v3p1-8b-instruct", "fireworks.client"),
    ],
)
def test_a_cloud_id_with_a_slash_and_no_base_url_goes_to_its_provider(
    model_id, adapter, wire_id, sdk, monkeypatch
) -> None:
    """A slash in a cloud id is not read as a local model."""
    if sdk is not None:
        pytest.importorskip(sdk)
    from effgen import Agent
    from effgen.core.agent_config import AgentConfig

    for name in ("EFFGEN_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    agent = Agent(AgentConfig(model=model_id, api_key="not-a-real-key"))
    assert type(agent.model).__name__ == adapter
    assert agent.model.model_name == wire_id
