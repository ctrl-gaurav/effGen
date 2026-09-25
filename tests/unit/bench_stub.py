"""An OpenAI-protocol endpoint that answers bench tasks it was given the answers to.

Not a model: a measuring standard for the bench command. It recognises a task
by its input text, asks for the ``calculator`` tool once when tools are offered
(so tool calls are counted), then answers ``Answer: <expected>``. Token usage is
a fixed function of the request, and a task whose id is listed in ``wrong``
gets a wrong answer, so a test knows every number the table must show.

Run standalone for a command-line smoke::

    python bench_stub.py suite.yaml [more.yaml ...] --port 8765
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_ID = "bench-stub"


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    answers: dict[str, tuple[str, Any]] = {}
    wrong: set[str] = set()
    delay_s: float = 0.0
    requests: list[dict[str, Any]] = []
    lock = threading.Lock()

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        with self.lock:
            self.requests.append(request)
        if self.delay_s:
            time.sleep(self.delay_s)
        messages = request.get("messages") or []
        everything = "\n".join(_text(m) for m in messages)
        task_id, expected = None, None
        for text, (tid, value) in self.answers.items():
            if text in everything:
                task_id, expected = tid, value
                break
        if task_id in self.wrong:
            expected = "-1"
        # A tool result comes back as a tool message or, when the run carries its own
        # steps as text, as an "Observation:" line.
        used_tool = any(m.get("role") == "tool" for m in messages) or "Observation:" in everything
        tools = request.get("tools") or []
        if tools and not used_tool and task_id is not None:
            message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{task_id}", "type": "function",
                "function": {"name": "calculator",
                             "arguments": json.dumps({"expression": f"{expected} * 1"})},
            }]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": f"Answer: {expected}"}
            finish = "stop"
        prompt_tokens = len(everything) // 4 + 1
        self._send({
            "id": "stub", "object": "chat.completion", "created": 0, "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 9,
                      "total_tokens": prompt_tokens + 9,
                      "prompt_tokens_details": {"cached_tokens": prompt_tokens // 2}},
        })


def serve(tasks: list[dict[str, Any]], *, wrong: set[str] | None = None,
          port: int = 0, delay_s: float = 0.0) -> tuple[ThreadingHTTPServer, str]:
    """Start the stub on a thread; return the server and its ``/v1`` base URL."""
    handler = type("Handler", (_Handler,), {
        "answers": {t["input"]: (t["id"], t["expected"]) for t in tasks},
        "wrong": set(wrong or ()),
        "delay_s": delay_s,
        "requests": [],
        "lock": threading.Lock(),
    })
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="+", help="suite files whose inline tasks it answers")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import yaml

    tasks = []
    for suite in args.suites:
        with open(suite, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        tasks += [{"id": t.get("id"), "input": t["input"], "expected": t.get("expected")}
                  for t in data["tasks"]]
    server, url = serve(tasks, port=args.port)
    print(f"stub serving {len(tasks)} tasks at {url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
