"""Tests for the ``effgen loadtest`` report rendering and CLI wiring.

Covers the drain note (requested vs. wall-clock duration when in-flight
requests are drained past the requested window), the per-category error
breakdown line (both read straight off ``LoadReport.to_dict()``), argument
validation for ``--url`` server mode, and ``_ServerTarget``'s HTTP behavior
against a real in-process ASGI app (an ``httpx.ASGITransport``, not a mock of
provider behavior -- it exercises the real client/server request path).
"""

from __future__ import annotations

import argparse

import pytest

from effgen.cli.loadtest import (
    _print_report,
    _ServerTarget,
    add_loadtest_subparser,
    run_loadtest_command,
)
from effgen.tools.loadgen import LoadReport

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - fastapi optional
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_loadtest_subparser(subparsers)
    return parser.parse_args(["loadtest", *argv])


def _report(**overrides) -> LoadReport:
    defaults = {
        "scenario": "fixed",
        "concurrency": 3,
        "duration": 10.0,
        "total_requests": 35,
        "successful_requests": 32,
        "failed_requests": 3,
        "error_rate": 0.0857,
        "throughput": 0.98,
        "p50_latency": 0.137,
        "p95_latency": 30.003,
        "p99_latency": 30.019,
        "min_latency": 0.093,
        "max_latency": 30.027,
        "mean_latency": 2.88,
        "stdev_latency": 8.46,
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "requested_duration": 10.0,
        "error_breakdown": {},
    }
    defaults.update(overrides)
    return LoadReport(**defaults)


def test_no_drain_note_when_duration_matches_requested(capsys):
    _print_report(_report(duration=10.0, requested_duration=10.0))
    out = capsys.readouterr().out
    assert "Duration      : 10.0s" in out
    assert "draining" not in out


def test_drain_note_on_overshoot(capsys):
    _print_report(_report(duration=35.636, requested_duration=10.0))
    out = capsys.readouterr().out
    assert "requested 10.0s" in out
    assert "wall 35.6s" in out
    assert "draining in-flight requests" in out


def test_no_drain_note_for_sub_precision_jitter(capsys):
    # A few ms of scheduler jitter must not print a "incl. 0.0s draining" note
    # that would display as zero and contradict itself.
    _print_report(_report(duration=2.003, requested_duration=2.0))
    out = capsys.readouterr().out
    assert "Duration      : 2.0s" in out
    assert "draining" not in out


def test_error_breakdown_line_present_when_failures(capsys):
    _print_report(_report(error_breakdown={"timeout": 2, "rate_limited": 1}))
    out = capsys.readouterr().out
    assert "Error types" in out
    assert "timeout=2" in out
    assert "rate_limited=1" in out


def test_error_breakdown_line_absent_when_no_failures(capsys):
    _print_report(_report(failed_requests=0, error_rate=0.0, error_breakdown={}))
    out = capsys.readouterr().out
    assert "Error types" not in out


# ---------------------------------------------------------------------------
# --url server mode: argument validation
# ---------------------------------------------------------------------------

def test_url_and_provider_are_mutually_exclusive(capsys):
    args = _parse(["--url", "http://127.0.0.1:8000", "--provider", "openai", "--model", "gpt-5-nano"])
    rc = run_loadtest_command(args)
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_url_requires_model(capsys):
    args = _parse(["--url", "http://127.0.0.1:8000"])
    rc = run_loadtest_command(args)
    assert rc == 2
    assert "requires --model" in capsys.readouterr().err


def test_url_mode_does_not_require_provider(capsys):
    # --provider is optional (and unused) in --url mode; only a bad/unreachable
    # URL should surface as request failures, not an argument error.
    args = _parse(
        ["--url", "http://127.0.0.1:1", "--model", "openai:gpt-5-nano", "--duration", "0.2", "--request-timeout", "0.2"]
    )
    rc = run_loadtest_command(args)
    out = capsys.readouterr().out
    assert "Server mode: url=http://127.0.0.1:1  model=openai:gpt-5-nano" in out
    assert rc == 1  # every request fails to connect; not an argument-validation error


# ---------------------------------------------------------------------------
# _ServerTarget: real HTTP behavior against an in-process ASGI app
# ---------------------------------------------------------------------------

@pytest.fixture
def _chat_app():
    pytest.importorskip("fastapi")

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _completions(request: Request):
        auth = request.headers.get("authorization", "")
        if auth != "Bearer good-key":
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "invalid api key", "type": "invalid_request_error",
                                    "param": None, "code": "invalid_api_key"}},
            )
        body = await request.json()
        prompt = body["messages"][-1]["content"]
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"echo: {prompt}"}}],
        }

    return app


async def _drive(target: _ServerTarget, prompt: str = "hi") -> str:
    try:
        return await target(prompt)
    finally:
        await target.aclose()


@pytest.mark.asyncio
async def test_server_target_success(_chat_app):
    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=_chat_app)
    target = _ServerTarget("http://testserver", "openai:gpt-5-nano", "good-key", transport=transport)
    text = await _drive(target, "hello")
    assert text == "echo: hello"


@pytest.mark.asyncio
async def test_server_target_surfaces_auth_failure(_chat_app):
    httpx = pytest.importorskip("httpx")
    from effgen.models.errors import classify_provider_error

    transport = httpx.ASGITransport(app=_chat_app)
    target = _ServerTarget("http://testserver", "openai:gpt-5-nano", "wrong-key", transport=transport)
    with pytest.raises(RuntimeError) as exc_info:
        await _drive(target)
    assert "invalid api key" in str(exc_info.value)
    # Classified the same way a live provider 401 would be -- so a load test
    # against a running server reports "auth", not a bare unknown failure.
    assert classify_provider_error(exc_info.value).category == "auth"
