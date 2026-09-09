"""Live tests for reusing one Agent across concurrent/repeated .run() calls —
exactly what BatchRunner does for every batch row (skipped without a key).

Cost/token accounting, sub-agent depth, retrieved citations, and the
execution tracker are per-call state, not instance state: two calls sharing
one Agent (concurrently, via threads, or one after another) must never read
or write each other's data. Also covers the `effgen batch` CLI surfaces that
depend on this: `--json` row output and the no-output-persisted warning.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests._harness.provider_unavailable import assert_cli_succeeded

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)

GROQ_MODEL = "openai/gpt-oss-20b"


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.api,
    pytest.mark.skipif(not _has_groq(), reason="SKIPPED: GROQ_API_KEY not set"),
]


def _distinct_length_prompts(n: int) -> list[str]:
    # Each prompt tokenizes to a different length, so two rows can only ever
    # report the same total_tokens by actually sharing state, never by
    # coincidence (unlike same-length trivia questions).
    return [
        "Answer briefly. " + " ".join(f"pad{j}" for j in range(i))
        + f" What is the capital of France? (marker {i})"
        for i in range(n)
    ]


def test_concurrent_run_calls_on_shared_agent_do_not_share_cost_state():
    from effgen import Agent
    from effgen.core.agent import AgentConfig

    agent = Agent(AgentConfig(name="race-test", model=GROQ_MODEL, provider="groq"))
    try:
        prompts = _distinct_length_prompts(10)

        async def one(q: str):
            return await asyncio.to_thread(agent.run, q, max_tokens=200)

        async def main():
            return await asyncio.gather(*[one(q) for q in prompts])

        responses = asyncio.run(main())
    finally:
        agent.close()

    # prompt_tokens is deterministic for a given prompt (unlike total_tokens,
    # which also depends on how long the model's own answer happens to be) —
    # since every prompt has a distinct, strictly increasing length, distinct
    # prompt_tokens is a non-flaky isolation check: any repeat here can only
    # come from two calls reading/writing the same shared accumulator.
    prompt_tokens = [r.metadata.get("prompt_tokens") for r in responses]
    assert all(t is not None for t in prompt_tokens), "every concurrent call must report its own tokens"
    assert len(set(prompt_tokens)) == len(prompt_tokens), (
        f"concurrent calls on a shared Agent reported overlapping prompt_tokens: {prompt_tokens}"
    )


def test_sequential_run_calls_on_shared_agent_do_not_leak_execution_trace():
    from effgen import Agent
    from effgen.core.agent import AgentConfig

    agent = Agent(AgentConfig(name="trace-test", model=GROQ_MODEL, provider="groq"))
    try:
        r1 = agent.run("Say hello.")
        r2 = agent.run("Say goodbye.")
    finally:
        agent.close()

    assert r1.execution_trace
    assert r2.execution_trace
    assert not any("hello" in str(e).lower() for e in r2.execution_trace), (
        "second run's execution_trace still carries the first run's events"
    )


def _run_batch_cli(*extra_args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "effgen.cli", "batch", *extra_args],
        cwd=cwd, capture_output=True, text=True, timeout=90,
    )


def test_batch_json_flag_emits_a_row_per_query(tmp_path):
    input_path = tmp_path / "tickets.jsonl"
    queries = _distinct_length_prompts(4)
    input_path.write_text("\n".join(json.dumps({"query": q}) for q in queries) + "\n")

    proc = _run_batch_cli(
        "-i", str(input_path), "-m", f"groq:{GROQ_MODEL}",
        "--concurrency", "4", "--json", "-q",
        cwd=Path(__file__).resolve().parents[2],
    )
    assert_cli_succeeded(proc, "effgen batch")
    data = json.loads(proc.stdout)
    assert data["total"] == 4
    assert len(data["rows"]) == 4
    indices = sorted(r["index"] for r in data["rows"])
    assert indices == [0, 1, 2, 3]
    for row in data["rows"]:
        assert "total_tokens" in row
        assert "cost_usd" in row
    prompt_tokens = [row["prompt_tokens"] for row in data["rows"]]
    assert len(set(prompt_tokens)) == len(prompt_tokens), (
        f"batch --json rows share prompt_tokens: {prompt_tokens}"
    )


def test_batch_warns_when_no_output_and_no_json(tmp_path):
    input_path = tmp_path / "tickets.jsonl"
    input_path.write_text(json.dumps({"query": "Say OK."}) + "\n")

    proc = _run_batch_cli(
        "-i", str(input_path), "-m", f"groq:{GROQ_MODEL}",
        cwd=Path(__file__).resolve().parents[2],
    )
    assert_cli_succeeded(proc, "effgen batch")
    assert "discarded" in proc.stdout.lower()
