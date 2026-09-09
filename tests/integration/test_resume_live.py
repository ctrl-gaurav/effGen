"""Live session-continuity and checkpoint-resume tests (skipped without a key).

These prove the durability surfaces against a real model: the model actually
*remembers* across a session round-trip, and a checkpointed task resumes and
completes. Bulk runs use the cheapest Groq model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests._harness.provider_unavailable import skip_if_provider_refused

# Load keys from the repo .env (canonical) then the per-user one, without
# overriding anything already in the environment.
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


def _agent(session_id=None, tools=None, **cfg):
    from effgen import Agent, AgentConfig

    return Agent(
        AgentConfig(name="live", model=GROQ_MODEL, provider="groq", tools=tools or [], **cfg),
        session_id=session_id,
    )


def test_session_continuity_live(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path))

    a1 = _agent(session_id="s-live")
    try:
        a1.run("Remember: my secret number is 4271. Reply OK.")
    finally:
        a1.close()

    # A brand-new agent (separate object) must recall the fact from the session.
    a2 = _agent(session_id="s-live")
    try:
        out = a2.run("What is my secret number?").output
    finally:
        a2.close()
    assert "4271" in out


def test_session_continuity_with_tools_live(tmp_path, monkeypatch):
    """A session must retain context even when a tool is attached.

    The CLI's default agent always carries a calculator tool, which routes the
    run through the native/hybrid tool-calling path. That path must still inject
    the prior conversation turns or multi-turn memory is silently lost the moment
    any tool is present (regression guard).
    """
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path))
    from effgen.tools import get_registry

    reg = get_registry()
    reg.discover_builtin_tools()
    calc = reg.get_tool_sync("calculator")

    a1 = _agent(session_id="s-tools", tools=[calc])
    try:
        a1.run("Remember: my secret number is 4271. Reply OK.")
    finally:
        a1.close()

    a2 = _agent(session_id="s-tools", tools=[calc])
    try:
        out = a2.run("What is my secret number? Reply with just the number.").output
    finally:
        a2.close()
    assert "4271" in out


def test_checkpoint_resume_live(tmp_path):
    ckpt = str(tmp_path / "ckpt")

    a1 = _agent()
    try:
        r1 = a1.run(
            "What is the capital of Japan? Answer in one word.",
            checkpoint_dir=ckpt,
        )
        skip_if_provider_refused(r1)
        assert r1.success
    finally:
        a1.close()

    # The checkpoint must have recorded the model so resume reuses it.
    from effgen.core.checkpoint import CheckpointManager

    latest = CheckpointManager(ckpt).load_latest()
    assert latest.model == GROQ_MODEL

    a2 = _agent()
    try:
        r2 = a2.resume(checkpoint_dir=ckpt)
        skip_if_provider_refused(r2)
        assert r2.success
        assert "tokyo" in r2.output.lower()
    finally:
        a2.close()
