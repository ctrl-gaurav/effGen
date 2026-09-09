"""``effgen workflow run`` — a node's YAML ``agent:`` value resolves to a real
model, not a silent local fallback.

Drives the real CLI subprocess against cheap cloud models to prove a node's
``agent: <provider>:<model>`` value is actually used to pick the model (a real
priced API call happens, `cost_usd` is nonzero) rather than being silently
replaced by the hardcoded local Transformers default. No mocking of model
behavior. Skipped when the relevant keys are absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)


def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _run_cli(*args, timeout=90):
    cmd = [sys.executable, "-m", "effgen.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.mark.skipif(not (_has_openai() and _has_groq()),
                     reason="SKIPPED: OPENAI_API_KEY/GROQ_API_KEY not set")
def test_node_agent_field_resolves_to_named_model_live(tmp_path):
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        "workflow:\n"
        "  name: agent_field_resolves\n"
        "  nodes:\n"
        "    - id: step1\n"
        "      agent: openai:gpt-5-nano\n"
        "      task: 'Reply with exactly one word: hello'\n"
        "    - id: step2\n"
        "      agent: groq:openai/gpt-oss-20b\n"
        "      depends_on: [step1]\n"
        "      task: 'Reply with exactly one word: world'\n"
    )
    result = _run_cli("workflow", "run", str(wf), "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["success"] is True
    # A real priced OpenAI call was made — a free local fallback reports 0.
    assert data["metadata"]["cost_usd"] > 0
    node_ids = {n["id"] for n in data["node_results"]}
    assert node_ids == {"step1", "step2"}


@pytest.mark.skipif(not _has_openai(), reason="SKIPPED: OPENAI_API_KEY not set")
def test_node_agent_field_that_is_not_a_model_fails_loudly(tmp_path):
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        "workflow:\n"
        "  name: bad_agent_field\n"
        "  nodes:\n"
        "    - id: step1\n"
        "      agent: not-a-real-model-id-xyz\n"
        "      task: 'Say hi'\n"
    )
    result = _run_cli("workflow", "run", str(wf), "--json")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["success"] is False
    assert "step1" in data["error"]
    assert "not-a-real-model-id-xyz" in data["error"]
