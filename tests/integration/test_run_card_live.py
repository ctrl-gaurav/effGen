"""Live coverage for `effgen run --card` and the run document's identity fields.

Runs a real model, so it is skipped when no key is present. The card is checked
for the things that make it shareable: it carries the answer and the tool trace,
it references no external asset, and writing it leaves the terminal and
``--json`` output alone.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)

MODEL = "openai/gpt-oss-20b"
PROVIDER = "groq"

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="SKIPPED: GROQ_API_KEY not set"
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "effgen.cli", "run", *args],
        capture_output=True, text=True, timeout=300,
    )


def _skip_if_upstream_hiccup(proc: subprocess.CompletedProcess) -> None:
    """Skip when the provider, not the code under test, ended the run.

    Only a failed command can be an upstream hiccup. A successful run prints its
    own document, whose run ids, token counts and costs contain arbitrary
    digits — matching a status code against those skips a test that passed.
    """
    if proc.returncode == 0:
        return
    combined = (proc.stdout + proc.stderr).lower()
    transient = ("rate limit", "rate_limit_exceeded", "service unavailable",
                 "overloaded")
    status = re.search(r"\b(429|500|502|503|529)\b", combined)
    if status or any(marker in combined for marker in transient):
        pytest.skip(f"Groq transient upstream condition: {combined[-300:]}")


@pytest.mark.integration
@pytest.mark.api
def test_run_card_carries_the_answer_and_the_tool_trace(tmp_path):
    card = tmp_path / "card.html"
    proc = _run(
        "What is 18723 * 4409? Use the calculator tool.",
        "-m", MODEL, "--provider", PROVIDER, "-t", "calculator",
        "--card", str(card),
    )
    _skip_if_upstream_hiccup(proc)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    html = card.read_text(encoding="utf-8")

    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    assert "18723 * 4409" in html
    assert MODEL in html and PROVIDER in html
    # The product reaches the card whether the model writes it grouped or not.
    assert "82549707" in html.replace(",", ""), "the answer belongs on the card"
    assert "calculator" in html, "the tool step belongs on the card"


@pytest.mark.integration
@pytest.mark.api
def test_run_card_opens_with_no_network_access(tmp_path):
    card = tmp_path / "card.html"
    proc = _run("Reply with exactly: OK", "-m", MODEL, "--provider", PROVIDER,
                "--card", str(card))
    _skip_if_upstream_hiccup(proc)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    html = card.read_text(encoding="utf-8")

    # No element may fetch anything: no external script/stylesheet/image/font.
    assert not re.search(r"<script[^>]+\bsrc\s*=", html)
    assert "<link" not in html
    assert not re.search(r"<img|@import|url\(\s*https?:", html)
    # A run with no sources has no links at all.
    assert "href=" not in html


@pytest.mark.integration
@pytest.mark.api
def test_card_leaves_json_stdout_untouched_and_composes_with_output(tmp_path):
    card = tmp_path / "card.html"
    saved = tmp_path / "run.json"
    proc = _run("Reply with exactly: OK", "-m", MODEL, "--provider", PROVIDER,
                "-q", "--json", "--card", str(card), "-o", str(saved))
    _skip_if_upstream_hiccup(proc)
    assert proc.returncode == 0, proc.stderr

    # stdout is the result document and nothing else — the card path is not on it.
    document = json.loads(proc.stdout)
    assert str(card) not in proc.stdout
    assert card.exists() and saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["output"] == document["output"]


@pytest.mark.integration
@pytest.mark.api
def test_saved_run_document_identifies_itself(tmp_path):
    saved = tmp_path / "run.json"
    proc = _run("Reply with exactly: OK", "-m", MODEL, "--provider", PROVIDER,
                "-q", "-o", str(saved))
    _skip_if_upstream_hiccup(proc)
    assert proc.returncode == 0, proc.stderr

    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["task"] == "Reply with exactly: OK"
    assert document["model"] == MODEL
    assert document["provider"] == PROVIDER
    # An ISO-8601 UTC start time, so a curated run can be dated later.
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", document["started_at"])


@pytest.mark.integration
@pytest.mark.api
def test_prefixed_model_id_names_the_provider_that_served_the_run(tmp_path):
    # `-m provider:model` carries the provider without `--provider`, and the
    # saved document has to name it: the card labels an absent cost by whether
    # the run was hosted or local, and reads the answer from this field.
    saved = tmp_path / "run.json"
    card = tmp_path / "card.html"
    proc = _run("Reply with exactly: OK", "-m", f"{PROVIDER}:{MODEL}", "-q",
                "-o", str(saved), "--card", str(card))
    _skip_if_upstream_hiccup(proc)
    assert proc.returncode == 0, proc.stderr

    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["provider"] == PROVIDER
    assert document["model"] == f"{PROVIDER}:{MODEL}"
    assert "unpriced (local)" not in card.read_text(encoding="utf-8")


@pytest.mark.integration
@pytest.mark.api
def test_saved_run_document_renders_as_a_card_after_the_fact(tmp_path):
    saved = tmp_path / "run.json"
    proc = _run("Name one prime number between 20 and 30.", "-m", MODEL,
                "--provider", PROVIDER, "-q", "-o", str(saved))
    _skip_if_upstream_hiccup(proc)
    assert proc.returncode == 0, proc.stderr

    out = tmp_path / "card.html"
    rendered = subprocess.run(
        [sys.executable, "-m", "effgen.cli", "report", str(saved), "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert rendered.returncode == 0, rendered.stderr
    assert "(run)" in rendered.stdout + rendered.stderr
    assert "Name one prime number" in out.read_text(encoding="utf-8")
