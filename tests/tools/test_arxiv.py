"""Tests for the new ArXivTool (search / fetch / download_pdf).

All tests hit the real arXiv Atom API. Transient errors skip.
"""

from __future__ import annotations

import asyncio
import os
import socket

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import ArXivTool


def _has_network(host: str = "export.arxiv.org", port: int = 80, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    return asyncio.run(coro)


def _ok(result: ToolResult):
    assert isinstance(result, ToolResult)
    if not result.success and result.error:
        # arXiv answers shared CI addresses with 406 or 503 at busy times while
        # the same request from elsewhere succeeds; that is the service's
        # refusal, not the tool's.
        transient_markers = (
            "HTTP 429",
            "HTTP 406",
            "HTTP 503",
            "Too Many Requests",
            "timed out",
            "Timeout",
        )
        if any(marker in result.error for marker in transient_markers):
            pytest.skip(f"arXiv transient error: {result.error}")
    assert result.success, f"tool failed: {result.error}"
    return result.output


def test_arxiv_metadata():
    assert ArXivTool().metadata.name == "arxiv"


def test_arxiv_search_requires_query():
    r = _run(ArXivTool().execute(operation="search"))
    assert not r.success


def test_arxiv_fetch_requires_id():
    r = _run(ArXivTool().execute(operation="fetch"))
    assert not r.success


@needs_net
def test_arxiv_search_returns_papers():
    tool = ArXivTool()
    out = _ok(_run(tool.execute(query="diffusion model image", max_results=5)))
    assert out["source"] == "arxiv"
    assert out["count"] >= 5
    first = out["results"][0]
    assert first["title"]
    assert isinstance(first["authors"], list)
    assert first["pdf_url"].startswith("http")
    fetched = _ok(_run(tool.execute(operation="fetch", arxiv_id=first["arxiv_id"])))
    assert fetched["result"]["title"]
    assert isinstance(fetched["result"]["authors"], list)
    assert fetched["result"]["authors"]


@needs_net
def test_arxiv_fetch_attention_is_all_you_need():
    out = _ok(_run(ArXivTool().execute(operation="fetch", arxiv_id="1706.03762")))
    assert out["arxiv_id"] == "1706.03762"
    paper = out["result"]
    assert "Attention Is All You Need" in paper["title"]
    assert any("Vaswani" in a for a in paper["authors"])
    assert "/pdf/" in paper["pdf_url"]


@needs_net
def test_arxiv_fetch_unknown_id_fails():
    r = _run(ArXivTool().execute(operation="fetch", arxiv_id="9999.99999"))
    assert not r.success


@needs_net
def test_arxiv_download_pdf_round_trip(tmp_path):
    dest = tmp_path / "paper.pdf"
    out = _ok(_run(ArXivTool().execute(
        operation="download_pdf", arxiv_id="1706.03762", dest=str(dest),
    )))
    assert out["arxiv_id"] == "1706.03762"
    assert os.path.exists(out["path"])
    # The PDF should be a real, non-trivial file.
    assert out["bytes"] > 50_000
    # PDF magic header.
    with open(out["path"], "rb") as f:
        assert f.read(4) == b"%PDF"
