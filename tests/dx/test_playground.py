"""Tests for the in-browser playground — static assets, routes, auth, and the
data plumbing it depends on (tool trace + model catalog over HTTP).

Validates:
1. Static files exist, carry the expected control IDs, and load no external
   network resource (self-contained, like the dashboard).
2. ``/playground`` serves the SPA shell; ``/playground/bootstrap`` serves the
   presets/tools/defaults JSON.
3. The static shell is public but the bootstrap data endpoint is auth-gated by
   default, opening only under ``EFFGEN_PUBLIC_PLAYGROUND`` — a flag separate
   from the dashboard's, so opening the playground does not open the dashboard.
4. ``GET /v1/models/catalog`` returns the catalog with accurate pricing.
5. The chat response surfaces the tool step trace in its ``effgen`` extension.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).parents[2] / "effgen" / "playground" / "static"

REQUIRED_CONTROL_IDS = [
    "preset-select",
    "model-select",
    "prompt",
    "tool-list",
    "temperature",
    "max-tokens",
    "stream-toggle",
    "run-btn",
    "answer",
    "stats",
    "trace",
    "snippets",
    "api-key",
]


# ---------------------------------------------------------------------------
# Static file checks (no server required)
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_assets_exist(self):
        for name in ("index.html", "app.js", "style.css"):
            assert (STATIC_DIR / name).exists(), f"missing {name}"

    def test_index_contains_control_ids(self):
        html = (STATIC_DIR / "index.html").read_text()
        for cid in REQUIRED_CONTROL_IDS:
            assert f'id="{cid}"' in html, f"control id '{cid}' not in index.html"

    def test_index_loads_only_local_assets(self):
        html = (STATIC_DIR / "index.html").read_text()
        # No <script src="http...">, <link href="http...">, or protocol-relative
        # asset. Every asset must be served by the same origin (self-contained).
        assert not re.search(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', html), (
            "index.html references an external asset"
        )

    def test_no_external_network_hosts(self):
        # No CDN/font-host references anywhere in the bundle. The only literal
        # URL allowed is the loopback example inside a copy-as-curl snippet.
        for name in ("index.html", "app.js", "style.css"):
            text = (STATIC_DIR / name).read_text()
            for host in ("jsdelivr", "unpkg", "cdnjs", "googleapis", "cloudflare"):
                assert host not in text, f"{name} references {host}"
            for url in re.findall(r"https?://[^\s\"'<>()]+", text):
                assert url.startswith("http://127.0.0.1"), f"{name} has external URL {url}"

    def test_style_is_theme_aware(self):
        css = (STATIC_DIR / "style.css").read_text()
        assert 'data-theme="light"' in css
        assert "prefers-color-scheme" in css
        assert "prefers-reduced-motion" in css

    def test_python_snippet_resolves_tool_instances(self):
        # AgentConfig(tools=...) takes Tool instances, not names, so the
        # copy-as-Python snippet must resolve each name through the registry;
        # passing bare name strings would raise TypeError when pasted.
        js = (STATIC_DIR / "app.js").read_text()
        assert "get_tool_sync(" in js, "python snippet must resolve tool names"
        assert 'tools=" + JSON.stringify(r.tools)' not in js, (
            "python snippet must not pass bare tool-name strings to AgentConfig"
        )

    def test_agent_tools_take_instances_not_names(self):
        # Guards the contract the copy-as-Python snippet depends on: bare tool
        # names raise when they reach the agent, so the snippet must resolve
        # each name to a Tool instance through the registry first.
        from effgen import Agent
        from effgen.core.agent import AgentConfig
        from effgen.tools import get_registry

        with pytest.raises(TypeError):
            Agent(AgentConfig(model="groq:openai/gpt-oss-20b", tools=["calculator"]))
        tool = get_registry().get_tool_sync("calculator")
        cfg = AgentConfig(model="groq:openai/gpt-oss-20b", tools=[tool])
        assert cfg.tools and cfg.tools[0] is tool


# ---------------------------------------------------------------------------
# Route tests (dev-mode app — auth bypassed)
# ---------------------------------------------------------------------------


def _make_dev_app():
    try:
        from effgen.server.app import create_app

        return create_app(dev_mode=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Cannot create test app: {exc}")


class TestPlaygroundRoutes:
    @pytest.fixture(autouse=True)
    def client(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        self._client = TestClient(_make_dev_app(), raise_server_exceptions=False)
        return self._client

    def test_index_200_html(self):
        resp = self._client.get("/playground")
        assert resp.status_code == 200
        assert "html" in resp.headers.get("content-type", "")
        assert 'id="run-btn"' in resp.text

    def test_static_assets(self):
        for asset in ("style.css", "app.js"):
            resp = self._client.get(f"/playground/{asset}")
            assert resp.status_code == 200, asset

    def test_bootstrap_shape(self):
        resp = self._client.get("/playground/bootstrap")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("presets", "tools", "default_model", "defaults", "catalog_url"):
            assert key in data
        assert isinstance(data["presets"], list) and data["presets"]
        # Each preset carries what the page needs to apply it client-side.
        first = data["presets"][0]
        assert {"name", "system_prompt", "tools"} <= set(first)
        # The always-safe tool options are offered.
        assert "calculator" in data["tools"]


# ---------------------------------------------------------------------------
# Auth gate: static shell public, bootstrap protected by default
# ---------------------------------------------------------------------------


class TestPlaygroundAuth:
    @pytest.fixture
    def _tc(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        return TestClient

    def _clear_env(self, monkeypatch):
        for var in (
            "EFFGEN_DEV_MODE",
            "EFFGEN_PUBLIC_PLAYGROUND",
            "EFFGEN_PUBLIC_DASHBOARD",
            "EFFGEN_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_static_shell_public_bootstrap_gated(self, _tc, monkeypatch):
        self._clear_env(monkeypatch)
        from effgen.server.app import create_app

        client = _tc(create_app(dev_mode=False), raise_server_exceptions=False)
        # Shell loads without a key so the page can render and prompt for one.
        assert client.get("/playground").status_code == 200
        assert client.get("/playground/style.css").status_code == 200
        # The bootstrap (which can carry a spend-authorizing key) requires auth.
        assert client.get("/playground/bootstrap").status_code == 401

    def test_lookalike_path_not_public(self, _tc, monkeypatch):
        self._clear_env(monkeypatch)
        from effgen.server.app import create_app

        client = _tc(create_app(dev_mode=False), raise_server_exceptions=False)
        assert client.get("/playgroundevil").status_code == 401

    def test_public_playground_opens_bootstrap_only(self, _tc, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("EFFGEN_PUBLIC_PLAYGROUND", "1")
        monkeypatch.setenv("EFFGEN_API_KEY", "secret-key")
        from effgen.server.app import create_app

        client = _tc(create_app(dev_mode=False), raise_server_exceptions=False)
        # Playground bootstrap is now public and injects the session key so a
        # local-view demo can Run.
        resp = client.get("/playground/bootstrap")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_key"] == "secret-key"
        assert data["spend_authorized"] is True
        # The dashboard data endpoint stays gated — the flags are independent.
        assert client.get("/dashboard/data.json").status_code == 401

    def test_no_key_injected_without_public_flag(self, _tc, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("EFFGEN_API_KEY", "secret-key")
        from effgen.server.app import create_app

        app = create_app(dev_mode=False)
        client = _tc(app, raise_server_exceptions=False)
        # With a key, reach the (protected) bootstrap and confirm no key leaks.
        resp = client.get(
            "/playground/bootstrap", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status_code == 200
        assert resp.json()["session_key"] is None


# ---------------------------------------------------------------------------
# Model catalog over HTTP: accurate pricing + provenance
# ---------------------------------------------------------------------------


class TestModelCatalog:
    def test_build_model_catalog_shape(self):
        from effgen.api.openai_compat import build_model_catalog

        cat = build_model_catalog()
        assert cat["object"] == "list"
        assert isinstance(cat["data"], list) and cat["data"]
        assert isinstance(cat["providers"], list) and cat["providers"]
        # Every record carries the fields a picker needs, and pricing is never
        # fabricated: an unpriced record reports None, not 0.
        for rec in cat["data"]:
            assert {"id", "provider", "is_priced", "price_source"} <= set(rec)
            if not rec["is_priced"]:
                assert rec["price_in_per_1m"] is None
                assert rec["price_out_per_1m"] is None

    def test_catalog_provider_filter(self):
        from effgen.api.openai_compat import build_model_catalog

        cat = build_model_catalog("groq")
        assert cat["data"]
        assert {r["provider"] for r in cat["data"]} == {"groq"}
        assert cat["providers"][0]["provider"] == "groq"

    def test_catalog_route(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        client = TestClient(_make_dev_app(), raise_server_exceptions=False)
        resp = client.get("/v1/models/catalog?provider=groq")
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["catalog"] >= 1
        assert all(r["provider"] == "groq" for r in data["data"])

    def test_catalog_route_requires_auth(self, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        for var in ("EFFGEN_DEV_MODE", "EFFGEN_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        from effgen.server.app import create_app

        client = TestClient(create_app(dev_mode=False), raise_server_exceptions=False)
        assert client.get("/v1/models/catalog").status_code == 401


# ---------------------------------------------------------------------------
# Tool trace surfacing
# ---------------------------------------------------------------------------


class TestToolTrace:
    def test_extract_tool_trace_pairs_events(self):
        from effgen.server.app import _extract_tool_trace

        class _Resp:
            execution_trace = [
                {
                    "type": "tool_call_start",
                    "timestamp": 100.0,
                    "data": {"tool_name": "calculator", "tool_input": "6*7"},
                },
                {
                    "type": "tool_call_complete",
                    "timestamp": 100.05,
                    "data": {"tool_name": "calculator", "result": "42"},
                },
            ]

        steps = _extract_tool_trace(_Resp())
        assert len(steps) == 1
        step = steps[0]
        assert step["tool"] == "calculator"
        assert step["args"] == "6*7"
        assert step["result_summary"] == "42"
        assert step["ok"] is True
        assert step["duration_ms"] == pytest.approx(50.0, abs=1.0)

    def test_extract_tool_trace_marks_failure(self):
        from effgen.server.app import _extract_tool_trace

        class _Resp:
            execution_trace = [
                {
                    "type": "tool_call_start",
                    "timestamp": 1.0,
                    "data": {"tool_name": "bash", "tool_input": "ls"},
                },
                {
                    "type": "tool_call_failed",
                    "timestamp": 1.01,
                    "data": {"tool_name": "bash", "error": "blocked"},
                },
            ]

        steps = _extract_tool_trace(_Resp())
        assert steps[0]["ok"] is False
        assert steps[0]["result_summary"] == "blocked"

    def test_extract_tool_trace_empty_for_direct_answer(self):
        from effgen.server.app import _extract_tool_trace

        class _Resp:
            execution_trace = [
                {"type": "task_start", "timestamp": 1.0, "data": {}},
                {"type": "task_complete", "timestamp": 2.0, "data": {}},
            ]

        assert _extract_tool_trace(_Resp()) == []

    def test_chat_response_surfaces_trace(self):
        """A runner that reports a trace has it echoed in the ``effgen`` object."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from effgen.api.openai_compat import RunnerResult
        from effgen.server.app import create_app

        def _runner(prompt, *, model, tools=None, stream=False, **_):
            return RunnerResult(
                text="1827993",
                prompt_tokens=10,
                completion_tokens=3,
                resolved_model=model,
                metadata={
                    "tool_calls": 1,
                    "trace": [
                        {
                            "tool": "calculator",
                            "args": "8347*219",
                            "result_summary": "1827993",
                            "ok": True,
                            "duration_ms": 4.0,
                        }
                    ],
                },
            )

        client = TestClient(
            create_app(dev_mode=True, runner=_runner), raise_server_exceptions=False
        )
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "groq:openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "8347*219?"}],
                "tools": [{"type": "function", "function": {"name": "calculator"}}],
            },
        )
        assert resp.status_code == 200
        eff = resp.json().get("effgen", {})
        assert eff.get("tool_calls") == 1
        assert eff["trace"][0]["tool"] == "calculator"
        assert eff["trace"][0]["result_summary"] == "1827993"
