"""Tests for the command palette and keyboard navigation on the web surfaces.

Covers:
1. The shared keyboard layer is served by both surfaces, from one source file,
   with no external asset.
2. Palette behavior — filtering, arrow-key movement, Enter, recently-used
   ordering — exercised directly against the state machine in ``webui.js``.
3. Modal semantics: dialog roles, focus trap, focus restoration, and a polite
   live region announcing the result count.
4. Constant-cost navigation: a section jump row whose targets take focus.
5. Accessibility details the surfaces share: skip links, one theme preference
   key, an announced theme change, the snippet tab pattern, and roving tabindex
   on the topology nodes.
6. The static routes serve shared assets and refuse a path escaping the
   static directory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2] / "effgen"
SHARED_DIR = ROOT / "webui" / "static"
DASH_DIR = ROOT / "dashboard" / "static"
PLAY_DIR = ROOT / "playground" / "static"

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node_path() -> str | None:
    """The global module directory, so ``require("jsdom")`` resolves there."""
    if NODE is None:
        return None
    npm = shutil.which("npm")
    if npm is None:
        return None
    try:
        out = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - npm absent
        return None
    return out.stdout.strip() or None


def _has_jsdom() -> bool:
    root = _node_path()
    if root is None:
        return False
    probe = subprocess.run(
        [NODE, "-e", 'require("jsdom"); console.log("ok")'],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "NODE_PATH": root},
    )
    return probe.returncode == 0


needs_jsdom = pytest.mark.skipif(
    not (NODE and _has_jsdom()),
    reason="jsdom is not installed (npm install -g jsdom)",
)

DASHBOARD_PANELS = [
    "summary-cards",
    "panel-slo",
    "panel-latency-chart",
    "panel-by-model",
    "panel-by-status",
    "panel-by-route",
    "panel-agent-runs",
    "panel-history",
    "panel-spans",
    "panel-waterfall",
    "panel-topology",
    "panel-catalog",
    "panel-metrics",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_node(script: str) -> dict:
    """Run a snippet against webui.js and return the JSON it prints."""
    prelude = (
        "globalThis.window = globalThis.window || {};\n"
        "const store = {};\n"
        "window.localStorage = {\n"
        "  getItem: (k) => (k in store ? store[k] : null),\n"
        "  setItem: (k, v) => { store[k] = String(v); },\n"
        "};\n"
        "window.__store = store;\n"
        f"const webui = require({str(SHARED_DIR / 'webui.js')!r});\n"
    )
    proc = subprocess.run(
        [NODE, "-e", prelude + script],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# One shared implementation, self-contained
# ---------------------------------------------------------------------------


class TestSharedAssets:
    def test_shared_files_exist(self):
        for name in ("webui.js", "webui.css"):
            assert (SHARED_DIR / name).is_file(), f"missing {name}"

    def test_both_surfaces_load_the_shared_layer(self):
        for directory in (DASH_DIR, PLAY_DIR):
            html = _read(directory / "index.html")
            assert 'src="webui.js"' in html
            assert 'href="webui.css"' in html
            # The shared layer defines the palette before the page wires it.
            assert html.index('src="webui.js"') < html.index('src="app.js"')

    def test_palette_is_not_duplicated_per_surface(self):
        """Neither surface reimplements the palette; both call the shared one."""
        for directory in (DASH_DIR, PLAY_DIR):
            js = _read(directory / "app.js")
            assert "window.effgenWebUI" in js
            assert "webui.init(" in js
            assert 'role="dialog"' not in js

    def test_no_external_asset_reference(self):
        for name in ("webui.js", "webui.css"):
            text = _read(SHARED_DIR / name)
            for host in ("jsdelivr", "unpkg", "cdnjs", "googleapis", "cloudflare"):
                assert host not in text, f"{name} references {host}"
            assert not re.findall(r"https?://", text), f"{name} has an external URL"
            assert not re.findall(r"""(?<![:\w])//[a-zA-Z0-9-]+\.""", text)


# ---------------------------------------------------------------------------
# Palette behavior, exercised against the state machine
# ---------------------------------------------------------------------------


ACTION_FIXTURE = """
const actions = [
  {id: "nav:catalog", group: "Navigate", label: "Go to Model catalog",
   keywords: "panel catalog section"},
  {id: "nav:history", group: "Navigate", label: "Go to History",
   keywords: "panel history runs"},
  {id: "model:gpt-5-nano", group: "Models", label: "gpt-5-nano",
   keywords: "openai tools vision"},
  {id: "run:abc123", group: "Runs", label: "Summarize the incident report",
   keywords: "groq openai/gpt-oss-20b abc123 ok"},
  {id: "action:theme", group: "Actions", label: "Switch color theme",
   keywords: "dark light appearance"},
];
const invoked = [];
actions.forEach((a) => { a.run = () => invoked.push(a.id); });
"""


@needs_node
class TestPaletteBehavior:
    def test_typing_filters_to_matching_commands(self):
        out = _run_node(ACTION_FIXTURE + """
        const s = webui.createPaletteState(actions, {});
        const seen = {};
        ["catalog", "nano", "gpt-oss", "theme", "go to"].forEach((q) => {
          s.setQuery(q);
          seen[q] = s.results().map((a) => a.id);
        });
        console.log(JSON.stringify(seen));
        """)
        assert out["catalog"] == ["nav:catalog"]
        assert out["nano"] == ["model:gpt-5-nano"]
        # A run is findable by the model that produced it, not only by its text.
        assert out["gpt-oss"] == ["run:abc123"]
        assert out["theme"] == ["action:theme"]
        assert out["go to"] == ["nav:catalog", "nav:history"]

    def test_label_match_outranks_a_keyword_match(self):
        out = _run_node(ACTION_FIXTURE + """
        const s = webui.createPaletteState(actions, {});
        s.setQuery("history");
        console.log(JSON.stringify({ids: s.results().map((a) => a.id)}));
        """)
        assert out["ids"][0] == "nav:history"

    def test_arrow_keys_move_and_wrap(self):
        out = _run_node(ACTION_FIXTURE + """
        const s = webui.createPaletteState(actions, {});
        const path = [s.active().id];
        s.move(1); path.push(s.active().id);
        s.move(-1); path.push(s.active().id);
        s.move(-1); path.push(s.active().id);   // wraps to the last row
        console.log(JSON.stringify({path: path, count: s.count()}));
        """)
        assert out["count"] == 5
        assert out["path"] == ["nav:catalog", "nav:history", "nav:catalog", "action:theme"]

    def test_enter_invokes_the_active_command(self):
        out = _run_node(ACTION_FIXTURE + """
        const s = webui.createPaletteState(actions, {});
        s.setQuery("incident");
        s.invoke();
        console.log(JSON.stringify({invoked: invoked, recent: s.recent()}));
        """)
        assert out["invoked"] == ["run:abc123"]
        assert out["recent"] == ["run:abc123"]

    def test_no_match_leaves_nothing_active(self):
        out = _run_node(ACTION_FIXTURE + """
        const s = webui.createPaletteState(actions, {});
        s.setQuery("no-such-command");
        console.log(JSON.stringify({count: s.count(), active: s.active(), index: s.index()}));
        """)
        assert out["count"] == 0
        assert out["active"] is None
        assert out["index"] == -1

    def test_recently_used_commands_lead_the_empty_query(self):
        out = _run_node(ACTION_FIXTURE + """
        const s = webui.createPaletteState(actions, {recent: ["action:theme", "run:abc123"]});
        console.log(JSON.stringify({ids: s.results().map((a) => a.id)}));
        """)
        assert out["ids"][:2] == ["action:theme", "run:abc123"]

    def test_invoking_records_the_command_as_recent(self):
        out = _run_node(ACTION_FIXTURE + """
        const s = webui.createPaletteState(actions, {recent: ["nav:history"]});
        s.setQuery("theme");
        s.invoke();
        s.setQuery("");
        console.log(JSON.stringify({ids: s.results().map((a) => a.id)}));
        """)
        assert out["ids"][:2] == ["action:theme", "nav:history"]


@needs_node
class TestSharedThemeKey:
    def test_one_key_is_shared_by_both_surfaces(self):
        out = _run_node("""
        webui.storeTheme("light");
        console.log(JSON.stringify({
          stored: window.__store, read: webui.readStoredTheme(), key: webui.THEME_KEY,
        }));
        """)
        assert out["key"] == "effgen-theme"
        assert out["stored"] == {"effgen-theme": "light"}
        assert out["read"] == "light"

    def test_a_choice_made_before_the_shared_key_carries_over(self):
        out = _run_node("""
        window.__store["effgen-playground-theme"] = "light";
        const first = webui.readStoredTheme();
        console.log(JSON.stringify({first: first, stored: window.__store}));
        """)
        assert out["first"] == "light"
        assert out["stored"]["effgen-theme"] == "light"

    def test_no_stored_choice_reads_as_none(self):
        out = _run_node('console.log(JSON.stringify({read: webui.readStoredTheme()}));')
        assert out["read"] is None

    def test_neither_surface_keeps_a_private_theme_key(self):
        for directory in (DASH_DIR, PLAY_DIR):
            js = _read(directory / "app.js")
            assert "effgen-dashboard-theme" not in js
            assert "effgen-playground-theme" not in js
            assert "readStoredTheme()" in js


# ---------------------------------------------------------------------------
# Modal semantics
# ---------------------------------------------------------------------------


class TestPaletteAccessibility:
    def test_overlay_is_a_modal_dialog_with_a_name(self):
        js = _read(SHARED_DIR / "webui.js")
        assert 'role="dialog"' in js
        assert 'aria-modal="true"' in js
        assert "aria-labelledby=" in js

    def test_results_are_a_listbox_with_an_active_descendant(self):
        js = _read(SHARED_DIR / "webui.js")
        assert 'role="listbox"' in js
        assert 'role="option"' in js
        assert 'role="combobox"' in js
        # Focus stays in the input; the active row is pointed at, not focused.
        assert 'setAttribute("aria-activedescendant"' in js
        assert 'aria-selected="' in js

    def test_result_count_is_announced_politely(self):
        js = _read(SHARED_DIR / "webui.js")
        assert 'aria-live="polite"' in js
        assert "status.textContent" in js

    def test_focus_is_trapped_and_restored(self):
        js = _read(SHARED_DIR / "webui.js")
        assert "function trapTab(" in js
        assert 'event.key !== "Tab"' in js
        assert "lastFocused = document.activeElement" in js
        assert "lastFocused.focus()" in js

    def test_escape_and_cmd_k_are_bound_on_the_document(self):
        js = _read(SHARED_DIR / "webui.js")
        assert 'document.addEventListener("keydown"' in js
        assert "event.metaKey || event.ctrlKey" in js
        assert 'key === "Escape"' in js
        assert 'key === "?"' in js

    def test_shortcut_reference_is_discoverable(self):
        js = _read(SHARED_DIR / "webui.js")
        assert "Open the command palette" in js
        assert "? shortcuts" in js
        dash = _read(DASH_DIR / "index.html")
        assert 'id="palette-hint"' in dash
        for html in (dash, _read(PLAY_DIR / "index.html")):
            assert 'id="palette-btn"' in html

    def test_backdrop_click_closes(self):
        js = _read(SHARED_DIR / "webui.js")
        assert 'data-close="1"' in js
        assert "closePalette()" in js


# ---------------------------------------------------------------------------
# Constant-cost navigation and skip links
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_dashboard_lists_every_panel_in_the_jump_row(self):
        html = _read(DASH_DIR / "index.html")
        assert 'id="panel-nav"' in html
        assert 'aria-label="Dashboard sections"' in html
        for panel in DASHBOARD_PANELS:
            assert f'data-panel-jump="{panel}"' in html, f"{panel} missing from the jump row"

    def test_every_jump_target_can_take_focus(self):
        html = _read(DASH_DIR / "index.html")
        for panel in DASHBOARD_PANELS:
            assert re.search(rf'id="{panel}" tabindex="-1"', html), panel

    def test_jump_moves_focus_and_respects_reduced_motion(self):
        js = _read(SHARED_DIR / "webui.js")
        assert "function jumpTo(" in js
        assert "el.focus({ preventScroll: true })" in js
        assert 'reducedMotion() ? "auto" : "smooth"' in js

    def test_both_surfaces_have_a_skip_link_to_the_main_landmark(self):
        for directory in (DASH_DIR, PLAY_DIR):
            html = _read(directory / "index.html")
            assert 'class="eff-skip-link" href="#main-content"' in html
            assert 'id="main-content"' in html
            # The skip link is the first focusable element on the page.
            body = html[html.index("<body>"):]
            assert body.index("eff-skip-link") < body.index("<button")

    def test_skip_link_is_hidden_until_focused(self):
        css = _read(SHARED_DIR / "webui.css")
        block = css[css.index(".eff-skip-link {"):]
        assert "top: -3rem;" in block[: block.index("}")]
        assert ".eff-skip-link:focus" in css

    def test_the_surfaces_link_to_each_other(self):
        assert 'href="/playground"' in _read(DASH_DIR / "index.html")
        assert 'href="/dashboard"' in _read(PLAY_DIR / "index.html")

    def test_dashboard_commands_cover_panels_runs_models_and_actions(self):
        js = _read(DASH_DIR / "app.js")
        for builder in ("navigationCommands", "runCommands", "modelCommands", "actionCommands"):
            assert f"function {builder}(" in js
        for panel in DASHBOARD_PANELS:
            assert f'"{panel}"' in js, f"{panel} not reachable from the palette"

    def test_playground_commands_cover_models_presets_and_snippets(self):
        js = _read(PLAY_DIR / "app.js")
        for builder in ("modelCommands", "presetCommands", "snippetCommands", "actionCommands"):
            assert f"function {builder}(" in js


# ---------------------------------------------------------------------------
# Per-surface keyboard details
# ---------------------------------------------------------------------------


class TestSurfaceDetails:
    def test_theme_control_announces_the_active_theme(self):
        for directory in (DASH_DIR, PLAY_DIR):
            html = _read(directory / "index.html")
            js = _read(directory / "app.js")
            assert 'id="theme-status"' in html
            assert 'role="status"' in html
            assert "Dark theme active. Switch to light theme." in js
            assert '"Dark theme" : "Light theme"' in js

    def test_playground_submit_shortcut_is_document_wide(self):
        js = _read(PLAY_DIR / "app.js")
        handler = js[js.index("function initKeyboard()"):]
        handler = handler[: handler.index("if (!webui) return;")]
        assert 'document.addEventListener("keydown"' in handler
        assert "e.ctrlKey || e.metaKey" in handler
        # The old prompt-only binding is gone.
        assert '$("prompt").addEventListener("keydown"' not in js

    def test_snippet_tabs_implement_the_full_tab_pattern(self):
        html = _read(PLAY_DIR / "index.html")
        js = _read(PLAY_DIR / "app.js")
        assert html.count('role="tab"') == 3
        assert html.count('aria-controls="snippet-panel"') == 3
        assert 'role="tabpanel"' in html
        assert 'aria-labelledby="snippet-tab-curl"' in html
        # One tab stop for the group; the arrow keys move within it.
        assert html.count('tabindex="-1"') >= 2
        assert 'aria-selected="true"' in html
        assert 'e.key === "ArrowRight"' in js
        assert 'setAttribute("aria-selected"' in js

    def test_topology_graph_is_one_tab_stop_with_arrow_key_movement(self):
        js = _read(DASH_DIR / "app.js")
        assert "topoFocusId" in js
        assert 'tabindex="-1"' in js
        assert "moveFocus(g, 1)" in js
        assert 'ev.key === "ArrowRight"' in js

    def test_escape_closes_an_open_detail_pane(self):
        js = _read(DASH_DIR / "app.js")
        assert "function closeTopoDetail()" in js
        assert "function closeRunDetail()" in js
        assert "palette.onEscape(" in js

    def test_run_disclosure_state_updates_without_rebuilding_the_table(self):
        js = _read(DASH_DIR / "app.js")
        assert "function syncRunDisclosure()" in js
        assert 'setAttribute("aria-expanded"' in js


# ---------------------------------------------------------------------------
# The palette driven from the keyboard, in a DOM
#
# The checks above read the source; these load the real pages and press real
# keys, which is the only way to show that focus, the trap, Escape and the
# recently-used order behave as the page promises.
# ---------------------------------------------------------------------------


DRIVER = r"""
const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs");
const payloads = JSON.parse(fs.readFileSync(process.env.EFF_PAYLOADS, "utf8"));
const dir = process.env.EFF_SURFACE_DIR;
const shared = process.env.EFF_SHARED_DIR;

(async () => {
  const vc = new VirtualConsole();
  const dom = new JSDOM(fs.readFileSync(dir + "/index.html", "utf8"), {
    url: "http://localhost/" + process.env.EFF_SURFACE,
    runScripts: "dangerously",
    pretendToBeVisual: true,
    virtualConsole: vc,
  });
  const w = dom.window;
  // Every network call the page makes is answered from a recorded payload,
  // so the run is offline and deterministic.
  w.fetch = async (url) => {
    const path = String(url).split("?")[0].replace(/^https?:\/\/[^/]+/, "");
    const body = payloads[path];
    return {
      ok: body !== undefined, status: body === undefined ? 404 : 200,
      json: async () => body, text: async () => JSON.stringify(body),
    };
  };
  w.EventSource = function () { this.close = () => {}; };
  // jsdom ships no canvas; the chart draw is not what is under test here.
  w.HTMLCanvasElement.prototype.getContext = function () {
    const noop = () => {};
    return new Proxy({}, { get: (_t, k) => (k === "measureText" ? () => ({ width: 8 }) : noop),
                           set: () => true });
  };
  w.HTMLElement.prototype.scrollIntoView = function () { w.__scrolled = this.id; };

  // Wait for the parse to finish before injecting, so the page's own
  // "run once the DOM is ready" guard fires exactly once — as it does in a
  // browser, where the scripts are parsed with the document.
  if (w.document.readyState === "loading") {
    await new Promise((r) => w.document.addEventListener("DOMContentLoaded", r));
  }
  for (const src of ["webui.js", "app.js"]) {
    const from = src === "webui.js" ? shared : dir;
    const el = w.document.createElement("script");
    el.textContent = fs.readFileSync(from + "/" + src, "utf8");
    w.document.body.appendChild(el);
  }
  await new Promise((r) => setTimeout(r, 250));
  if (w.document.querySelectorAll("#eff-palette").length !== 1) {
    throw new Error("the page initialised more than once");
  }

  const doc = w.document;
  const $ = (id) => doc.getElementById(id);
  const press = (k, o = {}) => {
    const ev = new w.KeyboardEvent("keydown",
      Object.assign({ key: k, bubbles: true, cancelable: true }, o));
    (doc.activeElement || doc.body).dispatchEvent(ev);
    return ev;
  };
  const typeQuery = (v) => {
    const input = $("eff-palette-input");
    input.value = v;
    input.dispatchEvent(new w.Event("input", { bubbles: true }));
  };
  const labels = () => Array.from(doc.querySelectorAll(".eff-palette-option"))
    .map((o) => o.querySelector(".eff-opt-label").textContent);
  const activeLabel = () => {
    const el = doc.querySelector(".eff-palette-option.is-active .eff-opt-label");
    return el ? el.textContent : null;
  };
  const focusId = () => (doc.activeElement ? doc.activeElement.id : null);
  const out = {};
  const ctx = { w, doc, $, press, typeQuery, labels, activeLabel, focusId, out,
                palette: () => $("eff-palette"), help: () => $("eff-shortcuts") };

  await (__SCRIPT__)(ctx);
  console.log(JSON.stringify(out));
  w.close();
})().catch((e) => { console.error(e && e.stack || e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def payload_file(tmp_path_factory):
    """Record every JSON payload the two surfaces fetch, straight from the app."""
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    client = TestClient(create_app(dev_mode=True), raise_server_exceptions=False)
    paths = [
        "/dashboard/data.json", "/dashboard/catalog.json",
        "/dashboard/history.json", "/dashboard/topology.json",
        "/playground/bootstrap",
    ]
    payloads = {}
    for path in paths:
        resp = client.get(path)
        if resp.status_code == 200:
            payloads[path] = resp.json()
    # A stored run and a two-node execution, so the Runs group and the topology
    # graph have something to show without needing a live model call.
    payloads["/dashboard/history.json"] = {
        "runs": [
            {"run_id": "r-0001", "task": "Summarize the incident report",
             "model": "openai/gpt-oss-20b", "status": "ok", "cost_usd": 1.2e-05,
             "output": "A brief summary.", "started": "2026-07-18T12:00:00"},
            {"run_id": "r-0002", "task": "Name one benefit of vector databases",
             "model": "gpt-5-nano", "status": "ok", "cost_usd": 9e-06,
             "output": "Fast similarity search.", "started": "2026-07-18T12:01:00"},
        ],
        "sessions": [], "run": None, "persisted": True,
    }
    payloads["/dashboard/topology.json"] = {
        "executions": [{
            "id": "e1", "kind": "team", "name": "brief", "status": "ok",
            "cost_usd": 9e-06, "tokens": 159,
            "nodes": [
                {"id": "manager", "label": "manager", "type": "agent",
                 "status": "ok", "model": "gpt-5-nano", "role": "manager"},
                {"id": "editor", "label": "editor", "type": "agent",
                 "status": "ok", "model": "openai/gpt-oss-20b", "role": "collab"},
            ],
            "edges": [{"source": "manager", "target": "editor", "kind": "delegation"}],
        }],
    }
    path = tmp_path_factory.mktemp("webui") / "payloads.json"
    path.write_text(json.dumps(payloads), encoding="utf-8")
    return path


def _drive(surface: str, payload_file: Path, script: str) -> dict:
    """Run ``script`` (an async function body) against a real page in jsdom."""
    directory = DASH_DIR if surface == "dashboard" else PLAY_DIR
    env = {
        **os.environ,
        "NODE_PATH": _node_path() or "",
        "EFF_PAYLOADS": str(payload_file),
        "EFF_SURFACE": surface,
        "EFF_SURFACE_DIR": str(directory),
        "EFF_SHARED_DIR": str(SHARED_DIR),
    }
    proc = subprocess.run(
        [NODE, "-e", DRIVER.replace("__SCRIPT__", script)],
        capture_output=True, text=True, timeout=180, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@needs_jsdom
class TestKeyboardOperationInADom:
    def test_ctrl_k_opens_the_palette_and_escape_restores_focus(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, focusId, palette, out, doc }) => {
          doc.getElementById("theme-btn").focus();
          out.before = focusId();
          press("k", { ctrlKey: true });
          out.opened = !palette().hidden;
          out.focus_on_open = focusId();
          out.modal = palette().querySelector('[role="dialog"]').getAttribute("aria-modal");
          press("Escape");
          out.closed = palette().hidden;
          out.focus_restored = focusId();
        }""")
        assert out["before"] == "theme-btn"
        assert out["opened"] is True
        assert out["focus_on_open"] == "eff-palette-input"
        assert out["modal"] == "true"
        assert out["closed"] is True
        # Focus returns to where it was, not to the top of the document.
        assert out["focus_restored"] == "theme-btn"

    def test_tab_is_trapped_inside_the_open_palette(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, focusId, out }) => {
          press("k", { ctrlKey: true });
          const ev = press("Tab");
          out.prevented = ev.defaultPrevented;
          out.focus = focusId();
          const back = press("Tab", { shiftKey: true });
          out.shift_focus = focusId();
          out.shift_prevented = back.defaultPrevented;
        }""")
        assert out["prevented"] is True
        assert out["focus"] == "eff-palette-input"
        assert out["shift_focus"] == "eff-palette-input"

    def test_typing_narrows_the_list_and_announces_the_count(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, typeQuery, labels, $, out }) => {
          press("k", { ctrlKey: true });
          out.all = labels().length;
          typeQuery("catalog");
          out.catalog = labels();
          out.announced = $("eff-palette-status").textContent;
          typeQuery("vector databases");
          out.run_hit = labels();
          typeQuery("zzz-no-such-thing");
          out.empty = !!document_query();
          function document_query() { return $("eff-palette-list").querySelector(".eff-palette-empty"); }
          out.no_activedescendant = $("eff-palette-input").getAttribute("aria-activedescendant");
        }""")
        assert out["all"] > 12
        assert out["catalog"] == ["Go to Model catalog", "Search the model catalog"]
        assert out["announced"] == "2 commands"
        # A stored run is reachable by its task text.
        assert out["run_hit"] == ["Name one benefit of vector databases"]
        assert out["empty"] is True
        # Nothing is active, so nothing is pointed at.
        assert out["no_activedescendant"] is None

    def test_arrows_move_the_active_row_without_moving_focus(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, typeQuery, activeLabel, focusId, $, doc, out }) => {
          press("k", { ctrlKey: true });
          typeQuery("go to");
          out.first = activeLabel();
          press("ArrowDown");
          out.second = activeLabel();
          out.focus = focusId();
          out.pointer = $("eff-palette-input").getAttribute("aria-activedescendant");
          out.selected = doc.querySelectorAll('.eff-palette-option[aria-selected="true"]').length;
          press("ArrowUp"); press("ArrowUp");
          out.wrapped = activeLabel();
        }""")
        assert out["first"] != out["second"]
        # Focus never leaves the input; the active row is pointed at instead.
        assert out["focus"] == "eff-palette-input"
        assert out["pointer"] == "eff-opt-1"
        assert out["selected"] == 1
        # Up from the first row wraps to the last.
        assert out["wrapped"] is not None

    def test_enter_runs_the_command_and_moves_focus_to_the_panel(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, typeQuery, focusId, palette, w, out }) => {
          press("k", { ctrlKey: true });
          typeQuery("prometheus metrics");
          press("Enter");
          out.closed = palette().hidden;
          out.focus = focusId();
          out.scrolled = w.__scrolled;
        }""")
        assert out["closed"] is True
        # The jump moves real focus, so the next Tab continues from the panel.
        assert out["focus"] == "panel-metrics"
        assert out["scrolled"] == "panel-metrics"

    def test_an_invoked_command_leads_the_next_open(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, typeQuery, labels, w, out }) => {
          press("k", { ctrlKey: true });
          typeQuery("prometheus metrics");
          press("Enter");
          press("k", { ctrlKey: true });
          out.first = labels()[0];
          out.stored = w.localStorage.getItem("effgen-recent-actions");
        }""")
        # Recency applies within the session, not only after a reload.
        assert out["first"] == "Go to Prometheus metrics"
        assert json.loads(out["stored"])[0] == "nav:panel-metrics"

    def test_question_mark_opens_the_shortcut_list_and_escape_closes_it(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, help, doc, out }) => {
          const ev = press("?");
          out.open = !help().hidden;
          out.prevented = ev.defaultPrevented;
          out.rows = doc.querySelectorAll("#eff-keys div").length;
          out.named = !!help().querySelector('[role="dialog"][aria-labelledby]');
          press("Escape");
          out.closed = help().hidden;
        }""")
        assert out["open"] is True
        assert out["prevented"] is True
        assert out["rows"] >= 5
        assert out["named"] is True
        assert out["closed"] is True

    def test_question_mark_typed_into_a_field_stays_text(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, help, $, out }) => {
          $("cat-search").focus();
          const ev = press("?");
          out.open = !help().hidden;
          out.prevented = ev.defaultPrevented;
        }""")
        assert out["open"] is False
        assert out["prevented"] is False

    def test_opening_a_run_expands_its_disclosure_and_escape_closes_it(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, typeQuery, doc, $, out }) => {
          press("k", { ctrlKey: true });
          typeQuery("incident report");
          press("Enter");
          out.detail_open = !$("history-detail").hidden;
          out.expanded = doc.querySelectorAll('#history-tbody button[aria-expanded="true"]').length;
          out.focus_class = doc.activeElement.className;
          press("Escape");
          out.detail_after = $("history-detail").hidden;
          out.expanded_after = doc.querySelectorAll('#history-tbody button[aria-expanded="true"]').length;
        }""")
        assert out["detail_open"] is True
        # Exactly the chosen row reports itself expanded, and focus lands on it.
        assert out["expanded"] == 1
        assert "link-btn" in out["focus_class"]
        assert out["detail_after"] is True
        assert out["expanded_after"] == 0

    def test_topology_is_one_tab_stop_and_arrows_move_between_nodes(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, doc, out }) => {
          const nodes = () => Array.from(doc.querySelectorAll("g.topo-node"));
          out.nodes = nodes().length;
          out.tab_stops = nodes().filter((n) => n.getAttribute("tabindex") === "0").length;
          const first = nodes()[0];
          first.focus();
          press("ArrowRight");
          out.moved_to = doc.activeElement.getAttribute("data-node");
          out.tab_stops_after = nodes().filter((n) => n.getAttribute("tabindex") === "0").length;
        }""")
        assert out["nodes"] == 2
        # A large team stays one tab stop however many nodes it draws.
        assert out["tab_stops"] == 1
        assert out["moved_to"] == "editor"
        assert out["tab_stops_after"] == 1

    def test_the_jump_row_moves_focus_to_the_panel(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, w, focusId, out }) => {
          const link = doc.querySelector('a[data-panel-jump="panel-topology"]');
          link.dispatchEvent(new w.MouseEvent("click", { bubbles: true, cancelable: true }));
          out.focus = focusId();
        }""")
        assert out["focus"] == "panel-topology"

    def test_the_theme_choice_is_shared_and_announced(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ press, typeQuery, w, $, out }) => {
          press("k", { ctrlKey: true });
          typeQuery("switch color theme");
          press("Enter");
          out.shared = w.localStorage.getItem("effgen-theme");
          out.legacy = w.localStorage.getItem("effgen-dashboard-theme");
          out.announced = $("theme-status").textContent;
          out.label = $("theme-btn").getAttribute("aria-label");
        }""")
        assert out["shared"] in ("dark", "light")
        # Nothing is written under the old per-surface key any more.
        assert out["legacy"] is None
        assert out["announced"] in ("Dark theme", "Light theme")
        assert "Switch to" in out["label"]


@needs_jsdom
class TestPlaygroundKeyboardInADom:
    def test_submit_shortcut_works_away_from_the_prompt(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ press, $, out }) => {
          $("prompt").value = "";           // run() stops on an empty prompt
          $("temperature").focus();
          press("Enter", { ctrlKey: true });
          out.reached_run = $("error-banner").textContent;
        }""")
        # The shortcut reached run() from the temperature field, which is what
        # the old prompt-only binding could not do.
        assert out["reached_run"] == "Enter a prompt first."

    def test_submit_shortcut_does_not_fire_while_the_palette_is_open(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ press, typeQuery, $, palette, focusId, out }) => {
          $("prompt").value = "";
          press("k", { ctrlKey: true });
          typeQuery("edit the prompt");
          press("Enter", { ctrlKey: true });
          out.closed = palette().hidden;
          out.focus = focusId();
          out.error = $("error-banner").textContent;
        }""")
        assert out["closed"] is True
        # The chosen command ran; the request shortcut did not also fire.
        assert out["focus"] == "prompt"
        assert out["error"] == ""

    def test_snippet_tabs_are_one_tab_stop_with_arrow_key_movement(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ press, doc, $, out }) => {
          const tabs = () => Array.from(doc.querySelectorAll(".snippet-tab"));
          out.stops = tabs().filter((t) => t.getAttribute("tabindex") === "0").length;
          tabs()[0].focus();
          press("ArrowRight");
          out.focus = doc.activeElement.id;
          out.selected = doc.querySelector('.snippet-tab[aria-selected="true"]').dataset.kind;
          out.labelled_by = $("snippet-panel").getAttribute("aria-labelledby");
          out.stops_after = tabs().filter((t) => t.getAttribute("tabindex") === "0").length;
          press("End");
          out.end = doc.activeElement.id;
        }""")
        assert out["stops"] == 1
        assert out["focus"] == "snippet-tab-cli"
        assert out["selected"] == "cli"
        # The panel names the tab that selected it.
        assert out["labelled_by"] == "snippet-tab-cli"
        assert out["stops_after"] == 1
        assert out["end"] == "snippet-tab-python"

    def test_the_palette_lists_models_and_presets_from_the_loaded_page(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ press, labels, doc, out }) => {
          press("k", { ctrlKey: true });
          const groups = Array.from(doc.querySelectorAll(".eff-palette-group-label"))
            .map((g) => g.textContent);
          out.groups = groups;
          out.count = labels().length;
        }""")
        assert "Actions" in out["groups"]
        assert "Navigate" in out["groups"]
        assert "Copy" in out["groups"]
        assert out["count"] > 5


@needs_jsdom
class TestStickyChrome:
    def test_the_jump_row_is_offset_below_the_sticky_header(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, out }) => {
          const root = doc.documentElement;
          out.header = root.style.getPropertyValue("--eff-header-h");
          out.sticky = root.style.getPropertyValue("--eff-sticky-h");
        }""")
        # Both surfaces publish the measurement; jsdom lays nothing out, so the
        # value is 0px here — what matters is that the properties are set, so
        # the jump row never renders on top of the header.
        assert out["header"].endswith("px")
        assert out["sticky"].endswith("px")

    def test_the_stylesheet_consumes_the_measurement(self):
        css = _read(SHARED_DIR / "webui.css")
        assert "top: var(--eff-header-h, 0px);" in css
        assert "scroll-margin-top: calc(var(--eff-sticky-h, 0px)" in css


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from effgen.server.app import create_app

    app = create_app()
    return fastapi_testclient.TestClient(app)


class TestSharedAssetRoutes:
    def test_both_surfaces_serve_the_shared_layer(self, client):
        for path in ("/dashboard/webui.js", "/playground/webui.js",
                     "/dashboard/webui.css", "/playground/webui.css"):
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert "effgenWebUI" in resp.text or ".eff-palette-option" in resp.text

    def test_surface_assets_still_serve(self, client):
        for path in ("/dashboard/app.js", "/dashboard/style.css",
                     "/playground/app.js", "/playground/style.css"):
            assert client.get(path).status_code == 200, path

    def test_asset_path_cannot_escape_the_static_directory(self, client):
        for path in ("/dashboard/../__init__.py", "/dashboard/..%2F__init__.py",
                     "/playground/../../server/app.py", "/dashboard/webui/../../__init__.py"):
            resp = client.get(path)
            # Either the client normalizes the path away from the surface (and
            # the request is rejected elsewhere) or the handler declines it —
            # in no case does a source file come back.
            assert resp.status_code in (400, 401, 404), f"{path} -> {resp.status_code}"
            assert "def " not in resp.text

    def test_unknown_asset_is_a_404(self, client):
        assert client.get("/dashboard/nope.js").status_code == 404
        assert client.get("/playground/nope.css").status_code == 404


def test_resolver_rejects_traversal_and_absolute_paths():
    from effgen.server.app import _resolve_web_asset

    assert _resolve_web_asset("webui.js", SHARED_DIR) is not None
    assert _resolve_web_asset("../__init__.py", SHARED_DIR) is None
    assert _resolve_web_asset("/etc/passwd", SHARED_DIR) is None
    assert _resolve_web_asset("", SHARED_DIR) is None
    # The first directory holding the file wins; the fallback covers the rest.
    assert _resolve_web_asset("app.js", DASH_DIR, SHARED_DIR).parent == DASH_DIR
    assert _resolve_web_asset("webui.js", DASH_DIR, SHARED_DIR).parent == SHARED_DIR
