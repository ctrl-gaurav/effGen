"""Accessibility guarantees the web surfaces make over time.

Two lanes:

1. **Contrast** — pure Python, always runs. Reads the design tokens straight out
   of the shipped stylesheets, composites translucent fills over their parent,
   and asserts every documented foreground/background pair clears its WCAG
   threshold in both themes on both surfaces.
2. **Behavior in a real DOM** — jsdom, gated the same way the palette tests are.
   Static markup says nothing about what a page does on its second, third and
   hundredth refresh, so these drive the real pages with recorded payloads and
   replayed streams and assert the behavior: focus survives a poll, an unchanged
   payload announces nothing, a race is quiet while it runs and states its
   verdict once, a streamed answer marks itself busy for the duration.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.dx.test_web_palette import (
    DASH_DIR,
    NODE,
    PLAY_DIR,
    SHARED_DIR,
    _node_path,
    needs_jsdom,
)

# ---------------------------------------------------------------------------
# Lane 1 — contrast, measured from the shipped stylesheets
# ---------------------------------------------------------------------------


def _tokens(css: str, selector: str) -> dict[str, str]:
    """Custom properties declared in the first block matching *selector*."""
    at = css.index(selector)
    block = css[css.index("{", at) + 1 : css.index("}", at)]
    return dict(re.findall(r"--([a-z0-9-]+)\s*:\s*([^;]+);", block))


def _rgb(value: str) -> tuple[float, float, float]:
    value = value.strip()
    if value.startswith("#"):
        digits = value[1:]
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        return (
            int(digits[0:2], 16),
            int(digits[2:4], 16),
            int(digits[4:6], 16),
        )
    match = re.match(r"rgba?\(([^)]+)\)", value)
    if match:
        parts = [p.strip() for p in match.group(1).split(",")]
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    raise ValueError(f"not a color: {value}")


def _alpha(value: str) -> float:
    match = re.match(r"rgba\(([^)]+)\)", value.strip())
    if match:
        parts = [p.strip() for p in match.group(1).split(",")]
        if len(parts) == 4:
            return float(parts[3])
    return 1.0


def _over(fg: str, bg: tuple[float, float, float]) -> tuple[float, float, float]:
    """Composite *fg* (possibly translucent) over the solid color *bg*."""
    alpha = _alpha(fg)
    front = _rgb(fg)
    return tuple(alpha * front[i] + (1 - alpha) * bg[i] for i in range(3))  # type: ignore[return-value]


def _luminance(color: tuple[float, float, float]) -> float:
    def channel(value: float) -> float:
        value = value / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(c) for c in color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _ratio(fg: tuple[float, float, float], bg: tuple[float, float, float]) -> float:
    a, b = _luminance(fg), _luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


# (what it is, foreground, background stack outermost-last, required ratio).
# A value starting with "--" names a token; anything else is a literal color.
# 4.5 is the AA threshold for body text, 3.0 the non-text threshold of WCAG
# 1.4.11 — which is what a control's boundary has to clear to be identifiable.
DASHBOARD_PAIRS: list[tuple[str, str, list[str], float]] = [
    ("body text on page", "--text", ["--bg"], 4.5),
    ("body text on panel", "--text", ["--bg-panel"], 4.5),
    ("panel heading", "--text-muted", ["--bg-panel"], 4.5),
    ("table header text", "--text-muted", ["--bg-card"], 4.5),
    ("card value", "--accent2", ["--bg-card"], 4.5),
    ("card sub-note", "--text-muted", ["--bg-card"], 4.5),
    ("hint text on panel", "--text-faint", ["--bg-panel"], 4.5),
    ("span timestamp on inset", "--text-faint", ["--bg-inset"], 4.5),
    ("span name on inset", "--accent2", ["--bg-inset"], 4.5),
    ("span duration on inset", "--ok", ["--bg-inset"], 4.5),
    ("span error on inset", "--err", ["--bg-inset"], 4.5),
    ("status chip 2xx code", "--ok", ["--bg-card"], 4.5),
    ("status chip 3xx code", "--accent2", ["--bg-card"], 4.5),
    ("status chip 4xx code", "--warn", ["--bg-card"], 4.5),
    ("status chip 5xx code", "--err", ["--bg-card"], 4.5),
    ("by-model outcome text", "--warn", ["--bg-panel"], 4.5),
    ("badge ok", "--ok", ["rgba(52, 211, 153, 0.18)", "--bg-panel"], 4.5),
    ("badge err", "--err", ["rgba(248, 113, 113, 0.18)", "--bg-panel"], 4.5),
    ("history link button", "--accent", ["--bg-panel"], 4.5),
    ("run id in waterfall head", "--accent", ["--bg-inset"], 4.5),
    ("waterfall track label", "--text-faint", ["--bg-inset"], 4.5),
    ("waterfall label on model bar", "--wf-label", ["--accent"], 4.5),
    ("waterfall label on tool bar", "--wf-label", ["--accent2"], 4.5),
    ("waterfall label on run bar", "--wf-label", ["--text-faint"], 4.5),
    ("waterfall label on router bar", "--wf-label", ["--warn"], 4.5),
    ("waterfall label on error bar", "--wf-label", ["--err"], 4.5),
    ("topology node name", "--text", ["--bg-card"], 4.5),
    ("topology node meta", "--text-muted", ["--bg-card"], 4.5),
    ("topology legend text", "--text-muted", ["--bg-panel"], 4.5),
    ("catalog toolbar label", "--text-muted", ["--bg-panel"], 4.5),
    ("catalog yes tick", "--ok", ["--bg-panel"], 4.5),
    ("empty-row text", "--text-muted", ["--bg-panel"], 4.5),
    ("auth banner text", "--text", ["rgba(251, 191, 36, 0.14)", "--bg"], 4.5),
    # Non-text contrast (WCAG 1.4.11).
    ("focus ring on page", "--focus", ["--bg"], 3.0),
    ("focus ring on panel", "--focus", ["--bg-panel"], 3.0),
    ("focus ring on inset", "--focus", ["--bg-inset"], 3.0),
    ("status dot ok", "--ok", ["--bg-panel"], 3.0),
    ("status dot error", "--err", ["--bg-panel"], 3.0),
    ("SLO bar fill (accent)", "--accent", ["--bg-card"], 3.0),
    ("SLO bar fill (err)", "--err", ["--bg-card"], 3.0),
    ("topology edge", "--text-faint", ["--bg-inset"], 3.0),
    ("topology node outline ok", "--ok", ["--bg-card"], 3.0),
    ("topology node outline skipped", "--warn", ["--bg-card"], 3.0),
    # Control boundaries: the border is what identifies the control.
    ("input border vs panel", "--border-strong", ["--bg-panel"], 3.0),
    ("input border vs its fill", "--border-strong", ["--bg-inset"], 3.0),
    ("input border vs card", "--border-strong", ["--bg-card"], 3.0),
    ("page button border vs panel", "--border-strong", ["--bg-panel"], 3.0),
    ("header icon button border", "--border-strong", ["--bg-panel"], 3.0),
]

PLAYGROUND_PAIRS: list[tuple[str, str, list[str], float]] = [
    ("run button label", "--on-accent", ["--accent"], 4.5),
    ("stat value", "--accent2", ["--bg-card"], 4.5),
    ("hint text", "--text-faint", ["--bg-panel"], 4.5),
    ("trace tool name", "--accent2", ["--bg-inset"], 4.5),
    ("trace duration", "--ok", ["--bg-inset"], 4.5),
    ("trace muted args", "--text-muted", ["--bg-inset"], 4.5),
    ("battle state streaming", "--accent2", ["--bg-card"], 4.5),
    ("battle state done", "--ok", ["--bg-card"], 4.5),
    ("battle state failed", "--err", ["--bg-card"], 4.5),
    ("battle footer", "--text-muted", ["--bg-card"], 4.5),
    ("battle error text", "--err", ["--bg-card"], 4.5),
    ("verdict key", "--text-muted", ["--bg-card"], 4.5),
    ("answer placeholder", "--text-faint", ["--bg-inset"], 4.5),
    ("streaming cursor", "--accent2", ["--bg-inset"], 4.5),
    ("snippet tab (inactive)", "--text-muted", ["--bg-panel"], 4.5),
    ("banner warn text", "--text", ["rgba(251, 191, 36, 0.14)", "--bg"], 4.5),
    ("banner err text", "--text", ["rgba(248, 113, 113, 0.14)", "--bg"], 4.5),
    ("badge local", "--accent2", ["rgba(34, 211, 238, 0.16)", "--bg-panel"], 4.5),
    ("badge free", "--ok", ["rgba(52, 211, 153, 0.18)", "--bg-panel"], 4.5),
    ("run button fill vs panel", "--accent", ["--bg-panel"], 3.0),
    ("textarea border vs panel", "--border-strong", ["--bg-panel"], 3.0),
    ("textarea border vs its fill", "--border-strong", ["--bg-inset"], 3.0),
    ("tool chip border vs panel", "--border-strong", ["--bg-panel"], 3.0),
    ("snippet tab border vs panel", "--border-strong", ["--bg-panel"], 3.0),
]

# Both themes are declared in every stylesheet; the media block repeats the
# light values for first paint, and a separate lockstep test pins that.
THEME_SELECTORS = {"dark": ":root", "light": ':root[data-theme="light"]'}


def _measure(css_path: Path, pairs: list) -> list[tuple[str, str, float, float]]:
    css = css_path.read_text(encoding="utf-8")
    rows: list[tuple[str, str, float, float]] = []
    for theme, selector in THEME_SELECTORS.items():
        token = _tokens(css, selector)

        def resolve(value: str, _t: dict[str, str] = token) -> str:
            return _t[value[2:]] if value.startswith("--") else value

        for label, fg, backgrounds, need in pairs:
            color = _rgb(resolve(backgrounds[-1]))
            for layer in reversed(backgrounds[:-1]):
                color = _over(resolve(layer), color)
            foreground = _over(resolve(fg), color)
            rows.append((theme, label, _ratio(foreground, color), need))
    return rows


class TestContrast:
    """Every documented pair clears its WCAG threshold in both themes."""

    @pytest.mark.parametrize(
        ("surface", "pairs"),
        [("dashboard", DASHBOARD_PAIRS), ("playground", PLAYGROUND_PAIRS)],
    )
    def test_every_pair_clears_its_threshold(self, surface, pairs):
        directory = DASH_DIR if surface == "dashboard" else PLAY_DIR
        failures = [
            f"{surface}/{theme}: {label} = {measured:.2f} (needs {need})"
            for theme, label, measured, need in _measure(directory / "style.css", pairs)
            if measured < need
        ]
        assert not failures, "\n".join(failures)

    def test_control_and_divider_borders_are_separate_tokens(self):
        """A decorative divider and a control boundary are not the same color."""
        for directory in (DASH_DIR, PLAY_DIR):
            css = (directory / "style.css").read_text(encoding="utf-8")
            for selector in THEME_SELECTORS.values():
                token = _tokens(css, selector)
                assert token["border"] != token["border-strong"]

    def test_shared_palette_input_uses_the_control_border(self):
        css = (SHARED_DIR / "webui.css").read_text(encoding="utf-8")
        block = css[css.index("#eff-palette-input") : css.index("}", css.index("#eff-palette-input"))]
        assert "var(--border-strong)" in block


def _rule(css: str, selector: str) -> dict[str, str]:
    """Declarations of the first rule whose selector list starts with *selector*."""
    at = css.index(selector + " {") if selector + " {" in css else css.index(selector)
    body = css[css.index("{", at) + 1 : css.index("}", at)]
    return {
        name.strip(): value.strip()
        for name, _, value in (d.partition(":") for d in body.split(";"))
        if name.strip()
    }


class TestFieldsetLooksLikeAField:
    """A checkbox group is a fieldset so the group carries a name.

    Naming it must not change how it looks: the browser draws a box around a
    fieldset and leaves the legend at body typography, which would make one
    control in the column look unlike all the others.
    """

    def test_the_tool_group_opts_out_of_the_default_box(self):
        html = (PLAY_DIR / "index.html").read_text(encoding="utf-8")
        assert 'class="field tool-field"' in html
        css = (PLAY_DIR / "style.css").read_text(encoding="utf-8")
        assert _rule(css, ".tool-field")["border"] == "0"

    def test_the_legend_reads_as_a_field_label(self):
        css = (PLAY_DIR / "style.css").read_text(encoding="utf-8")
        label = _rule(css, ".field label")
        legend = _rule(css, ".tool-field legend")
        for prop in ("display", "font-size", "color", "text-transform",
                     "letter-spacing", "margin-bottom"):
            assert legend[prop] == label[prop], prop

    def test_the_mode_picker_keeps_its_own_box(self):
        """The one fieldset that is meant to look like a box still does."""
        css = (PLAY_DIR / "style.css").read_text(encoding="utf-8")
        assert "var(--border-strong)" in _rule(css, ".mode-field")["border"]
        # Its legend rule is declared after the shared one, so it wins.
        assert css.index(".mode-field legend") > css.index(".tool-field legend")


# ---------------------------------------------------------------------------
# Lane 2 — behavior in a real DOM
# ---------------------------------------------------------------------------

DRIVER = r"""
const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs");
const payloads = JSON.parse(fs.readFileSync(process.env.EFF_PAYLOADS, "utf8"));
const dir = process.env.EFF_SURFACE_DIR;
const shared = process.env.EFF_SHARED_DIR;

// A recorded SSE body, handed back as a ReadableStream reader so the page's own
// streaming code runs unchanged.
function sseBody(w, chunks) {
  let i = 0;
  return {
    getReader: () => ({
      read: async () => {
        if (i >= chunks.length) return { done: true, value: undefined };
        const text = chunks[i++];
        return { done: false, value: new TextEncoder().encode(text) };
      },
    }),
  };
}

(async () => {
  const vc = new VirtualConsole();
  const dom = new JSDOM(fs.readFileSync(dir + "/index.html", "utf8"), {
    url: "http://localhost/" + process.env.EFF_SURFACE,
    runScripts: "dangerously",
    pretendToBeVisual: true,
    virtualConsole: vc,
  });
  const w = dom.window;
  w.__streams = {};   // model id -> array of raw SSE frames
  w.__fetches = [];
  w.fetch = async (url, init) => {
    const path = String(url).split("?")[0].replace(/^https?:\/\/[^/]+/, "");
    w.__fetches.push(path);
    if (path === "/v1/chat/completions") {
      const body = JSON.parse((init && init.body) || "{}");
      const frames = w.__streams[body.model] || w.__streams["*"] || [];
      return { ok: true, status: 200, body: sseBody(w, frames) };
    }
    const body = payloads[path];
    return {
      ok: body !== undefined, status: body === undefined ? 404 : 200,
      json: async () => body, text: async () => JSON.stringify(body),
    };
  };
  w.EventSource = function () { this.close = () => {}; };
  w.HTMLCanvasElement.prototype.getContext = function () {
    const noop = () => {};
    return new Proxy({}, { get: (_t, k) => (k === "measureText" ? () => ({ width: 8 }) : noop),
                           set: () => true });
  };
  w.HTMLElement.prototype.scrollIntoView = function () {};

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

  const doc = w.document;
  const $ = (id) => doc.getElementById(id);
  const settle = (ms) => new Promise((r) => setTimeout(r, ms == null ? 120 : ms));
  // Count DOM mutations inside each element, the way a screen reader watches a
  // live region.
  const watch = (els) => {
    const counts = new Map();
    const obs = els.map((el) => {
      counts.set(el, 0);
      const o = new w.MutationObserver((records) => {
        counts.set(el, counts.get(el) + records.length);
      });
      o.observe(el, { childList: true, characterData: true, subtree: true,
                      attributes: false });
      return o;
    });
    return {
      stop: () => { obs.forEach((o) => { o.takeRecords(); o.disconnect(); }); },
      get: (el) => counts.get(el),
    };
  };
  const liveRegions = () => Array.from(
    doc.querySelectorAll('[aria-live="polite"], [aria-live="assertive"], [role="status"]')
  ).filter((el) => el.getAttribute("aria-live") !== "off");
  const out = {};
  await (__SCRIPT__)({ w, doc, $, out, settle, watch, liveRegions });
  console.log(JSON.stringify(out));
  w.close();
})().catch((e) => { console.error((e && e.stack) || e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def payload_file(tmp_path_factory):
    """Record the JSON payloads both surfaces fetch, straight from the real app."""
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from effgen.observability.metrics import record_http_request, record_model_call, reset_all
    from effgen.observability.run_log import clear, record_run
    from effgen.server.app import create_app

    reset_all()
    clear()
    # A model served by two providers, one of which is failing, plus HTTP
    # traffic on two routes — so the by-model and by-route panels have rows.
    for _ in range(3):
        record_model_call(provider="openai", model="gpt-5-nano", outcome="ok", latency=0.4)
    record_model_call(provider="openai", model="gpt-5-nano", outcome="not_found", latency=0.2)
    record_model_call(provider="groq", model="gpt-5-nano", outcome="ok", latency=0.2)
    record_run(model="openai:gpt-5-nano", provider="openai", cost_usd=1.2e-05, duration_s=0.4)
    for _ in range(4):
        record_http_request(route="/v1/chat/completions", method="POST", status=200)
    record_http_request(route="/v1/chat/completions", method="POST", status=404)
    record_http_request(route="other", method="GET", status=404)

    client = TestClient(create_app(dev_mode=True), raise_server_exceptions=False)
    payloads = {}
    for path in (
        "/dashboard/data.json", "/dashboard/catalog.json",
        "/dashboard/history.json", "/dashboard/topology.json",
        "/playground/bootstrap", "/v1/models/catalog",
    ):
        resp = client.get(path)
        if resp.status_code == 200:
            payloads[path] = resp.json()
    reset_all()
    clear()

    payloads["/dashboard/history.json"] = {
        "runs": [
            {"run_id": "r-0001", "task": "Summarize the incident report",
             "model": "openai/gpt-oss-20b", "status": "ok", "cost_usd": 1.2e-05},
            {"run_id": "r-0002", "task": "Name one benefit of vector databases",
             "model": "gpt-5-nano", "status": "ok", "cost_usd": 9e-06},
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
    path = tmp_path_factory.mktemp("a11y") / "payloads.json"
    path.write_text(json.dumps(payloads), encoding="utf-8")
    return path


def _drive(surface: str, payload_file: Path, script: str) -> dict:
    """Run *script* (an async function body) against a real page in jsdom."""
    directory = DASH_DIR if surface == "dashboard" else PLAY_DIR
    env = {
        **os.environ,
        "NODE_PATH": _node_path() or "",
        "EFF_PAYLOADS": str(payload_file),
        "EFF_SURFACE": surface,
        "EFF_SURFACE_DIR": str(directory),
        "EFF_SHARED_DIR": str(SHARED_DIR),
        # Two recorded answer streams, so a battle and a single run replay
        # through the page's own streaming code with no network.
        "EFF_STREAM_A": json.dumps(_sse_frames(["hello", " there", " world"])),
        "EFF_STREAM_B": json.dumps(_sse_frames(["a", "b", "c", "d"])),
    }
    proc = subprocess.run(
        [NODE, "-e", DRIVER.replace("__SCRIPT__", script)],
        capture_output=True, text=True, timeout=180, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _sse_frames(words: list[str]) -> list[str]:
    """Recorded chat-completion SSE frames, one content delta per word."""
    frames = []
    for word in words:
        frames.append(
            "data: " + json.dumps(
                {"model": "m", "choices": [{"delta": {"content": word}}]}
            ) + "\n\n"
        )
    frames.append(
        "data: " + json.dumps(
            {"model": "m", "choices": [],
             "usage": {"prompt_tokens": 4, "completion_tokens": len(words),
                       "total_tokens": 4 + len(words)}}
        ) + "\n\n"
    )
    frames.append("data: [DONE]\n\n")
    return frames


@needs_jsdom
class TestDashboardBehavior:
    def test_focus_survives_a_refresh_of_the_history_table(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, $, out, settle }) => {
          await settle(200);
          const btn = doc.querySelector("#history-tbody button[data-run-index]");
          out.button_found = !!btn;
          btn.focus();
          out.focused_before = doc.activeElement === btn;
          const runId = btn.getAttribute("data-run-index");
          $("refresh-btn").click();
          await settle(250);
          const active = doc.activeElement;
          out.focus_left_the_body = active !== doc.body;
          out.focus_on_a_run_button = !!(active && active.matches
            && active.matches("#history-tbody button[data-run-index]"));
          out.same_run = !!(active && active.getAttribute("data-run-index") === runId);
        }""")
        assert out["button_found"] is True
        assert out["focused_before"] is True
        assert out["focus_left_the_body"] is True, "the poll dropped focus to <body>"
        assert out["focus_on_a_run_button"] is True
        assert out["same_run"] is True

    def test_three_identical_refreshes_announce_nothing(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, $, out, settle, watch, liveRegions }) => {
          await settle(250);
          const regions = liveRegions();
          out.region_ids = regions.map((el) => el.id || el.className);
          const w = watch(regions);
          for (let i = 0; i < 3; i++) { $("refresh-btn").click(); await settle(220); }
          w.stop();
          out.mutations = {};
          regions.forEach((el) => {
            out.mutations[el.id || el.className] = w.get(el);
          });
        }""")
        # Every polite region on the page was watched, and the payload never
        # changed, so none of them may have been rewritten.
        assert len(out["region_ids"]) >= 7, out["region_ids"]
        noisy = {k: v for k, v in out["mutations"].items() if v}
        assert not noisy, f"live regions rewritten with nothing new to say: {noisy}"

    def test_catalog_capability_cells_expose_yes_and_no_as_text(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, out, settle }) => {
          await settle(300);
          const cells = Array.from(doc.querySelectorAll("#catalog-tbody tr td"));
          out.rows = doc.querySelectorAll("#catalog-tbody tr").length;
          const text = cells.map((c) => c.textContent).join("|");
          out.has_yes = /yes/.test(text);
          out.has_no = /\\bno\\b/.test(text);
          out.aria_label_on_roleless = doc.querySelectorAll(
            "#catalog-tbody span[aria-label]").length;
        }""")
        assert out["rows"] > 0
        assert out["has_yes"] is True
        assert out["has_no"] is True
        assert out["aria_label_on_roleless"] == 0

    def test_status_chips_state_the_class_and_the_count_as_text(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, out, settle }) => {
          await settle(250);
          out.chips = Array.from(doc.querySelectorAll("#status-chips .status-chip"))
            .map((c) => c.textContent.trim());
        }""")
        assert out["chips"], "no status chips rendered"
        for chip in out["chips"]:
            assert re.search(r"\d{3} (success|redirect|client error|server error)", chip), chip
            assert "response" in chip

    def test_the_by_model_table_sorts_and_announces_the_sort(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, out, settle }) => {
          await settle(250);
          const th = doc.querySelector('#by-model-table th[data-sort="model"]');
          const btn = th.querySelector("button.sort-btn");
          btn.click();
          await settle(80);
          out.sorted_headers = doc.querySelectorAll("#by-model-table th[aria-sort]").length;
          out.aria_sort = th.getAttribute("aria-sort");
          out.focus_kept = doc.activeElement === btn;
          const live = doc.getElementById("eff-announce");
          out.announced = live ? live.textContent : null;
          out.announce_is_status = live ? live.getAttribute("role") : null;
        }""")
        # aria-sort moves rather than accumulating: exactly one header carries it.
        assert out["sorted_headers"] == 1
        assert out["aria_sort"] == "ascending"
        assert out["focus_kept"] is True
        assert "Sorted by Model" in (out["announced"] or "")
        assert out["announce_is_status"] == "status"

    def test_the_by_route_panel_separates_the_routes_behind_a_status(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, out, settle }) => {
          await settle(250);
          out.rows = Array.from(doc.querySelectorAll("#by-route-tbody tr"))
            .map((tr) => Array.from(tr.querySelectorAll("td")).map((td) => td.textContent.trim()));
        }""")
        routes = {row[0] for row in out["rows"]}
        assert "/v1/chat/completions" in routes
        assert "other" in routes, "unrouted traffic must be nameable, not folded in"

    def test_the_detail_panes_skip_no_heading_level(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, $, out, settle }) => {
          await settle(250);
          doc.querySelector("#history-tbody button[data-run-index]").click();
          await settle(80);
          out.detail_headings = Array.from($("history-detail").querySelectorAll("*"))
            .filter((el) => /^H[1-6]$/.test(el.tagName)).map((el) => el.tagName);
          out.page_headings = Array.from(doc.querySelectorAll("h1,h2,h3,h4,h5,h6"))
            .map((el) => Number(el.tagName.slice(1)));
        }""")
        assert "H4" not in out["detail_headings"]
        levels = out["page_headings"]
        for previous, current in zip(levels, levels[1:], strict=False):
            assert current - previous <= 1, f"heading level jumped {previous} → {current}"


@needs_jsdom
class TestPlaygroundBehavior:
    def test_a_replayed_battle_is_quiet_and_states_its_verdict_once(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ doc, $, w, out, settle, watch }) => {
          await settle(250);
          w.__streams = {
            "openai:gpt-5-nano": JSON.parse(process.env.EFF_STREAM_A),
            "groq:openai/gpt-oss-20b": JSON.parse(process.env.EFF_STREAM_B),
          };
          $("mode-battle").checked = true;
          $("mode-battle").dispatchEvent(new w.Event("change", { bubbles: true }));
          await settle(80);
          const sel = $("battle-models");
          for (const id of Object.keys(w.__streams)) {
            const opt = doc.createElement("option");
            opt.value = id; opt.textContent = id; opt.selected = true;
            sel.appendChild(opt);
          }
          const grid = $("battle-grid");
          out.grid_live = grid.getAttribute("aria-live");
          const gridWatch = watch([grid]);
          $("run-btn").click();
          await settle(900);
          gridWatch.stop();
          out.grid_mutations_while_racing = gridWatch.get(grid);
          out.grid_busy_after = grid.getAttribute("aria-busy");
          out.verdict_role = $("battle-verdict").getAttribute("role");
          out.verdict_hidden = $("battle-verdict").hidden;
          const live = doc.getElementById("eff-announce");
          out.announced = live ? live.textContent : null;
          out.column_headings = grid.querySelectorAll(".battle-col h3").length;
          out.column_groups = grid.querySelectorAll('.battle-col[role="group"]').length;
        }""")
        # aria-live="off" means the ten-per-second rebuild announces nothing…
        assert out["grid_live"] == "off"
        assert out["grid_mutations_while_racing"] > 0, "the race did not run"
        assert out["grid_busy_after"] == "false"
        # …and the verdict is what gets spoken, once, through a status region.
        assert out["verdict_role"] == "status"
        assert out["verdict_hidden"] is False
        assert "Battle complete" in (out["announced"] or "")
        assert out["announced"].count("Battle complete") == 1
        assert out["column_headings"] == 2
        assert out["column_groups"] == 2

    def test_a_streamed_answer_is_busy_while_it_streams(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ doc, $, w, out, settle }) => {
          await settle(250);
          w.__streams = { "*": JSON.parse(process.env.EFF_STREAM_A) };
          const answer = $("answer");
          out.atomic = answer.getAttribute("aria-atomic");
          out.live = answer.getAttribute("aria-live");
          out.busy_before = answer.getAttribute("aria-busy");
          const seen = [];
          const obs = new w.MutationObserver(() => {
            seen.push(answer.getAttribute("aria-busy"));
          });
          obs.observe(answer, { childList: true, characterData: true, subtree: true });
          $("run-btn").click();
          await settle(600);
          obs.disconnect();
          out.busy_during = seen.filter((v) => v === "true").length;
          out.busy_after = answer.getAttribute("aria-busy");
          out.text = answer.textContent;
          out.element_children = Array.from(answer.children).map((el) => el.tagName);
        }""")
        assert out["live"] == "polite"
        assert out["atomic"] == "false", "an atomic region re-reads the whole answer per delta"
        assert out["busy_before"] == "false"
        assert out["busy_during"] > 0, "the region was never marked busy while streaming"
        assert out["busy_after"] == "false"
        assert "hello" in out["text"]
        # Deltas are appended as text nodes; no markup is ever built from them.
        assert out["element_children"] == []

    def test_the_tool_checkbox_group_is_named(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ doc, out, settle }) => {
          await settle(250);
          const legends = Array.from(doc.querySelectorAll("fieldset legend"))
            .map((el) => el.textContent.trim());
          out.legends = legends;
          out.orphan_labels = Array.from(doc.querySelectorAll("label"))
            .filter((l) => !l.getAttribute("for") && !l.querySelector("input, select, textarea"))
            .map((l) => l.textContent.trim());
        }""")
        assert "Attach tools" in out["legends"]
        assert out["orphan_labels"] == []

    def test_the_contender_picker_offers_only_chat_models(self, payload_file):
        out = _drive("playground", payload_file, """
        async ({ doc, $, w, out, settle }) => {
          await settle(300);
          $("mode-battle").checked = true;
          $("mode-battle").dispatchEvent(new w.Event("change", { bubbles: true }));
          await settle(120);
          out.contenders = Array.from($("battle-models").querySelectorAll("option"))
            .map((o) => o.value);
          out.picker = Array.from($("model-select").querySelectorAll("option"))
            .map((o) => o.value);
        }""")
        embedding = re.compile(r"embedding|bge|minilm|sentence-transformers", re.I)
        assert not [m for m in out["contenders"] if embedding.search(m)], out["contenders"]
        assert not [m for m in out["picker"] if embedding.search(m)], out["picker"]


@needs_jsdom
class TestSharedAnnouncer:
    def test_the_same_message_twice_is_spoken_twice(self, payload_file):
        """Writing an identical string is not a change, so it would go unread.

        Sorting a table back to a column it already used, or two races ending
        the same way, must still reach the listener.
        """
        out = _drive("dashboard", payload_file, """
        async ({ doc, w, out, settle }) => {
          await settle(250);
          w.effgenWebUI.announce("Sorted by Calls, descending");
          await settle(120);
          const el = doc.getElementById("eff-announce");
          out.first = el.textContent;
          const writes = [];
          const obs = new w.MutationObserver(() => writes.push(el.textContent));
          obs.observe(el, { childList: true, characterData: true, subtree: true });
          w.effgenWebUI.announce("Sorted by Calls, descending");
          await settle(300);
          obs.disconnect();
          out.writes = writes;
          out.final = el.textContent;
          // A different message must win over a repeat still in flight.
          w.effgenWebUI.announce("Sorted by Cost, descending");
          w.effgenWebUI.announce("Sorted by Cost, descending");
          w.effgenWebUI.announce("Sorted by Model, ascending");
          await settle(300);
          out.after_interruption = el.textContent;
        }""")
        assert out["first"] == "Sorted by Calls, descending"
        # Emptied, then written back — two changes the region reports.
        assert out["writes"][0] == ""
        assert out["final"] == "Sorted by Calls, descending"
        assert out["after_interruption"] == "Sorted by Model, ascending"


@needs_jsdom
class TestPaletteExpandedState:
    def test_aria_expanded_follows_the_result_count(self, payload_file):
        out = _drive("dashboard", payload_file, """
        async ({ doc, w, out, settle }) => {
          await settle(250);
          const press = (k, o) => (doc.activeElement || doc.body).dispatchEvent(
            new w.KeyboardEvent("keydown",
              Object.assign({ key: k, bubbles: true, cancelable: true }, o)));
          press("k", { ctrlKey: true });
          await settle(60);
          const input = doc.getElementById("eff-palette-input");
          out.with_results = input.getAttribute("aria-expanded");
          input.value = "zzzzznotacommand";
          input.dispatchEvent(new w.Event("input", { bubbles: true }));
          await settle(60);
          out.options = doc.querySelectorAll(".eff-palette-option").length;
          out.with_no_results = input.getAttribute("aria-expanded");
        }""")
        assert out["with_results"] == "true"
        assert out["options"] == 0
        assert out["with_no_results"] == "false"
