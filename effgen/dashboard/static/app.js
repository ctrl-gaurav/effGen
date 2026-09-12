/**
 * effGen Dashboard — frontend app
 *
 * Polls /dashboard/data.json every 5 seconds and renders:
 *  - summary cards (requests, model-call errors, latency, real cost, tokens)
 *  - SLO burn-rate bars (true p99 from the latency histogram)
 *  - a latency trend chart drawn on a canvas (no external chart library)
 *  - per-model breakdown (calls, error rate, p95, tokens, cost)
 *  - HTTP responses by status code
 *  - recent agent runs table
 *  - history of stored runs and saved sessions, with per-run drill-in
 *  - multi-agent topology graph (hand-drawn SVG, no graph library)
 *  - live span stream (SSE from /dashboard/spans if available)
 *  - raw Prometheus metrics table
 *
 * Keyboard navigation — the section jump row, the Cmd/Ctrl-K command palette
 * and the "?" shortcut list — comes from webui.js, which the playground loads
 * as well. This file supplies the dashboard's own palette commands.
 *
 * All assets are served locally; the page has no external network dependency.
 */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Constants
  // ------------------------------------------------------------------
  const POLL_MS = 5000;
  const MAX_SPANS = 200;
  const MAX_CHART_POINTS = 30;
  // The keyboard layer shared with the playground. Absent only if webui.js
  // failed to load, in which case the page keeps working without shortcuts.
  const webui = window.effgenWebUI || null;

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  let latencyHistory = [];
  let spanCount = 0;
  let spanPaused = false;
  let eventSource = null;
  // Spans grouped by run id for the per-run waterfall. Insertion order is the
  // order runs first appeared; the newest few are drawn.
  const runs = new Map();
  const seenSpans = new Set();
  const MAX_RUNS = 8;
  // The last payload's rows, kept so a re-sort does not need a new fetch.
  let byModelRows = [];
  let byRouteRows = [];

  // ------------------------------------------------------------------
  // DOM helpers
  // ------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  // Several of these elements are polite live regions. Assigning an identical
  // string still replaces the text node, and a replaced text node is what a
  // screen reader announces — so an idle poll would re-read every card every
  // five seconds. Write only when the value actually changed.
  function setText(id, text) {
    const el = $(id);
    if (!el) return;
    const next = String(text);
    if (el.textContent === next) return;
    el.textContent = next;
  }

  // Same rule for a block rebuilt from markup: skip the write when the markup
  // is byte-identical to what is already there.
  function setHtml(el, html) {
    if (!el) return false;
    if (el.innerHTML === html) return false;
    el.innerHTML = html;
    return true;
  }

  // One polite live region for every announcement this page makes; the shared
  // keyboard layer owns it. Without webui.js the page still works, silently.
  function announce(message) {
    if (webui && webui.announce) webui.announce(message);
  }

  function setStatus(online) {
    const dot = $("status-dot");
    if (dot) {
      const cls = "status-dot " + (online ? "ok" : "err");
      if (dot.className !== cls) dot.className = cls;
    }
    // #status is a polite live region; only a real change is worth announcing.
    setText("status-text", online ? "Connected" : "Offline");
  }

  function showAuthBanner() {
    const el = $("auth-banner");
    if (!el) return;
    el.textContent = "Dashboard data requires authentication. Restart the server with "
      + "EFFGEN_PUBLIC_DASHBOARD=1 for local viewing, or supply an API key "
      + "(Authorization: Bearer <key> or X-API-Key: <key>).";
    el.hidden = false;
  }

  function hideAuthBanner() {
    const el = $("auth-banner");
    if (el) el.hidden = true;
  }

  // Escapes for both text and attribute positions: names reaching the page come
  // from agent configs and model output, so quotes must not end an attribute.
  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Quote a value for use inside a CSS attribute selector.
  function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === "function") {
      return window.CSS.escape(String(value));
    }
    return String(value).replace(/["\\]/g, "\\$&");
  }

  function fmt(v) {
    if (v == null) return "—";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
  }

  function fmtCost(v) {
    if (v == null) return "—";
    if (v === 0) return "$0.00";
    if (v < 0.01) return "$" + v.toFixed(6);
    return "$" + v.toFixed(4);
  }

  function fmtSeconds(v) {
    return v == null ? "—" : v.toFixed(3) + "s";
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------
  // The effective theme: an explicit choice (data-theme attribute) if set,
  // otherwise the OS preference the CSS is already following.
  function currentTheme() {
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr === "dark" || attr === "light") return attr;
    const prefersLight = window.matchMedia
      && window.matchMedia("(prefers-color-scheme: light)").matches;
    return prefersLight ? "light" : "dark";
  }

  function syncThemeButton(announceChange) {
    const theme = currentTheme();
    const icon = $("theme-icon");
    if (icon) icon.textContent = theme === "dark" ? "☾" : "☀";
    const btn = $("theme-btn");
    if (btn) btn.setAttribute("aria-label",
      theme === "dark" ? "Dark theme active. Switch to light theme."
                       : "Light theme active. Switch to dark theme.");
    // A change the viewer made is announced; the initial sync stays silent.
    if (announceChange) setText("theme-status", theme === "dark" ? "Dark theme" : "Light theme");
  }

  // Set an explicit theme (overriding the OS scheme). Persisted only on a click.
  function applyTheme(theme, persist) {
    document.documentElement.setAttribute("data-theme", theme);
    if (persist && webui) webui.storeTheme(theme);
    syncThemeButton(persist);
    drawChart();
  }

  function initTheme() {
    // One preference key is shared by every effGen web surface, so a choice
    // made here also applies on the playground.
    const stored = webui ? webui.readStoredTheme() : null;
    if (stored === "dark" || stored === "light") {
      // Honor the user's saved choice.
      applyTheme(stored, false);
    } else {
      // No stored choice: leave first paint to the CSS/OS scheme, just label
      // the toggle to match. Follow live OS-scheme changes until a choice is made.
      syncThemeButton();
      if (window.matchMedia) {
        window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
          if (!document.documentElement.getAttribute("data-theme")) {
            syncThemeButton();
            drawChart();
          }
        });
      }
    }
    const btn = $("theme-btn");
    if (btn) btn.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      applyTheme(next, true);
    });
  }

  // ------------------------------------------------------------------
  // Latency chart (canvas 2D — no external library)
  // ------------------------------------------------------------------
  function pushLatency(ts, value) {
    latencyHistory.push({ ts, value });
    if (latencyHistory.length > MAX_CHART_POINTS) {
      latencyHistory.shift();
    }
    drawChart();
  }

  function drawChart() {
    const canvas = $("latency-chart");
    const empty = $("chart-empty");
    if (!canvas || !canvas.getContext) return;

    const points = latencyHistory.map(p => p.value).filter(v => v != null);
    if (empty) empty.hidden = points.length > 0;
    if (!points.length) {
      const ctx0 = canvas.getContext("2d");
      ctx0.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    // Scale the backing store to device pixels for a crisp line.
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const cssH = parseInt(canvas.getAttribute("height"), 10) || 150;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 44, padR = 8, padT = 10, padB = 20;
    const plotW = cssW - padL - padR;
    const plotH = cssH - padT - padB;
    const maxV = Math.max.apply(null, points) * 1.15 || 1;
    const n = points.length;

    const grid = cssVar("--border") || "#2c3150";
    const axis = cssVar("--text-muted") || "#8892a4";
    const line = cssVar("--accent") || "#818cf8";

    // Horizontal grid lines + y labels
    ctx.strokeStyle = grid;
    ctx.fillStyle = axis;
    ctx.lineWidth = 1;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    const rows = 4;
    for (let i = 0; i <= rows; i++) {
      const y = padT + (plotH * i) / rows;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      const val = maxV * (1 - i / rows);
      ctx.fillText(val.toFixed(2) + "s", 4, y);
    }

    const xFor = (i) => n <= 1 ? padL + plotW : padL + (plotW * i) / (n - 1);
    const yFor = (v) => padT + plotH * (1 - v / maxV);

    // Filled area under the curve
    ctx.beginPath();
    ctx.moveTo(xFor(0), yFor(points[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(xFor(i), yFor(points[i]));
    ctx.lineTo(xFor(n - 1), padT + plotH);
    ctx.lineTo(xFor(0), padT + plotH);
    ctx.closePath();
    ctx.fillStyle = hexToRgba(line, 0.15);
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(xFor(0), yFor(points[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(xFor(i), yFor(points[i]));
    ctx.strokeStyle = line;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Points
    ctx.fillStyle = line;
    for (let i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.arc(xFor(i), yFor(points[i]), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function hexToRgba(hex, alpha) {
    const h = hex.replace("#", "");
    if (h.length !== 6) return "rgba(129,140,248," + alpha + ")";
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ------------------------------------------------------------------
  // Render helpers
  // ------------------------------------------------------------------
  function renderCards(data) {
    const m = data.metrics || {};
    setText("val-requests", fmt(m.total_requests));
    setText("val-errors",   fmt(m.total_errors));
    setText("val-latency",  fmtSeconds(m.avg_latency_s));
    setText("val-tokens",   fmt(m.total_tokens));

    // Real cost — sum of per-run cost_usd; never a fabricated flat rate.
    if (m.cost_usd == null) {
      setText("val-cost", "—");
      setText("sub-cost", (m.unpriced_runs ? m.unpriced_runs + " unpriced run(s)" : "no priced runs"));
    } else {
      setText("val-cost", fmtCost(m.cost_usd));
      const parts = [];
      if (m.priced_runs) parts.push(m.priced_runs + " priced");
      if (m.unpriced_runs) parts.push(m.unpriced_runs + " unpriced");
      setText("sub-cost", parts.join(" · "));
    }

    // HTTP error context for the errors card.
    const http4 = m.http_client_errors || 0;
    const http5 = m.http_server_errors || 0;
    setText("sub-errors", (http4 || http5)
      ? `HTTP ${http4} 4xx · ${http5} 5xx` : "");

    const version = data.version;
    if (version) setText("header-version", "v" + version);

    const ts = new Date().toLocaleTimeString([], { hour12: false });
    if (m.avg_latency_s != null) {
      pushLatency(ts, m.avg_latency_s);
    }
  }

  function renderSLO(data) {
    const slo = data.slo || {};

    // p99 latency burn (0→1, 1 = at the latency threshold), driven by the true p99.
    const p99 = Math.min((slo.p99_latency_burn || 0), 1);
    setSLOBar("slo-p99", "slo-p99-pct", "slo-p99-img", p99, false, "p99 latency burn");

    const errRate = Math.min((slo.error_rate_burn || 0), 1);
    setSLOBar("slo-err", "slo-err-pct", "slo-err-img", errRate, false, "error rate burn");

    const avail = Math.min((slo.availability != null ? slo.availability : 1), 1);
    setSLOBar("slo-avail", "slo-avail-pct", "slo-avail-img", avail, true, "availability");

    const detail = [];
    if (slo.p50_latency_s != null) detail.push("p50 " + slo.p50_latency_s.toFixed(2) + "s");
    if (slo.p95_latency_s != null) detail.push("p95 " + slo.p95_latency_s.toFixed(2) + "s");
    if (slo.p99_latency_s != null) detail.push("p99 " + slo.p99_latency_s.toFixed(2) + "s");
    if (slo.latency_threshold_s != null) detail.push("target " + slo.latency_threshold_s + "s");
    setText("slo-detail", detail.length ? detail.join("  ·  ") : "No latency samples yet.");
  }

  function setSLOBar(barId, pctId, imgId, ratio, invert, label) {
    const bar = $(barId);
    const pct = $(pctId);
    const img = $(imgId);
    const percent = (ratio * 100).toFixed(1) + "%";
    if (bar) bar.style.width = percent;
    if (pct) pct.textContent = percent;
    if (img) img.setAttribute("aria-label", label + ": " + percent);
    if (bar && !invert) {
      bar.style.background = ratio > 0.9
        ? cssVar("--err") : ratio > 0.5 ? cssVar("--warn") : cssVar("--accent");
    }
  }

  // ------------------------------------------------------------------
  // Sortable tables (W3C APG pattern: a button in the <th>, aria-sort on
  // exactly one header at a time, moved rather than duplicated on change)
  // ------------------------------------------------------------------
  // Sort state lives outside the renderers so a five-second poll re-renders
  // through the same view instead of resetting the user's choice.
  const sortState = {
    "by-model-table": { key: "calls", dir: -1 },
    "by-route-table": { key: "error_rate", dir: -1 },
  };

  function sortRows(tableId, rows) {
    const state = sortState[tableId];
    if (!state) return rows.slice();
    const { key, dir } = state;
    return rows.slice().sort((a, b) => {
      const av = a[key];
      const bv = b[key];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
  }

  // Reflect the current sort on the header cells and keep focus where it was.
  function syncSortHeaders(tableId) {
    const table = $(tableId);
    const state = sortState[tableId];
    if (!table || !state) return;
    table.querySelectorAll("th[data-sort]").forEach((th) => {
      if (th.getAttribute("data-sort") === state.key) {
        th.setAttribute("aria-sort", state.dir < 0 ? "descending" : "ascending");
      } else {
        th.removeAttribute("aria-sort");
      }
    });
  }

  function bindSortHeaders(tableId, rerender) {
    const table = $(tableId);
    const state = sortState[tableId];
    if (!table || !state) return;
    table.querySelectorAll("button.sort-btn[data-sort]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-sort");
        if (state.key === key) state.dir = -state.dir;
        else { state.key = key; state.dir = key === "model" || key === "provider" || key === "route" ? 1 : -1; }
        rerender();
        syncSortHeaders(tableId);
        announce(`Sorted by ${btn.textContent.trim()}, `
          + (state.dir < 0 ? "descending" : "ascending"));
        btn.focus({ preventScroll: true });
      });
    });
    syncSortHeaders(tableId);
  }

  function renderByModel(data) {
    if (data) byModelRows = data.by_model || [];
    const tbody = $("by-model-tbody");
    if (!tbody) return;
    const note = $("by-model-unattributed");
    if (note && data) {
      const spare = data.unattributed_cost_usd;
      if (typeof spare === "number" && spare > 0) {
        note.textContent = "Plus " + fmtCost(spare)
          + " recorded on runs that could not be matched to a row above.";
        note.hidden = false;
      } else {
        note.hidden = true;
      }
    }
    if (!byModelRows.length) {
      setHtml(tbody, '<tr><td colspan="8" class="empty-row">No model calls yet</td></tr>');
      return;
    }
    setHtml(tbody, sortRows("by-model-table", byModelRows).map(r => `
      <tr>
        <td>${esc(r.model || "—")}</td>
        <td>${esc(r.provider || "—")}</td>
        <td class="num">${fmt(r.calls)}</td>
        <td class="num">${(r.error_rate != null ? (r.error_rate * 100).toFixed(1) + "%" : "—")}</td>
        <td class="num">${fmtSeconds(r.p95_latency_s)}</td>
        <td>${outcomeCell(r)}</td>
        <td class="num">${fmt(r.input_tokens)} / ${fmt(r.output_tokens)}</td>
        <td class="num">${fmtCost(r.cost_usd)}</td>
      </tr>`).join(""));
  }

  // The dominant failure class for a model, with the same remediation sentence
  // the CLI prints for it — shown as a tooltip and, so it is not mouse-only, as
  // text only a screen reader reads.
  function outcomeCell(row) {
    if (!row.top_error) return '<span class="outcome-ok">ok</span>';
    const count = (row.outcomes && row.outcomes[row.top_error]) || row.errors || 0;
    const hint = row.top_error_hint || "";
    return `<span class="outcome-err"${hint ? ` title="${esc(hint)}"` : ""}>`
      + `${esc(row.top_error)} · ${fmt(count)}</span>`
      + (hint ? `<span class="eff-sr-only"> ${esc(hint)}</span>` : "");
  }

  function renderByRoute(data) {
    if (data) byRouteRows = data.by_route || [];
    const tbody = $("by-route-tbody");
    if (!tbody) return;
    if (!byRouteRows.length) {
      setHtml(tbody, '<tr><td colspan="6" class="empty-row">No HTTP responses recorded yet</td></tr>');
      return;
    }
    setHtml(tbody, sortRows("by-route-table", byRouteRows).map(r => {
      const classes = {};
      Object.keys(r.by_status || {}).forEach((code) => {
        const cls = String(code).charAt(0) + "xx";
        classes[cls] = (classes[cls] || 0) + r.by_status[code];
      });
      const breakdown = Object.keys(classes).sort()
        .map((cls) => `${cls} ${fmt(classes[cls])}`).join(" · ") || "—";
      return `
      <tr>
        <td class="cat-id">${esc(r.route || "—")}</td>
        <td>${esc(r.method || "—")}</td>
        <td class="num">${fmt(r.requests)}</td>
        <td class="num">${fmt(r.errors)}</td>
        <td class="num">${r.error_rate != null ? (r.error_rate * 100).toFixed(1) + "%" : "—"}</td>
        <td>${esc(breakdown)}</td>
      </tr>`;
    }).join(""));
  }

  function renderByStatus(data) {
    const chips = $("status-chips");
    if (!chips) return;
    const byStatus = data.by_status || {};
    const codes = Object.keys(byStatus);
    if (!codes.length) {
      chips.innerHTML = '<span class="empty-row">No HTTP responses recorded yet</span>';
      return;
    }
    chips.innerHTML = codes.sort().map(code => {
      const cls = "s-" + (code.charAt(0) || "2") + "xx";
      const count = byStatus[code];
      // The glyphs read as one run of digits ("2004140134…") when the chips are
      // spoken in order, and the color is the only thing stating the class. The
      // visible cells stay as they are; a hidden sentence carries the meaning.
      const spoken = `${code} ${statusClassLabel(code)} · `
        + `${count} ${count === 1 ? "response" : "responses"}`;
      return `<span class="status-chip ${cls}">`
        + `<span class="chip-code" aria-hidden="true">${esc(code)}</span>`
        + `<span class="chip-count" aria-hidden="true">${fmt(count)}</span>`
        + `<span class="eff-sr-only">${esc(spoken)}</span></span>`;
    }).join("");
  }

  // Plain-language name for a status class, so the chip does not rely on color.
  function statusClassLabel(code) {
    const lead = String(code).charAt(0);
    if (lead === "2") return "success";
    if (lead === "3") return "redirect";
    if (lead === "4") return "client error";
    if (lead === "5") return "server error";
    return "response";
  }

  function renderRuns(data) {
    const runs = data.recent_runs || [];
    const tbody = $("runs-tbody");
    if (!tbody) return;
    if (!runs.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No runs yet</td></tr>';
      return;
    }
    tbody.innerHTML = runs.map(r => `
      <tr>
        <td>${esc(r.ts || "—")}</td>
        <td>${esc(r.model || "—")}</td>
        <td>${fmt(r.input_tokens)} / ${fmt(r.output_tokens)}</td>
        <td>${fmtCost(r.cost_usd)}</td>
        <td>${fmtSeconds(r.duration_s)}</td>
        <td><span class="badge ${r.error ? "badge-err" : "badge-ok"}">${r.error ? "error" : "ok"}</span></td>
      </tr>`).join("");
  }

  function renderMetrics(data) {
    const raw = data.raw_metrics || {};
    const tbody = $("metrics-tbody");
    if (!tbody) return;
    const entries = Object.entries(raw);
    if (!entries.length) {
      tbody.innerHTML = '<tr><td colspan="2" class="empty-row">No metrics</td></tr>';
      return;
    }
    tbody.innerHTML = entries
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${typeof v === "number" ? v.toLocaleString() : esc(String(v))}</td></tr>`)
      .join("");
  }

  // ------------------------------------------------------------------
  // Model catalog browser
  // ------------------------------------------------------------------
  const CAT_PAGE_SIZE = 25;
  let catalogAll = [];        // every record from /dashboard/catalog.json
  let catalogView = [];       // filtered + sorted
  let catalogPage = 0;
  let catalogLoaded = false;

  function fmtPrice(v) {
    // A published nonzero rate shows as $<n>; null is unknown; 0 with no rate
    // is unpriced rather than a fabricated $0 (mirrors the CLI label).
    if (v == null) return null;
    if (v === 0) return null;
    return "$" + (Math.round(v * 1000) / 1000);
  }

  function priceCells(rec) {
    const pin = fmtPrice(rec.price_in_per_1m);
    const pout = fmtPrice(rec.price_out_per_1m);
    if (pin || pout) return [pin || "?", pout || "?"];
    const label = rec.free_tier ? "free" : (rec.is_priced ? "metered" : "unpriced");
    return [label, label];
  }

  function catalogFilterSort() {
    const q = ($("cat-search").value || "").toLowerCase().trim();
    const prov = $("cat-provider").value || "";
    const sort = $("cat-sort").value || "provider";
    const wantTools = $("cat-tools").checked;
    const wantVision = $("cat-vision").checked;
    const wantAudio = $("cat-audio").checked;
    const wantFree = $("cat-free").checked;
    const minCtx = parseInt($("cat-min-context").value, 10);

    let rows = catalogAll.filter((r) => {
      if (prov && r.provider !== prov) return false;
      if (wantTools && !r.supports_tools) return false;
      if (wantVision && !r.supports_vision) return false;
      if (wantAudio && !r.supports_audio) return false;
      if (wantFree && !r.free_tier) return false;
      if (!isNaN(minCtx) && (r.context_window || 0) < minCtx) return false;
      if (q) {
        const hay = (r.id + " " + (r.family || "") + " " + r.provider).toLowerCase();
        if (hay.indexOf(q) === -1) return false;
      }
      return true;
    });

    const big = Number.POSITIVE_INFINITY;
    const numIn = (r) => (r.price_in_per_1m == null ? big : r.price_in_per_1m);
    const numOut = (r) => (r.price_out_per_1m == null ? big : r.price_out_per_1m);
    const cmp = {
      provider: (a, b) => (a.provider + a.id).localeCompare(b.provider + b.id),
      id: (a, b) => a.id.localeCompare(b.id),
      context: (a, b) => (a.context_window || 0) - (b.context_window || 0),
      "max-out": (a, b) => (a.max_output || 0) - (b.max_output || 0),
      "price-in": (a, b) => numIn(a) - numIn(b),
      "price-out": (a, b) => numOut(a) - numOut(b),
    }[sort] || ((a, b) => 0);
    rows.sort(cmp);

    catalogView = rows;
    catalogPage = 0;
    renderCatalogPage();
  }

  function renderCatalogPage() {
    const tbody = $("catalog-tbody");
    if (!tbody) return;
    const total = catalogView.length;
    const pages = Math.max(1, Math.ceil(total / CAT_PAGE_SIZE));
    if (catalogPage >= pages) catalogPage = pages - 1;
    const start = catalogPage * CAT_PAGE_SIZE;
    const slice = catalogView.slice(start, start + CAT_PAGE_SIZE);

    if (!slice.length) {
      tbody.innerHTML = '<tr><td colspan="9" class="empty-row">'
        + (catalogAll.length ? "No models match those filters"
                             : "Catalog unavailable") + "</td></tr>";
    } else {
      tbody.innerHTML = slice.map((r) => {
        const [pin, pout] = priceCells(r);
        // aria-label on a span with no role is ignored, so the check glyph had
        // no accessible name at all: the cell has to carry real text.
        const check = (on) => on
          ? '<span class="cat-yes"><span class="eff-sr-only">yes</span>'
            + '<span aria-hidden="true">✓</span></span>'
          : '<span class="eff-sr-only">no</span>';
        return `<tr>
          <td>${esc(r.provider)}</td>
          <td class="cat-id">${esc(r.id)}</td>
          <td class="num">${r.context_window ? r.context_window.toLocaleString() : "—"}</td>
          <td class="num">${r.max_output ? r.max_output.toLocaleString() : "—"}</td>
          <td class="num">${esc(pin)}</td>
          <td class="num">${esc(pout)}</td>
          <td>${check(r.supports_tools)}</td>
          <td>${check(r.supports_vision)}</td>
          <td>${check(r.free_tier)}</td>
        </tr>`;
      }).join("");
    }

    const from = total ? start + 1 : 0;
    const to = Math.min(start + CAT_PAGE_SIZE, total);
    setText("cat-page-info", total
      ? `${from}–${to} of ${total} (of ${catalogAll.length})`
      : `0 of ${catalogAll.length}`);
    const prev = $("cat-prev");
    const next = $("cat-next");
    if (prev) prev.disabled = catalogPage <= 0;
    if (next) next.disabled = catalogPage >= pages - 1;
  }

  function populateProviderSelect() {
    const sel = $("cat-provider");
    if (!sel) return;
    const provs = Array.from(new Set(catalogAll.map((r) => r.provider))).sort();
    // Keep the leading "all" option; append the discovered providers once.
    provs.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      sel.appendChild(opt);
    });
  }

  async function loadCatalog() {
    if (catalogLoaded) return;
    try {
      const resp = await fetch("/dashboard/catalog.json", { cache: "no-store" });
      if (resp.status === 401) {
        setText("catalog-sub", "Catalog requires authentication (see the banner above).");
        return;
      }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      catalogAll = (data.data || []).slice();
      catalogLoaded = true;
      populateProviderSelect();
      const provCount = (data.providers || []).length;
      const verified = (data.providers || [])
        .map((p) => p.verified_on).filter(Boolean).sort();
      const asOf = verified.length ? verified[0] : "unknown";
      setText("catalog-sub",
        `${catalogAll.length} models across ${provCount} providers · `
        + `pricing from catalog snapshot (verified ${asOf}) · `
        + `run “effgen models refresh” to update`);
      catalogFilterSort();
    } catch (err) {
      setText("catalog-sub", "Catalog unavailable.");
      console.warn("[effGen dashboard] catalog fetch failed:", err);
    }
  }

  function initCatalog() {
    const ids = ["cat-search", "cat-provider", "cat-sort", "cat-tools",
                 "cat-vision", "cat-audio", "cat-free", "cat-min-context"];
    ids.forEach((id) => {
      const el = $(id);
      if (!el) return;
      const evt = (el.tagName === "INPUT" && (el.type === "search" || el.type === "number"))
        ? "input" : "change";
      el.addEventListener(evt, catalogFilterSort);
    });
    const prev = $("cat-prev");
    if (prev) prev.addEventListener("click", () => {
      if (catalogPage > 0) { catalogPage--; renderCatalogPage(); }
    });
    const next = $("cat-next");
    if (next) next.addEventListener("click", () => {
      catalogPage++; renderCatalogPage();
    });
  }

  // ------------------------------------------------------------------
  // Span stream
  // ------------------------------------------------------------------
  function spanKind(name) {
    const n = String(name || "");
    if (n.indexOf("model.call") !== -1) return { kind: "model", label: n.replace(/^effgen\.model\.call\s*/, "") };
    if (n.indexOf("tool.call") !== -1) return { kind: "tool", label: n.replace(/^effgen\.tool\.call\s*/, "") };
    if (n.indexOf("router") !== -1) return { kind: "router", label: n.replace(/^effgen\.router\.\w+\s*/, "") };
    if (n.indexOf("agent.run") !== -1) return { kind: "run", label: n.replace(/^effgen\.agent\.run\s*/, "") };
    if (n.indexOf("agent.iteration") !== -1) return { kind: "iter", label: n };
    return { kind: "span", label: n };
  }

  function appendSpan(span) {
    if (spanPaused) return;
    const stream = $("span-stream");
    if (!stream) return;

    spanCount++;
    setText("span-count", spanCount + " spans");

    const parsed = spanKind(span.name);
    const el = document.createElement("div");
    el.className = "span-entry" + (span.error ? " is-error" : "");
    const ts   = span.ts ? `<span class="span-ts">${esc(span.ts)}</span>` : "";
    const kind = `<span class="span-kind">${esc(parsed.kind)}</span>`;
    const name = `<span class="span-name">${esc(parsed.label || "span")}</span>`;
    const dur  = span.duration_ms != null
      ? `<span class="span-dur">${span.duration_ms.toFixed(1)}ms</span>` : "";
    const err  = span.error ? `<span class="span-err">[${esc(span.error)}]</span>` : "";
    el.innerHTML = `${ts}${kind}${name}${dur}${err}`;
    stream.appendChild(el);

    // Trim old spans
    while (stream.children.length > MAX_SPANS) {
      stream.removeChild(stream.firstChild);
    }
    stream.scrollTop = stream.scrollHeight;

    collectForWaterfall(span);
  }

  // ------------------------------------------------------------------
  // Per-run waterfall
  // ------------------------------------------------------------------
  function spanSignature(span) {
    return [span.run_id, span.name, span.offset_ms, span.duration_ms].join("|");
  }

  function collectForWaterfall(span) {
    const runId = span.run_id;
    if (!runId) return;
    // A run that streams live may replay buffered spans on reconnect; dedupe.
    const sig = spanSignature(span);
    if (seenSpans.has(sig)) return;
    seenSpans.add(sig);

    if (!runs.has(runId)) {
      runs.set(runId, { id: runId, ts: span.ts, spans: [] });
      // Bound memory: drop the oldest run once we exceed the cap.
      while (runs.size > MAX_RUNS) {
        const oldest = runs.keys().next().value;
        runs.delete(oldest);
      }
    }
    runs.get(runId).spans.push({
      kind: spanKind(span.name).kind,
      label: spanKind(span.name).label || span.name,
      offset: Number(span.offset_ms) || 0,
      duration: Number(span.duration_ms) || 0,
      error: !!span.error,
    });
    renderWaterfall();
  }

  function renderWaterfall() {
    const host = $("waterfall");
    if (!host) return;
    const list = Array.from(runs.values()).reverse();  // newest first
    if (!list.length) {
      host.innerHTML = '<p class="empty-row" id="waterfall-empty">No runs recorded yet.</p>';
      return;
    }
    const rows = list.map((run) => {
      // A run's total span is the agent.run bar; scale everything to it.
      const total = Math.max(
        1,
        ...run.spans.map((s) => s.offset + s.duration)
      );
      const bars = run.spans
        .slice()
        .sort((a, b) => a.offset - b.offset)
        .map((s) => {
          const left = (s.offset / total) * 100;
          const width = Math.max(0.6, (s.duration / total) * 100);
          const cls = s.error ? "wf-bar is-error" : "wf-bar wf-" + s.kind;
          const title = `${s.kind}: ${esc(s.label)} — ${s.duration.toFixed(1)}ms @ +${s.offset.toFixed(0)}ms`;
          return `<div class="wf-track"><span class="wf-track-label">${esc(s.kind)}</span>`
            + `<div class="wf-lane"><div class="${cls}" style="left:${left}%;width:${width}%" `
            + `title="${title}"><span class="wf-bar-label">${esc(s.label)} · ${s.duration.toFixed(0)}ms</span></div></div></div>`;
        })
        .join("");
      return `<div class="wf-run"><div class="wf-run-head">`
        + `<span class="wf-run-id">run ${esc(run.id)}</span>`
        + `<span class="wf-run-total">${total.toFixed(0)}ms · ${run.spans.length} spans</span></div>`
        + `${bars}</div>`;
    }).join("");
    host.innerHTML = rows;
  }

  function startSSE() {
    if (eventSource) return;
    try {
      eventSource = new EventSource("/dashboard/spans");
      eventSource.onmessage = (e) => {
        try { appendSpan(JSON.parse(e.data)); } catch { /* ignore */ }
      };
      eventSource.onerror = () => {
        eventSource.close();
        eventSource = null;
        // Retry after 10s
        setTimeout(startSSE, 10000);
      };
    } catch {
      // SSE not supported or path not found
    }
  }

  // ------------------------------------------------------------------
  // History (stored runs + saved sessions)
  // ------------------------------------------------------------------
  let historyRuns = [];
  let historyPayload = {};
  let openRunId = null;

  function fmtWhen(ts) {
    if (!ts) return "—";
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts).slice(0, 16);
    return d.toLocaleString(undefined, {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
    });
  }

  function clip(text, width) {
    if (text == null || text === "") return "—";
    const flat = String(text).replace(/\s+/g, " ").trim();
    return flat.length > width ? flat.slice(0, width - 1) + "…" : flat;
  }

  async function loadHistory() {
    const params = new URLSearchParams({ limit: "50" });
    const search = $("hist-search");
    const status = $("hist-status");
    if (search && search.value.trim()) params.set("search", search.value.trim());
    if (status && status.value) params.set("status", status.value);
    try {
      const resp = await fetch("/dashboard/history.json?" + params.toString(),
                               { cache: "no-store" });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      renderHistory(await resp.json());
    } catch (err) {
      setText("history-sub", "History unavailable.");
      console.warn("[effGen dashboard] history fetch failed:", err);
    }
  }

  // The run id of the disclosure button that currently holds focus, or null.
  function focusedHistoryRunId() {
    const active = document.activeElement;
    if (!active || !active.closest || !active.closest("#history-tbody")) return null;
    const index = active.getAttribute("data-run-index");
    if (index == null) return null;
    const run = historyRuns[Number(index)];
    return (run && run.run_id) || null;
  }

  // Put focus back on the same run's button after a rebuild. When that run is no
  // longer listed, focus moves to the panel heading rather than to <body>, so
  // the reader stays in the section they were in.
  function restoreHistoryFocus(runId) {
    if (!runId) return;
    const tbody = $("history-tbody");
    if (!tbody) return;
    const buttons = Array.from(tbody.querySelectorAll("button[data-run-index]"));
    const match = buttons.find((btn) => {
      const run = historyRuns[Number(btn.dataset.runIndex)];
      return run && run.run_id === runId;
    });
    if (match) {
      match.focus({ preventScroll: true });
      return;
    }
    const panel = $("panel-history");
    if (panel && panel.focus) panel.focus({ preventScroll: true });
  }

  function renderHistory(data) {
    // A poll rebuilds this table every five seconds. Note which run's disclosure
    // button holds focus before the rows are replaced, and give focus back
    // afterwards — otherwise a keyboard user is dropped to the top of the
    // document mid-interaction. The topology graph does the same thing.
    const focusedRunId = focusedHistoryRunId();
    historyPayload = data || {};
    historyRuns = data.runs || [];
    const sessions = data.sessions || [];
    const where = data.runs_dir ? " Stored in " + data.runs_dir + "." : "";
    const note = data.persisted === false
      ? " Persistence is off (EFFGEN_RUN_HISTORY=0); only this process's runs are listed."
      : where;
    setText("history-sub",
      historyRuns.length + " run(s), " + sessions.length + " session(s)." + note);

    const tbody = $("history-tbody");
    if (tbody) {
      tbody.innerHTML = historyRuns.length
        ? historyRuns.map((r, i) => `
          <tr class="history-row${r.run_id && r.run_id === openRunId ? " is-open" : ""}">
            <td>${esc(fmtWhen(r.ts))}</td>
            <td>${esc(clip(r.model, 28))}</td>
            <td><button type="button" class="link-btn" data-run-index="${i}"
                aria-expanded="${r.run_id && r.run_id === openRunId ? "true" : "false"}"
                >${esc(clip(r.task || r.run_id || "(no task recorded)", 60))}</button></td>
            <td class="num">${fmtCost(r.cost_usd)}</td>
            <td class="num">${fmtSeconds(r.duration_s)}</td>
            <td><span class="badge ${r.status === "error" ? "badge-err" : "badge-ok"}">${esc(r.status || "ok")}</span></td>
          </tr>`).join("")
        : '<tr><td colspan="6" class="empty-row">No runs recorded yet</td></tr>';
      tbody.querySelectorAll("button[data-run-index]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const run = historyRuns[Number(btn.dataset.runIndex)];
          openRunId = (run && run.run_id === openRunId) ? null : (run && run.run_id) || null;
          renderRunDetail(openRunId ? run : null);
        });
      });
      syncRunDisclosure();
      restoreHistoryFocus(focusedRunId);
    }

    const stbody = $("history-sessions-tbody");
    if (stbody) {
      stbody.innerHTML = sessions.length
        ? sessions.map((s) => `
          <tr>
            <td>${esc(s.session_id)}</td>
            <td class="num">${fmt(s.messages)}</td>
            <td>${esc(clip(s.model, 28))}</td>
            <td class="num">${fmtCost(s.cost_usd)}</td>
            <td>${esc(fmtWhen(s.updated_at))}</td>
          </tr>`).join("")
        : '<tr><td colspan="5" class="empty-row">No saved sessions</td></tr>';
    }
  }

  // Keep each row's disclosure state on the row itself, without rebuilding the
  // table — a rebuild would drop the keyboard focus mid-interaction.
  function syncRunDisclosure() {
    const tbody = $("history-tbody");
    if (!tbody) return;
    tbody.querySelectorAll("button[data-run-index]").forEach((btn) => {
      const run = historyRuns[Number(btn.dataset.runIndex)];
      const open = !!(run && run.run_id && run.run_id === openRunId);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      const row = btn.closest("tr");
      if (row) row.classList.toggle("is-open", open);
    });
  }

  // The steps a stored run took, in order. History keeps each run's step
  // kinds rather than its text, so the drill-in shows the shape of the
  // conversation — frame, question, thoughts, calls, results, answer — rather
  // than a flattened string. The full text comes from `effgen run
  // --show-thread` or `--card` at run time.
  function renderRunSteps(run) {
    const kinds = Array.isArray(run.thread_kinds) ? run.thread_kinds : [];
    if (!kinds.length) return "";
    const compactions = Number(run.compactions || 0);
    const note =
      kinds.length +
      " step" +
      (kinds.length === 1 ? "" : "s") +
      (compactions ? ", " + compactions + " compaction" + (compactions === 1 ? "" : "s") : "") +
      ". Kinds only — run with <code>--show-thread</code> for the text.";
    return (
      "<h3>Steps</h3>" +
      '<ol class="run-steps">' +
      kinds.map((k) => `<li>${esc(k)}</li>`).join("") +
      "</ol>" +
      `<p class="run-export-note">${note}</p>`
    );
  }

  function renderRunDetail(run) {
    const box = $("history-detail");
    if (!box) return;
    if (!run) {
      box.hidden = true;
      box.innerHTML = "";
      syncRunDisclosure();
      return;
    }
    const rows = [
      ["Run id", run.run_id || "—"],
      ["When", fmtWhen(run.ts)],
      ["Model", (run.model || "—") + (run.provider ? " (" + run.provider + ")" : "")],
      ["Agent", run.agent || "—"],
      ["Session", run.session_id || "—"],
      ["Tokens", (run.input_tokens || 0) + " in / " + (run.output_tokens || 0) + " out"],
      ["Cost", fmtCost(run.cost_usd)],
      ["Duration", fmtSeconds(run.duration_s)],
    ];
    // The history store keeps a truncated answer and no step trace, so the
    // card this offers is a summary and says so. The full card — answer, tool
    // trace, sources — comes from `effgen run --card` at run time.
    const exportCmd = run.run_id
      ? `effgen runs show ${run.run_id} --card run-${run.run_id}.html`
      : "";
    const exportBlock = exportCmd
      ? '<div class="run-export">' +
        `<button type="button" id="run-export-copy" data-cmd="${esc(exportCmd)}">` +
        "Copy summary-card command</button>" +
        '<span class="run-export-note">Writes a summary card from stored history ' +
        "(truncated answer, no step trace). Use <code>effgen run --card</code> at " +
        "run time for the full answer, trace and sources.</span></div>"
      : "";
    box.innerHTML =
      '<dl class="run-detail">' +
      rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("") +
      "</dl>" +
      exportBlock +
      (run.task ? `<h3>Task</h3><pre class="run-text">${esc(run.task)}</pre>` : "") +
      (run.output ? `<h3>Answer</h3><pre class="run-text">${esc(run.output)}</pre>` : "") +
      (run.error ? `<h3>Error</h3><pre class="run-text run-error">${esc(run.error)}</pre>` : "") +
      renderRunSteps(run);
    box.hidden = false;
    const copyBtn = $("run-export-copy");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        const cmd = copyBtn.dataset.cmd || "";
        const done = (ok) => {
          copyBtn.textContent = ok ? "Copied" : "Press Ctrl+C to copy";
          window.setTimeout(() => {
            copyBtn.textContent = "Copy summary-card command";
          }, 2000);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(cmd).then(() => done(true), () => done(false));
        } else {
          done(false);
        }
      });
    }
    syncRunDisclosure();
  }

  function initHistory() {
    ["hist-search", "hist-status"].forEach((id) => {
      const el = $(id);
      if (!el) return;
      el.addEventListener(el.tagName === "SELECT" ? "change" : "input", loadHistory);
    });
  }

  // ------------------------------------------------------------------
  // Multi-agent topology graph
  //
  // Draws one team or workflow execution as a node-link graph: agents (the
  // manager marked apart) and the tools they reached as nodes, delegation,
  // handoff, peer and tool-use links as edges. The SVG is built here — no
  // graph library and no external asset. Status is carried by a glyph and a
  // text label as well as color, and the layout is derived from the edges, so
  // the same execution draws the same way on every refresh.
  // ------------------------------------------------------------------
  let topologyData = [];
  let topoSelected = "";
  let topoNodeId = null;
  // The graph is one tab stop: the node carrying it is remembered here and the
  // arrow keys move between nodes, so a large team does not add N tab stops.
  let topoFocusId = null;

  const TOPO_GLYPH = { ok: "●", running: "◐", skipped: "⊘", error: "✗" };
  const TOPO_COL_W = 220;
  const TOPO_ROW_H = 78;
  const TOPO_NODE_W = 158;
  const TOPO_NODE_H = 46;

  function reducedMotion() {
    try {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch {
      return false;
    }
  }

  function topoColumns(nodes, edges) {
    // Longest-path depth from any node with no incoming edge. Cycles (peer
    // edges in a collaborative team) are bounded by the node count.
    const depth = new Map();
    nodes.forEach((n) => depth.set(n.id, 0));
    for (let pass = 0; pass < nodes.length; pass++) {
      let moved = false;
      edges.forEach((e) => {
        if (!depth.has(e.source) || !depth.has(e.target)) return;
        const next = depth.get(e.source) + 1;
        if (next > depth.get(e.target)) { depth.set(e.target, next); moved = true; }
      });
      if (!moved) break;
    }
    // Tools always sit one column right of the agent that called them.
    nodes.forEach((n) => { if (n.type === "tool") depth.set(n.id, depth.get(n.id) || 1); });
    return depth;
  }

  function topoLayout(execution) {
    const nodes = execution.nodes || [];
    const edges = execution.edges || [];
    const depth = topoColumns(nodes, edges);
    const byColumn = new Map();
    nodes.forEach((n) => {
      const col = depth.get(n.id) || 0;
      if (!byColumn.has(col)) byColumn.set(col, []);
      byColumn.get(col).push(n);
    });
    const placed = new Map();
    let rows = 1;
    Array.from(byColumn.keys()).sort((a, b) => a - b).forEach((col) => {
      const column = byColumn.get(col);
      rows = Math.max(rows, column.length);
      column.forEach((n, row) => {
        placed.set(n.id, {
          node: n,
          x: 20 + col * TOPO_COL_W,
          y: 16 + row * TOPO_ROW_H,
        });
      });
    });
    const cols = Math.max(...Array.from(byColumn.keys())) + 1;
    return {
      placed,
      width: 20 + cols * TOPO_COL_W,
      height: 16 + rows * TOPO_ROW_H,
    };
  }

  function topoNodeSvg(entry, selectedId, isTabStop) {
    const n = entry.node;
    const status = n.status || "ok";
    const glyph = TOPO_GLYPH[status] || TOPO_GLYPH.ok;
    const shape = n.type === "tool"
      ? `<ellipse cx="${TOPO_NODE_W / 2}" cy="${TOPO_NODE_H / 2}" `
        + `rx="${TOPO_NODE_W / 2}" ry="${TOPO_NODE_H / 2}"></ellipse>`
      : `<rect width="${TOPO_NODE_W}" height="${TOPO_NODE_H}" rx="${n.type === "manager" ? 4 : 10}"></rect>`;
    const kindLabel = n.type === "manager" ? "manager" : n.type;
    const meta = n.type === "tool"
      ? `${n.calls || 0} call${n.calls === 1 ? "" : "s"}`
      : (n.model || kindLabel);
    // The label repeats the status as text so it never depends on color alone.
    const label = `${kindLabel} ${n.label}, status ${status}`
      + (n.model ? `, model ${n.model}` : "")
      + (n.error ? ", failed" : "");
    const selected = n.id === selectedId ? " is-selected" : "";
    // Exactly one node holds the tab stop; the rest are reachable with the
    // arrow keys and stay out of the page's tab order.
    const tabStop = isTabStop ? `tabindex="0"` : `tabindex="-1"`;
    return `<g class="topo-node status-${esc(status)}${selected}" transform="translate(${entry.x},${entry.y})" `
      + `${tabStop} role="button" data-node="${esc(n.id)}" aria-label="${esc(label)}">`
      + shape
      + `<text class="topo-glyph" x="12" y="20">${glyph}</text>`
      + `<text class="topo-name" x="28" y="20">${esc(clip(n.label, 16))}</text>`
      + `<text class="topo-meta" x="28" y="34">${esc(clip(meta, 20))} · ${esc(status)}</text>`
      + `</g>`;
  }

  function topoEdgeSvg(edge, layout, animate) {
    const from = layout.placed.get(edge.source);
    const to = layout.placed.get(edge.target);
    if (!from || !to) return "";
    const x1 = from.x + TOPO_NODE_W;
    const y1 = from.y + TOPO_NODE_H / 2;
    const x2 = to.x;
    const y2 = to.y + TOPO_NODE_H / 2;
    const mid = (x1 + x2) / 2;
    const active = edge.count > 0 ? " is-active" : "";
    const pulse = edge.count > 0 && animate ? " topo-pulse" : "";
    return `<path class="topo-edge kind-${esc(edge.kind)}${active}${pulse}" `
      + `d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}" `
      + `marker-end="url(#topo-arrow)"><title>${esc(edge.kind)}</title></path>`;
  }

  function renderTopology() {
    const svg = $("topo-svg");
    const empty = $("topo-empty");
    const sub = $("topology-sub");
    if (!svg || !empty) return;

    const execution = topologyData.find((e) => e.id === topoSelected) || topologyData[0];
    if (!execution || !(execution.nodes || []).length) {
      svg.innerHTML = "";
      svg.removeAttribute("viewBox");
      empty.hidden = false;
      if (sub) {
        sub.textContent = topologyData.length
          ? "This execution recorded no agents yet."
          : "No multi-agent activity recorded yet. Run a team or a workflow to populate this view.";
      }
      return;
    }
    empty.hidden = true;
    if (sub) {
      sub.textContent = `${execution.kind} "${execution.name}" — `
        + `${execution.nodes.length} node(s), ${execution.edges.length} edge(s), `
        + `${fmtCost(execution.cost_usd)}, ${fmt(execution.tokens)} tokens`;
    }

    const layout = topoLayout(execution);
    const animate = !reducedMotion();
    // The arrowhead needs its own class: .topo-edge sets fill:none for the
    // connector line, and a CSS rule outranks a fill presentation attribute.
    const defs = '<defs><marker id="topo-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
      + 'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
      + '<path d="M0,0 L10,5 L0,10 z" class="topo-arrowhead"></path></marker></defs>';
    // Re-rendering replaces the SVG, so remember which node held focus and give
    // it back afterwards — otherwise a poll or a click drops a keyboard user
    // back to the top of the page mid-interaction.
    const focusedId = document.activeElement
      && document.activeElement.closest
      && document.activeElement.closest("g.topo-node")
      ? document.activeElement.closest("g.topo-node").getAttribute("data-node")
      : null;
    const order = Array.from(layout.placed.values());
    if (!order.some((entry) => entry.node.id === topoFocusId)) {
      topoFocusId = order.length ? order[0].node.id : null;
    }
    svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
    svg.setAttribute("height", String(layout.height));
    svg.innerHTML = defs
      + execution.edges.map((e) => topoEdgeSvg(e, layout, animate)).join("")
      + order.map((entry) => topoNodeSvg(entry, topoNodeId, entry.node.id === topoFocusId)).join("");

    const nodeEls = Array.from(svg.querySelectorAll("g.topo-node"));
    const moveFocus = (from, delta) => {
      if (nodeEls.length < 2) return;
      const at = nodeEls.indexOf(from);
      const next = nodeEls[((at + delta) % nodeEls.length + nodeEls.length) % nodeEls.length];
      nodeEls.forEach((el) => el.setAttribute("tabindex", el === next ? "0" : "-1"));
      topoFocusId = next.getAttribute("data-node");
      next.focus();
    };
    nodeEls.forEach((g) => {
      const open = () => {
        topoNodeId = g.getAttribute("data-node");
        topoFocusId = topoNodeId;
        renderTopology();
      };
      g.addEventListener("click", open);
      g.addEventListener("focus", () => { topoFocusId = g.getAttribute("data-node"); });
      g.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); open(); }
        else if (ev.key === "ArrowRight" || ev.key === "ArrowDown") {
          ev.preventDefault(); moveFocus(g, 1);
        } else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") {
          ev.preventDefault(); moveFocus(g, -1);
        }
      });
    });

    if (focusedId) {
      const restore = svg.querySelector(`g.topo-node[data-node="${cssEscape(focusedId)}"]`);
      if (restore) restore.focus();
    }

    renderTopoDetail(execution);
  }

  function renderTopoDetail(execution) {
    const host = $("topo-detail");
    if (!host) return;
    const node = (execution.nodes || []).find((n) => n.id === topoNodeId);
    if (!node) { host.hidden = true; host.innerHTML = ""; return; }
    const rows = [
      ["Type", node.type],
      ["Status", node.status],
      ["Role", node.role || "—"],
      ["Model", node.model || "—"],
      ["Runs", node.type === "tool" ? node.calls : node.runs],
      ["Cost", fmtCost(node.cost_usd)],
      ["Tokens", fmt(node.tokens)],
      ["Duration", fmtSeconds(node.duration_s)],
    ];
    host.hidden = false;
    setHtml(host, `<h3>${esc(node.label)}</h3><dl class="run-detail">`
      + rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v == null ? "—" : v))}</dd>`).join("")
      + `</dl>`
      + (node.task ? `<h3>Task</h3><pre class="run-text">${esc(node.task)}</pre>` : "")
      + (node.output ? `<h3>Output</h3><pre class="run-text">${esc(node.output)}</pre>` : "")
      + (node.error ? `<h3>Error</h3><pre class="run-text run-error">${esc(node.error)}</pre>` : ""));
  }

  async function loadTopology() {
    try {
      const resp = await fetch("/dashboard/topology.json", { cache: "no-store" });
      if (!resp.ok) return;
      const data = await resp.json();
      topologyData = data.executions || [];
      const select = $("topo-select");
      if (select) {
        const options = topologyData.map((e) =>
          `<option value="${esc(e.id)}">${esc(e.kind || "run")} · ${esc(e.name || e.id)}</option>`
        ).join("");
        // Rebuilt only when the set of executions actually changed, so polling
        // does not close an open dropdown or reset the reader's choice.
        if (select.innerHTML !== options) {
          select.innerHTML = options || '<option value="">—</option>';
        }
        if (topologyData.length && !topologyData.some((e) => e.id === topoSelected)) {
          topoSelected = topologyData[0].id;
        }
        select.value = topoSelected;
      }
      renderTopology();
    } catch (err) {
      console.warn("[effGen dashboard] topology fetch failed:", err);
    }
  }

  function initTopology() {
    const select = $("topo-select");
    if (select) {
      select.addEventListener("change", () => {
        topoSelected = select.value;
        topoNodeId = null;
        renderTopology();
      });
    }
  }

  // ------------------------------------------------------------------
  // Keyboard navigation: section jump row + command palette
  //
  // Commands are rebuilt each time the palette opens, from data the page has
  // already loaded, so runs and models added since the last open are listed.
  // ------------------------------------------------------------------
  function jump(panelId) {
    if (webui) webui.jumpTo(panelId);
  }

  function closeTopoDetail() {
    if (topoNodeId == null) return false;
    topoNodeId = null;
    renderTopology();
    return true;
  }

  function closeRunDetail() {
    if (!openRunId) return false;
    openRunId = null;
    renderRunDetail(null);
    return true;
  }

  function openHistoryRun(run) {
    openRunId = run.run_id || null;
    renderRunDetail(run);
    jump("panel-history");
    const tbody = $("history-tbody");
    const button = tbody && tbody.querySelector('button[aria-expanded="true"]');
    if (button) button.focus({ preventScroll: true });
  }

  function navigationCommands() {
    const panels = [
      ["summary-cards", "Summary metrics"],
      ["panel-slo", "SLO burn rates"],
      ["panel-latency-chart", "Latency chart"],
      ["panel-by-model", "By model"],
      ["panel-by-status", "HTTP responses by status"],
      ["panel-by-route", "Responses by route"],
      ["panel-agent-runs", "Recent agent runs"],
      ["panel-history", "History"],
      ["panel-spans", "Live span stream"],
      ["panel-waterfall", "Run timeline"],
      ["panel-topology", "Agent topology"],
      ["panel-catalog", "Model catalog"],
      ["panel-metrics", "Prometheus metrics"],
    ];
    const commands = panels.map(([id, label]) => ({
      id: "nav:" + id,
      group: "Navigate",
      label: "Go to " + label,
      keywords: id.replace(/-/g, " ") + " panel section",
      hint: "panel",
      run: () => jump(id),
    }));
    commands.push({
      id: "nav:playground",
      group: "Navigate",
      label: "Open the playground",
      keywords: "playground prompt compose run request",
      hint: "/playground",
      run: () => { window.location.href = "/playground"; },
    });
    return commands;
  }

  function runCommands() {
    return historyRuns.slice(0, 40).map((run, i) => ({
      id: "run:" + (run.run_id || i),
      group: "Runs",
      label: clip(run.task || run.run_id || "(no task recorded)", 70),
      keywords: [run.model, run.run_id, run.status, run.agent].filter(Boolean).join(" "),
      hint: clip(run.model, 24),
      run: () => openHistoryRun(run),
    }));
  }

  function modelCommands() {
    return catalogAll.slice(0, 400).map((rec) => ({
      id: "model:" + rec.provider + ":" + rec.id,
      group: "Models",
      label: rec.id,
      keywords: [rec.provider, rec.family, rec.free_tier ? "free" : "",
                 rec.supports_vision ? "vision" : "",
                 rec.supports_tools ? "tools" : ""].filter(Boolean).join(" "),
      hint: rec.provider,
      run: () => {
        const search = $("cat-search");
        if (search) { search.value = rec.id; catalogFilterSort(); }
        jump("panel-catalog");
      },
    }));
  }

  function actionCommands() {
    const pause = $("span-pause-cb");
    return [
      {
        id: "action:theme",
        group: "Actions",
        label: "Switch color theme",
        keywords: "dark light appearance",
        run: () => applyTheme(currentTheme() === "dark" ? "light" : "dark", true),
      },
      {
        id: "action:refresh",
        group: "Actions",
        label: "Refresh data now",
        keywords: "reload poll update",
        run: fetchData,
      },
      {
        id: "action:clear-spans",
        group: "Actions",
        label: "Clear the span stream",
        keywords: "spans trace empty",
        run: () => { const b = $("span-clear-btn"); if (b) b.click(); },
      },
      {
        id: "action:toggle-spans",
        group: "Actions",
        label: (pause && pause.checked ? "Resume" : "Pause") + " the span stream",
        keywords: "spans stream live pause resume",
        run: () => {
          if (!pause) return;
          pause.checked = !pause.checked;
          pause.dispatchEvent(new Event("change"));
        },
      },
      {
        id: "action:search-history",
        group: "Actions",
        label: "Search run history",
        keywords: "find run filter",
        run: () => { jump("panel-history"); const el = $("hist-search"); if (el) el.focus(); },
      },
      {
        id: "action:search-catalog",
        group: "Actions",
        label: "Search the model catalog",
        keywords: "models pricing context vision",
        run: () => { jump("panel-catalog"); const el = $("cat-search"); if (el) el.focus(); },
      },
    ];
  }

  function initKeyboard() {
    const nav = $("panel-nav");
    if (nav) {
      nav.addEventListener("click", (event) => {
        const link = event.target.closest("a[data-panel-jump]");
        if (!link) return;
        event.preventDefault();
        jump(link.dataset.panelJump);
      });
    }
    if (!webui) return;
    const palette = webui.init({
      surface: "dashboard",
      actions: () => navigationCommands()
        .concat(actionCommands(), runCommands(), modelCommands()),
      shortcuts: [
        { keys: "← → ↑ ↓", label: "Move between topology nodes once one has focus" },
      ],
    });
    palette.onEscape(() => closeTopoDetail() || closeRunDetail());
    setText("palette-hint", webui.chord("K") + " for commands · ? for shortcuts");
    const btn = $("palette-btn");
    if (btn) btn.addEventListener("click", palette.open);
  }

  // ------------------------------------------------------------------
  // Data polling
  // ------------------------------------------------------------------
  async function fetchData() {
    try {
      const resp = await fetch("/dashboard/data.json", { cache: "no-store" });
      if (resp.status === 401) {
        setStatus(false);
        showAuthBanner();
        return;
      }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      hideAuthBanner();
      setStatus(true);
      // The catalog shares the dashboard's access rule; load it once data
      // access is confirmed (a 401 there shows an inline note instead).
      loadCatalog();
      renderCards(data);
      renderSLO(data);
      renderByModel(data);
      renderByStatus(data);
      renderByRoute(data);
      renderRuns(data);
      renderMetrics(data);
      loadHistory();
      loadTopology();
      // Seed the span stream from the payload if SSE has not connected.
      if (!eventSource && data.recent_spans) {
        data.recent_spans.forEach(appendSpan);
      }
    } catch (err) {
      setStatus(false);
      console.warn("[effGen dashboard] fetch failed:", err);
    }
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  function init() {
    initTheme();
    initCatalog();
    initHistory();
    initTopology();
    initKeyboard();
    bindSortHeaders("by-model-table", () => renderByModel(null));
    bindSortHeaders("by-route-table", () => renderByRoute(null));

    // Wire up buttons
    const refreshBtn = $("refresh-btn");
    if (refreshBtn) refreshBtn.addEventListener("click", fetchData);

    const clearBtn = $("span-clear-btn");
    if (clearBtn) clearBtn.addEventListener("click", () => {
      const s = $("span-stream");
      if (s) s.innerHTML = "";
      spanCount = 0;
      setText("span-count", "0 spans");
      runs.clear();
      seenSpans.clear();
      renderWaterfall();
    });

    const pauseCb = $("span-pause-cb");
    if (pauseCb) pauseCb.addEventListener("change", () => { spanPaused = pauseCb.checked; });

    window.addEventListener("resize", drawChart);

    // Initial load + poll
    fetchData();
    setInterval(fetchData, POLL_MS);

    // Try SSE
    startSSE();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
