"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiActivity,
  FiArrowRight,
  FiExternalLink,
  FiFileText,
  FiGrid,
  FiLayers,
  FiShare2,
  FiWifiOff,
} from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Figure from "@/components/ui/Figure";
import ParamTable from "@/components/ui/ParamTable";
import CodeSample from "@/components/ui/CodeSample";
import RouteLink from "@/components/ui/RouteLink";
import { figureOf, webCapture } from "@/components/webCaptures";
import { siteData, version } from "@/components/siteData";
import {
  DOCS_AUTH_URL,
  DOCS_DASHBOARD_URL,
  DOCS_SERVER_URL,
  endpoints,
  panelGroups,
  panelNav,
  panelTitle,
  paletteGroups,
  snippetForms,
} from "./dashboardData";
import { accentTextStyle } from "@/components/accentText";

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

const web = siteData.web;

function Band({
  id,
  eyebrow,
  title,
  lede,
  tinted = false,
  children,
}: {
  id: string;
  eyebrow: string;
  title: React.ReactNode;
  lede?: React.ReactNode;
  tinted?: boolean;
  children: React.ReactNode;
}) {
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section
      id={id}
      className={`py-16 relative scroll-mt-24 ${tinted ? "bg-gray-50 dark:bg-[#030f07]" : ""}`}
      aria-labelledby={`${id}-heading`}
    >
      {SECTION_DIVIDER}
      <Container className="relative z-10">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-10 max-w-3xl"
        >
          <span className="text-[10px] font-mono uppercase tracking-widest text-green-700 dark:text-green-400">
            {eyebrow}
          </span>
          <h2
            id={`${id}-heading`}
            className="mt-2 text-3xl md:text-4xl font-black text-gray-900 dark:text-white leading-tight"
          >
            {title}
          </h2>
          {lede && <p className="mt-4 text-gray-600 dark:text-gray-400 leading-relaxed">{lede}</p>}
        </motion.div>
        {children}
      </Container>
    </section>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
      <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2">{title}</h3>
      <div className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed space-y-3">
        {children}
      </div>
    </div>
  );
}

export default function DashboardView() {
  const { ref: heroRef, inView: heroInView } = useInView({ triggerOnce: true, threshold: 0.05 });

  const kilobytes = Math.round(web.static_bytes / 1024);

  const headline = [
    {
      value: String(web.dashboard.panels.length),
      label: "Dashboard panels",
      accent: "#00ff88",
      icon: FiGrid,
    },
    {
      value: String(web.dashboard.data_endpoints.length),
      label: "Same-origin routes",
      accent: "#00e5ff",
      icon: FiActivity,
    },
    {
      value: `${kilobytes} kB`,
      label: `Both surfaces, ${web.static_files.length} files`,
      accent: "#a78bfa",
      icon: FiLayers,
    },
    {
      value: String(web.external_references),
      label: "Outside hosts reached",
      accent: "#ffd700",
      icon: FiWifiOff,
    },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        {/* Hero */}
        <section className="relative pt-32 pb-10 overflow-hidden">
          <div className="absolute inset-0 grid-pattern" />
          <Container className="relative z-10">
            <motion.div
              ref={heroRef}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="max-w-4xl"
            >
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8">
                <FiGrid size={14} />
                effgen serve · {version}
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                A dashboard, a playground and a model browser{" "}
                <span className="gradient-text">that fetch nothing</span>
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                <code className="font-mono text-green-700 dark:text-green-400">effgen serve</code>{" "}
                already carries its web surfaces:{" "}
                {web.dashboard.panels.length} panels of live traffic, cost and
                traces at <code className="font-mono text-sm">/dashboard</code>, and a
                browser playground at{" "}
                <code className="font-mono text-sm">/playground</code> that runs a model,
                shows the tools it called and hands the run back as{" "}
                {web.playground.snippet_kinds.join(", ")}. Both are{" "}
                {web.static_files.length} files inside the package —{" "}
                {kilobytes} kB, no CDN, no external font, nothing fetched when the page
                opens.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="#panels"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  All {web.dashboard.panels.length} panels
                  <FiArrowRight size={15} />
                </a>
                <a
                  href={DOCS_DASHBOARD_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-green-500/50 font-semibold text-sm transition-colors"
                >
                  Reference documentation
                  <FiExternalLink size={14} />
                </a>
              </div>
            </motion.div>
          </Container>
        </section>

        <section className="pb-12 relative">
          <Container className="relative z-10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {headline.map((stat, idx) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={heroInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: idx * 0.08 }}
                  className="rounded-2xl p-5 bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 text-center"
                >
                  <stat.icon className="mx-auto mb-2" style={accentTextStyle(stat.accent)} size={20} />
                  <div className="text-2xl font-black mb-0.5" style={accentTextStyle(stat.accent)}>
                    {stat.value}
                  </div>
                  <div className="text-[10px] text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </Container>
        </section>

        {/* The surface itself */}
        <section className="pb-4 relative">
          <Container className="relative z-10">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {(["dark", "light"] as const).map((theme) => {
                const capture = webCapture("dashboard-full", theme);
                return (
                  <Figure
                    key={theme}
                    {...figureOf(capture)}
                    frameClassName="max-h-[70vh] overflow-y-auto"
                    caption={
                      <>
                        The whole page in the {theme} theme, scrolled through in place.
                        The panels below are the same page, photographed one at a time.
                      </>
                    }
                    command={`effgen serve --port 8000  ·  ${capture.produced_by}`}
                  />
                );
              })}
            </div>
            <p className="mt-6 text-sm text-gray-600 dark:text-gray-400 max-w-3xl leading-relaxed">
              Every figure on this page is a screenshot of a running server, taken with a
              browser after real traffic went through it: six calls that succeeded, one
              naming a model that does not exist, one body with no{" "}
              <code className="font-mono text-xs">messages</code> field, a probe of a route
              that is not served, and a two-agent team run. That is why the error columns
              have numbers in them and the cost column has a{" "}
              <code className="font-mono text-xs">—</code> in it. The counters that include
              the dashboard&rsquo;s own polling keep climbing while a page is open, which
              is why they differ between one figure and the next. Nothing here is a
              mock-up.
            </p>
          </Container>
        </section>

        {/* The panels */}
        <Band
          id="panels"
          eyebrow="The dashboard"
          tinted
          title={
            <>
              {web.dashboard.panels.length} panels, and{" "}
              <span className="gradient-text">each one names what it measured</span>
            </>
          }
          lede={
            <>
              Figures from different sources are never added together. The per-model
              rows are scoped to a{" "}
              <code className="font-mono text-sm">(model, provider)</code> pair, spend
              that could not be matched to a row is stated below the table rather than
              spread across it, and the route panel carries the denominator that turns a
              status code into an error rate.
            </>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
            {panelGroups.map((group) => (
              <article
                key={group.id}
                className="relative rounded-xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-5"
              >
                <div
                  className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                  style={{ background: group.accent }}
                />
                <div className="pl-3">
                  <h3 className="text-base font-bold text-gray-900 dark:text-white">
                    {group.title}
                  </h3>
                  <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {group.lede}
                  </p>
                  <ul className="mt-3 flex flex-wrap gap-1.5">
                    {group.panels.map((id) => (
                      <li
                        key={id}
                        className="px-2 py-0.5 rounded-md text-[11px] font-mono bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-800"
                        title={panelNav(id)}
                      >
                        {panelTitle(id)}
                      </li>
                    ))}
                  </ul>
                </div>
              </article>
            ))}
          </div>

          <div className="space-y-10">
            <Figure
              {...figureOf(webCapture("dashboard-summary"))}
              caption={
                <>
                  The summary row. The cost card says how many of the runs behind it were
                  priced and how many were not, because a total that quietly counts an
                  unpriced call as $0 is a total nobody can act on.
                </>
              }
              command="GET /dashboard/data.json → .metrics"
            />

            <div>
              <Figure
                {...figureOf(webCapture("dashboard-by-model"))}
                caption={
                  <>
                    Per model and provider: calls, error rate, p95, the dominant failure
                    class, tokens and spend. The model that does not exist has a{" "}
                    <code className="font-mono text-xs">—</code> where its cost would be
                    and <code className="font-mono text-xs">not_found · 1</code> where its
                    outcome is. Model, Provider, Calls, Error rate, p95 and Cost all sort
                    on click.
                  </>
                }
                command="GET /dashboard/data.json → .by_model"
              />
              <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                <CodeSample
                  language="bash"
                  accent="#00e5ff"
                  code={`curl -s http://127.0.0.1:8000/dashboard/data.json | jq '.by_model[0]'`}
                  output={`{
  "model": "gemini-3.1-flash-lite",
  "provider": "gemini",
  "calls": 4,
  "errors": 0,
  "input_tokens": 234,
  "output_tokens": 50,
  "outcomes": {
    "ok": 4
  },
  "top_error": null,
  "top_error_hint": null,
  "error_rate": 0,
  "p95_latency_s": 2.4,
  "cost_usd": 0.000133
}`}
                  outputLabel="what that returned"
                />
                <Card title="The row a failure produces carries its own fix">
                  <p>
                    <code className="font-mono text-xs">outcomes</code> tallies the
                    recorded outcome label verbatim,{" "}
                    <code className="font-mono text-xs">top_error</code> names the most
                    frequent failure, and{" "}
                    <code className="font-mono text-xs">top_error_hint</code> is the same
                    remediation sentence the command line prints for that class — so the
                    panel says what to do, not only that something went wrong.
                  </p>
                  <p>
                    Spend that cannot be attributed to any row is reported apart, in{" "}
                    <code className="font-mono text-xs">unattributed_cost_usd</code>, so
                    the cost column always sums to money actually attributed.
                  </p>
                </Card>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <Figure
                {...figureOf(webCapture("dashboard-by-status"))}
                caption={
                  <>
                    One chip per status code, each stating the code, the class and the
                    count as text — so the panel does not depend on colour.
                  </>
                }
                command="GET /dashboard/data.json → .by_status"
              />
              <Figure
                {...figureOf(webCapture("dashboard-by-route"))}
                caption={
                  <>
                    The same failures with a denominator. Traffic outside the recorded
                    route list — including the dashboard&rsquo;s own polling — is labelled{" "}
                    <code className="font-mono text-xs">other</code> rather than being
                    folded into a route it did not touch.
                  </>
                }
                command="GET /dashboard/data.json → .by_route"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <Figure
                {...figureOf(webCapture("dashboard-waterfall"))}
                caption={
                  <>
                    Spans grouped by run, positioned by start offset and sized by
                    duration. The failed run is drawn as failed rather than as a very
                    fast one.
                  </>
                }
                command="GET /dashboard/data.json → .recent_spans"
              />
              <Figure
                {...figureOf(webCapture("dashboard-history"))}
                caption={
                  <>
                    Stored runs and saved sessions, filterable by text and status. This
                    is the same durable history{" "}
                    <code className="font-mono text-xs">effgen runs</code> and{" "}
                    <code className="font-mono text-xs">effgen sessions</code> read, so a
                    run started from a script shows up here.
                  </>
                }
                command="GET /dashboard/history.json"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <Figure
                {...figureOf(webCapture("dashboard-slo"))}
                caption={
                  <>
                    How much of the p99 latency, error-rate and availability budgets has
                    been spent, with the measured percentiles and the target underneath.
                  </>
                }
                command="GET /dashboard/data.json → .slo"
              />
              <Figure
                {...figureOf(webCapture("dashboard-latency"))}
                caption={
                  <>
                    Average latency over recent polling intervals, drawn on a canvas by
                    the page itself. No chart library is loaded, because none is shipped.
                  </>
                }
                command="GET /dashboard/data.json → .metrics.avg_latency_s"
              />
            </div>
          </div>

          <div className="mt-12">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              The {web.dashboard.data_endpoints.length} routes the page reads
            </h3>
            <ParamTable
              nameLabel="Route"
              params={endpoints().map((endpoint) => ({
                name: endpoint.path,
                description: endpoint.note,
              }))}
              caption="Every one is same-origin. The page issues no other request."
            />
          </div>
        </Band>

        {/* The playground */}
        <Band
          id="playground"
          eyebrow="The playground"
          title={
            <>
              Run a model in the browser, then{" "}
              <span className="gradient-text">take the run with you</span>
            </>
          }
          lede={
            <>
              Pick a model or a preset, attach tools, set temperature and a token cap,
              and run it — streamed or not. The answer arrives with its tokens, its cost
              and its latency, the tools that ran are listed with what they were given
              and what they returned, and the whole thing is offered back as{" "}
              {web.playground.snippet_kinds.join(", ")}.
            </>
          }
        >
          <Figure
            {...figureOf(webCapture("playground-run"))}
            caption={
              <>
                One real run. The model picker carries each model&rsquo;s price, context
                window and capabilities from the catalog; the tool checkboxes come from
                the registry the server has. The answer, the five figures under it and
                the trace row are what this run produced.
              </>
            }
            command={`GET /playground  ·  ${webCapture("playground-run").produced_by}`}
          />

          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                The tool trace is the run, not a summary of it
              </h3>
              <Figure
                {...figureOf(webCapture("playground-trace"))}
                caption={
                  <>
                    One row per tool step: the tool, the arguments it was called with, the
                    result, and how long it took. Tools run server-side, so the trace
                    shows what the agent actually did rather than what it said it did.
                  </>
                }
                command="the trace of the run above"
              />
            </div>
            <Card title="Two modes, one prompt">
              <p>
                <strong className="text-gray-900 dark:text-white">Single run</strong> is
                one model answering.{" "}
                <strong className="text-gray-900 dark:text-white">Battle</strong> sends
                the same prompt to every contender at once and lays the answers out side
                by side with each model&rsquo;s own tokens, cost and latency — so a
                battle spends once per model, which the form says before you press Run.
              </p>
              <p>
                The answer box appends deltas while a stream is open and marks itself
                busy for the duration, so a screen reader hears the answer once rather
                than hearing it re-read on every token. A battle grid is not a live
                region at all; the verdict is, and it is stated once.
              </p>
            </Card>
          </div>

          <div className="mt-8">
            <Figure
              {...figureOf(webCapture("playground-battle"))}
              caption={
                <>
                  Battle mode with two contenders on one prompt, each column carrying that
                  model&rsquo;s own answer and its own measurements.
                </>
              }
              command={webCapture("playground-battle").produced_by}
            />
          </div>

          <div className="mt-12">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              Copy this run
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6 max-w-3xl leading-relaxed">
              The three forms below are what the Copy button put on the clipboard after
              the run in the figure above — read back out of the page, then run. The model
              id, the tools, the temperature and the token cap are the ones the form was
              set to, so what you copy is the run you just watched.
            </p>
            <div className="space-y-6">
              {snippetForms.map((form) => (
                <div key={form.kind}>
                  <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                    {form.label}
                  </h4>
                  <CodeSample
                    language={form.language}
                    accent={form.accent}
                    code={form.code}
                    output={form.output}
                    outputLabel={form.outputLabel}
                  />
                </div>
              ))}
            </div>
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 max-w-3xl leading-relaxed">
              The Python form needs the provider key in the environment; the command-line
              form finds it the way{" "}
              <code className="font-mono text-xs">effgen</code> always does, and the{" "}
              <code className="font-mono text-xs">curl</code> form names the server&rsquo;s
              own key as an environment variable rather than carrying one.
            </p>
          </div>
        </Band>

        {/* The model browser */}
        <Band
          id="catalog"
          eyebrow="The model browser"
          tinted
          title={
            <>
              {siteData.models.models} models,{" "}
              <span className="gradient-text">searchable in the page</span>
            </>
          }
          lede={
            <>
              The catalog panel carries every model the catalog knows — context window,
              output limit, input and output price, and what each one can do — with a
              search box, provider and capability filters, a sort and paging. It is the
              same catalog{" "}
              <code className="font-mono text-sm">effgen models browse</code> reads.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <Figure
              {...figureOf(webCapture("dashboard-catalog"))}
              caption={
                <>
                  {siteData.models.models} models across{" "}
                  {siteData.models.with_catalog_count} providers, with the snapshot the
                  prices came from named above the table.
                </>
              }
              command="GET /dashboard/catalog.json"
            />
            <Figure
              {...figureOf(webCapture("dashboard-catalog-filtered"))}
              caption={<>The same panel with a search term typed into it.</>}
              command="the catalog panel, filtered in the page"
            />
          </div>
          <div className="mt-8 rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-6 flex flex-col md:flex-row md:items-center gap-4 justify-between">
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-2xl">
              The providers behind that table, what each is good for, the pricing rules
              the panel follows, and the same catalog on the command line are on the
              models page.
            </p>
            <RouteLink
              to="/models"
              className="inline-flex shrink-0 items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
            >
              Any model, anywhere
              <FiArrowRight size={15} />
            </RouteLink>
          </div>
        </Band>

        {/* Topology */}
        <Band
          id="topology"
          eyebrow="The topology graph"
          title={
            <>
              A team or a workflow,{" "}
              <span className="gradient-text">as the shape it ran in</span>
            </>
          }
          lede={
            <>
              Agents and the tools they reached are nodes; delegation, handoff and tool
              use are edges. It is built from the durable run store plus the buffered
              spans, so a team run from a script or the command line appears here too —
              not only work done inside the server process.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <Figure
              {...figureOf(webCapture("dashboard-topology"))}
              caption={
                <>
                  A two-agent team, run from a script while this server was up. Status is
                  carried by a glyph and a word as well as by colour, nodes are
                  keyboard-focusable and open a detail panel, and the picker above the
                  graph switches between recorded executions.
                </>
              }
              command="GET /dashboard/topology.json"
            />
            <div>
              <CodeSample
                language="bash"
                accent="#ffd700"
                code={`curl -s 'http://127.0.0.1:8000/dashboard/topology.json?limit=1' \\
  | jq '.executions[0] | {id, kind, name, status, cost_usd, tokens, edges}'`}
                output={`{
  "id": "ea8b59437040",
  "kind": "team",
  "name": "newsroom",
  "status": "ok",
  "cost_usd": 6.1e-05,
  "tokens": 88,
  "edges": [
    {
      "source": "writer",
      "target": "editor",
      "kind": "handoff",
      "count": 1
    }
  ]
}`}
                outputLabel="what that returned"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                <code className="font-mono text-xs">executions</code> is empty until
                something multi-agent has run — the panel says so rather than drawing an
                empty canvas. The graph is inline SVG the page builds itself; no graph
                library is involved.
              </p>
            </div>
          </div>

          <div className="mt-10">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              Spans arrive as they happen
            </h3>
            <CodeSample
              language="bash"
              accent="#a78bfa"
              code={`curl -N -s http://127.0.0.1:8000/dashboard/spans | head -1`}
              output={`data: {"ts": "05:59:32", "name": "effgen.agent.run api:openai:gpt-9-does-not-exist", "kind": "agent", "agent": "api:openai:gpt-9-does-not-exist", "tool": null, "model": null, "duration_ms": 61.8, "status": "error", "error": "openai error (model='gpt-9-does-not-exist'): The model \`gpt-9-does-not-exist\` does not exist or you do not have access to it. Did you mean: gpt-5-mini, gpt-5-nano, gpt-4o-mini? ...", "note": null, "run_id": "20f57f459322", "offset_ms": 0.0, "execution_id": null, "execution_kind": null, "execution_name": null, "parent_agent": null, "role": null}`}
              outputLabel="the first event, wrapped here to fit — one line on the wire"
            />
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 max-w-3xl leading-relaxed">
              <code className="font-mono text-xs">kind</code> is{" "}
              <code className="font-mono text-xs">agent</code>,{" "}
              <code className="font-mono text-xs">model</code>,{" "}
              <code className="font-mono text-xs">tool</code> or{" "}
              <code className="font-mono text-xs">router</code>, and the matching field
              names what the span timed — read those rather than parsing{" "}
              <code className="font-mono text-xs">name</code>, which is the display label.
              A run that reports a failure without raising is recorded here as an error,
              which is how the timeline above knows to draw it as one.
            </p>
          </div>
        </Band>

        {/* Keyboard */}
        <Band
          id="keyboard"
          eyebrow="Keyboard-first"
          tinted
          title={
            <>
              One command palette,{" "}
              <span className="gradient-text">shared by both surfaces</span>
            </>
          }
          lede={
            <>
              <kbd className="px-1.5 py-0.5 rounded border border-gray-300 dark:border-gray-700 font-mono text-xs">
                Ctrl-K
              </kbd>{" "}
              opens it on either page. It searches four groups built from data the page
              has already loaded, so it finds a stored run by its task text and a model by
              its capability without another request.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <Figure
              {...figureOf(webCapture("dashboard-palette"))}
              caption={
                <>
                  The palette with a query typed into it, grouped by where each match came
                  from. The commands used most recently lead the list when it opens empty.
                </>
              }
              command={webCapture("dashboard-palette").produced_by}
            />
            <Figure
              {...figureOf(webCapture("dashboard-shortcuts"))}
              caption={
                <>
                  <kbd className="px-1 py-0.5 rounded border border-gray-300 dark:border-gray-700 font-mono text-xs">
                    ?
                  </kbd>{" "}
                  shows the whole keyboard layer. Escape closes the palette, this list, or
                  an open detail pane.
                </>
              }
              command={webCapture("dashboard-shortcuts").produced_by}
            />
          </div>

          <div className="mt-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {paletteGroups.map((group) => (
              <div
                key={group.name}
                className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5"
              >
                <h3 className="text-sm font-bold text-gray-900 dark:text-white">
                  {group.name}
                </h3>
                <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  {group.what}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card title="Focus survives the five-second poll">
              <p>
                Focus on a run&rsquo;s disclosure button stays on that button when the
                poll rebuilds the history table, and the topology graph behaves the same
                way. If the run is no longer listed, focus moves to the history panel
                rather than to the top of the document.
              </p>
              <p>
                A jump row under the header links to every panel. Selecting one — from the
                row or from the palette — scrolls to it and moves focus into it, so the
                next Tab continues from there.
              </p>
            </Card>
            <Card title="Nothing is announced when nothing changed">
              <p>
                Every value the page writes goes through a write-if-changed rule, so an
                idle dashboard is silent instead of re-reading five cards, the SLO line
                and the connection status every five seconds.
              </p>
              <p>
                A sortable header is a button inside its{" "}
                <code className="font-mono text-xs">&lt;th&gt;</code>, exactly one header
                carries <code className="font-mono text-xs">aria-sort</code>, and the new
                order is spoken. Smooth scrolling is skipped for a visitor who prefers
                reduced motion, and the theme choice is stored under one key shared by
                both surfaces.
              </p>
            </Card>
          </div>
        </Band>

        {/* Self-contained */}
        <Band
          id="self-contained"
          eyebrow="Self-contained"
          title={
            <>
              {web.external_references} requests to{" "}
              <span className="gradient-text">anywhere but this machine</span>
            </>
          }
          lede={
            <>
              Both surfaces are {web.static_files.length} files inside the{" "}
              <code className="font-mono text-sm">effgen</code> package, {kilobytes} kB in
              total. There is no CDN script, no external stylesheet, no font host and no
              remote image, so the pages render the same on an air-gapped host as they do
              on a laptop. That is not a promise, it is a test.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <ParamTable
                nameLabel="File"
                params={web.static_files.map((file) => ({
                  name: file.file,
                  description: `${file.bytes.toLocaleString()} bytes`,
                }))}
                caption={
                  <>
                    Shipped inside the package. The two{" "}
                    <code className="font-mono text-[11px]">webui</code> files are the
                    shared keyboard layer, served to both surfaces under the access rule
                    of whichever page loaded them.
                  </>
                }
              />
            </div>
            <div className="space-y-6">
              <div>
                <CodeSample
                  language="bash"
                  accent="#00ff88"
                  code={`# what the shipped files reference — the framework's own check
python -m pytest -q \\
  "tests/dx/test_dashboard.py::TestStaticFiles::test_no_external_assets" \\
  "tests/dx/test_playground.py::TestStaticFiles::test_no_external_network_hosts" \\
  "tests/dx/test_web_palette.py::TestSharedAssets::test_no_external_asset_reference" \\
  "tests/dx/test_dashboard.py::TestStaticFiles::test_latency_chart_drawn_locally"`}
                  output={`....                                                                     [100%]
4 passed in 3.76s`}
                  outputLabel="what that printed"
                />
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  The first three scan every shipped file for an absolute or
                  protocol-relative URL; a same-origin path such as{" "}
                  <code className="font-mono text-xs">/dashboard/data.json</code> is
                  fine, anything else fails the build. The fourth is why there is no chart
                  library: the latency chart has to be drawn on a canvas.
                </p>
              </div>
              <Card title="And what a browser actually asked for">
                <p>
                  A file scan proves nothing about a page that builds a URL at run time,
                  so the surfaces were also driven by a real browser with every request it
                  made recorded. Loading{" "}
                  <code className="font-mono text-xs">/dashboard</code>, waiting through a
                  poll cycle and then loading{" "}
                  <code className="font-mono text-xs">/playground</code> produced{" "}
                  <strong className="text-gray-900 dark:text-white">23 requests</strong>,
                  of which{" "}
                  <strong className="text-gray-900 dark:text-white">
                    0 left the server
                  </strong>
                  .
                </p>
                <p>
                  The one absolute URL anywhere in the shipped files is{" "}
                  <code className="font-mono text-xs">http://127.0.0.1:8000</code>, printed
                  inside the playground&rsquo;s copy-as-curl snippet. It is text the page
                  displays, not a request it makes.
                </p>
              </Card>
            </div>
          </div>
        </Band>

        {/* Serving it */}
        <Band
          id="serve"
          eyebrow="Starting it"
          tinted
          title={
            <>
              One command, and{" "}
              <span className="gradient-text">it is authenticated by default</span>
            </>
          }
          lede={
            <>
              <code className="font-mono text-sm">effgen serve</code> binds loopback,
              serves the OpenAI-compatible API, the dashboard and the playground from one
              application, and is never unauthenticated: with no key configured it mints
              an ephemeral one and prints it once.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <CodeSample
                language="bash"
                accent="#00e5ff"
                code={`# local viewing: no auth, and the data routes open for the page
EFFGEN_DEV_MODE=1 EFFGEN_PUBLIC_DASHBOARD=1 effgen serve --port 8000

# then
open http://127.0.0.1:8000/dashboard
open http://127.0.0.1:8000/playground`}
                output={`{"status":"ok","version":"1.0.1"}`}
                outputLabel="curl -s http://127.0.0.1:8000/health"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                <code className="font-mono text-xs">--host 0.0.0.0</code> exposes it on
                every interface, which is why the help text tells you to set{" "}
                <code className="font-mono text-xs">EFFGEN_API_KEY</code> first.{" "}
                <code className="font-mono text-xs">effgen top</code> reads the same
                server&rsquo;s{" "}
                <code className="font-mono text-xs">/dashboard/data.json</code> from a
                terminal.
              </p>
            </div>
            <ParamTable
              nameLabel="Flag"
              params={siteData.cli.command_options.serve.map((option) => ({
                name: option.name,
                description: option.description,
              }))}
              caption={`effgen serve --help · effGen ${version}`}
            />
          </div>

          <div className="mt-10">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              The rest of the surface is environment
            </h3>
            <ParamTable
              nameLabel="Variable"
              params={siteData.cli.serve_env.map((setting) => ({
                name: setting.name,
                description: setting.description,
              }))}
              caption={`effgen serve --help · effGen ${version}`}
            />
          </div>
        </Band>

        {/* Failure */}
        <Band
          id="failure"
          eyebrow="When it will not show you anything"
          title={
            <>
              The page loads.{" "}
              <span className="gradient-text">The data does not.</span>
            </>
          }
          lede={
            <>
              The static shell is public so the page can render and ask for a key. The{" "}
              {web.dashboard.data_endpoints.length} data routes are authenticated by
              default and answer without one in the same typed envelope every other error
              uses — so a dashboard that shows nothing tells you why, instead of showing
              zeros that look like a quiet system.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <Figure
              {...figureOf(webCapture("dashboard-auth"))}
              caption={
                <>
                  The same dashboard against a server started with no key configured. The
                  banner names the environment variable that opens the routes for local
                  viewing and the two headers a key can be sent in.
                </>
              }
              command={webCapture("dashboard-auth").produced_by}
            />
            <div>
              <CodeSample
                language="bash"
                accent="#ff6b6b"
                code={`curl -s http://127.0.0.1:8000/dashboard/data.json | python -m json.tool
curl -s -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:8000/dashboard/data.json
curl -s -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:8000/dashboard`}
                output={`{
    "error": {
        "message": "Missing API key (send 'Authorization: Bearer <key>' or 'X-API-Key: <key>')",
        "type": "invalid_request_error",
        "param": null,
        "code": "invalid_api_key"
    }
}
401
200`}
                outputLabel="what those three commands printed"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The page itself is 200 and the data is 401, which is the split that lets
                the shell render and prompt.{" "}
                <code className="font-mono text-xs">EFFGEN_PUBLIC_DASHBOARD=1</code> opens
                the data routes for local viewing; in a shared deployment they stay closed
                and access is restricted at the ingress.
              </p>
              <a
                href={DOCS_AUTH_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-4 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                docs/server/auth.md
                <FiExternalLink size={13} />
              </a>
            </div>
          </div>
        </Band>

        {/* Documentation */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-8 md:p-10 flex flex-col md:flex-row md:items-center gap-6 justify-between">
              <div className="max-w-2xl">
                <h2 className="text-2xl font-black text-gray-900 dark:text-white">
                  The reference for <span className="gradient-text">the web surfaces</span>
                </h2>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  What each panel measures, the shape of every response, the span and
                  topology payloads, the keyboard table, the accessibility guarantees and
                  the tests that hold them, and{" "}
                  <code className="font-mono text-xs">record_run</code> for an integration
                  that bypasses <code className="font-mono text-xs">Agent</code>.
                </p>
                <a
                  href={DOCS_SERVER_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  <FiShare2 size={13} />
                  docs/server/openai-compat.md — the API the same server serves
                  <FiExternalLink size={13} />
                </a>
              </div>
              <a
                href={DOCS_DASHBOARD_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex shrink-0 items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
              >
                <FiFileText size={15} />
                docs/dx/dashboard.md
                <FiExternalLink size={14} />
              </a>
            </div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
