// The grouping and the prose `/agents` renders.
//
// Every parameter name, type, default, field, hook signature and strategy
// default on the page comes from `data/effgen.json`, which
// `scripts/gen_site_data.py` introspects off the installed classes. Nothing
// here restates any of them.
//
// What is written here is the one thing the classes do not state: which of the
// 52 `AgentConfig` fields this page documents, and in what order. The
// assertions below fail the build if a name here stops existing on the class,
// so a field renamed in the framework cannot leave a row on the page pointing
// at nothing.

import { siteData } from "@/components/siteData";

const FRAMEWORK_DOCS = "https://github.com/ctrl-gaurav/effGen/blob/main";

export const DOCS_REFERENCE_URL = `${FRAMEWORK_DOCS}/docs/api/reference.md`;
export const DOCS_MIDDLEWARE_URL = `${FRAMEWORK_DOCS}/docs/guides/middleware.md`;
export const DOCS_SESSIONS_URL = `${FRAMEWORK_DOCS}/docs/guides/sessions-and-checkpoints.md`;
export const DOCS_COMPACTION_URL = `${FRAMEWORK_DOCS}/docs/guides/context-compaction.md`;
export const DOCS_TOOLS_URL = `${FRAMEWORK_DOCS}/docs/tools/index.md`;
export const DOCS_RAG_URL = `${FRAMEWORK_DOCS}/docs/tutorials/rag-pipeline.md`;
export const DOCS_MULTI_AGENT_URL = `${FRAMEWORK_DOCS}/docs/tutorials/multi-agent.md`;

const api = siteData.api;

/* ── The configuration rows the page shows ── */

/**
 * The `AgentConfig` fields this page documents, grouped by what they decide.
 *
 * `AgentConfig` carries 52 fields. Printing all of them would be a reference
 * page, not an orientation, so this names the ones a first agent is written
 * with — and `configRows` reads each one's real annotation and default out of
 * the generated data rather than repeating them here.
 */
export interface ConfigGroup {
  id: string;
  title: string;
  what: string;
  fields: string[];
}

export const configGroups: ConfigGroup[] = [
  {
    id: "what-it-is",
    title: "What the agent is",
    what: "The model it runs on, the tools it may call, and what it is told it is for.",
    fields: ["model", "models", "tools", "system_prompt", "name"],
  },
  {
    id: "how-it-runs",
    title: "How the loop runs",
    what: "How many turns it may take, how the model samples, and whether it streams.",
    fields: ["max_iterations", "temperature", "max_tokens", "seed", "enable_streaming"],
  },
  {
    id: "what-it-remembers",
    title: "What it remembers",
    what: "Whether the conversation is kept, how it is measured, and what leaves when it stops fitting.",
    fields: ["enable_memory", "max_context_length", "compaction_strategy", "tokenizer"],
  },
  {
    id: "where-it-calls",
    title: "Where the calls go",
    what: "Which provider, which endpoint, which credential. All three are optional; a bare model id resolves them.",
    fields: ["provider", "base_url", "api_key"],
  },
  {
    id: "what-it-refuses",
    title: "What it refuses, and what it reports",
    what: "The hooks around the loop, the checks over the text, the approval gate, and what a failure does.",
    fields: ["middleware", "guardrails", "approval_mode", "raise_on_error"],
  },
  {
    id: "what-it-returns",
    title: "What comes back",
    what: "Ask for a shape, and the answer is validated against it before the run reports success. Ask for citations, and the answer marks the passages it used.",
    fields: ["output_format", "output_schema", "cite_sources"],
  },
];

/** Every field name any group names, so the assertion has one list to check. */
const documented = configGroups.flatMap((group) => group.fields);

/**
 * Fail the build if this page documents a field `AgentConfig` no longer has.
 *
 * The generated data is the class; this list is not. Without the check, a field
 * renamed upstream would leave a row here describing a parameter that does not
 * exist, which is exactly the failure this site has shipped before.
 */
function assertEveryDocumentedFieldExists(): void {
  const real = api.agent_config.map((field) => field.name);
  const missing = documented.filter((name) => !real.includes(name));
  if (missing.length) {
    throw new Error(
      "app/agents/agentsData.ts documents AgentConfig fields that the installed " +
        `framework does not have: ${missing.join(", ")}. Re-run ` +
        "scripts/gen_site_data.py and fix the list.",
    );
  }
}

assertEveryDocumentedFieldExists();

/** The generated row for one `AgentConfig` field. */
export function configField(name: string) {
  const field = api.agent_config.find((row) => row.name === name);
  if (!field) {
    throw new Error(`No AgentConfig field "${name}" in data/effgen.json.`);
  }
  return field;
}

/** The generated row for one `AgentResponse` field. */
export function responseField(name: string) {
  const field = api.agent_response.find((row) => row.name === name);
  if (!field) {
    throw new Error(`No AgentResponse field "${name}" in data/effgen.json.`);
  }
  return field;
}

/**
 * The response fields the page walks through, with what each one is for.
 *
 * The type and the default come from the class; only the sentence is written
 * here, because a dataclass carries no prose about why a field exists.
 */
export const responseNotes: { name: string; what: string }[] = [
  { name: "output", what: "The answer, as text. Passing the response to str() gives the same string." },
  { name: "success", what: "Whether the run answered, rather than being stopped by the loop or failing." },
  { name: "stop_reason", what: "The exit the run took — \"final_answer\" when it answered. response.outcome reads it as answered, stopped or failed, and since 1.3.0 response.termination as done, not_possible, stuck, tool_failed or error." },
  { name: "partial", what: "What a stopped run had reached: its tool observations, its last reasoning line and a one-line text." },
  { name: "tool_calls", what: "The calls the run made, as records. Still compares and casts as their count." },
  { name: "iterations", what: "How many turns of the loop it took." },
  { name: "tokens_used", what: "Prompt and completion tokens across every model call in the run." },
  { name: "execution_time", what: "Wall-clock seconds from the call to the answer." },
  { name: "sources", what: "The documents a retrieval-backed answer drew on." },
  { name: "citations", what: "The specific passages behind the answer, each with its source." },
  { name: "execution_trace", what: "Every step in order: the thought, the action, the observation." },
  { name: "metadata", what: "Cost, per-call token counts, the partial text under partial_output, what redaction removed, since 1.1.0 the run's thread and its context budget, since 1.2.0 its ledger, and since 1.3.0 tool_results, unavailable_tools, tool_calling and answer_source." },
];

/* ── The presets ── */

/** One accent per preset, in the order `effgen presets` lists them. */
export const presetAccents: Record<string, string> = {
  math: "#00ff88",
  research: "#00e5ff",
  coding: "#ffd700",
  general: "#a78bfa",
  rag: "#5eead4",
  minimal: "#94a3b8",
  multimodal: "#f472b6",
  notify: "#ff9500",
  media: "#ff6b6b",
};

/**
 * Fail the build if a preset ships without an accent, or an accent names a
 * preset that no longer ships.
 */
function assertAccentsCoverEveryPreset(): void {
  const shipped = siteData.presets.items.map((preset) => preset.name);
  const coloured = Object.keys(presetAccents);
  const missing = shipped.filter((name) => !coloured.includes(name));
  const unknown = coloured.filter((name) => !shipped.includes(name));
  if (missing.length || unknown.length) {
    throw new Error(
      "app/agents/agentsData.ts no longer matches the preset list in " +
        "data/effgen.json. " +
        [
          missing.length ? `Not coloured: ${missing.join(", ")}.` : "",
          unknown.length ? `Coloured but not shipped: ${unknown.join(", ")}.` : "",
        ]
          .filter(Boolean)
          .join(" "),
    );
  }
}

assertAccentsCoverEveryPreset();

/* ── Ways to give an agent a tool ── */

export interface ToolRoute {
  title: string;
  what: string;
  accent: string;
}

// Four routes, from `docs/tools/index.md` and the tool registry itself.
export const toolRoutes: ToolRoute[] = [
  {
    title: "The built-in registry",
    what:
      "Ask the registry for one of the tools the framework ships and put it in the list.",
    accent: "#00ff88",
  },
  {
    title: "A function of yours",
    what:
      "Decorate it with @tool, or wrap it with Tool.from_function. The name, the description and the parameter schema are read from the signature, the type hints and the docstring.",
    accent: "#00e5ff",
  },
  {
    title: "The provider's own tools",
    what:
      "Web search, code interpreter, file search, computer use and the text editor, executed by the provider rather than by your process.",
    accent: "#a78bfa",
  },
  {
    title: "A server that speaks a protocol",
    what:
      "MCP, A2A and ACP servers are mounted as tools, so an agent reaches what those servers expose without a wrapper per tool.",
    accent: "#ffd700",
  },
];

/* ── Orchestration ── */

export interface ResumeRow {
  state: string;
  onResume: string;
}

// From `docs/guides/sessions-and-checkpoints.md`, "What resuming does with each
// node", confirmed against a real interrupted run.
export const resumeRows: ResumeRow[] = [
  { state: "completed", onResume: "not run again — its output is restored and flows downstream" },
  { state: "skipped", onResume: "stays skipped, with the reason it was skipped for" },
  { state: "failed", onResume: "retried, which is usually why the run is being resumed" },
  { state: "never started", onResume: "run normally" },
];
