// The documentation's table of contents, and the only place it is written down.
//
// The sidebar groups, the route table in `App.tsx`, the breadcrumb trail, the
// previous/next pair at the foot of every page, the "see also" blocks and the
// placeholder a page shows before it is written all read this file. Adding a
// page means adding one entry here and one line to `PAGE_COMPONENTS` in
// `App.tsx`; nothing else has to be kept in step, because nothing else holds a
// second copy of the list.
//
// `source` names the file in the framework repository whose content the page is
// adapted from — `effgen --help` and the installed package outrank it, but it is
// where the writing starts, and it is what a reader is offered while the page is
// still a placeholder.

export interface DocPageEntry {
  /** Route under the docs base, always with a leading slash. */
  path: string
  /** Sidebar label, browser title and breadcrumb leaf. */
  title: string
  /** One sentence saying what the page is for. */
  lede: string
  /** Path in `github.com/ctrl-gaurav/effGen` this page is adapted from. */
  source?: string
}

export interface DocGroup {
  title: string
  /** Short id, used for the collapse state and as a CSS hook. */
  id: string
  pages: DocPageEntry[]
}

export const NAV: DocGroup[] = [
  {
    title: 'Getting started',
    id: 'getting-started',
    pages: [
      {
        path: '/introduction',
        title: 'Introduction',
        lede: 'What effGen is, what it ships, and what changed in 1.3.0.',
        source: 'README.md',
      },
      {
        path: '/installation',
        title: 'Installation',
        lede: 'Installing effGen and picking the extras your work needs, on CPU, Apple Silicon or a GPU.',
        source: 'docs/installation.md',
      },
      {
        path: '/quickstart',
        title: 'Quick start',
        lede: 'An agent that answers a question, from an empty shell to a printed result.',
        source: 'docs/tutorials/getting-started.md',
      },
      {
        path: '/first-project',
        title: 'Your first project',
        lede: 'What `effgen quickstart --init` writes into an empty directory, and how to run it.',
        source: 'docs/cli/new-project.md',
      },
      {
        path: '/configuration',
        title: 'Configuration',
        lede: 'Every setting an agent takes, where it can be set, and which source wins.',
        source: 'docs/cli/configuration.md',
      },
      {
        path: '/migration',
        title: 'Migrating to 1.3.0',
        lede: 'The ten changes in 1.3.0 existing code can see, the fourteen in 1.2.0, the eleven in 1.1.0, the four in 1.0.1, the three breaking changes in 1.0.0, and what each one asks you to change.',
        source: 'docs/migration.md',
      },
      {
        path: '/faq',
        title: 'FAQ',
        lede: 'The questions that come up first: hardware, keys, cost, offline use and model choice.',
        source: 'docs/faq.md',
      },
      {
        path: '/releases',
        title: 'Release notes',
        lede: 'What each release changed, newest first.',
        source: 'CHANGELOG.md',
      },
    ],
  },
  {
    title: 'Core concepts',
    id: 'core-concepts',
    pages: [
      {
        path: '/agents',
        title: 'Agents',
        lede: 'The `Agent` class, the config it takes and the response it returns.',
        source: 'docs/api/reference.md',
      },
      {
        path: '/presets',
        title: 'Presets',
        lede: 'Ready-made agent configurations, what each one turns on, and how to start from one.',
        source: 'docs/presets',
      },
      {
        path: '/models',
        title: 'Models and loading',
        lede: 'Naming a model, loading it, and what `load_model` gives you back.',
        source: 'docs/models/registry.md',
      },
      {
        path: '/providers',
        title: 'Providers',
        lede: 'Every provider adapter, the keys it reads, and what it supports.',
        source: 'docs/providers/parity.md',
      },
      {
        path: '/openai-compatible',
        title: 'Any OpenAI-compatible server',
        lede: 'Pointing an agent at vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio or a gateway with `base_url`.',
        source: 'docs/models/openai-compatible.md',
      },
      {
        path: '/local-models',
        title: 'Local models and engines',
        lede: 'Running weights on your own machine through the transformers, vLLM, GGUF and MLX engines.',
        source: 'docs/models/registry.md',
      },
      {
        path: '/catalog',
        title: 'The model catalog and pricing',
        lede: 'The bundled catalog: what it records about each model, and how to browse it.',
        source: 'docs/models/registry.md',
      },
      {
        path: '/routing',
        title: 'Model routing and fallback',
        lede: 'The policy router: how a model is chosen, and what happens when the first choice is unusable.',
        source: 'docs/models/router.md',
      },
      {
        path: '/tool-calling',
        title: 'Tool calling',
        lede: 'The strategies a model can be asked for tools with, and the one shape a turn reports them in.',
        source: 'docs/models/tool-calls.md',
      },
      {
        path: '/generation',
        title: 'Generation controls and structured output',
        lede: 'Sampling, limits, reasoning effort and asking a model for JSON that matches a schema.',
        source: 'docs/models/openai_advanced.md',
      },
    ],
  },
  {
    title: 'Tools and execution',
    id: 'tools',
    pages: [
      {
        path: '/tools',
        title: 'Tools',
        lede: 'What a tool is, how an agent is given one, and what calling one returns.',
        source: 'docs/tools/index.md',
      },
      {
        path: '/tools/gallery',
        title: 'Tool gallery',
        lede: 'Every built-in tool, with a runnable snippet for each.',
        source: 'docs/tools/gallery.md',
      },
      {
        path: '/native-provider-tools',
        title: 'Provider-native tools',
        lede: 'Web search, code execution and file search run by the provider rather than by effGen.',
        source: 'docs/tools/openai_native.md',
      },
      {
        path: '/custom-tools',
        title: 'Writing tools and plugins',
        lede: 'Turning a function into a tool, and packaging tools as an installable plugin.',
        source: 'docs/tutorials/custom-tools.md',
      },
      {
        path: '/execution',
        title: 'Code execution and the sandbox',
        lede: 'Running model-written code with a container, a timeout and a filesystem it cannot leave.',
        source: 'docs/security/codeexecutor.md',
      },
      {
        path: '/protocols',
        title: 'Protocols — MCP, A2A, ACP',
        lede: 'Connecting to tools and agents that speak Model Context Protocol, A2A or ACP.',
        source: 'docs/tools/protocols.md',
      },
    ],
  },
  {
    title: 'Knowledge and context',
    id: 'knowledge',
    pages: [
      {
        path: '/memory',
        title: 'Memory',
        lede: 'Short-term and long-term memory, what each stores, and when either is consulted.',
        source: 'docs/api/reference.md',
      },
      {
        path: '/sessions',
        title: 'Sessions and conversation history',
        lede: 'Carrying a conversation across runs with `run(session=...)`, and where it is kept.',
        source: 'docs/guides/sessions-and-checkpoints.md',
      },
      {
        path: '/compaction',
        title: 'Context compaction',
        lede: 'What happens when a conversation outgrows the context window, and how to choose what is dropped.',
        source: 'docs/guides/context-compaction.md',
      },
      {
        path: '/rag',
        title: 'RAG',
        lede: 'Indexing documents, retrieving passages, and answering with citations back to them.',
        source: 'docs/tutorials/rag-pipeline.md',
      },
      {
        path: '/multimodal',
        title: 'Multimodal',
        lede: 'Sending images, audio and video to a model, and what each provider accepts.',
        source: 'docs/multimodal/overview.md',
      },
    ],
  },
  {
    title: 'Prompts',
    id: 'prompts',
    pages: [
      {
        path: '/prompts',
        title: 'Prompt library',
        lede: 'The bundled templates, how they are rendered, and how to use one in an agent.',
        source: 'docs/prompts/library.md',
      },
      {
        path: '/prompts/gallery',
        title: 'Prompt gallery',
        lede: 'Every template in the library, by domain, with its variables.',
        source: 'docs/prompts/gallery.md',
      },
      {
        path: '/prompts/authoring',
        title: 'Authoring templates and the playground',
        lede: 'Writing your own template, and trying one against a model before you ship it.',
        source: 'docs/prompts/playground.md',
      },
    ],
  },
  {
    title: 'Orchestration',
    id: 'orchestration',
    pages: [
      {
        path: '/multi-agent',
        title: 'Multi-agent teams',
        lede: 'Several agents on one task: the orchestration patterns and how work is handed between them.',
        source: 'docs/tutorials/multi-agent.md',
      },
      {
        path: '/workflows',
        title: 'Workflows',
        lede: 'Declaring steps as a graph, running them in dependency order, and drawing the result.',
        source: 'docs/guides/sessions-and-checkpoints.md',
      },
      {
        path: '/checkpointing',
        title: 'Checkpointing and resumable runs',
        lede: 'Saving a workflow as it goes, and picking it up from the last step that finished.',
        source: 'docs/guides/sessions-and-checkpoints.md',
      },
      {
        path: '/middleware',
        title: 'Middleware',
        lede: 'The hooks a run passes through, what each one can change, and the order they fire in.',
        source: 'docs/guides/middleware.md',
      },
      {
        path: '/domains',
        title: 'Domains',
        lede: 'Domain packs: the prompt, tools and settings a subject area is set up with.',
        source: 'docs/api/reference.md',
      },
    ],
  },
  {
    title: 'Safety and reliability',
    id: 'safety',
    pages: [
      {
        path: '/guardrails',
        title: 'Guardrails',
        lede: 'Checking what goes into a model and what comes out of it, and what a block looks like.',
        source: 'docs/tutorials/guardrails.md',
      },
      {
        path: '/human-loop',
        title: 'Human in the loop',
        lede: 'Pausing a run for approval before a tool call, and resuming it afterwards.',
        source: 'docs/api/reference.md',
      },
      {
        path: '/security',
        title: 'Security',
        lede: 'Secret redaction, the supply-chain checks the project runs, and the SBOM it publishes.',
        source: 'docs/security/secrets.md',
      },
      {
        path: '/reliability',
        title: 'Reliability',
        lede: 'Retries, timeouts, circuit breakers and what the framework does when a provider is down.',
        source: 'docs/observability/reliability.md',
      },
      {
        path: '/errors',
        title: 'Errors and exceptions',
        lede: 'Every typed error effGen raises, what causes it, and what to do about it.',
        source: 'docs/api/conventions.md',
      },
    ],
  },
  {
    title: 'Server, clients and deployment',
    id: 'server',
    pages: [
      {
        path: '/api-server',
        title: 'API server',
        lede: 'Running `effgen serve`: authentication, roles, audit logging and rate limits.',
        source: 'docs/server/auth.md',
      },
      {
        path: '/openai-api',
        title: 'OpenAI-compatible API',
        lede: 'The endpoints `effgen serve` exposes, and talking to them with the official OpenAI SDK.',
        source: 'docs/server/openai-compat.md',
      },
      {
        path: '/clients',
        title: 'Clients and SDKs',
        lede: 'The Python and TypeScript clients: calling a running server from your own code.',
        source: 'docs/migration.md',
      },
      {
        path: '/deployment',
        title: 'Deployment',
        lede: 'Running the server on Docker, Kubernetes, AWS Lambda or a Cloudflare Worker.',
        source: 'docs/deploy/docker.md',
      },
      {
        path: '/hardware',
        title: 'Hardware and GPUs',
        lede: 'What effGen can see about the machine it is on, and how it decides what will fit.',
        source: 'docs/installation.md',
      },
    ],
  },
  {
    title: 'Operations',
    id: 'operations',
    pages: [
      {
        path: '/observability',
        title: 'Observability',
        lede: 'What a run records, where it goes, and how to look at it.',
        source: 'docs/observability/overview.md',
      },
      {
        path: '/metrics',
        title: 'Metrics',
        lede: 'The Prometheus instruments exposed at `/metrics`, their labels, and the structured log stream.',
        source: 'docs/observability/metrics.md',
      },
      {
        path: '/tracing',
        title: 'Tracing and spans',
        lede: 'The OpenTelemetry spans effGen emits, and exporting them to a collector.',
        source: 'docs/observability/tracing.md',
      },
      {
        path: '/slos',
        title: 'SLOs and alerting',
        lede: 'Error-budget tracking over a rolling window, and the alert rules that go with it.',
        source: 'docs/observability/slos.md',
      },
      {
        path: '/loadtest',
        title: 'Load testing, chaos and fuzz',
        lede: 'Putting a server or a provider under load, and injecting failure to see what holds.',
        source: 'docs/observability/loadtest.md',
      },
      {
        path: '/cost',
        title: 'Cost and budgets',
        lede: 'What a run cost, what the day has cost, and the budget that stops it going further.',
        source: 'docs/cli/cost.md',
      },
    ],
  },
  {
    title: 'Evaluation',
    id: 'evaluation',
    pages: [
      {
        path: '/evaluation',
        title: 'Evaluation and CI gates',
        lede: 'Scoring an agent against a test suite, and failing a build when the score drops.',
        source: 'docs/tutorials/evaluation.md',
      },
      {
        path: '/compare',
        title: 'Comparing models',
        lede: 'Running the same work through several models and reading the difference.',
        source: 'docs/tutorials/evaluation.md',
      },
    ],
  },
  {
    title: 'The command line',
    id: 'cli',
    pages: [
      {
        path: '/cli',
        title: 'CLI overview',
        lede: 'Every `effgen` command, what it is for, and the options they all share.',
        source: 'docs/dx/cli.md',
      },
      {
        path: '/cli/run',
        title: 'run and chat',
        lede: 'One-shot answers with `effgen run`, and a conversation with `effgen chat`.',
        source: 'docs/dx/cli.md',
      },
      {
        path: '/cli/code',
        title: 'effgen code',
        lede: 'A coding agent in the terminal: unified diffs, permission modes, and undo.',
        source: 'docs/cli/code.md',
      },
      {
        path: '/cli/top',
        title: 'effgen top',
        lede: 'A live view of runs, spend, providers and tool calls while they happen.',
        source: 'docs/cli/top.md',
      },
      {
        path: '/cli/reports',
        title: 'Reports and run cards',
        lede: 'Turning a saved result into a self-contained HTML report or a single-run card.',
        source: 'docs/cli/top.md',
      },
      {
        path: '/cli/history',
        title: 'Runs and sessions history',
        lede: 'What effGen keeps about past runs and conversations, and how to search it.',
        source: 'docs/guides/sessions-and-checkpoints.md',
      },
      {
        path: '/cli/appearance',
        title: 'Appearance and themes',
        lede: 'How the command line decides what to draw: themes, colour, animation and width.',
        source: 'docs/cli/appearance.md',
      },
      {
        path: '/cli/batch',
        title: 'Batch and automation',
        lede: 'Running many prompts from a file, and wiring `effgen` into a script.',
        source: 'docs/dx/cli.md',
      },
    ],
  },
  {
    title: 'Web and developer surfaces',
    id: 'surfaces',
    pages: [
      {
        path: '/dashboard',
        title: 'Dashboard',
        lede: 'The real-time view the server serves at `/dashboard`, and what each panel reads.',
        source: 'docs/dx/dashboard.md',
      },
      {
        path: '/playground',
        title: 'Playground',
        lede: 'Trying a model, a prompt and a set of tools in the browser, with nothing installed.',
        source: 'docs/dx/dashboard.md',
      },
      {
        path: '/jupyter',
        title: 'Jupyter',
        lede: 'The IPython magics that run an agent from a notebook cell.',
        source: 'docs/dx/jupyter.md',
      },
      {
        path: '/vscode',
        title: 'VS Code',
        lede: 'The editor extension: running a prompt, reading a result and watching cost.',
        source: 'docs/dx/vscode.md',
      },
      {
        path: '/debug',
        title: 'Debugging',
        lede: 'Stepping through a run to see the prompt, the tool calls and the decision at each turn.',
        source: 'docs/dx/cli.md',
      },
    ],
  },
  {
    title: 'Reference',
    id: 'reference',
    pages: [
      {
        path: '/api-reference',
        title: 'API reference',
        lede: 'Every name the `effgen` package exports, with its signature.',
        source: 'docs/api/reference.md',
      },
      {
        path: '/tutorials',
        title: 'Tutorials',
        lede: 'End-to-end builds: a research agent, a coding agent, RAG, guardrails and a deployment.',
        source: 'docs/tutorials/getting-started.md',
      },
      {
        path: '/cookbook',
        title: 'Cookbook',
        lede: 'Short recipes for one task each, ready to paste into a file and run.',
        source: 'docs/cookbook/README.md',
      },
      {
        path: '/examples',
        title: 'Examples',
        lede: 'The runnable programs that ship with the framework, and what each one shows.',
        source: 'docs/tutorials/getting-started.md',
      },
    ],
  },
]

/** Every page, in sidebar order — the order previous/next walks. */
export const PAGES: DocPageEntry[] = NAV.flatMap((group) => group.pages)

const BY_PATH = new Map(PAGES.map((page) => [page.path, page]))
const GROUP_BY_PATH = new Map(
  NAV.flatMap((group) => group.pages.map((page) => [page.path, group] as const)),
)

/** The entry for a route, or `undefined` for a route the navigation does not carry. */
export function pageFor(path: string): DocPageEntry | undefined {
  return BY_PATH.get(path)
}

/** The group a route sits in, which is what the breadcrumb trail names. */
export function groupFor(path: string): DocGroup | undefined {
  return GROUP_BY_PATH.get(path)
}

/** The pages either side of this one in reading order. */
export function neighboursOf(path: string): {
  prev?: DocPageEntry
  next?: DocPageEntry
} {
  const i = PAGES.findIndex((page) => page.path === path)
  if (i < 0) return {}
  return { prev: PAGES[i - 1], next: PAGES[i + 1] }
}

/**
 * Routes that no longer exist, and where each one goes instead.
 *
 * `/home` was a second overview competing with the introduction; `/dx` and
 * `/guides` were single pages holding what are now several. A link to any of
 * them — from a bookmark, a search result or an older release's README — lands
 * on the page that took the topic over rather than on a 404.
 */
export const REDIRECTS: Record<string, string> = {
  '/home': '/introduction',
  '/dx': '/cli',
  '/guides': '/tutorials',
}

/** Where the framework's own documentation for a page lives. */
export const FRAMEWORK_DOCS = 'https://github.com/ctrl-gaurav/effGen/blob/main'
