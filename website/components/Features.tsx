"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiActivity, FiBookOpen, FiBox, FiCode, FiCpu, FiDatabase, FiDollarSign,
  FiGitBranch, FiImage, FiLayers, FiLock, FiMonitor, FiServer, FiShield,
  FiTerminal, FiTool, FiUsers, FiX, FiZap,
} from "react-icons/fi";
import type { IconType } from "react-icons";
import { useRef, useState, useCallback, useEffect } from "react";
import Container from "./Container";
import { useReducedMotion } from "./useReducedMotion";
import { highlightCode } from "./syntaxHighlight";
import {
  commandCount, modelCount, presetCount, providerCount, providersWithCatalog,
  publicNameCount, siteData, subcommandCount, toolCount,
} from "./siteData";
import { accentTextStyle } from "./accentText";

// Nothing on this page is a number someone typed. `data/effgen.json` is written
// from the installed package by `scripts/gen_site_data.py`, so a count here is
// wrong only if the package changed and the file was not regenerated — which
// `gen_site_data.py --check` fails on.
const cat = siteData.tools.category_counts;
const localEngines = siteData.models.local_engines;
const promptTemplates = siteData.prompts.templates;
const promptLibrary = siteData.prompts.library;
const promptDomains = siteData.prompts.domains.length;
const presetNames = siteData.presets.items.map((preset) => preset.name);

/* ── Animated visuals for large cards ── */

function TaskDecompAnimation() {
  return (
    <div className="relative w-full h-40 flex items-center justify-center">
      <svg viewBox="0 0 400 180" className="w-full h-full" fill="none">
        <defs>
          <filter id="glow-decomp">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Main task box */}
        <motion.rect
          x="145" y="5" width="110" height="34" rx="6"
          stroke="#00ff88" strokeWidth="2" fill="#00ff8810"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        />
        <motion.text x="200" y="27" textAnchor="middle" fill="#00ff88" fontSize="11" fontFamily="monospace"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
        >Complex Task</motion.text>

        {/* Splitting lines */}
        {[{ x: 80 }, { x: 200 }, { x: 320 }].map((pos, i) => (
          <motion.line
            key={i}
            x1="200" y1="39" x2={pos.x} y2="75"
            stroke="#00ff8860" strokeWidth="1" strokeDasharray="4 3"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.5 + i * 0.15 }}
          />
        ))}

        {/* Sub-task boxes */}
        {[
          { x: 25, label: "Research", color: "#00e5ff" },
          { x: 145, label: "Analyze", color: "#a78bfa" },
          { x: 265, label: "Synthesize", color: "#ffd700" },
        ].map((st, i) => (
          <motion.g key={i}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 + i * 0.15 }}
          >
            <rect x={st.x} y="75" width="110" height="30" rx="5" stroke={st.color} strokeWidth="1" fill={`${st.color}10`} />
            <text x={st.x + 55} y="95" textAnchor="middle" fill={st.color} fontSize="11" fontFamily="monospace">{st.label}</text>
          </motion.g>
        ))}

        {/* Result lines flowing back up */}
        {[{ x: 80 }, { x: 200 }, { x: 320 }].map((pos, i) => (
          <motion.line
            key={`r-${i}`}
            x1={pos.x} y1="105" x2="200" y2="135"
            stroke="#00ff8840" strokeWidth="1" strokeDasharray="3 4"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 1.4 + i * 0.1 }}
          />
        ))}

        {/* Final result box */}
        <motion.rect
          x="145" y="130" width="110" height="30" rx="6"
          stroke="#00ff88" strokeWidth="1.5" fill="#00ff8820"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 1, 1, 0.8, 1] }}
          transition={{ delay: 1.8, duration: 0.6 }}
        />
        <motion.text x="200" y="150" textAnchor="middle" fill="#00ff88" fontSize="11" fontFamily="monospace" fontWeight="bold"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2 }}
        >Merged Result</motion.text>

        {/* Animated pulse traveling along lines */}
        <motion.circle
          r="3" fill="#00ff88"
          filter="url(#glow-decomp)"
          animate={{
            cx: [200, 80, 80, 200, 200, 320, 320, 200, 200],
            cy: [39, 75, 105, 135, 39, 75, 105, 135, 39],
            opacity: [1, 1, 1, 1, 1, 1, 1, 1, 0],
          }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", repeatDelay: 1 }}
        />
      </svg>
    </div>
  );
}

function OrchestrationAnimation() {
  const agents = [
    { label: "Research", angle: 0, color: "#00e5ff", desc: "Web & APIs" },
    { label: "Coding", angle: 120, color: "#a78bfa", desc: "Python & JS" },
    { label: "Analysis", angle: 240, color: "#ffd700", desc: "Data & Reports" },
  ];

  return (
    <div className="relative w-full h-44 flex items-center justify-center">
      <svg viewBox="0 0 340 220" className="w-full h-full" fill="none">
        {/* Center orchestrator */}
        <motion.circle
          cx="170" cy="110" r="28"
          stroke="#00ff88" strokeWidth="1.5" fill="#00ff8812"
          animate={{ r: [28, 30, 28] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <text x="170" y="107" textAnchor="middle" fill="#00ff88" fontSize="9" fontFamily="monospace">Orchestrator</text>
        <motion.circle
          cx="170" cy="118" r="2" fill="#00ff88"
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />

        {/* Orbit ring */}
        <motion.circle
          cx="170" cy="110" r="75"
          stroke="#00ff8820" strokeWidth="0.5" fill="none" strokeDasharray="4 4"
          animate={{ strokeDashoffset: [0, 20] }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />

        {/* Agent nodes */}
        {agents.map((agent, i) => {
          const rad = (agent.angle * Math.PI) / 180;
          const cx = 170 + Math.cos(rad) * 75;
          const cy = 110 + Math.sin(rad) * 75;
          return (
            <motion.g
              key={i}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + i * 0.2 }}
            >
              {/* Connection line */}
              <motion.line
                x1="170" y1="110" x2={cx} y2={cy}
                stroke={`${agent.color}50`} strokeWidth="1.5" strokeDasharray="3 3"
                animate={{ strokeDashoffset: [0, -12] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
              />
              {/* Node */}
              <circle cx={cx} cy={cy} r="18" stroke={agent.color} strokeWidth="1" fill={`${agent.color}12`} />
              <text x={cx} y={cy + 1} textAnchor="middle" fill={agent.color} fontSize="9" fontFamily="monospace">{agent.label}</text>
              {/* Description text */}
              <text x={cx} y={cy + 12} textAnchor="middle" fill="#888888" fontSize="7" fontFamily="monospace">{agent.desc}</text>
            </motion.g>
          );
        })}

        {/* Data pulses between orchestrator and agents */}
        {agents.map((agent, i) => {
          const rad = (agent.angle * Math.PI) / 180;
          const cx = 170 + Math.cos(rad) * 75;
          const cy = 110 + Math.sin(rad) * 75;
          return (
            <motion.circle
              key={`pulse-${i}`}
              r="2" fill={agent.color}
              style={{ filter: `drop-shadow(0 0 3px ${agent.color})` }}
              animate={{
                cx: [170, cx, cx, 170],
                cy: [110, cy, cy, 110],
                opacity: [1, 1, 0, 0],
              }}
              transition={{ duration: 3, delay: i * 1, repeat: Infinity, ease: "easeInOut" }}
            />
          );
        })}
      </svg>
    </div>
  );
}

function PresetStripVisual() {
  // The nine presets, their tool counts and their temperatures come from the
  // package; the colours are the only thing chosen here.
  const colors = [
    "#00ff88", "#00e5ff", "#a78bfa", "#ffd700", "#00c896",
    "#ff9500", "#f472b6", "#ec4899", "#f59e0b",
  ];

  return (
    <div className="flex gap-2 w-full overflow-x-auto py-1">
      {siteData.presets.items.map((preset, i) => {
        const color = colors[i % colors.length];
        return (
          <motion.div
            key={preset.name}
            className="flex-shrink-0 flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 dark:bg-black/30 border border-gray-200 dark:border-gray-800"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
            whileHover={{ y: -2, borderColor: `${color}50` }}
          >
            <div className="flex flex-col">
              <span className="text-[10px] font-bold uppercase" style={{ color }}>
                {preset.name}
              </span>
              <div className="flex items-center gap-2">
                <span className="text-[9px] text-gray-600 dark:text-gray-400">
                  {preset.tool_count} {preset.tool_count === 1 ? "tool" : "tools"}
                </span>
                <div className="w-8 h-1 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${preset.temperature * 100}%`,
                      background: preset.temperature <= 0.3 ? "#00e5ff" : preset.temperature <= 0.5 ? "#00ff88" : "#ffd700",
                    }}
                  />
                </div>
                <span className="text-[9px] text-gray-600 dark:text-gray-400">t={preset.temperature}</span>
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}

function MemoryFlowVisual() {
  return (
    <div className="flex items-center gap-1 text-[10px] font-mono py-2">
      <motion.span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-400 border border-violet-500/20"
        animate={{ opacity: [0.6, 1, 0.6] }} transition={{ duration: 2, repeat: Infinity }}
      >short-term</motion.span>
      <motion.span className="text-gray-600 dark:text-gray-400"
        animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1.5, repeat: Infinity }}
      >&rarr;</motion.span>
      <motion.span className="px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400 border border-violet-500/30"
        animate={{ opacity: [0.6, 1, 0.6] }} transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
      >long-term</motion.span>
      <motion.span className="text-gray-600 dark:text-gray-400"
        animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1.5, repeat: Infinity, delay: 0.3 }}
      >&rarr;</motion.span>
      <motion.span className="px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-400 border border-violet-500/40"
        animate={{ opacity: [0.6, 1, 0.6] }} transition={{ duration: 2, repeat: Infinity, delay: 1 }}
      >vector</motion.span>
    </div>
  );
}

/* ── Feature data ── */

interface FeatureItem {
  group: string;
  icon: IconType;
  title: string;
  description: string;
  features: string[];
  accent: string;
  visual?: "decomp" | "orchestration" | "presets" | "memory";
  code?: string;
  expandedContent?: string;
}

const groups = [
  "Agents and tools",
  "Models",
  "Knowledge and context",
  "Orchestration",
  "Running it for real",
  "Working with it",
];

const features: FeatureItem[] = [
  /* ── Agents and tools ── */
  {
    group: "Agents and tools",
    icon: FiZap,
    title: "Agents, presets and one result object",
    description: `Build an agent from a config, or take one of the ${presetCount} presets and be running in a line. Every run returns the same object, whatever the model behind it was.`,
    features: [
      `${presetCount} presets: ${presetNames.join(", ")}`,
      "AgentResponse carries .text, .success, .outcome, .termination, .thread, .ledger, .tool_calls, .sources, .citations and .metadata",
      `${publicNameCount} names exported from the top-level package`,
    ],
    accent: "#00ff88",
    visual: "presets",
    code: 'from effgen import create_agent\n\nagent = create_agent("math", "gemini:gemini-3.1-flash-lite")\nprint(agent.run("What is 17 * 23 + 144 ** 0.5?").text)',
    expandedContent:
      "A preset bundles the tools, the temperature, the iteration cap and the system prompt for one kind of work, so create_agent(preset, model) is a working agent. Build the config yourself when you want something else — AgentConfig takes the model, the tools, the prompt, guardrails, middleware, a compaction strategy and the generation controls. Either way run() returns an AgentResponse: str(response) is the answer, .text is the same string, .success says whether the run answered, .outcome says whether it answered, stopped or failed, .tool_calls lists the calls it made, .sources and .citations carry the URLs a grounded run actually retrieved, .partial carries what a stopped run had reached, and .metadata carries cost, tokens and latency. AgentResponse is imported from effgen.core.agent rather than the top-level package.",
  },
  {
    group: "Agents and tools",
    icon: FiTool,
    title: `${toolCount} built-in tools, and your own`,
    description: `Eight categories, from search and scraping to documents, code execution and messaging. A tool is awaited and returns a typed result — never a dictionary you have to guess the shape of.`,
    features: [
      `information retrieval ${cat.information_retrieval} · data processing ${cat.data_processing} · external API ${cat.external_api}`,
      `code execution ${cat.code_execution} · communication ${cat.communication} · system ${cat.system} · computation ${cat.computation} · files ${cat.file_operations}`,
      "@tool turns a function into one; MCP, A2A and ACP connect what you did not write",
    ],
    accent: "#00e5ff",
    code: 'from effgen import Agent, AgentConfig, tool\n\n\n@tool(description="Return the length of a string.")\nasync def strlen(text: str) -> int:\n    return len(text)\n\n\nagent = Agent(AgentConfig(\n    model="gemini:gemini-3.1-flash-lite",\n    tools=[strlen],\n))\nr = agent.run("Use the strlen tool on the word \'effgen\'.")\nprint(r.text, "|", [c.name for c in r.tool_calls])',
    expandedContent:
      "Every tool has the same shape: await tool.execute(**kwargs) returns a ToolResult with success, output, error, execution_time, metadata and timestamp. There is no data field and the result is not a dictionary — a failed call comes back with success=False and a message in error rather than raising into your loop. Write your own with the @tool decorator or Tool.from_function, and the schema the model sees is built from the signature and the docstring. Provider-native tools (OpenAI web search and file search and the code interpreter, Gemini URL context, the Anthropic text editor and bash tools) are wired in where the API executes them itself, and MCP, A2A and ACP servers plug in beside the built-ins.",
  },
  {
    group: "Agents and tools",
    icon: FiLayers,
    title: "Middleware, sessions and compaction",
    description:
      "Hook the run without patching the loop, serve many conversations from one agent, and decide what gets dropped when the context window fills.",
    features: [
      "AgentMiddleware and MiddlewareChain run before and after the run, each model call and each tool call",
      "run(session=...) keeps a conversation, across processes",
      "SummarizeOldest, DropOldest, KeepFirstAndLast, KeepToolResults",
    ],
    accent: "#a78bfa",
    code: 'from effgen import Agent, AgentConfig\n\nfirst = Agent(AgentConfig(model="gemini:gemini-3.1-flash-lite"))\nfirst.run("My dog is named Pixel.", session="user-123")\n\nsecond = Agent(AgentConfig(model="gemini:gemini-3.1-flash-lite"))\nprint(second.run("What is my dog\'s name?", session="user-123").text)  # Pixel',
    expandedContent:
      "Middleware is where anything effGen does not ship as a subsystem belongs — an approval gate, a cache, a redaction pass, a spend cap. LoggingMiddleware and ToolApprovalMiddleware come with it. Sessions are the other half: pass session= to run() and one agent instance serves many conversations, each with its own history, stored so a later process picks the thread back up; effgen sessions lists, shows, exports and cleans them from the command line. Compaction decides what happens when the conversation approaches the window: summarise the oldest turns, drop them, keep the first and last, or keep the tool results and drop the prose.",
  },

  /* ── Models ── */
  {
    group: "Models",
    icon: FiCpu,
    title: `${providerCount} provider adapters, one agent API`,
    description: `${providersWithCatalog} of them ship a bundled catalog — ${modelCount} models with their context windows, capabilities and prices. Switching provider is switching one string.`,
    features: [
      "openai, anthropic, gemini, cerebras, groq, together, fireworks, replicate, hf",
      "openai_compatible serves whatever your own endpoint serves",
      "Provider-prefixed ids (openai:gpt-5-nano) resolve a name that exists in more than one place",
    ],
    accent: "#00ff88",
    code: 'from effgen import Agent, AgentConfig\n\nagent = Agent(AgentConfig(model="openai:gpt-5-nano"))\nprint(agent.run("Name the three primary colours of light, comma separated.").text)',
    expandedContent:
      "Adapters register on import and the registry resolves a model id to one of them, raising a named error listing the valid providers when a bare id is ambiguous. The same call shape reaches every adapter — streaming, tool calls and multimodal parts included — so the agent, the tools and the result object do not change when the provider does. effgen doctor prints which keys are present and how many models each provider carries; effgen models list prints the catalog.",
  },
  {
    group: "Models",
    icon: FiServer,
    title: "Any OpenAI-compatible server, or your own hardware",
    description: `One base_url points effGen at a server you already run. Or load the weights in-process with one of ${localEngines.length} local engines and talk to no one.`,
    features: [
      "vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, LiteLLM, a gateway",
      `Local engines: ${localEngines.join(", ")}`,
      "list_served_models() asks the endpoint what it has; an unreachable one raises BackendUnreachableError",
    ],
    accent: "#00e5ff",
    code: 'import os\n\nfrom effgen import Agent, AgentConfig\n\nagent = Agent(AgentConfig(\n    model="openai:gpt-5-nano",\n    base_url="http://127.0.0.1:8000/v1",\n    api_key=os.environ["EFFGEN_API_KEY"],\n))\nprint(agent.run("Reply with the single word: ready").text)',
    expandedContent:
      "A shared server loads the weights once, batches every caller's requests together and outlives any single run — which in-process loading cannot do. effGen consults base_url first, then EFFGEN_BASE_URL, then OPENAI_BASE_URL, then OPENAI_API_BASE, so pointing effGen somewhere does not redirect every other OpenAI client on the machine. Nothing about the endpoint is assumed: the ids are the server's, the context window defaults to 32,768 and says so when it is assuming, and a call reports no price rather than inventing a zero. A refused connection, a host that does not resolve and a route that does not exist are all reported as unreachable, separately from a server that answered badly.",
  },
  {
    group: "Models",
    icon: FiDollarSign,
    title: "A catalog that will not invent a price",
    description: `${modelCount} catalogued models with dated, priced entries — and silence rather than a fabricated $0 for a model that is not in it.`,
    features: [
      "effgen models browse filters by provider, capability, context and price",
      "Routing and fallback across providers, on cost, latency or first-available",
      "Streamed cost and tokens on every provider; per-model spend that adds up",
    ],
    accent: "#ffd700",
    code: "effgen models list\neffgen models browse --tools --max-price-out 1.0\neffgen cost today",
    expandedContent:
      "Every provider ships a dated catalog snapshot, and effgen models refresh diffs it against the live API. A fine-tuned or uncatalogued model has no price entry, and effGen says nothing rather than reporting it as free — a $0 that is really 'unknown' is the kind of number that ends up in a budget. Cost and token counts land on streamed calls as well as buffered ones. effgen cost prints today, the week and lifetime totals by provider, and sets a daily cap; a project scaffolded with effgen quickstart --init gets a $1.00/day cap when none is configured.",
  },

  /* ── Knowledge and context ── */
  {
    group: "Knowledge and context",
    icon: FiDatabase,
    title: "Memory across turns and runs",
    description:
      "Short-term working memory for the current conversation, long-term memory for facts worth keeping, and a vector store for retrieving by meaning.",
    features: [
      "Short-term, long-term and vector memory, wired into the agent",
      "Checkpoints snapshot a run's state as JSON — no pickle",
      "Sessions and run history are shared by the library, the CLI and the server",
    ],
    accent: "#a78bfa",
    visual: "memory",
    expandedContent:
      "Memory is on the agent, not bolted beside it: enable_memory=True and the loop stores and recalls without the caller managing a transcript. A checkpoint is the other kind of persistence — a snapshot of an in-progress run (its steps, iteration, partial output, tool calls, tokens, memory) written as JSON so a resumed run continues where it stopped, is inspectable, and cannot execute anything on load. effgen runs and effgen sessions browse both from the command line.",
  },
  {
    group: "Knowledge and context",
    icon: FiBookOpen,
    title: "Retrieval that cites what it used",
    description:
      "Ingest a directory, chunk it the way the content wants, search it with dense embeddings and BM25 together, and answer with the sources attached.",
    features: [
      "Semantic, code, table and hierarchical chunkers",
      "Hybrid search — dense, BM25 and keyword, fused by reciprocal rank",
      "response.sources and .citations come from the URLs a run retrieved, not from the prose",
    ],
    accent: "#00e5ff",
    expandedContent:
      "The rag preset takes a knowledge_base directory and wires the retrieval tool over it. Documents are loaded, de-duplicated by hash, chunked, embedded and indexed; queries run against dense vectors, BM25 and keyword matching at once and the rankings are fused, then optionally re-ranked by a cross-encoder, an LLM or a rule. What comes back on the response object is the set of sources the run actually fetched — a model writing a plausible-looking URL into its answer does not put it there.",
  },
  {
    group: "Knowledge and context",
    icon: FiImage,
    title: "Images, audio and video as input",
    description:
      "A typed content-part schema carries an image, an audio file or a video into the same run() call, with per-provider preprocessing and a clear error when a model cannot take it.",
    features: [
      "TextPart, ImagePart, AudioPart, VideoPart, ToolCallPart, ToolResultPart",
      "Bytes, paths, URLs, PIL images and arrays all accepted",
      "The multimodal preset and the multimodal_describe tool",
    ],
    accent: "#f472b6",
    expandedContent:
      "Message(role, 'text') still works — the content-part union is additive. image_from, audio_from and video_from accept bytes, a path, a URL, a PIL image or an array, and each adapter resizes, downsamples or samples keyframes as that provider requires. A model that cannot accept a part raises a capability error naming what it does not support, rather than silently dropping the attachment and answering about nothing.",
  },
  {
    group: "Knowledge and context",
    icon: FiCode,
    title: `${promptTemplates} prompt templates, ${promptLibrary} in the library`,
    description: `Deterministic, versioned templates across ${promptDomains} domains, rendered with named variables and checked by a golden and live evaluation harness.`,
    features: [
      `${promptDomains} domains: ${siteData.prompts.domains.join(", ")}`,
      "Zero-shot, chain-of-thought, few-shot, tool and structured variants",
      "effgen prompts list, show, render and run; EFFGEN_PROMPTS_DIR adds your own",
    ],
    accent: "#00ff88",
    code: "effgen prompts list\neffgen prompts show data.sql_from_nl.v1\neffgen prompts render data.sql_from_nl.v1 -i vars.json",
    expandedContent:
      "Each template declares a name such as data.sql_from_nl.v1, its domain, its variant and the variables it renders with, so a prompt is a versioned artifact rather than a string in a file. Rendering is deterministic — the same inputs give the same text. The evaluation harness checks the structured ones by parsing what they produce rather than by looking at it, and prompts run fails closed and reports what the call cost. Point EFFGEN_PROMPTS_DIR at a directory to add your own alongside the library.",
  },

  /* ── Orchestration ── */
  {
    group: "Orchestration",
    icon: FiUsers,
    title: "Teams of agents",
    description:
      "Coordinate specialist agents through sequential, parallel, hierarchical, collaborative and pipeline patterns, with shared state and a message bus between them.",
    features: [
      "Team patterns with lifecycle management",
      "Shared state and agent-to-agent messaging",
      "A team that cannot route a task fails closed and says so",
    ],
    accent: "#ff6b6b",
    visual: "orchestration",
    expandedContent:
      "The orchestrator registers agents, assigns work by role and records what each one did. Patterns cover the common shapes: run them one after another, run the independent ones together, put one in charge of the rest, have them critique each other, or chain them as a pipeline. Failure is explicit — a task that no member can take is reported rather than absorbed, so a team does not return a confident answer that nobody produced.",
  },
  {
    group: "Orchestration",
    icon: FiGitBranch,
    title: "Workflows as a graph, resumable",
    description:
      "Declare the pipeline as a DAG, let independent nodes run together, and hand it a checkpoint store so a run that dies does not start again from the top.",
    features: [
      "Cycle detection, conditional edges, automatic parallelism",
      "FileCheckpointStore and InMemoryCheckpointStore, keyed by run id",
      "workflow run --diagram draws the graph; YAML definitions run from the CLI",
    ],
    accent: "#ffd700",
    code: 'from effgen import Agent, AgentConfig, FileCheckpointStore, WorkflowDAG, WorkflowNode\n\nresearcher = Agent(AgentConfig(model="gemini:gemini-3.1-flash-lite"))\nwriter = Agent(AgentConfig(model="gemini:gemini-3.1-flash-lite"))\n\nstore = FileCheckpointStore("./checkpoints")\n\ndag = WorkflowDAG("briefing")\ndag.add_node(WorkflowNode(id="research", agent=researcher))\ndag.add_node(WorkflowNode(id="draft", agent=writer))\ndag.connect("research", "draft")\n\nresult = dag.run("Why run an agent on a small model?",\n                 checkpoint=store, run_id="briefing-1")\n\nprint(result.success)\nfor node in result.node_results:\n    print(node["id"], node["status"])',
    expandedContent:
      "The graph is validated by topological sort, so a cycle is rejected before anything runs, and nodes with no dependency on each other execute together. Edges carry typed data and can be conditional. With a checkpoint store and a run id, progress is written after each level of the graph: a completed node is restored and its output flows on without a model call, a failed node is retried, and a node that never started runs. Re-running a finished run replays its stored outputs, which makes the whole workflow idempotent under a job runner that retries. Resuming into a graph whose node ids changed raises rather than mixing two runs together.",
  },
  {
    group: "Orchestration",
    icon: FiBox,
    title: "Sub-agents and domains",
    description:
      "Let a complex task be decomposed and routed to specialists, or take a knowledge domain and turn it into a configured agent in one call.",
    features: [
      "AgentMode.AUTO routes per call; single mode stays single unless you ask",
      "Decomposition, parallel or sequential subtasks, then synthesis",
      "A domain carries its prompt, its recommended tools and its guardrails",
    ],
    accent: "#a78bfa",
    visual: "decomp",
    expandedContent:
      "Sub-agent mode is opt-in: a plain Agent(config).run(task) stays in single-agent mode and never quietly decomposes. Set the mode on the config or pass mode= on one call and the router scores the task, picks a strategy, runs the subtasks and synthesises the result, with a depth cap so recursion is bounded. Domains are the smaller version of the same idea — a domain object knows the prompt, the tools and the guardrails its field needs, and to_agent(model) hands you an agent already configured for it.",
  },

  /* ── Running it for real ── */
  {
    group: "Running it for real",
    icon: FiServer,
    title: "An OpenAI-compatible API server",
    description:
      "effgen serve exposes /v1/chat/completions and /v1/embeddings, so anything already speaking the OpenAI protocol can call your agents without changing its client.",
    features: [
      "Static API key or OIDC/JWT with RBAC, plus a redacted per-request audit log",
      "Per-IP rate limiting, a priority queue with backpressure, a warm model pool",
      "Cross-origin is fail-closed; the dashboard and metrics need auth unless you opt out",
    ],
    accent: "#00ff88",
    code: 'export EFFGEN_API_KEY="$(python -c \'import secrets; print(secrets.token_urlsafe(24))\')"\neffgen serve --port 8000',
    expandedContent:
      "The server is never unauthenticated by accident: with no EFFGEN_API_KEY set it mints an ephemeral one and prints it once, and dev mode has to be asked for by name and says so loudly. Requests carry through to the same agents the library builds, so a tool, a preset or a guardrail behaves the same over HTTP as it does in a script. A model the server does not host is rejected with a clear 400 rather than quietly answered by something else. effgen loadtest --url drives a running server through its auth and rate limiting.",
  },
  {
    group: "Running it for real",
    icon: FiActivity,
    title: "Metrics, traces, SLOs and alerts",
    description:
      "Structured logs with secrets redacted at the encoder, Prometheus histograms, OpenTelemetry spans, and burn-rate tracking against objectives you set.",
    features: [
      "Latency percentiles, token counters, error breakdown, GPU memory",
      "OTLP, Jaeger and Zipkin exporters with explicit sampling",
      "Retries with jitter, circuit breakers, bulkheads and timeouts",
    ],
    accent: "#00e5ff",
    expandedContent:
      "Redaction happens in the log encoder, so a secret cannot escape through a path that forgot to sanitise. Metrics are exposed for scraping and the same numbers drive the dashboard. Traces propagate across agents, so a multi-agent run is one trace rather than several. Errors are classified into a stable taxonomy — auth, not found, rate limited, transient, timeout, fatal — which is what lets a retry fire only when retrying could possibly help, and what makes an error message name the thing to fix rather than describe the symptom.",
  },
  {
    group: "Running it for real",
    icon: FiShield,
    title: "Guardrails and a sandbox",
    description:
      "Offline input and output checks with no model behind them, and code execution confined to a container or a user namespace rather than trusted.",
    features: [
      "PII, toxicity, topic, length and prompt-injection checks, composable in chains",
      "Docker sandbox when it is available, an unprivileged namespace when it is not",
      "Credential stores masked and the process table isolated inside the sandbox",
    ],
    accent: "#ff6b6b",
    expandedContent:
      "Guardrails run locally: no external API, no model call, nothing leaves the process to decide whether something is allowed. They compose into chains and ship as presets, and tool-level guardrails cover a tool's inputs, its outputs and whether it may be called at all. Code execution defaults to the strongest confinement available — Docker read-only with no network and dropped capabilities, otherwise a subprocess in its own user namespace — and each result reports which confinement was actually enforced rather than which was requested.",
  },
  {
    group: "Running it for real",
    icon: FiLock,
    title: "Evaluation, budgets and deployment",
    description:
      "Score an agent against suites, gate a change on the result, cap what a day can cost, and ship the whole thing to the runtime you already use.",
    features: [
      "Exact, substring, regex, semantic and model-judged scoring; regression against a baseline",
      "Daily and monthly budgets, warned before they are hit and enforced when they are",
      "Docker, Kubernetes, AWS Lambda and a Cloudflare edge proxy",
    ],
    accent: "#a78bfa",
    code: "effgen eval --suite math -m openai:gpt-5-nano --report eval.html\neffgen compare --models openai:gpt-5-nano,gemini:gemini-3.1-flash-lite --suite math\neffgen cost set-budget 1.0",
    expandedContent:
      "Evaluation suites cover maths, tool use, reasoning, safety and conversation, and a run can be scored by exact match, substring, regex, semantic similarity or another model acting as judge. Results diff against a stored baseline, so a regression is a number rather than an impression, and the whole thing runs in CI. Budgets are enforced where the spend is recorded, shared across worker processes, so a cap holds when several workers are calling at once.",
  },

  /* ── Working with it ── */
  {
    group: "Working with it",
    icon: FiTerminal,
    title: `${commandCount} commands and ${subcommandCount} sub-commands`,
    description:
      "The command line is a way to use the framework, not a demonstration of it. Everything that prints can print JSON, and everything that runs an agent takes a session.",
    features: [
      "run, chat, code, serve, top, battle, workflow, batch, eval, compare, cost, report…",
      "--json on every command, one valid document on a pipe",
      "Named themes, shell completion for bash, zsh and fish",
    ],
    accent: "#00ff88",
    code: 'effgen run "Use the calculator tool to work out 24344 * 334." \\\n  -m gemini:gemini-3.1-flash-lite -t calculator',
    expandedContent:
      "The themes are there for a reason rather than for decoration — high-contrast targets low-vision readers, monochrome keeps the structure without relying on hue, and NO_COLOR still turns colour off entirely. --json is a contract: a command that emits JSON emits exactly one document on stdout, so a pipeline can parse it without stripping banners. effgen doctor reports which provider keys are readable, what the machine can do, and what the coding agent needs; effgen quickstart --init scaffolds a project with effgen.yaml, an .env.example and a daily spend cap.",
  },
  {
    group: "Working with it",
    icon: FiCode,
    title: "effgen code, in your repository",
    description:
      "A coding agent that reads the repository, shows a unified diff before it writes, runs what it wrote, and fixes what failed — with the permission level you choose.",
    features: [
      "Four permission modes: plan, ask, auto-edit, and yes",
      "--undo reverses recent edits from a 100-entry journal; --review runs read-only",
      "A git allow-list, repository awareness, AGENTS.md, and 26 slash commands",
    ],
    accent: "#a78bfa",
    code: 'effgen code -p "Create slugify.py with a slugify(text) function" \\\n  -m gemini:gemini-3.1-flash-lite -y',
    expandedContent:
      "Nothing is written outside the workspace, and a hunk that no longer applies is reported as a hunk that no longer applies rather than force-applied. --review takes away every tool that can write or run, so a review cannot change anything by accident, and it reads a diff, a revision range or files outside a repository. --session-id continues a saved session, sharing the store with effgen chat and effgen sessions. On a terminal with no task it opens an interactive session; a task, -p, piped stdin or --json runs once and exits.",
  },
  {
    group: "Working with it",
    icon: FiMonitor,
    title: "Dashboards, reports and history",
    description:
      "A live dashboard and a playground served from the package itself, HTML reports you can send to someone, and a durable record of every run and session.",
    features: [
      "Per-model and per-provider cost, latency percentiles, error breakdown, run waterfall",
      "Reports for compare, eval, cost and loadtest; run cards for a single run",
      "effgen top for the terminal, effgen battle to race models side by side",
    ],
    accent: "#ff9500",
    expandedContent:
      "The dashboard, the playground, the model browser, the topology graph and the command palette are all served from the installed package — no CDN, no external font, nothing fetched when the page loads, and a test in the framework enforces it. The playground copies a request out as curl, as a CLI invocation or as Python, so what you tried in the browser is what you run. Reports are single self-contained HTML files, and effgen report renders one again from a saved result document.",
  },
];

/* ── Feature Card ── */

/* ── One capability, on the card face ──
 *
 * The face carries what a reader needs to decide whether to open it: the icon,
 * the title and one sentence. Everything else — the capability list, the longer
 * account and the code — is in the dialog the card opens, so a section of
 * twenty-two capabilities reads as twenty-two lines rather than as a wall.
 *
 * The card is a real <button>: it is reachable by Tab, activates on Enter and
 * Space without a key handler, and announces itself as a control. The pointer
 * tilt is motion the visitor did not ask for, so it is off when they have asked
 * for less of it.
 */
function FeatureCard({ feature, index, onExpand }: { feature: FeatureItem; index: number; onExpand: () => void }) {
  const cardRef = useRef<HTMLButtonElement>(null);
  const reduced = useReducedMotion();
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    if (!cardRef.current || reduced) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = ((y - centerY) / centerY) * -6;
    const rotateY = ((x - centerX) / centerX) * 6;
    setTilt({ x: rotateX, y: rotateY });
    setMousePos({ x: (x / rect.width) * 100, y: (y / rect.height) * 100 });
  }, [reduced]);

  const handleMouseLeave = useCallback(() => {
    setTilt({ x: 0, y: 0 });
    setIsHovered(false);
  }, []);

  return (
    <div className="tilt-parent h-full">
      <motion.button
        ref={cardRef}
        type="button"
        initial={{ y: 40, opacity: 0, scale: 0.95 }}
        animate={{ y: 0, opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" as const, delay: Math.min(index, 8) * 0.05 }}
        className="group relative w-full p-6 rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden cursor-pointer shadow-sm dark:shadow-none tilt-card h-full flex flex-col text-center"
        style={{
          transform: `perspective(1000px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)`,
          transformStyle: "preserve-3d",
          transition: isHovered ? "none" : "transform 0.4s ease-out",
        }}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={handleMouseLeave}
        onClick={onExpand}
      >
        {/* The card's own words name this button. An `aria-label` would replace
            them rather than contain them, which is what WCAG 2.5.3 asks for
            when a control carries visible text; this adds only what pressing
            it does. */}
        <span className="sr-only"> — open details</span>
        {/* Mouse-tracking radial glow */}
        <div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl pointer-events-none"
          style={{
            background: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, ${feature.accent}15 0%, transparent 60%)`,
          }}
        />

        {/* Border glow on hover */}
        <motion.div
          className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
          style={{ boxShadow: `inset 0 0 0 1px ${feature.accent}30` }}
        />

        {/* Icon */}
        <motion.div
          className="relative w-12 h-12 rounded-xl flex items-center justify-center mb-4 mx-auto overflow-hidden flex-shrink-0"
          style={{
            background: `${feature.accent}15`,
            border: `1px solid ${feature.accent}30`,
          }}
          whileHover={{ rotate: 360, scale: 1.1 }}
          transition={{ duration: 0.5 }}
        >
          <feature.icon style={accentTextStyle(feature.accent)} size={22} />
          <motion.div
            className="absolute inset-[-2px] rounded-xl pointer-events-none"
            style={{
              background: `conic-gradient(from 0deg, transparent 60%, ${feature.accent}40 80%, transparent 100%)`,
            }}
            animate={{ rotate: 360 }}
            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
          />
        </motion.div>

        <h3 className="text-base font-bold mb-2 text-gray-900 dark:text-white group-hover:text-green-800 dark:group-hover:text-green-50 transition-colors">
          {feature.title}
        </h3>

        <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed group-hover:text-gray-700 dark:group-hover:text-gray-400 transition-colors">
          {feature.description}
        </p>

        {/* The capability list, the longer account and the code are in the dialog. */}
        <div className="mt-auto pt-4 text-[10px] text-gray-400 dark:text-gray-600 group-hover:text-gray-600 dark:group-hover:text-gray-400 transition-colors font-medium tracking-wide uppercase">
          Click for details
        </div>

        {/* Bottom accent line */}
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
          style={{ background: `linear-gradient(90deg, transparent, ${feature.accent}, transparent)` }}
        />
      </motion.button>
    </div>
  );
}


/* ── Expanded Detail Modal ── */

function ExpandedDetail({ feature, onClose }: { feature: FeatureItem; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const titleId = `feature-detail-title`;

  // Escape closes it, Tab stays inside it, and the element that was focused
  // when it opened is focused again when it closes. Without these three a
  // keyboard reader can tab out of an open dialog into a page they cannot see.
  useEffect(() => {
    const returnTo = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;

      const focusable = panelRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      returnTo?.focus?.();
    };
  }, [onClose]);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      />
      <motion.div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative w-full max-w-xl rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl overflow-hidden max-h-[85vh] overflow-y-auto"
        style={{ boxShadow: `0 0 60px ${feature.accent}15` }}
        initial={{ scale: 0.9, opacity: 0, y: 30 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.9, opacity: 0, y: 30 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        {/* Header bar */}
        <motion.div
          className="h-1"
          style={{ background: `linear-gradient(90deg, transparent, ${feature.accent}, transparent)` }}
        />

        <div className="p-6">
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute top-4 right-4 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
          >
            <FiX size={16} />
          </button>

          <div className="flex items-center gap-3 mb-4">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: `${feature.accent}15`, border: `1px solid ${feature.accent}30` }}
            >
              <feature.icon style={accentTextStyle(feature.accent)} size={20} />
            </div>
            <h3 id={titleId} className="text-xl font-black text-gray-900 dark:text-white">{feature.title}</h3>
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
            {feature.expandedContent}
          </p>

          {/* Interactive visual (only shown in expanded view) */}
          {feature.visual === "decomp" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-2">Live Visualization</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <TaskDecompAnimation />
              </div>
            </div>
          )}
          {feature.visual === "orchestration" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-2">Live Visualization</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <OrchestrationAnimation />
              </div>
            </div>
          )}
          {feature.visual === "presets" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-2">Available Presets</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <PresetStripVisual />
              </div>
            </div>
          )}
          {feature.visual === "memory" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-2">Memory Flow</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <MemoryFlowVisual />
              </div>
            </div>
          )}

          {/* Features list */}
          <div className="mb-4">
            <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-2">Key Capabilities</h4>
            <div className="space-y-1.5">
              {feature.features.map((f, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: feature.accent }} />
                  {f}
                </div>
              ))}
            </div>
          </div>

          {/* Code example if exists */}
          {feature.code && (
            <div className="px-4 py-3 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 font-mono text-xs overflow-x-auto">
              <pre className="leading-relaxed text-gray-700 dark:text-gray-300">
                <code
                  className="syntax-code"
                  dangerouslySetInnerHTML={{
                    __html: highlightCode(
                      feature.code,
                      /^\s*(pip|effgen|export|cd|sudo|brew|apt|docker|curl|sh|bash|#!)/m.test(feature.code) ? "bash" : "python"
                    ),
                  }}
                />
              </pre>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── Main Features Section ── */

export default function Features() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [expandedFeature, setExpandedFeature] = useState<FeatureItem | null>(null);

  return (
    <>
      <section id="features" className="py-24 bg-gray-50 dark:bg-[#020c08] relative overflow-hidden noise-overlay" ref={ref}>
        {/* Background elements */}
        <div className="absolute inset-0 grid-pattern" />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/30 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full opacity-[0.03] pointer-events-none"
          style={{ background: "radial-gradient(circle, #00ff88 0%, transparent 70%)" }}
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 8, repeat: Infinity }}
        />

        <Container className="relative z-10">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <motion.span
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6"
              whileHover={{ borderColor: "rgba(0,255,136,0.6)" }}
            >
              <FiZap size={14} />
              Features
            </motion.span>
            <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              Everything an agent needs,
              <br />
              <span className="gradient-text">and everything around it</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              {toolCount} tools, {presetCount} presets, {providerCount} provider adapters and
              a full production surface — grouped by what you reach for them for. Open a
              card for the detail.
            </p>
          </motion.div>

          {/* Grouped grid */}
          <div className="space-y-14">
            {groups.map((group, groupIndex) => {
              const inGroup = features.filter((feature) => feature.group === group);
              return (
                <div key={group}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={inView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.5, delay: 0.1 + groupIndex * 0.05 }}
                    className="flex items-center gap-4 mb-6"
                  >
                    <span className="h-px flex-1 bg-gradient-to-r from-transparent to-green-500/25" />
                    <h3 className="text-[11px] font-mono uppercase tracking-[0.2em] text-gray-600 dark:text-gray-400 whitespace-nowrap text-center">
                      {group}
                      <span className="ml-3 text-gray-400 dark:text-gray-600">
                        {String(inGroup.length).padStart(2, "0")}
                      </span>
                    </h3>
                    <span className="h-px flex-1 bg-gradient-to-r from-green-500/25 to-transparent" />
                  </motion.div>

                  {/* Flex rather than grid, because four of the six groups hold
                      three cards and a four-column grid would hang them to the
                      left with an empty column beside them. Wrapping and
                      centring keeps every card the same width and every row
                      centred under its heading. */}
                  <div className="flex flex-wrap justify-center gap-5">
                    {inGroup.map((feature, index) => (
                      <div
                        key={feature.title}
                        className="w-full sm:w-[calc(50%-0.625rem)] lg:w-[calc(25%-0.9375rem)]"
                      >
                        <FeatureCard
                          feature={feature}
                          index={index}
                          onExpand={() => setExpandedFeature(feature)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="flex flex-col items-center gap-4 mt-14"
          >
            <motion.a
              href="#quickstart"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
              style={{
                background: "linear-gradient(135deg, #00ff88, #00c96e)",
                boxShadow: "0 0 30px rgba(0,255,136,0.3)",
              }}
              whileHover={{ scale: 1.05, y: -2, boxShadow: "0 0 50px rgba(0,255,136,0.5)" }}
              whileTap={{ scale: 0.95 }}
            >
              Start here
              <FiZap />
            </motion.a>
          </motion.div>
        </Container>
      </section>

      {/* Expanded Detail Modal */}
      <AnimatePresence>
        {expandedFeature && (
          <ExpandedDetail
            feature={expandedFeature}
            onClose={() => setExpandedFeature(null)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
