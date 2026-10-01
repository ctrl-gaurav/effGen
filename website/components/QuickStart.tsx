"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiCode, FiServer, FiTerminal } from "react-icons/fi";
import type { IconType } from "react-icons";
import { useState } from "react";
import Container from "./Container";
import CodeSample from "./ui/CodeSample";
import { withBasePath } from "./basePath";
import { accentTextStyle } from "./accentText";

/* ── The three ways in ──
 *
 * Every command and every sample below was run before it was written down, and
 * the `output` under each one is what that run printed. Where a command prints
 * more than is reproduced here, the step says so rather than trimming quietly.
 */

interface Step {
  number: string;
  title: string;
  description: string;
  code: string;
  language: "bash" | "python";
  accent: string;
  output?: string;
  outputLabel?: string;
}

interface Tab {
  id: string;
  label: string;
  icon: IconType;
  accent: string;
  lede: string;
  steps: Step[];
}

const tabs: Tab[] = [
  {
    id: "cli",
    label: "CLI",
    icon: FiTerminal,
    accent: "#00ff88",
    lede: "Nothing to write. Install it, check what your keys reach, and run an agent from the shell.",
    steps: [
      {
        number: "01",
        title: "Install",
        description:
          "One package. Extras only when you want a particular local engine or provider client.",
        code: 'pip install -U effgen\neffgen --version',
        language: "bash",
        accent: "#00ff88",
        output: "effGen 1.3.0",
      },
      {
        number: "02",
        title: "See what your keys reach",
        description:
          "doctor reads the keys in your shell and your project .env and reports what each provider can serve. It goes on to print a system section and a check of what the coding agent needs; the provider table is the part reproduced here.",
        code: "effgen doctor",
        language: "bash",
        accent: "#00e5ff",
        output: `               effgen doctor — Provider Status
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Provider          ┃ Key     ┃ Env Var             ┃ Models ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ anthropic         │ missing │ —                   │     17 │
│ cerebras          │ present │ CEREBRAS_API_KEY    │      2 │
│ fireworks         │ present │ FIREWORKS_API_KEY   │     16 │
│ gemini            │ present │ GOOGLE_API_KEY      │      8 │
│ groq              │ present │ GROQ_API_KEY        │     14 │
│ hf                │ present │ HF_TOKEN            │    124 │
│ openai            │ present │ OPENAI_API_KEY      │     30 │
│ openai_compatible │ missing │ —                   │      0 │
│ replicate         │ present │ REPLICATE_API_TOKEN │     37 │
│ together          │ present │ TOGETHER_API_KEY    │    168 │
└───────────────────┴─────────┴─────────────────────┴────────┘`,
      },
      {
        number: "03",
        title: "Run an agent",
        description:
          "A task and a model. The result line says how long it took, how many tokens it used and what it cost.",
        code: 'effgen run "What is the capital of France? Answer in one word." \\\n  -m openai:gpt-5-nano',
        language: "bash",
        accent: "#a78bfa",
        output: `effGen v1.0.1 - Running Task

Initializing agent: cli-agent
Model: openai:gpt-5-nano
Tools: 1 available
Sub-agents: enabled

Task: What is the capital of France? Answer in one word.

Thinking...

Response
╭─────────────────────────────── Agent Response ───────────────────────────────╮
│ Paris                                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
✓ Done in 3.1s · 294 tokens · $0.000041`,
      },
      {
        number: "04",
        title: "Give it a tool",
        description:
          "-t names the tools the run may use. The result line counts the tool steps, and --trace prints their timeline.",
        code: 'effgen run "Use the calculator tool to work out 24344 * 334." \\\n  -m gemini:gemini-3.1-flash-lite -t calculator',
        language: "bash",
        accent: "#ffd700",
        output: `effGen v1.0.1 - Running Task
Loading tools: calculator
✓ Loaded tool: calculator

Initializing agent: cli-agent
Model: gemini:gemini-3.1-flash-lite
Tools: 1 available
Sub-agents: enabled

Task: Use the calculator tool to work out 24344 * 334.

Thinking...

Response
╭─────────────────────────────── Agent Response ───────────────────────────────╮
│ 8130896                                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
✓ Done in 3.5s · 2 tools · 508 tokens · $0.0002
2 tool steps — run with --trace to see the timeline`,
      },
    ],
  },
  {
    id: "python",
    label: "Python",
    icon: FiCode,
    accent: "#00e5ff",
    lede: "Four lines to an agent, six to an agent with a tool. The same object comes back either way.",
    steps: [
      {
        number: "01",
        title: "Your first agent",
        description:
          "A model id and nothing else. .text is the answer; str(response) is the same string.",
        code: 'from effgen import Agent, AgentConfig\n\nagent = Agent(AgentConfig(model="openai:gpt-5-nano"))\nprint(agent.run("Name the three primary colours of light, comma separated.").text)',
        language: "python",
        accent: "#00ff88",
        output: "red, green, blue",
      },
      {
        number: "02",
        title: "Your first tool call",
        description:
          "Hand the config a tool and the run may call it. What it called comes back on the response.",
        code: 'from effgen import Agent, AgentConfig\nfrom effgen.tools.builtin import Calculator\n\nagent = Agent(AgentConfig(\n    model="gemini:gemini-3.1-flash-lite",\n    tools=[Calculator()],\n))\nr = agent.run("Use the calculator tool to work out 24344 * 334.")\n\nprint(r.text)\nprint(r.tool_calls.total, "tool call")\nfor call in r.tool_calls:\n    print(call.name, call.arguments, "->", call.result)',
        language: "python",
        accent: "#00e5ff",
        output: '8130896\n1 tool call\ncalculator {"expression": "24344 * 334"} -> 8130896',
      },
      {
        number: "03",
        title: "Or start from a preset",
        description:
          "A preset carries the tools, the temperature, the iteration cap and the prompt for one kind of work.",
        code: 'from effgen import create_agent\n\nagent = create_agent("math", "gemini:gemini-3.1-flash-lite")\nprint(agent.run("What is 17 * 23 + 144 ** 0.5?").text)',
        language: "python",
        accent: "#a78bfa",
        output: "403.0",
      },
    ],
  },
  {
    id: "server",
    label: "Server",
    icon: FiServer,
    accent: "#a78bfa",
    lede: "One command puts the agents behind an OpenAI-compatible API — for your own clients, and for effGen itself.",
    steps: [
      {
        number: "01",
        title: "Start the server",
        description:
          "Set a key first, or it mints an ephemeral one and prints it once. The dashboard and the playground come up on the same port.",
        code: "export EFFGEN_API_KEY=\"$(python -c 'import secrets; print(secrets.token_urlsafe(24))')\"\neffgen serve --port 8000",
        language: "bash",
        accent: "#00ff88",
        output: `effGen v1.0.1 - API Server
✓ Auth: static API key (EFFGEN_API_KEY)
Starting server on 127.0.0.1:8000
  OpenAI-compatible API : http://127.0.0.1:8000/v1
  Interactive docs      : http://127.0.0.1:8000/docs
  Dashboard             : http://127.0.0.1:8000/dashboard  (data requires an API key; set EFFGEN_PUBLIC_DASHBOARD=1 for local viewing)
  Playground            : http://127.0.0.1:8000/playground  (paste an API key, or set EFFGEN_PUBLIC_PLAYGROUND=1 for local viewing)
  Both pages: Cmd/Ctrl-K opens the command palette, ? lists shortcuts.`,
      },
      {
        number: "02",
        title: "Call it like OpenAI",
        description:
          "Any client that already speaks the protocol works unchanged. The effgen block on the response says which model actually answered and what the call cost.",
        code: `curl -s http://127.0.0.1:8000/v1/chat/completions \\
  -H "Authorization: Bearer $EFFGEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "openai:gpt-5-nano", "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}'`,
        language: "bash",
        accent: "#00e5ff",
        output:
          '{"id":"chatcmpl-3d155ec23267489e9d2eb85f","object":"chat.completion","created":1787193322,"model":"openai:gpt-5-nano","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":27,"completion_tokens":202,"total_tokens":229},"effgen":{"requested_model":"openai:gpt-5-nano","resolved_model":"openai:gpt-5-nano","alias_applied":false,"cost_usd":0.00008215,"run_id":"99a3c0ceb5d4"}}',
      },
      {
        number: "03",
        title: "Point effGen at it",
        description:
          "The same base_url works for any OpenAI-protocol server — vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, a gateway, or this one.",
        code: 'import os\n\nfrom effgen import Agent, AgentConfig\n\nagent = Agent(AgentConfig(\n    model="openai:gpt-5-nano",\n    base_url="http://127.0.0.1:8000/v1",\n    api_key=os.environ["EFFGEN_API_KEY"],\n))\nprint(agent.run("Reply with the single word: ready").text)',
        language: "python",
        accent: "#a78bfa",
        output: "ready",
      },
    ],
  },
];

export default function QuickStart() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [active, setActive] = useState(tabs[0].id);

  const tab = tabs.find((t) => t.id === active) ?? tabs[0];

  return (
    <section id="quickstart" className="py-24 bg-gray-50 dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <motion.div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-[0.04] pointer-events-none"
        style={{ background: "radial-gradient(circle, #00ff88 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 8, repeat: Infinity }}
      />

      <Container className="relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
            <FiTerminal size={14} />
            Quick start
          </span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            Your first three <span className="gradient-text neon-text">commands</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            From the shell, from Python, or behind an API. Everything below was run
            before it was written down, and the output under each step is what that run
            printed.
          </p>
        </motion.div>

        {/* Tabs */}
        <div
          role="tablist"
          aria-label="Quick start"
          className="flex flex-wrap justify-center gap-2 mb-10"
        >
          {tabs.map((item) => {
            const selected = item.id === tab.id;
            return (
              <button
                key={item.id}
                type="button"
                role="tab"
                id={`quickstart-tab-${item.id}`}
                aria-selected={selected}
                aria-controls={`quickstart-panel-${item.id}`}
                onClick={() => setActive(item.id)}
                className={`inline-flex items-center gap-2 px-6 py-3 rounded-full text-sm font-semibold border transition-all ${
                  selected
                    ? "text-black"
                    : "text-gray-600 dark:text-gray-400 border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900/60 hover:border-green-400/60"
                }`}
                style={
                  selected
                    ? {
                        background: `linear-gradient(135deg, ${item.accent}, ${item.accent}cc)`,
                        borderColor: item.accent,
                        boxShadow: `0 0 24px ${item.accent}40`,
                      }
                    : undefined
                }
              >
                <item.icon size={14} />
                {item.label}
              </button>
            );
          })}
        </div>

        <div
          role="tabpanel"
          id={`quickstart-panel-${tab.id}`}
          aria-labelledby={`quickstart-tab-${tab.id}`}
          className="max-w-4xl mx-auto"
        >
          <p className="text-center text-sm text-gray-600 dark:text-gray-400 mb-8">
            {tab.lede}
          </p>

          <div className="space-y-6">
            {tab.steps.map((step, index) => (
              <motion.div
                key={`${tab.id}-${step.number}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.08 }}
                className="relative"
              >
                <div className="group flex flex-col lg:flex-row items-start gap-6 p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm relative overflow-hidden shadow-sm dark:shadow-none">
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                    style={{ background: `radial-gradient(circle at 10% 50%, ${step.accent}05 0%, transparent 60%)` }}
                  />

                  <div
                    className="flex-shrink-0 relative z-10 w-16 h-16 rounded-xl flex items-center justify-center"
                    style={{ background: `${step.accent}15`, border: `1px solid ${step.accent}30` }}
                  >
                    <span className="text-2xl font-black font-mono" style={accentTextStyle(step.accent)}>
                      {step.number}
                    </span>
                  </div>

                  <div className="flex-1 min-w-0 relative z-10">
                    <h3 className="text-xl font-bold mb-1.5 text-gray-900 dark:text-white">
                      {step.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                      {step.description}
                    </p>

                    <CodeSample
                      code={step.code}
                      language={step.language}
                      accent={step.accent}
                      output={step.output}
                      outputLabel={step.outputLabel}
                    />
                  </div>
                </div>

                {index < tab.steps.length - 1 && (
                  <div className="hidden lg:flex absolute -bottom-6 left-7 items-center">
                    <svg width="2" height="24" className="overflow-visible" aria-hidden="true">
                      <line x1="1" y1="0" x2="1" y2="24" stroke={step.accent} strokeWidth="2" strokeOpacity="0.5" />
                    </svg>
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          <div className="mt-10 text-center">
            <a
              href={withBasePath("/docs/quickstart")}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-semibold text-sm text-green-700 dark:text-green-300 border border-green-500/30 bg-green-500/5 hover:border-green-400/60 hover:bg-green-500/10 transition-all"
            >
              The full quick start, in the documentation
            </a>
          </div>
        </div>
      </Container>
    </section>
  );
}
