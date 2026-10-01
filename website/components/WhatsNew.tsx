"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiArrowRight, FiCode, FiGitBranch, FiLayers, FiMonitor, FiServer } from "react-icons/fi";
import type { IconType } from "react-icons";
import { ReactNode } from "react";
import Container from "./Container";
import CodeSample from "./ui/CodeSample";
import Terminal from "./ui/Terminal";
import RouteLink from "./ui/RouteLink";
import { version } from "./siteData";
import { accentTextStyle } from "./accentText";

interface Change {
  icon: IconType;
  accent: string;
  title: string;
  summary: string;
  detail: ReactNode;
  href: string;
  hrefLabel: string;
  /** The sample, run before it was written down, and what it printed. */
  code?: { source: string; language?: "python" | "bash"; output: string };
  /** Captured output from a real session, shown as text rather than a picture. */
  capture?: { command: string; output: string; maxLines?: number };
}

const changes: Change[] = [
  {
    icon: FiServer,
    accent: "#00e5ff",
    title: "Point it at any OpenAI-compatible server",
    summary:
      "A base_url is the whole instruction. vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, LiteLLM or a gateway — effGen drives the model you are already serving instead of loading a second copy of the weights.",
    detail: (
      <>
        The ids come from the server, so no catalog is consulted and no price is
        invented: a call through your own endpoint reports no cost rather than{" "}
        <code className="font-mono text-[13px]">$0</code>. Ask it what it serves with{" "}
        <code className="font-mono text-[13px]">list_served_models()</code>, and when
        nothing is listening you get{" "}
        <code className="font-mono text-[13px]">BackendUnreachableError</code> naming
        the endpoint it tried.
      </>
    ),
    href: "/models",
    hrefLabel: "Models and providers",
    code: {
      source: `import os

from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["EFFGEN_API_KEY"],
))
print(agent.run("Reply with the single word: ready").text)`,
      output: "ready",
    },
  },
  {
    icon: FiLayers,
    accent: "#00ff88",
    title: "A run tells you which calls it made",
    summary:
      "AgentResponse.tool_calls is the list of calls the run actually made, not a count. Each carries its name and iteration, plus the arguments, result, duration and error the provider reported.",
    detail: (
      <>
        <code className="font-mono text-[13px]">.failed</code> narrows it to the calls
        that went wrong, <code className="font-mono text-[13px]">.by_name()</code> to
        one tool, and <code className="font-mono text-[13px]">.total</code> to the
        count — so code written against the old integer still reads. How much each
        call carries depends on the provider: some return the arguments and the
        result, others report only that the call happened.
      </>
    ),
    href: "/agents",
    hrefLabel: "The agent surface",
    code: {
      source: `from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    tools=[Calculator()],
))
r = agent.run("Use the calculator tool to work out 24344 * 334.")

print(r.text)
print(r.tool_calls.total, "tool calls")
for call in r.tool_calls:
    print(call.name, call.arguments, "->", call.result)`,
      output: `8130896
2 tool calls
calculator {"expression": "24344 * 4"} -> 97376
calculator {"expression": "24344 * 334"} -> 8130896`,
    },
  },
  {
    icon: FiCode,
    accent: "#a78bfa",
    title: "A coding agent in the terminal",
    summary:
      "effgen code reads the repository, proposes a change, shows it as a unified diff, applies it, runs the result and fixes what fails. Four permission modes decide how much of that happens without you.",
    detail: (
      <>
        <code className="font-mono text-[13px]">--plan</code> proposes and writes
        nothing, the default asks before each change,{" "}
        <code className="font-mono text-[13px]">--auto-edit</code> applies writes but
        still asks before a shell command, and{" "}
        <code className="font-mono text-[13px]">-y</code> applies everything inside
        the workspace. <code className="font-mono text-[13px]">--undo</code> reverses
        the last edits, <code className="font-mono text-[13px]">--review</code> runs
        with no tool that can write, and{" "}
        <code className="font-mono text-[13px]">--session-id</code> continues where
        you left off.
      </>
    ),
    href: "/code",
    hrefLabel: "The coding agent",
    capture: {
      command:
        'effgen code -p "Create slugify.py …" -m gemini:gemini-3.1-flash-lite -y',
      output: `new file slugify.py (+18/-0)
--- a/slugify.py
+++ b/slugify.py
@@ -0,0 +1,18 @@
+import re
+
+def slugify(text):
+    # Lowercase the text
+    text = text.lower()
+    # Replace runs of non-alphanumeric characters with a single hyphen
+    text = re.sub(r'[^a-z0-9]+', '-', text)
+    # Strip leading and trailing hyphens
+    text = text.strip('-')
+    return text
+
+if __name__ == "__main__":
+    assert slugify("Hello World") == "hello-world"
+    assert slugify("  Hello   World  ") == "hello-world"
+    assert slugify("Hello-World!") == "hello-world"
+    assert slugify("---Hello---World---") == "hello-world"
+    assert slugify("123 456") == "123-456"
+    print("All tests passed!")
✓ Done in 21.5s · 2 tools · 5,390 tokens · $0.0017
Tool calling: hybrid — provider tool API first, falling back to the text
Files written in the workspace: slugify.py
slugify.py | All tests passed!`,
    },
  },
  {
    icon: FiGitBranch,
    accent: "#ffd700",
    title: "Workflows resume where they stopped",
    summary:
      "Hand a WorkflowDAG a checkpoint store and a run id. A pipeline that died half way through does not start again from the top — running the same line picks up at the node that had not finished.",
    detail: (
      <>
        A completed node is restored and its output flows downstream without a model
        call; a failed node is retried, which is usually why you are resuming; a node
        that never started runs normally. There is no separate resume call — a run id
        the store has not seen starts from the beginning, one it knows continues.
        Progress is written after each level of the graph, atomically.
      </>
    ),
    href: "/agents",
    hrefLabel: "Workflows and checkpoints",
    code: {
      source: `from effgen import Agent, AgentConfig, FileCheckpointStore, WorkflowDAG, WorkflowNode

store = FileCheckpointStore("./checkpoints")

dag = WorkflowDAG("briefing")
dag.add_node(WorkflowNode(id="research", agent=Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    system_prompt="Answer in three short bullet points. Nothing else.",
))))
dag.add_node(WorkflowNode(id="draft", agent=Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:9/v1",   # a server that is not up
    require_model=False,
))))
dag.connect("research", "draft")

result = dag.run("Why run an agent on a small model?",
                 checkpoint=store, run_id="briefing-1")

for node in result.node_results:
    print(f"{node['id']:9} {node['status']:9} {node['execution_time']:6.2f}s")`,
      output: `# first run
research  completed   3.08s
draft     failed     13.68s

# the same line again, in a new process
research  completed   0.00s
draft     failed     13.46s`,
    },
  },
  {
    icon: FiMonitor,
    accent: "#ff9500",
    title: "Surfaces to watch it through",
    summary:
      "effgen serve brings up the OpenAI-compatible API and, on the same port, a dashboard, a playground, a model browser and a topology graph. All of it is served from the package — no CDN, nothing fetched at view time.",
    detail: (
      <>
        In the terminal, <code className="font-mono text-[13px]">effgen top</code>{" "}
        reads that server and shows runs, traffic, spend and GPUs live;{" "}
        <code className="font-mono text-[13px]">effgen battle</code> races several
        models on one prompt; compare, eval, cost and loadtest each write a shareable
        HTML report, and any saved result renders again with{" "}
        <code className="font-mono text-[13px]">effgen report</code>.
      </>
    ),
    href: "/dashboard",
    hrefLabel: "The web surfaces",
    capture: {
      command: "effgen serve --port 8000",
      output: `effGen v1.0.1 - API Server
✓ Auth: static API key (EFFGEN_API_KEY)
Starting server on 127.0.0.1:8000
  OpenAI-compatible API : http://127.0.0.1:8000/v1
  Interactive docs      : http://127.0.0.1:8000/docs
  Dashboard             : http://127.0.0.1:8000/dashboard  (data requires an API key; set EFFGEN_PUBLIC_DASHBOARD=1 for local viewing)
  Playground            : http://127.0.0.1:8000/playground  (paste an API key, or set EFFGEN_PUBLIC_PLAYGROUND=1 for local viewing)
  Both pages: Cmd/Ctrl-K opens the command palette, ? lists shortcuts.`,
    },
  },
];

// Four of the ten 1.3.0 changes existing code can see — the ones most code
// meets first. The changelog lists all ten.
const breaking = [
  {
    title: "A run with no progress is asked for its answer",
    detail:
      "max_turns_without_progress defaults to 2, and a stuck run gets one closing request, so a run that raised RunStoppedError in 1.2.0 can now return an answer. None restores the earlier loop.",
  },
  {
    title: "A failing tool ends the run tool_failed",
    detail:
      "A tool that fails on its own side three times in a row is unavailable for the run; a run left with no usable tool stops with tool_failed and raises RunStoppedError by default.",
  },
  {
    title: "A broken tool call is read",
    detail:
      "recover_lost_tool_calls defaults to True, so some runs make more tool calls. A call missing a required argument is asked for again, never dispatched.",
  },
  {
    title: "Served and local models are probed",
    detail:
      "At tool_calling_mode=\"auto\" the first agent measures what the model does with a tool, once. capability_probe=False or EFFGEN_CAPABILITY_PROBE=0 turns it off.",
  },
];

export default function WhatsNew() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.02 });

  return (
    <section
      id="whats-new"
      className="py-24 bg-gray-50 dark:bg-[#020c08] relative overflow-hidden noise-overlay"
      ref={ref}
    >
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/30 to-transparent" />

      <Container className="relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <motion.span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
            <span className="font-mono">v{version}</span>
            What&rsquo;s new
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
            The stable line,
            <br />
            <span className="gradient-text">shown rather than listed</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            {version} makes every run say how it ended: a run that stops making progress is asked
            for its answer, a tool that keeps failing ends the run as tool_failed, and a tool call
            written in a broken shape is read rather than reported. Below, five of the changes 1.0.0 brought, each with the output it produces
            beside it. Every sample here was run before it was written down, on the release it
            was recorded on.
          </p>
        </motion.div>

        <div className="space-y-6">
          {changes.map((change, i) => (
            <motion.article
              key={change.title}
              initial={{ opacity: 0, y: 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: Math.min(i, 4) * 0.08 }}
              className="group relative rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden shadow-sm dark:shadow-none"
            >
              <div
                className="absolute top-0 left-0 right-0 h-px"
                style={{
                  background: `linear-gradient(90deg, transparent, ${change.accent}60, transparent)`,
                }}
              />

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 p-6 lg:p-8 items-start">
                <div>
                  <div className="flex items-center gap-3 mb-4">
                    <div
                      className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0"
                      style={{
                        background: `${change.accent}15`,
                        border: `1px solid ${change.accent}30`,
                      }}
                    >
                      <change.icon style={accentTextStyle(change.accent)} size={20} />
                    </div>
                    <span
                      className="text-[10px] font-mono uppercase tracking-widest"
                      style={accentTextStyle(change.accent)}
                    >
                      {String(i + 1).padStart(2, "0")}
                    </span>
                  </div>

                  <h3 className="text-xl md:text-2xl font-black text-gray-900 dark:text-white mb-3">
                    {change.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-3">
                    {change.summary}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
                    {change.detail}
                  </p>

                  <RouteLink
                    to={change.href}
                    className="inline-flex items-center gap-1.5 text-sm font-semibold transition-colors"
                    style={accentTextStyle(change.accent)}
                  >
                    {change.hrefLabel}
                    <FiArrowRight size={14} />
                  </RouteLink>
                </div>

                <div className="min-w-0">
                  {change.code && (
                    <CodeSample
                      code={change.code.source}
                      language={change.code.language ?? "python"}
                      accent={change.accent}
                      output={change.code.output}
                    />
                  )}
                  {change.capture && (
                    <Terminal
                      command={change.capture.command}
                      output={change.capture.output}
                      maxLines={change.capture.maxLines ?? 26}
                    />
                  )}
                </div>
              </div>
            </motion.article>
          ))}
        </div>

        {/* Changes existing code sees */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-10 rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-6 lg:p-8"
        >
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1">
            Four of the ten things existing code sees in {version}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
            Nothing was removed or renamed. Coming from 1.1.x, 1.2.0 also made a spent cap raise
            BudgetExceededError and an unpriced model report its cost as None; from 1.0.x, 1.1.0 also put the run&rsquo;s
            conversation in response.metadata as a live thread; from 1.0.0, 1.0.1 made a stopped run
            report failure and made citation markers opt-in; coming from 0.3.x, 1.0.0 raised
            the Python floor to 3.11 and made raise_on_error default to True.
          </p>

          <ol className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
            {breaking.map((item, i) => (
              <li
                key={item.title}
                className="rounded-xl bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-4"
              >
                <span className="text-[10px] font-mono uppercase tracking-widest text-orange-700 dark:text-orange-400">
                  Change {String(i + 1).padStart(2, "0")}
                </span>
                <h4 className="mt-2 text-sm font-bold text-gray-900 dark:text-white">
                  {item.title}
                </h4>
                <p className="mt-1.5 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                  {item.detail}
                </p>
              </li>
            ))}
          </ol>

          <div className="mt-6 flex flex-wrap gap-4">
            <RouteLink
              to="/changelog"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
            >
              The full {version} changelog
              <FiArrowRight size={14} />
            </RouteLink>
            <RouteLink
              to="/docs/migration"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
            >
              The migration guide
              <FiArrowRight size={14} />
            </RouteLink>
          </div>
        </motion.div>
      </Container>
    </section>
  );
}
