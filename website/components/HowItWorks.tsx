"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiAlertTriangle, FiCheckCircle, FiCpu, FiEye, FiMessageSquare, FiPlay,
  FiTool, FiX,
} from "react-icons/fi";
import { useEffect, useRef, useState } from "react";
import Container from "./Container";
import CodeSample from "./ui/CodeSample";
import { siteData, toolCount } from "./siteData";
import { accentTextStyle } from "./accentText";

/* ── The loop, one step at a time ── */

const steps = [
  {
    icon: FiMessageSquare,
    title: "The task",
    description: "A string goes in. The agent formats it with the schemas of the tools it has.",
    accent: "#00ff88",
    detail: 'agent.run("What is 24344 * 334?")',
    explanation:
      "run() takes plain text, a Message, or a list of content parts when the task carries an image, an audio file or a video. The tools on the config are turned into schemas the model can read, and the system prompt — the preset's, or yours — goes in front of it.",
    code: 'from effgen import Agent, AgentConfig\nfrom effgen.tools.builtin import Calculator\n\nagent = Agent(AgentConfig(\n    model="gemini:gemini-3.1-flash-lite",\n    tools=[Calculator()],\n))\nr = agent.run("Use the calculator tool to work out 24344 * 334.")',
    output: undefined as string | undefined,
  },
  {
    icon: FiCpu,
    title: "The model",
    description: "The model reads the task and either answers it or asks for a tool.",
    accent: "#00e5ff",
    detail: "reason → answer, or reason → call",
    explanation:
      "Nothing forces a tool call. If the model can answer from what it knows, it answers and the loop ends after one turn. A reasoning model that spends its first turn thinking without producing visible text is recognised as having done so rather than recorded as an empty answer.",
    code: undefined as string | undefined,
    output: undefined as string | undefined,
  },
  {
    icon: FiTool,
    title: "The call",
    description:
      "One of three strategies turns the model's output into a call against a tool it was given.",
    accent: "#a78bfa",
    detail: 'tool_calling_mode="auto"',
    explanation:
      "How a call is recognised depends on the model. A model with a function-calling API returns a structured call; one without it writes the call as text in a ReAct block that has to be parsed. auto picks per model, and hybrid tries the structured path and falls back to parsing when it fails. The tool the call names has to be one the agent was given — a name it was not given is reported, not invented.",
    code: undefined as string | undefined,
    output: undefined as string | undefined,
  },
  {
    icon: FiPlay,
    title: "Execution",
    description:
      "The tool is awaited with keyword arguments and returns a typed result. It does not raise into the loop.",
    accent: "#ffd700",
    detail: "await tool.execute(**kwargs)",
    explanation:
      "Every tool has the same signature and returns a ToolResult carrying success, output, error, execution_time, metadata and timestamp — no data field, and not a dictionary you index into. Code execution runs in the sandbox, and each result reports which confinement was actually enforced rather than which was asked for.",
    code: 'import asyncio\n\nfrom effgen.tools.builtin import Calculator\n\ncalc = Calculator()\n\nok = asyncio.run(calc.execute(expression="24344 * 334"))\nprint(ok.success, repr(ok.output), ok.error)\n\nbad = asyncio.run(calc.execute(expression="1 / 0"))\nprint(bad.success, repr(bad.output), bad.error)',
    output:
      "True {'result': 8130896, 'formatted': '8130896', 'expression': '24344 * 334'} None\nFalse None Tool execution failed: Calculation failed: division by zero. Check the expression for balanced brackets, a supported function and no stray characters.",
  },
  {
    icon: FiEye,
    title: "The observation",
    description:
      "The result goes back to the model as the next turn — whether it succeeded or not.",
    accent: "#ff9500",
    detail: "Observation: 8130896",
    explanation:
      "A failure is an observation like any other: the error text is what the model sees next, so it can correct the call rather than stall. Every call, successful or not, is recorded on the response, so what the run did is visible afterwards even when the answer looks fine.",
    code: undefined as string | undefined,
    output: undefined as string | undefined,
  },
  {
    icon: FiCheckCircle,
    title: "The answer",
    description:
      "One object comes back: the text, whether it succeeded, the calls it made and what it cost.",
    accent: "#00ff88",
    detail: "AgentResponse",
    explanation:
      "str(response) is the answer and .text is the same string. .success says whether the run answered, .outcome and .termination name how it ended, .thread is the run itself as typed steps, .ledger is what it spent and where its time went, .tool_calls lists what it called, .sources and .citations carry the URLs a grounded run retrieved, and .metadata carries the cost, the token counts, the latency, the stage the run ended at, and any partial output. AgentResponse is imported from effgen.core.agent, not from the top-level package.",
    // The run that binds `r` is repeated here so the block stands on its own:
    // it is five steps back on the page, with an unrelated tool demonstration
    // between, and a reader who copies this one should get the output below.
    code: 'from effgen import Agent, AgentConfig\nfrom effgen.tools.builtin import Calculator\n\nagent = Agent(AgentConfig(\n    model="gemini:gemini-3.1-flash-lite",\n    tools=[Calculator()],\n))\nr = agent.run("Use the calculator tool to work out 24344 * 334.")\n\nprint(r.text)\nprint(r.tool_calls.total, "tool calls")\nfor call in r.tool_calls:\n    print(call.name, call.arguments, "->", call.result)',
    output: '8130896\n2 tool calls\ncalculator {"expression": "24344 * 4"} -> 97376\ncalculator {"expression": "24344 * 334"} -> 8130896',
  },
];

/* ── How a call is recognised ── */

const strategies = [
  {
    name: "native",
    accent: "#00e5ff",
    headline: "The model's own function-calling API",
    body: "Tools become JSON-schema function definitions and the model returns a structured call. Nothing is parsed out of prose, so nothing can be mis-parsed.",
  },
  {
    name: "react",
    accent: "#a78bfa",
    headline: "Thought, Action, Action Input, in text",
    body: "The call is written in the model's output and read back out with a parser. It works on any model, including a small local one with no tool API at all.",
  },
  {
    name: "hybrid",
    accent: "#ffd700",
    headline: "Structured first, text as the fallback",
    body: "Try the provider's tool API; when the call does not come back parseable, fall back to reading it out of the text. This is what effgen code runs on.",
  },
];

/* ── What a failure looks like ── */

const failures = [
  {
    title: "The tool fails",
    body: "The call comes back with success=False and a message. The run does not stop: the error becomes the next observation, and the call is kept on the response so you can see it afterwards.",
    code: 'r = agent.run("Work out 1/0 with the calculator, then tell me what happened.")\n\nprint("success:", r.success)\nfor call in r.tool_calls:\n    print(call.name, "error:", call.error)\nprint("failed calls:", len(r.tool_calls.failed))',
    output:
      "success: True\ncalculator error: Error executing tool 'calculator': Tool execution failed: Calculation failed: division by zero. Check the expression for balanced brackets, a supported function and no stray characters.\nfailed calls: 1",
    accent: "#ff9500",
  },
  {
    title: "Nothing is listening",
    body: "A refused connection, a host that does not resolve and a route that does not exist are reported as unreachable — separately from a server that answered badly — and the error names the endpoint it tried. This one raises whatever raise_on_error is set to, because there is no result to return.",
    code: 'from effgen import Agent, AgentConfig\nfrom effgen.models.errors import BackendUnreachableError\n\nagent = Agent(AgentConfig(\n    model="Qwen/Qwen2.5-7B-Instruct",\n    base_url="http://127.0.0.1:9/v1",\n    require_model=False,\n))\ntry:\n    agent.run("anything")\nexcept BackendUnreachableError as e:\n    print(type(e).__name__, "->", e)',
    output:
      "BackendUnreachableError -> openai did not answer (model='Qwen/Qwen2.5-7B-Instruct'): OpenAI generation failed [will_retry]: Connection error.. Nothing answered at that endpoint — check the server is running and the base_url, host and port are right. The call was sent to http://127.0.0.1:9/v1.",
    accent: "#ff6b6b",
  },
  {
    title: "The run says how it ended",
    body: "Every result is answered, stopped or failed, and stop_reason names the exit it took. A run the loop ended before the model wrote an answer — at the iteration cap, on a repeated call — is not a success, and what it had reached is on the response as .partial rather than where an answer would be. With the default raise_on_error=True it raises RunStoppedError instead.",
    code: 'from effgen import Agent, AgentConfig\nfrom effgen.tools.builtin import Calculator\n\nagent = Agent(AgentConfig(\n    model="gemini:gemini-3.1-flash-lite",\n    tools=[Calculator()],\n    max_iterations=1,\n    raise_on_error=False,\n))\nr = agent.run("With the calculator: work out 24344 * 334, then multiply that by 7, "\n              "then subtract 19, then divide by 3.")\n\nprint("success:", r.success)\nprint("outcome:", r.outcome)\nprint("stop_reason:", r.stop_reason)\nprint("partial:", r.partial.text if r.partial else None)',
    output: "success: False\noutcome: stopped\nstop_reason: max_iterations_partial\npartial: 8130896",
    accent: "#00e5ff",
  },
];

function FlowPulse() {
  return (
    <motion.div
      className="absolute top-0 left-0 w-3 h-3 rounded-full z-20 pointer-events-none hidden lg:block"
      style={{
        background: "radial-gradient(circle, #00ff88, transparent)",
        boxShadow: "0 0 12px #00ff88, 0 0 24px rgba(0,255,136,0.4)",
      }}
      animate={{
        left: ["0%", "100%"],
        opacity: [0, 1, 1, 1, 1, 0],
      }}
      transition={{
        duration: 4,
        repeat: Infinity,
        ease: "easeInOut",
        repeatDelay: 1,
      }}
    />
  );
}

function StepDetail({ index, onClose }: { index: number; onClose: () => void }) {
  const step = steps[index];
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const StepIcon = step.icon;

  // Escape closes it, Tab stays inside it, and focus returns to whatever opened
  // it. A dialog a keyboard reader can tab out of is a dialog they are lost in.
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
        aria-labelledby="how-step-title"
        className="relative w-full max-w-xl rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl overflow-hidden max-h-[85vh] overflow-y-auto"
        style={{ boxShadow: `0 0 60px ${step.accent}15` }}
        initial={{ scale: 0.9, opacity: 0, y: 30 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.9, opacity: 0, y: 30 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        <div
          className="h-1"
          style={{ background: `linear-gradient(90deg, transparent, ${step.accent}, transparent)` }}
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
              style={{ background: `${step.accent}15`, border: `1px solid ${step.accent}30` }}
            >
              <StepIcon style={accentTextStyle(step.accent)} size={20} />
            </div>
            <div>
              <h3 id="how-step-title" className="text-xl font-black text-gray-900 dark:text-white">
                {step.title}
              </h3>
              <span className="text-xs font-mono" style={accentTextStyle(step.accent)}>
                Step {String(index + 1).padStart(2, "0")}
              </span>
            </div>
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
            {step.explanation}
          </p>

          {step.code && (
            <CodeSample code={step.code} accent={step.accent} output={step.output} />
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

export default function HowItWorks() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  const maxIterations =
    siteData.presets.items.find((preset) => preset.name === "general")?.max_iterations ?? 10;

  return (
    <section
      id="how-it-works"
      className="py-24 bg-white dark:bg-[#020c08] relative overflow-hidden noise-overlay"
      ref={ref}
    >
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <motion.div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-[0.04] pointer-events-none"
        style={{ background: "radial-gradient(circle, #00e5ff 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.15, 1] }}
        transition={{ duration: 10, repeat: Infinity }}
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
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-cyan-500/30 bg-cyan-500/5 text-cyan-600 dark:text-cyan-400 text-sm font-semibold mb-6"
            whileHover={{ borderColor: "rgba(0,229,255,0.6)" }}
          >
            <FiCpu size={14} />
            The agent loop
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
            How <span className="gradient-text">effGen</span> works
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            A model, {toolCount} tools it can be given, and a loop that keeps going until
            there is an answer — or until it runs out of turns and says so.
          </p>
        </motion.div>

        {/* Flow diagram */}
        <div className="relative max-w-6xl mx-auto">
          <div className="hidden lg:block absolute top-[60px] left-[8%] right-[8%] h-px z-0">
            <div className="w-full h-full bg-gradient-to-r from-green-500/30 via-cyan-500/20 to-green-500/30" />
            <div className="relative w-full" style={{ marginTop: "-7px" }}>
              <FlowPulse />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6 lg:gap-4">
            {steps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 40 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: index * 0.12 }}
                className="relative group"
              >
                <button
                  type="button"
                  className="relative w-full p-5 rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden text-center h-full cursor-pointer"
                  onClick={() => setSelectedStep(index)}
                >
                  {/* The card's own words name this button. An `aria-label` would replace
                      them rather than contain them, which is what WCAG 2.5.3 asks for
                      when a control carries visible text; this adds only what pressing
                      it does. */}
                  <span className="sr-only"> — open details</span>
                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                    style={{ background: `radial-gradient(circle at 50% 30%, ${step.accent}15 0%, transparent 70%)` }}
                  />

                  <div
                    className="absolute top-2 right-3 text-[10px] font-mono font-bold opacity-40"
                    style={accentTextStyle(step.accent)}
                  >
                    {String(index + 1).padStart(2, "0")}
                  </div>

                  <motion.div
                    className="relative w-12 h-12 rounded-xl flex items-center justify-center mb-4 mx-auto overflow-hidden"
                    style={{
                      background: `${step.accent}15`,
                      border: `1px solid ${step.accent}30`,
                    }}
                    whileHover={{ scale: 1.1, rotate: 5 }}
                  >
                    <step.icon style={accentTextStyle(step.accent)} size={20} />
                    <motion.div
                      className="absolute inset-[-2px] rounded-xl pointer-events-none"
                      style={{
                        background: `conic-gradient(from 0deg, transparent 70%, ${step.accent}40 90%, transparent 100%)`,
                      }}
                      animate={{ rotate: 360 }}
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    />
                  </motion.div>

                  <h3 className="text-sm font-bold mb-2 text-gray-900 dark:text-white">
                    {step.title}
                  </h3>

                  <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed mb-3">
                    {step.description}
                  </p>

                  <div className="px-2 py-1.5 rounded-lg bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                    <code className="text-[10px] font-mono block truncate" style={accentTextStyle(step.accent)}>
                      {step.detail}
                    </code>
                  </div>

                  <div className="mt-2 text-[9px] text-gray-400 dark:text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity font-medium tracking-wide uppercase">
                    Click for details
                  </div>

                  <motion.div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${step.accent}, transparent)` }}
                  />
                </button>

                {index < steps.length - 1 && (
                  <div className="flex justify-center py-2 lg:hidden">
                    <motion.div
                      className="w-px h-6 bg-gradient-to-b from-green-500/40 to-transparent"
                      initial={{ scaleY: 0 }}
                      animate={inView ? { scaleY: 1 } : {}}
                      transition={{ delay: index * 0.12 + 0.3 }}
                    />
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Loop indicator */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 1 }}
            className="mt-8 text-center"
          >
            <div className="inline-flex flex-wrap items-center justify-center gap-3 px-5 py-2.5 rounded-full bg-green-500/5 border border-green-500/20">
              <motion.div
                className="w-2 h-2 rounded-full bg-green-400"
                animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-xs font-mono text-green-700 dark:text-green-400">
                Steps 02–05 repeat until the model answers or max_iterations is reached
                (the general preset stops at {maxIterations})
              </span>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="text-green-400"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  <polyline points="21 3 21 12 12 12" />
                </svg>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* Step 03, in full: the three strategies */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="max-w-6xl mx-auto mt-20"
        >
          <div className="flex items-center gap-4 mb-6">
            <h3 className="text-[11px] font-mono uppercase tracking-[0.2em] text-gray-600 dark:text-gray-400 whitespace-nowrap">
              Step 03 — how a call is recognised
            </h3>
            <span className="h-px flex-1 bg-gradient-to-r from-cyan-500/25 to-transparent" />
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 max-w-3xl mb-6">
            Not every model has a function-calling API, and the ones that do do not all
            get it right. <code className="font-mono text-[13px]">tool_calling_mode</code>{" "}
            decides how a call is read out of the model&rsquo;s reply — the default,{" "}
            <code className="font-mono text-[13px]">&quot;auto&quot;</code>, picks per model.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {strategies.map((strategy) => (
              <div
                key={strategy.name}
                className="relative p-5 rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 overflow-hidden"
              >
                <div
                  className="absolute top-0 left-0 right-0 h-px"
                  style={{ background: `linear-gradient(90deg, transparent, ${strategy.accent}60, transparent)` }}
                />
                <code
                  className="text-xs font-mono font-semibold px-2 py-0.5 rounded"
                  style={{
                    ...accentTextStyle(strategy.accent),
                    background: `${strategy.accent}15`,
                    border: `1px solid ${strategy.accent}30`,
                  }}
                >
                  {strategy.name}
                </code>
                <h4 className="mt-3 text-sm font-bold text-gray-900 dark:text-white">
                  {strategy.headline}
                </h4>
                <p className="mt-2 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                  {strategy.body}
                </p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* When a step fails */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="max-w-6xl mx-auto mt-20"
        >
          <div className="flex items-center gap-4 mb-6">
            <h3 className="text-[11px] font-mono uppercase tracking-[0.2em] text-gray-600 dark:text-gray-400 whitespace-nowrap">
              When a step fails
            </h3>
            <span className="h-px flex-1 bg-gradient-to-r from-orange-500/25 to-transparent" />
            <FiAlertTriangle className="text-orange-500/60" size={14} aria-hidden="true" />
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 max-w-3xl mb-6">
            A tool that raised, a server that never answered and a run that stopped early
            are three different things, and effGen reports them as three different things.
            The output below is what each one prints.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {failures.map((failure) => (
              <div
                key={failure.title}
                className="relative rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 overflow-hidden p-5 flex flex-col"
              >
                <div
                  className="absolute top-0 left-0 right-0 h-px"
                  style={{ background: `linear-gradient(90deg, transparent, ${failure.accent}60, transparent)` }}
                />
                <h4 className="text-sm font-bold text-gray-900 dark:text-white">
                  {failure.title}
                </h4>
                <p className="mt-2 mb-4 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                  {failure.body}
                </p>
                <CodeSample
                  className="mt-auto"
                  code={failure.code}
                  accent={failure.accent}
                  output={failure.output}
                />
              </div>
            ))}
          </div>
        </motion.div>
      </Container>

      {/* Expanded Step Modal */}
      <AnimatePresence>
        {selectedStep !== null && (
          <StepDetail index={selectedStep} onClose={() => setSelectedStep(null)} />
        )}
      </AnimatePresence>
    </section>
  );
}
