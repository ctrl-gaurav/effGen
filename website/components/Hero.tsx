"use client";

import { motion } from "framer-motion";
import { FiArrowRight, FiStar, FiZap, FiCode, FiTrendingUp, FiBookOpen, FiPackage, FiFileText, FiGithub, FiTerminal, FiCopy, FiCheck } from "react-icons/fi";
import { useInView } from "react-intersection-observer";
import { useEffect, useState, useRef } from "react";
import Container from "./Container";
import Link from "next/link";
import GitHubStats from "./GitHubStats";
import { usePyPIVersion } from "./PyPIVersion";
import { siteData } from "./siteData";
import { useReducedMotion } from "./useReducedMotion";
import { useLowPower } from "./useLowPower";
import { accentTextStyle } from "./accentText";

// Fixed particle positions to avoid SSR/client hydration mismatch
const PARTICLE_POSITIONS = [
  49, 85, 36, 65, 17, 15, 44, 8, 91, 88, 3, 52, 42, 84, 6, 84, 35, 82, 5, 29,
  22, 73, 58, 11, 95, 38, 67, 78, 2, 60,
];
const PARTICLE_HEIGHTS = [
  520, 610, 480, 570, 540, 600, 490, 580, 530, 620,
  465, 555, 510, 590, 475, 615, 500, 560, 545, 570,
  495, 585, 525, 605, 460, 575, 505, 595, 485, 550,
];
const PARTICLE_DURATIONS = [
  10, 13, 9, 12, 11, 14, 10, 13, 12, 15, 9, 11, 10, 13, 12, 14, 11, 10, 13, 12,
  10, 14, 9, 11, 13, 12, 10, 15, 11, 9,
];
const PARTICLE_SIZES = [
  2, 3, 2, 4, 2, 5, 3, 2, 6, 3, 2, 4, 2, 3, 5, 2, 3, 4, 2, 6,
  3, 2, 4, 3, 2, 5, 3, 2, 4, 3,
];
const PARTICLE_BRIGHTNESS = [
  0.4, 0.8, 0.3, 0.6, 0.4, 1.0, 0.5, 0.3, 0.9, 0.5,
  0.3, 0.7, 0.4, 0.6, 0.9, 0.3, 0.5, 0.7, 0.4, 1.0,
  0.5, 0.3, 0.8, 0.4, 0.3, 0.9, 0.6, 0.4, 0.7, 0.5,
];

function Particle({ index, delay }: { index: number; delay: number }) {
  const x = PARTICLE_POSITIONS[index % PARTICLE_POSITIONS.length];
  const height = PARTICLE_HEIGHTS[index % PARTICLE_HEIGHTS.length];
  const duration = PARTICLE_DURATIONS[index % PARTICLE_DURATIONS.length];
  const size = PARTICLE_SIZES[index % PARTICLE_SIZES.length];
  const brightness = PARTICLE_BRIGHTNESS[index % PARTICLE_BRIGHTNESS.length];
  const isGlowing = brightness > 0.7;

  return (
    <motion.div
      className="absolute rounded-full"
      style={{
        left: `${x}%`,
        bottom: "-10px",
        width: `${size}px`,
        height: `${size}px`,
        background: isGlowing
          ? `radial-gradient(circle, rgba(0,255,136,${brightness}) 0%, rgba(0,255,136,${brightness * 0.3}) 70%)`
          : `rgba(0,255,136,${brightness * 0.6})`,
        boxShadow: isGlowing ? `0 0 ${size * 3}px rgba(0,255,136,${brightness * 0.5})` : 'none',
      }}
      animate={{
        y: [0, -height],
        opacity: [0, brightness, brightness, 0],
        scale: [0, 1, 1, 0],
      }}
      transition={{
        duration,
        delay,
        repeat: Infinity,
        ease: "easeOut",
      }}
    />
  );
}

// Counter hook for animated stats. A duration of 0 means the visitor asked for
// reduced motion: the figure lands at once instead of counting up to itself.
function useCounter(end: number, inView: boolean, duration: number = 1500) {
  const [count, setCount] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    if (!inView || started.current) return;
    if (duration <= 0) {
      started.current = true;
      setCount(end);
      return;
    }
    started.current = true;
    const startTime = Date.now();
    const timer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
      setCount(Math.round(eased * end));
      if (progress >= 1) clearInterval(timer);
    }, 16);
    return () => clearInterval(timer);
  }, [inView, end, duration]);

  return count;
}

// Terminal simulation with thinking + tool selection
function EnhancedTerminal({ inView }: { inView: boolean }) {
  const [phase, setPhase] = useState(0); // 0=typing, 1=thinking, 2=tool, 3=result, 4=done
  const [typedCmd, setTypedCmd] = useState("");
  const fullCmd = 'agent.run("What is 24344 * 334?")';
  const mountedRef = useRef(true);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!inView) return;

    // With motion off, the pane paints the run already finished — the command
    // typed, every phase shown, the answer in place. That is the frame the
    // animation settles on anyway, so a reader who asked for stillness gets the
    // whole thing at once instead of a loop they did not ask for.
    if (reducedMotion) {
      setTypedCmd(fullCmd);
      setPhase(4);
      return;
    }

    let timeoutId: ReturnType<typeof setTimeout>;
    let intervalId: ReturnType<typeof setInterval>;

    function runCycle() {
      if (!mountedRef.current) return;
      // Reset
      setPhase(0);
      setTypedCmd("");

      let i = 0;
      intervalId = setInterval(() => {
        if (!mountedRef.current) { clearInterval(intervalId); return; }
        if (i <= fullCmd.length) {
          setTypedCmd(fullCmd.slice(0, i));
          i++;
        } else {
          clearInterval(intervalId);
          // Phase chain with timeouts
          timeoutId = setTimeout(() => {
            if (!mountedRef.current) return;
            setPhase(1); // thinking
            timeoutId = setTimeout(() => {
              if (!mountedRef.current) return;
              setPhase(2); // tool selection
              timeoutId = setTimeout(() => {
                if (!mountedRef.current) return;
                setPhase(3); // result
                timeoutId = setTimeout(() => {
                  if (!mountedRef.current) return;
                  setPhase(4); // done
                  // Pause then restart the whole cycle
                  timeoutId = setTimeout(() => {
                    if (!mountedRef.current) return;
                    runCycle();
                  }, 2800);
                }, 1200);
              }, 1200);
            }, 1400);
          }, 400);
        }
      }, 35);
    }

    runCycle();

    return () => {
      clearTimeout(timeoutId);
      clearInterval(intervalId);
    };
  }, [inView, reducedMotion]);

  return (
    <div className="text-left text-sm font-mono leading-relaxed">
      {/* Command line */}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-green-700 dark:text-green-400">$</span>
        <span className="text-gray-800 dark:text-gray-200">{typedCmd}</span>
        {phase === 0 && (
          <motion.span className="w-2 h-4 bg-green-600 dark:bg-green-400" animate={{ opacity: [1, 0] }} transition={{ duration: 0.6, repeat: Infinity }} />
        )}
      </div>

      {/* Thinking phase */}
      {phase >= 1 && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-1"
        >
          <span className="text-gray-600 dark:text-gray-400 text-xs">
            <span className="text-yellow-500">[Thought]</span> I need to multiply 24344 by 334
            {phase === 1 && (
              <motion.span animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 0.8, repeat: Infinity }}>...</motion.span>
            )}
          </span>
        </motion.div>
      )}

      {/* Tool selection */}
      {phase >= 2 && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-1"
        >
          <span className="text-xs">
            <span className="text-cyan-400">[Action]</span>{" "}
            <span className="text-purple-400">Calculator</span>
            <span className="text-gray-600 dark:text-gray-400">(</span>
            <span className="text-green-400">expression=</span>
            <span className="text-orange-400">&quot;24344 * 334&quot;</span>
            <span className="text-gray-600 dark:text-gray-400">)</span>
          </span>
        </motion.div>
      )}

      {/* Result */}
      {phase >= 3 && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-1"
        >
          <span className="text-xs">
            <span className="text-blue-400">[Observation]</span>{" "}
            <span className="text-gray-400">8130896</span>
          </span>
        </motion.div>
      )}

      {/* Final answer */}
      {phase >= 4 && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <span className="text-xs">
            <span className="text-green-400">[Answer]</span>{" "}
            <span className="text-white font-semibold">The result of 24344 * 334 is 8,130,896</span>
          </span>
        </motion.div>
      )}
    </div>
  );
}

// Animated geometric shapes for background
function GeometricShapes() {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {/* Rotating hexagon */}
      <motion.div
        className="absolute top-[15%] right-[10%] hidden lg:block"
        animate={{ rotate: 360 }}
        transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
      >
        <svg width="120" height="120" viewBox="0 0 120 120" fill="none" style={{ opacity: 0.04 }}>
          <polygon points="60,5 110,30 110,90 60,115 10,90 10,30" stroke="#00ff88" strokeWidth="1" />
        </svg>
      </motion.div>

      {/* Rotating triangle */}
      <motion.div
        className="absolute bottom-[20%] left-[5%] hidden lg:block"
        animate={{ rotate: -360 }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      >
        <svg width="80" height="80" viewBox="0 0 80 80" fill="none" style={{ opacity: 0.03 }}>
          <polygon points="40,5 75,70 5,70" stroke="#00e5ff" strokeWidth="1" />
        </svg>
      </motion.div>

      {/* Diamond */}
      <motion.div
        className="absolute top-[40%] left-[8%] hidden lg:block"
        animate={{ rotate: 360, scale: [1, 1.1, 1] }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      >
        <svg width="60" height="60" viewBox="0 0 60 60" fill="none" style={{ opacity: 0.03 }}>
          <polygon points="30,5 55,30 30,55 5,30" stroke="#a78bfa" strokeWidth="1" />
        </svg>
      </motion.div>

      {/* Circle */}
      <motion.div
        className="absolute bottom-[30%] right-[8%] hidden lg:block"
        animate={{ scale: [1, 1.15, 1] }}
        transition={{ duration: 8, repeat: Infinity }}
      >
        <svg width="90" height="90" viewBox="0 0 90 90" fill="none" style={{ opacity: 0.03 }}>
          <circle cx="45" cy="45" r="40" stroke="#ffd700" strokeWidth="1" />
          <circle cx="45" cy="45" r="25" stroke="#ffd700" strokeWidth="0.5" strokeDasharray="3 3" />
        </svg>
      </motion.div>
    </div>
  );
}

// Every figure in the hero comes from data/effgen.json, which
// scripts/gen_site_data.py derives from the installed framework. A number here
// cannot drift from the package the way the hand-typed ones did.
const TOOL_COUNT = siteData.tools.count;
const PRESET_COUNT = siteData.presets.count;
const TEMPLATE_COUNT = siteData.prompts.library;
const MODEL_COUNT = siteData.models.models;
const PROVIDER_COUNT = siteData.models.adapter_count;
const CATALOG_PROVIDER_COUNT = siteData.models.with_catalog_count;
// Ten provider adapters plus the four engines that run weights on your own
// hardware — the number of places a model can come from.
const BACKEND_COUNT = PROVIDER_COUNT + siteData.models.local_engines.length;

// The particle field is the largest single item in the hero's frame budget: at
// full strength it is 30 of the 59 elements that animate continuously. On a
// device that reports few cores or little memory it is drawn at a third of that
// — the same field, thinner. Everything else in the composition is unchanged.
const PARTICLE_COUNT_FULL = 30;
const PARTICLE_COUNT_LOW = 10;

export default function Hero() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const reducedMotion = useReducedMotion();
  const lowPower = useLowPower();
  const particleCount = lowPower ? PARTICLE_COUNT_LOW : PARTICLE_COUNT_FULL;
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [typedText, setTypedText] = useState("");
  const pypiVersion = usePyPIVersion();

  const fullText = "pip install effgen";
  const [codeCopied, setCodeCopied] = useState(false);

  // Animated stat counters. Under reduced motion the count-up is skipped and the
  // figure is simply there — the number is the point, the counting is not.
  const countTools = useCounter(TOOL_COUNT, inView, reducedMotion ? 0 : 1500);
  const countPresets = useCounter(PRESET_COUNT, inView, reducedMotion ? 0 : 1500);
  const countTemplates = useCounter(TEMPLATE_COUNT, inView, reducedMotion ? 0 : 1500);
  const countBackends = useCounter(BACKEND_COUNT, inView, reducedMotion ? 0 : 1500);
  const countModels = useCounter(MODEL_COUNT, inView, reducedMotion ? 0 : 1500);

  const copyText = (text: string, onSuccess?: () => void) => {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(onSuccess).catch(() => {
        const el = document.createElement("textarea");
        el.value = text;
        el.style.position = "fixed";
        el.style.opacity = "0";
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
        onSuccess?.();
      });
    } else {
      const el = document.createElement("textarea");
      el.value = text;
      el.style.position = "fixed";
      el.style.opacity = "0";
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      onSuccess?.();
    }
  };

  const handleCopy = () => copyText(fullText);

  const handleCodeCopy = () => copyText("pip install effgen", () => {
    setCodeCopied(true);
    setTimeout(() => setCodeCopied(false), 2000);
  });

  useEffect(() => {
    // Two blurred orbs follow the pointer. With motion off they stay where they
    // are, so the listener is not attached at all rather than attached and
    // ignored — the whole point is that nothing under the cursor moves.
    if (reducedMotion) return;
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener("mousemove", handleMouseMove, { passive: true });
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [reducedMotion]);

  useEffect(() => {
    if (!inView) return;
    // The install line types itself out. With motion off it is simply there —
    // a reader who asked for stillness should not have to wait to read it.
    if (reducedMotion) {
      setTypedText(fullText);
      return;
    }
    let i = 0;
    const timer = setInterval(() => {
      if (i <= fullText.length) {
        setTypedText(fullText.slice(0, i));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 60);
    return () => clearInterval(timer);
  }, [inView, reducedMotion]);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15 } },
  };

  const itemVariants = {
    hidden: { y: 40, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.7, ease: "easeOut" as const } },
  };

  const stats = [
    { icon: FiCode, value: countTools.toString(), rawValue: TOOL_COUNT, label: "Built-in tools", color: "#00ff88" },
    { icon: FiBookOpen, value: countPresets.toString(), rawValue: PRESET_COUNT, label: "Presets", color: "#00e5ff" },
    { icon: FiZap, value: countTemplates.toString(), rawValue: TEMPLATE_COUNT, label: "Prompt templates", color: "#ffd700" },
    { icon: FiTrendingUp, value: countBackends.toString(), rawValue: BACKEND_COUNT, label: "Model backends", color: "#ff6b6b" },
    { icon: FiStar, value: countModels.toString(), rawValue: MODEL_COUNT, label: "Catalogued models", color: "#a78bfa" },
  ];

  // The words of the headline, animated in one at a time. The third carries the
  // accent treatment; the array is what the loop below reads, so a change here
  // is the only change a new headline needs.
  const headingLine1 = ["Build", "AI", "agents"];
  const ACCENT_WORD = "agents";

  return (
    <section
      id="home"
      className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20 noise-overlay"
      ref={ref}
    >
      {/* Deep dark background */}
      <div className="absolute inset-0 bg-white dark:bg-[#020c08]" />

      {/* Animated radial gradient */}
      <motion.div
        className="absolute inset-0"
        animate={{
          background: [
            "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,255,136,0.12) 0%, transparent 70%)",
            "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,201,110,0.15) 0%, transparent 70%)",
            "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,255,136,0.12) 0%, transparent 70%)",
          ],
        }}
        transition={{ duration: 6, repeat: Infinity }}
      />

      {/* Cyber grid */}
      <div className="absolute inset-0 grid-pattern" />

      {/* Geometric background shapes */}
      <GeometricShapes />

      {/* Pulsing concentric rings */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        {[0, 1, 2, 3].map((i) => (
          <motion.div
            key={i}
            className="absolute rounded-full border border-green-500/10"
            style={{ width: "300px", height: "300px" }}
            animate={{
              scale: [0.5, 2.5],
              opacity: [0.2, 0],
            }}
            transition={{
              duration: 4,
              delay: i * 1,
              repeat: Infinity,
              ease: "easeOut",
            }}
          />
        ))}
      </div>

      {/* Scan line */}
      <motion.div
        className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-400/30 to-transparent pointer-events-none z-10"
        animate={{ y: ["-100vh", "100vh"] }}
        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
      />

      {/* Mouse-following primary orb */}
      <motion.div
        className="absolute w-[700px] h-[700px] rounded-full blur-[120px] pointer-events-none"
        style={{ background: "radial-gradient(circle, rgba(0,255,136,0.12) 0%, transparent 70%)" }}
        animate={{
          x: mousePosition.x - 350,
          y: mousePosition.y - 350,
        }}
        transition={{ type: "spring", stiffness: 30, damping: 30 }}
      />

      {/* Secondary smaller orb */}
      <motion.div
        className="absolute w-[300px] h-[300px] rounded-full blur-[80px] pointer-events-none"
        style={{ background: "radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%)" }}
        animate={{
          x: mousePosition.x - 150,
          y: mousePosition.y - 150,
        }}
        transition={{ type: "spring", stiffness: 15, damping: 40 }}
      />

      {/* Floating particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {Array.from({ length: particleCount }).map((_, i) => (
          <Particle key={i} index={i} delay={i * 0.4} />
        ))}
      </div>

      {/* Corner decorations */}
      <div className="absolute top-20 left-8 text-green-500/20 font-mono text-xs hidden lg:block">
        {["[SYS:ONLINE]", "[AI:READY]", `[v${pypiVersion.loading ? "..." : pypiVersion.version}]`].map((t, i) => (
          <motion.div key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.3, 0.7, 0.3] }}
            transition={{ duration: 3, delay: i * 1, repeat: Infinity }}
          >{t}</motion.div>
        ))}
      </div>
      <div className="absolute top-20 right-8 text-green-500/20 font-mono text-xs text-right hidden lg:block">
        {[
          "[AGENTS:ACTIVE]",
          `[TOOLS:${TOOL_COUNT}]`,
          `[PROMPTS:${TEMPLATE_COUNT}]`,
          `[PROVIDERS:${PROVIDER_COUNT}]`,
        ].map((t, i) => (
          <motion.div key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.3, 0.7, 0.3] }}
            transition={{ duration: 3, delay: i * 1.2, repeat: Infinity }}
          >{t}</motion.div>
        ))}
      </div>

      <Container className="relative z-10 py-16">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
          className="text-center"
        >
          {/* Top badge */}
          <motion.div variants={itemVariants} className="flex flex-wrap items-center justify-center gap-3 mb-8">
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 backdrop-blur-sm"
              whileHover={{ scale: 1.05, borderColor: "rgba(0,255,136,0.6)" }}
            >
              <motion.div
                className="w-2 h-2 rounded-full bg-green-400"
                animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-sm font-semibold text-green-700 dark:text-green-400">
                effGen 1.3.0 is out — every run says how it ended
              </span>
              <FiZap className="text-green-700 dark:text-green-400" size={14} />
            </motion.div>
          </motion.div>

          {/* GitHub Stats */}
          <motion.div variants={itemVariants} className="flex justify-center mb-8">
            <GitHubStats showForks={true} showContributors={true} />
          </motion.div>

          {/* Main heading with staggered word animation */}
          <motion.h1
            variants={itemVariants}
            className="text-5xl sm:text-6xl lg:text-8xl font-black mb-6 leading-[0.95] tracking-tight text-gray-900 dark:text-white"
          >
            {headingLine1.map((word, i) => (
              <motion.span
                key={i}
                className={`inline-block mr-3 sm:mr-4 ${word === ACCENT_WORD ? "relative" : ""}`}
                initial={{ opacity: 0, y: 30, filter: "blur(10px)", scale: 0.9 }}
                animate={inView ? { opacity: 1, y: 0, filter: "blur(0px)", scale: 1 } : {}}
                transition={{ duration: 0.6, delay: 0.3 + i * 0.12, ease: "easeOut" }}
              >
                {word === ACCENT_WORD ? (
                  <span className="gradient-text neon-text glitch-text" data-text={word}>
                    {word}
                  </span>
                ) : (
                  word
                )}
              </motion.span>
            ))}
            <br />
            <motion.span
              className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-600 dark:text-gray-400 tracking-normal"
              initial={{ opacity: 0, y: 20, filter: "blur(8px)" }}
              animate={inView ? { opacity: 1, y: 0, filter: "blur(0px)" } : {}}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              on small language models
            </motion.span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            variants={itemVariants}
            className="text-lg sm:text-xl text-gray-600 dark:text-gray-400 mb-10 max-w-2xl mx-auto leading-relaxed"
          >
            Run a small model on your own hardware, or reach{" "}
            <span className="font-bold text-shimmer">
              {CATALOG_PROVIDER_COUNT} providers and any OpenAI-compatible server
            </span>{" "}
            through the same agent. Tools, memory, RAG, guardrails, evaluation and a
            production server come with it.
          </motion.p>

          {/* Terminal quick install */}
          <motion.div variants={itemVariants} className="flex justify-center mb-10">
            <motion.div
              className="relative flex items-center gap-3 px-6 py-3 rounded-xl bg-gray-100 dark:bg-black/80 border border-gray-300 dark:border-green-500/30 backdrop-blur-sm font-mono text-sm cursor-pointer select-none"
              whileHover={{ borderColor: "rgba(0,255,136,0.6)", boxShadow: "0 0 20px rgba(0,255,136,0.15)" }}
              onClick={handleCopy}
              title="Click to copy"
            >
              <FiTerminal className="text-green-700 dark:text-green-400 flex-shrink-0" size={16} />
              <span className="text-green-700 dark:text-green-400">$</span>
              <span className="text-gray-800 dark:text-gray-200">{typedText}</span>
              <motion.span
                className="w-2 h-4 bg-green-600 dark:bg-green-400 flex-shrink-0"
                animate={{ opacity: [1, 0, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              />

              <motion.div
                className="absolute inset-0 rounded-xl overflow-hidden"
                initial={false}
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-green-400/5 to-transparent"
                  animate={{ x: ["-100%", "200%"] }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                />
              </motion.div>
            </motion.div>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            variants={itemVariants}
            className="flex flex-wrap items-center justify-center gap-3 mb-16"
          >
            {/* Primary CTA */}
            <motion.a
              href="#quickstart"
              className="group relative px-8 py-4 rounded-full font-bold text-black text-lg flex items-center gap-2 overflow-hidden"
              style={{
                background: "linear-gradient(135deg, #00ff88, #00c96e)",
                boxShadow: "0 0 30px rgba(0,255,136,0.4)",
              }}
              whileHover={{
                scale: 1.05, y: -2,
                boxShadow: "0 0 50px rgba(0,255,136,0.6)",
              }}
              whileTap={{ scale: 0.95 }}
              animate={{
                boxShadow: [
                  "0 0 30px rgba(0,255,136,0.3)",
                  "0 0 50px rgba(0,255,136,0.5)",
                  "0 0 30px rgba(0,255,136,0.3)",
                ],
              }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <span className="absolute inset-0 bg-white/0 group-hover:bg-white/10 transition-colors" />
              <FiZap />
              Get Started
              <FiArrowRight className="group-hover:translate-x-1 transition-transform" />
            </motion.a>

            {/* arXiv */}
            <motion.a
              href="https://arxiv.org/abs/2602.00887"
              target="_blank" rel="noopener noreferrer"
              className="px-6 py-4 rounded-full font-semibold text-green-700 dark:text-green-400 border border-green-500/30 bg-green-500/5 flex items-center gap-2 hover:border-green-400 hover:bg-green-500/10 transition-all"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
            >
              <FiFileText size={16} />
              arXiv Paper
            </motion.a>

            {/* PyPI */}
            <motion.a
              href="https://pypi.org/project/effgen/"
              target="_blank" rel="noopener noreferrer"
              className="px-6 py-4 rounded-full font-semibold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/50 flex items-center gap-2 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
            >
              <FiPackage size={16} />
              PyPI
              <span className="px-2 py-0.5 text-xs rounded-full bg-green-500/20 text-green-700 dark:text-green-400 border border-green-500/30 font-mono">
                v{pypiVersion.loading ? "..." : pypiVersion.version}
              </span>
            </motion.a>

            {/* GitHub */}
            <motion.a
              href="https://github.com/ctrl-gaurav/effGen"
              target="_blank" rel="noopener noreferrer"
              className="px-6 py-4 rounded-full font-semibold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/50 flex items-center gap-2 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
            >
              <FiGithub size={16} />
              GitHub
              <FiStar size={14} className="text-yellow-400" />
            </motion.a>

            {/* Examples */}
            <Link href="/examples">
              <motion.span
                className="px-6 py-4 rounded-full font-semibold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/50 flex items-center gap-2 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all cursor-pointer"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
              >
                <FiCode size={16} />
                Examples
              </motion.span>
            </Link>
          </motion.div>

          {/* Enhanced Stats row */}
          <motion.div
            variants={itemVariants}
            className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 max-w-4xl mx-auto mb-16"
          >
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                className="group relative p-5 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/50 backdrop-blur-sm overflow-hidden shadow-sm dark:shadow-none"
                whileHover={{ y: -8, borderColor: stat.color + "50" }}
                initial={{ opacity: 0, y: 20, scale: 0.9 }}
                animate={inView ? { opacity: 1, y: 0, scale: 1 } : {}}
                transition={{ delay: 0.6 + index * 0.1, type: "spring", stiffness: 300 }}
              >
                {/* Glow on hover */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                  style={{ background: `radial-gradient(circle at 50% 50%, ${stat.color}15 0%, transparent 70%)` }}
                />

                <motion.div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-3 mx-auto"
                  style={{ background: `${stat.color}20`, border: `1px solid ${stat.color}30` }}
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                >
                  <stat.icon style={accentTextStyle(stat.color)} size={18} />
                </motion.div>

                <motion.div
                  className="text-3xl font-black mb-1"
                  style={accentTextStyle(stat.color)}
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={inView ? { scale: 1, opacity: 1 } : {}}
                  transition={{ delay: 0.8 + index * 0.1, type: "spring", stiffness: 200 }}
                >
                  {stat.value}
                </motion.div>
                <div className="text-xs text-gray-600 dark:text-gray-400 font-medium uppercase tracking-wide">
                  {stat.label}
                </div>

                {/* Bottom accent */}
                <motion.div
                  className="absolute bottom-0 left-0 right-0 h-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(90deg, transparent, ${stat.color}, transparent)` }}
                />
              </motion.div>
            ))}
          </motion.div>

          {/* Enhanced Terminal Preview */}
          <motion.div variants={itemVariants} className="max-w-3xl mx-auto">
            <motion.div
              className="relative group"
              whileHover={{ y: -6 }}
              transition={{ duration: 0.3 }}
            >
              {/* Outer glow */}
              <motion.div
                className="absolute -inset-1 rounded-2xl blur-xl opacity-0 dark:opacity-40 group-hover:opacity-20 dark:group-hover:opacity-70 transition-opacity pointer-events-none"
                animate={{
                  background: [
                    "linear-gradient(45deg, #00ff88, #00c96e, #00ff88)",
                    "linear-gradient(90deg, #00c96e, #00ff88, #00c96e)",
                    "linear-gradient(135deg, #00ff88, #00c96e, #00ff88)",
                  ],
                }}
                transition={{ duration: 4, repeat: Infinity }}
              />

              <div className="relative rounded-2xl bg-gray-50 dark:bg-[#0a1a0f] border border-gray-300 dark:border-green-500/20 overflow-hidden shadow-xl">
                {/* Top gradient line */}
                <motion.div
                  className="absolute top-0 left-0 right-0 h-px"
                  animate={{
                    background: [
                      "linear-gradient(90deg, transparent, #00ff88, #00c96e, transparent)",
                      "linear-gradient(90deg, transparent, #00c96e, #00ff88, transparent)",
                    ],
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                />

                {/* Scanline overlay */}
                <div
                  className="absolute inset-0 pointer-events-none z-20 opacity-[0.02] dark:opacity-[0.03]"
                  style={{
                    backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.1) 2px, rgba(0,255,136,0.1) 4px)",
                    backgroundSize: "100% 4px",
                  }}
                />

                <div className="p-6">
                  {/* Terminal header */}
                  <div className="flex items-center justify-between mb-5">
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1.5">
                        {["#ff5f57", "#febc2e", "#28c840"].map((c, i) => (
                          <motion.div
                            key={i}
                            className="w-3 h-3 rounded-full cursor-pointer"
                            style={{ background: c, boxShadow: `0 0 6px ${c}60` }}
                            whileHover={{ scale: 1.3 }}
                          />
                        ))}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-gray-600 dark:text-gray-400 dark:text-gray-600">effgen-terminal</span>
                        <motion.span
                          className="text-xs font-mono text-green-700/60 dark:text-green-400/60 bg-green-500/10 px-2 py-0.5 rounded border border-green-500/20"
                          animate={{ opacity: [0.6, 1, 0.6] }}
                          transition={{ duration: 2, repeat: Infinity }}
                        >
                          ● RUNNING
                        </motion.span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-gray-600 dark:text-gray-400 bg-gray-200 dark:bg-gray-800 px-3 py-1 rounded-full">
                        agent_demo.py
                      </span>
                      <motion.button
                        onClick={handleCodeCopy}
                        aria-label={codeCopied ? "Copied" : "Copy the example"}
                        className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-all text-xs border border-gray-300 dark:border-gray-700"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        {codeCopied ? <FiCheck size={12} className="text-green-400" /> : <FiCopy size={12} />}
                      </motion.button>
                    </div>
                  </div>

                  {/* Enhanced terminal with agent loop animation — fixed height to prevent layout shift */}
                  <div className="min-h-[140px]">
                    <EnhancedTerminal inView={inView} />
                  </div>

                  {/* Output bar */}
                  <motion.div
                    className="mt-4 p-3 rounded-xl bg-green-50 dark:bg-black/60 border border-green-200 dark:border-green-500/20"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 1.2 }}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs text-gray-600 dark:text-gray-400 font-mono">AGENT LOOP</span>
                      <motion.span
                        className="w-1.5 h-1.5 rounded-full bg-green-500 dark:bg-green-400"
                        animate={{ scale: [1, 1.5, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      />
                    </div>
                    <div className="flex items-center gap-3 text-[10px] font-mono text-gray-600 dark:text-gray-400">
                      <span className="text-yellow-400">Thought</span>
                      <span>&rarr;</span>
                      <span className="text-cyan-400">Action</span>
                      <span>&rarr;</span>
                      <span className="text-blue-400">Observation</span>
                      <span>&rarr;</span>
                      <span className="text-green-400">Answer</span>
                    </div>
                  </motion.div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      </Container>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <span className="text-xs font-mono text-green-500/50 tracking-widest">SCROLL</span>
        <div className="w-5 h-8 rounded-full border border-green-500/30 flex items-start justify-center p-1.5">
          <motion.div
            className="w-1 h-1.5 rounded-full bg-green-400"
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </section>
  );
}
