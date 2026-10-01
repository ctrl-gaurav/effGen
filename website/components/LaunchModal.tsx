"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiX,
  FiArrowRight,
  FiTerminal,
  FiActivity,
  FiCheck,
  FiCopy,
  FiStar,
  FiShield,
  FiCode,
  FiCpu,
  FiServer,
} from "react-icons/fi";
import { useFocusTrap } from "./useFocusTrap";
import { resolveRoute } from "./routes";
import { useReducedMotion } from "./useReducedMotion";
import { accentTextStyle } from "./accentText";

// One key per release, so a reader who dismissed the 1.2.0 modal still sees this
// one, and dismissing this one is remembered.
const STORAGE_KEY = "effgen_v130_launch_seen";

// Four, not the whole release. The full 1.3.0 story is a page — /changelog — and
// restating it here would mean two copies to keep true.
const HIGHLIGHTS = [
  { icon: FiActivity, label: "Every run says how it ended", sub: "response.termination is done, not_possible, stuck, tool_failed or error, and a run that stops making progress is asked for its answer", accent: "#00ff88" },
  { icon: FiCode, label: "A broken tool call is read", sub: "A call written in a broken or unexpected shape runs instead of ending the run, and string arguments are decoded", accent: "#22d3ee" },
  { icon: FiCpu, label: "Measured, not assumed", sub: "A served or local model is probed once for what it does with a tool, and tool_calling_mode=\"auto\" follows the result", accent: "#a78bfa" },
  { icon: FiServer, label: "One agent, many conversations", sub: "Overlapping run(session=...) and stream(session=...) calls on one agent no longer mix their turns", accent: "#f59e0b" },
];

// A minor release can still change what existing code sees. A launch modal that
// only lists the good news is how someone upgrades into a surprise. Four of the
// ten; the changelog has the rest.
const BREAKING = [
  "max_turns_without_progress defaults to 2: a run with no new tool result is asked for its answer. None restores 1.2.0's loop.",
  "A run whose tool keeps failing on its own side stops with tool_failed and raises RunStoppedError by default.",
  "recover_lost_tool_calls defaults to True, so some runs make more tool calls. False restores 1.2.0's reader.",
  "A served or local model is probed once at tool_calling_mode=\"auto\"; capability_probe=False turns it off.",
];

// Fixed positions to avoid SSR/hydration mismatch
const PARTICLES = Array.from({ length: 18 }, (_, i) => ({
  x: (i * 53) % 100,
  delay: (i % 6) * 0.4,
  duration: 6 + (i % 5),
  size: 1 + (i % 3),
}));

export default function LaunchModal() {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const seen = window.localStorage.getItem(STORAGE_KEY);
      if (!seen) {
        const t = setTimeout(() => setOpen(true), 700);
        return () => clearTimeout(t);
      }
    } catch {
      const t = setTimeout(() => setOpen(true), 700);
      return () => clearTimeout(t);
    }
  }, []);

  const close = useCallback(() => {
    setOpen(false);
    try {
      window.localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      /* ignore */
    }
  }, []);

  const copyInstall = () => {
    const text = "pip install -U effgen";
    const done = () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done).catch(() => {
        const el = document.createElement("textarea");
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
        done();
      });
    } else {
      const el = document.createElement("textarea");
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      done();
    }
  };

  // Escape closes, Tab stays inside the dialog, and focus returns to whatever
  // had it when the dialog opened. A modal that does none of those is a trap for
  // anyone not using a pointer.
  useFocusTrap(dialogRef, open, close);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[1000] flex items-center justify-center px-4 py-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Backdrop with animated radial gradient */}
          <motion.div
            className="absolute inset-0 cursor-pointer"
            onClick={close}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
            <motion.div
              className="absolute inset-0"
              animate={{
                background: [
                  "radial-gradient(ellipse 60% 60% at 30% 30%, rgba(0,255,136,0.18) 0%, transparent 60%)",
                  "radial-gradient(ellipse 60% 60% at 70% 70%, rgba(0,229,255,0.18) 0%, transparent 60%)",
                  "radial-gradient(ellipse 60% 60% at 30% 70%, rgba(167,139,250,0.18) 0%, transparent 60%)",
                  "radial-gradient(ellipse 60% 60% at 70% 30%, rgba(0,255,136,0.18) 0%, transparent 60%)",
                ],
              }}
              transition={{ duration: 14, repeat: Infinity, ease: "linear" }}
            />
            {/* Floating particles */}
            <div className="absolute inset-0 overflow-hidden">
              {(reducedMotion ? [] : PARTICLES).map((p, i) => (
                <motion.div
                  key={i}
                  className="absolute rounded-full bg-green-400"
                  style={{
                    left: `${p.x}%`,
                    bottom: -10,
                    width: p.size,
                    height: p.size,
                    boxShadow: `0 0 ${p.size * 4}px rgba(0,255,136,0.6)`,
                  }}
                  animate={{ y: [0, -700], opacity: [0, 0.7, 0] }}
                  transition={{
                    duration: p.duration,
                    delay: p.delay,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                />
              ))}
            </div>
          </motion.div>

          {/* Modal card */}
          <motion.div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="launch-modal-title"
            className="relative w-full max-w-xl rounded-3xl"
            initial={{ y: 30, opacity: 0, scale: 0.96 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 20, opacity: 0, scale: 0.97 }}
            transition={{ type: "spring", stiffness: 220, damping: 24 }}
          >
            {/* Conic-rotating gradient border */}
            <motion.div
              className="absolute -inset-[2px] rounded-3xl pointer-events-none"
              animate={{
                background: [
                  "conic-gradient(from 0deg at 50% 50%, #00ff88, #00e5ff, #a78bfa, #ff6b6b, #ffd700, #00ff88)",
                  "conic-gradient(from 360deg at 50% 50%, #00ff88, #00e5ff, #a78bfa, #ff6b6b, #ffd700, #00ff88)",
                ],
              }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              style={{ filter: "blur(14px)", opacity: 0.6 }}
            />
            <motion.div
              className="absolute -inset-px rounded-3xl pointer-events-none"
              animate={{
                background: [
                  "conic-gradient(from 0deg, #00ff88, #00e5ff, #a78bfa, #00ff88)",
                  "conic-gradient(from 360deg, #00ff88, #00e5ff, #a78bfa, #00ff88)",
                ],
              }}
              transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
            />

            {/* Card body */}
            <div className="relative rounded-3xl bg-[#040b08] border border-white/5 overflow-hidden shadow-[0_30px_80px_-20px_rgba(0,255,136,0.35)]">
              {/* Cyber grid */}
              <div
                className="absolute inset-0 opacity-[0.06] pointer-events-none"
                style={{
                  backgroundImage:
                    "linear-gradient(rgba(0,255,136,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,136,0.6) 1px, transparent 1px)",
                  backgroundSize: "28px 28px",
                  maskImage:
                    "radial-gradient(ellipse at 50% 35%, black 30%, transparent 80%)",
                }}
              />

              {/* Top scanline */}
              <motion.div
                className="absolute top-0 left-0 right-0 h-px"
                animate={{
                  background: [
                    "linear-gradient(90deg, transparent, #00ff88, #00e5ff, transparent)",
                    "linear-gradient(90deg, transparent, #00e5ff, #a78bfa, transparent)",
                    "linear-gradient(90deg, transparent, #a78bfa, #00ff88, transparent)",
                  ],
                }}
                transition={{ duration: 3, repeat: Infinity }}
              />
              <motion.div
                className="absolute left-0 right-0 h-[2px] pointer-events-none"
                style={{
                  background: "linear-gradient(90deg, transparent, rgba(0,255,136,0.5), transparent)",
                  filter: "blur(1px)",
                }}
                animate={{ top: ["-2%", "102%"] }}
                transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              />

              {/* Glow blobs */}
              <motion.div
                className="absolute -top-32 -left-32 w-80 h-80 rounded-full pointer-events-none"
                style={{
                  background: "radial-gradient(circle, rgba(0,255,136,0.35), transparent 70%)",
                  filter: "blur(50px)",
                }}
                animate={{ scale: [1, 1.15, 1], opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 6, repeat: Infinity }}
              />
              <motion.div
                className="absolute -bottom-32 -right-32 w-80 h-80 rounded-full pointer-events-none"
                style={{
                  background: "radial-gradient(circle, rgba(0,229,255,0.28), transparent 70%)",
                  filter: "blur(50px)",
                }}
                animate={{ scale: [1.1, 1, 1.1], opacity: [0.6, 0.9, 0.6] }}
                transition={{ duration: 7, repeat: Infinity }}
              />
              <motion.div
                className="absolute top-1/3 right-1/4 w-40 h-40 rounded-full pointer-events-none"
                style={{
                  background: "radial-gradient(circle, rgba(167,139,250,0.25), transparent 70%)",
                  filter: "blur(40px)",
                }}
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ duration: 8, repeat: Infinity }}
              />

              {/* Close button */}
              <motion.button
                onClick={close}
                aria-label="Close"
                className="absolute top-4 right-4 z-30 w-10 h-10 rounded-full border border-white/10 bg-black/50 text-gray-300 hover:text-white hover:border-green-400/60 flex items-center justify-center backdrop-blur-md"
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
                transition={{ type: "spring", stiffness: 400, damping: 20 }}
              >
                <FiX size={16} />
              </motion.button>

              <div className="relative z-10 p-6 sm:p-7">
                {/* Pulse badge */}
                <motion.div
                  className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-green-500/40 bg-green-500/10 mb-3 backdrop-blur-sm"
                  initial={{ y: -8, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.15 }}
                >
                  <motion.span
                    className="w-2 h-2 rounded-full bg-green-400"
                    animate={{ scale: [1, 1.6, 1], opacity: [1, 0.5, 1], boxShadow: ["0 0 0px #00ff88", "0 0 12px #00ff88", "0 0 0px #00ff88"] }}
                    transition={{ duration: 1.4, repeat: Infinity }}
                  />
                  <FiStar size={12} className="text-green-300" />
                  <span className="text-[11px] font-bold tracking-[0.2em] text-green-300 uppercase">
                    v1.3.0 · Minor release
                  </span>
                </motion.div>

                <motion.h2
                  id="launch-modal-title"
                  className="text-3xl sm:text-4xl font-black leading-[1.05] text-white mb-3 tracking-tight"
                  initial={{ y: 14, opacity: 0, filter: "blur(8px)" }}
                  animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
                  transition={{ delay: 0.22, duration: 0.6 }}
                >
                  effGen{" "}
                  <span className="relative inline-block">
                    <motion.span
                      className="absolute inset-0 bg-clip-text text-transparent blur-md"
                      style={{
                        backgroundImage: "linear-gradient(120deg, #00ff88, #00e5ff, #a78bfa)",
                      }}
                      animate={{ opacity: [0.6, 1, 0.6] }}
                      transition={{ duration: 3, repeat: Infinity }}
                    >
                      v1.3.0
                    </motion.span>
                    <span
                      className="relative bg-clip-text text-transparent"
                      style={{
                        backgroundImage:
                          "linear-gradient(120deg, #00ff88, #00e5ff, #a78bfa)",
                      }}
                    >
                      v1.3.0
                    </span>
                  </span>{" "}
                  is here!
                </motion.h2>

                <motion.p
                  className="text-gray-400 text-[13px] leading-relaxed mb-4"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.32 }}
                >
                  Out on 1 October 2026. A run that stops making progress is asked for its answer,
                  every run says how it ended, and a tool that keeps failing ends the run as
                  tool_failed instead of being answered around. Nothing was removed or renamed.
                  Small models given a search tool now cost more per run; the changelog says why.
                </motion.p>

                {/* Highlights */}
                <motion.ul
                  className="space-y-1 mb-5"
                  initial="hidden"
                  animate="visible"
                  variants={{
                    visible: { transition: { staggerChildren: 0.08, delayChildren: 0.55 } },
                  }}
                >
                  {HIGHLIGHTS.map(({ icon: Icon, label, sub, accent }) => (
                    <motion.li
                      key={label}
                      className="group flex items-center gap-3 px-2.5 py-1.5 -mx-2.5 rounded-lg hover:bg-white/[0.03] transition-colors"
                      variants={{
                        hidden: { x: -16, opacity: 0 },
                        visible: { x: 0, opacity: 1 },
                      }}
                    >
                      <motion.span
                        className="relative w-8 h-8 rounded-lg flex items-center justify-center border flex-shrink-0"
                        style={{
                          background: `${accent}1a`,
                          borderColor: `${accent}50`,
                          ...accentTextStyle(accent),
                          boxShadow: `0 0 0px ${accent}00`,
                        }}
                        whileHover={{
                          scale: 1.1,
                          boxShadow: `0 0 18px ${accent}80`,
                          rotate: [0, -6, 6, 0],
                        }}
                        transition={{ duration: 0.4 }}
                      >
                        <Icon size={13} />
                      </motion.span>
                      <div className="min-w-0 flex-1">
                        <div className="text-[13px] font-semibold text-gray-100 truncate">
                          {label}
                        </div>
                        <div className="text-[11px] text-gray-600 dark:text-gray-400 truncate">{sub}</div>
                      </div>
                      <FiArrowRight
                        size={14}
                        className="text-gray-600 group-hover:text-green-400 group-hover:translate-x-1 transition-all flex-shrink-0"
                      />
                    </motion.li>
                  ))}
                </motion.ul>

                {/* Changes existing code can see. Say so here rather than
                    letting an upgrade discover them. */}
                <motion.div
                  className="mb-5 rounded-xl border border-amber-500/30 bg-amber-500/[0.06] px-3.5 py-3"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.9 }}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <FiShield size={12} className="text-amber-400 flex-shrink-0" />
                    <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-amber-300">
                      Ten changes existing code sees, among them
                    </span>
                  </div>
                  <ul className="space-y-0.5">
                    {BREAKING.map((item) => (
                      <li key={item} className="text-[11px] leading-relaxed text-gray-400">
                        {item}
                      </li>
                    ))}
                  </ul>
                </motion.div>

                {/* Install row */}
                <motion.div
                  className="relative flex items-center justify-between gap-3 px-3 py-2.5 mb-4 rounded-xl bg-black/70 border border-green-500/30 font-mono text-sm overflow-hidden"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 1.0 }}
                >
                  <motion.div
                    className="absolute inset-0 pointer-events-none"
                    animate={{
                      background: [
                        "linear-gradient(90deg, transparent, rgba(0,255,136,0.08), transparent)",
                        "linear-gradient(90deg, transparent, rgba(0,229,255,0.08), transparent)",
                      ],
                      backgroundPosition: ["-100% 0", "200% 0"],
                    }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    style={{ backgroundSize: "60% 100%", backgroundRepeat: "no-repeat" }}
                  />
                  <div className="relative flex items-center gap-2 min-w-0">
                    <FiTerminal className="text-green-400 flex-shrink-0" size={14} />
                    <span className="text-green-400 flex-shrink-0">$</span>
                    <span className="text-gray-100 truncate">pip install -U effgen</span>
                  </div>
                  <motion.button
                    onClick={copyInstall}
                    className="relative flex-shrink-0 px-3 py-1.5 rounded-md text-xs font-bold border border-green-500/40 text-green-300 hover:text-black hover:bg-green-400 hover:border-green-300 transition-all flex items-center gap-1.5"
                    whileTap={{ scale: 0.94 }}
                  >
                    {copied ? (
                      <>
                        <FiCheck size={12} /> Copied
                      </>
                    ) : (
                      <>
                        <FiCopy size={12} /> Copy
                      </>
                    )}
                  </motion.button>
                </motion.div>

                {/* CTAs */}
                <motion.div
                  className="flex flex-wrap items-center justify-center gap-2.5"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 1.1 }}
                >
                  <motion.a
                    href={resolveRoute("/docs/migration").href}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={close}
                    className="group relative inline-flex items-center gap-2 px-6 py-3 rounded-full font-black text-black text-sm cursor-pointer overflow-hidden"
                    style={{
                      background: "linear-gradient(135deg, #00ff88, #00c96e)",
                    }}
                    whileHover={{ scale: 1.04, y: -2 }}
                    whileTap={{ scale: 0.97 }}
                    animate={{
                      boxShadow: [
                        "0 0 24px rgba(0,255,136,0.35)",
                        "0 0 40px rgba(0,255,136,0.6)",
                        "0 0 24px rgba(0,255,136,0.35)",
                      ],
                    }}
                    transition={{
                      boxShadow: { duration: 2.4, repeat: Infinity },
                    }}
                  >
                    <motion.span
                      className="absolute inset-0 bg-white/20"
                      initial={{ x: "-120%", skewX: -20 }}
                      animate={{ x: "120%" }}
                      transition={{ duration: 2.2, repeat: Infinity, repeatDelay: 1.2 }}
                    />
                    <span className="relative">Read the migration notes</span>
                    <FiArrowRight className="relative group-hover:translate-x-1 transition-transform" />
                  </motion.a>

                  <a
                    href={resolveRoute("/changelog").href}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={close}
                    className="inline-flex items-center gap-2 px-5 py-3 rounded-full font-bold text-sm text-green-300 border border-green-500/40 bg-green-500/5 hover:bg-green-500/15 hover:border-green-400 hover:text-white transition-all"
                  >
                    Everything in 1.3.0
                  </a>

                  <button
                    onClick={close}
                    className="px-4 py-3 rounded-full text-sm text-gray-600 dark:text-gray-400 hover:text-gray-200 transition-colors"
                  >
                    Maybe later
                  </button>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
