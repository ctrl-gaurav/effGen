"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiAlertTriangle, FiArrowRight, FiExternalLink, FiGitCommit, FiTag } from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import CodeSample from "@/components/ui/CodeSample";
import RouteLink from "@/components/ui/RouteLink";
import { siteData, version } from "@/components/siteData";
import {
  COMMITS_SINCE_0_3_2,
  COMMITS_SINCE_1_0_0,
  COMMITS_SINCE_1_0_1,
  COMMITS_SINCE_1_1_0,
  COMMITS_SINCE_1_2_0,
  PUBLIC_NAMES_1_0_1,
  PUBLIC_NAMES_1_1_0,
  PUBLIC_NAMES_1_2_0,
  RELEASE_DATE,
  RELEASE_DATE_1_0_1,
  RELEASE_DATE_1_1_0,
  RELEASE_DATE_1_2_0,
  RELEASE_DATE_1_3_0,
  breakingChanges,
  earlierReleases,
  newPublicNames,
  newPublicNames101,
  newPublicNames110,
  newPublicNames120,
  newPublicNames130,
  oneOneZeroGroups,
  oneThreeZeroGroups,
  oneTwoZeroGroups,
  oneZeroGroups,
  oneZeroOneGroups,
  visibleChanges101,
  visibleChanges110,
  visibleChanges120,
  visibleChanges130,
} from "./changelogData";
import type { ReactNode } from "react";
import type { VisibleChange } from "./changelogData";
import { accentTextStyle } from "@/components/accentText";

const CHANGELOG_URL = "https://github.com/ctrl-gaurav/effGen/blob/main/CHANGELOG.md";
const RELEASES_URL = "https://github.com/ctrl-gaurav/effGen/releases";

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

/* ── One themed group of changes ── */

function Group({ group, index }: { group: (typeof oneZeroGroups)[number]; index: number }) {
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <motion.section
      ref={ref}
      id={group.id}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5 }}
      className="scroll-mt-28"
      aria-labelledby={`${group.id}-heading`}
    >
      <div className="flex items-baseline gap-3 mb-2">
        <span
          className="text-xs font-mono font-bold tabular-nums"
          style={accentTextStyle(group.accent)}
        >
          {String(index + 1).padStart(2, "0")}
        </span>
        <h3
          id={`${group.id}-heading`}
          className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white"
        >
          {group.title}
        </h3>
      </div>
      <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">{group.lede}</p>

      <div className="space-y-6">
        {group.items.map((item) => (
          <article
            key={item.title}
            className="relative rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-6 shadow-sm dark:shadow-none"
          >
            <div
              className="absolute left-0 top-6 bottom-6 w-0.5 rounded-full"
              style={{ background: group.accent }}
            />
            <h4 className="text-base font-bold text-gray-900 dark:text-white mb-2 pl-3">
              {item.title}
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed pl-3">
              {item.body}
            </p>
            {item.code && (
              <div className="mt-4 pl-3">
                <CodeSample
                  code={item.code.source}
                  language={item.code.language ?? "python"}
                  accent={group.accent}
                  output={item.code.output}
                />
              </div>
            )}
          </article>
        ))}
      </div>
    </motion.section>
  );
}

/* ── The changes one release made that existing code sees ── */

function VisibleChanges({ changes }: { changes: VisibleChange[] }) {
  return (
    <ol className="space-y-6">
      {changes.map((change, i) => (
        <li
          key={change.title}
          className="rounded-xl bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-6"
        >
          <span className="text-[10px] font-mono uppercase tracking-widest text-orange-700 dark:text-orange-400">
            Change {String(i + 1).padStart(2, "0")}
          </span>
          <h3 className="mt-2 text-lg font-bold text-gray-900 dark:text-white">{change.title}</h3>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{change.why}</p>

          {change.migration && (
            <div className="mt-4">
              <CodeSample
                code={change.migration.source}
                language={change.migration.language}
                accent="#ff9500"
                output={change.migration.output}
              />
            </div>
          )}

          {change.note && (
            <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">{change.note}</p>
          )}
        </li>
      ))}
    </ol>
  );
}

/* ── One release's themed groups, with a jump list ── */

function ThemedGroups({
  groups,
  label,
  children,
}: {
  groups: (typeof oneZeroGroups)[number][];
  label: string;
  children: ReactNode;
}) {
  return (
    <section className="py-16 relative bg-gray-50 dark:bg-[#030f07]">
      {SECTION_DIVIDER}
      <div className="absolute inset-0 grid-pattern opacity-50" />
      <Container className="relative z-10">
        <div className="text-center mb-12">
          {children}
          <nav aria-label={label} className="mt-6 flex flex-wrap justify-center gap-2">
            {groups.map((group) => (
              <a
                key={group.id}
                href={`#${group.id}`}
                className="px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors"
                style={{
                  ...accentTextStyle(group.accent),
                  borderColor: `${group.accent}40`,
                  backgroundColor: `${group.accent}0d`,
                }}
              >
                {group.title}
              </a>
            ))}
          </nav>
        </div>

        <div className="max-w-4xl mx-auto space-y-16">
          {groups.map((group, index) => (
            <Group key={group.id} group={group} index={index} />
          ))}
        </div>
      </Container>
    </section>
  );
}

export default function ChangelogView() {
  const [heroRef, heroInView] = useInView({ triggerOnce: true, threshold: 0.05 });

  const headline = [
    { value: version, label: "Version", accent: "#00ff88", icon: FiTag },
    { value: RELEASE_DATE_1_3_0.replace(" 2026", ""), label: "Released", accent: "#00e5ff", icon: FiTag },
    { value: String(COMMITS_SINCE_1_2_0), label: "Commits since 1.2.0", accent: "#a78bfa", icon: FiGitCommit },
    { value: String(visibleChanges130.length), label: "Changes existing code sees", accent: "#ff9500", icon: FiAlertTriangle },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        {/* Hero */}
        <section className="relative pt-32 pb-16 overflow-hidden">
          <div className="absolute inset-0 grid-pattern" />
          <Container className="relative z-10">
            <motion.div
              ref={heroRef}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="max-w-4xl mx-auto text-center"
            >
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8">
                <FiTag size={14} />
                Changelog
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                <span className="gradient-text">effGen {version}</span> — how a run ends
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl mx-auto">
                Released {RELEASE_DATE_1_3_0}, {COMMITS_SINCE_1_2_0} commits after 1.2.0. A run that
                stops making progress is asked for its answer instead of going round to its iteration
                cap, and every run says how it ended:{" "}
                <code className="font-mono">response.termination</code> is done, not_possible, stuck,
                tool_failed or error. A tool that keeps failing on its own side ends the run as
                tool_failed. A tool call the model wrote in a broken or unexpected shape is read and
                run by default. A model you serve yourself, or run on a local engine, is measured once
                for what it does with a tool, and <code className="font-mono">tool_calling_mode=&quot;auto&quot;</code>{" "}
                follows what was measured. One agent can serve many overlapping conversations without
                mixing them. The public surface grew from {PUBLIC_NAMES_1_2_0} names to{" "}
                {siteData.public_names}, and nothing was removed or renamed.
              </p>
            </motion.div>
          </Container>
        </section>

        {/* Headline figures */}
        <section className="py-6 relative">
          {SECTION_DIVIDER}
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

        {/* The ten changes in 1.3.0 that existing code sees */}
        <section id="changed" className="py-16 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-6 lg:p-10">
              <div className="flex items-center gap-3 mb-2">
                <FiAlertTriangle className="text-orange-500 dark:text-orange-400" size={22} />
                <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white">
                  Ten things existing code sees
                </h2>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">
                Nothing was removed or renamed. Most of these change how a run ends or how a tool
                call is read rather than what your code has to call; where a setting restores the
                1.2.0 behaviour, the change shows it.
              </p>

              <VisibleChanges changes={visibleChanges130} />

              <div className="mt-8">
                <RouteLink
                  to="/docs/migration"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  The migration guide
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>

        {/* What 1.3.0 added, grouped by theme */}
        <ThemedGroups groups={oneThreeZeroGroups} label="1.3.0 changes by theme">
          <h2 className="text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-4">
            What 1.3.0 <span className="gradient-text">added</span>
          </h2>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            {oneThreeZeroGroups.length} themes, the last two of them where it falls short and what is
            still open. Jump to one:
          </p>
        </ThemedGroups>

        {/* The names 1.3.0 added */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                Two new names on the <code className="font-mono gradient-text">effgen</code> package
                in 1.3.0
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                The top-level surface grew from {PUBLIC_NAMES_1_2_0} names to{" "}
                {PUBLIC_NAMES_1_2_0 + newPublicNames130.length}. Existing types gained members too:{" "}
                <code className="font-mono">AgentResponse.termination</code>,{" "}
                <code className="font-mono">AgentConfig.capability_probe</code>,{" "}
                <code className="font-mono">BaseModel.capability_key()</code> and{" "}
                <code className="font-mono">.forwards_reasoning_effort()</code>, and{" "}
                <code className="font-mono">TERMINATIONS</code> in{" "}
                <code className="font-mono">effgen.core.agent</code>.
              </p>
              <ul className="flex flex-wrap gap-2">
                {newPublicNames130.map((name) => (
                  <li
                    key={name}
                    className="px-3 py-1.5 rounded-lg text-xs font-mono bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300"
                  >
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          </Container>
        </section>

        {/* 1.2.0 */}
        <section id="v1-2-0" className="pt-20 pb-4 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto text-center">
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
                <FiTag size={14} />
                <span className="font-mono">v1.2.0</span>
              </span>
              <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                <span className="gradient-text">effGen 1.2.0</span> — what a run costs
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl mx-auto">
                Released {RELEASE_DATE_1_2_0}, {COMMITS_SINCE_1_1_0} commits after 1.1.0. Every run
                now keeps a ledger of its model calls, tool calls, tokens and cost, and of where its
                time went: waiting on the model, on tools, on the caller, on child runs, or inside the
                framework. A tool result the model writes itself is never taken as the answer. A
                provider&apos;s prompt cache is kept warm, and its hits are read and priced where the
                provider reports them. A request carries less of the framework&apos;s own text, a
                model you serve yourself reads as unpriced rather than free, and a spent cap no longer
                refuses it. And <code className="font-mono">effgen bench</code> measures an agent on
                your own tasks. The public surface grew from {PUBLIC_NAMES_1_1_0} names to{" "}
                {PUBLIC_NAMES_1_1_0 + newPublicNames120.length}, and nothing was removed or renamed.
              </p>
            </div>
          </Container>
        </section>

        {/* The fourteen changes in 1.2.0 that existing code sees */}
        <section id="changed-120" className="py-16 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-6 lg:p-10">
              <div className="flex items-center gap-3 mb-2">
                <FiAlertTriangle className="text-orange-500 dark:text-orange-400" size={22} />
                <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white">
                  Fourteen things changed in 1.2.0
                </h2>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">
                Nothing was removed or renamed. Most of these change what a run reports or what a
                request carries rather than what your code has to call; where code does change, the
                change carries it.
              </p>

              <VisibleChanges changes={visibleChanges120} />

              <div className="mt-8">
                <RouteLink
                  to="/docs/migration"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  The migration guide
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>

        {/* What 1.2.0 added, grouped by theme */}
        <ThemedGroups groups={oneTwoZeroGroups} label="1.2.0 changes by theme">
          <h2 className="text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-4">
            What 1.2.0 <span className="gradient-text">added</span>
          </h2>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            {oneTwoZeroGroups.length} themes, the last of them what is still open. Jump to one:
          </p>
        </ThemedGroups>

        {/* The name 1.2.0 added */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                One new name on the <code className="font-mono gradient-text">effgen</code> package in
                1.2.0
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                The top-level surface grew from {PUBLIC_NAMES_1_1_0} names to{" "}
                {PUBLIC_NAMES_1_1_0 + newPublicNames120.length}. Existing types gained members too:{" "}
                <code className="font-mono">AgentResponse.ledger</code>,{" "}
                <code className="font-mono">Agent.last_stream_ledger</code>,{" "}
                <code className="font-mono">Checkpoint.ledger</code>,{" "}
                <code className="font-mono">AgentConfig.answer_style</code>,{" "}
                <code className="font-mono">.max_turns_without_progress</code> and{" "}
                <code className="font-mono">.recover_lost_tool_calls</code>,{" "}
                <code className="font-mono">BaseModel.prompt_cache_policy()</code>,{" "}
                <code className="font-mono">.prompt_detail()</code> and{" "}
                <code className="font-mono">.supports_stop_with_tools()</code>, and{" "}
                <code className="font-mono">SQLiteCostStore.flush()</code>.
              </p>
              <ul className="flex flex-wrap gap-2">
                {newPublicNames120.map((name) => (
                  <li
                    key={name}
                    className="px-3 py-1.5 rounded-lg text-xs font-mono bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300"
                  >
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          </Container>
        </section>

        {/* 1.1.0 */}
        <section id="v1-1-0" className="pt-20 pb-4 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto text-center">
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
                <FiTag size={14} />
                <span className="font-mono">v1.1.0</span>
              </span>
              <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                <span className="gradient-text">effGen 1.1.0</span> — a run keeps its conversation
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl mx-auto">
                Released {RELEASE_DATE_1_1_0}, {COMMITS_SINCE_1_0_1} commits after 1.0.1. A run now
                keeps its conversation as typed steps instead of one growing string. A finished run
                hands back what it did, and the command line, the run card, the debug inspector and
                the dashboard all render the same steps. A run is bounded by the prompt tokens it may
                send and gives up its oldest material first instead of failing at the provider. A
                saved run resumes where it stopped instead of restarting the task. And there is one
                agent loop rather than three, so a streamed run sends what a blocking one sends. The
                public surface grew from {PUBLIC_NAMES_1_0_1} names to {PUBLIC_NAMES_1_1_0}, and
                nothing was removed or renamed.
              </p>
            </div>
          </Container>
        </section>

        {/* The eleven changes in 1.1.0 that existing code sees */}
        <section id="changed-110" className="py-16 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-6 lg:p-10">
              <div className="flex items-center gap-3 mb-2">
                <FiAlertTriangle className="text-orange-500 dark:text-orange-400" size={22} />
                <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white">
                  Eleven things changed in 1.1.0
                </h2>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">
                Nothing was removed or renamed. Most of these change what a run reports or what a
                request carries rather than what your code has to call; where code does change, the
                change carries it.
              </p>

              <VisibleChanges changes={visibleChanges110} />

              <div className="mt-8">
                <RouteLink
                  to="/docs/migration"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  The migration guide
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>

        {/* What 1.1.0 added, grouped by theme */}
        <ThemedGroups groups={oneOneZeroGroups} label="1.1.0 changes by theme">
          <h2 className="text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-4">
            What 1.1.0 <span className="gradient-text">added</span>
          </h2>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            {oneOneZeroGroups.length} themes, the last of them what is still open. Jump to one:
          </p>
        </ThemedGroups>

        {/* The names 1.1.0 added */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                {newPublicNames110.length} new names on the{" "}
                <code className="font-mono gradient-text">effgen</code> package in 1.1.0
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                The top-level surface grew from {PUBLIC_NAMES_1_0_1} names to{" "}
                {PUBLIC_NAMES_1_0_1 + newPublicNames110.length}. Existing types gained members too:{" "}
                <code className="font-mono">AgentResponse.thread</code>,{" "}
                <code className="font-mono">Checkpoint.to_thread()</code>,{" "}
                <code className="font-mono">Session.last_thread()</code>,{" "}
                <code className="font-mono">AgentConfig.prompt_protocol</code>,{" "}
                <code className="font-mono">.context_budget</code> and{" "}
                <code className="font-mono">.compaction</code>, and{" "}
                <code className="font-mono">BaseModel.supports_message_protocol()</code>.
              </p>
              <ul className="flex flex-wrap gap-2">
                {newPublicNames110.map((name) => (
                  <li
                    key={name}
                    className="px-3 py-1.5 rounded-lg text-xs font-mono bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300"
                  >
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          </Container>
        </section>

        {/* 1.0.1 */}
        <section id="v1-0-1" className="pt-20 pb-4 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto text-center">
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
                <FiTag size={14} />
                <span className="font-mono">v1.0.1</span>
              </span>
              <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                <span className="gradient-text">effGen 1.0.1</span> — a run that stops says so
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl mx-auto">
                Released {RELEASE_DATE_1_0_1}, {COMMITS_SINCE_1_0_0} commits after 1.0.0. It fixed how
                the framework reports what a run did, what it puts in a prompt, and what its own
                bookkeeping costs: a run that stops without an answer says so instead of handing back
                its working notes, citation markers are opt-in, the loop guards no longer stop a run
                that is still making progress, every tool-calling path tells the model what the tools
                are for, the budget check reads an index instead of the whole spend ledger, and the
                Groq default points at a model Groq still serves. The public surface grew from 223
                names to {PUBLIC_NAMES_1_0_1}, and nothing was removed or renamed.
              </p>
            </div>
          </Container>
        </section>

        {/* The four changes in 1.0.1 that existing code sees */}
        <section id="changed-101" className="py-16 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-6 lg:p-10">
              <div className="flex items-center gap-3 mb-2">
                <FiAlertTriangle className="text-orange-500 dark:text-orange-400" size={22} />
                <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white">
                  Four things changed in 1.0.1
                </h2>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">
                One of them changes what <code className="font-mono">success</code> means for a run
                that stopped part way. Each carries the code that adapts to it.
              </p>

              <VisibleChanges changes={visibleChanges101} />
            </div>
          </Container>
        </section>

        {/* What 1.0.1 added and fixed, grouped by theme */}
        <ThemedGroups groups={oneZeroOneGroups} label="1.0.1 changes by theme">
          <h2 className="text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-4">
            What 1.0.1 <span className="gradient-text">added and fixed</span>
          </h2>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            {oneZeroOneGroups.length} themes. The two new names on the package are{" "}
            {newPublicNames101.map((name, i) => (
              <span key={name}>
                {i > 0 && " and "}
                <code className="font-mono">{name}</code>
              </span>
            ))}
            . Jump to one:
          </p>
        </ThemedGroups>

        {/* 1.0.0 */}
        <section id="v1-0-0" className="pt-20 pb-4 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto text-center">
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
                <FiTag size={14} />
                <span className="font-mono">v1.0.0</span>
              </span>
              <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                <span className="gradient-text">effGen 1.0.0</span> — the first stable release
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl mx-auto">
                Released {RELEASE_DATE}, {COMMITS_SINCE_0_3_2} commits after 0.3.2. The theme running
                through it is control over where a model runs and visibility into what a run did:
                drive a server you already operate, read back the calls a run made, wrap the agent
                loop in middleware, hand one agent many conversations, choose how history is
                compacted, and resume a workflow that died half way through. The public surface grew
                from 204 names to 223, and nothing was removed or renamed.
              </p>
            </div>
          </Container>
        </section>

        {/* The three breaking changes */}
        <section id="breaking" className="py-16 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-6 lg:p-10">
              <div className="flex items-center gap-3 mb-2">
                <FiAlertTriangle className="text-orange-500 dark:text-orange-400" size={22} />
                <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white">
                  Three things changed in 1.0.0
                </h2>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">
                Everything else in 1.0.0 is additive. Each of the three carries the line that
                migrates it, and what that line printed when it was run.
              </p>

              <ol className="space-y-6">
                {breakingChanges.map((change, i) => (
                  <li
                    key={change.title}
                    className="rounded-xl bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-6"
                  >
                    <span className="text-[10px] font-mono uppercase tracking-widest text-orange-700 dark:text-orange-400">
                      Breaking {String(i + 1).padStart(2, "0")}
                    </span>
                    <h3 className="mt-2 text-lg font-bold text-gray-900 dark:text-white">
                      {change.title}
                    </h3>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      {change.why}
                    </p>

                    <div className="mt-4">
                      <CodeSample
                        code={change.migration.source}
                        language={change.migration.language}
                        accent="#ff9500"
                        output={change.migration.output}
                      />
                    </div>

                    {change.note && (
                      <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                        {change.note}
                      </p>
                    )}
                  </li>
                ))}
              </ol>

              <div className="mt-8">
                <RouteLink
                  to="/docs/migration"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  The migration guide
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>

        {/* What 1.0.0 added, grouped the way the changelog groups it */}
        <section className="py-16 relative bg-gray-50 dark:bg-[#030f07]">
          {SECTION_DIVIDER}
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <Container className="relative z-10">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-4">
                What 1.0.0 <span className="gradient-text">added</span>
              </h2>
              <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
                {oneZeroGroups.length} themes, in the order the changelog files them. Jump to one:
              </p>
              <nav aria-label="Changes by theme" className="mt-6 flex flex-wrap justify-center gap-2">
                {oneZeroGroups.map((group) => (
                  <a
                    key={group.id}
                    href={`#${group.id}`}
                    className="px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors"
                    style={{
                      ...accentTextStyle(group.accent),
                      borderColor: `${group.accent}40`,
                      backgroundColor: `${group.accent}0d`,
                    }}
                  >
                    {group.title}
                  </a>
                ))}
              </nav>
            </div>

            <div className="max-w-4xl mx-auto space-y-16">
              {oneZeroGroups.map((group, index) => (
                <Group key={group.id} group={group} index={index} />
              ))}
            </div>
          </Container>
        </section>

        {/* The 19 names 1.0.0 added */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                {newPublicNames.length} new names on the{" "}
                <code className="font-mono gradient-text">effgen</code> package in 1.0.0
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                The top-level surface grew from 204 names to 223. Nothing was
                removed and nothing was renamed. <code className="font-mono">BaseModel</code> also
                gained <code className="font-mono">build_assistant_message</code> and{" "}
                <code className="font-mono">build_tool_result_message</code>, and{" "}
                <code className="font-mono">SandboxResult</code> gained{" "}
                <code className="font-mono">credential_reads_masked</code> and{" "}
                <code className="font-mono">process_table_isolated</code>.
              </p>
              <ul className="flex flex-wrap gap-2">
                {newPublicNames.map((name) => (
                  <li
                    key={name}
                    className="px-3 py-1.5 rounded-lg text-xs font-mono bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300"
                  >
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          </Container>
        </section>

        {/* Earlier releases */}
        <section id="earlier" className="py-16 relative bg-gray-50 dark:bg-[#030f07] scroll-mt-24">
          {SECTION_DIVIDER}
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                Earlier releases
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8">
                Every release before 1.0.0, with the date and the headline the changelog gives
                it. The full entry for each — every addition, change and fix — is in{" "}
                <a
                  href={CHANGELOG_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-700 dark:text-green-400 font-semibold inline-flex items-center gap-1"
                >
                  CHANGELOG.md
                  <FiExternalLink size={12} />
                </a>
                .
              </p>

              <ol className="relative border-l border-gray-200 dark:border-gray-800 ml-3">
                {earlierReleases.map((release) => (
                  <li key={release.version} className="relative pl-8 pb-8 last:pb-0">
                    <span
                      className="absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full bg-gray-300 dark:bg-gray-700"
                      aria-hidden="true"
                    />
                    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white font-mono">
                        v{release.version}
                      </h3>
                      <span className="text-xs text-gray-600 dark:text-gray-400">{release.date}</span>
                      <span className="text-sm font-semibold text-green-700 dark:text-green-400">
                        {release.title}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      {release.summary}
                    </p>
                  </li>
                ))}
              </ol>
            </div>
          </Container>
        </section>

        {/* Where to go next */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-3xl mx-auto text-center">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-4">
                Upgrade
              </h2>
              <div className="max-w-md mx-auto text-left mb-8">
                <CodeSample code="pip install -U effgen" language="bash" />
              </div>
              <div className="flex flex-wrap gap-4 justify-center">
                <a
                  href={RELEASES_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-black"
                  style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)" }}
                >
                  Releases on GitHub
                  <FiExternalLink size={14} />
                </a>
                <RouteLink
                  to="/docs/migration"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-green-700 dark:text-green-300 border border-green-500/30 bg-green-500/5 hover:bg-green-500/10 transition-colors"
                >
                  The migration guide
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
