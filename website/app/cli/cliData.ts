// The grouping and the prose `/cli` renders.
//
// The 29 commands, their one-line summaries, the global flags, the theme names
// and the completion shells all come from `data/effgen.json`, which
// `scripts/gen_site_data.py` parses out of the real `--help`. Nothing here
// restates any of them.
//
// What is written here is the one thing the framework does not state: which
// commands belong together. `assertGroupsCoverEveryCommand` below fails the
// build if that grouping ever stops matching the command list, so a command
// added to effGen cannot quietly go missing from this page.

import { siteData } from "@/components/siteData";

export const DOCS_CLI_URL =
  "https://github.com/ctrl-gaurav/effGen/blob/main/docs/dx/cli.md";
export const DOCS_TOP_URL =
  "https://github.com/ctrl-gaurav/effGen/blob/main/docs/cli/top.md";

export interface CommandGroup {
  id: string;
  title: string;
  lede: string;
  accent: string;
  commands: string[];
}

export const commandGroups: CommandGroup[] = [
  {
    id: "run",
    title: "Give an agent something to do",
    lede: "One task, a conversation, a coding session, or a file of tasks.",
    accent: "#00ff88",
    commands: ["run", "chat", "code", "debug", "batch", "examples"],
  },
  {
    id: "history",
    title: "Pick work back up",
    lede:
      "A run and a conversation are both durable, and both are addressable by id " +
      "afterwards — from the command line, from a script and from the server.",
    accent: "#00e5ff",
    commands: ["workflow", "resume", "sessions", "runs"],
  },
  {
    id: "catalog",
    title: "See what is available",
    lede: "The model catalog, the tool registry, the presets and the prompt library.",
    accent: "#a78bfa",
    commands: ["models", "tools", "presets", "prompts"],
  },
  {
    id: "measure",
    title: "Measure before you choose",
    lede:
      "A scored suite, an ad-hoc race, a graded evaluation, or a server under load. " +
      "Each writes a shareable HTML report.",
    accent: "#ffd700",
    commands: ["eval", "compare", "battle", "bench", "loadtest"],
  },
  {
    id: "serve",
    title: "Serve it and watch it",
    lede:
      "An OpenAI-compatible server, and a live terminal view of what is going " +
      "through it.",
    accent: "#ff6b6b",
    commands: ["serve", "top", "monitor", "health"],
  },
  {
    id: "spend",
    title: "Know what it cost",
    lede: "The local spend ledger, budgets, and a saved result rendered later.",
    accent: "#ff9500",
    commands: ["cost", "report"],
  },
  {
    id: "setup",
    title: "Set a machine up",
    lede:
      "Scaffold a project, check the environment, and read the configuration " +
      "effGen resolved.",
    accent: "#5eead4",
    commands: ["quickstart", "tutorial", "config", "doctor", "create-plugin"],
  },
];

/**
 * Fail loudly if the grouping above and the derived command list disagree.
 *
 * The command list is generated; this grouping is not. Without this check a
 * command added to effGen would simply be absent from the page, and nothing
 * would say so.
 */
function assertGroupsCoverEveryCommand(): void {
  const grouped = commandGroups.flatMap((group) => group.commands);
  const derived = siteData.cli.commands.map((command) => command.name);

  const missing = derived.filter((name) => !grouped.includes(name));
  const unknown = grouped.filter((name) => !derived.includes(name));
  const duplicated = grouped.filter((name, i) => grouped.indexOf(name) !== i);

  if (missing.length || unknown.length || duplicated.length) {
    throw new Error(
      "app/cli/cliData.ts no longer matches the command list in data/effgen.json. " +
        [
          missing.length ? `Not in any group: ${missing.join(", ")}.` : "",
          unknown.length ? `Grouped but not a command: ${unknown.join(", ")}.` : "",
          duplicated.length ? `In two groups: ${duplicated.join(", ")}.` : "",
        ]
          .filter(Boolean)
          .join(" "),
    );
  }
}

assertGroupsCoverEveryCommand();

/** The command's own one-line summary, from `effgen --help`. */
export function summaryOf(name: string): string {
  return siteData.cli.commands.find((command) => command.name === name)?.summary ?? "";
}

/** Sub-commands the command declares, from its own `--help`. */
export function subcommandsOf(name: string): string[] {
  return siteData.cli.subcommands[name] ?? [];
}

/* ── The named themes ── */

export interface ThemeNote {
  name: string;
  slug: string;
  ground: "dark" | "light";
  purpose: string;
  accent: string;
}

// What each theme is for, from the `--theme` help text and the palette module
// that defines them. The frames beside these are the real output of the same
// command under each one, with its colour codes read rather than discarded.
export const themeNotes: ThemeNote[] = [
  {
    name: "default",
    slug: "cli-theme-default",
    ground: "dark",
    purpose:
      "The unchanged look: cyan headings and table headers, magenta for a tool or " +
      "a title, green for a cost and a success, dim for anything secondary.",
    accent: "#00e5ff",
  },
  {
    name: "high-contrast",
    slug: "cli-theme-high-contrast",
    ground: "dark",
    purpose:
      "Aimed at low-vision readers. Every role moves to its bright pair and picks " +
      "up bold, and what the default theme dims becomes plain white instead.",
    accent: "#00ff88",
  },
  {
    name: "monochrome",
    slug: "cli-theme-monochrome",
    ground: "dark",
    purpose:
      "Keeps the structure without hue. Headings underline, an error reverses, a " +
      "tool and a model italicise — so the same distinctions survive on a terminal " +
      "with no colour and in a log file.",
    accent: "#a78bfa",
  },
  {
    name: "light",
    slug: "cli-theme-light",
    ground: "light",
    purpose:
      "For a light terminal, where cyan and yellow wash out. Blue replaces cyan, a " +
      "darker orange replaces yellow, and the muted role becomes a real grey rather " +
      "than dimmed white.",
    accent: "#2563eb",
  },
];

/* ── Where --json is guaranteed ── */

export interface JsonNote {
  command: string;
  note: string;
}

// Each of these was run for this page and its stdout parsed as one document.
export const jsonExamples: JsonNote[] = [
  {
    command:
      'effgen code -p "what does this package export?" --plan --json ' +
      "-m gemini:gemini-3.1-flash-lite",
    note: "The answer, the files written, every proposed diff, and the full action log.",
  },
  {
    command:
      'effgen battle "Explain a B-tree in two sentences." ' +
      "-m openai:gpt-5-nano,gemini:gemini-3.1-flash-lite --json",
    note: "Every contender's full answer, its cost and latency, the tally and the verdict.",
  },
  {
    command: "effgen top --json",
    note: "One snapshot: activity, traffic, per-model, spend and GPU.",
  },
  {
    command: "effgen runs list --json",
    note: "The run history, newest first, with the ids `runs show` takes.",
  },
];
