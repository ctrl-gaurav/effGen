# effgen code — the coding agent

`effgen code` runs a plan → act → observe loop over a workspace: the model
proposes an approach, writes files, executes code in the configured sandbox,
reads the **real** output, and iterates until the task is done or the iteration
cap is reached. Every file change is shown as a unified diff before it touches
disk, every write and command passes a permission gate, and the files the agent
edits stay inside the workspace root.

```bash
effgen code                                     # interactive session
effgen code "write fib.py with fib(n) and print fib(10)" --auto-edit
effgen code -p "add a --retries flag to cli.py" --auto-edit --json | jq .files_written
cat pytest.log | effgen code -p "why did this fail?"
effgen code --undo                              # revert the last applied edit
effgen code "fix the failing test" --auto-edit --commit
effgen code --review                            # read-only review of what is uncommitted
effgen code --session-id refactor-42            # continue a saved session
```

On a terminal with no task it opens an interactive session. A task argument,
`-p`, piped stdin, `--json` or `--review` runs once and exits.

## Stdin

When stdin is not a terminal it is read **to EOF before the run starts**. With a
task already given the piped text becomes context in front of it; with no task
the piped text *is* the task:

```bash
cat pytest.log | effgen code -p "why did this fail?"   # log becomes context
echo "add a --retries flag to cli.py" | effgen code    # stdin is the task
```

Reading to EOF is what lets a producer that writes slowly — a build log, `tail
-f` — be folded in whole. The consequence is that a pipe which never closes
holds the run: an open stdin inherited from a supervisor, an agent harness or a
job-control shell keeps the command waiting even though a task was supplied.
That wait is announced rather than silent — after about two seconds the command
prints a line to stderr naming what it is waiting for:

```
Reading piped stdin as context before starting; the stream has not closed.
Close it (Ctrl-D) or re-run with < /dev/null to skip it.
```

With no task given the piped text is the task, and the note names that instead:

```
Reading the task from piped stdin; the stream has not closed.
Close it (Ctrl-D), or pass the task with -p to skip the read.
```

Redirect stdin from `/dev/null` when a caller has no context to pipe in:

```bash
effgen code -p "write fib.py" -w WS --auto-edit --json < /dev/null
```

The note goes to stderr, so stdout still carries only the answer or the JSON
document.

## The workspace

The agent reads, edits and runs in one directory:

1. `-w/--workspace DIR` if given (created if it does not exist),
2. otherwise `EFFGEN_WORKSPACE`,
3. otherwise the directory the command was started in.

A file path outside that root is refused with the path named, not silently
redirected. The deny-list for credential files and sensitive locations
(`~/.ssh`, `~/.aws`, `.env`, …) applies inside the workspace too. Shell commands
and executed code start in the workspace, and how far they can reach is decided
by the sandbox backend below.

Executed code runs in the sandbox effGen already ships — Docker when its daemon
is reachable, otherwise a subprocess sandbox that isolates the network and
confines writes to the workspace, leaving the rest of the filesystem readable
but read-only. `EFFGEN_SANDBOX_BACKEND=docker|subprocess` pins one. A write
outside the workspace fails with a read-only-filesystem error naming where
writes are allowed, so the model corrects the path instead of silently
scattering files across the machine. Only Docker also confines *reads*.

Confinement is not assumed — the sandbox proves it can hold before claiming it,
and reports what each run actually enforced. A host that allows the namespaces
but refuses the locked read-only mounts keeps network isolation and says writes
are not confined; a host without unprivileged user namespaces at all isolates
neither, and says that too. On those hosts a workspace under the system temp
directory is written but not readable from executed code, and the command
reports that before the run rather than letting it fail mid-loop. `effgen
doctor` names whichever of the three states applies.

`effgen doctor` reports all of this — the workspace it would use, the sandbox
backend, and whether git is present — with the fix for anything that is not
ready.

## Permission modes

Pick at most one; naming two is an error rather than a silent precedence rule.

| Mode | Writes | Sandboxed runs | Shell commands | Commit |
|---|---|---|---|---|
| `--plan` | proposed only | no | no | no |
| default, with a terminal | confirm each | confirm each | confirm each | confirm |
| `--auto-edit` | applied | applied | confirm each | confirm |
| `--yes` | applied | applied | applied | applied |

Without a terminal there is nobody to confirm, so the default becomes `--plan`:
the run reports what it *would* do and writes nothing. Opt in explicitly with
`--auto-edit` or `--yes` to let a scripted run change files. A run that
completed but withheld its changes for that reason exits `2`.

## Reviewing instead of changing

`--review` runs read-only. It is not a permission mode with a different prompt:
the run holds no tool that writes a file, runs code or runs a shell command. The
file tool is narrowed to its reading operations — its schema carries no `write`
— and `git` is the read-only surface (`status`, `log`, `diff`, `branch`, `show`,
`remote`), pinned to the workspace. The permission gate still stands behind
them, so a review is read-only by the tools it holds *and* by the mode it runs
in.

```bash
effgen code --review                                  # everything not committed
effgen code --review staged -p "is this safe to merge?"
effgen code --review HEAD~3
effgen code --review main...HEAD --json | jq -r .answer
effgen code -f src/app.py -f src/db.py -p "where can this raise?"
effgen code --review staged -f docs/api.md            # a diff and a file set
```

`TARGET` is `uncommitted` (the default — `git diff HEAD`, staged and unstaged
together), `staged` (`git diff --cached`), or any revision or range git accepts.
`-f/--file PATH` is repeatable and includes a file in full; it works with a
target or on its own, which is how a directory that is not a repository is
reviewed. Outside a repository with no `-f/--file`, the run exits `1` naming the
three ways to give it a subject rather than reviewing something else.

The change under review is **handed to the model as context**, because the only
route to a diff would be a shell and a read-only run has none. A diff over the
budget is truncated with the cut marked and the remainder counted, never
silently. With no task the question defaults to a review brief: correctness
risks first, one finding per bullet, each naming `file:line`.

The record reports `"read_only": true` and a `review` block naming the target;
`files_written` is always `[]`. `permission_mode` keeps reporting one of the four
documented modes (`plan` for a review), so nothing that reads it has to learn a
fifth value. A model that asks for a write anyway gets a refusal it can read, the
action log records it as `refused`, and the run still exits `0`.

A model that will not write a review — it keeps asking for the same file or the
same repository report instead — leaves the loop with nothing to hand back but
what it read. A review never returns that as the review: the run says so, exits
`1`, and keeps the text it did reach under `partial_output`.

`--review` cannot be combined with `--plan`, `--auto-edit`, `--yes`, `--commit`
or `--undo`: each asks for something a read-only run will not do.

In a session, `/review [TARGET]` runs one read-only turn — the tools and the mode
are swapped for that turn and put back afterwards, including when it fails, so
the next turn can write again.

## When the run stops without an answer

A model that keeps repeating the same call, or returns no final answer, leaves
the loop with nothing to report but what it already has: the last tool result, or
its own recovered text. That is progress, not an answer, so the run stops and
says so. The panel is titled **Stopped**, the answer panel carries what happened
and what to do, and the line under it names what the run was holding:

```
The model did not write an answer; what the run had is the last tool result,
after the model repeated the same call.
```

`--json` carries `outcome: "stopped"` and `stop_reason` (`loop_detected`,
`repeated_tool_result`, `null_final_from_model`, `max_iterations_partial`,
`max_iterations_exhausted`), with the text under `partial_output` and
`answer_source` naming the same exit. This matters most in a review, where
reading the same file twice is a natural second move: the file is never handed
back as the review, and a tool whose output is source material rather than a
computed result gets the model one tool-free turn to write the answer from what
it has first.

## Continuing a session

`effgen code --session-id ID` (or `--resume ID`) continues a stored session — the
same store as `effgen chat --session-id` and `effgen sessions list`. On a terminal
with no task it reopens the session; with a task it appends that turn to it.

```bash
effgen code --session-id refactor-42 -p "start by listing what needs to change" --plan
effgen code --session-id refactor-42            # later, in a new shell
```

What is restored, and what is deliberately not:

| | |
|---|---|
| the conversation | restored — the next turn can answer from what earlier turns said |
| files in context | restored, minus any path that no longer exists, which is named |
| files the session wrote | restored, so `/git commit` still knows its own paths |
| model and provider | restored when `-m` was not given, and announced |
| permission mode | restored only on a terminal and only when no permission flag was given — a stored `yes` is never restored into a piped run |
| the workspace | **not** adopted: `-w/--workspace` or the current directory always wins, and a stored workspace that differs is reported in one line |
| edits staged by `/plan` | **not** stored; the files may have changed underneath, so re-run `/plan` |
| the undo journal | nothing to restore — it is per workspace and already on disk, so `--undo` and `/undo` work across restarts |

In a session, `/session <id>` on an id that already has turns loads it: the
stored conversation replaces live memory (rather than being appended to it, which
would store both twice) and the count is reported. `/session` with no argument
prints the id and the command that continues it later. The stored user message is
the line that was typed, with the files that were sent recorded beside it, so a
resumed conversation does not replay file bodies as the user's words.

`/save` and `/load` are a separate, file-based store under the chat history
directory; `effgen sessions` does not list those. Use `/session` and
`--session-id` for a session you want to continue by id.

## Whether a model can do the work

Not every model can complete a coding turn: some receive no tool definitions at
all (their chat template renders none), some receive them and answer the question
directly, and some write the call out as prose. Each of those finishes with an
answer, `files_written: []`, and exit `0` — which reads like success.

Before the first call, `effgen code` says one line when the chosen model is a
poor fit:

```
transformers:google/gemma-2-2b-it for coding: its chat template renders no tool
definitions, so it never receives the coding tools and can only describe a
change. Use Qwen/Qwen2.5-7B-Instruct locally, or a keyed cloud model.
(measured 2026-08-06)
```

It never blocks the run, `-q/--quiet` suppresses it, and `--json` carries the same
fact under `coding_suitability`. The verdict is one of `suitable` (nothing is
printed), `limited`, `unsuitable`, or `unknown` when the catalog does not know the
id. Where a model has been measured on this framework the note carries the date;
past that, a loaded model that reports no tool calling, a catalog record without
tool support, and a small local checkpoint each have their own rule.

`effgen models info <id>` shows the same verdict as a `Coding` row (and a `coding`
object under `--json`); `effgen models list --json` and `browse --json` carry the
verdict string per model.

## Diffs, apply/reject and undo

Each proposed edit is rendered as a colorized unified diff **before** the write
is decided, so a change is visible in every mode — including `--plan` and
including piped output (the diff goes to stderr, keeping stdout for the result).

In the interactive session `/plan` stages edits without writing them, `/diff`
shows what is staged, `/apply [n]` writes them (all, or one by number),
`/reject [n]` discards them. A hunk that no longer applies because the file
changed underneath is reported by name; the remaining hunks still apply.

Every applied edit is journaled per workspace, so it can be reversed:

```bash
effgen code --undo             # reverse the last applied edit
effgen code --undo --undo-count 3
```

`/undo [n]` does the same inside a session. A restored file returns to its
previous content; a file the run created is removed.

## Repository awareness

In a git repository the branch, a short `git status` and a bounded file layout
(ignored files excluded) are read before the first model call and become part of
the agent's context, along with an `AGENTS.md` in the workspace if there is one.
Outside a repository the same inventory is built from the workspace itself.

`/git` surfaces the read-only view (`status`, `diff`, `log`, `branch`, `show`,
`remote`) and `/git staged` pulls the staged patch into context.

The single repository change a session can make is a commit of the files it
wrote, and only after an explicit confirmation:

```bash
effgen code "fix the failing test" --auto-edit --commit
```

The plan — repository, exact paths, message — is printed before the y/N. Only
those paths are staged; work you had staged yourself stays staged and out of the
commit. Push, force-push, tag, reset, checkout, rebase and stash are refused in
every mode, including when the model tries to reach them through the shell.

## The interactive session

| Command | Does |
|---|---|
| `/plan` | Propose a change without writing it |
| `/review [TARGET]` | One read-only turn over a diff — nothing is written or run |
| `/diff` `/apply [n]` `/reject [n]` | Review and decide the staged edits |
| `/undo [n]` | Reverse the last applied edit(s) |
| `/run <cmd>` `/test [args]` | Run a command or the test suite in the workspace |
| `/context` `/add <file>` `/drop <file>` | Manage the files the agent sees, with a size estimate |
| `/clear` `/reset` `/compact` | Reset live context, clear memory, or summarize a long session. `/compact` summarizes only — it runs with the tools detached, so it cannot touch the workspace |
| `/mode [ask\|auto-edit\|yes\|plan]` | Show or change the permission mode |
| `/model <id>` | Hot-swap the model, carrying the conversation |
| `/git [...]` | Repository status/diff/log, staged diff, confirmed commit |
| `/save` `/session [id]` `/load` | Save to a file, or name and resume a stored session |
| `/cost` `/trace` `/tools` `/doctor` | Session totals, the last turn's steps, the tool list, an environment check |
| `/help` `/exit` | The command table; leave the session |

Type `/` on its own for the menu; tab completes command names.

### What a turn shows while it runs

An interactive turn shows a status line naming the tool in flight, each proposed
edit's diff before it is written, and a tick per decided action. On a model whose
provider streams its tool calls — openai, gemini, groq, together, fireworks and
cerebras — the answer is also written to the screen as the model produces it,
rather than appearing in one block when the turn ends. The status line and the
answer take turns owning the terminal: the status line runs while the model is
thinking and dispatching tools, and hands over from the first word of the answer.

Everything else keeps the previous behavior and prints the answer once the turn
finishes: a model whose tool calls are not streamed (the local engines, among
others), and every non-interactive surface — piped output, `--json`, `--quiet`,
`--no-animation` and `NO_COLOR`.

## Scripting and non-TTY behavior

Piped or with `--json`, stdout carries only the result — the answer text, or one
JSON document — and everything a human reads goes to stderr:

```bash
echo "summarize what this module does" | effgen code
effgen code -p "add type hints to utils.py" --auto-edit --json | jq '.files_written, .cost_usd'
```

The JSON document carries the answer, `files_written`, the diffs, the full
action log (what was allowed, withheld, declined or refused, and why),
iterations, tool calls, tokens, cost and duration. Every proposed edit appears
in `diffs`; the ones that reached disk carry `"applied": true`, so
`--plan --json` reports the changes it would make without writing any of them.
`tool_calling` names the path the model's tool calls travelled on — `hybrid` and
`native` send the tool definitions to the provider's tool-calling API, `react`
reads the calls out of the model's text — and the report prints the same line
under its summary. `outcome` and `stop_reason` say whether the run answered,
stopped without an answer or failed outright, `answer_source` names the exit the
loop took when it stopped holding a tool result, `read_only` and `review`
describe a `--review` run, and `coding_suitability` carries the note about the
chosen model.

## When the model describes a call instead of making one

Some small models answer with the tool call written out as text:

```
<file_operations> {"operation": "write", "path": "greet.py", …} </file_operations>
```

Nothing is written and nothing runs, so that turn is reported as a failure
(`"reason": "written_tool_call"`, exit code 1) naming the tool whose call was
written out and what to do about it, rather than as an answer describing work
that did not happen. With `recover_lost_tool_calls` set on the agent, such a
call is read where its arguments only differ from strict JSON by raw line
breaks or Python-style quoting, and where nothing can be read the turn is sent
back once with a call required before the failure is reported. The remedy depends on the path the run used: a model that
was sent the tool definitions natively and still answered in prose needs
replacing with one that calls tools, while a run on the `react` text path can
ask for the native path instead. Larger instruct models and the current cloud
models complete the loop; `effgen models list` marks the models that advertise
tool calling.

An answer that *recaps* a call the run really made — asking the agent to report
the arguments it used, for instance — keeps its result: the turn is reported as
a failure only when the tool named in the block never ran, or when the answer is
nothing but the block. A call inside backticks or a fenced block is
documentation and is left alone either way.

## When the run stops at its iteration cap

A run that spends every step without writing a final answer has no answer to
report (`"reason": "max_iterations_partial"`, or `max_iterations_exhausted` when
nothing was recovered). The answer states what stopped it and what to do — raise
the cap, or run the task on a model that needs fewer steps — and whatever the
run had reached is reported separately, as `partial_output` in `--json` and
under a *Partial progress* heading in the terminal. It is tool output and
reasoning, never presented as a result.

Exit codes: `0` completed, `1` failed, `2` completed but changes were withheld
because there was no terminal to confirm on and no `--auto-edit`/`--yes` — which
includes a `--commit` that could not be confirmed.

`NO_COLOR`, `--no-animation` and a non-terminal stdout all render plain text
with no escape codes.

## First run

`effgen quickstart` (and its alias `effgen tutorial`) offers a coding step that
writes and runs a small program end to end inside `~/.effgen/quickstart-code`,
so the whole loop can be seen before it is pointed at real code. `--code` runs
that step without asking (needed alongside `--yes`, which otherwise skips it);
`--no-code` skips it.
