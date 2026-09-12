# Sessions, Checkpoints & Background Tasks

This guide covers the durability surfaces effGen ships for long-lived work:
persistent **sessions** (multi-turn memory across processes), **checkpoints**
(resume a long task where it left off), and **background tasks** (run agent work
off the main thread).

## Sessions — persistent multi-turn memory

A session stores a conversation so a later process can pick it up and the model
*remembers* earlier turns.

```python
from effgen import Agent, AgentConfig

# Same session_id in two separate runs → the second remembers the first.
a1 = Agent(AgentConfig(model="gpt-5-nano", provider="openai"), session_id="user-123")
a1.run("My dog is named Pixel.")

a2 = Agent(AgentConfig(model="gpt-5-nano", provider="openai"), session_id="user-123")
print(a2.run("What is my dog's name?").output)   # -> "Pixel"
```

From a preset, pass `session_id=` the same way:

```python
from effgen import create_agent
agent = create_agent("math", "gpt-5-nano", session_id="user-123")
```

On the CLI, every command that runs an agent accepts `--session-id`:

```bash
effgen run --session-id user-123 "My favorite color is teal."
effgen run --session-id user-123 "What is my favorite color?"   # remembers
```

### Managing sessions

```bash
effgen sessions list                 # id, messages, model, cost, updated
effgen sessions show <id>            # read the conversation turn by turn
effgen sessions show <id> --last 4   # tail a long thread
effgen sessions browse               # list, then pick one to read
effgen sessions export <id>          # JSON (default) — a documented, reloadable format
effgen sessions export <id> --format text
effgen sessions delete <id>          # exit 0 if removed, 1 if not found
effgen sessions cleanup --days 30    # delete sessions not updated in N days
```

`list` narrows with `--search <text>` (matches ids, agent names and message
content), `--model`, `--since`/`--until` (YYYY-MM-DD) and `--limit`. `list`,
`show` and `browse` all accept `--json`; `browse` prints the list and stops when
stdin is not a terminal, so it stays usable in a pipeline. Session files that
cannot be parsed are named in the output rather than dropped from the count.

Export writes to stdout byte-for-byte, so message content containing square
brackets (`[bold]`, `[REDACTED]`, `[1]`) survives intact and the JSON form
round-trips.

Each turn is stamped with the model, token counts, cost and latency it was
answered with, which is what `sessions list` and `sessions show` report.

Exit codes follow the CLI convention: `0` success, `1` user error (e.g. a
missing id), `2` a corrupt file (see below).

### Run history

Every agent run is also recorded in a run history store, keyed by the same
`run_id` its trace spans carry:

```bash
effgen runs list                     # run id, time, model, task, cost, status
effgen runs list --status failed     # only the runs that failed
effgen runs list --search refund --since 2026-07-01
effgen runs show <run_id>            # task, answer, tokens, cost, error, session
effgen runs cleanup --days 30        # drop history older than N days
```

Records are appended to one JSONL file per day under `$EFFGEN_HOME/runs`
(default `~/.effgen/runs`), so runs from the CLI, a script and the server share
one history and survive a restart. A run that belongs to a session records its
session id, so `runs show` links back to the conversation it came from.

| Variable | Effect |
|---|---|
| `EFFGEN_RUN_HISTORY_DIR` | Exact directory for run history files. |
| `EFFGEN_RUN_HISTORY` | `0` keeps history in memory only (nothing written to disk). |
| `EFFGEN_RUN_HISTORY_MAX_DAYS` | Retention in days (default 30); older files are removed on write. |

History writes are best-effort: if the store cannot be written, the run itself
still succeeds. Task and answer text is stored truncated to a 500-character
preview.

The dashboard's History view reads the same store, with drill-in from a run to
its task, answer, tokens and cost.

### Where sessions are stored

By default sessions live in `~/.effgen/sessions/<id>.json`. Override the
location with environment variables (resolved at call time):

| Variable | Effect |
|---|---|
| `EFFGEN_SESSIONS_DIR` | Exact directory for session files. |
| `EFFGEN_HOME` | Base effGen state dir; sessions go in `$EFFGEN_HOME/sessions`, run history in `$EFFGEN_HOME/runs`, and the cost and rate-limit databases alongside them. |

Saves are **atomic** (written to a temp file then renamed) so a crash mid-write
can never leave a truncated session that fails to load later.

To cap unbounded growth, schedule `effgen sessions cleanup --days N` (e.g. from
cron) or call `SessionManager(...).cleanup(older_than_days=N)`.

### What a stored turn keeps

Each assistant turn stores the conversation its run had, so a later turn can
continue from the earlier run's structure rather than from a reading of its
text. `Session.last_thread()` reads it:

```python
thread = session.last_thread()
print(len(thread.steps), [a.tool for a in thread.actions()])
```

Only the **latest** turn keeps the whole of it. As a new turn arrives, each
earlier turn's record is reduced to its shape — the format version, the number
of steps and the kinds in order — because that is the only part of an older
turn anything reads, and keeping every one of them made the file grow with the
square of the conversation rather than with its length. What the turn said
stays where it always was, in its own `content`.

```python
from effgen.core.session import Session

Session(session_id="long-one", keep_thread_history=True)   # keep every one
```

Reading an older turn's record hands back an empty `AgentThread`: there is
nothing to rebuild from a shape, and saying so is better than raising on a
document that is exactly what it says it is.

### One agent, many conversations

`session_id=` on the constructor binds a conversation to the agent for its
whole life. A server handling many users wants the opposite: one agent, and the
conversation named per call. Pass `session=` to `run()`:

```python
from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))

agent.run("My dog is named Pixel.", session="user-123")
agent.run("My cat is named Mote.",  session="user-456")

print(agent.run("What is my pet's name?", session="user-123").output)  # -> Pixel
print(agent.run("What is my pet's name?", session="user-456").output)  # -> Mote
```

The run builds its prompt from that conversation's history and appends the turn
to it. The two conversations never see each other, and the agent's own memory is
untouched and restored when the call ends — including when the run fails.

A `Session` object works as well as an id, which is what you want when the
conversation is already in hand:

```python
from effgen.core.session import Session

session = Session.load_or_create("user-123")
response = agent.run("and how old is he?", session=session)
```

Without `session=`, `run()` behaves exactly as before and uses the agent's own
memory.

## Checkpoints & resume

A checkpoint snapshots an in-progress run — the run's own steps, its iteration,
its memory and the model id — so it can be resumed. Write checkpoints by passing
a directory:

```bash
effgen run --checkpoint-dir ./checkpoints --checkpoint-interval 1 "Long task..."
```

Resume — by default the latest checkpoint in the directory:

```bash
effgen resume --checkpoint ./checkpoints           # newest checkpoint
effgen resume --checkpoint ./checkpoints/<id>.json # a specific file
```

Resume reuses the **model the checkpoint was created with**. Pass `-m/--model`
to override; if it differs from the saved model you get a clear warning. If the
checkpoint predates model tracking, resume falls back to a small local model and
tells you to pass `-m`.

Programmatically:

```python
from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
agent.run("Long task...", checkpoint_dir="./checkpoints", checkpoint_interval=1)

# Later, on a fresh agent built with the same tools:
resumed = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
result = resumed.resume(checkpoint_dir="./checkpoints")
```

Checkpoints are **JSON only** (no pickle). A truncated or corrupt checkpoint
raises a clear `CorruptStateError` that names the file rather than a stack
trace; a missing one raises `FileNotFoundError`. A SQLite backend is also
available: `CheckpointManager(dir, backend="sqlite")`.

### What a checkpoint carries

A checkpoint holds the run's conversation under `thread`, as the same data
`response.thread.to_dict()` produces (see
[Reading a run back](reading-a-run.md)), carrying that format's own
version number. `Checkpoint.to_thread()` hands it back:

```python
from effgen.core.checkpoint import CheckpointManager

cp = CheckpointManager("./checkpoints").load_latest()
thread = cp.to_thread()
print(len(thread.steps), [a.tool for a in thread.actions()])
```

The transcript those steps render to is written beside them under `scratchpad`,
so a checkpoint written here still resumes on a 1.0.x build.

**A checkpoint written by 1.0.x resumes here too.** Those files carry only the
transcript, so the steps are read back out of it — which recovers the order of
thoughts, calls, rendered inputs and answers, and renders byte for byte to the
text it was read from. Five things were never in that text and do not come back:
a tool call's id (the recovered call is named from its position); the argument
names behind an input the model did not write as JSON, which comes back whole
under one key; a line the framework injected after an observation, which comes
back as part of that observation; the step the run ended on, with the answer it
reached and the reason it stopped; and the frame the run was asked in — the
persona, the tool contract, the session's earlier turns and the task.
The run resumes and completes; what it loses is structure underneath text it
still has.

## Resuming a workflow

The checkpoints above resume **one agent's run**. A `WorkflowDAG` runs several
agents, and it has its own store so a pipeline that died half way through does
not start again from the top. Pass a store and a run id:

```python
from effgen import FileCheckpointStore, WorkflowDAG, WorkflowNode

store = FileCheckpointStore()          # ~/.effgen/workflows by default
dag = WorkflowDAG("report")
dag.add_node(WorkflowNode(id="research", agent=researcher))
dag.add_node(WorkflowNode(id="draft", agent=writer))
dag.add_node(WorkflowNode(id="review", agent=editor))
dag.connect("research", "draft")
dag.connect("draft", "review")

result = dag.run("Write the Q3 summary.", checkpoint=store, run_id="q3-summary")
```

If that process is killed after `research` and `draft` finished, running the
**same line again** picks up at `review`. There is no separate resume call: a
run id the store has never seen starts from the beginning, and one it knows
continues.

What resuming does with each node:

| Node state when the run stopped | On resume |
|---|---|
| completed | not run again; its output is restored and flows downstream |
| skipped | stays skipped, with the original reason |
| failed | **retried** — that is usually why you are resuming |
| never started | run normally |

Progress is saved after each topological level, which is the point at which
every node in that level has finished and the next has not started. Writes are
atomic, so a crash during a save leaves the previous checkpoint intact rather
than a half-written file.

Two behaviours worth knowing:

- **Re-running a finished run is cheap.** Its stored outputs are replayed
  without calling a model, which makes a workflow run idempotent under a job
  runner that retries. Call `store.delete(run_id)` to genuinely start over.
- **The graph must match.** Resuming into a DAG with different node ids raises
  `ValueError` naming what was added or removed, rather than mixing outputs
  from two different workflows.

The store holds run *state*, never the graph — agents own sockets, model
handles and credentials, none of which survive a process boundary. Rebuild the
same `WorkflowDAG` in the new process and hand it the same `run_id`.

For a store that only needs to outlive a failed node rather than a failed
process, use `InMemoryCheckpointStore()`. To keep checkpoints somewhere else
entirely — a database, an object store — implement the three methods of
`CheckpointStore` (`save`, `load`, `delete`) and pass your own.

Set `EFFGEN_WORKFLOW_DIR` to move the on-disk location, which is what
containers and test suites do so a run cannot write into a developer's home
directory.

## Background tasks

Run agent tasks on worker threads without blocking:

```python
from effgen.core.background import BackgroundTaskRunner

with BackgroundTaskRunner(agent, max_workers=2) as runner:
    task_id = runner.submit("Summarize this report...")
    result = runner.get_result(task_id, wait=True, timeout=60)
    print(runner.get_status(task_id))   # COMPLETED / FAILED / CANCELLED
```

- `submit` / `get_status` / `get_result` / `cancel` / `pause` / `resume`.
- A failing task is reported as `FAILED` with a typed, **secret-redacted**
  error (never a silent success).
- The runner stops its worker threads when it is closed, used as a context
  manager, garbage-collected, or at interpreter exit — so a forgotten
  `shutdown()` never leaks threads. Prefer the `with` form for prompt cleanup.
