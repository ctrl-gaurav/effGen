# Reading a run back

Every finished run carries the conversation it had. That conversation is the
most useful thing a run produces when something went wrong: it is what the
model was framed by, what it was asked, what it reasoned, which tools it called
with which arguments, what came back, and how the run ended — in the order it
happened, rather than as one flattened transcript string.

Nothing has to be turned on to get it. No debug logging, no environment
variable, no second run.

## The accessor

`response.thread` is the documented way to reach it:

```python
from effgen import Agent, AgentConfig
from effgen.tools import get_registry

calculator = get_registry().get_tool_sync("calculator")
agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai", tools=[calculator]))
response = agent.run("What is 6 * 7?")

thread = response.thread                 # an AgentThread, or None
print(len(thread.steps))
print([step.kind for step in thread.steps])
print([action.tool for action in thread.actions()])
```

`response.metadata["thread"]` holds the same object and keeps working — it is
where the conversation has always lived. `response.thread` is the supported
name for it, and hands back `None` rather than raising for a response that
recorded no conversation.

### Serialising it

The thread is an object, not data, so **`json.dumps(response.metadata)` raises
`TypeError`**. The serialisable form of a whole run is `response.to_dict()`,
which writes the conversation through its own serialisation:

```python
import json

try:
    json.dumps(response.metadata)        # the thread is an object, not data
except TypeError as exc:
    print(exc)                           # Object of type AgentThread is not JSON serializable

document = response.to_dict()            # this is the documented path
json.dumps(document)                     # fine
document["metadata"]["thread"]["steps"]  # the same steps, as plain data
```

`AgentThread.from_dict()` reads that data back into a thread, so a saved run,
a checkpoint and a stored session all hand the conversation back as steps.

## Printing it

`effgen.thread_as_text` renders a conversation for a human:

```python
from effgen import thread_as_text

print(thread_as_text(response.thread))
```

```
  1. system (persona)
     | You are a careful assistant.
  2. system (contract)
     | First work the task out yourself, in your own words, step b…
  3. task
     | What is 6 * 7?
  4. thought
     | I should check this with the calculator.
  5. action (calculator)
     | {"expression": "6 * 7"}
  6. observation
     | 42
  7. answer  stop_reason=final_answer
     | 42
```

An agent holding tools is framed by two instructions — the persona it was given
and the tool-use contract — so both are steps, and both are what the model was
sent.

Two properties make that rendering worth keeping:

- **It diffs.** No timestamp, no run id and no provider-minted tool-call id
  appears, so two runs that took the same path render the same bytes and `diff`
  on two of them shows what the runs actually did differently. Pass
  `include_ids=True` when you do want the call ids.
- **Secrets do not travel.** Every rendered string goes through the same
  scrubber the structured logs use, so a provider key that reached a tool's
  output or an instruction is replaced by a labelled placeholder. Pass
  `redact=False` when you need the raw text.

`effgen.render_thread` returns the same steps as `RenderedStep` records —
`position`, `kind`, `label`, `body`, `detail`, `depth` — for a caller building
its own output. A delegated child's steps follow the delegation that produced
them, one level deeper.

## From the command line

```bash
effgen run "What is 6 * 7?" --show-thread
```

prints the answer and then the conversation, exactly as above.

```bash
effgen run "What is 6 * 7?" --json | jq '.metadata.thread.steps[].kind'
```

carries the same conversation as data. Every document `effgen run` writes —
`--json` on stdout, `-o` to a file, `--card` to an HTML page — is scrubbed on
the way out, because those are the copies that get piped, saved and shared. The
answer printed on the terminal is the run's own words, unchanged.

`effgen debug "<task>"` shows the conversation as it stood at the end of every
iteration, and the run card written by `--card` carries it as a table.

## Where else the conversation turns up

| Surface | How to reach it |
| --- | --- |
| A completed run | `response.thread` |
| A run that delegated | `response.sub_agent_threads()`, and `response.thread.delegations()` |
| A workflow | `result.thread`, `result.node_thread("<id>")` |
| A team | `result.thread`, `result.agent_threads()` |
| A checkpoint | `Checkpoint.to_thread()` |
| A stored run | `effgen run --json`, or the dashboard's run drill-in for its shape |

## See also

- [Sessions and checkpoints](sessions-and-checkpoints.md) — persisting a
  conversation and resuming from it.
- [Context compaction](context-compaction.md) — what happens to a conversation
  that outgrows the model's window, and how a shortened step says so.
