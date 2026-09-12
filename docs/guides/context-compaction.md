# Context compaction

Two different things outgrow a model's context window, and effGen bounds them
separately.

| What grows | What bounds it | Where |
|---|---|---|
| **The session's history** — the turns of a conversation held across runs | `compaction_strategy` and `memory_config["short_term_max_tokens"]` | this page, below |
| **One run's own transcript** — its reasoning, its tool calls and their results | `context_budget` and `compaction` | [One run's own budget](#one-runs-own-budget) |

They are not the same bound and they do not replace each other: a single run
holding one long tool result can outgrow the window without the session having
more than one turn in it, and a fifty-turn chat can outgrow it with no run
taking more than two steps.

## The session's history

A conversation eventually outgrows the model's context window, and something
has to go. Which turns survive changes the answer more for a small model than
for a frontier one, and different tasks want different answers, so effGen makes
the choice a strategy rather than a fixed rule.

The default, `SummarizeOldest`, is what effGen has always done: once the history
passes a fraction of the window, everything but the most recent few turns
becomes a summary.

```python
from effgen import Agent, AgentConfig
from effgen.memory.compaction import KeepFirstAndLast

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    compaction_strategy=KeepFirstAndLast(first=2, last=6),
))
```

Strategies can also be named:

```python
AgentConfig(model=..., compaction_strategy="drop_oldest")
```

## What ships

| Strategy | What survives | Model call? |
|---|---|---|
| `SummarizeOldest` *(default)* | The most recent turns verbatim; everything older becomes an extracted summary. | No |
| `DropOldest` | The most recent turns. The rest is forgotten. | No |
| `KeepFirstAndLast` | The opening turns and the recent ones; the middle is summarized or dropped. | No |
| `KeepToolResults` | Tool results and the recent turns; older reasoning is summarized or dropped. | No |

**None of the four asks a model anything.** A summary here is built from the
turns themselves: tool results kept near-verbatim, and the sentences carrying
numbers and names, sorted into facts, decisions and open questions. So
compaction adds no latency, no spend and nothing the model might have invented.
Supply a `summarize` of your own (see [Writing your own](#writing-your-own)) if
you want a model-written one; the trade is stated under [One run's own
budget](#one-runs-own-budget), and it applies here too.

**`DropOldest`** is the cheapest of the four and forgets rather than
abbreviates. Use it when older turns genuinely do not matter.

**`KeepFirstAndLast`** exists because the opening turns usually carry the task
— the document, the instruction, the constraint everything else refers to — and
a summary of those is a poor substitute. The redundancy in a long conversation
is in the middle.

**`KeepToolResults`** suits a tool-heavy run, where the reasoning is most of the
tokens and the tool results are the evidence the answer rests on.

## Measuring the history

By default the history is measured with the model's own tokenizer when there is
one, and otherwise estimated at four characters per token. Supply a tokenizer to
measure it in the units the window is actually measured in:

```python
import tiktoken

AgentConfig(model=..., tokenizer=tiktoken.get_encoding("cl100k_base"))
```

Anything with `count_tokens(text)` or `encode(text)` works. A tokenizer that
raises falls back to the estimate rather than failing the run.

## Writing your own

Override what differs; the defaults are `SummarizeOldest`'s.

```python
from effgen.memory.compaction import CompactionStrategy

class DropFailedToolTurns(CompactionStrategy):
    """Compact the turns where a tool errored; keep everything else."""

    def messages_to_compact(self, memory):
        return [m for m in memory.messages if "Error executing tool" in m.content]

    def summarize(self, memory, messages):
        return None   # drop them rather than summarize them
```

The three methods, called in this order:

| Method | Answers | Default |
|---|---|---|
| `should_compact(memory)` | Is it time? | Past `summarization_threshold` × the window |
| `messages_to_compact(memory)` | Which messages leave? | Everything but the recent few |
| `summarize(memory, messages)` | What replaces them? | A summary extracted from those messages; `None` drops them |

Returning an empty list from `messages_to_compact` cancels that round.

## One run's own budget

A run's transcript grows with every step it takes. `context_budget` says how
many prompt tokens one run may send; when a turn would go over it, the run
shortens its own conversation instead of growing until the provider refuses it.

```python
from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model=...,
    context_budget="auto",         # the default
))
```

| `context_budget` | Means |
|---|---|
| `"auto"` *(default)* | derived from the window the model declares, less what the run reserves for its own reply. A model that declares no usable window leaves the run unbounded rather than guessed at. |
| an `int` | that many prompt tokens, exactly |
| a `float` in `(0, 1]` | that share of the model's window |
| `None` | unbounded |

`max_context_length` overrides the window the model declares, for a model whose
adapter reports one you do not trust or does not report one at all.

### What a run gives up, in order

`compaction` names the policy. The default, `ShortenOldestFirst`, applies four
rungs oldest-first and stops the moment the prompt fits:

1. shorten an old tool result, keeping its opening and saying how much left;
2. drop an old thought — the tool call it led to keeps its own reasoning;
3. drop whole answered cycles, leaving one line saying what went;
4. nothing left to give up: raise `ContextBudgetExceededError`.

The question, the instructions the run is framed by, the session's earlier
turns, the most recent two complete cycles and the answer are never given up at
any rung — a run that dropped its own question would answer a different one. A
tool call and the result answering it always leave together, because a
conversation holding a call nothing replied to is one a provider rejects.

`ShortenOldestFirst` makes **no model call**, so a run that never reaches its
budget sends exactly the bytes it always sent, and one that does can be
replayed. `SummarizeWithModel` is the opt-in alternative: its marker carries a
model-written summary of what left, using the agent's own model unless you name
another.

```python
from effgen import ShortenOldestFirst, SummarizeWithModel

AgentConfig(model=..., compaction=ShortenOldestFirst(keep_recent_cycles=4))
AgentConfig(model=..., compaction="summarize_with_model")
```

A summary permanently becomes part of the conversation and is trusted like any
other step, so a summariser reading untrusted content is an indirect-prompt-
injection route that outlives the turn it was written in. Point it at a model
you would let write the run's own reasoning. A summariser that fails or refuses
never fails the run — the plain marker is written instead.

### What the caller sees

Every response carries `metadata["context_budget"]`, present whether or not a
budget was in force:

```python
response = agent.run("...")
budget = response.metadata["context_budget"]
budget["firings"]           # rounds of compaction; 0 when it never bound
budget["budget_tokens"]     # what the run was allowed to send
budget["measured_tokens"]   # the last prompt it actually sent
budget["source"]            # "auto", "config", "fraction" or "unbounded"
```

Each round logs one line beginning `[context] compacted the thread`, so the
firing count is greppable as well as readable off the response.

### When it will not fit

```python
from effgen import ContextBudgetExceededError

try:
    agent.run(task)
except ContextBudgetExceededError as exc:
    print(exc.budget_tokens, exc.measured_tokens, exc.window_tokens)
```

It is raised **before the request is sent**, so a run whose question alone is
too large costs nothing at the provider, and it is an `InvalidRequestError` —
code already catching a prompt that was refused for being too long keeps
catching it.

## Related

- [Sessions and checkpoints](sessions-and-checkpoints.md) — persisting a
  conversation across processes.
- [Architecture](../architecture/overview.md) — where short-term memory sits in the three-tier system.
