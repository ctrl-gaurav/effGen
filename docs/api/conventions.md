# API Conventions

effGen is a large framework, but it follows a small set of consistent
conventions. Learn these once and the rest of the surface is predictable.

## Importing

The common entry points live at the top level:

```python
from effgen import Agent, AgentConfig, load_model, create_agent, list_presets
from effgen import tool, Tool          # low-boilerplate tool authoring
from effgen.tools.builtin import Calculator, WebSearch  # built-in tools
```

`import effgen` is lazy — names resolve on first access, so importing the
package is cheap even though the public surface is large.

## Creating an agent

There are two equivalent paths; pick whichever reads better for you:

```python
# Preset + model id (shortest):
agent = create_agent("math", "gpt-5-nano")

# Explicit config (full control):
agent = Agent(AgentConfig(name="my-agent", model="gpt-5-nano", tools=[...]))
```

A `model` is always **required** — effGen never silently picks a paid cloud
model. Pass a model id (string) or a loaded model instance. To choose a default
once, set the `EFFGEN_DEFAULT_MODEL` environment variable.

## Models and providers

Model ids are strings. When an id is unambiguous it routes automatically;
otherwise prefix it with the provider:

```
"gpt-5-nano"                     # routes to OpenAI
"openai:gpt-5-nano"              # explicit provider prefix
"Qwen/Qwen2.5-1.5B-Instruct"     # local (Transformers/vLLM)
```

You can also pass `provider=` to `AgentConfig` / `load_model`, or `--provider`
on the CLI. A wrong id fails closed with a "did you mean…/available now…" hint —
run `effgen models list` to browse and `effgen doctor` to see which providers
are usable.

## Results

Every `agent.run(...)` returns an `AgentResponse`:

```python
result = agent.run("What is 17% of 250?")

print(result)            # the answer (str) — __str__ returns result.output
result.output            # the answer string
result.text              # read-only alias of .output
result.content           # read-only alias of .output
result.success           # bool — never True with an empty answer
result.outcome           # "answered" | "stopped" | "failed"
result.stop_reason       # what ended the run; never None
result.partial           # what a stopped run had reached, or None
result.tokens_used       # int
result.execution_time    # float seconds
result.to_dict()         # full structured detail (trace, cost, metadata)
```

On failure, `success` is `False`, the message is clear and redacted, and
`result.metadata["error"]` is a structured `{type, category, provider, model,
message, retryable}` dict — identical whether the failure came from the direct
or the tool path.

### Answered, stopped, failed

A run ends in one of three states, and `result.outcome` names which without you
having to read `output`:

| `outcome` | what it means | `output` | `partial` |
|---|---|---|---|
| `answered` | the model wrote an answer | the answer | `None` |
| `stopped` | the loop ended the run before the model wrote one | what happened, and what to do | the progress, when there was any |
| `failed` | the run could not be carried out | the classified failure | `None` |

`stop_reason` says which exit it took, and is present on **every** response — an
answered run reports `"final_answer"`. It is always equal to
`metadata["reason"]`, which earlier releases already carried. The vocabulary is
closed and published as `effgen.core.agent_response.STOP_REASONS`; the five
reasons that mean *stopped* are in `STOPPED_REASONS`:
`max_iterations_partial`, `max_iterations_exhausted`, `loop_detected`,
`repeated_tool_result`, `null_final_from_model`.

A **stopped** run has tool results and reasoning but no answer, so those never
go where the answer goes. They travel in `result.partial`, a `PartialResult`:

```python
result.partial.observations      # every tool result, in call order
result.partial.last_observation  # the final one
result.partial.last_thought      # the model's last reasoning line
result.partial.text              # the flattened form, one line
result.partial.iterations, result.partial.tool_calls
```

`result.metadata["partial_output"]` carries `partial.text` under the key earlier
releases used, and `metadata["partial"]` is `True` whenever there is progress.

```python
if result.outcome == "answered":
    print(result.output)
elif result.outcome == "stopped":
    print("no answer:", result.stop_reason)
    if result.partial:
        print("what it had:", result.partial.observations)
else:
    print("failed:", result.stop_reason, result.metadata["error"]["message"])
```

### `raise_on_error` and a stopped run

`AgentConfig.raise_on_error` defaults to `True`, so a run that produced no
answer raises rather than returning something that reads like one. For a
**stopped** run that exception is `effgen.RunStoppedError`, which subclasses
`RuntimeError` — anything already catching `RuntimeError` around the iteration
cap keeps working — and carries the run with it:

```python
from effgen import RunStoppedError

try:
    print(agent.run(task).output)
except RunStoppedError as exc:
    print(exc.stop_reason)                     # e.g. "loop_detected"
    print(exc.partial.text if exc.partial else "")
    print(exc.response.tool_calls)             # the calls it did make
```

With `raise_on_error=False` the same run comes back as a response with
`success=False` and `outcome == "stopped"`. Batch evaluation still wants the
flag off, so a stopped row is a row rather than an exception. A **failed** run
raises what it always raised: the typed provider error, or `RuntimeError`.

### Cost on the response

`result.metadata["cost_usd"]` is the run's cost in USD, summed across every
model call the run made. A run whose model publishes no per-token price carries
**no `cost_usd` key at all** and reports `metadata["unpriced_calls"]` instead —
the number of calls whose price is unknown. A run that mixed a priced model with
an unpriced one carries both: the cost of the calls that were priced, and the
count of the ones missing from it. A genuine free tier reports `cost_usd: 0.0`,
which is a real answer, not a placeholder.

Per model call, `GenerationResult.metadata["cost_usd"]` follows the same rule:
a float (possibly `0.0`) when a rate is published, `None` when it is not.

### Reasoning models that emit no visible token

A reasoning model can spend a whole turn on its internal chain and return an
empty answer you were still billed for. Providers report this in one of two
ways — the chain itself, or a count of what it cost:

| Provider | Signal |
| --- | --- |
| Together, Groq, Cerebras, Fireworks, HF router | `message.reasoning` (or `message.reasoning_content`) |
| OpenAI | `reasoning_tokens` in the usage block — chat completions and the Responses API alike |
| Gemini | thought parts plus `thoughts_token_count` |
| Anthropic | `thinking` content blocks |

Either way the answer is empty.

**The contract.** The chain is diagnosis, never the answer — effGen does not
return it as the result. A turn that produced only reasoning is reported, not
retried at settings that already failed:

- `GenerationResult.metadata["reasoning_only"]` is `True`, and
  `metadata["empty_response_reason"]` names the model, the `max_tokens` cap in
  force and the reasoning budget spent. `metadata["reasoning_tokens"]` carries
  the count whenever the provider reports one (on any turn, not only this one),
  `metadata["reasoning_chars"]` the length of the chain, and
  `metadata["reasoning"]` the chain itself when the provider sent it.
- Through `agent.run(...)`, the run fails with
  `metadata["error"]["type"] == "ReasoningOnlyResponse"` (category
  `reasoning_only`, `retryable: False`) carrying that message — or, when the
  budget was the cause, `"TruncatedResponse"`, whose message now also names the
  reasoning budget. Neither is the generic "empty response after retries".
- A native tool call with empty text is a complete turn and is never reported
  as reasoning-only. A server-side native-tool turn (OpenAI's Responses API,
  Gemini's built-in tools) that produced only reasoning fails the same typed
  way rather than reporting that there was simply no output.
- A streamed turn has no metadata channel, so the same message is logged when a
  stream ends without yielding one visible token.

**Stop sequences.** A provider that streams the chain and the answer through one
token stream matches stop sequences against the chain too, so a stop sequence
can end generation before the first visible token. For a reasoning model the
agent therefore holds its stop sequences back and applies them to the returned
answer instead — the same visible result, without the collision. A model the
catalog does not flag as a reasoning model is recognised from the first turn
that shows the shape, and the recovery is remembered for the rest of the
process, so at most one turn is spent on it.

## Streaming

`agent.stream(task)` yields successive **answer-text** `str` chunks; joining them
reconstructs the (sanitized) answer. The iterator ending is the "done" signal; a
provider failure raises a typed error rather than silently ending the stream.

```python
for chunk in agent.stream("Write a haiku about the sea"):
    print(chunk, end="", flush=True)
```

This holds for **tool-using agents** too: the default text stream is the answer
only — the internal ReAct scaffolding (`Thought:` / `Action:` / `Observation:` /
`Final Answer:`) is never part of the text payload. To observe the steps as they
happen, either pass the `on_thought` / `on_tool_call` / `on_observation`
callbacks, or opt into typed events:

```python
for event in agent.stream(task, include_events=True):
    if event.kind == "answer":
        print(event.text, end="", flush=True)
    elif event.kind == "tool_call":
        print(f"\n[calling {event.tool}]")
```

`include_events=True` yields `StreamEvent` objects with a `kind` of `answer`,
`thought`, `tool_call`, `observation`, `status`, or `usage`; concatenating the
`answer` events still reconstructs the final answer.

### Streaming with native tool calling

A tool-using stream takes one of two paths, chosen from the model:

- **The provider's tool calling.** When the adapter records the tool calls it
  streams — openai, gemini, groq, together, fireworks and cerebras do — the loop
  dispatches those calls the same way `agent.run()` does, and the assistant's
  text streams through as it is written. This is the same loop, the same repeat
  guards and the same failure vocabulary as a non-streamed run.
- **The ReAct text protocol.** Every other model — the local chat-template
  engines, and any provider whose stream drops its tool calls — keeps the
  prompt-based scaffold, where the answer arrives once the turn is parsed.

`agent.model.streams_tool_calls()` reports whether an adapter records what it
streams. The path is chosen per turn and needs no configuration.

On the native path a turn's text is held back until the turn can no longer
become a tool call: once a call has been declared the text is delivered as a
`thought` and never enters the answer, and once a text delta has arrived with no
call declared the turn is committed to answering. Text is also sanitized before
it is emitted, so what reaches the screen is what the answer ends up being.

### The record a streamed turn leaves

After a stream that took the native path, `agent.last_stream_response` holds the
`AgentResponse` the same task would have produced through `run()` — `output`,
`success`, `iterations`, `tool_calls`, `tokens_used`, `execution_time`, and the
`reason` / `error` / `partial` metadata described under [Results](#results). It
is `None` after a stream that did not take that path.

```python
events = list(agent.stream(task, include_events=True))
answer = "".join(e.text for e in events if e.kind == "answer")
response = agent.last_stream_response
assert answer == response.output          # a turn that answered
```

For a turn that answered, joining the `answer` events reproduces
`response.output` exactly. A turn the loop stopped — at its iteration cap, on a
repeated call, on a tool that reproduced its own result — and a turn whose model
wrote its tool call out as text instead of making it both have no answer to
stream: `output` carries the typed outcome and it arrives as a `status` event,
not as answer deltas. `response.outcome`, `response.stop_reason` and
`response.partial` read the same as they do after `run()`.

### Usage after a stream

The last event of an `include_events=True` stream is a `usage` event carrying
what the run cost, and the same dict is on `agent.last_stream_usage` after any
stream — including text mode — so a streamed turn can be tallied without running
the prompt a second time:

```python
chunks = list(agent.stream(task))
usage = agent.last_stream_usage
print(usage["total_tokens"], usage["cost_usd"], usage["ttft_ms"])
```

The keys are `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_usd`
(`None` for a model with no published price), `latency_ms`, `ttft_ms` (time to
the first answer token), `model_calls` (above one on a tool-using run), and
`estimated` — `True` when the token counts were counted locally because the
backend reported none, as local engines do.

Over the OpenAI-compatible server the same numbers arrive on the final
`stream_options.include_usage` chunk, whose `effgen` object carries `cost_usd`
alongside the standard `usage` block.

### Who decides what an answer looks like

effGen appends things to a prompt the model never asked for — the observation
block after a tool ran, most of all. It describes that machinery, because the
model cannot know what it is. It says nothing about the **form** of the answer,
because it does not know the question. Form is settled, in this order:

1. **`output_schema` / `output_model`**, when you declared one. This is the only
   machine-readable statement of shape effGen has, so it is stated to the model
   inside the loop, while the answer is being written, rather than being applied
   to prose afterwards. A run that answers in the declared shape on the first
   attempt costs one model call, not three.
2. **The task and your `system_prompt`.** Nothing effGen appends contradicts
   them. A task that ends "answer with the letter of the correct option" gets a
   letter, including after a retrieval tool has run.
3. **Nothing.** With no schema and no instruction of your own, the model follows
   the task as written.

The one thing effGen still says after retrieved passages is what the passages
are — source material, not the answer — and what to do if they do not answer the
question. Neither is a statement about length or wording.

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    option: str
    confidence: float

response = agent.run(question, output_model=Verdict)
response.metadata["structured_output_method"]    # "agent_output" — no repair call
response.metadata["structured_output_attempts"]  # 0
```

`structured_output_method` is `"agent_output"` when the answer already matched
the schema, and `"reprompt"` when effGen had to ask again; `structured_output_attempts`
counts those extra calls. Both describe how hard the framework had to work, not
the value you get back.

A schema set on `AgentConfig` is stated on a streamed run too, so the same agent
answers in the same shape whether you call `run()` or `stream()`. Streaming
yields the model's tokens as they arrive and does not validate or parse them, so
a stream gives you the declared shape as text; the parsed value and the
`structured_output_*` metadata come from `run()`.

### What effGen tells a model about your tools

Tool definitions do not travel in the prompt. They go through the provider's
tool-calling API, or through a local chat template, and the model is left to
work out what they are for. So effGen states it, once, on the turn the model
first sees them — the same sentences whether the run blocks or streams, and
whether the tools reach the model through an API or a template.

Which sentences depends on what the tools *are*, read from each tool's declared
`ToolCategory`:

| the tools you attached | what the model is told |
|---|---|
| `COMPUTATION` | work the task out first, then use the tools to check the steps you are least sure of |
| `CODE_EXECUTION`, `SYSTEM` | use the tools to do the task rather than working it out in your head, and read the answer off what they return |
| `INFORMATION_RETRIEVAL`, `EXTERNAL_API` | what comes back is source material, not the answer |
| anything else, **or a set that mixes the above** | work through the task one step at a time, one step per call |

A mixed set gets the general text on purpose. Each of the other three asserts
something about every tool the model is holding, and on a mixed set that
assertion is false of some of them — telling a model that owns a code executor
to work the answer out itself is how it stops calling the executor at all.

Your own configuration comes first:

- **`system_prompt`** — your persona still leads the prompt, and the contract is
  still stated. They are not in competition: the persona is who the model is,
  the contract describes machinery effGen attached.
- **`tool_contract="…"`** — that text is stated instead, in the same position.
- **`tool_contract=""`** — nothing is stated. Use this rather than
  `system_prompt_template`, which makes you rebuild the whole scaffold.
- **`system_prompt_template`** — you own the entire prompt, `{tools_description}`
  included, and effGen adds nothing to it.

An agent with no tools is unaffected: there is nothing to state a contract about.

## Tools

The recommended way to author a tool is the `@tool` decorator (it wraps the full
`BaseTool` machinery for you):

```python
from effgen import tool

@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())
```

The decorated object is a real tool instance — drop it into
`AgentConfig(tools=[...])` and it works with provider-native function-calling
too. `Tool.from_function(fn)` is the non-decorator equivalent. For rich
validation or lifecycle hooks, subclass `BaseTool` directly (see
[Custom Tools](../tutorials/custom-tools.md)). Multi-action built-in tools take
a canonical `operation` selector and accept common synonyms (`action`, and
natural verbs) so the obvious call works.

## Errors

User-facing errors are typed and actionable: a one-line cause plus how to fix
it. Unknown preset, model, or tool names raise typed errors with a fuzzy "did
you mean?" suggestion instead of a bare `KeyError`.

## Type hints

effGen ships a `py.typed` marker, so your editor and `mypy`/`pyright` see effGen's
annotations on the public surface (`from effgen import ...`). The public surface
is checked two ways in CI: a deterministic gate ensures every advertised name
resolves with an introspectable signature, and an advisory `mypy` lane
type-checks the public modules. Internal modules carry best-effort annotations
that may still tighten over time; rely on the documented public surface.

## Counts (tools, presets, providers, models)

The headline counts drift as the framework grows, so they are **generated**, not
hand-maintained. Get the exact current numbers from the live package:

```bash
python scripts/gen_counts.py           # human-readable table
python scripts/gen_counts.py --json    # machine-readable
```

This is the single source of truth for "how many tools/presets/providers/models
does effGen ship".
