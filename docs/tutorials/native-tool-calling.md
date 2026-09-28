# Native Tool Calling

effGen v0.2.0 supports native function calling for models that have built-in tool-use capabilities (Qwen, Llama, Mistral, and others). This bypasses text-based ReAct parsing for faster, more reliable tool execution.

## Tool Calling Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `"auto"` | Automatically selects native if model supports it, else ReAct | Default — works everywhere |
| `"native"` | Uses model's built-in function calling format | Qwen 2.5+, Llama 3.2+, Mistral |
| `"react"` | Text-based ReAct reasoning loop | Any model, maximum compatibility |
| `"hybrid"` | Tries native first, falls back to ReAct | Best accuracy with capable models |

## Basic Usage

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

model = load_model("Qwen/Qwen2.5-3B-Instruct")

config = AgentConfig(
    name="native_agent",
    model=model,
    tools=[Calculator()],
    tool_calling_mode="native",  # Use native function calling
)

agent = Agent(config=config)
result = agent.run("What is 42 * 58?")
print(result.output)  # 2436
```

## Structured Output

Force the agent to return JSON matching a schema:

```python
config = AgentConfig(
    name="structured_agent",
    model=model,
    tools=[Calculator()],
    output_format="json",
    output_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "number"},
            "explanation": {"type": "string"}
        },
        "required": ["answer"]
    },
)
```

### With Pydantic Models

```python
from pydantic import BaseModel

class MathResult(BaseModel):
    answer: float
    explanation: str

result = agent.run("What is 15% of 200?", output_model=MathResult)
parsed = result.metadata["parsed_output"]  # MathResult instance
print(parsed.answer)  # 30.0
```

## Checking Model Support

```python
model = load_model("Qwen/Qwen2.5-3B-Instruct")
print(model.supports_tool_calling())  # True
print(model.tool_call_support())      # "template"

model = load_model("google/gemma-2-2b-it")
print(model.supports_tool_calling())  # False — "auto" resolves to ReAct
print(model.tool_call_support())      # "none"
```

`tool_call_support()` names the mechanism behind the boolean, because the two
mechanisms behave differently:

| value | meaning |
|---|---|
| `"api"` | The provider takes tool definitions as a request parameter and returns any call as structured data. A cloud adapter reports this whenever it advertises tool calling for the model. |
| `"template"` | The definitions are rendered into the prompt by a local chat template. Nothing enforces the format — whether a call is emitted is up to the model. |
| `"none"` | No native tool calling. The ReAct text protocol is the only way to reach a tool. |

On a local engine, `supports_tool_calling()` asks whether the chat template
**renders** the definitions, not whether it accepts a `tools` argument. Some
templates — gemma-2 and Phi-3.5 among them — take the argument and discard it,
producing a prompt byte-identical to one built with no tools at all. Those
report `False`, so `"auto"` sends them down the ReAct path, where the tools are
described in the prompt text and the model can actually reach them.

Neither mechanism carries the definitions in the prompt text, so neither says
anything about *using* them. effGen states that itself, on the first turn of the
run and identically for `"api"` and `"template"`: what the tools are for, chosen
from their declared `ToolCategory`. A calculator is described as something to
check reasoning with, a code executor as something to run the work on, a search
tool as something that brings back material to answer from. The full table, and
how to replace or silence it with `AgentConfig.tool_contract`, is in
[Conventions](../api/conventions.md#what-effgen-tells-a-model-about-your-tools).

Whether the tool has to be used at all is the separate `AgentConfig.tool_use`
setting, also read from the declared categories by default: a code executor must
actually run, and nothing else is pushed either way. `tool_use="required"` makes
any attached tool one the run may not answer without — including on this path's
`"template"` mechanism, where there is no request parameter to constrain.

## Measured, not assumed: the capability probe

A declaration says how tool definitions reach a model, not whether the model
uses them. A model served behind a `base_url` is declared an `"api"` caller
whatever sits behind the URL, and two things can still go wrong: a small model
may answer from memory while holding a search tool that has the answer, and a
server may not carry a model's native calls back as structured calls, so the
same call repeats until a guard ends the run.

At `tool_calling_mode="auto"`, the first agent built for a model served behind a
URL, or run on a local engine, measures this once: eight questions about
fictional things, answerable only through a stub search tool, run as ordinary
agent runs. Each is `resolved` (called, and the answer carries the result),
`unresolved` (called, and the answer does not) or `skipped` (no call). Two
defaults follow from the counts, and only when you left them unset:

| measured | what `auto` does for this model |
|---|---|
| three or more native runs unresolved, and the text frame resolves at least three more | uses the ReAct text frame |
| any native run skipped | a run holding an information-retrieval tool that answers without calling it is sent back once and made to call |
| otherwise | as the declaration says |

Calculator and code tools are never moved by the probe. An explicit
`tool_calling_mode`, an explicit `tool_use`, `capability_probe=False` or
`EFFGEN_CAPABILITY_PROBE=0` keep the declared behaviour, and cloud adapters are
never probed. A probe costs about sixteen requests; it is stored in
`~/.effgen/capabilities.json` (or `$EFFGEN_HOME`, or `$EFFGEN_CAPABILITY_CACHE`)
and read by every later agent, across processes. It is measured again when the
weights, the chat template, the endpoint or the probe itself change, and after
30 days. A probe that cannot run — a refused request, a failing endpoint, over
its budget of 48 requests or 120 seconds — stores nothing and leaves `auto` as
declared, with one warning.

```bash
effgen doctor                                   # what was probed and learned
effgen doctor --probe Qwen/Qwen2.5-1.5B-Instruct --base-url http://127.0.0.1:8000/v1 --refresh
```

```python
from effgen import probe_tool_calling
from effgen.models import load_model

model = load_model("Qwen/Qwen2.5-1.5B-Instruct", provider="openai_compatible",
                   base_url="http://127.0.0.1:8000/v1", context_length=8192)
probe = probe_tool_calling(model)
print(probe.summary(), probe.strategy, probe.required_categories)
```

A run reports why it ran as it did in `response.metadata["tool_calling"]`:
the strategy, whether it came from your configuration, the declaration or a
probe, and the categories the probe made must-call.

The same store keeps two facts learned from real requests. A provider that
rejects stop sequences sent beside tool definitions is asked once more with the
stops applied locally; if that answers, later requests leave them off. A server
that rejects a pinned `reasoning_effort` is asked once more without it, and the
field is left off from then on. A small served model asked in words to use a
calculator may still answer by itself; `tool_use="required"` is the remedy for a
run that must use it.

## How It Works

1. Tools are converted to JSON Schema definitions via `tools_to_definitions()`
2. Definitions are passed to the model's chat template via the `tools` parameter
3. The model produces `<tool_call>` tokens in its native format
4. `NativeFunctionCallingStrategy` parses the model-specific format (Qwen, Llama, Mistral, or generic)
5. Tool is executed, result fed back to the model

In `"hybrid"` mode, if native parsing fails, the system falls back to ReAct text parsing automatically.

A model on this path sometimes writes its call as text (`Action: calculator` /
`Action Input: {...}`) instead of making a native call. Nothing has run at that
point, so the agent treats the turn as ending there:

- A tool-holding turn read by the `"hybrid"` or `"react"` reader is sent the
  stop sequence `"\nObservation:"`, so generation ends where the tool's result
  would begin. Only that one label is sent on the native path; the labels that
  would cut ordinary prose (`"\nQuestion:"` and the like) are not. A
  `stop_sequences` you pass yourself replaces it.
- Whatever the stop sequence did, the reader runs the written call and discards
  anything the model wrote after it — a result it made up under `Observation:`,
  or an answer built on one. The log line
  `[reader] a written action is run and what the model wrote after it is discarded`
  marks each time this happens.
- An adapter whose provider rejects `stop` beside `tools` says so with
  `supports_stop_with_tools()` returning `False`. Such a request then goes out
  without stop sequences and the returned text is cut at them instead.
