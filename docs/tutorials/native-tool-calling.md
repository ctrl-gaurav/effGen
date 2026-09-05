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

## How It Works

1. Tools are converted to JSON Schema definitions via `tools_to_definitions()`
2. Definitions are passed to the model's chat template via the `tools` parameter
3. The model produces `<tool_call>` tokens in its native format
4. `NativeFunctionCallingStrategy` parses the model-specific format (Qwen, Llama, Mistral, or generic)
5. Tool is executed, result fed back to the model

In `"hybrid"` mode, if native parsing fails, the system falls back to ReAct text parsing automatically.
