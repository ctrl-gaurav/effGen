# Tool calls

Every model adapter reports the tool calls a turn made in one shape, under
`GenerationResult.metadata["tool_calls"]`. One reader works across OpenAI,
Anthropic, Gemini, Cerebras, Groq, Together, Fireworks, Replicate and the
HuggingFace Inference API — the provider SDKs disagree about the shape, the
adapters do not.

```python
[
    {
        "id": "call_NdryBNer9yvtPNZpQNaSncrs",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression":"6*7"}',
        },
    },
]
```

## The rules

1. **The key is always present and always a list.** A turn that called nothing
   reports `[]`, so a reader needs no `KeyError` guard — including on the local
   engines (`transformers`, `vllm`, `gguf`, `mlx`), which report tool calls as
   text in `result.text` rather than as a structured list.
2. **Every element carries `id`, `type` and `function`.** `id` is the
   provider's call id, or `""` when the provider sends none. `type` is
   `"function"` for a function call. `function` carries `name` and `arguments`.
3. **`arguments` is a JSON string, exactly as the model generated it.** The
   adapter never parses it. That is what the OpenAI-compatible wire format
   requires (`arguments` is typed `str` in the OpenAI SDK), and it keeps a
   model's malformed JSON visible instead of silently arriving as an empty
   argument set.
4. **Order is the provider's order.** Parallel calls in one turn are separate
   elements.

## Reading a call

```python
import json

from effgen import load_model

TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

model = load_model("groq:openai/gpt-oss-20b")
result = model.generate_with_tools("What is 6*7? Use the calculator.", tools=TOOLS)

for call in result.metadata["tool_calls"]:
    name = call["function"]["name"]
    try:
        arguments = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError:
        # The model wrote invalid JSON. The raw text is still here to report.
        print(f"{name} was called with unparseable arguments: "
              f"{call['function']['arguments']!r}")
        continue
    print(call["id"], name, arguments)
```

Swap the model id for `openai:gpt-5-nano`, `gemini:gemini-3.1-flash-lite`,
`cerebras:gpt-oss-120b`, `together:...`, `fireworks:...` or `hf:...` and the
same loop runs unchanged.

Most callers never need this list. `Agent` dispatches the calls itself and
reports the count as `AgentResponse.tool_calls` (an `int`); this shape matters
when you drive an adapter directly, or when you forward the list onto an
OpenAI-compatible wire from a custom server runner.

Feeding the result back is portable too. Re-submitting the assistant turn plus
a tool result uses each provider's own wire format — a `role: "tool"` message on
the OpenAI-compatible providers, `functionResponse` parts on Gemini,
`tool_result` blocks on Anthropic — so every adapter builds its own:

```python
result = adapter.chat(messages=messages, tools=TOOLS)
messages.append(adapter.build_assistant_message(result))
for call in result.metadata["tool_calls"]:
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
    messages.append(
        adapter.build_tool_result_message(call["id"], name, run_tool(name, args))
    )
```

That loop is unchanged across providers; swapping the adapter is the only edit.
`Agent` still dispatches the calls for you and is the easier route — these are
for a caller driving `chat()` / `generate_with_tools()` by hand.

## Provider-specific additions

Two adapters carry extra keys **beside** the block above. The rules hold for
them too — the additions are read only by callers that ask for them.

| Provider | Addition | Why |
|---|---|---|
| Gemini | top-level `name` and `arguments` (the parsed mapping) | Gemini previously reported only this flat form; the keys stay for one release so callers written against it keep working. Prefer the nested block. |
| Anthropic | `metadata["tool_uses"]`, a separate list of `{id, name, input}` | Anthropic's own `tool_use` block shape, with `input` already parsed. |

The OpenAI Responses API path (`generate_with_native_tools`) reports its
entries in `metadata["native_tool_results"]` — the same list object as
`metadata["tool_calls"]`. Those entries keep the Responses API's own
`type: "function_call"` discriminator and its flat `name`/`arguments` keys,
because the list also carries server-side tool results (`web_search_call`,
`file_search_call`) that are not function calls at all. The nested `function`
block is present on the function-call entries as well.

## Streaming

`generate_stream()` yields text. An adapter that streams a tool call
accumulates its `arguments` deltas and finishes the call in the shape above —
the accumulated JSON string, unparsed — so a streamed call and the same call
made without streaming agree. To read the structured list, make the call
without streaming and read `metadata["tool_calls"]`.
