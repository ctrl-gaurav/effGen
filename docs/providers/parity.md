# Backend Parity — effGen v0.2.3

All effGen backends are interchangeable for the same task. This page documents which
capabilities each provider supports and the results of the cross-backend parity matrix.

## Parity Test

**Canonical task:** `(17 * 23) + sqrt(144) = 403`

The parity test runs an `Agent` with a `Calculator` tool against every provider using
both ReAct and native tool-calling strategies. All available providers produced the
correct answer (403) in validation.

## Provider Support Matrix

| Provider | ReAct | Native Tools | Streaming | Auth Key | Free Tier |
|----------|-------|-------------|-----------|----------|-----------|
| Cerebras | ✅ | ✅ | ✅ | `CEREBRAS_API_KEY` | ✅ Free |
| Groq | ✅ | ✅ | ✅ | `GROQ_API_KEY` | ✅ Free |
| Together AI | ✅ | ✅ | ✅ | `TOGETHER_API_KEY` | ✅ Free |
| Fireworks | ✅ | ✅ | ✅ | `FIREWORKS_API_KEY` | ✅ Free |
| HuggingFace | ✅ | ✅ | ✅ | `HF_TOKEN` | ✅ Free |
| Gemini | ✅ | ✅ | ✅ | `GOOGLE_API_KEY` | ✅ Free |
| OpenAI | ✅ | ✅ | ✅ | `OPENAI_API_KEY` | 💳 Paid |
| Anthropic | ✅ | ✅ | ✅ | `ANTHROPIC_API_KEY` | 💳 Paid |
| Replicate | ✅ | ⚠️ Limited | ✅ | `REPLICATE_API_TOKEN` | 💳 Paid/Credits |

## Strategy Notes

### ReAct (default)
All providers use the ReAct (Reason + Act) pattern by default. The agent loop generates
text responses following the `Thought: / Action: / Action Input: / Observation:` template.
Works with any model that can follow instructions.

### Native Tool Calling
Providers that expose a function-calling API surface pass tools in structured JSON format,
allowing the model to emit structured tool calls rather than parsing free text. Set
`tool_calling_mode="native"` in `AgentConfig` to use this.

Providers supporting native tools: Cerebras, Groq, Together, Fireworks, HF (Qwen models),
Gemini, OpenAI, Anthropic.

### The message protocol

Taking tool *definitions* as a request parameter is a separate question from carrying a
*conversation* that holds a tool call and its result. `AgentConfig.prompt_protocol="messages"`
sends the run as the conversation it was — a system turn, the task, the model's own reasoning
on the same assistant turn as the call it made, and each tool result answering that call id —
and it reaches only an adapter whose `supports_message_protocol()` says it carries both parts
through to the provider.

Every requested call is answered, including the ones the loop declines to dispatch — a repeat
it already has the result for, a call the loop breaker stopped, a tool the agent does not
hold — because a conversation carrying a call with no reply is rejected outright by a
provider. `AgentThread.unanswered_call_ids()` is the invariant; it is empty on every run
effGen produces.

Carrying the message protocol today, each in its own provider's shape:

| adapter | a tool call travels as | its result travels as |
|---|---|---|
| OpenAI, and every OpenAI-compatible endpoint | `tool_calls` on the assistant message | a `tool` message quoting `tool_call_id` |
| Groq, Together, Cerebras, Fireworks, HF inference | the same | the same |
| Anthropic | a `tool_use` content block | a `tool_result` block in a user turn |
| Gemini | a `function_call` part | a `function_response` part |
| Replicate | `tool_calls`, on the models whose input schema is a message array | a `tool` message quoting `tool_call_id` |

The local chat-template engines — Transformers, vLLM in-process, MLX, GGUF — answer `False`:
they render a prompt from a template and have no tool role to put a result in. A run that
asks for `"messages"` there gets the flat transcript and one log line saying which model and
why — never a rejected request.

## Switching Providers

All providers share the same `Agent` API. To switch, simply swap the adapter:

```python
from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Cerebras
from effgen.models.cerebras_adapter import CerebrasAdapter
model = CerebrasAdapter("gpt-oss-120b")

# Or Groq
# from effgen.models.groq_adapter import GroqAdapter
# model = GroqAdapter("openai/gpt-oss-120b")

# Or via registry
# from effgen.models import load_model
# model = load_model("groq:openai/gpt-oss-120b")

model.load()
agent = Agent(config=AgentConfig(
    name="my_agent",
    model=model,
    tools=[Calculator()],
    max_iterations=8,
))
result = agent.run("What is (17 * 23) + sqrt(144)?")
print(result.output)  # → 403
model.unload()
```

## Error Handling

All providers raise unified exception types on failure:

| Exception | Meaning |
|-----------|---------|
| `ModelAuthError` | Bad or missing API key |
| `ModelNotFoundError` | Model ID does not exist |
| `ModelTimeoutError` | Prediction timed out (Replicate) |
| `ModelUnavailableError` | Model not available on serverless tier (HF) |

```python
from effgen.models.errors import ModelAuthError, ModelNotFoundError

try:
    result = model.generate("hello")
except ModelAuthError as e:
    print(f"Fix your API key: {e}")
except ModelNotFoundError as e:
    print(f"Unknown model: {e}")
```

## Checking Provider Availability

```python
from effgen.models.auth import check_keys

keys = check_keys()
for provider, info in keys.items():
    status = "READY" if info["available"] else "MISSING"
    print(f"{provider:15s}  {status}")
```

Or from the CLI:
```bash
effgen doctor
```

## Streaming

All providers support streaming via `generate_stream()`:

```python
for chunk in model.generate_stream("Count 1 to 5"):
    print(chunk, end="", flush=True)
```

## Rate Limits

Each provider enforces its own rate limits. effGen's `RateLimitCoordinator` tracks
requests/tokens per minute and day, automatically waiting when limits approach.

| Provider | Free RPM | Free TPM |
|----------|----------|----------|
| Cerebras | 30 | 60,000 |
| Groq | 30 | 6,000–20,000 |
| Together | varies | varies |
| Fireworks | varies | varies |
| HuggingFace | varies (serverless) | — |
| Gemini | 10 (2.5-flash-lite) | 250,000 |

## Model Recommendations

For the canonical parity task, these models were validated:

| Provider | Model | Notes |
|----------|-------|-------|
| Cerebras | `gpt-oss-120b` | Fast, free-tier eligible |
| Groq | `openai/gpt-oss-120b` | Best quality/speed on free tier |
| Together | `Qwen/Qwen3.5-9B` | Cheapest tool-capable serverless model |
| Fireworks | `accounts/fireworks/models/kimi-k2p6` | Tool-capable serverless model |
| HuggingFace | `Qwen/Qwen2.5-72B-Instruct` | Best free HF model |
| Gemini | `gemini-2.5-flash-lite` | Fast, 20 req/day free |
| OpenAI | `gpt-4o-mini` | Reliable, low cost |
| Anthropic | `claude-3-haiku-20240307` | Fast Anthropic model |
