# Groq Backend

Groq provides extremely fast LLM inference via custom hardware (LPU). effGen's `GroqAdapter` wraps the Groq SDK and supports all chat models with streaming, native tool calling, rate-limit coordination, and cost tracking.

## Setup

```bash
pip install "effgen[groq]"
```

Set your API key (free at [console.groq.com](https://console.groq.com)):

```bash
export GROQ_API_KEY="gsk_..."
```

Or put it in your `.env` file.

## Quick Start

```python
from effgen.models.groq_adapter import GroqAdapter

adapter = GroqAdapter("openai/gpt-oss-20b")
adapter.load()

result = adapter.generate("What is the capital of France?")
print(result.text)

adapter.unload()
```

### Via `load_model`

```python
from effgen.models import load_model

model = load_model("openai/gpt-oss-120b", provider="groq")
result = model.generate("Explain quantum entanglement in one sentence.")
print(result.text)
model.unload()
```

## Streaming

```python
adapter = GroqAdapter("openai/gpt-oss-20b")
adapter.load()

for chunk in adapter.generate_stream("Count from 1 to 10."):
    print(chunk, end="", flush=True)

adapter.unload()
```

## Native Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"],
        },
    },
}]

adapter = GroqAdapter("openai/gpt-oss-120b")
adapter.load()

result = adapter.generate_with_tools("What is 17 * 23?", tools)
tool_calls = result.metadata["tool_calls"]
print(tool_calls)
# [{"id": "call_...", "type": "function",
#   "function": {"name": "calculator", "arguments": '{"expression": "17 * 23"}'}}]

adapter.unload()
```

The shape of `metadata["tool_calls"]` is the same for every adapter — see
[Tool calls](tool-calls.md).

## Models

Rate limits below are for the Developer plan. Catalog reconciled against Groq's
live model listing on 2026-09-08; `qwen/qwen3.6-27b` is the vision-capable entry.

### Chat Completion Models

| Model | Context | Max Output | Tools | RPM | RPD | TPM | TPD |
|-------|---------|-----------|-------|-----|-----|-----|-----|
| `qwen/qwen3.6-27b` | 131,072 | 16,384 | ✓ | 30 | 1,000 | 8k | — |
| `qwen/qwen3.8-27b` | 131,042 | 16,384 | ✓ | 30 | 1,000 | 8k | — |
| `openai/gpt-oss-120b` | 131,072 | 16,384 | ✓ | 30 | 1,000 | 8k | 200k |
| `openai/gpt-oss-20b` | 131,072 | 16,384 | ✓ | 30 | 1,000 | 8k | 200k |
| `openai/gpt-oss-safeguard-20b` | 131,072 | 4,096 | ✓ | 30 | 1,000 | 8k | 200k |
| `groq/compound` | 131,072 | 8,192 | ✓ | 30 | 250 | 70k | — |
| `groq/compound-mini` | 131,072 | 8,192 | ✓ | 30 | 250 | 70k | — |
| `allam-2-7b` | 4,096 | 2,048 | — | 30 | 7,000 | 6k | 500k |
| `meta-llama/llama-prompt-guard-2-86m` | 512 | 256 | — | 30 | 14,400 | 15k | 500k |
| `meta-llama/llama-prompt-guard-2-22m` | 512 | 256 | — | 30 | 14,400 | 15k | 500k |

### Speech-to-Text (not usable via chat completions)

| Model | RPM | RPD |
|-------|-----|-----|
| `whisper-large-v3` | 20 | 2,000 |
| `whisper-large-v3-turbo` | 20 | 2,000 |

### Text-to-Speech

| Model | RPM | RPD |
|-------|-----|-----|
| `canopylabs/orpheus-v1-english` | 10 | 100 |
| `canopylabs/orpheus-arabic-saudi` | 10 | 100 |

## Rate Limit Handling

The adapter automatically wires a `RateLimitCoordinator` that tracks RPM, RPD, TPM, and TPD windows per model. If a 429 rate-limit response is received, the adapter retries with exponential backoff (up to 6 attempts, capped at 60s delay).

To inspect current rate-limit status:

```python
status = adapter.rate_limit_status()
print(status)  # {"enabled": True, "status": "..."}
```

To disable rate limiting (e.g., in tests):

```python
adapter = GroqAdapter("openai/gpt-oss-20b", enable_rate_limiting=False)
```

## Programmatic Model Discovery

```python
from effgen.models.groq_models import available_models, chat_models, tool_capable_models, model_info

print(available_models())       # all 14 models
print(chat_models())            # 10 chat-capable models
print(tool_capable_models())    # 7 models with native tool support

info = model_info("openai/gpt-oss-120b")
print(info["context"])          # 131072
print(info["supports_native_tools"])  # True
print(info["rpm"])              # 30
print(info["notes"])            # "OpenAI GPT-OSS 120B open weights"
```

The `notes` field tells users about model characteristics; `context`, `max_output`, `rpm`/`rpd`/`tpm`/`tpd` are all machine-readable so the adapter and future router can make decisions automatically.

## Usage Metadata

Every `GenerationResult` returned by `GroqAdapter` includes:

```python
result.metadata = {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170,
    "provider": "groq",
    "cost_usd": 0.00001234, # None when the catalog publishes no rate for the model
    "tool_calls": [...],    # parsed tool calls, if any
}
```

## Cost Tracking

The adapter records every call in `CostTracker` and reports the catalog price as `metadata["cost_usd"]`. Groq serves ids the catalog carries no rate for (`allam-2-7b`, the `groq/compound` systems); those report `cost_usd: None` and read as `unpriced` on every surface, never as a `$0` that would look like a free call.
