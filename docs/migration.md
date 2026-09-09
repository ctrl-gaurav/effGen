# Migration Guide

## Coming from the OpenAI SDK / LangChain

effGen ships an OpenAI-compatible HTTP server, so most code that already talks to
the OpenAI API works by changing only the `base_url`. See
[`server/openai-compat.md`](server/openai-compat.md) for the full endpoint,
alias, streaming, and error-status reference.

### Point the official `openai` client at effGen

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_EFFGEN_API_KEY")

resp = client.chat.completions.create(
    model="openai:gpt-5-nano",                 # route by provider:model
    messages=[{"role": "user", "content": "Summarize the CAP theorem."}],
)
print(resp.choices[0].message.content)
```

- **Routing.** Send `provider:model` (`groq:openai/gpt-oss-20b`,
  `gemini:gemini-3.1-flash-lite`) or `provider/model`; a bare local id
  (`transformers:Qwen/Qwen2.5-1.5B-Instruct`) also loads. `effgen-default`
  routes to the server's configured default model. OpenAI flagship names
  (`gpt-4o-mini`, `gpt-3.5-turbo`) resolve to local models; the response's
  non-standard `effgen` object reports `resolved_model` and `alias_applied`.
- **Streaming** works unchanged, including `stream_options={"include_usage":
  True}` for a final usage chunk. `response_format={"type": "json_object"}`,
  legacy `/v1/completions`, and `/v1/embeddings` are supported.
- **Errors** map to the OpenAI status/type contract: unknown provider → 400,
  bad key → 401, unknown model → 404, rate limit → 429, upstream key
  missing/rejected → 503/502. `except openai.APIStatusError` code carries over.
- **Cost** rides along on each response as the `effgen` extension
  (`resp.effgen["cost_usd"]` for priced models); OpenAI-only clients ignore it.

### Tools run server-side

This is the one place the protocol differs. effGen executes its **own**
registered tools on the server and returns the final answer; it does **not**
forward client-defined function tools for the caller to run, and it does not
emit client-side `tool_calls` deltas. Request a registered tool by name:

```python
resp = client.chat.completions.create(
    model="groq:openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    tools=[{"type": "function", "function": {"name": "calculator"}}],
)
```

An unregistered tool name is refused with a 400 that points at
`effgen tools list`.

### Native client

For a lighter dependency than the `openai` SDK, `effgen.client.EffGenClient`
speaks the same server:

```python
from effgen.client import EffGenClient

c = EffGenClient(base_url="http://localhost:8000", api_key="YOUR_EFFGEN_API_KEY")
print(c.chat("Hello").content)                              # default model
print(c.chat("What is 17*23?", tools=["calculator"]).content)  # tool by name
```

### Coming from LangChain

Point `ChatOpenAI` at the effGen server the same way:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_EFFGEN_API_KEY",
    model="openai:gpt-5-nano",
)
```

Chains and prompt templates that call the model over the OpenAI protocol keep
working. Move any client-side LangChain tool that must run on the server into an
effGen registered tool (see `effgen tools list` and the tool authoring guide).

---

## v0.1.x → v0.2.0

### Breaking Changes

**None.** All existing `Agent`, `AgentConfig`, `load_model`, and tool APIs work without modification. v0.2.0 is fully backwards compatible.

### New AgentConfig Parameters (All Optional)

```python
config = AgentConfig(
    name="my_agent",
    model=model,
    tools=[Calculator()],

    # New in v0.2.0 (all optional, defaults preserve v0.1.x behavior):
    tool_calling_mode="auto",        # "auto", "native", "react", "hybrid"
    output_format=None,              # "json", "text", or None
    output_schema=None,              # JSON Schema dict
    guardrails=None,                 # GuardrailChain instance
    models=None,                     # List of additional models for routing
    speculative_execution=False,     # Run on 2 models, take fastest
    approval_mode="never",           # "never", "always", "first_time", "dangerous_only"
    approval_callback=None,          # Callable for human approval
    approval_timeout=60,             # Seconds to wait for approval
    session_id=None,                 # Persistent session ID
    checkpoint_interval=None,        # Checkpoint every N iterations
    checkpoint_dir=None,             # Directory for checkpoints
)
```

### New Agent.run() Parameters (All Optional)

```python
result = agent.run(
    "What is 2+2?",
    output_schema={"type": "object", ...},  # Per-call JSON schema
    output_model=MyPydanticModel,            # Per-call Pydantic model
    debug=True,                              # Capture DebugTrace
    checkpoint_interval=3,                   # Per-call checkpoint interval
)
```

### New AgentResponse Fields

```python
result = agent.run("query")
result.citations    # List[Citation] — RAG source citations (empty if no RAG)
result.sources      # List[str] — deduplicated source names
result.metadata["debug_trace"]  # DebugTrace (when debug=True)
result.metadata["parsed_output"]  # Pydantic model (when output_model used)
```

### New Modules

| Module | Import | Purpose |
|--------|--------|---------|
| Guardrails | `from effgen.guardrails import ...` | Safety & validation |
| RAG | `from effgen.rag import ...` | Retrieval Augmented Generation |
| Evaluation | `from effgen.eval import ...` | Benchmarking & regression |
| Domains | `from effgen.domains import ...` | Domain keyword expansion |
| Cache | `from effgen.cache import ...` | Prompt & result caching |
| Debug | `from effgen.debug import ...` | Interactive debugging |
| Hardware | `from effgen.hardware import ...` | Platform detection |
| Client SDK | `from effgen.client import ...` | API client |

### New Tools (17 Added)

Finance: `StockPriceTool`, `CurrencyConverterTool`, `CryptoTool`
Data Science: `DataFrameTool`, `PlotTool`, `StatsTool`
DevOps: `GitTool`, `DockerTool`, `SystemInfoTool`, `HTTPTool`
Knowledge: `ArxivTool`, `StackOverflowTool`, `GitHubTool`, `WolframAlphaTool`
Communication: `EmailDraftTool`, `SlackDraftTool`, `NotificationTool`

All imported from `effgen.tools.builtin`.

### New Optional Dependencies

```bash
pip install effgen[rag]       # sentence-transformers, faiss-cpu
pip install effgen[finance]   # yfinance
pip install effgen[data]      # matplotlib, plotly
pip install effgen[eval]      # rouge-score, nltk
pip install effgen[gguf]      # llama-cpp-python
pip install effgen[mlx]       # MLX for Apple Silicon
pip install effgen[mlx-vlm]   # MLX vision-language models
```

### New CLI Commands

```bash
# Workflows
effgen workflow run pipeline.yaml
effgen workflow validate pipeline.yaml

# Batch execution
effgen batch --input queries.jsonl --output results.jsonl

# Evaluation
effgen eval --suite math --model "Qwen/Qwen2.5-3B-Instruct"
effgen compare --models "model_a,model_b" --suite math

# Model management
effgen models load "Qwen/Qwen2.5-3B-Instruct"
effgen models status
effgen models unload "Qwen/Qwen2.5-3B-Instruct"

# Sessions
effgen sessions list
effgen sessions delete <id>
effgen sessions export <id>

# Debugging
effgen debug --preset math "What is 2+2?"

# Checkpointing
effgen run "Long task" --checkpoint-dir ./checkpoints
effgen resume --checkpoint ./checkpoints/latest.json
```

### API Server v2 Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat (new) |
| `POST /v1/completions` | OpenAI-compatible text completion (new) |
| `POST /v1/embeddings` | OpenAI-compatible embeddings (new) |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `WS /ws` | WebSocket streaming |

Model aliases map OpenAI model names to local SLMs (e.g., `gpt-3.5-turbo` → `Qwen2.5-3B-Instruct`).

---

## 0.0.2 → 0.1.0

### New Features
- **Presets**: Use `create_agent("math", model)` for instant agent setup
- **Plugin system**: Distribute tools as installable packages
- **CLI**: `--preset`, `--explain`, `--completion`, `create-plugin` commands
- **API server**: WebSocket streaming, API key auth, rate limiting, metrics
- **Tab completion**: `eval "$(effgen --completion bash)"`

### Breaking Changes
None. All existing `Agent`, `AgentConfig`, and `load_model` APIs remain unchanged.

### New Imports
```python
# Presets (new)
from effgen.presets import create_agent, list_presets

# Plugin system (new)
from effgen.tools.plugin import ToolPlugin, PluginManager, discover_plugins
```

### CLI Changes
```bash
# New commands
effgen presets                              # List available presets
effgen run --preset math "What is 2+2?"     # Use preset
effgen run --explain "..."                  # Show tool reasoning
effgen create-plugin my_tools               # Generate plugin scaffold
effgen --completion bash                    # Print completion script
```

### API Server Changes
- New endpoints: `WS /ws`, `GET /metrics`
- Auth: Set `EFFGEN_API_KEY` environment variable
- Rate limiting: Set `EFFGEN_RATE_LIMIT` to a requests/min cap per client (unset or `0` = disabled)
- `POST /run` now accepts `preset` field
