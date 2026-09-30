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

## 1.2 → 1.3: how a run ends

The tool loop now asks for the answer when a run stops making progress, and
says how a run ended. Code that reads `success`, `output` and `stop_reason`
keeps working; these are the differences a caller can see:

- **`AgentConfig.max_turns_without_progress` defaults to `2`** (it was `None`).
  A run whose turns stop bringing new tool results, or that declares no action
  after a result, is asked for its answer instead of going round to
  `max_iterations`. Set it to `None` — in the config or per
  `run(max_turns_without_progress=None)` — to keep the earlier loop.
- **One closing request before a stuck ending.** A run that would have stopped
  on `loop_detected`, `repeated_tool_result`, `max_iterations_*` or
  `null_final_from_model` while holding tool results is sent its calls and
  results once more without tools. When that reply is an answer the run
  succeeds (`metadata["answer_source"] == "closing_request"`), so it no longer
  raises `RunStoppedError` under the default `raise_on_error=True`.
- **A new stop reason, `tool_failed`**, in `STOPPED_REASONS`: the tools the run
  needed failed on their own side (connection, timeout, HTTP 5xx or 429,
  missing credentials). It raises `RunStoppedError` by default.
- **The per-tool circuit breaker counts only those failures.** A tool that was
  given bad input is no longer refused to later calls and runs.
- **`result.termination`** (`"done"`, `"not_possible"`, `"stuck"`,
  `"tool_failed"`, `"error"`) is new, and `to_dict()` carries it.
- **`Action: (continue reasoning)`** and other bracketed placeholders that name
  no held tool are read as "no action", as `Action: None` already was.
- **A stopped run's `partial` never carries a tool's error message**; a run
  whose only observations were errors has `partial=None`.

See [conventions](api/conventions.md) for the full contract.

## 1.2 → 1.3: what a failed tool call leads to

- **A failure is never a repeated result.** Two calls whose different inputs a
  tool rejects in the same words no longer stop the run's tool use; the next
  call runs. Some runs make one more tool call than before.
- **Four failures in a row on a tool's input withdraw it for the run.** A run
  left with no tool is asked for its answer, and ends `tool_failed` with
  `metadata["error"]["kind"] == "input"` when it writes none — where it used
  to go round to `max_iterations`.
- **An answer that is a tool's error message is not a success.** It is sent
  back once; given again the run stops with `null_final_from_model`. The
  direct calculator result is never a failure, and a failed call no longer
  earns "you have the answer from the tool" near the iteration cap.

## 1.2 → 1.3: how a tool call is read

- **`AgentConfig.recover_lost_tool_calls` defaults to `True`** (it was
  `False`). A tool call written as a Python literal, with raw line breaks
  inside its JSON, with unescaped double quotes inside a string, with its last
  string closed one bracket early, or as a whole
  object followed by text inside its tag, runs instead of being reported; a
  call nothing can read is sent back once, with a call required where the
  provider supports it, before the run reports `written_tool_call`. Some runs
  make more tool calls: the calls the model meant to make now run. When the
  agent holds exactly one code-execution tool, a program in a fenced block
  before an empty call tag runs as that call. `False` — in the config or per
  `run(recover_lost_tool_calls=False)` — restores the 1.2 reader; delegated
  sub-agents inherit the setting.
- **Arguments sent as a string are read.** A provider that returns
  `"arguments": "\"code='…'\""` or `"\"56*3+35\""` gets the call run with
  those values instead of with no arguments; `tool_calls` records carry the
  decoded arguments.
- **A call missing a required argument is not dispatched.** Tool code never
  sees an empty call for parameters it declared required; the model is asked
  for the call again.
- **Positional values** are named in the tool's parameter order; a call with
  more values than the tool has parameters (`calculator(2, 3, 4, 6)`) carries no
  arguments and is asked for again, instead of running with its first value or
  ending the run with an error.
- The capability probe keeps the 1.2 reader, so its stored results keep their
  meaning.

## 1.2 → 1.3: one agent, many conversations at once

- **`run(session=...)` holds the conversation on the call.** Overlapping calls
  on one agent — threads, `run_async()` tasks, streams — each read and write
  only their own session. In 1.2 they swapped `agent.session` and
  `agent.short_term_memory` on the shared object, so overlapping calls could
  read and record each other's turns. Inside a call, `agent.session` and
  `agent.short_term_memory` still name that call's conversation.
- **`stream()` honours `session=`.** It used to ignore it and write the turn to
  the agent's own memory; it now reads that conversation's history and appends
  the streamed turn to it.
- **A run whose input a guardrail blocks** leaves the agent's session as it was;
  in 1.2 a blocked `run(session=...)` left that session on the agent.
- Per-call middleware (`run(middleware=[...])`) applies to that call's model
  and tool calls only, also when another call on the agent overlaps it.

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
