# Architecture Guide

## System Overview

```
                    ┌─────────────────────────┐
                    │       User Input         │
                    │  (CLI / API / Python)    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │         Agent            │
                    │    (ReAct Loop)          │
                    │                         │
                    │  Thought → Action →     │
                    │  Observation → ...      │
                    └──┬──────┬──────┬────────┘
                       │      │      │
              ┌────────▼┐  ┌──▼───┐ ┌▼────────┐
              │  Model   │  │Tools │ │ Memory  │
              │ Backend  │  │      │ │         │
              └────────┬┘  └──┬───┘ └┬────────┘
                       │      │      │
    ┌──────────────────┼──────┼──────┼──────────┐
    │ Backends:        │      │      │          │
    │ - Transformers   │  Built-in   │ Short    │
    │ - vLLM           │  + Plugins  │ + Long   │
    │ - OpenAI API     │  + MCP      │ + Vector │
    │ - Anthropic API  │  + A2A/ACP  │          │
    │ - Gemini API     │             │          │
    └──────────────────┘─────────────┘──────────┘
```

## Core Components

### Agent (`effgen/core/agent.py`)

The central class. Implements the ReAct reasoning loop:

1. Receives a task from the user
2. Generates a **Thought** (reasoning about what to do)
3. Selects an **Action** (tool to call) with parameters
4. Receives an **Observation** (tool result)
5. Repeats until a **Final Answer** is reached or max iterations hit

Key features:
- Sub-agent decomposition for complex tasks
- Streaming output support
- Memory integration (short-term, long-term, vector)
- Configurable via `AgentConfig` dataclass

### Model Backends (`effgen/models/`)

Abstraction over multiple LLM backends:

| Backend | File | Use Case |
|---------|------|----------|
| `TransformersEngine` | `transformers_engine.py` (+ siblings) | Local GPU inference (default) |
| `VLLMEngine` | `vllm_engine.py` | High-throughput serving |
| `OpenAIAdapter` | `openai_adapter.py` | OpenAI API models |
| `AnthropicAdapter` | `anthropic_adapter.py` | Claude models |
| `GeminiAdapter` | `gemini_adapter.py` | Google Gemini models |
| `MLXEngine` | `mlx_engine.py` | Apple Silicon (MLX) |
| `MLXVLMEngine` | `mlx_vlm_engine.py` | Apple Silicon vision-language |
| `GGUFEngine` | `gguf_engine.py` | GGUF quantized models (llama-cpp) |

All implement `BaseModel` with: `generate()`, `generate_stream()`, `count_tokens()`, `get_context_length()`, `load()`, `unload()`, `supports_tool_calling()`, `tool_call_support()`

`tool_call_support()` names the mechanism behind the boolean — `"api"` for
provider-side tool calling, `"template"` when a local chat template renders the
definitions into the prompt, `"none"` for neither. It defaults to
`"api" if supports_tool_calling() else "none"`, so an adapter that implements
only the boolean needs no change. The chat-template engines override it, and
report `supports_tool_calling()` as True only when the template actually renders
the definitions — a template that accepts a `tools` argument and discards it
gives the model nothing to call.

Additional model infrastructure:
- `router.py`: `ModelRouter` — automatic model selection by query complexity
- `capabilities.py`: `MODEL_CAPABILITIES` — pre-populated profiles for 12+ models
- `pool.py`: `ModelPool` — LRU eviction, GPU memory management, hot-swap
- `lazy.py`: `LazyModel` — deferred loading until first use
- `batching.py`: `ContinuousBatcher` — coalesces concurrent requests

The two largest of these are each composed from siblings, so
`effgen.models.transformers_engine` and `effgen.models.model_loader` stay the one
import path for everything they publish:

- `transformers_engine.py` holds the engine class, its constructor and its
  capability reports, and composes it from `transformers_engine_support` (cache
  and offline detection, special-token stripping), `_placement` (device maps and
  CUDA-fault recovery), `_sampling` (generation config and chat templates),
  `_loading` (weights, quantization, teardown), `_generation` (`generate` and
  `generate_batch`) and `_streaming` (`generate_stream`)
- `model_loader.py` holds `ModelLoader`, its model tables and the `load_model()`
  function, and composes the class from `model_loader_routing` (model-id and
  prefix resolution), `_cloud` (provider adapters), `_local` (local engines) and
  `_capacity` (VRAM, quantization and tensor-parallel choices). Every adapter and
  engine import stays inside the function that constructs it, which is what keeps
  `from effgen import Agent` free of torch

### Tools (`effgen/tools/`)

- `base_tool.py`: `BaseTool` abstract class with metadata and validation
- `registry.py`: `ToolRegistry` for discovery, lazy loading, dependency management
- `builtin/`: 66 built-in tools (core, finance, data science, DevOps, knowledge, communication)
- `plugin.py`: External plugin loading via entry points
- `protocols/`: MCP, A2A, ACP protocol implementations

### Guardrails (`effgen/guardrails/`)

Safety and validation framework:
- `base.py`: `Guardrail` ABC, `GuardrailChain`, `GuardrailPosition`
- `content.py`: `ToxicityGuardrail`, `PIIGuardrail`, `LengthGuardrail`, `TopicGuardrail`
- `injection.py`: `PromptInjectionGuardrail` (low/medium/high sensitivity)
- `tool_safety.py`: `ToolInputGuardrail`, `ToolOutputGuardrail`, `ToolPermissionGuardrail`
- `presets.py`: `get_guardrail_preset()` — strict/standard/minimal/none

### RAG (`effgen/rag/`)

Production RAG pipeline:
- `ingest.py`: `DocumentIngester` — multi-format document loading with deduplication
- `chunking.py`: `SemanticChunker`, `CodeChunker`, `TableChunker`, `HierarchicalChunker`
- `search.py`: `HybridSearchEngine` — dense + BM25 + keyword + metadata via Reciprocal Rank Fusion
- `reranker.py`: `CrossEncoderReranker`, `LLMReranker`, `RuleBasedReranker`
- `context_builder.py`: `ContextBuilder` — token budget management with citations
- `attribution.py`: `Citation`, `CitationTracker`

### Evaluation (`effgen/eval/`)

- `evaluator.py`: `AgentEvaluator`, `TestCase`, `EvalResult`, `SuiteResults`
- `suites.py`: 5 built-in suites (math, tool_use, reasoning, safety, conversation)
- `regression.py`: `RegressionTracker` — baseline comparison with severity alerts
- `comparison.py`: `ModelComparison` — multi-model matrix benchmarking

### Memory (`effgen/memory/`)

Three tiers:
1. **ShortTermMemory**: Recent conversation context (token-limited)
2. **LongTermMemory**: Persistent facts across sessions (SQLite)
3. **VectorMemoryStore**: Semantic search over past interactions
- `token_budget.py`: `TokenBudget` — smart context window allocation

### Cache (`effgen/cache/`)

- `prompt_cache.py`: `PromptCache` — LRU + TTL with sha256 fingerprinting
- `result_cache.py`: `ResultCache` — per-tool TTL, optional semantic similarity

### Orchestration (`effgen/core/`)

- `message_bus.py`: `MessageBus` — pub/sub inter-agent communication
- `workflow.py`: `WorkflowDAG` — DAG execution with conditional branching
- `shared_state.py`: `SharedState` — thread-safe namespaced key-value store
- `lifecycle.py`: `AgentRegistry`, `AgentPool` — lifecycle management
- `checkpoint.py`: `CheckpointManager` — save/restore agent state
- `session.py`: `Session`, `SessionManager` — persistent conversations
- `human_loop.py`: `HumanApproval`, `HumanInput`, `HumanChoice`
- `batch.py`: `BatchRunner` — concurrent batch execution
- `execution_tracker.py`: `ExecutionTracker` — the run's event stream, the
  execution tree built from it, and the trace behind `run --explain` and chat's
  `/trace`. It holds the class, its constructor and its `__repr__`, and composes
  it from `execution_tracker_state` (event intake, parent links, the tree, live
  status), `_metrics` (summary, performance metrics, bottlenecks, critical path)
  and `_render` (display formatting and JSON/CSV/HTML export); the event types
  and the three dataclasses live in `execution_tracker_events`

### Observability (`effgen/observability/`)

- `tracing.py`: the OTel span layer — the `start_*` context managers every hot
  path opens, and the in-memory span stream the dashboard reads. It stays the one
  import path for all of it and composes the implementation from `tracing_otel`
  (the single optional-SDK import), `tracing_samplers` (the five samplers),
  `tracing_provider` (the `TracerProvider` lifecycle and the no-op tracer),
  `tracing_buffer` (the span ring buffer and the run/execution context vars) and
  `tracing_spans` (span construction and the outcome helpers). Not to be confused
  with `spans.py`, which declares the span- and attribute-name constants
- `metrics.py`, `logs.py`, `slo.py`, `alerts.py`: Prometheus series, structured
  logging with secret redaction, SLO tracking and alert delivery
- `topology.py`, `run_log.py`: the live multi-agent graph and the run history the
  dashboard reads, both built from the span stream

### Hardware (`effgen/hardware/`)

- `platform.py`: Apple Silicon/CUDA/MLX detection and backend recommendation

### Debug (`effgen/debug/`)

- `inspector.py`: `DebugAgent` — rich TUI step-through with `DebugTrace`

### API Server (`effgen/api/`)

- `openai_compat.py`: OpenAI-compatible `/v1/chat/completions` and `/v1/completions`
- `embeddings.py`: `/v1/embeddings` endpoint with caching
- `queue.py`: `RequestQueue` — priority-based with backpressure
- `pool.py`: `AgentPool` — pre-warmed agent instances
- `tenancy.py`: `TenantManager`, `APIKey` management
- `middleware.py`: CORS, request ID, GZip, graceful shutdown

### Client SDK (`effgen/client/`)

- `client.py`: `EffGenClient` — sync/async with retries, streaming, typed exceptions

### CLI (`effgen/cli/`)

- `_main.py`: argument parsing, the `_dispatch` routing table, the `effgen` /
  `effgen-agent` / `effgen-web` entry points, and `CLIInterface`. It re-exports
  every command's handler, so `effgen.cli._main` stays the one import path
- `commands/`: one module per command — the body of `run`, `chat`, `serve`,
  `doctor`, `quickstart`, `prompts`, `batch`, `eval`, `compare` and the rest
- `parsers/`: each command's `add_argument` declarations, grouped by family
  (`agent`, `catalog`, `history`, `jobs`, `ops`) and assembled by `create_parser`
- `_console.py`: `CLIConsoleMixin` — the print/render methods `CLIInterface` uses
- `_logging.py`: `--verbose`/`--quiet`/`--log-file` level policy
- `code/`: the `effgen code` session — REPL, engine, permissions, rendering.
  `code/repl.py` holds the session and composes `CodeREPL` from four siblings —
  `repl_commands` (the slash-command table and dispatcher), `repl_session`,
  `repl_turn` and `repl_view` — so `effgen.cli.code.repl` stays the one import path
- `chat.py`: the `effgen chat` session. It holds what the session is — the
  constructor, the agent it builds and rebuilds, readline and the loop — and
  composes `ChatREPL` from four siblings: `chat_commands` (the slash-command
  table, the dispatcher, and the commands that report or reset state),
  `chat_session` (`/model`, `/save`, `/load`, `/session`, and the history
  directory `effgen code` shares), `chat_turn` (the streamed and tool-bearing
  answer paths) and `chat_view` (banner, prompt, answer surface, footer), so
  `effgen.cli.chat` stays the one import path. Not to be confused with
  `commands/chat.py`, which validates `--provider` and launches this
- `monitor.py`: `effgen top` / `effgen monitor`. It holds the command body and the
  interval bounds it validates, and composes the rest from `monitor_collect` (the
  five sources, read into one snapshot document), `monitor_format` (the scope
  labels and the value formatters both rendering paths share), `monitor_panels`
  (the static labelled tables a pipe or `--once` gets) and `monitor_live` (the
  full-screen layout and the refresh loop), so `effgen.cli.monitor` stays the one
  import path

### Prompts (`effgen/prompts/`)

- `TemplateManager`: Prompt template management
- `ChainManager`: Prompt chaining
- `PromptOptimizer`: SLM-specific prompt optimization
- `AgentSystemPromptBuilder`: Auto-generates system prompts from tools —
  the role, the operational hints for the categories held, and the common
  mistakes list
- `tool_contract`: what a model is told to do with the tools attached to it,
  selected from their declared `ToolCategory` and stated by every tool-calling
  path

### Domains (`effgen/domains/`)

- `base.py`: `Domain` — keywords, system_prompt, tool_names, guardrails;
  `Domain.to_agent(model)` builds a runnable agent (or `create_agent(domain=...)`)
- `expander.py`: `KeywordExpander` — WordNet/template/LLM expansion (domain
  presets carry field-appropriate query templates)
- 5 built-in: `TechDomain`, `ScienceDomain`, `FinanceDomain`, `HealthDomain`, `LegalDomain`

### Configuration (`effgen/config/`)

YAML/JSON configuration loading with validation and defaults.

## Data Flow

```
User Query
    │
    ▼
AgentConfig (model, tools, prompts, memory)
    │
    ▼
Agent.__init__() → loads model, initializes tools & memory
    │
    ▼
Agent.run(task) → enters ReAct loop
    │
    ├──▶ Model.generate(prompt) → raw LLM output
    │        │
    │        ▼
    │    Parse: extract Thought, Action, Action Input
    │        │
    │        ▼
    │    Tool._execute(**params) → observation
    │        │
    │        ▼
    │    Append to context, check for Final Answer
    │        │
    │        └──▶ (loop back to generate)
    │
    ▼
AgentResponse (output, stats, trace)
```

## Plugin Architecture

Plugins are discovered from three sources:
1. Python entry points (`effgen.plugins` group)
2. User plugin directory (`~/.effgen/plugins/`)
3. Environment variable (`EFFGEN_PLUGINS_DIR`)

Each plugin provides a `ToolPlugin` subclass that registers `BaseTool` implementations into the global `ToolRegistry`.
