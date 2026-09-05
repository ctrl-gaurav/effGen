# Configuration Reference

## AgentConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | (required) | Agent identifier |
| `model` | `BaseModel \| str` | (required) | Model instance or ID |
| `tools` | `List[BaseTool]` | `[]` | Tools available to the agent |
| `system_prompt` | `str` | `"You are a helpful AI assistant."` | System prompt |
| `max_iterations` | `int` | `10` | Max ReAct loop iterations |
| `temperature` | `float` | `0.7` | Generation temperature |
| `enable_sub_agents` | `bool` | `True` | Allow task decomposition |
| `enable_memory` | `bool` | `True` | Enable conversation memory |
| `enable_streaming` | `bool` | `False` | Enable token streaming |
| `max_context_length` | `int \| None` | `None` | Override model context length |
| `system_prompt_template` | `str \| None` | `None` | Custom prompt template |
| `verbose_tools` | `bool \| None` | `None` | Verbose tool descriptions |
| `fallback_chain` | `Dict \| None` | `None` | Tool fallback mapping |
| `enable_fallback` | `bool` | `True` | Enable fallback chains |
| `cite_sources` | `bool` | `False` | Ask for inline `[1]` citation markers when answering from retrieved passages |
| `tool_contract` | `str \| None` | `None` | What the model is told about the attached tools. `None` selects it from the tools' declared categories; a string is stated verbatim; `""` states nothing |

## Memory Config

Nested in `AgentConfig.memory_config`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `short_term_max_tokens` | `int` | `4096` | Max tokens in short-term memory |
| `short_term_max_messages` | `int` | `100` | Max messages to retain |
| `summarization_threshold` | `float` | `0.8` | Share of `short_term_max_tokens` that triggers summarization |
| `keep_recent_messages` | `int` | `4` | Recent messages never summarized |
| `summary_budget_ratio` | `float` | `0.4` | Share of `short_term_max_tokens` the retained summaries may occupy; older summaries are folded together to stay inside it |
| `long_term_backend` | `str` | `"sqlite"` | Storage backend |
| `long_term_persist_path` | `str \| None` | `None` | Persistence directory |
| `auto_summarize` | `bool` | `True` | Auto-summarize old context |

## YAML Configuration File

```yaml
agent:
  name: my-agent
  model: Qwen/Qwen2.5-3B-Instruct
  temperature: 0.7
  max_iterations: 10
  system_prompt: "You are a helpful assistant."

tools:
  - calculator
  - python_repl
  - web_search

memory:
  short_term_max_tokens: 4096
  long_term_backend: sqlite
  auto_summarize: true
```

Load with:
```python
from effgen import ConfigLoader
config = ConfigLoader().load_config("config.yaml")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `EFFGEN_PLUGINS_DIR` | Custom plugin directory path |
| `EFFGEN_API_KEY` | API server authentication key |
| `EFFGEN_RATE_LIMIT` | Per-client API rate limit in requests/min (unset or `0` = disabled) |
| `CUDA_VISIBLE_DEVICES` | GPU selection for model loading |
| `OPENAI_API_KEY` | OpenAI API key (for OpenAI backend) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude backend) |
| `GOOGLE_API_KEY` | Google API key (for Gemini backend) |
| `CEREBRAS_API_KEY` | Cerebras Cloud API key (for Cerebras backend) |
