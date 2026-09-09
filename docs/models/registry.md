# Provider Registry

The `ProviderRegistry` is a singleton that consolidates all provider and model metadata
so that the router, presets, and CLI can introspect available providers and models from
a single source of truth.

## Overview

When any adapter module is imported, it automatically self-registers with the
`ProviderRegistry`. This means your code only needs to import the adapter (or call
`load_model`) to make the provider visible to the registry.

## Quick Start

```python
from effgen.models.registry import list_providers, list_models, lookup
from effgen.models.auth import check_keys

# See all registered providers
providers = list_providers()
# ['anthropic', 'cerebras', 'fireworks', 'gemini', 'groq', 'hf', 'openai', 'replicate', 'together']

# List all models for a provider (returns dicts with 'model_id' + metadata)
models = list_models("groq")

# Resolve a model by provider:model_id prefix
provider, adapter_cls, info = lookup("groq:openai/gpt-oss-120b")

# Check which API keys are present in the environment
keys = check_keys()
# {'groq': {'available': True, 'env_key': 'GROQ_API_KEY', ...}, ...}
```

## Loading Models with Provider Prefix

`load_model` now supports `"provider:model_id"` syntax to disambiguate models that
appear in multiple providers:

```python
from effgen.models import load_model

# Prefix syntax routes directly to Groq
model = load_model("groq:openai/gpt-oss-20b")

# Equivalent to:
model = load_model("openai/gpt-oss-20b", provider="groq")
```

## `effgen doctor`

The `effgen doctor` command prints a table of all registered providers and whether
their API keys are available in the environment:

```
$ effgen doctor

        effgen doctor — Provider API Key Status
┏━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Provider  ┃ Status      ┃ Env Key Found       ┃ Models ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ anthropic │ MISSING KEY │ —                   │     17 │
│ cerebras  │ READY       │ CEREBRAS_API_KEY    │      4 │
│ fireworks │ READY       │ FIREWORKS_API_KEY   │     80 │
│ gemini    │ READY       │ GOOGLE_API_KEY      │     16 │
│ groq      │ READY       │ GROQ_API_KEY        │     16 │
│ hf        │ READY       │ HF_TOKEN            │    124 │
│ openai    │ MISSING KEY │ —                   │     35 │
│ replicate │ READY       │ REPLICATE_API_TOKEN │     38 │
│ together  │ READY       │ TOGETHER_API_KEY    │    149 │
└───────────┴─────────────┴─────────────────────┴────────┘

Missing keys — set in ~/.effgen/.env or export:
  export ANTHROPIC_API_KEY=<your-key>
  export OPENAI_API_KEY=<your-key>
```

Options:
- `--json` — output as JSON
- `--provider <name>` — check a specific provider only

## ProviderRegistry API

### `ProviderRegistry.register(provider_name, adapter_cls, models, env_keys)`

Register a provider. Idempotent — safe to call multiple times.

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider_name` | `str` | Short lowercase label, e.g. `"groq"` |
| `adapter_cls` | `type` | The adapter class (subclass of `BaseModel`) |
| `models` | `dict[str, dict]` | Model registry dict `{model_id: info_dict}` |
| `env_keys` | `list[str]` | Env vars to check for API key availability |

### `ProviderRegistry.list_providers() -> list[str]`

Returns a sorted list of all registered provider names.

### `ProviderRegistry.list_models(provider) -> list[dict]`

Returns a list of model info dicts for the given provider. Each dict includes
the `model_id` key plus all metadata from the provider's model registry.

Raises `KeyError` if the provider is not registered.

### `ProviderRegistry.lookup(model_id, provider=None) -> (str, type, dict)`

Resolves a model ID to `(provider_name, adapter_cls, model_info)`.

- Supports `"provider:model_id"` prefix syntax.
- Raises `AmbiguousModelError` if the model exists in multiple providers and
  no provider is specified.
- Raises `KeyError` if the model ID is not found.

### `ProviderRegistry.reset() -> None`

Returns the registry to the state effGen starts in: every registration dropped
— including the circuit breaker and bulkhead held per provider — and the
built-in provider adapters registered again. Model lookups keep working
afterwards.

### `ProviderRegistry.clear() -> None`

Removes every registration and leaves the registry empty. Model lookups raise
until something registers; `reset()` brings the built-in providers back.

### `ProviderRegistry.register_builtins() -> list[str]`

Registers the provider adapters that ship with effGen and returns their names.
Adapter modules register themselves the first time they are imported, so
importing them again after the registry was emptied does nothing — this calls
their registration hooks directly. Idempotent.

### `ProviderRegistry.snapshot() -> dict` / `ProviderRegistry.restore(snapshot)`

Save and put back the exact set of registrations. `restore()` also undoes edits
made inside a provider record — an added or removed model, a changed capability
set, a tripped circuit breaker — which makes it the way to isolate registry
state in a test. A provider's model catalog is restored in the dict its adapter
module defines, so the adapter and the registry agree afterwards:

```python
snapshot = ProviderRegistry.snapshot()
ProviderRegistry.register("my_provider", MyAdapter, MY_MODELS, env_keys=["MY_KEY"])
...
ProviderRegistry.restore(snapshot)
```

## `check_keys(providers=None) -> dict`

Returns a mapping of `provider → {available, env_key, env_keys_checked}`.

```python
from effgen.models.auth import check_keys

result = check_keys()
if result["groq"]["available"]:
    print("Groq is ready!")
else:
    print("Set GROQ_API_KEY to use Groq")
```

## `AmbiguousModelError`

Raised when a model ID exists in multiple providers and no `provider:` prefix
or `provider=` kwarg disambiguates it:

```python
from effgen.models.errors import AmbiguousModelError
from effgen.models.registry import lookup

try:
    lookup("openai/gpt-oss-120b")   # served by Groq, HF, Replicate and Together
except AmbiguousModelError as e:
    print(e.model_id)    # "openai/gpt-oss-120b"
    print(e.providers)   # ["groq", "hf", "replicate", "together"]
    # Fix: be explicit
    prov, cls, info = lookup("groq:openai/gpt-oss-120b")
```

## Supported Providers

| Provider | Env Key | Models |
|----------|---------|--------|
| `anthropic` | `ANTHROPIC_API_KEY` | 17 |
| `cerebras` | `CEREBRAS_API_KEY` | 4 |
| `fireworks` | `FIREWORKS_API_KEY` | 80 |
| `gemini` | `GOOGLE_API_KEY` | 16 |
| `groq` | `GROQ_API_KEY` | 16 |
| `hf` | `HF_TOKEN` or `HUGGINGFACE_API_KEY` | 124 |
| `openai` | `OPENAI_API_KEY` | 35 |
| `replicate` | `REPLICATE_API_TOKEN` | 38 |
| `together` | `TOGETHER_API_KEY` | 149 |
