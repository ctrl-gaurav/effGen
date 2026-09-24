# Any server that speaks the OpenAI protocol

vLLM, SGLang, TGI, llama.cpp's server, Ollama, LM Studio, LiteLLM and most
corporate gateways all expose the OpenAI chat-completions API. Point effGen at
one with `base_url` and it drives the model you are already serving instead of
loading a second copy of the weights in the agent's process.

```python
from effgen.models import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)
print(model.generate("What is 6 times 7?").text)
```

Or straight from an agent, with no loader call:

```python
from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
))
print(agent.run("What is 6 times 7?").output)
```

## Why serve the model separately

Loading in-process means one copy of the weights per agent process, no sharing
between agents, no continuous batching across callers, and a GPU tied to the
agent's lifetime. A shared server fixes all four: the weights load once, every
caller's requests batch together, and the GPU outlives any individual run.

It is also the only way to have several frameworks — or several versions of
your own service — generate under identical settings, which is what a fair
comparison needs.

## Where the endpoint comes from

In order:

1. `base_url=` passed to `load_model()`, `AgentConfig` or the adapter;
2. `EFFGEN_BASE_URL`;
3. `OPENAI_BASE_URL`;
4. `OPENAI_API_BASE`.

effGen's own variable is consulted first, so you can point effGen at a server
without redirecting every other OpenAI client on the machine.

`provider="openai"` **with** a `base_url` also routes here, because a URL of
your own means the model ids, the context window and the pricing are the
server's rather than OpenAI's. Without one it stays on OpenAI, so a machine-wide
`OPENAI_BASE_URL` set for something unrelated cannot silently reroute a plain
OpenAI call — ask for `provider="openai_compatible"` to pick the environment up.

These all reach the same adapter:

```python
load_model(model_id, provider="openai_compatible", base_url=URL)
load_model(model_id, base_url=URL)                    # base_url is the whole instruction
load_model(model_id, provider="openai", base_url=URL, api_key="EMPTY")
load_model(f"openai_compatible:{model_id}", base_url=URL)
load_model(f"openai:{model_id}", base_url=URL)
```

In every one of them the server is asked for `model_id`: the `openai:` or
`openai_compatible:` prefix names the adapter and is not sent, while any slash
inside the id — `org/name`, `openai/<name>` — is part of the id and is. A prefix
naming a different provider (`together:<org>/<name>` with a `base_url`) is sent
as written, and the server answers that it has no such model.

`"openai-compatible"`, `"openai_compat"`, `"compatible"`, `"server"`,
`"vllm_server"` and `"local_server"` are accepted spellings of the provider.

## Credentials

A local server that authenticates nothing needs none — effGen sends a
placeholder, which vLLM, SGLang, TGI, llama.cpp and Ollama all accept. Pass a
real one for a gateway that checks it:

```python
model = load_model(
    "my-model", provider="openai_compatible",
    base_url="https://gateway.internal/v1", api_key=os.environ["GATEWAY_TOKEN"],
)
```

## What effGen does not assume

The server serves its own model ids, so no OpenAI catalog is consulted.

- **Context window** defaults to 32768. Pass `context_length=` when what you
  serve is different — effGen plans compaction against this number.
- **Sampling** — the full surface (`top_p`, `top_k`, `seed`, the penalties) is
  offered, which every implementation of the protocol accepts.
- **Reasoning** — pass `supports_reasoning=True` if the model you serve emits a
  reasoning stream.
- **Cost** — calls report no price. What your own server costs is not something
  effGen can derive from a token count, so it states nothing rather than `$0`.
  The calls are recorded under the provider `openai_compatible` (which is also
  what `response.provider` reads), never at the rate OpenAI charges for a model
  that happens to share the id, and a spent daily budget does not refuse them
  (see [`effgen cost`](../cli/cost.md)).

Ask the endpoint what it serves:

```python
model = load_model("whatever", provider="openai_compatible", base_url=URL)
print(model.list_served_models())
```

An endpoint that does not implement `/models` returns an empty list rather than
failing — some minimal servers have nothing to say about themselves.

## When nothing is listening

A connection that is refused, a host that does not resolve and a route that
does not exist are reported as **unreachable**, separately from a server that
answered badly, and they raise `BackendUnreachableError`:

```python
from effgen.models.errors import BackendUnreachableError

try:
    agent.run(task)
except BackendUnreachableError as e:
    print("the server is not up:", e)
```

This raises whatever `AgentConfig.raise_on_error` says. A task that ran and
failed is a result you can inspect; a backend that never answered is not, and
silently returning one is how a whole batch completes against nothing and looks
healthy in the summary.

## Serving a model to point at

vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Then `base_url="http://127.0.0.1:8000/v1"` and the model id vLLM serves it
under — whatever `--served-model-name` was given, defaulting to the repo id.

Ollama:

```bash
ollama serve
```

Then `base_url="http://127.0.0.1:11434/v1"` and the model id you pulled.

## Related

- [effGen's own OpenAI-compatible server](../server/openai-compat.md) — the other
  direction: serving *your* agents over this protocol.
- [OpenAI](openai.md) — calling OpenAI itself.
- [Model router](router.md) — choosing between several models at run time.
