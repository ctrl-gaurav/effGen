# OpenAI-compatible API (`/v1`)

`effgen serve` exposes OpenAI-compatible endpoints, so the official `openai`
client — or any OpenAI-compatible SDK/tool — can point at an effGen server
unchanged. `effgen serve` and the `effgen.server.app:create_app` factory build
the **same** secure application: `/v1/*`, auth, RBAC/budget, audit, metrics, and
the dashboard all come from one place.

```bash
effgen serve --port 8000          # binds 127.0.0.1 by default
```

## Authentication

The server is **authenticated by default** (fail-closed). Pick one posture:

| Posture | How | Notes |
|---|---|---|
| Static API key | `EFFGEN_API_KEY=<key> effgen serve` | Send `Authorization: Bearer <key>` **or** `X-API-Key: <key>` |
| OIDC / JWT | `EFFGEN_OIDC_ISSUER=…` (see [auth](auth.md)) | Bearer JWT validated against the issuer's JWKS |
| Dev (no auth) | `EFFGEN_DEV_MODE=1 effgen serve` | Loud warning; never use in production |
| *(nothing set)* | `effgen serve` | An **ephemeral** key is generated and printed once — the server is never unauthenticated by default |

Health probes (`/health`, `/healthz`, `/livez`, `/readyz`) and the API schema
(`/openapi.json`, `/docs`, `/redoc`) are always public.

```bash
curl -H "Authorization: Bearer $EFFGEN_API_KEY" http://127.0.0.1:8000/v1/models
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| POST | `/v1/completions` | Legacy text completions |
| GET | `/v1/models` | List the aliases + ids served this run (not exhaustive) |
| POST | `/v1/embeddings` | Text embeddings (local SentenceTransformer) |

## Python (official `openai` client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="YOUR_KEY")

# Any effGen-supported model id works directly:
r = client.chat.completions.create(
    model="groq/openai/gpt-oss-20b",          # or "openai/gpt-5-nano",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)                                                 # or a local id like
print(r.choices[0].message.content)               # "Qwen/Qwen2.5-1.5B-Instruct"
print(r.usage)                                    # real provider/tokenizer counts
```

`provider/model` ids are accepted (the slash is normalized to effGen's
`provider:model` routing). Bare ids load locally.

## Model aliases (compatibility shim)

OpenAI names map to concrete effGen models so OpenAI-only clients work:

| Alias | Resolves to |
|---|---|
| `gpt-4`, `gpt-4-turbo`, `gpt-4o` | `Qwen/Qwen2.5-7B-Instruct` |
| `gpt-4o-mini`, `gpt-3.5-turbo` | `Qwen/Qwen2.5-3B-Instruct` |
| `effgen-default`, `default` | the server's default model (`EFFGEN_DEFAULT_MODEL`, else `Qwen/Qwen2.5-3B-Instruct`) |

`GET /v1/models` lists these aliases plus every id the server has served a
successful response for this run. The list is **not** exhaustive: any
`provider:model` id the server can reach is callable whether or not it appears.

Aliasing is never silent. The response `model` field reports the model that
actually ran, and a non-standard `effgen` object documents the mapping (OpenAI
clients ignore unknown top-level keys):

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "effgen": {"requested_model": "gpt-4",
             "resolved_model": "Qwen/Qwen2.5-7B-Instruct",
             "alias_applied": true}
}
```

## Usage accounting

`usage` is real: provider-reported counts when the upstream API returns them,
otherwise tokenizer counts. It is never the legacy `len(text) // 4` estimate.

## Streaming

Streaming is truly incremental (SSE). To receive a final usage-only chunk, set
`stream_options.include_usage` — matching OpenAI, no usage chunk is sent unless
you ask:

```python
for ev in client.chat.completions.create(
        model="groq/openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Count 1 to 5."}],
        stream=True, stream_options={"include_usage": True}):
    if ev.choices and ev.choices[0].delta.content:
        print(ev.choices[0].delta.content, end="")
    if ev.usage:
        print("\nusage:", ev.usage)
```

A mid-stream failure is emitted as a terminal SSE event carrying an `error`
object and `finish_reason: "error"` (redacted), followed by `[DONE]`, rather
than truncating the stream. Both `/v1/chat/completions` and `/v1/completions`
behave this way.

## Tools — compatibility level

effGen runs tools **server-side** via its agent loop. Passing `tools` lets the
agent *use* them and return an already-resolved answer; the server does **not**
stream client-side `tool_calls` deltas for the client to execute.

```python
r = client.chat.completions.create(
    model="groq/openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Use the calculator to compute 127 * 43."}],
    tools=[{"type": "function",
            "function": {"name": "calculator",
                         "parameters": {"type": "object",
                                        "properties": {"expression": {"type": "string"}}}}}],
)
print(r.choices[0].message.content)   # -> "5461"
```

Tool names are resolved against effGen's built-in tool registry and are subject
to the caller's RBAC policy.

## Errors

**Every** error the server returns uses the OpenAI error envelope with an
accurate status and a redacted message — the model routes, `/v1/embeddings`,
the ops and RBAC routes, the dashboard and playground endpoints, the failures
raised before a route runs (auth, rate limit, RBAC, body-size cap, request
validation), and the ones raised outside any route (unknown URL, wrong method,
unhandled error):

```json
{"error": {"message": "…", "type": "model_not_found",
           "param": null, "code": "model_not_found"}}
```

Status codes: `400` invalid request, `401` authentication, `403` RBAC denial,
`404` model or URL not found, `405` method not allowed, `413` body too large,
`422` request validation, `429` rate limit, `502`/`503` upstream, `504` timeout,
`500` otherwise. Branch on `type` and `code` rather than on the message text.

The two upstream statuses say different things about the provider credential,
consistently across every provider: `503` with `code: "upstream_key_missing"`
means the server has no key configured for that provider (a configuration gap —
set the variable and retry), and `502` with `code: "upstream_auth_failed"` means
a key is configured and the provider rejected it. Neither ever echoes the key.

A request with nothing for the model to act on is refused with `400` before any
billed call: `empty_messages` / `empty_content` on `/v1/chat/completions`,
`empty_prompt` on `/v1/completions`, `empty_input` on `/v1/embeddings`.

## Performance

Loaded models are pooled and reused across requests (bounded LRU, set
`EFFGEN_MODEL_POOL_SIZE`), so the model-load cost is paid once rather than per
request.
