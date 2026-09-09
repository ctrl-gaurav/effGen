# effGen VSCode Extension (Experimental)

> **Status: experimental.** This editor extension is a developer-experience
> preview shipped from the effGen repository. It is **not** published to the
> VSCode Marketplace and is **not** covered by effGen's stability guarantees.
> Expect rough edges and breaking changes. It is intentionally excluded from
> user-facing feature claims in the main README.

A minimal extension that adds:

- **Prompt-template completion** — triggers on `LibraryPrompt(`, `effgen.prompts.`
  and `%effgen_` in Python / Jupyter files.
- **Run code lens** — an inline "▶ Run with effGen" action on
  `LibraryPrompt(...)` / `effgen_chat(...)` lines that POSTs to a running effGen
  server's OpenAI-compatible `/v1/chat/completions` endpoint.
- **Hover docs** — shows a prompt template's description on hover.

## Requirements

- A running effGen API server (`effgen serve`). Set `effgen.serverUrl`
  (default `http://localhost:8080`).
- `effgen.defaultModel` — the model id sent to the server (default `gpt-5-nano`;
  use any id your server can serve, e.g. `groq:openai/gpt-oss-20b` or a local
  HuggingFace repo id). Run `effgen models list` to see options.

## Build from source

```bash
cd tools/vscode-effgen
npm ci
npm run compile        # tsc -p ./  → out/extension.js
```

Then load the folder in VSCode via *Run Extension* (F5) or package it with
`vsce package` (install `@vscode/vsce` first).
