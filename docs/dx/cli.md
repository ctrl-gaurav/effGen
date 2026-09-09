# CLI developer-experience surfaces

Beyond `effgen run`/`chat`/`serve`, the CLI ships a few developer-focused
commands and a shell-completion generator. All of them stay in sync with the
real CLI automatically — the completion lists and preset choices are generated
by introspecting the live parser and registries, never hand-maintained.

## Short flags are per-command

Long option names mean the same thing everywhere. **Two short flags do not**, and
the long form is the portable spelling:

| Short | Means | In |
|---|---|---|
| `-c` | `--concurrency` | `batch`, `loadtest` |
| `-c` | `--config` | `run` |
| `-p` | `--port` | `serve`, `monitor`, `top` |
| `-p` | `--print` | `code` |

`-p` is the one to watch: three commands read it as a port number and a fourth
as a boolean, so `effgen code -p 8000` and `effgen serve -p 8000` mean unrelated
things. In a script, prefer `--port` and `--print`.

Two other short flags carry more than one long name but only one concept, and
are not a hazard: `-m` for `--model`/`--models`, and `-o` for
`--output`/`--output-dir`.

These bindings are **frozen**, not pending. Repointing `-p` would break
`effgen code -p`, which is documented and appears in shipped examples, for a
purely cosmetic gain; `tests/unit/test_cli_flag_consistency.py` pins both
collisions with the exact long names each separates, so a *third* concept
reaching for `-c` or `-p` — or any new collision on another short flag — fails
the build.

## Shell completion

Generate a completion script for your shell and source it:

```bash
# Bash — add to ~/.bashrc
eval "$(effgen --completion bash)"

# Zsh — add to ~/.zshrc
eval "$(effgen --completion zsh)"

# Fish
effgen --completion fish | source
```

The generated scripts complete the current subcommands (`run`, `chat`, `serve`,
`models`, `tools`, `doctor`, `compare`, `battle`, `debug`, `cost`, `eval`, …), the
`--preset` choices (from the preset registry), and `--tools` names (from the
tool registry). Because they are generated, they cannot drift out of date the
way a hardcoded list does.

## `effgen debug`

Run a task with full ReAct-step inspection — each iteration's thought, action,
tool input/observation, tokens and latency, plus a run summary.

```bash
effgen debug "What is 137 * 19? Use the calculator." -m gpt-5-nano
effgen debug "Plan a 3-step research task" --preset research --step
```

- `-m/--model` or `--preset` is required; with neither, the command prints an
  actionable message and exits `2`.
- `--step` pauses after each iteration so you can inspect the scratchpad.
- Exit code: `0` on success, `1` if the run failed, `2` for a config/usage error.

## `effgen compare`

Compare two or more models on a registered evaluation suite and print an
accuracy/latency matrix with a recommendation.

```bash
effgen compare \
  --models "gpt-5-nano,groq:openai/gpt-oss-20b" \
  --suite conversation \
  --scoring contains
```

- `--suite` accepts a registered suite name (`math`, `tool_use`, `reasoning`,
  `safety`, `conversation`); an unknown name lists the valid ones and exits `2`.
- `--preset` (registry-driven) wires each model into a preset agent.
- `-o/--output` writes the matrix in the format the extension names: `.html`
  renders the shareable report, `.md` writes Markdown, anything else JSON.
- `--report out.html` writes the shareable report alongside the usual output.
- `--judge MODEL` applies to `--scoring llm_judge`: without it every model
  grades its own answers, which is exactly the bias a bake-off is meant to
  remove. Naming a judge has one model grade the whole field. What did the
  grading is stated under the tables and recorded in the JSON as `judge_model`
  and `self_judged`.
- Exit code: `0` on success, `1` if no model loaded, `2` for an unknown suite.

Each score carries `responses` — what the model actually answered on every case
— so the JSON and the HTML report show the work behind the percentages. A model
that could not be run at all is reported as failed (`error`) rather than scored
`0%`, and is never recommended.

## `effgen battle`

Race several models on one ad-hoc prompt and watch the answers arrive side by
side. Where `compare` scores a suite against expected answers, `battle` answers
"which of these models should I use for this?" in one shot.

```bash
effgen battle "Explain a B-tree in two sentences." \
  -m openai:gpt-5-nano,groq:openai/gpt-oss-20b,gemini:gemini-3.1-flash-lite
```

On a terminal each model gets a column that fills in as its answer streams,
headed by the model id and footed with its time to first token, elapsed time,
tokens and cost. A verdict panel closes the race.

- `--judge MODEL` asks a separate model to pick the best answer. It is optional:
  the measured outcomes (fastest, cheapest, longest) need no judge, and a judged
  pick is reported apart from them, naming the judge.
- `--temperature`, `--max-tokens` and `--system-prompt` apply to every model.
- `-o out.json` / `-o out.md` saves the battle; `--report out.html` writes the
  self-contained report, with each model's full answer.
- A model that fails is reported as failed, does not end the race, and cannot
  win the verdict. Exit code `1` only when no model answered at all.

Loading and generating are timed separately, so a local model's start-up time is
visible as loading rather than counted as slow generation:

```
| Model                          | TTFT  | Latency | Cost      |
| groq:openai/gpt-oss-20b      | 0.42s | 1.87s   | $0.000008 |
| transformers:Qwen2.5-1.5B      | 0.48s | 1.23s   | unpriced  |   (load 12.7s)
```

Piped output, `--json`, `--no-animation` and `NO_COLOR` skip the live view and
print one structured result carrying every model's full answer, the tally and
the verdict:

```bash
effgen battle "..." -m a,b --json | jq '.contenders[] | {model, cost_usd, latency_s}'
```

The same race is available in the browser: open `/playground`, switch to
**Battle**, and pick contenders from the model catalog. Each column streams from
`POST /v1/chat/completions`, so the page inherits the endpoint's auth and spend
controls and adds no new model-execution path.

## Shareable HTML reports

`compare`, `battle`, `eval`, `cost`, and `loadtest` each take `--report out.html`,
which renders that run's result as a single HTML file: a headline verdict, the
tables the terminal shows, and inline charts.

```bash
effgen compare --models "gpt-5-nano,groq:openai/gpt-oss-20b" \
  --suite math --optimize cost --report bakeoff.html

effgen eval --suite math -m gpt-5-nano --provider openai --report eval.html
effgen cost today --report spend.html
effgen loadtest --duration 30 --concurrency 10 --report capacity.html
```

Each file is self-contained: every style, script, and chart is inline, so it
opens from disk or an email attachment with no network access. The page follows
the reader's light/dark system preference and carries a toggle. A model with no
published price reads `unpriced` and an absent metric reads `—`; neither is
rendered as `$0`.

The header stamps the generation time, the effGen version, and the command that
produced the result, so a shared file can be traced back and re-run.

`--report` is additive: terminal output, `--json`, and `-o` are unchanged.

### Rendering a saved result later

`effgen report` turns a result captured earlier into the same HTML file without
re-running any model. The report shape is inferred from the JSON.

```bash
effgen eval --suite math -m gpt-5-nano --provider openai --json > eval.json
effgen report eval.json                 # writes eval.html
effgen report eval.json -o for-team.html
effgen report spend.json --kind cost    # name the shape explicitly
effgen run "..." -o run.json && effgen report run.json   # a saved run
```

- Exit code: `0` on success, `2` if the file is missing, is not JSON, or is not
  a result the renderer recognizes (the message lists the accepted `--kind`
  values).
- A document that carries none of the fields the named `--kind` renders is
  refused with the fields that kind needs and the keys the document has, exits
  `2`, and writes no file — a report is never written blank.

## Run cards

`effgen run --card out.html` writes one run as a single shareable HTML file:

```bash
effgen run "What is 18723 * 4409? Use the calculator tool." \
  -m openai/gpt-oss-20b --provider groq -t calculator --card run.html
```

The card carries the task, the model and provider that answered it, a
succeeded/failed badge, the full answer, every tool step with its input, its
result or typed failure and its duration, the sources and quoted citations, and
the run's tokens, cost and latency. A run that failed shows the typed error —
type, category, provider, model, message, and whether it was retryable — in
place of an answer. A run on local hardware reads `unpriced (local)`; a hosted
model the catalog has no rate for reads `unpriced (no published rate)`.

Like the other reports the file is self-contained and opens with no network
access; links to a run's own sources are limited to `http`/`https`, and a
source with any other scheme is rendered as inert text. Buttons on the card
copy the task or the equivalent `effgen run` command.

`--card` is additive: terminal output and `--json` are unchanged, and it
composes with `-o`.

A run already in history can be exported as a summary card:

```bash
effgen runs show 3f9a1c2b --card run-3f9a1c2b.html
```

History keeps a truncated answer and no step trace, so that card says on its
face that it is a summary. Use `effgen run --card` at run time for the full
answer, trace and sources.
