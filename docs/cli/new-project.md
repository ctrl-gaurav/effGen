# Starting a new project

`effgen quickstart --init` turns an empty directory into a project that runs:
a configuration `effgen run` reads, an `.env` template naming the provider
variables, one runnable example, and a daily spend cap.

It makes no model call and asks no questions, so it works before any key is
set, in a pipe, and in CI.

```bash
mkdir my-agent && cd my-agent
effgen quickstart --init
```

Or name the directory (it is created if it does not exist):

```bash
effgen quickstart --init my-agent
```

```
effGen project
Directory: my-agent

  wrote    effgen.yaml   model, prompt and per-run caps
  wrote    .env.example  key names, no values
  wrote    example.py    a runnable agent script
  wrote    .gitignore    keeps .env out of git

Model:      groq:openai/gpt-oss-20b
            groq key detected
            change it on the 'model:' line of effgen.yaml

Spend cap:  $1.00 a day across all runs (set now)
            'effgen cost set-budget N' changes it
            'effgen cost clear-budget' removes it

Next three commands
  1. cp .env.example .env  # then paste one key into it
  2. effgen doctor  # confirm effGen sees it
  3. effgen run "What is 25 * 17?" -c effgen.yaml
```

## What it writes

| File | What it is |
|---|---|
| `effgen.yaml` | The agent configuration. `effgen run -c effgen.yaml` reads it. |
| `.env.example` | Every provider variable, named, with no value. Copy to `.env` and fill one in. |
| `example.py` | The same agent as a Python script — `python example.py`. |
| `.gitignore` | Excludes `.env`, `.effgen/` and the usual Python artifacts. |

Nothing else is written inside the directory. The one file written outside it
is the daily spend cap, which is the same `~/.effgen/budget.json` that
`effgen cost set-budget` writes.

### `effgen.yaml`

```yaml
# effGen project configuration.
#
# Used by:  effgen run "your task" -c effgen.yaml
#
# Next three commands:
#   1. cp .env.example .env  # then paste one key into it
#   2. effgen doctor  # confirm effGen sees it
#   3. effgen run "What is 25 * 17?" -c effgen.yaml
#
# max_tokens caps what one answer may cost; max_iterations caps how many steps
# the agent may take on one task. Both apply to every run that loads this file.
# A daily spend cap across all runs is separate: effgen cost set-budget 1.00

model: groq:openai/gpt-oss-20b
system_prompt: You are a helpful assistant. Answer concisely and say when you are unsure.
temperature: 0.2
max_tokens: 512
max_iterations: 5
```

The file carries only the keys a run applies: `model`, `provider`,
`system_prompt`, `temperature`, `max_tokens`, `max_iterations` and
`guardrails`. Any other key that names an `AgentConfig` field is reported when
the file loads, so a setting can never be a silent no-op.

`max_tokens` follows the model. A reasoning model (`gpt-5-nano`, the o-series)
spends part of its output budget on hidden reasoning before it emits a visible
token, so the scaffold writes the larger budget that model needs — and says why
in the file — instead of a budget that could run out before the first word and
bill you for an empty answer:

```yaml
# openai:gpt-5-nano spends part of its output budget on hidden reasoning,
# so max_tokens is 4096 here rather than the usual 512. A smaller
# budget can run out before the first visible word, and that empty
# answer is still billed.

model: openai:gpt-5-nano
max_tokens: 4096
```

`-m/--model` on the command line still wins over the `model:` line:

```bash
effgen run "What is 25 * 17?" -c effgen.yaml     # the model the file names
effgen run "What is 25 * 17?" -c effgen.yaml \
    -m gemini:gemini-3.1-flash-lite              # this one instead
```

The model id is written provider-prefixed (`groq:openai/gpt-oss-20b`,
`transformers:Qwen/Qwen2.5-1.5B-Instruct`) because a file travels without a
`--provider` flag beside it.

### `.env.example`

Every registered provider's variable is named and left empty:

```
# effGen provider keys.
#
#   cp .env.example .env
#   $EDITOR .env          # paste your key after the '=' of one line
#   effgen doctor         # confirms which keys effGen can see
...
# cerebras
CEREBRAS_API_KEY=

# gemini
GOOGLE_API_KEY=

# groq
GROQ_API_KEY=
```

No value is invented. An empty assignment counts as no key at all, and a
variable already exported in your shell wins over the file, so copying the
template unchanged cannot shadow a key you already have.

Until `.env` exists, `effgen doctor` names the template in its missing-key
hint:

```
Missing keys — set in ~/.effgen/.env or export:
  cp .env.example .env, then paste a key into it
  export OPENAI_API_KEY=<your-key>
  ...
```

## Cost control

Two caps apply, and they are independent:

* **per run** — `max_tokens` bounds one answer, `max_iterations` bounds how
  many steps the agent may take. Both live in `effgen.yaml`.
* **per day, across every run** — a spend cap in `~/.effgen/budget.json`. The
  scaffold sets `$1.00` a day when you have no cap yet, and never changes one
  you already have.

```bash
effgen quickstart --init my-agent --budget 5   # a different cap, when none is set
effgen quickstart --init my-agent --budget 0   # set no cap
effgen cost set-budget 2.50                    # change it later
effgen cost clear-budget                       # remove it
effgen cost today                              # what has been spent against it
```

## Re-running it

An existing file is kept, not replaced:

```bash
effgen quickstart --init my-agent           # keeps every file already there
effgen quickstart --init my-agent --force   # rewrites them
```

## Options

| Flag | Effect |
|---|---|
| `--init [DIR]` | Scaffold in `DIR` (default: the current directory). |
| `--force` | Replace a scaffolded file that already exists. |
| `--budget USD` | The daily cap to set when none is configured. `0` sets none. |
| `-m/--model ID` | Write this model id instead of the detected one. |

## Just the configuration

`effgen config init` writes the same document on its own:

```bash
effgen config init                 # config.yaml
effgen config init -o agent.yaml   # somewhere else
effgen config validate --file config.yaml
```

## Where to go next

* [CLI configuration & environment](configuration.md) — the full `.env` search
  order and the provider/model id forms.
* [Cost tracking and budgets](cost.md) — spend reports and budget behavior.
* [Getting started](../tutorials/getting-started.md) — the same ground from
  Python.
