# effgen bench - Measure an Agent on Your Own Tasks

`effgen bench` runs a suite of tasks against a model and reports what the agent
spent to answer them: accuracy beside LLM calls, tool calls, tokens and time.
`effgen bench compare` puts two runs side by side and prints a noise band beside
every difference, so a change that is within run-to-run variation is not
reported as a result.

| Subcommand | Description |
|---|---|
| `effgen bench init [PATH]` | Write the starter suite (default `bench-suite.yaml`) |
| `effgen bench run SUITE` | Run a suite and print its table; saves the run |
| `effgen bench compare A B` | Compare two saved runs, with a band beside every delta |

## Quick start

Save this as `arith.yaml`:

```yaml
name: arith
seed: 7
tools: [calculator]
scorer: number
agent:
  system_prompt: "Use the calculator for arithmetic. Finish with 'Answer: <number>'."
  temperature: 0.1
  max_iterations: 5
tasks:
  - id: rent
    input: Rent is $1,450 a month. How many dollars is a year of rent?
    expected: 17400
  - id: speed
    input: A train covers 270 km in 3 hours. What is its speed in km per hour?
    expected: 90
  - id: boxes
    input: 23 boxes hold 16 jars each. How many jars is that?
    expected: 368
```

Run it against any model effGen can reach — a server of your own speaking the
OpenAI protocol, a cloud provider, or a local model:

```bash
effgen bench run arith.yaml --model my-model --base-url http://127.0.0.1:8000/v1 --out runs/a
effgen bench run arith.yaml --model my-model --base-url http://127.0.0.1:8000/v1 --out runs/b
effgen bench compare runs/a runs/b
```

`effgen bench init` writes a longer starter suite in the same format.

## The suite file

A suite is YAML (`.yaml`, `.yml`) or JSON (`.json`). Loading is strict: an unknown
key, a task with no input, a repeated task id, or an `n` larger than the task list
is an error, so a typo never runs a different suite from the one the file names.

| Key | Meaning |
|---|---|
| `name` | The suite's name. Defaults to the file name. |
| `description` | Free text. |
| `tasks` | A list of tasks (below). |
| `tasks_file` | Instead of `tasks`: a JSON Lines file (or a JSON list), relative to the suite file. |
| `fields` | With `tasks_file`: which record keys hold `id`, `input`, `expected` and `context`, e.g. `{input: question, expected: answer}`. |
| `tools` | Tool names the agent may call, as `effgen tools list` shows them. Empty for none. |
| `scorer` | How an answer is scored (below). Default `exact`. |
| `n` | How many tasks to run. Default: all of them. |
| `seed` | Picks which `n` tasks run (without it, the first `n`) and is sent to the model as its sampling seed. |
| `agent` | Agent settings: `system_prompt`, `temperature`, `top_p`, `max_tokens`, `max_iterations`. Anything unset keeps the `AgentConfig` default. |
| `model`, `base_url` | Defaults for `--model` and `--base-url`. |

A task has an `id` (defaults to `task-<number>`; runs are paired on it), an `input`
(the text the agent receives), an `expected` value, and an optional `context`,
placed after the input with a blank line between them.

Every task runs in a fresh agent, so no task sees another's conversation. Memory
and sub-agent decomposition are off: the run measures one agent loop per task.

### Scorers

| Scorer | Scores 1 when |
|---|---|
| `exact` | the answer equals `expected`, ignoring case, surrounding space and a final full stop. When the answer has a line starting `Answer:` or `Final answer:`, the text after the last one is compared. |
| `contains` | `expected` appears in the answer, ignoring case. |
| `number` | the last number in the answer equals `expected` within a relative `tolerance` (default `1e-6`; `scorer: {name: number, tolerance: 0.01}`). Thousands separators are ignored. |
| `regex` | the regular expression `expected` matches the answer, ignoring case. |
| `choice` | the answer names the option letter `expected`: the letter after the last `Answer:`, else the last letter standing on its own. |

`expected` can be a list for `exact`, `contains` and `regex`; any match scores 1.

A scorer of your own is a function `score(answer, expected, task)` returning a
bool or a number from 0 to 1:

```yaml
scorer:
  callable: my_scorers.py:score    # a file next to the suite, or a module: pkg.mod:score
```

## `effgen bench run`

```bash
effgen bench run SUITE [--model ID] [--base-url URL] [--api-key-env VAR]
                       [-e transformers|vllm|auto-fast] [--n N] [--seed S]
                       [-c C] [--temperature T] [--max-tokens N]
                       [--max-iterations N] [--label NAME] [--out DIR | --no-save]
                       [--max-errors K] [--json]
```

The table it prints has one row per measured field, each as a per-task mean and a
total, read from every task's run ledger:

| Row | What it is |
|---|---|
| Accuracy (%) | Mean score, in percent. |
| Tasks | Tasks run. |
| Errors | Tasks that never produced a run: the model could not be reached, or the run failed outright. |
| LLM calls, Tool calls | Model requests (retries included) and tool executions. |
| Prompt, Completion, Cached input tokens | As the backend reported them; cached input is the part of the prompt served from a provider cache. |
| Wall time (s) | Seconds from the start of each task's run to its end. |
| Framework time (s) | Wall time spent outside model calls, tool calls and child runs. |
| Model wait, Tool wait (s) | Seconds inside model calls and inside tool executions. |
| Cost (USD) | Cost of priced calls; `unpriced` when the model has no published price. |
| Unpriced calls | Calls whose model has no published price. |
| Elapsed, whole run (s) | Clock time for the whole suite, with `--concurrency` tasks in flight. |
| Stop reasons | How the tasks' runs ended (`final_answer`, `max_iterations_partial`, ...). |

`--json` prints the run document: `table` (every row above), `tasks` (one record
per task: id, score, answer, expected, stop reason, error and every counter),
`suite` (name, fingerprint, n, seed, scorer, tools, agent settings), `config`
(model, endpoint, concurrency, settings), the effGen version and the start and
finish times. The same document is saved as `run.json`, next to `records.jsonl`,
which receives each task as it finishes; the printed document also carries
`saved_to`, the path of that `run.json`, which the file itself leaves out so a
run directory stays valid when it is moved. Without `--out`, runs are saved under
`$EFFGEN_BENCH_DIR`, else `$EFFGEN_HOME/bench`, else `~/.effgen/bench`.

The command exits 1 when more tasks than `--max-errors` (default 0) never produced
a run, because a score of 0 on a task the model never saw measures the connection,
not the model. `--api-key-env` names the environment variable holding the
endpoint's key; the key itself is never printed or saved.

## `effgen bench compare`

```bash
effgen bench compare A B [--json]
```

`A` and `B` are saved runs (a `run.json` file or its directory) of the same suite:
two runs of one configuration, or two configurations — a different model, prompt,
temperature or effGen version. The runs are paired on task id; tasks only one run
has are left out and counted.

For every field — accuracy, errors, calls, tokens, the four times, cost, and each
stop reason — it prints A, B, the delta `B - A` and a band:

```text
band = 2 × standard error of the per-task differences (B - A)
     = 2 × stdev(d) / sqrt(k),  d = value in B - value in A for each of the k paired tasks
```

The band comes from the two runs' own disagreement, task by task. Two runs that
agree on every task have a band of 0; two runs whose answers flip on many tasks
have a wide one. A delta within its band reads `within noise`: it is not a
measured difference, and repeating the same configuration can produce one that
size. Stop reasons are compared as counts, with the band scaled to a count.

The band is about a 95% interval for one field: two runs of the same configuration
put about 1 field in 20 outside it by chance. Reading all the fields of a comparison
at once, expect an occasional excursion; to judge many fields together, require each
delta to clear a wider band — for example `band × 1.43` (2.87 standard errors) keeps
the chance of any false excursion among 12 fields at or below about 5%.

The 95% reading also assumes enough paired tasks: with fewer than about 30 the band
covers less (about 92% at 10 tasks, about 82% at 3), so a small suite reads as a
smoke check, not a measurement. It further assumes many tasks differ between the
two runs. When only a few do —
a count that changed on four tasks out of a hundred, each by one — the band is too
narrow to read at that level: four changes that happen to share a direction land
outside it about one time in eight. Read such a row by how many tasks moved (the
per-task records are in `records.jsonl`) rather than by the band alone.

`--json` prints the comparison: `rows` and `stop_reasons`, each with `a`, `b`,
`delta`, `band` and `within_band`, plus how many tasks paired.

The band describes variation between the two runs you compare. Timings also move
with whatever else the machine and the server were doing, and a server's prompt
cache can make a second run's cached-input tokens and wall time differ from the
first's for reasons unrelated to the agent.

## What ships

effGen ships one starter suite (`effgen bench init`), about a kilobyte of
package data, and no datasets: a suite is your tasks, read from your files, and
nothing is downloaded when a suite runs.

## From Python

```python
from effgen.bench import BenchConfig, compare_runs, load_suite, run_suite

suite = load_suite("arith.yaml")
config = BenchConfig(model="my-model", base_url="http://127.0.0.1:8000/v1")
a = run_suite(suite, config)
b = run_suite(suite, config)
for row in compare_runs(a, b)["rows"]:
    print(row["field"], row["delta"], row["band"], row["within_band"])
```
