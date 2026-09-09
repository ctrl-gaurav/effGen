# effGen Release Notes

## v1.0.1 - September 8, 2026

This release fixes how the framework reports what a run did, what it puts in a prompt, and what its
own bookkeeping costs.

The main things: a run that stops without an answer now says so instead of handing back its working
notes. Citation markers are opt-in instead of being added to every retrieval answer. The loop guards
no longer stop a run that is still making progress. Every tool-calling path now tells the model what
the tools are for. The budget check before each model call reads an index instead of the whole spend
ledger. And the Groq default points at a model Groq still serves.

None of this is tuned for a benchmark. These are changes to how the framework behaves. We ran public
sample sets to check the changes helped rather than to chase a score, and where a change cost
something we say so.

Four changes are visible to existing code, and one of them changes what `success` means for a run
that stopped part way. Those are listed first.

The public surface grew from 223 names to 225. Nothing was removed or renamed.

```bash
pip install --upgrade effgen
effgen --version
```

### 1. A run that stops without an answer now reports failure, and raises by default

In 1.0.0 three paths returned `success=True` with internal state in `.output`: a computation tool
that tripped the repeat guard, a computation tool that returned a result it had already returned,
and a model that gave no final answer after its tools ran. On the same test 1.0.0 returns `'4'` with
`success=True`, no outcome and no stop reason. 1.0.1 returns `success=False`, `outcome="stopped"`,
`stop_reason="max_iterations_partial"`, and keeps what the model reached in `.partial`.

With the default `raise_on_error=True`, a stopped run raises `RunStoppedError`. It subclasses
`RuntimeError`, which is what the iteration cap has always raised, and it carries `.response`,
`.stop_reason` and `.partial`.

```python
from effgen import Agent, AgentConfig, RunStoppedError

agent = Agent(AgentConfig(model="openai:gpt-5-nano"))
try:
    response = agent.run("What is 17 * 23?")
    print(response.outcome, response.stop_reason)
    print(response.text)
except RunStoppedError as exc:
    print(exc.stop_reason)
    print(exc.partial.text if exc.partial else "nothing to report")
```

To upgrade: catch `RunStoppedError`, or pass `raise_on_error=False` and check `.outcome`, which is
`"answered"`, `"stopped"` or `"failed"`. The text the model reached is in `.partial.text`, and it is
still at `metadata["partial_output"]` byte for byte.

The outcome also shows up in the CLI, the run store (`effgen runs list --status stopped`), the batch
row and its CSV, `effgen code`, `EvalResult.stop_reason`, and the OpenAI-compatible server, whose
`effgen` envelope now carries `stop_reason`, `outcome` and `partial`.

### 2. Citation markers are opt-in

1.0.0 added "cite each passage you used inline as [1], [2], ..." to every turn that followed a
retrieval or search tool, whether or not you asked for citations. The markers did not point at
anything, so they were noise added to the answer. On a question with a one-word answer the model
followed the instruction and the answer stopped matching: 1.0.0 answers `'B [1], [2]'` where 1.0.1
answers `'B'`.

Ask for them with `AgentConfig(cite_sources=True)` or `run(..., cite_sources=True)`. The `rag`
preset asks already. When you do ask, the retrieval results are numbered with the citation indexes,
so `[n]` is `citations[n - 1]`, across calls and for rows with a URL. That was not true before.

Replaying recorded answers through the removal code recovers 192, 66 and 52 answers on three
question sets and breaks none. Across fifteen recorded retrieval runs it fires 481 times and breaks
nothing. How much this matters depends on the model. Over 4,769 answers at each size it fires 0
times at 1.5B, 135 at 3B, 12 at 7B, 334 at 14B and 0 at 32B, so a model that rarely wrote the
markers had little to gain.

### 3. A streamed run sends the model's working before its answer

Same task, same final answer, but 8 chunks became 134, starting with the model's reasoning. If you
join the chunks and show the result, you now get the working first.

### 4. A small local model writes more and reaches for tools more often

`Qwen2.5-1.5B-Instruct` on the local Transformers engine, four runs per release, same token counts
every time. "What is 144 divided by 12?" goes from 31 to 233 completion tokens and from about 3.5
seconds to about 27. "Name the largest planet in the solar system." goes from 11 to 145 tokens and
from 0 to 1 tool calls. Both answers stay correct.

## Fixed

- **The budget check no longer reads the whole ledger.** Against a 500,000 row ledger the check took
  1,278 ms warm and 1,246.9 ms cold, doing a full table scan every time. It is now 0.044 ms warm and
  37.4 ms cold, using a covering index instead of a scan. At 20,000 rows the cold read is 2.5 ms.
  `effgen cost prune` keeps the file small.
- **The loop guards no longer stop runs that are still working.** A repeat of a call that already
  succeeded is answered from the run's own record and the run keeps going. The drift thresholds are
  now bounded by the run's iteration budget. And when the loop does break, every tool category gets
  one turn to answer from what it has. Over a 200 run sample the two guards fired 69 times in 1.0.0
  and once in 1.0.1. Over a 125 run sample, 24 firings became 0.
- **A turn's own working is no longer read as its answer.** A turn that sent several tool calls at
  once could be treated as a final answer because of something as short as `=`.
- **An agent holding a code executor now runs the code.** A first answer that only describes what
  the tool would have returned is sent back once, naming the tool. The next turn is sent with
  `tool_choice="required"` on adapters that support it.
- **Every generation parameter reaches the provider.** `Agent._generate` copied one name out of its
  keyword arguments and dropped the rest, with no error and no log line.
- **A batched provider-side tool call records what it returned** instead of leaving
  `tool_calls[i].result` as `None`.
- **A result a tool computed but the answer left out is added back to the answer.**
- **A search that returns nothing is tried once more** with a different query.
- **Every tool-calling path says what the tools are for**, in the same words, picked from the tools'
  declared categories.
- **A declared `output_schema` is stated inside the loop**, on `stream()` as well as `run()`.
- **A tool with no declared category no longer raises from `Agent.__init__`.**
- **The Groq default names a model Groq still serves.** `GROQ_DEFAULT_MODEL`, the bundled catalog,
  the CLI help, the error messages and every shipped example moved off the two retired `llama` ids
  to `openai/gpt-oss-20b`.
- **A model id with a provider prefix now loads when you also pass the provider.**
  `load_model("groq:openai/gpt-oss-20b", provider="groq")` used to raise `Unknown Groq model`. This
  is the path `effgen run` and `effgen quickstart` take when you give no `--model`.

## Added

- `PartialResult` and `RunStoppedError`.
- `AgentConfig.cite_sources`, `.tool_contract` and `.tool_use`. `cite_sources` and `tool_choice` are
  now `run()` keywords.
- `effgen.prompts.tool_contract`, with four tool contracts picked from a tool's declared
  `ToolCategory`, and a `ToolUsePolicy` of `REQUIRED`, `AUTO` or `SPARING` set for every category.
  Every shipped default matches what 1.0.0 already did.
- `BaseModel.supports_forced_tool_call`.
- `SQLiteCostStore.spend_since`, `spend_today`, `spend_week`, `spend_month`, `count`, `count_since`
  and `prune`, plus `effgen cost prune`.
- `effgen runs list --status stopped`, `EvalResult.stop_reason` and `PresetConfig.cite_sources`.
- English and Spanish keyword matching in the complexity analyzer, decomposition engine, sub-agent
  router and prompt optimizer, with accents folded on both sides. There is no language detection
  step and English behaviour is unchanged. A root agent's `system_prompt` now also reaches the
  sub-agents it spawns. Both from @acdonaire.

## Known issues

1. **Retrieval got worse, and we know why but have not fixed it.** Two question sets lost 5.00 and
   4.50 points, both outside their noise bands. The cause is the loop, not the answer. Retrieval
   runs hit the iteration cap ten times as often as in 1.0.0, 30 of 600 runs against 3, and most of
   the lost answers are runs that ran out of iterations. The retrieval prompt changed in three ways
   at once in this release and we have not yet worked out which one costs the extra turns.
2. **A run that hits the iteration cap while holding a retrieval tool returns a retrieved passage**
   as its `partial`. The reporting is right, but the payload is source material shown as an answer.
   1.0.0 did this too, on the 3 capped retrieval runs it had.
3. **A retrieval run over a real set of documents can miss a fact 1.0.0 finds.** 2 of 3 against 3 of
   3 over seven documentation files, four runs out of four on each release.
4. **A spend cap can refuse a call that costs nothing.** The check compares money already spent
   against the cap without asking what the call would cost, so a used up cloud budget also blocks a
   `transformers`, `vllm` or `openai_compatible` call against a model you serve yourself. This was
   in 1.0.0 too.
5. **Groq retired `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`.** Both return
   `404 model_not_found`. Upgrading to 1.0.1 is what fixes the default, because
   `GROQ_DEFAULT_MODEL` is a module constant used as the default argument of `GroqAdapter.__init__`,
   and this release repoints it, the bundled catalog, the examples and the docs to
   `openai/gpt-oss-20b`. On 1.0.0, or in code that pinned a retired id, name a live id yourself with
   `GroqAdapter(model_name="openai/gpt-oss-20b")` or `--model groq:openai/gpt-oss-20b`.
   `effgen models refresh` does not fix it. It only rewrites the bundled snapshot, and the module
   default is never read back from that snapshot, so after a refresh the adapter still defaults to
   the retired id.
6. **The streamed loop does not send back an execution tool's no-call answer**, and does not strip a
   trailing citation marker. Its tokens are already out by the time the turn could be judged.
7. **Open-ended search answers are still well behind the best system we measured next to them**,
   44.00 against 64.00 on the same questions. Searching a second time helps, with a mean of 1.86
   search calls against a ceiling of 3, but the gap is structural and this release does not close
   it.

**What it cost.** Compared to 1.0.0 a run makes 37% more model calls and sends 57% more
prompt tokens. The full measurement is in the changelog.

[Full 1.0.1 changelog](CHANGELOG.md#101---2026-09-08)

---

## v1.0.0 — August 14, 2026

**effGen v1.0.0 is the first stable release.** It is more than 600 commits of work since v0.3.2. The theme is
control over where a model runs and visibility into what a run did: drive a server you already
operate, read back the calls a run made, extend the agent loop, and pick up a workflow that stopped
part way through.

Around it sits a terminal coding agent, a command line that works on any terminal, a real-time
dashboard, an in-browser playground, shareable reports, a cross-provider model and pricing browser, a
mission-control monitor and a browsable run history. Underneath both is the largest and least visible
part of the release: a long pass over everything that used to report the wrong thing confidently.

**Three changes are breaking.** They are listed at the end with their migrations.

### Point effGen at a server you already run

This is the change most people will feel first. If you already serve a model with vLLM, SGLang, TGI,
llama.cpp, Ollama, LM Studio or a gateway, effGen can drive it instead of loading a second copy of
the weights inside the agent process.

```python
from effgen.models import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)
```

The endpoint can come from `EFFGEN_BASE_URL`, `OPENAI_BASE_URL` or `OPENAI_API_BASE`, and
`AgentConfig(base_url=..., api_key=...)` reaches it without touching the loader. The server serves
its own ids, so effGen consults no OpenAI catalog: the full sampling surface is offered, and calls
report **no price** rather than a fabricated `$0`. Pass `context_length=` if your server's window is
not the 32,768 tokens effGen otherwise assumes. It now warns when it is assuming, naming the flag
that sets the real number, instead of failing later at a size nobody chose.

### The extension points people arrive expecting

Three of them, under the names they have elsewhere. **Middleware** wraps the run, each model call and
each tool call, with a *before* hook that can rewrite or short-circuit the request and an *after*
hook that can transform the result. **`run(session=...)`** lets one agent serve many conversations,
so a server no longer needs an agent object per user. **Compaction is a strategy**:
`SummarizeOldest` (the default), `DropOldest`, `KeepFirstAndLast`, `KeepToolResults`, or your own
subclass, chosen with `AgentConfig(compaction_strategy=...)`. Pass `AgentConfig(tokenizer=...)` and
the threshold is measured in the window's own units instead of characters divided by four.

```python
from effgen import Agent, AgentConfig
from effgen.core.middleware import AgentMiddleware

class SearchBudget(AgentMiddleware):
    def __init__(self, limit=3):
        self.limit, self.used = limit, 0

    def before_tool_call(self, ctx):
        if ctx.tool_name != "web_search":
            return None
        if self.used >= self.limit:
            return "Skipped: this run has spent its search budget."
        self.used += 1
        return None

agent = Agent(AgentConfig(model="gpt-5-nano", middleware=[SearchBudget()]))
```

### A workflow that died half way through picks up where it stopped

Pass a store and a run id. Run the same line again after a crash and it continues. Completed nodes
are not re-run, failed ones are retried, and a run that already finished replays its stored outputs
without calling a model, so a job runner that retries cannot double-bill you.

```python
from effgen import FileCheckpointStore, WorkflowDAG, WorkflowNode

store = FileCheckpointStore()
dag = WorkflowDAG("report")
dag.add_node(WorkflowNode(id="research", agent=researcher))
dag.add_node(WorkflowNode(id="draft", agent=writer))
dag.connect("research", "draft")

result = dag.run("Write the Q3 summary.", checkpoint=store, run_id="q3-summary")
```

There is no separate resume call. An unknown run id starts at the beginning and a known one
continues, so there is no second code path to get wrong. Resuming into a graph whose nodes changed is
refused by name rather than mixing outputs from two different workflows.

### Which calls a run made

`AgentResponse.tool_calls` used to be a number. It now carries the calls, each with its `name`,
`arguments`, `result`, `duration`, `error` and the iteration it was made on.

```python
for call in result.tool_calls:
    print(call.name, call.arguments, "->", call.error or call.result)
```

It still compares and casts as the count, so `tool_calls == 2` keeps working, and `tool_calls.total`
says the number plainly. For anyone writing their own loop, `build_assistant_message()` and
`build_tool_result_message()` on every adapter build each provider's own message shape, so a loop
written once runs on all of them rather than only on OpenAI.

### A coding agent in the terminal

`effgen code` reads your workspace, proposes edits as unified diffs, and writes nothing until you say
so. `--undo` rolls the last change back from a journal.

```bash
effgen code "add a --dry-run flag to the importer"
effgen code --review                      # one read-only pass
effgen code --session-id my-refactor      # continue where you left off
```

It runs in one of four permission modes that gate every write, every shell command and every commit.
An interactive session keeps one run record across turns and carries a full slash-command set:
`/plan`, `/diff`, `/apply`, `/reject`, `/undo`, `/run`, `/test`, `/context`, `/add`, `/drop`,
`/mode`, `/model`, `/cost`, `/trace`, `/git`, `/review`, `/save`, `/load` and more. It knows the
repository it is in, and git actions run through an allow-list, so push, reset, checkout and force
are refused before a subprocess starts, including when the model tries to reach them through the
shell. A turn streams its answer as it is written and names which tool-calling path it ran.

### Surfaces you can show someone

A real-time dashboard with per-model cost and latency; an in-browser playground; a cross-provider
model and pricing browser in both the terminal and the dashboard; self-contained HTML reports for
compare, eval, cost and loadtest, plus single-run cards you can send to a colleague; `effgen top` as
a terminal mission-control view; a live multi-agent topology graph; a command palette on both web
surfaces; terminal trace timelines and workflow diagrams; and `effgen battle`, which races several
models on one prompt.

```bash
effgen serve --port 8080     # dashboard + playground
effgen top                   # terminal mission control
effgen models browse --vision --min-context 128000 --sort price-out
effgen battle "Explain gradient clipping" -m groq:llama-3.1-8b-instant,gemini:gemini-3.1-flash-lite
```

Every web surface is self-contained. There is no CDN, no external font and nothing fetched at view
time, and a test checks that by inspecting what a browser would actually fetch.

### Starting a project, and finding your way back to a run

`effgen quickstart --init` writes a project that runs: a config the CLI reads, an `.env.example` with
one named variable per provider and no invented values, a runnable `example.py`, a `.gitignore`, and
a $1.00/day spend cap when none is configured. Every run is then recorded, so `effgen runs list`,
`effgen runs show <id>` and `effgen sessions browse` can find it again, with search, status, model
and date filters. Runs from the CLI, a script and the server share one history and survive a restart.

### Somewhere to read about it

effGen has a site now, built and published from this repository on every change: a landing page at
<https://ctrl-gaurav.github.io/effGen/> with the examples, the community links and the benchmark
leaderboard, and 36 documentation pages at <https://ctrl-gaurav.github.io/effGen/docs/> covering
installation and the quick start through to deployment, hardware, protocols and the API reference.

Underneath it, every public class, method and function in the package now states what it does, what
each argument means and what comes back, with a gate that fails when a public definition is added
without it. The 119 pages under `docs/` were re-run command by command against this release.

### Results that report what actually happened

This is the largest group of fixes in the release, and the least visible until it saves you.

A run that failed now says so. A turn whose every action failed, and a retrieval loop that produced
no answer, are reported as partial outcomes with the recovered text under
`metadata["partial_output"]`. A run stopped at the iteration cap reports the stop, not the last
passage it retrieved. A reasoning model that emitted no visible token says so rather than being
retried three times. A tool call the model *wrote out* instead of making is a failed turn, not an
answer. Code that exits non-zero reports failure.

Cost got the same treatment. An unpriced or uncatalogued model reports **no cost** rather than a
made-up one: a provider's placeholder rate used to make every unseen id read as priced, so a
fine-tuned `ft:` id was billed at an invented rate and that number was reported as a published price.
Streamed runs now report their cost and tokens on every provider, every model span carries its cost,
and a hierarchical team's total includes the manager's own calls.

One more change belongs here because it shows up on the bill: a plain `run()` no longer decomposes a
task into sub-agents on its own. A task over roughly a hundred words used to become six billed calls.
`--mode auto` opts back in.

### Tools that work on more models

Chat templates disagree about how a tool call is spelled. Many render JSON; others render nested tags
like `<function=calculator><parameter=expression>4817 * 236</parameter></function>`. effGen read only
the JSON spellings, so on a model whose template emits tags a turn parsed to nothing: no tool called,
and the run ending at the iteration cap. The reader is now keyed on the shape rather than on a model
family, so any family whose template writes that shape can use tools. The construct is also stripped
from the answer whole, and a streamed turn holds it back until it is cleaned. Measured across 15 local
families: three went from no tool call and no answer to a correct answer, and twelve were unchanged.

Gemma 4 had the same problem in its own dialect: it wraps reasoning in `<|channel>` and calls in
`<|tool_call>`, so the reasoning trace came back as the answer and no tool ever ran. Both are read
now, and the markers never reach the answer text.

Alongside it: one documented tool-call shape across every adapter, arguments that survive their own
commas and colons, `calculator(expression="1367 * 89")` keeping its arguments,
`stop_sequences="END"` no longer cutting at the first `E`, `generate_with_tools()` taking `config`
third everywhere (both spellings still work), a chat turn with tools that can stream, and a Groq
gpt-oss model that works on the ReAct path.

### Local models

Per-call sampling keywords now reach the local engines, including `seed` and `stop_sequences`, which
the Transformers engine used to read off the config before it looked at the call. A local reasoning
model is recognised from its own chat template rather than from its name, so it gets the larger
budget instead of spending the base one on a hidden chain and returning nothing, and
`chat_template_kwargs={"enable_thinking": False}` turns the thinking off.

On a multi-GPU node, `device_map="auto"` could place a model so that sampling read invalid logits and
the run died in a CUDA assert. The engine probes the logits after loading and pins the model to one
device before sampling. The MLX engine works with the current `mlx_lm`, whose sampler and tool
schemas both moved. Device memory comes back when a model unloads, and a model that does not fit
says so instead of falling back to the CPU in silence.

### Errors that name the fix, and a rate limit that stops multiplying

A URL with no `http://` or `https://` scheme is refused, naming the environment variable it came
from. A connection failure names the endpoint the call was sent to instead of pointing at a provider
status page, which is advice about the wrong machine when the server is yours. A blank
`OPENAI_BASE_URL` no longer sends every call to nowhere. A rate limit delivered as HTTP 413 is
classified as a rate limit. `effgen tools list --category typo` names the filter and the valid
categories instead of reporting an empty registry. Library warnings render as one line instead of a
traceback block.

Every message a user reads is now bounded, redacted and ends with what to do next. Quoted upstream
text is capped at 240 characters, where one real provider body reached 42 kB, and the credential you
submitted no longer travels back out through the chained SDK exception or a rendered traceback.

And retries stopped compounding: three layers each backed off a throttled call, so one client request
became twelve upstream requests and held the caller 20.5 seconds at a stated 2-second delay. One
layer owns provider retry now, and the same measurement reads four requests and 6.7 seconds.

### Sandboxing, security and supply chain

Code run in the subprocess sandbox no longer sees your credential stores (`~/.ssh`, `~/.aws`,
`~/.gnupg`, `~/.kube`, `~/.docker`, `~/.azure`, `~/.config/gcloud`, the credential files beside them,
`/etc/shadow`, mounted secrets) and runs in its own PID namespace, so it sees one process rather than
the host's process table. Both are reported on the result as `credential_reads_masked` and
`process_table_isolated`. It is a deny-list over a known set of paths, not read confinement, and the
documentation says so. Writes stay inside the run's scratch space.

The default guardrail preset now screens tool output for injection, not just input, so an instruction
planted in a tool's return value does not reach the model. Redaction covers every credit card rather
than the first, modern provider key shapes, labeled clinical identifiers in the forms real documents
use, and system prompts on the way out. Email scanning is linear time.

Five dependency floors moved past open advisories (aiohttp, cryptography, gitpython, h2, pypdf),
floors are now declared beside every extra that reaches a package, both lockfiles were regenerated,
and the vulnerability audit passes.

### The command line, on any terminal

Twenty-two commands used to exit non-zero purely because the console could not encode a character
effGen prints. Text is now folded where it becomes bytes, so a command added later is covered, and
`--json` escapes rather than transliterates, so a French or Chinese answer survives a hard-ASCII
console byte for byte. A styled line renders in its own colours instead of having its numbers and
brackets recoloured. Output reaches a redirected stdout while the command is still running. The whole
surface works without `rich` and without `torch`, and a module that cannot import names the package
it needs and how to install it.

Piped output is clean: no spinner, no chrome on stdout, one answer per input line under `-q`, and
zero colour codes under `NO_COLOR`.

### Installing it

`./install.sh` no longer fails when there is no terminal to answer its prompts, and
`--download-models` fetches models instead of quietly skipping. A reduced install now reports what it
cannot run instead of failing it.

**Python 3.11 is the floor**, and 3.11 through 3.14 are supported. 3.14 was installed and run rather
than assumed: the unit lane passes with `.[dev]` and with `[all]` through a shipped lock. One caveat
worth knowing before you type it, since plain `pip install effgen[all]` does not resolve on 3.14:

```bash
pip install -r requirements-all-py314-lock.txt
pip install --no-deps effgen
```

### The three breaking changes

1. **Python 3.10 is no longer supported.** The floor is 3.11. `tomllib`, `asyncio.timeout`,
   `datetime.UTC` and the `TimeoutError` unification are all stdlib from there, and effGen carried a
   hand-written fallback for each. *Migration:* upgrade the interpreter; the API is unchanged.
2. **`AgentConfig.raise_on_error` defaults to `True`.** A failed run raises its typed error instead
   of returning a response whose `.output` reads like an answer. *Migration:* pass
   `raise_on_error=False` to inspect the response yourself, and the failure shape is unchanged. That
   is also the documented setting for batch evaluation, where scoring a capped run as an error
   measures the reporting style rather than the model. With the flag off, a failed run's `output` is
   effGen's report of what stopped it, and the model's own text is in `metadata["partial_output"]`.
3. **A backend that never answered raises `BackendUnreachableError`** whatever that flag says. A task
   that ran and failed is a result you can inspect; a backend that was never reached is not, and
   returning one is how a whole batch completes against nothing and still looks healthy in the
   summary. *Migration:* there is no opt-out, by design. Catch the error where you want to handle it.

One smaller change is worth knowing: four public enums are now `enum.StrEnum`, so
`str(TaskStatus.RUNNING)` reads `"RUNNING"` rather than `"TaskStatus.RUNNING"`. Equality, membership
and JSON are unchanged.

The public surface grew from 204 names to 223. Nothing was removed or renamed.

### Thanks

Two people outside the maintainer contributed code to this release. **Yasuo Tabei**
([@tb-yasu](https://github.com/tb-yasu)) taught the parser Gemma 4's channel and tool-call format and
brought the MLX engine up to the current `mlx_lm`. **Aafiya Hussain**
([@Aafiya-H](https://github.com/Aafiya-H)) found and fixed the multi-GPU placement that made sampling
read invalid logits. Thank you both.

[Full 1.0.0 changelog →](CHANGELOG.md#100---2026-08-14)

---

## v0.3.2 — July 5, 2026

**effGen v0.3.2 is another usability, robustness & polish release.** Where v0.3.1 sanded down the edges a
first wave of professionals hit, v0.3.2 keeps going — a reliability engineer re-certifying results
integrity, a trust auditor tracing every source, a security engineer red-teaming the server, an ETL
engineer running batch at volume, a clinical analyst who cannot leak a patient identifier, an SRE living
in `/metrics`, a localization specialist, a CI gatekeeper, a non-technical operator, a game writer, a
plugin author, a FinOps owner watching spend, and a document specialist feeding it messy files. It adds no
new providers and no new subsystems; it makes the surfaces you already reach for predictable, and turns
every quiet trap into a clear, typed error. There are no breaking API changes.

### Structured output, cost gates, and document input reach the command line

The two things power users kept dropping into Python for are now on the CLI. `effgen batch --schema
schema.json` validates every row against a JSON Schema (or `--output-model module:Class`), and the written
file is lossless — each row carries its cost, tokens, the parsed object, and, if it failed, the reason.
`effgen eval --fail-under 0.8` turns evaluation into a real CI gate that drives the exit code, and a
detected regression under `--compare-baseline` now fails the build instead of passing silently. `effgen
compare --optimize cost` adds a `$/run` column and picks the cheapest good-enough model. And `effgen run
--file report.pdf` finally lets a CLI-first user hand effGen a document or an image without writing a line
of Python.

```bash
effgen batch --input tickets.jsonl --output out.jsonl -m groq:llama-3.1-8b-instant --schema schema.json
effgen eval --suite cases.jsonl -m groq:llama-3.1-8b-instant --fail-under 0.9   # exit 1 if it drops
effgen compare --models "groq:llama-3.1-8b-instant,gemini:gemini-3.1-flash-lite" --suite cases.jsonl --optimize cost
effgen run "What was Q3 revenue?" --file report.pdf -m groq:llama-3.1-8b-instant
```

### Redaction you can defend, and a server that fails with the right status code

For anyone handling regulated data, `PIIGuardrail` now removes the labeled clinical identifiers it used to
leave behind — patient name, date of birth, medical record number, address, member ID — matches
space-separated and undelimited SSNs, takes `custom_patterns` for site-specific formats, and can fail
closed in strict mode. A new `phi` guardrail preset packages it. On the server side, a failed
non-streaming completion now returns a real 4xx/5xx error envelope instead of an HTTP 200 with the error
stuffed into the answer, so an OpenAI-client pipeline raises instead of trusting a failure as a result.

```python
from effgen import PIIGuardrail, get_guardrail_preset

g = PIIGuardrail(action="redact", custom_patterns=[(r"MRN[:#]\s*\d+", "[MRN REDACTED]")])
print(g.check("Jane Doe  DOB: 1980-02-14  MRN: 55123").modified_content)
# "[NAME REDACTED]  DOB: [DOB REDACTED]  MRN: [MRN REDACTED]"

chain = get_guardrail_preset("phi")   # redaction + fail-closed strict mode
```

### Grounding that never vanishes, sampling that takes effect

Native web search used to hand back an empty `response.sources` whenever the model answered without inline
citations — even though the search ran. Now the URLs it searched are surfaced as sources, so an answer
built on a search always carries its provenance. And the creativity knobs a writer reaches for —
`seed`, `frequency_penalty`, `presence_penalty`, `top_k` — now actually reach the model through
`Agent.run()` and `AgentConfig`, instead of being silently accepted and ignored; an unknown keyword is now
rejected outright instead of swallowed.

```python
from effgen import Agent
from effgen.core.agent import AgentConfig

agent = Agent(config=AgentConfig(name="writer", model="groq:llama-3.1-8b-instant"))
agent.run("Write an atmospheric opening line.", seed=7, frequency_penalty=0.4)   # reproducible, less repetitive
```

### Observability an on-call rotation lives on, and batch that survives real data

Server `/metrics` now carries the provider, model, and status-code labels you actually alert on, so you
can graph error rate by status and cost by model straight from Prometheus — and the alerting and SLO
building blocks (`AlertWebhook`, `SLOTracker`, `check_slo_and_alert`) are exported top-level with a thin
bridge that fires a webhook when a rule trips. Batch jobs no longer die on one malformed input line (it's
skipped and reported by file and line number), print a per-job cost total, and can `--resume` a partial
run. Spreadsheets ingest into RAG, a directory ingest never silently drops a file it can't parse, the
`general` preset runs on Gemini, workflow YAML honors an `edges:` block, and the prompt library validates
your input against the template's own schema instead of billing silently-wrong output.

### Upgrade

`pip install --upgrade effgen`. Everything above is additive — no code changes required. The public API
grew by the seven alerting/SLO exports (197 → 204 names); nothing was removed or renamed.

---

## v0.3.1 — June 29, 2026

**effGen v0.3.1 is a real-world usability & polish release.** Where v0.3.0 hardened the framework, v0.3.1
sands down the edges that real professionals hit the moment they sit down with it — an analyst extracting
figures, a journalist tracing sources, a researcher serving local models, a founder shipping a weekend
MVP, a backend engineer deploying the server, a support lead wiring up agents, an educator building a
tutor, a security reviewer red-teaming the tools, and a legal knowledge manager standing up a domain
assistant. It adds no new providers or subsystems; it makes the things you already reach for honest and
delightful. There are no breaking API changes.

### Your results now carry their evidence

The single most-requested fix: `response.sources` and `response.citations` are no longer empty. A run now
surfaces the URLs its tools actually retrieved — and provider-native grounding (OpenAI citation
annotations, Gemini search grounding) — so you can verify and link them programmatically instead of
regexing prose. Only real, retrieved URLs land there; the research preset is told to cite only what its
tools returned, never to invent a source.

### Reasoning models finish the job, and every result is measurable

Reasoning models (the `gpt-5` family, `o`-series) spend output budget on hidden thinking, so the old
1024-token default could be consumed entirely and hand back an empty — but billed — answer on a
token-heavy task. They now get room to finish, a length-truncated empty is grown and retried once (or
fails with a clear "increase `max_tokens`"), and a starved budget is never retried three times. Every
`AgentResponse` now carries `cost_usd`, token counts, and `latency_ms` in its metadata (local models stay
honestly cost-free), teams and workflows report their summed cost, and sub-cent SLM costs finally show
real digits instead of `$0.0000`.

### Your persona is finally honored everywhere

A custom `system_prompt` — a Socratic tutor, a fixed-language assistant — was silently dropped on the
direct, streaming, and native-tool paths, so it *looked* applied but wasn't. Now it steers every response
on every provider. `effgen chat` gained `--system-prompt/--persona`, there's a new `education.*` prompt
set, and a knowledge domain becomes a runnable agent in one call: `LegalDomain().to_agent("gpt-5-nano")`
wires the domain's prompt, tools, and guardrails together.

### Honest teams, an honest server, and safe code execution

Multi-agent orchestration tells the truth now: a failed collaborator fails the team, hierarchical teams
route each subtask to the worker the manager *named*, and a workflow never runs a node downstream of a
failure (so an internal error can't become a customer-facing reply). The OpenAI-compatible server stops
silently downgrading — an unhosted client tool is rejected with a clear `400` instead of vanishing, and
`/v1/embeddings` reflects its real backend instead of quietly serving lexical hash vectors under a neural
model's name. And the Python REPL's sandbox switch is out of the model's hands entirely: unrestricted
execution is a developer-only opt-in, the `bash` env scrub now covers every provider credential, and the
guardrails catch more injections and redact leaked keys.

### Local-first truth and a dependable CI citizen

`effgen models status` shows physical GPU memory across all processes (so you can see which card is
actually free), `models info` recognizes a model in your own cache instead of routing you to the cloud,
local batch is thread-safe, and small local models can emit schema-valid JSON via the new optional
`effgen[grammar]` extra. For automation, the synchronous `Agent.run()` no longer hangs forever on an MCP
tool, installed tool plugins auto-discover, and `effgen run --json` (plus `eval`/`compare`/`workflow`/
`sessions list`) emits clean JSON to stdout for `jq`.

### Try it

```python
from effgen import create_agent, LegalDomain

# Grounded research with traceable sources and honest cost.
agent = create_agent("research", "openai:gpt-5-nano")
r = agent.run("What is the capital of France? Cite a source.")
print(r.text)                  # "...Paris (Source: https://en.wikipedia.org/wiki/Paris)."
print(r.sources)               # ['https://en.wikipedia.org/wiki/Paris']

# A knowledge domain → a runnable agent in one call.
print(LegalDomain().to_agent("openai:gpt-5-nano")
      .run("What does an NDA confidentiality clause protect?").text)
```

```bash
effgen run --json -q "What is 25 * 17?" | jq .output   # pure-JSON stdout for CI
effgen models status                                    # physical GPU memory; which card is free
```

```bash
pip install --upgrade effgen
```

---

## v0.3.0 — June 19, 2026

**effGen v0.3.0 is the stabilization release.** It adds no new features — instead it takes everything
effGen already does and makes it *robust, predictable, fast, secure, and genuinely pleasant to use*.
There are no breaking API changes; every ergonomic improvement is an additive alias.

### robust failures, never silent

The biggest change is one you'll feel immediately: effGen no longer lies about success. `Agent.run()`
can no longer return `success=True` with an empty answer. A bad model id, a missing key, or a provider
outage now produces `success=False` with a typed, redacted error and a consistent `reason` — the same
shape whether you used tools or not. Retries fire only when retrying could help; an auth error stops
once, clearly, instead of storming. A 404 model id suggests the nearest live alternative.

### A model catalog that updates itself

effGen ships a local snapshot of every provider's models — prices, context windows, capabilities,
free-tier flags — with a count and a "verified on" date. `effgen models refresh` pulls the live list
and tells you exactly what changed; effGen warns (once, never spammily) when its catalog looks stale.
Cerebras, OpenAI, and the rest now reflect what the live APIs actually serve, and private fine-tune
ids and non-chat models never pollute the catalog.

### Real GPU support, secure server, hardened tools

The documented GPU install now yields a usable GPU (or a loud, correct warning), `temperature=0`
decodes greedily instead of crashing, and the GPU allocator no longer deadlocks. The API server
**fails closed** — a forged JWT can't reach a protected route, CORS and the metrics dashboard are
locked down, and a missing upstream key returns 502, not a misleading 401. Built-in tools are
sandboxed: the Python REPL enforces its timeout from outside the code, every URL tool shares one
SSRF guard, file tools are path-confined, and unsafe pickle/`eval` paths are gone.

### Fast, consistent, and a joy to use

`import effgen` dropped from ~7.5 s to about **20 ms** thanks to lazy loading. Streaming is genuinely
incremental, the agent loop stops calling tools once it has a confident answer (a task that took 6
tool calls and 66 seconds now takes 1), and structured output is faster and honest about failures.
The CLI is quiet and scriptable (`--json` everywhere, `--provider` on `run`/`chat`/`debug`, non-zero
exit codes), the obvious constructor calls just work, and a new live "thinking" UX, rotating tips,
"did you mean?" suggestions, rich Markdown rendering, and a polished `effgen chat` make the first five
minutes feel easy. `pip-audit` is clean across the documented extras and `pypdf` is patched.

### Try it

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")
agent = Agent(config=AgentConfig(name="math_agent", model=model, tools=[Calculator(), PythonREPL()]))
print(agent.run("What is 24344 * 334?").output)
```

```bash
effgen models refresh                 # update the catalog from the live provider APIs
effgen doctor --live --cheap          # check which provider keys are actually usable
effgen run --provider groq "Summarize the theory of relativity in two sentences."
```

---

## v0.2.10 — May 27, 2026

**effGen v0.2.10** ships the **Security, Edge & Developer Experience** layer — hardening effGen end-to-end from secret scanning to production deployment to everyday developer ergonomics.

### What's new at a glance

**Secret scanning and SBOM.** Gitleaks pre-commit hook + CI workflow catch secrets before they reach the repo. A CycloneDX 1.5 SBOM (`sbom.cdx.json`) is generated and validated on every push and uploaded as a release artifact. `pip-audit` CI fails on any HIGH/CRITICAL vulnerability.

**Supply-chain integrity.** `EFFGEN_VERIFY_HASHES=1` compares installed-wheel hashes against the lockfile at startup and logs `hash_verification: ok` or `hash_verification: drift <package>`. `requirements-all-lock.txt` (uv-generated, with `google-protobuf` floors and `fireworks-ai<0.18` cap) makes `pip install .[all]` fully reproducible.

**Sandboxed CodeExecutor.** LLM-generated Python no longer runs on the host by default. `SubprocessSandbox` uses rootless user-namespace isolation (`unshare --map-root-user --net --pid --mount`) — no `CAP_SYS_ADMIN` required. `DockerSandbox` adds `--read-only --network=none --cap-drop=ALL --pids-limit=100 --memory=256m`. `FirecrackerSandbox` stub ships for v0.3. `EFFGEN_SANDBOX_BACKEND=docker|subprocess|off` selects the backend; `off` emits a loud warning and is never auto-selected.

**OAuth2/OIDC + RBAC + Audit Log.** The API server now validates Bearer JWTs via `authlib` (configurable issuer/JWKS) in non-dev mode. `Role` objects map JWT claims to `allowed_tools`, `allowed_models`, and `max_cost_per_day`. A `RBACBudgetMiddleware` enforces these per-request — returning 403 for disallowed tools and 429 for budget exhaustion. Every request/response pair is appended to `~/.effgen/audit/<date>.jsonl` (content redacted). `EFFGEN_DEV_MODE=1` disables auth for local development.

**Docker and Helm.** A multi-stage `deploy/docker/Dockerfile` produces a slim non-root image with a `/health` healthcheck. A full Helm chart (`deploy/k8s/helm/effgen/`) ships with Deployment, Service, Ingress, ConfigMap, Secret, ServiceAccount, NetworkPolicy, PDB, HPA (CPU + custom `effgen_model_call_latency_seconds` metric), and PVC templates.

**AWS Lambda.** `deploy/aws_lambda/handler.py` wraps the FastAPI app in Mangum (`lifespan="off"`). `ProviderRegistry` is preloaded at module level so cold starts land under 3 s; warm calls are under 100 ms. Per-invocation timeout budget is enforced — overruns return 504. A SAM template (`sam-template.yaml`) wires HTTP API → Lambda → SecretsManager.

**Cloudflare Worker edge proxy.** `deploy/cloudflare/worker.js` handles CORS, Bearer JWT validation, fixed-window KV-backed rate limiting, and upstream forwarding (with `duplex:"half"` for streaming bodies) at the edge. `wrangler.toml` defines routes, KV bindings, and staging/production environments.

**VSCode extension.** `tools/vscode-effgen/` provides prompt-template completion, an inline "Run" code lens on `LibraryPrompt` definitions, and hover docs. Compiled with TypeScript 5.3 strict (0 errors); publishable as a `.vsix`.

**Jupyter magics.** `%effgen_chat <message>` for one-shot chat, `%%effgen_agent <preset>` for cell-body task execution with a tool trace, and `%effgen_metrics` for a Prometheus snapshot. Load with `%load_ext effgen.jupyter`.

**Local dashboard.** The API server now serves a live SPA at `/dashboard` (public, no auth required). Panels: real-time span stream (SSE), `/metrics` summary, recent agent runs with token counts and cost, SLO burn rates. `/dashboard/data.json` exposes the same data as structured JSON.

### CLI quick-start

```bash
# Verify secret-scanning pre-commit hook is installed
pre-commit install && pre-commit run gitleaks

# Start server with dev mode (auth disabled)
EFFGEN_DEV_MODE=1 effgen serve --port 8000

# Check the dashboard
open http://localhost:8000/dashboard

# Docker
docker build -f deploy/docker/Dockerfile -t effgen:0.2.10 .
docker run -p 8000:8000 --env-file .env effgen:0.2.10

# Helm (Kubernetes)
helm lint deploy/k8s/helm/effgen/
helm install effgen deploy/k8s/helm/effgen/

# AWS Lambda (SAM)
cd deploy/aws_lambda && sam build && sam deploy --guided

# Cloudflare Worker
cd deploy/cloudflare && wrangler deploy
```

### Python API quick-start

```python
# Sandboxed code execution
import asyncio
from effgen.security.sandbox import get_sandbox, SandboxConfig

async def run_sandboxed():
    config = SandboxConfig(backend="subprocess", timeout=10)
    sandbox = await get_sandbox(config)
    result = await sandbox.run('print("hello, sandbox")', "python", config)
    print(result.stdout)  # hello, sandbox

asyncio.run(run_sandboxed())

# Jupyter (inside a notebook)
# %load_ext effgen.jupyter
# %effgen_chat "What is the square root of 169?"
# %%effgen_agent general
# Summarise the top HackerNews stories today.
```

### Upgrading from v0.2.9

No breaking API changes. All new modules are additive; existing `Agent`, `load_model`, and tool APIs are unaffected.

```bash
pip install --upgrade effgen
```

---

## v0.2.9 — May 23, 2026

**effGen v0.2.9** ships the **Observability & Reliability** layer — everything you need to run effGen agents confidently in production. Structured JSON logs with automatic secret redaction, OpenTelemetry traces with configurable sampling, Prometheus histograms, SLO burn-rate tracking, circuit breakers, bulkheads, jittered retries, a deterministic chaos harness, a Hypothesis-based fuzz suite, a load-testing CLI (`effgen loadtest`), and six Alertmanager-compatible alert rules. All telemetry is async/non-blocking — a failed export never fails inference.

### What's new at a glance

**Structured logging with secret redaction.** `get_logger(__name__)` emits JSON lines with `{ts, level, module, event, attributes, trace_id, span_id}`. A built-in `Redactor` catches OpenAI, Anthropic, Cerebras, Google, HuggingFace, Groq, Bearer token, Slack, and Discord webhook patterns at the encoder — secrets can't slip through any log path.

**Prometheus histograms.** Four new metrics with full label dimensions: `effgen_model_call_latency_seconds{provider,model,outcome}`, `effgen_tool_call_latency_seconds{tool,outcome}`, `effgen_agent_iteration_latency_seconds{preset}`, and `effgen_tokens_total{provider,model,kind}`. The existing `/metrics` endpoint now emits histogram buckets in valid Prometheus text format.

**SLO tracking.** `SLOTracker` maintains a rolling window per SLO (name, target%, window duration). `burn_rate(name)` returns the current error-budget consumption rate. Results are exposed at the `/slo` FastAPI endpoint.

**Configurable tracing samplers.** Choose `AlwaysOn`, `AlwaysOff`, `ParentBased(TraceIdRatio(p))`, or `ParentBased(RateLimited(per_second))` via `ObservabilityConfig`. A canonical span-attribute spec (`effgen/observability/spans.py`) is the single source of truth for all attribute names — no more scattered string literals.

**Reliability primitives.** Four building blocks now wrap every adapter call:
- **Timeouts** — `ReliabilityConfig.default_timeouts` with `{model_call: 60, tool_call: 30, http: 20}`. Explicit timeouts on every adapter httpx client; audit guard raises if any are missing.
- **Retries** — `@retryable(Retry(...))` with jittered exponential backoff. Handles 5xx, 429 + Retry-After, transient network errors. Emits OTel `effgen.retry.attempt` events.
- **Circuit breaker** — `CircuitBreaker` (CLOSED → OPEN → HALF_OPEN) per provider via `CircuitBreakerRegistry`. Isolates a misbehaving provider automatically.
- **Bulkhead** — `Bulkhead(max_concurrency, queue_size, queue_timeout)` per provider via `BulkheadRegistry`. Prevents one provider from starving others.

**Deterministic chaos harness.** `Chaos(seed)` injects `NetworkTimeout`, `Http5xx`, `Http429`, `SlowResponse`, `PartialResponse`, or `MalformedJSON` faults into the provider middleware. Four canonical scenarios (fallback on 5xx, Retry-After honoured, timeout fires cleanly, AllProvidersFailed no silent empty string) each pass across 10 seeds — 273 tests, all deterministic.

**Fuzz suite.** Hypothesis-based fuzz tests cover all 66 `BaseTool` subclasses (500 examples each), random `ContentPart` message sequences, and the router's provider-availability logic. No unhandled exceptions, no secret leaks across 164 test/500-example combinations.

**Load-testing CLI.** `effgen loadtest --concurrency 10 --duration 30 --scenario fixed` runs a mock or live load test and writes a JSON report with throughput, p50/p95/p99 latency, and error rate.

**Alerting.** `docs/observability/alert_rules.yaml` contains six Alertmanager-compatible rules (error rate, p95 latency, cost burn, SLO fast-burn, SLO slow-burn, circuit-breaker open). `AlertWebhook(url).fire(alert)` posts to Slack or Discord; never raises even if delivery fails.

### CLI quick-start

```bash
# Run a load test against the mock model
effgen loadtest --concurrency 10 --duration 30

# Run against Cerebras live
effgen loadtest --provider cerebras --model llama3.1-8b --concurrency 5 --duration 15

# Check SLOs
curl http://localhost:8000/slo

# Check Prometheus metrics
curl http://localhost:8000/metrics | grep effgen_model_call_latency
```

### Python API quick-start

```python
from effgen.observability import get_logger
from effgen.reliability.retry import Retry, retryable
from effgen.reliability.circuit import CircuitBreaker

log = get_logger(__name__)
log.event("demo.started")

breaker = CircuitBreaker("my_provider", failure_threshold=5, recovery_timeout=30)

@retryable(Retry(max_attempts=3, base_delay=1.0, jitter=True))
def call_api():
    if not breaker.is_call_permitted():
        raise RuntimeError("circuit open for my_provider")
    try:
        result = ...  # your adapter call here
        breaker.on_success()
        return result
    except Exception as exc:
        breaker.on_failure(exc)
        raise
```

### Upgrading from v0.2.8

No breaking API changes. All new modules are additive; existing code is unaffected.

```bash
pip install --upgrade effgen
```

---

## v0.2.8 — May 21, 2026

**effGen v0.2.8** ships first-class **multimodal input** — send images, audio, and video to any capable provider through a single, unified `Message` schema. Six providers (Gemini, OpenAI, Groq, Anthropic, Together, HuggingFace) gain structured multimodal routing with automatic preprocessing, capability-gated error surfaces, a new `multimodal` preset, and five end-to-end cookbook walkthroughs. No breaking API changes.

### What's new at a glance

**Unified `Message` schema.** `Message.content` is now a typed `List[ContentPart]` — a union of `TextPart`, `ImagePart`, `AudioPart`, `VideoPart`, `ToolCallPart`, and `ToolResultPart`. The old string constructor still works: `Message(role, "hello")` auto-wraps in a `TextPart`. Validation fires on construction, not at send time.

**Three `_from` helpers.** `image_from(source)`, `audio_from(source)`, and `video_from(source, fps=1)` accept `bytes`, a local path, a URL, a `PIL.Image`, or an `np.ndarray` — whichever is convenient. MIME type is inferred automatically.

**Preprocessing is explicit and loggable.** `image_pre.prepare()` enforces per-provider pixel/byte limits and Lanczos-downscales when needed, logging every action to `part.meta["preprocessing"]`. `audio_pre` downsamples to 16 kHz and chunks long clips. `video_pre` samples keyframes via ffmpeg (raising `MissingSystemDependency` with OS-specific install hints when absent).

**Image input across 6 providers.** Gemini, OpenAI gpt-4o, Groq Llama 4 / Llama 3.2-vision, Anthropic (code only), Together, and HF BLIP/LLaVA all accept `ImagePart`. Every adapter raises `CapabilityNotSupportedError` cleanly when the selected model doesn't support vision — no silent text fallback.

**Audio input across 3 providers.** Gemini native audio, OpenAI Whisper (`/audio/transcriptions`) + gpt-4o audio, and HF ASR. Anthropic raises `CapabilityNotSupportedError(Capability.audio_input)`.

**Video input — native + frame-sampling.** Gemini 2.x/3.x accepts raw video natively. All other adapters decompose a `VideoPart` into a sequence of `ImagePart`s (frame sampling) plus an optional `AudioPart` from the audio track.

**`multimodal` preset.** `create_agent("multimodal", model)` wires Gemini Flash-Lite as primary (vision + audio + video) with OpenAI gpt-4o-mini as vision fallback. The preset ships with `ImageInfoTool`, `ImageCaptionTool`, `OCRTool`, `AudioTranscribeTool`, `PDFTool`, `WeatherTool`, and the new `MultimodalDescribeTool` — which automatically chooses the right tool based on the input part type.

**MLX-VLM adapter.** `effgen/models/mlx_vlm_engine.py` wraps `mlx-vlm` for Apple Silicon vision-language inference. Raises `MissingSystemDependency` on non-Apple hardware or missing library. Live tests skipped on Linux; 28 unit tests with fakes pass.

**5 cookbook walkthroughs.** Image Q&A, audio transcribe + reason, video summarize, OCR + LLM structured extraction, chart reading from an image. Each is a runnable Python snippet with prose. See `docs/cookbook/README.md`.

### CLI quick-start

```bash
# Image Q&A via multimodal preset
effgen run --preset multimodal "What is in this image?" --image /tmp/photo.jpg

# Check which providers support vision
python -c "
from effgen.models.capabilities import Capability
from effgen import list_models
print([m for m in list_models('gemini') if Capability.vision in m.get('capabilities', [])][:3])
"
```

### Python API quick-start

```python
from effgen import image_from, audio_from, video_from, load_model
from effgen.core.messages import Message, Role
from effgen.presets import create_agent

model = load_model("gemini-2.0-flash", provider="gemini")
agent = create_agent("multimodal", model)

# Image
img = image_from("https://example.com/photo.jpg")
result = agent.run_message(Message(role=Role.USER, content=[img, "Describe this."]))

# Audio
aud = audio_from("/tmp/interview.mp3")
result = agent.run_message(Message(role=Role.USER, content=[aud, "Summarize in one line."]))

# Video (requires ffmpeg for frame-sampling fallback)
vid = video_from("/tmp/clip.mp4", fps=1)
result = agent.run_message(Message(role=Role.USER, content=[vid, "What happens in the first 5 seconds?"]))
```

### Upgrading from v0.2.7

No breaking API changes. The old string-based `Message` constructor is unchanged.

```bash
pip install --upgrade effgen
```

---

## v0.2.7 — May 20, 2026

**effGen v0.2.7** ships the **Prompt Library** — a curated, domain-organized catalog of **31 reusable prompt templates** covering research, coding, data/SQL, legal, medical, creative writing, and business. Every template is a Python callable that renders deterministically for fixed inputs, ships with a fixture and golden evaluation test, and is accessible through a rich CLI and an interactive playground.

### What's new at a glance

**31 templates across 7 domains.** Research (literature review, paper summary, citation extraction, methodology critique), Coding (code review, bug diagnosis, refactoring plan, test generation, docstring fill), Data (NL-to-SQL, SQL explain, SQL optimize, data profile, ETL plan), Legal (contract summary, clause classify, research brief), Medical (symptom triage, drug interaction, medical literature), Creative (story continuation ×2, poetry forms, character bio, world building), and Business (meeting summary, email draft, OKR generation, SWOT analysis, elevator pitch).

**Golden + live eval harness.** `effgen prompts eval` renders every template with its fixture and compares against a stored golden. Add `--live --model <name>` to run prompts through a real model and validate output shape — including `sqlglot.parse()` for SQL templates and `ast.parse()` for generated Python.

**Interactive playground.** `effgen prompts playground` opens a REPL where you can select any template, set its inputs, render a preview, run it against a model, and save the session to JSON. Non-interactive `effgen prompts render` and `effgen prompts run` modes are also available for scripts.

**Legal and medical safety.** Every legal and medical template renders the required non-advice disclaimer verbatim in the system prompt — enforced by unit tests, not convention.

**Auto-generated gallery.** `docs/prompts/gallery.md` lists all 31 templates with their variant and one-line description. Regenerate it any time with `effgen prompts list --format markdown`.

### CLI quick-start

```bash
# Discover templates
effgen prompts list
effgen prompts list --domain research --variant cot
effgen prompts list --format markdown

# Inspect a template
effgen prompts show research.literature_review.v1.cot

# Run golden evaluations (no model needed)
effgen prompts eval

# Run live evaluations (requires API key)
effgen prompts eval --domain coding --live --model llama3.1-8b

# Interactive playground
effgen prompts playground

# Non-interactive render
effgen prompts render data.sql_from_nl.v1 --input '{"schema_ddl": "CREATE TABLE orders (id INT, total FLOAT)", "question": "Total orders this month", "dialect": "sqlite"}'
```

### Python API quick-start

```python
from effgen.prompts.library import registry

# Browse all templates
for p in registry.all():
    print(p.name, p.variant, p.domain)

# Get and render a specific template
p = registry.get("research.literature_review.v1.cot")
prompt_text = p.template(
    topic="diffusion models",
    years_range="2022-2025",
    max_papers=10
)

# Search by domain and variant
sql_prompts = registry.search(domain="data", variant="structured")
```

### Upgrading from v0.2.6

No breaking API changes. All prompt library classes are opt-in additions.

```bash
pip install --upgrade effgen
```

---

## v0.2.6 — May 19, 2026

**effGen v0.2.6** is a document, media, and communication tools release that adds **14 new built-in tools** — OCR, audio transcription, image analysis, document parsing (PDF/DOCX/Excel), geo/weather, and email/webhook — raising the total built-in tool count from 44 to **58+**. Two new presets (`media`, `notify`) join the existing roster. No breaking API changes.

### New Tools at a Glance

**OCR** — `OCRTool` extracts text from images using Tesseract locally, with OCR.space as a free API fallback. Raises `OCRBackendUnavailable` with per-OS install instructions when neither backend is available. Added to `general` preset.

**Audio Transcription** — `AudioTranscribeTool` transcribes audio files locally via `faster-whisper` (CPU/GPU auto-detected), falling back to HuggingFace Inference when `HF_TOKEN` is set. Warns on CPU when a large model size is selected. Added to new `media` preset.

**Image Analysis** — `ImageInfoTool` extracts image metadata and performs local resize/thumbnail operations entirely via Pillow (zero network). `ImageCaptionTool` uses the effGen model router to select a vision-capable provider (Gemini / OpenAI / MLX-VLM) and generate a natural-language caption or description. `ImageInfoTool` is in `general`; `ImageCaptionTool` is in `media`.

**Document Parsing** — `PDFTool` (pypdf + pdfplumber), `DOCXTool` (python-docx), and `ExcelTool` (openpyxl + pandas) round-trip local documents with full text, table, and metadata extraction. All three added to both `research` and `general` presets.

**Geo / Weather** — `WeatherTool` fetches current, forecast, and historical weather from Open-Meteo (free, no auth). `GeocodeTool` forward/reverse geocodes via Nominatim (OSM) with 1 req/s token-bucket rate limiting and proper User-Agent header. `MapsTool` renders static PNG maps from OSM tiles via the `staticmap` library. All three added to `general`.

**Email** — `EmailSMTPTool` sends email via SMTP (stdlib `smtplib`, TLS on by default). `EmailIMAPTool` reads email via IMAP (stdlib `imaplib`). Both raise `MissingCredentialsError` when env vars are absent. Added to new `notify` preset.

**Webhooks** — `SlackWebhookTool` and `DiscordWebhookTool` post messages to Slack and Discord via incoming webhook URLs (no OAuth). Webhook URLs are redacted in all logs. Both added to `notify` preset.

### New Presets

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("llama3.1-8b", provider="cerebras")

# Media processing agent
media_agent = create_agent("media", model)   # AudioTranscribeTool + ImageCaptionTool

# Notification/alert agent
notify_agent = create_agent("notify", model) # EmailSMTP + EmailIMAP + Slack + Discord
```

### Upgrading from v0.2.5

No breaking API changes. All new tools are opt-in extras.

```bash
pip install --upgrade "effgen[all]"
# or selectively:
pip install --upgrade "effgen[documents]"   # PDFTool, DOCXTool, ExcelTool
pip install --upgrade "effgen[audio]"       # AudioTranscribeTool
pip install --upgrade "effgen[tools]"       # OCRTool, ImageInfoTool, and more
```

**System dependencies** (only needed for the relevant tool's primary path):
- `OCRTool` Tesseract: `apt-get install tesseract-ocr` / `brew install tesseract`
- `AudioTranscribeTool` ffmpeg (for non-WAV): `apt-get install ffmpeg` / `brew install ffmpeg`

---

## v0.2.5 — May 18, 2026

**effGen v0.2.5** is a tools-focused release that adds **13 free, no-auth-required tools** across six new categories — academic research, news aggregation, YouTube, social media, translation/language detection, and QR codes — bringing the total built-in tool count above 44. Every new tool ships with structured `{success, data, error}` output, preset integration, and a dedicated doc page.

### New Tools at a Glance

**Academic Research** — `PubMedTool` (NCBI E-utilities, 3 operations, built-in token-bucket rate limiter), `ArXivTool` (Atom feed search + PDF download), `SemanticScholarTool` (paper search + citations + references with polite backoff). All three are now part of the `research` preset.

**News & RSS** — `RSSFeedTool` fetches and full-text-searches any RSS/Atom feed; `NewsTool` aggregates top headlines across a curated list of reputable sources (Reuters, BBC, Hacker News, NPR, Al Jazeera, and more) with an optional NewsAPI.org key for better relevance. Both are in the `research` and `general` presets.

**YouTube** — `YouTubeTranscriptTool` pulls captions from public videos without a Google API key (via `youtube-transcript-api`), with URL extraction for watch?v=, youtu.be/, and shorts/ formats. `YouTubeMetadataTool` retrieves video and channel metadata via `yt-dlp` in metadata-only mode. Both added to `research`.

**Social Media** — `RedditTool` reads top/hot posts, user submissions, and thread comments from Reddit's public JSON endpoints (no OAuth). `HackerNewsTool` covers top/new stories, items, and user profiles from HN's Firebase API. Both added to `research` and `general`.

**Translation & Language Detection** — `TranslateTool` translates text between languages using LibreTranslate (configurable endpoint) with an offline `argostranslate` fallback; language packs are cached in `~/.effgen/argos/`. `LanguageDetectTool` detects language in text or batches, fully offline via `langdetect` (55+ languages). Both added to `general`.

**QR Codes** — `QRGenerateTool` generates QR codes locally from any text or URL, returning a base64 PNG or saving to a file path. `QRReadTool` decodes QR codes and barcodes from image files or base64 PNG using `pyzbar` + Pillow, with an OpenCV QR fallback when `libzbar` is unavailable. Both added to `general`.

### Tool Gallery

A new `docs/tools/gallery.md` file provides a one-line description and a working quickstart snippet for every tool in the effGen ecosystem — useful for discovering what's available at a glance.

### Upgrading from v0.2.4

No breaking API changes. All new tools are opt-in; existing code is unaffected. Install the new tool extras:

```bash
pip install --upgrade "effgen[tools]"   # feedparser, youtube-transcript-api, yt-dlp, pyzbar, qrcode, opencv, langdetect
# or grab everything:
pip install --upgrade "effgen[all]"
```

---

## v0.2.4 — May 14, 2026

**effGen v0.2.4** makes multi-provider AI inference production-grade with a composable **ModelRouter** — a new opt-in layer that sits between your application code and the 9 cloud providers effGen already supports. Instead of hard-coding a provider, you describe what you need (cheapest call within budget, fastest that meets an SLA, prefer free tier, fall back to paid) and the router picks the right provider, records its reasoning, and transparently retries or fails over when things go wrong.

### Top Highlights

1. **Three composable routing policies** — mix and match to build exactly the routing logic you need:

   ```python
   from effgen import PolicyBasedRouter, RoutingContext, CostBasedPolicy, LatencyBasedPolicy
   from effgen.models.capabilities import Capability

   router = PolicyBasedRouter(
       policies=[LatencyBasedPolicy(), CostBasedPolicy()],
   )
   context = RoutingContext(
       prompt_tokens_estimate=500,
       user_budget_usd=0.01,
       latency_budget_ms=3000,
       required_capabilities={Capability.chat, Capability.tools},
   )
   decision = router.route(context)
   print(decision.chosen)        # e.g., ProviderModelPair("cerebras", "llama3.1-8b")
   print(decision.eliminated)    # list of (pair, reason) — fully explainable
   ```

2. **Transparent failover** — `route_and_execute(context, fn)` automatically retries on `RateLimitExceeded`, 5xx errors, or timeouts and moves to the next-best provider. Each failover fires a `RouterEvent` to any registered subscribers so you can log or alert in real time.

3. **Cross-process rate-limit coordination** — `SQLiteRateLimitStore` (WAL-mode, `BEGIN IMMEDIATE`) lets multiple workers share a single rate-limit budget at `~/.effgen/rate_limits.sqlite`. Pass it into `RateLimitCoordinator(storage=store)` — the default in-memory mode is unchanged.

4. **Persistent cost tracking + `effgen cost` CLI** — every API call writes a row to `~/.effgen/costs.sqlite`. Query it instantly:

   ```bash
   effgen cost today          # per-provider per-model table
   effgen cost week           # rolling 7-day view
   effgen cost by-provider    # lifetime totals
   effgen cost set-budget 1.0 # set $1/day cap
   ```

   When cumulative daily spend hits 80% of your cap, effGen emits a warning; at 100% it raises `BudgetExceededError` — which the router treats as retriable and automatically fails over to a free-tier provider.

5. **Fully explainable decisions** — every `RouterDecision` carries the chosen provider, a list of eliminated candidates with per-provider reasons (`"rate_limited"`, `"no_key"`, `"cost_exceeds_budget"`, `"latency_exceeds_sla"`), the winning policy name, and a numeric score. Nothing is a black box.

### Upgrading from v0.2.3

No breaking API changes. All existing `load_model`, `Agent`, and direct adapter paths work without modification. The `ModelRouter` is a completely opt-in new layer.

```bash
pip install --upgrade effgen
```

`RateLimitCoordinator` and `CostTracker` both retain their existing in-memory defaults — existing code that constructs them without a `storage=` argument is unaffected.

---

## v0.2.3 — May 4, 2026

**effGen v0.2.3** grows the provider roster from 4 to **9 cloud inference backends** — Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference join the existing OpenAI, Anthropic, Gemini, and Cerebras adapters. Every new backend ships with streaming, native tool-calling where the provider supports it, automatic rate-limit coordination, and per-call cost tracking. A new `ProviderRegistry` consolidates all providers for clean introspection and the `effgen doctor` command tells you at a glance which API keys are wired up. A backend parity matrix proves that the canonical "What is (17 × 23) + sqrt(144)?" agentic task returns the correct answer (403) across every provider, with identical `ModelAuthError` raised on bad credentials.

### Top Highlights

1. **5 new cloud backends** — `GroqAdapter` (16 models, RPM/TPD windows), `TogetherAdapter` (163-model catalog with drift detection), `FireworksAdapter` (80 chat models), `ReplicateAdapter` (async run-poll + SSE streaming + timeout handling), `HFInferenceAdapter` (124-model HuggingFace Router catalog + custom Endpoint URL support). Each supports streaming and native tools.

   ```python
   from effgen import load_model

   # Groq — ultra-fast inference
   model = load_model("llama-3.1-8b-instant", provider="groq")

   # Together AI
   model = load_model("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider="together")

   # Fireworks
   model = load_model("accounts/fireworks/models/llama-v3p1-8b-instruct", provider="fireworks")

   # HuggingFace Inference Router
   model = load_model("Qwen/Qwen2.5-72B-Instruct", provider="hf")
   ```

2. **Unified ProviderRegistry** — `list_providers()`, `list_models(provider)`, `lookup(model_id)` in one place. All 9 adapters self-register on import. Duplicate model IDs across providers raise `AmbiguousModelError` with disambiguation instructions.

3. **`effgen doctor`** — new CLI command that prints a table of all 9 providers and whether their API key is available, with setup instructions for missing keys.

4. **Backend parity matrix** — 7/8 providers passed the canonical agentic task (Anthropic skipped — no key in dev env; Replicate xfail — billing credits). All 9 raise `ModelAuthError` uniformly on bad credentials. Full report in `docs/providers/parity.md`.

5. **HuggingFace Router support** — `HFInferenceAdapter` routes via `provider="auto"` (the new HF Inference Router), supports 124 bundled models with live `refresh_models()` + `check_drift()`, and raises helpful `ModelUnavailableError` with `suggest_alternatives()` when a model is temporarily offline.

### Installing New Backends

```bash
pip install "effgen[groq]"       # Groq: GROQ_API_KEY
pip install "effgen[together]"   # Together AI: TOGETHER_API_KEY
pip install "effgen[fireworks]"  # Fireworks: FIREWORKS_API_KEY
pip install "effgen[replicate]"  # Replicate: REPLICATE_API_TOKEN
pip install "effgen[hf]"         # HuggingFace: HF_TOKEN
```

Or grab everything at once:

```bash
pip install "effgen[all]"
```

### Upgrading from v0.2.2

No breaking API changes. All new providers are opt-in extras. Existing `load_model`, `Agent`, and tool calls work without modification.

```bash
pip install --upgrade effgen
```

---

## v0.2.2 — April 28, 2026

**effGen v0.2.2** brings Gemini's latest thinking and grounding capabilities to effGen, adds the Gemini Files API and three Gemini-native tools, and modernizes Anthropic support for the full Claude 4.x lineup.

### Top Highlights

1. **Gemini 3.x / 2.5 / 2.0 + Gemma 3/4 model registry** — `gemini-3.1-flash-lite`, `gemini-3.0-pro`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, and Gemma families all recognized with correct context windows, output limits, and feature flags. SDK migrated to `google-genai>=1.0.0`.

2. **Gemini `thinking_budget`** — pass `thinking_budget=8192` (or any token count) in `GenerationConfig` to activate Gemini's internal reasoning. Set `include_thoughts=True` to surface the thinking trace in `ModelResponse.metadata["thinking"]`.

   ```python
   from effgen import load_model
   from effgen.models.base import GenerationConfig
   model = load_model("gemini-3.1-flash-lite", provider="gemini")
   result = model.generate("Explain why π is irrational.", config=GenerationConfig(thinking_budget=8192, include_thoughts=True))
   ```

3. **Gemini Google Search grounding** — set `grounding=True` in `GenerationConfig` and the adapter injects Google Search; grounding attributions (URLs + snippets) arrive in `ModelResponse.metadata["grounding_chunks"]`.

4. **Gemini Files API** — `effgen.models.gemini_files.upload_file(path)` returns a `FileRef`; pass it in `generate(prompt, files=[...])` to give the model access to PDFs, images, and other documents (2 GiB limit enforced before upload).

5. **Gemini native tools** — `GoogleSearchTool`, `GeminiUrlContextTool`, `GeminiCodeExecutionTool` in `effgen.tools.builtin.gemini_native`. Use them directly in Agent — they activate Gemini's server-side capabilities with no extra API calls. Pairing with a non-Gemini model raises `ToolIncompatibleError` at init.

6. **Anthropic Claude 4.x registry** — claude-opus-4-7 (1M ctx), claude-sonnet-4-6, claude-haiku-4-5, and the full legacy 3.x / 4.x lineup in `effgen/models/anthropic_models.py`.

7. **Anthropic extended thinking** — `GenerationConfig.thinking = {"type": "enabled", "budget_tokens": N}` activates Claude's extended thinking; `redacted_thinking` blocks are preserved across multi-turn conversations.

8. **Anthropic prompt caching** — `mark_cached(block)` + `AgentConfig.cache_system_prompt=True` / `cache_tools=True` wire `cache_control` automatically; cache hit/creation tokens surfaced in `ModelResponse.usage`.

9. **Anthropic streaming polish** — `generate_stream_full()` handles thinking deltas, redacted-thinking, and parallel `tool_use` blocks in a unified `StreamChunk` API.

10. **Experimental Anthropic native tools** — `AnthropicBashTool`, `AnthropicTextEditorTool`, `AnthropicComputerTool` stubs in `effgen/tools/builtin/anthropic_native.py` (flag-gated, not registered by default).


### Upgrading from v0.2.1

No breaking API changes. All new fields (`thinking_budget`, `include_thoughts`, `grounding`, `thinking`, `cache_system_prompt`, `cache_tools`) default to safe backward-compatible values.

```bash
pip install --upgrade effgen
```

---

## v0.2.1 — April 25, 2026

**effGen v0.2.1** brings **Cerebras** to effGen as a first-class inference backend and modernizes the **OpenAI** adapter for the latest reasoning models.

### Top Highlights

1. **Cerebras backend** — All 4 free-tier Cerebras models (`gpt-oss-120b`, `llama3.1-8b`, `qwen-3-235b-a22b-instruct-2507`, `zai-glm-4.7`) with streaming, native function-calling, automatic rate-limit coordination (RPM/RPH/RPD + TPM/TPH/TPD sliding windows), and per-call cost tracking. `pip install effgen[cerebras]` and set `CEREBRAS_API_KEY`.

   ```python
   from effgen import load_model
   model = load_model("llama3.1-8b", provider="cerebras")
   ```

2. **OpenAI: gpt-5, gpt-5.4-nano, and o-series reasoning models** — full registry coverage with `reasoning_effort` (`minimal`/`low`/`medium`/`high`) and `max_reasoning_tokens` on `GenerationConfig`. Reasoning-only payloads are routed only to reasoning-capable models; chat models silently drop the field.

3. **OpenAI prompt caching** — `cached_input_tokens` is now surfaced in `ModelResponse.usage` and metadata. `AgentConfig.stable_system_prompt=True` keeps your system prompt anchored at position 0 so OpenAI's automatic ≥1024-token prefix cache stays warm.

4. **Structured outputs v2** — `OpenAIAdapter.generate_structured()` with strict JSON Schema; `to_openai_schema(pydantic_model)` inlines `$ref`s and forces `additionalProperties: false`. Refusals raise `ModelRefusalError` with the model's refusal text preserved.

5. **OpenAI native tools** — `OpenAIWebSearchTool`, `OpenAICodeInterpreterTool`, and `OpenAIFileSearchTool` route through OpenAI's Responses API and compose with effGen's local tools in the same agent. Pairing one with a non-OpenAI model raises `ToolIncompatibleError` at Agent init (no surprise mid-run failures).

### Other Improvements
- `load_model(..., provider="openai"/"anthropic"/"gemini"/"cerebras")` now routes correctly (was previously HF-only)
- HF-only kwargs are stripped before reaching API adapters
- `transformers` engine `unload()` removes accelerate hooks + syncs CUDA, eliminating cross-test GPU state leaks
- Stability sweep: ruff clean, mypy lenient-clean, multi-Python-version verified (3.10/3.11/3.12/3.13)

### Upgrading from v0.2.0

No breaking API changes. New parameters (`reasoning_effort`, `max_reasoning_tokens`, `stable_system_prompt`) all default to safe values. To use Cerebras:

```bash
pip install --upgrade "effgen[cerebras]"
export CEREBRAS_API_KEY=...
```

## v0.2.0 — April 9, 2026

**effGen v0.2.0** is a major release that transforms the framework into a production-grade agentic AI platform. 15 development phases deliver powerful new capabilities — all optimized for Small Language Models.

### Top 5 Features

1. **Native Tool Calling & Structured Output** — Models like Qwen, Llama, and Mistral can now use their built-in function calling instead of text-based ReAct parsing. Set `tool_calling_mode="native"` or `"hybrid"` in AgentConfig. JSON schema and Pydantic model output validation included.

2. **Guardrails & Safety** — Protect your agents with `PIIGuardrail`, `PromptInjectionGuardrail`, `ToxicityGuardrail`, `ToolPermissionGuardrail`, and more. Use presets: `get_guardrail_preset("strict")` for instant configuration.

3. **Advanced RAG Pipeline** — Full document ingestion (PDF, DOCX, HTML, Markdown, CSV, JSON), semantic/code/table/hierarchical chunking, hybrid search (dense + BM25 + keyword), reranking, source attribution with inline citations. One-liner: `create_agent("rag", model, knowledge_base="./docs/")`.

4. **Production API Server** — OpenAI-compatible `/v1/chat/completions` endpoint, request queuing with priority, agent pooling, multi-tenancy with API key management, CORS, GZip, graceful shutdown. Drop-in replacement for OpenAI API with local SLMs.

5. **Apple Silicon Native (MLX)** — Community-contributed MLX and MLX-VLM backends for Apple Silicon. Native Metal GPU acceleration with unified memory. `pip install effgen[mlx]` — no CUDA required.

### What's New

- **31 built-in tools** (up from 14) — finance (stock/currency/crypto), data science (DataFrame/Plot/Stats), DevOps (Git/Docker/SystemInfo/HTTP), knowledge (Arxiv/StackOverflow/GitHub/Wolfram), communication (EmailDraft/SlackDraft/Notification)
- **Multi-agent orchestration** — MessageBus pub/sub, DAG-based workflows (YAML), shared state, agent lifecycle management with pools and registries
- **Model router** — automatic model selection based on query complexity; multi-model agents with speculative execution; model pool with LRU eviction
- **Checkpointing & sessions** — save/restore agent state mid-task; persistent conversation sessions across processes; background task runner with pause/resume/cancel
- **Evaluation framework** — 5 built-in test suites (270 test cases), regression tracking, model comparison matrix; `effgen eval` and `effgen compare` CLI
- **Observability** — full OpenTelemetry tracing, structured JSON logging with correlation IDs, Prometheus metrics with percentiles, Grafana dashboard template, interactive debug mode
- **Human-in-the-loop** — approval workflows for dangerous tools, clarification requests, feedback collection
- **Performance** — prompt caching (LRU + TTL), result caching with semantic similarity, token budget management, lazy model loading, GGUF/AWQ/GPTQ quantization, continuous batching, speculative decoding hints
- **Python & TypeScript SDKs** — `EffGenClient` with sync/async, streaming, retries; TypeScript client for Node/Deno/Bun/browser
- **Local embedding API** — `/v1/embeddings` endpoint with sentence-transformers + TF-IDF fallback, LRU + SQLite caching
- **Domain keyword expansion** — 5 built-in domains (Tech/Science/Finance/Health/Legal) with WordNet/template/LLM-based expansion

### Upgrading from v0.1.x

No breaking API changes. All existing `Agent`, `AgentConfig`, `load_model`, and tool APIs work without modification. New features are opt-in. See the [migration guide](docs/migration.md) for details.

```bash
pip install --upgrade effgen==0.2.0
```

### New Optional Dependencies

```bash
pip install effgen[rag]       # RAG pipeline (sentence-transformers, faiss-cpu)
pip install effgen[finance]   # Finance tools (yfinance)
pip install effgen[data]      # Data science tools (matplotlib, plotly)
pip install effgen[eval]      # Evaluation extras (rouge-score, nltk)
pip install effgen[gguf]      # GGUF model support (llama-cpp-python)
pip install effgen[mlx]       # Apple Silicon MLX support
pip install effgen[mlx-vlm]   # Apple Silicon vision-language models
```

---

## v0.1.3 — March 25, 2026

v0.1.3 addresses 19 issues discovered during v0.1.2 verification, hardening the framework for real-world SLM agent usage.

### Highlights

- **Smarter loop detection** — allows 1 retry before flagging exact loops, raises threshold for data-processing tools, and normalizes inputs before comparison. Fewer false positives in multi-step pipelines.
- **"Skip the tool" prompting** — ReAct prompt now explicitly tells SLMs they can answer directly without tools. Reduces unnecessary tool calls for greetings, jokes, and recall tasks.
- **Model-aware token counting** — ShortTermMemory uses the loaded model's tokenizer instead of the `len//4` heuristic, improving summarization trigger accuracy.
- **Sub-agent depth limit** — configurable `max_sub_agent_depth` (default 3) prevents infinite sub-agent recursion.
- **Circuit breaker persistence** — optional JSON file persistence so breaker state survives agent restarts.

### What's Improved

- Partial answer extraction now finds day names and numeric results in tool observations
- Model-family prompt formatters differentiated (Qwen `<|tools|>` tags, Llama header/EOT tags)
- Removed `\n\n\n` stop sequence that truncated multi-paragraph output
- Streaming examples hardened with SIGALRM timeouts
- Integration test fixtures gracefully fall back to fp16 when bitsandbytes is missing
- NotImplementedError stubs in MCP and Retrieval now include descriptive messages

### What's Fixed

- Loop detection false positives on JSON data pipelines
- SLMs over-using tools for tasks that don't need them
- DateTimeTool date queries more reliable (better answer extraction)
- Silent model loading failures now logged with clear warning

---

## v0.1.2 — March 12, 2026

v0.1.2 is a test-driven hardening release. Every feature was built by creating a real agent, testing it across multiple models (0.5B to 8B), watching what breaks, and fixing the framework.

### Highlights

- **10 comprehensive example agents** — Q&A, calculator, multi-tool, file operations, code execution, conversational memory, error recovery, data processing, streaming, and multi-agent pipeline orchestration
- **19 framework bugs fixed** — discovered through real inference testing, not unit tests. Fixes cover tool parsing, answer extraction, memory management, and model-specific edge cases
- **Cross-model compatibility matrix** — 11 models tested across all 10 agents. 73% pass rate (80 PASS, 23 PARTIAL, 7 FAIL out of 110 combinations)
- **Top models (10/10 PASS):** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Phi-4-mini-instruct

### What's New

- 10 example agents in `examples/` with full documentation, model recommendations, and interactive modes
- Compatibility matrix at `examples/compatibility_matrix.md` with per-agent model recommendations
- User-explicit sub-agent trigger detection (e.g., "use 3 agents to parallelize this")
- Sweep runner (`examples/sweep_model.py`) for automated cross-model testing

### What's Improved

- ReAct loop is more robust — better loop detection, answer extraction, and error recovery
- Tool input parsing handles single-quoted JSON, non-JSON inputs, and markdown fences
- Conversation history is better managed — configurable turn limits, auto-summarization, response truncation
- Tool results are properly formatted for the model (no more raw dicts)

### What's Fixed

- 4-bit quantization now works correctly with TransformersEngine
- gemma-3 context length detection fixed for nested config
- DateTimeTool `now` operation respects date parameter
- PythonREPL no longer double-prints output
- Absolute file paths no longer get their leading slash stripped
- Many more — see [CHANGELOG.md](CHANGELOG.md) for the full list

### Model Recommendations

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Q&A (no tools) | 0.5B | 1.5B+ |
| Tool calling | 1.5B | 3B |
| Multi-turn conversation | 1.5B | 3B |
| Multi-agent pipeline | 1.5B | 3B |

---

## v0.1.1 — March 6, 2026

v0.1.1 is a stabilization release that fixes metadata inconsistencies, improves error handling, adds 6 new examples, and expands the test suite.

### What's Fixed
- License references now consistently say Apache-2.0 everywhere (was MIT in some files)
- `setup.py` entry points, Development Status, and dependency versions now match `pyproject.toml`
- 5 bare `except:` handlers in GPU monitoring replaced with specific exception types
- 15+ stray `print()` calls converted to structured logging

### What's New
- 6 example scripts: presets, streaming, memory, multi-tool, weather, and plugin usage
- 50+ new tests covering CLI, API server, plugins, presets, fallback chains, and circuit breakers
- Top-level convenience imports for `ToolFallbackChain`, `CircuitBreaker`, `ToolPromptGenerator`, `AgentSystemPromptBuilder`
- `NEWS.md` for user-friendly release summaries

### What's Changed
- Error handlers across execution modules now log exceptions instead of silently swallowing them
- Comprehensive lint cleanup via ruff (2200+ auto-fixes)

---

# effGen v0.1.0 Release Notes

**Release Date:** March 1, 2026

effGen v0.1.0 is the first feature-complete release, upgrading the framework from Alpha to Beta status. This release transforms effGen into a full-featured agentic AI framework optimized for Small Language Models (1B-7B parameters).

## Highlights

- **14 Built-in Tools** — 7 new tools added: BashTool, WeatherTool, JSONTool, DateTimeTool, TextProcessingTool, URLFetchTool, and WikipediaTool
- **Protocol Support** — Complete MCP, A2A, and ACP protocol implementations for tool and agent interoperability
- **Real Token Streaming** — True streaming via `generate_stream()` with callbacks for thoughts, tool calls, observations, and answers
- **Memory System** — ShortTermMemory, LongTermMemory, and VectorMemoryStore integrated into the Agent lifecycle
- **Agent Presets** — One-line agent creation with `create_agent("math", model)` for math, research, coding, general, and minimal configurations
- **Plugin System** — Extend effGen with custom tools via entry points or directory-based discovery
- **CLI Enhancements** — Rich progress display, `--preset`, `--explain`, `--verbose` flags, tab completion for bash/zsh/fish, and persistent chat history
- **API Server** — WebSocket streaming, API key authentication, rate limiting, and OpenAPI documentation
- **CI/CD & Testing** — 6 GitHub Actions workflows, 67 unit tests, health monitoring, OpenTelemetry tracing, and Prometheus metrics

## What's Changed

- Structured tool descriptions with parameter types and usage examples
- `stream()` now uses real token streaming (previously character-by-character)
- `run_async()` is natively async (previously wrapped sync in executor)
- Memory uses proper ShortTermMemory/LongTermMemory classes
- Development status upgraded from Alpha to Beta

## What's Fixed

- All `NotImplementedError` paths in retrieval tool
- ACP JSON Schema validation (was checking required fields only)
- Streaming placeholder removed (`time.sleep(0.01)`)
- Direct inference now retains multi-turn conversation context

## Upgrading from v0.0.2

No breaking API changes. Existing `Agent(config=AgentConfig(...))` and `load_model()` calls work without modification. New features are opt-in.

```bash
pip install --upgrade effgen
```
