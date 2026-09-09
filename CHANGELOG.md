# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] - 2026-09-08

### Highlights

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

### Changed: what existing code sees

#### 1. A run that stops without an answer now reports failure, and raises by default

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

#### 2. Citation markers are opt-in

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

#### 3. A streamed run sends the model's working before its answer

Same task, same final answer, but 8 chunks became 134, starting with the model's reasoning. If you
join the chunks and show the result, you now get the working first.

#### 4. A small local model writes more and reaches for tools more often

`Qwen2.5-1.5B-Instruct` on the local Transformers engine, four runs per release, same token counts
every time. "What is 144 divided by 12?" goes from 31 to 233 completion tokens and from about 3.5
seconds to about 27. "Name the largest planet in the solar system." goes from 11 to 145 tokens and
from 0 to 1 tool calls. Both answers stay correct.

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

### Measured

Ten sample sets at full size, 1,685 runs per arm, both arms against one served
`Qwen2.5-7B-Instruct` at the same settings. The band is the paired two sigma band worked out from
each set's own disagreeing runs. A change inside its band is not a result.

| set | 1.0.0 | 1.0.1 | delta | band (2 sigma) | |
|---|---|---|---|---|---|
| gsm8k | 72.00 | **85.00** | +13.00 | 6.63 | better |
| gsmplus | 54.50 | **65.50** | +11.00 | 7.87 | better |
| bb_hard | 37.14 | **80.00** | +42.86 | 19.79 | better |
| math500 | 54.50 | 52.00 | -2.50 | 7.00 | inside the band |
| bb_easy | 97.08 | 97.50 | +0.42 | 1.86 | inside the band |
| bb_med | 77.60 | 76.00 | -1.60 | 7.84 | inside the band |
| csqa | 92.00 | 90.00 | -2.00 | 4.47 | inside the band |
| simpleqa | 42.00 | 44.00 | +2.00 | 13.27 | inside the band |
| arc_c | 95.00 | 90.00 | -5.00 | 4.47 | **worse** |
| arc_e | 94.50 | 90.00 | -4.50 | 3.32 | **worse** |
| **all ten, unweighted** | **71.63** | **77.00** | | | |

By category: coding 70.61 to **84.50**, calculator 60.33 to **67.50**, agentic 42.00 to **44.00**,
retrieval 93.83 to **90.00**. Calculator is still below what the same model scores with no framework
at all, 67.50 against 79.00 on the same questions. This release does not change that, and it is
worth knowing before you put a calculator tool in front of a small model doing arithmetic.

What a run costs:

| | 1.0.0 | 1.0.1 |
|---|---|---|
| model calls | 2.22 | **3.05** (+37%) |
| prompt tokens | 1,120 | **1,762** (+57%) |
| completion tokens | 309 | **353** (+14%) |
| wall time | 13.30 s | **14.03 s** (+5%) |

These fixes buy their correctness with context. On these sets the framework did not get cheaper, it
got more expensive.

### Known issues

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

---

## [1.0.0] - 2026-08-14

### Highlights

**effGen v1.0.0 is the first stable release.** It is more than 600 commits of work since v0.3.2, and the theme
running through it is control over where a model runs and visibility into what a run did.

You can now point effGen at any server that speaks the OpenAI protocol, read back which tool calls a
run made, wrap the agent loop in middleware, hand a single agent many conversations, choose how
history is compacted, and resume a workflow that died half way through. A backend that never answered
raises instead of returning something that reads like an answer.

Around that sits a terminal coding agent (`effgen code`), a branded command line that works on any
terminal, a real-time dashboard, an in-browser playground, shareable HTML reports and run cards, a
cross-provider model and pricing browser, a terminal mission-control view, a live model battle, and a
browsable run and session history.

Underneath both is the least visible and largest part of the release: a long pass over everything
that used to report the wrong thing confidently. A run that failed now says so. An unpriced model
reports no cost instead of a made-up one. A turn whose every action failed is not a success. A tool
call written in a shape effGen could not read no longer ends a turn with nothing.

**Three changes are breaking.** Each is listed below with its one-line migration. The public surface
grew from 204 names to 223, and nothing was removed or renamed.

### Changed - breaking

1. **Python 3.10 is no longer supported.** The floor is 3.11, and the supported set is 3.11, 3.12,
   3.13 and 3.14. `tomllib`, `asyncio.timeout`, `datetime.UTC` and the `TimeoutError` unification are
   all stdlib from 3.11, and effGen carried a hand-written fallback for each.

   *Migration:* upgrade the interpreter. Nothing in the API changed.

2. **`AgentConfig.raise_on_error` now defaults to `True`.** A failed run raises its typed error
   instead of returning an `AgentResponse` with `success=False` and a plausible-looking string in
   `.output`, which a caller reading `.output` without checking `.success` never noticed.

   *Migration:* pass `raise_on_error=False` to inspect the response yourself. The failure shape is
   unchanged, and the CLI does exactly this at all fourteen of its construction sites.

   ```python
   Agent(AgentConfig(model="gpt-5-nano", raise_on_error=False))
   ```

   `raise_on_error=False` is also the documented setting for batch evaluation. Scoring a run that hit
   the iteration cap as an error rather than as a wrong answer measures the reporting style instead
   of the model, and a small model hits that cap often. With the flag off, a failed run's `output` is
   effGen's report of what stopped the run, and the model's own text is in
   `metadata["partial_output"]`.

3. **A backend that never answered raises whatever that flag says.** A refused connection, an
   unresolvable host or a missing route is classified `unreachable`, separately from a server that
   answered badly (which stays `transient` and still retries), and raises `BackendUnreachableError`.
   A task that ran and failed is a result you can inspect. A backend that was never reached is not,
   and returning one is how a whole batch completes against nothing and still looks healthy in the
   summary.

   *Migration:* there is no opt-out, by design. Catch the error where you want to handle it.

   ```python
   from effgen.models.errors import BackendUnreachableError

   try:
       result = agent.run("Summarise the Q3 report.")
   except BackendUnreachableError:
       pass   # the server is not up
   ```

   Classification reads the exception chain, because provider SDKs shorten a refused port to
   "Connection error." and keep the real cause on `__cause__`.

One smaller change is worth knowing before you upgrade: four public enums are now `enum.StrEnum`, so
`str(TaskStatus.RUNNING)` reads `"RUNNING"` rather than `"TaskStatus.RUNNING"`. Equality, membership
and JSON serialization are unchanged.

### Added - connecting to models

- **Point effGen at any OpenAI-compatible server.** `base_url` reaches `load_model()` and
  `AgentConfig`, so effGen can drive a model you already serve (vLLM, SGLang, TGI, llama.cpp, Ollama,
  LM Studio, LiteLLM, a gateway or a corporate proxy) instead of loading a second copy of the weights
  inside the agent process.

  ```python
  from effgen.models import load_model

  model = load_model(
      "Qwen/Qwen2.5-7B-Instruct",
      provider="openai_compatible",
      base_url="http://127.0.0.1:8000/v1",
  )
  ```

  The endpoint also comes from `EFFGEN_BASE_URL`, `OPENAI_BASE_URL` or `OPENAI_API_BASE`, in that
  order, and `provider="openai"` with a `base_url` routes here too. The server serves its own model
  ids, so no OpenAI catalog is consulted: the full sampling surface is offered, calls report **no
  price** rather than a fabricated `$0`, and `list_served_models()` asks the endpoint what it has.
  Pass `context_length=` when your server's window is not the assumed 32,768 tokens; effGen now warns
  when it is assuming, naming the value and the flag that sets the real one, instead of failing later
  at a size nobody chose. See [docs/models/openai-compatible.md](docs/models/openai-compatible.md).

- **A multi-turn tool loop you can write by hand, on any provider.** `build_assistant_message()` and
  `build_tool_result_message()` on `BaseModel`, and so on every adapter, build each provider's own
  message shape. A loop written once runs against OpenAI, Gemini, Anthropic, Groq, Together,
  Fireworks, Cerebras, Replicate and HF Inference instead of only the first. Gemini and Anthropic
  override them with their own shapes. See [docs/models/tool-calls.md](docs/models/tool-calls.md).

- **Python 3.14 is supported.** It was installed and run, not just resolved: the unit lane passes on
  3.14 with `.[dev]` (4,078 tests) and with `[all]` through a shipped lock (4,151). One caveat, since
  it changes the install line: plain `pip install effgen[all]` does not resolve on 3.14, because pip
  backtracks through the wide vLLM range into a release pinned to `numba==0.61`. On 3.14, install the
  extras with `pip install -r requirements-all-py314-lock.txt` followed by
  `pip install --no-deps effgen`. See [docs/installation.md](docs/installation.md).

### Added - the agent surface

- **Middleware around the agent loop.** Hooks at three points (the run, each model call, each tool
  call), each with a *before* and an *after*. A *before* hook can rewrite the request through its
  context or short-circuit it entirely; an *after* hook can transform the result. *Before* hooks run
  in order and *after* hooks in reverse, so middleware nest.

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

  Also available per call as `run(..., middleware=[...])`, which appends to the configured ones for
  that call only. `LoggingMiddleware` and `ToolApprovalMiddleware` ship. See
  [docs/guides/middleware.md](docs/guides/middleware.md).

- **One agent, many conversations.** `run(..., session=Session | str)` builds the prompt from that
  conversation's history and appends the turn to it, restoring the agent's own session and memory
  afterwards, including when the run fails. A server handling many users no longer needs an agent
  object per user, nor history bookkeeping outside the framework.

  ```python
  agent.run("My dog is named Pixel.", session="user-123")
  agent.run("My cat is named Mote.",  session="user-456")
  ```

- **Pluggable context compaction.** What gets dropped when a conversation outgrows the window is now
  a strategy: `SummarizeOldest` (the default, with behaviour unchanged), `DropOldest` (no model call,
  nothing invented), `KeepFirstAndLast` (the turns carrying the task survive verbatim) and
  `KeepToolResults` (the evidence stays, the reasoning is compacted). Choose one with
  `AgentConfig(compaction_strategy=DropOldest())`, or subclass `CompactionStrategy` for your own.
  `AgentConfig(tokenizer=...)` measures the history in the units the window is measured in rather
  than characters divided by four; anything with `count_tokens(text)` or `encode(text)` works. See
  [docs/guides/context-compaction.md](docs/guides/context-compaction.md).

- **A workflow that died part way through can be resumed.** `WorkflowDAG.run()` takes a `checkpoint=`
  store and a `run_id=`. Run the same line again after a crash and it continues where it stopped.

  ```python
  from effgen import FileCheckpointStore, WorkflowDAG, WorkflowNode

  store = FileCheckpointStore()          # ~/.effgen/workflows by default
  dag = WorkflowDAG("report")
  dag.add_node(WorkflowNode(id="research", agent=researcher))
  dag.add_node(WorkflowNode(id="draft", agent=writer))
  dag.connect("research", "draft")

  result = dag.run("Write the Q3 summary.", checkpoint=store, run_id="q3-summary")
  ```

  Completed nodes are not re-run and their outputs flow downstream, failed nodes are retried (which
  is usually the reason to resume), and a finished run replays its stored outputs without calling a
  model, so a retrying job runner cannot double-bill you. There is no separate resume call: an
  unknown run id starts from the beginning and a known one continues, so there is no second code path
  to get wrong. Progress is written after each topological level, writes are atomic, only state is
  stored (never the graph, which holds sockets and credentials), and resuming into a changed graph is
  refused by name rather than mixing outputs from two different workflows.
  `InMemoryCheckpointStore` is there for tests. See
  [docs/guides/sessions-and-checkpoints.md](docs/guides/sessions-and-checkpoints.md).

- **`AgentResponse.tool_calls` reports the calls, not just how many.** Each entry is a `ToolCall`
  carrying `name`, `arguments`, `result`, `duration`, `error` and the `iteration` it was made on,
  with `.failed` and `.by_name()` to narrow them.

  ```python
  for call in result.tool_calls:
      print(call.name, call.arguments, "->", call.error or call.result)
  ```

  Iterating the field used to raise `TypeError: 'int' object is not iterable`. It still compares and
  casts as the count, so `tool_calls == 2` and `tool_calls > 0` are unchanged; `tool_call_count` on
  the response, and `tool_calls.total` on the list, say the number plainly. `to_dict()` keeps the
  count under its original key and adds `tool_call_details`, so saved runs read back either way.
  Records are captured on the ReAct, native and streaming paths.

- **`load_env()`** runs the same `.env` search the CLI does, so a library script picks up the keys
  the CLI already finds. It honours `EFFGEN_NO_DOTENV` and never overwrites a value you exported.

- **A tool can declare its output to be retrieved context** (`is_context_retrieval`), so a loop that
  ends on a repeated retrieval knows it has evidence rather than an answer.

### Added - the coding agent

- **`effgen code` is a coding agent in the terminal.** It reads your workspace, proposes edits as
  unified diffs, and writes nothing until you say so. `--undo` rolls the last change back from a
  journal bounded to 100 entries.

  ```bash
  effgen code "add a --dry-run flag to the importer"
  effgen code --review                      # one read-only pass
  effgen code --session-id my-refactor      # continue where you left off
  ```

  It runs in one of four permission modes (plan, ask, auto-edit, yes) that gate every write, every
  shell command and every commit. Writes are confined to the workspace, and a hunk that no longer
  applies is reported rather than clobbering the file. An interactive session keeps one run record
  across turns and carries a slash-command set (`/plan`, `/diff`, `/apply`, `/reject`, `/undo`,
  `/run`, `/test`, `/context`, `/add`, `/drop`, `/mode`, `/model`, `/tools`, `/cost`, `/trace`,
  `/git`, `/review`, `/compact`, `/save`, `/session`, `/load`, `/doctor` and more). `--session-id`
  resumes it in a later process. `--review` (and `/review`) makes one read-only pass over a change
  with a tool set that holds nothing that writes, runs or executes.

  It is repository-aware: branch, status and a layout inventory that honours `.gitignore` go into the
  prompt, and an `AGENTS.md` brief is read when present. Git actions run through an allow-list, so
  push, reset, checkout, clean, rebase and force are refused before a subprocess starts, including
  when the model tries to reach them through the shell. A commit is confirmed like a write and uses
  the repository's own identity, leaving your other staged work alone.

  A turn streams its answer as it is written where the model's calls can be dispatched mid-stream,
  names which tool-calling path it ran, and reports each action as it happens. `-p`, `--json` and
  piped stdin run the single-shot path with byte-clean stdout. See [docs/cli/code.md](docs/cli/code.md).

- **`effgen doctor` reports coding readiness** (workspace, sandbox backend, git), and
  `quickstart`/`tutorial` include a coding step that writes and runs a real program.

### Added - surfaces you can show someone

- **A real-time dashboard** with per-model and per-provider cost, latency percentiles that are real
  percentiles, an error breakdown, a run waterfall, a model catalog panel and a history panel. Every
  chart is drawn locally.
- **An in-browser playground** on the existing chat endpoint, with model and preset pickers, tool
  toggles, the run's tool trace, and copy-as-curl, copy-as-CLI and copy-as-Python for the form you
  filled in.
- **A cross-provider model and pricing browser**, in the terminal (`effgen models browse`, with
  search, provider, capability, context and price filters, sorting and paging) and in the dashboard.
  `models info` shows every provider that serves a shared id.
- **Shareable HTML reports** for compare, eval, cost and loadtest (`--report out.html`), plus
  **single-run cards** (`run --card`, `runs show <id> --card`) and `effgen report <result.json>` to
  render a saved document after the fact.
- **`effgen top`** (alias `effgen monitor`), a terminal mission-control view over the telemetry you
  already collect: activity, traffic, per-model, spend and GPU panels, each stating the window and
  process it describes.
- **`effgen battle`**, which races several models on one prompt side by side and reports the tally,
  the cost and an optional judge's verdict separately from the measurements.
- **A live multi-agent topology graph**, terminal trace timelines, a workflow DAG diagram
  (`workflow run --diagram`) and a run waterfall.
- **A command palette and keyboard-first navigation** on both web surfaces, with a skip link, jump
  links, focus restoration and screen-reader announcements.
- **Named CLI themes** (`--theme`, `EFFGEN_THEME`: default, high-contrast, monochrome, light) drawn
  from one shared palette that the dashboard reads too, and a branded landing page and first-run
  welcome.

Every web surface is self-contained. There is no CDN, no external font and nothing fetched at view
time, and that is enforced by a test that inspects what a browser would fetch rather than by
searching for a substring.

### Added - history, projects and the command line

- **Durable run and session history.** Every run is recorded with its model, provider, tokens, cost,
  status and task, keyed by the same run id its trace spans carry: `effgen runs list/show/cleanup`
  and `effgen sessions list/show/browse/export/cleanup`, with search, status, model and date filters.
  Runs from the CLI, a script and the server share one history and survive a restart.
- **Project scaffolding.** `effgen quickstart --init [DIR]` writes `effgen.yaml`, `.env.example` (one
  named variable per registered provider, with no value invented), a runnable `example.py` and a
  `.gitignore`, puts a $1.00/day spend cap in force when none is configured, and prints the next three
  commands. `effgen config init` writes a document a run actually reads, and `effgen run -c` applies
  the `model` and `provider` that document names.
- **Flags and output that behave the same everywhere.** `--json` on every command that had no machine
  output, and `--json` stdout is now a single valid document on a pipe and on a terminal, with no
  spinner, table or warning mixed into it. `-o` picks its format from the extension (`.html` renders
  a report, `.md` writes Markdown, anything else JSON). `--guardrails` on `run`, `chat` and `batch`.
  `--provider` on `eval` and `compare`, `-m` on `prompts run`/`prompts eval`, `--temperature` on
  `eval`/`compare`, `--trace` on `run`, positional input on `batch`, `-t/--tools` on `chat`. A bare
  group command prints its own help and exits 0 instead of reporting an unknown subcommand. Thirteen
  short flags now mean the same thing across commands.
- **Your own prompt templates load beside the shipped ones.** `EFFGEN_PROMPTS_DIR` names one or more
  directories; each `*.py` in them is imported and its templates are registered under their own
  names, so a team's library sits next to the built-in one without a fork. `prompts run` now fails
  closed on an empty or truncated result rather than printing nothing and exiting 0, and reports the
  tokens, cost and latency of the call it made.
- **`effgen loadtest --url`** drives a running `effgen serve` over HTTP, through auth, rate limiting
  and the middleware stack, instead of only driving an adapter directly.
- **`effgen models status --json`**, `models info` on a local engine id, and `models browse
  --include-local`.

### Added - a documentation site

- **effGen has a project site and a documentation site**, both published from this repository:
  a landing page at <https://ctrl-gaurav.github.io/effGen/> with the examples, the community links
  and the benchmark leaderboard, and 36 documentation pages at
  <https://ctrl-gaurav.github.io/effGen/docs/> covering installation, the quick start, agents,
  models and providers, tools, RAG, memory, multi-agent work, workflows, checkpointing, guardrails,
  security, evaluation, observability, reliability, the API server, deployment, hardware, protocols,
  the human-in-the-loop path and the API reference. Both are static, both are built and published by
  the same lane on every change, and neither needs a server to read.
- **Every public definition documents its arguments and its result.** The package was walked module
  by module: each public class, method and function states what it does, what each argument means and
  what comes back, and a gate fails when a public definition is added without that. The 119 pages
  under `docs/` were re-run command by command against this release.

### Changed

- **A rate limit no longer multiplies.** Three layers each retried a throttled call and multiplied
  rather than shared a budget: one client request became twelve upstream requests and held the caller
  20.5 seconds at a stated 2-second delay. One layer now owns provider retry (the adapter's own
  backoff where it has one, the SDK's where it does not), and the agent no longer re-retries a call
  already classified as rate-limited. The same measurement now reads four requests and 6.7 seconds.
- **A plain `run()` no longer fans out into sub-agents on its own.** `AgentConfig.mode` defaults to
  `SINGLE` and `run()`/`run_async()`/`stream()` follow it, where they used to force automatic
  decomposition. A task over roughly a hundred words used to become six billed calls, and a
  decomposed run could report a number the source text never contained. `--mode auto` opts back in,
  and a genuinely multi-part task still decomposes.
- **`generate_with_tools()` takes `config` third on all ten adapters.** It was `messages` on Groq,
  Together and Fireworks, so a positional call misrouted its argument and failed as a retryable
  error. Both spellings still work, told apart by type, so there is no migration.
- **A local model run is labelled with its engine, not `provider="unknown"`.** On-device work
  reported `unknown` on the `effgen_model_*` metric series while the run store recorded `transformers`
  for the same call. A dashboard that grouped local runs under `unknown` now shows them under their
  engine. Nothing labelled with a real provider moves.
- **The `standard` guardrail preset screens tool output for injection**, not just input, so an
  instruction planted in a tool's return value no longer reaches the model under the default preset.
  `standard` also redacts personal data rather than blocking the message, so a customer quoting their
  own email address is answered instead of refused. `strict` still blocks.
- **Library warnings render as one line in the CLI**, on stderr, instead of Python's traceback block
  pointing at internals you did not write. Setting `-W` or `PYTHONWARNINGS` restores the default
  rendering, because someone who set those asked for it.
- **Catalogs refreshed against the live APIs** for Groq, Fireworks, Together, Replicate and Gemini:
  retired ids removed, undeployed models dropped, four prices corrected in both the catalog and the
  fallback table, and Fireworks and Cerebras models flagged as reasoning models so they get the
  larger first-token budget instead of truncating and costing an extra billed call. The bundled
  catalog now carries 417 models across 9 providers. Two drift checks that reported permanent false
  positives were fixed.
- **An incompatible protocol SDK now breaks only its own protocol.** `mcp` 2.0.0 removes the module
  the effGen MCP server is built on, so an uncapped install took 2.x and failed at import, and
  collecting the protocols package took thousands of unrelated tests down with it. The dependency
  reads `mcp>=1.28.1,<2`, the streamable-HTTP transport is imported under its current name with a
  fallback to the old alias, each protocol package imports its SDK only when that protocol is used,
  and the upper-bound guard now covers `mcp` so the next fast mover is caught by a check rather than
  by a user. A cross-loop `disconnect()` no longer hangs.
- **Five dependency floors were raised past open advisories** (`aiohttp>=3.14.3`,
  `cryptography>=50.0.0`, `gitpython>=3.1.58`, `h2>=4.4.1`, `pypdf>=6.15.0`), earlier floors were
  raised for `mcp`, `Pillow` and `httplib2`, every floor is now declared beside each extra that
  reaches the package, and both lockfiles were regenerated. The vulnerability audit passes.

### Fixed - results that report what actually happened

- **A turn that did nothing no longer reports success.** A coding turn whose every action failed, and
  a retrieval loop that produced no answer, are reported as partial outcomes with the recovered text
  under `metadata["partial_output"]` and a typed reason for what stopped the run.
- **A run stopped at the iteration cap reports the stop**, not the last passage it retrieved, and
  carries its progress through every surface that shows it: the terminal panel, `--json`, the run
  record, the chat session and the coding report.
- **A reasoning model that emitted no visible token says so** on every adapter, instead of being
  retried three times and reported as an empty answer. A tight token budget raises a heads-up before
  the call rather than after the bill.
- **A tool call the model wrote out instead of making is a failed turn**, not an answer, including
  the shapes that used to slip through: a stray angle bracket, a missing separator, a query string
  with HTML entities, call syntax whose arguments were dropped, and a tag named after the tool
  itself.
- **A code execution that exited non-zero reports failure.** A raise, a `sys.exit`, a syntax error, a
  non-zero bash exit and a JavaScript throw all return `success=False` with a named reason, and the
  full stdout, stderr and exit code are kept on every path.
- **An unpriced or uncatalogued model reports no cost rather than a fabricated one.** A provider's
  placeholder rate made every id the bundled catalog had not seen read as priced, so a fine-tuned
  `ft:` id was billed at a made-up rate and the invented number was reported as a published price.
  `call_cost` returns `None` for an unpriced model and `0.0` only for a genuine free tier, and every
  surface (the cost report, the ledger, the run card, the dashboard, the battle tally) says "no
  price" instead of `$0`.
- **Streamed runs report their cost and tokens** on every provider, including Replicate and HF
  Inference, which recorded neither. `model.total_tokens` is correct on every adapter (six never
  assigned it at all), a model span carries its cost every time, and a Groq response that reports an
  all-zero usage block for a call it billed is estimated and flagged rather than recorded as free.
- **Team and workflow totals include the manager's own calls**, so a hierarchical run's reported cost
  is the cost.
- **A model that could not run at all is reported as failed, not as scoring zero.** An evaluation
  where the key was missing or the provider refused every call used to print `0%` beside the models
  that did run, which reads as a bad model rather than as a model that never answered.
- **`load_config(validate=True)` actually validates**, and names the file it refused. Each validator's
  result used to be discarded, and a section was passed where the whole document was expected, so it
  validated an empty set.
- **A citation is a source the answer actually used.** `.sources` still carries every URL a search
  returned; `.citations` now carries the ones the answer references, and a PDF citation carries its
  page number.

### Fixed - tool calling across providers

- **A tool call written as XML tags is understood.** Chat templates disagree about how a call is
  spelled. Many render JSON; others render nested tags, such as
  `<function=calculator><parameter=expression>4817 * 236</parameter></function>`. effGen read only
  the JSON spellings, so on a model whose template emits tags the turn parsed to nothing: no tool was
  called, and the run ended at the iteration cap. The reader is keyed on the shape rather than on a
  model family (five call tags, four argument tags, both `<tag=NAME>` and `<tag name="NAME">`), so any
  family whose template writes that shape can use tools. Such a construct is also stripped from an
  answer whole rather than leaving its argument values behind as prose, and a streamed turn holds it
  back and delivers the cleaned answer instead of putting raw scaffolding on screen. Measured across
  15 local families: three went from no tool call and no answer to a correct answer, twelve were
  unchanged.
- **Gemma 4's channel format is read.** Gemma 4 wraps its reasoning in `<|channel>` and its calls in
  `<|tool_call>call:NAME{...}`, delimiters no reader knew. The whole reasoning trace was returned as
  the answer and no tool was ever called. The channel is now parsed and stripped on every answer
  path, loosely quoted arguments are accepted, and a verb-prefixed name is mapped onto the tool it
  names. The branch runs only when the Gemma markers are present, so no other family is affected.
- **One documented tool-call shape across every adapter.** Arguments arrive as a JSON string, five
  adapters stopped parsing them differently, and Gemini, Anthropic and Replicate report their calls
  the way the others do.
- **A tool call written in call syntax keeps its arguments.** `calculator(expression="1367 * 89")`
  used to resolve the tool and call it with `{}`, because every quote was stripped before the
  arguments were read.
- **JSON tool arguments survive their own punctuation.** A structural reader replaced a brace-matching
  regular expression that stopped at the first brace inside a string value, so
  `{"query": "Paris, France: population"}` no longer falls through to a raw-input fallback. Object and
  array arguments sent as JSON strings are coerced when they parse to the declared type.
- **A chat turn that uses tools streams its answer** where the model's calls can be dispatched
  mid-stream, and a streamed request now carries the same tool definitions, `tool_choice` and
  sampling settings as a non-streamed one.
- **`stop_sequences="END"` works.** A bare string used to be walked character by character and cut
  the text at the first matching letter.
- **A Groq gpt-oss model works on the ReAct path.** It used to fail every turn with a 400.
- **Tool definitions reach the model** when the chat template would otherwise drop them, and a model
  now reports how it receives them, so the framework can choose a strategy the model can actually
  follow. A small local model that would silently skip the tool and do the arithmetic itself is
  routed to the strategy it answers on.
- **A model that returns its reasoning in the answer field is asked not to.** Groq's qwen3 family
  used to return the chain of thought as the answer text.
- **`register_tool()` accepts a `@tool` instance**, and `@tool`/`Tool.from_function` carry
  `requires_approval`, `cost_estimate` and `timeout_seconds`, so human approval is reachable from the
  primary authoring API.

### Fixed - errors that name the fix

- **A URL with no `http://` or `https://` scheme is refused**, naming the environment variable it
  came from, rather than being sent and reported as a provider outage.
- **A connection failure names the endpoint the call was sent to** instead of pointing at the
  provider's status page, which is advice about the wrong machine when the server is yours.
- **A blank endpoint variable no longer redirects every OpenAI call.** effGen read a blank as "no
  override" and passed no `base_url`; the OpenAI SDK then read the same variable itself, treated `''`
  as an address and went there. The adapter now always passes an explicit endpoint, and the
  scaffolded `.env.example` writes those variables commented out.
- **A rate limit delivered as HTTP 413 is classified as one.** Groq reports a spent
  tokens-per-minute allowance that way. It used to be `unknown`, so there was no backoff and a
  throttle was reported as a permanent failure. A genuinely oversized body is still an invalid
  request.
- **Every message a user reads is bounded, redacted and actionable.** Provider errors, routing and
  retry failures, server auth, budget and RBAC denials, SDK errors, config validation, tool refusals
  and the coding agent's own refusals all end with what to do next. Quoted upstream text is bounded to
  240 characters (one real provider body reached 42 kB), and credentials are removed by shape rather
  than by neighbouring words.
- **The submitted credential never reaches the caller.** effGen redacted its own message, but
  `raise ... from exc` kept the SDK exception, and a 401 body quotes the key. The whole
  `__cause__`/`__context__` chain is now scrubbed, including per-SDK attributes and parsed JSON
  bodies, and a rendered traceback carries nothing.
- **Failures are classified consistently.** A connection reset, a gateway page, a device
  out-of-memory, a quantized load that does not fit, an absent API token, a missing repository, a
  non-Hugging-Face token and a spent account balance each now carry the right retry verdict, so
  effGen stops retrying what can never succeed and starts retrying what can.
- **A call can be bounded.** Gemini took no timeout at all and Replicate's deadline governed polling
  only, so a peer that never answers held the call for 90 seconds. Both take `timeout` and
  `max_retries` now, and `max_retries=0` still budgets one attempt instead of making no request at
  all.
- **`effgen tools list --category <unknown>` names the filter and the valid categories** instead of
  reporting an empty registry while 66 tools are registered.
- **A malformed input names its file.** A workflow YAML that is not a workflow, a config file that is
  not a mapping or not parseable, a drifted session, checkpoint or agent state, a damaged catalog
  snapshot, and a batch input row with no query text are each named with the file and the position,
  and either loaded usably or refused, instead of raising from somewhere unrelated. `depends_on:
  search` is one node named `search`, not six nodes named `s`, `e`, `a`, `r`, `c`, `h`.
- **A run refused before any model call shows its reason.** An empty task used to render an empty red
  panel.

### Fixed - the server and the API

- **The server answers every failure with one error envelope**, including unknown URLs, wrong
  methods, missing static assets, unhandled route errors, the metrics endpoint, RBAC denials, the
  shutdown drain, the legacy convenience routes, websockets and the edge adapters.
- **The server stays responsive during a long generation.** A non-streaming completion used to block
  the event loop, so `/health` timed out for the length of the call. It now runs off the loop:
  measured at 6 ms worst case during a 14.5 second completion.
- **Content-free requests are refused before they are billed.** An empty or whitespace prompt,
  absent content and a non-positive `max_tokens` each return a 4xx before any upstream call.
- **An absent provider key is a 503 on every provider**, an upstream 429 passes its delay on as
  `Retry-After`, and a mid-stream failure emits a terminal error event rather than truncating the
  stream.
- **`GET /v1/models` lists ids that were actually served**, marks legacy aliases and states that any
  reachable `provider:model` id is callable. `effgen-default` and `default` resolve to the server's
  default model, and the native client expands `tools=["calculator"]` into tool specs.
- **An `async def` runner works with `create_openai_router()`** instead of returning an opaque 500.
- **Rate limiting is not defeated by a header.** `X-Forwarded-For` is trusted only when you enable
  it, and `effgen serve` no longer lets uvicorn rewrite the client address behind that setting.
- **Body size limits cover `/v1/embeddings`**, which used to accept an unbounded body.

### Fixed - security, guardrails and sandboxing

- **The subprocess sandbox masks the credential stores** (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`,
  `~/.docker`, `~/.azure`, `~/.config/gcloud`, the credential files beside them, `/etc/shadow` and
  mounted secrets) **and runs in its own PID namespace**, so executed code sees one process rather
  than the host's process table. Both are reported on the result as `credential_reads_masked` and
  `process_table_isolated`. This is a deny-list over a known set of paths, not read confinement, and
  the documentation says so.
- **Executed code cannot write outside its scratch space**, and the model is told where it may write.
  The Python REPL's restricted mode was hardened against every dynamic-attribute escape route found,
  with an always-on audit hook that refuses process, shell and native execution.
- **The shell tool refuses obfuscated credential reads**: quoted string concatenation, glob wildcards
  against dot-files, and decode-to-file-then-execute chains.
- **File tools refuse credential filenames and credential content** inside an allowed directory, and
  a `.env`-shaped file renamed to `.csv` is refused by content.
- **Guardrails redact what they promised.** Every credit card rather than the first; labeled clinical
  identifiers in the shapes real documents use; modern provider key formats; credential tokens
  replaced whole in log output; injection attempts that name the constraint they are overriding
  rather than the word "instructions"; and a new `SystemPromptLeakGuardrail` that catches a system
  prompt on its way out. Re-checking already-redacted text no longer nests placeholders, and email
  scanning is linear time (a 32 kB pathological input went from 1.13 s to 0.015 s).
- **`/compact` can no longer modify the workspace**, and `/git` is framed like the other read-only
  views.

### Fixed - local models, GPUs and long runs

- **Per-call sampling keywords are honoured on the local engines** (vLLM, MLX and Transformers),
  including `seed` and `stop_sequences`, which the Transformers engine read off the config before it
  looked at the call. GGUF reproduces from a fixed seed.
- **A local reasoning model is recognised from its own chat template**, so it gets the larger budget
  instead of spending the base one on a hidden chain and returning nothing, and
  `chat_template_kwargs={"enable_thinking": False}` reaches the template.
- **A model that does not fit the GPU says so.** VRAM sizing reads free memory rather than total, the
  engine reconciles `.device` with where the parameters actually are, the run's metadata carries it,
  and `require_gpu=True` fails fast rather than falling back to CPU silently.
- **Automatic sharding across several GPUs no longer produces invalid output.** On a multi-GPU node,
  `device_map="auto"` could place a Transformers model so that sampling read invalid logits and the
  run died in a CUDA `multinomial` assert. The engine now probes the logits after loading and pins
  the model to one device before sampling, and an assert that has already poisoned the CUDA context
  is retried once and then reported with the restart the caller needs. Contributed by Aafiya Hussain.
- **The MLX engine works with the current `mlx_lm`.** Its sampler moved behind a new API and its
  native tool schemas changed shape, so generation on Apple silicon failed against any recent
  release. Contributed by Yasuo Tabei.
- **Device memory comes back when a local model unloads** (a 1.5B model used to keep 2.9 GB
  reserved), batched local prompts are prepared like single ones, an unusable model load is not
  retried, and an offline or uncached model reports a cache miss with the cached models listed rather
  than a connectivity error.
- **A long conversation stops growing its own prompt.** Session summaries were unbounded and were
  replayed into every prompt, so past a threshold each turn added another summary until every call
  was refused for exceeding the context window. Summaries are now folded within a token budget
  measured with the model's own tokenizer, and a 25,000-turn session stays flat.
- **Long runs hold up under concurrency.** Tool discovery, registry replacement and rate-limit
  accounting are serialised; costs and tokens are folded under a lock, so a shared adapter's totals
  match the calls; the in-memory cost store is one database rather than one per thread; two writers of
  one session no longer publish a blend, because every file is published through its own temporary;
  REPL sessions are bounded with LRU eviction and their workers do not outlive the tool or the thread
  that made the call.
- **Batch rows no longer contaminate each other.** Per-call state is isolated, so eight concurrent
  rows report eight distinct costs and token counts, and the job total reconciles with the sum.
- **A timeout actually fires.** `with_timeout()` re-arms rather than firing once into an SDK retry
  loop that swallowed it, so a 2-second bound stops a call at 2.25 seconds instead of 22 to 50.

### Fixed - documents, RAG and batch input

- **The `rag` preset refuses to run without a knowledge base** instead of succeeding with zero
  documents.
- **Retrieval keeps distinct topics.** The preset configures a wider `top_k` with MMR re-ranking, so a
  two-topic question returns both topics; `RetrievalTool` takes `default_top_k` and `diversity`.
- **Ingestion says what it skipped and why.** A corrupt file, an empty file, a file whose content
  duplicates an earlier one, an image, and an unsupported extension each have their own reason, and
  `DocumentIngester.last_summary` reports what was indexed. PDFs carry page numbers, and DOCX creation
  dates are ISO-8601.
- **A weak model no longer answers with the passages.** The prompt now ends with an answer-shaping
  instruction after a retrieval tool, and both loop fallbacks give the model one tool-free turn to
  answer from what it has. Measured on the worst case: verbatim passage dumps went from 8 of 9 runs to
  0 of 9, with citations on every run.
- **Batch input is read carefully.** A row keyed on `prompt`, `input`, `question` or `text` is
  recognised, a scalar or dict row does not become a prompt, CSV rows report the right line, a
  non-UTF-8 file names itself, a `.json` array reports item positions rather than line numbers, and
  `--strict` fails the job on any unusable row.
- **`run --file` reads source code and plain text**, not only documents, and refuses binaries.
- **An image, audio or video source can be an inline `data:` URI.** `inputs=["data:image/png;base64,…"]`
  used to be refused as a missing file, which sent the reader off to check a path that was never
  involved. Both the base64 and the percent-encoded text forms are decoded now, a malformed one is
  refused for the reason it is malformed, and a source the filesystem rejects outright raises the
  typed `InvalidMultimodalContent` instead of a bare `OSError`.
- **Local embeddings read the cache when the model hub is unreachable**, so an offline machine with
  the model already downloaded still builds an index, and a backend that cannot be imported reports
  that as a typed import error naming the package.
- **Structured output extraction stopped corrupting valid JSON.** Repairs used to run inside string
  literals, so a value containing `", note:"` was rewritten into something unparseable; a backtick
  inside a value was read as a code fence. Measured over 8,000 adversarial examples: 36 mis-parses
  before, 0 after, with 1,427 inputs newly recovered.

### Fixed - the built-in tools

- **A tool that cannot do its job says so rather than returning an empty success.** Translation with
  no language pair available, a knowledge-base search the API refused, a news fetch where every RSS
  source was unreachable, and a web search whose unset filters were sent to the backend as `None` all
  reported success or the wrong error. Each now fails with the reason.
- **A blocked request is not read as an empty result.** Reddit's redirect to a login page is reported
  as a block, and a YouTube network failure is no longer reported as an age restriction.
- **PubMed retries a truncated response** instead of failing on it, and a search that matched no
  record says that rather than returning nothing.
- **A refused tool call names what to pass.** Search, place and id lookups quote the argument they
  needed, and a call rejected by a tool's own validation states what that tool expects.

### Fixed - the terminal and the web surfaces

- **Every command works on a terminal that cannot encode the characters effGen prints.** Twenty-two
  commands used to exit non-zero purely because of the console encoding. Text is folded to ASCII where
  it becomes bytes, so a command added later is covered, and `--json` escapes rather than
  transliterates, so a French or Chinese answer survives a hard-ASCII console byte for byte.
- **A styled line renders in its own colours.** The value highlighter used to recolour numbers,
  brackets and identifiers inside lines effGen had already styled, so a version read as three colours
  and a session id came out bold magenta. Fragmented lines went from 25 to 3 on a real terminal, and
  the three that remain are lines whose author wrote two styles into them.
- **Output reaches a redirected stdout while the command is still running.** Without `rich`, a server
  banner sat in a block buffer until the process exited.
- **The whole command surface works without `rich` and without `torch`.** Twelve commands used to
  exit with `No module named 'rich'`, and a direct engine import reported the import system rather
  than naming PyTorch and how to install it.
- **Piped output is clean.** No spinner, no `Thinking...` placeholder, no chrome on stdout, one
  answer per input line under `-q`, and zero colour codes under `NO_COLOR`.
- **A closed pipe ends the command quietly.** `effgen ... | head` used to end in a `BrokenPipeError`
  traceback; the command now exits 141, the convention the shell expects.
- **The dashboard reports real numbers.** The estimated daily cost that read roughly 300 times the
  real spend is gone, "p99" is a percentile rather than the mean, the error count includes HTTP 4xx,
  per-model cost is not double-counted when two providers serve one model name, and generation counts
  are labelled separately from HTTP responses.
- **The web surfaces are usable by keyboard and by screen reader**: focus is not dropped by the poll,
  live regions announce only what changed, contrast clears WCAG AA on every bar and control, and the
  theme follows the operating system until you choose one.
- **A generated report is inert.** Model output that contains markup renders as text, no injected
  tag or event handler survives, and only `http` and `https` links keep an `href`.
- **`effgen top --once` prints all five documented panels** when no server is reachable, and a
  malformed URL degrades into the panel instead of ending the command.

### Fixed - installation, packaging and documentation

- **`./install.sh` no longer fails when run without a terminal.** Re-running it, or running it from
  CI or another script, met an interactive prompt with no one to answer, and under `set -e` that
  ended the install as "Installation failed" for a condition that is not a failure. Every prompt now
  sits behind a terminal check.
- **`--download-models` fetches models** instead of printing "not found, skipping". The helper it
  called had never been written, so `--full` read as supported while doing nothing.
- **A reduced install reports what it cannot run.** An absent optional extra is a skip naming the
  package rather than a failure, so a `.[dev]` checkout no longer shows around 94 red tests for
  extras that were never installed.
- **The install checks can run at once.** Both built a wheel inside the repository and raced each
  other; each now builds in its own directory, from the tracked tree, which is also what a user
  installs.
- **`activate.sh`**, which the installer writes into the clone, is no longer left as an untracked file
  in `git status`.
- **The docker compose file binds to loopback**, so `docker compose up` does not publish an
  unauthenticated server on every interface.
- **Every documentation snippet runs.** The 57 tool gallery snippets were rewritten to the awaited
  keyword API, all 30 network snippets now check `ToolResult.success` before reading output, and the
  CLI pages were re-run command by command. Two tool defects surfaced by that work were fixed: a news
  fetch reported success with zero articles when every RSS source was unreachable, and a web search
  sent unset filters to the backend and reported the resulting `NoneType` error instead of the
  connection failure.

### Contributor-facing

- `scripts/run_tests.sh` runs the suite and the checks around it, asking which lanes to include
  before it starts, with `scripts/watch_tests.sh` and `scripts/watch_tests_web.py` showing per-lane
  progress and time remaining. A lane the machine cannot run is listed with the reason rather than
  offered.
- The test suite is order-independent and runs with the machine's ambient state removed
  (`EFFGEN_TEST_HERMETIC=1`), with per-lane timing and a flake register. It grew from 260 test files
  to 399, and now covers endurance soaks, concurrency contention, a failure-injection matrix over
  every adapter, and the installation routes end to end.
- **The largest modules were split along their responsibilities**, in behaviour-preserving steps with
  a layout gate on each one: the CLI entry point went from 5,182 lines to 1,051 with one module per
  command, `core/agent.py` from 2,059 to 844, `core/agent_react.py` from 1,920 to 1,158, the
  Transformers engine from 1,053 to 267, `server/app.py` from 1,106 to 484, the OpenAI-compatible
  API from 812 to 508, tracing from 989 to 255, and the HTML report writer into one builder per
  report kind. Every moved definition was compared against the previous tree before and after.
- **Types are checked against a ratchet.** Signatures were annotated across the package and the
  recorded result is now a ceiling: a new error, or an old one becoming more frequent, fails the
  types job, so the number can only come down.
- **The language of the shipped tree is gated.** A test scans every tracked and untracked-but-not-
  ignored file for internal process references and for self-congratulatory wording, with an
  allowlist that has to state its reason. It is proven by planting each pattern in a real file of
  each scanned kind and watching the gate fail.
- **Provider trouble is told apart from a defect.** A test that fails because a provider refused to
  serve the call, or because an optional extra is not installed, is reported as a skip quoting the
  reason, while a connection error still fails, because that shape is usually local
  misconfiguration.

### Contributors

Thank you to the people outside the maintainer who contributed code to this release:

- **Yasuo Tabei** ([@tb-yasu](https://github.com/tb-yasu)): Gemma 4's channel and tool-call format,
  and the MLX engine against the current `mlx_lm` (#94).
- **Aafiya Hussain** ([@Aafiya-H](https://github.com/Aafiya-H)): the multi-GPU `device_map` sampling
  fix and the example teardown that goes with it (#44).

### New public names

`OpenAICompatibleAdapter`, `BackendUnreachableError`, `AgentMiddleware`, `MiddlewareChain`,
`LoggingMiddleware`, `ToolApprovalMiddleware`, `ToolCall`, `ToolCallList`, `CompactionStrategy`,
`SummarizeOldest`, `DropOldest`, `KeepFirstAndLast`, `KeepToolResults`, `WorkflowCheckpoint`,
`CheckpointStore`, `FileCheckpointStore`, `InMemoryCheckpointStore`, `SystemPromptLeakGuardrail`,
`load_env`. The top-level surface grew from 204 names to 223, and nothing was removed or renamed.
`BaseModel` gained `build_assistant_message` and `build_tool_result_message`; `SandboxResult` gained
`credential_reads_masked` and `process_table_isolated`.

---

## [0.3.2] - 2026-07-05

### Highlights

**effGen v0.3.2** is another **usability, robustness & polish** point release, again driven by living with
the framework as real professionals do — a reliability/QA engineer re-certifying results integrity, a
trust & grounding auditor, a platform/security engineer, an integration & local-serving engineer, a
data/ETL engineer running batch at volume, a clinical informatics analyst, a site-reliability engineer, a
localization specialist, a model-evaluation engineer wiring CI gates, a non-technical product/ops user, a
narrative/game writer, a plugin & tool author, a multi-provider FinOps cost optimizer, and an
accessibility & document-processing specialist. It adds no new providers and no new heavyweight
subsystems; it seals the sharp edges those users hit and makes the surfaces they already reach for
predictable. **No breaking API changes** — every addition is additive, and every previously-silent trap
now surfaces a clear, typed error.

### Added — new surface

- **Structured output from the CLI.** `effgen batch --schema <file.json>` / `--output-model
  module:Class` validates every row against a JSON Schema or a Pydantic model; a row that can't be
  coerced is flagged failed with a reason instead of a silently off-schema string. `batch` also gained
  `--temperature`, `--system-prompt`/`--persona`, `-i`/`-o` aliases, and `--resume` to skip rows already
  present in the output file.
- **A CI accuracy gate.** `effgen eval --fail-under 0.8` sets a suite-level accuracy threshold that drives
  the exit code, and `--compare-baseline` now returns a non-zero exit when it detects a blocking
  regression — so a real accuracy drop fails the build. `eval`/`compare` gained `--temperature` for
  reproducible scoring and `--baseline-dir` to keep baselines in your own repo.
- **Cost-aware model selection.** `effgen compare --optimize {accuracy,cost,latency}` adds an average
  `$/run` column and factors price into the recommendation, so a bake-off between two equally-accurate
  models can pick the cheaper one; `compare --json` now carries per-model cost.
- **Document and file input on the CLI.** `effgen run --file report.pdf` (repeatable, also `--input`)
  reads a PDF/DOCX/XLSX/text document into the task, or feeds an image through the vision path — document
  work without writing Python.
- **A clinical de-identification posture.** A new `phi` guardrail preset (`get_guardrail_preset("phi")` /
  `phi_guardrails()`) pairs redaction with a fail-closed strict mode, and `PIIGuardrail` gained
  `custom_patterns` / `custom_terms` and a `strict` option.
- **Alerting & SLO building blocks are now public.** `Alert`, `AlertSeverity`, `AlertWebhook`, `SLO`,
  `SLOTracker`, `check_slo_and_alert`, and `validate_alert_rules_yaml` are exported top-level, and a thin
  SLO-to-alert bridge evaluates rules against the live meter and fires a webhook.
- **Sampling controls and a config `max_tokens`.** `AgentConfig` accepts `max_tokens`, `top_p`, `seed`,
  `frequency_penalty`, and `presence_penalty` so they can be pinned once for an agent.

The top-level public surface grew from 197 to 204 names (the seven alerting/SLO exports); no public name
was removed or renamed.

### Changed — results integrity on the generation path

- **A dict `output_schema` now populates `metadata["parsed"]`** with the parsed object, matching the
  Pydantic `output_model` behavior — a caller reading `parsed` no longer gets `None` for a schema that
  extracted correctly.
- **An empty or whitespace-only task is rejected before any model call** with a typed error, instead of
  being sent to the model and billed as a `success=True` result.
- **`AgentResponse.tokens_used` reports total tokens**, matching its documentation and the Prometheus
  token counters, so usage dashboards no longer under-count by the prompt tokens.
- A raw `GenerationResult` carries a truncation signal when `finish_reason="length"`, `run_async()` lists
  `output_schema`/`output_model` in its signature, `AgentConfig` no longer requires `name`, and
  `cost_usd` is the one canonical cost key on raw metadata.

### Changed — grounding you can trace

- **Native web search now surfaces the URLs it searched even when the model answers without inline
  citations.** On the OpenAI native/hybrid path, `response.sources` is populated from the search results
  (`web_search_call.action.sources`) so an answer built on a search is never handed back with empty
  provenance; inline `url_citation` annotations remain the higher-precision `.citations`.
- **Partial document ingestion tells you what it skipped.** When some files index and others fail, the
  skipped files are named with a reason (previously discarded unless *every* file failed), and a
  pre-built `VectorMemoryStore` uses a `title`/`doc`/`name` metadata value for human-traceable citations.

### Changed — a consistent server contract

- **A failed non-streaming completion returns a real HTTP status.** A generation failure is returned as a
  4xx/5xx error envelope (mirroring the streaming path) instead of an HTTP 200 whose `message.content` is
  the error text — so an OpenAI-client pipeline raises instead of treating the error string as an answer.
- An unknown provider prefix returns a typed 4xx that lists the known providers (instead of a 500 leaking
  the local-loader internals), `/v1/models` reflects the servable catalog, credential redaction covers the
  AWS secret access key and plaintext role-label injections, and throttled responses carry the standard
  `RateLimit-*` headers.

### Changed — batch that survives real data

- **A per-job cost and token total** print on the "Batch complete" line and land in `BatchResult`, and the
  written output file is lossless — each row carries `cost_usd`, token counts, the validated `parsed`
  object, and a failure reason.
- **One malformed input row no longer aborts the whole job.** A bad line is skipped and reported (named by
  file and line number) as a failed row, with `--strict` to hard-fail instead.

### Changed — clinical-grade redaction

- **PHI redaction covers the labeled clinical identifiers.** Label-anchored patterns catch patient name,
  date of birth, medical record number, address, and member/beneficiary ID; SSN detection now matches
  space-separated and undelimited forms; and the strict mode fails closed when a high-risk field can't be
  fully redacted. A redaction summary (types and counts) is surfaced in `response.metadata`.

### Changed — observability an on-call rotation can use

- **Server metrics carry the labels you alert on.** The request path records provider/model/status-labeled
  token, latency, and request counters, so you can graph error rate by status and cost/latency by model
  straight from `/metrics`. The Helm chart's Prometheus scrape works out of the box, the AWS Lambda
  runtime moved to a supported version, and `EFFGEN_NO_DOTENV=1` disables the filesystem `.env` walk for
  a production process.

### Changed — orchestration, evaluation & selection

- **The `general` preset runs on Gemini.** Array-typed tool parameters missing an `items` subschema are
  sanitized in the Gemini adapter, so a preset with array-param tools no longer 400s.
- **Workflow YAML honors an `edges:` block** (and warns on unknown top-level keys) instead of silently
  dropping the DAG's wiring and reporting the workflow valid; custom `eval`/`compare` datasets accept
  `input`/`prompt`/`question` as aliases for `query` and name the missing field on a load error; and a
  single oversized request is classified as payload-too-large (413) rather than routed through the
  rate-limit/failover path.

### Changed — trustworthy prompt library & terminal UX

- **The prompt library validates `--input` against the template's own schema.** `prompts run`/`render`
  rejects a missing required field, a wrong type, or an out-of-enum value with a field-named error and a
  non-zero exit, instead of rendering silently-wrong text and billing it (a list-valued field passed as a
  string no longer renders one bullet per character).
- The `chat` `/model` hot-swap honors a `provider:` prefix (or refuses the swap) rather than printing
  "Switched" and then failing every turn, the advertised dashboard shows an on-screen reason when its data
  endpoints require auth (with `EFFGEN_PUBLIC_DASHBOARD` documented in `serve --help`), a failed run shows
  the classified message instead of the raw provider JSON, and adapter warnings stay out of normal stdout.

### Changed — sampling controls that take effect

- **`seed`, `frequency_penalty`, `presence_penalty`, and `top_k` now reach the model** through
  `Agent.run()` and `AgentConfig`, matching what `GenerationConfig` already advertised — a writer can pin
  a seed to reproduce a generation or dial down repetition in long prose. An unrecognized `run()` keyword
  is now rejected with a clear message instead of being silently swallowed, and the Groq adapter forwards
  the penalties the OpenAI adapter already did.

### Changed — extend effGen without hitting a seam

- **A `@tool`/`Tool.from_function` instance can be registered** (and published to the CLI, plugin
  discovery, and the MCP server), the plugin-development guide's tool example runs as written, a
  `list[str]`/`list[int]` annotation carries its element type into the tool schema, and a mistyped
  `@tool(category=...)` is warned about instead of silently coerced.

### Changed — document intake & FinOps accuracy

- **Spreadsheets ingest.** `.xlsx` files are read into a RAG corpus (rows to text, workbook metadata
  carried through), and a directory ingest never silently drops a file — anything it can't parse is named
  in the skip report rather than quietly excluded. PDF dates are normalized to ISO-8601.
- **The budget config location is honored end-to-end.** `EFFGEN_BUDGET_CONFIG` now redirects the CLI
  display and every write (not just the enforcement read), so setting a budget no longer clobbers your real
  `~/.effgen/budget.json`; an unpriced catalog entry reads as unpriced rather than a fabricated
  `$0.000000`; and a pre-flight budget gate refuses a call once the period is already over budget.

### Fixed

- Corrected the priced-vs-unpriced classification for catalog entries carrying a `0/0` price on a
  non-free-tier model (they now read as unpriced), and the `/bin/bash` local-run pricing.
- Closed the evaluator's agent in a `finally` block (no more garbage-collected-without-close warning),
  added `--no-animation` to `compare` for flag parity with `eval`, and surfaced a
  `semantic_similarity`-to-`contains` scoring fallback in the results.
- Quieted the MCP client's manual-teardown traceback and reconciled `EffGenMCPServerConfig` with the
  server constructor.

## [0.3.1] - 2026-06-29

### Highlights

**effGen v0.3.1** is a **real-world usability & polish** release, driven by living with the framework as
real professionals do — a finance analyst, a journalist, a researcher, a founder, a backend engineer, a
support lead, an ML engineer, an educator, a security reviewer, an integration engineer, and a legal
knowledge manager. It adds no new providers or subsystems. Instead it seals the sharp edges those users
hit first: grounded results now carry the sources they were built from, reasoning models finish
token-heavy work, a custom persona is honored on every path, multi-agent teams fail honestly, the server
stops silently downgrading, code-execution is safe to enable, and a knowledge domain becomes a runnable
agent in one call. **No breaking API changes** — every addition is additive.

### Added — traceable evidence on every result

- **`response.sources` and `response.citations` are populated** from the URLs a run actually retrieved
  (`web_search`, `url_fetch`, `news`, `wikipedia`) and from provider-native grounding (OpenAI
  `url_citation` annotations, surfaced as `metadata["grounding_chunks"]`; Gemini search grounding). Only
  tool-returned URLs land here — never URLs scraped from the model's prose — so a caller can verify and
  link them programmatically. The research preset is instructed to cite **only** a URL one of its tools
  returned this run and never to invent one.

### Changed — reasoning models finish the job

- **Reasoning models (the `gpt-5` family, `o`-series) no longer return empty, billed results** on
  token-heavy tasks. These models spend output budget on hidden reasoning, so the old fixed 1024-token
  default could be fully consumed and return an empty but billed response. They now get a larger default
  output budget (4096) across every path (direct, ReAct, streaming, speculative, native-tool, structured),
  an empty result with `finish_reason="length"` is treated as **truncation** — the budget grows once and
  retries, or fails with an actionable "increase `max_tokens`" message — and a starved budget is never
  retried three times. `effgen batch` gained `--max-tokens`.

### Changed — honest, costed, measurable results

- **Cost and tokens on every result.** `AgentResponse.metadata` now carries `cost_usd` and prompt/
  completion/total token counts (summed across the run, tool loops included); local models report no
  `cost_usd` key rather than a fake `$0`. Per-run `latency_ms`/`duration_s` is folded onto both
  `AgentResponse` and raw `GenerationResult` metadata, so throughput is computable without a manual timer.
- **Readable sub-cent costs.** A shared adaptive formatter shows real SLM costs (e.g. `$0.000049`) at the
  per-turn footer, `/cost`, and the notebook card instead of rounding them to `$0.0000`; genuinely
  free/local stays an honest `$0.00`.
- **Team and workflow results report summed `cost_usd`/tokens** in metadata, mirroring the single-agent
  surface.
- `str(GenerationResult)` returns the generated text (matching `AgentResponse`), and a notebook card
  (`_repr_html_`) renders `model.generate()` cleanly instead of a dataclass repr.
- Passing a Pydantic class to `output_schema=` now also populates `metadata["parsed"]` with a typed
  instance.

### Changed — your persona is honored everywhere

- **A custom `system_prompt` now steers every response.** It was silently dropped on the no-tool direct
  path, the no-tool streaming path that `chat` uses, and the native/hybrid tool path — so a carefully
  written tutor or fixed-language assistant looked applied but was ignored. The persona is now captured at
  construction and applied as a system message (or prepended) on every path; default agents are unchanged.
- New `chat --system-prompt/--persona` to stand up a custom assistant from the terminal, an
  `education.*` prompt set (`socratic_tutor`, `lesson_plan`, `quiz_generate`, `explain_simply`), and
  `prompts list --json` as an alias for `--format json`.

### Changed — trustworthy multi-agent teams & workflows

- **Collaborative teams fail closed.** A failed collaborator now sets the team `success=False` with a
  discoverable per-agent error and a redacted reason, instead of always returning `success=True` with the
  failure invisible.
- **Hierarchical teams route by name.** Each subtask goes to the worker the manager *named* (round-robin
  only as a fallback), every subtask runs instead of extras being dropped, and a failed worker fails the
  team — making hierarchical the real triage→specialist handoff. The `pipeline` pattern's docstring is
  corrected and a "route to one specialist" recipe is documented.
- **Workflow DAGs don't run downstream of failure.** A node whose required upstream failed (or was
  skipped) is marked skipped with a reason, so an internal error is never rewritten into a downstream,
  customer-facing answer. On failure, sequential/hierarchical no longer echo the caller's own input back
  as the answer.
- `effgen chat --session-id/--resume` continues a persisted conversation (the same store `run --session-id`
  and `sessions` use); streamed turns are saved and the session-vs-checkpoint help is aligned.

### Added — one-call agents from a knowledge domain

- **A knowledge domain becomes a runnable agent in one call:** `LegalDomain().to_agent("gpt-5-nano")`
  (or `create_agent(domain=...)`) wires the domain's system prompt, recommended tools, and guardrails
  into an agent — giving the bundled domain guardrails their first real consumer.
- A RAG agent accepts a pre-built `VectorMemoryStore` as its `knowledge_base`, connecting the memory and
  retrieval subsystems (an empty store still fails loudly).
- The everyday guardrail classes (`PIIGuardrail`, `GuardrailChain`, the presets, …) are exported at the
  top level for the same discoverability as the domains. The non-tech domains (legal, finance, health,
  science) expand seed keywords into field-appropriate query variants (`"{kw} clause"`,
  `"{kw} obligations"`, `"{kw} regulation"`) instead of borrowing the tech how-to templates.

### Changed — honest OpenAI-compatible server

- **No silent tool downgrade.** `/v1/chat/completions` no longer silently drops a client-defined function
  tool it does not host; the unhosted tool is rejected with a clear `400` (`unknown_tool`) naming it.
  Built-in tools still resolve and run server-side.
- **Honest embeddings.** `/v1/embeddings` strips a `provider:` prefix so `openai:text-embedding-3-small`
  reaches the real neural model; when the neural backend can't load it warns once and reflects the lexical
  fallback to the caller (`effgen.degraded`/`backend` + `x-effgen-embedding-backend` header), or fails
  closed with `503` under `EFFGEN_EMBEDDINGS_STRICT=1` — no more near-zero hash vectors served under a
  neural model's name.
- Auth (401), validation (422), rate-limit (429), and RBAC/budget (403/429) now share the same
  `{"error":{message,type,code}}` envelope as model errors; per-call `cost_usd` is surfaced in the
  `effgen` response extension for priced models; an empty `messages` array returns `400`; and
  `effgen serve --help` documents the operational env knobs and adds `--rate-limit`.

### Changed — local-first truth: GPU, catalog, structured output

- **Grammar-constrained structured output across model families.** With the new optional
  `effgen[grammar]` extra (`outlines`), small local models that won't follow a JSON schema by prompting
  alone (Llama-3.2-3B, gemma-2-2b) emit schema-valid output in one constrained pass; when the extra is
  absent the honest `success=False` ends with a one-line fix hint.
- **`models status` shows physical GPU memory** (the driver's view across all processes, via
  `mem_get_info`) plus a utilization column — so it shows which GPU is actually free, not the calling
  process's reservations.
- **`models info` is local-aware:** a model in the local HuggingFace cache is described as
  locally-runnable (engines, on-disk size, context window) instead of "not found" or a cloud-only route;
  incomplete downloads are flagged rather than counted as ready.
- **Thread-safe local batch.** A per-engine lock serializes the thread-unsafe fast tokenizer, so local
  Transformers batch at the default concurrency no longer emits `"Already borrowed"` errors.
- `effgen compare` breaks accuracy ties on lower latency then fewer tokens, gained `--max-cases`/
  `--difficulty`, and (with `eval`) accepts your own `.jsonl`/`.json` dataset; `eval --suite list` shows
  real per-suite case counts.

### Changed — RAG ingestion & newcomer ergonomics

- **PDFs ingest out of the box.** RAG `knowledge_base` no longer hard-requires `pymupdf`; PDFs fall back
  to `pypdf`/`pdfplumber`, and when a file is skipped the error names *why* each was skipped instead of a
  bare "0 documents to index".
- `create_agent(extra_tools=["calculator"])` accepts tool **name** strings (with "did you mean" on a
  typo), `tools=` is accepted as an alias for `extra_tools`, and bare `Agent()` teaches how to construct
  one. A mistyped model id surfaces a single clean "did you mean" up front, `create_agent` reports an
  unknown preset before demanding a model, a failed construction no longer tails a
  "garbage-collected without close()" warning, and `effgen run` with no `-m` mirrors `quickstart` (prefer
  a detected cheap cloud model, then a small local one) and says which and why.
- `Agent.run(inputs=["photo.png"])` auto-wraps a bare image/audio/video path by extension; `agent.aclose()`
  / `async with agent:` clean up asynchronously, and awaiting the sync `run()` raises a clear
  "use `run_async()`" message. A dangling `Final Answer:` label and internal ReAct loop nudges are stripped
  from answers.

### Changed — automation & integration

- **Sync `Agent.run()` no longer hangs forever** when handed a tool whose async resource is bound to the
  calling event loop (e.g. an MCP stdio session): it runs on a daemon thread bounded by the timeout and
  raises a clear `TimeoutError` pointing at `await agent.run_async(...)` instead of an indefinite silent
  hang.
- **Installed tool plugins auto-discover.** A package published with an `effgen.plugins` entry point —
  what `effgen create-plugin` scaffolds — has its tools folded into the registry on first use (set
  `EFFGEN_DISABLE_PLUGINS=1` to opt out).
- **`effgen run --json`** emits the full result document to stdout for piping to `jq` (combine with `-q`
  for pristine stdout), and `--json` is added to `eval`, `compare`, `workflow`, and `sessions list`. The
  official MCP server gained a package entry point so `python -m effgen.tools.protocols.mcp_official`
  starts without the runpy double-import warning; unknown background-task ids raise a clear "no such task"
  message.

### Security — hardened code-execution, secrets, and guardrails

- **The Python REPL sandbox toggle is out of the model's hands.** `restricted_mode` is no longer in the
  model-facing schema; unrestricted execution is a developer-only opt-in
  (`PythonREPL(allow_unrestricted=True)` or `EFFGEN_REPL_ALLOW_UNRESTRICTED`), and a model-supplied
  `restricted_mode=False` is ignored — fail-closed.
- **The `bash` env scrub is exhaustive and drift-proof:** it strips every provider key plus anything that
  looks like a credential (`*_API_KEY`/`*_TOKEN`/`*SECRET*`/`*PASSWORD*`), refuses reads of common secret
  files (`.env`, `~/.ssh`, `~/.aws/credentials`, private keys, `.netrc`), no longer claims to run
  "safely", and is no longer bundled in the `general` preset (it stays in `coding`).
- The **prompt-injection guardrail** catches the common textbook phrasings (disregard/forget/pretend,
  role-delimiter spoofing, `System:`/`### New system prompt` headers, repeat-the-text-above leaks) with no
  false positives, documented as best-effort defense-in-depth. The **PII guardrail** can optionally treat
  leaked API keys / cloud credentials as sensitive, and now redacts an IPv4 that **ends a sentence**
  (`"Server at 10.2.3.4."`) while still leaving version strings (`1.2.3.4.5`) alone.

### Quick start

```python
from effgen import create_agent, LegalDomain

# Grounded research: response.sources/.citations are filled from the URLs the
# tools actually retrieved (never from the model's prose), with honest cost.
agent = create_agent("research", "openai:gpt-5-nano")
r = agent.run("What is the capital of France? Cite a source.")
print(r.text)                      # "...Paris (Source: https://en.wikipedia.org/wiki/Paris)."
print(r.sources)                   # ['https://en.wikipedia.org/wiki/Paris']
print(r.metadata["cost_usd"], r.metadata["latency_ms"])

# A knowledge domain becomes a runnable agent in one call.
legal = LegalDomain().to_agent("openai:gpt-5-nano")
print(legal.run("What does an NDA confidentiality clause protect?").text)
```

```bash
effgen run --json -q "What is 25 * 17?" | jq .output   # pure-JSON stdout for CI
effgen models status                                    # physical GPU memory; which card is free
```

### Upgrading from v0.3.0

No breaking API changes; every change is additive or makes a previously-silent failure honest. If you
relied on `response.sources`/`.citations` being empty, they are now populated from retrieved URLs; if you
gave `Agent.run()` an MCP stdio tool on the synchronous path, you now get a clear `TimeoutError` pointing
at `run_async()` instead of a hang.

```bash
pip install --upgrade effgen
```

## [0.3.0] - 2026-06-19

### Highlights

**effGen v0.3.0** is a major **stabilization & hardening** release. It adds no new providers, tools, prompt
templates, or subsystems — instead it makes everything already in effGen **robust, predictable, fast,
secure, and pleasant to use**. Failures are now loud and typed instead of silently succeeding; the
model catalog updates itself and warns when it drifts; local GPUs work out of the box; the API server
fails closed; the built-in tools are sandboxed and SSRF-safe; streaming and the agent loop are
dramatically faster; the CLI is quiet and scriptable; and `import effgen` is effectively instant. No
breaking API changes — every ergonomic addition is an additive alias.

### Changed — fail-closed behavior

- **No more silent success.** `Agent.run()` never returns `success=True` with empty output. Both the
  direct and tool paths now return the *same* shape on failure: `success=False`, a typed redacted
  error, `metadata["error"]`, and a consistent `metadata["reason"]`. A new
  `classify_provider_error()` maps provider exceptions to a stable taxonomy (`auth`, `not_found`,
  `rate_limited`, `retryable`, `fatal`).
- **Smarter retries.** Retries fire only on retryable / rate-limited errors; auth and not-found errors
  fast-stop with a single clear message instead of a retry storm. `AgentConfig.raise_on_error` lets
  callers opt into exceptions.
- **Helpful not-found.** A wrong or 404 model id suggests the nearest live alternative instead of
  crashing (`Unknown Cerebras model 'llama3.1-8b'. Did you mean: gpt-oss-120b, zai-glm-4.7?`).
- **Engine import errors** distinguish "module not installed" from "installed but failed to import
  (ABI/CUDA mismatch)".

### Added — self-updating, drift-aware model catalog

- Every provider ships a **local catalog snapshot** (id, context window, max output, input/output
  price per 1M tokens, tool/vision/audio support, free-tier flag, rate limits) with a **count** and a
  **"verified on" date**.
- **`effgen models refresh [--provider X]`** fetches the live list, updates the snapshot, and reports
  models added / removed / changed. `check_drift()` reports divergence; effGen shows a single,
  non-spammy warning when its catalog looks stale.
- Refresh/drift now **filter to chat/text-generation base models** and never persist private `ft:`
  fine-tune ids, embeddings, audio, or image models.
- **Pricing completeness** — every record is priced or explicitly flagged free/unpriced; the cost
  tracker reads pricing from the catalog (no silent `$0`), and token accounting prefers provider
  usage over heuristics.

### Changed — real GPU support

- The documented GPU install selects a **driver-compatible torch**; a runtime guard warns once when
  NVML sees GPUs but `torch.cuda` cannot (CPU fallback). CPU / CUDA-12.4 / CUDA-13 / vLLM install
  matrices are documented.
- **`temperature=0` works** — `temperature <= 0` is treated as greedy decoding (`do_sample=False`)
  across local backends instead of raising "temperature must be strictly positive".
- **GPU allocator** no longer deadlocks on `reset()`, reads real free memory via NVML /
  `torch.cuda.mem_get_info` (respecting other processes), and no longer mutates the global CUDA
  device during discovery.
- Local tool answers no longer leak chat-template special tokens (e.g. Gemma `<end_of_turn>`/`<eos>`,
  Qwen `<|im_start|>`); the strip set is derived from the tokenizer while preserving tool-call
  delimiters.

### Security — server fails closed

- Outside dev mode with no configured issuer/JWKS, the server **rejects all bearer tokens** — a forged
  HS256 JWT can no longer reach `/whoami` or `/v1/chat/completions`. `/v1/*` return 401 without
  credentials.
- Production CORS no longer combines wildcard origin with credentials; metrics and the dashboard
  require auth unless explicitly opened; the `viewer` role can no longer run tools and unknown roles
  are rejected (strict mode).
- Budget enforcement **reserves then reconciles** so failed calls are not charged; request bodies are
  size-limited before buffering; the server version is sourced from `effgen.__version__`.
- Upstream/provider-auth/missing-key failures map to **502/503** (server-side) — `401` is reserved for
  genuine client-auth failures.

### Security — hardened built-in tools

- **PythonREPL** runs user code in a worker subprocess with a hard wall-clock timeout, process-group
  kill, and memory/output caps enforced *outside* the executed code (`while True: pass` now dies at
  its timeout, not ~30 s later).
- **One shared SSRF guard** (`tools/builtin/_net.py`) — DNS resolution, per-IP classification, and
  re-validation on every redirect — protects *every* URL-taking builtin; private/loopback/link-local/
  metadata addresses are blocked by default (opt-out for user-URL tools, host-pinned for fixed
  endpoints).
- **Path confinement** (`tools/builtin/_fs.py`) on every file-path tool; the un-gated `pickle.load`
  path was removed (JSON-only state); prompt-chain conditions are evaluated with an AST-whitelist
  comparator instead of `eval()` over interpolated model output. A security-pattern gate test prevents
  regressions.

### Performance

- **`import effgen` is effectively instant** — the public surface is resolved lazily, so a bare import
  / version check dropped from ~7.5 s / ~800 MB to **~0.02 s / ~12 MB**; heavy provider/torch imports
  are deferred until first use.
- **Faster streaming** — removed per-chunk overhead, true incrementality across OpenAI/Gemini/Groq and
  local backends, typed mid-stream errors (no longer buffered into one final chunk), and
  optional usage in the final chunk.
- **Efficient agent loop** — repeated identical tool calls are short-circuited and the loop stops once
  a confident answer exists ("compute 15 squared" went from **6 tool calls / 66 s** to **1 tool call**).
- **Faster, structured output** — native JSON modes are preferred; unmatched schemas return
  `success=False` with the raw text and validation error instead of silently reprompting.

### Changed — consistent, scriptable surfaces

- **Input robustness** — `Agent.run()` accepts `str | Message | list[ContentPart]` and a first-class
  `inputs=` kwarg for media; clear `TypeError` otherwise.
- **CLI** — quiet by default (clean tables, no INFO spam); `--json` on `doctor`/`models`/`tools`/
  `cost`; `run`/`chat`/`debug` accept `--provider`; `models list`/`info`/`refresh` are catalog-backed
  with split views (cached / configured / live) showing status, price, context, deprecation, and
  verified-on; `doctor` distinguishes "key present" from "key usable" and gains `doctor --live
  --cheap` plus a CUDA/torch/vLLM system report; user errors exit non-zero.
- **Construction ergonomics** — the obvious calls work: `create_agent(preset, model, name="X")`,
  `TemplateManager()` is populated by default, `ConfigLoader.load`, `ShortTermMemory.get_messages`,
  `TestCase(input=, expected=)`, and a `@tool` / `Tool.from_function()` helper. A bare/invalid
  constructor raises a clear accepted-kwargs error instead of a cryptic dataclass `TypeError`.
- **KeywordExpander** validates its input (a bare string is one keyword, not character-iterated) and
  `get_guardrail_preset("default")` resolves.
- **Tool registry auto-discovers** lazily (no INFO spam); `effgen tools info/test` discover tools and
  exit non-zero on failure; the plugin scaffold round-trips (create → install → import → run).

### Changed — internals & DX

- One implementation per concern: a single `CircuitBreaker`, metrics/logging/tracing consolidated with
  import-compatible shims, a `NullHandler` on the package logger, and **no INFO logging on import**.
- Multi-agent teams, workflows, and the DAG are exported top-level, reconciled to consistent shapes,
  and **fail honestly** (a failing node/sub-agent → `success=False` with a typed per-node error; an
  empty workflow → `success=False, reason="empty_workflow"`).
- The MCP server round-trips with the **official `mcp` client** (stdio + HTTP) behind a fail-closed
  tool allowlist; sessions, checkpoints, and `effgen resume` are durable with clear errors on
  corrupt/absent files; Jupyter magics, the dashboard, `effgen debug`/`compare`, and shell completion
  all work against real runs.
- **Live progress UX** — TTY-aware "thinking"/working status, a live elapsed/tokens/cost meter that
  freezes into a final summary, graceful Ctrl-C (partial result + "Stopped."), rotating tips,
  "did you mean?" suggestions, a first-run welcome, `effgen quickstart`, rich Markdown/code rendering,
  `print(result)` → the answer, a Jupyter `_repr_html_` card, and a polished `effgen chat` REPL with
  slash commands and per-turn cost. All animation honors `NO_COLOR`, `--no-animation`, `--quiet`, and
  non-interactive/CI output.

### Packaging & dependency security

- Reconciled extras with a single source of truth (`[api]`, `[server]`, `[rag]`, `[tools-web]`,
  `[tools-docs]`, `[local]`, `[vllm]`, `[all]`), tested upper bounds for fast-moving deps, a
  `constraints-cu1xx.txt` flow that prevents an extras re-install from replacing a working torch, one
  canonical installer, and a conda recipe that sources the real version.
- `pip-audit` runs clean across the documented extras; `pypdf` was bumped to `>=6.13.3`
  (GHSA-jm82-fx9c-mx94). The `requirements-all-lock.txt` reproduces `.[all]` on Python 3.11.

### Quick start

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")
agent = Agent(config=AgentConfig(name="math_agent", model=model, tools=[Calculator(), PythonREPL()]))

result = agent.run("What is 24344 * 334?")
print(result.output)
```

```bash
effgen models refresh                 # update the catalog from the live provider APIs
effgen doctor --live --cheap          # check which provider keys are actually usable
effgen run --provider groq "Summarize the theory of relativity in two sentences."
```

## [0.2.10] - 2026-05-27

### Highlights

**effGen v0.2.10** is the **Security, Edge & Developer Experience** release — hardening effGen end-to-end with secret scanning, dependency auditing, a SBOM pipeline, supply-chain integrity verification, a sandboxed `CodeExecutor`, OAuth2/OIDC auth with RBAC and a per-request audit log, Docker and Helm production deployments, AWS Lambda (Mangum adapter), a Cloudflare Worker edge proxy, a VSCode extension with prompt-template completion, Jupyter magics, and a live local dashboard. No breaking API changes; every security and DX feature is additive.

### Added

#### Security — Secret Scanning (`effgen/security/`, `.gitleaks.toml`, `.pre-commit-config.yaml`)

- **`.gitleaks.toml`** — tuned rule set covering OpenAI, Anthropic, Cerebras, Google, HuggingFace, Groq, Slack, Discord, and Bearer-token patterns. Allowlist for test fixtures with obviously fake keys.
- **`.pre-commit-config.yaml`** — gitleaks pre-commit hook; blocks commits containing secret-like strings.
- **`.github/workflows/secret-scan.yml`** — CI secret-scan workflow; scans both working tree and full git history. Fails on any detected real secret.
- **`tests/security/test_secret_patterns.py`** — plants fake secrets in a temp file, runs gitleaks, asserts detected; repo-history scan asserts clean.

#### Security — SBOM (`sbom.cdx.json`, `.github/workflows/sbom.yml`)

- **`sbom.cdx.json`** — CycloneDX 1.5 SBOM; every runtime dependency listed with name, version, and PURL.
- **`.github/workflows/sbom.yml`** — CI SBOM workflow; generates `sbom.cdx.json` on each push; validates against CycloneDX schema; uploads as release artifact.
- **`tests/security/test_sbom.py`** — asserts every runtime dep from `pyproject.toml` appears in the SBOM.

#### Security — Dependency Pinning (`requirements-lock.txt`, `requirements-all-lock.txt`)

- **`requirements-lock.txt`** — hash-verified lock for base dependencies. `pip install -e . -c requirements-lock.txt` is reproducible.
- **`requirements-all-lock.txt`** — hash-verified lock for `.[all]` extras bucket (uv-generated; `google-protobuf` floors + `fireworks-ai<0.18` cap resolve deep-resolution issues).

#### Security — Vulnerability Audit (`docs/security/`)

- **`.github/workflows/deps-audit.yml`** — `pip-audit` CI; fails on HIGH/CRITICAL; reports MEDIUM. Excludes `fastapi==0.136.3` (known malicious, pinned away in lock).
- **`.github/workflows/sbom.yml`** — SBOM generation and CycloneDX schema validation.
- **Startup hash verification** — `EFFGEN_VERIFY_HASHES=1` on startup compares installed-wheel hashes against the lockfile; logs `hash_verification: ok` or `hash_verification: drift` with the first drifted package.
- **`tests/security/test_vuln_audit.py`** — runs `pip-audit --format json`; asserts no HIGH/CRITICAL in the current environment.
- **`tests/security/test_supply_chain.py`** — asserts `pyproject.toml` contains required fields (`license`, `authors`, `urls`).
- **`docs/security/secrets.md`** — secret-scanning guide and pre-commit setup.
- **`docs/security/sbom.md`** — SBOM generation and validation guide.
- **`docs/security/supply_chain.md`** — supply-chain hardening, hash verification, and Dependabot configuration.

#### Security — Sandbox for CodeExecutor (`effgen/security/sandbox.py`)

- **`SubprocessSandbox`** — rootless user-namespace isolation via `unshare --map-root-user --net --pid --mount`; separate `/tmp` bind-mount; network blocked; no `CAP_SYS_ADMIN` required. Loud warning when falling back from Docker.
- **`DockerSandbox`** — `--read-only --network=none --cap-drop=ALL --pids-limit=100 --memory=256m`; non-root user; automatically selected when Docker daemon is available.
- **`FirecrackerSandbox`** — stub interface; `NotImplementedError` with install instructions for v0.3 roadmap.
- **`OffSandbox`** — `EFFGEN_SANDBOX_BACKEND=off`; executes on host with a loud startup warning; never auto-selected.
- **`SandboxConfig`** — env-driven: `EFFGEN_SANDBOX_BACKEND=docker|subprocess|off`, `EFFGEN_SANDBOX_TIMEOUT=10`.
- **`CodeExecutor.run(code, language)`** — dispatches to the configured sandbox backend.
- **`tests/security/test_sandbox.py`** — network-block test (subprocess path: `urllib.request.urlopen` → `OSError`); filesystem isolation test (`rm -rf /tmp/evil` leaves host untouched). Docker paths skip when daemon unavailable.
- **`docs/security/codeexecutor.md`** — threat model, sandbox architecture, and configuration reference.

#### Auth — OAuth2/OIDC, RBAC, Audit Log (`effgen/server/auth.py`, `effgen/server/rbac.py`, `effgen/server/budget.py`, `effgen/server/audit.py`)

- **OIDC JWT validation** (`effgen/server/auth.py`) — Bearer JWT validation on every non-public endpoint via `authlib`. Configurable issuer, `client_id`, JWKS endpoint via env vars (`EFFGEN_OIDC_ISSUER`, `EFFGEN_OIDC_CLIENT_ID`, `EFFGEN_OIDC_JWKS_URI`). Public endpoints: `/health`, `/metrics` (configurable).
- **RBAC** (`effgen/server/rbac.py`) — `Role(name, allowed_tools, allowed_models, max_cost_per_day)`. JWT claim `roles: [...]` resolved per-request. `RBACBudgetMiddleware` (pure-ASGI, no body-consumption issue) enforces `allowed_tools` (403) and daily cost cap (429 `BudgetExceeded`).
- **Budget tracking** (`effgen/server/budget.py`) — per-principal daily cost accumulation; 429 on cap breach.
- **Audit log** (`effgen/server/audit.py`) — every request/response pair appended to `~/.effgen/audit/<date>.jsonl`. Fields: `ts, principal, role, endpoint, request_summary, response_summary, outcome`. Content redacted via `Redactor`.
- **Dev mode** — `EFFGEN_DEV_MODE=1` disables auth with a loud startup warning. Default off in CI/prod.
- **`tests/server/test_auth.py`** — JWT verify, role matrix, unauthenticated rejected, reader/deny_tools 403, cost-cap 429.
- **`tests/server/test_audit.py`** — audit log fields present, no secrets in log.
- **`docs/server/auth.md`** — OIDC configuration guide (Auth0 walkthrough).
- **`docs/server/rbac.md`** — role definition, JWT claims, tool/model allow-lists.
- **`docs/server/audit.md`** — audit log format, location, rotation.

#### Deploy — Docker (`deploy/docker/Dockerfile`, `docs/deploy/docker.md`)

- **Multi-stage Dockerfile** — `python:3.11-slim` builder + slim runtime; non-root user (`uid=1000`); read-only filesystem; `HEALTHCHECK` on `/health`; regular (non-editable) install with `EXTRAS=server`.
- **`prometheus-client`** added to `server` extras (required for `/metrics`).
- **`tests/deploy/test_dockerfile.py`** — Dockerfile structure: 19 checks (non-root, HEALTHCHECK, EXPOSE, multi-stage); integration path: `docker run --health-cmd` (skipped without Docker group access).
- **`docs/deploy/docker.md`** — one-liner quickstart, environment variable reference, multi-platform build instructions.

#### Deploy — Kubernetes / Helm (`deploy/k8s/helm/effgen/`, `docs/deploy/kubernetes.md`)

- **Helm chart** — `Chart.yaml`, `values.yaml`; templates: `Deployment`, `Service`, `Ingress`, `ConfigMap`, `Secret`, `ServiceAccount`, `NetworkPolicy`, `PodDisruptionBudget`, `HPA` (CPU + `effgen_model_call_latency_seconds`), `PersistentVolumeClaim`.
- Default replicas=2; resource requests `cpu:100m / memory:256Mi`; HPA min=2, max=10.
- **`kubeconform`** validates all rendered manifests against Kubernetes 1.29 strict schema.
- **`tests/deploy/test_helm_lint.py`** — `helm lint` (40 tests); `helm template` produces valid YAML; all resource kinds present; HPA custom metric configured.
- **`docs/deploy/kubernetes.md`** — minikube walkthrough, OIDC secret wiring, HPA tuning guide.

#### Deploy — AWS Lambda (`deploy/aws_lambda/`, `docs/deploy/lambda.md`)

- **`deploy/aws_lambda/handler.py`** — Mangum-adapter wrapping the FastAPI app; `lifespan="off"`; `ProviderRegistry` preloaded at module level for cold-start optimisation (first call < 3 s, warm call < 100 ms); per-invocation timeout budget enforced → 504 on overrun.
- **`deploy/aws_lambda/sam-template.yaml`** — AWS SAM template: HTTP API + Lambda function + CloudWatch Log Group + Outputs. SecretsManager reference for API keys.
- **`deploy/aws_lambda/_smoke_runner.py`** — local smoke runner for live Cerebras calls through the handler.
- **`tests/deploy/test_lambda_handler.py`** — 41 tests: v1 + v2 APIGateway events, 4xx invalid body, 404 unknown path, cold-start timing, timeout 504, SAM struct + cfn-lint.
- **`docs/deploy/lambda.md`** — build zip, deploy via SAM, expected cold-start times, SecretsManager wiring.

#### Deploy — Cloudflare Worker (`deploy/cloudflare/`, `docs/deploy/cloudflare.md`)

- **`deploy/cloudflare/worker.js`** — thin edge proxy: CORS headers, Bearer JWT auth, fixed-window rate limiting (KV-backed), upstream forward with `duplex:"half"` for streaming bodies, security response headers.
- **`deploy/cloudflare/wrangler.toml`** — routes, KV `RATE_LIMIT` namespace, env vars, staging and production environments.
- **`tests/deploy/test_cloudflare_worker.py`** — 30 tests: 11 structural + 9 unit (stubbed) + 2 real-fetch round-trip regression guards (file:// → worker.js → live origin).
- **`docs/deploy/cloudflare.md`** — wrangler deploy guide, KV setup, environment variable reference, JWT configuration.

#### DX — VSCode Extension (`tools/vscode-effgen/`, `docs/dx/vscode.md`)

- **TypeScript extension** (`tools/vscode-effgen/src/extension.ts`) — prompt-template completion from the registry, inline "Run" code lens on `LibraryPrompt` definitions, hover docs with template description + input schema.
- **`npm run compile`** — TypeScript 5.3 strict, 0 errors.
- **`tests/dx/test_vscode_build.py`** — asserts `npm ci && npm run compile` succeed; compiled JS present.
- **`docs/dx/vscode.md`** — install from `.vsix`, feature walkthrough, development guide.

#### DX — Jupyter Magics (`effgen/jupyter/magics.py`, `docs/dx/jupyter.md`)

- **`%effgen_chat <message>`** — one-shot chat; displays formatted response.
- **`%%effgen_agent <preset>`** — cell body as task; displays final answer + tool trace.
- **`%effgen_metrics`** — snapshot of current Prometheus counters inline.
- **`effgen[jupyter]` extra** — `ipython` dependency added.
- **`tests/dx/test_jupyter_magics.py`** — 31 tests: magic loading, `%effgen_chat` with mock model, `%%effgen_agent`, `%effgen_metrics`.
- **`docs/dx/jupyter.md`** — `%load_ext effgen.jupyter`, magic reference, Cerebras live example.

#### DX — Local Dashboard (`effgen/dashboard/`, `docs/dx/dashboard.md`)

- **Static SPA** (`effgen/dashboard/static/`) — served at `/dashboard` (public, no auth required). Panels: live span stream (SSE), `/metrics` summary, recent agent runs with token counts and cost, SLO burn rates. Chart.js from CDN.
- **`/dashboard/data.json`** — `{ts, metrics, slo, recent_runs, recent_spans, raw_metrics}` JSON snapshot.
- **SSE endpoint** at `/dashboard/spans` — pushes new spans in real time.
- **Auth exemption** — `/dashboard` and `/dashboard/*` paths bypassed by `AuthMiddleware`.
- **`tests/dx/test_dashboard.py`** — 46 tests: `/dashboard` 200, panel IDs present, `/dashboard/data.json` schema, SSE stream, auth boundary.
- **`docs/dx/dashboard.md`** — panel reference, SSE protocol, customisation guide.

### Tests Added

| File | Tests | Coverage |
|------|-------|----------|
| `tests/security/test_secret_patterns.py` | 4 | gitleaks planted-secret detection, repo-history clean |
| `tests/security/test_sbom.py` | 4 | SBOM structure, CycloneDX schema, dep coverage |
| `tests/security/test_vuln_audit.py` | 6 | pip-audit JSON, no HIGH/CRITICAL |
| `tests/security/test_supply_chain.py` | 30 | pyproject required fields, hash verification startup |
| `tests/security/test_sandbox.py` | 35 | SubprocessSandbox net+fs isolation; Docker paths skip |
| `tests/server/test_auth.py` | 57 | JWT, RBAC, unauthenticated rejected, 429 budget |
| `tests/server/test_audit.py` | 28 | Audit fields, no secrets, anonymisation |
| `tests/deploy/test_dockerfile.py` | 19 | Dockerfile structure; integration skipped w/o Docker |
| `tests/deploy/test_helm_lint.py` | 40 | helm lint + template, resource kinds, HPA metric |
| `tests/deploy/test_lambda_handler.py` | 41 | v1+v2 events, 4xx, 404, timing, 504, SAM + cfn-lint |
| `tests/deploy/test_cloudflare_worker.py` | 30 | Structural + unit + real-fetch round-trip |
| `tests/dx/test_vscode_build.py` | 20 | npm compile, compiled JS, manifest |
| `tests/dx/test_jupyter_magics.py` | 31 | Magic loading, chat, agent, metrics |
| `tests/dx/test_dashboard.py` | 46 | /dashboard routes, panels, data.json schema, SSE, auth |

### Validation Results

| Check | Result |
|-------|--------|
| `effgen.__version__` | **0.2.10** |
| gitleaks pre-commit | Detects planted secrets ✓ |
| gitleaks CI (dir + history) | Both exit 0 on clean repo ✓ |
| CycloneDX SBOM | Generates + validates against 1.5 schema ✓ |
| pip-audit | 0 HIGH/CRITICAL ✓ |
| `EFFGEN_VERIFY_HASHES=1` | ok/drift logged correctly ✓ |
| SubprocessSandbox network block | `OSError` on `urlopen` ✓ |
| SubprocessSandbox filesystem isolation | Host `/tmp` unchanged ✓ |
| DockerSandbox tests | Skip without Docker group (not an error) ✓ |
| API server (unauthenticated) | 401 in non-dev mode ✓ |
| RBAC deny_tools | 403 ✓ |
| Budget exceeded | 429 `BudgetExceeded` ✓ |
| Audit log | Fields correct, no secrets ✓ |
| Dockerfile | Builds; /health 200 via uvicorn smoke ✓ |
| Helm chart | `helm lint` clean; `kubeconform` strict K8s 1.29 ✓ |
| Lambda handler | 41/41 tests pass; live Cerebras call through handler ✓ |
| Cloudflare Worker | wrangler --dry-run validates; live round-trip via worker ✓ |
| VSCode extension | `npm run compile` 0 errors; .vsix buildable ✓ |
| Jupyter magics | Live Cerebras llama3.1-8b smoke ✓ |
| Dashboard | /dashboard 200 + /dashboard/data.json valid JSON ✓ |
| Full regression suite | **3721 passed, 0 failed** (88 skipped, 14 xfailed) ✓ |
| Wheel build | `effgen-0.2.10-py3-none-any.whl` built cleanly ✓ |
| Wheel smoke | `python -c "import effgen; assert effgen.__version__ == '0.2.10'"` ✓ |

### Upgrading from v0.2.9

No breaking API changes. All security and DX features are additive.

```bash
pip install --upgrade effgen
```

#### Security Quick Start

```python
# CodeExecutor now sandboxed by default (DockerSandbox if available, else SubprocessSandbox)
import asyncio
from effgen.security.sandbox import get_sandbox, SandboxConfig

async def main():
    config = SandboxConfig(backend="subprocess", timeout=10)
    sandbox = await get_sandbox(config)
    result = await sandbox.run('print("hello")', "python", config)
    print(result.stdout)  # hello

asyncio.run(main())

# Or use EFFGEN_SANDBOX_BACKEND=docker for Docker isolation
```

#### Auth Quick Start

```bash
# Start server with OIDC auth
export EFFGEN_OIDC_ISSUER=https://your-auth0-domain.auth0.com/
export EFFGEN_OIDC_CLIENT_ID=your-client-id
export EFFGEN_OIDC_JWKS_URI=https://your-auth0-domain.auth0.com/.well-known/jwks.json
effgen serve --port 8000

# Dev mode (auth disabled — for local development only)
EFFGEN_DEV_MODE=1 effgen serve --port 8000
```

#### Deploy Quick Start

```bash
# Docker
docker build -f deploy/docker/Dockerfile -t effgen:0.2.10 .
docker run -p 8000:8000 --env-file .env effgen:0.2.10

# Helm (Kubernetes)
helm install effgen deploy/k8s/helm/effgen/ -f deploy/k8s/helm/effgen/values.yaml

# AWS Lambda
cd deploy/aws_lambda && sam build && sam deploy

# Cloudflare Worker edge proxy
cd deploy/cloudflare && wrangler deploy
```

#### DX Quick Start

```python
# Jupyter
%load_ext effgen.jupyter
%effgen_chat "What is 17 * 23?"
%%effgen_agent general
Summarise the top HackerNews stories today.

# Python API
from effgen.jupyter.magics import EffgenMagics
```

---

## [0.2.9] - 2026-05-23

### Highlights

**effGen v0.2.9** is the **Observability & Reliability** release — turning effGen into something you can operate in production. Structured JSON logs with secret redaction, OpenTelemetry tracing with configurable samplers, Prometheus histograms, SLO tracking, circuit breakers, bulkheads, jittered retries, timeout propagation, a deterministic chaos harness, a Hypothesis fuzz suite, a load-testing harness with CLI, and Alertmanager-compatible alert rules ship in this release. No breaking API changes; every telemetry path is async/non-blocking.

### Added

#### Observability — Structured Logging (`effgen/observability/logs.py`)

- **`StructuredFormatter`** — emits JSON lines `{ts, level, module, event, attributes, trace_id, span_id}` with OTel span-context integration. Every log line is structured; no ad-hoc `print()` in critical paths.
- **`get_logger(__name__)` helper** — `log.event("model.call.started", model=..., cached_tokens=128)` API; module-level structured logger with trace-context injection.
- **Migration pass** — agent loop, model adapters, router, and tool call sites migrated from ad-hoc `print`/`logger.info` to the structured logger.

#### Observability — Secret Redaction (`effgen/observability/redact.py`)

- **`Redactor`** — built-in patterns for OpenAI (`sk-`), Anthropic (`sk-ant-`), Cerebras (`csk-`), Google (`AIza`), HuggingFace (`hf_`), Groq (`gsk_`), Bearer tokens, Slack webhook URLs, Discord webhook URLs. Replaces with `<REDACTED:openai_key>` etc. User-extensible via `Redactor.add_pattern(name, regex)`.
- **Applied at the log encoder** — every path is covered; secrets never appear in log output.

#### Observability — Metrics: Histograms + SLO (`effgen/observability/metrics.py`, `effgen/observability/slo.py`)

- **`effgen_model_call_latency_seconds{provider,model,outcome}`** — Histogram (buckets 50 ms–60 s).
- **`effgen_tool_call_latency_seconds{tool,outcome}`** — Histogram.
- **`effgen_agent_iteration_latency_seconds{preset}`** — Histogram.
- **`effgen_tokens_total{provider,model,kind}`** — Counter (kind ∈ input, output, cached).
- **`SLO(name, target_pct, window_seconds, query)`** and **`SLOTracker`** — rolling-window error-budget tracking. `burn_rate(name)` returns the ratio against target. `/slo` endpoint on the FastAPI server.

#### Observability — Tracing: Sampling + Span Spec (`effgen/observability/tracing.py`, `effgen/observability/spans.py`)

- **Samplers** — `AlwaysOn`, `AlwaysOff`, `ParentBased(TraceIdRatio(p))`, `ParentBased(RateLimited(per_second))` configurable via `ObservabilityConfig.tracing.sampler`. No implicit `head=1.0` in production.
- **Canonical span-attribute spec** (`effgen/observability/spans.py`) — single source of truth for every attribute name: `effgen.agent.*`, `effgen.model.*`, `effgen.tool.*`, `effgen.router.*`, `effgen.retry.*`.
- **Span emission** — all adapters, tools, and router decisions emit correct spans with declared attributes; multimodal `effgen.model.parts_count` where relevant.

#### Reliability — Timeouts (`effgen/reliability/timeouts.py`)

- **`ReliabilityConfig.default_timeouts`** — `{model_call: 60, tool_call: 30, http: 20}`. Propagated into adapter `httpx` clients + tool executions.
- **`with_timeout`, `async_timeout`, `apply_timeout`** wrappers; `audit_no_none_timeouts()` guard. Every adapter missing an explicit timeout fails the timeout audit test.

#### Reliability — Retries (`effgen/reliability/retry.py`)

- **`Retry(max_attempts, base_delay, max_delay, jitter, retryable)`** — configurable policy.
- **`@retryable(Retry(...))`** decorator with jittered exponential backoff.
- Emits `effgen.retry.attempt` OTel span event per retry attempt.
- Default policy: transient network / 5xx / 429 after Retry-After header.

#### Reliability — Circuit Breaker (`effgen/reliability/circuit.py`)

- **`CircuitBreaker(name, failure_threshold, recovery_timeout, half_open_probes)`** — three-state (CLOSED → OPEN → HALF_OPEN) per-provider breaker.
- **`CircuitBreakerRegistry`** — one breaker per provider, wired into `ProviderRegistry`.

#### Reliability — Bulkhead (`effgen/reliability/bulkhead.py`)

- **`Bulkhead(name, max_concurrency, queue_size, queue_timeout)`** — semaphore-based concurrency limiter with bounded queue, sync + async variants.
- **`BulkheadRegistry`** — one bulkhead per provider so one misbehaving provider can't starve the others.

#### Chaos Harness (`effgen/reliability/chaos.py`)

- **`Chaos(seed)`** — deterministic fault injection with reproducible outcomes across seeds.
- **Fault types**: `NetworkTimeout`, `Http5xx`, `Http429(retry_after)`, `SlowResponse(ms)`, `PartialResponse`, `MalformedJSON`.
- **`registry.with_chaos(Chaos(...))`** — attaches as middleware to `ProviderRegistry`.
- **4 canonical scenarios** validated: A (5xx fallback), B (429 + Retry-After), C (SlowResponse timeout), D (AllProvidersFailed — no silent empty string).

#### Load-Testing Harness (`effgen/tools/loadgen.py`, `effgen/cli/loadtest.py`)

- **`effgen loadtest`** CLI — concurrency, duration, scenario (fixed/synthetic/multi_tool).
- Reports throughput, p50/p95/p99 latency, error rate to stdout (JSON) or file.
- Runs against local mock model by default; `--provider` switches to live inference.
- Live smoke: 30 s, c=10, mock model → ~69 k req, 0% error, p95 ≈ 4.3 ms.

#### Alerting (`effgen/observability/alerting.py`, `docs/observability/alert_rules.yaml`)

- **6 Alertmanager-compatible rules**: `HighErrorRate` (>5% for 10 min), `HighP95Latency` (>10 s for 5 min), `CostBurnHigh` (>$10/day), `SLOFastBurn` (>14.4× error budget), `SLOSlowBurn`, `CircuitBreakerOpen`.
- **`AlertWebhook(url).fire(alert)`** — posts to Slack/Discord via Phase v0.2.6 webhook tools; generic `httpx` fallback; non-raising (fire never throws).
- Webhook URL redacted in logs.

#### Documentation (`docs/observability/`)

- `overview.md` — architecture, quickstart, configuration reference.
- `metrics.md` — all metrics with label dimensions and bucket definitions.
- `tracing.md` — sampler selection guide, span attribute spec.
- `alerting.md` — Alertmanager integration, webhook configuration.
- `loadtest.md` — load-testing harness guide with examples.

### Tests Added

| File | Coverage |
|------|----------|
| `tests/observability/test_logs.py` | JSON shape, trace_id propagation, 26 tests |
| `tests/observability/test_redact.py` | Every pattern on known fixtures, 37 tests |
| `tests/observability/test_metrics.py` | Histograms, counters, Prometheus text format |
| `tests/observability/test_slo.py` | Rolling-window math, burn-rate formula |
| `tests/observability/test_tracing.py` | In-memory span exporter, 3-tool agent span tree |
| `tests/reliability/test_timeouts.py` | Timeout wrappers, adapter audit (0 `timeout=None` violations) |
| `tests/reliability/test_retry.py` | Jitter, backoff, Retry-After, OTel event emission |
| `tests/reliability/test_circuit.py` | CLOSED→OPEN→HALF_OPEN state machine |
| `tests/reliability/test_bulkhead.py` | Concurrency limits, queue overflow, sync + async |
| `tests/reliability/test_chaos.py` | 4 scenarios × 10 seeds, 273 tests — all deterministic |
| `tests/fuzz/test_tool_fuzz.py` | All 66 BaseTool subclasses × 500 examples, no secret leaks |
| `tests/fuzz/test_message_fuzz.py` | Random ContentPart sequences, no unhandled exceptions |
| `tests/fuzz/test_router_fuzz.py` | Random availability + capabilities → valid decision or NoEligibleProvider |
| `tests/tools/test_loadgen.py` | Loadgen library + CLI, 47 tests |
| `tests/observability/test_alerting.py` | Alert rules, AlertWebhook, 34 tests |

### Validation Results

| Check | Result |
|-------|--------|
| `effgen.__version__` | **0.2.9** |
| Every log line in `agent.run()` | JSON + no raw secrets ✓ |
| `/metrics` scrape | Prometheus-valid histograms ✓ |
| SLO burn-rate math | Spot-checked against rolling-window fixtures ✓ |
| Span tree (Calculator+WebSearch, 3-tool) | All declared attributes present ✓ |
| Timeout audit | 0 `timeout=None` violations in source ✓ |
| Breaker CLOSED→OPEN→HALF_OPEN | Verified across synthetic faults ✓ |
| Bulkhead concurrency limit | Verified sync + async ✓ |
| Chaos Scenario A–D × 10 seeds | 273/273 pass, deterministic ✓ |
| Fuzz × 500 examples per test | 164/164 pass, no unhandled exceptions ✓ |
| Load harness mock smoke (c=10, 30 s) | ~69 k req, 0% error, p95 ≈ 4.3 ms ✓ |
| Alert rules YAML | Syntactically valid, 6 rules ✓ |
| Regression suite | All prior tests pass (p=1300+, f=0) ✓ |
| Wheel build | `effgen-0.2.9-py3-none-any.whl` built cleanly ✓ |
| Wheel smoke | `python -c "import effgen; assert effgen.__version__ == '0.2.9'"` ✓ |

### Upgrading from v0.2.8

No breaking API changes. All observability and reliability features are additive.

```bash
pip install --upgrade effgen
```

#### Observability Quick Start

```python
from effgen.observability import get_logger
from effgen.observability import record_model_call, export_metrics
from effgen.observability.slo import SLOTracker, SLO

log = get_logger(__name__)
log.event("agent.started", preset="general", model="llama3.1-8b")

# Histograms auto-record on agent/model/tool calls; you can also record directly:
record_model_call(provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.42)
print(export_metrics())  # Prometheus text format

tracker = SLOTracker()
tracker.register(SLO("model_success", target_pct=99.0, window_seconds=3600))
tracker.record("model_success", ok=True)
print(tracker.burn_rate("model_success"))  # e.g. 0.0 when all calls succeed
```

#### Reliability Quick Start

```python
from effgen.reliability.retry import Retry, retryable
from effgen.reliability.circuit import CircuitBreaker
from effgen.reliability.bulkhead import Bulkhead

@retryable(Retry(max_attempts=3, base_delay=0.5, jitter=True))
def call_model(prompt):
    ...

breaker = CircuitBreaker("cerebras", failure_threshold=5, recovery_timeout=30)
if breaker.is_call_permitted():
    try:
        result = call_model("hello")
        breaker.on_success()
    except Exception as exc:
        breaker.on_failure(exc)
        raise

bulkhead = Bulkhead("cerebras", max_concurrency=10, queue_size=50)
with bulkhead.acquire():
    call_model("hello")
```

---

## [0.2.8] - 2026-05-21

### Highlights

**effGen v0.2.8** is the **Multimodal Input** release — image, audio, and video are now first-class citizens across 6 cloud providers (Gemini, OpenAI, Groq, Anthropic, Together, HF). A unified `Message` content schema, provider-specific adapters, automatic preprocessing (resize, downsample, frame-sampling), capability-gating errors, a new `multimodal` preset, `MultimodalDescribeTool`, a local MLX-VLM adapter (Apple Silicon), and 5 cookbook walkthroughs ship in this release. No breaking API changes.

### Added

#### Core — Unified Message Schema (`effgen/core/messages.py`)

- **Structured `ContentPart` union** — `TextPart`, `ImagePart`, `AudioPart`, `VideoPart`, `ToolCallPart`, `ToolResultPart` form the typed `ContentPart` union. `Message.content` is always `List[ContentPart]`.
- **Backwards-compatible constructor** — `Message(role, "text string")` auto-wraps in `TextPart`; `Message.text` property joins all text parts; `Message.from_str(text)` classmethod.
- **Validation on construction** — `ImagePart` validates MIME ∈ {image/png, image/jpeg, image/gif, image/webp}; `AudioPart` validates MIME ∈ {audio/mp3, audio/wav, audio/flac, audio/ogg, audio/m4a}; `VideoPart.frames` must be non-empty. Raises `InvalidMultimodalContent` on failure.

#### Core — Multimodal Helpers (`effgen/core/multimodal.py`)

- **`image_from(source) → ImagePart`** — accepts `bytes`, local path, URL, `PIL.Image`, `np.ndarray`. MIME sniffed automatically.
- **`audio_from(source) → AudioPart`** — accepts `bytes`, local path, URL. Duration extracted from metadata when available.
- **`video_from(source, fps=1) → VideoPart`** — accepts `bytes`, local path, URL; samples keyframes at `fps` via ffmpeg; raises `MissingSystemDependency` with install hints if ffmpeg is absent.

#### Multimodal Preprocessing

- **`effgen/multimodal/image_pre.py`** — `prepare(part, provider, model) → ImagePart`. Applies per-provider constraints (max bytes, max pixel dims, supported MIMEs); PIL Lanczos downscale when needed; records preprocessing steps in `part.meta["preprocessing"]` for observability.
- **`effgen/multimodal/audio_pre.py`** — Downsamples to 16 kHz mono if provider requires; chunks audio longer than provider max duration into sequential requests and concatenates results. Uses `pydub`.
- **`effgen/multimodal/video_pre.py`** — `VideoSource(path_or_url).sample_frames(fps, max_frames) → List[ImagePart]`; `VideoSource.extract_audio() → AudioPart | None`. Raises `MissingSystemDependency("ffmpeg", ...)` when ffmpeg is absent.

#### Provider Adapters — Image Input

- **Gemini** — native `inline_data` image parts (base64 + mime_type); all Gemini 2.x/3.x vision models.
- **OpenAI** — `content: [{type: "image_url"}]` format for gpt-4o family; base64 data-URL encoding.
- **Anthropic** — base64 media blocks in content list (code-only; live tests skipped — no key in dev env).
- **Groq** — Llama 4 / Llama 3.2 vision model support via image_url content blocks.
- **Together** — vision-capable Together models via image_url content blocks.
- **HuggingFace Inference** — BLIP / LLaVA family via multimodal inference payload.
- **Capability gating** — every adapter raises `CapabilityNotSupportedError(Capability.vision)` when the selected model doesn't support images; no silent text downcast.

#### Provider Adapters — Audio Input

- **Gemini** — native `Part.from_bytes(audio_bytes, mime_type)` inline audio; full conversation with audio context.
- **OpenAI** — Whisper via `/audio/transcriptions` (`transcribe_audio()` method) + gpt-4o audio in chat completions.
- **HuggingFace Inference** — `automatic_speech_recognition` task endpoint.
- **Anthropic** — raises `CapabilityNotSupportedError(Capability.audio_input)` (no audio support).

#### Provider Adapters — Video Input

- **Gemini** — native video inline data for Gemini 2.x/3.x; video MIME passed directly.
- **All others** — `VideoPart` converted to sequence of `ImagePart`s (frame sampling) + optional `AudioPart` (from audio track); sent as multi-image message.

#### New Preset: `multimodal` (`effgen/presets/multimodal.py`)

- **Primary model** — Gemini Flash-Lite (vision + audio + video).
- **Fallback** — OpenAI gpt-4o-mini (vision), HF BLIP (vision-only).
- **Tools** — `ImageInfoTool`, `ImageCaptionTool`, `OCRTool`, `AudioTranscribeTool`, `PDFTool`, `WeatherTool`, `MultimodalDescribeTool`.
- **`MultimodalDescribeTool`** — auto-selects between `ImageCaption`, `OCR`, and `AudioTranscribe` based on the input part type; returns structured description.
- **`create_agent("multimodal", model=...)`** — factory wires the preset end-to-end.

#### Local MLX-VLM Adapter (`effgen/models/mlx_vlm_engine.py`)

- Thin wrapper around the `mlx-vlm` library for Apple Silicon vision-language inference.
- Raises `MissingSystemDependency` on non-Apple-Silicon / missing `mlx-vlm`. Live tests skipped on Linux; fully unit-tested with fakes.

#### Cookbook (`docs/cookbook/`)

- **`multimodal_01_image_qa.md`** — image Q&A walk-through with Gemini and OpenAI.
- **`multimodal_02_audio_transcribe_reason.md`** — audio → transcript → sentiment analysis.
- **`multimodal_03_video_summarize.md`** — video → keyframes → narrative summary.
- **`multimodal_04_ocr_plus_llm.md`** — OCR text extraction then structured extraction via `contract_summarize_v1` prompt.
- **`multimodal_05_bullet_chart_read.md`** — read a bar chart from an image and answer comparison questions.
- **`docs/cookbook/README.md`** — index of all cookbook walkthroughs with quick-start links.

#### Documentation

- **`docs/multimodal/overview.md`** — unified Message schema, ContentPart types, capability gating, provider support matrix, preprocessing pipeline, and quick-start examples.
- **`docs/multimodal/images.md`** — per-provider image input guide.
- **`docs/multimodal/audio.md`** — per-provider audio input guide.
- **`docs/multimodal/video.md`** — video frame-sampling and native video path guide.

### Tests Added

| File | Coverage |
|------|----------|
| `tests/core/test_message_schema.py` | ContentPart construction, validation, back-compat |
| `tests/core/test_multimodal_helpers.py` | `image_from`, `audio_from`, `video_from` on all source types |
| `tests/core/test_image_input.py` | Adapter translation per provider (fakes) |
| `tests/core/test_audio_input.py` | Audio adapter translation per provider (fakes) |
| `tests/core/test_video_input.py` | VideoPart → ImagePart fallback; Gemini native path (fakes) |
| `tests/multimodal/test_image_pre.py` | Resize, MIME, size constraints |
| `tests/multimodal/test_audio_pre.py` | Chunking, downsample |
| `tests/multimodal/test_video_pre.py` | ffmpeg missing → clean error; frame sampling rates |
| `tests/presets/test_multimodal.py` | Preset construction, tool wiring, MultimodalDescribeTool |
| `tests/models/test_mlx_vlm.py` | MLX-VLM adapter unit tests (28 tests) |
| `tests/cookbook/test_cookbook_runs.py` | Cookbook snippet extraction, `pytest.mark.live` gating |

### Validation Results

| Check | Result |
|-------|--------|
| `effgen.__version__` | **0.2.8** |
| Image input — live | Gemini ✓, OpenAI ✓, Groq ✓ (≥3 providers) |
| Audio input — live | Gemini ✓, OpenAI Whisper ✓ |
| Video input — live | Gemini native ✓, OpenAI frame-sampling ✓ |
| Multimodal preset — live | image ✓, audio ✓, video ✓ (all 3 modalities) |
| Cookbook live runs | 7/8 pass (1 skipped — ffmpeg not installed in test env) |
| `CapabilityNotSupportedError` | Raised cleanly on vision-incapable provider/model |
| `MissingSystemDependency("ffmpeg")` | Raised with install hints when ffmpeg absent |
| Wheel build | `effgen-0.2.8-py3-none-any.whl` built cleanly |
| Wheel smoke | `python -c "import effgen; assert effgen.__version__ == '0.2.8'"` ✓ |
| Regression suite | All prior tests pass (p=2489+, f=0) |

### Upgrading from v0.2.7

No breaking API changes. The old `Message(role, content: str)` constructor still works.

```bash
pip install --upgrade effgen
```

#### Quick Start

```python
from effgen import image_from, audio_from
from effgen.core.messages import Message, Role
from effgen.presets import create_agent
from effgen import load_model

model = load_model("gemini-2.0-flash", provider="gemini")
agent = create_agent("multimodal", model)

# Image Q&A
img = image_from("https://example.com/photo.jpg")
msg = Message(role=Role.USER, content=[img, "Describe this image."])
result = agent.run_message(msg)

# Audio
aud = audio_from("/tmp/recording.mp3")
msg = Message(role=Role.USER, content=[aud, "Transcribe and give the sentiment."])
result = agent.run_message(msg)
```

---

## [0.2.7] - 2026-05-20

### Highlights

**effGen v0.2.7** is the **Prompt Library** release — a curated, domain-organized catalog of **31 reusable prompt templates** across 7 domains, paired with a golden evaluation harness, a rich CLI, and an interactive playground. No breaking API changes.

### Added

#### Prompt Library (`effgen/prompts/library/`)

- **`LibraryPrompt` dataclass** (`base.py`) — structured prompt definition with `name`, `domain`, `variant`, `description`, `template` (callable), `input_schema` (JSON Schema), `fixture`, `expected_shape`, and `tags`. Fully validated on registration.
- **`PromptRegistry` singleton** (`registry.py`) — auto-discovers all domain packages under `effgen/prompts/library/domains/` at startup; `register`, `get`, `search`, `all`, `domains`, `__len__`.
- **`PromptEval` harness** (`eval.py`) — `eval_golden` (renders with fixture, compares against `.txt` golden, writes on first run); `eval_live` (renders + runs via model, checks `expected_shape`); `eval_all_golden` with pass/fail table.
- **CLI** — `effgen prompts list [--domain X] [--variant Y] [--format table|json|markdown]`, `effgen prompts show <name>`, `effgen prompts eval [--domain X] [--live --model M]`.

#### Research Domain (`effgen/prompts/library/domains/research/`)

- **`research.literature_review.v1.zero_shot`** — zero-shot literature review; inputs: `topic`, `years_range`, `max_papers`.
- **`research.literature_review.v1.cot`** — chain-of-thought literature review with step-by-step reasoning.
- **`research.paper_summary.v1`** — structured output: `{abstract_summary, key_findings, limitations, future_work}`.
- **`research.citation_extract.v1`** — tool-augmented; instructs agent to retrieve live ArXiv/PubMed metadata.
- **`research.methodology_critique.v1`** — CoT critique covering design, sampling, measurement, analysis, generalizability.

#### Coding Domain (`effgen/prompts/library/domains/coding/`)

- **`coding.code_review.v1`** — structured output: `{issues: [{severity, location, suggestion}]}`.
- **`coding.bug_diagnose.v1`** — CoT diagnosis; inputs: `code`, `error_message`, `repro_steps`.
- **`coding.refactor_plan.v1`** — tool-augmented; reads the source file then produces a structured plan with risk assessment.
- **`coding.test_generate.v1`** — few-shot; two exemplar pytest suites; live eval asserts `ast.parse()` passes on generated Python.
- **`coding.docstring_fill.v1`** — zero-shot; adds Google/NumPy/Sphinx-style docstrings to undocumented functions.

#### Data Domain (`effgen/prompts/library/domains/data/`)

- **`data.sql_from_nl.v1`** — structured output: `{sql, warnings[]}`; inputs: `schema_ddl`, `question`, `dialect`; live eval validates via `sqlglot.parse()`.
- **`data.sql_explain.v1`** — zero-shot; explains SQL in plain English for developer or business audience.
- **`data.sql_optimize.v1`** — CoT; identifies anti-patterns, explains execution impact, produces rewritten query and index hints.
- **`data.data_profile.v1`** — tool-augmented; takes ExcelTool/CSV column stats, produces structured data-quality report.
- **`data.etl_plan.v1`** — few-shot; two exemplar ETL designs covering Extract → Transform → Load → Validate → Cleanup.

#### Legal Domain (`effgen/prompts/library/domains/legal/`)

> All legal prompts include the verbatim disclaimer: *"This output is for informational purposes only and does not constitute legal advice. Consult a qualified attorney for guidance specific to your situation."*

- **`legal.contract_summarize.v1`** — structured output: `{parties, term, obligations, termination, risks}`.
- **`legal.clause_classify.v1`** — zero-shot clause classification with characteristic flags.
- **`legal.legal_research_brief.v1`** — tool-augmented; produces structured research brief grounded in pre-retrieved sources.

#### Medical Domain (`effgen/prompts/library/domains/medical/`)

> All medical prompts include the verbatim disclaimer: *"This output is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional."*

- **`medical.symptom_triage.v1`** — structured output with mandatory `disclaimer` field and `see_doctor_if` list.
- **`medical.drug_interaction_query.v1`** — structured output with severity levels and recommendations.
- **`medical.medical_literature.v1`** — tool-augmented; synthesizes retrieved PubMed abstracts into a clinical evidence brief.

#### Creative Domain (`effgen/prompts/library/domains/creative/`)

- **`creative.story_continuation.v1.zero_shot`** — zero-shot story continuation maintaining genre and tone.
- **`creative.story_continuation.v1.few_shot`** — few-shot with craft exemplars from multiple genres.
- **`creative.poetry_forms.v1`** — few-shot with exemplars for haiku, sonnet, and free verse; inputs: `theme`, `form`, `mood`.
- **`creative.character_bio.v1`** — structured output: `{name, age, background, personality_traits, goals, flaws, relationships}`.
- **`creative.world_building.v1`** — CoT; develops geography, politics, magic/tech, culture, and story hooks step by step.

#### Business Domain (`effgen/prompts/library/domains/business/`)

- **`business.meeting_summary.v1`** — structured output: `{decisions, action_items[{owner, item, due}], risks}`; inputs: `transcript`, `meeting_title`, `attendees`.
- **`business.email_draft.v1`** — few-shot; two tone exemplars (formal, casual); inputs: `purpose`, `recipient`, `key_points`, `tone`.
- **`business.okr_generate.v1`** — CoT; produces aligned objectives and measurable key results from mission and strategic priorities.
- **`business.swot_analysis.v1`** — structured output: `{strengths, weaknesses, opportunities, threats, strategic_insights}`; perspective-aware.
- **`business.elevator_pitch.v1`** — zero-shot; strict ≤150-word constraint; live eval asserts word count.

#### Playground CLI (`effgen prompts playground`)

- **Interactive REPL** — `select`, `set`, `render`, `run`, `save`, `list`, `show`, `help`, `quit` commands.
- **Non-interactive mode** — `effgen prompts render <name> [--input input.json]` and `effgen prompts run <name> [--input input.json] [--model M]`.
- **Session persistence** — sessions saved to `~/.effgen/playground/<timestamp>.json`; `effgen prompts playground --load <session>` reloads.
- **Hot-reload** — template edits are picked up without REPL restart (importlib-based re-import).

#### Gallery Doc

- **`docs/prompts/gallery.md`** — auto-generated from registry; one row per template with name, domain, variant, and description. Regenerate with `effgen prompts list --format markdown`.

### Tests

- `tests/prompts/test_registry.py` — discovery, search, validation.
- `tests/prompts/test_eval.py` — golden and live eval harness.
- `tests/prompts/test_research.py`, `test_coding.py`, `test_data.py`, `test_legal.py`, `test_medical.py`, `test_creative.py`, `test_business.py` — domain golden + live checks.
- `tests/prompts/test_playground.py` — scripted non-interactive walk-through.

### Documentation

- `docs/prompts/library.md` — framework overview, key classes, CLI reference, adding-new-domain guide.
- `docs/prompts/research.md`, `coding.md`, `data.md`, `legal.md`, `medical.md`, `creative.md`, `business.md`, `playground.md` — per-domain guides.

---

## [0.2.6] - 2026-05-19

### Highlights

**effGen v0.2.6** is a document, media, and communication tools release adding **14 new built-in tools** across six categories — OCR, audio transcription, image analysis, document parsing, geo/weather, and email/webhook communication — raising the total built-in tool count from 44 to **58+**. Two new presets (`media`, `notify`) are introduced. Every tool follows the established `BaseTool` pattern with structured `{success, data, error}` output, async `_execute()`, unit + integration tests, a dedicated doc page, and preset integration. No breaking API changes.

### Added

#### OCR Tools
- **`OCRTool`** (`effgen/tools/builtin/ocr.py`) — extract text from images using Tesseract (local, primary) with OCR.space free API as fallback (`OCR_SPACE_API_KEY`). Raises `OCRBackendUnavailable` with per-OS install instructions when no backend is available. Operations: `extract`, `extract_regions`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.ocr import OCRTool
  result = OCRTool().execute({"operation": "extract", "image_path": "/tmp/scan.png", "lang": "eng"})
  print(result["data"]["text"])
  ```

  **System dep install:**
  ```bash
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  # macOS
  brew install tesseract
  # Windows
  choco install tesseract
  ```

#### Audio Transcription Tools
- **`AudioTranscribeTool`** (`effgen/tools/builtin/audio_transcribe.py`) — transcribe audio files locally via `faster-whisper` (CPU/GPU auto-detected) with HuggingFace Inference fallback (`HF_TOKEN`). Detects GPU via `nvidia-smi`; warns when `model_size > "base"` on CPU. Operations: `transcribe`. Added to `media` preset.

  ```python
  from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
  result = AudioTranscribeTool().execute({"operation": "transcribe", "audio_path": "/tmp/clip.mp3", "model_size": "base"})
  print(result["data"]["text"])
  ```

  **System dep install (for non-WAV formats):**
  ```bash
  sudo apt-get install ffmpeg   # Ubuntu/Debian
  brew install ffmpeg           # macOS
  ```

#### Image Analysis Tools
- **`ImageInfoTool`** (`effgen/tools/builtin/image_info.py`) — extract image metadata (size, format, mode, EXIF, color histogram) and perform local resize/thumbnail operations using Pillow. Zero network calls. Operations: `info`, `resize`, `thumbnail`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.image_info import ImageInfoTool
  result = ImageInfoTool().execute({"operation": "info", "image_path": "/tmp/photo.jpg"})
  print(result["data"]["size"], result["data"]["format"])
  ```

- **`ImageCaptionTool`** (`effgen/tools/builtin/image_caption.py`) — generate natural-language descriptions of images via the effGen model router (selects a vision-capable provider: Gemini, OpenAI, or MLX-VLM). Raises `NoVisionProviderAvailable` when no vision-capable provider is configured. Operations: `caption`, `describe`. Added to `media` preset.

  ```python
  from effgen.tools.builtin.image_caption import ImageCaptionTool
  result = ImageCaptionTool().execute({"operation": "caption", "image_path": "/tmp/photo.jpg"})
  print(result["data"]["caption"])
  ```

#### Document Parsing Tools
- **`PDFTool`** (`effgen/tools/builtin/pdf.py`) — extract text, tables, and metadata from PDF files using `pypdf` (primary) with `pdfplumber` for structured table extraction. Operations: `text`, `metadata`, `tables`, `extract_images`. Added to `research` and `general` presets.

  ```python
  from effgen.tools.builtin.pdf import PDFTool
  result = PDFTool().execute({"operation": "text", "path": "/tmp/paper.pdf"})
  print(result["data"]["text"][:500])
  ```

- **`DOCXTool`** (`effgen/tools/builtin/docx.py`) — parse Word documents (`.docx`) using `python-docx`. Operations: `text`, `paragraphs`, `tables`, `metadata`. Added to `research` and `general` presets.

  ```python
  from effgen.tools.builtin.docx import DOCXTool
  result = DOCXTool().execute({"operation": "text", "path": "/tmp/report.docx"})
  print(result["data"]["text"])
  ```

- **`ExcelTool`** (`effgen/tools/builtin/excel.py`) — read Excel workbooks (`.xlsx`) using `openpyxl` with tabular DataFrame output via `pandas`. Operations: `sheets`, `read_sheet`, `headers`. Added to `research` and `general` presets.

  ```python
  from effgen.tools.builtin.excel import ExcelTool
  result = ExcelTool().execute({"operation": "read_sheet", "path": "/tmp/data.xlsx", "sheet_name": "Sheet1"})
  print(result["data"]["rows"][:3])
  ```

#### Geo / Weather Tools
- **`WeatherTool`** (`effgen/tools/builtin/weather.py`) — fetch current conditions, forecasts, and historical weather data from Open-Meteo (free, no auth required). Integrates with `GeocodeTool` for place-name → lat/lon resolution. Operations: `current`, `forecast`, `historical`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.weather import WeatherTool
  result = WeatherTool().execute({"operation": "current", "lat": 37.42, "lon": -122.08})
  print(result["data"]["temperature_c"], result["data"]["weather_description"])
  ```

- **`GeocodeTool`** (`effgen/tools/builtin/geocode.py`) — forward/reverse geocoding using Nominatim (OpenStreetMap). Sets `effGen/<version>` User-Agent as required; built-in 1 req/s token-bucket rate limiter. Operations: `geocode`, `reverse`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.geocode import GeocodeTool
  result = GeocodeTool().execute({"operation": "geocode", "address": "1600 Amphitheatre Pkwy, Mountain View, CA"})
  print(result["data"]["lat"], result["data"]["lon"])
  ```

- **`MapsTool`** (`effgen/tools/builtin/maps.py`) — render static PNG maps from OpenStreetMap tiles using the `staticmap` library. Operations: `render`, `bounding_box`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.maps import MapsTool
  result = MapsTool().execute({"operation": "render", "lat": 37.42, "lon": -122.08, "zoom": 13, "dest": "/tmp/map.png"})
  print(result["data"]["path"])
  ```

#### Email Tools
- **`EmailSMTPTool`** (`effgen/tools/builtin/email_smtp.py`) — send email via SMTP using stdlib `smtplib`. TLS-on by default. Config: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`. Raises `MissingCredentialsError` when config is absent. Operations: `send`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.email_smtp import EmailSMTPTool
  result = EmailSMTPTool().execute({"operation": "send", "to": "alice@example.com", "subject": "Hello", "body": "Hi there!"})
  ```

- **`EmailIMAPTool`** (`effgen/tools/builtin/email_imap.py`) — read email via IMAP using stdlib `imaplib`. Config: `IMAP_HOST`, `IMAP_PORT`, `IMAP_USER`, `IMAP_PASSWORD`. Operations: `list_folders`, `fetch_recent`, `search`, `get`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.email_imap import EmailIMAPTool
  result = EmailIMAPTool().execute({"operation": "fetch_recent", "folder": "INBOX", "n": 5})
  for msg in result["data"]["messages"]:
      print(msg["subject"], msg["from"])
  ```

#### Webhook Tools
- **`SlackWebhookTool`** (`effgen/tools/builtin/slack_webhook.py`) — post messages to Slack via incoming webhook URL (no OAuth required). Config: `SLACK_WEBHOOK_URL`. URL is redacted in all logs. Operations: `post`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.slack_webhook import SlackWebhookTool
  result = SlackWebhookTool().execute({"operation": "post", "text": "Deploy complete!"})
  ```

- **`DiscordWebhookTool`** (`effgen/tools/builtin/discord_webhook.py`) — post messages to Discord via webhook URL. Config: `DISCORD_WEBHOOK_URL`. URL is redacted in all logs. Operations: `post`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.discord_webhook import DiscordWebhookTool
  result = DiscordWebhookTool().execute({"operation": "post", "content": "Deployment succeeded!"})
  ```

#### New Presets
- **`media` preset** — bundles `AudioTranscribeTool` and `ImageCaptionTool` for media-processing agents.
- **`notify` preset** — bundles `EmailSMTPTool`, `EmailIMAPTool`, `SlackWebhookTool`, and `DiscordWebhookTool` for notification/alert agents.

#### Documentation
- **`docs/tools/gallery.md`** — updated with all 14 new tools (OCR, AudioTranscribe, ImageInfo, ImageCaption, PDF, DOCX, Excel, Weather, Geocode, Maps, EmailSMTP, EmailIMAP, SlackWebhook, DiscordWebhook).
- **`docs/tools/ocr.md`** — OCRTool reference with per-OS Tesseract install instructions.
- **`docs/tools/audio_transcribe.md`** — AudioTranscribeTool reference with ffmpeg install notes.
- **`docs/tools/image.md`** — ImageInfoTool + ImageCaptionTool reference.
- **`docs/tools/documents.md`** — PDFTool + DOCXTool + ExcelTool reference.
- **`docs/tools/weather.md`** — WeatherTool reference.
- **`docs/tools/geocode.md`** — GeocodeTool reference.
- **`docs/tools/maps.md`** — MapsTool reference.
- **`docs/tools/email.md`** — EmailSMTPTool + EmailIMAPTool reference.
- **`docs/tools/webhooks.md`** — SlackWebhookTool + DiscordWebhookTool reference (with security note: webhook URLs are secrets).

### Changed
- **`general` preset** — now includes OCRTool, ImageInfoTool, PDFTool, DOCXTool, ExcelTool, WeatherTool, GeocodeTool, MapsTool, EmailSMTPTool, EmailIMAPTool, SlackWebhookTool, DiscordWebhookTool in addition to existing tools.
- **`research` preset** — now includes PDFTool, DOCXTool, ExcelTool for document parsing alongside existing academic/web tools.
- **`effgen/__init__.py`** — version bumped to `0.2.6`.

### New Errors
- **`OCRBackendUnavailable`** — raised when neither Tesseract nor OCR.space is available; includes per-OS install instructions.
- **`MissingSystemDependency`** — raised by audio/document tools when a required system binary (ffmpeg, tesseract) is absent.
- **`NoVisionProviderAvailable`** — raised by `ImageCaptionTool` when no vision-capable provider is configured.
- **`MissingCredentialsError`** — raised by email/webhook tools when required env vars are absent.
- **`CorruptDocumentError`** — raised by PDF/DOCX/Excel tools on unreadable files.

---

## [0.2.5] - 2026-05-18

### Highlights

**effGen v0.2.5** adds **13 new free/no-auth tools** spanning academic research, news & RSS, YouTube, social media, translation, language detection, and QR codes — bringing the total built-in tool count to **44+**. All tools are `BaseTool` subclasses with structured `{success, data, error}` output, integrated into the `research` and `general` presets, and covered by unit + integration tests.

### Added

#### Academic Research Tools
- **`PubMedTool`** (`effgen/tools/builtin/pubmed.py`) — search PubMed via NCBI E-utilities, fetch article metadata, retrieve abstracts. Operations: `search`, `fetch`, `abstract`. Built-in token-bucket rate limiter (3 req/s without key, 10/s with `NCBI_API_KEY`). Added to `research` preset.
- **`ArXivTool`** (`effgen/tools/builtin/arxiv.py`) — search arXiv Atom feed, fetch paper metadata by ID, download PDFs. Operations: `search`, `fetch`, `download_pdf`. Added to `research` preset.
- **`SemanticScholarTool`** (`effgen/tools/builtin/semantic_scholar.py`) — search papers, fetch paper details, retrieve citations and references via Semantic Scholar Graph API. Operations: `search`, `paper`, `citations`, `references`. Built-in backoff (100 req/5 min unauth). Added to `research` preset.

#### News & RSS Tools
- **`RSSFeedTool`** (`effgen/tools/builtin/rss.py`) — fetch, browse, and full-text search any RSS/Atom feed by URL. Operations: `fetch`, `latest`, `search_in_feed`. Handles malformed feeds gracefully. Added to `research` and `general` presets.
- **`NewsTool`** (`effgen/tools/builtin/news.py`) — aggregate top headlines and search news across curated reputable RSS sources (Reuters, BBC, HN, NPR, etc.); optional `NEWS_API_KEY` for NewsAPI.org. Operations: `top_headlines`, `search`. Added to `research` and `general` presets.

#### YouTube Tools
- **`YouTubeTranscriptTool`** (`effgen/tools/builtin/youtube_transcript.py`) — fetch YouTube captions/transcripts without a Google API key via `youtube-transcript-api`. Operations: `get_transcript`, `list_available_languages`, `translated`. Handles watch?v=, youtu.be/, and shorts/ URL formats. Added to `research` preset.
- **`YouTubeMetadataTool`** (`effgen/tools/builtin/youtube_metadata.py`) — fetch video/channel metadata using yt-dlp in metadata-only mode. Operations: `metadata`, `channel`. No auth required for public content. Added to `research` preset.

#### Social Media Tools
- **`RedditTool`** (`effgen/tools/builtin/reddit.py`) — access Reddit top/hot posts, user submissions, and thread comments via public JSON endpoints (no OAuth for reads). Operations: `subreddit_top`, `subreddit_hot`, `user_submissions`, `thread_comments`. Sets `effGen/<version>` User-Agent; exponential backoff on 429. Added to `research` and `general` presets.
- **`HackerNewsTool`** (`effgen/tools/builtin/hackernews.py`) — fetch top/new stories, story details, and user profiles from HN Firebase API. Operations: `top_stories`, `new_stories`, `story`, `user`. No auth required. Added to `research` and `general` presets.

#### Translation & Language Detection Tools
- **`TranslateTool`** (`effgen/tools/builtin/translate.py`) — translate text between languages with LibreTranslate as primary backend (configurable via `LIBRE_TRANSLATE_URL`) and `argostranslate` as an offline fallback. Operations: `translate`, `available_pairs`. Language pack cache at `~/.effgen/argos/`. Added to `general` preset.
- **`LanguageDetectTool`** (`effgen/tools/builtin/language_detect.py`) — detect language of text or a batch of texts, fully offline via `langdetect` (55+ languages). Operations: `detect`, `detect_batch`. Added to `general` preset.

#### QR Code Tools
- **`QRGenerateTool`** (`effgen/tools/builtin/qr_generate.py`) — generate QR codes locally from any text or URL; returns base64 PNG or file path. Operations: `generate`. Supports `data_url_return=True` for inline embedding. No network required. Added to `general` preset.
- **`QRReadTool`** (`effgen/tools/builtin/qr_read.py`) — decode QR codes and barcodes from image files or base64 PNG using `pyzbar` + Pillow, with OpenCV QR fallback when `libzbar` is unavailable. Operations: `read`. Fully local. Added to `general` preset.

#### Documentation
- **`docs/tools/gallery.md`** — tool gallery with one-line description and quickstart snippet for every built-in tool (all 44+).
- **`docs/tools/index.md`** — updated with all 13 new tools.
- Per-tool docs: `pubmed.md`, `arxiv.md`, `semantic_scholar.md`, `rss.md`, `news.md`, `youtube.md`, `reddit.md`, `hackernews.md`, `translate.md`, `language_detect.md`, `qr.md`.

### Changed
- **Preset registry** — `research` preset now includes PubMed, ArXiv, SemanticScholar, RSS, News, YouTubeTranscript, YouTubeMetadata, Reddit, HackerNews tools. `general` preset now includes RSS, News, Reddit, HackerNews, Translate, LanguageDetect, QRGenerate, QRRead tools.
- **`effgen/__init__.py`** — version bumped to `0.2.5`.

---

## [0.2.4] - 2026-05-14

### Highlights

**effGen v0.2.4** introduces a production-ready **ModelRouter** with three composable routing policies (FirstAvailable, CostBased, LatencyBased), transparent provider failover with retry logic, persisted cross-process rate-limit coordination via SQLite, and a persistent cost tracker with a `effgen cost` CLI dashboard.

### Added

#### ModelRouter + Routing Policies
- **`PolicyBasedRouter`** (`effgen/models/router.py`) — composable policy engine; `route(context)` returns an explainable `RouterDecision` recording which providers were eliminated and why; `route_and_execute(context, fn)` wraps any callable with transparent failover across `failover_hops` (default 3).
- **`RoutingPolicy` ABC** — base class for all policies; implement `select(candidates, context) → RouterDecision`.
- **`RoutingContext`** — carries `prompt_tokens_estimate`, `user_budget_usd`, `latency_budget_ms`, `required_capabilities`.
- **`RouterDecision`** — records `chosen`, `eliminated` (with per-provider reasons), `policy_name`, and `score`. Every routing decision is fully explainable.
- **`RouterEvent`** — emitted on failover; subscribers register via `PolicyBasedRouter.subscribe(callback)`.
- **`FirstAvailablePolicy`** (`effgen/models/routing/first_available.py`) — returns the first provider with a valid API key that meets the required capabilities.
- **`CostBasedPolicy`** (`effgen/models/routing/cost.py`) — estimates cost per call from pricing registry; ranks cheapest-first; free-tier providers rank ahead of equally-priced paid providers; raises `NoCandidateWithinBudgetError` when no candidate fits `user_budget_usd`.
- **`LatencyBasedPolicy`** (`effgen/models/routing/latency.py`) — picks the fastest provider by observed p50 latency; eliminates candidates exceeding `latency_budget_ms`; warm-up probe seeds empty-history tiebreaks.
- **`RetryPolicy`** (`effgen/models/routing/retry.py`) — configurable `max_retries`, exponential backoff with jitter; retries `RateLimitExceeded`, `ProviderTransientError`, `ModelTimeoutError`; does **not** retry `ModelAuthError`, `ModelRefusalError`, `InvalidRequestError`.

#### Capability Model
- **`Capability` enum** (`effgen/models/capabilities.py`) — `{chat, tools, streaming, vision, grounding, thinking, json_schema}`; all 9 adapters register their capability sets in `ProviderRegistry`.
- **`ProviderRegistry.register(..., capabilities=..., pricing=...)`** — extended with capability and pricing fields; all 9 adapters updated with current published pricing (2026-05-14).

#### Latency Tracker
- **`LatencyTracker`** (`effgen/models/latency_tracker.py`) — rolling window (last 50 calls) per `(provider, model)`; records total latency and time-to-first-token (TTFT); `p50(provider, model)` returns `float | None`; all 9 adapters instrumented.

#### Cross-Process Rate-Limit Coordination (SQLite)
- **`SQLiteRateLimitStore`** (`effgen/models/_rate_limit_store.py`) — WAL-mode SQLite at `~/.effgen/rate_limits.sqlite`; `BEGIN IMMEDIATE` row-locking prevents double-spend across processes; schema: `rate_events(provider, model, kind, timestamp, tokens)`.
- **`RateLimitCoordinator`** gains `storage=` parameter (default in-memory for back-compat); pass `SQLiteRateLimitStore` for cross-process coordination.
- Background housekeeping removes events older than 24 h × 1.1.

#### Cost Tracker Persistence + Dashboard
- **`SQLiteCostStore`** (`effgen/models/_cost_store.py`) — WAL-mode SQLite at `~/.effgen/costs.sqlite`; schema: `cost_events(provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp)`.
- **`CostTracker`** gains `storage=` parameter (default in-memory for back-compat); every `record()` writes a row; 80% daily budget → warning; 100% → `BudgetExceededError`.
- **`effgen cost` CLI** — `today`, `week`, `by-provider`, `set-budget`, `clear-budget` subcommands; rich table output.
- **`effgen config set budget.daily <USD>`** — configure daily spend cap.

#### New Errors
- `AllCandidatesExhaustedError` — raised when failover exhausts all providers.
- `BudgetExceededError` — raised when cumulative spend exceeds the configured cap; `RetryPolicy` treats this as retriable (failover to free tier).
- `ProviderTransientError` — base class for 5xx / transient provider failures.
- `InvalidRequestError` — bad request (4xx); not retried.

### Changed
- **`RateLimitCoordinator`** — gains `storage: RateLimitStore` parameter; default behavior (in-memory) unchanged.
- **`CostTracker`** — gains `storage: CostStore` parameter; default behavior (in-memory) unchanged.
- **`ProviderRegistry.register()`** — extended with `capabilities=set[Capability]` and `pricing=dict` optional fields; existing callers unaffected.
- **All 9 adapters** — instrumented with `LatencyTracker.record()` on every `generate()` call; TTFT recorded in streaming adapters on first yielded chunk.
- **Top-level `effgen` namespace** — new exports: `PolicyBasedRouter`, `RoutingPolicy`, `RoutingContext`, `RouterDecision`, `RouterEvent`, `ProviderModelPair`, `FirstAvailablePolicy`, `CostBasedPolicy`, `LatencyBasedPolicy`, `RetryPolicy`, `LatencyTracker`, `CostTracker`, `SQLiteCostStore`, `AllCandidatesExhaustedError`, `BudgetExceededError`, `ProviderTransientError`, `InvalidRequestError`.

### Fixed
- **Stability sweep** — all pre-existing ruff / mypy warnings resolved at the v0.2.3 baseline.
- **Back-compat guarantee** — `load_model(...)`, `Agent(config)`, direct adapter paths all unaffected by the router layer.

---

## [0.2.3] - 2026-05-04

### Highlights

**effGen v0.2.3** expands the provider ecosystem from 4 to **9 inference backends** — adding Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference — each with full streaming, native tool-calling, cost tracking, and rate-limit coordination. A unified `ProviderRegistry` consolidates all adapters for first-class introspection, and a backend parity matrix proves cross-provider correctness on a canonical agentic task.

### Added

#### New Backends (5 providers)
- **`GroqAdapter`** — 16 chat models (llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b, gemma2-9b-it, qwen3-32b, and the full free-tier roster); native tool-calling; streaming with timestamp-verified chunk delivery; RPM/RPD/TPM/TPD rate-limit windows; free-tier `CostTracker` ($0). `pip install effgen[groq]`.
- **`TogetherAdapter`** — 163-model catalog (149 chat + 13 language + 1 embedding) with live `refresh_models()` drift detection; native tools; streaming; per-model pricing. `pip install effgen[together]`.
- **`FireworksAdapter`** — 80 chat models (54 tool-capable); live catalog via `refresh_models()`; OpenAI-compatible interface; streaming; per-model pricing. `pip install effgen[fireworks]`.
- **`ReplicateAdapter`** — 38 models (25 tool-capable, 34 streaming); async run-then-poll with exponential backoff; SSE streaming; configurable prediction timeout (default 300 s); `ModelTimeoutError` with prediction cancellation; `compute_seconds` in metadata. `pip install effgen[replicate]`.
- **`HFInferenceAdapter`** — 124-model dynamic registry from HuggingFace Router (refresh + drift detection + `~/.effgen/cache` hot-reload); `chat_completion` + `text_generation` + streaming + native tools; custom Inference Endpoint URL support; `ModelUnavailableError` with `suggest_alternatives()`; `ModelNotFoundError`. `pip install effgen[hf]`.

#### Unified ProviderRegistry + Auth
- **`ProviderRegistry`** (`effgen/models/registry.py`) — singleton with `register`, `list_providers`, `list_models`, `lookup`; handles duplicate model IDs across providers; `AmbiguousModelError` on bare ambiguous IDs.
- **Adapter self-registration** — all 9 adapters register on import; idempotent.
- **`check_keys()`** (`effgen/models/auth.py`) — `provider → {available, env_key, env_keys_checked}` map.
- **`effgen doctor`** CLI command — prints provider auth table; `--json` and `--provider` filter flags; loads `.env` from `~/.effgen/.env` + project root.

#### Backend Parity Matrix
- **`tests/integration/parity/canonical_task.py`** — shared Calculator + ReAct task: "What is (17 × 23) + sqrt(144)?" Expected answer: 403.
- **`tests/integration/parity/test_backend_parity.py`** — parametrized across (provider, model) pairs; per-parametrization skip on missing key.
- **Parity reports** — `outputs/7-parity-matrix.md` (7/8 providers correct; Anthropic=no key; Replicate=billing), `outputs/7-stream-parity.md` (7/7 providers streaming), `outputs/7-error-parity.md` (9/9 providers raise `ModelAuthError`).
- **`docs/providers/parity.md`** — full provider capability table.

### Changed
- **`load_model`** now uses `ProviderRegistry` for `provider:model_id` prefix parsing; existing per-provider branches unchanged (no behavior change for callers).
- **Provider table in README expanded** to 9 providers with `effgen[groq]` / `[together]` / `[fireworks]` / `[replicate]` / `[hf]` install instructions.

### Fixed
- **`cli.py`** — `BatchConfig` variable reference from the stability sweep.
- **`aggregation.py`** — `sources` variable shadowing.
- **`CostTracker._rate`** — Fireworks pricing path added.
- **`GroqAdapter.supports_tool_calling()`** — was missing; added.
- **`ModelAuthError`** — unified across all 9 adapters for consistent error surface.
- **`AmbiguousModelError`** raised by `ModelLoader` on bare IDs shared across multiple providers (previously fell through to HF download path).

---

## [0.2.2] - 2026-04-28

### Highlights

**effGen v0.2.2** expands the **Gemini** adapter with the latest model families (3.x, 2.5, 2.0, Gemma 3/4), `thinking_budget`, Google Search grounding, the Files API, and three Gemini-native tools (`GoogleSearchTool`, `GeminiUrlContextTool`, `GeminiCodeExecutionTool`). It also modernizes the **Anthropic** adapter — Claude 4.7 / 4.x registry, extended thinking, prompt caching via `cache_control`, streaming polish, and experimental native tools — implemented and unit-tested; Anthropic live tests are skipped (no key in dev env).

### Added

#### Gemini — New Model Families
- **Expanded model registry** (`effgen/models/gemini_models.py`) — Gemini 3.1-flash-lite, 3.0-pro, 2.5-flash, 2.5-pro, 2.0-flash, Gemma 3 and Gemma 4 families; `available_models()`, `free_tier_models()`, `recommended_models()`, `model_info()` helpers; context/output/feature flags per model
- **Migrated SDK** from `google.generativeai` (legacy) to `google.genai` (`google-genai>=1.0.0`)

#### Gemini — Thinking
- **`GenerationConfig.thinking_budget: int | None`** — pass tokens for Gemini's internal reasoning; wired through `ThinkingConfig` in the adapter
- **`GenerationConfig.include_thoughts: bool`** (default `False`) — surface thinking trace in `ModelResponse.metadata["thinking"]`

#### Gemini — Grounding
- **`GenerationConfig.grounding: bool`** (default `False`) — injects Google Search tool when model supports it; grounding attributions surfaced in `ModelResponse.metadata["grounding_chunks"]`

#### Gemini — File Upload
- **`effgen.models.gemini_files.upload_file(path) → FileRef`** — wraps `genai` Files API; 2 GiB pre-upload guard; accepts `FileRef` objects in `generate(prompt, files=[...])`

#### Gemini — Native Tools
- **`GoogleSearchTool`** — activates Gemini's built-in search; `ToolIncompatibleError` at Agent init with non-Gemini model
- **`GeminiUrlContextTool`** — server-side URL content fetching
- **`GeminiCodeExecutionTool`** — server-side Python execution; output surfaced in `generated_text`
- All three in `effgen/tools/builtin/gemini_native.py`; re-exported from `effgen`
- Parallel function calls handled: adapter encodes all parallel `functionCall` parts; `metadata["tool_calls"]` is a list

#### Anthropic — Claude 4.x Registry
- **`effgen/models/anthropic_models.py`** — claude-opus-4-7 (1M ctx / 128K out), claude-sonnet-4-6 (1M ctx / 64K out), claude-haiku-4-5, legacy 4.x and 3.x lineup; `supports_thinking`, `supports_native_tools`, `supports_prompt_caching`, pricing fields

#### Anthropic — Extended Thinking
- **`GenerationConfig.thinking: dict | None`** — accepts `{"type": "enabled", "budget_tokens": N}`; temperature is forced to 1.0 when thinking is active
- Thinking trace surfaced in `ModelResponse.metadata["thinking"]`
- **`redacted_thinking` multi-turn preservation** — `raw_content_blocks` in metadata; `build_assistant_message()` helper re-inserts redacted blocks on next turn

#### Anthropic — Prompt Caching
- **`effgen/models/anthropic_cache.py`** — `mark_cached(block, ttl="5m"|"1h")`, `apply_cache_to_system()`, `apply_cache_to_tools()`, `validate_breakpoint_count()` (max 4; raises `ValueError` on 5th)
- **`AgentConfig.cache_system_prompt: bool = True`** — auto-inserts `cache_control` on the system prompt's final block
- **`AgentConfig.cache_tools: bool = True`** — auto-inserts `cache_control` on the last tool spec
- **Usage surfacing** — `ModelResponse.usage.cached_input_tokens` + `ModelResponse.usage.cache_creation_tokens`

#### Anthropic — Streaming + Native Tools
- **`generate_stream_full()`** — returns typed `StreamChunk` objects; thinking, tool-use, redacted-thinking, and text deltas all handled; parallel `tool_use` blocks accumulated per-index
- **Experimental native tools** — `AnthropicBashTool`, `AnthropicTextEditorTool`, `AnthropicComputerTool` in `effgen/tools/builtin/anthropic_native.py`; `IS_ANTHROPIC_NATIVE` sentinel; `ToolIncompatibleError` at Agent init with non-Anthropic model; marked `experimental=True`

### Changed
- **`GenerationConfig`** gains `thinking_budget`, `include_thoughts`, `grounding`, `thinking` (all `None`/`False` by default — fully back-compat)
- **`AgentConfig`** gains `cache_system_prompt: bool = True`, `cache_tools: bool = True` (additive, safe defaults)
- **Gemini adapter** migrated to `google-genai` SDK; existing `GeminiAdapter` public API unchanged

### Fixed
- **`cli.py`** — `ToolMetadata.input_schema` missing field from the stability sweep
- **`aggregation.py`** — `sources` variable shadowing / redefinition from the stability sweep
- **Gemini mixed native + function-calling** — `tool_config.include_server_side_tool_invocations=True` set when mixing built-in and user-defined tools (Gemini API requirement)
- **Anthropic `top_k`** — removed unsupported parameter that caused 400 errors
- **Gemini model aliases** — short aliases (e.g. `gemini-3.1-flash-lite`) now resolve to canonical registry IDs (e.g. `gemini-3.1-flash-lite-preview`) at `GeminiAdapter` init, so API calls succeed when callers pass the friendly short form

---

## [0.2.1] - 2026-04-25

### Highlights

**effGen v0.2.1** adds the **Cerebras** inference backend (4 free-tier models with streaming, native tool-calling, and cost tracking) and modernizes the **OpenAI** adapter (gpt-5/gpt-5.4-nano + o-series reasoning models, `reasoning_effort`, prompt caching surfacing, structured outputs v2, and OpenAI native tools — web_search, code_interpreter, file_search).

### Added

#### Cerebras Backend (new provider)
- **`CerebrasAdapter`** — full async adapter on top of `cerebras-cloud-sdk>=1.0`; `load`/`generate`/`generate_stream`/`generate_with_tools`/`unload`; OpenAI-compatible message format
- **All 4 free-tier models** in `effgen.models.cerebras_models` — `gpt-oss-120b`, `llama3.1-8b`, `qwen-3-235b-a22b-instruct-2507`, `zai-glm-4.7`; `available_models()`, `free_tier_models()`, `model_info()` helpers
- **`RateLimitCoordinator`** (`effgen/models/_rate_limit.py`) — sliding-window per-(provider,model) RPM/RPH/RPD + TPM/TPH/TPD throttling; `asyncio.Lock`-guarded; raises `RateLimitExceeded` on daily-budget exhaustion; wired into both sync and async Cerebras paths
- **Streaming** — `generate_stream()` yields token deltas via SDK `stream=True`; preserves usage on the terminal chunk
- **Native function-calling** — `generate_with_tools()` integrates with the Agent loop in `hybrid` mode; `supports_native_tools` flag per model
- **`CostTracker`** (`effgen/models/_cost.py`) — thread-safe singleton with per-provider rate table (Cerebras free-tier $0); `record/total_cost/summary/reset`
- **Loader integration** — `load_model(..., provider="cerebras")` and `effgen.CerebrasAdapter` re-exported
- **Docs** — `docs/models/cerebras.md` (all 4 models, rate-limit table, streaming + tool examples)

#### OpenAI Modernization
- **Expanded model registry** (`effgen/models/openai_models.py`) — gpt-5, gpt-5.4-nano, gpt-4.1, gpt-4o family, o1/o1-mini/o3/o3-mini/o4-mini reasoning models; pricing + `supports_reasoning`/`supports_native_tools`/`supports_prompt_caching` flags
- **`reasoning_effort`** + **`max_reasoning_tokens`** on `GenerationConfig` — `Literal["minimal","low","medium","high"]`; routed only to reasoning models; chat models silently drop with debug log; unknown values raise `ValueError`
- **`_pick_default_max_output()`** — family-aware default output budget (reasoning models default to 100k)
- **Prompt caching surfacing** — `AgentConfig.stable_system_prompt` (default True); `cached_input_tokens` exposed via `ModelResponse.usage` and metadata
- **Structured outputs v2** — `OpenAIAdapter.generate_structured()`; `to_openai_schema()` helper inlines `$ref`s and forces `additionalProperties: false`; `ModelRefusalError` raised on refusal
- **OpenAI native tools** — `OpenAIWebSearchTool`, `OpenAICodeInterpreterTool`, `OpenAIFileSearchTool` in `effgen.tools.builtin.openai_native`; routed through Responses API; `ToolIncompatibleError` at Agent init when paired with non-OpenAI models
- **Docs** — `docs/models/openai.md`, `docs/models/openai_advanced.md`, `docs/tools/openai_native.md`

### Changed
- **`GenerationConfig`** gains `reasoning_effort` and `max_reasoning_tokens` (both `None` by default — fully back-compat)
- **`AgentConfig`** gains `stable_system_prompt: bool = True`
- **OpenAI adapter** uses `max_completion_tokens` (replaces deprecated `max_tokens`); drops unsupported `stop`, `temperature`, `top_p` params for reasoning/gpt-5 models
- **`load_model(..., provider=...)`** routes correctly to OpenAI/Anthropic/Gemini/Cerebras (previously HF-only); HF-specific kwargs are stripped before reaching API adapters; OpenAI auto-detection prefix list extended to gpt-5/o-series

### Fixed
- **Stability sweep** — ruff cleanup (F821, B023, F841, C401, C408, I001, F401, F541); 2 real bug fixes (missing `Any` import; loop-closure variable capture)
- **`load_model(..., provider="openai"/"anthropic"/"gemini")`** — was silently treated as HF-only
- **OpenAI** — warns rather than errors on unknown model ids
- **Transformers engine** — `unload()` now removes `accelerate` hooks and syncs CUDA, eliminating cross-test CUDA-state bleed
- **GPU e2e/integration test fixtures** — disabled bitsandbytes 4-bit (CUDA state leak) and narrowed scope to class-level, fixing intermittent Qwen2 RMSNorm aborts
- **Cerebras streaming test** — retries on Cerebras `429`/`queue_exceeded`

---

## [0.2.0] - 2026-04-09

### Highlights

**effGen v0.2.0** is a major release that transforms the framework from a capable agent toolkit into a **production-grade agentic AI platform** — with native tool calling, guardrails, multi-agent orchestration, RAG pipelines, evaluation, and a production API server — all optimized for Small Language Models.

### Added

#### Critical Bug Fixes & Foundation Repairs
- **ReAct parser hardening** — improved `Final Answer:` extraction with `Observation:`/`Human:` boundary splitting; `_clean_json_input()` handles trailing commas, markdown fences, unquoted keys; 28-case parser test suite
- **Async/sync race condition fix** — replaced direct `asyncio.run()` with `_run_coroutine_sync()` for sub-agent parallel execution; works inside Jupyter/FastAPI/async contexts; configurable timeout (120s default)
- **Memory performance fix** — `get_token_count()` uses cached `_current_token_count` instead of O(n) recalculation; structured summary format preserving facts/decisions/pending items
- **Agent resource cleanup** — `Agent.close()` + sync context manager (`with Agent(config) as agent:`)
- **MCP transport fix** — correlation-ID-based pending request tracking; SSE exponential backoff reconnection (max 5 retries)
- **Tool security hardening** — BashTool blocks `${VAR:-$(cmd)}`, heredoc injection, process substitution; PythonREPL blocks `__import__`, `importlib`, `__builtins__`, `__subclasses__`; standardized 30s timeout / 100KB output limits
- **Sub-agent depth tracking** — try/finally cleanup; reset on run() start
- **Vision pass-through** — OpenAI, Anthropic adapters now support image_url/image blocks
- **New examples** — `async_concurrent_agent.py`, Docker Compose deployment, `agent_communication.py`

#### Native Tool Calling & Structured Output
- **`ToolCallingStrategy`** — abstract strategy with `ReActStrategy`, `NativeFunctionCallingStrategy`, `HybridStrategy` implementations
- **Native function calling** — `supports_tool_calling()` on all model backends; Qwen/Llama/Mistral/generic format parsers; tool JSON Schema definitions passed via chat template `tools` parameter
- **`tool_calling_mode`** in `AgentConfig` — `"auto"`, `"native"`, `"react"`, `"hybrid"` modes
- **Structured output** — `StructuredOutputConfig`, `constrain_output()`, `validate_json_schema()`; `output_schema` and `output_model` (Pydantic) parameters on `Agent.run()`; `output_format` and `output_schema` on `AgentConfig`
- **`ToolDefinition`** — with OpenAI/Anthropic format converters and `tools_to_definitions()` utility

#### Guardrails, Safety & Input/Output Validation
- **`effgen.guardrails`** module — `Guardrail` ABC, `GuardrailChain`, `GuardrailPosition` enum
- **Content guardrails** — `ToxicityGuardrail`, `PIIGuardrail` (SSN/email/phone/CC with Luhn/IP), `LengthGuardrail`, `TopicGuardrail`
- **`PromptInjectionGuardrail`** — low/medium/high sensitivity with zero false positives on normal queries
- **Tool safety** — `ToolInputGuardrail`, `ToolOutputGuardrail` (PII stripping, size limit), `ToolPermissionGuardrail` (allow/deny/require_approval)
- **Agent integration** — `AgentConfig.guardrails` param; pre-run input check, pre/post-tool checks, pre-return output check
- **Presets** — `get_guardrail_preset("strict"|"standard"|"minimal"|"none")`

#### Advanced Multi-Agent Orchestration
- **`MessageBus`** — pub/sub, mailbox, broadcast inter-agent communication with topic-based wildcard subscriptions and optional persistence
- **`WorkflowDAG`** — DAG-based workflow engine with cycle detection (Kahn's topological sort), conditional branching, auto-parallelization via `asyncio.gather`; YAML workflow definitions; `effgen workflow run/validate` CLI
- **`SharedState`** — thread-safe namespaced key-value store with per-namespace RLock, snapshots for rollback, event-sourced mutation log
- **Agent lifecycle management** — `AgentLifecycleState` (8 states), `AgentEntry` state machine, `AgentPool` (pre-warmed), `AgentRegistry` (thread-safe); per-agent timeout and cancellation

#### Batch Execution & Domain Scaling
- **`BatchRunner`** — asyncio-based concurrent batch execution with semaphore, retry, timeout; JSONL/CSV/JSON/text I/O; `Agent.run_batch()` convenience; `effgen batch` CLI
- **`ResultAggregator`** — exact hash + fuzzy Jaccard deduplication, ranking (confidence/relevance/speed/custom), merge strategies (first/best/consensus/union)
- **`ToolResultCache`** — thread-safe LRU + TTL for cross-query tool result sharing
- **`effgen.domains`** module — `Domain` base class, `KeywordExpander` (WordNet/template/LLM expansion); 5 built-in domains: `TechDomain`, `ScienceDomain`, `FinanceDomain`, `HealthDomain`, `LegalDomain`

#### Observability, Tracing & Debugging
- **OpenTelemetry upgrade** — full OTel SDK with Resource, BatchSpanProcessor, configurable exporters (OTLP/Jaeger/Zipkin/console); cross-agent trace propagation; no-op fallback
- **Structured logging** — `EffGenJSONFormatter`, `StructuredLogger` with agent/tool/model/iteration events; `LogRunContext` with run_id/workflow_id/agent_name/session_id correlation
- **Prometheus metrics upgrade** — response_latency/token_usage/tool_execution_time histograms with percentiles; GPU memory gauge; labels support
- **Grafana dashboard** — 12 panels: latency p50/p95/p99, throughput, error rate, tool breakdown
- **`effgen.debug`** module — `DebugAgent` wrapper with rich TUI step-through; `Agent.run(debug=True)` captures `DebugTrace` with per-iteration raw_prompt, raw_response, thought, action, observation, tokens, latency; `effgen debug` CLI

#### Model Router & Auto-Selection
- **`ModelRouter`** — routing by complexity, capabilities, loaded state, model size; `RoutingConfig`, `RoutingDecision`
- **`estimate_complexity()`** — heuristic keyword analysis (code/math/reasoning/multilingual), query length, structural patterns; < 1ms execution
- **`MODEL_CAPABILITIES`** — registry with pre-populated profiles for 12 models (Qwen 0.5B-7B, Llama 1B-3B, Phi-3/3.5/4, Mistral 7B, Gemma 2B/9B)
- **Multi-model agent** — `models` list and `speculative_execution` in AgentConfig; ModelRouter auto-created; `_generate_speculative()` runs on 2 models via `asyncio.wait(FIRST_COMPLETED)`
- **`ModelPool`** — LRU eviction, GPU memory-based eviction, hot-swap; `effgen models load|unload|status` CLI

#### Community Contribution: MLX & MLX-VLM Backends (PR #4, commit e5b54f5)
- **`MLXEngine`** — MLX (mlx-lm) text generation engine with streaming/batch support for Apple Silicon
- **`MLXVLMEngine`** — MLX-VLM vision-language engine with image support (30+ architectures)
- **`effgen.hardware`** module — `platform.py` with Apple Silicon/CUDA/MLX detection helpers and backend recommendation
- **Model loader integration** — MLX/MLX-VLM auto-selection on Apple Silicon; `ModelType.MLX` and `ModelType.MLX_VLM`
- **Optional deps** — `pip install effgen[mlx]` and `pip install effgen[mlx-vlm]` (darwin/arm64 only)
- **5 new GUI examples** — `chat_gui_mlx.py`, `agent_viz_mlx.py`, `tool_builder_gui.py`, `tool_tester_gui.py`, `basic_agent_mlx.py` (Gradio-based)
- **Unit tests** — `test_hardware_platform.py`, `test_mlx_engine.py`, `test_mlx_vlm_engine.py`

#### Persistent Agent State & Checkpointing
- **`CheckpointManager`** — save/restore full agent state (scratchpad, memory, tool states, iteration count, partial results); filesystem + SQLite backends
- **Agent checkpoint/resume** — `agent.run("...", checkpoint_interval=3)` for periodic checkpointing; `agent.resume(checkpoint_id="...")` to resume; CLI: `effgen run --checkpoint-dir --checkpoint-interval`, `effgen resume --checkpoint`
- **`Session`** / `SessionManager` — persistent conversation sessions with UUID management, expiry, cleanup; `Agent(config, session_id="user-123")` auto-loads/persists per turn; CLI: `effgen sessions list|delete|export|cleanup`
- **`BackgroundTaskRunner`** — priority queue, pause/resume/cancel, threading workers; `Agent.run_background()` / `get_task_status()` / `get_task_result()` / `cancel_task()`

#### Advanced RAG Pipeline
- **`effgen.rag`** module — complete RAG pipeline
- **`DocumentIngester`** — txt/md/json/jsonl/csv/html built-in loaders; pdf/docx/epub optional; SHA-256 deduplication; progress tracking
- **Advanced chunking** — `SemanticChunker`, `CodeChunker` (py/js/ts/go/rust/java), `TableChunker`, `HierarchicalChunker`
- **`HybridSearchEngine`** — dense + BM25 + keyword + metadata filter fused via Reciprocal Rank Fusion
- **Reranking** — `CrossEncoderReranker` (optional), `LLMReranker` (free default), `RuleBasedReranker` (recency/authority/keyword/title)
- **`ContextBuilder`** — token budget management, source deduplication, relevance/chronological ordering, inline `[N]` citations
- **Source attribution** — `Citation` dataclass, `CitationTracker` with verify/extract; `AgentResponse.citations` and `.sources` fields
- **RAG preset** — `create_agent("rag", model, knowledge_base="./docs/")`

#### Human-in-the-Loop & Approval Workflows
- **Human interaction points** — `HumanApproval`, `HumanInput`, `HumanChoice` (all with timeout via ThreadPoolExecutor)
- **Tool approval** — `requires_approval` on `ToolMetadata`; `approval_callback`, `approval_mode` (`always`/`first_time`/`never`/`dangerous_only`), `approval_timeout` in `AgentConfig`; `ApprovalManager` wired into tool execution path
- **Clarification** — `ClarificationRequest` (options + free-text), `ClarificationDetector` with heuristic ambiguity detection (short query, vague words, multiple-tool-match)
- **Feedback collection** — `FeedbackCollector` (thumbs/rate/comment), `FeedbackEntry`, export to JSONL

#### New Domain Tools (17 New Tools — 31 Total)
- **Finance** — `StockPriceTool` (yfinance + Yahoo Finance v8 fallback), `CurrencyConverterTool` (frankfurter.app/ECB), `CryptoTool` (CoinGecko); all include "not financial advice" disclaimer
- **Data Science** — `DataFrameTool` (pandas: load/head/describe/filter/aggregate), `PlotTool` (matplotlib: line/bar/scatter/hist → PNG), `StatsTool` (numpy: mean/median/std/correlation/regression)
- **DevOps** — `GitTool` (read-only: status/log/diff/branch/show), `DockerTool` (read-only: ps/images/logs), `SystemInfoTool` (psutil: cpu/memory/disk/network), `HTTPTool` (urllib GET/POST)
- **Knowledge** — `ArxivTool` (Atom feed), `StackOverflowTool` (SE API), `GitHubTool` (public search API), `WolframAlphaTool` (optional, requires API key)
- **Communication** — `EmailDraftTool` (draft only, does NOT send), `SlackDraftTool` (draft only), `NotificationTool` (plyer desktop notifications, optional)
- All external libraries handled as optional with clear install hints

#### Evaluation, Benchmarking & Regression Testing
- **`effgen.eval`** module — `AgentEvaluator`, `EvalResult`, `SuiteResults`, `TestCase`, `TestSuite`
- **Scoring modes** — `EXACT_MATCH`, `CONTAINS`, `REGEX`, `SEMANTIC_SIMILARITY` (sentence-transformers optional), `LLM_JUDGE`
- **5 built-in test suites** — `MathSuite` (77 cases), `ToolUseSuite` (93), `ReasoningSuite` (40), `SafetySuite` (40), `ConversationSuite` (20)
- **`RegressionTracker`** — save/load/compare baselines; severity levels (warning/high/critical); thresholds: >5% accuracy drop, >20% latency increase
- **`ModelComparison`** — multi-model matrix comparison with recommendations; markdown/JSON export
- **CLI** — `effgen eval --suite <name>` and `effgen compare --models "a,b,c" --suite <name>`
- **Nightly CI** — eval-regression job compares against stored baselines, opens GitHub issue on failure

#### API Server v2 — Production Gateway
- **OpenAI-compatible API** — `/v1/chat/completions` and `/v1/completions` with `tools` param and `stream: true` (SSE); model aliases (gpt-4 → Qwen2.5-7B, gpt-3.5-turbo → Qwen2.5-3B)
- **`RequestQueue`** — priority queue with fair scheduling, deadlines, backpressure (`QueueFullError`)
- **`AgentPool`** — min/max size, factory, idle TTL, health checking, acquire/release
- **Multi-tenancy** — `TenantManager` (rate limits, model restrictions, tool permissions); `APIKey` management with hashed storage and constant-time resolution
- **Production middleware** — CORS, request ID injection (X-Request-ID), GZip compression, graceful shutdown

#### SDK, Client Libraries & Embedding API
- **Python client SDK** — `EffGenClient` with sync + async via httpx; `chat()`, `embed()`, `health()`, `chat_stream_sync()`, `achat()`, `chat_stream()` (async iterator); retries with exponential backoff; 7 typed exception classes
- **TypeScript/JavaScript client** — `clients/typescript/` with fetch-based `EffGenClient`; chat/embed/health/streaming; works in Node 18+/Deno/Bun/browser
- **Local embedding API** — `/v1/embeddings` endpoint (OpenAI-compatible); `SentenceTransformerEmbedder` + `TFIDFEmbedder` fallback; model aliases; `LRUCache` + `SQLiteCache` for embedding caching

#### Performance Optimization & Caching
- **`effgen.cache`** module — `PromptCache` (LRU + TTL, sha256 fingerprint, thread-safe, hit/miss stats); `ResultCache` (LRU + per-tool TTL, optional semantic similarity via embed_fn + cosine)
- **`TokenBudget`** — smart context window allocation (system 20% / tools 30% / history 40% / response 10%); `smart_truncate()` preserves head+tail; `fit_to_budget()` per-section truncation
- **`LazyModel`** — defers `.load()` until first generate/count_tokens; idle_timeout-based eviction (default 600s)
- **GGUF support** — `GGUFEngine` via optional llama-cpp-python; auto-routed by model_loader for `.gguf` files
- **AWQ / GPTQ quantization** — `quantization="awq"` and `quantization="gptq"` in model_loader; optional deps with friendly install hints
- **Speculative decoding** — `GenerationConfig.draft_model` field for backends that support draft-model decoding
- **`ContinuousBatcher`** — coalesces concurrent submit() calls in background worker; max_batch_size / max_wait_ms flush; `BatchModel` fast path + sequential fallback

### Changed
- **7 inference backends** (was 5) — added MLX and MLX-VLM for Apple Silicon
- **31 built-in tools** (was 14) — added 17 domain tools (finance, data science, DevOps, knowledge, communication)
- Model backends now support `supports_tool_calling()` for native function calling
- `AgentConfig` extended with `tool_calling_mode`, `output_format`, `output_schema`, `guardrails`, `models`, `speculative_execution`, `approval_mode`, `approval_callback`, `session_id`, `checkpoint_interval`, `checkpoint_dir`
- `AgentResponse` extended with `citations` and `sources` fields
- `Agent.run()` accepts `output_schema`, `output_model`, `debug`, `checkpoint_interval` parameters
- Prometheus metrics now include histograms with percentiles, GPU memory gauge, and labels

### Fixed
- `asyncio.run()` crash when Agent used inside existing event loops (Jupyter, FastAPI)
- ShortTermMemory `get_token_count()` O(n) recalculation on every call (now O(1))
- MCP HTTP transport race condition with concurrent requests
- Sub-agent `_current_depth` not reset on completion/failure
- BashTool vulnerable to nested command substitution (`${VAR:-$(cmd)}`)
- PythonREPL sandbox escape via `__import__`, `importlib`, `__builtins__`

### Internal
- 487+ unit tests passing (up from 157 in v0.1.3)
- Real GPU integration tests (A40 GPUs)
- Fresh isolated environment validation for each feature set
- Nightly CI with eval regression detection and automated GitHub issue creation

---

## [0.1.3] - 2026-03-25

### Added
- **Sub-agent depth limiting** — `max_sub_agent_depth` config option (default 3) prevents unbounded sub-agent recursion (ISSUE-005)
- **"No tool needed" guidance** in ReAct prompt — explicit instruction and example for direct answers, reducing unnecessary tool calls by SLMs (ISSUE-016)
- **Model-aware token counting** — `ShortTermMemory` now accepts an optional `model` parameter for accurate tokenization instead of the `len(text)//4` heuristic (ISSUE-009)
- **Circuit breaker persistence** — optional JSON file persistence for circuit breaker state via `persist_path` parameter (ISSUE-012)
- **Streaming timeout safety** — all streaming examples now use `signal.SIGALRM` timeouts to prevent indefinite hangs (ISSUE-013)
- **`pytest-timeout`** added to dev dependencies with 120s default timeout (ISSUE-001)
- **`bitsandbytes`** added to dev dependencies for 4-bit quantization testing (ISSUE-002)

### Improved
- **Loop detection** — exact loop now allows 1 retry before triggering (was zero-tolerance); fuzzy loop threshold raised to 7 for `DATA_PROCESSING` category tools; action inputs normalized (JSON key sorting, whitespace stripping) before comparison (ISSUE-004, ISSUE-019)
- **Partial answer extraction** — observations now scanned for day names and numeric results; multiple valid observations combined for multi-tool tasks (ISSUE-017)
- **"Answer now" nudge** — when iterations are running low and a tool returned successfully, the scratchpad hints the model to emit `Final Answer:` (ISSUE-017)
- **Model-family prompt formatters** — Qwen format uses `<|tools|>` section markers; Llama format uses `<|begin_of_text|>` header/EOT tags (ISSUE-010)
- **Stop sequences** — removed overly aggressive `\n\n\n` stop sequence that could truncate legitimate multi-paragraph output (ISSUE-015)
- **System prompt** — added "Do NOT use tools for greetings, jokes, opinions, or recalling information" to mistakes section (ISSUE-016)
- **Model loading warning** — logs a clear warning when `require_model=False` and loading fails, instead of silently setting `self.model = None` (ISSUE-011)
- **Integration test robustness** — `real_model` fixture falls back to fp16 if bitsandbytes is not installed (ISSUE-002)

### Fixed
- **NotImplementedError messages** — MCP transport stubs and Retrieval tool stubs now include descriptive messages instead of bare `raise NotImplementedError` (ISSUE-006, ISSUE-008)

### Internal
- 19 issues from v0.1.2 verification addressed across 12 files
- Streaming examples hardened with timeout handling
- Conversational agent example tuned for memory summarization

## [0.1.2] - 2026-03-12

### Added
- **10 comprehensive example agents** covering Q&A, calculator, multi-tool, file operations, code execution, conversational memory, error recovery, data processing, streaming, and multi-agent pipeline orchestration
- **Cross-model compatibility matrix** — 11 models tested across all 10 agents (110 combinations), 73% pass rate ([compatibility_matrix.md](examples/compatibility_matrix.md))
- **User-explicit sub-agent trigger detection** in `SubAgentRouter` — regex-based fuzzy matching for phrases like "use sub-agents", "launch 3 agents", "spawn agents" (router.py)
- **Compatibility sweep runner** (`examples/sweep_model.py`) for automated cross-model testing

### Improved
- **ReAct loop robustness** — loop detection breaks repeated identical actions (BUG-003), fuzzy loop detection for 5+ calls with different inputs (BUG-017)
- **Tool input parsing** — single-quoted JSON via `ast.literal_eval` fallback (BUG-016), non-JSON input mapping, markdown fence stripping for code params
- **Conversation history** — `max_turns` increased from 5 to 25, summary inclusion, assistant response truncation (300 chars), configurable `keep_recent_messages` (BUG-014, BUG-015)
- **Answer extraction** — line-start anchor for "Answer:" regex with `re.MULTILINE` (BUG-004), trailing text trimming (BUG-005), newline boundary fix for `Action: Final Answer` (BUG-008)
- **Tool result formatting** — proper extraction of `data`/`message` keys from FileOperations dict results (BUG-010), stderr extraction for CodeExecutor errors (BUG-013), stdout preference over None for PythonREPL (BUG-012)
- **Default max_tokens** increased from 512 to 1024 for long tool data (BUG-018)

### Fixed
- **BUG-001:** `quantization="4bit"` silently ignored by TransformersEngine — now properly passed through (model_loader.py)
- **BUG-002:** gemma-3 context length detection fails when config uses nested `text_config` (transformers_engine.py)
- **BUG-006:** DateTimeTool `now` operation ignores `date` parameter (datetime_tool.py)
- **BUG-007:** `validate_parameters` rejects unknown parameters hallucinated by SLMs — now warns instead of failing (base_tool.py)
- **BUG-009:** `_map_input_to_parameters` strips leading slash from absolute paths via `lstrip('/')` (agent.py)
- **BUG-011:** PythonREPL `_execute()` re-evaluates last `ast.Call` expression causing double `print()` output (python_repl.py)

### Internal
- Test-driven development with real GPU inference across 12 development iterations
- 19 framework bugs discovered and fixed through systematic agent testing
- Compatibility testing across 11 model families (0.5B to 8B parameters)
- Verification sweep: 116 unit tests pass, all integration tests pass

## [0.1.1] - 2026-03-06

### Fixed
- Fixed license inconsistency: all files now correctly reference Apache-2.0 (was MIT in some files)
- Fixed `setup.py` entry point mismatch: `effgen-agent` now correctly points to `agent_main` (was `main`)
- Fixed `setup.py` Development Status: now correctly says Beta (was Alpha)
- Fixed `setup.py` dependency version mismatches with `pyproject.toml` (duckduckgo-search, cloud-secrets, monitoring groups)
- Fixed missing `fastapi` and `uvicorn` in `pyproject.toml` dependencies (`effgen serve` now works out of the box with `pip install effgen`)
- Replaced 5 bare `except:` in `gpu/monitor.py` with specific exception handlers
- Replaced 15+ `print()` calls with proper logger calls in `docker_sandbox`, `decomposition_engine`, `router`, `complexity_analyzer`, `gpu/utils`
- Added logging to silent `except Exception:` handlers in execution modules (`docker_sandbox.py`, `sandbox.py`, `code_executor.py`)

### Added
- `NEWS.md` with user-friendly release summaries
- 6 new example scripts: `preset_agents`, `streaming_agent`, `memory_agent`, `multi_tool_agent`, `weather_agent`, `plugin_example`
- Updated `examples/README.md` with descriptions for all examples
- Top-level imports for `ToolFallbackChain`, `CircuitBreaker`, `ToolPromptGenerator`, `AgentSystemPromptBuilder`
- CLI smoke tests (`tests/integration/test_cli.py`)
- API server tests (`tests/integration/test_api_server.py`)
- Plugin system tests (`tests/unit/test_plugin.py`)
- Preset tests (`tests/unit/test_presets.py`)
- Fallback chain tests (`tests/unit/test_fallback.py`)
- Circuit breaker tests (`tests/unit/test_circuit_breaker.py`)
- Benchmark baseline (`tests/benchmarks/baseline.json`)

### Changed
- All error handlers in `gpu/monitor.py` now catch specific exceptions instead of bare `except:`
- Diagnostic output in `docker_sandbox`, `decomposition_engine`, `router`, `complexity_analyzer`, and `gpu/utils` now uses structured logging

### Internal
- Lint cleanup via ruff (2200+ auto-fixes)
- mypy fixes on modified files
- Validated all GitHub Actions YAML files

---

## [0.1.0] - 2026-03-01

### Added

#### Foundation Hardening
- **ToolPromptGenerator**: Dynamic system prompts with exact tool usage examples for SLMs
- **Model-Specific Prompts**: Optimized prompt formatting for Qwen, Llama, and Phi model families
- **Tool Fallback Chains**: Automatic fallback when tools fail (e.g., calculator → python_repl → code_executor)
- **CircuitBreaker**: Tracks tool failure rates and temporarily disables failing tools
- **Enhanced Tool Descriptions**: Structured format with parameter types, defaults, and usage examples
- **Retry Logic**: Exponential backoff for empty model responses with temperature adjustment
- **Partial Answer Extraction**: Extracts best answer from scratchpad when max iterations reached
- **Input Sanitization**: Validates and sanitizes all tool inputs before execution
- **Async Context Manager**: `async with Agent(config) as agent:` support
- **True Async**: `run_async()` is now natively async (not executor-wrapped)

#### Tool Ecosystem (7 New Tools — 14 Total)
- **BashTool**: Shell command execution with security controls (command whitelist/blacklist)
- **WeatherTool**: Weather data via Open-Meteo API (free, no API key required)
- **JSONTool**: Parse, query (JSONPath), transform, and validate JSON
- **DateTimeTool**: Current time, timezone conversion, date arithmetic
- **TextProcessingTool**: Word count, regex operations, text comparison
- **URLFetchTool**: Fetch and extract text from web pages
- **WikipediaTool**: Search and retrieve Wikipedia articles (free API)
- **Enhanced Retrieval**: Document loaders (txt, md, pdf, csv, json), chunking strategies, hybrid search (vector + BM25)
- **Enhanced AgenticSearch**: ripgrep backend, multi-query, file-type awareness, summarization
- **AgentSystemPromptBuilder**: Auto-generates tool-aware system prompts per agent configuration

#### Protocols & Streaming
- **ACP Protocol Complete**: Full JSON Schema validation, server/client modes, async task polling
- **MCP Client Enhanced**: Auto-reconnection, MCP→effGen tool bridge, resource→context bridge, health monitoring
- **Real Streaming**: True token streaming via generate_stream() (replaces placeholder)
- **Streaming Callbacks**: on_thought, on_tool_call, on_observation, on_answer
- **SSE Streaming**: Server-Sent Events endpoint for real-time API streaming
- **Memory Integration**: ShortTermMemory, LongTermMemory, VectorMemoryStore connected to Agent
- **Memory Configuration**: Configurable backends, persistence paths, auto-summarization

#### Infrastructure
- **CI/CD Pipelines**: GitHub Actions for CI, releases, docs, nightly tests, health checks, PR gates
- **Health Monitoring**: Website, DNS, SSL checks for effgen.org and docs.effgen.org
- **Test Suite**: 67 unit tests, 8 benchmarks, integration and e2e tests with MockModel and fixtures
- **Observability**: OpenTelemetry tracing (no-op fallback), Prometheus metrics
- **`effgen health` Command**: CLI health checker for all infrastructure
- **Code Quality**: Pre-commit hooks (black, isort, flake8, mypy, bandit), CONTRIBUTING.md

#### Developer Experience
- **Plugin System**: ToolPlugin base class with entry point and directory discovery
- **Agent Presets**: Ready-to-use configs — math, research, coding, general, minimal
- **`create_agent()` Factory**: One-line agent creation from presets
- **CLI Enhancements**: Rich progress, verbose/explain modes, tab completion (bash/zsh/fish), session persistence
- **API Server**: WebSocket streaming, API key authentication, rate limiting, OpenAPI docs, /health, /metrics
- **Documentation**: API reference, 6 tutorials, architecture guide, configuration reference, FAQ, migration guide
- **Packaging**: py.typed (PEP 561), Dockerfile, conda-forge recipe, optional dependency groups

### Changed
- `_get_tools_description()` now outputs structured format with parameter details
- `stream()` now uses real token streaming (previously character-by-character placeholder)
- `run_async()` is now truly async (previously wrapped sync in executor)
- Memory system uses proper ShortTermMemory/LongTermMemory classes (previously plain list)
- ACP `validate_request()` now does full JSON Schema validation (previously only checked required fields)
- User-Agent strings now use dynamic version from `effgen.__version__`
- Development status upgraded from Alpha to Beta

### Fixed
- All `NotImplementedError` paths in retrieval tool
- ACP TODO for JSON schema validation
- Streaming placeholder (`time.sleep(0.01)`)
- Memory as plain list (`self.short_term_memory = []`)
- Direct inference path now includes conversation history for multi-turn context retention

---

## [0.0.2] - 2026-02-03

### Added
- **Retrieval Tool**: RAG-based semantic search tool for knowledge base Q&A
- **Agentic Search Tool**: Grep-based exact match search with async support

### Fixed
- **vLLM Backend**: Fixed automatic chat template support for instruction-tuned models
- **GPU Memory Control**: Improved `gpu_memory_utilization` parameter handling
- **OOM Error Handling**: Better error messages and suggestions for CUDA out-of-memory errors
- **Tensor Parallel Auto-Selection**: Fixed auto-detection of tensor parallel size for small models (1.7B, 4B, etc.)
- **vLLM Cache Directory**: Resolved issues with vLLM cache directory handling

### Changed
- **Model Loader**: Improved small model detection for tensor parallel size selection
- **Version Management**: Consolidated `__version__` to single source in main `effgen/__init__.py`

### Compatibility
- Tested with multiple model families:
  - Qwen (Qwen3-1.7B, Qwen2.5-3B-Instruct)
  - Meta Llama (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct)
  - Microsoft Phi (Phi-4-mini-instruct)
  - HuggingFace SmolLM (SmolLM2-1.7B-Instruct, SmolLM3-3B)
  - Google Gemma (Gemma-3-4b-it)

---

## [0.0.1] - 2026-01-31

### Added

#### Core Framework
- **Agent System**: Complete agentic framework optimized for Small Language Models (1B-7B parameters)
- **Task Management**: Task and SubTask classes with priority levels and status tracking
- **Agent State**: Comprehensive state management for agent execution
- **ReAct Pattern**: Reasoning and Acting pattern implementation for structured problem-solving

#### Model Support
- **Multi-Backend Support**:
  - HuggingFace Transformers (local models)
  - vLLM (fast inference with 5-10x speedup)
  - OpenAI API adapter
  - Anthropic API adapter
  - Google Gemini API adapter
- **Model Loader**: Automatic model detection and loading with intelligent fallback
- **Generation Configuration**: Flexible configuration for temperature, tokens, sampling, etc.

#### Tool System
- **Built-in Tools**:
  - Calculator (basic math, conversions, financial calculations)
  - Web Search (DuckDuckGo integration with caching)
  - Code Executor (Python, JavaScript, Bash in sandboxed environment)
  - File Operations (read, write, list, search)
  - Python REPL (interactive Python execution)
- **Tool Registry**: Dynamic tool registration and discovery
- **Protocol Support**:
  - MCP (Model Context Protocol) - Official Anthropic SDK integration
  - A2A (Agent-to-Agent) protocol
  - ACP (Agent Communication Protocol)

#### Prompt Engineering
- **Template Manager**: Jinja2-based template system with versioning
- **Chain Manager**: Multi-step prompt chaining with conditional execution
- **Prompt Optimizer**: SLM-specific optimization techniques
- **Few-Shot Learning**: Dynamic example selection for improved performance

#### Memory Systems
- **Short-Term Memory**: Conversation history and context management
- **Long-Term Memory**: Persistent storage with importance-based retrieval
- **Vector Store**: Semantic search with FAISS, ChromaDB, and Qdrant support
- **Storage Backends**: JSON and SQLite storage options

#### Task Decomposition
- **Complexity Analysis**: Automatic task complexity assessment
- **Decomposition Engine**: Break complex tasks into manageable subtasks
- **Sub-Agent Manager**: Specialized sub-agents for different task types
- **Orchestrator**: Coordinate multi-agent execution with parallel/sequential strategies

#### GPU Management
- **GPU Allocator**: Intelligent GPU allocation with memory requirements
- **GPU Monitor**: Real-time monitoring of utilization, temperature, and power
- **Multi-GPU Support**: Automatic distribution across available GPUs

#### Code Execution
- **Sandboxed Execution**: Safe code execution with Docker containers
- **Code Validator**: Static analysis and security checks
- **Multiple Languages**: Support for Python, JavaScript, Bash, and more
- **Resource Limits**: Configurable CPU, memory, and timeout limits

#### Configuration
- **YAML Configuration**: Hierarchical configuration with validation
- **JSON Schema Validation**: Type-safe configuration with comprehensive schemas
- **Environment Variables**: Secure secret management with .env support
- **Cloud Secrets**: AWS Secrets Manager, HashiCorp Vault, Azure Key Vault integration

#### CLI Interface
- **Interactive Chat**: Real-time chat interface with rich formatting
- **One-Shot Execution**: Direct task execution from command line
- **API Server**: FastAPI-based REST API server
- **Web Agent**: Autonomous web browsing and interaction
- **Tool Management**: List, inspect, and test tools

#### Utilities
- **Logging System**: Rich, structured logging with multiple levels and formats
- **Metrics Tracking**: Performance metrics, token usage, and cost tracking
- **Error Handling**: Comprehensive error handling with retry logic
- **Async Support**: Full async/await support for concurrent operations

#### Examples & Documentation
- **Basic Agent Example**: Simple agent with calculator and web search
- **Web Agent Example**: Agent that can browse and extract information
- **Installation Script**: Interactive installer with animations
- **Security Policy**: Comprehensive security guidelines and vulnerability reporting

### Configuration Files
- `pyproject.toml`: Modern Python packaging with build system configuration
- `setup.py`: Traditional setuptools configuration for compatibility
- `.gitignore`: Comprehensive ignore patterns for Python, IDEs, and system files
- `requirements.txt`: Core dependencies with version specifications

### Package Metadata
- **License**: MIT License
- **Python Support**: 3.10, 3.11, 3.12, 3.13
- **Development Status**: Alpha
- **Keywords**: ai, agents, llm, slm, language-models, tool-use, multi-agent

### Optional Dependencies
- `dev`: Development tools (pytest, black, isort, flake8, mypy)
- `vllm`: Fast inference engine
- `flash-attn`: Flash Attention for faster transformer inference
- `vector-db`: Vector database backends (FAISS, ChromaDB, Qdrant)
- `search`: Advanced search engines (Google, DuckDuckGo)
- `cloud-secrets`: Cloud secret management (AWS, Azure, Vault)
- `monitoring`: Experiment tracking (Weights & Biases, TensorBoard)
- `all`: All optional dependencies combined

### Entry Points
- `effgen`: Main CLI entry point
- `effgen-agent`: Agent-specific commands
- `effgen-web`: Web agent interface

---

## Version History

### Version Naming Convention
- **Major.Minor.Patch** (Semantic Versioning)
- **Major**: Breaking changes, major new features
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, minor improvements

### Release Schedule
- **Patch releases**: As needed for critical bugs
- **Minor releases**: Monthly feature updates
- **Major releases**: Quarterly for significant changes

---

## Links

- **GitHub**: https://github.com/ctrl-gaurav/effGen
- **PyPI**: https://pypi.org/project/effgen/
- **Documentation**: https://effgen.org/docs/
- **Issues**: https://github.com/ctrl-gaurav/effGen/issues

---

## Contributors

Thank you to all contributors who helped make effGen possible!

- Gaurav Srivastava (@ctrl-gaurav) - Creator and maintainer
- Yasuo Tabei (@tb-yasu) - Gemma 4 tool-call format, MLX engine (1.0.0)
- Aafiya Hussain (@Aafiya-H) - multi-GPU device placement (1.0.0)

---

[Unreleased]: https://github.com/ctrl-gaurav/effGen/compare/v1.0.0...HEAD
[1.0.1]: https://github.com/ctrl-gaurav/effGen/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ctrl-gaurav/effGen/compare/v0.3.2...v1.0.0
[0.3.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/ctrl-gaurav/effGen/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.10...v0.3.0
[0.2.10]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ctrl-gaurav/effGen/releases/tag/v0.0.1
