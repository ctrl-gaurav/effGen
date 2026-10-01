# effGen Release Notes

## v1.3.0 - October 1, 2026

This release is about how a run ends, and how a tool call is read: a run that stops making
progress is asked for its answer, and every run says how it ended.

The main things: `max_turns_without_progress` now defaults to 2, so a stalled run is asked for its
answer instead of going round to its iteration cap, and `response.termination` says whether it was
done, not possible, stuck, stopped by a failing tool or an error. A run whose tool keeps failing on
its own side ends `tool_failed`. `recover_lost_tool_calls` now defaults to on, so a tool call written
in a broken shape is read and run. A model you serve yourself, or run on a local engine, is measured
once for what it does with a tool, and `auto` tool calling follows what was measured. One agent
serves overlapping conversations without mixing them, and a streamed turn is saved to the session the
agent is bound to. `reasoning_effort` reaches a model behind `base_url`, and Groq.

On task types kept out of this release's development, the accuracy gain over 1.2.0 comes from tasks
whose tool fails, and elsewhere it is flat. Small models given a search tool cost noticeably more per
run, and tool-using runs still make more model calls than they need to; what this release cost
against 1.2.0 is set out at the end.

Ten changes are visible to existing code, and they are listed first.

The public surface grew from 251 names to 253. Nothing was removed or renamed.

```bash
pip install --upgrade effgen
effgen --version
```

### Read this before upgrading: ten changes you will notice

#### 1. A run that stops making progress is asked for its answer — `max_turns_without_progress` defaults to `2`

It was `None`. After two turns in a row that bring no new tool result, the next turn offers no tools
and asks for the answer. A turn that declares no action after a result — `Action: None`, `Action:
(continue reasoning)` or another bracketed placeholder that names no tool the agent holds — is asked
at once. Before the run's first result, only a turn whose every call was declined counts.

A run that would have stopped on `loop_detected`, `repeated_tool_result`, `max_iterations_*` or
`null_final_from_model` while holding tool results gets one **closing request**: its own calls and
results, with no tools, whose reply is the answer. When that reply is an answer the run succeeds,
with `metadata["answer_source"] == "closing_request"`, so a run that raised `RunStoppedError` in
1.2.0 under the default `raise_on_error=True` can now return an answer. A closing reply that is a
call, or a program for a code tool the agent holds, is not taken as the answer; after a program the
run carries on as it would have without the request, so such a run can send one model call more than
`max_iterations`. On the text scaffold, the closing request is how the run is asked for its answer.

In testing, on 32 of the 33 task-and-model combinations measured, no correct answer came from text a
loop guard put together from tool results any more; those runs now end with an answer the model
wrote. On the larger model measured, runs on one arithmetic sample set make 6.5% fewer model calls.

*Migration:* `max_turns_without_progress=None` — in `AgentConfig` or per
`run(max_turns_without_progress=None)` — restores 1.2.0's loop.

#### 2. A run whose tool keeps failing ends `tool_failed`

A tool that fails on its own side — a connection error, a timeout, an HTTP 5xx or 429, missing
credentials — three times in a row, or whose circuit breaker is open, is unavailable for the rest of
the run. A run left with no usable tool and no result now stops with the new stop reason
`tool_failed`, which is in `STOPPED_REASONS` and raises `RunStoppedError` under the default
`raise_on_error=True`. In 1.2.0 the run went on and returned whatever the model wrote, often a
statement that it could not get the information. Such a run gets **no** closing request: it reports
the tool's failure, typed, with `metadata["error"]["kind"] == "tool"`, and
`metadata["unavailable_tools"]` names the tools.

The agent's per-tool circuit breaker now counts only those tool-side failures. A tool given bad
input by the model is no longer refused to later calls and later runs. A call that is retried after
a tool-side failure is a retry, not a repeated call, so the loop guard no longer stops it.

*Migration:* a caller that read the model's text from such a run catches `RunStoppedError`, or
passes `raise_on_error=False` and branches on `response.termination == "tool_failed"`.

#### 3. `AgentResponse.termination` says how a run ended

`"done"` (the model answered), `"not_possible"` (the model answered, but every call was declined,
failed on the tool's side or returned nothing — usually an answer saying the task cannot be done
with these tools), `"stuck"` (the run kept proposing work that brought nothing new and wrote no
answer), `"tool_failed"` (change 2) and `"error"` (the run could not be carried out). `to_dict()`
carries it, and a saved run read back reports the same value. A run whose calls reached a tool that
rejected their input used the tool, and its answer is `"done"`. Alongside it,
`metadata["tool_results"]` counts a run's attempted, usable and input-rejected calls.

A stopped run's `partial` never carries a tool's error message; a run whose only observations were
errors has `partial=None`.

#### 4. A tool call is read before it is reported — `recover_lost_tool_calls` defaults to `True`

It was `False`. A tool call written as a Python literal, with raw line breaks inside its JSON, with
unescaped double quotes inside a string value, with its last string closed one bracket early, or as
a whole object followed by text inside its tag, now runs instead of ending the run with
`written_tool_call`. A call nothing can read is sent back once, with a call required where the
provider supports it. When the agent holds exactly one code-execution tool, a program in a fenced
block before an empty call tag runs as that call. Some runs make more tool calls: the calls the
model meant to make now run.

Under either setting:

* **Arguments that arrive as a string are read** — as keywords, as an object's JSON, or as a raw
  value — instead of the call running with no arguments. `tool_calls` records carry the decoded
  arguments.
* **A call missing a required argument is not dispatched.** The model is asked for the call again,
  so tool code never sees an empty call for a parameter it declared required.
* **Positional values are named** in the tool's parameter order. A call with more values than the
  tool has parameters carries no arguments and is asked for again, where it used to run with its
  first value or end the run with an error.

In testing, no measured run ended `written_tool_call`. On the hardest coding task measured, 15.7
points more answers were correct, at 15% more model calls.

*Migration:* `recover_lost_tool_calls=False` — in the config or per run — restores 1.2.0's reader;
delegated sub-agents inherit the setting.

#### 5. A failed tool call is retried, bounded, and never an answer

* **A failure is never a repeated result.** Two calls whose different inputs a tool rejects in the
  same words no longer withdraw the run's tools. In testing, no run on a calculator task stopped on
  `repeated_tool_result`; 1.2.0 stopped 25 runs that way on one math sample set, 12% of them right.
* **Four failures in a row on a tool's input withdraw it for the run.** A run left with no tool is
  asked for its answer, and ends `tool_failed` with `metadata["error"]["kind"] == "input"` when it
  writes none, where it used to go round to `max_iterations`.
* **An answer that is a tool's error message is not a success.** It is sent back once; given again,
  the run stops with `null_final_from_model`. A failed call no longer earns "you have the answer
  from the tool" near the iteration cap, and the direct calculator result is never a failure.

#### 6. Served and local models are measured once for how they use a tool

At `tool_calling_mode="auto"`, the first agent with tools built for a model served behind a
`base_url`, or run on a local engine, runs a short probe: questions about fictional things,
answerable only through a stub search tool, as ordinary agent runs. Two defaults follow from it, and
only when you left them unset. A model that answers from memory while holding a search tool has its
information-retrieval tools made must-call: a run that answers without calling one is sent back once
and made to call. A model whose native calls the server does not carry back is run in the ReAct text
frame. Calculator and code tools are never moved by the probe.

The result is stored in `~/.effgen/capabilities.json` (or under `$EFFGEN_HOME`, or at
`$EFFGEN_CAPABILITY_CACHE`) and read by every later agent, across processes, without a request; it
is measured again when the weights, the chat template, the endpoint or the probe change, and after
30 days. A probe is bounded at 48 requests and 120 seconds; one that cannot run stores nothing,
leaves `auto` as declared and logs one warning. Its requests are made by the first agent, in its
process. `response.metadata["tool_calling"]` says which strategy a run used and why.

The same store keeps what the framework learns from real requests. A provider that rejects stop
sequences beside tool definitions is asked once more with the stops applied locally, and later
requests leave them off; in 1.2.0 the run reported the provider's HTTP 400. A server that rejects
`reasoning_effort` is asked once more without it.

In testing, a small served model that answered from memory while holding a web-search tool answered
14 more questions in 100 correctly, at 1.6 more model calls a run.

*Migration:* `AgentConfig(capability_probe=False)` or `EFFGEN_CAPABILITY_PROBE=0` resolves `auto`
from the model's declaration alone, as in 1.2.0; so does an explicit `tool_calling_mode` or
`tool_use`. First-party cloud adapters are never probed. A test suite that scripts a served endpoint
sees the probe's requests first unless it sets one of these.

#### 7. `reasoning_effort` reaches a model behind `base_url`, and Groq

A `reasoning_effort` you set is now sent by the OpenAI-compatible adapter, whatever the model is
called, and by the Groq adapter for a model its catalog marks as reasoning. In 1.2.0 both dropped
it. A server that refuses the field costs one retry, once, and is remembered (change 6). Every other
adapter that drops a `reasoning_effort` you set now says so once per model at WARNING level.

#### 8. One agent serves overlapping conversations without mixing them

`run(session=...)`, `run_async(session=...)` and `stream(session=...)` hold their conversation on
the call, not on the shared agent. Overlapping calls on one agent — threads, `run_async()` tasks,
streams — each read and record only their own session. In 1.2.0 they swapped `agent.session` and
`agent.short_term_memory` on the shared object, so overlapping calls could read and record each
other's turns, and the agent could be left on another conversation's session after the calls ended.
Inside a call, `agent.session` and `agent.short_term_memory` still name that call's conversation;
outside one, they are the agent's own. Per-call middleware applies to that call only.

`stream()` now honours `session=`; in 1.2.0 it ignored it and wrote the turn to the agent's own
memory. A run whose input a guardrail blocks leaves the agent's session as it was.

In testing, a reproduction with overlapping calls on one agent returned 94 answers carrying another
conversation's turn, and lost 118 turns, on 1.2.0 in a single scenario; 1.3.0 mixed none in eleven.

*Migration:* none. Calls without `session=` still share the agent's own memory, as before.

#### 9. A streamed turn is saved to the agent's bound session

On an agent created with `session_id=`, or given `agent.session`, `stream()` now appends each
answered turn to that session and saves it, as `run()` always did. In 1.2.0 streamed turns stayed in
memory and were missing from the session file, so a later process continuing the conversation never
saw them.

#### 10. Smaller changes

* `effgen code --json` reports a `tool_failed` run as stopped, as the library does.
* The OpenAI SDK's response types are built once when the adapter loads. In 1.2.0 the first
  concurrent streams in a fresh process could fail with
  `'BaseModel' has no attribute '__pydantic_core_schema__'`.
* The agent's circuit breaker keeps every first failure when several calls fail on one tool at once,
  and no longer raises `dictionary changed size during iteration` under concurrent writers.

---

### Added

**Two new names** — `from effgen import ToolCallingProbe, probe_tool_calling`:

- `probe_tool_calling(model, refresh=False)` — measures what a loaded model does when handed a tool,
  or returns what was measured before. It returns a `ToolCallingProbe`, or `None` for a model that
  is not probed (a first-party cloud adapter) or a probe that could not run.
- `ToolCallingProbe` — the stored result: how many runs resolved, went unresolved or skipped the
  tool in the native frame (and in the text frame, when that was measured), the strategy and the
  must-call tool categories `auto` derives from them, and what the probe itself cost in requests,
  tokens and time.

**Configuration and results:**

- `AgentConfig.capability_probe` (default `True`) and the environment variables
  `EFFGEN_CAPABILITY_PROBE` and `EFFGEN_CAPABILITY_CACHE`.
- `AgentResponse.termination`, and the tuple `TERMINATIONS` in `effgen.core.agent`; the stop reason
  `tool_failed`; the metadata keys `tool_results`, `unavailable_tools`, `tool_calling` and
  `answer_source`.
- `stream(session=...)`.
- `BaseModel.capability_key()` and `BaseModel.forwards_reasoning_effort()`.

**`effgen doctor`** lists what each served or local model was measured to do with a tool and what
was learned from its endpoint, and `--json` carries the same. `--probe MODEL` measures one model now
— a local model, or one served at `--base-url URL` (`--api-key-env VAR` names the variable holding
the endpoint's key) — and `--refresh` measures it again.

```python
from effgen import AgentConfig
from effgen.core.agent import TERMINATIONS

config = AgentConfig(model="Qwen/Qwen2.5-1.5B-Instruct", base_url="http://127.0.0.1:8000/v1")
print(config.max_turns_without_progress)   # 2: a run with no new result is asked for its answer
print(config.recover_lost_tool_calls)      # True: a broken tool call is read before it is reported
print(config.capability_probe)             # True: a served or local model is measured once
print(TERMINATIONS)                        # the values response.termination can take

as_in_1_2 = AgentConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    max_turns_without_progress=None,
    recover_lost_tool_calls=False,
    capability_probe=False,
)
```

```bash
effgen doctor
```

---

### Known issues

These are open. Each is understood well enough to say what it is.

1. **Small models given a search tool cost more.** When the probe finds that a model answers from
   memory while holding a search or retrieval tool, every run holding one is made to call it. On the
   small model measured, that made a run about three times as many model calls on question-answering
   tasks, and across all the tasks measured on that model, 1.5 times the model calls and 1.9 times
   the wall time of 1.2.0. `tool_use="auto"` or `capability_probe=False` turns it off for an agent.
2. **A run whose calls keep failing on their own input ends stuck, not typed.** A tool that rejects
   the same input twice, or every input in the same words, is stopped by the loop guard before the
   four-failure bound, so the run ends `loop_detected` (`termination == "stuck"`) rather than
   `tool_failed` with `kind="input"`.
3. **A run whose tools all fail on their own side gets no closing request.** It ends `tool_failed`
   at once (change 2), even when the model had already written what it would answer. Whether such a
   run should be asked once, as a run whose tool was withdrawn for bad input is, is open.
4. **On one coding task set, a mid-size model answers less often correctly.** 5.4 points fewer
   correct answers, short of significance once corrected for the number of comparisons, while its
   correctly written final answers rose and its model calls fell by a quarter. The difference is
   runs that 1.2.0's loop guard ended with a usable result; 1.3.0 asks them for their answer, and
   some answer from the wrong result.
5. **The framework's own time on a long run grew.** On a 100-step run it is 9.9% higher than 1.2.0's
   (a median of 302 to 332 ms), within its budget. A batch of 16 runs is flat at about 19 ms of CPU
   a run.
6. **A probe's requests are booked in the first agent's cost ledger** with no tag marking them as
   the probe's.
7. **The readers now on by default have edges.** The lenient reader reads `"print("")"` as
   `print()`. A truncated call naming the code tool itself still runs the fenced block before it. A
   closing reply on the text scaffold that is another call ends the run stuck after one request.
8. **A Groq call's retry wait is booked as framework time** in the run ledger.
9. **One agent, many sessions, at the edges.** `stream()` runs no middleware. `resume(session=...)`
   restores the checkpoint's memory onto the agent's own memory. A run made while a stream is held
   open writes its trace to a tracker of its own, so `agent.execution_tracker` does not show it.
10. **Not aimed at by this release, and not re-checked:** known issues 3 to 11, 13 and 14 of 1.2.0,
    and the Groq free-tier HTTP 413 in its issue 12.

---

**What it cost.** Against 1.2.0, on the same tasks and the same samples, with two models served locally. The larger
model measured makes 3% fewer model calls per run, sends 4% fewer prompt tokens and takes 7% less
wall time, and answers more questions correctly, outside run-to-run noise. The smaller model makes
52% more model calls, sends 70% more prompt tokens and takes 94% more wall time; nearly all of that
is the search tool being made must-call on question-answering tasks, where calls tripled. On its
tasks that hold no search tool it makes 7% more calls and takes 11% more wall time, and answers more
questions correctly, outside run-to-run noise. No task got worse by more than run-to-run noise
allows once corrected for the number of comparisons. On a further set of task types kept out of this
release's development, accuracy rose on both models, all of it on the type whose tool breaks or
keeps failing, the shape this release targets; on the other types it is flat, and those runs cost
less on both models. Groq and Gemini were rate-limited throughout and no cloud model was measured at
full size, so nothing here is a claim about a cloud provider.

**Full changelog:** [CHANGELOG.md](CHANGELOG.md#130---2026-10-01)

## v1.2.0 - September 27, 2026

This release is about what a run costs, and whether you can see it: every run now keeps a ledger
of its calls, tokens, cost and time.

The main things: `response.ledger` says what a run spent and where its time went — waiting on the
model, on tools, on the caller, on child runs, or inside the framework — and it matches what goes over the wire. A tool result the model writes itself is
never taken as the answer. A provider's prompt cache is kept warm, and its hits are read and priced
where the provider reports them. A request carries less of the framework's own text, and
`reasoning_effort` reaches a blocking run. A model you serve yourself reads as unpriced rather than
free, and a spent cap no longer refuses it. Many agents in one process no longer queue on the shared
spend ledger. And `effgen bench` measures an agent on your own tasks, with a noise band beside every
difference.

Runs that use tools still make more model calls than they need to; reducing that is the main work
of the next release, and what this one cost against 1.1.0 is set out at the end.

Fourteen changes are visible to existing code, and they are listed first.

The public surface grew from 250 names to 251. Nothing was removed or renamed.

```bash
pip install --upgrade effgen
effgen --version
```

### Read this before upgrading: fourteen changes you will notice

#### 1. A tool result the model writes itself is never taken as the answer

On a model with native tool calling, the default hybrid strategy also reads a tool call the model
writes out as text (`Action:` / `Action Input:`). A model that does that can carry on and write the
tool's result itself; taken as the answer, that invented result would end the run with
`stop_reason="final_answer"`, no tool call recorded and a tool that never ran. Two things stop it.

* A turn that holds tools and whose reply may be read as text is sent one stop sequence,
  `"\nObservation:"`, so generation ends where the tool's result would begin. 1.1.0 sent four labels
  (`"\nObservation:"`, `"\nQuestion:"`, `"\nHuman:"`, `"\nUser:"`) on such turns. A tool-free turn,
  and an agent whose reader reads only native calls, are sent none. A caller's own `stop_sequences`
  replace it.
* When a reply holds a written action anyway — a model that ignores the stop, or a provider that
  cannot take one — the action runs and whatever the model wrote after it is discarded, logged as
  `[reader] a written action is run and what the model wrote after it is discarded`.

`BaseModel.supports_stop_with_tools()` is new and answers `True`. An adapter for a provider that
rejects `stop` beside `tools` answers `False`; the framework then sends no stop sequence and cuts the
returned text itself.

In testing, no run was answered from an invented result, including runs replayed from recordings in
which the model had written one. A long answer containing a line that begins
`Question:` is no longer cut there.

*Migration:* none. A caller who relied on `"\nQuestion:"`, `"\nHuman:"` or `"\nUser:"` stopping a
tool-holding turn passes them in `stop_sequences`.

#### 2. `reasoning_effort` reaches `run()` and `run_async()`

In 1.1.0 it reached a streamed turn and not a blocking one. `run()`, `run_async()` and `stream()` now
all send it when the caller passes it and the model declares that it reasons; on the wire all three
paths carry it, and so does the blocking follow-up turn inside a stream. On a reasoning model this
changes how long the answer is. On any other model nothing changes, because the adapter drops it.

#### 3. The output budget follows what the run declared

When the caller pins no `max_tokens`, a run that declares an `output_schema` asks for a budget sized
from the schema — a one-integer schema asks for 256 tokens rather than 1,024 — and a model that
declares it reasons is never sent fewer than 4,096. A published maximum caps either. The native-tool
and structured-output paths go through the same rule. An explicit `max_tokens`, on the call or on
`AgentConfig`, still wins on every call the run makes; the calls that repair an answer to fit its
schema afterwards are sized from the schema, as they took their own budget before.

Where a model would have written more than a schema-sized budget, the answer is now cut at the
budget with the existing typed truncation message. The budget and the OpenAI adapter's stop decision
read one declaration of whether a model reasons; where a model is judged from its name instead, that
is logged, naming the model.

#### 4. A request carries less of the framework's own text

* **The tool contracts no longer ask for text the task did not ask for.** Two clauses went with that:
  the lookup contract's "if what comes back does not answer the question, say so and name what is
  missing", and the computation contract's instruction to work the task out step by step and finish
  by stating the answer. Every run with tools is affected; a run with no tools or with
  `tool_contract=""` is not.
* **The tool contract comes before the caller's task** on every frame, so the task is the last thing
  the model reads, followed only by an answer style when one is set.
* **Tool rules are stated once.** Where the system prompt in force is the one the framework
  generated, the text scaffold no longer repeats its five tool rules. The line pointing at earlier
  conversation appears only when there is earlier conversation.
* **A repeated tool result is sent once.** A result identical to one the request already carries,
  from the same tool with the same arguments, is written once and then referred back to. Only the
  rendering changes: the step keeps the whole result, so nothing stored, checkpointed or read back
  differs.
* **How fully a tool is described comes from the adapter**, through the new
  `BaseModel.prompt_detail()`, rather than from substrings of the model's name.

Replayed over recorded runs, requests are 2% to 22% shorter, with every answer and every call count
unchanged. Without the lookup clause, a model given a search tool uses it more often: it makes more
model calls and answers more questions correctly. Restoring the computation clause made no
measurable difference to accuracy, so it stays out.

*Migration:* none. `AgentConfig(answer_style=...)`, under Added, is the way to ask for a shorter or a
fuller answer.

#### 5. On a provider with a prompt cache, a run keeps one request shape

* **A run with tools sends its conversation as messages from its first run** when memory is on (the
  default), `prompt_protocol` is `"auto"` (the default), and the provider's adapter declares a prompt
  cache and takes messages. In 1.1.0 only a run continuing a session did. The session's next
  question then extends the cached prefix instead of starting a new one, logged as
  `[cache] the session keeps one request shape: this first run sends the conversation as messages, so the next question extends its prefix instead of restarting it`.
  On a self-hosted server with prefix caching it made no measurable difference, because nearly every
later prompt was already served from the cache.
* **The turn that asks for the answer keeps its shape.** When the guards stop offering tools, an
  adapter that can forbid a call gets the same message list and tool definitions with
  `tool_choice="none"`. The prefix that turn shares with the one before it went from 0.03% to 82%.
* **`AgentConfig.cache_system_prompt` and `cache_tools` now work on Anthropic.** The tool
  definitions, the system prompt and the last completed step carry cache breakpoints.
* **Cached tokens are read** on Groq, Together, Fireworks, Cerebras and Gemini, which reported 0
  before, and priced at the provider's cached rate where the model catalog carries one. The ledger
  gains `cache_write_tokens`. Through Fireworks, nearly all of a run's later prompt
tokens were served from the cache, and the ledger matched the provider's own counts.

*Migration:* none. A reader that sums prompt tokens sees no change: cached tokens are part of
`prompt_tokens`, not added to it. `AgentConfig(prompt_protocol="flat")` keeps a run's own steps in
one string.

#### 6. The loop stops fewer runs that are still finding things

* **A tool whose calls keep returning new results is no longer withdrawn after 12 calls** (16 for a
  data-processing tool). It is withdrawn after that many calls in a row that brought nothing new; a
  run whose calls keep finding things is bounded by `max_iterations`, so with a high ceiling it can
  make more calls than before. A scripted run whose every one of 20 calls
  returned something new was stopped at 12 with `loop_detected`; it now makes all 20 and answers. A
  run that only repeats itself ends exactly as before.
* **`run(max_iterations=N)` now moves the loop's own thresholds too.** They used to read the
  configuration's value, so raising it for one call did not stop a tool being withdrawn early.
* **`Action: None` is read as "no action"**, not as a call to a tool named `None`, and logged as
  `[progress] the turn declared no action, which names no tool`.

#### 7. Groq hands a tool call it could not parse back to the loop

When a request carried tools and Groq answers that it could not parse the model's tool call, the
adapter returns the call as the model wrote it, in the `<tool_call>` form the loop reads, instead of
failing the run. The loop runs the call if it can read it and otherwise asks for it again. The call's
usage is estimated and recorded, and logged as
`Groq rejected a tool call it could not parse; the call is handed back as the model wrote it`.

#### 8. A spent cap refuses only calls that cost money, and says so with a typed error

Once a configured spend cap is spent, a call to a local engine, a free tier or a server reached with
`base_url=` is no longer refused.

A refused call now raises `BudgetExceededError` from `run()` whatever `raise_on_error` says. In 1.1.0
it raised `RuntimeError("BudgetExceededError: …")` or came back as a failed response.
`BudgetExceededError` is not a `RuntimeError`. `stream()` raises it; `run_batch()` raises and stops;
the server answers HTTP 429 with `budget_exceeded`; `effgen run --json` prints the error document
instead of the run document. No request reaches the provider.

*Migration:* catch `BudgetExceededError` (from `effgen`) where you caught `RuntimeError`.

#### 9. A model with no published price reads as unknown, not as free

* `CostTracker.total_cost()` returns `None` when every call it covers was unpriced; it returned
  `0.0`. `CostEvent.cost_usd`, `RunLedger.cost_usd`, `total_cost_usd` in `effgen cost --json`, and
  `cost_usd` in the run executions and topology can all be `None` for the same reason. A free model
  still reads `0.0`.
* A server reached with `base_url=` records as `openai_compatible` — in `response.provider`, the run
  store, the cost tracker, the ledger file and the Prometheus labels — where 1.1.0 recorded `openai`.
  An existing ledger file shows the same served model under both names across the upgrade.

Every consumer of those values in the package was run on an unpriced run: none raised, and none
showed `$0` where 1.1.0 showed it in three places.

*Migration:* treat `None` as "not priced" wherever you add up cost.

#### 10. Every run keeps a ledger, and two totals changed with it

`response.ledger` and `response.metadata["ledger"]` carry the run's `RunLedger`; the flat metadata
keys are unchanged. Children — sub-agents, workflow nodes, team members, an agent run inside a
tool — are attached once, and `total()` adds them up.

* `effgen_model_call_latency_seconds` now observes each model call — its count is the number of
  calls and its value each call's wait. It observed each run's whole wall time once.
* A decomposed run's `tokens_used` includes its sub-agents' tokens: 48 where it read 16 on a
  three-call example. `effgen_tokens_used_total` takes only the run's own calls, so nothing is
  counted twice.
* A synchronous `@tool` runs in the caller's context, so an agent it starts is a child of the run and
  its cost is part of the run's total.

A run against a model you serve yourself, and what it spent:

```python
from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    tools=[Calculator()],
))
response = agent.run("What is 17 * 23?")
ledger = response.ledger

print(response.output)
print(ledger.llm_calls, ledger.tool_calls, ledger.prompt_tokens, ledger.completion_tokens)
print(f"model {ledger.model_wait_s:.2f} s, tools {ledger.tool_wait_s:.3f} s, "
      f"framework {ledger.framework_s * 1000:.1f} ms")
print(ledger.cost_usd)   # None: a model you serve yourself has no published price
```

#### 11. A run's reported time includes the work after the model's last answer

`execution_time` now includes `after_run` middleware, the session save and the final checkpoint, and
equals the ledger's `wall_s`. With that work slowed on purpose, 1.1.0 reported 250 to 558 ms less
than the caller waited; 1.2.0 reports it to within a millisecond.

#### 12. The spend ledger stops growing at 250,000 rows

At 250,000 rows the ledger file folds its oldest rows into per-model totals, keeping every total
exact. `EFFGEN_COST_MAX_ROWS` sets the ceiling, and `0` keeps every row. Opening an existing file adds
two columns, `calls` and `unpriced_calls`, and rows in `effgen cost --json` gain
`unpriced_requests`. Over 72 simulated hours and 1,036,800 calls the file held at most 242,245 rows
and 33.5 MB, flat for the last 18 hours, with every total exact; 1.1.0 kept all 1,036,800 rows and
reached 133.5 MB, still growing. A write that triggers a fold waits for it, about 0.2 to 0.3 s once
every 50,000 or so calls.

#### 13. A server named with `openai:` gets the id without the prefix, and a streamed call keeps its own arguments

* `AgentConfig(model="openai:<id>", base_url=...)` now sends `<id>`. 1.1.0 sent `openai:<id>`, and
  the server answered that the model does not exist.
* Concurrent streamed runs no longer hand a tool another stream's arguments. With 64 streamed agents
  on one model, 102 of 115 calls carried another agent's argument before; 0 of 64 do now. A streamed
  argument that arrives broken is now exactly what the server sent.
* In-process local engines serve concurrent agents: eight agents sharing one in-process vLLM engine
  finished in 2.07 s where they aborted before, and concurrent GGUF runs no longer crash. A GGUF run
  reuses its cache across turns, evaluating 609 prompt tokens per run where it evaluated 1,173.

#### 14. An out-of-budget failure says what to do and is not retried

A reply cut off by its token budget, or one that spent the whole budget reasoning, now carries its
own guidance and is not marked retryable. It used to read "Unexpected provider error" and be retried.

---

### Added

**One new name** — `from effgen import RunLedger`:

- `RunLedger` — what one run spent and where its time went: model and tool calls, prompt,
  completion and cached tokens, cost, and wall time split into model, tool, caller, child and
  framework time, per run and per iteration

**A new command, `effgen bench`:**

- `effgen bench init` — writes a starter suite
- `effgen bench run SUITE` — runs a suite of your own tasks against a model and prints accuracy
  beside LLM calls, tool calls, tokens, time and cost; saves the run
- `effgen bench compare A B` — pairs two saved runs task by task and prints a noise band beside
  every difference

```bash
effgen bench init
effgen bench run bench-suite.yaml --model Qwen/Qwen2.5-1.5B-Instruct --base-url http://127.0.0.1:8000/v1 --out runs/a
```

The command is built on the `effgen.bench` package, which is importable and not part of
`effgen.__all__`. [`docs/cli/bench.md`](https://github.com/ctrl-gaurav/effGen/blob/main/docs/cli/bench.md)
documents the suite format.

**How a run is asked to answer:**

- `AgentConfig.answer_style`, and `answer_style=` on `run()`, `stream()` and `run_async()` — one
  line about the form of the answer, stated last: `"brief"` ("Answer in the form the question asks
  for, and nothing else."), `"full"` ("Explain your reasoning in the answer."), or your own
  sentence. The default states nothing: a brief-answer line on every run shortens answers but also
  makes a model answer from what it knows instead of using the tool it was given.
- `AgentConfig.max_turns_without_progress`, default `None` — after that many turns in a row that
  brought no new result, the next turn offers no tools and asks for the answer.
- `AgentConfig.recover_lost_tool_calls`, default `False` — reads a tool call written with raw line
  breaks or Python-style quoting, and asks once more, with a call required where the provider
  enforces one, for a call that could not be read at all.

Both of the last two can be set for one call as `run()` keywords, and a child run inherits them.

```python
from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano", answer_style="brief")
print(config.answer_style)                 # brief: one line, stated last
print(config.max_turns_without_progress)   # None: off unless you set it
print(config.recover_lost_tool_calls)      # False: off unless you set it
```

Also new, on types that already existed: `AgentResponse.ledger`, `Agent.last_stream_ledger`,
`Checkpoint.ledger`, `BaseModel.prompt_cache_policy()`, `BaseModel.prompt_detail()`,
`BaseModel.supports_stop_with_tools()`, `SQLiteCostStore.flush()`, `CostEvent.calls` and
`.unpriced_calls`, `cache_write_tokens` in the ledger, the run-store fields `llm_calls`,
`tool_calls`, `cached_input_tokens`, `model_wait_s`, `tool_wait_s` and `framework_s`, the Prometheus
series `effgen_run_framework_seconds`, `effgen_model_cost_usd_total` and
`effgen_model_unpriced_calls_total`, the run card's "Model calls" and "Framework time", and the
environment variable `EFFGEN_COST_MAX_ROWS`.

---

### Known issues

These are open. Each is understood well enough to say what it is.

1. **One `Agent` serving concurrent `run(session=...)` calls mixes the conversations.** With many
   sessions in flight on one agent, 27 to 29 of 48 answers carried another conversation's turn. This
   was already true in 1.1.0. Use one agent per concurrent session.
2. **A provider that rejects `stop` beside `tools` answers HTTP 400 through a stock adapter.** Until
   its adapter declares `supports_stop_with_tools()` as `False`, the run reports the provider's 400;
   it never returns an invented answer. The framework does not yet retry without the stop sequence.
3. **Four edges of change 1.** A written action whose argument contains the text `Final Answer:` is
   cut there (not seen in recorded runs). A tool-holding agent's planning and direct
   calls, which carry no tools, are sent the observation stop too. The speculative path ignores
   `supports_stop_with_tools()`. A reply that writes an `Action:` line naming no tool and then an
   answer is read as a call to that name, so the run is told there is no such tool and takes one
   more turn to answer (rare in recorded runs). None of these can make an invented result the
   answer.
4. **Streamed runs are missing from Prometheus.** The stream path records no Prometheus series, so
   the new cost and unpriced-call series leave streamed calls out: over three runs and one stream on
   Groq, the tracker read $1.295e-4 and Prometheus $1.054e-4.
5. **The run store and Prometheus record a run's time before its post-run work**, so they can read up
   to 558 ms less than `execution_time`.
6. **A few surfaces still print `$0` for unpriced work.** The chat and `effgen code` `/cost` commands
   print `$0.00` for a session of unpriced turns, and the monitor's spend panel shows `$0.000000`
   beside "1 model(s) excluded from the total". `effgen cost prune` counts rows as events, though
   after a fold one row can stand for many calls. The post-call budget check logs "spend cap: refused
   a call" for the call that crossed the cap, which was made and billed, and on the command line the
   `spend cap:` warnings print above the error panel. Spans and the usage log still name `openai` for
   an `openai_compatible` call.
7. **A streamed tool argument the server broke still reaches the tool.** On a served model, a coding
   task is answered correctly less often when streamed than when blocking: nearly every difference is
   an argument the server's streaming parser broke, which the blocking path's text reader recovers.
8. **Cached tokens are priced at the cached rate only where the catalog carries one** — OpenAI and
   Anthropic today. On Groq, Gemini and Fireworks the hits are counted and billed in effGen's estimate
   at the full input rate. `pricing_status("fireworks", "gpt-oss-120b")` reads unpriced while the
   fully-qualified id is priced, and some bundled catalogs carry `0.30000000000000004` for an Anthropic
   cached rate.
9. **Prompt caching has gaps.** A session with no tools gains nothing from change 5. Anthropic does
   not declare that it can forbid a call, so the answer turn's shape repair does not run there, and a
   `tool_choice` word on the run path reaches the Messages API untranslated. Once any agent in a
   process meets a provider that answers a forbidden call with a call anyway, every later agent
   against that model and endpoint drops its tool definitions on the answer turn.
10. **The turn that asks for the answer on the text-scaffold fallback still narrates the run's own
    calls.** It is 17.3% shorter than in 1.1.0 (2,651 to 2,193 characters in a recorded example), not
    gone. It applies to a provider that ignores a request to forbid a call.
11. **Local engines.** On a host whose driver lists a GPU that torch cannot start,
    `engine="auto-fast"` now picks vLLM and falls back to Transformers only after that load fails.
    `load_model(<id>, base_url=...).generate(prompt)` with no configuration asks for the whole
    context as its completion budget, which vLLM refuses. A fork while threads write the spend ledger
    hangs the child, as it did in 1.1.0. The Transformers engine answers concurrent agents one at a
    time and keeps no cache of the shared prompt between turns.
12. **Reasoning models, in two places.** On Groq's free tier, when a reply is truncated, the
    framework's escalation asks for 8,192 tokens, which the free tier refuses with HTTP 413; pin
    `max_tokens` below it. A reasoning model reached with `base_url=` gets the OpenAI-compatible
    adapter, which does not declare that it reasons, so `reasoning_effort` is not sent there; 1.1.0
    behaved the same.
13. **`effgen bench`** has no way to pass `context_length`, so every run against a served model logs
    the adapter's warning; its compare prints "not every paired task was priced" when none was; and
    the documentation site has no page for it yet.
14. **Carried unchanged from 1.1.0, not aimed at by this release:** `ToolCall.arguments` is a string
    on the ReAct path and a mapping on the native path; a tool-free stream yields the model's own
    text rather than the sanitized answer; `GuardrailChain.check(position=...)` is not forwarded; and
    the retrieval and open-ended gaps 1.0.1 recorded.

---

**What it cost.** Measured against 1.1.0 on the same tasks and models, accuracy holds: every task measured again after
the final fix is within its run-to-run noise. The cost moved both ways. Answers that need no tool
are far shorter, with 82% to 93% fewer completion tokens on short-answer questions. Arithmetic and
math tasks that use a calculator write 7% to 15% fewer completion tokens but, on three of the four
measured, make 16% to 27% more model calls and send 10% to 22% more prompt tokens; a coding task on
the larger model makes 14% more calls. Question-answering tasks with a search tool search more often
and answer more questions correctly, at up to half again as many model calls. Savings in model calls
measured part-way through the release came partly from the defect change 1 fixes, and are not
claimed. Tool-using runs still make more model calls than they need to, and reducing that is the
focus of the next release. No cloud model was measured at full size, so nothing here is a claim
about a cloud provider.

**Full changelog:** [CHANGELOG.md](CHANGELOG.md#120---2026-09-27)

## v1.1.0 - September 14, 2026

This release changes how a run holds its conversation: as typed steps, instead of one string that
grows with every turn.

The main things: a finished run hands back what it did, as steps that the command line, the run
card, the debug inspector and the dashboard all render the same way. A run is bounded by the prompt
tokens it may send, and gives up its oldest material first instead of failing at the provider. A
saved run resumes where it stopped instead of restarting the task. There is one agent loop instead
of three, so a streamed run sends what a blocking one sends. And a run that continues a session
sends that conversation as the messages it was, on a model that takes them.

Eleven changes are visible to existing code, and they are listed first.

The public surface grew from 225 names to 250. Nothing was removed or renamed.

```bash
pip install --upgrade effgen
effgen --version
```

### 1. A run carries its conversation, and `response.metadata` is no longer plain data

`AgentResponse.thread` is the run's `AgentThread`, or `None` for a run that recorded none.
`response.metadata["thread"]` holds the same object.

The thread opens with the run's frame: a `SystemStep` when tools are attached, and then always a
`TaskStep`, before any step the model produced. Code that reads `thread.steps[0]` expecting the
first thought will find the frame there instead.

`to_text()` is unchanged and still renders the transcript a 1.0.x reader would recognise.
**`to_dict()` is the documented serialisation**: `json.dumps(response.metadata)` raises on the
thread object, because `metadata["thread"]` is the live object that `thread.delegations()`,
`.child_threads()` and `.to_text()` are called on. `json.dumps(response.to_dict())` works and
writes the thread through its own serialisation.

*Migration:* serialise through `response.to_dict()`, or `response.thread.to_dict()`, rather than
dumping `response.metadata` directly.

A run against a model you serve yourself, start to finish:

```python
from effgen import Agent, AgentConfig, thread_as_text

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
))
response = agent.run("What is 17 * 23?")

print(response.output)
print(thread_as_text(response.thread))        # the run, step by step
print(response.metadata["context_budget"])    # what it was allowed to send
```

`thread_as_text()` is one block of text; `render_thread()` hands back the same steps one at a time,
with a position, a kind, a label, a body and a depth, so a caller can lay them out itself. Both
redact by default and omit tool-call ids unless asked. The same rendering is on the command line as
`effgen run --show-thread`, and the steps themselves are in `effgen run --json` under
`metadata.thread`.

```python
from effgen import AgentThread, TaskStep, AnswerStep, render_thread, thread_as_text

thread = AgentThread(steps=[TaskStep(text="What is 17 * 23?"), AnswerStep(text="391")])
for step in render_thread(thread):
    print(step.position, step.kind, step.label, "|", step.body)
print(thread_as_text(thread))
```

`AgentThread.to_text()` is a different thing and is unchanged: it renders the run's **working** as
the flat transcript a 1.0.x reader would recognise, which for a run that answered without using a
tool is empty.

### 2. A session's earlier turns render differently inside the prompt

A run continuing a session used to paste its history into the prompt under
`=== Previous Conversation Context ===` with `[Turn n]` markers. It now renders as
`Earlier in this conversation:` followed by `User:` / `Assistant:` lines. The old block is gone.

That rendering is for a request that carries one string. On a model whose adapter takes a
conversation, a run whose tools travel as a request parameter puts neither the history nor a
persona inside the prompt: the session's earlier turns go out as their own `user` and `assistant`
messages, and a caller's `system_prompt` goes out as the system message. That holds on every request
of the run, including the one that asks for the answer after the guards stop offering tools, and
whatever `prompt_protocol` says — the protocol decides how the run's own steps travel (change 11). A
run with no tools, a run whose tools are written into the prompt, a caller's own
`system_prompt_template` and a model whose adapter takes one string keep the rendering above.

*Migration:* only code that matched on that header text, or that reads the request a provider
receives, is affected. The history itself is unchanged, and `Session.last_thread()` reads the steps
directly. No `prompt_protocol` value sends such a run as one string on a model that takes a
conversation; `tool_calling_mode="react"`, which writes the tools into the prompt, does.

### 3. `AgentConfig(guardrails=...)` accepts a plain list and rejects what is not a guardrail

A list of guardrails no longer has to be wrapped, and a non-guardrail in the list raises
`TypeError` at construction instead of failing later inside a run.

### 4. A 1.0.x checkpoint still resumes, and what the reconstruction drops is documented

`Checkpoint` has a new `thread` field carrying the same data as `response.thread.to_dict()`, and it
writes the flat transcript beside it under `scratchpad`, so a file this release writes still
resumes on a build that only knows the transcript. A file written by 1.0.x has no steps, so
`Checkpoint.to_thread()` rebuilds them from the transcript and logs
`[compat] rebuilt a thread from a flat transcript`.

**Four things the rebuild cannot recover**, because the transcript never held them:

* the provider's own call id — the recovered call is named from its position in the run;
* a tool's arguments as values; they come back as the rendered string they were printed as;
* whether a line was the framework's own, so an injected nudge is indistinguishable from a thought;
* the whole frame — the persona, the tool contract, any earlier turns, and the task itself.

Measured on runs a release really wrote: a 1.1.0 checkpoint round-trips **12/12 as text, 12/12 as a
dict and 12/12 in its step kinds**; the same runs stored the way 1.0.x stored them come back
**12/12 as text and 0/12 as a dict**. Both directions were checked against real files — a
checkpoint written inside `v1.0.1` resumes here and completes, and a checkpoint this release writes
was resumed by the released `v1.0.1` and reached the same answer.

### 5. `agent.resume()` continues an unfinished run instead of restarting the task

The final checkpoint stores the run's steps where it used to store `scratchpad=""`, so resuming
picks up the conversation the run had. On a fixed seed at temperature 0, a resumed run makes fewer
model calls than an uninterrupted one because it does not redo the turns the checkpoint holds: over
ten runs on `openai:gpt-5-nano`, **2.7 → 1.0** mean model calls, the same answer 10/10 and the same
wording 10/10. An independent probe resumed mid-loop on 5/5 runs there, 4/4 on
`gemini:gemini-3.1-flash-lite` and 3/3 on `groq:openai/gpt-oss-20b`, with equal or fewer tool calls
on the resumed arm — where a 1.0.x build resumed mid-loop on none of them. A larger n on those two
families is not measured: both rate-limited.

### 6. One agent loop, so `stream()` now behaves like `run()`

The ReAct loop was written three times: once for `run()`, once for the prompt-scaffold branch of
`stream()`, and once for the branch that dispatches a provider's streamed tool calls. There is one
now, in a private module, and streaming is a decision taken at the end of a turn rather than a fork
taken before the first prompt.

**If your code called `run()`, nothing changed.** Replayed over a recorded 366-run corpus against
the previous tree: **366/366 runs identical in every compared field and 1,628/1,628 prompts
byte-identical**.

**If your code called `stream()`, it changed — in its favour.** Measured before and after on the
same profiles, a streamed run's first prompt now matches `run()`'s on 45 of 45, where 31 matched and
12 differed, and every turn's prompt and the tool definitions sent with it match on 65 of 65. The
sampling fields that differed from `run()` went from 6 of 9 on the text stream and 1 of 9 on the
native stream to none. The loop guards fire on a streamed run exactly as on a blocking one, where
two of three profiles never reached them before. 42 of 45 streamed runs now carry an
`AgentResponse`, against 13 before; the three that do not are tool-free streams, which record
nothing by design. Output guardrails are checked twice, as `run()` checks them, and a block raises
from the iterator.

*Migration:* a streamed run that was relying on the old behaviour — different sampling settings, a
guard that never fired, an output guardrail that was never checked — will now behave as the
blocking path always did. A streamed run is also now stoppable by an output guardrail, which raises
out of the iterator.

### 7. A run is bounded by what it may send

`AgentConfig.context_budget` (`"auto" | int | float | None`, default `"auto"`) and
`AgentConfig.compaction` are new. `AgentConfig.max_context_length` — declared since 1.0 and read by
nothing — is now the window override. The budget is `(window − output reserve) × 0.85`, and it is
**unbounded when the model declares no window**, because guessing one would silently truncate a
conversation that would have fitted.

`response.metadata["context_budget"]` is reported on every outcome, including a run that used no
tools. When the conversation will not fit, the run gives up the oldest material first — an old tool
result is shortened, then an old thought dropped, then whole answered cycles replaced by one
`NudgeStep` — and never touches the frame, the task, the most recent two complete cycles or the
answer. A call and the result answering it always leave together. `ObservationStep` gained
`compacted` and `original_chars` so a shortened result says so.

`Session.keep_thread_history` defaults to `False`: an earlier turn's stored steps are reduced to
their shape, and reading that turn's thread hands back an empty one. This is what keeps session
files from growing — 12 turns went from 78,281 to 27,175 bytes, and 40 turns from 470,518 to
80,877.

A prompt larger than the model's window is now classified non-retryable, so it is sent once instead
of three times.

```python
from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano")
print(config.context_budget)        # auto — bounded by the model's own window
print(AgentConfig(model="openai:gpt-5-nano", context_budget=8000).context_budget)
print(AgentConfig(model="openai:gpt-5-nano", context_budget=None).context_budget)
```

*Migration:* none required; `"auto"` reproduces 1.0.x behaviour on any conversation that already
fitted. Pass `context_budget=None` for the old unbounded behaviour, or an integer to name a budget
yourself. Set `Session(keep_thread_history=True)` if you read the stored steps of earlier turns.

### 8. Orchestration results carry threads

* a run with **no tools** now reports `response.metadata["thread"]`, where it reported none; its
  prompts and its answer are unchanged;
* `AgentResponse.sub_agent_threads()`, and a decomposed run's own `metadata["thread"]`;
* `WorkflowResult.thread`, `.threads`, `.node_thread()` and `.failed_nodes()`, and
  `WorkflowNode.thread` — all serialised into `to_dict()`;
* `TeamResponse.thread` and `.agent_threads()`, serialised the same way;
* `WorkflowCheckpoint.threads` and `.tasks`; an older reader ignores both and still resumes;
* `SubAgentResult.thread`;
* `WorkflowDAG(projection=...)`, `TeamConfig(projection=...)` and `SubAgentManager(projection=...)`,
  all defaulting to carrying nothing into a child run — which is what every pattern did before.

### 9. `effgen run --json` works on a run that used a tool, and the documents are scrubbed

`AgentResponse.to_dict()["execution_tree"]` carried `ToolCall` objects rather than data, so
`json.dumps(response.to_dict())` raised `TypeError: Object of type ToolCall is not JSON
serializable` for any run that called a tool — taking `effgen run --json`, `-o` and `--card` with
it. `ExecutionNode.to_dict()` and `ExecutionEvent.to_dict()` now render through a converter that
asks a record for its own `to_dict`, reads a dataclass field by field, and falls back to a string
rather than dropping anything.

`effgen run`'s `--json`, `-o` and `--card` documents now go through one scrubber, and
`--show-thread` renders through the same one. **The terminal answer panel still prints the run's
own words unredacted** — a tool that puts a key in the answer puts it in the answer. That asymmetry
is deliberate and it is the one place a secret can still reach a terminal.

### 10. The debug trace carries steps

`DebugIteration` gained `thread_snapshot`, and `DebugIteration.to_dict()` gained a `thread` key.
The inspector's panel renders every iteration from it, not only behind `--step`.

### 11. A run continuing a session sends its conversation as messages, from its first request to its last

Every 1.0.x run sent one flat string. At the new default, `prompt_protocol="auto"`, a run with tools
that continues a session sends the session's earlier turns and its own steps as the messages they
were, on a model that declares the message protocol and takes its tools as a request parameter. A
run that continues nothing keeps its own steps in the flat string, and a run with no tools sends the
one string it always sent: measured on ten public sample sets at two model sizes, the default sent
messages **0** times, and a run with no tools attached sent them 0 times at either setting. Whether
a caller's `system_prompt` and a session's earlier turns travel as their own messages is not this
setting's to decide; change 2 says when they do.

The protocol holds for the whole run. When the guards stop offering tools — the model has spent its
allowance of multi-call turns, or repeated a call with nothing usable to fall back on — the turn
that asks for the answer still goes out as messages: the session's earlier turns as separate
messages, a persona stated once as the system turn, the answer scaffold as the last user message,
and no `tools` or `tool_choice` on the request. The same holds for every run at an explicit
`prompt_protocol="messages"`. That turn logs
`[protocol] the run's conversation is already travelling as messages; this turn keeps it there, with no tool definitions on the request`
at INFO; a run that never went out as messages still logs
`[protocol] this turn's tool definitions do not travel as a request parameter; the turn sends the flat transcript`.
A run whose persona or earlier turns travelled as their own messages keeps them there on that turn
as well, at every `prompt_protocol`, and logs
`[frame] the turn asking for the answer keeps the run's frame as messages`, so the turn does not
change the request's shape. If the provider refuses the message list on that turn, the turn is sent
again as the flat string.

*Migration:* none for a run without a session. `AgentConfig(prompt_protocol="flat")` keeps a
session run's own steps in one string, as 1.0.x did; its earlier turns and a persona still travel
as their own messages where change 2 says they do.

---

## The prompt protocol

`AgentConfig.prompt_protocol` is new: `"flat"`, `"messages"` or `"auto"`, **default `"auto"`**. It
decides how a run's conversation reaches the model.

* **`"flat"`** renders the run's own steps — its thoughts, calls and results — into one string. This
  is what every effGen release before this one did, and it is still what a single-turn run does. On
  a model whose adapter takes a conversation, a caller's `system_prompt` and a session's earlier
  turns travel beside that string as their own messages (change 2).
* **`"messages"`** sends the conversation as the turns it was: the frame as a system message, the
  task as a user message, the run's own steps as the assistant/tool exchange they were, and the
  closing instruction as its own user turn. `OpenAIAdapter` now carries a tool call into
  `tool_calls` and a tool result into a `tool` message with its `tool_call_id`; both were dropped
  silently before.
* **`"auto"`** means one protocol for the whole of one conversation: a run **continuing a session**
  sends its own steps as the turns they were, and a run **continuing nothing** keeps them in the
  flat string.

```python
from effgen import AgentConfig

print(AgentConfig(model="openai:gpt-5-nano").prompt_protocol)   # auto
print(AgentConfig(model="openai:gpt-5-nano", prompt_protocol="messages").prompt_protocol)
```

**Why the default is not `messages`.** The rule was written down before anything was measured:
the default would move to `"messages"` only if no sample set got worse beyond its noise band, mean
accuracy rose by at least 1.66 points at one model size and fell by no more than that at the other,
and prompt tokens and model calls rose by no more than 10%. Compared with `"flat"` on the same
samples at two model sizes, `"messages"` met only the condition on mean accuracy: three sets got
worse beyond their bands, and on others it sent up to half as many prompt tokens again. So
`"messages"` ships opt-in, as `AgentConfig(prompt_protocol="messages")` — a configuration setting
rather than a `run()` keyword, because the protocol has to hold for a whole conversation. The
comparison was taken before the turn described in change 11 was fixed, and was not re-run after it.

---

## Added

**The conversation itself** — `from effgen import ...`:

- `AgentThread` — the conversation of one run, as typed steps
- `Step` — one entry in a run's conversation
- `SystemStep` — instruction the run is framed by: a persona, a contract, a format spec
- `TaskStep` — what the caller asked, with any non-text parts it arrived with
- `TurnStep` — one message from earlier in the session, before this run started
- `ThoughtStep` — the model's own reasoning for a turn
- `ActionStep` — a tool the turn asked for, with the arguments it asked for
- `ObservationStep` — what a tool returned for the call before it
- `NudgeStep` — a line the framework injected, not something the model or a tool said
- `AnswerStep` — how the run ended: the answer it reached, or why it stopped without one
- `DelegationStep` — work this run handed to another agent, and the conversation it had

**Keeping a conversation inside its budget:**

- `ContextBudgetExceededError` — raised when a run's conversation will not fit the tokens it may send
- `CompactionPolicy` — how a run's thread is brought back under its budget
- `ShortenOldestFirst` — the default policy: four rungs, and not one model call
- `SummarizeWithModel` — opt-in: the dropped material leaves behind a model-written summary

**What a child run starts with:**

- `ThreadProjection` — which of a parent run's steps a child run starts with
- `NoParentContext` — carry nothing: the child sees only the question it was asked (the default)
- `ParentTask` — carry the job the parent was given, as one user turn
- `ParentAnswers` — carry the parent's task and what its finished children answered
- `LastCycles` — carry the parent's task and the last *n* complete cycles of its own work

**Reading a conversation back:**

- `RenderedStep` — one step of a conversation, ready to print
- `render_thread` — render a conversation as steps, redacted by default, ids on request
- `thread_as_text` — the conversation as one block of text, one heading and body per step

**Managing sub-agents** — importable from `effgen`, where before they were reachable only through
`effgen.core.sub_agent_manager`:

- `SubAgentManager` — spawns the sub-agents a decomposed task is split across, runs them in
  parallel or in sequence, and combines their results
- `SubAgentResult` — what one sub-agent returned: its result or error, its time and tokens, and
  its own conversation as `.thread`

Also new, on types that already existed: `AgentResponse.thread` and `.sub_agent_threads()`,
`Checkpoint.thread` and `.to_thread()`, `Session.last_thread()` and `Session.keep_thread_history`,
`WorkflowResult.thread` / `.threads` / `.node_thread()` / `.failed_nodes()`, `WorkflowNode.thread`,
`TeamResponse.thread` / `.agent_threads()`, `WorkflowCheckpoint.threads` / `.tasks`,
`SubAgentResult.thread`, `ObservationStep.compacted` / `.original_chars`,
`DebugIteration.thread_snapshot`, `AgentConfig.prompt_protocol` / `.context_budget` /
`.compaction`, `BaseModel.supports_message_protocol()`, and `effgen run --show-thread`.

`docs/guides/reading-a-run.md` is the guide to all of it.

## Known issues

These are open. Each is understood well enough to say what it is. The first is fixed in part, and the
entry says what is still open.

1. **A turn whose tools the guards have stopped offering still shows the model its own tool calls as
   text.** Once a run spends its allowance of multi-call turns, or repeats a call with nothing
   usable to fall back on, the turn that asks for the answer carries no tool definitions. On the
   build the protocol comparison was taken on, that turn moved a run that was on messages to the
   flat string for the rest of the run — on every run at `prompt_protocol="messages"`, and on any
   run continuing a session at the default `"auto"`. In that comparison the switch is what the 31
   fallbacks on one 7B set, and 1–5 on six others, recorded. **The switch is fixed:** the turn stays
   on messages, with the session's earlier turns as separate messages, a persona stated once as the
   system turn, and no `tools` or `tool_choice` on the request (change 11). **What remains** is the
   last user message of that turn. It is the whole answer scaffold, which carries this run's own
   calls and results as `Thought:` / `Action:` / `Observation:` lines and lists the tools in prose,
   so on that one turn the model still reads its own calls as text inside the user's message. A run
   with no session and no persona sends the same text the flat string carried, inside a message
   list. And if a provider refuses that tool-free message list, the turn is sent again as the flat
   string and the model is treated as refusing the message protocol for the rest of the process; a
   run that carried only its persona or earlier turns as messages falls back the same way for the
   rest of that run, without marking the model.
2. **A streamed run can hand a tool a truncated argument.** On an arithmetic sample set at full
   size, **62 of 557** streamed 7B tool calls carried an argument the tool could not use, against
   **0 of 486** on the blocking path. The arguments arrive cut mid-JSON and are wrapped as
   `{"__raw_input__": …}`. On the same set the streamed path answered 6.50 points below the blocking
   one at 1.5B — a direction, not a result, at that sample size — and the samples with a truncated
   argument account for only 1.00 of that.
3. **`AgentConfig(model="openai:<id>", base_url=...)` sends the engine prefix on the wire.** Naming
   a self-hosted OpenAI-protocol server with a prefixed id makes the server answer
   `The model 'openai:<id>' does not exist`. Write the id without the prefix —
   `AgentConfig(model="<id>", base_url=...)` — which is the form the documentation uses.
4. **`ToolCall.arguments` is a string on one path and a mapping on the other**: text for the ReAct
   path, whose action input is text, and a parsed dict for native tool calling. Fixing it changes a
   document shape every caller of `AgentResponse.tool_calls` can read.
5. **`reasoning_effort` reaches a streamed turn and not a blocking one.** Pre-existing; the archive
   behaves identically.
6. **A tool-free stream yields the model's own text, not the sanitized answer.** `run()` returns
   `'36'` where the stream yields `'Thought: …\nFinal Answer: 36 '`. Pre-existing and unchanged.
7. **`GuardrailChain.check(position=...)` is not forwarded.**
8. **A spend cap still refuses a call that costs nothing.** Once a configured daily budget is spent,
   the preflight refuses every call — including one to a model you serve yourself, because an
   `openai_compatible` adapter is priced as `openai`. The refusal arrives as a failed generation
   rather than an error, so a batch can complete with no errors and measure nothing. This was in
   1.0.0 and 1.0.1 too.
9. **The retrieval and open-ended gaps 1.0.1 recorded are unchanged.** Nothing in this release was
   aimed at them.

**What it cost.** On the same ten public sample sets and the same samples as the 1.0.1 baseline, a
run sends 26% fewer prompt tokens at 1.5B and 18% fewer at 7B, and makes about 16% fewer model calls
at both sizes. The saving is in tokens sent, not in time spent generating, so the framework is
cheaper to run and no faster. Mean accuracy moved −2.67 at 1.5B and −3.53 at 7B with the sets
weighted equally, and −0.42 and −1.48 with the samples weighted equally. Three sets got worse and
two got better beyond every noise band we computed, and the three that got worse are the sets where
this release sends more, not less. No cloud model was measured, so nothing here is a claim about a
cloud provider.

[Full 1.1.0 changelog](CHANGELOG.md#110---2026-09-14)

---

## v1.0.1 - September 8, 2026

This release fixes how the framework reports what a run did, what it puts in a prompt, and what its
own bookkeeping costs.

The main things: a run that stops without an answer now says so instead of handing back its working
notes. Citation markers are opt-in instead of being added to every retrieval answer. The loop guards
no longer stop a run that is still making progress. Every tool-calling path now tells the model what
the tools are for. The budget check before each model call reads an index instead of the whole spend
ledger. And the Groq default points at a model Groq still serves.

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
