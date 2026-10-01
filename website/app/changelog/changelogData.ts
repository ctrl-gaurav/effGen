// What changed in each release, adapted from the framework's own CHANGELOG.md
// and NEWS.md. Nothing here is a summary written from memory: each group below
// carries the heading the changelog files it under, and each item is one of the
// entries under that heading.
//
// Counts and version numbers are not written here — the page reads them from
// data/effgen.json, so this file cannot disagree with the installed package.

export interface ChangeItem {
  title: string;
  body: string;
  /** A sample, run before it was written down, and what it printed. */
  code?: { source: string; language?: "python" | "bash"; output?: string };
}

export interface ChangeGroup {
  id: string;
  title: string;
  lede: string;
  accent: string;
  items: ChangeItem[];
}

export interface BreakingChange {
  title: string;
  why: string;
  /** The one line that migrates it, and what it printed when it was run. */
  migration: { source: string; language: "python" | "bash"; output?: string };
  note?: string;
}

/** A change existing code sees. Where no code has to change, there is no migration sample. */
export type VisibleChange = Omit<BreakingChange, "migration"> & {
  migration?: BreakingChange["migration"];
};

/* ── 1.3.0 ───────────────────────────────────────────────────────────────── */

/** Released 1 October 2026. */
export const RELEASE_DATE_1_3_0 = "1 October 2026";

/** `git rev-list --count v1.2.0..v1.3.0` in the framework repository. */
export const COMMITS_SINCE_1_2_0 = 28;

/** The public surface 1.2.0 shipped, which 1.3.0 grew from. */
export const PUBLIC_NAMES_1_2_0 = 251;

/** The ten changes in 1.3.0 that existing code can see, in the order the changelog lists them. */
export const visibleChanges130: VisibleChange[] = [
  {
    title: "A run that stops making progress is asked for its answer — max_turns_without_progress defaults to 2",
    why:
      "It was None. After two turns in a row that bring no new tool result, the next turn offers no tools and asks for the answer. A turn that declares no action after a result — Action: None, Action: (continue reasoning) or another bracketed placeholder that names no tool the agent holds — is asked at once. Before the run's first result, only a turn whose every call was declined counts. A run that would have stopped on loop_detected, repeated_tool_result, max_iterations_* or null_final_from_model while holding tool results gets one closing request: its own calls and results, with no tools, whose reply is the answer. When that reply is an answer the run succeeds, with metadata[\"answer_source\"] == \"closing_request\", so a run that raised RunStoppedError in 1.2.0 under the default raise_on_error=True can now return an answer. A closing reply that is a call, or a program for a code tool the agent holds, is not taken as the answer; after a program the run carries on, so such a run can send one model call more than max_iterations.",
    migration: {
      source: `from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano", max_turns_without_progress=None)   # 1.2.0's loop`,
      language: "python",
    },
    note: "run(max_turns_without_progress=None) restores it for one call.",
  },
  {
    title: "A run whose tool keeps failing ends tool_failed",
    why:
      "A tool that fails on its own side — a connection error, a timeout, an HTTP 5xx or 429, missing credentials — three times in a row, or whose circuit breaker is open, is unavailable for the rest of the run. A run left with no usable tool and no result now stops with the new stop reason tool_failed, which is in STOPPED_REASONS and raises RunStoppedError under the default raise_on_error=True. In 1.2.0 the run went on and returned whatever the model wrote, often a statement that it could not get the information. Such a run gets no closing request: it reports the tool's failure, typed, with metadata[\"error\"][\"kind\"] == \"tool\", and metadata[\"unavailable_tools\"] names the tools. The per-tool circuit breaker now counts only those tool-side failures, so a tool given bad input by the model is no longer refused to later calls and later runs, and a call retried after a tool-side failure is a retry, not a repeated call.",
    migration: {
      source: `from effgen import RunStoppedError

try:
    response = agent.run("What is the weather in Paris?")
except RunStoppedError as exc:   # stop_reason "tool_failed" when the tool kept failing
    print("stopped:", exc)`,
      language: "python",
    },
    note:
      "Or pass raise_on_error=False and branch on response.termination == \"tool_failed\".",
  },
  {
    title: "AgentResponse.termination says how a run ended",
    why:
      "\"done\" (the model answered), \"not_possible\" (the model answered, but every call was declined, failed on the tool's side or returned nothing — usually an answer saying the task cannot be done with these tools), \"stuck\" (the run kept proposing work that brought nothing new and wrote no answer), \"tool_failed\" and \"error\" (the run could not be carried out). to_dict() carries it, and a saved run read back reports the same value. A run whose calls reached a tool that rejected their input used the tool, and its answer is \"done\". Alongside it, metadata[\"tool_results\"] counts a run's attempted, usable and input-rejected calls. A stopped run's partial never carries a tool's error message; a run whose only observations were errors has partial=None.",
  },
  {
    title: "A tool call is read before it is reported — recover_lost_tool_calls defaults to True",
    why:
      "It was False. A tool call written as a Python literal, with raw line breaks inside its JSON, with unescaped double quotes inside a string value, with its last string closed one bracket early, or as a whole object followed by text inside its tag, now runs instead of ending the run with written_tool_call. A call nothing can read is sent back once, with a call required where the provider supports it. When the agent holds exactly one code-execution tool, a program in a fenced block before an empty call tag runs as that call. Under either setting, arguments that arrive as a string are read (as keywords, as an object's JSON, or as a raw value) and tool_calls records carry the decoded arguments; a call missing a required argument is not dispatched and is asked for again; and positional values are named in the tool's parameter order, while a call with more values than the tool has parameters carries no arguments and is asked for again.",
    migration: {
      source: `from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano", recover_lost_tool_calls=False)   # 1.2.0's reader`,
      language: "python",
    },
    note:
      "Some runs make more tool calls: the calls the model meant to make now run. Delegated sub-agents inherit the setting.",
  },
  {
    title: "A failed tool call is retried, bounded, and never an answer",
    why:
      "Two calls whose different inputs a tool rejects in the same words are no longer a repeated result, so they no longer withdraw the run's tools. Four failures in a row on a tool's input withdraw it for the run; a run left with no tool is asked for its answer, and ends tool_failed with metadata[\"error\"][\"kind\"] == \"input\" when it writes none, where it used to go round to max_iterations. An answer that is a tool's error message is not a success: it is sent back once, and given again the run stops with null_final_from_model. A failed call no longer earns \"you have the answer from the tool\" near the iteration cap, and the direct calculator result is never a failure.",
  },
  {
    title: "Served and local models are measured once for how they use a tool",
    why:
      "At tool_calling_mode=\"auto\", the first agent with tools built for a model served behind a base_url, or run on a local engine, runs a short probe: questions about fictional things, answerable only through a stub search tool, as ordinary agent runs. A model that answers from memory while holding a search tool has its information-retrieval tools made must-call; a model whose native calls the server does not carry back is run in the ReAct text frame. Both follow only when you left the setting unset, and calculator and code tools are never moved. The result is stored in ~/.effgen/capabilities.json (or under $EFFGEN_HOME, or at $EFFGEN_CAPABILITY_CACHE), read by every later agent across processes, and measured again when the weights, the chat template, the endpoint or the probe change, and after 30 days. A probe is bounded at 48 requests and 120 seconds; one that cannot run stores nothing and logs one warning. response.metadata[\"tool_calling\"] says which strategy a run used and why. The same store remembers a provider that rejects stop sequences beside tool definitions, where 1.2.0 reported the provider's HTTP 400, and a server that rejects reasoning_effort.",
    migration: {
      source: `from effgen import AgentConfig

config = AgentConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    capability_probe=False,   # or EFFGEN_CAPABILITY_PROBE=0
)`,
      language: "python",
    },
    note:
      "An explicit tool_calling_mode or tool_use also keeps the declared behaviour, and first-party cloud adapters are never probed. A test suite that scripts a served endpoint sees the probe's requests first unless it sets one of these.",
  },
  {
    title: "reasoning_effort reaches a model behind base_url, and Groq",
    why:
      "A reasoning_effort you set is now sent by the OpenAI-compatible adapter, whatever the model is called, and by the Groq adapter for a model its catalog marks as reasoning; in 1.2.0 both dropped it. A server that refuses the field costs one retry, once, and is remembered. Every other adapter that drops a reasoning_effort you set now says so once per model at WARNING level.",
  },
  {
    title: "One agent serves overlapping conversations without mixing them",
    why:
      "run(session=...), run_async(session=...) and stream(session=...) hold their conversation on the call, not on the shared agent, so overlapping calls on one agent — threads, run_async() tasks, streams — each read and record only their own session. In 1.2.0 they swapped agent.session and agent.short_term_memory on the shared object, so overlapping calls could read and record each other's turns. Inside a call, agent.session and agent.short_term_memory still name that call's conversation; outside one, they are the agent's own. Per-call middleware applies to that call only. stream() now honours session=, where 1.2.0 ignored it, and a run whose input a guardrail blocks leaves the agent's session as it was.",
    note: "No code has to change. Calls without session= still share the agent's own memory, as before.",
  },
  {
    title: "A streamed turn is saved to the agent's bound session",
    why:
      "On an agent created with session_id=, or given agent.session, stream() now appends each answered turn to that session and saves it, as run() always did. In 1.2.0 streamed turns stayed in memory and were missing from the session file, so a later process continuing the conversation never saw them.",
  },
  {
    title: "Smaller changes",
    why:
      "effgen code --json reports a tool_failed run as stopped, as the library does. The OpenAI SDK's response types are built once when the adapter loads, so the first concurrent streams in a fresh process no longer fail with 'BaseModel' has no attribute '__pydantic_core_schema__'. The agent's circuit breaker keeps every first failure when several calls fail on one tool at once, and no longer raises dictionary changed size during iteration under concurrent writers.",
  },
];

export const oneThreeZeroGroups: ChangeGroup[] = [
  {
    id: "probe-130",
    title: "Measuring what a model does with a tool",
    lede: "The probe behind tool_calling_mode=\"auto\" is public, and effgen doctor shows and runs it.",
    accent: "#00ff88",
    items: [
      {
        title: "probe_tool_calling and ToolCallingProbe",
        body:
          "probe_tool_calling(model, refresh=False) measures what a loaded model does when handed a tool, or returns what was measured before. It returns a ToolCallingProbe — how many runs resolved, went unresolved or skipped the tool in the native frame (and in the text frame, when that was measured), the strategy and the must-call tool categories auto derives from them, and what the probe itself cost in requests, tokens and time — or None for a model that is not probed (a first-party cloud adapter) or a probe that could not run.",
      },
      {
        title: "effgen doctor --probe",
        body:
          "effgen doctor lists what each served or local model was measured to do with a tool and what was learned from its endpoint, and --json carries the same. --probe MODEL measures one model now — a local model, or one served at --base-url URL, with --api-key-env VAR naming the variable that holds the endpoint's key — and --refresh measures it again.",
        code: {
          source: `effgen doctor
effgen doctor --probe Qwen/Qwen2.5-1.5B-Instruct --base-url http://127.0.0.1:8000/v1 --refresh`,
          language: "bash",
        },
      },
    ],
  },
  {
    id: "config-130",
    title: "Configuration and results",
    lede: "What is new on AgentConfig, AgentResponse and the model adapters.",
    accent: "#00e5ff",
    items: [
      {
        title: "Settings",
        body:
          "AgentConfig.capability_probe (default True) and the environment variables EFFGEN_CAPABILITY_PROBE and EFFGEN_CAPABILITY_CACHE. stream() takes session=. BaseModel.capability_key() and BaseModel.forwards_reasoning_effort() are new on the adapter interface.",
        code: {
          source: `from effgen import AgentConfig
from effgen.core.agent import TERMINATIONS

config = AgentConfig(model="Qwen/Qwen2.5-1.5B-Instruct", base_url="http://127.0.0.1:8000/v1")
print(config.max_turns_without_progress)   # 2: a run with no new result is asked for its answer
print(config.recover_lost_tool_calls)      # True: a broken tool call is read before it is reported
print(config.capability_probe)             # True: a served or local model is measured once
print(TERMINATIONS)                        # the values response.termination can take`,
        },
      },
      {
        title: "Results",
        body:
          "AgentResponse.termination and the tuple TERMINATIONS in effgen.core.agent; the stop reason tool_failed; and the metadata keys tool_results, unavailable_tools, tool_calling and answer_source.",
      },
    ],
  },
  {
    id: "short-130",
    title: "Where it falls short",
    lede: "What the release does not do.",
    accent: "#ff9500",
    items: [
      {
        title: "More calls in some places",
        body:
          "Small models given a search tool now cost noticeably more per run, because they are made to use it. Runs that use tools still make more model calls than they need to.",
      },
    ],
  },
  {
    id: "known-130",
    title: "Known issues",
    lede: "Open in 1.3.0, and each understood well enough to say what it is.",
    accent: "#ff6b6b",
    items: [
      {
        title: "Small models given a search tool cost more",
        body:
          "When the probe finds that a model answers from memory while holding a search or retrieval tool, every run holding one is made to call it, which makes such runs several times as many model calls on question-answering tasks and slower overall. tool_use=\"auto\" or capability_probe=False turns it off for an agent.",
      },
      {
        title: "A run whose calls keep failing on their own input ends stuck, not typed",
        body:
          "A tool that rejects the same input twice, or every input in the same words, is stopped by the loop guard before the four-failure bound, so the run ends loop_detected (termination == \"stuck\") rather than tool_failed with kind=\"input\".",
      },
      {
        title: "A run whose tools all fail on their own side gets no closing request",
        body:
          "It ends tool_failed at once, even when the model had already written what it would answer. Whether such a run should be asked once, as a run whose tool was withdrawn for bad input is, is open.",
      },
      {
        title: "Some runs answer from the wrong result",
        body:
          "On one set of coding tasks, a mid-size model answers correctly less often than on 1.2.0. The difference is runs that 1.2.0's loop guard ended with a usable result; 1.3.0 asks them for their answer, and some answer from the wrong result.",
      },
      {
        title: "The framework's own time on a long run grew",
        body:
          "On a long run the framework's own time is higher than in 1.2.0, within its budget; a batch of runs is flat.",
      },
      {
        title: "Smaller open items",
        body:
          "A probe's requests are booked in the first agent's cost ledger with no tag marking them as the probe's. The readers now on by default have edges: the lenient reader reads \"print(\"\")\" as print(), a truncated call naming the code tool itself still runs the fenced block before it, and a closing reply on the text scaffold that is another call ends the run stuck after one request. A Groq call's retry wait is booked as framework time in the run ledger. With one agent and many sessions, stream() runs no middleware, resume(session=...) restores the checkpoint's memory onto the agent's own memory, and a run made while a stream is held open writes its trace to a tracker of its own. Most of 1.2.0's known issues, and the Groq free-tier HTTP 413, were not aimed at by this release and were not re-checked.",
      },
    ],
  },
];

/** The names 1.3.0 added to `effgen.__all__`. */
export const newPublicNames130 = ["ToolCallingProbe", "probe_tool_calling"];

/* ── 1.2.0 ───────────────────────────────────────────────────────────────── */

/** Released 27 September 2026. */
export const RELEASE_DATE_1_2_0 = "27 September 2026";

/** `git rev-list --count v1.1.0..v1.2.0` in the framework repository. */
export const COMMITS_SINCE_1_1_0 = 69;

/** The public surface 1.1.0 shipped, which 1.2.0 grew from. */
export const PUBLIC_NAMES_1_1_0 = 250;

/** The fourteen changes in 1.2.0 that existing code can see, in the order the changelog lists them. */
export const visibleChanges120: VisibleChange[] = [
  {
    title: "A tool result the model writes itself is never taken as the answer",
    why:
      "On a model with native tool calling, the default hybrid strategy also reads a tool call written out as text (Action: / Action Input:). A model that does that can go on to write the tool's result itself, which used to end the run as a final answer from a tool that never ran. A turn that holds tools and whose reply may be read as text is now sent one stop sequence, \"\\nObservation:\", where 1.1.0 sent four labels; a tool-free turn is sent none, and a caller's own stop_sequences replace it. When a reply holds a written action anyway, the action runs and whatever the model wrote after it is discarded. BaseModel.supports_stop_with_tools() is new: an adapter for a provider that rejects stop beside tools answers False, and the framework then cuts the returned text itself.",
    note:
      "A caller who relied on \"\\nQuestion:\", \"\\nHuman:\" or \"\\nUser:\" stopping a tool-holding turn passes them in stop_sequences. A long answer containing a line that begins Question: is no longer cut there.",
  },
  {
    title: "reasoning_effort reaches run() and run_async()",
    why:
      "In 1.1.0 it reached a streamed turn and not a blocking one. run(), run_async() and stream() now all send it when the caller passes it and the model declares that it reasons, and so does the blocking follow-up turn inside a stream. On a reasoning model this changes how long the answer is; on any other model the adapter drops it.",
  },
  {
    title: "The output budget follows what the run declared",
    why:
      "When the caller pins no max_tokens, a run that declares an output_schema asks for a budget sized from the schema — a one-integer schema asks for 256 tokens rather than 1,024 — and a model that declares it reasons is never sent fewer than 4,096. A published maximum caps either. An explicit max_tokens, on the call or on AgentConfig, still wins on every call the run makes.",
    note:
      "Where a model would have written more than a schema-sized budget, the answer is now cut at the budget with the existing typed truncation message.",
  },
  {
    title: "A request carries less of the framework's own text",
    why:
      "The tool contracts no longer ask for text the task did not ask for: the lookup contract's request to name what is missing, and the computation contract's request to work step by step and restate the answer, are gone. The tool contract comes before the caller's task, so the task is the last thing the model reads. Tool rules are stated once, a tool result identical to one the request already carries is written once and referred back to, and how fully a tool is described comes from the adapter through the new BaseModel.prompt_detail(). A run with no tools, or with tool_contract=\"\", is unaffected.",
    note:
      "Only the rendering changes: a step keeps the whole tool result, so nothing stored, checkpointed or read back differs. AgentConfig(answer_style=...) is the way to ask for a shorter or a fuller answer.",
  },
  {
    title: "On a provider with a prompt cache, a run keeps one request shape",
    why:
      "A run with tools sends its conversation as messages from its first run when memory is on, prompt_protocol is \"auto\" and the provider's adapter declares a prompt cache and takes messages, so a session's next question extends the cached prefix instead of starting a new one. The turn that asks for the answer keeps the same messages and tool definitions with tool_choice=\"none\" where the adapter can forbid a call. AgentConfig.cache_system_prompt and cache_tools now work on Anthropic. Cached tokens are read on Groq, Together, Fireworks, Cerebras and Gemini, and priced at the provider's cached rate where the model catalog carries one; the ledger gains cache_write_tokens.",
    migration: {
      source: `from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano", prompt_protocol="flat")`,
      language: "python",
    },
    note:
      "A reader that sums prompt tokens sees no change: cached tokens are part of prompt_tokens, not added to it. prompt_protocol=\"flat\" keeps a run's own steps in one string.",
  },
  {
    title: "The loop stops fewer runs that are still finding things",
    why:
      "A tool whose calls keep returning new results is no longer withdrawn after 12 calls (16 for a data-processing tool); it is withdrawn after that many calls in a row that brought nothing new, and a run that keeps finding things is bounded by max_iterations. run(max_iterations=N) now moves the loop's own thresholds too. Action: None is read as no action rather than as a call to a tool named None. A run that only repeats itself ends exactly as before.",
  },
  {
    title: "Groq hands a tool call it could not parse back to the loop",
    why:
      "When a request carried tools and Groq answers that it could not parse the model's tool call, the adapter returns the call as the model wrote it instead of failing the run. The loop runs it if it can read it and otherwise asks for it again; the call's usage is estimated and recorded.",
  },
  {
    title: "A spent cap refuses only calls that cost money, and says so with a typed error",
    why:
      "Once a configured spend cap is spent, a call to a local engine, a free tier or a server reached with base_url= is no longer refused. A refused call raises BudgetExceededError from run() whatever raise_on_error says, where 1.1.0 raised RuntimeError or returned a failed response. stream() raises it, run_batch() raises and stops, the server answers HTTP 429 with budget_exceeded, and effgen run --json prints the error document. No request reaches the provider.",
    migration: {
      source: `from effgen import BudgetExceededError

try:
    response = agent.run("Summarise the report.")
except BudgetExceededError as exc:   # not a RuntimeError
    print("spend cap reached:", exc)`,
      language: "python",
    },
  },
  {
    title: "A model with no published price reads as unknown, not as free",
    why:
      "CostTracker.total_cost() returns None when every call it covers was unpriced; it returned 0.0. CostEvent.cost_usd, RunLedger.cost_usd, total_cost_usd in effgen cost --json and cost_usd in the run executions and topology can all be None for the same reason. A free model still reads 0.0. A server reached with base_url= now records as openai_compatible, where 1.1.0 recorded openai, so an existing ledger file shows the same served model under both names across the upgrade.",
    note: "Treat None as \"not priced\" wherever you add up cost.",
  },
  {
    title: "Every run keeps a ledger, and two totals changed with it",
    why:
      "response.ledger and response.metadata[\"ledger\"] carry the run's RunLedger; the flat metadata keys are unchanged. Children — sub-agents, workflow nodes, team members, an agent run inside a tool — are attached once, and total() adds them up. effgen_model_call_latency_seconds now observes each model call rather than each run's wall time, and a decomposed run's tokens_used includes its sub-agents' tokens, while effgen_tokens_used_total takes only the run's own calls. A synchronous @tool runs in the caller's context, so an agent it starts is a child of the run.",
    migration: {
      source: `from effgen import Agent, AgentConfig
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
print(ledger.cost_usd)   # None: a model you serve yourself has no published price`,
      language: "python",
    },
  },
  {
    title: "A run's reported time includes the work after the model's last answer",
    why:
      "execution_time now includes after_run middleware, the session save and the final checkpoint, and equals the ledger's wall_s.",
  },
  {
    title: "The spend ledger stops growing at 250,000 rows",
    why:
      "At that size the ledger file folds its oldest rows into per-model totals, keeping every total exact. EFFGEN_COST_MAX_ROWS sets the ceiling, and 0 keeps every row. Opening an existing file adds two columns, calls and unpriced_calls, and rows in effgen cost --json gain unpriced_requests. A write that triggers a fold waits for it.",
  },
  {
    title: "A server named with openai: gets the id without the prefix, and a streamed call keeps its own arguments",
    why:
      "AgentConfig(model=\"openai:<id>\", base_url=...) now sends <id>; 1.1.0 sent the prefix and the server answered that the model does not exist. Concurrent streamed runs no longer hand a tool another stream's arguments. In-process local engines serve concurrent agents: several agents can share one in-process vLLM engine, concurrent GGUF runs no longer crash, and a GGUF run reuses its cache across turns.",
  },
  {
    title: "An out-of-budget failure says what to do and is not retried",
    why:
      "A reply cut off by its token budget, or one that spent the whole budget reasoning, now carries its own guidance and is not marked retryable. It used to read \"Unexpected provider error\" and be retried.",
  },
];

export const oneTwoZeroGroups: ChangeGroup[] = [
  {
    id: "ledger-120",
    title: "What a run spent",
    lede: "Every run keeps a ledger of its model calls, tool calls, tokens, cost and where its time went, and children roll up into their parent.",
    accent: "#00ff88",
    items: [
      {
        title: "RunLedger",
        body:
          "Model and tool calls; prompt, completion and cached tokens; cost; and wall time split into model, tool, caller, child and framework time — per run and per iteration. It is on AgentResponse.ledger, Agent.last_stream_ledger and Checkpoint.ledger, and the run store gains llm_calls, tool_calls, cached_input_tokens, model_wait_s, tool_wait_s and framework_s.",
      },
      {
        title: "On the dashboards",
        body:
          "Prometheus gains effgen_run_framework_seconds, effgen_model_cost_usd_total and effgen_model_unpriced_calls_total, and the run card shows \"Model calls\" and \"Framework time\".",
      },
      {
        title: "Many agents, one spend ledger",
        body:
          "Many agents in one process no longer queue on the shared spend ledger. SQLiteCostStore.flush() writes what is buffered, and CostEvent gains calls and unpriced_calls.",
      },
    ],
  },
  {
    id: "bench-120",
    title: "effgen bench",
    lede: "Measures an agent on your own tasks and prints accuracy beside model calls, tool calls, tokens, time and cost, with a noise band beside every difference.",
    accent: "#00e5ff",
    items: [
      {
        title: "init, run and compare",
        body:
          "effgen bench init writes a starter suite; effgen bench run SUITE runs it against a model and saves the run; effgen bench compare A B pairs two saved runs task by task. The command is built on the effgen.bench package, which is importable and not part of effgen.__all__.",
        code: {
          source: `effgen bench init
effgen bench run bench-suite.yaml --model Qwen/Qwen2.5-1.5B-Instruct --base-url http://127.0.0.1:8000/v1 --out runs/a`,
          language: "bash",
        },
      },
    ],
  },
  {
    id: "answer-120",
    title: "How a run is asked to answer",
    lede: "Three new AgentConfig settings, each off unless you set it, and each also a keyword on the run.",
    accent: "#a78bfa",
    items: [
      {
        title: "answer_style",
        body:
          "One line about the form of the answer, stated last: \"brief\", \"full\", or your own sentence. The default states nothing, because a brief-answer line on every run also makes a model answer from what it knows instead of using the tool it was given.",
        code: {
          source: `from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano", answer_style="brief")
print(config.answer_style)                 # brief: one line, stated last
print(config.max_turns_without_progress)   # None: off unless you set it
print(config.recover_lost_tool_calls)      # False: off unless you set it`,
        },
      },
      {
        title: "max_turns_without_progress",
        body:
          "After that many turns in a row that brought no new result, the next turn offers no tools and asks for the answer.",
      },
      {
        title: "recover_lost_tool_calls",
        body:
          "Reads a tool call written with raw line breaks or Python-style quoting, and asks once more for a call that could not be read at all. Both of the last two can be set for one call as run() keywords, and a child run inherits them.",
      },
    ],
  },
  {
    id: "known-120",
    title: "Known issues",
    lede: "Open in 1.2.0, and each understood well enough to say what it is.",
    accent: "#ff6b6b",
    items: [
      {
        title: "Tool-using runs still make more model calls than they need to",
        body:
          "The ledger added here is what makes that visible, and reducing it is the main work of the next release. Answers that need no tool are far shorter than in 1.1.0; some tool-using tasks make more model calls and send more prompt tokens.",
      },
      {
        title: "One Agent serving concurrent sessions mixes the conversations",
        body:
          "With many run(session=...) calls in flight on one agent, answers can carry another conversation's turn. This was already true in 1.1.0. Use one agent per concurrent session.",
      },
      {
        title: "A provider that rejects stop beside tools answers HTTP 400 through a stock adapter",
        body:
          "Until its adapter declares supports_stop_with_tools() as False, the run reports the provider's 400; it never returns an invented answer.",
      },
      {
        title: "Streamed runs and post-run time are missing from some records",
        body:
          "The stream path records no Prometheus series, and the run store and Prometheus record a run's time before its post-run work, so they can read less than execution_time.",
      },
      {
        title: "Smaller open items",
        body:
          "The chat and effgen code /cost commands still print $0.00 for unpriced turns; cached tokens are priced at the cached rate only where the catalog carries one (OpenAI and Anthropic today); a reasoning model reached with base_url= is not sent reasoning_effort; effgen bench cannot pass context_length; ToolCall.arguments is a string on the ReAct path and a mapping on native tool calling; and GuardrailChain.check(position=...) is not forwarded.",
      },
    ],
  },
];

/** The name 1.2.0 added to `effgen.__all__`. */
export const newPublicNames120 = ["RunLedger"];

/* ── 1.1.0 ───────────────────────────────────────────────────────────────── */

/** Released 14 September 2026. */
export const RELEASE_DATE_1_1_0 = "14 September 2026";

/** `git rev-list --count v1.0.1..v1.1.0` in the framework repository. */
export const COMMITS_SINCE_1_0_1 = 59;

/** The public surface 1.0.1 shipped, which 1.1.0 grew from. */
export const PUBLIC_NAMES_1_0_1 = 225;

/** The eleven changes in 1.1.0 that existing code can see, in the order the changelog lists them. */
export const visibleChanges110: VisibleChange[] = [
  {
    title: "A run carries its conversation, and response.metadata is no longer plain data",
    why:
      "AgentResponse.thread is the run's AgentThread, or None for a run that recorded none, and response.metadata[\"thread\"] holds the same live object. The thread opens with the run's frame — a SystemStep when tools are attached, then always a TaskStep — so thread.steps[0] is no longer the first thought. json.dumps(response.metadata) raises on the thread object; to_dict() is the documented serialisation and writes the thread through its own.",
    migration: {
      source: `import json
from effgen import Agent, AgentConfig, thread_as_text

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
))
response = agent.run("What is 17 * 23?")

print(thread_as_text(response.thread))     # the run, step by step
json.dumps(response.to_dict())             # not json.dumps(response.metadata)`,
      language: "python",
    },
    note:
      "AgentThread.to_text() is unchanged and still renders the flat transcript a 1.0.x reader would recognise. The same steps are on the command line as effgen run --show-thread, and in effgen run --json under metadata.thread.",
  },
  {
    title: "A session's earlier turns render differently inside the prompt",
    why:
      "The === Previous Conversation Context === block with [Turn n] markers is gone; a one-string request now reads Earlier in this conversation: followed by User: / Assistant: lines. On a model whose adapter takes a conversation, a run whose tools travel as a request parameter sends the earlier turns as their own user and assistant messages and a caller's system_prompt as the system message, on every request of the run.",
    note:
      "Only code that matched on the old header text, or that reads the request a provider receives, is affected. The history itself is unchanged, and Session.last_thread() reads the steps directly.",
  },
  {
    title: "AgentConfig(guardrails=...) accepts a plain list and rejects what is not a guardrail",
    why:
      "A list of guardrails no longer has to be wrapped, and a non-guardrail in the list raises TypeError at construction instead of failing later inside a run.",
  },
  {
    title: "A 1.0.x checkpoint still resumes, and what the reconstruction drops is documented",
    why:
      "Checkpoint has a new thread field and still writes the flat transcript beside it, so a file 1.1.0 writes resumes on a build that only knows the transcript. A 1.0.x file has no steps, so Checkpoint.to_thread() rebuilds them and logs [compat] rebuilt a thread from a flat transcript. The rebuild cannot recover the provider's call id, a tool's arguments as values, which lines were the framework's own, or the run's frame.",
  },
  {
    title: "agent.resume() continues an unfinished run instead of restarting the task",
    why:
      "The final checkpoint stores the run's steps where it used to store an empty transcript, so resuming picks up the conversation the run had and does not redo the turns the checkpoint already holds.",
  },
  {
    title: "One agent loop, so stream() now behaves like run()",
    why:
      "The ReAct loop was written three times; there is one now. A blocking run is unchanged. A streamed run now sends the same prompt, the same tool definitions and the same sampling settings as run(), the loop guards fire on it as they do on a blocking run, and output guardrails are checked as run() checks them — a block raises out of the iterator.",
    note:
      "A streamed run that relied on the old behaviour — different sampling settings, a guard that never fired, an output guardrail that was never checked — now behaves as the blocking path always did.",
  },
  {
    title: "A run is bounded by what it may send",
    why:
      "AgentConfig.context_budget (\"auto\" by default) and AgentConfig.compaction are new, and max_context_length is now the window override. When the conversation will not fit, the run gives up the oldest material first and never touches the frame, the task, the two most recent complete cycles or the answer. The budget is unbounded when the model declares no window. Session.keep_thread_history defaults to False, which keeps session files from growing.",
    migration: {
      source: `from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano")
print(config.context_budget)        # auto — bounded by the model's own window
print(AgentConfig(model="openai:gpt-5-nano", context_budget=8000).context_budget)
print(AgentConfig(model="openai:gpt-5-nano", context_budget=None).context_budget)`,
      language: "python",
      output: "auto\n8000\nNone",
    },
    note:
      "\"auto\" reproduces 1.0.x behaviour on any conversation that already fitted. Pass context_budget=None for the old unbounded behaviour, and Session(keep_thread_history=True) if you read the stored steps of earlier turns. A prompt larger than the model's window is now sent once rather than retried.",
  },
  {
    title: "Orchestration results carry threads",
    why:
      "A run with no tools now reports metadata[\"thread\"]. AgentResponse.sub_agent_threads(), WorkflowResult.thread / .threads / .node_thread() / .failed_nodes(), WorkflowNode.thread, TeamResponse.thread / .agent_threads(), WorkflowCheckpoint.threads / .tasks and SubAgentResult.thread are new and serialised into to_dict(). WorkflowDAG, TeamConfig and SubAgentManager take a projection that defaults to carrying nothing into a child run — what every pattern did before.",
  },
  {
    title: "effgen run --json works on a run that used a tool, and its documents are scrubbed",
    why:
      "In 1.0.x, to_dict()[\"execution_tree\"] carried ToolCall objects, so serialising any run that called a tool raised TypeError — taking effgen run --json, -o and --card with it. Those documents now serialise, and all three go through one scrubber, as does --show-thread.",
    note:
      "The terminal answer panel still prints the run's own words unredacted: a tool that puts a key in the answer puts it in the answer.",
  },
  {
    title: "The debug trace carries steps",
    why:
      "DebugIteration gained thread_snapshot, and DebugIteration.to_dict() gained a thread key. The inspector renders every iteration from it, not only behind --step.",
  },
  {
    title: "A run continuing a session sends its conversation as messages",
    why:
      "At the new default, prompt_protocol=\"auto\", a run with tools that continues a session sends the session's earlier turns and its own steps as the messages they were, on a model that declares the message protocol and takes its tools as a request parameter — from its first request to its last, including the turn that asks for the answer after the guards stop offering tools. A run that continues nothing, and a run with no tools, sends the one string it always sent.",
    migration: {
      source: `from effgen import AgentConfig

config = AgentConfig(model="openai:gpt-5-nano", prompt_protocol="flat")`,
      language: "python",
    },
    note:
      "prompt_protocol=\"flat\" keeps a session run's own steps in one string, as 1.0.x did. \"messages\" sends every run's conversation as turns, and ships opt-in.",
  },
];

export const oneOneZeroGroups: ChangeGroup[] = [
  {
    id: "thread-110",
    title: "The conversation a run keeps",
    lede: "One ordered list of typed steps, which the loop builds, the prompt is rendered from, the checkpoint stores, and the caller can read.",
    accent: "#00ff88",
    items: [
      {
        title: "AgentThread and its step types",
        body:
          "SystemStep, TaskStep, TurnStep, ThoughtStep, ActionStep, ObservationStep, NudgeStep, DelegationStep and AnswerStep, all on Step. The command line, the run card, the debug inspector and the dashboard render the same steps.",
      },
      {
        title: "Reading a run back",
        body:
          "render_thread() hands back the steps one at a time as RenderedStep rows — a position, a kind, a label, a body and a depth — and thread_as_text() is the same thing as one block. Both redact by default and omit tool-call ids unless asked.",
        code: {
          source: `from effgen import AgentThread, TaskStep, AnswerStep, render_thread, thread_as_text

thread = AgentThread(steps=[TaskStep(text="What is 17 * 23?"), AnswerStep(text="391")])
for step in render_thread(thread):
    print(step.position, step.kind, step.label, "|", step.body)
print(thread_as_text(thread))`,
          output: `1 task task | What is 17 * 23?
2 answer answer | 391
  1. task
     | What is 17 * 23?
  2. answer
     | 391`,
        },
      },
      {
        title: "effgen run --show-thread",
        body: "Prints the run's conversation, step by step, through the same scrubber as the --json, -o and --card documents.",
        code: { source: 'effgen run "What is 17 * 23?" --show-thread', language: "bash" },
      },
    ],
  },
  {
    id: "budget-110",
    title: "Keeping a conversation inside its budget",
    lede: "A run says how many prompt tokens it may send and stays inside it, giving up the oldest material first rather than failing at the provider.",
    accent: "#00e5ff",
    items: [
      {
        title: "CompactionPolicy, ShortenOldestFirst and SummarizeWithModel",
        body:
          "The default policy, ShortenOldestFirst, shortens an old tool result, then drops an old thought, then replaces whole answered cycles with one NudgeStep, and makes no model call. SummarizeWithModel is opt-in and leaves a model-written summary behind. A call and the result answering it always leave together, and a shortened ObservationStep says so through compacted and original_chars.",
      },
      {
        title: "ContextBudgetExceededError",
        body:
          "Raised when a run's conversation will not fit the tokens it may send. response.metadata[\"context_budget\"] reports the budget on every outcome, including a run that used no tools.",
      },
    ],
  },
  {
    id: "protocol-110",
    title: "The prompt protocol",
    lede: "AgentConfig.prompt_protocol decides how a run's conversation reaches the model: \"flat\", \"messages\" or \"auto\", with \"auto\" the default.",
    accent: "#a78bfa",
    items: [
      {
        title: "Three settings, one per conversation",
        body:
          "\"flat\" renders the run's own steps into one string, as every earlier release did. \"messages\" sends the frame, the task, the run's steps and the closing instruction as the turns they were. \"auto\" sends a run continuing a session as turns and keeps a run continuing nothing in the flat string. It is a configuration setting rather than a run() keyword because the protocol holds for a whole conversation.",
        code: {
          source: `from effgen import AgentConfig

print(AgentConfig(model="openai:gpt-5-nano").prompt_protocol)
print(AgentConfig(model="openai:gpt-5-nano", prompt_protocol="messages").prompt_protocol)`,
          output: "auto\nmessages",
        },
      },
      {
        title: "Tool calls and results reach an OpenAI-protocol request",
        body:
          "OpenAIAdapter now carries a tool call into tool_calls and a tool result into a tool message with its tool_call_id; both were dropped silently before. BaseModel.supports_message_protocol() says whether a model takes a conversation.",
      },
      {
        title: "Why the default is not \"messages\"",
        body:
          "The condition for moving the default was written down before anything was compared, and \"messages\" did not meet all of it, so it ships opt-in.",
      },
    ],
  },
  {
    id: "delegation-110",
    title: "What a child run starts with",
    lede: "A parent run decides which of its steps a child it delegates to starts with. The default carries nothing, which is what every pattern did before.",
    accent: "#ff9500",
    items: [
      {
        title: "ThreadProjection and its four forms",
        body:
          "NoParentContext (the default) carries only the question the child was asked. ParentTask carries the parent's job as one user turn, ParentAnswers adds what its finished children answered, and LastCycles carries the last n complete cycles of the parent's own work. WorkflowDAG, TeamConfig and SubAgentManager each take projection=.",
      },
      {
        title: "SubAgentManager and SubAgentResult on the package",
        body:
          "Importable from effgen, where before they were reachable only through effgen.core.sub_agent_manager. A SubAgentResult carries its own conversation as .thread.",
      },
    ],
  },
  {
    id: "known-110",
    title: "Known issues",
    lede: "Open in 1.1.0, and each understood well enough to say what it is.",
    accent: "#ff6b6b",
    items: [
      {
        title: "The answer turn still shows the model its own tool calls as text",
        body:
          "Once the guards stop offering tools, the turn that asks for the answer now stays on messages; what remains is that its last user message is the whole answer scaffold, which carries the run's calls and results as Thought: / Action: / Observation: lines.",
      },
      {
        title: "A streamed run can hand a tool a truncated argument",
        body:
          "Some streamed tool calls arrive cut mid-JSON and are wrapped as {\"__raw_input__\": …}; the blocking path does not show this.",
      },
      {
        title: "A prefixed id with base_url sends the prefix on the wire",
        body:
          "AgentConfig(model=\"openai:<id>\", base_url=...) makes a self-hosted server answer that the model does not exist. Write the id without the prefix, which is the form the documentation uses.",
      },
      {
        title: "Smaller open items",
        body:
          "ToolCall.arguments is a string on the ReAct path and a mapping on native tool calling; reasoning_effort reaches a streamed turn and not a blocking one; a tool-free stream yields the model's own text rather than the sanitized answer; GuardrailChain.check(position=...) is not forwarded; and a spend cap still refuses a call that costs nothing.",
      },
    ],
  },
];

/** The names 1.1.0 added to `effgen.__all__`. */
export const newPublicNames110 = [
  "AgentThread",
  "Step",
  "SystemStep",
  "TaskStep",
  "TurnStep",
  "ThoughtStep",
  "ActionStep",
  "ObservationStep",
  "NudgeStep",
  "AnswerStep",
  "DelegationStep",
  "ContextBudgetExceededError",
  "CompactionPolicy",
  "ShortenOldestFirst",
  "SummarizeWithModel",
  "ThreadProjection",
  "NoParentContext",
  "ParentTask",
  "ParentAnswers",
  "LastCycles",
  "RenderedStep",
  "render_thread",
  "thread_as_text",
  "SubAgentManager",
  "SubAgentResult",
];

/* ── 1.0.1 ───────────────────────────────────────────────────────────────── */

/** Released 8 September 2026. */
export const RELEASE_DATE_1_0_1 = "8 September 2026";

/** `git rev-list --count v1.0.0..v1.0.1` in the framework repository. */
export const COMMITS_SINCE_1_0_0 = 80;

/** The four changes in 1.0.1 that existing code can see, in the order the changelog lists them. */
export const visibleChanges101: BreakingChange[] = [
  {
    title: "A run that stops without an answer reports failure, and raises by default",
    why:
      "In 1.0.0 three paths returned success=True with the loop's internal state in .output: a computation tool that tripped the repeat guard, a computation tool that returned a result it had already returned, and a model that gave no final answer after its tools ran. 1.0.1 returns success=False, outcome=\"stopped\" and a stop_reason naming the exit, and keeps what the model reached in .partial. Under the default raise_on_error=True the run raises RunStoppedError, which subclasses RuntimeError — what the iteration cap has always raised — and carries .response, .stop_reason and .partial.",
    migration: {
      source: `from effgen import Agent, AgentConfig, RunStoppedError

agent = Agent(AgentConfig(model="openai:gpt-5-nano"))
try:
    response = agent.run("What is 17 * 23?")
    print(response.outcome, response.stop_reason)
    print(response.text)
except RunStoppedError as exc:
    print(exc.stop_reason)
    print(exc.partial.text if exc.partial else "nothing to report")`,
      language: "python",
    },
    note:
      "Or pass raise_on_error=False and branch on .outcome, which is \"answered\", \"stopped\" or \"failed\". The text the model reached is in .partial.text, and still at metadata[\"partial_output\"] byte for byte. The same outcome reaches the CLI, the run store, batch rows and their CSV, effgen code, EvalResult.stop_reason and the server's effgen envelope.",
  },
  {
    title: "Citation markers are opt-in",
    why:
      "1.0.0 told the model to cite each passage inline as [1], [2], ... on every turn that followed a retrieval or search tool, whether or not you asked for citations. The markers pointed at nothing, and a question with a one-word answer came back with them attached. 1.0.1 adds them only when you ask, and a trailing marker on an answer you did not ask to be cited is removed.",
    migration: {
      source: `from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="openai:gpt-5-nano", cite_sources=True))

# or for a single call
agent.run("Summarise the retrieved notes.", cite_sources=True)`,
      language: "python",
    },
    note:
      "The rag preset asks already. When you do ask, the retrieval results are numbered with the citation indexes, so [n] is citations[n - 1], across calls and for rows with a URL.",
  },
  {
    title: "A streamed run sends the model's working before its answer",
    why:
      "Same task, same final answer, but the chunks a stream yields now start with the model's reasoning. Code that joins every chunk and shows the result shows the working first.",
    migration: {
      source: `events = list(agent.stream(task, include_events=True))
answer = "".join(e.text for e in events if e.kind == "answer")`,
      language: "python",
    },
    note: "Joining only the answer events reproduces response.output for a turn that answered.",
  },
  {
    title: "A small local model writes more and reaches for tools more often",
    why:
      "Every tool-calling path now tells the model what its tools are for. On a small model running on the local Transformers engine that means longer completions, and a tool call on questions it used to answer directly. The answers stay the same; the token count and the wall time go up.",
    migration: {
      source: `from effgen import Agent, AgentConfig, load_model

agent = Agent(AgentConfig(
    model=load_model("Qwen/Qwen2.5-1.5B-Instruct"),
    tools=tools,
    tool_use="sparing",   # a run that already has the answer gives it
))`,
      language: "python",
    },
    note:
      "tool_contract=\"\" states nothing about the tools at all, and tool_use=\"auto\" requires nothing whatever the tools declare.",
  },
];

export const oneZeroOneGroups: ChangeGroup[] = [
  {
    id: "outcomes-101",
    title: "What a run reports",
    lede: "A run that answered, a run the loop stopped, and a run that failed are three different results, and every surface now says which.",
    accent: "#00ff88",
    items: [
      {
        title: "outcome, stop_reason and partial on every response",
        body:
          "AgentResponse.outcome is \"answered\", \"stopped\" or \"failed\". stop_reason names the exit the run took and is present on every response — an answered run reports \"final_answer\". A stopped run's tool results and last reasoning travel in .partial, a PartialResult, instead of arriving where an answer would be. PartialResult and RunStoppedError are the two new names on the effgen package.",
      },
      {
        title: "The outcome on the command line, in the run store and over the server",
        body:
          "effgen runs list --status stopped finds the runs the loop ended before the model wrote an answer. Batch rows and their CSV carry the outcome, effgen code titles such a run Stopped, EvalResult.stop_reason records it, and the OpenAI-compatible server's effgen envelope carries stop_reason, outcome and partial.",
        code: { source: "effgen runs list --status stopped", language: "bash" },
      },
      {
        title: "A turn's own working is no longer read as its answer",
        body:
          "A turn that sent several tool calls at once could be treated as a final answer because of something as short as \"=\".",
      },
      {
        title: "A batched provider-side tool call records what it returned",
        body:
          "tool_calls[i].result is filled in instead of being left as None, and a result a tool computed but the answer left out is added back to the answer.",
      },
    ],
  },
  {
    id: "tools-101",
    title: "What a model is told about its tools",
    lede: "Tool definitions travel through the provider's API, not the prompt, so effGen now says what they are for — once, in the same words on every path.",
    accent: "#a78bfa",
    items: [
      {
        title: "Tool contracts picked from the tools' declared category",
        body:
          "effgen.prompts.tool_contract carries four contracts, chosen from each tool's ToolCategory: a tool that checks the model's work, one that does work the model cannot do, one that brings back material to answer from, and a general one for anything else or a mixed set. AgentConfig.tool_contract replaces the text, and an empty string states nothing.",
      },
      {
        title: "A ToolUsePolicy for whether a tool has to be called",
        body:
          "REQUIRED, AUTO or SPARING, set for every category and overridable with AgentConfig.tool_use. Every shipped default matches what 1.0.0 already did: a code executor or a system tool must actually run, and nothing else is pushed either way. tool_choice is now a run() keyword, and BaseModel.supports_forced_tool_call reports whether an adapter can enforce it.",
        code: {
          source: `from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[Calculator()],
    tool_use="required",   # answer only after calling a tool
))`,
          language: "python",
        },
      },
      {
        title: "An agent holding a code executor runs the code",
        body:
          "A first answer that only describes what the tool would have returned is sent back once, naming the tool, and the next turn is sent with tool_choice=\"required\" on adapters that support it.",
      },
      {
        title: "A declared output_schema is stated inside the loop",
        body:
          "On stream() as well as run(), so the same agent answers in the same shape either way. A tool with no declared category no longer raises from Agent.__init__.",
      },
    ],
  },
  {
    id: "loop-101",
    title: "The agent loop",
    lede: "The guards that end a run stop the runs that are stuck, not the ones still making progress.",
    accent: "#00e5ff",
    items: [
      {
        title: "The loop guards no longer stop runs that are still working",
        body:
          "A repeat of a call that already succeeded is answered from the run's own record and the run keeps going. The drift thresholds are bounded by the run's iteration budget, and when the loop does break, every tool category gets one turn to answer from what it has.",
      },
      {
        title: "A search that returns nothing is tried once more",
        body: "With a different query, and only once, before the run's answer is accepted.",
      },
      {
        title: "Every generation parameter reaches the provider",
        body:
          "Agent._generate copied one name out of its keyword arguments and dropped the rest, with no error and no log line.",
      },
    ],
  },
  {
    id: "cost-101",
    title: "The spend ledger",
    lede: "The budget check before each model call reads an index instead of the whole ledger.",
    accent: "#ffd700",
    items: [
      {
        title: "Budget checks use a covering index",
        body:
          "The check summed the ledger with a full table scan on every call, so its cost grew with the file. It now reads a total against an index on the timestamp, so its cost follows the window it asks about.",
      },
      {
        title: "effgen cost prune keeps the file small",
        body:
          "The ledger gains a row per model call and nothing removes one, so effGen logs a line naming the command once it grows large, and deletes nothing on its own. SQLiteCostStore gains spend_since, spend_today, spend_week, spend_month, count, count_since and prune.",
        code: {
          source: `effgen cost prune --dry-run           # what would go, keeping the last 90 days
effgen cost prune --older-than-days 30
effgen cost prune --keep-rows 100000  # keep the newest 100,000 events`,
          language: "bash",
        },
      },
    ],
  },
  {
    id: "models-101",
    title: "Models and languages",
    lede: "A default that points at a model the provider still serves, and keyword routing that reads Spanish as well as English.",
    accent: "#ff9500",
    items: [
      {
        title: "The Groq default names a model Groq still serves",
        body:
          "Groq retired llama-3.1-8b-instant and llama-3.3-70b-versatile. GROQ_DEFAULT_MODEL, the bundled catalog, the CLI help, the error messages and every shipped example moved to openai/gpt-oss-20b. Code that pinned a retired id names a live one itself; effgen models refresh does not change the module default.",
        code: { source: 'effgen run "What is 17 * 23?" -m groq:openai/gpt-oss-20b', language: "bash" },
      },
      {
        title: "A provider-prefixed model id loads when you also pass the provider",
        body:
          "load_model(\"groq:openai/gpt-oss-20b\", provider=\"groq\") used to raise Unknown Groq model. This is the path effgen run and effgen quickstart take when you give no --model.",
      },
      {
        title: "English and Spanish keyword matching",
        body:
          "The complexity analyzer, the decomposition engine, the sub-agent router and the prompt optimizer match both languages, with accents folded on both sides, so codigo and código agree. There is no language detection step and English behaviour is unchanged. A root agent's system_prompt now also reaches the sub-agents it spawns. Both from @acdonaire.",
      },
    ],
  },
];

/** The names 1.0.1 added to `effgen.__all__`. */
export const newPublicNames101 = ["PartialResult", "RunStoppedError"];

/* ── 1.0.0 ───────────────────────────────────────────────────────────────── */

/** Released 14 August 2026. The tag commit is dated the 15th. */
export const RELEASE_DATE = "14 August 2026";

/** `git rev-list --count v0.3.2..v1.0.0` in the framework repository. */
export const COMMITS_SINCE_0_3_2 = 640;

export const breakingChanges: BreakingChange[] = [
  {
    title: "Python 3.10 is no longer supported",
    why:
      "The floor is 3.11. tomllib, asyncio.timeout, datetime.UTC and the TimeoutError unification are all standard library from 3.11, and the package carried a hand-written fallback for each.",
    migration: {
      source: "python --version   # 3.11, 3.12, 3.13 or 3.14",
      language: "bash",
      output: "Python 3.11.15",
    },
    note: "Nothing in the API changed.",
  },
  {
    title: "AgentConfig.raise_on_error defaults to True",
    why:
      "A failed run raises its typed error instead of returning an AgentResponse with success=False and a plausible-looking string in .output — which a caller reading .output without checking .success never noticed.",
    migration: {
      source: `from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="openai:gpt-5-nano", raise_on_error=False))
response = agent.run("Reply with the single word: ready")

print(response.success, "|", response.text)`,
      language: "python",
      output: "True | ready",
    },
    note:
      "raise_on_error=False is also the documented setting for batch evaluation: with the flag off, a failed run's output is effGen's report of what stopped it, and the model's own text is in metadata[\"partial_output\"].",
  },
  {
    title: "An unreachable backend raises whatever that flag says",
    why:
      "A refused connection, an unresolvable host or a missing route is classified separately from a server that answered badly, and raises BackendUnreachableError. A task that ran and failed is a result you can inspect; a backend that was never reached is not, and returning one is how a whole batch completes against nothing and still looks healthy.",
    migration: {
      source: `from effgen import Agent, AgentConfig
from effgen.models.errors import BackendUnreachableError

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    base_url="http://127.0.0.1:9/v1",
    api_key="not-used",
    raise_on_error=False,
))

try:
    agent.run("Anything.")
except BackendUnreachableError as error:
    print(type(error).__name__)`,
      language: "python",
      output: `BackendUnreachableError`,
    },
    note: "There is no opt-out, by design. Catch the error where you want to handle it.",
  },
];

export const oneZeroGroups: ChangeGroup[] = [
  {
    id: "models",
    title: "Connecting to models",
    lede: "Where the weights live stopped being effGen's decision.",
    accent: "#00e5ff",
    items: [
      {
        title: "Point effGen at any OpenAI-compatible server",
        body:
          "base_url reaches load_model() and AgentConfig, so effGen drives a model you already serve — vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, LiteLLM, a gateway or a corporate proxy — instead of loading a second copy of the weights inside the agent process. The endpoint also comes from EFFGEN_BASE_URL, OPENAI_BASE_URL or OPENAI_API_BASE, in that order.",
        code: {
          source: `from effgen.models import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)`,
          language: "python",
        },
      },
      {
        title: "The server's ids are the server's",
        body:
          "No OpenAI catalog is consulted: the full sampling surface is offered, calls report no price rather than a fabricated $0, and list_served_models() asks the endpoint what it has. Pass context_length= when your server's window is not the assumed 32,768 tokens; effGen now warns when it is assuming, naming the value and the flag that sets the real one, instead of failing later at a size nobody chose.",
      },
      {
        title: "A tool loop you can write by hand, on any provider",
        body:
          "build_assistant_message() and build_tool_result_message() on BaseModel, and so on every adapter, build each provider's own message shape. A loop written once runs against OpenAI, Gemini, Anthropic, Groq, Together, Fireworks, Cerebras, Replicate and HF Inference instead of only the first.",
      },
      {
        title: "Python 3.14 is supported",
        body:
          "Installed and run, not just resolved. One caveat, because it changes the install line: plain pip install effgen[all] does not resolve on 3.14, because pip backtracks through the wide vLLM range into a release pinned to numba==0.61. On 3.14, install the extras through the shipped lock file first.",
        code: {
          source: `pip install -r requirements-all-py314-lock.txt
pip install --no-deps effgen`,
          language: "bash",
        },
      },
    ],
  },
  {
    id: "agents",
    title: "The agent surface",
    lede: "The extension points people arrive expecting, under the names they have elsewhere.",
    accent: "#00ff88",
    items: [
      {
        title: "Middleware around the agent loop",
        body:
          "Hooks at three points — the run, each model call, each tool call — each with a before and an after. A before hook can rewrite the request or short-circuit it entirely; an after hook can transform the result. Before hooks run in order and after hooks in reverse, so middleware nest. LoggingMiddleware and ToolApprovalMiddleware ship, and run(..., middleware=[...]) adds one for a single call.",
        code: {
          source: `from effgen import Agent, AgentConfig, LoggingMiddleware

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    middleware=[LoggingMiddleware()],
))
print(agent.run("Reply with one word: hello").text)`,
          language: "python",
          output: "Hello",
        },
      },
      {
        title: "One agent, many conversations",
        body:
          "run(..., session=...) builds the prompt from that conversation's history and appends the turn to it, restoring the agent's own session and memory afterwards, including when the run fails. A server handling many users no longer needs an agent object per user, nor history bookkeeping outside the framework.",
        code: {
          source: `agent.run("My dog is named Pixel.", session="user-123")
agent.run("My cat is named Mote.",  session="user-456")`,
          language: "python",
        },
      },
      {
        title: "Compaction is a strategy",
        body:
          "What gets dropped when a conversation outgrows the window is now yours to choose: SummarizeOldest (the default, unchanged), DropOldest (no model call, nothing invented), KeepFirstAndLast (the turns carrying the task survive verbatim) and KeepToolResults (the evidence stays, the reasoning is compacted) — or subclass CompactionStrategy. AgentConfig(tokenizer=...) measures the history in the units the window is measured in, rather than characters divided by four.",
      },
      {
        title: "A workflow that died part way through can be resumed",
        body:
          "WorkflowDAG.run() takes a checkpoint= store and a run_id=. Run the same line again after a crash and it continues where it stopped. Completed nodes are not re-run and their outputs flow downstream, failed nodes are retried, and a finished run replays its stored outputs without calling a model, so a retrying job runner cannot double-bill you. There is no separate resume call: an unknown run id starts from the beginning and a known one continues.",
        code: {
          source: `from effgen import FileCheckpointStore, WorkflowDAG, WorkflowNode

store = FileCheckpointStore()          # ~/.effgen/workflows by default
result = dag.run("Write the Q3 summary.", checkpoint=store, run_id="q3-summary")`,
          language: "python",
        },
      },
      {
        title: "AgentResponse.tool_calls reports the calls, not just how many",
        body:
          "Each entry is a ToolCall carrying name, arguments, result, duration, error and the iteration it was made on, with .failed and .by_name() to narrow them. Iterating the field used to raise TypeError: 'int' object is not iterable. It still compares and casts as the count, so tool_calls == 2 is unchanged, and .total says the number plainly. How much a call carries depends on the provider.",
        code: {
          source: `for call in result.tool_calls:
    print(call.name, call.arguments, "->", call.error or call.result)`,
          language: "python",
        },
      },
      {
        title: "load_env()",
        body:
          "Runs the same .env search the command line does, so a library script picks up the keys the CLI already finds. It honours EFFGEN_NO_DOTENV and never overwrites a value you exported.",
      },
    ],
  },
  {
    id: "code",
    title: "The coding agent",
    lede: "effgen code reads your workspace, proposes edits as unified diffs, and writes nothing until you say so.",
    accent: "#a78bfa",
    items: [
      {
        title: "Four permission modes, and an undo",
        body:
          "plan, ask, auto-edit and yes gate every write, every shell command and every commit. Writes are confined to the workspace, and a hunk that no longer applies is reported rather than clobbering the file. --undo rolls the last change back from a journal bounded to 100 entries.",
        code: {
          source: `effgen code "add a --dry-run flag to the importer"
effgen code --review                      # one read-only pass
effgen code --session-id my-refactor      # continue where you left off`,
          language: "bash",
        },
      },
      {
        title: "It knows the repository it is in",
        body:
          "Branch, status and a layout inventory that honours .gitignore go into the prompt, and an AGENTS.md brief is read when present. Git actions run through an allow-list, so push, reset, checkout, clean, rebase and force are refused before a subprocess starts — including when the model tries to reach them through the shell. A commit is confirmed like a write, uses the repository's own identity, and leaves your other staged work alone.",
      },
      {
        title: "A session that survives the process",
        body:
          "An interactive session keeps one run record across turns and carries a slash-command set — /plan, /diff, /apply, /reject, /undo, /run, /test, /context, /mode, /model, /cost, /trace, /git, /review, /compact, /save, /session and more. --session-id resumes it later. --review makes one read-only pass with a tool set that holds nothing that writes, runs or executes.",
      },
      {
        title: "Scriptable",
        body:
          "-p, --json and piped stdin run the single-shot path with byte-clean stdout. effgen doctor reports coding readiness — workspace, sandbox backend, git — and quickstart and tutorial include a coding step that writes and runs a real program.",
      },
    ],
  },
  {
    id: "surfaces",
    title: "Surfaces you can show someone",
    lede: "Everything here is self-contained: no CDN, no external font, nothing fetched at view time — enforced by a test that inspects what a browser would fetch.",
    accent: "#ffd700",
    items: [
      {
        title: "A real-time dashboard",
        body:
          "Per-model and per-provider cost, latency percentiles that are real percentiles, an error breakdown, a run waterfall, a model catalog panel and a history panel. Every chart is drawn locally.",
      },
      {
        title: "An in-browser playground",
        body:
          "On the existing chat endpoint, with model and preset pickers, tool toggles, the run's tool trace, and copy-as-curl, copy-as-CLI and copy-as-Python for the form you filled in.",
      },
      {
        title: "A cross-provider model and pricing browser",
        body:
          "In the terminal and in the dashboard: search, provider, capability, context and price filters, sorting and paging. models info shows every provider that serves a shared id.",
        code: {
          source: `effgen models browse --tools --min-context 100000 --sort price-in`,
          language: "bash",
        },
      },
      {
        title: "Shareable reports and run cards",
        body:
          "--report out.html for compare, eval, cost and loadtest, plus run --card and runs show <id> --card, and effgen report <result.json> to render a saved document after the fact. A generated report is inert: model output containing markup renders as text, and only http and https links keep an href.",
      },
      {
        title: "effgen top, and effgen battle",
        body:
          "top (alias monitor) is a terminal mission-control view over the telemetry you already collect — activity, traffic, per-model, spend and GPU panels, each stating the window and process it describes. battle races several models on one prompt side by side and reports the tally, the cost and an optional judge's verdict separately from the measurements.",
      },
      {
        title: "Graphs, palettes and themes",
        body:
          "A live multi-agent topology graph, terminal trace timelines, a workflow DAG diagram (workflow run --diagram) and a run waterfall. A command palette and keyboard-first navigation on both web surfaces, with a skip link, jump links, focus restoration and screen-reader announcements. Named terminal themes — default, high-contrast, monochrome, light — drawn from one palette the dashboard reads too.",
      },
    ],
  },
  {
    id: "history",
    title: "History, projects and the command line",
    lede: "A run is a record you can find again, and every command answers the same flags.",
    accent: "#f472b6",
    items: [
      {
        title: "Durable run and session history",
        body:
          "Every run is recorded with its model, provider, tokens, cost, status and task, keyed by the same run id its trace spans carry. Runs from the command line, a script and the server share one history and survive a restart, with search, status, model and date filters.",
        code: {
          source: `effgen runs list --status failed --since 7d
effgen sessions show <id>`,
          language: "bash",
        },
      },
      {
        title: "Project scaffolding",
        body:
          "effgen quickstart --init [DIR] writes effgen.yaml, a .env.example carrying one named variable per registered provider with no value invented, a runnable example.py and a .gitignore; puts a $1.00/day spend cap in force when none is configured; and prints the next three commands. effgen config init writes a document a run actually reads.",
      },
      {
        title: "Flags and output that behave the same everywhere",
        body:
          "--json on every command that had no machine output, and --json stdout is now a single valid document on a pipe and on a terminal, with no spinner, table or warning mixed in. -o picks its format from the extension. Thirteen short flags now mean the same thing across commands, and a bare group command prints its own help and exits 0 instead of reporting an unknown subcommand.",
      },
      {
        title: "Your own prompt templates load beside the shipped ones",
        body:
          "EFFGEN_PROMPTS_DIR names one or more directories; each *.py in them is imported and its templates registered under their own names, so a team's library sits next to the built-in one without a fork. prompts run now fails closed on an empty or truncated result rather than printing nothing and exiting 0, and reports the tokens, cost and latency of the call it made.",
      },
      {
        title: "Load testing through the server, not around it",
        body:
          "effgen loadtest --url drives a running effgen serve over HTTP, through auth, rate limiting and the middleware stack, instead of only driving an adapter directly.",
      },
    ],
  },
  {
    id: "truthful",
    title: "Results that report what actually happened",
    lede: "The largest and least visible part of the release: a pass over everything that used to report the wrong thing confidently.",
    accent: "#ff9500",
    items: [
      {
        title: "A turn that did nothing no longer reports success",
        body:
          "A coding turn whose every action failed, and a retrieval loop that produced no answer, are reported as partial outcomes with the recovered text under metadata[\"partial_output\"] and a typed reason for what stopped the run. A run stopped at the iteration cap reports the stop, not the last passage it retrieved, and carries that through every surface that shows it.",
      },
      {
        title: "A tool call the model wrote out instead of making is a failed turn",
        body:
          "Not an answer — including the shapes that used to slip through: a stray angle bracket, a missing separator, a query string with HTML entities, call syntax whose arguments were dropped, and a tag named after the tool itself.",
      },
      {
        title: "An unpriced model reports no cost, not a fabricated one",
        body:
          "A provider's placeholder rate made every id the bundled catalog had not seen read as priced, so a fine-tuned ft: id was billed at a made-up rate and the invented number reported as a published price. call_cost returns None for an unpriced model and 0.0 only for a genuine free tier, and every surface says \"no price\" instead of $0.",
      },
      {
        title: "Streamed runs report their cost and tokens",
        body:
          "On every provider, including Replicate and HF Inference, which recorded neither. model.total_tokens is correct on every adapter — six never assigned it at all — and a response that reports an all-zero usage block for a call it billed is estimated and flagged rather than recorded as free. Team and workflow totals now include the manager's own calls.",
      },
      {
        title: "A citation is a source the answer actually used",
        body:
          ".sources still carries every URL a search returned; .citations now carries the ones the answer references, and a PDF citation carries its page number.",
      },
      {
        title: "A model that could not run at all is reported as failed, not as scoring zero",
        body:
          "An evaluation where the key was missing or the provider refused every call used to print 0% beside the models that did run, which reads as a bad model rather than as a model that never answered.",
      },
    ],
  },
  {
    id: "toolcalls",
    title: "Tool calling across providers",
    lede: "Chat templates disagree about how a call is spelled. effGen now reads the shapes, not the model families.",
    accent: "#00c896",
    items: [
      {
        title: "A tool call written as XML tags is understood",
        body:
          "Many chat templates render JSON; others render nested tags. effGen read only the JSON spellings, so on a model whose template emits tags the turn parsed to nothing: no tool was called, and the run ended at the iteration cap. The reader is keyed on the shape rather than on a model family — five call tags, four argument tags, both <tag=NAME> and <tag name=\"NAME\"> — so any family whose template writes that shape can use tools.",
      },
      {
        title: "One call shape across every adapter",
        body:
          "generate_with_tools() takes config third on all ten adapters; it was messages on Groq, Together and Fireworks, so a positional call misrouted its argument and failed as a retryable error. Both spellings still work, told apart by type, so there is no migration.",
      },
      {
        title: "A rate limit no longer multiplies",
        body:
          "Three layers each retried a throttled call and multiplied rather than shared a budget: one client request became twelve upstream requests and held the caller 20.5 seconds at a stated 2-second delay. One layer now owns provider retry. The same measurement reads four requests and 6.7 seconds.",
      },
      {
        title: "A plain run() no longer fans out into sub-agents on its own",
        body:
          "AgentConfig.mode defaults to SINGLE. A task over roughly a hundred words used to become six billed calls, and a decomposed run could report a number the source text never contained. --mode auto opts back in, and a genuinely multi-part task still decomposes.",
      },
    ],
  },
  {
    id: "errors",
    title: "Errors that name the fix",
    lede: "Every message a reader sees is bounded, redacted and ends with what to do next.",
    accent: "#ff6b6b",
    items: [
      {
        title: "A connection failure names the endpoint the call was sent to",
        body:
          "Instead of pointing at the provider's status page, which is advice about the wrong machine when the server is yours. A URL with no scheme is refused, naming the environment variable it came from, and a blank endpoint variable no longer redirects every OpenAI call.",
      },
      {
        title: "A rate limit delivered as HTTP 413 is classified as one",
        body:
          "One provider reports a spent tokens-per-minute allowance that way. It used to be unknown, so there was no backoff and a throttle was reported as a permanent failure. A genuinely oversized body is still an invalid request.",
      },
      {
        title: "The submitted credential never reaches the caller",
        body:
          "effGen redacted its own message, but raise ... from exc kept the SDK exception, and a 401 body quotes the key. The whole __cause__/__context__ chain is now scrubbed, including per-SDK attributes and parsed JSON bodies, and a rendered traceback carries nothing. Quoted upstream text is bounded to 240 characters — one real provider body reached 42 kB.",
      },
      {
        title: "A malformed input names its file",
        body:
          "A workflow YAML that is not a workflow, a config file that is not a mapping, a drifted session or checkpoint, a damaged catalog snapshot and a batch row with no query text are each named with the file and the position, instead of raising from somewhere unrelated.",
      },
      {
        title: "A call can be bounded",
        body:
          "One adapter took no timeout at all and another's deadline governed polling only, so a peer that never answers held the call for 90 seconds. Both take timeout and max_retries now, and with_timeout() re-arms rather than firing once into an SDK retry loop that swallowed it — a 2-second bound now stops a call at 2.25 seconds instead of 22 to 50.",
      },
    ],
  },
  {
    id: "server",
    title: "The server, security and sandboxing",
    lede: "The OpenAI-compatible server, the guardrail presets and the execution sandbox.",
    accent: "#22d3ee",
    items: [
      {
        title: "One error envelope, and a loop that stays responsive",
        body:
          "The server answers every failure with the same envelope — unknown URLs, wrong methods, missing static assets, unhandled route errors, RBAC denials, the shutdown drain, websockets and the edge adapters. A non-streaming completion used to block the event loop, so /health timed out for the length of the call; it now runs off the loop, measured at 6 ms worst case during a 14.5 second completion.",
      },
      {
        title: "Content-free requests are refused before they are billed",
        body:
          "An empty or whitespace prompt, absent content and a non-positive max_tokens each return a 4xx before any upstream call. An absent provider key is a 503 on every provider, an upstream 429 passes its delay on as Retry-After, and a mid-stream failure emits a terminal error event rather than truncating the stream.",
      },
      {
        title: "Rate limiting is not defeated by a header",
        body:
          "X-Forwarded-For is trusted only when you enable it, and effgen serve no longer lets the ASGI server rewrite the client address behind that setting. Body size limits now cover /v1/embeddings, which used to accept an unbounded body.",
      },
      {
        title: "The sandbox masks credential stores and isolates the process table",
        body:
          "Executed code sees one process rather than the host's, and the credential directories beside it are masked. Both are reported on the result as credential_reads_masked and process_table_isolated. This is a deny-list over a known set of paths, not read confinement, and the documentation says so.",
      },
      {
        title: "The standard guardrail preset screens tool output for injection",
        body:
          "Not just input, so an instruction planted in a tool's return value no longer reaches the model under the default preset. standard also redacts personal data rather than blocking the message, so a customer quoting their own email address is answered instead of refused. strict still blocks.",
      },
    ],
  },
  {
    id: "local",
    title: "Local models, GPUs and long runs",
    lede: "What happens when the weights are on your own machine, and the run goes on for a while.",
    accent: "#a3e635",
    items: [
      {
        title: "A model that does not fit the GPU says so",
        body:
          "VRAM sizing reads free memory rather than total, the engine reconciles .device with where the parameters actually are, the run's metadata carries it, and require_gpu=True fails fast rather than falling back to CPU silently. Device memory comes back when a local model unloads — a 1.5B model used to keep 2.9 GB reserved.",
      },
      {
        title: "Automatic sharding across several GPUs no longer produces invalid output",
        body:
          "On a multi-GPU node, device_map=\"auto\" could place a model so that sampling read invalid logits and the run died in a CUDA assert. The engine now probes the logits after loading and pins the model to one device before sampling.",
      },
      {
        title: "Per-call sampling keywords are honoured on the local engines",
        body:
          "Including seed and stop_sequences, which one engine read off the config before it looked at the call. A local reasoning model is recognised from its own chat template, so it gets the larger budget instead of spending the base one on a hidden chain and returning nothing.",
      },
      {
        title: "A long conversation stops growing its own prompt",
        body:
          "Session summaries were unbounded and replayed into every prompt, so past a threshold each turn added another summary until every call was refused for exceeding the context window. Summaries are now folded within a token budget measured with the model's own tokenizer, and a 25,000-turn session stays flat.",
      },
      {
        title: "Long runs hold up under concurrency",
        body:
          "Tool discovery, registry replacement and rate-limit accounting are serialised; costs and tokens are folded under a lock, so a shared adapter's totals match the calls; two writers of one session no longer publish a blend. Batch rows no longer contaminate each other: eight concurrent rows report eight distinct costs, and the job total reconciles with the sum.",
      },
    ],
  },
  {
    id: "rag",
    title: "Documents, retrieval and batch input",
    lede: "What goes into a knowledge base, and what comes back out of it.",
    accent: "#818cf8",
    items: [
      {
        title: "The rag preset refuses to run without a knowledge base",
        body:
          "Instead of succeeding with zero documents. Retrieval also keeps distinct topics: the preset configures a wider top_k with MMR re-ranking, so a two-topic question returns both topics.",
      },
      {
        title: "Ingestion says what it skipped and why",
        body:
          "A corrupt file, an empty file, a file whose content duplicates an earlier one, an image and an unsupported extension each have their own reason, and DocumentIngester.last_summary reports what was indexed. PDFs carry page numbers.",
      },
      {
        title: "A weak model no longer answers with the passages",
        body:
          "The prompt now ends with an answer-shaping instruction after a retrieval tool, and both loop fallbacks give the model one tool-free turn to answer from what it has. Measured on the worst case: verbatim passage dumps went from 8 of 9 runs to 0 of 9, with citations on every run.",
      },
      {
        title: "Structured output extraction stopped corrupting valid JSON",
        body:
          "Repairs used to run inside string literals, so a value containing \", note:\" was rewritten into something unparseable. Measured over 8,000 adversarial examples: 36 mis-parses before, 0 after, with 1,427 inputs newly recovered.",
      },
      {
        title: "Batch input is read carefully",
        body:
          "A row keyed on prompt, input, question or text is recognised, a scalar or dict row does not become a prompt, CSV rows report the right line, a non-UTF-8 file names itself, and --strict fails the job on any unusable row. run --file reads source code and plain text, not only documents, and refuses binaries.",
      },
    ],
  },
  {
    id: "terminal",
    title: "The terminal, the tools and the install",
    lede: "The parts you meet before you write any code.",
    accent: "#fb923c",
    items: [
      {
        title: "Every command works on a terminal that cannot encode what effGen prints",
        body:
          "Twenty-two commands used to exit non-zero purely because of the console encoding. Text is folded to ASCII where it becomes bytes, and --json escapes rather than transliterates, so a French or Chinese answer survives a hard-ASCII console byte for byte.",
      },
      {
        title: "Piped output is clean, and a closed pipe is quiet",
        body:
          "No spinner, no placeholder, no chrome on stdout, one answer per input line under -q, and zero colour codes under NO_COLOR. effgen ... | head used to end in a BrokenPipeError traceback; the command now exits 141, the convention the shell expects.",
      },
      {
        title: "The whole command surface works without rich and without torch",
        body:
          "Twelve commands used to exit with a missing-module error, and a direct engine import reported the import system rather than naming PyTorch and how to install it.",
      },
      {
        title: "A tool that cannot do its job says so",
        body:
          "Translation with no language pair available, a knowledge-base search the API refused, a news fetch where every source was unreachable, and a web search whose unset filters were sent to the backend all reported success or the wrong error. Each now fails with the reason. A blocked request is not read as an empty result.",
      },
      {
        title: "Every documentation snippet runs",
        body:
          "The 57 tool gallery snippets were rewritten to the awaited keyword API, all 30 network snippets now check ToolResult.success before reading output, and the command-line pages were re-run command by command. ./install.sh no longer fails when run without a terminal, and the docker compose file binds to loopback.",
      },
    ],
  },
];

/** The 19 names 1.0.0 adds to the top-level package. */
export const newPublicNames = [
  "OpenAICompatibleAdapter",
  "BackendUnreachableError",
  "AgentMiddleware",
  "MiddlewareChain",
  "LoggingMiddleware",
  "ToolApprovalMiddleware",
  "ToolCall",
  "ToolCallList",
  "CompactionStrategy",
  "SummarizeOldest",
  "DropOldest",
  "KeepFirstAndLast",
  "KeepToolResults",
  "WorkflowCheckpoint",
  "CheckpointStore",
  "FileCheckpointStore",
  "InMemoryCheckpointStore",
  "SystemPromptLeakGuardrail",
  "load_env",
];

/* ── Earlier releases ────────────────────────────────────────────────────── */

export interface EarlierRelease {
  version: string;
  date: string;
  title: string;
  summary: string;
}

/** Every release before 1.0.0, with the date and headline CHANGELOG.md gives it. */
export const earlierReleases: EarlierRelease[] = [
  {
    version: "0.3.2",
    date: "5 July 2026",
    title: "Usability, robustness and polish",
    summary:
      "A point release driven by living with the framework: results integrity re-certified, grounding traceable, a consistent server contract, batch that survives real data, and observability an on-call rotation can use. No breaking API changes.",
  },
  {
    version: "0.3.1",
    date: "29 June 2026",
    title: "Real-world usability and polish",
    summary:
      "Grounded results carry the sources they were built from, reasoning models finish token-heavy work, a custom persona is honoured on every path, a multi-agent team reports the failure instead of a partial result, and a knowledge domain becomes a runnable agent in one call. No breaking API changes.",
  },
  {
    version: "0.3.0",
    date: "19 June 2026",
    title: "Stabilization and hardening",
    summary:
      "No new providers, tools or subsystems. Failures became loud and typed instead of silently succeeding, the model catalog updates itself, local GPUs work out of the box, the server fails closed, the built-in tools are sandboxed, and import effgen is effectively instant.",
  },
  {
    version: "0.2.10",
    date: "27 May 2026",
    title: "Security, edge and developer experience",
    summary:
      "Secret scanning, dependency auditing, an SBOM pipeline, supply-chain integrity verification, a sandboxed code executor, OIDC auth with RBAC and per-request audit logging, four deploy targets, and three developer-experience surfaces.",
  },
  {
    version: "0.2.9",
    date: "23 May 2026",
    title: "Observability and reliability",
    summary:
      "Structured JSON logs with secret redaction, OpenTelemetry tracing with configurable samplers, Prometheus histograms, SLO tracking, circuit breakers, bulkheads, a deterministic chaos harness and load testing.",
  },
  {
    version: "0.2.8",
    date: "21 May 2026",
    title: "Multimodal input",
    summary:
      "Image, audio and video become input types in their own right across six cloud providers plus local MLX-VLM, with a unified content schema, per-provider preprocessing and capability gating — no silent downcast when a model lacks vision or audio.",
  },
  {
    version: "0.2.7",
    date: "20 May 2026",
    title: "The prompt library",
    summary:
      "A curated, domain-organised catalog of reusable prompt templates, paired with a golden evaluation harness, a command-line surface and an interactive playground.",
  },
  {
    version: "0.2.6",
    date: "19 May 2026",
    title: "Documents, media and communication tools",
    summary:
      "Fourteen new built-in tools across OCR, audio transcription, image analysis, document parsing, geo and weather, and email and webhook communication — plus the media and notify presets.",
  },
  {
    version: "0.2.5",
    date: "18 May 2026",
    title: "Thirteen free, no-auth tools",
    summary:
      "Academic search, news and RSS, YouTube, social media, translation, language detection and QR codes, all wired into the research and general presets.",
  },
  {
    version: "0.2.4",
    date: "14 May 2026",
    title: "Model routing and cost tracking",
    summary:
      "A ModelRouter with three composable routing policies, transparent provider failover with retry logic, cross-process rate-limit coordination, and a persisted cost ledger behind effgen cost.",
  },
  {
    version: "0.2.3",
    date: "4 May 2026",
    title: "Nine inference backends",
    summary:
      "Groq, Together, Fireworks, Replicate and HF Inference join the provider set, each with streaming, native tool calling, cost tracking and rate-limit coordination, behind one provider interface.",
  },
  {
    version: "0.2.2",
    date: "28 April 2026",
    title: "The Gemini adapter, expanded",
    summary:
      "The current model families, thinking budgets, search grounding, the Files API and three provider-native tools.",
  },
  {
    version: "0.2.1",
    date: "25 April 2026",
    title: "Cerebras, and a modern OpenAI adapter",
    summary:
      "The Cerebras backend with streaming, native tool calling and cost tracking, alongside reasoning models, reasoning effort, prompt-cache reporting and structured outputs on OpenAI.",
  },
  {
    version: "0.2.0",
    date: "9 April 2026",
    title: "From toolkit to platform",
    summary:
      "Native tool calling, guardrails, multi-agent orchestration, RAG pipelines, evaluation and an API server.",
  },
  {
    version: "0.1.0",
    date: "1 March 2026",
    title: "Prompts built for small models",
    summary:
      "Dynamic system prompts carrying exact tool-usage examples, per-family prompt formatting, and tool fallback chains for when a tool fails.",
  },
  {
    version: "0.0.1",
    date: "31 January 2026",
    title: "The first release",
    summary:
      "The agent loop, task management, agent state and the ReAct pattern, built for models between one and seven billion parameters.",
  },
];
