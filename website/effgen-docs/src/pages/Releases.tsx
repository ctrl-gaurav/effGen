import { Tag } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { publicNameCount, pythonVersions, siteData, version } from '../siteData';

/** The published release history, newest first. Dates are from CHANGELOG.md. */
const HISTORY = [
  ['0.3.2', '2026-07-05', 'Results integrity, traceable grounding, a consistent server contract, and batch that survives real data.'],
  ['0.3.1', '2026-06-29', 'Evidence on every result: populated sources and citations, cost and latency in metadata, and a system prompt that steers every path.'],
  ['0.3.0', '2026-06-19', 'Stabilisation and hardening. No new providers or subsystems — fail-closed behaviour, a drift-aware model catalog and real GPU support.'],
  ['0.2.10', '2026-05-27', 'Security and supply chain: secret scanning, dependency auditing, an SBOM pipeline and a sandboxed code executor.'],
  ['0.2.9', '2026-05-23', 'Observability: structured logs with secret redaction, OpenTelemetry tracing and Prometheus metrics.'],
  ['0.2.8', '2026-05-21', 'Multimodal input — image, audio and video as ordinary message content across six cloud providers.'],
  ['0.2.7', '2026-05-20', 'The prompt library: a domain-organised template catalog with an evaluation harness and a playground.'],
  ['0.2.6', '2026-05-19', 'Document, media and communication tools — OCR, transcription, image analysis, document parsing, geo and mail.'],
  ['0.2.5', '2026-05-18', 'Free, no-auth tools for academic research, news, RSS, YouTube, social media, translation and QR codes.'],
  ['0.2.4', '2026-05-14', 'The model router: three composable policies, provider failover with retry, and cross-process rate-limit coordination.'],
  ['0.2.3', '2026-05-04', 'The provider ecosystem grew to nine backends — Groq, Together, Fireworks, Replicate and HuggingFace Inference.'],
  ['0.2.2', '2026-04-28', 'Gemini: newer model families, a thinking budget, Google Search grounding, the Files API and three native tools.'],
  ['0.2.1', '2026-04-25', 'The Cerebras backend, and a modernised OpenAI adapter with the reasoning tier.'],
  ['0.2.0', '2026-04-09', 'Native tool calling, guardrails, multi-agent orchestration, RAG pipelines and evaluation.'],
  ['0.1.3', '2026-03-25', 'Sub-agent depth limiting, and guidance for answering without a tool.'],
  ['0.1.2', '2026-03-12', 'Ten example agents and cross-model prompt work.'],
  ['0.1.1', '2026-03-06', 'Licence and packaging fixes.'],
  ['0.1.0', '2026-03-01', 'Foundation hardening: dynamic tool prompts, model-specific formatting and tool fallback.'],
  ['0.0.2', '2026-02-03', 'The retrieval and agentic-search tools.'],
  ['0.0.1', '2026-01-31', 'The first release: the agent system, task management and agent state.'],
] as const;

export default function Releases() {
  return (
    <DocPage
      subtitle="What each release changed, newest first."
      icon={<Tag size={48} />}
    >
      <h2>{version} — 1 October 2026</h2>
      <p>
        This release is about how a run ends, and how a tool call is read. A run that stops making
        progress is asked for its answer instead of going round to its iteration cap, and every run
        says how it ended: <code>response.termination</code> is <code>"done"</code>,{' '}
        <code>"not_possible"</code>, <code>"stuck"</code>, <code>"tool_failed"</code> or{' '}
        <code>"error"</code>. A tool that keeps failing on its own side ends the run as{' '}
        <code>tool_failed</code> rather than leaving the model to answer around it. A tool call the
        model wrote in a broken or unexpected shape is read and run by default. A model you serve
        yourself, or run on a local engine, is measured once for what it actually does with a tool,
        and <code>tool_calling_mode="auto"</code> follows what was measured. One agent can serve
        many overlapping conversations without mixing them.
      </p>

      <Callout type="warning" title="Ten changes are visible to existing code">
        <p>
          The ones most code meets first: <code>max_turns_without_progress</code> defaults to{' '}
          <code>2</code> and a stuck run gets one closing request, so a run that raised{' '}
          <code>RunStoppedError</code> in 1.2.0 can now return an answer; a run whose tool keeps
          failing stops with the new stop reason <code>tool_failed</code>;{' '}
          <code>recover_lost_tool_calls</code> defaults to <code>True</code>, so some runs make more
          tool calls; and a served or local model is probed once at{' '}
          <code>tool_calling_mode="auto"</code>. <Link to="/migration">Migrating to {version}</Link>{' '}
          walks all ten and the setting that restores each 1.2.0 behaviour. The public surface grew
          from 251 names to {publicNameCount}, and nothing was removed or renamed.
        </p>
      </Callout>

      <Callout type="note" title="Where it falls short">
        <p>
          Small models given a search tool now cost noticeably more per run, because they are made
          to use it. Runs that use tools still make more model calls than they need to.
        </p>
      </Callout>

      <h3>How a run ends</h3>
      <p>
        After <code>max_turns_without_progress</code> turns in a row (now 2) that bring no new tool
        result, the next turn offers no tools and asks for the answer; a turn that declares no
        action after a result, such as <code>Action: None</code> or{' '}
        <code>Action: (continue reasoning)</code>, is asked at once. A run that would have stopped
        on a loop guard or its iteration cap while holding tool results gets one closing request —
        its own calls and results with no tools — and when the reply is an answer the run succeeds
        with <code>metadata["answer_source"] == "closing_request"</code>. A tool that fails on its
        own side three times in a row is unavailable for the run, and a run left with no usable
        tool stops with <code>tool_failed</code>; four failures in a row on a tool’s input withdraw
        that tool for the run. An answer that is a tool’s error message is never a success.{' '}
        <code>metadata["tool_results"]</code>, <code>unavailable_tools</code>,{' '}
        <code>tool_calling</code> and <code>answer_source</code> are new metadata keys.
      </p>

      <h3>How a tool call is read</h3>
      <p>
        With <code>recover_lost_tool_calls</code> now on, a call written as a Python literal, with
        raw line breaks or unescaped quotes inside its JSON, with its last string closed one bracket
        early, or followed by text inside its tag, runs instead of ending the run with{' '}
        <code>written_tool_call</code>. Under either setting, arguments that arrive as a string are
        decoded, a call missing a required argument is asked for again rather than dispatched, and
        positional values are named in the tool’s parameter order.
      </p>

      <h3>The capability probe</h3>
      <p>
        At <code>tool_calling_mode="auto"</code>, the first agent with tools built for a model
        behind a <code>base_url</code>, or on a local engine, runs a short probe and stores the
        result in <code>~/.effgen/capabilities.json</code> for every later agent. A model that
        answers from memory while holding a search tool has its information-retrieval tools made
        must-call; a model whose native calls the server does not carry back runs in the ReAct text
        frame. <code>probe_tool_calling()</code> and <code>ToolCallingProbe</code> are the two new
        names, and <code>effgen doctor --probe MODEL</code> measures a model now.{' '}
        <code>AgentConfig(capability_probe=False)</code> or <code>EFFGEN_CAPABILITY_PROBE=0</code>{' '}
        turns it off; first-party cloud adapters are never probed. The same store remembers a
        provider that rejects stop sequences beside tools, and a server that rejects{' '}
        <code>reasoning_effort</code> — which now reaches a model behind <code>base_url</code>, and
        Groq.
      </p>

      <h3>Sessions</h3>
      <p>
        <code>run(session=...)</code>, <code>run_async(session=...)</code> and{' '}
        <code>stream(session=...)</code> hold their conversation on the call, so overlapping calls on
        one agent each read and record only their own session. <code>stream()</code> now honours{' '}
        <code>session=</code>, and on an agent bound to a session it appends and saves each answered
        turn, as <code>run()</code> always did.
      </p>

      <h3>Known issues</h3>
      <ul>
        <li>
          Small models given a search tool cost more: when the probe makes the search tool
          must-call, runs on question-answering tasks make several times as many model calls.{' '}
          <code>tool_use="auto"</code> or <code>capability_probe=False</code> turns it off for an
          agent.
        </li>
        <li>
          A run whose calls keep failing on their own input can be stopped by the loop guard before
          the four-failure bound, ending <code>stuck</code> rather than <code>tool_failed</code>.
        </li>
        <li>
          A run whose tools all fail on their own side gets no closing request, even when the model
          had already written what it would answer.
        </li>
        <li>
          On one set of coding tasks, a mid-size model answers correctly less often than on 1.2.0:
          runs the old loop guard ended with a usable result are now asked for their answer, and
          some answer from the wrong result.
        </li>
        <li>The framework’s own time on a long run is higher than in 1.2.0, within its budget.</li>
        <li>
          A probe’s requests are booked in the first agent’s cost ledger with no tag marking them as
          the probe’s, and a Groq call’s retry wait is booked as framework time.
        </li>
        <li>
          The readers now on by default have edges: <code>"print("")"</code> is read as{' '}
          <code>print()</code>, and a closing reply on the text scaffold that is another call ends
          the run stuck after one request.
        </li>
        <li>
          With one agent and many sessions, <code>stream()</code> runs no middleware and{' '}
          <code>resume(session=...)</code> restores the checkpoint’s memory onto the agent’s own
          memory.
        </li>
      </ul>

      <h2>1.2.0 — 27 September 2026</h2>
      <p>
        This release is about what a run costs, and whether you can see it. Every run now keeps a
        ledger — <code>response.ledger</code>, a <code>RunLedger</code> — of its model calls, tool
        calls, prompt, completion and cached tokens, its cost, and where its time went: waiting on
        the model, on tools, on the caller, on child runs, or inside the framework. Around it, a
        provider’s prompt cache is kept warm and its hits are read and priced where the provider
        reports them, a request carries less of the framework’s own text, a tool result the model
        writes itself is never taken as the answer, and a model you serve yourself reads as unpriced
        rather than free.
      </p>

      <Callout type="warning" title="Fourteen changes are visible to existing code">
        <p>
          The ones most code meets first: a call refused by a spent cap raises{' '}
          <code>BudgetExceededError</code>, which is not a <code>RuntimeError</code>; a model with no
          published price reports its cost as <code>None</code> rather than <code>0.0</code>; the
          tool contracts ask for less and come before the caller’s task; and a decomposed run’s{' '}
          <code>tokens_used</code> includes its sub-agents’ tokens.{' '}
          The migration guide walks all fourteen. The public surface grew from 250 names to 251,
          and nothing was removed or renamed.
        </p>
      </Callout>

      <h3>What a run spent</h3>
      <p>
        <code>RunLedger</code> counts a run’s model and tool calls and its tokens, prices them, and
        splits its wall time into model, tool, caller, child and framework time, per run and per
        iteration. It is on <code>AgentResponse.ledger</code>, <code>Agent.last_stream_ledger</code>{' '}
        and <code>Checkpoint.ledger</code>; sub-agents, workflow nodes, team members and an agent run
        inside a tool attach to their parent once, and <code>total()</code> adds them up. The run
        store gains the same counts, Prometheus gains <code>effgen_run_framework_seconds</code>,{' '}
        <code>effgen_model_cost_usd_total</code> and <code>effgen_model_unpriced_calls_total</code>,
        and <code>execution_time</code> now includes the work after the model’s last answer.
      </p>

      <CodeBlock
        code={`from effgen import Agent, AgentConfig
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
print(ledger.cost_usd)   # None: a model you serve yourself has no published price`}
      />

      <h3>Cost and the spend cap</h3>
      <p>
        A model with no published price reads as unknown: <code>CostTracker.total_cost()</code>,{' '}
        <code>RunLedger.cost_usd</code> and <code>effgen cost --json</code> report{' '}
        <code>None</code> where 1.1.0 reported <code>0.0</code>. A spent cap refuses only calls that
        cost money, and raises <code>BudgetExceededError</code> before any request is sent. The
        spend ledger file folds its oldest rows into per-model totals at 250,000 rows (
        <code>EFFGEN_COST_MAX_ROWS</code>), and many agents in one process no longer queue on it.
      </p>

      <h3>What a request carries</h3>
      <p>
        The tool contracts no longer ask for text the task did not ask for, and come before the
        caller’s task. Tool rules are stated once, a repeated tool result is sent once, and a
        tool-holding turn is sent the single stop sequence <code>"\nObservation:"</code>, so a
        result the model writes itself is cut off; an action written anyway runs and what follows
        it is discarded. On a provider with a prompt cache, a run keeps one request shape so the
        cached prefix is reused, <code>cache_system_prompt</code> and <code>cache_tools</code> work
        on Anthropic, and cached tokens are read on Groq, Together, Fireworks, Cerebras and Gemini.{' '}
        <code>reasoning_effort</code> now reaches <code>run()</code> and <code>run_async()</code>,
        and the output budget follows a declared <code>output_schema</code>.
      </p>

      <h3>New settings, and effgen bench</h3>
      <p>
        <code>AgentConfig.answer_style</code> (<code>"brief"</code>, <code>"full"</code> or your own
        sentence), <code>max_turns_without_progress</code> and <code>recover_lost_tool_calls</code>{' '}
        are new, each off unless set in 1.2.0 (1.3.0 turns the last two on) and each also a{' '}
        <code>run()</code> keyword. A tool whose calls
        keep returning new results is no longer withdrawn at a fixed count.{' '}
        <code>effgen bench init</code>, <code>run</code> and <code>compare</code> measure an agent
        on a suite of your own tasks and print a noise band beside every difference; the suite
        format is in the framework’s <code>docs/cli/bench.md</code>.
      </p>

      <h3>Fixed</h3>
      <p>
        <code>AgentConfig(model="openai:&lt;id&gt;", base_url=...)</code> sends the id without the
        prefix. Concurrent streamed runs no longer hand a tool another stream’s arguments, and
        in-process local engines serve concurrent agents. Groq hands a tool call it could not parse
        back to the loop. A reply cut off by its token budget carries its own guidance and is not
        retried.
      </p>

      <h3>Known issues</h3>
      <ul>
        <li>
          Runs that use tools still make more model calls than they need to; reducing that is the
          main work of the next release.
        </li>
        <li>
          One <code>Agent</code> serving concurrent <code>run(session=...)</code> calls can mix the
          conversations. Use one agent per concurrent session. Fixed in 1.3.0.
        </li>
        <li>
          A provider that rejects <code>stop</code> beside <code>tools</code> answers HTTP 400
          through a stock adapter until it declares <code>supports_stop_with_tools()</code> as{' '}
          <code>False</code>. Since 1.3.0 the request is retried with the stops applied locally.
        </li>
        <li>Streamed runs record no Prometheus series.</li>
        <li>
          The chat and <code>effgen code</code> <code>/cost</code> commands still print{' '}
          <code>$0.00</code> for unpriced turns.
        </li>
        <li>
          <code>ToolCall.arguments</code> is a string on the ReAct path and a mapping on native tool
          calling.
        </li>
      </ul>

      <h2>1.1.0 — 14 September 2026</h2>
      <p>
        A run now keeps its conversation as typed steps instead of one growing string. That string
        was assembled in three places, read back with regular expressions in four, and thrown away
        at the end of the run. In its place is an <code>AgentThread</code>, which the loop builds,
        the prompt is rendered from, the checkpoint stores, and the caller can read. A finished run
        hands back what it did. A run is bounded by the prompt tokens it may send. A saved run
        resumes where it stopped. And there is one agent loop rather than three, so a streamed run
        sends what a blocking one sends.
      </p>

      <Callout type="warning" title="Eleven changes are visible to existing code">
        <p>
          The ones most code meets first: <code>response.metadata["thread"]</code> is the live
          thread, so serialise with <code>response.to_dict()</code>; a session’s earlier turns
          render differently in the prompt and travel as their own messages on a model that takes a
          conversation; <code>stream()</code> now behaves like <code>run()</code>; and a run is
          bounded by <code>context_budget="auto"</code>.{' '}
          <Link to="/migration">Migrating to {version}</Link> walks all eleven. The public surface
          grew from 225 names to 250, and nothing was removed or renamed.
        </p>
      </Callout>

      <h3>The conversation a run keeps</h3>
      <p>
        <code>AgentThread</code> is an ordered list of <code>SystemStep</code>,{' '}
        <code>TaskStep</code>, <code>TurnStep</code>, <code>ThoughtStep</code>,{' '}
        <code>ActionStep</code>, <code>ObservationStep</code>, <code>NudgeStep</code>,{' '}
        <code>DelegationStep</code> and <code>AnswerStep</code>. <code>response.thread</code> is the
        run’s; <code>render_thread()</code> hands the steps back one at a time as{' '}
        <code>RenderedStep</code> rows and <code>thread_as_text()</code> as one block, both redacted
        by default. The command line (<code>effgen run --show-thread</code>), the run card, the debug
        inspector and the dashboard render the same steps.
      </p>

      <CodeBlock
        code={`from effgen import AgentThread, TaskStep, AnswerStep, render_thread, thread_as_text

thread = AgentThread(steps=[TaskStep(text="What is 17 * 23?"), AnswerStep(text="391")])
for step in render_thread(thread):
    print(step.position, step.kind, step.label, "|", step.body)
print(thread_as_text(thread))`}
      />

      <Terminal
        command="python thread_demo.py"
        output={`1 task task | What is 17 * 23?
2 answer answer | 391
  1. task
     | What is 17 * 23?
  2. answer
     | 391`}
        caption="Run against effGen 1.1.0."
      />

      <h3>Keeping a conversation inside its budget</h3>
      <p>
        <code>AgentConfig.context_budget</code> (default <code>"auto"</code>) bounds what a run may
        send, and <code>AgentConfig.compaction</code> takes a <code>CompactionPolicy</code>. The
        default, <code>ShortenOldestFirst</code>, shortens an old tool result, then drops an old
        thought, then replaces whole answered cycles with one <code>NudgeStep</code>, without a model
        call; <code>SummarizeWithModel</code> is opt-in. The frame, the task, the two most recent
        complete cycles and the answer are never touched, and a run that still cannot fit raises{' '}
        <code>ContextBudgetExceededError</code>. <Link to="/compaction">Compaction</Link> has the
        details.
      </p>

      <h3>Resuming and orchestration</h3>
      <p>
        A checkpoint stores the run’s steps, so <code>agent.resume()</code> continues the
        conversation instead of restarting the task, and a 1.0.x checkpoint still resumes through{' '}
        <code>Checkpoint.to_thread()</code>. Workflow, team and sub-agent results carry their runs’
        threads. A <code>ThreadProjection</code> — <code>NoParentContext</code> (the default),{' '}
        <code>ParentTask</code>, <code>ParentAnswers</code> or <code>LastCycles</code> — decides
        what a child run starts with. <code>SubAgentManager</code> and <code>SubAgentResult</code>{' '}
        are now importable from <code>effgen</code>.
      </p>

      <h3>The prompt protocol</h3>
      <p>
        <code>AgentConfig.prompt_protocol</code> is <code>"flat"</code>, <code>"messages"</code> or{' '}
        <code>"auto"</code>, default <code>"auto"</code>. A run continuing a session sends its
        conversation as the messages it was, on a model that takes them; a run continuing nothing
        keeps the flat string every earlier release sent. <code>"messages"</code> ships opt-in.{' '}
        <code>OpenAIAdapter</code> now carries a tool call into <code>tool_calls</code> and a tool
        result into a <code>tool</code> message, both of which were dropped before.
      </p>

      <h3>Fixed</h3>
      <p>
        <code>effgen run --json</code>, <code>-o</code> and <code>--card</code> work on a run that
        called a tool, and the three documents go through one scrubber. A prompt larger than the
        model’s window is sent once instead of retried.
      </p>

      <h3>Known issues</h3>
      <ul>
        <li>
          Once the guards stop offering tools, the turn that asks for the answer stays on messages,
          but its last user message still carries the run’s own calls and results as text.
        </li>
        <li>A streamed run can hand a tool a truncated argument.</li>
        <li>
          <code>AgentConfig(model="openai:&lt;id&gt;", base_url=...)</code> sends the prefix on the
          wire; write the id without it when you name your own server.
        </li>
        <li>
          <code>ToolCall.arguments</code> is a string on the ReAct path and a mapping on native tool
          calling.
        </li>
        <li>
          A configured daily spend cap, once spent, also refuses calls to a model you serve
          yourself.
        </li>
      </ul>

      <h2>1.0.1 — 8 September 2026</h2>
      <p>
        This release fixed how the framework reports what a run did, what it puts in a prompt, and
        what its own bookkeeping costs. A run that stops without an answer now says so instead of
        handing back its working notes. Citation markers are opt-in instead of being added to every
        retrieval answer. The loop guards no longer stop a run that is still making progress. Every
        tool-calling path now tells the model what the tools are for. The budget check before each
        model call reads an index instead of the whole spend ledger. And the Groq default points at
        a model Groq still serves.
      </p>

      <Callout type="warning" title="Four changes are visible to existing code">
        <p>
          A run that stops without an answer returns <code>success=False</code> and raises{' '}
          <code>RunStoppedError</code> under the default <code>raise_on_error=True</code>; citation
          markers are added only with <code>cite_sources=True</code>; a stream sends the
          model’s working before its answer; and a small local model writes more and calls tools
          more often. <Link to="/migration">Migrating to {version}</Link> has the code for each. The
          public surface grew from 223 names to 225, and nothing was removed or
          renamed.
        </p>
      </Callout>

      <h3>What a run reports</h3>
      <p>
        <strong>Answered, stopped or failed.</strong> <code>AgentResponse.outcome</code> names how a
        run ended, <code>stop_reason</code> names the exit it took and is present on every response,
        and a stopped run’s tool results and last reasoning travel in <code>.partial</code>, a{' '}
        <code>PartialResult</code>, instead of arriving where an answer would be. The same outcome
        reaches <code>effgen runs list --status stopped</code>, batch rows and their CSV,{' '}
        <code>effgen code</code>, <code>EvalResult.stop_reason</code> and the OpenAI-compatible
        server’s <code>effgen</code> envelope. <Link to="/errors">Errors</Link> has the vocabulary.
      </p>

      <CodeBlock
        code={`from effgen import Agent, AgentConfig, RunStoppedError

agent = Agent(AgentConfig(model="openai:gpt-5-nano"))
try:
    response = agent.run("What is 17 * 23?")
    print(response.outcome, response.stop_reason)
    print(response.text)
except RunStoppedError as exc:
    print(exc.stop_reason)
    print(exc.partial.text if exc.partial else "nothing to report")`}
      />

      <h3>What a model is told about its tools</h3>
      <p>
        <strong>Tool contracts.</strong> <code>effgen.prompts.tool_contract</code> carries four
        contracts, picked from each tool’s declared <code>ToolCategory</code>, and every
        tool-calling path states the same one. <code>AgentConfig.tool_contract</code> replaces the
        text, and an empty string states nothing.
      </p>
      <p>
        <strong>Whether a tool has to be called.</strong> A <code>ToolUsePolicy</code> of{' '}
        <code>REQUIRED</code>, <code>AUTO</code> or <code>SPARING</code> is set for every category
        and overridden with <code>AgentConfig.tool_use</code>; every shipped default matches what
        1.0.0 already did. <code>cite_sources</code> and <code>tool_choice</code> are now{' '}
        <code>run()</code> keywords, and <code>BaseModel.supports_forced_tool_call</code> reports
        whether an adapter can enforce a required call. An agent holding a code executor now runs
        the code rather than describing what it would print, and a declared{' '}
        <code>output_schema</code> is stated inside the loop on <code>stream()</code> as well as{' '}
        <code>run()</code>.
      </p>

      <h3>The loop and the ledger</h3>
      <p>
        A repeat of a call that already succeeded is answered from the run’s own record and the
        run keeps going, the drift thresholds are bounded by the run’s iteration budget, and a
        search that returns nothing is tried once more with a different query. Every generation
        parameter now reaches the provider. The budget check reads a total against an index rather
        than scanning the spend ledger, <code>SQLiteCostStore</code> gains{' '}
        <code>spend_since</code>, <code>spend_today</code>, <code>spend_week</code>,{' '}
        <code>spend_month</code>, <code>count</code>, <code>count_since</code> and{' '}
        <code>prune</code>, and <code>effgen cost prune</code> keeps the file small.{' '}
        <Link to="/cost">Cost</Link> has it.
      </p>

      <h3>Models and languages</h3>
      <p>
        Groq retired <code>llama-3.1-8b-instant</code> and <code>llama-3.3-70b-versatile</code>, so
        the Groq default, the bundled catalog, the CLI help, the error messages and every shipped
        example moved to <code>openai/gpt-oss-20b</code>. A provider-prefixed id now loads when you
        also pass the provider. The complexity analyzer, the decomposition engine, the sub-agent
        router and the prompt optimizer match English and Spanish keywords with accents folded on
        both sides, and a root agent’s <code>system_prompt</code> now reaches the sub-agents it
        spawns — both from @acdonaire.
      </p>

      <h2>1.0.0 — 14 August 2026</h2>
      <p>
        The first stable release. The theme running through it is
        control over where a model runs and visibility into what a run did. You can point effGen at
        any server speaking the OpenAI protocol, read back which tool calls a run made, wrap the
        agent loop in middleware, hand a single agent many conversations, choose how history is
        compacted, and resume a workflow that died half way through. A backend that never answered
        raises instead of returning something that reads like an answer.
      </p>
      <p>
        Around that sits a terminal coding agent, a branded command line that works on any
        terminal, a real-time dashboard, an in-browser playground, shareable HTML reports and run
        cards, a cross-provider model and pricing browser, a terminal mission-control view, a live
        model battle, and a browsable run and session history.
      </p>
      <p>
        Underneath both is the least visible and largest part of the release: a pass over
        everything that used to report the wrong thing confidently. A run that failed now says so.
        An unpriced model reports no cost instead of a made-up one. A turn whose every action
        failed is not a success. A tool call written in a shape effGen could not read no longer
        ends a turn with nothing.
      </p>

      <Callout type="warning" title="Three changes are breaking">
        <p>
          The Python floor is 3.11; <code>AgentConfig.raise_on_error</code> defaults to{' '}
          <code>True</code>; and a backend that was never reached raises{' '}
          <code>BackendUnreachableError</code> whatever that flag says.{' '}
          <Link to="/migration">The migration guide</Link> carries all three with the code each
          one asks you to change. The public surface grew from 204 names to 223 and nothing was
          removed or renamed.
        </p>
      </Callout>

      <h3>Connecting to models</h3>
      <p>
        <strong>Point effGen at any OpenAI-compatible server.</strong> <code>base_url</code> reaches{' '}
        <code>load_model()</code> and <code>AgentConfig</code>, so effGen can drive a model you
        already serve — vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, LiteLLM, a gateway or a
        corporate proxy — instead of loading a second copy of the weights inside the agent process.
      </p>

      <CodeBlock
        code={`from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)`}
        caption={
          <>
            The endpoint also comes from <code>EFFGEN_BASE_URL</code>,{' '}
            <code>OPENAI_BASE_URL</code> or <code>OPENAI_API_BASE</code>, in that order. Calls
            report no price rather than a fabricated $0, and{' '}
            <code>list_served_models()</code> asks the endpoint what it has. See{' '}
            <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
          </>
        }
      />

      <p>
        <strong>A multi-turn tool loop you can write by hand, on any provider.</strong>{' '}
        <code>build_assistant_message()</code> and <code>build_tool_result_message()</code> are on{' '}
        every adapter and build each provider's own message shape, so one loop runs against all of
        them rather than only the first. <Link to="/tool-calling">Tool calling</Link> has it.
      </p>

      <p>
        <strong>Python 3.14 is supported</strong> — installed and run, not just resolved. The
        supported set is {pythonVersions.join(', ')}.
      </p>

      <h3>The agent surface</h3>

      <p>
        <strong>Middleware around the agent loop.</strong> Hooks at three points — the run, each
        model call, each tool call — each with a <em>before</em> and an <em>after</em>. A{' '}
        <em>before</em> hook can rewrite the request or short-circuit it entirely; an{' '}
        <em>after</em> hook can transform the result. <em>Before</em> hooks run in order and{' '}
        <em>after</em> hooks in reverse, so middleware nest.
      </p>

      <CodeBlock
        filename="budget.py"
        code={`from effgen import Agent, AgentConfig
from effgen.core.middleware import AgentMiddleware
from effgen.tools.builtin import Calculator

class ToolBudget(AgentMiddleware):
    def __init__(self, limit=1):
        self.limit, self.used = limit, 0

    def before_tool_call(self, ctx):
        if self.used >= self.limit:
            return "Skipped: this run has spent its tool budget."
        self.used += 1
        return None

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[Calculator()],
    tool_calling_mode="react",
    temperature=0.0,
    middleware=[ToolBudget(limit=1)],
))
r = agent.run("What is 4817 * 236, and then what is that plus 1000?")
print(r.tool_calls.names)
print(r.output.strip()[:90])`}
      />

      <Terminal
        command="python budget.py"
        output={`['calculator']
1136812`}
        caption={
          <>
            One call was allowed and one was made; the second was refused by the hook.{' '}
            <code>LoggingMiddleware</code> and <code>ToolApprovalMiddleware</code> ship, and{' '}
            <code>run(..., middleware=[...])</code> adds hooks for one call. See{' '}
            <Link to="/middleware">Middleware</Link>.
          </>
        }
      />

      <p>
        <strong>One agent, many conversations.</strong> <code>run(..., session=...)</code> builds
        the prompt from that conversation's history and appends the turn to it, restoring the
        agent's own session and memory afterwards — including when the run fails. A server handling
        many users no longer needs an agent object per user.
      </p>

      <CodeBlock
        filename="sessions.py"
        code={`from effgen import create_agent

agent = create_agent("minimal", "openai:gpt-5-nano")

agent.run("My dog is named Pixel.", session="user-123")
agent.run("My cat is named Mote.", session="user-456")

print(agent.run("What is my dog called?", session="user-123").text.strip())
print(agent.run("What is my cat called?", session="user-456").text.strip())`}
      />

      <Terminal command="python sessions.py" output={`Pixel
Mote.`} />

      <p>
        <strong>Pluggable context compaction.</strong> What gets dropped when a conversation
        outgrows the window is now a strategy: <code>SummarizeOldest</code> (the default, with
        behaviour unchanged), <code>DropOldest</code> (no model call, nothing invented),{' '}
        <code>KeepFirstAndLast</code> (the turns carrying the task survive verbatim) and{' '}
        <code>KeepToolResults</code> (the evidence stays, the reasoning is compacted). Choose one
        with <code>AgentConfig(compaction_strategy=DropOldest())</code>, or subclass{' '}
        <code>CompactionStrategy</code>. <code>AgentConfig(tokenizer=...)</code> measures the
        history in the units the window is measured in rather than characters divided by four. See{' '}
        <Link to="/compaction">Context compaction</Link>.
      </p>

      <p>
        <strong>A workflow that died part way through can be resumed.</strong>{' '}
        <code>WorkflowDAG.run()</code> takes a <code>checkpoint=</code> store and a{' '}
        <code>run_id=</code>; running the same line again after a crash continues where it stopped.
        Completed nodes are not re-run and their outputs flow downstream, failed nodes are retried,
        and a finished run replays its stored outputs without calling a model — so a retrying job
        runner cannot double-bill you. There is no separate resume call: an unknown run id starts
        from the beginning and a known one continues. See{' '}
        <Link to="/checkpointing">Checkpointing and resumable runs</Link>.
      </p>

      <p>
        <strong>
          <code>AgentResponse.tool_calls</code> reports the calls, not just how many.
        </strong>{' '}
        Each entry carries <code>name</code>, <code>arguments</code>, <code>result</code>,{' '}
        <code>duration</code>, <code>error</code> and the <code>iteration</code> it was made on,
        with <code>failed</code> and <code>by_name()</code> to narrow them. Iterating the field used
        to raise <code>TypeError: 'int' object is not iterable</code>. It still compares and casts
        as the count.
      </p>

      <h3>The coding agent</h3>
      <p>
        <strong>
          <code>effgen code</code> is a coding agent in the terminal.
        </strong>{' '}
        It reads your workspace, proposes edits as unified diffs, and writes nothing until you say
        so. <code>--undo</code> rolls the last change back from a journal bounded to{' '}
        {siteData.code.undo_journal_entries} entries.
      </p>

      <CodeBlock
        language="bash"
        code={`effgen code "add a --dry-run flag to the importer"
effgen code --review                      # one read-only pass
effgen code --session-id my-refactor      # continue where you left off`}
      />

      <p>
        It runs in one of four permission modes that gate every write, every shell command and every
        commit. Writes are confined to the workspace, and a hunk that no longer applies is reported
        rather than clobbering the file. An interactive session keeps one run record across turns
        and carries {siteData.code.slash_command_count} slash commands. It is repository-aware —
        branch, status and a layout inventory that honours <code>.gitignore</code> go into the
        prompt, and an <code>AGENTS.md</code> brief is read when present. Git actions run through an
        allow-list, so push, reset, checkout, clean, rebase and force are refused before a
        subprocess starts, including when the model tries to reach them through the shell. See{' '}
        <Link to="/cli/code">effgen code</Link>.
      </p>

      <h3>Surfaces you can show someone</h3>

      <ApiTable
        headers={['Surface', 'What it is']}
        rows={[
          [
            <Link to="/dashboard">A real-time dashboard</Link>,
            'Per-model and per-provider cost, latency percentiles that are real percentiles, an error breakdown, a run waterfall, a model catalog panel and a history panel. Every chart is drawn locally.',
          ],
          [
            <Link to="/playground">An in-browser playground</Link>,
            'Model and preset pickers, tool toggles, the run’s tool trace, and copy-as-curl, copy-as-CLI and copy-as-Python for the form you filled in.',
          ],
          [
            <Link to="/catalog">A model and pricing browser</Link>,
            'In the terminal and in the dashboard, with search, provider, capability, context and price filters, sorting and paging.',
          ],
          [
            <Link to="/cli/reports">HTML reports and run cards</Link>,
            <>
              <code>--report out.html</code> for compare, eval, cost and loadtest;{' '}
              <code>run --card</code> for a single run; and <code>effgen report</code> to render a
              saved result after the fact.
            </>,
          ],
          [
            <Link to="/cli/top">effgen top</Link>,
            'A terminal mission-control view over the telemetry you already collect — activity, traffic, per-model, spend and GPU panels, each stating the window and process it describes.',
          ],
          [
            <Link to="/compare">effgen battle</Link>,
            'Races several models on one prompt and reports the tally, the cost and an optional judge’s verdict separately from the measurements.',
          ],
          [
            'A live topology graph',
            'Multi-agent topology, terminal trace timelines, a workflow DAG diagram and a run waterfall.',
          ],
          [
            <Link to="/cli/appearance">Named terminal themes</Link>,
            <>
              {siteData.cli.themes.map((theme, i) => (
                <span key={theme}>
                  {i > 0 ? ', ' : ''}
                  <code>{theme}</code>
                </span>
              ))}
              , drawn from one shared palette the dashboard reads too.
            </>,
          ],
        ]}
        caption={`Every web surface is self-contained: no CDN, no external font and nothing fetched at view time — enforced by a test that inspects what a browser would fetch rather than by searching for a substring. Across the shipped files it finds ${siteData.web.external_references} external references.`}
      />

      <h3>History, projects and the command line</h3>
      <ul>
        <li>
          <strong>Durable run and session history.</strong> Every run is recorded with its model,
          provider, tokens, cost, status and task, keyed by the same run id its trace spans carry.
          Runs from the command line, a script and the server share one history and survive a
          restart — <Link to="/cli/history">Runs and sessions history</Link>.
        </li>
        <li>
          <strong>Project scaffolding.</strong>{' '}
          <code>effgen quickstart --init</code> writes a configuration, an{' '}
          <code>.env</code> template, a runnable example and a <code>.gitignore</code>, and puts a
          daily spend cap in force when none is configured —{' '}
          <Link to="/first-project">Your first project</Link>.
        </li>
        <li>
          <strong>Flags and output that behave the same everywhere.</strong>{' '}
          <code>--json</code> on every command that had no machine output, and{' '}
          <code>--json</code> stdout is now a single valid document on a pipe and on a terminal,
          with no spinner, table or warning mixed into it. <code>-o</code> picks its format from the
          extension. Thirteen short flags now mean the same thing across commands, and a bare group
          command prints its own help and exits 0.
        </li>
        <li>
          <strong>Your own prompt templates load beside the shipped ones.</strong>{' '}
          <code>EFFGEN_PROMPTS_DIR</code> names one or more directories, so a team's library sits
          next to the built-in one without a fork.
        </li>
      </ul>

      <h3>A documentation site</h3>
      <p>
        effGen has a project site and a documentation site, both static and both published from the
        framework repository. Every public definition documents its arguments and its result: the
        package was walked module by module, and a gate fails when a public definition is added
        without that.
      </p>

      <h3>What was fixed</h3>
      <p>
        The largest part of the release is a pass over reporting. In summary:
      </p>
      <ul>
        <li>
          <strong>Results that report what actually happened.</strong> A turn whose every action
          failed is not a success; an unpriced model reports no cost rather than <code>$0</code>;
          and a failed run carries the reason it stopped.
        </li>
        <li>
          <strong>Tool calling across providers.</strong> Call shapes effGen could not read no
          longer end a turn with nothing, and the reported shape is the same on every adapter.
        </li>
        <li>
          <strong>Errors that name the fix.</strong> A wrong model id suggests near matches, a
          missing key names the variable, and a CUDA mismatch names the torch build against the
          driver.
        </li>
        <li>
          <strong>A rate limit no longer multiplies.</strong> Three layers each retried a throttled
          call and multiplied rather than shared a budget.
        </li>
        <li>
          The server and the API, security, guardrails and sandboxing, local models and GPUs,
          documents, RAG and batch input, the built-in tools, the terminal and web surfaces, and
          installation and packaging each have their own section in the changelog.
        </li>
      </ul>

      <h2>Earlier releases</h2>

      <ApiTable
        headers={['Version', 'Date', 'What it was']}
        rows={HISTORY.map(([release, date, summary]) => [<code>{release}</code>, date, summary])}
        caption={
          <>
            Summarised from the framework's <code>CHANGELOG.md</code>, which carries every entry in
            full. Code samples on this page are from the release they describe; earlier releases' examples are in
            the changelog, and some of them describe APIs that have since gained better ones.
          </>
        }
      />

      <Callout type="note" title="Where the record lives">
        <p>
          <code>CHANGELOG.md</code> in the framework repository is the full record, and{' '}
          <code>NEWS.md</code> is the readable version of the current release. This page summarises
          both; where they disagree with anything here, they are right.
        </p>
      </Callout>

      <SeeAlso paths={['/migration', '/introduction', '/api-reference']} />
    </DocPage>
  );
}
