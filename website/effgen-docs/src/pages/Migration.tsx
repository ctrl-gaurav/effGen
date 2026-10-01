import { ArrowUpCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Steps,
  Step,
  Terminal,
} from '../components/docs';
import { pythonVersions, publicNameCount, version } from '../siteData';

export default function Migration() {
  return (
    <DocPage
      subtitle={`The ten changes in ${version} that existing code can see, the fourteen in 1.2.0, the eleven in 1.1.0, the four in 1.0.1, the three breaking changes in 1.0.0, and what each one asks you to change.`}
      icon={<ArrowUpCircle size={48} />}
    >
      <p>
        The public surface is {publicNameCount} names and nothing has been removed or renamed since
        0.3.x, so most code upgrades by changing the version. Coming from 1.2.0, ten changes in{' '}
        {version} are visible to existing code; most change how a run ends or how a tool call is
        read, and each that changes behaviour has a setting that restores 1.2.0’s. Coming from
        1.1.0, 1.2.0 added fourteen more, most in what a run reports, what it costs or what a
        request carries, and two ask for a code change where you catch errors or add up cost.
        Coming from 1.0.1, 1.1.0 added eleven more, most of them in what a run reports.
        Coming from 1.0.0, 1.0.1 added four more, one of
        which changes what <code>success</code> means for a run that stopped part way. Coming from
        0.3.x, 1.0.0 also carried three breaking changes, each with a one-line migration.
      </p>

      <h2>The upgrade</h2>

      <CodeBlock
        language="bash"
        code={`pip install --upgrade effgen
python -c "import effgen; print(effgen.__version__)"`}
      />

      <Terminal
        command={'python -c "import effgen; print(effgen.__version__)"'}
        output={version}
      />

      <h2>From 1.2.0 to {version}</h2>
      <p>
        A run that stops making progress is asked for its answer, and every run says how it ended.
        Nothing was removed or renamed. <Link to="/releases">Releases</Link> has what was added;
        these are the changes existing code can see, in the order the changelog lists them. Code
        that reads <code>success</code>, <code>output</code> and <code>stop_reason</code> keeps
        working.
      </p>

      <h3>1. A run that stops making progress is asked for its answer</h3>
      <p>
        <code>AgentConfig.max_turns_without_progress</code> defaults to <code>2</code>; it was{' '}
        <code>None</code>. After two turns in a row that bring no new tool result, the next turn
        offers no tools and asks for the answer. A turn that declares no action after a result —{' '}
        <code>Action: None</code>, <code>Action: (continue reasoning)</code> or another bracketed
        placeholder that names no held tool — is asked at once. A run that would have stopped on{' '}
        <code>loop_detected</code>, <code>repeated_tool_result</code>,{' '}
        <code>max_iterations_*</code> or <code>null_final_from_model</code> while holding tool
        results gets one closing request: its own calls and results, with no tools. When the reply
        is an answer the run succeeds (<code>metadata["answer_source"] == "closing_request"</code>),
        so a run that raised <code>RunStoppedError</code> in 1.2.0 can now return an answer. A
        closing reply that is a program for a code tool gives the turn back, so such a run can send
        one model call more than <code>max_iterations</code>.
      </p>
      <p>
        <strong>Migration:</strong> <code>max_turns_without_progress=None</code>, in the config or
        per <code>run()</code>, restores 1.2.0’s loop.
      </p>

      <h3>2. A run whose tool keeps failing ends tool_failed</h3>
      <p>
        A tool that fails on its own side — a connection error, a timeout, an HTTP 5xx or 429,
        missing credentials — three times in a row, or whose circuit breaker is open, is unavailable
        for the rest of the run. A run left with no usable tool and no result stops with the new
        stop reason <code>tool_failed</code>, which is in <code>STOPPED_REASONS</code> and raises{' '}
        <code>RunStoppedError</code> under the default <code>raise_on_error=True</code>. Such a run
        gets no closing request; <code>metadata["error"]["kind"] == "tool"</code> and{' '}
        <code>metadata["unavailable_tools"]</code> names the tools. The per-tool circuit breaker now
        counts only those failures, so a tool given bad input is no longer refused to later calls
        and runs.
      </p>
      <p>
        <strong>Migration:</strong> a caller that read the model’s text from such a run catches{' '}
        <code>RunStoppedError</code>, or passes <code>raise_on_error=False</code> and branches on the
        termination.
      </p>
      <CodeBlock
        code={`response = agent.run(task, raise_on_error=False)
if response.termination == "tool_failed":
    print("unavailable:", response.metadata["unavailable_tools"])`}
      />

      <h3>3. response.termination says how a run ended</h3>
      <p>
        <code>"done"</code> (the model answered), <code>"not_possible"</code> (it answered, but every
        call was declined, failed on the tool’s side or returned nothing), <code>"stuck"</code> (the
        run kept proposing work that brought nothing new and wrote no answer),{' '}
        <code>"tool_failed"</code> and <code>"error"</code> (the run could not be carried out).{' '}
        <code>to_dict()</code> carries it, and <code>metadata["tool_results"]</code> counts a run’s
        attempted, usable and input-rejected calls. A stopped run’s <code>partial</code> never
        carries a tool’s error message. <strong>Migration:</strong> none.
      </p>

      <h3>4. A tool call is read before it is reported</h3>
      <p>
        <code>AgentConfig.recover_lost_tool_calls</code> defaults to <code>True</code>; it was{' '}
        <code>False</code>. A call written as a Python literal, with raw line breaks or unescaped
        double quotes inside its JSON, with its last string closed one bracket early, or as a whole
        object followed by text inside its tag, runs instead of ending the run with{' '}
        <code>written_tool_call</code>; a call nothing can read is sent back once. Some runs make
        more tool calls: the calls the model meant to make now run. Under either setting, arguments
        that arrive as a string are decoded, a call missing a required argument is asked for again
        rather than dispatched, and positional values are named in the tool’s parameter order.
      </p>
      <p>
        <strong>Migration:</strong> <code>recover_lost_tool_calls=False</code>, in the config or per
        run, restores 1.2.0’s reader; delegated sub-agents inherit it.
      </p>

      <h3>5. A failed tool call is retried, bounded, and never an answer</h3>
      <p>
        Two calls whose different inputs a tool rejects in the same words are no longer a repeated
        result. Four failures in a row on a tool’s input withdraw it for the run; a run left with no
        tool is asked for its answer and ends <code>tool_failed</code> with{' '}
        <code>metadata["error"]["kind"] == "input"</code> when it writes none. An answer that is a
        tool’s error message is sent back once, and given again the run stops with{' '}
        <code>null_final_from_model</code>. <strong>Migration:</strong> none.
      </p>

      <h3>6. Served and local models are measured once for how they use a tool</h3>
      <p>
        At <code>tool_calling_mode="auto"</code>, the first agent with tools built for a model behind
        a <code>base_url</code>, or on a local engine, runs a short probe and stores the result in{' '}
        <code>~/.effgen/capabilities.json</code>. A model that answers from memory while holding a
        search tool has its information-retrieval tools made must-call; a model whose native calls
        the server does not carry back runs in the ReAct text frame. A probe is bounded at 48
        requests and 120 seconds, and one that cannot run stores nothing. The same store remembers a
        provider that rejects stop sequences beside tools — 1.2.0 reported its HTTP 400 — and a
        server that rejects <code>reasoning_effort</code>.
      </p>
      <p>
        <strong>Migration:</strong> <code>AgentConfig(capability_probe=False)</code> or{' '}
        <code>EFFGEN_CAPABILITY_PROBE=0</code> resolves <code>auto</code> from the model’s declaration
        alone; so does an explicit <code>tool_calling_mode</code> or <code>tool_use</code>. A test
        suite that scripts a served endpoint sees the probe’s requests first unless it sets one of
        these. First-party cloud adapters are never probed.
      </p>

      <h3>7. reasoning_effort reaches a model behind base_url, and Groq</h3>
      <p>
        The OpenAI-compatible adapter now sends a <code>reasoning_effort</code> you set whatever the
        model is called, and the Groq adapter sends it for a model its catalog marks as reasoning;
        in 1.2.0 both dropped it. Every other adapter that drops it says so once per model at
        WARNING level. <strong>Migration:</strong> none.
      </p>

      <h3>8. One agent serves overlapping conversations without mixing them</h3>
      <p>
        <code>run(session=...)</code>, <code>run_async(session=...)</code> and{' '}
        <code>stream(session=...)</code> hold their conversation on the call, so overlapping calls on
        one agent each read and record only their own session; in 1.2.0 they could read and record
        each other’s turns. <code>stream()</code> now honours <code>session=</code>, and a run whose
        input a guardrail blocks leaves the agent’s session as it was.{' '}
        <strong>Migration:</strong> none; calls without <code>session=</code> still share the agent’s
        own memory.
      </p>

      <h3>9. A streamed turn is saved to the agent’s bound session</h3>
      <p>
        On an agent created with <code>session_id=</code>, or given <code>agent.session</code>,{' '}
        <code>stream()</code> appends each answered turn to that session and saves it, as{' '}
        <code>run()</code> always did. <strong>Migration:</strong> none.
      </p>

      <h3>10. Smaller changes</h3>
      <p>
        <code>effgen code --json</code> reports a <code>tool_failed</code> run as stopped. The first
        concurrent streams in a fresh process no longer fail on the OpenAI SDK’s response types, and
        the agent’s circuit breaker is safe under concurrent writers.
      </p>

      <h3>All of 1.2.0’s loop, in one place</h3>
      <CodeBlock
        code={`from effgen import AgentConfig

as_in_1_2 = AgentConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    max_turns_without_progress=None,
    recover_lost_tool_calls=False,
    capability_probe=False,
)`}
      />

      <h2>From 1.1.0 to 1.2.0</h2>
      <p>
        Every run now keeps a ledger of what it spent and where its time went. Nothing was removed
        or renamed. <Link to="/releases">Releases</Link> has what was added; these are the changes
        existing code can see, in the order the changelog lists them.
      </p>

      <h3>1. A tool result the model writes itself is never taken as the answer</h3>
      <p>
        A turn that holds tools and whose reply may be read as text is sent one stop sequence,{' '}
        <code>"\nObservation:"</code>, where 1.1.0 sent four labels; a tool-free turn is sent none,
        and a caller’s own <code>stop_sequences</code> replace it. When a reply holds a written
        action anyway, the action runs and whatever the model wrote after it is discarded.{' '}
        <code>BaseModel.supports_stop_with_tools()</code> is new and answers <code>True</code>; an
        adapter for a provider that rejects <code>stop</code> beside <code>tools</code> answers{' '}
        <code>False</code>.
      </p>
      <p>
        <strong>Migration:</strong> none. A caller who relied on <code>"\nQuestion:"</code>,{' '}
        <code>"\nHuman:"</code> or <code>"\nUser:"</code> stopping a tool-holding turn passes them
        in <code>stop_sequences</code>.
      </p>

      <h3>2. reasoning_effort reaches run() and run_async()</h3>
      <p>
        In 1.1.0 it reached a streamed turn and not a blocking one. All three paths now send it when
        the caller passes it and the model declares that it reasons; on any other model the adapter
        drops it.
      </p>

      <h3>3. The output budget follows what the run declared</h3>
      <p>
        With no <code>max_tokens</code> pinned, a run that declares an <code>output_schema</code>{' '}
        asks for a budget sized from the schema, and a model that declares it reasons is never sent
        fewer than 4,096 tokens. An explicit <code>max_tokens</code> still wins on every call.
      </p>

      <h3>4. A request carries less of the framework’s own text</h3>
      <p>
        The tool contracts no longer ask for text the task did not ask for, and come before the
        caller’s task. Tool rules are stated once, and a repeated tool result is sent once and
        referred back to; the step itself keeps the whole result. How fully a tool is described
        comes from the adapter, through <code>BaseModel.prompt_detail()</code>. A run with no tools,
        or with <code>tool_contract=""</code>, is unaffected.
      </p>
      <p>
        <strong>Migration:</strong> none. <code>AgentConfig(answer_style=...)</code> asks for a
        shorter or a fuller answer.
      </p>

      <h3>5. On a provider with a prompt cache, a run keeps one request shape</h3>
      <p>
        A run with tools sends its conversation as messages from its first run when the provider’s
        adapter declares a prompt cache and takes messages, so a session’s next question extends
        the cached prefix. The answer turn keeps the same messages and tools with{' '}
        <code>tool_choice="none"</code> where the adapter can forbid a call.{' '}
        <code>cache_system_prompt</code> and <code>cache_tools</code> now work on Anthropic, and
        cached tokens are read on Groq, Together, Fireworks, Cerebras and Gemini.
      </p>
      <p>
        <strong>Migration:</strong> none. Cached tokens are part of <code>prompt_tokens</code>, not
        added to it. <code>AgentConfig(prompt_protocol="flat")</code> keeps a run’s own steps in
        one string.
      </p>

      <h3>6. The loop stops fewer runs that are still finding things</h3>
      <p>
        A tool is withdrawn after 12 calls in a row that brought nothing new (16 for a
        data-processing tool), not after 12 calls in all; a run whose calls keep finding things is
        bounded by <code>max_iterations</code>. <code>run(max_iterations=N)</code> moves the loop’s
        own thresholds too, and <code>Action: None</code> is read as no action.
      </p>

      <h3>7. Groq hands a tool call it could not parse back to the loop</h3>
      <p>
        Instead of failing the run, the adapter returns the call as the model wrote it; the loop runs
        it if it can read it and otherwise asks for it again.
      </p>

      <h3>8. A spent cap refuses only calls that cost money, with a typed error</h3>
      <p>
        A call to a local engine, a free tier or a <code>base_url=</code> server is no longer
        refused. A refused call raises <code>BudgetExceededError</code> from <code>run()</code>{' '}
        whatever <code>raise_on_error</code> says; <code>stream()</code> raises it,{' '}
        <code>run_batch()</code> raises and stops, and the server answers HTTP 429 with{' '}
        <code>budget_exceeded</code>.
      </p>
      <p>
        <strong>Migration:</strong> catch <code>BudgetExceededError</code> where you caught{' '}
        <code>RuntimeError</code>.
      </p>
      <CodeBlock
        code={`from effgen import BudgetExceededError

try:
    response = agent.run(task)
except BudgetExceededError as exc:   # not a RuntimeError
    print("spend cap reached:", exc)`}
      />

      <h3>9. A model with no published price reads as unknown, not as free</h3>
      <p>
        <code>CostTracker.total_cost()</code>, <code>CostEvent.cost_usd</code>,{' '}
        <code>RunLedger.cost_usd</code>, <code>total_cost_usd</code> in{' '}
        <code>effgen cost --json</code> and <code>cost_usd</code> in the run executions can be{' '}
        <code>None</code> when every call they cover was unpriced. A free model still reads{' '}
        <code>0.0</code>. A server reached with <code>base_url=</code> records as{' '}
        <code>openai_compatible</code> where 1.1.0 recorded <code>openai</code>.
      </p>
      <p>
        <strong>Migration:</strong> treat <code>None</code> as “not priced” wherever you add up
        cost.
      </p>
      <CodeBlock
        code={`costs = [r.ledger.cost_usd for r in responses]
priced = [c for c in costs if c is not None]
print(sum(priced), len(costs) - len(priced), "unpriced")`}
      />

      <h3>10. Every run keeps a ledger, and two totals changed with it</h3>
      <p>
        <code>response.ledger</code> and <code>response.metadata["ledger"]</code> carry the run’s{' '}
        <code>RunLedger</code>; the flat metadata keys are unchanged.{' '}
        <code>effgen_model_call_latency_seconds</code> observes each model call rather than each
        run, and a decomposed run’s <code>tokens_used</code> includes its sub-agents’ tokens while{' '}
        <code>effgen_tokens_used_total</code> counts only the run’s own calls. A synchronous{' '}
        <code>@tool</code> runs in the caller’s context, so an agent it starts is a child of the run.
      </p>

      <h3>11. A run’s reported time includes the work after the model’s last answer</h3>
      <p>
        <code>execution_time</code> now includes <code>after_run</code> middleware, the session save
        and the final checkpoint, and equals the ledger’s <code>wall_s</code>.
      </p>

      <h3>12. The spend ledger stops growing at 250,000 rows</h3>
      <p>
        The file folds its oldest rows into per-model totals, keeping every total exact.{' '}
        <code>EFFGEN_COST_MAX_ROWS</code> sets the ceiling, and <code>0</code> keeps every row.
        Opening an existing file adds the columns <code>calls</code> and{' '}
        <code>unpriced_calls</code>.
      </p>

      <h3>13. A server named with openai: gets the id without the prefix</h3>
      <p>
        <code>AgentConfig(model="openai:&lt;id&gt;", base_url=...)</code> now sends{' '}
        <code>&lt;id&gt;</code>. Concurrent streamed runs keep their own tool arguments, and
        in-process local engines serve concurrent agents.
      </p>

      <h3>14. An out-of-budget failure says what to do and is not retried</h3>
      <p>
        A reply cut off by its token budget, or one that spent the whole budget reasoning, carries
        its own guidance and is not marked retryable. It used to read “Unexpected provider error”
        and be retried.
      </p>

      <h2>From 1.0.1 to 1.1.0</h2>
      <p>
        A run now keeps its conversation as an <code>AgentThread</code> of typed steps instead of
        one growing string. Nothing was removed or renamed. <Link to="/releases">Releases</Link> has
        what was added; these are the changes existing code can see, in the order the changelog
        lists them.
      </p>

      <h3>1. A run carries its conversation, and response.metadata is no longer plain data</h3>
      <p>
        <code>AgentResponse.thread</code> is the run’s <code>AgentThread</code>, or{' '}
        <code>None</code> for a run that recorded none, and <code>response.metadata["thread"]</code>{' '}
        holds the same live object. The thread opens with the run’s frame — a{' '}
        <code>SystemStep</code> when tools are attached, then always a <code>TaskStep</code> — so{' '}
        <code>thread.steps[0]</code> is no longer the first thought.{' '}
        <code>AgentThread.to_text()</code> is unchanged.
      </p>
      <p>
        <strong>Migration:</strong> serialise through <code>response.to_dict()</code> or{' '}
        <code>response.thread.to_dict()</code>; <code>json.dumps(response.metadata)</code> raises on
        the thread object.
      </p>
      <CodeBlock
        code={`import json

json.dumps(response.to_dict())            # writes the thread through its own to_dict()
json.dumps(response.thread.to_dict())     # the steps alone`}
      />

      <h3>2. A session’s earlier turns render differently inside the prompt</h3>
      <p>
        The <code>=== Previous Conversation Context ===</code> block with <code>[Turn n]</code>{' '}
        markers is gone. A request that carries one string now reads{' '}
        <code>Earlier in this conversation:</code> followed by <code>User:</code> /{' '}
        <code>Assistant:</code> lines. On a model whose adapter takes a conversation, a run whose
        tools travel as a request parameter sends the earlier turns as their own{' '}
        <code>user</code> and <code>assistant</code> messages, and a caller’s{' '}
        <code>system_prompt</code> as the system message, on every request of the run.
      </p>
      <p>
        <strong>Migration:</strong> only code that matched on the old header text, or that reads
        the request a provider receives, is affected. <code>Session.last_thread()</code> reads the
        steps directly. <code>tool_calling_mode="react"</code>, which writes the tools into the
        prompt, keeps the one-string rendering.
      </p>

      <h3>3. AgentConfig(guardrails=...) accepts a plain list</h3>
      <p>
        A list of guardrails no longer has to be wrapped, and a non-guardrail in the list raises{' '}
        <code>TypeError</code> at construction instead of failing later inside a run.
      </p>

      <h3>4. A 1.0.x checkpoint still resumes</h3>
      <p>
        <code>Checkpoint</code> has a new <code>thread</code> field and still writes the flat
        transcript beside it, so a file 1.1.0 writes resumes on a build that only knows the
        transcript. A 1.0.x file has no steps, so <code>Checkpoint.to_thread()</code> rebuilds them
        from the transcript and logs <code>[compat] rebuilt a thread from a flat transcript</code>.
        The rebuild cannot recover the provider’s call id, a tool’s arguments as values, which lines
        were the framework’s own, or the run’s frame — the persona, the tool contract, earlier turns
        and the task. <Link to="/checkpointing">Checkpointing</Link> has the details.
      </p>

      <h3>5. agent.resume() continues an unfinished run</h3>
      <p>
        The final checkpoint stores the run’s steps where it used to store an empty transcript, so
        resuming picks up the conversation the run had instead of restarting the task, and does not
        redo the turns the checkpoint holds.
      </p>

      <h3>6. One agent loop, so stream() behaves like run()</h3>
      <p>
        If your code calls <code>run()</code>, nothing changed. A streamed run now sends the same
        prompt, the same tool definitions and the same sampling settings as <code>run()</code>, its
        loop guards fire as on a blocking run, and output guardrails are checked as{' '}
        <code>run()</code> checks them.
      </p>
      <p>
        <strong>Migration:</strong> a stream that relied on different sampling settings, a guard
        that never fired, or an output guardrail that was never checked now behaves as the blocking
        path always did — and an output guardrail that blocks raises out of the iterator.
      </p>

      <h3>7. A run is bounded by what it may send</h3>
      <p>
        <code>AgentConfig.context_budget</code> (<code>"auto"</code>, an integer, a float or{' '}
        <code>None</code>; default <code>"auto"</code>) and <code>AgentConfig.compaction</code> are
        new, and <code>max_context_length</code> — declared since 1.0 and read by nothing — is now
        the window override. The budget is unbounded when the model declares no window.{' '}
        <code>response.metadata["context_budget"]</code> is reported on every outcome.{' '}
        <code>Session.keep_thread_history</code> defaults to <code>False</code>, so an earlier
        turn’s stored steps are reduced to their shape. A prompt larger than the model’s window is
        sent once instead of retried.
      </p>
      <CodeBlock
        code={`from effgen import AgentConfig
from effgen.core.session import Session

AgentConfig(model="openai:gpt-5-nano", context_budget=None)   # unbounded, as in 1.0.x
AgentConfig(model="openai:gpt-5-nano", context_budget=8000)   # a budget you name
Session(keep_thread_history=True)                              # keep earlier turns' steps`}
      />
      <p>
        <strong>Migration:</strong> none required; <code>"auto"</code> reproduces 1.0.x behaviour
        on any conversation that already fitted. <Link to="/compaction">Compaction</Link> covers the
        policies.
      </p>

      <h3>8. Orchestration results carry threads</h3>
      <p>
        A run with no tools now reports <code>metadata["thread"]</code>. Workflow, team and
        sub-agent results carry their runs’ threads and serialise them into{' '}
        <code>to_dict()</code>. <code>WorkflowDAG</code>, <code>TeamConfig</code> and{' '}
        <code>SubAgentManager</code> take a <code>projection</code> that defaults to carrying
        nothing into a child run — which is what every pattern did before.
      </p>

      <h3>9. effgen run --json works on a run that used a tool</h3>
      <p>
        <code>to_dict()["execution_tree"]</code> used to carry <code>ToolCall</code> objects, so
        serialising any run that called a tool raised <code>TypeError</code>, taking{' '}
        <code>effgen run --json</code>, <code>-o</code> and <code>--card</code> with it. Those
        documents now serialise, and all three go through one scrubber. The terminal answer panel
        still prints the run’s own words unredacted.
      </p>

      <h3>10. The debug trace carries steps</h3>
      <p>
        <code>DebugIteration</code> gained <code>thread_snapshot</code>, and its{' '}
        <code>to_dict()</code> gained a <code>thread</code> key.
      </p>

      <h3>11. A run continuing a session sends its conversation as messages</h3>
      <p>
        <code>AgentConfig.prompt_protocol</code> is new: <code>"flat"</code>,{' '}
        <code>"messages"</code> or <code>"auto"</code>, default <code>"auto"</code>. At the
        default, a run with tools that continues a session sends the session’s earlier turns and its
        own steps as the messages they were, on a model that declares the message protocol and
        takes its tools as a request parameter. A run that continues nothing, and a run with no
        tools, sends the one string it always sent.
      </p>
      <p>
        <strong>Migration:</strong> none for a run without a session.{' '}
        <code>AgentConfig(prompt_protocol="flat")</code> keeps a session run’s own steps in one
        string, as 1.0.x did.
      </p>

      <h2>From 1.0.0 to 1.0.1</h2>

      <h3>A run that stops without an answer reports failure, and raises by default</h3>
      <p>
        In 1.0.0 three paths returned <code>success=True</code> with the loop’s internal state
        in <code>.output</code>: a computation tool that tripped the repeat guard, a computation tool
        that returned a result it had already returned, and a model that gave no final answer after
        its tools ran. Since 1.0.1 they return <code>success=False</code>,{' '}
        <code>outcome="stopped"</code> and a <code>stop_reason</code> naming the exit, and keep what
        the model reached in <code>.partial</code>. Under the default{' '}
        <code>raise_on_error=True</code> the run raises <code>RunStoppedError</code>, which
        subclasses <code>RuntimeError</code> — what the iteration cap has always raised — and
        carries <code>.response</code>, <code>.stop_reason</code> and <code>.partial</code>.
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

      <p>
        <strong>Migration:</strong> catch <code>RunStoppedError</code>, or pass{' '}
        <code>raise_on_error=False</code> and branch on <code>.outcome</code>, which is{' '}
        <code>"answered"</code>, <code>"stopped"</code> or <code>"failed"</code>. The text the model
        reached is in <code>.partial.text</code>, and still at{' '}
        <code>metadata["partial_output"]</code> byte for byte. The outcome also reaches the CLI,{' '}
        <code>effgen runs list --status stopped</code>, batch rows and their CSV,{' '}
        <code>effgen code</code>, <code>EvalResult.stop_reason</code> and the server’s{' '}
        <code>effgen</code> envelope. <Link to="/errors">Errors</Link> has the full vocabulary.
      </p>

      <h3>Citation markers are opt-in</h3>
      <p>
        1.0.0 told the model to cite each passage inline as <code>[1]</code>, <code>[2]</code>, ...
        on every turn that followed a retrieval or search tool, whether or not you asked. The
        markers pointed at nothing, and a question with a one-word answer came back with them
        attached. They are now added only when you ask; the <code>rag</code> preset asks already.
        When you do ask, <code>[n]</code> is <code>citations[n - 1]</code>.
      </p>

      <CodeBlock
        code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="openai:gpt-5-nano", cite_sources=True))

# or for a single call
agent.run("Summarise the retrieved notes.", cite_sources=True)`}
      />

      <h3>A streamed run sends the model’s working before its answer</h3>
      <p>
        Same task, same final answer, but the chunks start with the model’s reasoning, so code
        that joins every chunk shows the working first. Ask for events and join only the answer:
      </p>

      <CodeBlock
        code={`events = list(agent.stream(task, include_events=True))
answer = "".join(e.text for e in events if e.kind == "answer")`}
      />

      <h3>A small local model writes more and reaches for tools more often</h3>
      <p>
        Every tool-calling path now tells the model what its tools are for, picked from their
        declared <code>ToolCategory</code>. On a small model on the local Transformers engine that
        means longer completions and a tool call on questions it used to answer directly; the
        answers stay the same. <code>tool_use="sparing"</code> tells a run that already has the
        answer to give it, <code>tool_use="auto"</code> requires nothing whatever the tools declare,
        and <code>tool_contract=""</code> states nothing about the tools at all.
      </p>

      <h2>From 0.3.x to 1.0.0</h2>
      <p>
        1.0.0 was the first stable release, and three of its changes are breaking.
      </p>

      <h2>1. Python 3.10 is no longer supported</h2>
      <p>
        The floor is 3.11, and the supported set is {pythonVersions.join(', ')}.{' '}
        <code>tomllib</code>, <code>asyncio.timeout</code>, <code>datetime.UTC</code> and the{' '}
        <code>TimeoutError</code> unification are all standard library from 3.11, and effGen
        carried a hand-written fallback for each.
      </p>
      <p>
        <strong>Migration:</strong> upgrade the interpreter. Nothing in the API changed.
      </p>

      <Callout type="note" title="On 3.14, the [all] extra needs a lock file">
        <p>
          The base install and every named extra install normally on 3.14. Only{' '}
          <code>effgen[all]</code> needs{' '}
          <code>pip install -r requirements-all-py314-lock.txt</code> followed by{' '}
          <code>pip install --no-deps effgen</code> —{' '}
          <Link to="/installation">Installation</Link> explains why.
        </p>
      </Callout>

      <h2>2. raise_on_error now defaults to True</h2>
      <p>
        A failed run raises its typed error instead of returning an{' '}
        <code>AgentResponse</code> with <code>success=False</code> and a plausible-looking string
        in <code>.output</code> — which a caller reading <code>.output</code> without checking{' '}
        <code>.success</code> never noticed.
      </p>
      <p>
        <strong>Migration:</strong> pass <code>raise_on_error=False</code> to inspect the response
        yourself. The failure shape is unchanged.
      </p>

      <CodeBlock code={`from effgen import Agent, AgentConfig

Agent(AgentConfig(model="openai:gpt-5-nano", raise_on_error=False))`} />

      <Callout type="warning" title="With the flag off, read partial_output rather than output">
        <p>
          A failed run's <code>output</code> is effGen's report of what stopped it. The model's own
          text is in <code>metadata["partial_output"]</code>.
        </p>
      </Callout>

      <CodeBlock
        filename="inspect.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[Calculator()],
    tool_calling_mode="react",
    max_iterations=1,
    temperature=0.0,
    raise_on_error=False,          # the 1.0.0 default is True
))
r = agent.run("What is 4817 * 236? Use the calculator, then explain the result.")

print("success:", r.success)
print("reason:", r.metadata.get("reason"))
print("output:", r.output[:90])
print("the model's own text:", r.metadata.get("partial_output"))`}
      />

      <Terminal command="python inspect.py" output={`success: False
reason: max_iterations_partial
output: Stopped after 1 iteration without a final answer: 'gpt-5-nano' was still taking tool steps
the model's own text: 1136812`} />

      <Callout type="tip" title="Batch evaluation wants raise_on_error=False">
        <p>
          Scoring a run that hit the iteration cap as an error rather than as a wrong answer
          measures the reporting style instead of the model, and a small model hits that cap often.
          What makes turning the flag off safe is the change below: an ordinary failure comes back
          to be inspected, while a backend that was never reached still raises — so a broken
          endpoint cannot be silently scored as a wrong answer. The command line does exactly this
          at all fourteen of its construction sites.
        </p>
      </Callout>

      <h2>3. A backend that never answered raises</h2>
      <p>
        A refused connection, an unresolvable host or a missing route is classified{' '}
        <strong>unreachable</strong> — separately from a server that answered badly, which stays
        transient and is still retried — and raises <code>BackendUnreachableError</code>.
      </p>
      <p>
        <strong>Migration:</strong> there is no opt-out, by design. Catch the error where you want
        to handle it.
      </p>

      <CodeBlock
        filename="unreachable.py"
        code={`from effgen import Agent, AgentConfig
from effgen.models.errors import BackendUnreachableError

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:9/v1",     # nothing is listening here
))

try:
    agent.run("What is 6 times 7?")
except BackendUnreachableError as e:
    print(type(e).__name__)
    print(str(e)[:220])`}
      />

      <Terminal command="python unreachable.py" output={`BackendUnreachableError
openai did not answer (model='Qwen/Qwen2.5-7B-Instruct'): OpenAI generation failed [will_retry]: Connection error.. Nothing answered at that endpoint — check the server is running and the base_url, host and port are righ`} />

      <p>
        A task that ran and failed is a result you can inspect. A backend that was never reached is
        not, and returning one is how a whole batch completes against nothing and still looks
        healthy in the summary. Classification reads the exception chain, because provider SDKs
        shorten a refused port to "Connection error." and keep the real cause on{' '}
        <code>__cause__</code>.
      </p>

      <h2>One smaller change worth knowing</h2>
      <p>
        Four enums are now <code>enum.StrEnum</code>: <code>TaskStatus</code> (from{' '}
        <code>effgen.core.background</code>), <code>AlertSeverity</code>,{' '}
        <code>PermissionMode</code> and <code>LoadScenario</code>. Equality, membership and JSON
        serialization are unchanged; what changed is that <code>str()</code> now gives the value
        rather than <code>ClassName.MEMBER</code>.
      </p>

      <CodeBlock
        filename="strenum.py"
        code={`from effgen.core.background import TaskStatus

print(str(TaskStatus.RUNNING))            # "RUNNING" since 1.0.0
print(TaskStatus.RUNNING == "RUNNING", TaskStatus.RUNNING in ("RUNNING", "FAILED"))`}
      />

      <Terminal
        command="python strenum.py"
        output={`RUNNING
True True`}
        caption="Anything that formatted one of these into a log line or a filename gets a shorter string than it used to."
      />

      <Callout type="note" title="Two different TaskStatus classes exist">
        <p>
          The one exported from the top-level package is a plain <code>Enum</code> used by the
          multi-agent task graph. The <code>StrEnum</code> is{' '}
          <code>effgen.core.background.TaskStatus</code>, which is what background jobs report. The
          import path decides which you get.
        </p>
      </Callout>

      <h2>Upgrading from 0.3.x, in order</h2>

      <Steps>
        <Step title="Move to Python 3.11 or newer">
          <p>Nothing in the API changed with it, so do this first and separately.</p>
        </Step>
        <Step title="Find every place that reads .output without checking .success">
          <p>
            Those are the call sites the second change affects. Either let them raise — which is
            usually what you want — or pass <code>raise_on_error=False</code> and check{' '}
            <code>.success</code> explicitly.
          </p>
        </Step>
        <Step title="Decide where an unreachable backend should be handled">
          <p>
            A batch runner, a job queue and a server all want to handle it differently. Catching{' '}
            <code>BackendUnreachableError</code> at the boundary is usually right.
          </p>
        </Step>
        <Step title="Search for str() on the four enums">
          <p>Only if you format them into logs, filenames or a wire format.</p>
        </Step>
      </Steps>

      <h2>What did not change</h2>

      <ApiTable
        headers={['Concern', 'Status']}
        rows={[
          ['Public names', `Grown to ${publicNameCount}. Nothing removed, nothing renamed.`],
          [
            <code>Agent</code>,
            <>
              <code>AgentConfig</code>, <code>load_model</code> and every tool API work unchanged.
            </>,
          ],
          [
            <code>AgentResponse.tool_calls</code>,
            'Now the calls themselves rather than only a count — but it still compares and casts as the count, so tool_calls == 2 and tool_calls > 0 are unchanged, and to_dict() keeps the count under its original key.',
          ],
          ['The failure shape', 'Unchanged. It is when you get it that changed.'],
          ['Configuration files', 'Unchanged.'],
          ['The server API', 'Unchanged for existing endpoints.'],
        ]}
      />

      <h2>Coming from another framework</h2>
      <p>
        If you are not upgrading but arriving — from the OpenAI SDK, from LangChain, or from
        anything that speaks the OpenAI protocol — the route in is effGen's own server, which most
        client code reaches by changing only its <code>base_url</code>.{' '}
        <Link to="/openai-api">OpenAI-compatible API</Link> is the full endpoint, alias, streaming
        and error-status reference, and <Link to="/clients">Clients and SDKs</Link> covers the
        lighter native client. The one place the protocol differs is tools: effGen runs its own
        registered tools on the server and returns the final answer, rather than forwarding
        client-defined function tools back for the caller to run.
      </p>

      <SeeAlso paths={['/releases', '/errors', '/installation']} />
    </DocPage>
  );
}
