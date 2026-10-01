import { Bot } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  MermaidDiagram,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import type { Param } from '../components/docs';
import { siteData, version } from '../siteData';
import { siteHref } from '../siteLinks';

/**
 * What each `AgentConfig` field is for, in the words of the class's own
 * docstring.
 *
 * The name, the type and the default are **not** written here — they come from
 * `data/effgen.json`, which is generated from the installed dataclass. This map
 * supplies only the sentence, and `paramsFor` below reports a field that has
 * gained or lost a description rather than letting the table drift.
 */
const CONFIG_NOTES: Record<string, string> = {
  model: 'A loaded model instance, or an id string such as "openai:gpt-5-nano". The only field with no default.',
  name: 'Agent identifier. Defaults to the model id, or "agent" for a model instance.',
  tools: 'The tools this agent may call. Each one is a BaseTool instance.',
  system_prompt: 'System-level instructions, applied on every path including streaming and the native tool loop.',
  max_iterations: 'How many times the loop may go round on one task before it stops.',
  temperature: 'Generation temperature. A run() keyword of the same name overrides it for one call.',
  max_tokens: 'Output-token budget for every run. None lets the model pick a size-aware default.',
  top_p: 'Nucleus-sampling threshold.',
  top_k: 'Top-k sampling cutoff. Providers that do not support it ignore it.',
  seed: 'Sampling seed. With temperature=0 this reproduces a generation exactly on Gemini, Groq and the local engines; OpenAI documents its seed as best-effort rather than a guarantee.',
  presence_penalty: 'Penalises tokens already present anywhere in the text.',
  frequency_penalty: 'Penalises tokens in proportion to how often they already appeared — the anti-repetition knob for long text.',
  repetition_penalty: 'Multiplicative repeat penalty, used by the local and HuggingFace engines.',
  mode: 'Default execution mode for run(). SINGLE never decomposes on its own; AUTO lets the router decide per call.',
  enable_sub_agents: 'Whether the agent may spawn sub-agents for parts of a task.',
  enable_memory: 'Whether the memory subsystem is active.',
  enable_streaming: 'Whether tokens are streamed as they arrive.',
  max_context_length: 'The model’s context window, overriding what the model declares. Read since 1.1.0, where it bounds context_budget.',
  context_budget: 'How many prompt tokens a run may send: "auto" (default) derives it from the model’s window, an int names it, a float is a fraction of the window, and None leaves it unbounded. Unbounded when the model declares no window.',
  compaction: 'The CompactionPolicy that brings a run’s thread back under its budget. None uses ShortenOldestFirst, which makes no model call; SummarizeWithModel is opt-in.',
  router_config: 'Settings for the sub-agent router.',
  sub_agent_config: 'Settings for the sub-agent manager.',
  model_config: 'Engine options passed through when the model is loaded from an id.',
  require_model: 'Whether a string model must load at construction. True means a typo or a missing key fails immediately instead of at the first run.',
  provider: 'Explicit provider for a bare model id — the same choice the "provider:model" prefix makes.',
  base_url: 'Endpoint for a server speaking the OpenAI protocol. Giving one loads the model through that server instead of in this process.',
  api_key: 'Credential for that endpoint. A local server that checks nothing needs none.',
  middleware: 'Hooks around the run, each model call and each tool call.',
  compaction_strategy: 'How the conversation is shortened as it approaches the window. Accepts a strategy, a class, or a name.',
  tokenizer: 'Anything with count_tokens(text) or encode(text), used to measure history in the units the window is measured in.',
  raise_on_error: 'Whether a run that produced no answer raises. True since 1.0.0: a failed run raises its typed error, and since 1.0.1 a stopped run raises RunStoppedError.',
  cite_sources: 'Ask for inline [1], [2] citation markers when answering from retrieved passages. Off by default; run(cite_sources=...) overrides it for one call, and the rag preset turns it on.',
  tool_contract: 'What the model is told about the attached tools. None picks the text from the tools’ declared categories, a string is stated verbatim instead, and "" states nothing.',
  tool_use: 'Whether a run holding these tools has to call one: "required", "auto" or "sparing", or a ToolUsePolicy. None reads it from the tools’ declared categories.',
  system_prompt_template: 'A template for the assembled system prompt, replacing the built-in one.',
  verbose_tools: 'Whether tool descriptions are sent in full. None follows the model.',
  fallback_chain: 'A mapping from tool name to the tools tried when it fails.',
  enable_fallback: 'Whether those tool fallback chains apply.',
  max_sub_agent_depth: 'How deep sub-agents may nest.',
  tool_calling_mode: '"auto", "native", "react" or "hybrid" — how tools are offered to the model.',
  output_format: 'Default output format for every run: "json", "yaml", "csv" or None.',
  output_schema: 'Default JSON Schema every run must produce.',
  guardrails: 'A GuardrailChain, a plain list of guardrails, or the name of a guardrail preset. Anything in a list that is not a guardrail raises TypeError at construction.',
  memory_config: 'Memory settings — token and message caps, the long-term backend, and whether old context is summarised.',
  models: 'Additional models this agent may fall back to or route between.',
  speculative_execution: 'Run on two models and take the first that succeeds.',
  approval_callback: 'Called before a tool runs, to approve or refuse it.',
  approval_mode: '"never", "always", "first_time" or "dangerous_only".',
  approval_timeout: 'Seconds to wait for approval. 0 waits forever.',
  clarification_callback: 'Called when the agent needs the user to choose between options.',
  input_callback: 'Called when the agent needs a line of input from the user.',
  stable_system_prompt: 'Keep the system prompt at a fixed position so a provider can cache the prefix.',
  cache_system_prompt: "Mark the system prompt for Anthropic prompt caching. Since 1.2.0 the last completed step carries a breakpoint too.",
  cache_tools: 'Mark the tool definitions for Anthropic prompt caching.',
  answer_style: 'One line about the form of the answer, stated last: "brief", "full" or your own sentence. None (the default) states nothing. Also a run(), stream() and run_async() keyword. New in 1.2.0.',
  max_turns_without_progress: 'After this many turns in a row that brought no new result, the next turn offers no tools and asks for the answer; a turn that declares no action after a result is asked at once, and a run about to stop stuck gets one closing request with its calls and results and no tools. 2 by default since 1.3.0 (None in 1.2.0); None turns all of it off and restores the earlier loop.',
  recover_lost_tool_calls: 'Read a tool call written in a broken shape — a Python literal, raw line breaks or unescaped quotes inside its JSON, its last string closed early, or text after the object in its tag — and ask once more for a call that could not be read at all. On by default since 1.3.0 (off in 1.2.0); False restores the strict reader. Delegated sub-agents inherit it.',
  capability_probe: 'Whether tool_calling_mode="auto" may measure, once per model, what a model behind a base_url or on a local engine does when handed a tool, and resolve against it. The result is stored in ~/.effgen/capabilities.json and shown by effgen doctor. False, or EFFGEN_CAPABILITY_PROBE=0, resolves auto from the declaration alone; cloud adapters are never probed. New in 1.3.0.',
  prompt_protocol: 'How a run’s conversation reaches the model: "flat" (one string), "messages" (the turns it was) or "auto" (default) — a run continuing a session sends messages on a model that takes them, and a run continuing nothing sends the flat string.',
};

const RESPONSE_NOTES: Record<string, string> = {
  output: 'The answer. `text` and `content` are read-only aliases, and str(response) is the same string.',
  success: 'Whether the run produced an answer. A stopped or failed run is returned with False only when raise_on_error is off.',
  stop_reason: 'The exit the run took, on every response: "final_answer" for an answer, a stopped reason such as "max_iterations_partial", "loop_detected" or (since 1.3.0) "tool_failed", or a failure such as "generation_failed". .outcome reads it as "answered", "stopped" or "failed", and .termination (since 1.3.0) as "done", "not_possible", "stuck", "tool_failed" or "error".',
  partial: 'What a stopped run had reached — its tool observations, last thought and a one-line text — as a PartialResult. None otherwise.',
  mode: 'The mode the run actually used.',
  iterations: 'How many times the loop went round.',
  tool_calls: 'The calls the run made. Iterable, and still compares and casts as the count.',
  tokens_used: 'Total tokens across every model call in the run, including its sub-agents’ calls since 1.2.0.',
  execution_time: 'Wall-clock seconds for the whole run, including after_run middleware, the session save and the final checkpoint. Equals response.ledger.wall_s.',
  execution_trace: 'One entry per step, for reconstructing what happened.',
  execution_tree: 'The same steps as a tree, when sub-agents were involved.',
  routing_decision: 'Which model was chosen and why, when routing was in play.',
  metadata: 'Cost, tokens, latency, partial_output, input_redaction, the run’s thread and context_budget, its ledger (also response.ledger, a RunLedger, since 1.2.0), since 1.3.0 tool_results, unavailable_tools, tool_calling and answer_source, and anything a subsystem attached. The thread is a live object, so serialise through to_dict() rather than json.dumps(metadata).',
  citations: 'Citations built from what the run retrieved, never scraped from the prose.',
  sources: 'The deduplicated source URLs behind those citations.',
  task: 'The task the run was given.',
  model: 'The model id that answered.',
  provider: 'The provider that served it.',
  started_at: 'When the run started, as an ISO-8601 string.',
};

/**
 * Join the generated field list to the sentences above.
 *
 * A field the framework no longer has cannot appear, because the list is the
 * framework's. A field that has appeared and has no sentence yet is marked
 * rather than silently described as nothing.
 */
function paramsFor(
  fields: typeof siteData.api.agent_config,
  notes: Record<string, string>,
): Param[] {
  return fields.map((field) => ({
    name: field.name,
    type: field.type,
    default: field.required ? undefined : (field.default ?? '(empty)'),
    required: field.required,
    description: notes[field.name] ?? 'Not described on this page yet.',
  }));
}

const LOOP = `flowchart TD
    Start["agent.run(task)"] --> Build["Build the prompt:<br/>system prompt + history + task"]
    Build --> Call["Call the model"]
    Call --> Decide{"Answer or<br/>tool call?"}
    Decide -->|"answer"| Done["AgentResponse"]
    Decide -->|"tool call"| Run["Run the tool"]
    Run --> Obs["Append the ToolResult"]
    Obs --> Cap{"iterations<br/>&lt; max_iterations?"}
    Cap -->|"yes"| Call
    Cap -->|"no"| Stop["Stop and report why"]
`;

export default function Agents() {
  return (
    <DocPage
      subtitle="The Agent class, the config it takes and the response it returns."
      icon={<Bot size={48} />}
    >
      <p>
        An agent is a model, a set of tools and a loop between them.{' '}
        <code>Agent</code> holds the loop, <code>AgentConfig</code> holds every setting it obeys,
        and <code>run()</code> returns an <code>AgentResponse</code> carrying the answer and
        everything that is true about how it was reached.
      </p>

      <p className="doc-crosslink">
        This page is the reference: every field, every return value, every failure. For what an
        agent is for and one worked run end to end, see <a href={siteHref('/agents')}>the agents
        page</a> on the main site.
      </p>

      <h2>The shortest agent</h2>

      <CodeBlock
        filename="agent.py"
        code={`from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator

model = load_model("openai:gpt-5-nano")

agent = Agent(AgentConfig(
    model=model,
    tools=[Calculator()],
    system_prompt="You are a careful arithmetic assistant.",
    temperature=0.0,
    max_iterations=5,
))

response = agent.run("What is (17 * 23) + 12?")
print(response.output)`}
      />

      <Terminal
        command="python agent.py"
        output={`403

Explanation:
- 17 × 23 = 391
- 391 + 12 = 403`}
        caption={`Run against effGen 1.0.0.`}
      />

      <p>
        <code>model</code> is the only field with no default. Everything else has one, so a
        one-field config is legal: <code>Agent(AgentConfig(model="openai:gpt-5-nano"))</code> is a
        working agent with no tools. For a configuration someone has already worked out, start
        from a <Link to="/presets">preset</Link> instead.
      </p>

      <h2>The loop</h2>

      <MermaidDiagram
        chart={LOOP}
        title="What run() does"
        description="run() builds a prompt from the system prompt, the history and the task, and calls the model. If the model answers, the run returns an AgentResponse. If it asks for a tool, the tool runs, its result is appended, and the loop calls the model again until max_iterations is reached, at which point the run stops and reports why."
      />

      <p>
        How tools are offered to the model — as native function definitions, as text the model
        writes back in, or both — is <code>tool_calling_mode</code>, covered on{' '}
        <Link to="/tool-calling">Tool calling</Link>. Nothing else about the loop changes with it.
      </p>

      <h3>How a run ends</h3>
      <p>
        Since 1.3.0 every response says how its run ended in <code>response.termination</code>,
        derived from <code>success</code>, <code>stop_reason</code> and{' '}
        <code>metadata["tool_results"]</code>, so it never disagrees with them.
      </p>
      <ApiTable
        headers={['termination', 'What it means', 'success']}
        rows={[
          [<code>"done"</code>, 'The model wrote an answer. A run whose calls reached a tool that rejected their input used the tool, and is done.', <code>True</code>],
          [<code>"not_possible"</code>, 'The model wrote an answer, but every call was declined, failed on the tool’s side or returned nothing — usually an answer saying the task cannot be done with these tools.', <code>True</code>],
          [<code>"stuck"</code>, 'The run kept proposing work that brought nothing new, was asked for its answer, and wrote none (loop_detected, repeated_tool_result, max_iterations_*, null_final_from_model).', <code>False</code>],
          [<code>"tool_failed"</code>, 'The tools the run needed failed — on their own side, or on every input the model gave them — and the run has no answer.', <code>False</code>],
          [<code>"error"</code>, 'The run could not be carried out.', <code>False</code>],
        ]}
      />
      <p>
        Before a run ends stuck, the loop asks for the answer. After{' '}
        <code>max_turns_without_progress</code> turns in a row (2 by default) that bring no new tool
        result, the next turn offers no tools and asks for the answer, and a turn that declares no
        action after a result (<code>Action: None</code>, <code>Action: (continue reasoning)</code>)
        is asked at once. A run that would still stop on a loop guard or its iteration cap while
        holding tool results gets one closing request — its own calls and results, with no tools —
        and when the reply is an answer the run succeeds with{' '}
        <code>metadata["answer_source"] == "closing_request"</code>.
      </p>
      <p>
        A tool that fails on its own side — a connection error, a timeout, an HTTP 5xx or 429,
        missing credentials — three times in a row is not called again in the run, and when every
        tool the agent holds is in that state the run ends <code>tool_failed</code> without another
        model call; <code>metadata["unavailable_tools"]</code> names them. Four failures in a row on
        a tool’s input withdraw that tool for the run; a run left with no tool is asked for its
        answer and ends <code>tool_failed</code> with <code>metadata["error"]["kind"] == "input"</code>{' '}
        when it writes none. An answer that is a tool’s error message is never a success.{' '}
        <code>max_turns_without_progress=None</code> turns the answer request and the closing request
        off, which is the loop 1.2.0 ran.
      </p>

      <h2>Constructing an agent</h2>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'config',
            type: 'AgentConfig | None',
            default: 'None',
            description: 'The settings. Omitting it is only useful in subclasses that supply one.',
          },
          {
            name: 'session_id',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                A stored conversation to load or create, so multi-turn context survives across
                processes. Per-call conversations use <code>run(session=...)</code> instead — see{' '}
                <Link to="/sessions">Sessions</Link>.
              </>
            ),
          },
        ]}
        caption="Agent(config=None, session_id=None)"
      />

      <h2>Running a task</h2>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'task',
            type: 'str | Message | list[ContentPart]',
            required: true,
            description:
              'The task. A plain string, a multimodal Message, or a list of content parts — text is extracted and any image, audio or video parts go through the multimodal path.',
          },
          {
            name: 'mode',
            type: 'AgentMode | None',
            default: 'None',
            description:
              'Overrides config.mode for this call. AgentMode.AUTO lets the router decide from task complexity.',
          },
          {
            name: 'context',
            type: 'dict[str, Any] | None',
            default: 'None',
            description: 'Extra context for this call.',
          },
          {
            name: 'output_schema',
            type: 'dict | type[BaseModel] | None',
            default: 'None',
            description:
              'A JSON Schema dict or a Pydantic model class. The final output is then valid JSON matching it. Any other type raises TypeError.',
          },
          {
            name: 'output_model',
            type: 'type[BaseModel] | None',
            default: 'None',
            description: (
              <>
                A Pydantic model class. The output is validated and the parsed instance is stored
                in <code>response.metadata["parsed"]</code>. See{' '}
                <Link to="/generation">Generation controls</Link>.
              </>
            ),
          },
          {
            name: 'inputs',
            type: 'list[ContentPart] | None',
            default: 'None',
            description:
              'Multimodal parts made by image_from, audio_from or video_from. With these present the agent sends a structured Message.',
          },
          {
            name: 'session',
            type: 'Session | str',
            description:
              'A conversation, by object or by id. The prompt is built from that conversation and this turn is appended to it, so one agent can serve many conversations.',
          },
          {
            name: 'middleware',
            type: 'list[AgentMiddleware]',
            description: 'Hooks for this call only, appended to any on the config.',
          },
          {
            name: 'debug',
            type: 'bool',
            default: 'False',
            description: 'Attach a DebugTrace to the response.',
          },
        ]}
        caption={
          <>
            <code>
              Agent.run(task, mode=None, context=None, output_schema=None, output_model=None,
              inputs=None, **kwargs)
            </code>{' '}
            — <code>session</code>, <code>middleware</code> and <code>debug</code> arrive through{' '}
            <code>**kwargs</code>. Sampling keywords such as <code>temperature</code> and{' '}
            <code>max_tokens</code> do too, overriding the config for one call.
          </>
        }
      />

      <h2>AgentConfig</h2>
      <p>
        Every field, with the type and default the dataclass declares. Anything not passed keeps
        the default shown.
      </p>

      <ParamTable
        nameLabel="Field"
        params={paramsFor(siteData.api.agent_config, CONFIG_NOTES)}
        caption={
          <>
            Generated from the installed <code>AgentConfig</code> dataclass. A field listed as{' '}
            <code>(empty)</code> is built per instance — an empty list or dict.
          </>
        }
      />

      <Callout type="warning" title="raise_on_error defaults to True">
        <p>
          Since 1.0.0 a failed run raises its typed error rather than
          returning a response with <code>success=False</code> and a plausible-looking string in{' '}
          <code>output</code>. Set it to <code>False</code> to inspect the response yourself — and
          note that with the flag off, a failed run's <code>output</code> is effGen's report of
          what stopped it, while the model's own text is in{' '}
          <code>metadata["partial_output"]</code>. A backend that was never reached raises either
          way. Since 1.0.1 a run the loop stopped before the model wrote an answer raises{' '}
          <code>RunStoppedError</code>, which carries the response, its <code>stop_reason</code>{' '}
          and its <code>partial</code>. <Link to="/migration">Migrating to {version}</Link> has the
          migration.
        </p>
      </Callout>

      <h2>AgentResponse</h2>
      <p>
        Not a string. <code>print(response)</code> prints the answer, and every field below says
        something about the run that produced it. It is imported from{' '}
        <code>effgen.core.agent</code> — it is not exported from the top-level package.
      </p>

      <ParamTable
        nameLabel="Field"
        params={paramsFor(siteData.api.agent_response, RESPONSE_NOTES)}
        caption={
          <>
            Generated from the installed <code>AgentResponse</code> dataclass. On top of these it
            carries <code>text</code> and <code>content</code> (aliases for <code>output</code>),{' '}
            <code>termination</code> (how the run ended, since 1.3.0),{' '}
            <code>tool_call_count</code>, <code>to_dict()</code>, and <code>show()</code> /{' '}
            <code>trace()</code> for printing a run in a terminal.
          </>
        }
      />

      <CodeBlock
        filename="response.py"
        code={`from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
r = agent.run("What is 6 * 7?")

print(r.text)                        # the answer; r.output and r.content are the same string
print(r.success, r.iterations)
print(r.tool_calls.total, r.tool_call_count)
print(r.model, r.provider)
print(r.metadata["cost_usd"], r.metadata["latency_ms"])`}
      />

      <Terminal
        command="python response.py"
        output={`42

Verification:
- Calculator result: 6 * 7 = 42
- Python result: print(6 * 7) -> 42

Conclusion: Indeed, 6 multiplied by 7 equals 42.
True 2
2 2
openai:gpt-5-nano openai
0.00065765 12403.7`}
        caption="Two tool calls over two loop iterations. r.model carries the id as it was given, prefix and all."
      />

      <h2>Reading the tool calls</h2>
      <p>
        <code>tool_calls</code> is a <code>ToolCallList</code>: iterate it for the calls
        themselves, or use it as the count. Each entry is a <code>ToolCall</code>.
      </p>

      <ApiTable
        headers={['Field', 'What it is']}
        rows={siteData.api.tool_call.map((field) => [
          <code>{field.name}</code>,
          <code>{field.type}</code>,
        ])}
        caption={
          <>
            The reading surface <code>ToolCallList</code> adds on top of a list:{' '}
            {siteData.api.tool_call_list.map((name, i) => (
              <span key={name}>
                {i > 0 ? ', ' : ''}
                <code>{name}</code>
              </span>
            ))}
            .
          </>
        }
      />

      <CodeBlock
        filename="calls.py"
        code={`from effgen import create_agent

agent = create_agent("math", "gemini:gemini-3.1-flash-lite")
r = agent.run("What is 4817 * 236?")

for call in r.tool_calls:
    print(call.iteration, call.name, call.arguments, "->", call.error or call.result)

print("names:", r.tool_calls.names)
print("failed:", r.tool_calls.failed.total)
print("calculator calls:", r.tool_calls.by_name("calculator").total)`}
      />

      <Terminal command="python calls.py" output={`1 calculator {"expression": "4817 * 236"} -> 1136812
names: ['calculator']
failed: 0
calculator calls: 1`} />

      <Callout type="note" title="How much of a call is recorded depends on the provider">
        <p>
          <code>iteration</code>, <code>arguments</code> and <code>result</code> are filled in by
          the adapter that made the call. The Gemini adapter records all three. The OpenAI adapter
          records the name and leaves the rest <code>None</code>, so the same loop against{' '}
          <code>openai:gpt-5-nano</code> prints <code>None calculator None -&gt; None</code>.{' '}
          <code>names</code>, <code>failed</code> and <code>by_name()</code> report the same thing
          on both.
        </p>
      </Callout>

      <Callout type="note" title="tool_calls changed in 1.0.0">
        <p>
          It used to be an integer, and iterating it raised{' '}
          <code>TypeError: 'int' object is not iterable</code>. It still compares and casts as the
          count, so <code>tool_calls == 2</code> and <code>tool_calls &gt; 0</code> are unchanged,
          and <code>to_dict()</code> keeps the count under its original key while adding{' '}
          <code>tool_call_details</code>.
        </p>
      </Callout>

      <h2>When a run fails</h2>

      <ApiTable
        headers={['Error', 'When', 'What to do']}
        rows={[
          [
            <code>BackendUnreachableError</code>,
            'A refused connection, an unresolvable host or a missing route — the backend was never reached.',
            <>
              Raises whatever <code>raise_on_error</code> says, by design: there is no result to
              inspect. Catch it where you want to handle it.
            </>,
          ],
          [
            <code>ModelAuthError</code>,
            'The provider rejected the credential.',
            <>
              Check the key with <code>effgen doctor</code>.
            </>,
          ],
          [
            <code>ModelNotFoundError</code>,
            'The provider does not serve that model id.',
            <>
              <code>effgen models list --provider &lt;name&gt;</code>, then{' '}
              <code>effgen models refresh</code>.
            </>,
          ],
          [
            <code>RateLimitExceeded</code>,
            "The provider's limit was hit.",
            <>
              Retryable. A <Link to="/routing">router</Link> fails over to another provider on
              this.
            </>,
          ],
          [
            <>A run whose tools failed</>,
            <>
              Since 1.3.0: the tools the run needed kept failing on their own side, or on every
              input the model gave them, and the run has no answer.
            </>,
            <>
              Raises <code>RunStoppedError</code> with <code>stop_reason="tool_failed"</code>;{' '}
              <code>metadata["unavailable_tools"]</code> names the tools and{' '}
              <code>metadata["error"]["kind"]</code> says <code>"tool"</code> or{' '}
              <code>"input"</code>. Check the tool’s service, or read{' '}
              <code>response.termination</code> with <code>raise_on_error=False</code>.
            </>,
          ],
          [
            <>A stopped run</>,
            <>
              The loop reached <code>max_iterations</code>, or a guard ended a run that repeated
              itself, without the model answering — since 1.3.0, also after the closing request
              brought no answer.
            </>,
            <>
              Raises <code>RunStoppedError</code>, a <code>RuntimeError</code>. Raise the cap,
              simplify the task, or read <code>exc.partial</code> — or set{' '}
              <code>raise_on_error=False</code> and read <code>response.partial</code>.
            </>,
          ],
        ]}
      />

      <CodeBlock
        filename="inspect_failure.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    tools=[Calculator()],
    max_iterations=1,
    raise_on_error=False,          # the default is True
))
r = agent.run("With the calculator: work out 24344 * 334, then multiply that by 7, "
              "then subtract 19, then divide by 3.")

print("success:", r.success)
print("outcome:", r.outcome, "· stop_reason:", r.stop_reason)
print("output:", r.output[:100])
print("partial:", r.partial.text if r.partial else None)`}
      />

      <Terminal command="python inspect_failure.py" output={`success: False
outcome: stopped · stop_reason: max_iterations_partial
output: Stopped after 1 iteration without a final answer: 'gemini-3.1-flash-lite' was still taking tool step
partial: 8130896`} caption={`Run against effGen 1.0.1. output says what stopped the run; what it had reached is in partial.`} />

      <SeeAlso paths={['/presets', '/configuration', '/tool-calling']} />
    </DocPage>
  );
}
