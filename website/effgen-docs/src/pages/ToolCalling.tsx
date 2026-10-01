import { Wrench } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function ToolCalling() {
  return (
    <DocPage
      subtitle="The strategies a model can be asked for tools with, and the one shape a turn reports them in."
      icon={<Wrench size={48} />}
    >
      <p>
        Providers disagree about how a model asks for a tool — some expose a function-calling API,
        some only generate text you have to read a call out of, and the chat templates that do the
        latter disagree with each other. effGen hides both halves: one setting chooses how tools
        are offered, and every adapter reports what came back in the same shape.
      </p>

      <h2>Reading the calls a turn made</h2>

      <CodeBlock
        filename="calls.py"
        code={`import json

from effgen import load_model

TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

model = load_model("openai:gpt-5-nano")
result = model.generate_with_tools("What is 6*7? Use the calculator.", tools=TOOLS)

for call in result.metadata["tool_calls"]:
    name = call["function"]["name"]
    try:
        arguments = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError:
        print(f"{name} was called with unparseable arguments: {call['function']['arguments']!r}")
        continue
    print(call["id"], name, arguments)`}
      />

      <Terminal command="python calls.py" output={`call_YHN5s6lhtcadKd5d5OURdc8r calculator {'expression': '6*7'}`} caption={`Run against effGen 1.0.0.`} />

      <p>
        Swap the model id for <code>gemini:gemini-3.1-flash-lite</code>,{' '}
        <code>groq:…</code>, <code>together:…</code>, <code>fireworks:…</code> or{' '}
        <code>hf:…</code> and the loop runs unchanged.
      </p>

      <Callout type="tip" title="Most callers never need this">
        <p>
          <code>Agent</code> dispatches tool calls itself and reports them on{' '}
          <code>AgentResponse.tool_calls</code> — see <Link to="/agents">Agents</Link>. The list
          below matters when you drive an adapter by hand, or forward calls onto a wire of your
          own.
        </p>
      </Callout>

      <h2>The shape</h2>
      <p>
        Every adapter reports the calls a turn made under{' '}
        <code>GenerationResult.metadata["tool_calls"]</code>, in this shape:
      </p>

      <CodeBlock
        code={`[
    {
        "id": "call_YHN5s6lhtcadKd5d5OURdc8r",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression":"6*7"}',
        },
    },
]`}
      />

      <ApiTable
        headers={['Rule', 'What it means for a reader']}
        rows={[
          [
            'The key is always present and always a list',
            'A turn that called nothing reports []. No KeyError guard is needed, including on the local engines, which report tool calls as text in result.text rather than as a structured list.',
          ],
          [
            <>
              Every element carries <code>id</code>, <code>type</code> and <code>function</code>
            </>,
            <>
              <code>id</code> is the provider's call id, or <code>""</code> when the provider sends
              none. <code>type</code> is <code>"function"</code> for a function call.{' '}
              <code>function</code> carries <code>name</code> and <code>arguments</code>.
            </>,
          ],
          [
            <>
              <code>arguments</code> is a JSON string, exactly as the model generated it
            </>,
            'The adapter never parses it. That is what the wire format requires, and it keeps a model’s malformed JSON visible instead of arriving silently as an empty argument set.',
          ],
          ["Order is the provider's order", 'Parallel calls in one turn are separate elements.'],
        ]}
      />

      <Callout type="warning" title="Parse the arguments defensively">
        <p>
          A model can and does emit invalid JSON. Because the string is handed to you unparsed, a{' '}
          <code>json.JSONDecodeError</code> is yours to catch — and the raw text is still there to
          report, which is the point of not parsing it for you.
        </p>
      </Callout>

      <h3>What two adapters add</h3>

      <ApiTable
        headers={['Provider', 'Addition', 'Why']}
        rows={[
          [
            'Gemini',
            <>
              top-level <code>name</code> and <code>arguments</code>, the parsed mapping
            </>,
            'Gemini previously reported only this flat form; the keys stay for one release so callers written against it keep working. Prefer the nested block.',
          ],
          [
            'Anthropic',
            <>
              <code>metadata["tool_uses"]</code>, a list of{' '}
              <code>{'{id, name, input}'}</code>
            </>,
            <>
              Anthropic's own <code>tool_use</code> block shape, with <code>input</code> already
              parsed.
            </>,
          ],
        ]}
        caption="Additions sit beside the block above, not instead of it, and are read only by callers that ask for them."
      />

      <p>
        The OpenAI Responses API path (<code>generate_with_native_tools</code>) reports its entries
        in <code>metadata["native_tool_results"]</code> — the same list object as{' '}
        <code>metadata["tool_calls"]</code>. Those entries keep the Responses API's own{' '}
        <code>type: "function_call"</code> discriminator and its flat keys, because the list also
        carries server-side results such as <code>web_search_call</code> that are not function
        calls at all. The nested <code>function</code> block is present on the function-call
        entries as well. See <Link to="/native-provider-tools">Provider-native tools</Link>.
      </p>

      <h2>A tool loop that runs on any provider</h2>
      <p>
        Feeding a result back is where the wire formats disagree most — a <code>role: "tool"</code>{' '}
        message on the OpenAI-compatible providers, <code>functionResponse</code> parts on Gemini,{' '}
        <code>tool_result</code> blocks on Anthropic. Two methods on every model build the right
        shape, so one loop is portable.
      </p>

      <CodeBlock
        filename="loop.py"
        code={`import json

from effgen import load_model

TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

def run_tool(name, args):
    return str(eval(args["expression"], {"__builtins__": {}}))   # a real tool would not use eval

adapter = load_model("openai:gpt-5-nano")
messages = [{"role": "user", "content": "What is 4817 * 236? Use the calculator."}]

result = adapter.generate_with_tools("", tools=TOOLS, messages=messages)
messages.append(adapter.build_assistant_message(result))
for call in result.metadata["tool_calls"]:
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
    messages.append(adapter.build_tool_result_message(call["id"], name, run_tool(name, args)))

print(adapter.generate_with_tools("", tools=TOOLS, messages=messages).text.strip())`}
      />

      <Terminal command="python loop.py" output={`1,136,812`} />

      <p>
        Swapping the adapter is the only edit that loop needs. Gemini and Anthropic override both
        builders with their own shapes; the call site does not change.
      </p>

      <h2>How tools are offered: tool_calling_mode</h2>

      <ParamTable
        nameLabel="Mode"
        params={[
          {
            name: '"auto"',
            type: 'the default',
            description:
              'Ask the model whether it supports native tool calling. If it does, use the hybrid strategy; if not, use ReAct. Since 1.3.0, for a model behind a base_url or on a local engine, it also follows what the capability probe measured (below).',
          },
          {
            name: '"native"',
            type: '',
            description:
              'Send the tools as function definitions and read structured calls back. Only for a model that supports it.',
          },
          {
            name: '"react"',
            type: '',
            description:
              'Describe the tools in the prompt and read the call out of the text the model writes — Thought / Action / Action Input / Observation. Works with any model that can follow instructions.',
          },
          {
            name: '"hybrid"',
            type: '',
            description: 'Try native first, and fall back to reading the text when the parse fails.',
          },
        ]}
        caption={
          <>
            <code>AgentConfig(tool_calling_mode=...)</code>. An unrecognised value logs a warning
            and falls back to ReAct.
          </>
        }
      />

      <CodeBlock
        filename="modes.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

for mode in ("native", "react", "hybrid"):
    agent = Agent(AgentConfig(
        model="openai:gpt-5-nano",
        tools=[Calculator()],
        tool_calling_mode=mode,
        temperature=0.0,
    ))
    r = agent.run("What is 1234 * 5678?")
    print(f"{mode:7s} {r.output.strip()[:40]:42s} tool calls: {r.tool_call_count}")`}
      />

      <Terminal
        command="python modes.py"
        output={`native  7,006,652                                  tool calls: 0
react   7006652                                    tool calls: 1
hybrid  7,006,652                                  tool calls: 0`}
        caption="A capable model may simply answer rather than reach for the calculator, which is why native and hybrid report no calls here and ReAct — which puts the tool in the prompt and asks for a step — reports one. The answer is the same either way."
      />

      <Callout type="note" title="A mode is not a guarantee that a tool is used">
        <p>
          Whether a tool is called is the model's decision unless the run says otherwise. If a run
          must go through its tools, set <code>tool_use="required"</code> (below), or check{' '}
          <code>response.tool_calls</code> and treat an empty list as a failure.
        </p>
      </Callout>

      <h3>Measured, not assumed: the capability probe</h3>
      <p>
        A declaration says how tool definitions reach a model, not whether the model uses them. A
        model served behind a <code>base_url</code> is declared an API caller whatever sits behind
        the URL, and two things can still go wrong: a small model may answer from memory while
        holding a search tool that has the answer, and a server may not carry a model’s native
        calls back as structured calls. Since 1.3.0, at <code>tool_calling_mode="auto"</code>, the
        first agent with tools built for such a model, or for a model on a local engine, measures
        this once: a few questions about fictional things, answerable only through a stub search
        tool, run as ordinary agent runs. Each is resolved, unresolved or skipped.
      </p>
      <ApiTable
        headers={['Measured', 'What auto does for this model']}
        rows={[
          ['Three or more native runs unresolved, and the text frame resolves at least three more', 'Uses the ReAct text frame.'],
          ['Any native run skipped', 'A run holding an information-retrieval tool that answers without calling it is sent back once and made to call.'],
          ['Otherwise', 'As the declaration says.'],
        ]}
      />
      <p>
        Calculator and code tools are never moved by the probe. An explicit{' '}
        <code>tool_calling_mode</code>, an explicit <code>tool_use</code>,{' '}
        <code>capability_probe=False</code> or <code>EFFGEN_CAPABILITY_PROBE=0</code> keep the
        declared behaviour, and first-party cloud adapters are never probed. The result is stored
        in <code>~/.effgen/capabilities.json</code> (or under <code>$EFFGEN_HOME</code>, or at{' '}
        <code>$EFFGEN_CAPABILITY_CACHE</code>) and read by every later agent, across processes; it
        is measured again when the weights, the chat template, the endpoint or the probe change,
        and after 30 days. A probe is bounded at 48 requests and 120 seconds, and one that cannot
        run stores nothing, leaves <code>auto</code> as declared and logs one warning.{' '}
        <code>response.metadata["tool_calling"]</code> says which strategy a run used and why.
      </p>
      <CodeBlock
        filename="probe.py"
        code={`from effgen import probe_tool_calling
from effgen.models import load_model

model = load_model("Qwen/Qwen2.5-1.5B-Instruct", provider="openai_compatible",
                   base_url="http://127.0.0.1:8000/v1", context_length=8192)
probe = probe_tool_calling(model)
print(probe.summary(), probe.strategy, probe.required_categories)`}
      />
      <p>
        <code>effgen doctor --probe MODEL</code> does the same from the command line. The same store
        keeps two facts learned from real requests: a provider that rejects stop sequences sent
        beside tool definitions is asked once more with the stops applied locally, and a server
        that rejects a pinned <code>reasoning_effort</code> is asked once more without it.
      </p>
      <Callout type="warning" title="Small models given a search tool cost more">
        <p>
          When the probe finds that a model answers from memory while holding a search or
          retrieval tool, every run holding one is made to call it, so such runs make several times
          as many model calls on question-answering tasks. <code>tool_use="auto"</code> or{' '}
          <code>capability_probe=False</code> turns it off for an agent. A probe’s own requests are
          booked in the first agent’s cost ledger, with no tag marking them as the probe’s.
        </p>
      </Callout>

      <h2>What the model is told about its tools</h2>
      <p>
        Tool definitions travel through the provider's tool-calling API or a local chat template,
        not the prompt, so the model is left to work out what they are for. effGen states it once,
        on the turn the model first sees them — the same sentences whether the run blocks or
        streams, and whether the tools reach the model through an API or a template. Which
        sentences depends on each tool's declared <code>ToolCategory</code>:
      </p>

      <ApiTable
        headers={['The tools you attached', 'What the model is told']}
        rows={[
          [<code>COMPUTATION</code>, 'Use the tools to check the steps you are least sure of, one step per call, and correct yourself if a tool disagrees.'],
          [
            <>
              <code>CODE_EXECUTION</code>, <code>SYSTEM</code>
            </>,
            'Use the tools to do the task rather than working it out in your head, and read the answer off what they return.',
          ],
          [
            <>
              <code>INFORMATION_RETRIEVAL</code>, <code>EXTERNAL_API</code>
            </>,
            'What comes back is source material, not the answer.',
          ],
          ['Anything else, or a set that mixes the above', 'Work through the task one step at a time, one step per call.'],
        ]}
        caption={
          <>
            Since 1.2.0 the contracts no longer ask for text the task did not ask for: the lookup
            contract no longer asks the model to name what is missing, and the computation contract
            no longer asks it to work the task out step by step and restate the answer. The
            contract comes before the caller’s task, so the task is the last thing the model reads; <code>AgentConfig(answer_style=...)</code> is where the
            form of the answer is set. <code>AgentConfig(tool_contract="...")</code> states your
            text in the same position instead, and <code>tool_contract=""</code> states nothing. A{' '}
            <code>system_prompt</code> persona still leads the prompt either way, and a{' '}
            <code>system_prompt_template</code> owns the whole prompt.
          </>
        }
      />

      <p>
        Whether a tool <em>has</em> to be called is a separate setting, <code>tool_use</code>. By
        default it is read from the same categories: a code executor or a system tool is{' '}
        <code>required</code> — an answer written with no call is sent back once, naming the tool,
        and the next turn is required to call where the provider can enforce it — and everything
        else is <code>auto</code>. A set that mixes categories takes the strictest policy present.
      </p>

      <ParamTable
        nameLabel="tool_use"
        params={[
          {
            name: '"required"',
            description:
              'Every attached tool is one the run may not answer without — on the ReAct and chat-template paths too, where there is no request parameter to constrain.',
          },
          {
            name: '"auto"',
            description: 'effGen states nothing and requires nothing, whatever the tools declare.',
          },
          {
            name: '"sparing"',
            description:
              'One further sentence after the contract: calling a tool is optional, and a run that already knows the answer should give it.',
          },
        ]}
        caption={
          <>
            A string or a <code>ToolUsePolicy</code> member, from{' '}
            <code>effgen.prompts.tool_contract</code>. <code>run(tool_choice="required")</code>{' '}
            constrains a single call on adapters where{' '}
            <code>model.supports_forced_tool_call()</code> is true.
          </>
        }
      />

      <h2>Reading a call out of text</h2>
      <p>
        Since 1.2.0 a turn that holds tools and whose reply may be read as text is sent one stop
        sequence, <code>"\nObservation:"</code>, so generation ends where a tool’s result would
        begin; a caller’s own <code>stop_sequences</code> replace it. When a reply holds a written
        action anyway, the action runs and whatever the model wrote after it is discarded, so a
        result the model invents is never taken as the answer. An adapter for a provider that
        rejects <code>stop</code> beside <code>tools</code> answers{' '}
        <code>supports_stop_with_tools()</code> with <code>False</code>, and the framework cuts the
        text itself.
      </p>
      <p>
        Since 1.3.0 <code>recover_lost_tool_calls</code> is on by default. A call written as a
        Python literal, with raw line breaks inside its JSON, with unescaped double quotes inside a
        string value, with its last string closed one bracket early, or as a whole object followed
        by text inside its tag, runs instead of ending the run with <code>written_tool_call</code>;
        a call nothing can read is sent back once, with a call required where the provider supports
        it. When the agent holds exactly one code-execution tool, a program in a fenced block before
        an empty call tag runs as that call. <code>recover_lost_tool_calls=False</code> restores the
        strict reader.
      </p>
      <p>
        Under either setting a call is read or refused, never half-read. Arguments a provider
        returns as a string are the call’s arguments: keyword syntax (<code>code='…'</code>) is read
        as those keywords, an object’s JSON text as that object, and anything else is bound to the
        tool’s parameter; <code>tool_calls</code> records carry the decoded arguments. A call that
        leaves a required parameter without a value is not dispatched and is asked for again.
        Positional values are given the tool’s parameter names in order; a call with more values
        than the tool has parameters carries none and is asked for again.
      </p>
      <p>
        A model without a function-calling API writes its call into the text, and chat templates
        disagree about the spelling. effGen reads all of the common shapes: the ReAct{' '}
        <code>Action:</code> / <code>Action Input:</code> form, a bare JSON object with{' '}
        <code>name</code> and <code>arguments</code>, the Qwen-style{' '}
        <code>&lt;tool_call&gt;</code> JSON wrapper, and an XML dialect that nests one tag per
        argument.
      </p>

      <CodeBlock
        filename="xml.py"
        code={`from effgen.core.tool_calling import NativeFunctionCallingStrategy

TEXT = """<tool_call>
<function=calculator>
<parameter=expression>
4817 * 236
</parameter>
</function>
</tool_call>"""

parsed = NativeFunctionCallingStrategy().parse_response(TEXT)
print(parsed.is_tool_call, parsed.tool_name, parsed.arguments)`}
      />

      <Terminal command="python xml.py" output={`True calculator {'expression': '4817 * 236'}`} />

      <p>
        Both ways of naming a tag are accepted — <code>&lt;function=NAME&gt;</code> and{' '}
        <code>&lt;function name="NAME"&gt;</code> — across{' '}
        <code>function</code>, <code>tool_call</code>, <code>invoke</code>, <code>tool</code> and{' '}
        <code>function_call</code> for a call, and <code>parameter</code>, <code>param</code>,{' '}
        <code>argument</code> and <code>arg</code> for an argument. The closing tag is optional, so
        a call the token budget cut short still reads. Each value is the tag's text: one that is
        valid JSON is decoded, so an integer argument arrives as an <code>int</code> rather than as{' '}
        <code>"3"</code>.
      </p>

      <Callout type="note" title="Why this matters more than it sounds">
        <p>
          Without a reader for that dialect the turn parses to nothing at all — the wrapper tag
          also stops the text being taken as a final answer, so the loop nudges itself to its
          iteration cap and the tool is never called, on every model whose template writes that
          shape.
        </p>
      </Callout>

      <h2>Streaming</h2>
      <p>
        <code>generate_stream()</code> yields text. An adapter that streams a tool call
        accumulates the <code>arguments</code> deltas and finishes the call in the shape above —
        the accumulated JSON string, unparsed — so a streamed call and the same call made without
        streaming agree. To read the structured list, make the call without streaming and read{' '}
        <code>metadata["tool_calls"]</code>. <code>streams_tool_calls()</code> on the model says
        whether it does this at all.
      </p>

      <h2>When tool calling goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'The run ends at the iteration cap with no answer',
            'The model wrote a call in a shape nothing could read, so the turn parsed to nothing.',
            <>
              Try <code>tool_calling_mode="hybrid"</code>, or a model whose template writes a shape
              effGen reads. The raw text is in the execution trace.
            </>,
          ],
          [
            <><code>stop_reason="tool_failed"</code></>,
            'Since 1.3.0: the tools the run needed kept failing on their own side, or on every input the model gave them.',
            <>
              <code>metadata["unavailable_tools"]</code> names them and{' '}
              <code>metadata["error"]["kind"]</code> says whether the tool or its input failed. See{' '}
              <Link to="/errors">Errors</Link>.
            </>,
          ],
          [
            <><code>json.JSONDecodeError</code> on the arguments</>,
            'The model emitted invalid JSON.',
            'Catch it and report the raw string, which is preserved for exactly this. A stricter system prompt, or a smaller schema, usually fixes it.',
          ],
          [
            'A message telling you to use native mode',
            'The adapter was given tool definitions on a path that cannot carry them.',
            <>
              Set <code>tool_calling_mode="native"</code> so the adapter routes the definitions the
              way the provider expects.
            </>,
          ],
          [
            <code>ToolIncompatibleError</code>,
            'The tool cannot be offered to this model — a provider-native tool on a provider that does not have it, for instance.',
            <>
              <Link to="/native-provider-tools">Provider-native tools</Link> says which are
              available where.
            </>,
          ],
          [
            'No tool calls at all',
            'The model decided it did not need one.',
            <>
              Not an error. Check <code>response.tool_calls</code> if a particular tool has to have
              run.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/tools', '/agents', '/native-provider-tools']} />
    </DocPage>
  );
}
