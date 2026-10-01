import { GraduationCap } from 'lucide-react';
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

/**
 * The twelve tutorials the framework ships, and where each one is on this site.
 *
 * Nine of them are the same ground as a page that already exists here, in more
 * depth than a tutorial has room for, so the row points at that page. The other
 * three build an agent end to end and are written out below.
 */
const TUTORIALS: { file: string; what: string; where: string; to?: string }[] = [
  {
    file: 'getting-started.md',
    what: 'Install it, configure a model, run one agent.',
    where: 'Quick start',
    to: '/quickstart',
  },
  {
    file: 'building-math-agent.md',
    what: 'An agent that computes instead of guessing.',
    where: 'On this page',
  },
  {
    file: 'building-research-agent.md',
    what: 'An agent that looks things up and says where it looked.',
    where: 'On this page',
  },
  {
    file: 'building-code-agent.md',
    what: 'An agent that runs the code it writes before answering.',
    where: 'On this page',
  },
  {
    file: 'custom-tools.md',
    what: 'Turning one of your functions into a tool.',
    where: 'Writing tools & plugins',
    to: '/custom-tools',
  },
  {
    file: 'native-tool-calling.md',
    what: 'The three tool-calling strategies and when each is used.',
    where: 'Tool calling',
    to: '/tool-calling',
  },
  {
    file: 'rag-pipeline.md',
    what: 'Indexing your own documents and answering from them.',
    where: 'RAG',
    to: '/rag',
  },
  {
    file: 'multi-agent.md',
    what: 'Several agents on one task, and how the answers are combined.',
    where: 'Multi-agent teams',
    to: '/multi-agent',
  },
  {
    file: 'multi-agent-workflows.md',
    what: 'A DAG of steps, with checkpoints it can resume from.',
    where: 'Workflows',
    to: '/workflows',
  },
  {
    file: 'guardrails.md',
    what: 'Checks on the way in and on the way out.',
    where: 'Guardrails',
    to: '/guardrails',
  },
  {
    file: 'evaluation.md',
    what: 'Test cases, suites and a gate that fails a build.',
    where: 'Evaluation & CI gates',
    to: '/evaluation',
  },
  {
    file: 'production-deployment.md',
    what: 'Containers, health checks and what to watch.',
    where: 'Deployment',
    to: '/deployment',
  },
];

export default function Tutorials() {
  return (
    <DocPage
      subtitle="Twelve end-to-end builds. Three of them are written out here in full; the other nine are the subject of a page of their own."
      icon={<GraduationCap size={48} />}
    >
      <p>
        A tutorial is a whole program: something to build, the code to build it with, and the
        output it produced when it was run. The three below each take one task from an empty file
        to a printed answer, and every block on this page is a real run against effGen 1.0.1 —
        including the one where a tool fails.
      </p>

      <Callout type="tip" title="The model these use">
        <p>
          Every sample here runs on <code>openai:gpt-5-nano</code>, because it is cheap and it
          answers in seconds. Any model works: swap the two <code>AgentConfig</code> lines for a
          local one (<Link to="/local-models">Local models</Link>) or point them at your own
          endpoint (<Link to="/openai-compatible">Any OpenAI-compatible server</Link>). Nothing else
          in the code changes.
        </p>
      </Callout>

      <h2>The twelve tutorials, and where each one is</h2>

      <ApiTable
        headers={['Tutorial', 'What it builds', 'Where']}
        rows={TUTORIALS.map((row) => [
          <code>{row.file}</code>,
          row.what,
          row.to ? <Link to={row.to}>{row.where}</Link> : <em>{row.where}</em>,
        ])}
      />

      <h2>Build a math agent</h2>

      <p>
        A language model asked for a standard deviation will produce a number that looks right. An
        agent with <code>Calculator</code> and <code>PythonREPL</code> computes one instead. The
        difference shows up in <code>response.tool_calls</code>: if the list is empty, nothing was
        computed.
      </p>

      <Steps>
        <Step title="Wire the two tools in">
          <p>
            <code>tools=</code> takes tool <em>instances</em>, not names. Pass a string and you get
            a <code>TypeError</code> that tells you so.
          </p>
        </Step>
        <Step title="Lower the temperature">
          <p>
            Arithmetic has one right answer. <code>temperature=0.3</code> keeps the model from
            wandering off the tool's result.
          </p>
        </Step>
        <Step title="Read the calls, not just the answer">
          <p>
            <code>response.tool_calls.names</code> is the list of tools the run actually reached
            for, in order.
          </p>
        </Step>
      </Steps>

      <CodeBlock
        filename="math_agent.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

agent = Agent(AgentConfig(
    name="math-tutor",
    model="gpt-5-nano",
    provider="openai",
    tools=[Calculator(), PythonREPL()],
    system_prompt=(
        "You are a precise math tutor. Show your working. Use the calculator for "
        "arithmetic and python_repl for anything statistical."
    ),
    max_iterations=8,
    temperature=0.3,
))

response = agent.run("What is the standard deviation of [4, 8, 15, 16, 23, 42]?")
print(response.output)
print()
print("success   :", response.success)
print("tools     :", response.tool_calls.names)
print("calls     :", response.tool_calls.total)
print("iterations:", response.iterations)`}
      />

      <Terminal
        command="python math_agent.py"
        output={`- Population standard deviation ≈ 12.315302
- Sample standard deviation ≈ 13.490738

If you need a specific convention (population or sample) for reporting, tell me which one to use.

success   : True
tools     : ['calculator', 'python_repl']
calls     : 2
iterations: 3`}
        caption={`Run against effGen 1.0.0. Both tools were used, in three passes of the loop.`}
      />

      <h3>The same agent, from a preset</h3>

      <p>
        <Link to="/presets">A preset</Link> is a tool set someone has already chosen, with a system
        prompt and generation settings to match. <code>math</code> is the two tools above.
      </p>

      <CodeBlock
        filename="from_preset.py"
        code={`from effgen import create_agent, list_presets

print("presets:", ", ".join(list_presets()))

agent = create_agent("math", "gpt-5-nano", provider="openai")
print("tools  :", [t.name for t in agent.config.tools])

response = agent.run("If I invest $1000 at 5% a year compounded monthly, what is it worth after 10 years?")
print()
print(response.output)`}
      />

      <Terminal
        command="python from_preset.py"
        output={`presets: math, research, coding, general, rag, minimal, multimodal, notify, media

tools  : ['calculator', 'python_repl']

$1,647.01 after 10 years.`}
      />

      <h2>Build a research agent</h2>

      <p>
        Research tools return sources, and a run that used them carries those sources back on the
        response. <code>response.sources</code> is what lets you check the answer rather than
        believe it.
      </p>

      <CodeBlock
        filename="research_agent.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import ArXivTool, URLFetchTool, WikipediaTool

agent = Agent(AgentConfig(
    name="researcher",
    model="gpt-5-nano",
    provider="openai",
    tools=[WikipediaTool(), ArXivTool(), URLFetchTool()],
    system_prompt=(
        "You are a research assistant. Look things up before answering, "
        "prefer primary sources, and name what you used."
    ),
    max_iterations=10,
    temperature=0.5,
))

response = agent.run(
    "Find one 2024-or-later arXiv paper about speculative decoding and summarise its claim in two sentences."
)
print(response.output)
print()
print("tools  :", response.tool_calls.names)
print("sources:", response.sources[:3])`}
      />

      <Terminal
        command="python research_agent.py"
        output={`Speculative Safety-Aware Decoding (SSD) is a decoding-time approach that aims to imbue a large language model with a desired safety property by leveraging a smaller, safety-focused model and applying speculative sampling to speed up inference. It gauges jailbreak risk via the match ratio between the small and large models, dynamically switches decoding schemes to prioritize safety or utility, and samples the next token from a blended distribution of both models [1].

tools  : ['arxiv', 'arxiv']
sources: ['http://arxiv.org/abs/2203.16487v6', 'http://arxiv.org/abs/2605.01106v1', 'http://arxiv.org/abs/2508.17739v2']`}
        maxLines={16}
        caption="Two searches, three sources kept. arXiv, Wikipedia and URL fetching need no API key; a web search tool does — see the tool gallery for which."
      />

      <p>
        <code>research</code> is also a preset, and it adds PubMed and Semantic Scholar to the
        three above. Every one of those tools is listed with its parameters in the{' '}
        <Link to="/tools/gallery">tool gallery</Link>.
      </p>

      <h2>Build a coding agent</h2>

      <p>
        The point of a coding agent is not that it writes code — it is that it runs the code before
        telling you it works. Give it <code>PythonREPL</code> for quick expressions and{' '}
        <code>CodeExecutor</code> for a whole program in a{' '}
        <Link to="/execution">sandbox</Link>, and the answer becomes something the machine printed.
      </p>

      <CodeBlock
        filename="code_agent.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import CodeExecutor, PythonREPL

agent = Agent(AgentConfig(
    name="code-agent",
    model="gpt-5-nano",
    provider="openai",
    tools=[PythonREPL(), CodeExecutor()],
    system_prompt=(
        "You are an expert programmer. Write the code, run it on a real input, "
        "and answer with what it printed rather than with what you expect it to print."
    ),
    max_iterations=12,
    temperature=0.4,
))

response = agent.run(
    "Write a Python function that returns the longest palindromic substring of a string, "
    "then run it on 'forgeeksskeegfor' and tell me what it returned."
)
print(response.output)
print()
for call in response.tool_calls:
    print(f"  {call.name}: {'ok' if not call.error else 'error: ' + str(call.error)[:60]}")`}
      />

      <Terminal
        command="python code_agent.py"
        output={`geeksskeeg

  code_executor: ok
  code_executor: ok`}
        caption="The answer is the string the executed program printed. Two executions: one to write and run the function, one to run it on the input."
      />

      <h3>When a step fails</h3>

      <p>
        A tool that raises does not end the run. The call is recorded with its error, the model
        sees the message, and <code>response.tool_calls.failed</code> is how you find out
        afterwards — including when the agent decided to stop rather than try again.
      </p>

      <CodeBlock
        filename="failing_step.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import PythonREPL

agent = Agent(AgentConfig(
    name="recovers",
    model="gpt-5-nano",
    provider="openai",
    tools=[PythonREPL()],
    max_iterations=6,
))

response = agent.run(
    "Compute 1/0 with python_repl, then report exactly what the tool said, "
    "then compute 1/4 the same way and report that."
)
print(response.output)
print()
print("calls:", response.tool_calls.total, "| failed:", len(response.tool_calls.failed))
for call in response.tool_calls:
    outcome = f"error: {call.error}" if call.error else f"ok: {call.result}"
    print(f"  {call.name}  {str(outcome)[:90]}")`}
      />

      <Terminal
        command="python failing_step.py"
        output={`I will first attempt 1/0 in the python_repl to observe the error output, then I will compute 1/4 to observe the result.

calls: 1 | failed: 1
  python_repl  error: Error executing tool 'python_repl': Tool execution failed: ZeroDivisionError: divis`}
        caption="One call, and it failed. The answer text describes a plan rather than a result — which is exactly the case the counters exist to make visible."
      />

      <Callout type="warning" title="Read the counters, not the prose">
        <p>
          The run above is a success by every field that describes the transport:{' '}
          <code>success</code> is <code>True</code>, nothing raised, an answer came back. It is the
          two numbers that tell you the work did not happen — one call, one failure, and a reply
          that describes what it is about to do. Any check worth having asserts on{' '}
          <code>tool_calls</code>, not on the text. <Link to="/evaluation">Evaluation &amp; CI
          gates</Link> is how you make that a build failure rather than something you notice later.
        </p>
      </Callout>

      <h2>What to read next</h2>

      <p>
        Each of these takes one of the agents above further:{' '}
        <Link to="/memory">Memory</Link> and <Link to="/sessions">Sessions</Link> for an agent that
        remembers, <Link to="/rag">RAG</Link> for one that answers from your own documents,{' '}
        <Link to="/custom-tools">Writing tools</Link> for one that reaches into your systems,{' '}
        <Link to="/multi-agent">Multi-agent teams</Link> for several at once, and{' '}
        <Link to="/deployment">Deployment</Link> for putting one behind an endpoint.
      </p>

      <SeeAlso paths={['/quickstart', '/cookbook', '/examples']} />
    </DocPage>
  );
}
