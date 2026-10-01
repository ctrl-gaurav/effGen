import { Rocket } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  Callout,
  CodeBlock,
  DocPage,
  FeatureList,
  MermaidDiagram,
  QuickLinks,
  SeeAlso,
  Terminal,
} from '../components/docs';
import {
  commandCount,
  modelCount,
  presetCount,
  providerCount,
  providersWithCatalog,
  publicNameCount,
  pythonVersions,
  siteData,
  toolCount,
  version,
} from '../siteData';

const RUN_LOOP = `flowchart LR
    T["Task"] --> A["Agent"]
    A --> M["Model"]
    M -->|"answer"| R["AgentResponse"]
    M -->|"tool call"| X["Tool"]
    X -->|"ToolResult"| A
    A -.->|"history"| Mem["Memory / session"]
`;

export default function Introduction() {
  return (
    <DocPage
      subtitle={`What effGen is, what it ships, and what changed in ${version}.`}
      icon={<Rocket size={48} />}
    >
      <p>
        effGen is a Python framework for building agents on small language models — and on
        cloud models, and on anything you already serve yourself. An agent is a model, a set of
        tools and a loop that runs between them; effGen supplies the loop, {toolCount} tools,{' '}
        {presetCount} ready-made configurations, and one way of saying which model you mean
        whether it runs on your GPU, behind an API key, or on a server on your own network.
      </p>

      <h2>The shortest thing that works</h2>
      <p>
        Two lines make an agent, and a third runs it. The preset supplies the tools and the
        system prompt; the model id says where the generation happens.
      </p>

      <CodeBlock
        language="bash"
        code={`pip install effgen
export OPENAI_API_KEY=...`}
      />

      <CodeBlock
        filename="hello.py"
        code={`from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
result = agent.run("What is 17% of 250?")

print(result)                  # printing the response prints the answer
print(result.success, result.tool_call_count)`}
      />

      <Terminal
        command="python hello.py"
        output={`42.5
True 1`}
        caption={`Run against effGen 1.0.0. The agent reached for its calculator once, which is the 1.`}
      />

      <Callout type="tip" title="No key yet?">
        <p>
          Swap the model id for a local one —{' '}
          <code>create_agent("math", "Qwen/Qwen2.5-1.5B-Instruct")</code> downloads the weights
          once and runs them on your own machine, with no account anywhere.{' '}
          <Link to="/local-models">Local models and engines</Link> covers the four engines that
          can run them.
        </p>
      </Callout>

      <h2>How a run is put together</h2>
      <p>
        Every path through effGen is the same shape. The agent sends the task and the tool
        schemas to the model; the model either answers or asks for a tool; a tool that is asked
        for returns a <code>ToolResult</code> that goes back into the conversation; the loop ends
        when the model answers or when <code>max_iterations</code> is reached. What comes back is
        an <code>AgentResponse</code> carrying the answer, the calls that were made, the sources,
        and what the run cost.
      </p>

      <MermaidDiagram
        chart={RUN_LOOP}
        title="One agent run"
        description="A task enters the agent, which calls the model. The model either returns an answer, which becomes the AgentResponse, or requests a tool, whose ToolResult returns to the agent for another pass. The agent's memory or session records the conversation."
      />

      <h2>What effGen gives you</h2>

      <FeatureList
        features={[
          {
            icon: '🧠',
            title: 'One name for any model',
            description: (
              <>
                {providerCount} provider adapters, {providersWithCatalog} of which ship a bundled
                catalog of {modelCount} models, plus the{' '}
                {siteData.models.local_engines.join(', ')} engines for weights on your own
                machine, plus any server that speaks the OpenAI protocol.
              </>
            ),
          },
          {
            icon: '🔧',
            title: `${toolCount} tools that already work`,
            description: (
              <>
                Search, documents, code execution, HTTP, mail, images, audio and more, across{' '}
                {Object.keys(siteData.tools.category_counts).length} categories — and a decorator
                that turns one of your own functions into another.
              </>
            ),
          },
          {
            icon: '🎛️',
            title: 'Control over the loop',
            description: (
              <>
                Middleware around every run, model call and tool call; a compaction strategy for
                what gets dropped when a conversation outgrows the window; sessions so one agent
                can serve many conversations; and checkpoints so a workflow that died can be
                resumed.
              </>
            ),
          },
          {
            icon: '📟',
            title: `A command line of ${commandCount} commands`,
            description: (
              <>
                Run a task, hold a conversation, edit a repository with{' '}
                <code>effgen code</code>, watch live traffic with <code>effgen top</code>, compare
                models, render an HTML report — each with <code>--json</code> for scripting.
              </>
            ),
          },
          {
            icon: '📊',
            title: 'Surfaces you can show someone',
            description: (
              <>
                A real-time dashboard, an in-browser playground, a model and pricing browser, and
                shareable HTML reports and run cards. They are served by effGen itself and fetch
                nothing from a third party.
              </>
            ),
          },
          {
            icon: '🛡️',
            title: 'The operational half',
            description: (
              <>
                An OpenAI-compatible server with auth, roles, audit and rate limits; Prometheus
                metrics, tracing, SLOs and alerting; guardrails, sandboxed execution and a spend
                cap.
              </>
            ),
          },
        ]}
      />

      <h2>What {version} changed</h2>
      <p>
        {version}, released on 1 October 2026, is about how a run ends, and how a tool call is
        read. It supports Python {pythonVersions.join(', ')} and exports {publicNameCount} public
        names; nothing was removed or renamed. A run that stops making progress is asked for its
        answer instead of going round to its iteration cap, and every run says how it ended in{' '}
        <code>response.termination</code>. A tool that keeps failing on its own side ends the run as{' '}
        <code>tool_failed</code>. A tool call written in a broken or unexpected shape is read and run
        by default. A model you serve yourself, or run on a local engine, is measured once for what
        it does with a tool, and <code>tool_calling_mode="auto"</code> follows the result. One agent
        can serve many overlapping conversations without mixing them. Small models given a search
        tool now cost more per run, because they are made to use it.
      </p>

      <p>
        <strong>Ten changes are visible to existing code.</strong> The ones most code meets first:
      </p>
      <ul>
        <li>
          <strong><code>max_turns_without_progress</code> defaults to <code>2</code></strong>: a run
          with no new tool result is asked for its answer, and a stuck run gets one closing request.{' '}
          <code>None</code> restores the earlier loop.
        </li>
        <li>
          <strong>A run whose tool keeps failing stops with <code>tool_failed</code></strong>, which
          raises <code>RunStoppedError</code> under the default <code>raise_on_error=True</code>.
        </li>
        <li>
          <strong><code>recover_lost_tool_calls</code> defaults to <code>True</code></strong>, so some
          runs make more tool calls; <code>False</code> restores the earlier reader.
        </li>
        <li>
          <strong>A served or local model is probed once</strong> at{' '}
          <code>tool_calling_mode="auto"</code>; <code>capability_probe=False</code> turns it off.
        </li>
      </ul>
      <p>
        1.2.0, released on 27 September 2026, gave every run a ledger of its calls, tokens, cost
        and time, made a model with no published price read as unpriced rather than free, kept a
        provider’s prompt cache warm, and added <code>effgen bench</code>.
      </p>
      <p>
        1.1.0, released on 14 September 2026, made a run keep its conversation as an{' '}
        <code>AgentThread</code> of typed steps, bounded a run by the prompt tokens it may send, made
        a saved run resume where it stopped, and put <code>stream()</code> and <code>run()</code> on
        one agent loop.
      </p>
      <p>
        1.0.1, released on 8 September 2026, made a run that stops without an answer report
        failure and raise <code>RunStoppedError</code>, made citation markers opt-in, and told the
        model what its tools are for on every tool-calling path.
      </p>
      <p>
        1.0.0, released on 14 August 2026, was the first stable release, and it carried three
        breaking changes: the Python floor is 3.11, <code>AgentConfig.raise_on_error</code>{' '}
        defaults to <code>True</code>, and a backend that was never reached raises{' '}
        <code>BackendUnreachableError</code> whatever that flag says.{' '}
        <Link to="/migration">Migrating to {version}</Link> carries every one with the code each
        one asks you to change, and <Link to="/releases">Releases</Link> has the full record.
      </p>

      <h2>Where to go next</h2>

      <QuickLinks
        links={[
          {
            icon: '⚡',
            title: 'Quick start',
            description: 'An agent that answers a question, from an empty shell to a result.',
            path: '/quickstart',
          },
          {
            icon: '📦',
            title: 'Installation',
            description: 'The extras matrix, GPU wheels and the Apple Silicon path.',
            path: '/installation',
          },
          {
            icon: '🤖',
            title: 'Agents',
            description: 'Every AgentConfig field and every AgentResponse field.',
            path: '/agents',
          },
          {
            icon: '🎯',
            title: 'Presets',
            description: `The ${presetCount} ready-made configurations and what each one turns on.`,
            path: '/presets',
          },
          {
            icon: '🔌',
            title: 'Any OpenAI-compatible server',
            description: 'Point an agent at vLLM, Ollama, TGI or a gateway with base_url.',
            path: '/openai-compatible',
          },
          {
            icon: '📖',
            title: 'API reference',
            description: `All ${publicNameCount} names the package exports.`,
            path: '/api-reference',
          },
        ]}
      />

      <SeeAlso paths={['/quickstart', '/installation', '/migration']} />
    </DocPage>
  );
}
