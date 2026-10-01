import { Users } from 'lucide-react';
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

export default function MultiAgent() {
  return (
    <DocPage
      subtitle="Several agents on one task: the orchestration patterns and how work is handed between them."
      icon={<Users size={48} />}
    >
      <p>
        A team is a named list of agents and a pattern that says how they are run. You build it once
        with <code>create_team</code> and hand it work with <code>assign_task</code>; what comes back
        is one <code>TeamResponse</code> carrying the final answer, every member's output, and the
        summed cost and tokens.
      </p>

      <h2>Two agents, in order</h2>

      <CodeBlock filename="team.py" code={`from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern

draft = Agent(AgentConfig(
    name="draft", model="gpt-5-nano", provider="openai", enable_sub_agents=False,
    system_prompt="Reply with exactly one short sentence. No preamble.",
))
translate = Agent(AgentConfig(
    name="translate", model="gpt-5-nano", provider="openai", enable_sub_agents=False,
    system_prompt="Translate whatever you are given into French. "
                  "Reply with the translation and nothing else.",
))

orchestrator = MultiAgentOrchestrator()
orchestrator.create_team("desk", [draft, translate], pattern=OrchestrationPattern.SEQUENTIAL)

result = orchestrator.assign_task("Describe what a tide is.", "desk")
print("success:", result.success)
for step in result.agent_responses:
    print(f"  {step['agent_name']:10} {step['output']}")
print("cost_usd   :", result.metadata["cost_usd"])
print("tokens_used:", result.metadata["tokens_used"])`} />

      <Terminal
        command="python team.py"
        output={`success: True
  draft      A tide is the regular rise and fall of sea level caused by the gravitational pull of the Moon and Sun on Earth's oceans.
  translate  Une marée est la montée et la descente régulières du niveau de la mer causée par l'attraction gravitationnelle de la Lune et du Soleil sur les océans de la Terre.
cost_usd   : 0.000293
tokens_used: 794`}
        caption={`Run against effGen 1.0.0. Two calls to gpt-5-nano, summed onto one result.`}
      />

      <Callout type="warning" title="A stage that decomposes is not a stage">
        <p>
          Team members are ordinary agents, and an ordinary agent has{' '}
          <code>enable_sub_agents=True</code> — so a stage can decide to split its own input into
          subtasks and synthesise them, which turns a two-word rewrite into an essay. For a pipeline
          stage that should do one narrow thing, set <code>enable_sub_agents=False</code>, as above.
        </p>
      </Callout>

      <h2>The six patterns</h2>

      <ApiTable
        headers={['Pattern', 'What it does', 'What each agent is given']}
        rows={[
          [
            <code>SEQUENTIAL</code>,
            'Runs the agents in order and stops at the first failure.',
            <>
              The first agent gets the task. Every agent after it gets the{' '}
              <strong>previous agent's output</strong> as its whole task — the original task is not
              repeated.
            </>,
          ],
          [
            <code>PIPELINE</code>,
            <>
              An alias for <code>SEQUENTIAL</code> — the two run identically.
            </>,
            'The same. Give each stage a role-specific system prompt.',
          ],
          [
            <code>PARALLEL</code>,
            'Runs every agent on the same task at once, then combines the results.',
            'The task, unchanged.',
          ],
          [
            <code>HIERARCHICAL</code>,
            'A manager splits the task into named subtasks and dispatches each to the worker of that name.',
            'One subtask, labelled with the worker who should do it.',
          ],
          [
            <code>COLLABORATIVE</code>,
            <>
              Agents discuss over rounds until they converge, up to{' '}
              <code>max_rounds</code>.
            </>,
            'The task, plus what the others have said so far.',
          ],
          [
            <code>COMPETITIVE</code>,
            <>
              Every agent solves the task and the best answer is selected, by{' '}
              <code>voting_strategy</code>.
            </>,
            'The task, unchanged.',
          ],
        ]}
        caption={
          <>
            <code>effgen.OrchestrationPattern</code>. The default is <code>SEQUENTIAL</code>.
          </>
        }
      />

      <Callout type="note" title="No pattern hides a failed member">
        <p>
          A member that fails is recorded with its own <code>success</code> and typed, redacted{' '}
          <code>error</code>, and the team's <code>success</code> is <code>False</code> with a{' '}
          <code>reason</code> in <code>metadata</code>. This holds for every one of the six — there
          is no pattern where a failure is averaged away into a <code>True</code>. An empty team is{' '}
          <code>False</code> too, rather than a vacuous success.
        </p>
      </Callout>

      <CodeBlock filename="failure.py" code={`from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern, load_model

good = Agent(AgentConfig(name="good", model="gpt-5-nano", provider="openai",
                         system_prompt="Answer in one word."))
broken = Agent(AgentConfig(
    name="broken",
    model=load_model("gpt-5-nano", provider="openai", api_key="sk-not-a-real-key"),
    raise_on_error=False,
))

orchestrator = MultiAgentOrchestrator()
orchestrator.create_team("pair", [good, broken], pattern=OrchestrationPattern.SEQUENTIAL)
result = orchestrator.assign_task("Name a colour.", "pair")

print("success :", result.success)
print("reason  :", result.metadata.get("reason"))
for step in result.agent_responses:
    print(f"  {step['agent_name']:7} success={step['success']}  error={str(step.get('error'))[:60]}")`} />

      <Terminal
        command="python failure.py"
        output={`success : False
reason  : sub_agent_failed
  good    success=True  error=None
  broken  success=False  error={'type': 'ModelAuthError', 'category': 'auth', 'provider': '`}
        caption={
          <>
            The partial work is kept, not discarded — <code>good</code>'s answer is still in{' '}
            <code>agent_responses</code>. The per-member <code>error</code> has the same seven keys
            as <Link to="/errors">any structured error</Link>.
          </>
        }
      />

      <h2>Routing one ticket to one specialist</h2>
      <p>
        A support flow usually wants one agent to answer, not all of them.{' '}
        <code>HIERARCHICAL</code> does that: the manager labels each subtask with the name of the
        worker who should handle it, and the subtask is dispatched to the agent of{' '}
        <em>that name</em> — never by position in the list.
      </p>

      <CodeBlock filename="triage.py" code={`from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern

manager = Agent(AgentConfig(name="manager", model="gpt-5-nano", provider="openai"))
team = [
    Agent(AgentConfig(name="billing", model="gpt-5-nano", provider="openai",
                      system_prompt="You handle refunds and billing only. Answer in one sentence.")),
    Agent(AgentConfig(name="tech", model="gpt-5-nano", provider="openai",
                      system_prompt="You handle login and app bugs only. Answer in one sentence.")),
]

orchestrator = MultiAgentOrchestrator()
orchestrator.create_team("support", team, pattern=OrchestrationPattern.HIERARCHICAL,
                         manager_agent=manager)
result = orchestrator.assign_task("I was charged twice for order ORD-7788.", "support")

for step in result.agent_responses:
    print(f"{step['agent_name']:9} <- {step.get('subtask', '')[:70]}")
print()
print("cost_usd   :", result.metadata.get("cost_usd"))
print("tokens_used:", result.metadata.get("tokens_used"))
print("execution  :", result.metadata.get("execution_id"))`} />

      <Terminal
        command="python triage.py"
        output={`billing   <- billing: Confirm duplicate charges for ORD-7788 (amounts, timestamps,
tech      <- tech: Retrieve and analyze payment gateway and system logs for ORD-778
billing   <- billing: Process refund for the duplicate charge, update customer/orde
tech      <- tech: Implement a fix to prevent future duplicates (improve idempotenc
billing   <- billing: Reconcile accounting entries and close the incident with a su

cost_usd   : 0.004279
tokens_used: 17371
execution  : 809685bed2a1`}
        caption="The manager decided this ticket needed five subtasks across both specialists, and each one went to the agent named in its label. How many it produces is the manager's call, and it is worth watching the cost when you first wire one up."
      />

      <p>
        If you would rather make the routing decision yourself, run a triage agent first and pick the
        specialist from its answer — one cheap call, and a decision you can log and test.
      </p>

      <CodeBlock filename="manual_triage.py" continues code={`ticket = "I was charged twice for order ORD-7788."

triage = Agent(AgentConfig(name="triage", model="gpt-5-nano", provider="openai",
    system_prompt="Reply with exactly one word: 'billing' or 'tech'."))
choice = triage.run(ticket).output.strip().lower()
specialist = {"billing": team[0], "tech": team[1]}.get(choice, team[0])
answer = specialist.run(ticket)
print(choice, "->", answer.output)`} />

      <Terminal
        command="python manual_triage.py"
        output={`billing -> I’ll investigate the duplicate charge on ORD-7788 and issue a refund for the duplicate payment; you should see the credit back to your original payment method within 5–7 business days.`}
        caption="Two calls instead of six, and the routing decision is one word you can log, assert on and change without a prompt rewrite."
      />

      <h2>Seeing the shape before you spend anything</h2>
      <p>
        <code>TeamConfig.to_dict()</code> serialises the topology — members, their models and tools,
        and the edges the pattern implies — without running anything. <code>diagram()</code> draws
        the same structure for a terminal, and takes a response to annotate it with what happened.
      </p>

      <CodeBlock filename="topology.py" code={`from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern
from effgen.tools.builtin import Calculator

lead = Agent(AgentConfig(name="lead", model="gpt-5-nano", provider="openai"))
researcher = Agent(AgentConfig(name="researcher", model="gpt-5-nano", provider="openai",
                               tools=[Calculator()]))
writer = Agent(AgentConfig(name="writer", model="gpt-5-nano", provider="openai"))

orchestrator = MultiAgentOrchestrator()
team = orchestrator.create_team(
    "desk", [researcher, writer],
    pattern=OrchestrationPattern.HIERARCHICAL, manager_agent=lead,
)

shape = team.to_dict()
print("pattern:", shape["pattern"])
print("manager:", shape["manager"]["name"])
for agent in shape["agents"]:
    print("  agent:", agent["name"], agent["model"], agent["tools"])
for edge in shape["edges"]:
    print("  edge :", edge["source"], "->", edge["target"], f"({edge['kind']})")
print()
print(team.diagram())`} />

      <Terminal
        command="python topology.py"
        output={`pattern: hierarchical
manager: lead
  agent: researcher gpt-5-nano ['calculator']
  agent: writer gpt-5-nano []
  edge : lead -> researcher (delegation)
  edge : lead -> writer (delegation)

Team: desk
hierarchical · 3 agent(s), 2 edge(s)
  ○ lead   manager  pending  gpt-5-nano
      ├─▶ researcher
      └─▶ writer
  ○ researcher   member  pending  gpt-5-nano
      ⚒ calculator
  ○ writer   member  pending  gpt-5-nano`}
        caption={
          <>
            <code>pending</code> because nothing has run. Pass a <code>TeamResponse</code> —{' '}
            <code>team.diagram(response)</code> — to see the same shape with each node's status.
            The structure is also on <code>response.metadata["topology"]</code>.
          </>
        }
      />

      <h2>The API</h2>

      <ParamTable
        nameLabel="Method"
        params={[
          {
            name: 'create_team(name, agents, pattern=SEQUENTIAL, manager_agent=None, **kwargs)',
            type: 'TeamConfig',
            description: (
              <>
                Registers the team under <code>name</code>. Extra keyword arguments go on the{' '}
                <code>TeamConfig</code>.
              </>
            ),
          },
          {
            name: 'assign_task(task, team, context=None)',
            type: 'TeamResponse',
            description: (
              <>
                <code>team</code> is a <code>TeamConfig</code> or the name it was created under.
              </>
            ),
          },
          {
            name: 'assign_task_async(task, team, context=None)',
            type: 'TeamResponse',
            description: 'The same run, awaited. Same result object.',
          },
          {
            name: 'register_agent(agent)',
            type: 'None',
            description: 'Adds an agent to the orchestrator without putting it in a team.',
          },
          { name: 'get_team(name)', type: 'TeamConfig | None', description: 'Look one up.' },
          { name: 'list_teams()', type: 'list[str]', description: 'Every registered team name.' },
          { name: 'remove_team(name)', type: 'None', description: 'Forget one.' },
          {
            name: 'cancel_workflow(team_name=None)',
            type: 'int',
            description:
              'Sets the cooperative cancellation flag, so a run in progress stops before launching its next agent. Returns how many were signalled.',
          },
        ]}
        caption={
          <>
            <code>MultiAgentOrchestrator</code>, from <code>effgen</code>.
          </>
        }
      />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'output', type: 'str', description: 'The team’s answer.' },
          {
            name: 'success',
            type: 'bool',
            default: 'True',
            description: 'False if any member failed, the team was empty, or the run was cancelled.',
          },
          {
            name: 'pattern',
            type: 'OrchestrationPattern',
            default: 'SEQUENTIAL',
            description: 'The pattern that ran.',
          },
          {
            name: 'agent_responses',
            type: 'list[dict]',
            default: '[]',
            description: (
              <>
                One entry per member run: <code>agent_name</code>, <code>output</code>,{' '}
                <code>success</code>, <code>tokens_used</code>, <code>cost_usd</code>,{' '}
                <code>execution_time</code>, and <code>error</code> when it failed.
              </>
            ),
          },
          {
            name: 'execution_time',
            type: 'float',
            default: '0.0',
            description: 'Wall clock for the whole team run.',
          },
          {
            name: 'rounds',
            type: 'int',
            default: '1',
            description: 'How many discussion rounds ran, for COLLABORATIVE.',
          },
          {
            name: 'selected_response',
            type: 'dict | None',
            default: 'None',
            description: 'The winning member, for COMPETITIVE.',
          },
          {
            name: 'consensus_score',
            type: 'float',
            default: '0.0',
            description: 'How closely the members agreed, for COLLABORATIVE.',
          },
          {
            name: 'metadata',
            type: 'dict',
            default: '{}',
            description: (
              <>
                <code>cost_usd</code>, <code>tokens_used</code>, <code>execution_id</code>,{' '}
                <code>topology</code>, and <code>reason</code> on a failure.
              </>
            ),
          },
        ]}
        caption={
          <>
            <code>TeamResponse</code>, from <code>effgen.core.orchestrator</code>.
          </>
        }
      />

      <Callout type="note" title="Cost is summed, not re-summed">
        <p>
          <code>metadata["cost_usd"]</code> is the whole team's spend. A member on a local engine, or
          on a model with no published rate, contributes nothing; when no member reported a cost at
          all it is <code>None</code> rather than <code>0.0</code>, so a free run and an unpriced one
          are distinguishable. <Link to="/cost">Cost and budgets</Link> has the rule.
        </p>
      </Callout>

      <h2>Grouping one execution's runs</h2>
      <p>
        Every team run issues an execution id, and each member's run record carries it along with the
        agent that delegated the work and the role it played. That is what turns five separate run
        records back into one team run — in the run history, in traces, and in the dashboard's
        topology panel.
      </p>

      <CodeBlock continues filename="execution.py" code={`from effgen.observability import run_log

response = orchestrator.assign_task("Draft the brief.", team)
members = run_log.read_runs(execution_id=response.metadata["execution_id"])
for row in members:
    print(row["agent"], row["role"], "←", row["parent_agent"], row["status"])`} />

      <p>
        Because the records are on disk, this works for teams run from a script or the command line,
        not only for work done inside the server process. See{' '}
        <Link to="/observability">Observability</Link>.
      </p>

      <h2>Sub-agents inside one agent</h2>
      <p>
        A team is several agents you built. Sub-agents are one agent deciding to split its own task
        up — no team, no orchestrator. It is on by default, and <code>mode=</code> on{' '}
        <code>run()</code> overrides the decision for one call.
      </p>

      <CodeBlock filename="modes.py" code={`from effgen.core.agent import AgentMode

agent.run(task, mode=AgentMode.AUTO)         # let the router decide (the default)
agent.run(task, mode=AgentMode.SINGLE)       # one agent, no decomposition
agent.run(task, mode=AgentMode.SUB_AGENTS)   # decompose, whatever the router thinks`} />

      <ApiTable
        headers={['Where', 'How to turn decomposition off']}
        rows={[
          [
            'For one call',
            <>
              <code>agent.run(task, mode=AgentMode.SINGLE)</code>
            </>,
          ],
          [
            'For an agent',
            <>
              <code>AgentConfig(enable_sub_agents=False)</code>
            </>,
          ],
          [
            'On the command line',
            <>
              <code>effgen run --mode single</code> or <code>--no-sub-agents</code>
            </>,
          ],
        ]}
      />

      <p>
        Decomposition costs a call to plan and one per subtask, so it earns its keep on a task with
        genuinely separable parts and loses on a short one. Larger models plan better; a 1B model
        asked to decompose usually should not be.
      </p>

      <h2>What a child run starts with, and what comes back</h2>
      <p>
        Since 1.1.0 a parent decides which of its steps a child it delegates to starts with, through
        a <code>ThreadProjection</code>. <code>WorkflowDAG</code>, <code>TeamConfig</code> and{' '}
        <code>SubAgentManager</code> each take <code>projection=</code>, and the default carries
        nothing — which is what every pattern did before.
      </p>

      <ApiTable
        headers={['Projection', 'What the child starts with']}
        rows={[
          [<code>NoParentContext()</code>, 'Only the question it was asked. The default.'],
          [<code>ParentTask()</code>, 'The job the parent was given, as one user turn.'],
          [<code>ParentAnswers()</code>, 'The parent’s task and what its finished children answered.'],
          [<code>LastCycles(n=2)</code>, 'The parent’s task and the last n complete cycles of its own work.'],
        ]}
      />

      <CodeBlock filename="projection.py" code={`from effgen import LastCycles
from effgen.core.workflow import WorkflowDAG

dag = WorkflowDAG("report", projection=LastCycles(1))`} />

      <p>
        Each result carries its runs’ conversations and serialises them into <code>to_dict()</code>:{' '}
        <code>WorkflowResult.thread</code>, <code>.threads</code>, <code>.node_thread()</code> and{' '}
        <code>.failed_nodes()</code>; <code>TeamResponse.thread</code> and{' '}
        <code>.agent_threads()</code>; <code>AgentResponse.sub_agent_threads()</code> for a
        decomposed run; and <code>SubAgentResult.thread</code>. <code>SubAgentManager</code> and{' '}
        <code>SubAgentResult</code> are importable from <code>effgen</code>.
      </p>

      <h2>Talking between agents</h2>
      <p>
        <code>MessageBus</code> is a publish–subscribe bus with topic wildcards and per-agent
        mailboxes. The orchestrator already publishes every task assignment and every result on it,
        so subscribing is how you watch a team without changing it.
      </p>

      <CodeBlock filename="bus.py" code={`from effgen.core.message_bus import AgentMessage, MessageBus, MessageType

bus = MessageBus()
bus.subscribe("results.*", lambda msg: print("got:", msg.payload))

bus.publish(AgentMessage(
    sender="math_agent", recipient="coordinator",
    type=MessageType.RESULT, payload={"answer": 42}, topic="results.math",
))

bus.send(AgentMessage(
    sender="coordinator", recipient="research_agent",
    type=MessageType.TASK_ASSIGNMENT, payload={"task": "Search for recent papers"},
))
messages = bus.receive("research_agent")
print("mailbox:", [(m.sender, m.type.value, m.payload) for m in messages])`} />

      <Terminal
        command="python bus.py"
        output={`got: {'answer': 42}
mailbox: [('coordinator', 'task_assignment', {'task': 'Search for recent papers'})]`}
        caption={
          <>
            <code>publish</code> goes to every subscriber whose topic pattern matches;{' '}
            <code>send</code> goes to one agent's mailbox and waits there until it is read. The five
            message types are <code>TASK_ASSIGNMENT</code>, <code>RESULT</code>,{' '}
            <code>STATUS_UPDATE</code>, <code>ERROR</code> and <code>HANDOFF</code>.
          </>
        }
      />

      <p>
        <code>SharedState</code> is the other half: a thread-safe namespaced key–value store with
        snapshots, which the orchestrator writes each member's result into.
      </p>

      <CodeBlock filename="state.py" code={`from effgen.core.shared_state import SharedState

state = SharedState()
state.set("research", "papers_found", 42, agent_id="researcher")
count = state.get("research", "papers_found")

snapshot_id = state.snapshot()
# … work that might need undoing …
state.rollback(snapshot_id)          # True when the snapshot was found`} />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'A later stage ignores its instructions and writes an essay',
            <>
              The stage decomposed into sub-agents and synthesised them. Its system prompt was
              competing with a task that read like an open question.
            </>,
            <>
              <code>enable_sub_agents=False</code> on every pipeline stage.
            </>,
          ],
          [
            'A stage answers the original question again',
            <>
              Under <code>SEQUENTIAL</code> a stage's task <em>is</em> the previous output, with no
              instruction attached.
            </>,
            'Put the transformation in the stage’s system prompt — that is the only place it can go.',
          ],
          [
            <><code>success=False</code> with <code>reason: sub_agent_failed</code></>,
            'One member failed. The rest of the work is still in agent_responses.',
            <>
              Read that member’s <code>error</code> dict — it names the type, the provider and
              whether it is retryable.
            </>,
          ],
          [
            <>
              <code>success=False</code> with <code>reason: empty_team</code>
            </>,
            <>
              The team has no agents, so nothing ran. <code>metadata["error"]</code> says{' '}
              <code>Team has no agents.</code>
            </>,
            'A team with no agents is a configuration error, and is reported rather than returning a vacuous success.',
          ],
          [
            'HIERARCHICAL cost far more than expected',
            'The manager chose to produce many subtasks, and each one is a run.',
            <>
              Watch <code>metadata["cost_usd"]</code>, and consider triaging with one cheap call
              yourself instead.
            </>,
          ],
          [
            'A subtask went to the wrong specialist',
            'Dispatch is by the name in the manager’s label. A worker whose name the manager never uses gets nothing.',
            'Give the workers names the manager will actually pick, and say what each one handles in its system prompt.',
          ],
          [
            <><code>cost_usd</code> is <code>None</code></>,
            'No member reported a cost — local engines, or models with no published rate.',
            <>
              This is not zero. <code>0.0</code> means a genuine free tier.
            </>,
          ],
          [
            'A team run will not stop',
            'Cancellation is cooperative: it takes effect between agents, not inside one.',
            <>
              <code>cancel_workflow(team_name)</code> stops the next agent from launching. A
              per-agent timeout is what bounds the current one.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/workflows', '/agents', '/checkpointing']} />
    </DocPage>
  );
}
