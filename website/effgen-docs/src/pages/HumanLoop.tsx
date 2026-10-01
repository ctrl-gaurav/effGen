import { UserCheck } from 'lucide-react';
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

export default function HumanLoop() {
  return (
    <DocPage
      subtitle="Pausing a run for approval before a tool call, and resuming it afterwards."
      icon={<UserCheck size={48} />}
    >
      <p>
        Some tool calls should not happen without a person saying so — a shell command, a payment, a
        delete. effGen puts the decision in a callback you supply: the loop stops, asks, and carries
        on with whatever the callback returned. The same mechanism collects free-text answers,
        choices and feedback.
      </p>

      <h2>Approving a tool call</h2>
      <p>
        Three fields on <code>AgentConfig</code> are the whole setup. The callback is handed the
        tool's name and its arguments, and returns a boolean.
      </p>

      <CodeBlock filename="approval.py" code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

asked = []


def approve(tool_name: str, tool_args: str) -> bool:
    """A real gate prompts a person. This one records and refuses."""
    asked.append((tool_name, tool_args))
    return False


agent = Agent(AgentConfig(
    model="gpt-5-nano", provider="openai",
    tools=[Calculator()],
    approval_mode="always",
    approval_callback=approve,
    approval_timeout=30.0,
))
response = agent.run("Use the calculator tool to compute 81234 * 9317.")
print("asked   :", asked)
print("answer  :", response.output)`} />

      <Terminal
        command="python approval.py"
        output={`asked   : [('calculator', '{"expression": "81234 * 9317", "operation": "calculate"}')]
answer  : Error executing tool 'calculator': execution denied by human approval (denied)`}
        caption={`Run against effGen 1.0.0. A refusal is reported back to the model as the tool's result, so the run continues and can answer without it.`}
      />

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'approval_mode',
            type: 'str',
            default: '"never"',
            description: (
              <>
                <code>"always"</code>, <code>"first_time"</code>, <code>"dangerous_only"</code> or{' '}
                <code>"never"</code>.
              </>
            ),
          },
          {
            name: 'approval_callback',
            type: 'Callable[[str, str], bool] | None',
            default: 'None',
            description: (
              <>
                Called with <code>(tool_name, tool_args)</code>. No callback means nothing is ever
                approved.
              </>
            ),
          },
          {
            name: 'approval_timeout',
            type: 'float',
            default: '0.0',
            description: (
              <>
                Seconds to wait for the callback. <code>0.0</code> waits indefinitely.
              </>
            ),
          },
          {
            name: 'clarification_callback',
            type: 'Callable[[str, list[str]], int] | None',
            default: 'None',
            description: (
              <>
                Called with a question and its options; returns the index chosen. See{' '}
                <a href="#asking-a-question-mid-run">below</a>.
              </>
            ),
          },
        ]}
        caption={
          <>
            On <code>AgentConfig</code>, from <code>effgen.core.agent</code>.
          </>
        }
      />

      <h2>The four modes</h2>

      <CodeBlock filename="modes.py" code={`from effgen.core.human_loop import ApprovalManager, ApprovalMode

for mode in ApprovalMode:
    manager = ApprovalManager(mode=mode, callback=lambda name, args: True)
    safe = manager.should_request_approval("calculator", requires_approval=False)
    flagged = manager.should_request_approval("bash", requires_approval=True)
    print(f"{mode.value:14} calculator={safe!s:5} bash={flagged}")`} />

      <Terminal
        command="python modes.py"
        output={`always         calculator=True  bash=True
first_time     calculator=True  bash=True
never          calculator=False bash=False
dangerous_only calculator=False bash=True`}
      />

      <ApiTable
        headers={['Mode', 'When the callback is called']}
        rows={[
          [<code>"always"</code>, 'Every tool call, every time.'],
          [
            <code>"first_time"</code>,
            <>
              The first call to each tool <strong>name</strong>. Once a tool is approved, later calls
              to it go through without asking — including with different arguments.
            </>,
          ],
          [
            <code>"dangerous_only"</code>,
            <>
              When the tool declares <code>requires_approval</code>, or when its name matches the
              built-in dangerous list.
            </>,
          ],
          [<code>"never"</code>, 'Never. This is the default.'],
        ]}
      />

      <Callout type="warning" title="first_time is per tool, not per call">
        <p>
          <code>ApprovalManager</code> remembers the tool's <em>name</em> once it is approved. A
          shell tool approved for <code>ls</code> will not ask again before{' '}
          <code>rm -rf</code>. Use <code>"always"</code> where the arguments are what matters.{' '}
          <code>manager.reset()</code> clears the memory.
        </p>
      </Callout>

      <CodeBlock filename="first_time.py" code={`from effgen.core.human_loop import ApprovalManager, ApprovalMode

manager = ApprovalManager(mode=ApprovalMode.FIRST_TIME, callback=lambda name, args: True)

for args in ("effgen release", "effgen changelog"):
    if manager.should_request_approval("web_search"):
        print("asking about", args, "->", manager.request_approval("web_search", args).value)
    else:
        print("not asking about", args, "— web_search was approved earlier")

manager.reset()
print("after reset:", manager.should_request_approval("web_search"))`} />

      <Terminal
        command="python first_time.py"
        output={`asking about effgen release -> approved
not asking about effgen changelog — web_search was approved earlier
after reset: True`}
      />

      <h2>What counts as dangerous</h2>
      <p>
        <code>dangerous_only</code> asks about a tool whose metadata sets{' '}
        <code>requires_approval</code>, and about any tool whose name contains one of fifteen
        keywords. The list is <code>effgen.core.human_loop.DANGEROUS_TOOL_KEYWORDS</code> and{' '}
        <code>is_tool_dangerous(name)</code> answers for one name.
      </p>

      <CodeBlock filename="dangerous.py" code={`from effgen.core.human_loop import DANGEROUS_TOOL_KEYWORDS, is_tool_dangerous

for name in ("calculator", "bash", "file_write", "wikipedia", "code_executor"):
    print(f"{name:14} dangerous={is_tool_dangerous(name)}")
print()
print(len(DANGEROUS_TOOL_KEYWORDS), "keywords:", ", ".join(sorted(DANGEROUS_TOOL_KEYWORDS)))`} />

      <Terminal
        command="python dangerous.py"
        output={`calculator     dangerous=False
bash           dangerous=True
file_write     dangerous=True
wikipedia      dangerous=False
code_executor  dangerous=True

15 keywords: bash, cancel, charge, code_executor, delete, exec, execute, file_delete, file_write, payment, purchase, refund, shell, system, transfer`}
        caption="The match is on the tool's name, so a tool you write called `delete_customer` is covered without doing anything."
      />

      <h2>When nobody answers</h2>
      <p>
        A gate that blocks forever is a gate that takes the service down with it.{' '}
        <code>timeout</code> runs the callback on a worker thread and applies{' '}
        <code>default_decision</code> when it does not return in time. The default default is{' '}
        <code>DENIED</code>.
      </p>

      <CodeBlock filename="timeout.py" code={`import time

from effgen.core.human_loop import ApprovalDecision, HumanApproval


def nobody_is_watching(tool_name: str, tool_args: str) -> bool:
    time.sleep(5)
    return True


request = HumanApproval(
    tool_name="file_delete",
    tool_args="/var/log/app.log",
    reason="the agent wants to delete a file",
    timeout=1.0,
    default_decision=ApprovalDecision.DENIED,
)
print(request.request(callback=nobody_is_watching))`} />

      <Terminal command="python timeout.py" output={`ApprovalDecision.TIMEOUT`} />

      <p>
        The three decisions are <code>APPROVED</code>, <code>DENIED</code> and{' '}
        <code>TIMEOUT</code> — a timeout is reported as itself rather than being folded into a
        denial, so a run that was never seen can be told apart from one that was refused.
      </p>

      <h2>A gate as middleware</h2>
      <p>
        <code>ToolApprovalMiddleware</code> is the same idea written as{' '}
        <Link to="/middleware">middleware</Link>: it takes a list of tool names to gate and a
        refusal message, and it composes with anything else on the run.
      </p>

      <CodeBlock filename="gate.py" code={`from effgen import Agent, AgentConfig
from effgen.core.middleware import ToolApprovalMiddleware
from effgen.tools.builtin import Calculator

denied = []


def ask(tool_name, tool_input):
    denied.append(tool_name)
    return False          # a real gate would prompt; this one refuses everything


agent = Agent(AgentConfig(
    model="gpt-5-nano", provider="openai",
    tools=[Calculator()],
    middleware=[ToolApprovalMiddleware(approve=ask, tools=["calculator"])],
))
response = agent.run("Use the calculator to work out 19 * 23.")
print("asked about:", denied)
print(response.output)`} />

      <Terminal
        command="python gate.py"
        output={`asked about: ['calculator']
This call was not approved, so the tool did not run.`}
        caption="The refusal string is the middleware's `refusal` argument, and it is what the model is told."
      />

      <ApiTable
        headers={['Reach for', 'When']}
        rows={[
          [
            <code>approval_mode</code>,
            'The gate is a property of the agent, and the same policy applies to every run it makes.',
          ],
          [
            <code>ToolApprovalMiddleware</code>,
            'The gate belongs to one caller or one request — a server that approves for some users and not others — or it needs to sit beside other middleware in a known order.',
          ],
          [
            <code>ToolPermissionGuardrail</code>,
            <>
              The answer is a policy, not a person: a fixed allow or deny list. See{' '}
              <Link to="/guardrails">Guardrails</Link>.
            </>,
          ],
        ]}
      />

      <h2 id="asking-a-question-mid-run">Asking a question mid-run</h2>
      <p>
        A <code>ClarificationRequest</code> is a question with options and a default. Answer it with
        a choice callback, a free-text callback, or both — it uses the choice callback when it has
        options and falls back to free text when it does not.
      </p>

      <CodeBlock filename="clarify.py" code={`from effgen.core.clarification import ClarificationRequest

request = ClarificationRequest(
    question="Which environment should I deploy to?",
    options=["staging", "production"],
    default=0,
)
print(request.ask(choice_callback=lambda q, opts: 1))`} />

      <Terminal command="python clarify.py" output={`production`} />

      <p>
        <code>ClarificationDetector</code> decides when a task needs one:{' '}
        <code>detect_ambiguity(query, available_tools)</code> returns a request or{' '}
        <code>None</code>, and <code>request_clarification(...)</code> asks it and returns the
        answer. Wiring it into an agent is the <code>clarification_callback</code> field.
      </p>

      <h2>The three primitives</h2>

      <ApiTable
        headers={['Class', 'Constructor', 'request(callback=…) returns']}
        rows={[
          [
            <code>HumanApproval</code>,
            <>
              <code>tool_name</code>, <code>tool_args</code>, <code>reason=""</code>,{' '}
              <code>timeout=0.0</code>, <code>default_decision=DENIED</code>
            </>,
            <code>ApprovalDecision</code>,
          ],
          [
            <code>HumanInput</code>,
            <>
              <code>prompt</code>, <code>timeout=0.0</code>, <code>default=""</code>
            </>,
            <code>str</code>,
          ],
          [
            <code>HumanChoice</code>,
            <>
              <code>prompt</code>, <code>options</code>, <code>default=0</code>,{' '}
              <code>timeout=0.0</code>
            </>,
            <>
              <code>int</code> — the index chosen
            </>,
          ],
        ]}
        caption={
          <>
            All three are in <code>effgen.core.human_loop</code>, alongside{' '}
            <code>cli_approval_callback</code>, <code>cli_input_callback</code> and{' '}
            <code>cli_choice_callback</code>, which prompt on a terminal.
          </>
        }
      />

      <h2>Collecting feedback after the fact</h2>
      <p>
        <code>FeedbackCollector</code> records thumbs, ratings and comments against a response id
        and writes them out as JSONL — one line per entry, for offline analysis or a training set.
      </p>

      <CodeBlock filename="feedback.py" code={`from effgen.core.feedback import FeedbackCollector

collector = FeedbackCollector(agent_name="support")
collector.thumbs(response_id="r-1", thumbs_up=True, query="Where is my refund?")
collector.thumbs(response_id="r-2", thumbs_up=False, query="Why was I charged twice?")
collector.rate(response_id="r-2", rating=2, query="Why was I charged twice?")
collector.comment(response_id="r-2", text="quoted the wrong policy")

print(collector.summary())
print("written:", collector.export_jsonl("./feedback.jsonl"), "entries")`} />

      <Terminal
        command="python feedback.py"
        output={`{'total': 4, 'thumbs_up': 1, 'thumbs_down': 1, 'average_rating': 2.0, 'total_ratings': 1, 'total_comments': 1}
written: 4 entries`}
        caption={
          <>
            Use a stable <code>response_id</code> — <code>response.metadata["run_id"]</code> is one,
            and it is the same id <Link to="/checkpointing">run history</Link> keys on.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Every tool call is denied and nothing prompts',
            <>
              <code>approval_mode</code> is set but <code>approval_callback</code> is not. With no
              callback there is nothing to approve with, and the default decision applies.
            </>,
            'Pass a callback, or set the mode back to "never".',
          ],
          [
            <code>ApprovalDecision.TIMEOUT</code>,
            <>
              The callback did not return within <code>approval_timeout</code>.
            </>,
            <>
              Raise the timeout, or set <code>default_decision</code> deliberately —{' '}
              <code>DENIED</code> is the safe one and is the default.
            </>,
          ],
          [
            'The run hangs and never finishes',
            <>
              <code>approval_timeout=0.0</code> waits indefinitely, and the callback is blocking on
              a person who is not there.
            </>,
            'Always set a timeout on an unattended service.',
          ],
          [
            <code>Error executing tool '…': execution denied by human approval</code>,
            'Working as intended: the refusal is reported to the model as the tool result.',
            'The model usually answers without the tool. If it should stop instead, raise from the callback — an exception in a middleware hook is not caught.',
          ],
          [
            'A dangerous tool was never gated',
            <>
              <code>dangerous_only</code> matched neither the tool’s{' '}
              <code>requires_approval</code> flag nor a keyword in its name.
            </>,
            <>
              <code>is_tool_dangerous("your_tool")</code> tells you which. Use <code>"always"</code>{' '}
              if the list is not the right test.
            </>,
          ],
          [
            'Feedback entries all overwrite each other',
            <>
              The same <code>response_id</code> is being reused across turns.
            </>,
            <>
              <code>get_by_response(id)</code> returns every entry for one id — that is the shape.
              Use a per-run id if you want them separate.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/guardrails', '/middleware', '/agents']} />
    </DocPage>
  );
}
