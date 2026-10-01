import { Layers } from 'lucide-react';
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

export default function Middleware() {
  return (
    <DocPage
      subtitle="The hooks a run passes through, what each one can change, and the order they fire in."
      icon={<Layers size={48} />}
    >
      <p>
        effGen ships guardrails, observability, reliability and cost tracking as subsystems.
        Middleware is the general form of the same idea: somewhere to put behaviour effGen does not
        ship — an approval gate, a cache, a redaction pass, a per-run spend cap, a trace exporter of
        your own — without patching the loop.
      </p>

      <h2>Three points, six hooks</h2>

      <ApiTable
        headers={['Point', 'Fires', 'Hooks']}
        rows={[
          [
            'the run',
            <>
              Once per <code>run()</code>.
            </>,
            <>
              <code>before_run</code>, <code>after_run</code>
            </>,
          ],
          [
            'each model call',
            'Every generation, retries included.',
            <>
              <code>before_model_call</code>, <code>after_model_call</code>
            </>,
          ],
          [
            'each tool call',
            'Every dispatch.',
            <>
              <code>before_tool_call</code>, <code>after_tool_call</code>
            </>,
          ],
        ]}
      />

      <p>
        Subclass <code>AgentMiddleware</code> and override only what you need — every hook has a
        default that does nothing.
      </p>

      <CodeBlock filename="budget.py" code={`from effgen import Agent, AgentConfig
from effgen.core.middleware import AgentMiddleware
from effgen.tools.builtin import Calculator


class CallBudget(AgentMiddleware):
    """Let a tool run a fixed number of times, then answer for it."""

    def __init__(self, tool: str, limit: int):
        self.tool, self.limit, self.used = tool, limit, 0

    def before_tool_call(self, ctx):
        if ctx.tool_name != self.tool:
            return None
        if self.used >= self.limit:
            return f"Skipped: this run has already used its {self.limit} {self.tool} call(s)."
        self.used += 1
        return None


budget = CallBudget("calculator", limit=1)
agent = Agent(AgentConfig(
    model="gpt-5-nano", provider="openai",
    tools=[Calculator()], middleware=[budget], max_iterations=6,
))
response = agent.run(
    "Use the calculator tool for each step. First compute 81234 * 9317, "
    "then compute 44021 * 88913. Report both products."
)
print(response.output)
print("calculator calls allowed through:", budget.used)
print("calculator calls attempted      :", response.tool_calls.total)`} />

      <Terminal
        command="python budget.py"
        output={`- 81234 * 9317 = 756,857,178
- 44021 * 88913 = 3,914,039,173

Note: The calculator tool limit in this run prevented a second tool call; the second product was computed manually. If you'd like, I can re-run the calculation in a new session to show the tool result for the second multiplication.
calculator calls allowed through: 1
calculator calls attempted      : 2`}
        caption={`Run against effGen 1.0.0. The second call was intercepted: the model was handed the middleware's sentence in place of the tool's result, and answered around it.`}
      />

      <h2>Modifying and short-circuiting</h2>
      <p>
        A <code>before_</code> hook receives a context it can edit in place, and what it returns
        decides what happens next.
      </p>

      <ApiTable
        headers={['A before_ hook returns', 'What happens']}
        rows={[
          [
            <code>None</code>,
            'Carry on. The real call is made, with whatever edits the hook made to the context.',
          ],
          [
            'anything else',
            <>
              Short-circuit. The real call does not happen and the returned value is used as its
              result. The matching <code>after_</code> hook still runs.
            </>,
          ],
        ]}
      />

      <p>
        An <code>after_</code> hook receives the result and returns the one to use, so it can
        transform as well as observe.
      </p>

      <CodeBlock filename="cache.py" code={`from effgen.core.agent import AgentResponse
from effgen.core.middleware import AgentMiddleware, MiddlewareChain, RunContext


class AnswerFromCache(AgentMiddleware):
    def __init__(self, cache):
        self.cache = cache

    def before_run(self, ctx):
        hit = self.cache.get(ctx.task)
        return AgentResponse(output=hit, success=True) if hit else None

    def after_run(self, ctx, response):
        self.cache[ctx.task] = response.output
        return response


cache = {"What is the capital of France?": "Paris"}
chain = MiddlewareChain([AnswerFromCache(cache)])

hit = chain.before_run(RunContext(task="What is the capital of France?"))
print("cache hit  :", hit.output if hit else None)

miss = chain.before_run(RunContext(task="What is the capital of Peru?"))
print("cache miss :", miss)`} />

      <Terminal
        command="python cache.py"
        output={`cache hit  : Paris
cache miss : None`}
        caption={
          <>
            <code>MiddlewareChain</code> is what the agent runs your middleware through, and it is
            importable — so a hook can be exercised in a unit test without a model.
          </>
        }
      />

      <h2>Ordering</h2>
      <p>
        <code>before_</code> hooks run in the order the middleware were given;{' '}
        <code>after_</code> hooks run in reverse. They nest the way context managers do — the first
        one listed is the outermost.
      </p>

      <CodeBlock filename="order.py" code={`from effgen.core.middleware import AgentMiddleware, MiddlewareChain, RunContext


class Trace(AgentMiddleware):
    def __init__(self, label):
        self.label = label

    def before_run(self, ctx):
        print("before", self.label)
        return None

    def after_run(self, ctx, response):
        print("after ", self.label)
        return response


chain = MiddlewareChain([Trace("outer"), Trace("inner")])
ctx = RunContext(task="anything")
chain.before_run(ctx)
print("[the run]")
chain.after_run(ctx, None)`} />

      <Terminal
        command="python order.py"
        output={`before outer
before inner
[the run]
after  inner
after  outer`}
        caption="The first before_ hook to short-circuit wins, and the ones after it do not run — so the outermost middleware gets the first say."
      />

      <h2>Adding one for a single call</h2>
      <p>
        <code>run(..., middleware=[...])</code> <strong>adds</strong> to the configured middleware for
        that call only. It does not replace them and it does not affect the next call.
      </p>

      <CodeBlock filename="per_call.py" code={`agent.run(task, middleware=[Timing()])`} />

      <h2>Passing values between your own hooks</h2>
      <p>
        <code>ctx.metadata</code> on a <code>RunContext</code> is free space effGen never reads. Use
        it to get something from a <code>before_</code> hook to its <code>after_</code>.
      </p>

      <CodeBlock filename="timed.py" code={`import time

from effgen.core.agent import AgentResponse
from effgen.core.middleware import AgentMiddleware, MiddlewareChain, RunContext


class Timed(AgentMiddleware):
    def before_run(self, ctx):
        ctx.metadata["started"] = time.monotonic()
        return None

    def after_run(self, ctx, response):
        response.metadata["wall_clock_s"] = round(time.monotonic() - ctx.metadata["started"], 1)
        return response


chain = MiddlewareChain([Timed()])
ctx = RunContext(task="Summarise the release notes.")
chain.before_run(ctx)
time.sleep(0.5)
response = chain.after_run(ctx, AgentResponse(output="…", success=True))
print(response.metadata["wall_clock_s"], "seconds")`} />

      <Terminal command="python timed.py" output={`0.5 seconds`} />

      <h2>The three contexts</h2>

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'task',
            type: 'str',
            required: true,
            description: 'The task. Editable — rewrite it and the run uses the rewrite.',
          },
          { name: 'agent_name', type: 'str', default: "''", description: 'The agent’s name.' },
          {
            name: 'mode',
            type: 'Any',
            default: 'None',
            description: 'The AgentMode the run is executing under.',
          },
          {
            name: 'metadata',
            type: 'dict[str, Any]',
            default: '{}',
            description: 'Free space. effGen never reads it.',
          },
        ]}
        caption={<><code>RunContext</code> — the argument to <code>before_run</code> and <code>after_run</code>.</>}
      />

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'prompt',
            type: 'Any',
            required: true,
            description: 'What is about to be sent. Editable.',
          },
          { name: 'model_name', type: 'str', default: "''", description: 'The model being called.' },
          {
            name: 'kwargs',
            type: 'dict[str, Any]',
            default: '{}',
            description: 'The generation arguments. Editable — this is where temperature lives.',
          },
          {
            name: 'attempt',
            type: 'int',
            default: '1',
            description: 'Which attempt this is. Above one means a retry.',
          },
          {
            name: 'run',
            type: 'RunContext | None',
            default: 'None',
            description: 'The run this call belongs to, so its metadata is reachable from here.',
          },
        ]}
        caption={
          <>
            <code>ModelCallContext</code> — <code>before_model_call</code> /{' '}
            <code>after_model_call</code>.
          </>
        }
      />

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'tool_name',
            type: 'str',
            required: true,
            description: 'The tool about to be dispatched.',
          },
          {
            name: 'tool_input',
            type: 'str',
            default: "''",
            description: 'Its arguments. Editable.',
          },
          {
            name: 'run',
            type: 'RunContext | None',
            default: 'None',
            description: 'The run this call belongs to.',
          },
        ]}
        caption={
          <>
            <code>ToolCallContext</code> — <code>before_tool_call</code> /{' '}
            <code>after_tool_call</code>. It has no <code>metadata</code> of its own; use{' '}
            <code>ctx.run.metadata</code>.
          </>
        }
      />

      <ApiTable
        headers={['Hook', 'Signature', 'Returning a value means']}
        rows={[
          [
            <code>before_run</code>,
            <code>(ctx: RunContext) -&gt; AgentResponse | None</code>,
            'The run does not happen; that response is the result.',
          ],
          [
            <code>after_run</code>,
            <code>(ctx, response: AgentResponse) -&gt; AgentResponse</code>,
            'Return the response to use. Return the one you were given to observe only.',
          ],
          [
            <code>before_model_call</code>,
            <code>(ctx: ModelCallContext) -&gt; Any</code>,
            'The provider is not called; that value is the generation result.',
          ],
          [<code>after_model_call</code>, <code>(ctx, result: Any) -&gt; Any</code>, 'The result to use.'],
          [
            <code>before_tool_call</code>,
            <code>(ctx: ToolCallContext) -&gt; str | None</code>,
            'The tool does not run; that string is what the model is told it returned.',
          ],
          [<code>after_tool_call</code>, <code>(ctx, result: str) -&gt; str</code>, 'The tool output to use.'],
        ]}
      />

      <h2>What ships</h2>

      <ApiTable
        headers={['Middleware', 'What it does']}
        rows={[
          [
            <code>LoggingMiddleware(level=logging.INFO, logger_name=None)</code>,
            'Logs the run, every model call and every tool call. Also a worked example: it touches all six hooks and changes nothing.',
          ],
          [
            <code>ToolApprovalMiddleware(approve, tools=None, refusal=…)</code>,
            <>
              Calls <code>approve(tool_name, tool_input)</code> before the named tools run — every
              tool when <code>tools</code> is <code>None</code> — and reports the refusal string to
              the model instead of the tool's output.
            </>,
          ],
        ]}
        caption={<>Both in <code>effgen.core.middleware</code>.</>}
      />

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
        caption={
          <>
            The default <code>refusal</code> string, unchanged. <Link to="/human-loop">Human in the
            loop</Link> covers the other way of writing the same gate.
          </>
        }
      />

      <h2>Failure</h2>
      <p>
        A hook that raises is not caught. The exception reaches the caller like any other error in
        the run — which is exactly what lets an approval gate that refuses stop the run outright
        rather than letting it continue without the tool.
      </p>

      <Callout type="note" title="What it costs">
        <p>
          An agent with no middleware pays one boolean test at each of the three points. The hooks are
          safe to leave in place on the hot path.
        </p>
      </Callout>

      <Callout type="note" title="New in 1.0.0">
        <p>
          Middleware is new in 1.0.0. Nothing that existed before behaves differently — an agent
          that passes no <code>middleware</code> runs the same loop it always did.
        </p>
      </Callout>

      <h2>Middleware, guardrails or a gate?</h2>

      <ApiTable
        headers={['Reach for', 'When']}
        rows={[
          [
            <Link to="/guardrails">Guardrails</Link>,
            'The rule is about content — PII, injection, length, a blocked topic. It is a policy, it has a preset, and it reports a refusal in a shape the rest of the framework understands.',
          ],
          [
            'Middleware',
            'The behaviour is about the mechanics of the run — caching, budgets, timing, an exporter, a gate. It sees the task, the prompt, the tool arguments and the results.',
          ],
          [
            <Link to="/human-loop">approval_mode</Link>,
            'A person decides, and the same policy applies to every run the agent makes.',
          ],
        ]}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'A hook never fires',
            <>
              The method name is misspelled. <code>AgentMiddleware</code> has default
              implementations of all six, so an override that does not match one is simply a method
              nobody calls.
            </>,
            'The six names are on this page. Python will not warn you about this one.',
          ],
          [
            'A run returns nothing at all',
            <>
              A <code>before_</code> hook returned something falsy that is not <code>None</code> —
              an empty string short-circuits just as a response does.
            </>,
            <>
              Return <code>None</code> to carry on. Only <code>None</code> means carry on.
            </>,
          ],
          [
            'An after_ hook’s change is lost',
            <>
              It observed the result but returned nothing, so <code>None</code> became the result.
            </>,
            'Always return the response or the result, even when you only meant to look at it.',
          ],
          [
            <code>AttributeError: 'ToolCallContext' object has no attribute 'metadata'</code>,
            'Only RunContext carries metadata.',
            <>
              <code>ctx.run.metadata</code> — the tool context points back at its run.
            </>,
          ],
          [
            'A middleware’s counter is wrong across runs',
            'The instance is shared by every run of that agent, and nothing resets it.',
            <>
              Reset in <code>before_run</code>, or key the state by{' '}
              <code>ctx.run</code>.
            </>,
          ],
          [
            <><code>before_model_call</code> fires more often than expected</>,
            'It fires on every generation, retries included, and a tool-using run makes several.',
            <>
              <code>ctx.attempt</code> distinguishes a retry from a first attempt.
            </>,
          ],
          [
            'A per-call middleware replaced the configured ones',
            <>
              It does not — <code>run(middleware=[...])</code> adds.
            </>,
            'Both lists run, the configured ones first. There is no per-call way to remove one.',
          ],
        ]}
      />

      <SeeAlso paths={['/guardrails', '/human-loop', '/observability']} />
    </DocPage>
  );
}
