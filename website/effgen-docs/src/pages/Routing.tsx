import { GitBranch } from 'lucide-react';
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

const FLOW = `flowchart TD
    Ctx["RoutingContext:<br/>capabilities, budget, latency"] --> P1["Policy 1"]
    P1 -->|"a candidate"| Dec["RouterDecision:<br/>chosen + every rejection"]
    P1 -->|"none"| P2["Policy 2"]
    P2 -->|"a candidate"| Dec
    P2 -->|"none"| Err["NoCandidate…Error"]
    Dec --> Call["Call the provider"]
    Call -->|"retryable error"| Hop["Block it, re-route,<br/>emit a RouterEvent"]
    Hop --> Call
    Call -->|"hops exhausted"| Ex["AllCandidatesExhaustedError"]
    Call -->|"answer"| Done["Result"]
`;

export default function Routing() {
  return (
    <DocPage
      subtitle="The policy router: how a model is chosen, and what happens when the first choice is unusable."
      icon={<GitBranch size={48} />}
    >
      <p>
        The router picks a provider and model for a request from what the request needs — the
        capabilities, a cost ceiling, a latency budget — and from which credentials this machine
        actually holds. Every decision is explainable: it records the chosen pair and every
        candidate it rejected, with the reason.
      </p>

      <h2>One decision</h2>

      <CodeBlock
        filename="route.py"
        code={`import effgen.models                                     # registers every adapter
from effgen.models.capabilities import Capability
from effgen.models.router import ModelRouter, RoutingContext
from effgen.models.routing.cost import CostBasedPolicy

router = ModelRouter(policies=[CostBasedPolicy()])
decision = router.route(RoutingContext(
    prompt_tokens_estimate=2000,
    required_capabilities={Capability.chat, Capability.grounding},
))

print(decision.policy_name, "chose", f"{decision.chosen.provider}/{decision.chosen.model_id}")
print(f"estimated cost \${decision.score:.6f}, {len(decision.eliminated)} candidates eliminated")
for pair, reason in decision.eliminated[:3]:
    print(f"  {pair.provider}/{pair.model_id}: {reason}")`}
      />

      <Terminal
        command="python route.py"
        output={`cost_based chose gemini/gemini-2.5-flash-lite
estimated cost $0.000610, 416 candidates eliminated
  anthropic/claude-opus-4-7: no API key (ANTHROPIC_API_KEY)
  anthropic/claude-sonnet-4-6: no API key (ANTHROPIC_API_KEY)
  anthropic/claude-haiku-4-5-20251001: no API key (ANTHROPIC_API_KEY)`}
        caption={`Run against effGen 1.0.0 on a machine holding keys for eight of the ten adapters. Grounding narrows the field to the one provider that supports it; the eliminations name every model that was not chosen and why.`}
      />

      <MermaidDiagram
        chart={FLOW}
        title="How a request reaches a provider"
        description="A RoutingContext of capabilities, budget and latency is offered to each policy in order. The first policy that finds a candidate produces a RouterDecision naming the chosen pair and every rejection; if none does, a NoCandidate error is raised. The chosen provider is then called, and a retryable error blocks that provider, re-routes, emits a RouterEvent and retries until the hop limit, at which point AllCandidatesExhaustedError is raised."
      />

      <h2>RoutingContext</h2>

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'prompt_tokens_estimate',
            type: 'int',
            default: '0',
            description: 'Estimated input tokens, which is what a cost estimate is computed from.',
          },
          {
            name: 'user_budget_usd',
            type: 'float | None',
            default: 'None',
            description: 'The most this call may cost. None is unlimited.',
          },
          {
            name: 'latency_budget_ms',
            type: 'int | None',
            default: 'None',
            description: 'The slowest acceptable p50. None is unlimited.',
          },
          {
            name: 'required_capabilities',
            type: 'set[Capability]',
            default: 'set()',
            description: 'What the model must be able to do. A model missing any of these is eliminated by name.',
          },
        ]}
      />

      <h3>Capabilities</h3>

      <ApiTable
        headers={['Flag', 'What it means']}
        rows={[
          [<code>chat</code>, 'Text completion and chat.'],
          [<code>tools</code>, 'Function or tool calling.'],
          [<code>streaming</code>, 'Token streaming.'],
          [<code>vision</code>, 'Image inputs.'],
          [<code>audio_input</code>, 'Audio inputs.'],
          [<code>video_input</code>, 'Video inputs.'],
          [<code>grounding</code>, 'Web grounding — a search the provider runs itself.'],
          [<code>thinking</code>, 'Extended reasoning.'],
          [<code>json_schema</code>, 'Output constrained to a JSON schema.'],
        ]}
        caption={<>From <code>effgen.models.capabilities.Capability</code>.</>}
      />

      <h2>RouterDecision</h2>

      <ApiTable
        headers={['Field', 'What it is']}
        rows={[
          [<code>chosen</code>, <>The selected <code>ProviderModelPair</code>.</>],
          [<code>eliminated</code>, 'Every rejected candidate, paired with the reason it was rejected.'],
          [<code>policy_name</code>, 'Which policy produced the decision.'],
          [<code>score</code>, 'The policy’s own number — an estimated cost for CostBasedPolicy, a p50 in milliseconds for LatencyBasedPolicy.'],
        ]}
        caption="A decision is never a bare answer: eliminated is what lets you explain, in production, why a request went where it went."
      />

      <h2>The policies</h2>

      <ApiTable
        headers={['Policy', 'Chooses', 'Import']}
        rows={[
          [
            <code>FirstAvailablePolicy</code>,
            'The first provider, alphabetically, that has a key and supports every required capability.',
            <code>effgen.models.routing.first_available</code>,
          ],
          [
            <code>CostBasedPolicy</code>,
            'The cheapest pair that has a key, supports the capabilities, and fits the budget.',
            <code>effgen.models.routing.cost</code>,
          ],
          [
            <code>LatencyBasedPolicy</code>,
            'The fastest pair by observed p50 that satisfies the latency budget.',
            <code>effgen.models.routing.latency</code>,
          ],
        ]}
        caption={
          <>
            <code>PolicyBasedRouter</code> is exported for callers who prefer the explicit class
            name. Policies are tried in the order you give them: the first that finds a candidate
            wins, and the rest are fallbacks.
          </>
        }
      />

      <h3>Cost</h3>
      <p>
        The estimate is{' '}
        <code>
          input_per_1m × prompt_tokens ÷ 1,000,000 + output_per_1m × expected_output ÷ 1,000,000
        </code>
        . Free-tier providers rank ahead of paid ones at equal cost, and remaining ties break
        deterministically — provider priority, then published list cost, then provider name, then
        model id — so the same context routes the same way twice. A model marked as needing a
        dedicated endpoint is eliminated rather than chosen and then failed on.
      </p>

      <CodeBlock
        filename="budget.py"
        code={`import effgen.models
from effgen.models.capabilities import Capability
from effgen.models.errors import NoCandidateWithinBudgetError
from effgen.models.router import ModelRouter, RoutingContext
from effgen.models.routing.cost import CostBasedPolicy

router = ModelRouter(policies=[CostBasedPolicy()])
try:
    router.route(RoutingContext(
        prompt_tokens_estimate=2000,
        user_budget_usd=0.0000001,
        required_capabilities={Capability.chat, Capability.grounding},
    ))
except NoCandidateWithinBudgetError as e:
    print(type(e).__name__)
    print(f"  cheapest available: \${e.cheapest_cost_usd:.6f} from {e.cheapest_pair[0]}/{e.cheapest_pair[1]}")`}
      />

      <Terminal command="python budget.py" output={`NoCandidateWithinBudgetError
  cheapest available: $0.000610 from gemini/gemini-2.5-flash-lite`} />

      <Callout type="note" title="A free quota is not the same as a free price">
        <p>
          Gemini's Flash models have free quotas, but the free tier carries stricter rate limits and
          different data-use terms. <code>CostBasedPolicy</code> therefore budgets against the
          published paid prices, so a routing decision does not depend on a quota you may have
          already spent.
        </p>
      </Callout>

      <h3>Latency</h3>
      <p>
        <code>LatencyBasedPolicy</code> scores on observed p50 from <code>LatencyTracker</code>,
        which every <code>generate()</code> and <code>generate_stream()</code> feeds — the latter
        recording time-to-first-token separately, so a streaming application can route on TTFT.
        Where a provider has no measurement yet the policy uses a conservative seed, and any real
        measurement outranks every seed. On the first latency route with no history it fires tiny
        10-token probes in parallel to the eligible providers and scores against those instead.
      </p>

      <CodeBlock
        code={`from effgen.models.latency_tracker import LatencyTracker
from effgen.models.routing._probe import warm_up_providers
from effgen.models.capabilities import Capability

warm_up_providers(context_caps={Capability.chat})   # optional startup warm-up

tracker = LatencyTracker.get()
print(tracker.all_stats())`}
        caption="The probe runs candidates in a thread pool with a 15-second per-provider timeout and caches results for the session."
      />

      <p>
        When no measured candidate meets <code>latency_budget_ms</code>, the policy raises{' '}
        <code>NoCandidateWithinLatencyError</code> with the fastest p50 it did find. It is not
        converted into a first-available fallback, so an SLA miss is visible to the caller rather
        than silently absorbed.
      </p>

      <h3>Combining them</h3>

      <CodeBlock
        code={`from effgen.models.capabilities import Capability
from effgen.models.router import ModelRouter, RoutingContext
from effgen.models.routing.cost import CostBasedPolicy
from effgen.models.routing.latency import LatencyBasedPolicy

router = ModelRouter(policies=[CostBasedPolicy(), LatencyBasedPolicy()])
decision = router.route(RoutingContext(
    prompt_tokens_estimate=200,
    user_budget_usd=0.0001,
    latency_budget_ms=10_000,
    required_capabilities={Capability.chat},
))`}
      />

      <h2>Failover</h2>
      <p>
        <code>route_and_execute()</code> combines routing with automatic failover. When the chosen
        provider raises a retryable error, the router blocks that provider for the moment,
        re-routes, and retries. <code>failover_hops</code> is the number of failovers allowed{' '}
        <em>after</em> the first attempt, so <code>failover_hops=1</code> permits two provider
        attempts.
      </p>

      <ApiTable
        headers={['Error', 'Retryable', 'What the router does']}
        rows={[
          [<code>RateLimitExceeded</code>, 'yes', 'Fail over to the next provider.'],
          [<code>ProviderTransientError</code>, 'yes', 'Fail over — a 5xx from the provider.'],
          [<code>ModelTimeoutError</code>, 'yes', 'Fail over.'],
          [<code>ModelAuthError</code>, 'no', 'Raise immediately. Another provider will not fix a bad key.'],
          [<code>ModelRefusalError</code>, 'no', 'Raise immediately.'],
          [<code>InvalidRequestError</code>, 'no', 'Raise immediately.'],
        ]}
      />

      <CodeBlock
        code={`from effgen.models.router import ModelRouter, RoutingContext
from effgen.models.routing.cost import CostBasedPolicy
from effgen.models.routing.retry import RetryPolicy

router = ModelRouter(
    policies=[CostBasedPolicy()],
    failover_hops=2,
    retry_policy=RetryPolicy(max_retries=2, backoff_base=1.0, jitter=0.5),
)

events = []
router.subscribe(events.append)

result = router.route_and_execute(ctx, call_provider)
for event in events:
    print(event.as_dict())`}
        caption="RetryPolicy retries within one provider before the router fails over to another. Exponential backoff with additive jitter keeps a fleet from retrying in lockstep."
      />

      <ApiTable
        headers={['RouterEvent field', 'What it is']}
        rows={[
          [<code>from_provider</code> , 'The provider that failed.'],
          [<code>from_model</code>, 'The model that failed.'],
          [<code>to_provider</code>, 'The provider being tried next.'],
          [<code>to_model</code>, 'The model being tried next.'],
          [<code>reason</code>, <>Why — <code>rate_limited</code>, <code>transient_error_503</code>, <code>timeout</code>.</>],
          [<code>hop</code>, 'Which failover hop this is, counting from 1.'],
          [<code>exception</code>, 'The exception that triggered it.'],
        ]}
        caption={<>Every subscriber gets one per failover. <code>as_dict()</code> makes it loggable.</>}
      />

      <p>
        When every candidate inside the hop limit fails retryably,{' '}
        <code>AllCandidatesExhaustedError</code> carries <code>attempts</code>,{' '}
        <code>hop_limit</code> and <code>failures</code> — a list of{' '}
        <code>(provider, model, exception)</code> — so the whole attempt sequence can be reported
        rather than only the last failure.
      </p>

      <h2>Fallback on an agent</h2>
      <p>
        The router is the explicit, inspectable path. For an agent that should simply keep working
        when its first model does not, <code>AgentConfig</code> takes a list:
      </p>

      <CodeBlock
        filename="fallback.py"
        code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    models=["gemini:gemini-3.1-flash-lite"],   # tried in order when the first fails
    enable_fallback=True,
))
print(agent.run("Name one prime number between 20 and 30.").text)`}
      />

      <Terminal command="python fallback.py" output={`23`} />

      <p>
        <code>speculative_execution=True</code> goes further and runs two models at once, taking
        the first that succeeds — it trades money for latency, so it belongs on the calls where
        that trade is worth making.
      </p>

      <Callout type="note" title="Tool fallback is a different setting">
        <p>
          <code>fallback_chain</code> on <code>AgentConfig</code> maps a tool name to the tools to
          try when it fails. It has nothing to do with model routing, despite the similar name.
        </p>
      </Callout>

      <h2>Writing a policy</h2>

      <CodeBlock
        code={`from effgen.models.router import NoCandidateError, RouterDecision, RoutingPolicy

class PreferEuropean(RoutingPolicy):
    @property
    def name(self) -> str:
        return "prefer_european"

    def select(self, candidates, context):
        for pair in candidates:
            if pair.provider in {"mistral", "aleph_alpha"}:
                return RouterDecision(chosen=pair, eliminated=[], policy_name=self.name, score=0.0)
        raise NoCandidateError("no European provider is configured")`}
        caption="A policy receives the candidates that already passed the capability and credential filters, and either returns a decision or raises NoCandidateError so the next policy gets a turn."
      />

      <h2>Several workers, one rate limit</h2>
      <p>
        Rate limits are per account, not per process, so several workers sharing one key have to
        coordinate. <code>RateLimitCoordinator</code> backed by{' '}
        <code>SQLiteRateLimitStore</code> reserves a request slot before the call starts and
        reconciles the real token usage afterwards, with every worker pointing at the same database
        file.
      </p>

      <h2>When routing fails</h2>

      <ApiTable
        headers={['Error', 'When', 'What to do']}
        rows={[
          [
            <code>NoCandidateWithinBudgetError</code>,
            'Nothing that can do the job fits the budget.',
            <>
              <code>e.cheapest_cost_usd</code> and <code>e.cheapest_pair</code> say what the
              cheapest option actually was. Raise the budget or drop a capability.
            </>,
          ],
          [
            <code>NoCandidateWithinLatencyError</code>,
            'Nothing measured meets the latency budget.',
            'Raise the budget, or warm up the probe so the decision is made on real measurements rather than seeds.',
          ],
          [
            <code>NoCandidateError</code>,
            'A policy found nothing at all.',
            <>
              Usually a capability nothing keyed supports.{' '}
              <Link to="/providers">Providers</Link> shows what each catalog carries.
            </>,
          ],
          [
            <code>AllCandidatesExhaustedError</code>,
            'Every candidate inside the hop limit failed retryably.',
            <>
              <code>e.failures</code> lists each provider, model and exception. Raise{' '}
              <code>failover_hops</code>, or fix what is failing.
            </>,
          ],
          [
            'Every candidate eliminated for "no API key"',
            'The adapters are registered but this machine holds no credential for them.',
            <>
              <code>effgen doctor</code>, and <Link to="/configuration">Configuration</Link> for
              where keys are read from.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/providers', '/catalog', '/reliability']} />
    </DocPage>
  );
}
