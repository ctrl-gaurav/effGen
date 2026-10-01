import { SlidersHorizontal } from 'lucide-react';
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

export default function Generation() {
  return (
    <DocPage
      subtitle="Sampling, limits, reasoning effort and asking a model for JSON that matches a schema."
      icon={<SlidersHorizontal size={48} />}
    >
      <p>
        The same sampling controls exist in three places: pinned on an <code>AgentConfig</code> so
        they apply to every run, passed to one <code>run()</code> call, or given to a model
        directly as a <code>GenerationConfig</code>. Later wins over earlier. Everything on this
        page works that way.
      </p>

      <h2>Pinning them</h2>

      <CodeBlock
        filename="sampling.py"
        code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    temperature=0.0,
    top_p=1.0,
    seed=7,
    max_tokens=64,
))

print(agent.run("Name three primes, comma separated.").text.strip())
print(agent.run("Name three primes, comma separated.").text.strip())`}
      />

      <Terminal
        command="python sampling.py"
        output={`2, 3, 5
7, 11, 13`}
        caption={`Run against effGen 1.0.0. Two runs of one agent, same settings, different answers — a seed is not a promise that the same question is asked twice in the same conversation state.`}
      />

      <h2>The sampling surface</h2>

      <ParamTable
        nameLabel="Setting"
        params={[
          {
            name: 'temperature',
            type: 'float',
            default: '0.7',
            description: 'How much randomness there is in the choice of the next token. 0 is as close to deterministic as a provider offers.',
          },
          {
            name: 'top_p',
            type: 'float',
            default: '0.9',
            description: 'Nucleus sampling: consider only the most likely tokens whose probabilities sum to this.',
          },
          {
            name: 'top_k',
            type: 'int',
            default: '50',
            description: 'Consider only this many candidates. Providers that do not support it ignore it.',
          },
          {
            name: 'max_tokens',
            type: 'int | None',
            default: 'None',
            description: 'The output budget. None lets the model pick a size-aware default.',
          },
          {
            name: 'stop_sequences',
            type: 'list[str] | None',
            default: 'None',
            description: 'Strings that end generation when produced. GenerationConfig only.',
          },
          {
            name: 'seed',
            type: 'int | None',
            default: 'None',
            description: 'Sampling seed.',
          },
          {
            name: 'presence_penalty',
            type: 'float',
            default: '0.0',
            description: 'Penalises tokens already present anywhere in the text.',
          },
          {
            name: 'frequency_penalty',
            type: 'float',
            default: '0.0',
            description: 'Penalises tokens in proportion to how often they already appeared — the anti-repetition knob for long text.',
          },
          {
            name: 'repetition_penalty',
            type: 'float',
            default: '1.0',
            description: 'Multiplicative repeat penalty, used by the local and HuggingFace engines.',
          },
          {
            name: 'reasoning_effort',
            type: '"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | None',
            default: 'None',
            description: 'How much hidden reasoning a reasoning model should spend before answering. GenerationConfig only. Since 1.3.0 it is also sent to a model served behind base_url, whatever it is called, and by Groq for a model its catalog marks as reasoning; a server that refuses it is asked once more without it and remembered, and any other adapter that drops it says so once per model at WARNING level.',
          },
          {
            name: 'max_reasoning_tokens',
            type: 'int | None',
            default: 'None',
            description: 'A ceiling on that hidden reasoning.',
          },
          {
            name: 'thinking_budget',
            type: 'int | None',
            default: 'None',
            description: "Gemini's own thinking budget.",
          },
          {
            name: 'include_thoughts',
            type: 'bool',
            default: 'False',
            description: 'Return the reasoning stream alongside the answer, where the provider exposes it.',
          },
          {
            name: 'thinking',
            type: 'dict | None',
            default: 'None',
            description: "Anthropic's extended-thinking block, passed through.",
          },
          {
            name: 'grounding',
            type: 'bool',
            default: 'False',
            description: 'Let the provider run its own web search — Gemini only.',
          },
          {
            name: 'response_mime_type',
            type: 'str | None',
            default: 'None',
            description: 'Ask for a media type, such as application/json.',
          },
          {
            name: 'response_schema',
            type: 'dict | None',
            default: 'None',
            description: 'A schema the provider itself constrains the output to.',
          },
          {
            name: 'draft_model',
            type: 'Any',
            default: 'None',
            description: 'A smaller model to draft with, for speculative decoding where the engine supports it.',
          },
        ]}
        caption={
          <>
            <code>GenerationConfig</code> carries all of these. <code>AgentConfig</code> carries the
            sampling subset — <code>temperature</code>, <code>top_p</code>, <code>top_k</code>,{' '}
            <code>max_tokens</code>, <code>seed</code> and the three penalties — and a{' '}
            <code>run()</code> keyword of the same name overrides it for one call.
          </>
        }
      />

      <Callout type="warning" title="A seed is best-effort on OpenAI">
        <p>
          A fixed seed with <code>temperature=0</code> reproduces a generation exactly on Gemini,
          Groq and the local engines. OpenAI's chat models accept <code>seed</code> and usually
          reproduce output, but the same request can still return a different completion — OpenAI
          documents this as best-effort, not a guarantee, especially at the reasoning tier. Treat an
          OpenAI seed as "usually reproducible".
        </p>
      </Callout>

      <h2>Per call, on a model</h2>

      <CodeBlock
        filename="per_call.py"
        code={`from effgen import load_model
from effgen.models.base import GenerationConfig

model = load_model("gemini:gemini-3.1-flash-lite")
config = GenerationConfig(temperature=0.0, max_tokens=32, stop_sequences=["."])

result = model.generate("The capital of Japan is", config=config)
print(repr(result.text))
print(result.finish_reason, result.tokens_used)`}
      />

      <Terminal command="python per_call.py" output={`'The capital of Japan is **Tokyo**'
stop 7`} />

      <p>
        <code>finish_reason</code> is how you tell a finished answer from a truncated one:{' '}
        <code>stop</code> means the model ended, and a length reason means{' '}
        <code>max_tokens</code> ran out mid-sentence.
      </p>

      <h2>Reasoning models</h2>
      <p>
        A reasoning model spends part of its output budget on hidden reasoning before it emits a
        visible token. Two consequences follow, and both cost money if you miss them: the budget has
        to be generous, and a budget that runs out during the reasoning returns an empty answer that
        is still billed.
      </p>

      <CodeBlock
        filename="effort.py"
        code={`from effgen import load_model
from effgen.models.base import GenerationConfig

model = load_model("openai:gpt-5-nano")

for effort in ("low", "high"):
    result = model.generate(
        "In one sentence: why is the sky blue?",
        config=GenerationConfig(reasoning_effort=effort, max_tokens=2048),
    )
    print(f"{effort:5s} {result.tokens_used:5d} tokens  {result.text.strip()[:56]}")`}
      />

      <Terminal
        command="python effort.py"
        output={`low     428 tokens  Because Earth's atmosphere scatters sunlight and the sho
high    295 tokens  Because Rayleigh scattering in Earth's atmosphere scatte`}
        caption="Effort is a request, not an accounting rule — more effort can produce a shorter answer, and here it did."
      />

      <Callout type="tip" title="4096 is a workable floor">
        <p>
          <code>effgen quickstart --init</code> writes <code>max_tokens: 4096</code> when it detects
          a reasoning model, rather than the 512 it writes otherwise, and says why in the file. Use
          the same number when you set it by hand.
        </p>
      </Callout>

      <h2>Streaming</h2>

      <CodeBlock
        filename="stream.py"
        code={`from effgen import load_model

model = load_model("openai:gpt-5-nano")
for chunk in model.generate_stream("Count from 1 to 5, comma separated."):
    print(chunk, end="", flush=True)
print()`}
      />

      <Terminal command="python stream.py" output={`1, 2, 3, 4, 5`} />

      <p>
        On an agent, <code>AgentConfig(enable_streaming=True)</code> streams the run instead. The
        structured tool-call list is only available from a non-streaming call — see{' '}
        <Link to="/tool-calling">Tool calling</Link>.
      </p>

      <p>
        <code>agent.stream(task)</code> yields the model’s working as well as its answer, the
        working first. To show only the answer, ask for events and join the answer events — for a
        turn that answered, that reproduces <code>response.output</code> exactly:
      </p>

      <CodeBlock
        code={`events = list(agent.stream(task, include_events=True))
answer = "".join(e.text for e in events if e.kind == "answer")`}
      />

      <h2>Structured output</h2>
      <p>
        Two ways to insist on JSON. <code>output_model</code> takes a Pydantic class, validates the
        output against it and gives you the parsed instance; <code>output_schema</code> takes a JSON
        Schema dict or a model class and guarantees the output is valid JSON matching it. Both are{' '}
        <code>run()</code> keywords, and both have an <code>AgentConfig</code> equivalent that
        applies to every run.
      </p>

      <CodeBlock
        filename="structured.py"
        code={`from pydantic import BaseModel

from effgen import create_agent

class Sentiment(BaseModel):
    label: str
    confidence: float

agent = create_agent("minimal", "openai:gpt-5-nano")
r = agent.run(
    "Classify the sentiment of: 'I absolutely love this product!'",
    output_model=Sentiment,
    max_tokens=4096,
)

parsed = r.metadata["parsed"]
print(type(parsed).__name__, parsed.label, parsed.confidence)
print(r.output)`}
      />

      <Terminal command="python structured.py" output={`Sentiment Positive 0.99
{"label":"Positive","confidence":0.99}`} />

      <ApiTable
        headers={['Setting', 'Where', 'What it does']}
        rows={[
          [
            <code>output_model</code>,
            <>
              <code>run()</code>
            </>,
            <>
              A Pydantic class. The output is validated and the instance is stored in{' '}
              <code>metadata["parsed"]</code>.
            </>,
          ],
          [
            <code>output_schema</code>,
            <>
              <code>run()</code> and <code>AgentConfig</code>
            </>,
            <>
              A JSON Schema dict or a Pydantic class. Anything else raises{' '}
              <code>TypeError</code>.
            </>,
          ],
          [
            <code>output_format</code>,
            <code>AgentConfig</code>,
            <>
              A default format for every run: <code>"json"</code>, <code>"yaml"</code>,{' '}
              <code>"csv"</code> or <code>None</code>.
            </>,
          ],
          [
            <code>response_schema</code>,
            <code>GenerationConfig</code>,
            'A schema the provider itself constrains generation to, where it supports one.',
          ],
        ]}
      />

      <Callout type="warning" title="A nested schema needs a generous budget">
        <p>
          The model spends tokens on the JSON structure before it fills in any value, and a
          reasoning model spends more before that. When an extraction validates but every field is
          empty, <code>response.metadata["structured_output_empty"]</code> is set to{' '}
          <code>True</code> — read it rather than trusting a schema-shaped answer.
        </p>
      </Callout>

      <h3>Turning a model class into a schema</h3>

      <CodeBlock
        filename="schema.py"
        code={`import json

from pydantic import BaseModel

from effgen import to_openai_schema

class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

print(json.dumps(to_openai_schema(Person), indent=2))`}
      />

      <Terminal
        command="python schema.py"
        output={`{
  "properties": {
    "name": {
      "title": "Name",
      "type": "string"
    },
    "address": {
      "properties": {
        "street": {
          "title": "Street",
          "type": "string"
        },
        "city": {
          "title": "City",
          "type": "string"
        }
      },
      "required": [
        "street",
        "city"
      ],
      "title": "Address",
      "type": "object",
      "additionalProperties": false
    }
  },
  "required": [
    "name",
    "address"
  ],
  "title": "Person",
  "type": "object",
  "additionalProperties": false
}`}
        maxLines={22}
        caption="Nested models are inlined — no $ref survives — every object gets additionalProperties: false, and every object gets an explicit required array. That is what a provider’s strict mode needs."
      />

      <h2>Prompt caching</h2>
      <p>
        OpenAI caches prompt prefixes of 1,024 tokens or more automatically. Consecutive calls that
        share a prefix bill the cached part at a lower rate, and the saving is reported as{' '}
        <code>metadata["cached_input_tokens"]</code> alongside{' '}
        <code>metadata["prompt_tokens"]</code>. <code>AgentConfig.stable_system_prompt</code> —{' '}
        <code>True</code> by default — keeps the system prompt at a fixed position so the prefix
        stays cache-eligible across calls.
      </p>
      <p>
        Anthropic's caching is explicit rather than inferred, and is covered on{' '}
        <Link to="/providers">Providers</Link>.
      </p>

      <h2>When generation goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'An empty answer that was still billed',
            'A reasoning model spent the whole output budget on hidden reasoning.',
            <>
              Raise <code>max_tokens</code>. 4096 is a workable floor.
            </>,
          ],
          [
            <>
              <code>finish_reason</code> is a length reason
            </>,
            'The answer was cut off mid-sentence.',
            <>
              Raise <code>max_tokens</code>, or ask for a shorter answer.
            </>,
          ],
          [
            <code>ModelRefusalError</code>,
            'The provider returned a refusal instead of content.',
            <>
              <code>e.refusal_message</code> carries the text and <code>e.model_name</code> the
              model. Rephrase, or route to another model.
            </>,
          ],
          [
            <><code>structured_output_empty</code> is True</>,
            'The output validated against the schema but every field is empty.',
            'Raise the budget, or flatten the schema. Treat it as a failure rather than as data.',
          ],
          [
            <><code>TypeError</code> from output_schema</>,
            'It was given something that is neither a JSON Schema dict nor a Pydantic class.',
            <>
              Pass one of those two. <code>to_openai_schema()</code> converts a class if you need
              the dict.
            </>,
          ],
          [
            'The same seed gives a different answer',
            'The provider does not guarantee determinism — OpenAI documents its seed as best-effort.',
            <>
              Use Gemini, Groq or a local engine when reproducibility has to hold, and pair the seed
              with <code>temperature=0</code>.
            </>,
          ],
          [
            <><code>top_k</code> appears to do nothing</>,
            'Not every provider supports it.',
            'It is ignored rather than rejected, so the same config works everywhere.',
          ],
        ]}
      />

      <SeeAlso paths={['/agents', '/models', '/tool-calling']} />
    </DocPage>
  );
}
