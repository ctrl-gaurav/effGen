import { ChefHat } from 'lucide-react';
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
import { siteData } from '../siteData';

export default function Cookbook() {
  return (
    <DocPage
      subtitle="Seven short recipes, each one task: five that read pictures, sound and video, and two about running a lot of work and testing it."
      icon={<ChefHat size={48} />}
    >
      <p>
        Every recipe below is a whole file. Copy it, run it, change the input. The five multimodal
        ones use <code>gemini-3.1-flash-lite</code>, which reads images, audio and video on the free
        tier; the last two need no provider key at all.
      </p>

      <ApiTable
        headers={['Recipe', 'What it does', 'Needs']}
        rows={[
          [<a href="#image-questions">Image questions</a>, 'Ask about a picture in words.', 'A vision model'],
          [<a href="#audio">Transcribe and reason</a>, 'Turn a recording into text, then answer about it.', 'An audio model'],
          [<a href="#video">Summarise a clip</a>, 'Sample frames and describe what happens.', 'ffmpeg, a vision model'],
          [<a href="#ocr">Document to fields</a>, 'Read a scan, then pull structured values out of it.', 'A vision model'],
          [<a href="#chart">Read a chart</a>, 'Get the numbers back off a bar chart.', 'A vision model'],
          [<a href="#batches">Run a lot of queries</a>, 'One agent, many questions, bounded concurrency.', 'Any model'],
          [<a href="#testing">Test an agent</a>, 'Assertions that need no network and no key.', 'Nothing'],
        ]}
      />

      <Callout type="note" title="What builds a part">
        <p>
          <code>image_from()</code>, <code>audio_from()</code> and <code>video_from()</code> each
          take a path, a URL, raw bytes or an in-memory object, and return a part you hand to{' '}
          <code>run(inputs=[…])</code>. <Link to="/multimodal">Multimodal</Link> has the full
          message schema and the provider support matrix; these are the recipes.
        </p>
      </Callout>

      <h2 id="image-questions">Image questions</h2>

      <p>
        Two ways to ask, and the difference matters. A direct model call sends the picture and the
        question and gets a sentence back. An agent does the same but can also reach for the{' '}
        <code>ocr</code>, <code>image_caption</code> and <code>image_info</code> tools when the
        question needs them.
      </p>

      <CodeBlock
        filename="image_qa.py"
        code={`from PIL import Image, ImageDraw

from effgen import Agent, AgentConfig, image_from, load_model
from effgen.core.messages import Message, Role, TextPart

# A picture with something in it worth asking about.
picture = Image.new("RGB", (320, 200), "white")
draw = ImageDraw.Draw(picture)
draw.rectangle([40, 60, 120, 170], fill="#c0392b")
draw.ellipse([170, 50, 280, 160], fill="#2980b9")
picture.save("/tmp/shapes.png")

part = image_from(picture)          # a PIL image, a path, a URL or raw bytes
print("mime:", part.mime, "| bytes:", len(part.image))

# A: the model on its own, with no agent and no tools.
model = load_model("gemini-3.1-flash-lite", provider="gemini")
answer = model.generate([
    Message(role=Role.USER, content=[part, TextPart(text="Name the two shapes and their colours, in one sentence.")])
])
print("\\n[direct]", answer.text.strip())

# B: an agent, which can also reach for the ocr and image_info tools.
agent = Agent(AgentConfig(name="vision", model="gemini-3.1-flash-lite", provider="gemini"))
response = agent.run("Which of the two shapes is larger, and by roughly how much?", inputs=[part])
print("\\n[agent]", response.output.strip())
print("[agent] success:", response.success, "| tools:", response.tool_calls.names)`}
      />

      <Terminal
        command="python image_qa.py"
        output={`mime: image/png | bytes: 1181

[direct] There is a red rectangle and a blue circle.

[agent] The image shows a red rectangle on the left and a blue circle on the right against a white background.

Objectively, the red rectangle is larger than the blue circle. By estimating the pixel area:
* The red rectangle is approximately 150 pixels wide by 215 pixels tall, resulting in an area of roughly 32,250 square pixels.
* The blue circle has a diameter of approximately 185 pixels, resulting in an area of roughly 26,880 square pixels.

The red rectangle is larger by approximately 20% in terms of total surface area.
[agent] success: True | tools: []`}
        maxLines={20}
        caption={`Run against effGen 1.0.0.`}
      />

      <Callout type="tip" title="An empty tool list is not a failure">
        <p>
          <code>tools: []</code> on that run is correct. The model reads images itself, so it
          answered from the picture rather than calling <code>image_caption</code> to describe it
          first. The vision tools are there for a model that cannot — and for the questions a
          caption cannot answer, like the pixel dimensions <code>image_info</code> reports. Note
          also that the estimate above is the model's arithmetic on shapes it measured by eye: the
          two areas are roughly right, the "20%" is not.
        </p>
      </Callout>

      <h2 id="audio">Transcribe a recording, then reason about it</h2>

      <p>
        One run, two jobs: the transcript, and something said about it. Passing an{' '}
        <code>AudioPart</code> to a model with native audio is a single request — there is no
        separate transcription call to wait for.
      </p>

      <CodeBlock
        filename="audio_qa.py"
        code={`from effgen import Agent, AgentConfig, audio_from

# audio_from takes a path, raw bytes or a URL.
clip = audio_from("https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg")
print("mime:", clip.mime, "| bytes:", f"{len(clip.audio):,}", "| duration:", clip.duration_s)

agent = Agent(AgentConfig(name="listener", model="gemini-3.1-flash-lite", provider="gemini"))
response = agent.run(
    "Transcribe this recording, then say in one sentence what it is about "
    "and whether the tone is positive, negative or neutral.",
    inputs=[clip],
)
print("\\n", response.output.strip(), sep="")
print("\\nsuccess:", response.success, "| tools:", response.tool_calls.names)`}
      />

      <Terminal
        command="python audio_qa.py"
        output={`mime: audio/ogg | bytes: 104,793 | duration: None

This is an example sound file in Ogg Vorbis format from Wikipedia, the free encyclopedia.

This recording explains what the audio file is in a neutral tone.

success: True | tools: []`}
        caption="A public-domain Ogg Vorbis clip, fetched by audio_from() from its URL. mp3, wav, flac, ogg, m4a and webm are all accepted."
      />

      <p>
        <code>duration_s</code> is <code>None</code> here: it is filled in when the container
        carries readable metadata and left empty rather than guessed at when it does not. Nothing
        downstream depends on it.
      </p>

      <h2 id="video">Summarise a clip</h2>

      <p>
        <code>video_from()</code> shells out to <code>ffmpeg</code> to pull frames at the rate you
        ask for, and caps them at <code>max_frames</code>. Gemini takes the frames in order and can
        reason across them; a vision model without native video sees the same frames as a sequence
        of images.
      </p>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'source',
            type: 'str | Path | bytes',
            required: true,
            description: 'The clip: a path, raw bytes or a URL.',
          },
          {
            name: 'fps',
            type: 'float',
            default: '1.0',
            description: 'Frames sampled per second of video. 0.25 gives one frame every four seconds.',
          },
          {
            name: 'mime',
            type: 'str',
            default: "'image/jpeg'",
            description: 'The encoding the sampled frames are handed over in.',
          },
          {
            name: 'max_frames',
            type: 'int',
            default: '16',
            description: 'Hard cap, whatever the clip length and fps would otherwise produce.',
          },
        ]}
        caption={
          <>
            From <code>inspect.signature(video_from)</code>.
          </>
        }
      />

      <CodeBlock
        filename="video_summary.py"
        code={`from effgen import Agent, AgentConfig, video_from

clip = video_from("sample_video.mp4", fps=1, max_frames=8)
print("frames:", len(clip.frames), "| fps:", clip.fps, "| mime:", clip.mime)

agent = Agent(AgentConfig(name="watcher", model="gemini-3.1-flash-lite", provider="gemini"))
response = agent.run(
    "Describe what happens in this clip: the subject, and any motion you can see between frames.",
    inputs=[clip],
)
print("\\n", response.output.strip(), sep="")`}
      />

      <Terminal
        command="python video_summary.py"
        output={`frames: 2 | fps: 1 | mime: image/jpeg

The clip displays a series of abstract, solid-colored vertical bars that shift and change color with each frame. There is no central subject or physical movement; instead, the visuals consist of rapid, rhythmic color transitions.`}
        caption="Run against the framework's own test clip — a two-second moving test pattern, which is why two frames came back at 1 fps and why there is no subject to name."
      />

      <Callout type="warning" title="ffmpeg has to be on PATH">
        <p>
          Frame sampling is a subprocess. Without <code>ffmpeg</code> installed,{' '}
          <code>video_from()</code> raises with the install command for your platform rather than
          returning an empty clip. <code>apt-get install ffmpeg</code> on Debian and Ubuntu,{' '}
          <code>brew install ffmpeg</code> on macOS.
        </p>
      </Callout>

      <h2 id="ocr">A document, then its fields</h2>

      <p>
        Two steps, because they are two different jobs. First get the text off the page. Then hand
        that text to a <Link to="/prompts">prompt template</Link> that knows what to pull out of it
        — here <code>legal.contract_summarize.v1</code>, one of the {siteData.prompts.library} in
        the library.
      </p>

      <CodeBlock
        filename="document_fields.py"
        code={`import json
import re

from PIL import Image, ImageDraw, ImageFont

from effgen import Agent, AgentConfig, image_from, load_model
from effgen.core.messages import Message, Role, TextPart
from effgen.prompts.library import registry

# Step 0 — a document to read. Any scan or photo works; this one is drawn so
# the recipe runs with no file to hand.
lines = [
    "SERVICE AGREEMENT",
    "",
    "Parties: Acme Corp. (Provider) and Beta LLC (Client).",
    "Effective Date: 2026-06-01",
    "Termination: either party may terminate with 30 days written notice.",
    "Payment Terms: $5,000 per month, due on the 1st.",
    "Governing Law: State of California.",
]
page = Image.new("RGB", (720, 300), "white")
draw = ImageDraw.Draw(page)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except OSError:
    font = ImageFont.load_default()
for i, line in enumerate(lines):
    draw.text((24, 24 + i * 34), line, fill="black", font=font)
page.save("/tmp/agreement.png")

# Step 1 — pull the text out of the image.
agent = Agent(AgentConfig(name="reader", model="gemini-3.1-flash-lite", provider="gemini"))
ocr = agent.run(
    "Use the ocr tool to extract every line of text from this document. Return the text only.",
    inputs=[image_from("/tmp/agreement.png")],
)
print("=== extracted text ===")
print(ocr.output.strip())
print("\\ntools:", ocr.tool_calls.names)

# Step 2 — hand the text to a template from the prompt library.
template = registry.get("legal.contract_summarize.v1")
prompt = template.render(contract_text=ocr.output)

model = load_model("gemini-3.1-flash-lite", provider="gemini")
summary = model.generate([Message(role=Role.USER, content=[TextPart(text=prompt)])])

print("\\n=== legal.contract_summarize.v1 ===")
print(summary.text.strip())

block = re.search(r"\\{[\\s\\S]+\\}", summary.text)
if block:
    print("\\nparsed keys:", sorted(json.loads(block.group())))`}
      />

      <Terminal
        command="python document_fields.py"
        output={`=== extracted text ===
The image contains the following text:

SERVICE AGREEMENT

Parties: Acme Corp. (Provider) and Beta LLC (Client).
Effective Date: 2026-06-01
Termination: either party may terminate with 30 days written notice.
Payment Terms: $5,000 per month, due on the 1st.
Governing Law: State of California.

tools: []

=== legal.contract_summarize.v1 ===
{
  "parties": [
    "Acme Corp.",
    "Beta LLC"
  ],
  "term": "Effective Date: 2026-06-01",
  "obligations": [
    "Provider (Acme Corp.) to provide services",
    "Client (Beta LLC) to pay $5,000 per month due on the 1st"
  ],
  "termination": "Either party may terminate the agreement with 30 days written notice.",
  "risks": [
    "Governing law is set to the State of California, which may impact legal proceedings or venue for disputes."
  ]
}

parsed keys: ['obligations', 'parties', 'risks', 'term', 'termination']`}
        maxLines={24}
      />

      <p>
        Two things worth noticing in that output. The model did the reading itself rather than
        calling the <code>ocr</code> tool — same as the image recipe, and for the same reason. And
        the extracted text keeps the model's own preamble line, which is why step two renders the
        template rather than parsing the string: the template's job is to find the fields in
        whatever came back.
      </p>

      <h2 id="chart">Read a chart</h2>

      <p>
        No chart-parsing library. The numbers are printed on the bars, and a vision model reads
        them the way you would — which also means it is worth asking the same chart more than one
        question and checking the answers against each other.
      </p>

      <CodeBlock
        filename="read_chart.py"
        code={`import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from effgen import Agent, AgentConfig
from effgen.core.messages import ImagePart

regions = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
units = [42, 87, 31, 65, 53]

figure, axes = plt.subplots(figsize=(7, 4))
bars = axes.bar(regions, units, color="#00a86b")
axes.set_title("Units sold by region")
axes.set_ylabel("Units")
axes.set_ylim(0, 100)
for bar, value in zip(bars, units):
    axes.text(bar.get_x() + bar.get_width() / 2, value + 1.5, str(value), ha="center")

buffer = io.BytesIO()
plt.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
plt.close(figure)
chart = ImagePart(image=buffer.getvalue(), mime="image/png")
print("true values:", dict(zip(regions, units)))

agent = Agent(AgentConfig(name="chart-reader", model="gemini-3.1-flash-lite", provider="gemini"))
for question in [
    "Which region sold the most, and how many units?",
    "How many more units did the top region sell than the bottom one?",
    "List every region and its value as a JSON object, numbers only.",
]:
    response = agent.run(question, inputs=[chart])
    print(f"\\nQ {question}\\nA {response.output.strip()}")`}
      />

      <Terminal
        command="python read_chart.py"
        output={`true values: {'Alpha': 42, 'Beta': 87, 'Gamma': 31, 'Delta': 65, 'Epsilon': 53}

Q Which region sold the most, and how many units?
A Based on the provided bar chart, the region that sold the most is Beta, with 87 units.

Q How many more units did the top region sell than the bottom one?
A Based on the provided bar chart:

* The region with the highest number of units sold is **Beta**, with **87** units.
* The region with the lowest number of units sold is **Gamma**, with **31** units.

The difference between these two values is 56 (87 - 31 = 56). Therefore, the top region sold 56 more units than the bottom one.

Q List every region and its value as a JSON object, numbers only.
A The image is a bar chart titled "Units sold by region." The x-axis lists five regions (Alpha, Beta, Gamma, Delta, Epsilon), and the y-axis represents the number of units sold on a scale from 0 to 100. Each bar is teal and has its corresponding numerical value displayed above it.

\`\`\`json
{
 "Alpha": 42,
 "Beta": 87,
 "Gamma": 31,
 "Delta": 65,
 "Epsilon": 53
}
\`\`\``}
        maxLines={24}
        caption="All five values read back exactly. The chart is drawn from known numbers so the answer can be checked, which is the only reason to trust it on a chart whose numbers you do not already have."
      />

      <p>
        If you need the JSON without the paragraph in front of it, ask for structured output:{' '}
        <code>run(output_schema=…)</code> constrains the answer to a schema.{' '}
        <Link to="/generation">Generation controls</Link> has the shape.
      </p>

      <h2 id="batches">Run a lot of queries through one agent</h2>

      <p>
        <code>run_batch()</code> takes a list, runs it with bounded concurrency, retries the ones
        that fail and hands back a <code>BatchResult</code> — not a list. The totals on it are the
        reason it exists: cost and tokens for the whole batch, in one place.
      </p>

      <ParamTable
        nameLabel="Parameter"
        params={[
          { name: 'queries', type: 'list[str]', required: true, description: 'The tasks to run.' },
          {
            name: 'max_concurrency',
            type: 'int',
            default: '5',
            description: 'How many run at once. Match it to the provider rate limit, not to the machine.',
          },
          {
            name: 'batch_size',
            type: 'int',
            default: '0',
            description: 'Group size for progress reporting; 0 runs them as one batch.',
          },
          {
            name: 'retry_failed',
            type: 'int',
            default: '1',
            description: 'How many times a failed query is retried before it is recorded as failed.',
          },
          {
            name: 'timeout_per_item',
            type: 'float',
            default: '120.0',
            description: 'Seconds one query may take before it counts as failed.',
          },
          {
            name: 'progress_callback',
            type: 'Callable[[int, int], None] | None',
            default: 'None',
            description: 'Called with (done, total) as the batch advances.',
          },
        ]}
        caption={
          <>
            From <code>inspect.signature(Agent.run_batch)</code>.
          </>
        }
      />

      <CodeBlock
        filename="batch.py"
        code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))

batch = agent.run_batch(
    queries=[
        "Name the capital of Japan.",
        "Name the capital of Peru.",
        "Name the capital of Kenya.",
        "Name the capital of Norway.",
    ],
    max_concurrency=4,
    retry_failed=1,
    timeout_per_item=60,
)

print("type   :", type(batch).__name__)
print("fields :", sorted(vars(batch)))
print()
for item in batch.results:
    print(str(getattr(item, "output", item)).strip()[:70])`}
      />

      <Terminal
        command="python batch.py"
        output={`type   : BatchResult
fields : ['failed', 'per_query_times', 'results', 'succeeded', 'total', 'total_completion_tokens', 'total_cost_usd', 'total_prompt_tokens', 'total_time', 'total_tokens']

Tokyo.
Lima
Nairobi.
Oslo`}
      />

      <Callout type="warning" title="BatchResult is not a list">
        <p>
          Iterating the return value raises <code>TypeError: 'BatchResult' object is not
          iterable</code>. The answers are <code>batch.results</code>; <code>batch.succeeded</code>{' '}
          and <code>batch.failed</code> are the counts, and{' '}
          <code>batch.total_cost_usd</code> is what the whole run cost. For the same thing from a
          shell, with a JSONL file in and a JSONL file out, see{' '}
          <Link to="/cli/batch">Batch &amp; automation</Link>.
        </p>
      </Callout>

      <h3>Pools, when the work outlives one script</h3>

      <p>
        A long-running service wants the agents and the weights kept warm between requests rather
        than rebuilt per call. Four pieces do that, and their constructors are worth reading before
        you wire them together:
      </p>

      <CodeBlock
        filename="pools.py"
        code={`import inspect

from effgen.api.pool import AgentPool
from effgen.api.queue import RequestPriority, RequestQueue
from effgen.models import ContinuousBatcher, LazyModel
from effgen.models.pool import ModelPool, PoolConfig

for cls in (RequestQueue, AgentPool, ModelPool, PoolConfig, LazyModel, ContinuousBatcher):
    print(f"{cls.__name__:18s}{inspect.signature(cls)}")

print()
print("priorities:", [p.name for p in RequestPriority])

queue = RequestQueue(max_size=1000, default_timeout=30.0)
models = ModelPool(config=PoolConfig(max_loaded_models=4, gpu_memory_limit_gb=40))
agents = AgentPool(factory=lambda: None, min_size=2, max_size=10, idle_ttl=300)
print("model pool loaded:", models.loaded_model_names())
print("agent pool stats :", agents.stats())`}
      />

      <Terminal
        command="python pools.py"
        output={`RequestQueue      (max_size: 'int' = 1000, default_timeout: 'float | None' = 30.0) -> 'None'
AgentPool         (factory: 'Callable[[], Any]', *, min_size: 'int' = 1, max_size: 'int' = 8, idle_ttl: 'float' = 300.0, health_check_interval: 'float' = 60.0, health_checker: 'Callable[[Any], bool] | None' = None) -> 'None'
ModelPool         (config: 'PoolConfig | None' = None, loader: 'ModelLoader | None' = None) -> 'None'
PoolConfig        (max_loaded_models: 'int' = 4, gpu_memory_limit_gb: 'float | None' = None, eviction_headroom_gb: 'float' = 2.0, auto_evict: 'bool' = True) -> None
LazyModel         (inner: 'BaseModel', idle_timeout: 'float | None' = 600.0) -> 'None'
ContinuousBatcher (model: 'BaseModel', max_batch_size: 'int' = 8, max_wait_ms: 'float' = 20.0) -> 'None'

priorities: ['HIGH', 'NORMAL', 'LOW']
model pool loaded: []
agent pool stats : {'total': 0, 'in_use': 0, 'healthy': 0, 'min_size': 2, 'max_size': 10}`}
        maxLines={18}
        caption="Two of these are easy to get wrong from memory: RequestQueue takes a default timeout, not a default priority, and LazyModel wraps a model you already built rather than taking a model name."
      />

      <p>
        <code>ModelPool</code> keeps at most <code>max_loaded_models</code> in memory and evicts the
        least recently used one when a new load would cross the budget;{' '}
        <code>prewarm()</code> loads one before the first request needs it, and{' '}
        <code>status()</code> reports what is resident. GPU targeting is{' '}
        <code>CUDA_VISIBLE_DEVICES</code>, the same as everywhere else —{' '}
        <Link to="/hardware">Hardware &amp; GPUs</Link> covers the allocation side.
      </p>

      <h2 id="testing">Test an agent</h2>

      <p>
        Most of what you want to assert about an agent needs no model at all:{' '}
        <code>require_model=False</code> builds the configuration without resolving a backend, so
        the wiring can be checked in a test that runs offline, in milliseconds, with no key.
      </p>

      <CodeBlock
        filename="test_agent.py"
        code={`import asyncio

from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator


def test_calculator_is_wired_in() -> None:
    agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai",
                              tools=[Calculator()], require_model=False))
    assert [tool.name for tool in agent.config.tools] == ["calculator"]


def test_the_tool_itself() -> None:
    result = asyncio.run(Calculator().execute(expression="6 * 7"))
    assert result.success
    assert "42" in str(result.output)`}
      />

      <Terminal
        command="python -m pytest test_agent.py -q"
        output={`..                                                                       [100%]
2 passed, 1 warning in 2.57s`}
        caption="No network, no key, no model loaded."
      />

      <p>
        What that cannot check is whether the agent gets the answer right, which needs a real run
        against a real model. That belongs in a suite with a marker on it so it is not part of the
        fast lane — <Link to="/evaluation">Evaluation &amp; CI gates</Link> is that pattern, with
        the scoring, the thresholds and the exit code a build reads.
      </p>

      <SeeAlso paths={['/multimodal', '/tutorials', '/examples']} />
    </DocPage>
  );
}
