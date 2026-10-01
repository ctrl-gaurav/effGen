import { Image } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function Multimodal() {
  return (
    <DocPage
      subtitle="Sending images, audio and video to a model, and what each provider accepts."
      icon={<Image size={48} />}
    >
      <p>
        Providers disagree about how media reaches a model — inline data on Gemini, an image URL on
        OpenAI, a separate transcription endpoint for audio. effGen gives you one shape:{' '}
        <code>image_from()</code>, <code>audio_from()</code> and <code>video_from()</code> build a
        content part, the adapter translates it, and a model that cannot take it says so before the
        request is sent.
      </p>

      <h2>Asking about an image</h2>

      <CodeBlock filename="describe.py" code={`from effgen import Agent, AgentConfig, image_from

agent = Agent(AgentConfig(name="vision", model="gemini-3.1-flash-lite", provider="gemini"))

response = agent.run(
    "What single colour dominates this image?",
    inputs=[image_from("/tmp/chart.png")],
)
print(response.text)`} />

      <Terminal
        command="python describe.py"
        output={`The image displays a bar graph with four vertical bars of increasing height, all colored in a dark shade of green. These bars are set against a plain white background, and a single black horizontal line runs along the bottom.

The single color that dominates the image is dark green.`}
        caption={`Run against effGen 1.0.0, on a 300×200 PNG bar chart.`}
      />

      <h2>The three helpers</h2>

      <CodeBlock filename="helpers.py" code={`from effgen import audio_from, image_from

image = image_from("/tmp/chart.png")
print("image", image.mime, len(image.image), "bytes")

audio = audio_from("/tmp/clip.mp3")
print("audio", audio.mime, len(audio.audio), "bytes", audio.duration_s, "s")`} />

      <Terminal command="python helpers.py" output={`image image/png 711 bytes
audio audio/mpeg 12582 bytes None s`} />

      <ApiTable
        headers={['Helper', 'Takes', 'Returns']}
        rows={[
          [
            <code>image_from(source)</code>,
            <>
              bytes, a path, a URL, a <code>PIL.Image.Image</code>, or an{' '}
              <code>np.ndarray</code> of shape H×W×C
            </>,
            <code>ImagePart</code>,
          ],
          [<code>audio_from(source)</code>, 'bytes, a path, or a URL', <code>AudioPart</code>],
          [
            <code>video_from(source, fps=1)</code>,
            'bytes, a path, or a URL — keyframes are sampled',
            <code>VideoPart</code>,
          ],
        ]}
        caption={<>All three are exported from the top-level <code>effgen</code> package.</>}
      />

      <h2>Three ways to pass it</h2>

      <CodeBlock
        filename="three_ways.py"
        code={`from effgen import Agent, AgentConfig, image_from
from effgen.core.messages import Message, Role

agent = Agent(AgentConfig(name="vision", model="gemini-3.1-flash-lite", provider="gemini"))

# 1) text plus inputs= — the canonical form
agent.run("What single colour dominates this image?", inputs=[image_from("/tmp/chart.png")])

# 2) a Message as the task; a bare string in the content list is fine
agent.run(Message(role=Role.USER, content=[image_from("/tmp/chart.png"), "Describe this."]))

# 3) a list of content parts as the task
agent.run([image_from("/tmp/chart.png"), "What is this?"])`}
        caption="Form 1 is the one run above. All three reach the same path; inputs= is an explicit keyword parameter of run() and run_async(), so it shows up in autocomplete and in inspect.signature."
      />

      <Callout type="note" title="Streaming is text only">
        <p>
          <code>agent.stream()</code> yields text. Media goes through <code>run()</code> or{' '}
          <code>run_async()</code>.
        </p>
      </Callout>

      <h2>The message schema</h2>
      <p>
        <code>Message.content</code> is a typed list of content parts, and a plain string still
        works everywhere it used to.
      </p>

      <CodeBlock filename="parts.py" code={`from effgen import image_from
from effgen.core.messages import ImagePart, Message, Role, TextPart

message = Message(
    role=Role.USER,
    content=[image_from("/tmp/chart.png"), TextPart(text="What is in this image?")],
)

print(message.text)
for part in message.content:
    print(" ", type(part).__name__, getattr(part, "mime", ""))`} />

      <Terminal command="python parts.py" output={`What is in this image?
  ImagePart image/png
  TextPart `} />

      <ApiTable
        headers={['Part', 'Fields', 'Validated']}
        rows={[
          [<code>TextPart</code>, <><code>text: str</code></>, '—'],
          [
            <code>ImagePart</code>,
            <>
              <code>image: bytes</code>, <code>mime: str</code>, <code>meta: dict</code>
            </>,
            'MIME is one of png, jpeg, gif, webp',
          ],
          [
            <code>AudioPart</code>,
            <>
              <code>audio: bytes</code>, <code>mime: str</code>,{' '}
              <code>duration_s: float | None</code>
            </>,
            'MIME is one of mp3, wav, flac, ogg, m4a',
          ],
          [
            <code>VideoPart</code>,
            <>
              <code>frames: list[bytes]</code>, <code>fps: float</code>, <code>mime: str</code>
            </>,
            'Frames must be non-empty',
          ],
          [
            <code>ToolCallPart</code>,
            <>
              <code>id</code>, <code>name</code>, <code>arguments</code>
            </>,
            '—',
          ],
          [
            <code>ToolResultPart</code>,
            <>
              <code>id</code>, <code>content</code>
            </>,
            '—',
          ],
        ]}
        caption={
          <>
            <code>message.text</code> joins every <code>TextPart</code>, so reading the text of a
            mixed message is unchanged.
          </>
        }
      />

      <CodeBlock filename="invalid.py" code={`from effgen.core.messages import ImagePart, VideoPart
from effgen.errors import InvalidMultimodalContent

for build in (
    lambda: ImagePart(image=b"\\x89PNG", mime="image/tiff"),
    lambda: VideoPart(frames=[], fps=1.0, mime="image/jpeg"),
):
    try:
        build()
    except InvalidMultimodalContent as exc:
        print(type(exc).__name__, "-", exc)`} />

      <Terminal command="python invalid.py" output={`InvalidMultimodalContent - Invalid image content: MIME type 'image/tiff' is not supported. Allowed: ['image/gif', 'image/jpeg', 'image/png', 'image/webp']. Check that the image source is reachable and in a supported format.
InvalidMultimodalContent - Invalid video_frames content: frames list must be non-empty. Check that the video_frames source is reachable and in a supported format.`} />

      <h2>A model that cannot take it</h2>
      <p>
        No adapter quietly turns an image into <code>"[image not supported]"</code>. The capability
        is checked before the request goes out, and the error names a model that would work.
      </p>

      <CodeBlock filename="capability.py" code={`from effgen import image_from, load_model
from effgen.core.messages import Message, Role
from effgen.errors import CapabilityNotSupportedError

model = load_model("gpt-3.5-turbo", provider="openai")     # a text-only model
message = Message(role=Role.USER, content=[image_from("/tmp/chart.png"), "Describe this."])

try:
    model.generate([message])
except CapabilityNotSupportedError as exc:
    print(type(exc).__name__)
    print("capability:", exc.capability)
    print("provider:  ", exc.provider)
    print(exc)`} />

      <Terminal command="python capability.py" output={`CapabilityNotSupportedError
capability: vision
provider:   openai
Capability 'vision' is not supported by provider 'openai'.

Model 'gpt-3.5-turbo' does not support vision. Use 'gpt-4o-mini' or 'gpt-4o' for image inputs.

Choose a model that declares this capability — run \`effgen models list --capability <name>\` to see which do.`} />

      <h2>Which provider takes what</h2>

      <ApiTable
        headers={['Provider', 'Image', 'Audio', 'Video (native)', 'Video (frames)']}
        rows={[
          ['Gemini 2.x / 3.x', 'Yes', 'Yes', 'Yes', 'Yes'],
          ['OpenAI gpt-4o family', 'Yes', 'Yes, via Whisper', 'No', 'Yes'],
          ['Groq (Llama 4, 3.2-vision)', 'Yes', 'No', 'No', 'Yes'],
          ['Anthropic', 'Yes', 'No', 'No', 'No'],
          ['Together (vision models)', 'Yes', 'No', 'No', 'Yes'],
          ['HuggingFace Inference (BLIP, LLaVA)', 'Yes', 'Yes, via ASR', 'No', 'Yes'],
          ['Cerebras', 'No', 'No', 'No', 'No'],
          ['MLX-VLM (Apple Silicon)', 'Yes', 'No', 'No', 'Yes'],
        ]}
        caption={
          <>
            "Video (frames)" means effGen samples keyframes and sends them as images, so a
            vision-only model can still answer about a clip. The catalog knows this per model —{' '}
            <code>effgen models list --capability vision</code>, and{' '}
            <Link to="/catalog">The model catalog</Link>.
          </>
        }
      />

      <h2>Preprocessing</h2>
      <p>
        Between your part and the wire, effGen fits the media to what the provider accepts. Every
        step is recorded in <code>part.meta["preprocessing"]</code>, so a resize is visible rather
        than assumed.
      </p>

      <ApiTable
        headers={['Modality', 'What is done']}
        rows={[
          [
            'Image',
            <>
              Downscaled to the provider's pixel maximum with PIL Lanczos (2048×2048 on Gemini),
              converted between PNG and JPEG when the MIME or the byte limit requires it.
            </>,
          ],
          [
            'Audio',
            'Downsampled to 16 kHz mono where the provider requires it, and clips past the provider maximum are chunked into sequential calls whose results are concatenated.',
          ],
          [
            'Video',
            'Keyframes sampled at the requested fps up to a frame cap; the audio track extracted separately when there is one.',
          ],
        ]}
      />

      <CodeBlock filename="video.py" code={`from effgen.multimodal.video_pre import VideoSource

source = VideoSource("/tmp/clip.mp4")
frames = source.sample_frames(fps=1, max_frames=16)
audio = source.extract_audio()

print(len(frames), "frames sampled;", frames[0].mime)
print("audio track:", audio.mime if audio else None)`} />

      <Terminal
        command="python video.py"
        output={`2 frames sampled; image/jpeg
audio track: None`}
        caption="A short clip with no audio track, so extract_audio() returns None rather than raising."
      />

      <Callout type="warning" title="Video needs ffmpeg">
        <p>
          <code>VideoSource</code> requires <code>ffmpeg</code> on <code>PATH</code> and raises{' '}
          <code>MissingSystemDependency("ffmpeg", …)</code> with per-OS install instructions when
          it is absent.
        </p>
      </Callout>

      <h2>The multimodal preset</h2>

      <CodeBlock filename="preset.py" code={`from effgen import create_agent

agent = create_agent("multimodal", "gemini-3.1-flash-lite", provider="gemini")
print(sorted(tool.name for tool in agent.config.tools))`} />

      <Terminal command="python preset.py" output={`['audio_transcribe', 'image_caption', 'image_info', 'multimodal_describe', 'ocr', 'pdf', 'weather']`} />

      <p>
        Seven tools, with Gemini Flash-Lite as the primary model and gpt-4o-mini and HF BLIP as
        vision fallbacks. <code>multimodal_describe</code> looks at the input's type and calls the
        right one of the others, so a mixed folder needs no routing of your own.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>CapabilityNotSupportedError</code>,
            'The model does not declare the capability the part needs.',
            <>
              <code>exc.capability</code> and <code>exc.provider</code> say which and whose, and
              the message names a model that would work.
            </>,
          ],
          [
            <code>InvalidMultimodalContent</code>,
            'The MIME type is not one of the accepted ones, the frame list is empty, or the file is not there.',
            'The message says which of those it is. It is raised when the part is built, before any request.',
          ],
          [
            <code>MissingSystemDependency</code>,
            <>
              <code>ffmpeg</code> is not on <code>PATH</code>.
            </>,
            'The error carries per-OS install instructions. Video sampling and audio extraction both need it.',
          ],
          [
            'A very large bill for one image',
            'A high-resolution image is a lot of tokens.',
            <>
              Preprocessing downscales to the provider maximum, not to what you need. Resize first
              — <code>ImageInfoTool</code> does it locally.
            </>,
          ],
          [
            'An empty transcription',
            'The clip has no speech, or the audio track is silent.',
            <>
              <code>result.success</code> is <code>True</code> with an empty{' '}
              <code>text</code>: a silent clip is not an error. Check the file plays.
            </>,
          ],
          [
            'A truncated answer about a long video',
            'Frame sampling is capped, so a long clip is represented by a bounded number of frames.',
            <>
              Raise <code>fps</code> or <code>max_frames</code> on{' '}
              <code>sample_frames</code>, or split the clip.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/providers', '/tools/gallery', '/presets']} />
    </DocPage>
  );
}
