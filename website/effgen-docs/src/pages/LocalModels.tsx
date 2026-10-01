import { HardDrive } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  CodeTabs,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData, version } from '../siteData';

export default function LocalModels() {
  return (
    <DocPage
      subtitle="Running weights on your own machine through the transformers, vLLM, GGUF and MLX engines."
      icon={<HardDrive size={48} />}
    >
      <p>
        A bare HuggingFace repo id means "run this here". effGen downloads the weights once and
        generates in your own process, through one of{' '}
        {siteData.models.local_engines.length} engines:{' '}
        {siteData.models.local_engines.map((engine, i) => (
          <span key={engine}>
            {i > 0 ? ', ' : ''}
            <code>{engine}</code>
          </span>
        ))}
        . No key, no account, and after the first download no network.
      </p>

      <h2>Running one</h2>

      <CodeBlock
        filename="local.py"
        code={`from effgen import load_model

model = load_model("Qwen/Qwen2.5-0.5B-Instruct", engine="transformers")
result = model.generate("The capital of France is", max_tokens=16)

print(repr(result.text.strip()))
print(result.model_name, result.tokens_used, result.finish_reason)
print("cost:", result.metadata.get("cost"))
model.unload()`}
      />

      <Terminal command="python local.py" output={`'The capital of France is Paris.'
Qwen/Qwen2.5-0.5B-Instruct 8 stop
cost: None`} caption={`Run against effGen 1.0.0.`} />

      <p>
        Cost is <code>None</code>, not <code>0</code>: a model running on hardware you already own
        has no token price to report, and effGen says nothing rather than inventing a zero.
      </p>

      <h2>Naming the engine</h2>

      <CodeTabs
        tabs={[
          {
            label: 'load_model',
            code: `from effgen import load_model

model = load_model("Qwen/Qwen2.5-0.5B-Instruct", engine="transformers")`,
          },
          {
            label: 'A prefix',
            filename: 'prefix.py',
            code: `from effgen import create_agent

agent = create_agent("minimal", "transformers:Qwen/Qwen2.5-0.5B-Instruct")
print(agent.run("Name one colour. Answer with the word only.").text.strip())`,
          },
          {
            label: 'Command line',
            language: 'bash',
            code: `effgen run "Name one colour." -m transformers:Qwen/Qwen2.5-0.5B-Instruct`,
          },
        ]}
        caption="The engine prefix on a model id is the same choice load_model's engine= parameter makes."
      />

      <Terminal command="python prefix.py" output={`Red`} />

      <h2>The four engines</h2>

      <ApiTable
        headers={['Engine', 'Hardware', 'Weights', 'Install', 'Use it when']}
        rows={[
          [
            <code>transformers</code>,
            'CPU or NVIDIA GPU',
            'Any HuggingFace repo',
            'in the base install',
            'You want the widest model support and the least setup. This is the default.',
          ],
          [
            <code>vllm</code>,
            'NVIDIA GPU',
            'Any HuggingFace repo',
            <code>effgen[vllm]</code>,
            'Throughput matters — continuous batching across many concurrent requests.',
          ],
          [
            <code>gguf</code>,
            'CPU, with optional GPU offload',
            'GGUF-quantized files',
            <code>effgen[gguf]</code> ,
            'You are running a quantized model on a machine with no usable GPU.',
          ],
          [
            <code>mlx</code>,
            'Apple Silicon',
            'MLX or HuggingFace repos',
            <code>effgen[mlx]</code>,
            'You are on a Mac. Add [mlx-vlm] for vision-language models.',
          ],
        ]}
        caption={
          <>
            Read from the installed package. <Link to="/installation">Installation</Link> has the
            torch and vLLM version tables, which are the part that goes wrong.
          </>
        }
      />

      <Callout type="tip" title="auto-fast picks vLLM only when it will actually work">
        <p>
          <code>load_model(..., engine="auto-fast")</code> uses vLLM when it imports without error
          and a
          GPU is usable, and falls back to the transformers engine otherwise — so opting into speed
          never hard-fails your program. <code>engine=None</code> defaults to transformers.
        </p>
      </Callout>

      <h2>What each engine takes</h2>

      <h3>transformers</h3>
      <ParamTable
        nameLabel="Parameter"
        params={[
          { name: 'quantization_bits', type: 'int | None', default: 'None', description: 'Load in 4-bit or 8-bit. load_model accepts quantization="4bit" as the friendlier spelling.' },
          { name: 'device_map', type: 'str', default: '"auto"', description: 'How layers are placed across devices. "cpu" forces the CPU.' },
          { name: 'use_flash_attention', type: 'bool', default: 'True', description: 'Use FlashAttention where it is installed and the model supports it.' },
          { name: 'torch_dtype', type: 'torch.dtype | None', default: 'None', description: 'Weight dtype. None lets transformers choose.' },
          { name: 'trust_remote_code', type: 'bool', default: 'False', description: 'Allow a repo to execute its own modelling code. Off unless you ask.' },
          { name: 'low_cpu_mem_usage', type: 'bool', default: 'True', description: 'Stream weights in rather than materialising them twice.' },
          { name: 'max_memory', type: 'dict[int, str] | None', default: 'None', description: 'Per-GPU memory ceiling, as {device: "20GiB"}.' },
          { name: 'offload_folder', type: 'str | None', default: 'None', description: 'Where layers that do not fit are offloaded to disk.' },
          { name: 'require_gpu', type: 'bool', default: 'False', description: 'Fail instead of falling back to the CPU when the GPU cannot hold the model.' },
        ]}
        caption={<><code>TransformersEngine(model_name, ...)</code></>}
      />

      <h3>vLLM</h3>
      <ParamTable
        nameLabel="Parameter"
        params={[
          { name: 'tensor_parallel_size', type: 'int', default: '1', description: 'How many GPUs to shard across. load_model auto-detects from model size when you do not pass one.' },
          { name: 'quantization', type: 'str | None', default: 'None', description: 'A vLLM quantization scheme.' },
          { name: 'max_model_len', type: 'int | None', default: 'None', description: 'Cap the context window, which lowers the KV-cache reservation.' },
          { name: 'gpu_memory_utilization', type: 'float', default: '0.9', description: 'Fraction of GPU memory vLLM may reserve. Lower it on CUDA out-of-memory.' },
          { name: 'trust_remote_code', type: 'bool', default: 'True', description: 'Allow a repo to execute its own modelling code.' },
          { name: 'download_dir', type: 'str | None', default: 'None', description: 'Where weights are cached.' },
          { name: 'dtype', type: 'str', default: '"auto"', description: 'Weight dtype.' },
          { name: 'seed', type: 'int', default: '0', description: 'Engine seed.' },
          { name: 'max_num_seqs', type: 'int', default: '256', description: 'How many sequences may be in flight at once.' },
          { name: 'max_num_batched_tokens', type: 'int | None', default: 'None', description: 'Token ceiling for one batch.' },
          { name: 'use_tqdm', type: 'bool', default: 'True', description: 'Show vLLM’s own progress bar.' },
          { name: 'apply_chat_template', type: 'bool', default: 'True', description: 'Apply the model’s chat template, which instruction-tuned models need.' },
          { name: 'system_prompt', type: 'str | None', default: 'None', description: 'A system prompt applied at the engine level.' },
        ]}
        caption={<><code>VLLMEngine(model_name, ...)</code></>}
      />

      <h3>GGUF</h3>
      <ParamTable
        nameLabel="Parameter"
        params={[
          { name: 'n_ctx', type: 'int', default: '4096', description: 'Context window to allocate.' },
          { name: 'n_gpu_layers', type: 'int', default: '0', description: 'How many layers to offload to the GPU. 0 is CPU only.' },
          { name: 'n_threads', type: 'int | None', default: 'None', description: 'CPU threads. None lets llama.cpp choose.' },
          { name: 'verbose', type: 'bool', default: 'False', description: 'llama.cpp’s own logging.' },
        ]}
        caption={<><code>GGUFEngine(model_name, ...)</code></>}
      />

      <h3>MLX</h3>
      <ParamTable
        nameLabel="Parameter"
        params={[
          { name: 'max_tokens', type: 'int | None', default: 'None', description: 'Default output budget.' },
          { name: 'trust_remote_code', type: 'bool', default: 'True', description: 'Allow a repo to execute its own modelling code.' },
          { name: 'apply_chat_template', type: 'bool', default: 'True', description: 'Apply the model’s chat template.' },
          { name: 'system_prompt', type: 'str | None', default: 'None', description: 'A system prompt applied at the engine level.' },
          { name: 'adapter_path', type: 'str | None', default: 'None', description: 'A LoRA adapter to apply on top of the base weights.' },
          { name: 'lazy_load', type: 'bool', default: 'False', description: 'Defer materialising the weights until the first generation.' },
        ]}
        caption={<><code>MLXEngine(model_name, ...)</code></>}
      />

      <h2>What the machine can do</h2>

      <CodeBlock
        filename="hardware.py"
        code={`from effgen.hardware import get_best_local_backend, is_apple_silicon, is_cuda_available

print("cuda:", is_cuda_available())
print("apple silicon:", is_apple_silicon())
print("best local backend:", get_best_local_backend())`}
      />

      <Terminal command="python hardware.py" output={`cuda: True
apple silicon: False
best local backend: vllm`} />

      <h2>From the command line</h2>

      <ParamTable
        nameLabel="Command"
        params={[
          {
            name: 'effgen models load NAME',
            type: '-e, --engine ENGINE',
            description: 'Pre-load a model into memory so the first task does not pay for the load.',
          },
          {
            name: 'effgen models unload NAME',
            type: '—',
            description: 'Release it, and the GPU memory with it.',
          },
          {
            name: 'effgen models status',
            type: '--json',
            description: 'What is loaded in this process, and the physical GPU state beside it.',
          },
          {
            name: 'effgen models list',
            type: '--provider · --free · -t/--tools · --json',
            description: 'The provider registry, and below it every model already in your HuggingFace cache with its size and whether the download completed.',
          },
          {
            name: 'effgen models browse',
            type: '--include-local',
            description: 'Include those locally cached models in the browse table.',
          },
        ]}
        caption={<><code>effgen models --help</code>, {version}.</>}
      />

      <Terminal command="effgen models status" output={`Model & GPU Status
          GPU Status (physical, all processes)           
┏━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ GPU ┃ Name       ┃ Total   ┃ Used    ┃ Free    ┃ Util ┃
┡━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ 0   │ NVIDIA A40 │ 44.4 GB │ 44.3 GB │ 0.2 GB  │ 100% │
│ 1   │ NVIDIA A40 │ 44.4 GB │ 44.2 GB │ 0.3 GB  │ 100% │
│ 2   │ NVIDIA A40 │ 44.4 GB │ 0.0 GB  │ 44.4 GB │ 100% │
│ 3   │ NVIDIA A40 │ 44.4 GB │ 44.4 GB │ 0.1 GB  │ 100% │
│ 4   │ NVIDIA A40 │ 44.4 GB │ 44.4 GB │ 0.1 GB  │ 100% │
│ 5   │ NVIDIA A40 │ 44.4 GB │ 0.0 GB  │ 44.4 GB │ 100% │
│ 6   │ NVIDIA A40 │ 44.4 GB │ 44.1 GB │ 0.3 GB  │ 100% │
│ 7   │ NVIDIA A40 │ 44.4 GB │ 43.9 GB │ 0.5 GB  │ 100% │
└─────┴────────────┴─────────┴─────────┴─────────┴──────┘

No models currently loaded in this process.

Capability profiles registered: 12`} />

      <h2>Running offline</h2>
      <p>
        After the first download nothing on this path needs the network. The catalog, the presets
        and the tool registry all ship inside the package, so a local model plus the built-in tools
        that do not call out is a complete offline agent. The HuggingFace cache is where the weights
        live; <code>effgen models list</code> shows what is in it, including a download that did not
        finish.
      </p>

      <h2>When a local model will not run</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>CUDA out of memory</code>,
            'The weights do not fit in the GPU you have.',
            <>
              Load in 4-bit with <code>quantization="4bit"</code>, pick a smaller model, lower{' '}
              <code>gpu_memory_utilization</code> on vLLM, or select one GPU with{' '}
              <code>CUDA_VISIBLE_DEVICES</code>.
            </>,
          ],
          [
            <code>CUBLAS_STATUS_ALLOC_FAILED</code>,
            'CUDA could not allocate — usually another process is already holding the card.',
            <>
              <code>effgen models status</code> shows what each GPU is doing. Choose a free one, or
              force the CPU with <code>CUDA_VISIBLE_DEVICES=""</code>.
            </>,
          ],
          [
            <>a warning that torch cannot use the GPUs it can see</>,
            'The installed torch is built for a CUDA runtime the driver does not support, so everything runs on the CPU.',
            <>
              Install the matching torch wheel — <Link to="/installation">Installation</Link> has
              the table. <code>EFFGEN_NO_GPU_WARN=1</code> silences the warning when CPU-only is
              deliberate.
            </>,
          ],
          [
            <code>libcudart.so.13: cannot open shared object file</code>,
            'vLLM was built against a CUDA-13 torch this driver cannot run.',
            <>
              Pin the vLLM and torch pair together. <code>VLLMEngine.load()</code> reports this as
              an ABI failure rather than as "vLLM is not installed".
            </>,
          ],
          [
            'The model answers, slowly, on the CPU',
            'No usable GPU was found and the engine fell back.',
            <>
              <code>require_gpu=True</code> makes that a failure instead of a slow success.
            </>,
          ],
          [
            <code>ModuleNotFoundError</code>,
            'The engine you asked for lives behind an extra.',
            <>
              Install it — <code>effgen[vllm]</code>, <code>effgen[gguf]</code>,{' '}
              <code>effgen[mlx]</code>.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="Serving instead of loading">
        <p>
          If several agents need the same weights, serve them once and point effGen at the server
          rather than loading a copy per process — see{' '}
          <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
        </p>
      </Callout>

      <SeeAlso paths={['/openai-compatible', '/installation', '/hardware']} />
    </DocPage>
  );
}
