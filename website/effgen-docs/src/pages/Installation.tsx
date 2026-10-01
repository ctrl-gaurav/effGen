import { Package } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  CodeTabs,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { pythonVersions, version } from '../siteData';

export default function Installation() {
  return (
    <DocPage
      subtitle="Installing effGen and picking the extras your work needs, on CPU, Apple Silicon or a GPU."
      icon={<Package size={48} />}
    >
      <p>
        The base install is one line and carries the agent loop, the tool registry, the command
        line and every provider adapter. The heavy optional stacks — vLLM, vector databases, OCR,
        OpenCV, the speech models — are left out of it deliberately, and each is an extra you add
        when you need it.
      </p>

      <CodeBlock language="bash" code={`pip install effgen`} />

      <Terminal
        command={'python -c "import effgen; print(effgen.__version__)"'}
        output={version}
        caption="If this prints a version, the install is done."
      />

      <h2>Supported Python versions</h2>
      <p>
        effGen supports Python {pythonVersions.join(', ')}. 3.10 was dropped for 1.0.0:{' '}
        <code>tomllib</code>, <code>asyncio.timeout</code>, <code>datetime.UTC</code> and the{' '}
        <code>TimeoutError</code> unification are all standard library from 3.11, and effGen
        carried a hand-written fallback for each.
      </p>

      <Callout type="warning" title="Python 3.14 needs a lock file for the [all] extra">
        <p>
          The base install, the command line, <code>[dev]</code> and every provider extra install
          normally on 3.14. <code>effgen[all]</code> is the exception, and the reason is pip's
          resolver rather than any one package: <code>all</code> pulls <code>vllm</code> on a wide
          range, and when pip backtracks over an unrelated conflict it walks that range backwards
          until it reaches a release pinned to <code>numba==0.61</code>, which predates 3.14 and
          whose source build refuses to run there. Pin the whole set instead.
        </p>
        <CodeBlock
          language="bash"
          showLineNumbers={false}
          code={`pip install -r requirements-all-py314-lock.txt
pip install --no-deps effgen`}
        />
      </Callout>

      <h2>Extras</h2>
      <p>
        Install the slice you need rather than the everything-extra. Several extras can be
        combined in one bracket — <code>pip install "effgen[rag,tools-docs]"</code>.
      </p>

      <h3>Models and providers</h3>
      <ApiTable
        headers={['Extra', 'What it adds']}
        rows={[
          [<code>api</code>, 'Every hosted-inference SDK at once — Groq, Together, Fireworks, Replicate, Cerebras and the HuggingFace hub client.'],
          [<code>groq</code>, 'The Groq SDK on its own.'],
          [<code>together</code>, 'The Together SDK on its own.'],
          [<code>fireworks</code>, 'The Fireworks SDK on its own.'],
          [<code>replicate</code>, 'The Replicate SDK on its own.'],
          [<code>cerebras</code>, 'The Cerebras Cloud SDK on its own.'],
          [<code>hf</code>, 'The HuggingFace hub client, for the Inference API and for downloads.'],
        ]}
        caption={
          <>
            OpenAI and Gemini need no extra — their clients are base dependencies. Nothing here is
            needed to talk to a server of your own; see{' '}
            <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
          </>
        }
      />

      <h3>Running weights locally</h3>
      <ApiTable
        headers={['Extra', 'What it adds']}
        rows={[
          [<code>vllm</code>, 'The vLLM engine, for NVIDIA GPUs. Read the version table below before installing it.'],
          [<code>local</code>, 'Quantization and GGUF: bitsandbytes and llama-cpp-python.'],
          [<code>gguf</code>, 'llama-cpp-python on its own, for GGUF weights.'],
          [<code>mlx</code>, 'The MLX engine, for Apple Silicon.'],
          [<code>mlx-vlm</code>, 'MLX plus vision-language model support.'],
          [<code>flash-attn</code>, 'FlashAttention. Installed separately — see below.'],
          [<code>grammar</code>, 'Constrained decoding through outlines.'],
        ]}
      />

      <h3>Retrieval, documents and media</h3>
      <ApiTable
        headers={['Extra', 'What it adds']}
        rows={[
          [<code>rag</code>, 'sentence-transformers and faiss-cpu — enough to index and search a corpus.'],
          [<code>vector-db</code>, 'faiss-cpu plus the Chroma and Qdrant clients.'],
          [<code>documents</code>, 'PDF, DOCX and XLSX reading, pandas and reportlab.'],
          [<code>tools-docs</code>, 'The document-parsing tools only — the same readers without reportlab.'],
          [<code>tools-web</code>, 'The web-search and browsing tools.'],
          [<code>search</code>, 'tools-web plus the DuckDuckGo client.'],
          [<code>tools</code>, 'OCR through pytesseract and Pillow.'],
          [<code>qr</code>, 'QR generation and reading, with OpenCV.'],
          [<code>audio</code>, 'faster-whisper and pydub, for the transcription tools.'],
          [<code>youtube</code>, 'Transcript and metadata tools for YouTube.'],
          [<code>rss</code>, 'The RSS reader.'],
          [<code>translate</code>, 'Offline translation and language detection.'],
          [<code>geo</code>, 'Static map rendering.'],
          [<code>finance</code>, 'yfinance, for the market-data tools.'],
          [<code>data</code>, 'matplotlib and plotly, for the plotting tool.'],
          [<code>prompts-data</code>, 'SQL parsing for the data-domain prompt templates.'],
        ]}
      />

      <h3>Serving, operating and developing</h3>
      <ApiTable
        headers={['Extra', 'What it adds']}
        rows={[
          [<code>server</code>, 'OIDC/JWT authentication and Prometheus metrics for the API server.'],
          [<code>monitoring</code>, 'Weights & Biases, TensorBoard and gitpython.'],
          [<code>cloud-secrets</code>, 'Secret backends: AWS, Vault and Azure Key Vault.'],
          [<code>lambda</code>, 'The Mangum adapter, for running the server on AWS Lambda.'],
          [<code>jupyter</code>, 'IPython, a kernel and the Jupyter console.'],
          [<code>eval</code>, 'rouge-score and nltk, for the evaluation metrics.'],
          [<code>dev</code>, 'The test and lint stack: pytest and its plugins, ruff, mypy.'],
          [<code>all</code>, 'Every optional dependency, so the full test suite runs. Large, slow to resolve, and needs the lock below.'],
        ]}
      />

      <h2>Installing the all extra</h2>
      <p>
        <code>[all]</code> pulls vLLM plus every provider SDK and the full Google client stack.
        Under the <code>protobuf&gt;=5.29.5</code> security floor that dependency graph is too deep
        for pip to resolve on its own, and the install ends in{' '}
        <code>resolution-too-deep</code>. Install it with the committed constraints lock, which
        pins one consistent solution:
      </p>

      <CodeTabs
        tabs={[
          {
            label: 'From a clone',
            language: 'bash',
            code: `pip install -e ".[all]" -c requirements-all-lock.txt`,
          },
          {
            label: 'From PyPI',
            language: 'bash',
            code: `pip install "effgen[all]" -c https://raw.githubusercontent.com/ctrl-gaurav/effGen/main/requirements-all-lock.txt`,
          },
          {
            label: 'On Python 3.14',
            language: 'bash',
            code: `pip install -r requirements-all-py314-lock.txt
pip install --no-deps effgen`,
          },
        ]}
      />

      <p>
        The two locks are not interchangeable: the 3.11 lock pins{' '}
        <code>torch==2.2.1</code> and <code>vllm==0.4.1</code>, neither of which has a 3.14 wheel.
        Regenerate a lock after changing dependencies with{' '}
        <code>uv pip compile pyproject.toml --extra all --output-file requirements-all-lock.txt</code>.
      </p>

      <h2>NVIDIA GPUs: matching torch to your driver</h2>
      <p>
        PyTorch wheels are built against a specific CUDA runtime, and an NVIDIA driver is forward
        compatible only. A driver reporting <code>CUDA Version: 12.4</code> can run a torch built
        for CUDA 12.x but not one built for CUDA 13 — and the failure is quiet:{' '}
        <code>torch.cuda.is_available()</code> is <code>False</code> while{' '}
        <code>nvidia-smi</code> still lists the GPUs, and everything runs slowly on the CPU.
      </p>

      <CodeBlock
        language="bash"
        code={`nvidia-smi    # top right: "CUDA Version: 12.4"`}
      />

      <ApiTable
        headers={['Your environment', 'Index URL', 'Install']}
        rows={[
          ['CPU only', <code>cpu</code>, <code>pip install torch --index-url https://download.pytorch.org/whl/cpu</code>],
          ['CUDA 12.1–12.7 driver', <code>cu124</code>, <code>pip install "torch&gt;=2.0,&lt;3" --index-url https://download.pytorch.org/whl/cu124</code>],
          ['CUDA 12.8+ driver', <code>cu128</code>, <code>pip install "torch&gt;=2.0,&lt;3" --index-url https://download.pytorch.org/whl/cu128</code>],
          ['CUDA 13.x driver', <code>cu130</code>, <code>pip install "torch&gt;=2.0,&lt;3" --index-url https://download.pytorch.org/whl/cu130</code>],
        ]}
        caption="Install the matching torch wheel before installing effGen. The repository's ./install.sh detects the driver and does this for you."
      />

      <CodeBlock
        language="bash"
        code={`# a host with a CUDA 12.4 driver
pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu124
pip install effgen
python -c "import torch; print(torch.cuda.is_available())"   # True`}
      />

      <h3>Keeping the GPU torch when you install an extra</h3>
      <p>
        Installing the wrong wheel is half the trap. The other half is that once you have a
        working GPU torch, a later extras install that pulls a torch-pinning dependency can let
        pip's resolver upgrade torch back to a newer-CUDA wheel with no warning. A constraints
        file per CUDA line prevents it:
      </p>

      <CodeBlock
        language="bash"
        code={`pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[local]" -c constraints-cu124.txt
python -c "import torch; print(torch.cuda.is_available())"   # still True`}
      />

      <ApiTable
        headers={['Driver CUDA version', 'Constraints file']}
        rows={[
          ['12.1–12.7', <code>constraints-cu124.txt</code>],
          ['12.8+', <code>constraints-cu128.txt</code>],
          ['13.x', <code>constraints-cu130.txt</code>],
          ['CPU only', <code>constraints-cpu.txt</code>],
        ]}
        caption="Each file carries its own --extra-index-url, so it also works on a fresh environment with no torch installed."
      />

      <Callout type="note" title="effGen tells you when torch and the driver disagree">
        <p>
          When effGen sees physical NVIDIA GPUs but <code>torch.cuda</code> cannot use them, it
          prints one warning naming the torch CUDA build against the driver's CUDA version, rather
          than running silently on the CPU. Set <code>EFFGEN_NO_GPU_WARN=1</code> to silence it if
          CPU-only on a GPU box is what you meant.
        </p>
      </Callout>

      <h2>Installing vLLM</h2>
      <p>
        vLLM gives much higher throughput than the transformers engine and is the trickiest
        optional stack to install, because each vLLM release pins one exact torch version and that
        torch build decides the CUDA runtime. The latest vLLM pins a CUDA-13 torch, so a plain{' '}
        <code>pip install effgen[vllm]</code> on a CUDA-12 driver pulls{' '}
        <code>torch ...+cu130</code>, which then fails to import vLLM's compiled extension with{' '}
        <code>libcudart.so.13: cannot open shared object file</code>.
      </p>

      <ApiTable
        headers={['Driver CUDA', 'torch build', 'Known-good vLLM']}
        rows={[
          ['12.1–12.7', <code>torch==2.6.0+cu124</code>, <code>vllm==0.8.5.post1</code>],
          ['12.8+', <code>torch==2.7.1+cu128</code>, <code>vllm==0.10.1.1</code>],
          ['13.x (driver ≥ 580)', <code>torch&gt;=2.11+cu130</code>, <>latest — <code>pip install effgen[vllm]</code> works directly</>],
        ]}
      />

      <CodeBlock
        language="bash"
        code={`# CUDA 12.4 box: a CUDA-12.4 torch first, then a matching vLLM
pip install effgen
pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \\
    --index-url https://download.pytorch.org/whl/cu124
pip install "vllm==0.8.5.post1"
python -c "import torch; from vllm import LLM; print('vLLM ready:', torch.cuda.is_available())"`}
      />

      <p>
        <code>pip check</code> must report no conflicts afterwards; if vLLM and torch disagree on
        versions, the pair is mismatched. Two behaviours make a mismatch diagnosable rather than
        mysterious: <code>VLLMEngine.load()</code> reports an ABI or CUDA import failure as exactly
        that instead of "vLLM is not installed", and{' '}
        <code>load_model(..., engine="auto-fast")</code> uses vLLM only when it imports and a GPU
        is usable, falling back to the transformers engine otherwise.
      </p>

      <h2>Installing flash-attn</h2>
      <p>
        <code>flash-attn</code> is kept out of <code>[all]</code> on purpose. Its own{' '}
        <code>setup.py</code> imports <code>torch</code> while pip is generating wheel metadata,
        and pip's isolated build environment has no torch at that moment — so any package that
        lists it as a dependency breaks <code>pip install</code> for everyone. Install it in a
        second step with build isolation off:
      </p>

      <CodeBlock
        language="bash"
        code={`pip install -e ".[all]" -c requirements-all-lock.txt
pip install flash-attn --no-build-isolation`}
      />

      <p>It needs an NVIDIA GPU of compute capability 7.5 or newer (Turing and later), a{' '}
        <code>nvcc</code> matching your torch CUDA version, GCC 9 or newer, and a few GB of RAM for
        a compile that can take 10–30 minutes. If the build fails, prefer the pre-built wheel from
        the FlashAttention releases page matching your exact Python, torch and CUDA triple.
      </p>

      <h2>Apple Silicon</h2>
      <p>
        MLX is the local engine for Apple Silicon; <code>[mlx-vlm]</code> adds vision-language
        models on top. Neither needs CUDA, and neither is installed by the base package.
      </p>
      <CodeBlock language="bash" code={`pip install "effgen[mlx]"
pip install "effgen[mlx-vlm]"   # also vision-language models`} />
      <p>
        effGen picks the engine for you when you do not name one —{' '}
        <Link to="/local-models">Local models and engines</Link> shows how to check what it chose.
      </p>

      <h2>Checking the install</h2>
      <CodeBlock
        language="bash"
        code={`python -c "import effgen; print(effgen.__version__)"
python -c "from effgen import Agent; print(Agent)"
effgen --version
effgen doctor`}
      />
      <p>
        <code>effgen doctor</code> reports which provider keys effGen can see and prints a system
        report. It prints no key value.{' '}
        <Link to="/configuration">Configuration</Link> covers where it looks for them.
      </p>

      <h2>When it goes wrong</h2>
      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>resolution-too-deep</code>,
            <>pip could not resolve <code>[all]</code> on its own.</>,
            <>Install it with the constraints lock, as above.</>,
          ],
          [
            <code>torch.cuda.is_available() == False</code>,
            'The torch wheel is built for a newer CUDA runtime than the driver supports.',
            'Install the torch wheel matching the driver, then reinstall extras under the matching constraints file.',
          ],
          [
            <code>libcudart.so.13: cannot open shared object file</code>,
            'vLLM was built against a CUDA-13 torch that the driver cannot run.',
            'Pin the vLLM/torch pair from the table above.',
          ],
          [
            <>a <code>flash-attn</code> build failing on <code>import torch</code></>,
            "pip's isolated build environment has no torch.",
            <>Install it separately with <code>--no-build-isolation</code>.</>,
          ],
          [
            <><code>ModuleNotFoundError</code> for an optional dependency at run time</>,
            'The feature you used lives behind an extra.',
            'The error names the extra to install. The tables above say what each one carries.',
          ],
        ]}
      />

      <SeeAlso paths={['/quickstart', '/first-project', '/local-models']} />
    </DocPage>
  );
}
