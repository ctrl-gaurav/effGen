import { ShieldCheck } from 'lucide-react';
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
import { version } from '../siteData';

export default function Execution() {
  return (
    <DocPage
      subtitle="Running model-written code with a container, a timeout and a filesystem it cannot leave."
      icon={<ShieldCheck size={48} />}
    >
      <p>
        A prompt an attacker can reach becomes code an agent runs, so effGen never hands model
        output straight to an interpreter: <code>CodeExecutor</code> runs it inside a sandbox with
        no network, a memory and CPU cap, and one writable directory, and every result says which
        of those were actually enforced.
      </p>

      <h2>Running something</h2>

      <CodeBlock filename="run_code.py" code={`import asyncio

from effgen.tools.builtin.code_executor import CodeExecutor

result = asyncio.run(CodeExecutor().execute(
    language="python", code="print(sum(range(100)))"
))
print(result.success, result.output["stdout"].strip())`} />

      <Terminal
        command="python run_code.py"
        output={`True 4950`}
        caption={`Run against effGen 1.0.0.`}
      />

      <h3>What comes back</h3>

      <CodeBlock filename="result.py" code={`import asyncio

from effgen.tools.builtin.code_executor import CodeExecutor

result = asyncio.run(CodeExecutor().execute(
    language="python", code="import sys; print('out'); print('err', file=sys.stderr)"
))
for key, value in result.output.items():
    print(f"{key:16} {value!r}")`} />

      <Terminal command="python result.py" output={`stdout           'out\\n'
stderr           'err\\n'
exit_code        0
execution_time   0.36937657510861754
timed_out        False
sandbox_backend  'subprocess'
filesystem_confined True
writable_root    '/tmp/effgen-workspace'
success          True`} />

      <h3>Three languages</h3>

      <CodeBlock filename="languages.py" code={`import asyncio

from effgen.tools.builtin.code_executor import CodeExecutor

executor = CodeExecutor()
for language, code in [
    ("python", "print('from python')"),
    ("javascript", "console.log('from node')"),
    ("bash", "echo from bash"),
]:
    result = asyncio.run(executor.execute(language=language, code=code))
    print(f"{language:11} success={result.success} {result.output['stdout'].strip()!r}")`} />

      <Terminal command="python languages.py" output={`python      success=True 'from python'
javascript  success=True 'from node'
bash        success=True 'from bash'`} />

      <ParamTable
        nameLabel="Parameter"
        params={[
          { name: 'code', type: 'string', required: true, description: 'The source to run.' },
          {
            name: 'language',
            type: 'string',
            default: "'python'",
            description: 'One of: python, javascript, bash. Node is required for javascript.',
          },
          {
            name: 'timeout',
            type: 'integer',
            default: '10',
            description: (
              <>
                Seconds before the run is killed. Overrides <code>EFFGEN_SANDBOX_TIMEOUT</code> for
                this call.
              </>
            ),
          },
          {
            name: 'memory_limit',
            type: 'string',
            default: "'256m'",
            description: 'Docker-style size — 256m, 1g. Enforced by cgroups under Docker, advisory otherwise.',
          },
          {
            name: 'network_enabled',
            type: 'boolean',
            default: 'False',
            description: 'Whether the code may reach the network. Leave it off unless the task genuinely needs it.',
          },
        ]}
        caption={<><code>CodeExecutor().execute(...)</code>, from its own metadata.</>}
      />

      <h2>Which sandbox ran it</h2>
      <p>
        The backend is chosen at call time: Docker when the daemon is reachable, otherwise a
        subprocess in an unprivileged user namespace. Rather than assume, read what the run
        reported.
      </p>

      <CodeBlock filename="sandbox.py" code={`import asyncio

from effgen.security.sandbox import SandboxConfig, get_sandbox


async def main():
    config = SandboxConfig(
        backend="auto",          # docker when the daemon is reachable, else subprocess
        timeout=15,
        memory_limit="256m",
        network_enabled=False,
    )
    sandbox = await get_sandbox(config)
    result = await sandbox.run(code="print(sum(range(100)))", language="python", config=config)

    print("stdout                 ", result.stdout.strip())
    print("backend_used           ", result.backend_used)
    print("filesystem_confined    ", result.filesystem_confined)
    print("writable_root          ", result.writable_root)
    print("credential_reads_masked", result.credential_reads_masked)
    print("process_table_isolated ", result.process_table_isolated)


asyncio.run(main())`} />

      <Terminal
        command="python sandbox.py"
        output={`stdout                  4950
backend_used            subprocess
filesystem_confined     True
writable_root           /tmp/effgen-workspace
credential_reads_masked True
process_table_isolated  True`}
        caption="On a host with no reachable Docker daemon, so the subprocess backend was chosen — and said so."
      />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'stdout', type: 'str', description: 'What the code printed.' },
          { name: 'stderr', type: 'str', description: 'What it wrote to standard error.' },
          { name: 'backend_used', type: 'str', description: 'docker, subprocess or off — which backend actually ran it.' },
          {
            name: 'filesystem_confined',
            type: 'bool',
            description: 'Whether writes outside the scratch space were actually refused. False means the isolation could not be set up.',
          },
          { name: 'writable_root', type: 'str', description: 'The one directory the code could write to.' },
          {
            name: 'credential_reads_masked',
            type: 'bool',
            description: 'Whether the named credential stores were covered for this run.',
          },
          {
            name: 'process_table_isolated',
            type: 'bool',
            description: 'Whether the run got a private PID namespace, so it could not read another process’s environment.',
          },
        ]}
        caption={<><code>effgen.security.sandbox.SandboxResult</code>. A caller never has to assume what was enforced.</>}
      />

      <h2>What the sandbox stops</h2>

      <h3>The network</h3>

      <CodeBlock filename="network.py" code={`import asyncio

from effgen.security.sandbox import SandboxConfig, get_sandbox

CODE = """
import urllib.request
try:
    urllib.request.urlopen("https://example.com", timeout=3)
    print("network reachable")
except Exception as exc:
    print("network blocked:", type(exc).__name__)
"""


async def main():
    config = SandboxConfig(backend="auto", timeout=15, network_enabled=False)
    sandbox = await get_sandbox(config)
    result = await sandbox.run(code=CODE, language="python", config=config)
    print(result.stdout.strip())


asyncio.run(main())`} />

      <Terminal command="python network.py" output={`network blocked: URLError`} />

      <h3>Writes outside one directory</h3>

      <CodeBlock filename="writes.py" code={`import asyncio

from effgen.security.sandbox import SandboxConfig, get_sandbox

CODE = """
for path in ("scratch.txt", "/etc/effgen-probe"):
    try:
        open(path, "w").write("x")
        print("wrote", path)
    except OSError as exc:
        print("refused", path, "-", exc.__class__.__name__)
"""


async def main():
    config = SandboxConfig(backend="auto", timeout=15, network_enabled=False)
    sandbox = await get_sandbox(config)
    result = await sandbox.run(code=CODE, language="python", config=config)
    print(result.stdout.strip())
    print("filesystem_confined:", result.filesystem_confined)


asyncio.run(main())`} />

      <Terminal command="python writes.py" output={`wrote scratch.txt
refused /etc/effgen-probe - OSError
filesystem_confined: True`} />

      <h3>A run that will not end</h3>

      <CodeBlock filename="timeout.py" code={`import asyncio

from effgen.tools.builtin.code_executor import CodeExecutor

result = asyncio.run(CodeExecutor().execute(
    language="python", code="import time; time.sleep(30)", timeout=3
))
print(result.success)
print(result.error)`} />

      <Terminal command="python timeout.py" output={`False
Execution timed out after 3s (sandbox 'subprocess' killed the process)`} />

      <h2>The two backends, side by side</h2>

      <ApiTable
        headers={['Control', 'DockerSandbox', 'SubprocessSandbox']}
        rows={[
          [
            'Network isolation',
            <><code>--network=none</code></>,
            <><code>unshare --map-root-user --net</code> — no privileges needed</>,
          ],
          [
            'Write confinement',
            <><code>--read-only</code></>,
            'Every mount remounted read-only except the scratch space, locked by a nested unshare',
          ],
          [
            'Read confinement',
            <>Yes — <code>--read-only</code>, no host mount</>,
            'No. The host filesystem stays readable; the credential stores are masked instead',
          ],
          ['Memory cap', <><code>--memory=256m</code>, enforced</>, <><code>ulimit -v</code>, advisory</>],
          ['CPU cap', <><code>--cpus=1</code>, enforced</>, <><code>ulimit -t</code>, advisory</>],
          ['Process limit', <><code>--pids-limit=100</code></>, <><code>ulimit -u 256</code></>],
          ['Capability drop', <><code>--cap-drop=ALL</code></>, 'Unprivileged user namespace — no host root'],
          [
            'Privilege escalation',
            <><code>--no-new-privileges</code></>,
            'User-namespace UID mapping',
          ],
          [
            'Process table',
            'The container’s own',
            <>
              Private PID namespace (<code>--pid --fork --mount-proc</code>) — the run sees one
              process
            </>,
          ],
        ]}
        caption="Docker is the one to run in production. The subprocess backend is a development fallback, and every run reports which of these it actually enforced."
      />

      <Callout type="warning" title="Reads are masked, not confined, on the subprocess backend">
        <p>
          Code in the subprocess sandbox can read any ordinary file the calling user can read. What
          it cannot read are the per-user credential stores — <code>~/.ssh</code>,{' '}
          <code>~/.aws</code>, <code>~/.gnupg</code>, <code>~/.kube</code>, <code>~/.docker</code>,{' '}
          <code>~/.azure</code>, <code>~/.config/gcloud</code>, the credential files beside them,{' '}
          <code>/etc/shadow</code> and the mounted-secret directories. Each is covered by an empty
          tmpfs or by <code>/dev/null</code>, so a read succeeds and returns nothing rather than
          confirming the path exists. That is a deny list, reported as{' '}
          <code>credential_reads_masked</code>. A secret in a location that is not on it is still
          readable — use Docker when reads have to be confined.
        </p>
      </Callout>

      <h3>The other two backends</h3>

      <ApiTable
        headers={['Backend', 'What it is']}
        rows={[
          [
            <code>off</code>,
            <>
              No sandbox at all. Code runs on the host with the effGen process's privileges. Never
              chosen by auto-resolution, warns loudly on every execution, and is set only by{' '}
              <code>EFFGEN_SANDBOX_BACKEND=off</code>.
            </>,
          ],
          [
            <code>firecracker</code>,
            <>
              An interface stub for MicroVM isolation. It raises <code>NotImplementedError</code>{' '}
              if called; it is not a working backend in {version}.
            </>,
          ],
        ]}
      />

      <h2>Configuration</h2>

      <ParamTable
        nameLabel="Variable"
        params={[
          {
            name: 'EFFGEN_SANDBOX_BACKEND',
            default: "'auto'",
            description: 'auto, docker, subprocess, firecracker or off. auto picks Docker when the daemon is reachable.',
          },
          { name: 'EFFGEN_SANDBOX_TIMEOUT', default: "'10'", description: 'Execution timeout in seconds.' },
          { name: 'EFFGEN_SANDBOX_MEMORY', default: "'256m'", description: 'Memory limit, Docker-style.' },
          {
            name: 'EFFGEN_SANDBOX_IMAGE',
            default: "'effgen/sandbox:python-3.11'",
            description: 'The image DockerSandbox runs. Pin it to a digest in production.',
          },
          {
            name: 'EFFGEN_WORKSPACE',
            description: 'The scratch space — the one directory executed code may write to. The file and shell tools use the same root.',
          },
        ]}
        caption="Read at call time, so a change takes effect without a restart."
      />

      <p>Build the image the Docker backend prefers:</p>

      <CodeBlock
        language="bash"
        code={`docker build -f deploy/sandbox/Dockerfile.sandbox -t effgen/sandbox:python-3.11 .`}
      />

      <p>
        Without it the backend falls back to stock <code>python:3.11-slim</code>, which works and
        carries fewer preinstalled packages.
      </p>

      <h2>In an agent</h2>

      <CodeBlock filename="analyst.py" code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin.code_executor import CodeExecutor

agent = Agent(AgentConfig(
    name="analyst",
    model="gpt-5-nano",
    provider="openai",
    tools=[CodeExecutor()],
))

response = agent.run("Use the code executor to compute the sum of the first 100 prime numbers. Reply with just the number.")
print(response.text)`} />

      <Terminal command="python analyst.py" output={`24133`} />

      <Callout type="tip" title="Two other ways to run code">
        <p>
          <code>PythonREPL</code> keeps variables between calls in a per-session worker process —
          useful when a task builds up state. <code>BashTool</code> runs shell commands against an
          allow list. Both are in <Link to="/tools/gallery">the gallery</Link>, and the
          provider-hosted interpreters are on{' '}
          <Link to="/native-provider-tools">Provider-native tools</Link>.
        </p>
      </Callout>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>A startup <code>WARNING</code> naming inactive protections</>,
            'Docker was unreachable and unprivileged user namespaces are unavailable, so the sandbox degraded to ulimit-only.',
            <>
              Install Docker, or enable user namespaces (<code>kernel.unprivileged_userns_clone=1</code>,{' '}
              <code>user.max_user_namespaces&gt;0</code>). The warning lists exactly what is off.
            </>,
          ],
          [
            <><code>filesystem_confined</code> is <code>False</code></>,
            'The write isolation could not be set up, so the previous, unconfined behaviour applies.',
            'Availability is probed by really writing inside and outside the scratch space. Anything short of both halves leaves confinement off rather than claiming it.',
          ],
          [
            <><code>success</code> is <code>False</code> with a timeout message</>,
            'The run passed its timeout and was killed.',
            <>
              Raise <code>timeout=</code> on the call, or <code>EFFGEN_SANDBOX_TIMEOUT</code>. A
              loop that never ends is the usual cause.
            </>,
          ],
          [
            'A permission error writing a file',
            'The code wrote outside the scratch space.',
            <>
              Widen <code>EFFGEN_WORKSPACE</code> to a directory containing the target, or have the
              task write into the scratch space.
            </>,
          ],
          [
            'A network call fails inside the sandbox',
            'The run has no network namespace, which is the default.',
            <>
              Pass <code>network_enabled=True</code> only after reading the code that will run.
            </>,
          ],
          [
            <code>NotImplementedError</code>,
            <>
              <code>EFFGEN_SANDBOX_BACKEND=firecracker</code> was set.
            </>,
            'That backend is a stub. Use docker, subprocess or auto.',
          ],
          [
            'javascript fails to run at all',
            'Node is not on PATH.',
            <>
              Install Node, or use <code>language="python"</code>.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/tools/gallery', '/security', '/native-provider-tools']} />
    </DocPage>
  );
}
