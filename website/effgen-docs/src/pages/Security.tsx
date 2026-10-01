import { Lock } from 'lucide-react';
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

export default function Security() {
  return (
    <DocPage
      subtitle="Where credentials live, how they are kept out of logs and errors, and what the project publishes about its own supply chain."
      icon={<Lock size={48} />}
    >
      <p>
        Two different questions live on this page. The first is how your keys are handled by code
        that runs on your machine — where they are read from, and what stops them reaching a log
        line, a terminal panel or an API response. The second is what you can check about the
        package itself before you install it.
      </p>

      <h2>Where credentials come from</h2>
      <p>
        effGen reads provider keys from the environment. Nothing is ever written to a source file,
        and no key is stored by the framework.
      </p>

      <ApiTable
        headers={['Location', 'When to use it']}
        rows={[
          [
            'The process environment',
            'CI, containers, anything with a secret manager in front of it. Highest precedence.',
          ],
          [
            <code>./.env</code>,
            'One project on one machine. The nearest .env above the working directory is loaded before a command runs, so a fresh install works without exporting anything by hand.',
          ],
          [
            <code>~/.effgen/.env</code>,
            <>
              Keys shared across every project on a machine. Give it <code>chmod 600</code>.
            </>,
          ],
        ]}
      />

      <CodeBlock
        language="bash"
        filename=".env"
        code={`# Never committed. Add .env to .gitignore before you write the first key into it.
OPENAI_API_KEY=...
GROQ_API_KEY=...
GOOGLE_API_KEY=...`}
      />

      <p>
        A snippet in this documentation that needs a key names the variable, never a value. So
        should yours: a key that reaches a repository is compromised the moment it is pushed, and
        rotation at the provider is the only fix — removing the commit does not undo it.
      </p>

      <h2>Redaction</h2>
      <p>
        Every string effGen logs, traces, or puts in an error message passes through a redactor
        first. It carries fifteen patterns covering the providers effGen talks to, plus bearer
        tokens, webhook URLs and <code>NAME_SECRET=value</code> assignments.
      </p>

      <CodeBlock filename="redact.py" code={`from effgen.observability.redact import get_redactor

redactor = get_redactor()
print(redactor.scrub("Authorization: Bearer sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH"))
print(redactor.scrub("GROQ_API_KEY=gsk_0123456789abcdefghijABCDEFGHIJKLMNOPQRSTUVWXYZ01"))
print(redactor.scrub_dict({"model": "gpt-5-nano", "api_key": "sk-live-abcdefghijklmnop"}))
print()
print(len(redactor.pattern_names()), "patterns:", ", ".join(redactor.pattern_names()))`} />

      <Terminal
        command="python redact.py"
        output={`Authorization: <REDACTED:bearer_token>
GROQ_API_KEY=<REDACTED:groq_key>
{'model': 'gpt-5-nano', 'api_key': '<REDACTED:openai_key>'}

15 patterns: anthropic_key, cerebras_key, google_key, hf_key, groq_key, replicate_key, fireworks_key, github_token, slack_token, aws_access_key, openai_key, bearer_token, slack_webhook, discord_webhook, env_secret`}
        caption={`Run against effGen 1.0.0. The marker names the pattern that matched, so a redacted log still says what kind of credential was there.`}
      />

      <p>
        Add your own — an employee id, an internal account number, a customer reference — and it is
        redacted everywhere the framework redacts.
      </p>

      <CodeBlock filename="custom_pattern.py" code={`from effgen.observability.redact import get_redactor

redactor = get_redactor()
redactor.add_pattern("employee_id", r"\\bEMP-\\d{6}\\b", "[EMPLOYEE ID]")
print(redactor.scrub("Ticket raised by EMP-004821 about the invoice."))`} />

      <Terminal
        command="python custom_pattern.py"
        output={`Ticket raised by [EMPLOYEE ID] about the invoice.`}
      />

      <ApiTable
        headers={['Method', 'Takes', 'Returns']}
        rows={[
          [<code>scrub(text)</code>, 'A string', 'The string with every match replaced'],
          [
            <code>scrub_dict(data)</code>,
            'A dict',
            'A copy with every value scrubbed, keys untouched',
          ],
          [
            <code>scrub_value(value)</code>,
            'Anything',
            'The value scrubbed if it is text, unchanged otherwise',
          ],
          [
            <code>add_pattern(name, regex, replacement=None)</code>,
            'A name, a regex, an optional marker',
            <>
              <code>None</code> — the pattern is added to the shared redactor
            </>,
          ],
          [
            <code>pattern_names()</code>,
            '—',
            'The names of every pattern, in the order they are applied',
          ],
        ]}
        caption={
          <>
            <code>get_redactor()</code> returns the process-wide instance, so a pattern added once
            applies to everything after it.
          </>
        }
      />

      <h3>Error messages are redacted and bounded</h3>
      <p>
        A provider that rejects a request often echoes part of it back, and sometimes the credential
        with it. Every message that quotes text effGen did not write passes through{' '}
        <code>quote_for_message</code>: redacted first, then cut to 240 characters. The order
        matters — redacting first means a key that straddles the cut is replaced rather than
        half-printed.
      </p>

      <CodeBlock filename="messages.py" code={`from effgen.errors import quote_for_message

body = "401 Unauthorized: key sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH is not valid for this org"
print(quote_for_message(body))
print()
print(quote_for_message("x" * 400))`} />

      <Terminal
        command="python messages.py"
        output={`401 Unauthorized: key <REDACTED:openai_key> is not valid for this org

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx… (400 characters)`}
        caption={
          <>
            The cut is marked with the real length, so nothing looks complete when it is not. The
            limit is <code>effgen.errors.MESSAGE_ECHO_LIMIT</code>.
          </>
        }
      />

      <Callout type="note" title="Redaction never replaces the failure">
        <p>
          If the redactor itself cannot be loaded, the original text is returned rather than an error
          about redacting it. A broken redactor makes a message less safe; it does not make the
          failure invisible.
        </p>
      </Callout>

      <h2>Screening for committed secrets</h2>
      <p>
        The project scans itself with <a href="https://github.com/gitleaks/gitleaks">gitleaks</a>,
        both as a pre-commit hook and in CI over the full history. The rules are in{' '}
        <code>.gitleaks.toml</code> and the hook is in <code>.pre-commit-config.yaml</code>; the same
        two files work in a project of your own.
      </p>

      <CodeBlock
        language="bash"
        code={`pip install pre-commit
pre-commit install                 # runs gitleaks on every git commit

gitleaks dir . --config .gitleaks.toml --verbose      # the working tree
gitleaks git . --config .gitleaks.toml --verbose      # the whole history
gitleaks protect --staged --config .gitleaks.toml     # what you are about to commit`}
      />

      <ApiTable
        headers={['Provider', 'Pattern the rules match']}
        rows={[
          [<code>OpenAI</code>, <code>sk-proj-… / sk-svcacct-…</code>],
          [<code>Anthropic</code>, <code>sk-ant-api03-…</code>],
          [<code>Groq</code>, <code>gsk_…</code>],
          [<code>Cerebras</code>, <code>csk-… / CEREBRAS_API_KEY=…</code>],
          [<code>Google Gemini</code>, <code>AIza…</code>],
          [<code>HuggingFace</code>, <code>hf_…</code>],
          [<code>Replicate</code>, <code>r8_…</code>],
          [<code>Together</code>, <code>TOGETHER_API_KEY=…</code>],
          [<code>Fireworks</code>, <code>FIREWORKS_API_KEY=…</code>],
          [<code>AWS</code>, <code>AKIA…</code>],
        ]}
      />

      <Callout type="danger" title="If a key does reach a remote">
        <p>
          Rotate it at the provider first, before anything else. Then rewrite the history and force
          push. The order is not interchangeable: the key is readable from the moment it is pushed,
          and a rewritten history does not un-publish it.
        </p>
      </Callout>

      <h2>Checking the install you got</h2>
      <p>
        <code>requirements-lock.txt</code> pins every transitive dependency of the base install with
        a hash. Installing against it gives a build that resolves to exactly those versions.
      </p>

      <CodeBlock language="bash" code={`pip install effgen -r requirements-lock.txt`} />

      <p>
        <code>EFFGEN_VERIFY_HASHES=1</code> turns on a startup check that compares what is installed
        against those pins and warns on drift. It never blocks startup — an operator is told, and the
        application runs.
      </p>

      <CodeBlock filename="verify.py" code={`import warnings

from effgen.security import HashDriftWarning, verify_installed_hashes

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = verify_installed_hashes()

print("checked        :", result.checked)
print("matching lock  :", result.ok)
print("drifted        :", len(result.drifted))
print("not in lockfile:", len(result.not_in_lockfile))
print("clean          :", result.clean)
print("warnings       :", sum(1 for w in caught if issubclass(w.category, HashDriftWarning)))`} />

      <Terminal
        command="python verify.py"
        output={`checked        : 449
matching lock  : 112
drifted        : 36
not in lockfile: 301
clean          : False`}
        caption="Taken in a development checkout with optional extras installed past the lockfile, which is exactly the situation the check exists to report. An environment installed from the lockfile reports clean=True and emits one ok line."
      />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'checked', type: 'int', description: 'Distributions examined.' },
          { name: 'ok', type: 'int', description: 'Matching the pinned version exactly.' },
          {
            name: 'drifted',
            type: 'list[str]',
            description:
              'Package names whose version, or whose build fingerprint at the same version, does not match.',
          },
          {
            name: 'not_in_lockfile',
            type: 'list[str]',
            description:
              'Installed but not pinned — extras and development tools live here, and are not a finding.',
          },
          { name: 'skipped', type: 'int', description: 'Distributions whose metadata was unreadable.' },
          {
            name: 'clean',
            type: 'bool',
            description: (
              <>
                <code>True</code> when <code>drifted</code> is empty.
              </>
            ),
          },
        ]}
        caption={
          <>
            <code>VerificationResult</code>, from <code>effgen.security</code>. Each drift also
            raises a <code>HashDriftWarning</code> naming the package and both versions.
          </>
        }
      />

      <ApiTable
        headers={['What is checked', 'What a mismatch means']}
        rows={[
          [
            'Version pinning',
            'The installed version is not the pinned one. Usually a dependency was upgraded outside the lockfile.',
          ],
          [
            'Build fingerprint',
            'A SHA-256 of the distribution’s WHEEL metadata, recorded on the first run and compared afterwards. Same version, different fingerprint, means the wheel was replaced.',
          ],
        ]}
        caption={
          <>
            The fingerprint snapshot lives at{' '}
            <code>~/.effgen/supply_chain/installed_hashes.json</code>.
          </>
        }
      />

      <h2>The bill of materials</h2>
      <p>
        Every release is accompanied by a CycloneDX JSON v1.5 SBOM listing every Python package in
        the runtime environment, with its version, its <code>purl</code> and its licence. You can
        generate the same file for the environment you actually installed into.
      </p>

      <CodeBlock
        language="bash"
        code={`pip install cyclonedx-bom
cyclonedx-py environment --of json --sv 1.5 -o sbom.cdx.json --pyproject pyproject.toml

grype sbom:sbom.cdx.json          # or
osv-scanner --sbom sbom.cdx.json  # or
trivy sbom sbom.cdx.json`}
      />

      <p>
        Alongside it, <code>pip-audit</code> runs against the lockfile on every push to{' '}
        <code>main</code> and daily, and fails the build on a CVE in any non-exempt package. The
        exemptions are packaging tools, development-only test utilities, and optional extras whose
        advisories are in code paths effGen does not execute.
      </p>

      <CodeBlock
        language="bash"
        code={`pip install pip-audit
pip-audit --format json --output pip-audit-report.json`}
      />

      <Callout type="note" title="Wheel signing">
        <p>
          Sigstore keyless signing is written into the release workflow, but that workflow is not
          enabled for {version}. Do not plan a deployment gate around verifying a published
          signature; verify the SBOM and the lockfile instead.
        </p>
      </Callout>

      <h2>Running untrusted code</h2>
      <p>
        Anything an agent executes — a snippet it wrote, a shell command it chose — runs through the
        sandbox layer, not through a bare <code>subprocess</code>.{' '}
        <Link to="/execution">Code execution and the sandbox</Link> covers the four backends and what
        each one isolates.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>… API key not provided. Set OPENAI_API_KEY …</code>,
            'No key was found in the environment or in any .env that was discovered.',
            <>
              Check which directory you ran from — the <code>.env</code> lookup starts there. An
              exported variable always wins.
            </>,
          ],
          [
            'A key appears in your own log line',
            'Redaction covers what effGen logs. Text you format and log yourself is yours.',
            <>
              Put it through <code>get_redactor().scrub(...)</code> first.
            </>,
          ],
          [
            <><code>HashDriftWarning</code> on startup</>,
            'An installed version does not match the lockfile pin, or a wheel was replaced at the same version.',
            <>
              Reinstall against <code>requirements-lock.txt</code>. A version bump you made on
              purpose means regenerating the lockfile.
            </>,
          ],
          [
            <>
              <code>not_in_lockfile</code> is large
            </>,
            'The lockfile pins the base install. Extras and development tools are not in it.',
            <>
              This is not a finding. Only <code>drifted</code> is.
            </>,
          ],
          [
            'A pattern you added stops being applied',
            <>
              <code>add_pattern</code> is per process, and a new process starts from the fifteen
              built-ins.
            </>,
            'Add it at import time, in a module every entry point loads.',
          ],
          [
            'An error message is cut off mid-sentence',
            'It exceeded the 240-character echo limit and was bounded.',
            <>
              The full length is printed after the ellipsis. The untruncated provider body is in the
              logs at <code>EFFGEN_LOG_LEVEL=DEBUG</code>.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/guardrails', '/execution', '/errors']} />
    </DocPage>
  );
}
