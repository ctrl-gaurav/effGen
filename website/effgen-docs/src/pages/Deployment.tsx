import { Ship } from 'lucide-react';
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
import { version } from '../siteData';

export default function Deployment() {
  return (
    <DocPage
      subtitle="Running the server on Docker, Kubernetes, AWS Lambda or a Cloudflare Worker."
      icon={<Ship size={48} />}
    >
      <p>
        Four recipes ship in the framework repository under <code>deploy/</code>, and all four run
        the same FastAPI application <Link to="/api-server"><code>effgen serve</code></Link> runs — a
        container image, a Helm chart, an AWS SAM template with a Mangum adapter, and a Cloudflare
        Worker that fronts any of the other three. Nothing about auth, roles, the audit log or the{' '}
        <code>/v1</code> routes changes between them.
      </p>

      <Callout type="note" title="These files come from a checkout">
        <p>
          <code>pip install effgen</code> installs the Python package. The <code>Dockerfile</code>,
          the Helm chart, the SAM template and the Worker are repository artefacts — clone{' '}
          <code>github.com/ctrl-gaurav/effGen</code> to get them, and read every path on this page as
          relative to that checkout.
        </p>
      </Callout>

      <h2>Docker</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`docker build -f deploy/docker/Dockerfile --build-arg EXTRAS=server -t effgen:${version} .

docker run --rm -p 8080:8080 -e EFFGEN_DEV_MODE=1 effgen:${version}

curl http://localhost:8080/health   # {"status":"ok"}`}
        caption="Dev mode disables auth, which is fine on a loopback port and nowhere else."
      />

      <p>
        The image is multi-stage: a builder installs the package into a virtualenv at{' '}
        <code>/opt/effgen/venv</code>, and the runtime stage copies only that. It runs as{' '}
        <code>effgen</code> (uid 1001), exposes 8080, and carries a <code>HEALTHCHECK</code> that
        polls <code>/health</code> every 30 seconds and marks the container unhealthy after three
        misses. The entrypoint is uvicorn against the app factory, so it is the same application by
        construction.
      </p>

      <ParamTable
        nameLabel="Build argument"
        params={[
          {
            name: 'VERSION',
            type: 'str',
            default: version,
            description: 'Baked into the image labels as the OCI version.',
          },
          {
            name: 'EXTRAS',
            type: 'str',
            default: 'server',
            description: (
              <>
                pip extras installed into the image. <code>server</code> adds OIDC/JWT auth and the
                Prometheus client; FastAPI and uvicorn are core. Add a provider SDK by naming it —{' '}
                <code>--build-arg EXTRAS=server,openai</code>.
              </>
            ),
          },
        ]}
        caption={
          <>
            From <code>deploy/docker/Dockerfile</code>. The heavy local-inference extras are not
            installed by default, which is what keeps the image small; add them if the container is
            meant to run a <Link to="/local-models">local engine</Link>.
          </>
        }
      />

      <h3>A run that faces a network</h3>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`docker run --rm -p 8080:8080 \\
  -e EFFGEN_DEV_MODE=0 \\
  -e EFFGEN_OIDC_ISSUER=https://your-issuer.example.com \\
  -e EFFGEN_OIDC_CLIENT_ID=your-client-id \\
  -e EFFGEN_OIDC_JWKS_URI=https://your-issuer.example.com/.well-known/jwks.json \\
  -v ~/.effgen:/home/effgen/.effgen:ro \\
  --read-only \\
  --tmpfs /tmp \\
  --tmpfs /home/effgen/.effgen/audit \\
  effgen:${version}`}
        caption={
          <>
            The audit directory needs to be writable, which is why it is a tmpfs mount under{' '}
            <code>--read-only</code>. Ship the lines out with a log agent rather than keeping them in
            the container.
          </>
        }
      />

      <ApiTable
        headers={['Variable', 'Default in the image', 'What it does']}
        rows={[
          [<code>EFFGEN_DEV_MODE</code>, <code>0</code>, 'Auth is on unless you turn it off.'],
          [<code>EFFGEN_PORT</code>, <code>8080</code>, 'Listening port.'],
          [<code>EFFGEN_HOST</code>, <code>0.0.0.0</code>, 'Bound inside the container; publish it with -p.'],
          [
            <code>EFFGEN_SANDBOX_BACKEND</code>,
            <code>subprocess</code>,
            <>
              How <Link to="/execution">code execution</Link> is confined. Docker-in-Docker is not
              available inside the image.
            </>,
          ],
          [<code>EFFGEN_AUDIT_DIR</code>, <code>~/.effgen/audit</code>, 'Where the audit JSONL goes.'],
          [
            <>
              <code>EFFGEN_OIDC_ISSUER</code> / <code>_CLIENT_ID</code> / <code>_JWKS_URI</code>
            </>,
            '—',
            'OIDC configuration; the JWKS URI is discovered when omitted.',
          ],
        ]}
        caption={
          <>
            Provider keys are injected at runtime and never baked into an image.{' '}
            <Link to="/api-server">The API server page</Link> has the full environment table.
          </>
        }
      />

      <h3>Compose</h3>

      <CodeBlock
        language="yaml"
        filename="deploy/docker/docker-compose.yml"
        code={`services:
  effgen:
    build:
      context: ../..
      dockerfile: deploy/docker/Dockerfile
      args:
        EXTRAS: server
    image: effgen:local
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      EFFGEN_DEV_MODE: "1"
      EFFGEN_PORT: "8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3`}
        caption={
          <>
            <code>docker compose -f deploy/docker/docker-compose.yml up --build</code>. The port is
            published on loopback only because this file runs in dev mode; for a real deployment,
            drop <code>EFFGEN_DEV_MODE</code>, set the <code>EFFGEN_OIDC_*</code> variables and widen
            the mapping to <code>"8080:8080"</code>.
          </>
        }
      />

      <h2>Kubernetes</h2>

      <p>
        The chart is at <code>deploy/k8s/helm/effgen/</code>. It needs Kubernetes 1.24 or newer and
        Helm 3.12 or newer. With default values it renders six objects; the ConfigMap, Secret,
        Ingress and PersistentVolumeClaim templates are there too and switch on through{' '}
        <code>values.yaml</code>.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`helm lint deploy/k8s/helm/effgen/

helm install effgen deploy/k8s/helm/effgen \\
  --set env.EFFGEN_OIDC_ISSUER=https://your-issuer.example.com \\
  --set env.EFFGEN_OIDC_CLIENT_ID=your-client-id \\
  --set image.tag=${version} \\
  -n effgen --create-namespace`}
      />

      <Terminal command="helm lint deploy/k8s/helm/effgen/" output={`==> Linting deploy/k8s/helm/effgen/

1 chart(s) linted, 0 chart(s) failed`} />

      <Terminal
        command="helm template effgen deploy/k8s/helm/effgen/ | grep '^kind:' | sort | uniq -c"
        output={`      1 kind: Deployment
      1 kind: HorizontalPodAutoscaler
      1 kind: NetworkPolicy
      1 kind: PodDisruptionBudget
      1 kind: Service
      1 kind: ServiceAccount`}
        caption="What the chart actually renders, from this checkout."
      />

      <ParamTable
        nameLabel="Value"
        params={[
          { name: 'replicaCount', type: 'int', default: '2', description: 'Pods before the autoscaler takes over.' },
          { name: 'autoscaling.enabled', type: 'bool', default: 'true', description: 'Render the HorizontalPodAutoscaler.' },
          { name: 'autoscaling.minReplicas', type: 'int', default: '2', description: 'HPA floor.' },
          { name: 'autoscaling.maxReplicas', type: 'int', default: '10', description: 'HPA ceiling.' },
          {
            name: 'autoscaling.customMetrics',
            type: 'list',
            description: (
              <>
                Scales on <code>effgen_model_call_latency_seconds</code> as well as CPU. Needs a
                Prometheus Adapter or KEDA to expose it as a custom metric.
              </>
            ),
          },
          { name: 'networkPolicy.enabled', type: 'bool', default: 'true', description: 'Restrict ingress and egress.' },
          { name: 'podDisruptionBudget.enabled', type: 'bool', default: 'true', description: 'Keep at least one pod during a drain.' },
          { name: 'env.EFFGEN_DEV_MODE', type: 'str', default: '"0"', description: 'Auth on by default.' },
        ]}
        caption={
          <>
            The keys most deployments touch. <code>helm show values deploy/k8s/helm/effgen/</code>{' '}
            prints all of them.
          </>
        }
      />

      <p>
        The pods run as uid 1001 with a read-only root filesystem, every Linux capability dropped and{' '}
        <code>seccompProfile: RuntimeDefault</code>. The NetworkPolicy allows ingress from the same
        namespace and egress only to DNS and HTTPS — which is what provider APIs and the OIDC issuer
        need and nothing else. <code>automountServiceAccountToken</code> is off, because effGen never
        calls the Kubernetes API.
      </p>

      <Callout type="tip" title="Validate before you apply">
        <p>
          <code>helm template … | kubectl apply --dry-run=client -f -</code> checks against a live
          cluster; <code>helm template … | kubeconform -strict -summary</code> checks with no cluster
          at all.
        </p>
      </Callout>

      <h2>AWS Lambda</h2>

      <p>
        <code>deploy/aws_lambda/handler.py</code> wraps the same FastAPI app in{' '}
        <a href="https://github.com/Kludex/mangum">Mangum</a>, so an API Gateway HTTP API v2 event
        becomes an ASGI call and back. Install the extra with{' '}
        <code>pip install "effgen[lambda]"</code>, plus whichever provider extra the function needs —{' '}
        the Lambda extra carries the adapter and the server, not any provider SDK.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`EFFGEN_DEV_MODE=1 python deploy/aws_lambda/_smoke_runner.py`}
        caption="Runs the handler against a bundled mock event. No AWS account, no SAM CLI."
      />

      <Terminal
        command="EFFGEN_DEV_MODE=1 python deploy/aws_lambda/_smoke_runner.py"
        output={`… UserWarning:
╔══════════════════════════════════════════════════════════════╗
║  ⚠  EFFGEN_DEV_MODE=1: AUTHENTICATION IS DISABLED  ⚠       ║
║  Never run with EFFGEN_DEV_MODE=1 in production!            ║
╚══════════════════════════════════════════════════════════════╝
  app = cls(app, *args, **kwargs)
{
  "event_path": "/health",
  "event_method": "GET",
  "event_format": "API Gateway HTTP API v2.0",
  "response_status_code": 200,
  "response_body_parsed": {
    "status": "ok",
    "version": "1.0.1"
  },
  "elapsed_ms": 6.9,
  "mangum_version": "0.21.0",
  "test": "PASS"
}`}
        caption={
          <>
            The warning comes from the dev-mode setting the runner defaults to; the interpreter path
            that preceded it on the captured line is elided. <code>PASS</code> means the real handler
            answered the real event with a 200.
          </>
        }
      />

      <CodeTabs
        tabs={[
          {
            label: 'build & deploy',
            language: 'bash',
            code: `sam build --template deploy/aws_lambda/sam-template.yaml --use-container

sam deploy \\
  --template deploy/aws_lambda/sam-template.yaml \\
  --stack-name effgen-prod \\
  --capabilities CAPABILITY_IAM \\
  --parameter-overrides \\
    Environment=prod \\
    OidcIssuer=https://YOUR_TENANT.auth0.com/ \\
    OidcClientId=YOUR_CLIENT_ID \\
    ApiKeySecretArn=arn:aws:secretsmanager:us-east-1:123456789012:secret:effgen-keys-XXXXXX`,
          },
          {
            label: 'invoke locally',
            language: 'bash',
            code: `sam local invoke EffgenFunction \\
  --template deploy/aws_lambda/sam-template.yaml \\
  --event tests/deploy/fixtures/apigw-http-event.json \\
  --env-vars <(echo '{"EffgenFunction":{"EFFGEN_DEV_MODE":"1"}}')`,
          },
          {
            label: 'tear down',
            language: 'bash',
            code: `sam delete --stack-name effgen-prod`,
          },
        ]}
      />

      <ParamTable
        nameLabel="Variable"
        params={[
          { name: 'EFFGEN_TIMEOUT_SECONDS', type: 'int', default: '29', description: 'Per-invocation budget. Keep it under the function Timeout.' },
          { name: 'EFFGEN_AUDIT_DIR', type: 'str', default: '/tmp/effgen-audit', description: 'The only writable place in a Lambda container.' },
          {
            name: 'EFFGEN_SANDBOX_BACKEND',
            type: 'str',
            default: 'subprocess',
            description: 'Set by the SAM template — a Lambda container cannot run Docker-in-Docker.',
          },
          { name: 'EFFGEN_DEV_MODE', type: 'str', default: '"0"', description: 'As everywhere else.' },
        ]}
        caption="The Lambda-specific settings, on top of the standard server environment."
      />

      <Callout type="note" title="Why the timeout budget exists">
        <p>
          The handler runs each request with a limit of{' '}
          <code>min(EFFGEN_TIMEOUT_SECONDS, remaining Lambda time − 0.5s)</code>. An overrun returns a{' '}
          <code>504</code> with a JSON body. Without it the runtime hard-kills the container and the
          caller sees an opaque <code>502</code> with nothing in it.
        </p>
      </Callout>

      <p>
        Provider keys go in Secrets Manager as one JSON secret and are passed by ARN at deploy time;
        the template grants the execution role <code>secretsmanager:GetSecretValue</code> on that
        secret only. The module preloads the provider registry once per container, so a cold start
        pays roughly 3–6 seconds and a warm <code>/health</code> answers in under 10 milliseconds.
      </p>

      <h2>Cloudflare Worker</h2>

      <p>
        effGen is Python and Workers run JavaScript, so <code>deploy/cloudflare/worker.js</code> is a
        proxy rather than a port: it sits in front of a Docker, Kubernetes or Lambda deployment and
        handles CORS, a structural JWT check, fixed-window rate limiting in KV, security headers and
        forwarding. Signature verification stays at the origin, which has the JWKS.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`cd deploy/cloudflare

wrangler kv namespace create RATE_LIMIT           # note the id it prints
wrangler kv namespace create RATE_LIMIT --preview # and the preview id

wrangler secret put EFFGEN_BACKEND_TOKEN          # optional M2M token to the origin
wrangler deploy`}
        caption={
          <>
            Put both ids into the <code>[[kv_namespaces]]</code> block of{' '}
            <code>wrangler.toml</code> first. <code>wrangler dev</code> runs it locally on{' '}
            <code>http://localhost:8787</code> with a local KV binding and no Cloudflare account.
          </>
        }
      />

      <ParamTable
        nameLabel="Variable"
        params={[
          {
            name: 'EFFGEN_BACKEND_URL',
            type: 'str',
            default: 'https://api.example.com',
            description: 'The upstream effGen server, with no trailing slash.',
          },
          {
            name: 'EFFGEN_CORS_ORIGINS',
            type: 'str',
            default: '*',
            description: 'Comma-separated allowed origins. Narrow it.',
          },
          { name: 'EFFGEN_RATE_LIMIT_WINDOW_SECONDS', type: 'str', default: '60', description: 'Fixed-window size.' },
          { name: 'EFFGEN_RATE_LIMIT_IP_MAX', type: 'str', default: '100', description: 'Requests per IP per window; 0 disables.' },
          { name: 'EFFGEN_RATE_LIMIT_TOKEN_MAX', type: 'str', default: '200', description: 'Requests per token per window; 0 disables.' },
          {
            name: 'EFFGEN_BACKEND_TOKEN',
            type: 'secret',
            description: (
              <>
                Injected upstream as <code>X-Effgen-Proxy-Token</code>. Set it with{' '}
                <code>wrangler secret put</code>, never in <code>[vars]</code>.
              </>
            ),
          },
        ]}
        caption={
          <>
            From <code>deploy/cloudflare/wrangler.toml</code>. The <code>staging</code> and{' '}
            <code>production</code> environments in that file override them per deployment.
          </>
        }
      />

      <p>
        The Worker's public paths are the server's own —{' '}
        <code>/health</code>, <code>/healthz</code>, <code>/livez</code>, <code>/ready</code>,{' '}
        <code>/readyz</code>, <code>/slo</code>, <code>/favicon.ico</code> — so one rule decides
        access at the edge and at the origin. <code>/metrics</code> is deliberately not in that list:
        the Worker forwards a backend token, so exempting it would hand metrics to an
        unauthenticated caller. Errors raised at the edge use the same{' '}
        <code>{'{"error": {message, type, param, code}}'}</code> envelope the origin uses, and a
        rate-limited request carries <code>Retry-After</code> and the three{' '}
        <code>X-RateLimit-*</code> headers.
      </p>

      <Callout type="warning" title="KV counters are best-effort">
        <p>
          Cloudflare KV is eventually consistent, so the per-IP and per-token windows are
          approximate across data centres. That is usually what you want for abuse control. For a
          limit that has to hold globally, back the counter with a Durable Object — or rely on the
          origin's own <code>--rate-limit</code>, which counts in one process.
        </p>
      </Callout>

      <Terminal
        command="pytest -q tests/deploy/test_cloudflare_worker.py"
        output={`...............................                                          [100%]
31 passed in 2.46s`}
        caption="CORS preflight, missing and malformed and expired tokens, upstream 200 and 502, public-path bypass, IP rate limiting and the security headers — run with Node, no Cloudflare account."
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>500</code> on every Lambda request
            </>,
            'Handler initialisation failed, usually a missing dependency in the zip.',
            <>
              Read the function log. <code>ModuleNotFoundError: mangum</code> means{' '}
              <code>effgen[lambda]</code> was not in the deployment package.
            </>,
          ],
          [
            <>
              <code>504</code> with a timeout-budget body
            </>,
            <>
              The request outran <code>EFFGEN_TIMEOUT_SECONDS</code>.
            </>,
            'Raise the function Timeout and that variable together, or use a faster model.',
          ],
          [
            <>
              A container answers <code>401</code> to everything
            </>,
            'Auth is on and nothing is configured, so the only valid key is the ephemeral one printed at startup — which a container restart replaces.',
            <>
              Set <code>EFFGEN_API_KEY</code> or the <code>EFFGEN_OIDC_*</code> variables.
            </>,
          ],
          [
            <>
              The container starts, then exits writing the audit log
            </>,
            <>
              <code>--read-only</code> with no writable audit directory.
            </>,
            <>
              Mount a tmpfs at <code>EFFGEN_AUDIT_DIR</code>, as in the production run above.
            </>,
          ],
          [
            'The HPA never scales on latency',
            <>
              <code>effgen_model_call_latency_seconds</code> is a Prometheus metric, not a
              Kubernetes one.
            </>,
            <>
              Deploy the Prometheus Adapter or KEDA to expose it. CPU scaling works without either.
            </>,
          ],
          [
            <>
              <code>502 bad_gateway</code> from the Worker
            </>,
            <>
              The Worker cannot reach <code>EFFGEN_BACKEND_URL</code>.
            </>,
            'Check the origin is up and the URL has no trailing slash.',
          ],
        ]}
      />

      <SeeAlso paths={['/api-server', '/openai-api', '/observability']} />
    </DocPage>
  );
}
