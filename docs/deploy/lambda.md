# Deploying effGen on AWS Lambda

effGen's FastAPI application can run on AWS Lambda via the
[Mangum](https://github.com/Kludex/mangum) ASGI adapter.  The same codebase
that runs with `uvicorn` locally is deployed unchanged — Mangum translates API
Gateway HTTP events into ASGI calls and converts the ASGI response back.

---

## Architecture

```
Client → API Gateway HTTP API (v2) → Lambda → Mangum → FastAPI (effGen)
```

All of the effGen server features (OIDC auth, RBAC, audit log, OpenAI-compat
`/v1` routes) work identically on Lambda.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.11 | `conda create -n effgen python=3.11` |
| AWS SAM CLI | <https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html> |
| AWS CLI v2 | <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html> |
| Docker (for `sam build`) | <https://docs.docker.com/get-docker/> |

---

## Quick-start (local smoke test)

```bash
# Install the Lambda extra (adds mangum + the FastAPI server deps)
pip install "effgen[lambda]"

# Run the handler against a mock API Gateway event
EFFGEN_DEV_MODE=1 python deploy/aws_lambda/_smoke_runner.py
```

> **Provider SDKs:** `effgen[lambda]` bundles only the Mangum adapter and the
> FastAPI server. To serve a model you must also install the relevant provider
> extra in the deployment package, e.g. `pip install "effgen[lambda,cerebras]"`.
> The `/health` smoke test above works without any provider SDK, but a
> `/v1/chat/completions` call against `cerebras/...` requires
> `cerebras-cloud-sdk` to be present.

Expected output:

```json
{
  "event_path": "/health",
  "response_status_code": 200,
  "response_body_parsed": {"status": "ok", "version": "1.0.1"},
  "test": "PASS"
}
```

---

## Build & deploy via SAM

### 1. Build the deployment package

```bash
# From the repo root
sam build \
  --template deploy/aws_lambda/sam-template.yaml \
  --use-container                  # builds inside a Lambda-compatible Docker image
```

SAM installs all Python dependencies into `.aws-sam/build/` and packages the
`effgen/` source tree alongside them.

### 2. Deploy to AWS

```bash
sam deploy \
  --template deploy/aws_lambda/sam-template.yaml \
  --stack-name effgen-prod \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    Environment=prod \
    OidcIssuer=https://YOUR_TENANT.auth0.com/ \
    OidcClientId=YOUR_CLIENT_ID \
    ApiKeySecretArn=arn:aws:secretsmanager:us-east-1:123456789012:secret:effgen-keys-XXXXXX
```

> **Tip:** Run `sam deploy --guided` on first deploy to write a `samconfig.toml`
> that remembers your parameters.

### 3. Test the deployed endpoint

```bash
ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name effgen-prod \
  --query "Stacks[0].Outputs[?OutputKey=='EffgenApiEndpoint'].OutputValue" \
  --output text)

curl "$ENDPOINT/health"
# {"status":"ok","version":"1.0.1"}
```

---

## Environment variables

All standard effGen server variables are supported.  Set them via the SAM
`Globals.Function.Environment.Variables` block or as SecretsManager references.

| Variable | Description | Default |
|----------|-------------|---------|
| `EFFGEN_DEV_MODE` | `"1"` disables JWT auth (dev only) | `"0"` |
| `EFFGEN_OIDC_ISSUER` | OIDC issuer URL | — |
| `EFFGEN_OIDC_CLIENT_ID` | JWT audience claim | — |
| `EFFGEN_OIDC_JWKS_URI` | JWKS endpoint (auto-discovered if blank) | — |
| `EFFGEN_TIMEOUT_SECONDS` | Per-invocation budget (< 30) | `29` |
| `EFFGEN_AUDIT_DIR` | Audit log directory | `/tmp/effgen-audit` |
| `EFFGEN_SANDBOX_BACKEND` | Code executor backend | `subprocess` |

> **Note:** `EFFGEN_SANDBOX_BACKEND=subprocess` is set automatically by the
> SAM template — Lambda containers cannot run Docker-in-Docker.

> **Timeout budget:** the handler enforces a per-invocation budget of
> `min(EFFGEN_TIMEOUT_SECONDS, remaining Lambda time − 0.5s)`. When a request
> overruns it, the handler returns a clean `504` with a JSON body rather than
> letting the Lambda runtime hard-kill the container (which would surface as an
> opaque `502` with no body). Keep `EFFGEN_TIMEOUT_SECONDS` below the Lambda
> function `Timeout`.

---

## Provider API keys via SecretsManager

Store provider keys as a JSON secret:

```bash
aws secretsmanager create-secret \
  --name effgen-keys \
  --secret-string '{"CEREBRAS_API_KEY":"cs-...","GROQ_API_KEY":"gsk_..."}'
```

Pass the ARN as `ApiKeySecretArn` at deploy time.  The SAM template grants
the Lambda execution role `secretsmanager:GetSecretValue` on that secret.

---

## Cold-start optimisation

The handler preloads the ProviderRegistry at module load time (once per
Lambda container lifetime).  This reduces per-request latency on warm
invocations.

Typical cold-start times:

| Phase | Typical latency |
|-------|----------------|
| Module import + registry preload | ~3–6 s |
| Warm invocation — `/health` | < 10 ms |
| Warm invocation — `/v1/chat/completions` | depends on model provider |

Provision Concurrency (AWS Lambda feature) can eliminate cold starts
entirely for latency-sensitive workloads.

---

## SAM local invoke

```bash
# Invoke with the bundled fixture event
sam local invoke EffgenFunction \
  --template deploy/aws_lambda/sam-template.yaml \
  --event tests/deploy/fixtures/apigw-http-event.json \
  --env-vars <(echo '{"EffgenFunction":{"EFFGEN_DEV_MODE":"1"}}')
```

---

## Validate the SAM template

```bash
# Requires SAM CLI
sam validate --template deploy/aws_lambda/sam-template.yaml

# Structural YAML check (no SAM CLI required)
python -c "
import yaml
from tests.deploy.test_lambda_handler import _load_cfn_yaml
t = _load_cfn_yaml(open('deploy/aws_lambda/sam-template.yaml').read())
print('Resources:', list(t['Resources'].keys()))
"
```

---

## Tear down

```bash
sam delete --stack-name effgen-prod
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| 500 on every request | Handler init failed | Check Lambda logs: `aws logs tail /aws/lambda/... --follow` |
| 401 Unauthorized | OIDC not configured | Set `EFFGEN_DEV_MODE=1` for testing or configure OIDC env vars |
| 403 RBAC denied | Role policy | Check `/rbac/roles` and `/rbac/policy` |
| 504 with "timeout budget" body | Request exceeded `EFFGEN_TIMEOUT_SECONDS` | Raise the Lambda `Timeout` and `EFFGEN_TIMEOUT_SECONDS`, or speed up the model call |
| `ModuleNotFoundError: mangum` | Wrong Lambda layer | Ensure `effgen[lambda]` or `mangum>=0.17.0` is in the deployment zip |
