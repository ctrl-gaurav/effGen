# Docker Deployment

effGen ships a Docker image built with a multi-stage Dockerfile.

## Quick start

```bash
# Dev mode (auth disabled — fine for local testing)
docker build -f deploy/docker/Dockerfile \
  --build-arg EXTRAS=server \
  -t effgen:1.0.1 .

docker run --rm -p 8080:8080 \
  -e EFFGEN_DEV_MODE=1 \
  effgen:1.0.1

curl http://localhost:8080/health
# {"status":"ok"}
```

## Image tags

| Tag | Description |
|-----|-------------|
| `effgen:1.0.1` | Exact release |
| `effgen:1.0` | Latest 1.0.x |
| `effgen:latest` | Latest release |

## Build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `VERSION` | `1.0.1` | Label baked into image metadata |
| `EXTRAS` | `server` | pip extras to install (auth + Prometheus metrics; FastAPI/uvicorn are core) |

To add a provider SDK (e.g. Cerebras):
```bash
docker build -f deploy/docker/Dockerfile \
  --build-arg EXTRAS=server,cerebras \
  -t effgen:1.0.1-full .
```

## Production run

```bash
docker run --rm -p 8080:8080 \
  -e EFFGEN_DEV_MODE=0 \
  -e EFFGEN_OIDC_ISSUER=https://your-issuer.example.com \
  -e EFFGEN_OIDC_CLIENT_ID=your-client-id \
  -e EFFGEN_OIDC_JWKS_URI=https://your-issuer.example.com/.well-known/jwks.json \
  -v ~/.effgen:/home/effgen/.effgen:ro \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /home/effgen/.effgen/audit \
  effgen:1.0.1
```

## Security posture

- **Non-root user**: runs as `effgen` (uid=1001).
- **Read-only filesystem**: use `--read-only` with `--tmpfs` for `/tmp` and audit log.
- **No secrets in image**: API keys must be injected at runtime via environment variables or mounted files.
- **HEALTHCHECK**: polls `/health` every 30s; 3 failures mark the container unhealthy.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EFFGEN_DEV_MODE` | `0` | Set to `1` to disable auth (loud warning) |
| `EFFGEN_PORT` | `8080` | Listening port |
| `EFFGEN_OIDC_ISSUER` | — | OIDC issuer URL (required in prod) |
| `EFFGEN_OIDC_CLIENT_ID` | — | JWT audience |
| `EFFGEN_OIDC_JWKS_URI` | — | JWKS endpoint (auto-discovered if blank) |
| `EFFGEN_SANDBOX_BACKEND` | `subprocess` | `docker`, `subprocess`, or `off` |
| `EFFGEN_AUDIT_DIR` | `~/.effgen/audit` | Audit log directory |

## Docker Compose

A ready-to-run `deploy/docker/docker-compose.yml` builds the image and starts
the server (dev mode, auth disabled, published on loopback only) in one command:

```bash
docker compose -f deploy/docker/docker-compose.yml up --build
curl http://localhost:8080/health   # {"status":"ok"}
```

The file:

```yaml
services:
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
      retries: 3
```

For a deployment that faces a network, drop `EFFGEN_DEV_MODE`, set the
`EFFGEN_OIDC_*` variables (see "Production run" above), and widen the port
mapping to `"8080:8080"`.

## Building for multiple architectures

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f deploy/docker/Dockerfile \
  -t effgen:1.0.1 \
  --push .
```
