# syntax=docker/dockerfile:1
# effGen — Multi-stage Docker image
#
# Build (from repo root):
#   docker build -f deploy/docker/Dockerfile \
#     --build-arg VERSION=1.0.1 \
#     -t effgen:1.0.1 -t effgen:1.0 -t effgen:latest .
#
# Run (dev mode, no auth):
#   docker run --rm -p 8080:8080 \
#     -e EFFGEN_DEV_MODE=1 \
#     -v ~/.effgen:/home/effgen/.effgen \
#     effgen:latest
#
# Run (production — requires OIDC env vars):
#   docker run --rm -p 8080:8080 \
#     -e EFFGEN_OIDC_ISSUER=https://... \
#     -e EFFGEN_OIDC_CLIENT_ID=... \
#     -e EFFGEN_OIDC_JWKS_URI=https://... \
#     -v ~/.effgen:/home/effgen/.effgen:ro \
#     effgen:latest

# ─────────────────────────────────────────────────────────
# Stage 1: builder — install deps into a venv
# ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG VERSION=1.0.1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv in a known location
RUN python -m venv /opt/effgen/venv
ENV PATH="/opt/effgen/venv/bin:$PATH"

# Copy only the dependency manifests first (layer cache)
WORKDIR /build
COPY pyproject.toml setup.py requirements.txt ./
# Copy the package source so pip can build + install it into the venv.
COPY effgen/ ./effgen/
# Both READMEs are needed at build time: pyproject's readme is README_PYPI.md,
# and setup.py reads README.md for its long_description. Omitting README.md
# fails metadata generation with FileNotFoundError.
COPY README_PYPI.md README.md ./

# Install a minimal server subset to keep the image lean.
# FastAPI + uvicorn + starlette + cryptography are core deps; the "server"
# extra adds OIDC/JWT auth (authlib, PyJWT) + the Prometheus metrics client.
# Heavy ML libs (torch/transformers/vllm) come in via core/other extras and
# are only needed for local inference; override EXTRAS at build time as needed.
#
# NOTE: a *regular* (non-editable) install copies the package into the venv's
# site-packages. An editable install (`-e`) would bake the absolute build path
# (/build) into a finder file, which breaks at runtime once the source tree is
# relocated — so it must NOT be used here.
ARG EXTRAS="server"
RUN pip install --upgrade pip setuptools wheel && \
    pip install ".[${EXTRAS}]" --no-cache-dir

# ─────────────────────────────────────────────────────────
# Stage 2: runtime — lean image, non-root user
# ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG VERSION=1.0.1

LABEL org.opencontainers.image.title="effGen" \
      org.opencontainers.image.description="Agentic AI framework for Small Language Models" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/ctrl-gaurav/effGen" \
      org.opencontainers.image.licenses="Apache-2.0" \
      maintainer="Gaurav Srivastava <gks@vt.edu>"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/effgen/venv/bin:$PATH" \
    # Security defaults — override at runtime
    EFFGEN_DEV_MODE=0 \
    EFFGEN_SANDBOX_BACKEND=subprocess \
    # Bind address inside container
    EFFGEN_HOST=0.0.0.0 \
    EFFGEN_PORT=8080

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1001 effgen \
    && useradd --uid 1001 --gid effgen \
               --create-home --shell /usr/sbin/nologin \
               effgen

# Copy virtualenv from builder. The package lives inside the venv's
# site-packages (regular install), so the source tree is not needed at runtime.
COPY --from=builder --chown=effgen:effgen /opt/effgen/venv /opt/effgen/venv

# Data directories (audit log, config)
RUN mkdir -p /home/effgen/.effgen/audit \
    && chown -R effgen:effgen /home/effgen/.effgen

# Use non-root user
USER effgen
WORKDIR /home/effgen

EXPOSE 8080

# Healthcheck: poll /health every 30s; fail after 3 misses
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entrypoint: run uvicorn directly (no shell wrapper needed)
ENTRYPOINT ["python", "-m", "uvicorn", "effgen.server.app:create_app", \
            "--factory", \
            "--host", "0.0.0.0", \
            "--port", "8080", \
            "--log-level", "info"]
