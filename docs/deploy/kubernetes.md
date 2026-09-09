# Kubernetes / Helm Deployment

effGen includes a Helm chart at `deploy/k8s/helm/effgen/`.

## Prerequisites

- Kubernetes ≥ 1.24
- Helm ≥ 3.12
- `kubectl` configured to point at your cluster
- For HPA custom metrics: [KEDA](https://keda.sh/) or Prometheus Adapter

## Quick start (minikube)

```bash
# 1. Start minikube
minikube start --cpus=2 --memory=4g

# 2. Build and load the image into minikube
docker build -f deploy/docker/Dockerfile \
  --build-arg EXTRAS=server \
  -t effgen:1.0.1 .
minikube image load effgen:1.0.1

# 3. Install in dev mode
helm install effgen deploy/k8s/helm/effgen \
  --set env.EFFGEN_DEV_MODE=1 \
  --set image.pullPolicy=Never \
  -n effgen --create-namespace

# 4. Port-forward and test
kubectl port-forward -n effgen svc/effgen 8080:80 &
curl http://localhost:8080/health
# {"status":"ok"}
```

## Production install

```bash
helm install effgen deploy/k8s/helm/effgen \
  --set env.EFFGEN_OIDC_ISSUER=https://your-issuer.example.com \
  --set env.EFFGEN_OIDC_CLIENT_ID=your-client-id \
  --set env.EFFGEN_OIDC_JWKS_URI=https://your-issuer.example.com/.well-known/jwks.json \
  --set image.tag=1.0.1 \
  -n effgen --create-namespace
```

## Configuration reference

See `deploy/k8s/helm/effgen/values.yaml` for all options.

### Key values

| Key | Default | Description |
|-----|---------|-------------|
| `replicaCount` | `2` | Default pod count (overridden by HPA) |
| `autoscaling.enabled` | `true` | Enable HPA |
| `autoscaling.minReplicas` | `2` | HPA minimum |
| `autoscaling.maxReplicas` | `10` | HPA maximum |
| `autoscaling.customMetrics` | see values.yaml | Scale on `effgen_model_call_latency_seconds` |
| `networkPolicy.enabled` | `true` | Enforce ingress/egress rules |
| `podDisruptionBudget.enabled` | `true` | Ensure minAvailable=1 during disruptions |
| `env.EFFGEN_DEV_MODE` | `"0"` | Auth enabled by default |

## Auto-scaling

The HPA scales on two signals:

1. **CPU utilization** — scale up above 70%.
2. **Custom metric** `effgen_model_call_latency_seconds` — scale up when average latency exceeds 2s.

For the custom metric to work, deploy the [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter) and configure it to expose `effgen_model_call_latency_seconds` from your Prometheus instance.

```yaml
# prometheus-adapter ConfigMap excerpt
rules:
  - seriesQuery: 'effgen_model_call_latency_seconds_bucket{..}'
    name:
      matches: "^(.*)_bucket$"
      as: "${1}"
    metricsQuery: histogram_quantile(0.95, sum(rate(<<.Series>>[2m])) by (le, pod))
```

## Upgrading

```bash
helm upgrade effgen deploy/k8s/helm/effgen \
  --set image.tag=0.2.11 \
  -n effgen
```

## Uninstalling

```bash
helm uninstall effgen -n effgen
kubectl delete namespace effgen
```

## Security notes

- Pods run as `uid=1001` (non-root), `readOnlyRootFilesystem=true`.
- All Linux capabilities dropped.
- `seccompProfile: RuntimeDefault` applied.
- NetworkPolicy limits ingress to same-namespace pods and allows egress only to DNS + HTTPS (for LLM provider APIs and OIDC).
- PDB ensures at least 1 pod is always available during node drain.
- `ServiceAccount.automountServiceAccountToken=false` — effGen doesn't need the K8s API.

## Linting the chart

```bash
helm lint deploy/k8s/helm/effgen/
# 1 chart(s) linted, 0 chart(s) failed

# Validate rendered manifests against a live cluster:
helm template effgen-test deploy/k8s/helm/effgen/ | kubectl apply --dry-run=client -f -

# Or validate offline (no cluster needed) with kubeconform:
helm template effgen-test deploy/k8s/helm/effgen/ \
  | kubeconform -strict -summary -kubernetes-version 1.29.0
# Summary: 6 resources found parsing stdin - Valid: 6, Invalid: 0, Errors: 0, Skipped: 0
```
