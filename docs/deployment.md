# Deployment Guide

## 1. Configuration (environment variables)

All serving configuration is environment-driven (Twelve-Factor). The server validates
everything at startup and **fails fast** on misconfiguration.

| Variable | Default | Description |
| --- | --- | --- |
| `WHISPERLITE_CHECKPOINT` | — (required) | Path to the model checkpoint |
| `WHISPERLITE_DEVICE` | `auto` | `auto` / `cpu` / `cuda[:N]` |
| `WHISPERLITE_API_KEYS` | — | Comma-separated bearer keys, each ≥ 16 chars. **Required** unless auth is disabled |
| `WHISPERLITE_AUTH_ENABLED` | `1` | Set `0` only for local development |
| `WHISPERLITE_RATE_LIMIT_RPM` | `60` | Sustained requests/minute per key, per process |
| `WHISPERLITE_RATE_LIMIT_BURST` | `10` | Instantaneous burst per key |
| `WHISPERLITE_MAX_UPLOAD_MB` | `25` | Upload size cap (enforced before reading the body) |
| `WHISPERLITE_MAX_AUDIO_SECONDS` | `600` | Decoded-duration cap |
| `WHISPERLITE_BEAM_SIZE` | `1` | Default beam width (1–8) |
| `WHISPERLITE_TEMPERATURE` | `0.0` | Default sampling temperature |
| `WHISPERLITE_MAX_CONCURRENCY` | `2` | Concurrent transcriptions per worker process |
| `WHISPERLITE_CORS_ORIGINS` | empty | Comma-separated allowed origins (empty = CORS off) |
| `WHISPERLITE_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `WHISPERLITE_LOG_JSON` | `1` | JSON-lines logs (set `0` for human-readable) |

Generate keys with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

## 2. Bare process (dev / small deployments)

```bash
pip install "whisperlite[serve] @ ."           # or make install
export WHISPERLITE_API_KEYS=...
whisperlite serve --checkpoint models/best.pt --host 0.0.0.0 --port 8000
```

One process per GPU (or per few CPU cores). Run several processes behind nginx/haproxy
for more throughput; they are fully stateless.

## 3. Docker

```bash
docker build -t whisperlite:1.0.0 .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/models:ro" \
  -e WHISPERLITE_API_KEYS="$API_KEY" \
  whisperlite:1.0.0
```

Image properties: multi-stage build, CPU-only torch wheels (≈ 5–10× smaller than CUDA),
non-root user (uid 10001), `HEALTHCHECK` wired to `/healthz`, checkpoint expected at
`/models/model.pt` (override `WHISPERLITE_CHECKPOINT`).

For GPU serving: change the builder's torch index URL to the matching CUDA wheel index,
base the runtime stage on `nvidia/cuda:<ver>-runtime-ubuntu22.04` + Python, and run with
`--gpus all`.

### docker compose

`docker-compose.yml` wires the image, `.env`, a read-only model volume, resource limits,
and the health check:

```bash
cp .env.example .env      # set WHISPERLITE_API_KEYS
docker compose up --build
```

## 4. Kubernetes

Reference manifests in [`deploy/kubernetes.yaml`](../deploy/kubernetes.yaml):

* **Deployment** — 2+ replicas, rolling update with `maxUnavailable: 0`, non-root +
  read-only rootfs + dropped capabilities, resources requests/limits, Prometheus scrape
  annotations.
* **Probes** — `startupProbe` on `/readyz` tolerates slow model loads; readiness gates
  traffic on the model being loaded; liveness restarts a wedged process.
* **Service + HPA** — CPU-based autoscaling 2→8 replicas.
* API keys come from a `Secret` (`whisperlite-keys`); the checkpoint from a PVC (or an
  initContainer that downloads it from object storage — preferred for immutability).

```bash
kubectl create secret generic whisperlite-keys --from-literal=API_KEYS="$KEY1,$KEY2"
kubectl apply -f deploy/kubernetes.yaml
```

TLS termination, WAF, and *global* rate limiting belong on the Ingress/API gateway in
front of the Service.

## 5. CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

| Job | Checks |
| --- | --- |
| `lint` | `ruff check`, `ruff format --check`, `mypy` |
| `test` | full pytest suite with coverage on Python 3.10 / 3.11 / 3.12 (CPU torch wheels, pip cache) |
| `openapi-drift` | regenerates the OpenAPI spec and diffs it against the committed `docs/openapi.json` |
| `docker` | builds the production image (GHA layer cache), no push |

Suggested release flow (add a `release.yml` when a registry exists): tag `vX.Y.Z` →
build + push image with the tag and digest → update the Deployment image by digest.

## 6. Model rollout and rollback

Model artifacts should be **immutable and versioned** (e.g.
`s3://models/whisperlite/2026-07-01-step80k.pt`).

* **Roll out**: point the new ReplicaSet/compose service at the new checkpoint path or
  image tag. Readiness probes guarantee zero-downtime cutover.
* **Verify**: `GET /v1/models` reports `checkpoint_step`; compare WER on a canary
  manifest with `whisperlite eval` *before* promoting.
* **Roll back**: repoint to the previous artifact — pods are stateless, so rollback is
  exactly a redeploy (`kubectl rollout undo deployment/whisperlite`).

## 7. Capacity planning

Start with: 1 pod = 2 vCPU / 2 GiB (tiny model ≈ 150 MB fp32 + working memory), 
`WHISPERLITE_MAX_CONCURRENCY=2`. Load-test with your real audio-length distribution and
scale replicas until p95 latency meets the SLO; then set the HPA range around that.
Long uploads dominate memory (`MAX_UPLOAD_MB` × concurrency bounds the worst case).
