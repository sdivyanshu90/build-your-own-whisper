# Security

## Threat model

Assets: the model checkpoint, API keys, service availability, and (transiently) user
audio. Adversaries: unauthenticated internet clients, clients with stolen keys, and
malicious artifacts (checkpoints/uploads). User audio is processed in memory only and
never persisted by the service.

## Controls by layer

### Authentication & authorization

* Bearer API keys, compared with `hmac.compare_digest` (constant-time — no timing
  side channel), minimum 16 characters enforced at startup.
* **Fail closed**: with auth enabled and no keys configured, the server refuses to
  start. Disabling auth requires an explicit `WHISPERLITE_AUTH_ENABLED=0`.
* Raw keys never appear in logs or internal maps — only a SHA-256 digest prefix is used
  for rate-limit bucketing and access logs.
* Authorization model is single-tier (any valid key may transcribe). Per-key scopes
  would be added at the gateway or as a keyed policy map in `ServingSettings`.

### Input handling (OWASP API4/API8)

* Upload size capped **before** the body is buffered: Content-Length pre-check plus a
  chunked read that aborts at the limit (413).
* Decoded audio duration capped (413) — prevents "zip-bomb" style tiny-file/huge-audio
  amplification (e.g. heavily compressed FLAC/OGG).
* Audio is decoded by libsndfile from bytes in memory; decode failures are a clean 400.
  Nothing derived from the upload is ever used as a path, shell argument, or query —
  no injection surface (SQLi/XSS/SSRF do not apply: no DB, no HTML rendering, no
  outbound requests from request data).
* Form parameters are range-validated (temperature 0–2, beam 1–8) and semantically
  cross-checked (beam × temperature) with stable 422 error codes.

### Rate limiting & DoS (OWASP API4)

* Per-key token bucket (burst + sustained RPM) per process; 429 + `Retry-After`.
* `WHISPERLITE_MAX_CONCURRENCY` bounds simultaneous inference, preventing memory
  exhaustion; excess requests queue in the thread pool.
* Bucket map is memory-bounded (idle buckets evicted past 10k keys).
* Global/distributed limits belong on the API gateway — documented, not faked in-process.

### Transport & headers

* TLS terminates at the ingress/gateway (the app never sees private keys).
* Every response sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, `Cache-Control: no-store`.
* CORS is **off by default**; enable per-origin via `WHISPERLITE_CORS_ORIGINS`.

### Artifact safety

* Checkpoints are loaded with `torch.load(weights_only=True)` — the unpickler only
  reconstructs tensors/primitives, closing the classic PyTorch pickle-RCE vector for
  untrusted checkpoints. Structure and format-version are validated before use.
* Tokenizer files are plain JSON with strict schema validation.
* Checkpoint/tokenizer writes are atomic; partially-written artifacts can't be loaded.

### Secrets management

* Secrets only enter via environment variables (K8s `Secret`, compose `.env` which is
  gitignored; `.env.example` documents shape without values).
* Rotation: `WHISPERLITE_API_KEYS` accepts multiple keys — add the new key, roll
  clients, remove the old key, restart (rolling).

### Logging & auditing

* Structured JSON access logs with request id, method, path, status, duration and the
  hashed caller identity — sufficient for audit trails without storing credentials or
  audio.
* 500-level errors log full tracebacks server-side but return only a generic envelope
  to clients (no stack/board disclosure).

### Container & runtime hardening

* Non-root user (uid 10001), read-only root filesystem, all capabilities dropped,
  `allowPrivilegeEscalation: false` (see `deploy/kubernetes.yaml`).
* Model volume mounted read-only.
* `/metrics`, `/healthz`, `/readyz` must not be exposed publicly — restrict at the
  ingress.

### Supply chain

* Runtime dependency surface is deliberately small (torch, numpy, soundfile, PyYAML +
  serve extras); versions have lower bounds in `pyproject.toml` — pin exact versions
  with a lock/constraints file in your deployment pipeline.
* CI builds the container from source; pin base images and actions by digest for
  stricter environments, and add `pip-audit`/Dependabot for CVE monitoring.
* `pre-commit` includes secret-detection and large-file checks.

## Reporting

See [SECURITY.md](../SECURITY.md) for the vulnerability disclosure process.
