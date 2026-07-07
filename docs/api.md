# API Reference

Machine-readable spec: [`docs/openapi.json`](openapi.json) (regenerate with
`make openapi`; CI fails if it drifts from the code). Interactive docs are served at
`/docs` (Swagger UI) and `/redoc`.

## Conventions

* **Base URL**: all business endpoints are versioned under `/v1`. Breaking changes ship
  as `/v2` alongside `/v1`, never in place.
* **Authentication**: `Authorization: Bearer <api-key>` on every `/v1` endpoint.
  Keys are configured server-side (`WHISPERLITE_API_KEYS`, comma-separated) and compared
  in constant time. Operational endpoints (`/healthz`, `/readyz`, `/metrics`) are
  unauthenticated and must only be reachable from the internal network.
* **Request IDs**: every response carries `X-Request-ID`; include it in bug reports —
  it is attached to every server-side log line for that request.
* **Errors**: every non-2xx response uses one envelope:

```json
{
  "error": {
    "code": "invalid_audio",
    "message": "could not decode audio: Format not recognised.",
    "request_id": "9f2b1c0a4d6e8f01"
  }
}
```

### Error codes

| HTTP | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_audio` | Upload could not be decoded, or was empty |
| 401 | `unauthorized` | Missing or invalid API key (`WWW-Authenticate: Bearer`) |
| 404 | `not_found` | Unknown path |
| 413 | `payload_too_large` | Upload exceeds `WHISPERLITE_MAX_UPLOAD_MB` |
| 413 | `audio_too_long` | Decoded duration exceeds `WHISPERLITE_MAX_AUDIO_SECONDS` |
| 422 | `validation_error` | Malformed request (missing file, out-of-range form field) |
| 422 | `invalid_parameters` | Semantically invalid combination (e.g. beam search with temperature > 0) |
| 429 | `rate_limited` | Token bucket empty; honor the `Retry-After` header |
| 500 | `internal_error` | Unexpected server error (logged with request id) |
| 503 | `not_ready` | Model not loaded (only during startup) |

### Rate limiting

Each API key gets a token bucket: `WHISPERLITE_RATE_LIMIT_BURST` immediate requests,
refilled at `WHISPERLITE_RATE_LIMIT_RPM / 60` per second, **per server process**. 429
responses include `Retry-After` (seconds). Put a gateway limiter in front for a global
bound across replicas.

---

## POST `/v1/audio/transcriptions`

Transcribe one uploaded audio file (any libsndfile format: wav, flac, ogg, …; any sample
rate — resampled server-side). Recordings longer than the model's chunk length are
transcribed window-by-window (see `chunks`).

**Request** — `multipart/form-data`:

| Field | Type | Required | Constraints | Description |
| --- | --- | --- | --- | --- |
| `file` | binary | yes | ≤ `WHISPERLITE_MAX_UPLOAD_MB` | The audio file |
| `temperature` | float | no | 0.0 – 2.0 | Sampling temperature; `0` = deterministic (default from server config) |
| `beam_size` | int | no | 1 – 8 | Beam width; `1` = greedy. Requires `temperature = 0` when > 1 |

**Response 200** (`TranscriptionResponse`):

```json
{
  "text": "hello world this is a test",
  "duration": 42.7,
  "chunks": [
    {"start": 0.0, "end": 30.0, "text": "hello world this", "avg_logprob": -0.19},
    {"start": 30.0, "end": 42.7, "text": "is a test", "avg_logprob": -0.24}
  ],
  "request_id": "9f2b1c0a4d6e8f01"
}
```

`avg_logprob` is the mean per-token log-probability of the decoded hypothesis; values
below ≈ −1.0 usually indicate unreliable output (silence, noise, out-of-domain audio).

**Example**:

```bash
curl -X POST https://asr.example.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@meeting.flac" \
  -F "beam_size=4"
```

```python
import httpx

response = httpx.post(
    "https://asr.example.com/v1/audio/transcriptions",
    headers={"Authorization": f"Bearer {api_key}"},
    files={"file": ("meeting.flac", open("meeting.flac", "rb"), "audio/flac")},
    data={"beam_size": 4},
    timeout=120,
)
response.raise_for_status()
print(response.json()["text"])
```

## GET `/v1/models`

Metadata about the loaded model — useful for client-side capability checks and for
verifying which checkpoint a deployment is running.

**Response 200** (`ModelInfoResponse`):

```json
{
  "name": "whisperlite",
  "version": "1.0.0",
  "parameters": 37184256,
  "checkpoint_step": 80000,
  "vocab_size": 8192,
  "sample_rate": 16000,
  "chunk_length": 30.0,
  "device": "cuda"
}
```

## Operational endpoints (unauthenticated — keep internal)

| Endpoint | Purpose | Success |
| --- | --- | --- |
| `GET /healthz` | Liveness: process responsive | `{"status": "ok"}` |
| `GET /readyz` | Readiness: model loaded | `{"status": "ready"}`, else 503 `not_ready` |
| `GET /metrics` | Prometheus exposition | text format, see docs/operations.md |

## Pagination and filtering

The API is intentionally request/response — there are no list endpoints, so pagination
and filtering do not apply. If batch/async transcription queues are added later they
will appear under `/v1/jobs` with cursor pagination.

## Versioning policy

* The **package version** (`GET /v1/models` → `version`) follows semver.
* The **URL version** (`/v1`) changes only for breaking wire-format changes.
* New optional response fields may appear in a minor version; clients must ignore
  unknown fields.
