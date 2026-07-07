# Operations Runbook

## 1. Monitoring

### Prometheus metrics (`GET /metrics`)

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `whisperlite_requests_total` | counter | `route`, `status` | HTTP requests by route template and status code |
| `whisperlite_request_duration_seconds` | histogram | `route` | End-to-end request latency |
| `whisperlite_transcribed_audio_seconds_total` | counter | — | Seconds of audio successfully transcribed |
| `whisperlite_requests_inflight` | gauge | — | Requests currently being processed |
| `whisperlite_model_loaded` | gauge | — | 1 after the checkpoint loads, 0 otherwise |

Useful derived signals (PromQL):

```promql
# Error rate (5m)
sum(rate(whisperlite_requests_total{status=~"5.."}[5m]))
  / sum(rate(whisperlite_requests_total[5m]))

# p95 latency on the transcription route
histogram_quantile(0.95, sum by (le) (
  rate(whisperlite_request_duration_seconds_bucket{route="/v1/audio/transcriptions"}[5m])))

# Real-time factor processed (audio seconds per wall second)
rate(whisperlite_transcribed_audio_seconds_total[5m])
```

### Suggested alerts

| Alert | Condition (for 5–10 min) | First response |
| --- | --- | --- |
| High error rate | 5xx ratio > 1% | Check logs by `request_id`; roll back last deploy/model |
| Latency SLO breach | p95 > SLO on transcription route | Check `requests_inflight` vs `MAX_CONCURRENCY`; scale replicas |
| Not ready | `whisperlite_model_loaded == 0` on a serving pod | Pod stuck loading — describe pod, check checkpoint volume |
| Saturation | `requests_inflight` pinned at `MAX_CONCURRENCY` | Scale out; verify no single caller is flooding (429 counts) |
| Abuse | high `status="429"` rate for one key hash | Contact/rotate that key; consider gateway limits |

### Logs

JSON lines on stderr. Every request log carries `request_id`, `method`, `path`,
`status`, `duration_s`. Client error reports include the `X-Request-ID` header value —
grep for it to reconstruct the request's lifecycle.

## 2. Health & readiness

* `GET /healthz` — process liveness only. Restart the pod if it stops answering.
* `GET /readyz` — 200 once the checkpoint is loaded and warmed. During deploys, pods
  receive no traffic until ready.

## 3. Troubleshooting (serving)

| Symptom | Diagnosis | Fix |
| --- | --- | --- |
| Startup: `WHISPERLITE_API_KEYS is empty` | Fail-closed auth | Provide keys or explicitly disable auth for dev |
| Startup: `checkpoint not found` / `missing keys` | Wrong path or truncated file | Verify mount; re-download artifact; checkpoints are atomic so re-copy the source |
| Startup: `tokenizer vocab ... does not match model n_vocab` | Mixed artifacts from different runs | Use the checkpoint's embedded tokenizer (automatic) — this indicates a hand-edited file |
| 400 `invalid_audio` on valid-looking files | Container lacks the codec (rare; libsndfile is bundled with soundfile wheels) | Transcode client-side to wav/flac; check `pip show soundfile` |
| All requests 401 | Key mismatch/whitespace in env | Compare `sha256(key)[:16]` against the hash in access logs |
| Slow first request | No warmup (only if lifespan was bypassed) | Always start via `whisperlite serve` / `create_app` |
| Garbage transcripts, `avg_logprob` ≈ −5 | Untrained/wrong checkpoint, or audio far from training domain | `GET /v1/models` → `checkpoint_step`; run `whisperlite eval` on a known-good manifest |
| Memory growth under load | Concurrency × upload size | Lower `MAX_UPLOAD_MB` / `MAX_CONCURRENCY`; verify limits in the pod spec |
| CUDA OOM at load | Model bigger than GPU | Serve on CPU (`WHISPERLITE_DEVICE=cpu`) or use a smaller preset |

For training-side troubleshooting see [training.md §8](training.md#8-troubleshooting-training).

## 4. Maintenance

* **Dependency updates** — monthly: bump minima in `pyproject.toml`, run `make test`,
  `make lint`, rebuild the image. CI's 3-version Python matrix catches most breakage.
* **Key rotation** — quarterly (or on personnel change): append new key to
  `WHISPERLITE_API_KEYS`, roll clients, remove old key, rolling restart.
* **Model refresh** — retrain/fine-tune, run `whisperlite eval` on the frozen test
  manifest, compare WER against the incumbent, promote via the rollout procedure in
  [deployment.md §6](deployment.md#6-model-rollout-and-rollback).
* **Disk hygiene (training hosts)** — `keep_checkpoints` bounds run size automatically;
  archive `best.pt` + `metrics.jsonl` + `train_config.yaml` to object storage per run.

## 5. Upgrade guide

1. Read `CHANGELOG.md` for the target version; breaking changes are flagged.
2. Checkpoints carry `format_version` — a new library reading an old checkpoint either
   works or fails loudly with a clear message; never silently reinterprets.
3. Upgrade order: staging first, run the API test suite against staging
   (`pytest tests/test_api.py` pointed at a real checkpoint), then production canary
   (1 pod), then full rollout.
4. Rollback is a redeploy of the previous image + checkpoint (stateless service).

## 6. Backup & disaster recovery

| Asset | Backup | Restore |
| --- | --- | --- |
| Model checkpoints | Object storage, versioned, immutable | Repoint deployment |
| Training runs | `best.pt`, `metrics.jsonl`, `train_config.yaml` per run | Resume from latest step checkpoint |
| Manifests/corpus | Source-of-truth data lake | Regenerate manifests via scripts |
| API keys | Secret manager | Recreate `Secret`, rolling restart |

RTO for serving = time to schedule pods + model load (seconds to minutes). No databases,
no migrations, no stateful recovery.
