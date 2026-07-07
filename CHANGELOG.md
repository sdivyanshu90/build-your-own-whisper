# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-07-07

### Added

- Whisper-style encoder–decoder ASR model (`tiny`/`base`/`small` presets) with
  SDPA attention and explicit KV caching.
- From-scratch audio front-end: Slaney mel filterbank, Whisper-exact log-mel
  normalization, SpecAugment.
- Trainable byte-level BPE tokenizer with fixed special-token layout and
  incremental-pair-count training.
- Decoding: batched greedy, temperature sampling, beam search with length penalty;
  long-form chunked transcription with timestamps.
- Step-based trainer: AMP (bf16/fp16), gradient accumulation/clipping, warmup +
  cosine/linear/constant LR schedules, WER/CER evaluation, atomic self-contained
  checkpoints (weights + configs + tokenizer), resume, retention pruning.
- CLI: `train`, `eval`, `transcribe`, `serve`, `tokenizer train`, `manifest validate`,
  `version`.
- FastAPI serving: bearer-key auth (constant-time), per-key token-bucket rate limiting,
  upload/duration caps, uniform error envelope, request IDs, security headers,
  Prometheus metrics, JSON logging, health/readiness probes, committed OpenAPI spec.
- Ops: multi-stage non-root Dockerfile, docker-compose, Kubernetes manifests (probes,
  HPA, hardening), GitHub Actions CI (lint, typecheck, 3-version test matrix, OpenAPI
  drift check, Docker build).
- Test suite: 216 tests including end-to-end training convergence, resume, and live
  API behavior; ~90% branch coverage.
- Documentation: architecture, API, training, deployment, security, operations,
  development guides.
