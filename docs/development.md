# Development Guide

## Setup

```bash
git clone <repo> && cd build-your-own-whisper
make install-dev          # venv + editable install + pre-commit hooks
make test                 # full suite (~10 s CPU)
make lint typecheck       # ruff + mypy
```

`make test-fast` skips the `slow`-marked end-to-end training tests during tight loops.

## Module walkthrough

| Module | Responsibility | Key entry points |
| --- | --- | --- |
| `config.py` | Frozen dataclass configs; strict YAML→dataclass loader (unknown keys error); model presets and derived-field resolution | `TrainConfig`, `load_train_config`, `resolve_model_config` |
| `audio/features.py` | Slaney mel filterbank (from formula), Whisper-exact log-mel, audio decode/resample, pad/trim | `log_mel_spectrogram`, `load_audio`, `mel_filterbank` |
| `audio/augment.py` | SpecAugment (freq/time masking, mean fill) | `spec_augment` |
| `text/tokenizer.py` | Byte-level BPE: fixed special IDs (pad=0, sot=1, eot=2, bytes 3–258, merges after); incremental-pair-count trainer; JSON serialization | `BPETokenizer.train/encode/decode/save/load` |
| `model/layers.py` | SDPA attention with caller-owned KV caches, pre-LN blocks, sinusoids, causal masks | `MultiHeadAttention`, `ResidualAttentionBlock`, `causal_mask` |
| `model/encoder.py` | Conv subsampling (×2) + transformer encoder | `AudioEncoder` |
| `model/decoder.py` | Causal decoder w/ cross-attention, learned positions, tied output projection, `DecoderCache` | `TextDecoder` |
| `model/asr.py` | Composition + GPT-2-style init | `WhisperLite` |
| `model/generation.py` | Batched greedy/sampling, per-utterance beam search, suppression, confidence | `generate`, `GenerationOptions` |
| `model/checkpoint.py` | Self-contained atomic checkpoints; `weights_only=True` loading; format versioning | `save_checkpoint`, `load_model` |
| `transcribe.py` | Long-form windowed transcription with timestamps | `transcribe_file`, `transcribe_waveform` |
| `data/manifest.py` | JSONL manifest parse/serialize with line-precise errors | `read_manifest` |
| `data/dataset.py` | Lazy audio decode, up-front token filtering, teacher-forcing collation | `SpeechDataset`, `collate_batch` |
| `training/trainer.py` | Step-based loop: AMP, accumulation, clipping, eval, checkpoints, resume, JSONL metrics | `Trainer.train/evaluate` |
| `training/scheduler.py` | Warmup + cosine/linear/constant `LambdaLR` | `create_lr_scheduler` |
| `training/metrics.py` | Text normalizer, Levenshtein, corpus WER/CER | `word_error_rate` |
| `serving/` | App factory, env settings (fail-fast), constant-time auth, token-bucket limiter, bounded-concurrency service, Prometheus, error envelope | `serving.app.create_app` |
| `cli.py` | argparse CLI, exit code 2 for domain errors | `main` |

## Testing strategy

216 tests in `tests/`, organized by behavior rather than coverage lines:

* **Unit** — pure functions (filterbank, WER, scheduler curves, token round-trips,
  masks) checked against hand-computed values.
* **Property/invariant** — decoder causality (future tokens can't affect past logits),
  KV-cache ≡ full forward (exact-match), batched greedy ≡ per-utterance greedy,
  tokenizer round-trips on arbitrary Unicode, determinism under fixed seeds.
* **Negative** — every domain error path (bad manifests, corrupt tokenizer/checkpoint
  files, invalid configs, out-of-range API params) asserts the specific error type and
  message fragment.
* **Integration (`-m slow`)** — a real 60-step training run on synthetic sine-wave
  speech asserting the loss drops >30%, checkpoint pruning, best-checkpoint tracking,
  and bit-compatible resume.
* **API** — full FastAPI stack over a real (tiny) checkpoint via `TestClient`: auth,
  429s with `Retry-After`, 413 upload/duration caps, error envelope, metrics
  exposition, OpenAPI presence.
* Fixtures are deliberately miniature (1 s chunks, 64-dim model) so the entire suite
  runs in seconds — no mocks around the model itself; the tests exercise production code.

Coverage: `make coverage` (branch coverage on; ~90% branches, higher line coverage).
New code should come with tests for its failure modes, not just its happy path.

## Coding standards

* Formatting/lint: `ruff` (line length 100, py310 target) — `make format`.
* Types: `mypy` clean on `src/whisperlite`; public functions are annotated.
* Errors: raise the module's domain error (`ConfigError`, `TokenizerError`,
  `ManifestError`, `AudioError`, `CheckpointError`, `GenerationError`) — all derive
  from `ValueError` so the CLI maps them to exit code 2.
* Docstrings explain *why* and contracts, not restatements of the code.
* No mutable global state; caches are `lru_cache` over immutable keys.

## Regenerating artifacts

* OpenAPI spec: `make openapi` (CI fails on drift).
* Tokenizer/checkpoints for manual testing: see `tests/conftest.py` for the minimal
  recipe, or run the quickstart on synthetic data.
