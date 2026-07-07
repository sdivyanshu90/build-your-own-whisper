# Training Guide

End-to-end: raw audio → manifests → tokenizer → trained checkpoint → evaluation.

## 1. Data preparation

### Manifest format

Training consumes JSONL manifests, one utterance per line:

```json
{"audio_filepath": "/abs/path/utt1.flac", "text": "reference transcript", "duration": 4.21}
```

| Field | Required | Notes |
| --- | --- | --- |
| `audio_filepath` | yes | Any libsndfile-readable format; absolute paths recommended |
| `text` | yes | Reference transcript; normalize casing/punctuation consistently |
| `duration` | no | Seconds; enables filtering over-long clips without opening files |

Validate before training:

```bash
whisperlite manifest validate data/manifests/train.jsonl --check-audio
```

### Recommendations

* Resample offline to 16 kHz with a proper filter (`ffmpeg -ar 16000 -ac 1`) — the
  built-in linear resampler is a convenience fallback, not a mastering tool.
* Keep utterances **within one chunk length** (30 s default); longer ones are skipped
  with a logged count.
* Lowercase the transcripts (the LibriSpeech script does) — WER scoring normalizes
  case anyway, and a smaller effective character set helps small models.
* LibriSpeech shortcut: `python scripts/prepare_librispeech.py --root data/librispeech
  --subset train-clean-100 --subset dev-clean`.

## 2. Tokenizer

```bash
whisperlite tokenizer train \
    --manifest data/manifests/train.jsonl \
    --vocab-size 8192 --output artifacts/tokenizer.json
```

* Train the tokenizer on the **training transcripts only** (no test leakage).
* `--vocab-size` includes the 3 specials + 256 bytes; 4k–8k works well for monolingual
  English. Larger vocabularies shorten sequences (faster decoding) but grow the softmax.
* The tokenizer is embedded in every checkpoint afterwards — you never ship the JSON to
  serving, but keep it with the run for reproducibility.

## 3. Configuration reference

`whisperlite train --config <yaml>` loads a `TrainConfig`. **Unknown keys are hard
errors** — typos cannot silently no-op. All fields, with defaults:

```yaml
output_dir: runs/default        # run artifacts: checkpoints/, metrics.jsonl, train_config.yaml
tokenizer_path: <required>      # tokenizer JSON from step 2
seed: 42                        # seeds python/numpy/torch (+cuda)

data:
  train_manifest: <required>
  val_manifest: <required>
  batch_size: 16                # per step, before grad_accum
  num_workers: 2                # DataLoader processes (0 = in-process)
  max_text_tokens: 446          # transcripts longer than this (BPE tokens) are skipped;
                                # must satisfy max_text_tokens + 2 <= n_text_ctx
  augment:                      # SpecAugment (training split only)
    enabled: true
    freq_masks: 2               # number of frequency masks
    freq_width: 27              # max mel bins per mask
    time_masks: 2               # number of time masks
    time_ratio: 0.05            # max fraction of frames per mask

audio:                          # must match between training and any fine-tune
  sample_rate: 16000
  n_fft: 400
  hop_length: 160
  n_mels: 80
  chunk_length: 30.0            # seconds; smaller = cheaper training, shorter max utterance

model_preset: tiny              # tiny | base | small | null (fully manual)
model_overrides: {}             # any ModelConfig field, e.g. {dropout: 0.1, n_text_ctx: 448}
                                # n_mels/n_audio_ctx/n_vocab are derived; conflicts error

optim:
  lr: 1.0e-3                    # peak LR after warmup
  weight_decay: 0.01            # applied to >=2-D params only (not biases/norms)
  betas: [0.9, 0.98]
  eps: 1.0e-8
  warmup_steps: 500             # linear ramp from ~0
  scheduler: cosine             # cosine | linear | constant
  min_lr_ratio: 0.05            # LR floor as a fraction of peak
  clip_norm: 1.0                # global grad-norm clip

max_steps: 10000                # optimizer steps (not micro-batches)
grad_accum: 1                   # micro-batches per step; effective batch = batch_size*grad_accum
amp: auto                       # auto | bf16 | fp16 | off  (auto: bf16 on capable CUDA, else fp16; off on CPU)
device: auto                    # auto | cpu | cuda | cuda:N
log_interval: 25                # steps between train metric records
eval_interval: 500              # steps between validation (loss + greedy WER/CER)
eval_max_batches: 50            # cap on validation batches per eval
save_interval: 500              # steps between periodic checkpoints
keep_checkpoints: 3             # newest step checkpoints kept (best.pt never pruned)
resume_from: null               # checkpoint path (or use --resume)
```

## 4. What a run produces

```
runs/tiny/
├── train_config.yaml       # fully-resolved config incl. derived ModelConfig
├── metrics.jsonl           # {"type":"train",step,loss,lr,grad_norm,utt_per_sec}
│                           # {"type":"eval",step,loss,wer,cer,utterances}
└── checkpoints/
    ├── step-00002000.pt    # periodic, contains full training state
    ├── step-00004000.pt
    └── best.pt             # lowest validation WER so far
```

Every checkpoint is **self-contained** (weights + model config + audio config +
tokenizer) and loads for inference with `whisperlite transcribe/serve` directly.
Checkpoint writes are atomic (temp + rename) — a crash cannot corrupt them.

## 5. Mixed precision, throughput, hardware

* `amp: auto` picks bf16 on Ampere+ GPUs (no loss-scaling pathologies), fp16 with
  dynamic loss scaling on older GPUs, and full fp32 on CPU.
* Effective batch sizes of 64–256 utterances (via `grad_accum`) are typical for
  `tiny`/`base`; scale `lr` roughly with the square root of the batch size.
* Guidance: `tiny` + LibriSpeech train-clean-100 (~100 h) reaches useful dev-clean WER
  in ~80k steps on one 12–24 GB GPU. `base` wants ≥ 360 h of speech to shine.
* Watch `utt_per_sec` in the train records; if it drops when `num_workers > 0`, the
  bottleneck is audio decoding — pre-resample offline or raise workers.

## 6. Resume and reproducibility

```bash
whisperlite train --config configs/train_tiny.yaml \
    --resume runs/tiny/checkpoints/step-00040000.pt
```

Resume restores model, optimizer, scheduler, scaler, step counter, best-WER and the
torch RNG state. The architecture in the config must match the checkpoint exactly —
mismatches are rejected, not silently reinterpreted. For end-to-end determinism also
keep `num_workers` and batch size unchanged.

## 7. Evaluation

During training, every eval interval computes validation loss and **greedy** WER/CER
(fast, comparable across steps). For final numbers use beam search:

```bash
whisperlite eval --checkpoint runs/tiny/checkpoints/best.pt \
    --manifest data/manifests/test.jsonl --beam-size 4 --batch-size 8
# → {"utterances": 2620, "wer": 0.087, "cer": 0.031}
```

Scoring normalizes both sides (lowercase, punctuation stripped except apostrophes,
whitespace collapsed) so formatting differences don't count as errors.

## 8. Troubleshooting training

| Symptom | Likely cause / fix |
| --- | --- |
| Loss stuck near `ln(vocab_size)` | LR too low, or data/tokenizer mismatch — spot-check `tokenizer.decode(encode(text))` round-trips |
| Loss spikes then NaN (fp16) | Use `amp: bf16` or `off`; lower `lr`; `clip_norm` is already on |
| WER 1.0 while loss falls | Model emits empty/short hypotheses early in training — normal before convergence; check again after warmup×10 steps |
| `dataset filtering: skipped N...` warnings | Clips longer than `chunk_length` or transcripts over `max_text_tokens`; fix data or raise limits |
| OOM on GPU | Halve `batch_size`, double `grad_accum` (same effective batch) |
| Eval very slow | Lower `eval_max_batches`; eval decodes autoregressively, unlike training |
