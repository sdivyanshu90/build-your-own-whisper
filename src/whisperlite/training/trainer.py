"""Step-based training loop with AMP, gradient accumulation, and checkpoints.

Design notes
------------
* **Step-based, not epoch-based** — speech corpora vary wildly in size, and
  LR schedules are defined over optimizer steps; the loader is simply cycled.
* **Mixed precision** — bf16 where supported (no loss scaling needed), fp16
  with dynamic loss scaling otherwise, controlled by ``TrainConfig.amp``.
* **Checkpoints are self-contained** (weights + configs + tokenizer) with
  training state attached, so both "resume training" and "ship to serving"
  are single-file operations. ``best.pt`` tracks the lowest validation WER.
* **Metrics** stream to stdout via logging and to ``metrics.jsonl`` in the
  run directory as machine-readable JSON lines.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from whisperlite.config import TrainConfig, resolve_model_config
from whisperlite.data.dataset import SpeechDataset, collate_batch
from whisperlite.data.manifest import read_manifest
from whisperlite.model.asr import WhisperLite
from whisperlite.model.checkpoint import load_checkpoint, save_checkpoint
from whisperlite.model.generation import GenerationOptions, generate
from whisperlite.text.tokenizer import BPETokenizer
from whisperlite.training.metrics import char_error_rate, word_error_rate
from whisperlite.training.scheduler import create_lr_scheduler
from whisperlite.utils import count_parameters, format_count, resolve_device, set_seed

logger = logging.getLogger(__name__)

_STEP_CKPT_RE = re.compile(r"^step-(\d+)\.pt$")


class Trainer:
    """Owns one training run described by a :class:`TrainConfig`."""

    def __init__(self, config: TrainConfig):
        self.config = config
        set_seed(config.seed)
        self.device = resolve_device(config.device)
        self.amp_dtype, self.amp_enabled = self._resolve_amp(config.amp)

        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = BPETokenizer.load(config.tokenizer_path)
        self.model_config = resolve_model_config(
            config.model_preset,
            config.model_overrides,
            config.audio,
            n_vocab=self.tokenizer.vocab_size,
        )
        if config.data.max_text_tokens + 2 > self.model_config.n_text_ctx:
            raise ValueError(
                f"max_text_tokens ({config.data.max_text_tokens}) + sot/eot does not fit "
                f"in n_text_ctx ({self.model_config.n_text_ctx})"
            )
        self.model = WhisperLite(self.model_config).to(self.device)
        logger.info(
            "model: %s parameters on %s (amp=%s)",
            format_count(count_parameters(self.model)),
            self.device,
            self.amp_dtype if self.amp_enabled else "off",
        )

        self.train_loader = self._build_loader(config.data.train_manifest, train=True)
        self.val_loader = self._build_loader(config.data.val_manifest, train=False)

        self.optimizer = torch.optim.AdamW(
            self._param_groups(),
            lr=config.optim.lr,
            betas=config.optim.betas,
            eps=config.optim.eps,
        )
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            name=config.optim.scheduler,
            warmup_steps=config.optim.warmup_steps,
            total_steps=config.max_steps,
            min_lr_ratio=config.optim.min_lr_ratio,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        self.global_step = 0
        self.best_wer = float("inf")
        self._metrics_path = self.output_dir / "metrics.jsonl"

        self._dump_resolved_config()
        if config.resume_from:
            self._resume(Path(config.resume_from))

    # -- Setup ---------------------------------------------------------------

    def _resolve_amp(self, mode: str) -> tuple[torch.dtype, bool]:
        if mode == "off":
            return torch.float32, False
        if mode == "auto":
            if self.device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16, True
                return torch.float16, True
            return torch.float32, False
        if mode == "bf16":
            return torch.bfloat16, True
        if mode == "fp16":
            if self.device.type != "cuda":
                raise ValueError("amp=fp16 requires a CUDA device; use bf16 or off on CPU")
            return torch.float16, True
        raise ValueError(f"unknown amp mode {mode!r}")

    def _param_groups(self) -> list[dict[str, Any]]:
        """Standard AdamW hygiene: no weight decay on 1-D params (biases, norms)."""
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.ndim >= 2 else no_decay).append(param)
        return [
            {"params": decay, "weight_decay": self.config.optim.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _build_loader(self, manifest_path: str, *, train: bool) -> DataLoader:
        dataset = SpeechDataset(
            read_manifest(manifest_path),
            self.tokenizer,
            self.config.audio,
            max_text_tokens=self.config.data.max_text_tokens,
            augment=self.config.data.augment,
            train=train,
        )
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=train,
            num_workers=self.config.data.num_workers,
            collate_fn=lambda items: collate_batch(items, self.tokenizer.pad_id),
            pin_memory=self.device.type == "cuda",
            drop_last=train and len(dataset) >= self.config.data.batch_size,
            generator=generator if train else None,
            persistent_workers=self.config.data.num_workers > 0,
        )

    def _dump_resolved_config(self) -> None:
        resolved = {
            "train": json.loads(json.dumps(asdict(self.config))),
            "model": asdict(self.model_config),
        }
        (self.output_dir / "train_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )

    # -- Core loop -----------------------------------------------------------

    def _autocast(self):
        if not self.amp_enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        mel = batch["mel"].to(self.device, non_blocking=True)
        tokens_in = batch["tokens_in"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        logits = self.model(mel, tokens_in)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_id,
        )

    def train(self) -> dict[str, Any]:
        """Run to ``max_steps``; returns a summary of the run."""
        cfg = self.config
        self.model.train()
        data_iter = self._cycle(self.train_loader)
        window_loss, window_start = 0.0, time.perf_counter()
        window_steps = 0

        while self.global_step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(cfg.grad_accum):
                batch = next(data_iter)
                with self._autocast():
                    loss = self._compute_loss(batch)
                self.scaler.scale(loss / cfg.grad_accum).backward()
                step_loss += float(loss.detach()) / cfg.grad_accum

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            window_loss += step_loss
            window_steps += 1

            if self.global_step % cfg.log_interval == 0:
                elapsed = time.perf_counter() - window_start
                utt_per_sec = (
                    window_steps * cfg.grad_accum * cfg.data.batch_size / max(elapsed, 1e-9)
                )
                record = {
                    "type": "train",
                    "step": self.global_step,
                    "loss": round(window_loss / max(window_steps, 1), 4),
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": round(float(grad_norm), 4),
                    "utt_per_sec": round(utt_per_sec, 2),
                }
                self._log_metrics(record)
                window_loss, window_steps = 0.0, 0
                window_start = time.perf_counter()

            if self.global_step % cfg.eval_interval == 0 or self.global_step == cfg.max_steps:
                self._eval_and_track_best()
                self.model.train()

            if self.global_step % cfg.save_interval == 0 or self.global_step == cfg.max_steps:
                self._save(self.checkpoint_dir / f"step-{self.global_step:08d}.pt")
                self._prune_checkpoints()

        return {
            "steps": self.global_step,
            "best_wer": self.best_wer,
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
        }

    @staticmethod
    def _cycle(loader: DataLoader):
        while True:
            yield from loader

    # -- Evaluation ----------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        """Validation loss plus greedy-decoding WER/CER on the val split."""
        max_batches = max_batches or self.config.eval_max_batches
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        references: list[str] = []
        hypotheses: list[str] = []
        decode_options = GenerationOptions(max_new_tokens=self.config.data.max_text_tokens + 8)
        for batch in self.val_loader:
            with self._autocast():
                total_loss += float(self._compute_loss(batch))
            n_batches += 1
            results = generate(
                self.model,
                self.tokenizer,
                batch["mel"].to(self.device),
                decode_options,
            )
            references.extend(batch["texts"])
            hypotheses.extend(result.text for result in results)
            if n_batches >= max_batches:
                break
        return {
            "loss": total_loss / max(n_batches, 1),
            "wer": word_error_rate(references, hypotheses),
            "cer": char_error_rate(references, hypotheses),
            "utterances": float(len(references)),
        }

    def _eval_and_track_best(self) -> None:
        metrics = self.evaluate()
        record = {
            "type": "eval",
            "step": self.global_step,
            "loss": round(metrics["loss"], 4),
            "wer": round(metrics["wer"], 4),
            "cer": round(metrics["cer"], 4),
            "utterances": int(metrics["utterances"]),
        }
        self._log_metrics(record)
        if metrics["wer"] < self.best_wer:
            self.best_wer = metrics["wer"]
            self._save(self.checkpoint_dir / "best.pt")
            logger.info("new best WER %.4f at step %d", self.best_wer, self.global_step)

    def _log_metrics(self, record: dict[str, Any]) -> None:
        logger.info(
            "%s", " ".join(f"{key}={value}" for key, value in record.items() if key != "type")
        )
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # -- Checkpointing -------------------------------------------------------

    def _save(self, path: Path) -> None:
        save_checkpoint(
            path,
            model=self.model,
            audio_config=self.config.audio,
            tokenizer=self.tokenizer,
            step=self.global_step,
            train_state={
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.global_step,
                "best_wer": self.best_wer,
                "torch_rng_state": torch.get_rng_state(),
            },
        )

    def _prune_checkpoints(self) -> None:
        """Keep the newest ``keep_checkpoints`` step files; never touch best.pt."""
        step_files = sorted(
            (p for p in self.checkpoint_dir.iterdir() if _STEP_CKPT_RE.match(p.name)),
            key=lambda p: int(_STEP_CKPT_RE.match(p.name).group(1)),  # type: ignore[union-attr]
        )
        for stale in step_files[: -self.config.keep_checkpoints]:
            stale.unlink()

    def _resume(self, path: Path) -> None:
        payload = load_checkpoint(path)
        if payload["model_config"] != asdict(self.model_config):
            raise ValueError(
                f"cannot resume: checkpoint model config {payload['model_config']} "
                f"differs from configured {asdict(self.model_config)}"
            )
        self.model.load_state_dict(payload["model_state"])
        train_state = payload.get("train_state")
        if train_state is None:
            raise ValueError(f"{path} has no training state; it is an inference checkpoint")
        self.optimizer.load_state_dict(train_state["optimizer"])
        self.scheduler.load_state_dict(train_state["scheduler"])
        self.scaler.load_state_dict(train_state["scaler"])
        self.global_step = int(train_state["step"])
        self.best_wer = float(train_state["best_wer"])
        rng_state = train_state.get("torch_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.to(torch.uint8).cpu())
        logger.info(
            "resumed from %s at step %d (best WER %.4f)", path, self.global_step, self.best_wer
        )
