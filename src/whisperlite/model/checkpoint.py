"""Self-contained checkpoint serialization.

A checkpoint bundles everything needed for inference — model weights, model
and audio configuration, and the tokenizer — so a single ``.pt`` file can be
shipped to the serving tier without side-channel artifacts. Training state
(optimizer, scheduler, scaler, step counters) is stored under a separate
``train_state`` key and stripped for release checkpoints.

Checkpoints are written atomically (temp file + rename) so a crash mid-save
never corrupts the latest good checkpoint, and loaded with
``weights_only=True`` so a malicious checkpoint cannot execute code via
pickle.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from whisperlite.config import AudioConfig, ModelConfig, dataclass_from_dict
from whisperlite.model.asr import WhisperLite
from whisperlite.text.tokenizer import BPETokenizer
from whisperlite.version import __version__

CHECKPOINT_FORMAT_VERSION = 1
_REQUIRED_KEYS = ("format_version", "model_state", "model_config", "audio_config", "tokenizer")


class CheckpointError(ValueError):
    """Raised when a checkpoint file is missing keys or incompatible."""


def save_checkpoint(
    path: str | Path,
    *,
    model: WhisperLite,
    audio_config: AudioConfig,
    tokenizer: BPETokenizer,
    step: int = 0,
    train_state: dict[str, Any] | None = None,
) -> None:
    """Atomically write a checkpoint to *path*."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "whisperlite_version": __version__,
        "step": int(step),
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "audio_config": asdict(audio_config),
        "tokenizer": tokenizer.to_dict(),
    }
    if train_state is not None:
        payload["train_state"] = train_state
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load and structurally validate a checkpoint payload."""
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"checkpoint not found: {p}")
    payload = torch.load(p, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointError(f"{p}: checkpoint payload must be a dict")
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise CheckpointError(f"{p}: checkpoint is missing keys {missing}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"{p}: unsupported checkpoint format_version {payload['format_version']!r} "
            f"(this build reads version {CHECKPOINT_FORMAT_VERSION})"
        )
    return payload


def load_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[WhisperLite, BPETokenizer, AudioConfig]:
    """Reconstruct a ready-to-run model + tokenizer + audio config."""
    payload = load_checkpoint(path)
    model_config = dataclass_from_dict(ModelConfig, payload["model_config"])
    audio_config = dataclass_from_dict(AudioConfig, payload["audio_config"])
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    if tokenizer.vocab_size != model_config.n_vocab:
        raise CheckpointError(
            f"tokenizer vocab ({tokenizer.vocab_size}) does not match model "
            f"n_vocab ({model_config.n_vocab})"
        )
    model = WhisperLite(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, audio_config
