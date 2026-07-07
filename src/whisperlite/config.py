"""Typed configuration objects and strict YAML loading.

All runtime behaviour is driven by frozen-ish dataclasses defined here. YAML
files are deserialized through :func:`dataclass_from_dict`, which rejects
unknown keys so that typos in configuration files fail fast instead of being
silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


T = TypeVar("T")

# ---------------------------------------------------------------------------
# Audio front-end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioConfig:
    """Parameters of the log-mel spectrogram front-end.

    The defaults reproduce OpenAI Whisper's front-end: 16 kHz audio, 25 ms
    windows (400 samples), 10 ms hop (160 samples), 80 mel bins, and fixed
    30-second chunks.
    """

    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    chunk_length: float = 30.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.n_fft <= 0:
            raise ConfigError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ConfigError(f"hop_length must be positive, got {self.hop_length}")
        if self.hop_length > self.n_fft:
            raise ConfigError(
                f"hop_length ({self.hop_length}) must not exceed n_fft ({self.n_fft})"
            )
        if self.n_mels <= 0:
            raise ConfigError(f"n_mels must be positive, got {self.n_mels}")
        if self.chunk_length <= 0:
            raise ConfigError(f"chunk_length must be positive, got {self.chunk_length}")
        if self.chunk_samples % self.hop_length != 0:
            raise ConfigError(
                "chunk_length * sample_rate must be a multiple of hop_length "
                f"(got {self.chunk_samples} samples with hop {self.hop_length})"
            )
        if self.n_frames % 2 != 0:
            raise ConfigError(
                "the mel frame count must be even because the encoder downsamples "
                f"by 2 (got {self.n_frames} frames)"
            )

    @property
    def chunk_samples(self) -> int:
        """Number of waveform samples in one fixed-length chunk."""
        return round(self.chunk_length * self.sample_rate)

    @property
    def n_frames(self) -> int:
        """Number of mel frames produced for one chunk."""
        return self.chunk_samples // self.hop_length

    @property
    def n_audio_ctx(self) -> int:
        """Encoder sequence length after the stride-2 convolution."""
        return self.n_frames // 2


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Whisper-style encoder-decoder transformer hyperparameters."""

    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_audio_state: int = 384
    n_audio_head: int = 6
    n_audio_layer: int = 4
    n_vocab: int = 8192
    n_text_ctx: int = 448
    n_text_state: int = 384
    n_text_head: int = 6
    n_text_layer: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "n_mels",
            "n_audio_ctx",
            "n_audio_state",
            "n_audio_head",
            "n_audio_layer",
            "n_vocab",
            "n_text_ctx",
            "n_text_state",
            "n_text_head",
            "n_text_layer",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(f"{name} must be a positive integer, got {value!r}")
        if self.n_audio_state % self.n_audio_head != 0:
            raise ConfigError(
                f"n_audio_state ({self.n_audio_state}) must be divisible by "
                f"n_audio_head ({self.n_audio_head})"
            )
        if self.n_text_state % self.n_text_head != 0:
            raise ConfigError(
                f"n_text_state ({self.n_text_state}) must be divisible by "
                f"n_text_head ({self.n_text_head})"
            )
        if self.n_audio_state % 2 != 0:
            raise ConfigError("n_audio_state must be even for sinusoidal embeddings")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")


#: Named architecture presets, mirroring the layer/width scaling of the
#: corresponding official Whisper checkpoints (vocabulary and context sizes
#: are resolved separately from the tokenizer and audio configuration).
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "n_audio_state": 384,
        "n_audio_head": 6,
        "n_audio_layer": 4,
        "n_text_state": 384,
        "n_text_head": 6,
        "n_text_layer": 4,
    },
    "base": {
        "n_audio_state": 512,
        "n_audio_head": 8,
        "n_audio_layer": 6,
        "n_text_state": 512,
        "n_text_head": 8,
        "n_text_layer": 6,
    },
    "small": {
        "n_audio_state": 768,
        "n_audio_head": 12,
        "n_audio_layer": 12,
        "n_text_state": 768,
        "n_text_head": 12,
        "n_text_layer": 12,
    },
}

#: Fields of :class:`ModelConfig` that are derived from the audio config and
#: tokenizer rather than chosen freely.
_DERIVED_MODEL_FIELDS = ("n_mels", "n_audio_ctx", "n_vocab")


def resolve_model_config(
    preset: str | None,
    overrides: dict[str, Any] | None,
    audio: AudioConfig,
    n_vocab: int,
) -> ModelConfig:
    """Build the final :class:`ModelConfig` for training or inference.

    ``n_mels`` and ``n_audio_ctx`` are derived from *audio*, and ``n_vocab``
    from the tokenizer; supplying conflicting values in *overrides* is an
    error rather than a silent mismatch.
    """
    kwargs: dict[str, Any] = {}
    if preset is not None:
        if preset not in MODEL_PRESETS:
            raise ConfigError(
                f"unknown model preset {preset!r}; available: {sorted(MODEL_PRESETS)}"
            )
        kwargs.update(MODEL_PRESETS[preset])

    overrides = dict(overrides or {})
    valid_fields = {f.name for f in fields(ModelConfig)}
    unknown = set(overrides) - valid_fields
    if unknown:
        raise ConfigError(f"unknown model override keys: {sorted(unknown)}")

    derived = {"n_mels": audio.n_mels, "n_audio_ctx": audio.n_audio_ctx, "n_vocab": n_vocab}
    for name, value in derived.items():
        supplied = overrides.pop(name, None)
        if supplied is not None and supplied != value:
            raise ConfigError(
                f"model override {name}={supplied} conflicts with derived value {value}"
            )
    kwargs.update(overrides)
    kwargs.update(derived)
    return ModelConfig(**kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentConfig:
    """SpecAugment configuration applied to training mel spectrograms."""

    enabled: bool = True
    freq_masks: int = 2
    freq_width: int = 27
    time_masks: int = 2
    time_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.freq_masks < 0 or self.time_masks < 0:
            raise ConfigError("mask counts must be non-negative")
        if self.freq_width < 0:
            raise ConfigError("freq_width must be non-negative")
        if not 0.0 <= self.time_ratio <= 1.0:
            raise ConfigError(f"time_ratio must be in [0, 1], got {self.time_ratio}")


@dataclass(frozen=True)
class DataConfig:
    """Dataset and dataloader configuration."""

    train_manifest: str
    val_manifest: str
    batch_size: int = 16
    num_workers: int = 2
    max_text_tokens: int = 446
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ConfigError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.max_text_tokens <= 0:
            raise ConfigError(f"max_text_tokens must be positive, got {self.max_text_tokens}")


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer and learning-rate schedule configuration."""

    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    warmup_steps: int = 500
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.05
    clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ConfigError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ConfigError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.scheduler not in ("cosine", "linear", "constant"):
            raise ConfigError(
                f"scheduler must be one of cosine|linear|constant, got {self.scheduler!r}"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ConfigError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.clip_norm <= 0:
            raise ConfigError(f"clip_norm must be positive, got {self.clip_norm}")


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training run configuration (usually loaded from YAML)."""

    data: DataConfig
    tokenizer_path: str
    output_dir: str = "runs/default"
    audio: AudioConfig = field(default_factory=AudioConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model_preset: str | None = "tiny"
    model_overrides: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    max_steps: int = 10_000
    grad_accum: int = 1
    amp: str = "auto"
    device: str = "auto"
    log_interval: int = 25
    eval_interval: int = 500
    eval_max_batches: int = 50
    save_interval: int = 500
    keep_checkpoints: int = 3
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ConfigError(f"max_steps must be positive, got {self.max_steps}")
        if self.grad_accum <= 0:
            raise ConfigError(f"grad_accum must be positive, got {self.grad_accum}")
        if self.amp not in ("auto", "bf16", "fp16", "off"):
            raise ConfigError(f"amp must be one of auto|bf16|fp16|off, got {self.amp!r}")
        for name in ("log_interval", "eval_interval", "save_interval"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")
        if self.eval_max_batches <= 0:
            raise ConfigError("eval_max_batches must be positive")
        if self.keep_checkpoints <= 0:
            raise ConfigError("keep_checkpoints must be positive")


# ---------------------------------------------------------------------------
# Generic strict dict -> dataclass deserialization
# ---------------------------------------------------------------------------


def _unwrap_optional(tp: Any) -> Any:
    """Return ``X`` for ``Optional[X]``; otherwise return *tp* unchanged."""
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def dataclass_from_dict(cls: type[T], data: Any, *, path: str = "") -> T:
    """Recursively build a dataclass from a plain dict, rejecting unknown keys."""
    label = path or cls.__name__
    if not isinstance(data, dict):
        raise ConfigError(f"{label}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    field_map = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(field_map)
    if unknown:
        raise ConfigError(f"{label}: unknown keys {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        target = _unwrap_optional(hints[name])
        key_path = f"{label}.{name}"
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = dataclass_from_dict(target, value, path=key_path)
        elif get_origin(target) is tuple and isinstance(value, list | tuple):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required fields
        raise ConfigError(f"{label}: {exc}") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file that must contain a mapping at the top level."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top-level YAML value must be a mapping")
    return data


def load_train_config(path: str | Path) -> TrainConfig:
    """Load and validate a :class:`TrainConfig` from a YAML file."""
    return dataclass_from_dict(TrainConfig, load_yaml(path), path=str(path))
