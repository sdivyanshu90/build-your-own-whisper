"""SpecAugment: frequency and time masking on log-mel spectrograms.

Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for
Automatic Speech Recognition" (2019). Time warping is intentionally omitted —
it contributes little for its implementation cost, per the paper's ablations.
"""

from __future__ import annotations

import torch

from whisperlite.config import AugmentConfig


def _rand_int(high: int, generator: torch.Generator | None) -> int:
    """Uniform integer in ``[0, high)``; returns 0 when the range is empty."""
    if high <= 0:
        return 0
    return int(torch.randint(0, high, (1,), generator=generator).item())


def spec_augment(
    mel: torch.Tensor,
    config: AugmentConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a masked copy of an ``(n_mels, n_frames)`` spectrogram.

    Masked regions are filled with the spectrogram mean so that the global
    feature statistics stay stable, which matters because the model has no
    per-utterance feature normalization.
    """
    if not config.enabled:
        return mel
    if mel.ndim != 2:
        raise ValueError(f"expected (n_mels, n_frames), got shape {tuple(mel.shape)}")

    n_mels, n_frames = mel.shape
    out = mel.clone()
    fill = mel.mean()

    max_freq_width = min(config.freq_width, n_mels)
    for _ in range(config.freq_masks):
        width = _rand_int(max_freq_width + 1, generator)
        if width == 0:
            continue
        start = _rand_int(n_mels - width + 1, generator)
        out[start : start + width, :] = fill

    max_time_width = max(1, int(n_frames * config.time_ratio))
    for _ in range(config.time_masks):
        width = _rand_int(max_time_width + 1, generator)
        if width == 0:
            continue
        start = _rand_int(n_frames - width + 1, generator)
        out[:, start : start + width] = fill

    return out
