"""Audio I/O, feature extraction, and augmentation."""

from whisperlite.audio.augment import spec_augment
from whisperlite.audio.features import (
    AudioError,
    load_audio,
    log_mel_spectrogram,
    mel_filterbank,
    pad_or_trim,
    resample_linear,
)

__all__ = [
    "AudioError",
    "load_audio",
    "log_mel_spectrogram",
    "mel_filterbank",
    "pad_or_trim",
    "resample_linear",
    "spec_augment",
]
