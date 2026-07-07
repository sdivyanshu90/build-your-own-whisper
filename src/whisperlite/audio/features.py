"""Waveform loading and log-mel spectrogram extraction.

The feature pipeline reproduces OpenAI Whisper's front-end exactly (up to the
mel filterbank, which is computed here from the Slaney formula instead of
being shipped as a binary asset):

1. 16 kHz mono float32 waveform, padded/trimmed to a fixed chunk length.
2. STFT with a periodic Hann window (``n_fft=400``, ``hop=160``, centered),
   dropping the final frame so a 30 s chunk yields exactly 3000 frames.
3. Power spectrum projected through an 80-bin Slaney-normalized mel
   filterbank.
4. ``log10`` with a 1e-10 floor, clamped to 8 dB of dynamic range below the
   per-chunk maximum, then affinely mapped to roughly ``[-1, 1]`` via
   ``(x + 4) / 4``.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Union, overload

import numpy as np
import soundfile as sf
import torch

from whisperlite.config import AudioConfig

AudioSource = Union[str, Path, BinaryIO, io.BytesIO]  # noqa: UP007 - runtime alias


class AudioError(ValueError):
    """Raised when audio cannot be decoded or is unusable."""


# ---------------------------------------------------------------------------
# Mel filterbank (Slaney-style, librosa-compatible)
# ---------------------------------------------------------------------------


def _hertz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    freq = np.asarray(freq, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = freq / f_sp
    log_region = min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) / logstep
    return np.where(freq >= min_log_hz, log_region, linear)


def _mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_hertz_to_mel`."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = mels * f_sp
    log_region = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log_region, linear)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build an ``(n_mels, n_fft // 2 + 1)`` triangular mel filterbank.

    Uses the Slaney mel scale with Slaney area normalization, matching
    ``librosa.filters.mel`` defaults (and therefore Whisper's shipped
    filters).
    """
    if n_mels <= 0 or n_fft <= 0 or sample_rate <= 0:
        raise ValueError("sample_rate, n_fft and n_mels must all be positive")
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_points = np.linspace(
        _hertz_to_mel(np.array(0.0)), _hertz_to_mel(np.array(sample_rate / 2.0)), n_mels + 2
    )
    hz_points = _mel_to_hertz(mel_points)

    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1][:, None]
    upper = ramps[2:] / fdiff[1:][:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: scale each filter to constant energy per channel.
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


@lru_cache(maxsize=8)
def _mel_filters(sample_rate: int, n_fft: int, n_mels: int, device: str) -> torch.Tensor:
    return torch.from_numpy(mel_filterbank(sample_rate, n_fft, n_mels)).to(device)


@lru_cache(maxsize=8)
def _hann_window(n_fft: int, device: str) -> torch.Tensor:
    return torch.hann_window(n_fft, periodic=True, device=torch.device(device))


# ---------------------------------------------------------------------------
# Waveform I/O
# ---------------------------------------------------------------------------


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample by linear interpolation.

    Linear interpolation is dependency-free and adequate for speech features;
    for maximum fidelity, resample offline with a polyphase filter (e.g.
    ``ffmpeg -ar 16000``) before building manifests.
    """
    if orig_sr == target_sr:
        return audio
    if orig_sr <= 0 or target_sr <= 0:
        raise AudioError(f"invalid sample rates: {orig_sr} -> {target_sr}")
    n_out = round(audio.shape[0] * target_sr / orig_sr)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_positions = np.arange(n_out, dtype=np.float64) * (orig_sr / target_sr)
    resampled = np.interp(src_positions, np.arange(audio.shape[0], dtype=np.float64), audio)
    return resampled.astype(np.float32)


def load_audio(source: AudioSource, target_sample_rate: int = 16_000) -> np.ndarray:
    """Decode an audio file (wav/flac/ogg/...) to mono float32 at *target_sample_rate*.

    Accepts a filesystem path or a binary file-like object (used by the API
    server to decode uploads in memory). Raises :class:`AudioError` on any
    decode failure so callers can map it to a client error.
    """
    try:
        data, sr = sf.read(source, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, ValueError, EOFError) as exc:
        raise AudioError(f"could not decode audio: {exc}") from exc
    if data.size == 0:
        raise AudioError("audio stream contains no samples")
    mono = data.mean(axis=1)
    return resample_linear(mono, int(sr), target_sample_rate)


@overload
def pad_or_trim(array: torch.Tensor, length: int, *, axis: int = ...) -> torch.Tensor: ...


@overload
def pad_or_trim(array: np.ndarray, length: int, *, axis: int = ...) -> np.ndarray: ...


def pad_or_trim(
    array: torch.Tensor | np.ndarray, length: int, *, axis: int = -1
) -> torch.Tensor | np.ndarray:
    """Pad with zeros or trim *array* to exactly *length* along *axis*."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    current = array.shape[axis]
    if current == length:
        return array
    if current > length:
        indexer: list[slice] = [slice(None)] * array.ndim
        indexer[axis] = slice(0, length)
        return array[tuple(indexer)]
    pad_amount = length - current
    if isinstance(array, torch.Tensor):
        pad = [0, 0] * array.ndim
        # torch.nn.functional.pad orders dims from last to first.
        pad_index = (array.ndim - 1 - (axis % array.ndim)) * 2 + 1
        pad[pad_index] = pad_amount
        return torch.nn.functional.pad(array, pad)
    widths = [(0, 0)] * array.ndim
    widths[axis % array.ndim] = (0, pad_amount)
    return np.pad(array, widths)


# ---------------------------------------------------------------------------
# Log-mel spectrogram
# ---------------------------------------------------------------------------


def log_mel_spectrogram(
    audio: np.ndarray | torch.Tensor,
    config: AudioConfig,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute an ``(n_mels, n_frames)`` normalized log-mel spectrogram.

    ``n_frames == len(audio) // hop_length`` (the trailing STFT frame is
    dropped, matching Whisper). The waveform should already be at
    ``config.sample_rate``.
    """
    if isinstance(audio, np.ndarray):
        waveform = torch.from_numpy(np.ascontiguousarray(audio))
    else:
        waveform = audio
    if waveform.ndim != 1:
        raise AudioError(f"expected a mono 1-D waveform, got shape {tuple(waveform.shape)}")
    if waveform.numel() == 0:
        raise AudioError("cannot compute features of an empty waveform")
    waveform = waveform.to(dtype=torch.float32)
    if device is not None:
        waveform = waveform.to(device)
    # Reflection padding used by the centered STFT needs n_fft // 2 samples on
    # each side; zero-pad very short inputs instead of erroring out.
    if waveform.shape[0] <= config.n_fft // 2:
        waveform = torch.nn.functional.pad(waveform, (0, config.n_fft // 2 + 1 - waveform.shape[0]))

    device_str = str(waveform.device)
    window = _hann_window(config.n_fft, device_str)
    stft = torch.stft(
        waveform,
        config.n_fft,
        config.hop_length,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2

    filters = _mel_filters(config.sample_rate, config.n_fft, config.n_mels, device_str)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
