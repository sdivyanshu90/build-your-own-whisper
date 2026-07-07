"""Long-form transcription: fixed-window chunking over arbitrary-length audio.

The model consumes fixed-length chunks (30 s by default). Longer recordings
are split into consecutive non-overlapping windows, each window is decoded
independently, and the texts are concatenated. This is intentionally simpler
than Whisper's timestamp-conditioned sliding window — it needs no timestamp
tokens in the vocabulary — at the cost of possible word breakage exactly at
chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from whisperlite.audio.features import AudioSource, load_audio, log_mel_spectrogram, pad_or_trim
from whisperlite.config import AudioConfig
from whisperlite.model.asr import WhisperLite
from whisperlite.model.generation import GenerationOptions, generate
from whisperlite.text.tokenizer import BPETokenizer


@dataclass(frozen=True)
class ChunkTranscription:
    """Transcription of one fixed-length window of audio."""

    start: float
    end: float
    text: str
    avg_logprob: float


@dataclass(frozen=True)
class TranscriptionOutput:
    """Full-recording transcription."""

    text: str
    duration: float
    chunks: tuple[ChunkTranscription, ...]


def transcribe_waveform(
    model: WhisperLite,
    tokenizer: BPETokenizer,
    waveform: np.ndarray,
    audio_config: AudioConfig,
    *,
    options: GenerationOptions | None = None,
    batch_size: int = 4,
) -> TranscriptionOutput:
    """Transcribe a mono float32 waveform at ``audio_config.sample_rate``."""
    if waveform.ndim != 1:
        raise ValueError(f"expected a mono waveform, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("cannot transcribe an empty waveform")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    options = options or GenerationOptions()

    device = next(model.parameters()).device
    chunk_samples = audio_config.chunk_samples
    duration = waveform.shape[0] / audio_config.sample_rate
    n_chunks = max(1, -(-waveform.shape[0] // chunk_samples))  # ceil division

    mels = []
    spans: list[tuple[float, float]] = []
    for index in range(n_chunks):
        start = index * chunk_samples
        segment = waveform[start : start + chunk_samples]
        spans.append(
            (
                start / audio_config.sample_rate,
                min((start + segment.shape[0]) / audio_config.sample_rate, duration),
            )
        )
        padded = pad_or_trim(segment, chunk_samples)
        mels.append(log_mel_spectrogram(padded, audio_config, device=device))

    # Beam search decodes utterances one at a time internally, so batching
    # only helps greedy/sampling paths.
    effective_batch = 1 if options.beam_size > 1 else batch_size
    chunks: list[ChunkTranscription] = []
    for offset in range(0, len(mels), effective_batch):
        batch = torch.stack(mels[offset : offset + effective_batch])
        for i, result in enumerate(generate(model, tokenizer, batch, options)):
            start_s, end_s = spans[offset + i]
            chunks.append(
                ChunkTranscription(
                    start=round(start_s, 3),
                    end=round(end_s, 3),
                    text=result.text,
                    avg_logprob=result.avg_logprob,
                )
            )

    text = " ".join(chunk.text for chunk in chunks if chunk.text).strip()
    return TranscriptionOutput(text=text, duration=round(duration, 3), chunks=tuple(chunks))


def transcribe_file(
    model: WhisperLite,
    tokenizer: BPETokenizer,
    source: AudioSource | str | Path,
    audio_config: AudioConfig,
    *,
    options: GenerationOptions | None = None,
    batch_size: int = 4,
) -> TranscriptionOutput:
    """Decode an audio file (any libsndfile-supported format) and transcribe it."""
    waveform = load_audio(source, target_sample_rate=audio_config.sample_rate)
    return transcribe_waveform(
        model, tokenizer, waveform, audio_config, options=options, batch_size=batch_size
    )
