"""Transcription service: model lifecycle + bounded-concurrency inference."""

from __future__ import annotations

import io
import logging
import threading

import torch

from whisperlite.audio.features import AudioError, load_audio
from whisperlite.model.generation import GenerationOptions
from whisperlite.serving.settings import ServingSettings
from whisperlite.transcribe import TranscriptionOutput, transcribe_waveform
from whisperlite.utils import resolve_device

logger = logging.getLogger(__name__)


class AudioTooLongError(ValueError):
    """Uploaded audio exceeds the configured duration limit."""


class TranscriptionService:
    """Owns the loaded model and runs inference with bounded concurrency.

    PyTorch inference releases the GIL inside kernels, so a small semaphore
    lets a few requests overlap while preventing a thundering herd from
    exhausting memory. All methods are thread-safe.
    """

    def __init__(self, settings: ServingSettings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        # Imported lazily so importing this module never touches the filesystem.
        from whisperlite.model.checkpoint import load_checkpoint, load_model

        self.model, self.tokenizer, self.audio_config = load_model(
            settings.checkpoint_path, self.device
        )
        self.checkpoint_step = int(load_checkpoint(settings.checkpoint_path).get("step", 0))
        self._semaphore = threading.Semaphore(settings.max_concurrency)
        logger.info(
            "loaded checkpoint %s (step %d) on %s",
            settings.checkpoint_path,
            self.checkpoint_step,
            self.device,
        )

    def warmup(self) -> None:
        """Run one dummy inference so the first user request isn't slow."""
        import numpy as np

        silence = np.zeros(self.audio_config.sample_rate // 2, dtype=np.float32)
        self.transcribe_waveform_array(silence, GenerationOptions(max_new_tokens=1))
        logger.info("warmup inference complete")

    def default_options(
        self, temperature: float | None = None, beam_size: int | None = None
    ) -> GenerationOptions:
        return GenerationOptions(
            beam_size=beam_size if beam_size is not None else self.settings.beam_size,
            temperature=temperature if temperature is not None else self.settings.temperature,
        )

    def transcribe_bytes(self, data: bytes, options: GenerationOptions) -> TranscriptionOutput:
        """Decode an uploaded audio file held in memory and transcribe it.

        Raises :class:`AudioError` for undecodable input and
        :class:`AudioTooLongError` for over-long audio; the API layer maps
        these to 400/413 responses.
        """
        waveform = load_audio(io.BytesIO(data), self.audio_config.sample_rate)
        duration = waveform.shape[0] / self.audio_config.sample_rate
        if duration > self.settings.max_audio_seconds:
            raise AudioTooLongError(
                f"audio is {duration:.1f}s long; the limit is "
                f"{self.settings.max_audio_seconds:.0f}s"
            )
        return self.transcribe_waveform_array(waveform, options)

    def transcribe_waveform_array(self, waveform, options: GenerationOptions):
        with self._semaphore, torch.inference_mode():
            return transcribe_waveform(
                self.model, self.tokenizer, waveform, self.audio_config, options=options
            )


__all__ = ["AudioError", "AudioTooLongError", "TranscriptionService"]
