"""Audio front-end: filterbank, features, I/O, augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import make_sine, write_wav
from whisperlite.audio.augment import spec_augment
from whisperlite.audio.features import (
    AudioError,
    load_audio,
    log_mel_spectrogram,
    mel_filterbank,
    pad_or_trim,
    resample_linear,
)
from whisperlite.config import AugmentConfig


class TestMelFilterbank:
    def test_shape_and_nonnegativity(self):
        fb = mel_filterbank(16_000, 400, 80)
        assert fb.shape == (80, 201)
        assert fb.dtype == np.float32
        assert (fb >= 0).all()
        assert (fb.sum(axis=1) > 0).all()

    def test_filter_centers_increase_monotonically(self):
        fb = mel_filterbank(16_000, 400, 80)
        centers = fb.argmax(axis=1)
        assert (np.diff(centers) >= 0).all()

    def test_invalid_args_rejected(self):
        with pytest.raises(ValueError):
            mel_filterbank(16_000, 400, 0)


class TestLogMel:
    def test_shape_matches_config(self, audio_config):
        audio = make_sine(440.0, 1.0, audio_config.sample_rate)
        mel = log_mel_spectrogram(audio, audio_config)
        assert mel.shape == (audio_config.n_mels, audio_config.n_frames)
        assert mel.dtype == torch.float32

    def test_normalization_range(self, audio_config):
        audio = make_sine(440.0, 1.0, audio_config.sample_rate)
        mel = log_mel_spectrogram(audio, audio_config)
        # (log10 + 4) / 4 with 8 dB dynamic range => span <= 2.
        assert float(mel.max() - mel.min()) <= 2.0 + 1e-6

    def test_tone_energy_localized_in_frequency(self, audio_config):
        low = log_mel_spectrogram(make_sine(200.0, 1.0, 16_000), audio_config)
        high = log_mel_spectrogram(make_sine(4000.0, 1.0, 16_000), audio_config)
        assert low.mean(dim=1).argmax() < high.mean(dim=1).argmax()

    def test_accepts_torch_input(self, audio_config):
        audio = torch.from_numpy(make_sine(440.0, 1.0, 16_000))
        mel = log_mel_spectrogram(audio, audio_config)
        assert mel.shape == (80, 100)

    def test_very_short_audio_zero_padded(self, audio_config):
        mel = log_mel_spectrogram(np.zeros(10, dtype=np.float32), audio_config)
        assert mel.shape[0] == audio_config.n_mels

    def test_empty_audio_rejected(self, audio_config):
        with pytest.raises(AudioError):
            log_mel_spectrogram(np.zeros(0, dtype=np.float32), audio_config)

    def test_stereo_input_rejected(self, audio_config):
        with pytest.raises(AudioError, match="mono"):
            log_mel_spectrogram(np.zeros((2, 100), dtype=np.float32), audio_config)


class TestLoadAudio:
    def test_wav_roundtrip(self, tmp_path):
        original = make_sine(440.0, 0.5, 16_000)
        path = tmp_path / "tone.wav"
        write_wav(path, original, 16_000)
        loaded = load_audio(path, 16_000)
        assert loaded.dtype == np.float32
        assert abs(loaded.shape[0] - original.shape[0]) <= 1
        assert np.corrcoef(loaded[:7000], original[:7000])[0, 1] > 0.99

    def test_stereo_downmixed_to_mono(self, tmp_path):
        import soundfile as sf

        stereo = np.stack([make_sine(440.0, 0.2, 16_000)] * 2, axis=1)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, 16_000)
        loaded = load_audio(path, 16_000)
        assert loaded.ndim == 1

    def test_resampled_on_load(self, tmp_path):
        path = tmp_path / "tone8k.wav"
        write_wav(path, make_sine(440.0, 0.5, 8_000), 8_000)
        loaded = load_audio(path, 16_000)
        assert abs(loaded.shape[0] - 8_000) <= 2

    def test_garbage_bytes_rejected(self, tmp_path):
        path = tmp_path / "junk.wav"
        path.write_bytes(b"this is not audio data")
        with pytest.raises(AudioError):
            load_audio(path, 16_000)


class TestResample:
    def test_identity_when_rates_match(self):
        audio = make_sine(440.0, 0.1, 16_000)
        assert resample_linear(audio, 16_000, 16_000) is audio

    def test_length_scales_with_rate(self):
        audio = make_sine(440.0, 1.0, 8_000)
        up = resample_linear(audio, 8_000, 16_000)
        assert abs(up.shape[0] - 16_000) <= 2

    def test_preserves_tone_frequency(self):
        audio = make_sine(100.0, 1.0, 8_000)
        up = resample_linear(audio, 8_000, 16_000)
        # Count zero crossings: a 100 Hz tone has ~200 per second.
        crossings = int(np.sum(np.abs(np.diff(np.signbit(up)))))
        assert 190 <= crossings <= 210


class TestPadOrTrim:
    def test_pads_numpy(self):
        out = pad_or_trim(np.ones(5, dtype=np.float32), 8)
        assert out.shape == (8,)
        assert out[5:].sum() == 0

    def test_trims_torch(self):
        out = pad_or_trim(torch.ones(10), 4)
        assert out.shape == (4,)

    def test_noop_when_exact(self):
        x = torch.ones(6)
        assert pad_or_trim(x, 6) is x

    def test_other_axis(self):
        out = pad_or_trim(torch.ones(3, 5), 7, axis=0)
        assert out.shape == (7, 5)

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            pad_or_trim(torch.ones(3), 0)


class TestSpecAugment:
    def _mel(self):
        torch.manual_seed(7)
        return torch.randn(80, 100)

    def test_masks_change_values_but_not_shape(self):
        mel = self._mel()
        config = AugmentConfig(enabled=True, freq_masks=2, freq_width=10, time_masks=2)
        generator = torch.Generator().manual_seed(1)
        out = spec_augment(mel, config, generator=generator)
        assert out.shape == mel.shape
        assert not torch.equal(out, mel)

    def test_original_not_mutated(self):
        mel = self._mel()
        snapshot = mel.clone()
        spec_augment(mel, AugmentConfig(), generator=torch.Generator().manual_seed(1))
        assert torch.equal(mel, snapshot)

    def test_disabled_passthrough(self):
        mel = self._mel()
        assert spec_augment(mel, AugmentConfig(enabled=False)) is mel

    def test_masked_regions_use_mean_fill(self):
        mel = torch.ones(80, 100) * 3.0
        config = AugmentConfig(enabled=True, freq_masks=4, freq_width=20, time_masks=0)
        out = spec_augment(mel, config, generator=torch.Generator().manual_seed(3))
        assert torch.allclose(out, mel)  # mean of constant input == input

    def test_rejects_batched_input(self):
        with pytest.raises(ValueError):
            spec_augment(torch.randn(2, 80, 100), AugmentConfig())
