"""Long-form transcription: chunking, spans, file decoding."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_sine, write_wav
from whisperlite.model.generation import GenerationOptions
from whisperlite.transcribe import transcribe_file, transcribe_waveform


class TestTranscribeWaveform:
    def test_single_chunk(self, model, tokenizer, audio_config):
        waveform = make_sine(440.0, 0.6, audio_config.sample_rate)
        output = transcribe_waveform(
            model,
            tokenizer,
            waveform,
            audio_config,
            options=GenerationOptions(max_new_tokens=4),
        )
        assert output.duration == pytest.approx(0.6, abs=0.01)
        assert len(output.chunks) == 1
        assert output.chunks[0].start == 0.0
        assert output.chunks[0].end == pytest.approx(0.6, abs=0.01)

    def test_multi_chunk_spans_cover_audio(self, model, tokenizer, audio_config):
        # 2.5 s of audio with 1 s chunks -> 3 windows.
        waveform = make_sine(440.0, 2.5, audio_config.sample_rate)
        output = transcribe_waveform(
            model,
            tokenizer,
            waveform,
            audio_config,
            options=GenerationOptions(max_new_tokens=4),
        )
        assert len(output.chunks) == 3
        assert output.chunks[0].start == 0.0
        assert output.chunks[1].start == pytest.approx(1.0)
        assert output.chunks[2].end == pytest.approx(2.5, abs=0.01)

    def test_beam_options_supported(self, model, tokenizer, audio_config):
        waveform = make_sine(440.0, 1.2, audio_config.sample_rate)
        output = transcribe_waveform(
            model,
            tokenizer,
            waveform,
            audio_config,
            options=GenerationOptions(beam_size=2, max_new_tokens=4),
        )
        assert len(output.chunks) == 2

    def test_empty_waveform_rejected(self, model, tokenizer, audio_config):
        with pytest.raises(ValueError, match="empty"):
            transcribe_waveform(model, tokenizer, np.zeros(0, dtype=np.float32), audio_config)

    def test_stereo_rejected(self, model, tokenizer, audio_config):
        with pytest.raises(ValueError, match="mono"):
            transcribe_waveform(
                model, tokenizer, np.zeros((2, 100), dtype=np.float32), audio_config
            )


class TestTranscribeFile:
    def test_decodes_and_transcribes(self, model, tokenizer, audio_config, tmp_path):
        path = tmp_path / "tone.wav"
        write_wav(path, make_sine(440.0, 0.5, audio_config.sample_rate), audio_config.sample_rate)
        output = transcribe_file(
            model,
            tokenizer,
            path,
            audio_config,
            options=GenerationOptions(max_new_tokens=4),
        )
        assert isinstance(output.text, str)
        assert output.duration == pytest.approx(0.5, abs=0.01)

    def test_resamples_foreign_rates(self, model, tokenizer, audio_config, tmp_path):
        path = tmp_path / "tone8k.wav"
        write_wav(path, make_sine(440.0, 0.5, 8_000), 8_000)
        output = transcribe_file(
            model,
            tokenizer,
            path,
            audio_config,
            options=GenerationOptions(max_new_tokens=4),
        )
        assert output.duration == pytest.approx(0.5, abs=0.01)
