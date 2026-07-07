"""Shared fixtures: tiny tokenizer, tiny model, synthetic audio corpus.

Everything here is deliberately miniature (1-second chunks, 64-dim model)
so the whole suite runs in seconds on CPU while still exercising the exact
production code paths.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from whisperlite.config import AudioConfig, resolve_model_config
from whisperlite.model.asr import WhisperLite
from whisperlite.model.checkpoint import save_checkpoint
from whisperlite.text.tokenizer import BPETokenizer

CORPUS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "speech recognition from scratch",
    "one two three four five six seven eight nine ten",
    "alpha beta gamma delta",
    "testing testing one two three",
    "a journey of a thousand miles begins with a single step",
    "to be or not to be that is the question",
    "pack my box with five dozen liquor jugs",
    "the rain in spain stays mainly in the plain",
]

#: (frequency in Hz, transcript) pairs for the synthetic training corpus.
SYNTH_UTTERANCES = [
    (220.0, "alpha beta"),
    (330.0, "gamma delta"),
    (440.0, "hello world"),
    (550.0, "one two three"),
    (660.0, "quick brown fox"),
    (770.0, "lazy dog"),
    (880.0, "testing one"),
    (990.0, "single step"),
]


@pytest.fixture(scope="session")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=320, min_frequency=1)


@pytest.fixture(scope="session")
def audio_config() -> AudioConfig:
    # 1-second chunks -> 100 mel frames -> encoder context of 50.
    return AudioConfig(sample_rate=16_000, n_fft=400, hop_length=160, n_mels=80, chunk_length=1.0)


@pytest.fixture(scope="session")
def model_config(tokenizer, audio_config):
    return resolve_model_config(
        None,
        {
            "n_audio_state": 64,
            "n_audio_head": 2,
            "n_audio_layer": 2,
            "n_text_state": 64,
            "n_text_head": 2,
            "n_text_layer": 2,
            "n_text_ctx": 48,
            "dropout": 0.0,
        },
        audio_config,
        n_vocab=tokenizer.vocab_size,
    )


@pytest.fixture(scope="session")
def model(model_config) -> WhisperLite:
    torch.manual_seed(0)
    return WhisperLite(model_config).eval()


def make_sine(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")


def wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def synth_corpus(tmp_path_factory, audio_config):
    """Synthetic wav files + train/val manifests; returns their paths."""
    root = tmp_path_factory.mktemp("synth_corpus")
    entries = []
    for index, (frequency, text) in enumerate(SYNTH_UTTERANCES):
        wav_path = root / f"utt{index}.wav"
        write_wav(
            wav_path,
            make_sine(frequency, 0.8, audio_config.sample_rate),
            audio_config.sample_rate,
        )
        entries.append({"audio_filepath": str(wav_path), "text": text, "duration": 0.8})

    train_manifest = root / "train.jsonl"
    val_manifest = root / "val.jsonl"
    with train_manifest.open("w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    with val_manifest.open("w") as fh:
        for entry in entries[:4]:
            fh.write(json.dumps(entry) + "\n")
    return {"root": root, "train": train_manifest, "val": val_manifest, "entries": entries}


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory, model, tokenizer, audio_config) -> Path:
    """A saved (untrained) checkpoint for API/CLI tests."""
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    save_checkpoint(path, model=model, audio_config=audio_config, tokenizer=tokenizer, step=123)
    return path


@pytest.fixture(scope="session")
def sample_wav_bytes(audio_config) -> bytes:
    return wav_bytes(make_sine(440.0, 0.5, audio_config.sample_rate), audio_config.sample_rate)
