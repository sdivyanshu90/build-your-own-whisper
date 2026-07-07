"""Manifests, dataset items, and batch collation."""

from __future__ import annotations

import json

import pytest
import torch

from whisperlite.config import AugmentConfig
from whisperlite.data.dataset import SpeechDataset, collate_batch
from whisperlite.data.manifest import ManifestEntry, ManifestError, read_manifest, write_manifest


class TestManifest:
    def test_roundtrip(self, tmp_path):
        entries = [
            ManifestEntry("a.wav", "hello", 1.5),
            ManifestEntry("b.wav", "world", None),
        ]
        path = tmp_path / "m.jsonl"
        write_manifest(path, entries)
        assert read_manifest(path) == entries

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"audio_filepath": "a.wav", "text": "hi"}\n\n\n')
        assert len(read_manifest(path)) == 1

    def test_error_reports_line_number(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"audio_filepath": "a.wav", "text": "ok"}\n{broken\n')
        with pytest.raises(ManifestError, match=":2:"):
            read_manifest(path)

    def test_missing_fields_rejected(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"text": "no audio"}\n')
        with pytest.raises(ManifestError, match="audio_filepath"):
            read_manifest(path)

    def test_negative_duration_rejected(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"audio_filepath": "a.wav", "text": "x", "duration": -1}\n')
        with pytest.raises(ManifestError, match="duration"):
            read_manifest(path)

    def test_empty_manifest_rejected(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text("\n")
        with pytest.raises(ManifestError, match="no entries"):
            read_manifest(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            read_manifest(tmp_path / "nope.jsonl")


class TestSpeechDataset:
    def test_item_shapes(self, synth_corpus, tokenizer, audio_config):
        dataset = SpeechDataset(
            read_manifest(synth_corpus["train"]), tokenizer, audio_config, max_text_tokens=24
        )
        item = dataset[0]
        assert item["mel"].shape == (audio_config.n_mels, audio_config.n_frames)
        assert item["tokens"][0] == tokenizer.sot_id
        assert item["tokens"][-1] == tokenizer.eot_id
        assert item["text"] == synth_corpus["entries"][0]["text"]

    def test_augmentation_applied_only_in_train_mode(self, synth_corpus, tokenizer, audio_config):
        entries = read_manifest(synth_corpus["train"])
        heavy = AugmentConfig(
            enabled=True, freq_masks=8, freq_width=40, time_masks=8, time_ratio=0.5
        )
        torch.manual_seed(0)
        train_ds = SpeechDataset(
            entries, tokenizer, audio_config, augment=heavy, train=True, max_text_tokens=24
        )
        eval_ds = SpeechDataset(
            entries, tokenizer, audio_config, augment=heavy, train=False, max_text_tokens=24
        )
        assert not torch.equal(train_ds[0]["mel"], eval_ds[0]["mel"])

    def test_overlong_transcripts_filtered(self, synth_corpus, tokenizer, audio_config):
        entries = read_manifest(synth_corpus["train"])
        short = ManifestEntry(entries[0].audio_filepath, "a", duration=0.8)
        long = ManifestEntry(
            entries[0].audio_filepath,
            "an extremely long transcript that certainly exceeds the token budget",
            duration=0.8,
        )
        limit = len(tokenizer.encode(short.text))
        dataset = SpeechDataset([short, long], tokenizer, audio_config, max_text_tokens=limit)
        assert len(dataset) == 1
        assert dataset[0]["text"] == "a"

    def test_overlong_audio_filtered_by_duration(self, tokenizer, audio_config, synth_corpus):
        entries = read_manifest(synth_corpus["train"])
        long_entry = ManifestEntry(entries[0].audio_filepath, "text", duration=999.0)
        dataset = SpeechDataset([*entries, long_entry], tokenizer, audio_config, max_text_tokens=24)
        assert len(dataset) == len(entries)

    def test_empty_after_filtering_rejected(self, synth_corpus, tokenizer, audio_config):
        entries = [
            ManifestEntry(e.audio_filepath, e.text, duration=999.0)
            for e in read_manifest(synth_corpus["train"])
        ]
        with pytest.raises(ValueError, match="empty"):
            SpeechDataset(entries, tokenizer, audio_config, max_text_tokens=24)


class TestCollate:
    def test_teacher_forcing_layout(self, tokenizer):
        items = [
            {
                "mel": torch.zeros(80, 100),
                "tokens": torch.tensor([1, 10, 11, 12, 2]),
                "text": "a",
            },
            {
                "mel": torch.zeros(80, 100),
                "tokens": torch.tensor([1, 20, 2]),
                "text": "b",
            },
        ]
        batch = collate_batch(items, pad_id=tokenizer.pad_id)
        assert batch["mel"].shape == (2, 80, 100)
        assert batch["tokens_in"].shape == (2, 4)
        assert batch["tokens_in"][0].tolist() == [1, 10, 11, 12]
        assert batch["targets"][0].tolist() == [10, 11, 12, 2]
        assert batch["tokens_in"][1].tolist() == [1, 20, 0, 0]
        assert batch["targets"][1].tolist() == [20, 2, 0, 0]
        assert batch["texts"] == ["a", "b"]

    def test_json_serializable_manifest(self, synth_corpus):
        for line in synth_corpus["train"].read_text().splitlines():
            record = json.loads(line)
            assert set(record) == {"audio_filepath", "text", "duration"}
