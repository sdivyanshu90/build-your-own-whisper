"""CLI subcommands, run in-process through main()."""

from __future__ import annotations

import json

import pytest

from tests.conftest import make_sine, write_wav
from whisperlite.cli import main
from whisperlite.text.tokenizer import BPETokenizer
from whisperlite.version import __version__


class TestVersion:
    def test_prints_version(self, capsys):
        assert main(["version"]) == 0
        assert capsys.readouterr().out.strip() == __version__


class TestTokenizerTrain:
    def test_trains_from_text_file(self, tmp_path, capsys):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("hello world\nhello again\nworld again\n" * 5)
        output = tmp_path / "tok.json"
        code = main(
            [
                "tokenizer",
                "train",
                "--text-file",
                str(corpus),
                "--vocab-size",
                "280",
                "--output",
                str(output),
            ]
        )
        assert code == 0
        tokenizer = BPETokenizer.load(output)
        assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"

    def test_trains_from_manifest(self, synth_corpus, tmp_path):
        output = tmp_path / "tok.json"
        code = main(
            [
                "tokenizer",
                "train",
                "--manifest",
                str(synth_corpus["train"]),
                "--vocab-size",
                "270",
                "--output",
                str(output),
                "--min-frequency",
                "1",
            ]
        )
        assert code == 0
        assert output.exists()

    def test_no_corpus_is_domain_error(self, tmp_path):
        code = main(
            ["tokenizer", "train", "--vocab-size", "280", "--output", str(tmp_path / "t.json")]
        )
        assert code == 2

    def test_missing_text_file_is_domain_error(self, tmp_path):
        code = main(
            [
                "tokenizer",
                "train",
                "--text-file",
                str(tmp_path / "nope.txt"),
                "--vocab-size",
                "280",
                "--output",
                str(tmp_path / "t.json"),
            ]
        )
        assert code == 2


class TestManifestValidate:
    def test_valid_manifest(self, synth_corpus, capsys):
        code = main(["manifest", "validate", str(synth_corpus["train"]), "--check-audio"])
        assert code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["entries"] == 8
        assert report["missing_audio"] == 0

    def test_missing_audio_detected(self, tmp_path):
        manifest = tmp_path / "bad.jsonl"
        manifest.write_text('{"audio_filepath": "/nope/missing.wav", "text": "x"}\n')
        assert main(["manifest", "validate", str(manifest), "--check-audio"]) == 2

    def test_malformed_manifest_is_domain_error(self, tmp_path):
        manifest = tmp_path / "broken.jsonl"
        manifest.write_text("{not json}\n")
        assert main(["manifest", "validate", str(manifest)]) == 2


class TestTranscribeCommand:
    def test_transcribes_to_stdout(self, checkpoint_path, audio_config, tmp_path, capsys):
        wav = tmp_path / "tone.wav"
        write_wav(wav, make_sine(440.0, 0.5, audio_config.sample_rate), audio_config.sample_rate)
        code = main(
            ["transcribe", "--checkpoint", str(checkpoint_path), str(wav), "--device", "cpu"]
        )
        assert code == 0
        results = json.loads(capsys.readouterr().out)
        assert len(results) == 1
        assert results[0]["audio_filepath"] == str(wav)
        assert "text" in results[0]

    def test_output_json_file(self, checkpoint_path, audio_config, tmp_path):
        wav = tmp_path / "tone.wav"
        write_wav(wav, make_sine(440.0, 0.5, audio_config.sample_rate), audio_config.sample_rate)
        out_file = tmp_path / "result.json"
        code = main(
            [
                "transcribe",
                "--checkpoint",
                str(checkpoint_path),
                str(wav),
                "--device",
                "cpu",
                "--output-json",
                str(out_file),
            ]
        )
        assert code == 0
        assert json.loads(out_file.read_text())[0]["duration"] == pytest.approx(0.5, abs=0.01)

    def test_missing_checkpoint_is_domain_error(self, tmp_path):
        wav = tmp_path / "missing.wav"
        assert main(["transcribe", "--checkpoint", "/nope.pt", str(wav)]) == 2


class TestEvalCommand:
    def test_reports_wer(self, checkpoint_path, synth_corpus, capsys):
        code = main(
            [
                "eval",
                "--checkpoint",
                str(checkpoint_path),
                "--manifest",
                str(synth_corpus["val"]),
                "--device",
                "cpu",
                "--max-utterances",
                "2",
            ]
        )
        assert code == 0
        report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert report["utterances"] == 2
        assert report["wer"] >= 0.0
