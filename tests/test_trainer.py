"""End-to-end training: overfit a tiny model on synthetic audio, resume, evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisperlite.config import (
    AugmentConfig,
    DataConfig,
    OptimConfig,
    TrainConfig,
)
from whisperlite.model.checkpoint import load_model
from whisperlite.training.trainer import Trainer

TINY_OVERRIDES = {
    "n_audio_state": 64,
    "n_audio_head": 2,
    "n_audio_layer": 2,
    "n_text_state": 64,
    "n_text_head": 2,
    "n_text_layer": 2,
    "n_text_ctx": 48,
    "dropout": 0.0,
}


def make_train_config(synth_corpus, tokenizer_path: Path, output_dir: Path, **kwargs):
    audio_kwargs = {"chunk_length": 1.0}
    defaults = dict(
        data=DataConfig(
            train_manifest=str(synth_corpus["train"]),
            val_manifest=str(synth_corpus["val"]),
            batch_size=4,
            num_workers=0,
            max_text_tokens=24,
            augment=AugmentConfig(enabled=False),
        ),
        tokenizer_path=str(tokenizer_path),
        output_dir=str(output_dir),
        model_preset=None,
        model_overrides=dict(TINY_OVERRIDES),
        optim=OptimConfig(lr=3e-3, warmup_steps=5, scheduler="cosine", clip_norm=1.0),
        seed=7,
        max_steps=60,
        amp="off",
        device="cpu",
        log_interval=10,
        eval_interval=60,
        eval_max_batches=1,
        save_interval=30,
        keep_checkpoints=2,
    )
    defaults.update(kwargs)
    from whisperlite.config import AudioConfig

    defaults.setdefault("audio", AudioConfig(**audio_kwargs))
    return TrainConfig(**defaults)


@pytest.fixture(scope="module")
def tokenizer_file(tokenizer, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tokenizer.save(path)
    return path


@pytest.fixture(scope="module")
def trained_run(synth_corpus, tokenizer_file, tmp_path_factory):
    """One shared 60-step training run inspected by several tests."""
    output_dir = tmp_path_factory.mktemp("run")
    config = make_train_config(synth_corpus, tokenizer_file, output_dir)
    summary = Trainer(config).train()
    return {"config": config, "summary": summary, "output_dir": output_dir}


def read_metrics(output_dir: Path) -> list[dict]:
    lines = (output_dir / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.slow
class TestTraining:
    def test_loss_decreases_substantially(self, trained_run):
        train_records = [r for r in read_metrics(trained_run["output_dir"]) if r["type"] == "train"]
        assert len(train_records) >= 3
        assert train_records[-1]["loss"] < train_records[0]["loss"] * 0.7

    def test_eval_metrics_recorded(self, trained_run):
        eval_records = [r for r in read_metrics(trained_run["output_dir"]) if r["type"] == "eval"]
        assert eval_records, "expected at least one eval record"
        final = eval_records[-1]
        assert final["wer"] >= 0.0
        assert final["utterances"] == 4

    def test_checkpoints_written_and_pruned(self, trained_run):
        checkpoint_dir = Path(trained_run["summary"]["checkpoint_dir"])
        step_files = sorted(checkpoint_dir.glob("step-*.pt"))
        assert len(step_files) == 2  # keep_checkpoints=2 out of steps 30, 60
        assert (checkpoint_dir / "best.pt").exists()

    def test_best_checkpoint_loadable_and_runs(self, trained_run, audio_config):
        model, tokenizer, loaded_audio = load_model(trained_run["summary"]["best_checkpoint"])
        assert loaded_audio.chunk_length == 1.0
        assert model.config.n_vocab == tokenizer.vocab_size

    def test_resolved_config_dumped(self, trained_run):
        dumped = (trained_run["output_dir"] / "train_config.yaml").read_text()
        assert "n_audio_state: 64" in dumped

    def test_resume_continues_from_saved_step(
        self, synth_corpus, tokenizer_file, tmp_path_factory, trained_run
    ):
        checkpoint_dir = Path(trained_run["summary"]["checkpoint_dir"])
        last_step_ckpt = sorted(checkpoint_dir.glob("step-*.pt"))[-1]
        output_dir = tmp_path_factory.mktemp("resume_run")
        config = make_train_config(
            synth_corpus,
            tokenizer_file,
            output_dir,
            max_steps=70,
            resume_from=str(last_step_ckpt),
        )
        trainer = Trainer(config)
        assert trainer.global_step == 60
        summary = trainer.train()
        assert summary["steps"] == 70

    def test_resume_with_mismatched_architecture_rejected(
        self, synth_corpus, tokenizer_file, tmp_path_factory, trained_run
    ):
        checkpoint_dir = Path(trained_run["summary"]["checkpoint_dir"])
        last_step_ckpt = sorted(checkpoint_dir.glob("step-*.pt"))[-1]
        overrides = dict(TINY_OVERRIDES, n_audio_layer=3)
        config = make_train_config(
            synth_corpus,
            tokenizer_file,
            tmp_path_factory.mktemp("bad_resume"),
            model_overrides=overrides,
            resume_from=str(last_step_ckpt),
        )
        with pytest.raises(ValueError, match="cannot resume"):
            Trainer(config)


@pytest.mark.slow
class TestEvaluateAPI:
    def test_evaluate_returns_all_metrics(
        self, trained_run, synth_corpus, tokenizer_file, tmp_path_factory
    ):
        config = make_train_config(
            synth_corpus, tokenizer_file, tmp_path_factory.mktemp("eval_run"), max_steps=10
        )
        trainer = Trainer(config)
        metrics = trainer.evaluate(max_batches=1)
        assert set(metrics) == {"loss", "wer", "cer", "utterances"}
        assert metrics["loss"] > 0
