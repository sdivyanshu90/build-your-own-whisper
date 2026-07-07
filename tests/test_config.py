"""Configuration loading, validation, and preset resolution."""

from __future__ import annotations

import pytest

from whisperlite.config import (
    MODEL_PRESETS,
    AudioConfig,
    ConfigError,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    dataclass_from_dict,
    load_train_config,
    resolve_model_config,
)


class TestAudioConfig:
    def test_whisper_defaults_derive_expected_shapes(self):
        config = AudioConfig()
        assert config.chunk_samples == 480_000
        assert config.n_frames == 3000
        assert config.n_audio_ctx == 1500

    def test_rejects_nonpositive_values(self):
        with pytest.raises(ConfigError):
            AudioConfig(sample_rate=0)
        with pytest.raises(ConfigError):
            AudioConfig(chunk_length=-1.0)

    def test_rejects_hop_larger_than_window(self):
        with pytest.raises(ConfigError, match="hop_length"):
            AudioConfig(n_fft=200, hop_length=400)

    def test_rejects_misaligned_chunk(self):
        with pytest.raises(ConfigError, match="multiple of hop_length"):
            AudioConfig(chunk_length=1.005)


class TestModelConfig:
    def test_head_divisibility_enforced(self):
        with pytest.raises(ConfigError, match="divisible"):
            ModelConfig(n_audio_state=100, n_audio_head=3)

    def test_dropout_range(self):
        with pytest.raises(ConfigError, match="dropout"):
            ModelConfig(dropout=1.0)


class TestResolveModelConfig:
    def test_preset_and_derived_fields(self):
        audio = AudioConfig(chunk_length=1.0)
        config = resolve_model_config("base", None, audio, n_vocab=500)
        assert config.n_audio_state == MODEL_PRESETS["base"]["n_audio_state"]
        assert config.n_audio_ctx == audio.n_audio_ctx
        assert config.n_vocab == 500

    def test_unknown_preset_rejected(self):
        with pytest.raises(ConfigError, match="unknown model preset"):
            resolve_model_config("huge", None, AudioConfig(chunk_length=1.0), n_vocab=500)

    def test_override_conflict_with_derived_value_rejected(self):
        with pytest.raises(ConfigError, match="conflicts"):
            resolve_model_config(
                "tiny", {"n_vocab": 999}, AudioConfig(chunk_length=1.0), n_vocab=500
            )

    def test_matching_derived_override_allowed(self):
        config = resolve_model_config(
            "tiny", {"n_vocab": 500}, AudioConfig(chunk_length=1.0), n_vocab=500
        )
        assert config.n_vocab == 500

    def test_unknown_override_key_rejected(self):
        with pytest.raises(ConfigError, match="unknown model override"):
            resolve_model_config("tiny", {"depth": 3}, AudioConfig(chunk_length=1.0), 500)


class TestDataclassFromDict:
    def test_unknown_keys_rejected_with_path(self):
        with pytest.raises(ConfigError, match="unknown keys"):
            dataclass_from_dict(OptimConfig, {"learning_rate": 1e-3})

    def test_nested_dataclass_and_tuple_coercion(self):
        config = dataclass_from_dict(
            TrainConfig,
            {
                "data": {"train_manifest": "a.jsonl", "val_manifest": "b.jsonl"},
                "tokenizer_path": "tok.json",
                "optim": {"betas": [0.9, 0.95]},
            },
        )
        assert isinstance(config.data, DataConfig)
        assert config.optim.betas == (0.9, 0.95)

    def test_missing_required_field_reported(self):
        with pytest.raises(ConfigError):
            dataclass_from_dict(TrainConfig, {"tokenizer_path": "tok.json"})

    def test_non_mapping_rejected(self):
        with pytest.raises(ConfigError, match="expected a mapping"):
            dataclass_from_dict(OptimConfig, [1, 2, 3])


class TestYamlLoading:
    def test_load_train_config_roundtrip(self, tmp_path):
        config_file = tmp_path / "train.yaml"
        config_file.write_text(
            "data:\n"
            "  train_manifest: train.jsonl\n"
            "  val_manifest: val.jsonl\n"
            "tokenizer_path: tok.json\n"
            "max_steps: 100\n"
        )
        config = load_train_config(config_file)
        assert config.max_steps == 100
        assert config.data.batch_size == 16  # default preserved

    def test_missing_file_errors(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_train_config(tmp_path / "nope.yaml")

    def test_non_mapping_yaml_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_train_config(bad)


class TestTrainConfigValidation:
    def _base(self, **kwargs):
        return TrainConfig(
            data=DataConfig(train_manifest="a", val_manifest="b"),
            tokenizer_path="tok.json",
            **kwargs,
        )

    def test_amp_mode_validated(self):
        with pytest.raises(ConfigError, match="amp"):
            self._base(amp="float8")

    def test_scheduler_validated(self):
        with pytest.raises(ConfigError, match="scheduler"):
            OptimConfig(scheduler="exponential")
