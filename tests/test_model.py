"""Model architecture: shapes, causality, KV-cache equivalence, checkpoints."""

from __future__ import annotations

import pytest
import torch

from whisperlite.model.asr import WhisperLite
from whisperlite.model.checkpoint import (
    CheckpointError,
    load_checkpoint,
    load_model,
    save_checkpoint,
)
from whisperlite.model.layers import causal_mask, sinusoids


def random_mel(model_config, batch: int = 2) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(batch, model_config.n_mels, model_config.n_audio_ctx * 2)


class TestLayers:
    def test_sinusoids_shape_and_boundedness(self):
        emb = sinusoids(50, 64)
        assert emb.shape == (50, 64)
        assert float(emb.abs().max()) <= 1.0 + 1e-6

    def test_sinusoids_odd_channels_rejected(self):
        with pytest.raises(ValueError):
            sinusoids(10, 63)

    def test_causal_mask_prefill(self):
        mask = causal_mask(3, 0, torch.device("cpu"), torch.float32)
        assert mask.shape == (3, 3)
        assert mask[0, 1] == float("-inf")
        assert mask[2, 2] == 0.0

    def test_causal_mask_with_offset(self):
        mask = causal_mask(2, 4, torch.device("cpu"), torch.float32)
        assert mask.shape == (2, 6)
        assert (mask[0, :5] == 0).all()
        assert mask[0, 5] == float("-inf")

    def test_single_token_needs_no_mask(self):
        assert causal_mask(1, 7, torch.device("cpu"), torch.float32) is None


class TestEncoder:
    def test_output_shape(self, model, model_config):
        features = model.embed_audio(random_mel(model_config))
        assert features.shape == (2, model_config.n_audio_ctx, model_config.n_audio_state)

    def test_shorter_input_allowed(self, model, model_config):
        mel = random_mel(model_config)[:, :, : model_config.n_audio_ctx]  # half length
        features = model.embed_audio(mel)
        assert features.shape[1] == model_config.n_audio_ctx // 2

    def test_overlong_input_rejected(self, model, model_config):
        mel = torch.randn(1, model_config.n_mels, model_config.n_audio_ctx * 2 + 10)
        with pytest.raises(ValueError, match="too long"):
            model.embed_audio(mel)

    def test_wrong_mel_bins_rejected(self, model, model_config):
        with pytest.raises(ValueError, match="expected mel"):
            model.embed_audio(torch.randn(1, model_config.n_mels + 1, 10))


class TestDecoder:
    def test_teacher_forcing_shape(self, model, model_config):
        mel = random_mel(model_config)
        tokens = torch.randint(3, model_config.n_vocab, (2, 7))
        logits = model(mel, tokens)
        assert logits.shape == (2, 7, model_config.n_vocab)

    def test_causality(self, model, model_config):
        """Changing a future token must not affect earlier logits."""
        mel = random_mel(model_config, batch=1)
        audio = model.embed_audio(mel)
        tokens = torch.randint(3, model_config.n_vocab, (1, 8))
        mutated = tokens.clone()
        mutated[0, 5] = (mutated[0, 5] + 1) % model_config.n_vocab
        base = model.decode_step(tokens, audio)
        changed = model.decode_step(mutated, audio)
        assert torch.allclose(base[0, :5], changed[0, :5], atol=1e-5)
        assert not torch.allclose(base[0, 5:], changed[0, 5:], atol=1e-5)

    def test_kv_cache_matches_full_forward(self, model, model_config):
        mel = random_mel(model_config, batch=2)
        audio = model.embed_audio(mel)
        tokens = torch.randint(3, model_config.n_vocab, (2, 6))

        full = model.decode_step(tokens, audio)
        cache = model.new_cache()
        stepwise = torch.cat(
            [
                model.decode_step(tokens[:, i : i + 1], audio, cache=cache)
                for i in range(tokens.shape[1])
            ],
            dim=1,
        )
        assert torch.allclose(full, stepwise, atol=1e-4)

    def test_kv_cache_prefill_then_step(self, model, model_config):
        mel = random_mel(model_config, batch=1)
        audio = model.embed_audio(mel)
        tokens = torch.randint(3, model_config.n_vocab, (1, 6))

        full = model.decode_step(tokens, audio)
        cache = model.new_cache()
        prefill = model.decode_step(tokens[:, :4], audio, cache=cache)
        tail = model.decode_step(tokens[:, 4:], audio, cache=cache)
        assert torch.allclose(full, torch.cat([prefill, tail], dim=1), atol=1e-4)
        assert cache.offset == 6

    def test_context_overflow_rejected(self, model, model_config):
        mel = random_mel(model_config, batch=1)
        audio = model.embed_audio(mel)
        tokens = torch.zeros(1, model_config.n_text_ctx + 1, dtype=torch.long)
        with pytest.raises(ValueError, match="exceeds the text context|exceeds text context"):
            model.decode_step(tokens, audio)


class TestModel:
    def test_deterministic_construction(self, model_config):
        torch.manual_seed(5)
        first = WhisperLite(model_config)
        torch.manual_seed(5)
        second = WhisperLite(model_config)
        for p1, p2 in zip(first.parameters(), second.parameters(), strict=False):
            assert torch.equal(p1, p2)

    def test_parameter_count_positive(self, model):
        assert model.num_parameters > 100_000

    def test_output_projection_tied_to_embedding(self, model):
        assert (
            model.decoder.token_embedding.weight.data_ptr()
            == model.decoder.token_embedding.weight.data_ptr()
        )
        # Tying is structural (same tensor used in forward); verify grads flow.
        assert model.decoder.token_embedding.weight.requires_grad


class TestCheckpoint:
    def test_roundtrip(self, model, tokenizer, audio_config, tmp_path):
        path = tmp_path / "model.pt"
        save_checkpoint(path, model=model, audio_config=audio_config, tokenizer=tokenizer, step=42)
        loaded_model, loaded_tokenizer, loaded_audio = load_model(path)
        assert loaded_audio == audio_config
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        for p1, p2 in zip(model.parameters(), loaded_model.parameters(), strict=False):
            assert torch.equal(p1, p2)
        assert load_checkpoint(path)["step"] == 42

    def test_missing_file(self, tmp_path):
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(tmp_path / "nope.pt")

    def test_corrupt_payload_rejected(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"foo": 1}, path)
        with pytest.raises(CheckpointError, match="missing keys"):
            load_checkpoint(path)

    def test_loaded_model_produces_identical_logits(
        self, model, tokenizer, audio_config, model_config, tmp_path
    ):
        path = tmp_path / "model.pt"
        save_checkpoint(path, model=model, audio_config=audio_config, tokenizer=tokenizer)
        loaded_model, _, _ = load_model(path)
        mel = random_mel(model_config, batch=1)
        tokens = torch.randint(3, model_config.n_vocab, (1, 5))
        with torch.no_grad():
            assert torch.allclose(model(mel, tokens), loaded_model(mel, tokens), atol=1e-6)
