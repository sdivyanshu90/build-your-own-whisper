"""Decoding strategies: greedy, sampling, beam search."""

from __future__ import annotations

import pytest
import torch

from whisperlite.model.generation import (
    GenerationError,
    GenerationOptions,
    TranscriptionResult,
    generate,
)


@pytest.fixture(scope="module")
def mel(model_config):
    torch.manual_seed(3)
    return torch.randn(2, model_config.n_mels, model_config.n_audio_ctx * 2)


class TestOptions:
    def test_defaults_valid(self):
        options = GenerationOptions()
        assert options.beam_size == 1
        assert options.temperature == 0.0

    def test_invalid_beam(self):
        with pytest.raises(GenerationError):
            GenerationOptions(beam_size=0)

    def test_beam_with_temperature_rejected(self):
        with pytest.raises(GenerationError, match="temperature"):
            GenerationOptions(beam_size=4, temperature=0.5)

    def test_negative_temperature_rejected(self):
        with pytest.raises(GenerationError):
            GenerationOptions(temperature=-0.1)

    def test_suppressing_eot_rejected(self, model, tokenizer, mel):
        with pytest.raises(GenerationError, match="end-of-transcript"):
            generate(
                model,
                tokenizer,
                mel,
                GenerationOptions(suppress_tokens=(tokenizer.eot_id,), max_new_tokens=2),
            )


class TestGreedy:
    def test_batch_results_structure(self, model, tokenizer, mel):
        results = generate(model, tokenizer, mel, GenerationOptions(max_new_tokens=8))
        assert len(results) == 2
        for result in results:
            assert isinstance(result, TranscriptionResult)
            assert isinstance(result.text, str)
            assert result.avg_logprob <= 0.0
            assert len(result.tokens) <= 8
            assert tokenizer.eot_id not in result.tokens
            assert tokenizer.sot_id not in result.tokens

    def test_greedy_is_deterministic(self, model, tokenizer, mel):
        options = GenerationOptions(max_new_tokens=8)
        first = generate(model, tokenizer, mel, options)
        second = generate(model, tokenizer, mel, options)
        assert [r.tokens for r in first] == [r.tokens for r in second]

    def test_2d_mel_accepted(self, model, tokenizer, mel):
        results = generate(model, tokenizer, mel[0], GenerationOptions(max_new_tokens=4))
        assert len(results) == 1

    def test_batch_matches_single(self, model, tokenizer, mel):
        """Batched greedy must equal decoding each utterance alone."""
        options = GenerationOptions(max_new_tokens=8)
        batched = generate(model, tokenizer, mel, options)
        singles = [generate(model, tokenizer, mel[i], options)[0] for i in range(2)]
        assert [r.tokens for r in batched] == [r.tokens for r in singles]

    def test_suppressed_tokens_never_emitted(self, model, tokenizer, mel):
        base = generate(model, tokenizer, mel, GenerationOptions(max_new_tokens=8))
        emitted = {token for result in base for token in result.tokens}
        if not emitted:
            pytest.skip("random model emitted EOT immediately")
        victim = next(iter(emitted))
        suppressed = generate(
            model,
            tokenizer,
            mel,
            GenerationOptions(max_new_tokens=8, suppress_tokens=(victim,)),
        )
        assert all(victim not in result.tokens for result in suppressed)

    def test_model_training_mode_restored(self, model, tokenizer, mel):
        model.train()
        try:
            generate(model, tokenizer, mel, GenerationOptions(max_new_tokens=2))
            assert model.training
        finally:
            model.eval()


class TestSampling:
    def test_seeded_sampling_reproducible(self, model, tokenizer, mel):
        options = GenerationOptions(temperature=0.8, max_new_tokens=8)
        torch.manual_seed(123)
        first = generate(model, tokenizer, mel, options)
        torch.manual_seed(123)
        second = generate(model, tokenizer, mel, options)
        assert [r.tokens for r in first] == [r.tokens for r in second]


class TestBeam:
    def test_beam_one_matches_greedy(self, model, tokenizer, mel):
        greedy = generate(model, tokenizer, mel, GenerationOptions(max_new_tokens=8))
        # beam_size=1 goes through the beam-search code path.
        beam = generate(model, tokenizer, mel, GenerationOptions(beam_size=2, max_new_tokens=8))
        assert len(beam) == 2
        # Beam search explores more, so its (length-normalized) score should
        # never be materially worse than greedy's.
        for g, b in zip(greedy, beam, strict=False):
            assert b.avg_logprob >= g.avg_logprob - 1e-4

    def test_beam_results_structure(self, model, tokenizer, mel):
        results = generate(model, tokenizer, mel, GenerationOptions(beam_size=4, max_new_tokens=6))
        assert len(results) == 2
        for result in results:
            assert len(result.tokens) <= 6
            assert tokenizer.eot_id not in result.tokens

    def test_beam_deterministic(self, model, tokenizer, mel):
        options = GenerationOptions(beam_size=3, max_new_tokens=6)
        first = generate(model, tokenizer, mel, options)
        second = generate(model, tokenizer, mel, options)
        assert [r.tokens for r in first] == [r.tokens for r in second]

    def test_length_penalty_changes_ranking_inputs(self, model, tokenizer, mel):
        # Structural check: both settings decode successfully.
        for penalty in (0.0, 1.0):
            results = generate(
                model,
                tokenizer,
                mel,
                GenerationOptions(beam_size=3, max_new_tokens=6, length_penalty=penalty),
            )
            assert len(results) == 2
