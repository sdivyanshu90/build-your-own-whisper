"""WER/CER metrics, text normalization, LR schedules."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from whisperlite.training.metrics import (
    char_error_rate,
    edit_distance,
    normalize_text,
    word_error_rate,
)
from whisperlite.training.scheduler import create_lr_scheduler


class TestNormalizeText:
    def test_case_punctuation_whitespace(self):
        assert normalize_text("  Hello,   WORLD!  ") == "hello world"

    def test_apostrophes_preserved(self):
        assert normalize_text("don't stop") == "don't stop"

    def test_empty(self):
        assert normalize_text("...") == ""


class TestEditDistance:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("abc", "abc", 0),
            ("abc", "", 3),
            ("", "abc", 3),
            ("kitten", "sitting", 3),
            (["a", "b"], ["a", "c"], 1),
        ],
    )
    def test_known_values(self, a, b, expected):
        assert edit_distance(a, b) == expected

    def test_symmetry(self):
        assert edit_distance("sunday", "saturday") == edit_distance("saturday", "sunday")


class TestErrorRates:
    def test_perfect_match(self):
        assert word_error_rate(["hello world"], ["hello world"]) == 0.0

    def test_known_wer(self):
        # 1 substitution over 3 reference words.
        assert word_error_rate(["a b c"], ["a x c"]) == pytest.approx(1 / 3)

    def test_normalization_applied(self):
        assert word_error_rate(["Hello, World!"], ["hello world"]) == 0.0

    def test_corpus_level_weighting(self):
        # 1 error over 5 total reference words, not mean of per-utterance rates.
        wer = word_error_rate(["a", "b c d e"], ["x", "b c d e"])
        assert wer == pytest.approx(1 / 5)

    def test_cer(self):
        assert char_error_rate(["abcd"], ["abed"]) == pytest.approx(1 / 4)

    def test_empty_reference_edge_cases(self):
        assert word_error_rate([""], [""]) == 0.0
        assert word_error_rate([""], ["hello"]) == 1.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            word_error_rate(["a"], ["a", "b"])

    def test_zero_utterances_rejected(self):
        with pytest.raises(ValueError, match="zero utterances"):
            word_error_rate([], [])


class TestScheduler:
    def _optimizer(self, lr: float = 1.0):
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)

    def _lrs(self, name: str, warmup: int, total: int, min_ratio: float = 0.0):
        optimizer = self._optimizer()
        scheduler = create_lr_scheduler(
            optimizer,
            name=name,
            warmup_steps=warmup,
            total_steps=total,
            min_lr_ratio=min_ratio,
        )
        values = []
        for _ in range(total):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return values

    def test_warmup_ramps_to_peak(self):
        values = self._lrs("cosine", warmup=10, total=100)
        assert values[0] == pytest.approx(0.1)
        assert values[9] == pytest.approx(1.0)
        assert max(values) == pytest.approx(1.0)

    def test_cosine_reaches_min_ratio(self):
        values = self._lrs("cosine", warmup=10, total=100, min_ratio=0.1)
        assert values[-1] == pytest.approx(
            0.1 + (1 - 0.1) * 0.5 * (1 + math.cos(math.pi * 89 / 90)), rel=1e-6
        )

    def test_linear_decays_monotonically(self):
        values = self._lrs("linear", warmup=5, total=50)
        post_warmup = values[5:]
        assert all(a >= b for a, b in itertools.pairwise(post_warmup))

    def test_constant_holds_after_warmup(self):
        values = self._lrs("constant", warmup=5, total=20)
        assert all(v == pytest.approx(1.0) for v in values[5:])

    def test_invalid_warmup_rejected(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            create_lr_scheduler(self._optimizer(), name="cosine", warmup_steps=100, total_steps=50)
