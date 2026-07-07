"""Byte-level BPE tokenizer: training, round-trips, serialization."""

from __future__ import annotations

import pytest

from tests.conftest import CORPUS
from whisperlite.text.tokenizer import (
    BASE_VOCAB_SIZE,
    BPETokenizer,
    TokenizerError,
)


class TestTraining:
    def test_vocab_size_reached(self, tokenizer):
        assert BASE_VOCAB_SIZE < tokenizer.vocab_size <= 320

    def test_min_vocab_is_bytes_only(self):
        tok = BPETokenizer.train(["abc"], vocab_size=BASE_VOCAB_SIZE)
        assert tok.vocab_size == BASE_VOCAB_SIZE

    def test_vocab_below_base_rejected(self):
        with pytest.raises(TokenizerError, match="vocab_size"):
            BPETokenizer.train(["abc"], vocab_size=100)

    def test_empty_corpus_rejected(self):
        with pytest.raises(TokenizerError, match="empty"):
            BPETokenizer.train([], vocab_size=300)

    def test_min_frequency_stops_early(self):
        # Every chunk is unique, so no pair reaches frequency 2.
        tok = BPETokenizer.train(["ab cd ef"], vocab_size=400, min_frequency=5)
        assert tok.vocab_size == BASE_VOCAB_SIZE

    def test_training_is_deterministic(self):
        first = BPETokenizer.train(CORPUS, vocab_size=300, min_frequency=1)
        second = BPETokenizer.train(CORPUS, vocab_size=300, min_frequency=1)
        assert first.to_dict() == second.to_dict()

    def test_merges_compress_training_text(self, tokenizer):
        text = CORPUS[1]
        assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


class TestRoundTrip:
    @pytest.mark.parametrize("text", [*CORPUS, "", " ", "  double  spaces  "])
    def test_corpus_roundtrip(self, tokenizer, text):
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_unseen_unicode_roundtrip(self, tokenizer):
        text = "héllo wörld 世界 🚀 → done"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_newlines_and_tabs_roundtrip(self, tokenizer):
        text = "line one\nline\ttwo"
        assert tokenizer.decode(tokenizer.encode(text)) == text


class TestSpecialTokens:
    def test_fixed_ids(self, tokenizer):
        assert tokenizer.pad_id == 0
        assert tokenizer.sot_id == 1
        assert tokenizer.eot_id == 2
        assert tokenizer.special_ids == (0, 1, 2)

    def test_specials_skipped_in_decode(self, tokenizer):
        ids = [tokenizer.sot_id, *tokenizer.encode("hello"), tokenizer.eot_id]
        assert tokenizer.decode(ids) == "hello"

    def test_encode_never_emits_specials(self, tokenizer):
        for text in CORPUS:
            assert not set(tokenizer.encode(text)) & set(tokenizer.special_ids)

    def test_out_of_range_id_rejected(self, tokenizer):
        with pytest.raises(TokenizerError, match="out of range"):
            tokenizer.decode([tokenizer.vocab_size])
        with pytest.raises(TokenizerError, match="out of range"):
            tokenizer.decode([-1])


class TestSerialization:
    def test_save_load_identity(self, tokenizer, tmp_path):
        path = tmp_path / "tok.json"
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.to_dict() == tokenizer.to_dict()
        text = "the quick brown fox"
        assert loaded.encode(text) == tokenizer.encode(text)

    def test_missing_file(self, tmp_path):
        with pytest.raises(TokenizerError, match="not found"):
            BPETokenizer.load(tmp_path / "nope.json")

    def test_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(TokenizerError, match="invalid tokenizer JSON"):
            BPETokenizer.load(path)

    def test_wrong_type_rejected(self, tokenizer, tmp_path):
        data = tokenizer.to_dict()
        data["type"] = "wordpiece"
        with pytest.raises(TokenizerError, match="unsupported tokenizer type"):
            BPETokenizer.from_dict(data)

    def test_invalid_merge_reference_rejected(self):
        with pytest.raises(TokenizerError, match="invalid token id"):
            BPETokenizer([(1000, 3)])
