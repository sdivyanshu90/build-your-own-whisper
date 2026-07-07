"""Byte-level BPE tokenizer, trainable from scratch.

Design
------
* **Byte-level**: the base alphabet is all 256 byte values, so any Unicode
  text round-trips without an ``<unk>`` token.
* **Pre-tokenization**: text is split into whitespace-anchored chunks (each
  word keeps its leading whitespace) and BPE merges never cross chunk
  boundaries. This is the same idea as GPT-2's regex pre-tokenizer, kept
  deliberately simple.
* **Fixed ID layout** (stable across vocab sizes, so decoding logic never
  needs a lookup table for specials)::

      0            <|pad|>   padding / ignored label positions
      1            <|sot|>   start of transcript
      2            <|eot|>   end of transcript
      3   .. 258   the 256 raw byte tokens
      259 ..       merged tokens, in merge-creation order

The trainer uses incremental pair-count maintenance, so training an 8k
vocabulary on a few hundred MB of text is minutes, not hours.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD_TOKEN = "<|pad|>"
SOT_TOKEN = "<|sot|>"
EOT_TOKEN = "<|eot|>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, SOT_TOKEN, EOT_TOKEN)

NUM_SPECIAL = len(SPECIAL_TOKENS)
NUM_BYTES = 256
BASE_VOCAB_SIZE = NUM_SPECIAL + NUM_BYTES  # 259

_FORMAT_VERSION = 1
_TOKENIZER_TYPE = "byte_bpe"

# Splits text losslessly into chunks of "optional leading whitespace + word"
# plus trailing pure-whitespace runs.
_PRETOKEN_RE = re.compile(r"\s*\S+|\s+")

Pair = tuple[int, int]


class TokenizerError(ValueError):
    """Raised for invalid tokenizer files, ids, or training inputs."""


class BPETokenizer:
    """A byte-level BPE tokenizer with fixed special-token ids."""

    def __init__(self, merges: list[Pair]):
        bytes_table: list[bytes] = [b""] * NUM_SPECIAL
        bytes_table.extend(bytes([i]) for i in range(NUM_BYTES))
        for index, (left, right) in enumerate(merges):
            new_id = BASE_VOCAB_SIZE + index
            for part in (left, right):
                if not (NUM_SPECIAL <= part < new_id):
                    raise TokenizerError(
                        f"merge #{index} references invalid token id {part} "
                        f"(must be in [{NUM_SPECIAL}, {new_id}))"
                    )
            bytes_table.append(bytes_table[left] + bytes_table[right])

        self._merges: list[Pair] = [(int(left), int(right)) for left, right in merges]
        self._ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self._merges)}
        self._bytes_table = bytes_table

    # -- Properties ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self._merges)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def sot_id(self) -> int:
        return 1

    @property
    def eot_id(self) -> int:
        return 2

    @property
    def special_ids(self) -> tuple[int, ...]:
        return tuple(range(NUM_SPECIAL))

    # -- Encoding / decoding ------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids (no special tokens added)."""
        ids: list[int] = []
        for chunk in _PRETOKEN_RE.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def _encode_chunk(self, raw: bytes) -> list[int]:
        parts = [NUM_SPECIAL + b for b in raw]
        while len(parts) >= 2:
            best_pair: Pair | None = None
            best_rank = len(self._merges)
            for pair in itertools.pairwise(parts):
                rank = self._ranks.get(pair, -1)
                if rank != -1 and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged_id = BASE_VOCAB_SIZE + best_rank
            parts = _merge_pair(parts, best_pair, merged_id)
        return parts

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids to text, silently skipping special tokens."""
        pieces: list[bytes] = []
        for token_id in ids:
            token_id = int(token_id)
            if not 0 <= token_id < self.vocab_size:
                raise TokenizerError(
                    f"token id {token_id} out of range for vocab of {self.vocab_size}"
                )
            pieces.append(self._bytes_table[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": _FORMAT_VERSION,
            "type": _TOKENIZER_TYPE,
            "special_tokens": list(SPECIAL_TOKENS),
            "merges": [list(pair) for pair in self._merges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> BPETokenizer:
        if not isinstance(data, dict):
            raise TokenizerError("tokenizer data must be a mapping")
        if data.get("type") != _TOKENIZER_TYPE:
            raise TokenizerError(f"unsupported tokenizer type: {data.get('type')!r}")
        if data.get("format_version") != _FORMAT_VERSION:
            raise TokenizerError(
                f"unsupported tokenizer format_version: {data.get('format_version')!r}"
            )
        if data.get("special_tokens") != list(SPECIAL_TOKENS):
            raise TokenizerError("special token layout does not match this library version")
        merges = data.get("merges")
        if not isinstance(merges, list):
            raise TokenizerError("merges must be a list of [left, right] pairs")
        parsed: list[Pair] = []
        for entry in merges:
            if not (isinstance(entry, list | tuple) and len(entry) == 2):
                raise TokenizerError(f"invalid merge entry: {entry!r}")
            parsed.append((int(entry[0]), int(entry[1])))
        return cls(parsed)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        p = Path(path)
        if not p.is_file():
            raise TokenizerError(f"tokenizer file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TokenizerError(f"invalid tokenizer JSON in {p}: {exc}") from exc
        return cls.from_dict(data)

    # -- Training -----------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        *,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Learn BPE merges from a text corpus.

        *vocab_size* is the total size including the 3 specials and 256 byte
        tokens, so it must be at least ``BASE_VOCAB_SIZE``. Training stops
        early when no remaining pair occurs *min_frequency* times.
        """
        if vocab_size < BASE_VOCAB_SIZE:
            raise TokenizerError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE} (specials + bytes), got {vocab_size}"
            )
        if min_frequency < 1:
            raise TokenizerError(f"min_frequency must be >= 1, got {min_frequency}")

        chunk_freqs: Counter[bytes] = Counter()
        for text in texts:
            for chunk in _PRETOKEN_RE.findall(text):
                chunk_freqs[chunk.encode("utf-8")] += 1
        if not chunk_freqs:
            raise TokenizerError("training corpus is empty")

        words: list[list[int]] = []
        freqs: list[int] = []
        for raw, freq in chunk_freqs.items():
            words.append([NUM_SPECIAL + b for b in raw])
            freqs.append(freq)

        # Incrementally-maintained statistics: total pair counts, and for each
        # pair the set of word indices containing it.
        pair_counts: Counter[Pair] = Counter()
        pair_to_words: dict[Pair, set[int]] = {}
        for idx, word in enumerate(words):
            freq = freqs[idx]
            for pair in itertools.pairwise(word):
                pair_counts[pair] += freq
                pair_to_words.setdefault(pair, set()).add(idx)

        merges: list[Pair] = []
        target_merges = vocab_size - BASE_VOCAB_SIZE
        while len(merges) < target_merges and pair_counts:
            # Deterministic winner: highest count, then lowest (left, right).
            best_pair, best_count = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if best_count < min_frequency:
                break
            new_id = BASE_VOCAB_SIZE + len(merges)
            merges.append(best_pair)

            affected = pair_to_words.pop(best_pair, set())
            del pair_counts[best_pair]
            for idx in affected:
                word = words[idx]
                freq = freqs[idx]
                for pair in itertools.pairwise(word):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    bucket = pair_to_words.get(pair)
                    if bucket is not None:
                        bucket.discard(idx)
                        if not bucket:
                            del pair_to_words[pair]
                new_word = _merge_pair(word, best_pair, new_id)
                words[idx] = new_word
                for pair in itertools.pairwise(new_word):
                    pair_counts[pair] += freq
                    pair_to_words.setdefault(pair, set()).add(idx)

        return cls(merges)


def _merge_pair(parts: list[int], pair: Pair, merged_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of *pair* (left to right)."""
    out: list[int] = []
    i = 0
    n = len(parts)
    while i < n:
        if i + 1 < n and parts[i] == pair[0] and parts[i + 1] == pair[1]:
            out.append(merged_id)
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return out
