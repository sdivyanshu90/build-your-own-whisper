"""ASR evaluation metrics: WER and CER with a standard text normalizer."""

from __future__ import annotations

import re
from collections.abc import Sequence

_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation (keeping apostrophes), collapse whitespace.

    Applied to both references and hypotheses before scoring so that casing
    and punctuation differences don't count as recognition errors.
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_item != hyp_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _error_rate(references: Sequence[str], hypotheses: Sequence[str], unit: str) -> float:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references ({len(references)}) and hypotheses ({len(hypotheses)}) differ in length"
        )
    if not references:
        raise ValueError("cannot compute an error rate over zero utterances")
    errors = 0
    total = 0
    for ref, hyp in zip(references, hypotheses, strict=False):
        ref_units: Sequence = normalize_text(ref).split() if unit == "word" else normalize_text(ref)
        hyp_units: Sequence = normalize_text(hyp).split() if unit == "word" else normalize_text(hyp)
        errors += edit_distance(ref_units, hyp_units)
        total += len(ref_units)
    if total == 0:
        # All references empty: perfect iff all hypotheses are empty too.
        return 0.0 if errors == 0 else 1.0
    return errors / total


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level WER (total edits / total reference words)."""
    return _error_rate(references, hypotheses, "word")


def char_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level CER (total edits / total reference characters)."""
    return _error_rate(references, hypotheses, "char")
