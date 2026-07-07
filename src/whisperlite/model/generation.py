"""Autoregressive decoding: batched greedy/sampling and beam search.

All strategies share the same KV-cached decode loop shape:

1. Encode the mel spectrogram once.
2. Feed ``<|sot|>``, then repeatedly feed the single most recent token,
   reusing cached keys/values for all earlier positions.
3. Stop at ``<|eot|>`` or the decoder context limit.

Special tokens (and any user-supplied ``suppress_tokens``) are masked to
``-inf`` before selection so the model can never emit padding or a second
start-of-transcript token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from whisperlite.model.asr import WhisperLite
from whisperlite.text.tokenizer import BPETokenizer


class GenerationError(ValueError):
    """Raised for invalid generation options."""


@dataclass(frozen=True)
class GenerationOptions:
    """Decoding parameters.

    * ``beam_size=1`` selects greedy decoding (or multinomial sampling when
      ``temperature > 0``); larger values enable beam search.
    * ``length_penalty`` follows the convention ``score = logprob_sum /
      length**alpha`` — ``1.0`` ranks beams by mean log-probability.
    """

    beam_size: int = 1
    temperature: float = 0.0
    max_new_tokens: int | None = None
    length_penalty: float = 1.0
    suppress_tokens: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise GenerationError(f"beam_size must be >= 1, got {self.beam_size}")
        if self.temperature < 0:
            raise GenerationError(f"temperature must be >= 0, got {self.temperature}")
        if self.beam_size > 1 and self.temperature > 0:
            raise GenerationError("beam search requires temperature == 0")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise GenerationError("max_new_tokens must be >= 1 when set")
        if self.length_penalty < 0:
            raise GenerationError("length_penalty must be >= 0")


@dataclass(frozen=True)
class TranscriptionResult:
    """One decoded hypothesis for one audio chunk."""

    text: str
    tokens: tuple[int, ...]
    avg_logprob: float


def _suppress_ids(tokenizer: BPETokenizer, options: GenerationOptions) -> list[int]:
    ids = sorted({tokenizer.pad_id, tokenizer.sot_id, *options.suppress_tokens})
    for token_id in ids:
        if not 0 <= token_id < tokenizer.vocab_size:
            raise GenerationError(f"suppress token id {token_id} out of range")
    if tokenizer.eot_id in ids:
        raise GenerationError("cannot suppress the end-of-transcript token")
    return ids


@torch.no_grad()
def generate(
    model: WhisperLite,
    tokenizer: BPETokenizer,
    mel: Tensor,
    options: GenerationOptions | None = None,
) -> list[TranscriptionResult]:
    """Transcribe a batch of mel spectrograms.

    *mel* is ``(n_mels, n_frames)`` or ``(batch, n_mels, n_frames)``; one
    result per batch element is returned.
    """
    options = options or GenerationOptions()
    was_training = model.training
    model.eval()
    try:
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        if mel.ndim != 3:
            raise GenerationError(f"mel must be 2-D or 3-D, got shape {tuple(mel.shape)}")
        device = next(model.parameters()).device
        mel = mel.to(device)

        audio_features = model.embed_audio(mel)
        max_new = model.config.n_text_ctx - 1
        if options.max_new_tokens is not None:
            max_new = min(max_new, options.max_new_tokens)
        suppress = _suppress_ids(tokenizer, options)

        if options.beam_size == 1:
            return _greedy_or_sample(model, tokenizer, audio_features, options, max_new, suppress)
        return [
            _beam_search(model, tokenizer, audio_features[i : i + 1], options, max_new, suppress)
            for i in range(audio_features.shape[0])
        ]
    finally:
        model.train(was_training)


def _step_logprobs(
    model: WhisperLite,
    tokens: Tensor,
    audio_features: Tensor,
    cache,
    suppress: list[int],
) -> Tensor:
    """One decoder step -> float32 log-probabilities with suppression applied."""
    logits = model.decode_step(tokens, audio_features, cache=cache)[:, -1].float()
    logits[:, suppress] = float("-inf")
    return F.log_softmax(logits, dim=-1)


def _greedy_or_sample(
    model: WhisperLite,
    tokenizer: BPETokenizer,
    audio_features: Tensor,
    options: GenerationOptions,
    max_new: int,
    suppress: list[int],
) -> list[TranscriptionResult]:
    batch = audio_features.shape[0]
    device = audio_features.device
    eot = tokenizer.eot_id

    cache = model.new_cache()
    current = torch.full((batch, 1), tokenizer.sot_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    sum_logprob = torch.zeros(batch, device=device)
    lengths = torch.zeros(batch, dtype=torch.long, device=device)
    steps: list[Tensor] = []

    for _ in range(max_new):
        logprobs = _step_logprobs(model, current, audio_features, cache, suppress)
        if options.temperature > 0:
            probs = (logprobs / options.temperature).softmax(dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = logprobs.argmax(dim=-1)

        step_logprob = logprobs.gather(1, next_tokens[:, None]).squeeze(1)
        active = ~finished
        sum_logprob += step_logprob.masked_fill(finished, 0.0)
        lengths += active.long()

        # Already-finished rows keep emitting EOT so tensor shapes stay fixed.
        next_tokens = torch.where(finished, torch.full_like(next_tokens, eot), next_tokens)
        steps.append(next_tokens)
        finished = finished | (next_tokens == eot)
        current = next_tokens[:, None]
        if bool(finished.all()):
            break

    generated = (
        torch.stack(steps, dim=1)
        if steps
        else torch.empty((batch, 0), dtype=torch.long, device=device)
    )
    results: list[TranscriptionResult] = []
    for i in range(batch):
        row = generated[i].tolist()
        tokens = row[: row.index(eot)] if eot in row else row
        # ``lengths`` counts every scored step including the EOT emission.
        denom = max(int(lengths[i].item()), 1)
        results.append(
            TranscriptionResult(
                text=tokenizer.decode(tokens).strip(),
                tokens=tuple(tokens),
                avg_logprob=float(sum_logprob[i].item()) / denom,
            )
        )
    return results


def _beam_search(
    model: WhisperLite,
    tokenizer: BPETokenizer,
    audio_features: Tensor,
    options: GenerationOptions,
    max_new: int,
    suppress: list[int],
) -> TranscriptionResult:
    """Beam search over one utterance; beams live on the batch dimension."""
    k = options.beam_size
    device = audio_features.device
    eot = tokenizer.eot_id
    vocab = model.config.n_vocab

    audio = audio_features.expand(k, -1, -1)
    cache = model.new_cache()
    current = torch.full((k, 1), tokenizer.sot_id, dtype=torch.long, device=device)
    # All beams start identical, so only beam 0 gets a finite score.
    scores = torch.full((k,), float("-inf"), device=device)
    scores[0] = 0.0
    beam_tokens = torch.empty((k, 0), dtype=torch.long, device=device)
    finished: list[tuple[float, list[int]]] = []  # (sum logprob incl. EOT, tokens)

    for _ in range(max_new):
        logprobs = _step_logprobs(model, current, audio, cache, suppress)
        candidates = (scores[:, None] + logprobs).view(-1)
        # 2k candidates guarantee >= k non-EOT continuations (each source beam
        # contributes at most one EOT candidate).
        top_scores, top_indices = candidates.topk(min(2 * k, candidates.numel()))

        keep_sources: list[int] = []
        keep_tokens: list[int] = []
        keep_scores: list[float] = []
        for score, flat_index in zip(top_scores.tolist(), top_indices.tolist(), strict=False):
            source, token = divmod(flat_index, vocab)
            if token == eot:
                finished.append((score, beam_tokens[source].tolist()))
            else:
                keep_sources.append(source)
                keep_tokens.append(token)
                keep_scores.append(score)
            if len(keep_sources) == k:
                break

        if not keep_sources or len(finished) >= k:
            break

        source_index = torch.tensor(keep_sources, dtype=torch.long, device=device)
        cache.reorder(source_index)
        new_tokens = torch.tensor(keep_tokens, dtype=torch.long, device=device)
        beam_tokens = torch.cat(
            [beam_tokens.index_select(0, source_index), new_tokens[:, None]], dim=1
        )
        scores = torch.tensor(keep_scores, device=device)
        current = new_tokens[:, None]

    if not finished:
        # Context limit reached with no EOT: fall back to the live beams.
        finished = [
            (float(scores[b].item()), beam_tokens[b].tolist())
            for b in range(k)
            if scores[b].item() > float("-inf")
        ]

    def ranked(entry: tuple[float, list[int]]) -> float:
        score, tokens = entry
        length = max(len(tokens) + 1, 1)  # +1 for the EOT emission
        return score / (length**options.length_penalty)

    best_score, best_tokens = max(finished, key=ranked)
    return TranscriptionResult(
        text=tokenizer.decode(best_tokens).strip(),
        tokens=tuple(best_tokens),
        avg_logprob=best_score / max(len(best_tokens) + 1, 1),
    )
