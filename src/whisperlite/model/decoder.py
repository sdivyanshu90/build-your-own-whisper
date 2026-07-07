"""Text decoder: causal transformer with cross-attention over audio states."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from whisperlite.config import ModelConfig
from whisperlite.model.layers import BlockCache, ResidualAttentionBlock, causal_mask


class DecoderCache:
    """Key/value cache for incremental decoding.

    ``offset`` tracks how many token positions have already been fed to the
    decoder; it determines both position-embedding indices and the causal
    mask for multi-token prefills.
    """

    def __init__(self, n_layers: int):
        self.blocks: list[BlockCache] = [BlockCache() for _ in range(n_layers)]
        self.offset: int = 0

    def reorder(self, indices: Tensor) -> None:
        """Reorder the batch/beam dimension of every cached tensor."""
        for block in self.blocks:
            block.reorder(indices)


class TextDecoder(nn.Module):
    """Decode token sequences conditioned on encoder output.

    Uses learned position embeddings (as in Whisper) and ties the output
    projection to the token embedding matrix, which regularizes the softmax
    layer and saves ``n_vocab * n_state`` parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n_state = config.n_text_state
        self.token_embedding = nn.Embedding(config.n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(config.n_text_ctx, n_state))
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(
                n_state,
                config.n_text_head,
                cross_attention=True,
                kv_dim=config.n_audio_state,
                dropout=config.dropout,
            )
            for _ in range(config.n_text_layer)
        )
        self.ln = nn.LayerNorm(n_state)

    def forward(
        self,
        tokens: Tensor,
        audio_features: Tensor,
        cache: DecoderCache | None = None,
    ) -> Tensor:
        """Return logits of shape ``(batch, n_tokens, n_vocab)``.

        When *cache* is provided, *tokens* holds only the not-yet-processed
        positions and the cache is updated in place.
        """
        if tokens.ndim != 2:
            raise ValueError(f"expected tokens of shape (batch, seq), got {tuple(tokens.shape)}")
        offset = cache.offset if cache is not None else 0
        n_new = tokens.shape[1]
        if offset + n_new > self.config.n_text_ctx:
            raise ValueError(
                f"sequence length {offset + n_new} exceeds text context {self.config.n_text_ctx}"
            )

        x = self.token_embedding(tokens) + self.positional_embedding[offset : offset + n_new]
        x = self.dropout(x)

        mask = causal_mask(n_new, offset, x.device, x.dtype)
        for i, block in enumerate(self.blocks):
            block_cache = cache.blocks[i] if cache is not None else None
            x = block(x, xa=audio_features, mask=mask, cache=block_cache)

        if cache is not None:
            cache.offset += n_new

        x = self.ln(x)
        return x @ self.token_embedding.weight.to(x.dtype).T
