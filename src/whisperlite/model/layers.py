"""Transformer building blocks shared by the audio encoder and text decoder.

The attention implementation uses ``torch.nn.functional.scaled_dot_product_attention``
(FlashAttention / memory-efficient kernels where available) and supports an
explicit key/value cache for fast autoregressive decoding:

* **Self-attention cache** — keys/values of previously decoded positions are
  appended each step, so each step costs O(T) instead of O(T^2).
* **Cross-attention cache** — keys/values are computed from the encoder
  output once and reused for every decoded token.

Caches are plain dataclasses owned by the caller (see
:class:`whisperlite.model.decoder.DecoderCache`), which keeps the modules
stateless and makes beam-search reordering a simple ``index_select``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoids(length: int, channels: int, max_timescale: float = 10_000.0) -> Tensor:
    """Fixed sinusoidal position embeddings (identical to Whisper's)."""
    if channels % 2 != 0:
        raise ValueError(f"channels must be even, got {channels}")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None].float() * inv_timescales[None, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


def causal_mask(n_new: int, offset: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    """Additive attention mask for *n_new* queries at positions ``offset..``.

    Entry ``(i, j)`` is ``-inf`` when key position ``j`` is in the future of
    query position ``offset + i``. Returns ``None`` for single-token steps,
    where every cached position is attendable and no mask is needed.
    """
    if n_new == 1:
        return None
    total = offset + n_new
    mask = torch.full((n_new, total), float("-inf"), device=device, dtype=dtype)
    return mask.triu_(offset + 1)


@dataclass
class AttentionCache:
    """Cached key/value tensors of shape ``(batch, seq, n_state)``."""

    k: Tensor | None = None
    v: Tensor | None = None

    def reorder(self, indices: Tensor) -> None:
        """Reorder the batch dimension (used by beam search)."""
        if self.k is not None:
            self.k = self.k.index_select(0, indices)
        if self.v is not None:
            self.v = self.v.index_select(0, indices)


@dataclass
class BlockCache:
    """Per-decoder-block cache: self-attention and cross-attention halves."""

    self_attn: AttentionCache = field(default_factory=AttentionCache)
    cross_attn: AttentionCache = field(default_factory=AttentionCache)

    def reorder(self, indices: Tensor) -> None:
        self.self_attn.reorder(indices)
        self.cross_attn.reorder(indices)


class MultiHeadAttention(nn.Module):
    """Multi-head attention for self- and cross-attention with optional cache.

    Following Whisper, the key projection has no bias. ``kv_dim`` allows the
    decoder's cross-attention to consume encoder states of a different width.
    """

    def __init__(self, n_state: int, n_head: int, kv_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if n_state % n_head != 0:
            raise ValueError(f"n_state ({n_state}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.dropout = dropout
        kv_dim = kv_dim if kv_dim is not None else n_state
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(kv_dim, n_state, bias=False)
        self.value = nn.Linear(kv_dim, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor | None = None,
        mask: Tensor | None = None,
        cache: AttentionCache | None = None,
    ) -> Tensor:
        q = self.query(x)

        if xa is None:
            # Self-attention: append the new keys/values to the cache.
            k = self.key(x)
            v = self.value(x)
            if cache is not None:
                if cache.k is not None and cache.v is not None:
                    k = torch.cat([cache.k, k], dim=1)
                    v = torch.cat([cache.v, v], dim=1)
                cache.k, cache.v = k, v
        else:
            # Cross-attention: encoder states are static, project them once.
            if cache is not None and cache.k is not None:
                k, v = cache.k, cache.v
            else:
                k = self.key(xa)
                v = self.value(xa)
                if cache is not None:
                    cache.k, cache.v = k, v

        batch, n_q, n_state = q.shape
        head_dim = n_state // self.n_head
        q = q.view(batch, n_q, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, k.shape[1], self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, v.shape[1], self.n_head, head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(batch, n_q, n_state)
        return self.out(attn)


class ResidualAttentionBlock(nn.Module):
    """Pre-LayerNorm transformer block: self-attn, optional cross-attn, MLP."""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        kv_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_ln = nn.LayerNorm(n_state)
        self.attn = MultiHeadAttention(n_state, n_head, dropout=dropout)

        self.cross_attn_ln: nn.LayerNorm | None = None
        self.cross_attn: MultiHeadAttention | None = None
        if cross_attention:
            self.cross_attn_ln = nn.LayerNorm(n_state)
            self.cross_attn = MultiHeadAttention(n_state, n_head, kv_dim=kv_dim, dropout=dropout)

        self.mlp_ln = nn.LayerNorm(n_state)
        self.mlp = nn.Sequential(
            nn.Linear(n_state, 4 * n_state), nn.GELU(), nn.Linear(4 * n_state, n_state)
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        xa: Tensor | None = None,
        mask: Tensor | None = None,
        cache: BlockCache | None = None,
    ) -> Tensor:
        self_cache = cache.self_attn if cache is not None else None
        x = x + self.residual_dropout(self.attn(self.attn_ln(x), mask=mask, cache=self_cache))
        if self.cross_attn is not None and self.cross_attn_ln is not None:
            if xa is None:
                raise ValueError("cross-attention block requires encoder states (xa)")
            cross_cache = cache.cross_attn if cache is not None else None
            x = x + self.residual_dropout(
                self.cross_attn(self.cross_attn_ln(x), xa=xa, cache=cross_cache)
            )
        x = x + self.residual_dropout(self.mlp(self.mlp_ln(x)))
        return x
