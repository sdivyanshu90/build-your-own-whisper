"""Audio encoder: convolutional subsampling + transformer stack."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from whisperlite.config import ModelConfig
from whisperlite.model.layers import ResidualAttentionBlock, sinusoids


class AudioEncoder(nn.Module):
    """Encode ``(batch, n_mels, n_frames)`` log-mels into hidden states.

    Two 1-D convolutions (the second with stride 2) halve the temporal
    resolution — each encoder position covers 20 ms of audio — followed by
    fixed sinusoidal position embeddings and pre-LN transformer blocks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n_state = config.n_audio_state
        self.conv1 = nn.Conv1d(config.n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer(
            "positional_embedding", sinusoids(config.n_audio_ctx, n_state), persistent=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(n_state, config.n_audio_head, dropout=config.dropout)
            for _ in range(config.n_audio_layer)
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, mel: Tensor) -> Tensor:
        if mel.ndim != 3 or mel.shape[1] != self.config.n_mels:
            raise ValueError(
                f"expected mel of shape (batch, {self.config.n_mels}, n_frames), "
                f"got {tuple(mel.shape)}"
            )
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, ctx, n_state)

        n_ctx = x.shape[1]
        if n_ctx > self.config.n_audio_ctx:
            raise ValueError(
                f"audio is too long: {n_ctx} encoder positions exceed the "
                f"maximum context {self.config.n_audio_ctx}"
            )
        x = x + self.positional_embedding[:n_ctx].to(x.dtype)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)
