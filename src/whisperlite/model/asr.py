"""The complete WhisperLite ASR model."""

from __future__ import annotations

from torch import Tensor, nn

from whisperlite.config import ModelConfig
from whisperlite.model.decoder import DecoderCache, TextDecoder
from whisperlite.model.encoder import AudioEncoder


class WhisperLite(nn.Module):
    """Whisper-style sequence-to-sequence speech recognizer.

    ``forward`` implements the teacher-forced training path; incremental
    decoding goes through :meth:`embed_audio` + :meth:`decode_step` with a
    :class:`DecoderCache` (see :mod:`whisperlite.model.generation`).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2-style initialization: N(0, 0.02) weights, zero biases."""
        if isinstance(module, nn.Linear | nn.Conv1d):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def embed_audio(self, mel: Tensor) -> Tensor:
        """Encode ``(batch, n_mels, n_frames)`` mels to encoder states."""
        return self.encoder(mel)

    def decode_step(
        self, tokens: Tensor, audio_features: Tensor, cache: DecoderCache | None = None
    ) -> Tensor:
        """Decoder logits for *tokens* given cached state."""
        return self.decoder(tokens, audio_features, cache=cache)

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        """Teacher-forced logits: ``(batch, seq, n_vocab)``."""
        return self.decoder(tokens, self.encoder(mel))

    def new_cache(self) -> DecoderCache:
        return DecoderCache(self.config.n_text_layer)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
