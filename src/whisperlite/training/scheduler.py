"""Learning-rate schedules: linear warmup into cosine/linear/constant decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_lr_scheduler(
    optimizer: Optimizer,
    *,
    name: str,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Build a per-step ``LambdaLR``.

    All schedules ramp linearly from ~0 to the base LR over *warmup_steps*,
    then decay to ``base_lr * min_lr_ratio`` at *total_steps*:

    * ``cosine`` — half-cosine decay (the standard choice for speech models),
    * ``linear`` — straight-line decay,
    * ``constant`` — hold the base LR after warmup.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if warmup_steps >= total_steps and name != "constant":
        raise ValueError(f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        if name == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif name == "linear":
            decay = 1.0 - progress
        else:
            raise ValueError(f"unknown scheduler {name!r}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, lr_lambda)
