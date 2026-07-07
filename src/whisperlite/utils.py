"""Small shared utilities: seeding, device resolution, formatting."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Seed every RNG that influences training (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    """Resolve a device spec such as ``auto``, ``cpu``, ``cuda`` or ``cuda:1``."""
    if spec == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {spec!r} requested but CUDA is not available")
    return device


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Total number of (trainable) parameters in *module*."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def format_count(n: int) -> str:
    """Human-readable parameter/sample counts, e.g. ``37.2M``."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)
