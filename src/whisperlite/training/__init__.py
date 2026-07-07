"""Training loop, LR schedules, and evaluation metrics."""

from whisperlite.training.metrics import (
    char_error_rate,
    edit_distance,
    normalize_text,
    word_error_rate,
)
from whisperlite.training.scheduler import create_lr_scheduler
from whisperlite.training.trainer import Trainer

__all__ = [
    "Trainer",
    "char_error_rate",
    "create_lr_scheduler",
    "edit_distance",
    "normalize_text",
    "word_error_rate",
]
