"""Environment-driven serving configuration (Twelve-Factor style).

Every knob is a ``WHISPERLITE_*`` environment variable so the same container
image runs unchanged across environments. Validation is strict and fails at
startup: a misconfigured server should never come up half-working.

Security posture: authentication is ON by default and the server refuses to
start without API keys unless ``WHISPERLITE_AUTH_ENABLED=0`` is set
explicitly (development only).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class ServingConfigError(ValueError):
    """Raised when serving configuration is missing or invalid."""


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    if raw in ("1", "true", "True", "yes"):
        return True
    if raw in ("0", "false", "False", "no"):
        return False
    raise ServingConfigError(f"{name} must be a boolean flag, got {raw!r}")


def _get_int(env: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ServingConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ServingConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(env: Mapping[str, str], name: str, default: float, minimum: float = 0.0) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ServingConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ServingConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_csv(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = env.get(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class ServingSettings:
    """Validated serving configuration."""

    checkpoint_path: Path
    device: str = "auto"
    api_keys: tuple[str, ...] = field(default_factory=tuple)
    auth_enabled: bool = True
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    max_upload_bytes: int = 25 * 1024 * 1024
    max_audio_seconds: float = 600.0
    beam_size: int = 1
    temperature: float = 0.0
    max_concurrency: int = 2
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    log_level: str = "INFO"
    log_json: bool = True

    def __post_init__(self) -> None:
        if self.auth_enabled and not self.api_keys:
            raise ServingConfigError(
                "authentication is enabled but WHISPERLITE_API_KEYS is empty; "
                "set API keys or explicitly disable auth with "
                "WHISPERLITE_AUTH_ENABLED=0 (development only)"
            )
        for key in self.api_keys:
            if len(key) < 16:
                raise ServingConfigError(
                    "API keys must be at least 16 characters; generate one with "
                    '`python -c "import secrets; print(secrets.token_urlsafe(32))"`'
                )
        if not 1 <= self.beam_size <= 8:
            raise ServingConfigError(f"beam_size must be in [1, 8], got {self.beam_size}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ServingConfigError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.max_upload_bytes < 1024:
            raise ServingConfigError("max_upload_bytes is implausibly small")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServingSettings:
        env = env if env is not None else os.environ
        checkpoint = env.get("WHISPERLITE_CHECKPOINT", "")
        if not checkpoint:
            raise ServingConfigError("WHISPERLITE_CHECKPOINT must point to a model checkpoint")
        return cls(
            checkpoint_path=Path(checkpoint),
            device=env.get("WHISPERLITE_DEVICE", "auto") or "auto",
            api_keys=_get_csv(env, "WHISPERLITE_API_KEYS"),
            auth_enabled=_get_bool(env, "WHISPERLITE_AUTH_ENABLED", True),
            rate_limit_rpm=_get_int(env, "WHISPERLITE_RATE_LIMIT_RPM", 60),
            rate_limit_burst=_get_int(env, "WHISPERLITE_RATE_LIMIT_BURST", 10),
            max_upload_bytes=_get_int(env, "WHISPERLITE_MAX_UPLOAD_MB", 25) * 1024 * 1024,
            max_audio_seconds=_get_float(env, "WHISPERLITE_MAX_AUDIO_SECONDS", 600.0, 1.0),
            beam_size=_get_int(env, "WHISPERLITE_BEAM_SIZE", 1),
            temperature=_get_float(env, "WHISPERLITE_TEMPERATURE", 0.0),
            max_concurrency=_get_int(env, "WHISPERLITE_MAX_CONCURRENCY", 2),
            cors_origins=_get_csv(env, "WHISPERLITE_CORS_ORIGINS"),
            log_level=env.get("WHISPERLITE_LOG_LEVEL", "INFO") or "INFO",
            log_json=_get_bool(env, "WHISPERLITE_LOG_JSON", True),
        )
