"""In-process token-bucket rate limiter.

Each caller identity gets a bucket of ``burst`` tokens refilled at
``rpm / 60`` tokens per second. This bounds abuse per worker process; for a
global limit across replicas, put a shared limiter (e.g. an API gateway or
Redis-based limiter) in front — see docs/security.md.
"""

from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    """Thread-safe token bucket keyed by caller identity."""

    _MAX_TRACKED_KEYS = 10_000

    def __init__(self, rpm: int, burst: int):
        if rpm < 1 or burst < 1:
            raise ValueError("rpm and burst must both be >= 1")
        self._rate = rpm / 60.0
        self._capacity = float(burst)
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Consume one token if available; returns ``(allowed, retry_after_s)``."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                self._maybe_evict(now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, (1.0 - tokens) / self._rate

    def _maybe_evict(self, now: float) -> None:
        """Drop buckets that have fully refilled, bounding memory usage."""
        if len(self._buckets) <= self._MAX_TRACKED_KEYS:
            return
        refilled = [
            key
            for key, (tokens, last) in self._buckets.items()
            if tokens + (now - last) * self._rate >= self._capacity
        ]
        for key in refilled:
            del self._buckets[key]
