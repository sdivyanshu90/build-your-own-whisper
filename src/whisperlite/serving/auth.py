"""Bearer-token authentication.

Keys are compared with :func:`hmac.compare_digest` (constant-time) to avoid
timing side channels, and only SHA-256 digests of keys are used for logging
and rate-limit bucketing so raw credentials never leave the comparison path.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request, status


def hash_key(key: str) -> str:
    """Short, non-reversible identifier for a key (for logs and buckets)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def authenticate(request: Request) -> str:
    """FastAPI dependency: validate ``Authorization: Bearer <key>``.

    Returns the caller identity (hashed key, or ``anonymous`` when auth is
    disabled) used downstream for rate limiting and access logs.
    """
    settings = request.app.state.settings
    if not settings.auth_enabled:
        client = request.client
        return f"anonymous:{client.host if client else 'unknown'}"

    header = request.headers.get("Authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    for key in settings.api_keys:
        if hmac.compare_digest(credential, key):
            return hash_key(key)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
