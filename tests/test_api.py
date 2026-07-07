"""HTTP API: auth, rate limiting, limits, transcription, operations endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from whisperlite.serving.app import create_app
from whisperlite.serving.auth import hash_key
from whisperlite.serving.ratelimit import TokenBucketLimiter
from whisperlite.serving.settings import ServingConfigError, ServingSettings

API_KEY = "test-key-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


def make_settings(checkpoint_path: Path, **kwargs) -> ServingSettings:
    defaults = dict(
        checkpoint_path=checkpoint_path,
        device="cpu",
        api_keys=(API_KEY,),
        rate_limit_rpm=6000,
        rate_limit_burst=1000,
        max_concurrency=1,
        log_json=False,
    )
    defaults.update(kwargs)
    return ServingSettings(**defaults)


@pytest.fixture(scope="module")
def client(checkpoint_path):
    app = create_app(make_settings(checkpoint_path))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestSettings:
    def test_auth_enabled_requires_keys(self, checkpoint_path):
        with pytest.raises(ServingConfigError, match="WHISPERLITE_API_KEYS"):
            ServingSettings(checkpoint_path=checkpoint_path, api_keys=())

    def test_short_keys_rejected(self, checkpoint_path):
        with pytest.raises(ServingConfigError, match="16 characters"):
            ServingSettings(checkpoint_path=checkpoint_path, api_keys=("short",))

    def test_from_env_parses_everything(self, checkpoint_path):
        settings = ServingSettings.from_env(
            {
                "WHISPERLITE_CHECKPOINT": str(checkpoint_path),
                "WHISPERLITE_API_KEYS": f"{API_KEY}, second-key-9876543210fedcba",
                "WHISPERLITE_RATE_LIMIT_RPM": "30",
                "WHISPERLITE_MAX_UPLOAD_MB": "10",
                "WHISPERLITE_DEVICE": "cpu",
                "WHISPERLITE_BEAM_SIZE": "2",
            }
        )
        assert settings.api_keys == (API_KEY, "second-key-9876543210fedcba")
        assert settings.rate_limit_rpm == 30
        assert settings.max_upload_bytes == 10 * 1024 * 1024
        assert settings.beam_size == 2

    def test_from_env_requires_checkpoint(self):
        with pytest.raises(ServingConfigError, match="WHISPERLITE_CHECKPOINT"):
            ServingSettings.from_env({})

    def test_invalid_env_types_rejected(self, checkpoint_path):
        with pytest.raises(ServingConfigError, match="integer"):
            ServingSettings.from_env(
                {
                    "WHISPERLITE_CHECKPOINT": str(checkpoint_path),
                    "WHISPERLITE_API_KEYS": API_KEY,
                    "WHISPERLITE_RATE_LIMIT_RPM": "sixty",
                }
            )


class TestRateLimiterUnit:
    def test_burst_then_deny_then_refill(self, monkeypatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("whisperlite.serving.ratelimit.time.monotonic", lambda: clock["now"])
        limiter = TokenBucketLimiter(rpm=60, burst=2)  # 1 token/s, burst 2
        assert limiter.check("k")[0]
        assert limiter.check("k")[0]
        allowed, retry_after = limiter.check("k")
        assert not allowed
        assert retry_after == pytest.approx(1.0)
        clock["now"] += 1.5
        assert limiter.check("k")[0]

    def test_keys_are_independent(self):
        limiter = TokenBucketLimiter(rpm=60, burst=1)
        assert limiter.check("a")[0]
        assert limiter.check("b")[0]

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(rpm=0, burst=1)


class TestAuth:
    def test_missing_token_401(self, client):
        response = client.post("/v1/audio/transcriptions")
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == "unauthorized"
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_wrong_token_401(self, client, sample_wav_bytes):
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer wrong-key-000000000000"},
            files={"file": ("t.wav", sample_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 401

    def test_hash_key_stable_and_opaque(self):
        assert hash_key(API_KEY) == hash_key(API_KEY)
        assert API_KEY not in hash_key(API_KEY)
        assert len(hash_key(API_KEY)) == 16


class TestTranscriptionEndpoint:
    def test_happy_path(self, client, sample_wav_bytes):
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("tone.wav", sample_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body) == {"text", "duration", "chunks", "request_id"}
        assert body["duration"] == pytest.approx(0.5, abs=0.01)
        assert len(body["chunks"]) == 1
        assert body["request_id"] == response.headers["X-Request-ID"]

    def test_decode_params_accepted(self, client, sample_wav_bytes):
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("tone.wav", sample_wav_bytes, "audio/wav")},
            data={"temperature": "0.0", "beam_size": "2"},
        )
        assert response.status_code == 200

    def test_incompatible_params_422(self, client, sample_wav_bytes):
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("tone.wav", sample_wav_bytes, "audio/wav")},
            data={"temperature": "0.5", "beam_size": "2"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_parameters"

    def test_out_of_bounds_params_422(self, client, sample_wav_bytes):
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("tone.wav", sample_wav_bytes, "audio/wav")},
            data={"beam_size": "50"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_undecodable_audio_400(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("junk.wav", b"definitely not audio", "audio/wav")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_audio"

    def test_missing_file_422(self, client):
        response = client.post("/v1/audio/transcriptions", headers=AUTH)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_security_headers_present(self, client):
        response = client.get("/healthz")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Cache-Control"] == "no-store"


class TestUploadLimit:
    def test_oversized_upload_413(self, checkpoint_path, sample_wav_bytes):
        app = create_app(make_settings(checkpoint_path, max_upload_bytes=1024))
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                headers=AUTH,
                files={"file": ("big.wav", sample_wav_bytes, "audio/wav")},
            )
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "payload_too_large"

    def test_audio_duration_limit_413(self, checkpoint_path, sample_wav_bytes):
        app = create_app(make_settings(checkpoint_path, max_audio_seconds=0.2))
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                headers=AUTH,
                files={"file": ("tone.wav", sample_wav_bytes, "audio/wav")},
            )
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "audio_too_long"


class TestRateLimitEndpoint:
    def test_burst_exhaustion_returns_429(self, checkpoint_path, sample_wav_bytes):
        app = create_app(make_settings(checkpoint_path, rate_limit_rpm=1, rate_limit_burst=2))
        with TestClient(app) as client:
            statuses = []
            for _ in range(3):
                response = client.post(
                    "/v1/audio/transcriptions",
                    headers=AUTH,
                    files={"file": ("tone.wav", sample_wav_bytes, "audio/wav")},
                )
                statuses.append(response.status_code)
            assert statuses[:2] == [200, 200]
            assert statuses[2] == 429
            assert response.json()["error"]["code"] == "rate_limited"
            assert "Retry-After" in response.headers


class TestOperationalEndpoints:
    def test_healthz_no_auth(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_readyz_ready(self, client):
        assert client.get("/readyz").json() == {"status": "ready"}

    def test_models_endpoint(self, client):
        response = client.get("/v1/models", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "whisperlite"
        assert body["checkpoint_step"] == 123
        assert body["parameters"] > 0

    def test_models_requires_auth(self, client):
        assert client.get("/v1/models").status_code == 401

    def test_metrics_exposition(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "whisperlite_requests_total" in response.text
        assert "whisperlite_model_loaded 1.0" in response.text

    def test_openapi_served(self, client):
        spec = client.get("/openapi.json").json()
        assert "/v1/audio/transcriptions" in spec["paths"]
