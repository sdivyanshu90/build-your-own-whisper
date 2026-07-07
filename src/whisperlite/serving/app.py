"""FastAPI application factory.

Responsibilities wired here (and only here):

* model lifecycle (load + warm up during lifespan startup),
* request-ID assignment, access logging, and Prometheus instrumentation,
* security headers on every response,
* a uniform JSON error envelope for all failure paths,
* liveness (``/healthz``), readiness (``/readyz``), and ``/metrics``.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from whisperlite.logging_utils import setup_logging
from whisperlite.serving.metrics import MetricsBundle
from whisperlite.serving.ratelimit import TokenBucketLimiter
from whisperlite.serving.routes import router
from whisperlite.serving.settings import ServingSettings
from whisperlite.version import __version__

logger = logging.getLogger("whisperlite.access")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

_DEFAULT_ERROR_CODES = {
    status.HTTP_401_UNAUTHORIZED: "unauthorized",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: "payload_too_large",
    status.HTTP_422_UNPROCESSABLE_ENTITY: "validation_error",
    status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_error",
}


def _error_response(
    status_code: int, code: str, message: str, request_id: str | None, headers=None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers=headers,
    )


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    """Build a fully configured application (used by CLI, Docker, and tests)."""
    settings = settings or ServingSettings.from_env()
    setup_logging(settings.log_level, settings.log_json)
    metrics = MetricsBundle()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from whisperlite.serving.service import TranscriptionService

        service = TranscriptionService(settings)
        service.warmup()
        app.state.service = service
        metrics.model_loaded.set(1)
        logger.info("whisperlite %s ready to serve", __version__)
        yield
        app.state.service = None
        metrics.model_loaded.set(0)

    app = FastAPI(
        title="WhisperLite API",
        version=__version__,
        description=(
            "Speech-to-text API backed by a from-scratch Whisper-style model. "
            "Authenticate with `Authorization: Bearer <api-key>`."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.limiter = TokenBucketLimiter(settings.rate_limit_rpm, settings.rate_limit_burst)
    app.state.service = None

    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # -- Observability middleware -------------------------------------------

    @app.middleware("http")
    async def observability(request: Request, call_next):
        request_id = uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        metrics.inflight.inc()
        started = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            metrics.inflight.dec()
        elapsed = time.perf_counter() - started

        route = request.scope.get("route")
        route_label = route.path if route is not None else "unmatched"
        metrics.requests_total.labels(route=route_label, status=str(response.status_code)).inc()
        metrics.request_duration.labels(route=route_label).observe(elapsed)

        response.headers["X-Request-ID"] = request_id
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)

        logger.info(
            "%s %s -> %d in %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_s": round(elapsed, 4),
            },
        )
        return response

    # -- Error envelope ------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code = getattr(exc, "error_code", None) or _DEFAULT_ERROR_CODES.get(
            exc.status_code, "error"
        )
        return _error_response(
            exc.status_code,
            code,
            str(exc.detail),
            getattr(request.state, "request_id", None),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = f"{location}: {first.get('msg', 'invalid request')}"
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_error",
            message,
            getattr(request.state, "request_id", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logging.getLogger("whisperlite.serving").exception(
            "unhandled error", extra={"request_id": request_id}
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "an internal error occurred",
            request_id,
        )

    # -- Operational endpoints ------------------------------------------------

    @app.get("/healthz", tags=["operations"], summary="Liveness probe")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", tags=["operations"], summary="Readiness probe")
    async def readyz(request: Request):
        if request.app.state.service is None:
            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready", "model is not loaded", None
            )
        return {"status": "ready"}

    @app.get(
        "/metrics",
        tags=["operations"],
        summary="Prometheus metrics",
        description="Expose only on an internal network; see docs/security.md.",
    )
    async def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)
    return app
