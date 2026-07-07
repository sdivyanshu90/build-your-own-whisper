"""API route handlers (versioned under ``/v1``)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from whisperlite.serving.auth import authenticate
from whisperlite.serving.schemas import (
    ChunkSchema,
    ErrorResponse,
    ModelInfoResponse,
    TranscriptionResponse,
)
from whisperlite.version import __version__

router = APIRouter(prefix="/v1", tags=["transcription"])

#: Slack allowed between Content-Length and the file payload for multipart
#: boundaries and form fields.
_MULTIPART_OVERHEAD = 16 * 1024
_READ_CHUNK = 1024 * 1024


class ApiError(HTTPException):
    """HTTPException carrying a stable machine-readable error code."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = code


def _service(request: Request):
    service = request.app.state.service
    if service is None:  # pragma: no cover - lifespan always runs before routes
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready", "model is not loaded")
    return service


def _enforce_rate_limit(request: Request, caller: str) -> None:
    allowed, retry_after = request.app.state.limiter.check(caller)
    if not allowed:
        exc = ApiError(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate_limited",
            "rate limit exceeded; retry later",
        )
        exc.headers = {"Retry-After": str(max(1, round(retry_after)))}
        raise exc


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, aborting with 413 as soon as the cap is exceeded."""
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ApiError(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "payload_too_large",
                f"upload exceeds the {max_bytes // (1024 * 1024)} MiB limit",
            )
        parts.append(chunk)
    if total == 0:
        raise ApiError(status.HTTP_400_BAD_REQUEST, "invalid_audio", "uploaded file is empty")
    return b"".join(parts)


@router.post(
    "/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
    responses={
        400: {"model": ErrorResponse, "description": "Undecodable or empty audio"},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        413: {"model": ErrorResponse, "description": "Upload or audio duration too large"},
        422: {"model": ErrorResponse, "description": "Invalid decoding parameters"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, flac, ogg, ...)"),
    temperature: float | None = Form(
        default=None, ge=0.0, le=2.0, description="Sampling temperature; 0 = deterministic"
    ),
    beam_size: int | None = Form(
        default=None, ge=1, le=8, description="Beam width; 1 = greedy decoding"
    ),
    caller: str = Depends(authenticate),
) -> TranscriptionResponse:
    """Upload one audio file and receive its transcript.

    Long recordings are transcribed in fixed windows (see ``chunks`` in the
    response). ``temperature`` and ``beam_size`` override the server defaults
    within safe bounds; beam search requires ``temperature`` to be 0.
    """
    _enforce_rate_limit(request, caller)
    settings = request.app.state.settings
    service = _service(request)

    declared = request.headers.get("content-length")
    if (
        declared
        and declared.isdigit()
        and int(declared) > (settings.max_upload_bytes + _MULTIPART_OVERHEAD)
    ):
        raise ApiError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "payload_too_large",
            f"upload exceeds the {settings.max_upload_bytes // (1024 * 1024)} MiB limit",
        )
    data = await _read_limited(file, settings.max_upload_bytes)

    from whisperlite.audio.features import AudioError
    from whisperlite.model.generation import GenerationError
    from whisperlite.serving.service import AudioTooLongError

    try:
        options = service.default_options(temperature=temperature, beam_size=beam_size)
    except GenerationError as exc:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_parameters", str(exc)
        ) from exc
    try:
        result = await run_in_threadpool(service.transcribe_bytes, data, options)
    except AudioTooLongError as exc:
        raise ApiError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "audio_too_long", str(exc)
        ) from exc
    except AudioError as exc:
        raise ApiError(status.HTTP_400_BAD_REQUEST, "invalid_audio", str(exc)) from exc

    request.app.state.metrics.audio_seconds_total.inc(result.duration)
    return TranscriptionResponse(
        text=result.text,
        duration=result.duration,
        chunks=[ChunkSchema(**chunk.__dict__) for chunk in result.chunks],
        request_id=getattr(request.state, "request_id", ""),
    )


@router.get(
    "/models",
    response_model=ModelInfoResponse,
    summary="Describe the loaded model",
    responses={401: {"model": ErrorResponse}},
)
async def get_model_info(
    request: Request, caller: str = Depends(authenticate)
) -> ModelInfoResponse:
    service = _service(request)
    return ModelInfoResponse(
        name="whisperlite",
        version=__version__,
        parameters=service.model.num_parameters,
        checkpoint_step=service.checkpoint_step,
        vocab_size=service.tokenizer.vocab_size,
        sample_rate=service.audio_config.sample_rate,
        chunk_length=service.audio_config.chunk_length,
        device=str(service.device),
    )
