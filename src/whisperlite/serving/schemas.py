"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkSchema(BaseModel):
    """Transcription of one fixed-length audio window."""

    start: float = Field(description="Chunk start time in seconds")
    end: float = Field(description="Chunk end time in seconds")
    text: str = Field(description="Transcribed text for this chunk")
    avg_logprob: float = Field(
        description="Mean token log-probability; low values indicate low confidence"
    )


class TranscriptionResponse(BaseModel):
    """Successful transcription result."""

    text: str = Field(description="Full transcript of the uploaded audio")
    duration: float = Field(description="Decoded audio duration in seconds")
    chunks: list[ChunkSchema] = Field(description="Per-window transcription details")
    request_id: str = Field(description="Server-assigned request identifier")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "hello world",
                    "duration": 2.4,
                    "chunks": [
                        {"start": 0.0, "end": 2.4, "text": "hello world", "avg_logprob": -0.21}
                    ],
                    "request_id": "9f2b1c0a4d6e8f01",
                }
            ]
        }
    }


class ModelInfoResponse(BaseModel):
    """Metadata about the loaded model."""

    name: str
    version: str
    parameters: int
    checkpoint_step: int
    vocab_size: int
    sample_rate: int
    chunk_length: float
    device: str


class HealthResponse(BaseModel):
    status: str


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable explanation")
    request_id: str | None = Field(default=None)


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for all non-2xx responses."""

    error: ErrorDetail
