# Multi-stage build: heavy toolchain in the builder, minimal runtime image.
#
# The image installs CPU-only PyTorch (~10x smaller than the CUDA build); for
# GPU serving, switch the index URL to the matching CUDA wheel index and base
# the runtime stage on nvidia/cuda:*-runtime.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
# Install torch first from the CPU wheel index so pip never pulls CUDA wheels.
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN /opt/venv/bin/pip install ".[serve]"

# ---------------------------------------------------------------------------

FROM python:3.11-slim

LABEL org.opencontainers.image.title="whisperlite" \
      org.opencontainers.image.description="Whisper-style ASR serving" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    WHISPERLITE_CHECKPOINT=/models/model.pt

# Run as an unprivileged user; the model volume is mounted read-only.
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv

USER app
WORKDIR /home/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

ENTRYPOINT ["whisperlite", "serve", "--host", "0.0.0.0", "--port", "8000"]
