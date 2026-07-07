"""Production HTTP serving for WhisperLite (FastAPI).

Import :func:`whisperlite.serving.app.create_app` to build an application
instance; the ``whisperlite serve`` CLI and the Docker image both go through
that factory.
"""

from whisperlite.serving.settings import ServingConfigError, ServingSettings

__all__ = ["ServingConfigError", "ServingSettings"]
