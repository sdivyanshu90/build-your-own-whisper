"""Prometheus metrics, scoped to one application instance.

A dedicated :class:`CollectorRegistry` (instead of the global default) keeps
tests and multi-app processes free of duplicate-registration errors.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class MetricsBundle:
    """All Prometheus instruments exposed at ``/metrics``."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "whisperlite_requests_total",
            "HTTP requests by route and status code",
            ["route", "status"],
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "whisperlite_request_duration_seconds",
            "HTTP request latency by route",
            ["route"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )
        self.audio_seconds_total = Counter(
            "whisperlite_transcribed_audio_seconds_total",
            "Total seconds of audio transcribed",
            registry=self.registry,
        )
        self.inflight = Gauge(
            "whisperlite_requests_inflight",
            "Requests currently being processed",
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "whisperlite_model_loaded",
            "1 once the model checkpoint has been loaded",
            registry=self.registry,
        )
