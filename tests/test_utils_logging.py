"""Utilities and structured logging."""

from __future__ import annotations

import json
import logging

import pytest
import torch

from whisperlite.logging_utils import JsonFormatter, setup_logging
from whisperlite.utils import count_parameters, format_count, resolve_device, set_seed


class TestSeeding:
    def test_torch_reproducibility(self):
        set_seed(123)
        first = torch.randn(4)
        set_seed(123)
        assert torch.equal(first, torch.randn(4))


class TestResolveDevice:
    def test_auto_resolves(self):
        device = resolve_device("auto")
        assert device.type in ("cpu", "cuda")

    def test_cpu_explicit(self):
        assert resolve_device("cpu").type == "cpu"

    def test_cuda_unavailable_raises(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA present on this machine")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(999, "999"), (1_500, "1.5K"), (37_200_000, "37.2M"), (2_000_000_000, "2.0B")],
    )
    def test_format_count(self, value, expected):
        assert format_count(value) == expected

    def test_count_parameters(self):
        module = torch.nn.Linear(10, 5)
        assert count_parameters(module) == 55


class TestJsonLogging:
    def test_json_formatter_includes_extras(self):
        formatter = JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        record.request_id = "abc123"
        payload = json.loads(formatter.format(record))
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"
        assert payload["request_id"] == "abc123"
        assert payload["ts"].endswith("Z")

    def test_json_formatter_serializes_exceptions(self):
        formatter = JsonFormatter()
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            record = logging.LogRecord(
                "test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
            )
        payload = json.loads(formatter.format(record))
        assert "RuntimeError: boom" in payload["exc_info"]

    def test_setup_logging_replaces_handlers(self):
        setup_logging("WARNING", json_format=True)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.WARNING
        setup_logging("INFO", json_format=False)  # restore for other tests
