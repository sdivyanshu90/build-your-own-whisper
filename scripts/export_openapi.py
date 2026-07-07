#!/usr/bin/env python3
"""Export the committed OpenAPI specification from the live FastAPI app.

Usage: python scripts/export_openapi.py docs/openapi.json

The app factory is invoked with a placeholder checkpoint path; the model is
only loaded during lifespan startup, which this script never triggers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from whisperlite.serving.app import create_app
from whisperlite.serving.settings import ServingSettings


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} OUTPUT.json", file=sys.stderr)
        return 2
    settings = ServingSettings(
        checkpoint_path=Path("/nonexistent/model.pt"),
        auth_enabled=False,
        log_json=False,
    )
    spec = create_app(settings).openapi()
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
