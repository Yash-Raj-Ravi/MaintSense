"""Production ASGI server entry point honoring a platform-provided PORT."""

from __future__ import annotations

import os

import uvicorn


def configured_port(value: str | None = None) -> int:
    """Parse and validate the platform's HTTP port (default 8000)."""
    raw_port = value if value is not None else os.getenv("PORT", "8000")
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError("PORT must be an integer from 1 to 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer from 1 to 65535")
    return port


def main() -> None:
    """Start the API without a development reload process."""
    uvicorn.run(
        "maintsense.api:app",
        host="0.0.0.0",
        port=configured_port(),
        reload=False,
    )


if __name__ == "__main__":
    main()
