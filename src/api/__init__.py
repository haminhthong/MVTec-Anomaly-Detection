"""Schema API công khai và FastAPI được nạp theo nhu cầu."""

from __future__ import annotations

from .schemas import BatchInspectionResponse, HealthResponse, InspectionResponse


def __getattr__(name: str):
    """Nạp app sau khi schema đã sẵn sàng."""
    if name in {"MODEL_DIR", "app", "health", "live", "ready"}:
        from .app import MODEL_DIR, app, health, live, ready

        return {
            "MODEL_DIR": MODEL_DIR,
            "app": app,
            "health": health,
            "live": live,
            "ready": ready,
        }[name]
    raise AttributeError(name)


__all__ = [
    "app",
    "health",
    "live",
    "ready",
    "MODEL_DIR",
    "HealthResponse",
    "InspectionResponse",
    "BatchInspectionResponse",
]
