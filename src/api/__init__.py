"""Public API schema và FastAPI app lazy-load."""

from __future__ import annotations

from .schemas import BatchInspectionResponse, HealthResponse, InspectionResponse


def __getattr__(name: str):
    """Nạp app sau khi schema đã sẵn sàng."""
    if name in {"MODEL_DIR", "app", "health"}:
        from .app import MODEL_DIR, app, health

        return {"MODEL_DIR": MODEL_DIR, "app": app, "health": health}[name]
    raise AttributeError(name)


__all__ = [
    "app",
    "health",
    "MODEL_DIR",
    "HealthResponse",
    "InspectionResponse",
    "BatchInspectionResponse",
]
