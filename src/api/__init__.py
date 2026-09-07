"""Schema API nhẹ; FastAPI app được lazy-load khi khởi động service."""

from __future__ import annotations

from .schemas import (
    BatchInspectionResponse,
    HealthResponse,
    InspectionResponse,
    LocalizationBreakdown,
    ModelBreakdown,
    ReadinessResponse,
    ScoreBreakdown,
)


def __getattr__(name: str):
    """Nạp FastAPI app và health khi caller chạy server."""
    if name in {"MODEL_DIR", "app", "health"}:
        from .app import MODEL_DIR, app, health

        return {"MODEL_DIR": MODEL_DIR, "app": app, "health": health}[name]
    raise AttributeError(name)


__all__ = [
    "app",
    "health",
    "MODEL_DIR",
    "HealthResponse",
    "ReadinessResponse",
    "InspectionResponse",
    "BatchInspectionResponse",
    "ScoreBreakdown",
    "LocalizationBreakdown",
    "ModelBreakdown",
]
