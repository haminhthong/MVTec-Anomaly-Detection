"""Schema HTTP ổn định cho inspection triage và human QC."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Thông tin health tổng quát."""

    status: str
    model_ready: bool
    model_version: str
    categories: list[str] = Field(default_factory=list)


class ReadinessResponse(BaseModel):
    """Readiness của service."""

    ready: bool
    categories: list[str]
    active_device: str


class ScoreBreakdown(BaseModel):
    """Score và ngưỡng auto-pass; alias cũ chỉ để tương thích client."""

    anomaly_score: float | None = None
    auto_pass_threshold: float
    review_threshold: float | None = None
    fail_threshold: float | None = None


class LocalizationBreakdown(BaseModel):
    """Bằng chứng không gian, không diễn giải thành business severity."""

    anomalous_area_ratio: float = 0.0
    peak_anomaly_score: float = 0.0
    peak_score: float = 0.0
    pixel_threshold: float


class ModelBreakdown(BaseModel):
    """Identity của model release."""

    version: str
    category: str
    release_id: str | None = None


class InspectionResponse(BaseModel):
    """Contract V1: RECAPTURE_REQUIRED, AUTO_PASS hoặc HUMAN_REVIEW."""

    inspection_id: str = Field(default_factory=lambda: f"insp_{uuid.uuid4().hex[:12]}")
    category: str
    decision: str
    severity: str | None = Field(
        default=None,
        description="Deprecated; không phải nhãn major/minor và không dùng để reject.",
    )
    scores: ScoreBreakdown
    localization: LocalizationBreakdown
    capture_quality: dict[str, Any] = Field(default_factory=dict)
    model: ModelBreakdown
    line_id: str | None = None
    camera_id: str | None = None
    timestamp: str | None = None
    overlay_b64: str | None = None

    # Alias phẳng cho client cũ.
    anomaly_score: float | None = None
    threshold: float | None = None
    heatmap_shape: list[int] | None = None
    model_version: str | None = None


class BatchInspectionResponse(BaseModel):
    """Response của batch inspection."""

    batch_size: int
    category: str
    items: list[InspectionResponse]
