"""Schema HTTP tối giản cho suy luận và kiểm tra thủ công."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Trạng thái service và các category đã có artifact."""

    status: str
    model_ready: bool
    model_version: str
    categories: list[str] = Field(default_factory=list)


class InspectionResponse(BaseModel):
    """Kết quả kiểm tra ảnh, anomaly score và quyết định triage."""

    inspection_id: str = Field(default_factory=lambda: f"insp_{uuid.uuid4().hex[:12]}")
    category: str
    decision: str
    anomaly_score: float | None = None
    image_threshold: float
    anomalous_area_ratio: float = 0.0
    peak_anomaly_score: float = 0.0
    pixel_threshold: float
    capture_quality: dict[str, Any] = Field(default_factory=dict)
    model_version: str
    camera_id: str | None = None
    timestamp: str | None = None
    overlay_b64: str | None = None
    heatmap_shape: list[int] | None = None


class BatchInspectionResponse(BaseModel):
    """Kết quả batch giữ nguyên thứ tự input."""

    batch_size: int
    category: str
    items: list[InspectionResponse]
