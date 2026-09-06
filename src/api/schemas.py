"""Pydantic Schemas for Industrial Visual Anomaly Detection API.

Provides stable data contracts:
- Monitoring (/health, /health/live, /health/ready)
- Model registry (/models, /models/{category})
- Single and batch inspection endpoints (/inspect, /inspect/batch)
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for basic health check endpoint."""

    status: str = Field(..., description="Service status: 'ok' or 'degraded'")
    model_ready: bool = Field(..., description="True if at least one model is available")
    model_version: str = Field(..., description="Active model version")
    categories: list[str] = Field(default_factory=list, description="Available product categories")


class ReadinessResponse(BaseModel):
    """Schema for readiness probe endpoint."""

    ready: bool = Field(..., description="True if models and index are ready to serve")
    categories: list[str] = Field(..., description="Loaded category models")
    active_device: str = Field(..., description="Hardware device ('cuda' or 'cpu')")


class ScoreBreakdown(BaseModel):
    """Anomaly scores and operating thresholds."""

    anomaly_score: float = Field(..., description="99th percentile image anomaly score")
    review_threshold: float = Field(..., description="P95 normal calibration review threshold")
    fail_threshold: float = Field(..., description="P99 normal calibration fail threshold")


class LocalizationBreakdown(BaseModel):
    """Localization details and surface defect area ratio."""

    anomalous_area_ratio: float = Field(..., description="Fraction of surface exceeding pixel threshold")
    peak_score: float = Field(..., description="Peak anomaly distance in heatmap")
    pixel_threshold: float = Field(..., description="P99 normal heatmap pixel threshold")


class ModelBreakdown(BaseModel):
    """Model version and category identity."""

    version: str = Field(..., description="Model artifact version")
    category: str = Field(..., description="Product category name")


class InspectionResponse(BaseModel):
    """Stable production response schema for /inspect."""

    inspection_id: str = Field(
        default_factory=lambda: f"insp_{uuid.uuid4().hex[:12]}",
        description="Unique inspection transaction identifier",
    )
    category: str = Field(..., description="Product category inspected")
    decision: str = Field(
        ..., description="Operational decision: 'PASS', 'REVIEW', or 'FAIL'"
    )
    severity: str = Field(
        ..., description="Severity classification: 'PASS', 'REVIEW', 'FAIL_MINOR', 'FAIL_MAJOR'"
    )
    scores: ScoreBreakdown = Field(..., description="Breakdown of anomaly score and thresholds")
    localization: LocalizationBreakdown = Field(
        ..., description="Defect localization metrics and anomalous area ratio"
    )
    model: ModelBreakdown = Field(..., description="Model version and metadata")
    overlay_b64: str | None = Field(
        None, description="Base64 encoded PNG heatmap overlay image"
    )

    # Convenience backward compatibility aliases
    anomaly_score: float | None = Field(default=None, description="Flat anomaly score alias")
    threshold: float | None = Field(default=None, description="Flat threshold alias")
    heatmap_shape: list[int] | None = Field(default=None, description="Heatmap grid dimensions")
    model_version: str | None = Field(default=None, description="Flat model version alias")


class BatchInspectionResponse(BaseModel):
    """Schema for /inspect/batch endpoint."""

    batch_size: int = Field(..., description="Number of items inspected in this batch")
    category: str = Field(..., description="Product category inspected")
    items: list[InspectionResponse] = Field(..., description="Inspection responses for each input image")
