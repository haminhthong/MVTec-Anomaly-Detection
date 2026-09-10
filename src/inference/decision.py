"""Quyết định triage rõ ràng cho anomaly score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.artifacts import Thresholds


@dataclass(frozen=True)
class OperationalPolicy:
    """Policy runtime chỉ có image threshold và pixel threshold."""

    image_threshold: float
    pixel_threshold: float

    @classmethod
    def from_thresholds(cls, thresholds: Thresholds) -> "OperationalPolicy":
        """Tạo policy từ metadata model."""
        return cls(
            image_threshold=thresholds.image_threshold,
            pixel_threshold=thresholds.pixel_threshold,
        )


def classify_decision(
    anomaly_score: float,
    thresholds: Thresholds | None = None,
    operational_policy: OperationalPolicy | None = None,
    image_threshold: float | None = None,
) -> str:
    """Phân loại thành PASS_CANDIDATE hoặc REVIEW_REQUIRED."""
    if operational_policy is not None:
        threshold = operational_policy.image_threshold
    elif thresholds is not None:
        threshold = thresholds.image_threshold
    elif image_threshold is not None:
        threshold = float(image_threshold)
    else:
        raise ValueError("Cần cung cấp thresholds, operational_policy hoặc image_threshold.")
    return "PASS_CANDIDATE" if float(anomaly_score) < threshold else "REVIEW_REQUIRED"


def classify_decision_and_severity(
    anomaly_score: float,
    thresholds: Thresholds | None = None,
    operational_policy: OperationalPolicy | None = None,
    image_threshold: float | None = None,
    anomalous_area_ratio: float = 0.0,
    peak_score: float = 0.0,
) -> tuple[str, str | None]:
    """Trả decision và mô tả extent; extent không phải nhãn business."""
    decision = classify_decision(
        anomaly_score,
        thresholds=thresholds,
        operational_policy=operational_policy,
        image_threshold=image_threshold,
    )
    if decision == "PASS_CANDIDATE":
        return decision, "NORMAL"
    extent = "ANOMALY_EXTENSIVE" if anomalous_area_ratio >= 0.05 else "ANOMALY_LOCALIZED"
    return decision, extent
