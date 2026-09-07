"""Các hàm inference nhẹ; detector torch được lazy-load khi cần runtime."""

from __future__ import annotations

from .decision import OperationalPolicy, classify_decision, classify_decision_and_severity
from .localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)
def __getattr__(name: str):
    """Chỉ nạp detector khi caller thực sự cần runtime torch."""
    if name == "AnomalyDetector":
        from .detector import AnomalyDetector

        return AnomalyDetector
    if name in {"compute_image_score", "compute_patch_distances"}:
        from .scoring import compute_image_score, compute_patch_distances

        return {
            "compute_image_score": compute_image_score,
            "compute_patch_distances": compute_patch_distances,
        }[name]
    raise AttributeError(name)

__all__ = [
    "AnomalyDetector",
    "OperationalPolicy",
    "apply_heatmap_smoothing",
    "compute_anomalous_area_ratio",
    "create_heatmap_overlay_b64",
    "classify_decision_and_severity",
    "classify_decision",
    "compute_patch_distances",
    "compute_image_score",
]
