"""Inference package for PatchCore-style visual anomaly detection."""

from __future__ import annotations

from .decision import OperationalPolicy, classify_decision_and_severity
from .detector import AnomalyDetector
from .localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)
from .scoring import compute_image_score, compute_patch_distances

__all__ = [
    "AnomalyDetector",
    "OperationalPolicy",
    "apply_heatmap_smoothing",
    "compute_anomalous_area_ratio",
    "create_heatmap_overlay_b64",
    "classify_decision_and_severity",
    "compute_patch_distances",
    "compute_image_score",
]
