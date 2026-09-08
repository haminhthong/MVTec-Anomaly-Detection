"""Kiểm thử các chỉ số localization không phụ thuộc model runtime."""

from __future__ import annotations

import numpy as np

from src.inference.localization import compute_anomalous_area_ratio


def test_zero_pixel_threshold_counts_nonnegative_heatmap() -> None:
    """Threshold bằng 0 phải đếm các pixel anomaly không âm, không trả 0 giả."""
    heatmap = np.zeros((2, 2), dtype=np.float32)
    assert compute_anomalous_area_ratio(heatmap, pixel_threshold=0.0) == 1.0
