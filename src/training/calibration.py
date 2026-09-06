"""Held-out Normal Calibration module for operational threshold calibration.

Principles:
1. One-Class calibration: Thresholds are established solely on normal (non-defective) samples.
2. Zero Defect Leakage: Defect samples and test images are NEVER used during calibration.
3. Operational Policy:
   - review_threshold: 95th percentile of normal image anomaly scores.
   - fail_threshold: 99th percentile of normal image anomaly scores.
   - pixel_threshold: 99th percentile of all pixels across normal heatmaps.
4. Note: P95/P99 are operating policy choices calibrated on normal data, NOT defect-optimized thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from ..model.artifacts import ThresholdPolicy


def split_normal_paths(
    paths: list[Path],
    calibration_fraction: float = 0.2,
    seed: int = 42,
    min_calibration_samples: int = 20,
) -> tuple[list[Path], list[Path]]:
    """Split normal training images reproducibly into Memory Set and Calibration Set.

    Args:
        paths: List of normal image file paths (train/good).
        calibration_fraction: Fraction of images reserved for held-out calibration.
        seed: Random seed for shuffling.
        min_calibration_samples: Minimum required calibration images for reliable quantiles.

    Returns:
        tuple[list[Path], list[Path]]: (memory_paths, calibration_paths).

    Raises:
        ValueError: If total sample count or calibration count is below min_calibration_samples.
    """
    if len(paths) < min_calibration_samples:
        raise ValueError(
            f"Total normal images ({len(paths)}) is less than minimum required calibration samples ({min_calibration_samples})."
        )

    memory, calibration = train_test_split(
        sorted(paths), test_size=calibration_fraction, random_state=seed, shuffle=True
    )

    if len(calibration) < min_calibration_samples:
        raise ValueError(
            f"Resulting calibration set ({len(calibration)}) is smaller than required minimum ({min_calibration_samples}). "
            "Please increase calibration_fraction or provide more train/good images."
        )

    return sorted(memory), sorted(calibration)


def calibrate_thresholds(
    normal_scores: list[float],
    normal_heatmaps: list[np.ndarray],
    review_quantile: float = 0.95,
    fail_quantile: float = 0.99,
    pixel_quantile: float = 0.99,
) -> ThresholdPolicy:
    """Calibrate operational threshold policy based on normal distribution percentiles.

    Args:
        normal_scores: Anomaly scores of held-out normal calibration images.
        normal_heatmaps: 2D smoothed anomaly maps of normal calibration images.
        review_quantile: Quantile for REVIEW warning threshold (default: 0.95).
        fail_quantile: Quantile for FAIL / Image defect threshold (default: 0.99).
        pixel_quantile: Quantile across all normal heatmap pixels (default: 0.99).

    Returns:
        ThresholdPolicy: Configured policy object.

    Raises:
        ValueError: If normal_scores is empty.
    """
    if not normal_scores:
        raise ValueError("normal_scores list is empty, cannot calibrate thresholds.")

    review_threshold = float(np.quantile(normal_scores, review_quantile))
    fail_threshold = float(np.quantile(normal_scores, fail_quantile))

    if normal_heatmaps:
        all_pixels = np.concatenate([h.ravel() for h in normal_heatmaps])
        pixel_threshold = float(np.quantile(all_pixels, pixel_quantile))
    else:
        pixel_threshold = fail_threshold

    return ThresholdPolicy(
        review_threshold=review_threshold,
        fail_threshold=fail_threshold,
        pixel_threshold=pixel_threshold,
    )
