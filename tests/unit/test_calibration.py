"""Unit tests for calibration and ThresholdPolicy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.model.artifacts import ThresholdPolicy
from src.training.calibration import calibrate_thresholds, split_normal_paths


def test_split_normal_paths_reproducible(tmp_path: Path) -> None:
    """Test reproducibility and disjoint sets."""
    paths = [tmp_path / f"{i:03d}.png" for i in range(100)]
    mem1, cal1 = split_normal_paths(paths, calibration_fraction=0.2, seed=42, min_calibration_samples=20)
    mem2, cal2 = split_normal_paths(paths, calibration_fraction=0.2, seed=42, min_calibration_samples=20)

    assert mem1 == mem2
    assert cal1 == cal2
    assert set(mem1).isdisjoint(cal1)
    assert len(cal1) == 20
    assert len(mem1) == 80


def test_calibrate_thresholds_returns_policy() -> None:
    """Test calibrate_thresholds produces valid ThresholdPolicy."""
    scores = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    heatmaps = [np.ones((10, 10), dtype=np.float32) * s for s in scores]

    policy = calibrate_thresholds(
        normal_scores=scores,
        normal_heatmaps=heatmaps,
        review_quantile=0.90,
        fail_quantile=0.99,
        pixel_quantile=0.99,
    )

    assert isinstance(policy, ThresholdPolicy)
    assert policy.review_threshold < policy.fail_threshold
    assert policy.fail_threshold > 0
    assert policy.pixel_threshold > 0


def test_calibration_rejects_nonfinite_or_mismatched_inputs() -> None:
    """Calibration phải fail-fast khi score hoặc heatmap không hợp lệ."""
    with pytest.raises(ValueError, match="số hữu hạn"):
        calibrate_thresholds([1.0, float("nan")], [])

    with pytest.raises(ValueError, match="cùng số mẫu"):
        calibrate_thresholds([1.0, 2.0], [np.ones((2, 2), dtype=np.float32)])


def test_threshold_policy_rejects_nonfinite_values() -> None:
    """Artifact không được lưu threshold NaN hoặc vô cực."""
    with pytest.raises(ValueError, match="số hữu hạn"):
        ThresholdPolicy(auto_pass_threshold=float("nan"), pixel_threshold=1.0)
