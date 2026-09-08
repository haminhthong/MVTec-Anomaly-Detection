"""Unit tests kiểm tra tính toán chỉ số đánh giá 3 tầng và ma trận nhầm lẫn."""

from __future__ import annotations

import numpy as np
import pytest
from src.evaluation.metrics import calculate_3tier_metrics


def test_calculate_3tier_metrics_logic() -> None:
    """Kiểm tra độ chính xác của 3 tầng metrics: Detection, Localization, Operational QC."""
    y_true = [0, 0, 1, 1]
    scores = [1.0, 2.0, 4.0, 5.0]
    masks = np.zeros((4, 8, 8), dtype=bool)
    masks[2:, 2:6, 2:6] = True
    maps = masks.astype(np.float32) * 5.0

    threshold = 3.0  # Hoàn hảo: 2 normal < 3.0, 2 defect >= 3.0
    res = calculate_3tier_metrics(y_true, scores, masks, maps, threshold)

    assert "detection" in res
    assert "localization" in res
    assert "operational_decision" in res

    # Tier 1
    assert res["detection"]["image_auroc"] == 1.0
    assert res["detection"]["image_average_precision"] == 1.0

    # Tier 2
    assert res["localization"]["pixel_auroc"] > 0.99

    # Tier 3
    op = res["operational_decision"]
    assert op["accuracy"] == 1.0
    assert op["precision"] == 1.0
    assert op["defect_recall"] == 1.0
    assert op["specificity"] == 1.0
    assert op["false_reject_rate"] == 0.0
    assert op["false_accept_rate"] == 0.0

    cm = op["confusion_matrix"]
    assert cm["tp"] == 2
    assert cm["tn"] == 2
    assert cm["fp"] == 0
    assert cm["fn"] == 0


def test_metrics_reject_invalid_labels_and_shape() -> None:
    """Metrics phải fail fast khi label hoặc kích thước map không hợp lệ."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    maps = np.zeros((2, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="nhãn 0"):
        calculate_3tier_metrics([0, 2], [0.1, 0.2], masks, masks, threshold=0.5)
    with pytest.raises(ValueError, match="cùng số mẫu"):
        calculate_3tier_metrics([0, 1], [0.1, 0.2], masks, maps, threshold=0.5)
