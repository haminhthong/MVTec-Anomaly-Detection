"""Unit tests cho detection, localization và triage metrics."""

from __future__ import annotations

import numpy as np
import pytest
from src.evaluation.metrics import calculate_workflow_metrics


def test_calculate_workflow_metrics_logic() -> None:
    """Kiểm tra benchmark và boundary PASS_CANDIDATE."""
    y_true = [0, 0, 1, 1]
    scores = [1.0, 2.0, 4.0, 5.0]
    masks = np.zeros((4, 8, 8), dtype=bool)
    masks[2:, 2:6, 2:6] = True
    maps = masks.astype(np.float32) * 5.0

    threshold = 3.0  # Hoàn hảo: 2 normal < 3.0, 2 defect >= 3.0
    res = calculate_workflow_metrics(y_true, scores, masks, maps, threshold)

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
    assert op["false_pass_candidate_rate"] == 0.0

    cm = op["confusion_matrix"]
    assert cm["tp_review_required_defect"] == 2
    assert cm["tn_pass_candidate_normal"] == 2
    assert cm["fp_review_required_normal"] == 0
    assert cm["fn_pass_candidate_defect"] == 0


def test_metrics_reject_invalid_labels_and_shape() -> None:
    """Metrics phải fail fast khi label hoặc kích thước map không hợp lệ."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    maps = np.zeros((2, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="nhãn 0"):
        calculate_workflow_metrics([0, 2], [0.1, 0.2], masks, masks, image_threshold=0.5)
    with pytest.raises(ValueError, match="cùng số mẫu"):
        calculate_workflow_metrics([0, 1], [0.1, 0.2], masks, maps, image_threshold=0.5)
