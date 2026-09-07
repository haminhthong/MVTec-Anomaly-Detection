"""Unit tests kiểm tra công cụ ra quyết định và tính toán diện tích khuyết tật."""

from __future__ import annotations

import numpy as np
from src.inference.decision import classify_decision_and_severity
from src.inference.localization import compute_anomalous_area_ratio


def test_classify_decision_cases() -> None:
    """Kiểm tra policy V1 chỉ có AUTO_PASS và HUMAN_REVIEW."""
    auto_pass_th = 3.0

    # Điểm dưới ngưỡng được tự động thông qua.
    dec, sev = classify_decision_and_severity(
        anomaly_score=2.5, review_threshold=auto_pass_th
    )
    assert dec == "AUTO_PASS"
    assert sev == "NORMAL"

    # Điểm bằng hoặc vượt ngưỡng luôn chuyển người kiểm tra.
    dec, sev = classify_decision_and_severity(
        anomaly_score=3.5, review_threshold=auto_pass_th
    )
    assert dec == "HUMAN_REVIEW"
    assert sev == "ANOMALY_LOCALIZED"

    # Diện tích/peak chỉ là evidence, không tạo business severity.
    dec, sev = classify_decision_and_severity(
        anomaly_score=4.2,
        review_threshold=auto_pass_th,
        anomalous_area_ratio=0.01,
        peak_score=4.5,
    )
    assert dec == "HUMAN_REVIEW"
    assert sev == "ANOMALY_LOCALIZED"


def test_compute_anomalous_area_ratio() -> None:
    """Kiểm tra tính tỷ lệ diện tích pixel vượt ngưỡng trên heatmap."""
    heat = np.zeros((10, 10), dtype=np.float32)
    heat[:2, :5] = 5.0  # 10 pixels trên tổng 100 pixels = 0.10 (10%)
    ratio = compute_anomalous_area_ratio(heat, pixel_threshold=4.0)
    assert abs(ratio - 0.10) < 1e-5
