"""Tính chỉ số AUPRO (Area Under Per-Region Overlap).

AUPRO@0.3 đánh giá mức độ bao phủ của bản đồ nhiệt dự đoán trên từng thành phần liên thông
(connected component) của ground-truth mask, không bị thiên lệch bởi kích thước vùng lỗi lớn/nhỏ.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def compute_aupro(masks: np.ndarray, maps: np.ndarray, max_fpr: float = 0.3) -> float:
    """Tính AUPRO trong miền FPR ``[0, max_fpr]``.

    Args:
        masks: Mask ground truth boolean [N, H, W], True là pixel lỗi.
        maps: Bản đồ nhiệt dự đoán float32 [N, H, W].
        max_fpr: Giới hạn tỷ lệ dương tính giả, mặc định là 0.3.

    Returns:
        float: Giá trị chỉ số AUPRO chuẩn hóa trong khoảng [0, 1].
    """
    normal = ~masks
    thresholds = np.quantile(maps, np.linspace(0, 1, 200))
    points: list[tuple[float, float]] = []

    for threshold in thresholds:
        predicted = maps >= threshold
        fpr = float(predicted[normal].mean()) if normal.any() else 0.0
        if fpr > max_fpr:
            continue

        overlaps: list[float] = []
        for mask, pred in zip(masks, predicted, strict=True):
            components, count = label(mask)
            for component_id in range(1, count + 1):
                region = components == component_id
                overlaps.append(float(pred[region].mean()))

        points.append((fpr, float(np.mean(overlaps)) if overlaps else 0.0))

    if len(points) < 2:
        return 0.0

    best_by_fpr: dict[float, float] = {}
    for fpr, pro in points:
        best_by_fpr[fpr] = max(pro, best_by_fpr.get(fpr, 0.0))

    x = np.array(sorted(best_by_fpr))
    y = np.array([best_by_fpr[value] for value in x])

    if x[-1] < max_fpr:
        x = np.append(x, max_fpr)
        y = np.append(y, y[-1])

    return float(np.trapezoid(y, x) / max_fpr)
