"""Tính điểm anomaly bằng nearest-neighbor theo phong cách PatchCore.

Module chuyển patch embedding thành heatmap khoảng cách và điểm anomaly cấp ảnh.
"""

from __future__ import annotations

import numpy as np

from ..model.memory_bank import MemoryBank


def compute_patch_distances(
    query_patches: np.ndarray, memory_bank: MemoryBank
) -> np.ndarray:
    """Tính khoảng cách Euclid tới láng giềng gần nhất cho từng patch.

    Args:
        query_patches: Mảng 2D embedding query [N_patches, Dim].
        memory_bank: MemoryBank đã fit.

    Returns:
        np.ndarray: Mảng khoảng cách gần nhất [N_patches].
    """
    distances, _ = memory_bank.kneighbors(query_patches)
    return distances.ravel()


def compute_image_score(
    smoothed_heatmap: np.ndarray, percentile: float = 99.0
) -> float:
    """Tính điểm anomaly cấp ảnh từ heatmap 2D đã smoothing.

    Dùng percentile 99 để giảm ảnh hưởng pixel nhiễu đơn lẻ nhưng vẫn nhạy
    với cụm defect cục bộ.

    Args:
        smoothed_heatmap: Heatmap anomaly 2D [H, W].
        percentile: Percentile cần dùng, mặc định 99.0.

    Returns:
        float: Điểm anomaly cấp ảnh.
    """
    if smoothed_heatmap.size == 0:
        return 0.0
    return float(np.percentile(smoothed_heatmap, percentile))
