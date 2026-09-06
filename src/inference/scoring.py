"""Scoring logic for PatchCore nearest-neighbor anomaly detection.

Maps patch embeddings to distance heatmaps and computes aggregate image-level anomaly scores.
"""

from __future__ import annotations

import numpy as np

from ..model.memory_bank import MemoryBank


def compute_patch_distances(
    query_patches: np.ndarray, memory_bank: MemoryBank
) -> np.ndarray:
    """Compute Euclidean distance to nearest neighbor in memory bank for each patch.

    Args:
        query_patches: 2D array of query patch embeddings [N_patches, Dim].
        memory_bank: Fitted MemoryBank instance.

    Returns:
        np.ndarray: 1D array of nearest neighbor distances [N_patches].
    """
    distances, _ = memory_bank.kneighbors(query_patches)
    return distances.ravel()


def compute_image_score(
    smoothed_heatmap: np.ndarray, percentile: float = 99.0
) -> float:
    """Compute image-level anomaly score from 2D smoothed anomaly map.

    Uses the 99th percentile compromise: robust against isolated noisy pixels
    while remaining highly sensitive to localized defect clusters.

    Args:
        smoothed_heatmap: 2D smoothed anomaly map [H, W].
        percentile: Percentile value to use (default: 99.0).

    Returns:
        float: Image anomaly score.
    """
    if smoothed_heatmap.size == 0:
        return 0.0
    return float(np.percentile(smoothed_heatmap, percentile))
