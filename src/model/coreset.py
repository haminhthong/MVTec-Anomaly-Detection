"""Greedy K-Center Coreset selection module for Memory Bank subsampling.

Data flow:
Full Patches [N, D] (e.g. D = 384)
    ↓
Random Projection (Johnson-Lindenstrauss) for fast distance calculation [N, 64]
    ↓
Greedy K-Center Selection on 64D projected space
    ↓
Selected Indices [K]
    ↓
Slice original D-dimensional features (e.g. 384D)
    ↓
Final Memory Bank [K, D]

The 64D random projection is strictly used for coreset selection speedup;
all subsequent 1-NN nearest-neighbor lookups operate on original D-dimensional space.
"""

from __future__ import annotations

import numpy as np


def select_coreset_indices(
    features: np.ndarray,
    size: int,
    seed: int = 42,
    projection_dim: int = 64,
) -> np.ndarray:
    """Select representative subset indices using Greedy K-Center with Random Projection.

    Args:
        features: 2D array of patch embeddings [N, D].
        size: Target number of coreset samples to retain (K).
        seed: Random seed for projection matrix and starting index.
        projection_dim: Dimension for Johnson-Lindenstrauss random projection.

    Returns:
        np.ndarray: 1D array of selected integer indices of shape [K].

    Raises:
        ValueError: If size is non-positive or exceeds array length.
    """
    if size <= 0:
        raise ValueError(f"Kích thước coreset phải là một số nguyên dương > 0, nhận được: {size}.")
    n_samples, dim = features.shape
    if n_samples <= size:
        return np.arange(n_samples, dtype=int)

    rng = np.random.default_rng(seed)

    # 1. Johnson-Lindenstrauss Random Projection to accelerate Euclidean distance calculations
    if dim > projection_dim:
        proj_matrix = rng.normal(size=(dim, projection_dim)).astype(np.float32)
        proj_matrix /= np.sqrt(float(projection_dim))
        projected = features @ proj_matrix
    else:
        projected = features

    # 2. Select initial point randomly
    selected: list[int] = [int(rng.integers(n_samples))]
    min_distances = np.full(n_samples, np.inf, dtype=np.float32)

    # 3. Iteratively pick point maximizing minimum distance to current selected centers
    for _ in range(1, size):
        last_center = projected[selected[-1]]
        dist_sq = np.sum((projected - last_center) ** 2, axis=1)
        min_distances = np.minimum(min_distances, dist_sq)
        selected.append(int(np.argmax(min_distances)))

    return np.asarray(selected, dtype=int)


def greedy_coreset(features: np.ndarray, size: int, seed: int = 42) -> np.ndarray:
    """Greedy K-Center Coreset returning subsampled original feature vectors [K, D].

    Args:
        features: 2D array of original patch embeddings [N, D] (e.g. 384D).
        size: Target number of patches (K).
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Coreset patch embeddings in original dimension [K, D].
    """
    indices = select_coreset_indices(features=features, size=size, seed=seed)
    return features[indices]
