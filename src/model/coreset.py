"""Chọn coreset Greedy K-Center để rút gọn Memory Bank.

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
Memory Bank cuối [K, D]

Phép chiếu ngẫu nhiên 64D chỉ dùng để tăng tốc chọn coreset; toàn bộ truy vấn
1-NN sau đó vẫn hoạt động trên vector D chiều gốc.
"""

from __future__ import annotations

import numpy as np


def select_coreset_indices(
    features: np.ndarray,
    size: int,
    seed: int = 42,
    projection_dim: int = 64,
) -> np.ndarray:
    """Chọn index đại diện bằng Greedy K-Center và Random Projection.

    Args:
        features: Mảng embedding patch 2D [N, D].
        size: Số mẫu coreset cần giữ lại (K).
        seed: Seed cho ma trận chiếu và index khởi đầu.
        projection_dim: Số chiều phép chiếu Johnson-Lindenstrauss.

    Returns:
        np.ndarray: Mảng index nguyên [K].

    Raises:
        ValueError: Nếu size không dương hoặc vượt số phần tử.
    """
    if size <= 0:
        raise ValueError(f"Kích thước coreset phải là một số nguyên dương > 0, nhận được: {size}.")
    n_samples, dim = features.shape
    if n_samples <= size:
        return np.arange(n_samples, dtype=int)

    rng = np.random.default_rng(seed)

    # 1. Chiếu ngẫu nhiên Johnson-Lindenstrauss để tăng tốc khoảng cách Euclid.
    if dim > projection_dim:
        proj_matrix = rng.normal(size=(dim, projection_dim)).astype(np.float32)
        proj_matrix /= np.sqrt(float(projection_dim))
        projected = features @ proj_matrix
    else:
        projected = features

    # 2. Chọn điểm khởi đầu theo seed.
    selected: list[int] = [int(rng.integers(n_samples))]
    min_distances = np.full(n_samples, np.inf, dtype=np.float32)

    # 3. Lặp và chọn điểm có khoảng cách tối thiểu lớn nhất tới tâm đã chọn.
    for _ in range(1, size):
        last_center = projected[selected[-1]]
        dist_sq = np.sum((projected - last_center) ** 2, axis=1)
        min_distances = np.minimum(min_distances, dist_sq)
        selected.append(int(np.argmax(min_distances)))

    return np.asarray(selected, dtype=int)


def greedy_coreset(features: np.ndarray, size: int, seed: int = 42) -> np.ndarray:
    """Trả về vector feature gốc [K, D] sau khi chọn Greedy K-Center.

    Args:
        features: Mảng embedding patch gốc [N, D], ví dụ D=384.
        size: Số patch mục tiêu (K).
        seed: Seed để tái lập kết quả.

    Returns:
        np.ndarray: Embedding patch coreset ở số chiều gốc [K, D].
    """
    indices = select_coreset_indices(features=features, size=size, seed=seed)
    return features[indices]
