"""Unit tests for Greedy K-Center Coreset selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.coreset import greedy_coreset, select_coreset_indices


def test_select_coreset_indices() -> None:
    """Test select_coreset_indices returns unique indices in valid range."""
    features = np.random.randn(100, 384).astype(np.float32)
    indices = select_coreset_indices(features, size=15, seed=42)

    assert len(indices) == 15
    assert len(set(indices)) == 15
    assert all(0 <= idx < 100 for idx in indices)


def test_greedy_coreset_size_and_dimension() -> None:
    """Test coreset returns original dimensionality (384D) sliced from input."""
    features = np.random.randn(100, 384).astype(np.float32)
    selected = greedy_coreset(features, size=15, seed=42)

    assert selected.shape == (15, 384)
    for row in selected:
        assert any(np.allclose(row, orig) for orig in features)


def test_greedy_coreset_invalid_size() -> None:
    """Test ValueError raised on size <= 0."""
    features = np.random.randn(20, 384).astype(np.float32)
    with pytest.raises(ValueError, match="Kích thước coreset"):
        select_coreset_indices(features, size=0)
