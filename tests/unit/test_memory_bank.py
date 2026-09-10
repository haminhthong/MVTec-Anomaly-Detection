"""Kiểm thử MemoryBank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.model.memory_bank import MemoryBank


def test_memory_bank_nearest_neighbors() -> None:
    """Kiểm tra tạo chỉ mục 1-NN và truy vấn khoảng cách."""
    vectors = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    bank = MemoryBank(vectors)

    assert bank.size == 2
    assert bank.dim == 2

    query = np.array([[0.1, 0.0], [9.9, 10.0]], dtype=np.float32)
    distances, indices = bank.kneighbors(query)

    assert indices[0][0] == 0
    assert indices[1][0] == 1
    assert distances[0][0] < 0.2


def test_memory_bank_save_and_load(tmp_path: Path) -> None:
    """Kiểm tra lưu ra .npy và nạp lại."""
    vectors = np.random.randn(50, 64).astype(np.float32)
    bank = MemoryBank(vectors)
    save_path = tmp_path / "memory_bank.npy"
    bank.save(save_path)

    loaded = MemoryBank.load(save_path)
    assert loaded.size == 50
    assert loaded.dim == 64
    assert np.allclose(loaded.vectors, bank.vectors)


def test_memory_bank_rejects_invalid_query_shape_or_values() -> None:
    """MemoryBank phải báo lỗi rõ khi query sai dimension hoặc chứa NaN."""
    bank = MemoryBank(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        bank.kneighbors(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="hữu hạn"):
        bank.kneighbors(np.array([[0.0, 0.0, 0.0, np.nan]], dtype=np.float32))
