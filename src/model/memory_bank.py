"""Memory bank và chỉ mục 1-NN cho patch embeddings.

Memory bank được lưu dưới dạng mảng NumPy; chỉ mục NearestNeighbors được dựng
lại khi detector khởi động.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


class MemoryBank:
    """Quản lý tập patch đại diện bình thường và chỉ mục tìm kiếm láng giềng 1-NN.

    Attributes:
        vectors: Mảng numpy 2D [K, Dim] chứa các vector đặc trưng coreset.
        index: Đối tượng NearestNeighbors đã fit trên vectors.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0 or array.shape[1] == 0:
            raise ValueError("Memory bank phải là mảng 2 chiều không rỗng [K, Dim].")
        if not np.isfinite(array).all():
            raise ValueError("Memory bank chỉ được chứa giá trị số hữu hạn.")
        self.vectors: np.ndarray = np.ascontiguousarray(array)
        self.index: NearestNeighbors = NearestNeighbors(
            n_neighbors=1, metric="euclidean", n_jobs=-1
        ).fit(self.vectors)

    @property
    def size(self) -> int:
        """Số lượng vector patch trong memory bank."""
        return len(self.vectors)

    @property
    def dim(self) -> int:
        """Số chiều của vector patch embedding."""
        return self.vectors.shape[1]

    @property
    def features(self) -> np.ndarray:
        """Trả về mảng vector trong memory bank."""
        return self.vectors

    def kneighbors(self, query_patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tìm khoảng cách tới vector láng giềng gần nhất trong memory bank.

        Args:
            query_patches: Mảng [N_patches, Dim] các vector cần tra cứu.

        Returns:
            tuple[np.ndarray, np.ndarray]: (distances [N, 1], indices [N, 1]).
        """
        query = np.asarray(query_patches, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != self.dim:
            raise ValueError(
                f"Query patches phải có shape [N, {self.dim}], nhận được {query.shape}."
            )
        if not np.isfinite(query).all():
            raise ValueError("Query patches chỉ được chứa giá trị số hữu hạn.")
        return self.index.kneighbors(query)

    def save(self, file_path: str | Path) -> None:
        """Lưu trữ memory bank ra tập tin numpy .npy."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, self.vectors)

    @classmethod
    def load(cls, file_path: str | Path) -> MemoryBank:
        """Tải memory bank từ tập tin numpy .npy và khởi tạo 1-NN index.

        Args:
            file_path: Đường dẫn tới file ``memory_bank.npy``.

        Raises:
            FileNotFoundError: Nếu file không tồn tại.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy tập tin memory bank tại '{p}'.")
        vectors = np.load(p)
        return cls(vectors)
