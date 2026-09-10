"""Dataset và bộ nạp ảnh MVTec AD.

Dataset đọc ảnh từ danh sách đường dẫn, áp dụng transform và dùng manifest
đã được validate để bảo đảm đúng boundary dữ liệu.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .transforms import TFM
from .validation import _find_category_root


class ImageFolderDataset(Dataset):
    """Dataset PyTorch đọc ảnh từ một danh sách đường dẫn.

    Args:
        paths: Danh sách đường dẫn ảnh dạng ``Path`` hoặc chuỗi.
        transform: Hàm transform từ ``PIL.Image`` sang tensor.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        transform: Callable[[Image.Image], Tensor] | transforms.Compose | None = None,
    ) -> None:
        self.paths: list[Path] = [Path(p) for p in paths]
        self.transform: Callable[[Image.Image], Tensor] = (
            transform if transform is not None else TFM
        )

    def __len__(self) -> int:
        """Trả về tổng số ảnh trong dataset."""
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Tensor, str]:
        """Đọc và transform ảnh tại vị trí ``i``.

        Args:
            i: Chỉ số ảnh.

        Returns:
            tuple[Tensor, str]: Tensor ảnh [3, H, W] và đường dẫn file.
        """
        p = self.paths[i]
        with Image.open(p) as img:
            tensor_img = self.transform(img.convert("RGB"))
        return tensor_img, str(p)


def find_category_root(raw: str | Path = "data/raw", category: str = "bottle") -> Path:
    """Tìm thư mục category mà không đọc test hoặc ground-truth.

    Args:
        raw: Đường dẫn thư mục dữ liệu raw.
        category: Tên category.

    Returns:
        Path: Đường dẫn tới thư mục gốc của category.
    """
    return _find_category_root(raw, category)
