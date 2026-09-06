"""Dataset management and loader for MVTec AD images.

Provides PyTorch Dataset reading images from paths and applying transforms.
Integrates with DatasetManifest and validate_mvtec_category.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .transforms import TFM, build_transform
from .validation import DatasetManifest, validate_mvtec_category


class ImageFolderDataset(Dataset):
    """PyTorch Dataset reading images from a list of paths.

    Args:
        paths: Sequence of image file paths (Path or str).
        transform: Image transform function (PIL.Image -> Tensor).
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
        """Total number of images in dataset."""
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Tensor, str]:
        """Read and transform image at index i.

        Args:
            i: Image index.

        Returns:
            tuple[Tensor, str]: Transformed image tensor [3, H, W] and file path string.
        """
        p = self.paths[i]
        with Image.open(p) as img:
            tensor_img = self.transform(img.convert("RGB"))
        return tensor_img, str(p)


def find_category_root(raw: str | Path = "data/raw", category: str = "bottle") -> Path:
    """Find and validate category directory under data directory.

    Args:
        raw: Path to raw data directory.
        category: Name of category.

    Returns:
        Path: Path to category root directory.
    """
    manifest = validate_mvtec_category(data_dir=raw, category=category)
    return manifest.root_path
