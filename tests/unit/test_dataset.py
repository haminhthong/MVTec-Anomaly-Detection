"""Unit tests for ImageFolderDataset."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from src.data.dataset import ImageFolderDataset


def test_image_folder_dataset_loading(tmp_path: Path) -> None:
    """Test ImageFolderDataset loads images and outputs expected tensor shape."""
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (64, 64), color="red").save(img1)
    Image.new("RGB", (64, 64), color="blue").save(img2)

    dataset = ImageFolderDataset([img1, img2])
    assert len(dataset) == 2

    tensor1, path1 = dataset[0]
    assert isinstance(tensor1, torch.Tensor)
    assert tensor1.shape == (3, 224, 224)
    assert str(img1) == path1
