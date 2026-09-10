"""Kiểm thử DatasetManifest và validate_mvtec_category."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data.validation import (
    DatasetManifest,
    DatasetValidationError,
    validate_mvtec_category,
)


def create_dummy_mvtec_structure(
    root: Path,
    category: str = "bottle",
    include_masks: bool = True,
    train_count: int = 25,
    defect_count: int = 5,
) -> Path:
    """Tạo thư mục category MVTec AD tối thiểu nhưng hợp lệ."""
    cat_dir = root / category
    train_good = cat_dir / "train" / "good"
    train_good.mkdir(parents=True, exist_ok=True)
    for i in range(train_count):
        Image.new("RGB", (32, 32), color=(i, i, i)).save(train_good / f"{i:03d}.png")

    test_good = cat_dir / "test" / "good"
    test_good.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        Image.new("RGB", (32, 32), color=(255, i, i)).save(test_good / f"{i:03d}.png")

    test_defect = cat_dir / "test" / "broken"
    test_defect.mkdir(parents=True, exist_ok=True)
    for i in range(defect_count):
        Image.new("RGB", (32, 32), color=(0, 255, i)).save(test_defect / f"{i:03d}.png")

    if include_masks:
        gt_dir = cat_dir / "ground_truth" / "broken"
        gt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(defect_count):
            Image.new("L", (32, 32), color=255).save(gt_dir / f"{i:03d}_mask.png")

    return cat_dir


def test_validate_mvtec_category_success(tmp_path: Path) -> None:
    """Kiểm tra validation thành công và tạo DatasetManifest."""
    create_dummy_mvtec_structure(tmp_path, category="bottle", include_masks=True)
    manifest = validate_mvtec_category(data_dir=tmp_path, category="bottle")

    assert isinstance(manifest, DatasetManifest)
    assert manifest.category == "bottle"
    assert manifest.total_train == 25
    assert manifest.total_test_good == 5
    assert manifest.total_test_defect == 5
    assert manifest.total_test == 10
    assert "broken" in manifest.defect_types
    assert len(manifest.masks) == 5


def test_validate_mvtec_missing_mask_raises_error(tmp_path: Path) -> None:
    """Ảnh lỗi thiếu ground-truth mask bắt buộc phải báo lỗi."""
    create_dummy_mvtec_structure(tmp_path, category="cable", include_masks=False)

    with pytest.raises(DatasetValidationError, match="ground_truth"):
        validate_mvtec_category(data_dir=tmp_path, category="cable")


def test_validate_mvtec_nonexistent_category(tmp_path: Path) -> None:
    """Category không tồn tại phải phát sinh FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Không tìm thấy category"):
        validate_mvtec_category(data_dir=tmp_path, category="nonexistent_cat")


def test_validate_mvtec_empty_train_raises_error(tmp_path: Path) -> None:
    """Thư mục train/good rỗng phải phát sinh DatasetValidationError."""
    cat_dir = tmp_path / "bottle"
    (cat_dir / "train" / "good").mkdir(parents=True, exist_ok=True)
    (cat_dir / "test" / "good").mkdir(parents=True, exist_ok=True)

    with pytest.raises(DatasetValidationError, match="Không có ảnh hợp lệ trong train/good"):
        validate_mvtec_category(data_dir=tmp_path, category="bottle")
