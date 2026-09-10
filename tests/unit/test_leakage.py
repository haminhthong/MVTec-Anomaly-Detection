"""Kiểm tra chống rò rỉ: train offline không được đọc test hoặc ground_truth."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from src.config import TrainConfig
from src.training.trainer import train_patchcore


def test_training_never_reads_test_directory(tmp_path: Path) -> None:
    """Kiểm tra quan trọng: pipeline train không bao giờ đọc test/ hoặc ground_truth/."""
    cat_root = tmp_path / "data" / "raw" / "leakage_check"
    train_good = cat_root / "train" / "good"
    test_good = cat_root / "test" / "good"
    test_defect = cat_root / "test" / "broken"
    ground_truth = cat_root / "ground_truth" / "broken"

    for d in (train_good, test_good, test_defect, ground_truth):
        d.mkdir(parents=True, exist_ok=True)

    # Tạo các ảnh giả lập.
    for i in range(25):
        Image.new("RGB", (32, 32), color=(i, i, i)).save(train_good / f"{i:03d}.png")
    for i in range(5):
        Image.new("RGB", (32, 32), color=(255, i, i)).save(test_good / f"{i:03d}.png")
        Image.new("RGB", (32, 32), color=(0, 255, i)).save(test_defect / f"{i:03d}.png")
        Image.new("L", (32, 32), color=255).save(ground_truth / f"{i:03d}_mask.png")

    models_dir = tmp_path / "models"

    accessed_image_paths: list[str] = []
    original_open = Image.open

    def recording_open(fp, *args, **kwargs):
        path_str = str(fp)
        if path_str.endswith(".png"):
            accessed_image_paths.append(path_str)
        return original_open(fp, *args, **kwargs)

    cfg = TrainConfig(
        category="leakage_check",
        batch_size=4,
        min_calibration_samples=5,
        coreset_size=20,
    )

    with patch("PIL.Image.open", side_effect=recording_open):
        _ = train_patchcore(
            config=cfg,
            models_dir=models_dir,
            data_dir=tmp_path / "data" / "raw",
        )

    assert len(accessed_image_paths) > 0, "Training must have loaded images."

    # Xác nhận không có rò rỉ dữ liệu.
    for accessed in accessed_image_paths:
        normalized = accessed.replace("\\", "/")
        assert "/test/" not in normalized, f"LEAKAGE DETECTED: Training accessed test image '{accessed}'!"
        assert "/ground_truth/" not in normalized, f"LEAKAGE DETECTED: Training accessed ground_truth mask '{accessed}'!"
        assert "/train/good/" in normalized, f"Training accessed unexpected image: '{accessed}'"
