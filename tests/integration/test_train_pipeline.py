"""Kiểm thử tích hợp pipeline train."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.config import TrainConfig
from src.training.trainer import train_patchcore


def test_train_pipeline_end_to_end(tmp_path: Path) -> None:
    """Chạy train_patchcore trên dataset giả lập và kiểm tra artifact theo category."""
    raw_dir = tmp_path / "data" / "raw"
    category = "dummy_cat"
    train_good = raw_dir / category / "train" / "good"
    test_good = raw_dir / category / "test" / "good"
    test_defect = raw_dir / category / "test" / "bad"
    gt_dir = raw_dir / category / "ground_truth" / "bad"

    for d in (train_good, test_good, test_defect, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(25):
        Image.new("RGB", (32, 32), color=(i * 10, i * 5, 200)).save(train_good / f"{i:03d}.png")
    for i in range(5):
        Image.new("RGB", (32, 32), color=(255, i, 0)).save(test_good / f"{i:03d}.png")
        Image.new("RGB", (32, 32), color=(0, 255, i)).save(test_defect / f"{i:03d}.png")
        Image.new("L", (32, 32), color=255).save(gt_dir / f"{i:03d}_mask.png")

    models_dir = tmp_path / "models"
    cfg = TrainConfig(
        category=category,
        batch_size=4,
        min_calibration_samples=5,
        image_quantile=0.98,
        pixel_quantile=0.98,
        coreset_size=20,
    )

    artifact = train_patchcore(config=cfg, models_dir=models_dir, data_dir=raw_dir)
    assert artifact.metadata.category == category
    assert artifact.thresholds.image_threshold > 0

    cat_dir = models_dir / category
    assert (cat_dir / "memory_bank.npy").exists()
    assert (cat_dir / "metadata.json").exists()
    assert (tmp_path / "reports" / category / "training_split.json").exists()

    loaded_mem = np.load(cat_dir / "memory_bank.npy")
    assert loaded_mem.ndim == 2
    assert loaded_mem.shape[1] == 384
