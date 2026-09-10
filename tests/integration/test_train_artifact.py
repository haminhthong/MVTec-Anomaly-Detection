"""Kiểm thử tích hợp việc train offline và tạo artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.config import TrainConfig
from src.model.artifacts import ModelArtifact, SplitManifest
from src.training.trainer import train_patchcore


def test_train_artifact_generation(tmp_path: Path) -> None:
    """Chạy train_patchcore trên category giả lập và kiểm tra artifact."""
    raw_dir = tmp_path / "data" / "raw"
    category = "test_item"
    train_good = raw_dir / category / "train" / "good"
    test_good = raw_dir / category / "test" / "good"
    test_defect = raw_dir / category / "test" / "flaw"
    gt_dir = raw_dir / category / "ground_truth" / "flaw"

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

    cat_dir = models_dir / category
    assert (cat_dir / "metadata.json").exists()
    assert (cat_dir / "memory_bank.npy").exists()
    report_dir = tmp_path / "reports" / category
    assert (report_dir / "training_split.json").exists()

    # Kiểm tra split manifest.
    split = SplitManifest.from_dict(
        json.loads((report_dir / "training_split.json").read_text(encoding="utf-8"))
    )
    assert split.memory_count + split.dev_count + split.calibration_count == 25
    assert len(split.memory_files) == split.memory_count
    assert len(split.dev_files) == split.dev_count
    assert len(split.calibration_files) == split.calibration_count

    # Kiểm tra memory bank.
    mem = np.load(cat_dir / "memory_bank.npy")
    assert mem.ndim == 2
    assert mem.shape[1] == 384
    assert len(mem) <= 20

    # Kiểm tra artifact sau khi nạp lại.
    loaded_art = ModelArtifact.load(cat_dir)
    assert loaded_art.metadata.category == category
    assert loaded_art.thresholds.image_threshold > 0
