"""Integration tests verifying AnomalyDetector loads artifact and scores samples correctly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.config import TrainConfig
from src.inference.detector import AnomalyDetector
from src.training.trainer import train_patchcore


def test_detector_from_saved_artifact(tmp_path: Path) -> None:
    """Train dummy artifact, load detector, and inspect an image."""
    raw_dir = tmp_path / "data" / "raw"
    category = "widget"
    train_good = raw_dir / category / "train" / "good"
    test_good = raw_dir / category / "test" / "good"
    test_defect = raw_dir / category / "test" / "crack"
    gt_dir = raw_dir / category / "ground_truth" / "crack"

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
        coreset_fraction=0.1,
        min_coreset_size=5,
        max_coreset_size=20,
    )
    _ = train_patchcore(config=cfg, models_dir=models_dir, data_dir=raw_dir)

    detector = AnomalyDetector(model_dir=models_dir, category=category)
    assert detector.category == category
    assert detector.threshold > 0

    test_img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    result = detector.inspect(test_img, include_overlay=True)

    assert "inspection_id" in result
    assert result["category"] == category
    assert result["decision"] in {"AUTO_PASS", "HUMAN_REVIEW", "RECAPTURE_REQUIRED"}
    assert result["scores"]["anomaly_score"] >= 0
    assert result["overlay_b64"] is not None
    assert result["overlay_b64"].startswith("data:image/png;base64,")
