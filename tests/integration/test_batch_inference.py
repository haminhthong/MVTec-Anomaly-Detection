"""Integration tests verifying batch inference parity with single-image inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.config import TrainConfig
from src.inference.detector import AnomalyDetector
from src.training.trainer import train_patchcore


def test_batch_inference_parity(tmp_path: Path) -> None:
    """Ensure inspect_batch returns results matching single-image inspect() within numerical tolerance."""
    raw_dir = tmp_path / "data" / "raw"
    category = "batch_test"
    train_good = raw_dir / category / "train" / "good"
    test_good = raw_dir / category / "test" / "good"
    test_defect = raw_dir / category / "test" / "scratch"
    gt_dir = raw_dir / category / "ground_truth" / "scratch"

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

    # Prepare 3 distinct test images
    img1 = Image.new("RGB", (48, 48), color=(10, 20, 30))
    img2 = Image.new("RGB", (48, 48), color=(200, 100, 50))
    img3 = Image.new("RGB", (48, 48), color=(0, 250, 10))
    test_images = [img1, img2, img3]

    # Single inspections
    single_results = [detector.inspect(im, include_overlay=False) for im in test_images]

    # Batch inspection
    batch_results = detector.inspect_batch(test_images, include_overlay=False)

    assert len(batch_results) == len(test_images)

    for single_r, batch_r in zip(single_results, batch_results, strict=True):
        assert single_r["decision"] == batch_r["decision"]
        assert single_r["severity"] == batch_r["severity"]
        # Score parity within small floating point difference
        assert np.isclose(
            single_r["scores"]["anomaly_score"],
            batch_r["scores"]["anomaly_score"],
            atol=1e-3,
        )
