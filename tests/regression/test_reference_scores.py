"""Regression tests kiểm tra tính tất định và ổn định của điểm số suy luận."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from src.inference.detector import AnomalyDetector


def test_scoring_determinism_and_reproducibility(tmp_path: Path) -> None:
    """Kiểm tra cùng một mô hình và ảnh đầu vào luôn tạo ra điểm số trùng khớp 100%."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    memory = rng.normal(size=(50, 384)).astype(np.float32)
    np.save(model_dir / "memory_bank.npy", memory)

    config_data = {
        "model": {
            "category": "regr_test",
            "model_version": "1.0.0",
            "backbone": "resnet18",
            "pretrained": True,
            "feature_layers": ["layer2", "layer3"],
        },
        "category": "regr_test",
        "smooth_sigma": 1.0,
        "thresholds": {"image_threshold": 3.0, "pixel_threshold": 2.8},
        "preprocessing": {
            "image_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(config_data), encoding="utf-8")

    det1 = AnomalyDetector(model_dir=str(model_dir))
    det2 = AnomalyDetector(model_dir=str(model_dir))

    img = Image.new("RGB", (224, 224), color=(50, 100, 150))

    score1, heat1 = det1.score(img)
    score2, heat2 = det2.score(img)

    assert abs(score1 - score2) < 1e-6
    assert np.allclose(heat1, heat2, atol=1e-6)
