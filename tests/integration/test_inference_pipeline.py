"""Kiểm thử tích hợp cho luồng suy luận."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.inference.detector import AnomalyDetector


def test_inference_with_artifact(tmp_path: Path) -> None:
    """Đảm bảo detector đọc artifact và trả schema V1 ổn định."""
    model_dir = tmp_path / "models" / "test_box"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Memory bank giả lập [20, 384].
    memory = np.random.randn(20, 384).astype(np.float32)
    np.save(model_dir / "memory_bank.npy", memory)

    # Artifact runtime hiện tại chỉ chứa metadata và thresholds.
    config_data = {
        "model": {
            "model_version": "1.0.0",
            "category": "test_box",
            "backbone": "resnet18",
            "pretrained": True,
            "feature_layers": ["layer2", "layer3"],
        },
        "category": "test_box",
        "model_version": "1.0.0",
        "smooth_sigma": 1.0,
        "thresholds": {
            "image_threshold": 3.5,
            "pixel_threshold": 3.0,
        },
        "preprocessing": {
            "image_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(config_data), encoding="utf-8")

    det = AnomalyDetector(model_dir=str(model_dir))
    assert det.image_threshold == 3.5
    assert det.pixel_threshold == 3.0
    assert det.memory_bank.size == 20

    img = Image.new("RGB", (200, 200), color="white")
    s, heat = det.score(img)
    assert isinstance(s, float)
    assert heat.shape == (28, 28)

    res = det.inspect(img, include_overlay=True)
    assert "inspection_id" in res
    assert "decision" in res
    assert "image_threshold" in res
    assert "anomalous_area_ratio" in res
    assert res["decision"] in {"PASS_CANDIDATE", "REVIEW_REQUIRED", "RECAPTURE_REQUIRED"}
    assert res["overlay_b64"].startswith("data:image/png;base64,")
