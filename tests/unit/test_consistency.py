"""Preprocessing consistency tests between training configuration, saved artifact, and runtime detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.transforms import PreprocessingConfig
from src.inference.detector import AnomalyDetector
from src.model.artifacts import ModelArtifact, ModelMetadata, ThresholdPolicy


def test_train_inference_preprocessing_consistency(tmp_path: Path) -> None:
    """Ensure training transform configuration matches artifact and detector runtime transforms 100%."""
    custom_prep = PreprocessingConfig(
        image_size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
    )

    models_dir = tmp_path / "models"
    cat_dir = models_dir / "custom_cat"
    cat_dir.mkdir(parents=True, exist_ok=True)

    artifact = ModelArtifact(
        metadata=ModelMetadata(model_version="1.0.0", category="custom_cat"),
        threshold_policy=ThresholdPolicy(review_threshold=2.0, fail_threshold=2.5, pixel_threshold=2.5),
        preprocessing=custom_prep,
        coreset_info={"size": 10},
    )
    artifact.save(cat_dir)
    np.save(cat_dir / "memory_bank.npy", np.zeros((10, 384), dtype=np.float32))

    detector = AnomalyDetector(model_dir=cat_dir, category="custom_cat")

    # Verify identical PreprocessingConfig
    assert detector.preprocessing_config.image_size == custom_prep.image_size
    assert detector.preprocessing_config.mean == custom_prep.mean
    assert detector.preprocessing_config.std == custom_prep.std
    assert detector.preprocessing_config == custom_prep
