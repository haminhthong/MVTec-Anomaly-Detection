"""Unit tests kiểm tra tính hợp lệ và cơ chế validation của cấu hình hệ thống."""

from __future__ import annotations

import pytest
from src.config import TrainConfig
from src.data.transforms import PreprocessingConfig


def test_preprocessing_config_defaults() -> None:
    """Kiểm tra giá trị mặc định của PreprocessingConfig."""
    cfg = PreprocessingConfig()
    assert cfg.image_size == (224, 224)
    assert len(cfg.mean) == 3
    assert len(cfg.std) == 3

    d = cfg.to_dict()
    assert d["image_size"] == [224, 224]

    reconstructed = PreprocessingConfig.from_dict(d)
    assert reconstructed == cfg


def test_train_config_validations() -> None:
    """Kiểm tra các quy tắc bắt lỗi trong TrainConfig."""
    # Category rỗng
    with pytest.raises(ValueError, match="category"):
        TrainConfig(category="").validate()

    # Batch size <= 0
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0).validate()

    # Calibration fraction >= 0.5
    with pytest.raises(ValueError, match="calibration_fraction"):
        TrainConfig(calibration_fraction=0.6).validate()

    # Min calibration samples < 5
    with pytest.raises(ValueError, match="min_calibration_samples"):
        TrainConfig(min_calibration_samples=2).validate()

    # Image quantile ngoài dải
    with pytest.raises(ValueError, match="image_quantile"):
        TrainConfig(image_quantile=0.2).validate()

    # Pixel quantile ngoài dải
    with pytest.raises(ValueError, match="pixel_quantile"):
        TrainConfig(pixel_quantile=0.2).validate()

    # Smooth sigma âm
    with pytest.raises(ValueError, match="smooth_sigma"):
        TrainConfig(smooth_sigma=-1.0).validate()
