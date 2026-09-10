"""Kiểm thử FeatureExtractor với backbone cấu hình được và kích thước động."""

from __future__ import annotations

import pytest
import torch

from src.model.feature_extractor import FeatureExtractor


def test_feature_extractor_default_shapes() -> None:
    """Kiểm tra kích thước feature mặc định của ResNet18 layer2 + layer3."""
    extractor = FeatureExtractor(backbone="resnet18", layers=("layer2", "layer3"), pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)

    patches, (h, w) = extractor.extract_spatial_features(dummy)
    assert (h, w) == (28, 28)
    assert patches.shape == (2 * 28 * 28, 384)

    forward_patches = extractor(dummy)
    assert forward_patches.shape == (2 * 28 * 28, 384)


def test_feature_extractor_custom_resolution() -> None:
    """Kiểm tra FeatureExtractor với ảnh không phải 224x224, ví dụ 256x256."""
    extractor = FeatureExtractor(backbone="resnet18", layers=("layer2", "layer3"), pretrained=False)
    dummy = torch.randn(1, 3, 256, 256)

    patches, (h, w) = extractor.extract_spatial_features(dummy)
    # 256 / 8 = 32 cho layer2.
    assert (h, w) == (32, 32)
    assert patches.shape == (1 * 32 * 32, 384)


def test_feature_extractor_weights_are_frozen() -> None:
    """Kiểm tra toàn bộ tham số đều có requires_grad=False."""
    extractor = FeatureExtractor(backbone="resnet18", pretrained=False)
    for p in extractor.parameters():
        assert p.requires_grad is False
