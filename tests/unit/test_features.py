"""Unit tests kiểm tra FeatureExtractor trích xuất đặc trưng đa tầng."""

from __future__ import annotations

import torch
from src.model.feature_extractor import FeatureExtractor


def test_feature_extractor_dimensions() -> None:
    """Kiểm tra FeatureExtractor hợp nhất Layer 2 (128D) và Layer 3 (256D upsampled)."""
    extractor = FeatureExtractor(backbone="resnet18", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)

    patches = extractor(dummy_input)
    # 2 ảnh * 28*28 (784 patches) = 1568 patches; 128 + 256 = 384 dimensions
    assert patches.shape == (2 * 784, 384)

    spatial_patches, (h, w) = extractor.extract_spatial_features(dummy_input)
    assert (h, w) == (28, 28)
    assert spatial_patches.shape == (2 * 784, 384)
