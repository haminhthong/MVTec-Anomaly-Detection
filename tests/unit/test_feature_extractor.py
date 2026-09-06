"""Unit tests for FeatureExtractor with configurable backbone and dynamic shapes."""

from __future__ import annotations

import pytest
import torch

from src.model.feature_extractor import FeatureExtractor


def test_feature_extractor_default_shapes() -> None:
    """Test default ResNet18 layer2 + layer3 extraction shape."""
    extractor = FeatureExtractor(backbone="resnet18", layers=("layer2", "layer3"), pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)

    patches, (h, w) = extractor.extract_spatial_features(dummy)
    assert (h, w) == (28, 28)
    assert patches.shape == (2 * 28 * 28, 384)

    forward_patches = extractor(dummy)
    assert forward_patches.shape == (2 * 28 * 28, 384)


def test_feature_extractor_custom_resolution() -> None:
    """Test FeatureExtractor with dynamic non-224 input resolution (e.g. 256x256)."""
    extractor = FeatureExtractor(backbone="resnet18", layers=("layer2", "layer3"), pretrained=False)
    dummy = torch.randn(1, 3, 256, 256)

    patches, (h, w) = extractor.extract_spatial_features(dummy)
    # 256 / 8 = 32 for layer2
    assert (h, w) == (32, 32)
    assert patches.shape == (1 * 32 * 32, 384)


def test_feature_extractor_weights_are_frozen() -> None:
    """Test all parameters have requires_grad=False."""
    extractor = FeatureExtractor(backbone="resnet18", pretrained=False)
    for p in extractor.parameters():
        assert p.requires_grad is False
