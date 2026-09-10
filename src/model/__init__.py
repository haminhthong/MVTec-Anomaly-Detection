"""Các thành phần CV cốt lõi của detector."""

from __future__ import annotations

from .artifacts import ModelArtifact, ModelMetadata, SplitManifest, Thresholds
from .backbone_registry import BACKBONE_REGISTRY, BackboneSpec, get_backbone_spec
from .coreset import greedy_coreset, select_coreset_indices
from .memory_bank import MemoryBank


def __getattr__(name: str):
    """Nạp FeatureExtractor lazy để import nhẹ khi chỉ dùng config."""
    if name == "FeatureExtractor":
        from .feature_extractor import FeatureExtractor

        return FeatureExtractor
    raise AttributeError(name)


__all__ = [
    "FeatureExtractor",
    "select_coreset_indices",
    "greedy_coreset",
    "MemoryBank",
    "BackboneSpec",
    "BACKBONE_REGISTRY",
    "get_backbone_spec",
    "Thresholds",
    "SplitManifest",
    "ModelMetadata",
    "ModelArtifact",
]
