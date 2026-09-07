"""Các thành phần model; backbone torch được lazy-load khi cần."""

from __future__ import annotations

from .artifact_resolver import ModelNotFoundError, resolve_artifact_dir
from .artifacts import (
    ModelArtifact,
    ModelMetadata,
    SplitManifest,
    ThresholdPolicy,
    verify_artifact_integrity,
    write_integrity_manifest,
)
from .backbone_registry import BACKBONE_REGISTRY, BackboneSpec, get_backbone_spec
from .coreset import greedy_coreset, select_coreset_indices
from .memory_bank import MemoryBank


def __getattr__(name: str):
    """Nạp phần phụ thuộc torch chỉ khi caller dùng runtime."""
    if name == "FeatureExtractor":
        from .feature_extractor import FeatureExtractor

        return FeatureExtractor
    if name == "ModelRegistry":
        from .registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(name)

__all__ = [
    "FeatureExtractor",
    "select_coreset_indices",
    "greedy_coreset",
    "MemoryBank",
    "ModelRegistry",
    "ModelNotFoundError",
    "resolve_artifact_dir",
    "BackboneSpec",
    "BACKBONE_REGISTRY",
    "get_backbone_spec",
    "ThresholdPolicy",
    "SplitManifest",
    "ModelMetadata",
    "ModelArtifact",
    "verify_artifact_integrity",
    "write_integrity_manifest",
]
