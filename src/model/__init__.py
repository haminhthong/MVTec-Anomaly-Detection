"""Model package for PatchCore-style Anomaly Detection."""

from __future__ import annotations

from .artifact_resolver import ModelNotFoundError, resolve_artifact_dir
from .artifacts import (
    ModelArtifact,
    ModelMetadata,
    SplitManifest,
    ThresholdPolicy,
)
from .backbone_registry import BACKBONE_REGISTRY, BackboneSpec, get_backbone_spec
from .coreset import greedy_coreset, select_coreset_indices
from .feature_extractor import FeatureExtractor
from .memory_bank import MemoryBank
from .registry import ModelRegistry

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
]
