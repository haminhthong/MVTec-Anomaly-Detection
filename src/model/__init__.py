"""Model package for PatchCore-style Anomaly Detection."""

from __future__ import annotations

from .artifacts import (
    ModelArtifact,
    ModelMetadata,
    SplitManifest,
    ThresholdPolicy,
)
from .coreset import greedy_coreset, select_coreset_indices
from .feature_extractor import FeatureExtractor
from .memory_bank import MemoryBank
from .registry import ModelNotFoundError, ModelRegistry

__all__ = [
    "FeatureExtractor",
    "select_coreset_indices",
    "greedy_coreset",
    "MemoryBank",
    "ModelRegistry",
    "ModelNotFoundError",
    "ThresholdPolicy",
    "SplitManifest",
    "ModelMetadata",
    "ModelArtifact",
]
