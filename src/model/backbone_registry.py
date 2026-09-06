"""Backbone architecture specification and registry for PatchCore feature extraction.

Enforces strict backbone and layer compatibility:
- Avoids arbitrary layer mismatches across different vision architectures
- Pairs architectures with their verified ImageNet weights enum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackboneSpec:
    """Specification contract for supported vision backbones."""

    name: str
    default_layers: tuple[str, ...]
    weights: str
    feature_dim: int | None = None


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        name="resnet18",
        default_layers=("layer2", "layer3"),
        weights="ResNet18_Weights.IMAGENET1K_V1",
        feature_dim=384,  # 128 + 256
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        default_layers=("layer2", "layer3"),
        weights="ResNet50_Weights.IMAGENET1K_V2",
        feature_dim=1536,  # 512 + 1024
    ),
    "wide_resnet50_2": BackboneSpec(
        name="wide_resnet50_2",
        default_layers=("layer2", "layer3"),
        weights="Wide_ResNet50_2_Weights.IMAGENET1K_V2",
        feature_dim=1536,
    ),
}


def get_backbone_spec(name: str) -> BackboneSpec:
    """Retrieve verified BackboneSpec for an architecture.

    Args:
        name: Name of architecture (e.g. 'resnet18', 'resnet50').

    Returns:
        BackboneSpec: Registered specification.

    Raises:
        ValueError: If architecture is not officially supported.
    """
    key = name.lower().strip()
    if key not in BACKBONE_REGISTRY:
        supported = list(BACKBONE_REGISTRY.keys())
        raise ValueError(
            f"Unsupported backbone '{name}'. Officially supported architectures: {supported}."
        )
    return BACKBONE_REGISTRY[key]


def list_supported_backbones() -> list[str]:
    """List all officially supported backbone names."""
    return sorted(BACKBONE_REGISTRY.keys())
