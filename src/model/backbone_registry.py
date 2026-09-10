"""Danh mục backbone được hỗ trợ để trích xuất feature PatchCore-style.

Danh mục này mô tả kiến trúc, layer và bộ trọng số ImageNet tương ứng; không
cho phép chọn backbone ngoài các contract đã kiểm tra.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    """Mô tả contract của một backbone được hỗ trợ."""

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
    """Lấy BackboneSpec đã được xác thực cho một kiến trúc.

    Args:
        name: Tên kiến trúc, ví dụ ``resnet18`` hoặc ``resnet50``.

    Returns:
        BackboneSpec: Contract tương ứng trong danh mục backbone.

    Raises:
        ValueError: Nếu kiến trúc chưa được hỗ trợ chính thức.
    """
    key = name.lower().strip()
    if key not in BACKBONE_REGISTRY:
        supported = list(BACKBONE_REGISTRY.keys())
        raise ValueError(
            f"Backbone '{name}' chưa được hỗ trợ. Các kiến trúc hợp lệ: {supported}."
        )
    return BACKBONE_REGISTRY[key]


def list_supported_backbones() -> list[str]:
    """Liệt kê tên các backbone được hỗ trợ chính thức."""
    return sorted(BACKBONE_REGISTRY.keys())
