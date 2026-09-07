"""Multi-layer patch feature extractor for PatchCore-style anomaly detection.

Extracts intermediate feature maps from frozen pretrained backbones (e.g. ResNet18, ResNet50),
spatially aligns them to the highest-resolution target layer via bilinear interpolation,
and concatenates them into dense local patch embeddings capturing textures and semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


from .backbone_registry import BACKBONE_REGISTRY, get_backbone_spec


class FeatureExtractor(nn.Module):
    """Configurable multi-layer patch feature extractor with frozen weights.

    Attributes:
        backbone_name: Name of the CNN backbone architecture.
        layers: Sequence of layer names to extract features from.
        pretrained: Whether ImageNet weights are loaded.
        weights: Weights enum identifier or string.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] | None = None,
        pretrained: bool = True,
        weights: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name: str = backbone
        self.pretrained: bool = pretrained

        # Resolve thông số backbone nếu có trong registry.
        try:
            spec = get_backbone_spec(backbone)
            default_layers = spec.default_layers
            default_weights = spec.weights
        except ValueError:
            spec = None
            default_layers = ("layer2", "layer3")
            default_weights = "DEFAULT"

        self.layers: tuple[str, ...] = tuple(layers) if layers is not None else default_layers

        # Khởi tạo backbone.
        if not hasattr(models, backbone):
            raise ValueError(f"Backbone '{backbone}' is not supported by torchvision.models.")

        if not pretrained:
            resolved_weights = None
            self.weights_name: str | None = None
        elif weights is not None:
            self.weights_name = str(weights)
            if "." in self.weights_name:
                try:
                    resolved_weights = models.get_weight(self.weights_name)
                except Exception:
                    resolved_weights = self.weights_name
            else:
                resolved_weights = self.weights_name
        else:
            self.weights_name = default_weights
            if "." in self.weights_name:
                try:
                    resolved_weights = models.get_weight(self.weights_name)
                except Exception:
                    resolved_weights = self.weights_name
            else:
                resolved_weights = self.weights_name

        self.model: nn.Module = getattr(models, backbone)(weights=resolved_weights)

        # Đóng băng toàn bộ tham số.
        self.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Đăng ký forward hook trên các layer mục tiêu.
        self._feature_maps: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []
        named_modules = dict(self.model.named_modules())

        for layer_name in self.layers:
            if layer_name not in named_modules:
                raise ValueError(
                    f"Layer '{layer_name}' not found in backbone '{backbone}'. "
                    f"Available modules: {list(named_modules.keys())[:15]}..."
                )
            hook = named_modules[layer_name].register_forward_hook(self._save_feature(layer_name))
            self._hooks.append(hook)

    def _save_feature(self, layer_name: str):
        def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._feature_maps[layer_name] = output
        return hook_fn

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning patch embeddings tensor [B * H_map * W_map, C_total].

        Args:
            x: Input image tensor [Batch, 3, Height, Width].

        Returns:
            torch.Tensor: Flattened patch embeddings [N_patches, C_total].
        """
        patches, _ = self.extract_spatial_features(x)
        return patches

    @torch.inference_mode()
    def extract_spatial_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Extract patch features along with spatial grid dimensions (H_map, W_map).

        Args:
            x: Input image tensor [Batch, 3, Height, Width].

        Returns:
            tuple[torch.Tensor, tuple[int, int]]:
                - patches: Tensor of patch embeddings [B * H_map * W_map, C_total].
                - (h_map, w_map): Spatial resolution of aligned patch grid.
        """
        self._feature_maps.clear()
        _ = self.model(x)

        extracted_maps = [self._feature_maps[layer_name] for layer_name in self.layers]
        target_shape = extracted_maps[0].shape[2:]  # (H, W) of the first feature layer

        # Nội suy song tuyến để căn chỉnh kích thước không gian.
        aligned_maps: list[torch.Tensor] = []
        for feat in extracted_maps:
            if feat.shape[2:] != target_shape:
                feat = F.interpolate(
                    feat, size=target_shape, mode="bilinear", align_corners=False
                )
            aligned_maps.append(feat)

        combined = torch.cat(aligned_maps, dim=1)
        b, c, h, w = combined.shape
        # Đổi trục [B, C, H, W] -> [B, H, W, C] -> [-1, C].
        patches = combined.permute(0, 2, 3, 1).reshape(-1, c)
        return patches, (h, w)
