"""Trích xuất multi-layer patch feature cho anomaly detection PatchCore-style.

Module lấy feature map trung gian từ backbone frozen, căn chỉnh về layer có
độ phân giải cao nhất rồi ghép thành embedding patch cục bộ.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .backbone_registry import get_backbone_spec


class FeatureExtractor(nn.Module):
    """Bộ trích xuất patch feature nhiều layer với trọng số frozen.

    Attributes:
        backbone_name: Tên kiến trúc backbone CNN.
        layers: Các layer được dùng để lấy feature.
        pretrained: Có nạp trọng số ImageNet hay không.
        weights: Tên enum hoặc chuỗi định danh trọng số.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] | None = None,
        pretrained: bool = True,
        weights: str | None = None,
    ) -> None:
        super().__init__()
        backbone = backbone.strip().lower()
        self.backbone_name: str = backbone
        self.pretrained: bool = pretrained

        # Chỉ dùng backbone đã có contract trong registry.
        spec = get_backbone_spec(backbone)
        default_layers = spec.default_layers
        default_weights = spec.weights

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
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Weights không hợp lệ: {self.weights_name}") from exc
            else:
                resolved_weights = self.weights_name
        else:
            self.weights_name = default_weights
            if "." in self.weights_name:
                try:
                    resolved_weights = models.get_weight(self.weights_name)
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Weights registry không hợp lệ: {self.weights_name}") from exc
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
        """Forward và trả về tensor embedding patch [B * H_map * W_map, C_total].

        Args:
            x: Tensor ảnh đầu vào [Batch, 3, Height, Width].

        Returns:
            torch.Tensor: Embedding patch đã phẳng [N_patches, C_total].
        """
        patches, _ = self.extract_spatial_features(x)
        return patches

    @torch.inference_mode()
    def extract_spatial_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Lấy patch feature kèm kích thước lưới không gian (H_map, W_map).

        Args:
            x: Tensor ảnh đầu vào [Batch, 3, Height, Width].

        Returns:
            tuple[torch.Tensor, tuple[int, int]]:
                - patches: Tensor embedding patch [B * H_map * W_map, C_total].
                - (h_map, w_map): Độ phân giải lưới patch sau khi căn chỉnh.
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
