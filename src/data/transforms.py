"""Cấu hình tiền xử lý ảnh dùng chung cho train và inference.

Mọi stage đều dùng cùng kích thước và chuẩn hóa ImageNet để feature không bị
lệch giữa lúc xây memory bank và lúc chấm ảnh mới.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    """Thông số chuẩn hóa và kích thước ảnh của pipeline.

    Attributes:
        image_size: Kích thước (height, width) sau khi resize ảnh đầu vào.
        mean: Vector trung bình RGB của ImageNet.
        std: Vector độ lệch chuẩn RGB của ImageNet.
    """

    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def to_dict(self) -> dict:
        """Chuyển cấu hình thành dictionary để lưu trong metadata model."""
        return {
            "image_size": list(self.image_size),
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PreprocessingConfig:
        """Khôi phục cấu hình từ metadata model."""
        return cls(
            image_size=tuple(data.get("image_size", (224, 224))),
            mean=tuple(data.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(data.get("std", (0.229, 0.224, 0.225))),
        )


def build_transform(config: PreprocessingConfig | None = None) -> Any:
    """Tạo transform; chỉ import torchvision khi thật sự chạy feature pipeline."""
    from torchvision import transforms

    cfg = config or PreprocessingConfig()
    return transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
        ]
    )


# Khởi tạo mặc định mà không import torchvision cho tới khi thật sự transform ảnh.
DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()


class _LazyTransform:
    """Proxy trì hoãn transform mặc định cho Dataset."""

    def __call__(self, image: Any) -> Any:
        return build_transform(DEFAULT_PREPROCESSING_CONFIG)(image)


TFM = _LazyTransform()
