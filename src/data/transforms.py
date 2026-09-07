"""Module cấu hình tiền xử lý hình ảnh (Preprocessing Pipeline).

Đảm bảo tính nhất quán 100% về kích thước, chuẩn hóa ImageNet giữa quá trình
huấn luyện (training) và suy luận (inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    """Dataclass chứa thông số chuẩn hóa và kích thước ảnh cho toàn hệ thống.

    Attributes:
        image_size: Kích thước (height, width) sau khi resize ảnh đầu vào (mặc định 224x224).
        mean: Vector trung bình RGB cho ImageNet Normalization.
        std: Vector độ lệch chuẩn RGB cho ImageNet Normalization.
    """

    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def to_dict(self) -> dict:
        """Chuyển đổi cấu hình thành dictionary để lưu trữ trong config artifact."""
        return {
            "image_size": list(self.image_size),
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PreprocessingConfig:
        """Tái tạo PreprocessingConfig từ dictionary đã lưu trong config artifact."""
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


# Giữ tên cũ nhưng trì hoãn việc khởi tạo torchvision.
DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()


class _LazyTransform:
    """Proxy trì hoãn transform mặc định cho Dataset tương thích ngược."""

    def __call__(self, image: Any) -> Any:
        return build_transform(DEFAULT_PREPROCESSING_CONFIG)(image)


TFM = _LazyTransform()
