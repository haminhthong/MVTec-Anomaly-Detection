"""Cấu hình tập trung cho pipeline PatchCore-style normal-only."""

from __future__ import annotations

from dataclasses import dataclass, field

from .data.transforms import PreprocessingConfig
from .model.backbone_registry import get_backbone_spec
from .path_safety import ensure_safe_segment

# Các giá trị mặc định của hệ thống
DEFAULT_CATEGORY: str = "bottle"
DEFAULT_SEED: int = 42
DEFAULT_BACKBONE: str = "resnet18"
DEFAULT_FEATURE_LAYERS: tuple[str, ...] = ("layer2", "layer3")


@dataclass(frozen=True)
class TrainConfig:
    """Dataclass chứa toàn bộ tham số cấu hình cho pipeline PatchCore-style.

    Attributes:
        category: Tên danh mục sản phẩm cần phát hiện lỗi (mặc định: 'bottle').
        seed: Seed cho các bộ sinh số ngẫu nhiên nhằm đảm bảo tính tái lập.
        backbone: Tên kiến trúc CNN trích xuất đặc trưng (ví dụ: 'resnet18', 'resnet50').
        feature_layers: Danh sách tên các tầng trích xuất đặc trưng trung gian.
        pretrained: Sử dụng trọng số pretrained ImageNet cho backbone.
        batch_size: Kích thước batch khi trích xuất đặc trưng hình ảnh.
        dev_fraction: Tỷ lệ normal dành cho phát triển và synthetic stress.
        calibration_fraction: Tỷ lệ normal held-out dùng để calibration.
        image_quantile: Quantile normal-only dùng cho image threshold.
        pixel_quantile: Phân vị pixel heatmap normal dùng làm Pixel Threshold (mặc định: 0.99).
        min_calibration_samples: Số lượng ảnh calibration tối thiểu yêu cầu (mặc định: 20).
        coreset_size: Số patch cụ thể giữ lại trong memory bank.
        smooth_sigma: Độ lệch chuẩn Sigma cho bộ lọc Gaussian Smoothing làm mịn anomaly map.
        preprocessing: Cấu hình tiền xử lý ảnh (PreprocessingConfig).
    """

    category: str = DEFAULT_CATEGORY
    seed: int = DEFAULT_SEED
    backbone: str = DEFAULT_BACKBONE
    feature_layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS
    pretrained: bool = True
    batch_size: int = 8
    dev_fraction: float = 0.15
    calibration_fraction: float = 0.15
    image_quantile: float = 0.99
    pixel_quantile: float = 0.99
    min_calibration_samples: int = 20
    coreset_size: int | None = None
    smooth_sigma: float = 1.0
    weights: str | None = None
    scoring_percentile: float = 99.0
    capture_contract: dict[str, object] = field(default_factory=dict)

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của tất cả các tham số cấu hình.

        Raises:
            ValueError: Nếu bất kỳ tham số nào nằm ngoài dải hợp lệ.
        """
        ensure_safe_segment(self.category, "category")
        if not self.backbone.strip():
            raise ValueError("Tên backbone không được để trống.")
        get_backbone_spec(self.backbone)
        if not self.feature_layers:
            raise ValueError("Danh sách feature_layers không được rỗng.")
        if self.batch_size <= 0:
            raise ValueError("Kích thước batch (batch_size) phải lớn hơn 0.")
        if not 0 <= self.dev_fraction < 0.5:
            raise ValueError("dev_fraction phải thuộc khoảng [0, 0.5).")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction phải thuộc khoảng (0, 0.5).")
        if self.dev_fraction + self.calibration_fraction >= 1.0:
            raise ValueError("dev_fraction + calibration_fraction phải nhỏ hơn 1.")
        if self.min_calibration_samples < 5:
            raise ValueError(
                "min_calibration_samples phải >= 5 để đảm bảo ước lượng quantile có ý nghĩa."
            )
        if not 0.5 <= self.image_quantile < 1.0:
            raise ValueError("image_quantile phải thuộc khoảng [0.5, 1.0).")
        if not 0.5 <= self.pixel_quantile < 1.0:
            raise ValueError("pixel_quantile phải thuộc khoảng [0.5, 1.0).")
        if self.coreset_size is not None and self.coreset_size <= 0:
            raise ValueError("coreset_size phải là số nguyên dương.")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma không được âm.")
        if not 50.0 <= self.scoring_percentile <= 100.0:
            raise ValueError("scoring_percentile phải thuộc khoảng [50.0, 100.0].")

    def resolved_coreset_size(self, full_memory_size: int) -> int:
        """Tính K rõ ràng cho memory bank."""
        if full_memory_size <= 0:
            raise ValueError("full_memory_size phải lớn hơn 0.")
        if self.coreset_size is not None:
            return min(self.coreset_size, full_memory_size)
        return min(1000, full_memory_size)
