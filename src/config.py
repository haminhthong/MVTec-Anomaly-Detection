"""Cấu hình và kiểm soát tham số cho hệ thống kiểm tra lỗi ngoại quan MVTec AD (PatchCore-style).

Module này cung cấp dataclass TrainConfig và PreprocessingConfig để quản lý toàn diện
các tham số tiền xử lý, kiến trúc backbone, huấn luyện, hiệu chỉnh ngưỡng vận hành kép
(Dual-threshold calibration: P95 review, P99 fail), phân vị pixel và kích thước coreset memory bank.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from collections.abc import Sequence

from .data.transforms import PreprocessingConfig

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
        calibration_fraction: Tỷ lệ ảnh normal held-out dùng để căn chỉnh threshold.
        review_quantile: Phân vị normal score dùng làm ngưỡng cảnh báo REVIEW (mặc định: 0.95).
        threshold_quantile: Phân vị normal score dùng làm ngưỡng lỗi FAIL / Image Threshold (mặc định: 0.99).
        pixel_quantile: Phân vị pixel heatmap normal dùng làm Pixel Threshold (mặc định: 0.99).
        min_calibration_samples: Số lượng ảnh calibration tối thiểu yêu cầu (mặc định: 20).
        coreset_fraction: Tỷ lệ patch trích xuất để tạo coreset đại diện.
        min_coreset_size: Kích thước tối thiểu của tập patch memory bank coreset.
        max_coreset_size: Kích thước tối đa của tập patch memory bank coreset.
        smooth_sigma: Độ lệch chuẩn Sigma cho bộ lọc Gaussian Smoothing làm mịn anomaly map.
        preprocessing: Cấu hình tiền xử lý ảnh (PreprocessingConfig).
    """

    category: str = DEFAULT_CATEGORY
    seed: int = DEFAULT_SEED
    backbone: str = DEFAULT_BACKBONE
    feature_layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS
    pretrained: bool = True
    batch_size: int = 8
    calibration_fraction: float = 0.2
    review_quantile: float = 0.95
    threshold_quantile: float = 0.99
    pixel_quantile: float = 0.99
    min_calibration_samples: int = 20
    coreset_fraction: float = 0.05
    min_coreset_size: int = 100
    max_coreset_size: int = 1000
    smooth_sigma: float = 1.0
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của tất cả các tham số cấu hình.

        Raises:
            ValueError: Nếu bất kỳ tham số nào nằm ngoài dải hợp lệ.
        """
        if not self.category.strip():
            raise ValueError("Tên danh mục (category) không được để trống.")
        if not self.backbone.strip():
            raise ValueError("Tên backbone không được để trống.")
        if not self.feature_layers:
            raise ValueError("Danh sách feature_layers không được rỗng.")
        if self.batch_size <= 0:
            raise ValueError("Kích thước batch (batch_size) phải lớn hơn 0.")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction phải thuộc khoảng (0, 0.5).")
        if self.min_calibration_samples < 5:
            raise ValueError(
                "min_calibration_samples phải >= 5 để đảm bảo ước lượng quantile có ý nghĩa."
            )
        if not (0.5 <= self.review_quantile < self.threshold_quantile < 1.0):
            raise ValueError(
                "review_quantile phải thuộc [0.5, threshold_quantile) và nhỏ hơn threshold_quantile."
            )
        if not 0.5 <= self.pixel_quantile < 1.0:
            raise ValueError("pixel_quantile phải thuộc khoảng [0.5, 1.0).")
        if not 0 < self.coreset_fraction <= 1:
            raise ValueError("coreset_fraction phải thuộc khoảng (0, 1].")
        if not 1 <= self.min_coreset_size <= self.max_coreset_size:
            raise ValueError(
                "Kích thước coreset tối thiểu/tối đa không hợp lệ (min phải <= max và min >= 1)."
            )
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma không được âm.")


def parse_args() -> TrainConfig:
    """Đọc tham số dòng lệnh CLI và trả về cấu hình TrainConfig đã kiểm tra hợp lệ."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phát hiện lỗi ngoại quan PatchCore-style cho MVTec AD"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_CATEGORY,
        help="Tên danh mục sản phẩm trong MVTec AD",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Giá trị seed ngẫu nhiên"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE,
        help="Kiến trúc mạng backbone (mặc định: resnet18)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Kích thước batch cho DataLoader"
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.2,
        help="Tỷ lệ ảnh normal giữ riêng cho calibration",
    )
    parser.add_argument(
        "--review-quantile",
        type=float,
        default=0.95,
        help="Phân vị score normal dùng làm review_threshold",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Phân vị score normal dùng làm fail_threshold (image_threshold)",
    )
    parser.add_argument(
        "--pixel-quantile",
        type=float,
        default=0.99,
        help="Phân vị pixel heatmap normal dùng làm pixel_threshold",
    )
    parser.add_argument(
        "--min-calibration-samples",
        type=int,
        default=20,
        help="Số lượng ảnh calibration tối thiểu yêu cầu",
    )
    parser.add_argument(
        "--coreset-fraction",
        type=float,
        default=0.05,
        help="Tỷ lệ mẫu patch giữ lại qua coreset",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Độ mịn Gaussian smoothing cho anomaly map",
    )

    args = parser.parse_args()
    config = TrainConfig(
        category=args.category,
        seed=args.seed,
        backbone=args.backbone,
        batch_size=args.batch_size,
        calibration_fraction=args.calibration_fraction,
        review_quantile=args.review_quantile,
        threshold_quantile=args.threshold_quantile,
        pixel_quantile=args.pixel_quantile,
        min_calibration_samples=args.min_calibration_samples,
        coreset_fraction=args.coreset_fraction,
        smooth_sigma=args.smooth_sigma,
    )
    config.validate()
    return config
