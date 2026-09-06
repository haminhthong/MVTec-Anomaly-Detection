"""Configuration for the PatchCore-style offline model-building pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .data.transforms import PreprocessingConfig

DEFAULT_CATEGORY = "bottle"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class TrainConfig:
    """Single source of truth for training and normal-only calibration."""

    category: str = DEFAULT_CATEGORY
    data_root: str = "data/raw"
    model_root: str = "models"
    seed: int = DEFAULT_SEED
    batch_size: int = 8
    num_workers: int = 0
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
        if not self.category.strip():
            raise ValueError("category must not be empty")
        if not self.data_root.strip():
            raise ValueError("data_root must not be empty")
        if not self.model_root.strip():
            raise ValueError("model_root must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in (0, 0.5)")
        if self.min_calibration_samples < 5:
            raise ValueError("min_calibration_samples must be >= 5")
        if not (0.5 <= self.review_quantile < self.threshold_quantile < 1.0):
            raise ValueError(
                "review_quantile must be >= 0.5 and smaller than threshold_quantile < 1.0"
            )
        if not 0.5 <= self.pixel_quantile < 1.0:
            raise ValueError("pixel_quantile must be in [0.5, 1.0)")
        if not 0 < self.coreset_fraction <= 1:
            raise ValueError("coreset_fraction must be in (0, 1]")
        if not 1 <= self.min_coreset_size <= self.max_coreset_size:
            raise ValueError("coreset size bounds are invalid")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be >= 0")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Build a category-scoped PatchCore-style model from MVTec AD normal images"
    )
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--review-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--pixel-quantile", type=float, default=0.99)
    parser.add_argument("--min-calibration-samples", type=int, default=20)
    parser.add_argument("--coreset-fraction", type=float, default=0.05)
    parser.add_argument("--min-coreset-size", type=int, default=100)
    parser.add_argument("--max-coreset-size", type=int, default=1000)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    args = parser.parse_args()

    if args.image_size <= 0:
        parser.error("--image-size must be > 0")

    cfg = TrainConfig(
        category=args.category,
        data_root=args.data_root,
        model_root=args.model_root,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calibration_fraction=args.calibration_fraction,
        review_quantile=args.review_quantile,
        threshold_quantile=args.threshold_quantile,
        pixel_quantile=args.pixel_quantile,
        min_calibration_samples=args.min_calibration_samples,
        coreset_fraction=args.coreset_fraction,
        min_coreset_size=args.min_coreset_size,
        max_coreset_size=args.max_coreset_size,
        smooth_sigma=args.smooth_sigma,
        preprocessing=PreprocessingConfig(image_size=(args.image_size, args.image_size)),
    )
    cfg.validate()
    return cfg
