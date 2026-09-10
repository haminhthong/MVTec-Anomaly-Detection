"""Các thành phần dữ liệu của MVTec AD."""

from __future__ import annotations

from .manifest import DatasetManifest, EvaluationManifest, NormalReferenceManifest
from .validation import (
    DatasetValidationError,
    validate_evaluation,
    validate_mvtec_category,
    validate_reference_category,
)

__all__ = [
    "DatasetManifest",
    "NormalReferenceManifest",
    "EvaluationManifest",
    "DatasetValidationError",
    "validate_mvtec_category",
    "validate_reference_category",
    "validate_evaluation",
]


def __getattr__(name: str):
    """Lazy import phần phụ thuộc torch để manifest/hash vẫn dùng được offline."""
    if name in {"ImageFolderDataset", "find_category_root"}:
        from .dataset import ImageFolderDataset, find_category_root

        return {"ImageFolderDataset": ImageFolderDataset, "find_category_root": find_category_root}[name]
    if name in {"PreprocessingConfig", "build_transform", "DEFAULT_PREPROCESSING_CONFIG", "TFM"}:
        from .transforms import DEFAULT_PREPROCESSING_CONFIG, TFM, PreprocessingConfig, build_transform

        return {
            "PreprocessingConfig": PreprocessingConfig,
            "build_transform": build_transform,
            "DEFAULT_PREPROCESSING_CONFIG": DEFAULT_PREPROCESSING_CONFIG,
            "TFM": TFM,
        }[name]
    raise AttributeError(name)
