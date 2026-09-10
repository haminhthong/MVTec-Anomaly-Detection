"""Validator tách biệt reference normal khỏi official test.

Điểm quan trọng: ``validate_reference_category`` chỉ duyệt ``train/good``.
Code xây model dùng hàm này, vì vậy không vô tình phụ thuộc vào test/mask.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .manifest import (
    DatasetManifest,
    EvaluationManifest,
    NormalReferenceManifest,
    SUPPORTED_EXTENSIONS,
)
from ..path_safety import ensure_safe_segment


class DatasetValidationError(Exception):
    """Lỗi cấu trúc, định dạng hoặc tính toàn vẹn của dataset."""


def _find_category_root(data_dir: str | Path, category: str) -> Path:
    """Tìm thư mục category theo hai layout phổ biến của MVTec AD."""
    category = ensure_safe_segment(category, "category")
    raw_path = Path(data_dir)
    candidates = [
        raw_path / category,
        raw_path / "mvtec_anomaly_detection" / category,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Không tìm thấy category '{category}' dưới '{raw_path}'. "
        "Hãy kiểm tra đường dẫn hoặc chạy scripts/download_data.py."
    )


def _image_files(directory: Path, description: str) -> list[Path]:
    """Lấy danh sách ảnh hợp lệ trong một thư mục bắt buộc."""
    if not directory.is_dir():
        raise DatasetValidationError(f"Thiếu thư mục bắt buộc: '{directory}'.")
    files = sorted(path for path in directory.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise DatasetValidationError(f"Không có ảnh hợp lệ trong {description} '{directory}'.")
    return files


def _source_metadata(cat_root: Path) -> dict[str, str | None]:
    """Đọc thông tin nguồn dữ liệu do script tải ghi lại, nếu có."""
    candidates = (cat_root.parent / "DATASET_SOURCE.json", cat_root / "DATASET_SOURCE.json")
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {
                    "source": data.get("source", "MVTec AD"),
                    "source_version": data.get("source_version", "unknown"),
                    "download_date": data.get("download_date"),
                    "license": data.get("license", "CC BY-NC-SA 4.0"),
                }
            except (OSError, json.JSONDecodeError):
                break
    return {
        "source": "MVTec AD",
        "source_version": "unknown",
        "download_date": None,
        "license": "CC BY-NC-SA 4.0",
    }


def _verify_images(paths: list[Path]) -> None:
    """Đọc metadata và kiểm tra từng ảnh để phát hiện file hỏng."""
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise DatasetValidationError(f"File ảnh hỏng hoặc không đọc được: '{path}'.") from exc


def validate_reference_category(
    data_dir: str | Path = "data/raw",
    category: str = "bottle",
    check_image_integrity: bool = False,
) -> NormalReferenceManifest:
    """Kiểm tra chỉ ``train/good`` và tạo normal reference manifest.

    Hàm này cố ý không kiểm tra ``test`` hay ``ground_truth``. Đây là boundary
    chống leakage của model-building pipeline.
    """
    cat_root = _find_category_root(data_dir, category)
    train_good = _image_files(cat_root / "train" / "good", "train/good")
    if check_image_integrity:
        _verify_images(train_good)
    metadata = _source_metadata(cat_root)
    return NormalReferenceManifest(
        category=category,
        root_path=cat_root,
        train_good=train_good,
        **metadata,
    )


def validate_evaluation(
    data_dir: str | Path = "data/raw",
    category: str = "bottle",
    check_image_integrity: bool = False,
) -> EvaluationManifest:
    """Kiểm tra official test và mask cho bước đánh giá cuối.

    Hàm này không được import vào training module.
    """
    cat_root = _find_category_root(data_dir, category)
    test_root = cat_root / "test"
    test_good = _image_files(test_root / "good", "test/good")
    ground_truth_root = cat_root / "ground_truth"
    test_defect: dict[str, list[Path]] = {}
    masks: dict[str, Path] = {}

    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        defect_type = defect_dir.name
        defect_images = _image_files(defect_dir, f"test/{defect_type}")
        test_defect[defect_type] = defect_images
        mask_dir = ground_truth_root / defect_type
        if not mask_dir.is_dir():
            raise DatasetValidationError(
                f"Thiếu thư mục ground_truth cho defect '{defect_type}': '{mask_dir}'."
            )
        for image_path in defect_images:
            candidates = (
                mask_dir / f"{image_path.stem}_mask.png",
                mask_dir / f"{image_path.stem}_mask{image_path.suffix}",
                mask_dir / image_path.name,
            )
            mask_path = next((candidate for candidate in candidates if candidate.exists()), None)
            if mask_path is None:
                raise DatasetValidationError(
                    f"Thiếu ground-truth mask cho ảnh lỗi '{image_path}'."
                )
            masks[str(image_path)] = mask_path

    if check_image_integrity:
        defect_images = [path for paths in test_defect.values() for path in paths]
        _verify_images(test_good + defect_images + list(masks.values()))

    metadata = _source_metadata(cat_root)
    return EvaluationManifest(
        category=category,
        root_path=cat_root,
        test_good=test_good,
        test_defect=test_defect,
        masks=masks,
        **metadata,
    )


def validate_mvtec_category(
    data_dir: str | Path = "data/raw",
    category: str = "bottle",
    check_image_integrity: bool = False,
) -> DatasetManifest:
    """Tạo manifest tổng hợp cho lệnh kiểm tra đầy đủ category.

    Training không gọi hàm này; nó gọi
    ``validate_reference_category`` để giữ test isolation ở mức kiến trúc.
    """
    reference = validate_reference_category(data_dir, category, check_image_integrity)
    evaluation = validate_evaluation(data_dir, category, check_image_integrity)
    all_paths = list(reference.train_good) + list(evaluation.test_good)
    all_paths.extend(path for paths in evaluation.test_defect.values() for path in paths)
    all_paths.extend(evaluation.masks.values())
    from .manifest import build_file_records, fingerprint_records

    records = build_file_records(reference.root_path, all_paths)
    return DatasetManifest(
        category=category,
        root_path=reference.root_path,
        train_good=reference.train_good,
        test_good=evaluation.test_good,
        test_defect=evaluation.test_defect,
        masks=evaluation.masks,
        fingerprint=fingerprint_records(category, records),
        source=reference.source,
        source_version=reference.source_version,
        download_date=reference.download_date,
        license=reference.license,
        file_records=records,
    )
