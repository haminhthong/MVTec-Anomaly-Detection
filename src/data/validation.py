"""Data Pipeline validation and DatasetManifest creation for MVTec AD datasets.

Enforces strict integrity:
- Category directory existence
- train/good contains valid images
- test/good contains valid images
- test defect folders contain defect images
- Every defect test image MUST have a corresponding ground-truth mask (missing mask -> fail)
- Image file extensions and readability verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError

from .manifest import DatasetManifest, SUPPORTED_EXTENSIONS


class DatasetValidationError(Exception):
    """Raised when MVTec AD dataset fails structural or integrity validation."""


def validate_mvtec_category(
    data_dir: str | Path = "data/raw",
    category: str = "bottle",
    check_image_integrity: bool = False,
) -> DatasetManifest:
    """Validate structure and integrity of an MVTec AD category dataset.

    Args:
        data_dir: Root directory containing raw MVTec datasets (e.g., data/raw).
        category: Name of the category (e.g. 'bottle', 'cable').
        check_image_integrity: If True, attempts to open each image to detect corruption.

    Returns:
        DatasetManifest: Validated manifest with resolved paths.

    Raises:
        DatasetValidationError: If directory layout, files, or masks fail validation.
        FileNotFoundError: If the category directory cannot be located.
    """
    raw_path = Path(data_dir)
    candidates = [
        raw_path / category,
        raw_path / "mvtec_anomaly_detection" / category,
    ]

    cat_root: Path | None = None
    for c in candidates:
        if c.exists() and c.is_dir():
            cat_root = c
            break

    if cat_root is None:
        raise FileNotFoundError(
            f"Category '{category}' not found under '{raw_path}'. "
            "Please check the path or run 'python scripts/download_data.py --category <category>'."
        )

    # 1. Validate train/good
    train_good_dir = cat_root / "train" / "good"
    if not train_good_dir.exists() or not train_good_dir.is_dir():
        raise DatasetValidationError(
            f"Required directory '{train_good_dir}' does not exist for category '{category}'."
        )

    train_good_files = sorted(
        p for p in train_good_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not train_good_files:
        raise DatasetValidationError(
            f"No valid images found in train directory '{train_good_dir}'."
        )

    # 2. Validate test directory
    test_dir = cat_root / "test"
    if not test_dir.exists() or not test_dir.is_dir():
        raise DatasetValidationError(
            f"Required test directory '{test_dir}' does not exist for category '{category}'."
        )

    test_good_dir = test_dir / "good"
    if not test_good_dir.exists() or not test_good_dir.is_dir():
        raise DatasetValidationError(
            f"Required test/good directory '{test_good_dir}' does not exist for category '{category}'."
        )

    test_good_files = sorted(
        p for p in test_good_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not test_good_files:
        raise DatasetValidationError(
            f"No valid images found in test/good directory '{test_good_dir}'."
        )

    # 3. Validate defect directories & ground truth masks
    gt_root = cat_root / "ground_truth"
    test_defect: dict[str, list[Path]] = {}
    masks: dict[str, Path] = {}

    for sub_dir in sorted(test_dir.iterdir()):
        if not sub_dir.is_dir() or sub_dir.name == "good":
            continue

        defect_type = sub_dir.name
        defect_images = sorted(
            p for p in sub_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not defect_images:
            raise DatasetValidationError(
                f"Defect directory '{sub_dir}' exists but contains no valid images."
            )

        test_defect[defect_type] = defect_images

        # Verify ground-truth masks for each defect image
        gt_defect_dir = gt_root / defect_type
        if not gt_defect_dir.exists() or not gt_defect_dir.is_dir():
            raise DatasetValidationError(
                f"Missing ground_truth directory for defect type '{defect_type}' at '{gt_defect_dir}'."
            )

        for img_path in defect_images:
            # Standard MVTec AD mask naming: <stem>_mask.png
            mask_candidates = [
                gt_defect_dir / f"{img_path.stem}_mask.png",
                gt_defect_dir / f"{img_path.stem}_mask{img_path.suffix}",
                gt_defect_dir / img_path.name,
            ]
            mask_found: Path | None = None
            for mc in mask_candidates:
                if mc.exists():
                    mask_found = mc
                    break

            if mask_found is None:
                raise DatasetValidationError(
                    f"Missing ground-truth mask for defect image '{img_path}'. "
                    f"Expected candidate: '{gt_defect_dir / (img_path.stem + '_mask.png')}'."
                )

            masks[str(img_path)] = mask_found

    # 4. Integrity check (optional image readability)
    if check_image_integrity:
        all_paths: list[Path] = (
            train_good_files + test_good_files + list(masks.values())
        )
        for d_list in test_defect.values():
            all_paths.extend(d_list)

        for img_p in all_paths:
            try:
                with Image.open(img_p) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise DatasetValidationError(
                    f"Corrupted or unreadable image file: '{img_p}'"
                ) from exc

    return DatasetManifest(
        category=category,
        root_path=cat_root,
        train_good=train_good_files,
        test_good=test_good_files,
        test_defect=test_defect,
        masks=masks,
    )
