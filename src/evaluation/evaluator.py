"""Evaluation Pipeline for industrial visual anomaly detection.

IMPORTANT ANTI-LEAKAGE POLICY:
# REPORT-ONLY:
# This module must never modify, retune, or optimize model thresholds.
# It evaluates frozen artifacts strictly against test samples and ground-truth masks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..data.validation import DatasetManifest, validate_mvtec_category
from ..inference.detector import AnomalyDetector
from .metrics import calculate_3tier_metrics


def evaluate_category(
    category: str | None = None,
    model_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    """Run comprehensive 3-tier evaluation on the test split for a category.

    # REPORT-ONLY:
    # This function must never modify model thresholds or leak test labels to model building.

    Args:
        category: Name of product category (if None, resolved from detector artifact).
        model_dir: Path to directory containing model artifacts.
        data_dir: Path to raw datasets directory.
        output_report: Path to output JSON file (defaults to reports/<category>/test_metrics.json).

    Returns:
        dict[str, Any]: 3-tier metrics dictionary.
    """
    # 1. Load frozen detector artifact
    det = AnomalyDetector(model_dir=model_dir, category=category)
    resolved_category = det.category
    image_size = det.preprocessing_config.image_size  # (H, W) dynamically resolved

    # 2. Validate test dataset via DatasetManifest
    manifest: DatasetManifest = validate_mvtec_category(
        data_dir=data_dir, category=resolved_category
    )

    ys: list[int] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []

    print(
        f"\n[EVALUATION] Evaluating frozen PatchCore-style model for '{resolved_category}'..."
    )
    print(f"  - Artifact version: {det.model_version}")
    print(f"  - Calibrated Image Fail Threshold: {det.threshold:.4f} (P99 normal)")
    print(f"  - Calibrated Review Threshold: {det.review_threshold:.4f} (P95 normal)")
    print(f"  - Calibrated Pixel Threshold: {det.pixel_threshold:.4f} (P99 normal)")

    h_target, w_target = image_size
    test_items = manifest.get_all_test_paths()

    for img_path, is_defective, mask_path in test_items:
        with Image.open(img_path) as img:
            s, heat = det.score(img)

        ys.append(is_defective)
        scores.append(s)

        # Process ground-truth mask
        if is_defective:
            if mask_path is None or not mask_path.exists():
                raise FileNotFoundError(
                    f"Defect test image '{img_path}' is missing its required ground-truth mask."
                )
            with Image.open(mask_path) as m_img:
                mask = (
                    np.asarray(
                        m_img.convert("L").resize(
                            (w_target, h_target), Image.Resampling.NEAREST
                        )
                    )
                    > 0
                )
        else:
            mask = np.zeros((h_target, w_target), dtype=bool)

        # Resize anomaly heatmap to match image target resolution
        anomaly_map = np.asarray(
            Image.fromarray(heat.astype(np.float32)).resize(
                (w_target, h_target), Image.Resampling.BILINEAR
            )
        )

        masks.append(mask)
        maps.append(anomaly_map)

    masks_array = np.asarray(masks)
    maps_array = np.asarray(maps)

    # 3. Compute 3-tier metrics
    metrics_result = calculate_3tier_metrics(
        y_true=ys,
        scores=scores,
        masks=masks_array,
        maps=maps_array,
        threshold=det.threshold,
    )

    metrics_result["category"] = resolved_category
    metrics_result["model_version"] = det.model_version
    metrics_result["test_samples_total"] = len(ys)
    metrics_result["test_defect_count"] = sum(ys)
    metrics_result["test_normal_count"] = len(ys) - sum(ys)

    # 4. Save report
    if output_report is None:
        report_file = Path("reports") / resolved_category / "test_metrics.json"
    else:
        report_file = Path(output_report)

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(metrics_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary
    det_tier = metrics_result["detection"]
    loc_tier = metrics_result["localization"]
    op = metrics_result["operational_decision"]
    cm = op["confusion_matrix"]

    print("\n" + "=" * 68)
    print(f"      3-TIER EVALUATION REPORT: {resolved_category.upper()}")
    print("=" * 68)
    print(" [TIER 1: DETECTION (Image-level Classification)]")
    print(f"  - Image AUROC                : {det_tier['image_auroc']:.4f}")
    print(f"  - Image Average Precision    : {det_tier['image_average_precision']:.4f}")
    print("\n [TIER 2: LOCALIZATION (Pixel-level Segmentation)]")
    print(f"  - Pixel AUROC                : {loc_tier['pixel_auroc']:.4f}")
    print(f"  - Pixel Average Precision    : {loc_tier['pixel_average_precision']:.4f}")
    print(f"  - AUPRO (max_fpr=0.3)        : {loc_tier['aupro_0.3']:.4f}")
    print("\n [TIER 3: OPERATIONAL QC (Operating Policy at Normal-Calibrated Threshold)]")
    print(f"  - Calibrated Fail Threshold  : {op['threshold']:.4f}")
    print(f"  - Accuracy                   : {op['accuracy']:.4f}")
    print(f"  - Precision                  : {op['precision']:.4f}")
    print(f"  - Defect Recall (TPR)        : {op['defect_recall']:.4f} (Sensitivity)")
    print(f"  - Specificity (TNR)          : {op['specificity']:.4f}")
    print(f"  - F1 Score                   : {op['f1_score']:.4f}")
    print(f"  - False Reject Rate (FRR)    : {op['false_reject_rate']:.4f} (Scrap waste)")
    print(f"  - False Accept Rate (FAR)    : {op['false_accept_rate']:.4f} (Customer risk)")
    print("\n [CONFUSION MATRIX]")
    print(f"                 Pred PASS    Pred FAIL")
    print(f"  Normal (Good):   TN={cm['tn']:<4}      FP={cm['fp']:<4}  (Total: {cm['tn']+cm['fp']})")
    print(f"  Defect:          FN={cm['fn']:<4}      TP={cm['tp']:<4}  (Total: {cm['tp']+cm['fn']})")
    print("=" * 68 + "\n")

    return metrics_result
