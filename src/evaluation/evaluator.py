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

from ..data.manifest import DatasetManifest
from ..data.validation import validate_mvtec_category
from ..inference.detector import AnomalyDetector
from ..model.artifacts import ModelArtifact
from .metrics import calculate_3tier_metrics


def evaluate_category(
    category: str | DatasetManifest | None = None,
    model_dir: str | Path | ModelArtifact | None = "models",
    data_dir: str | Path = "data/raw",
    output_report: str | Path | None = None,
    manifest: DatasetManifest | None = None,
    artifact: ModelArtifact | str | Path | None = None,
) -> dict[str, Any]:
    """Run comprehensive 3-tier evaluation on the test split for a category.

    # REPORT-ONLY:
    # This function must never modify model thresholds or leak test labels to model building.

    Args:
        category: Name of product category (or DatasetManifest if passed positionally).
        model_dir: Path to directory containing model artifacts (or ModelArtifact if passed positionally).
        data_dir: Path to raw datasets directory.
        output_report: Path to output JSON file (defaults to reports/<category>/test_metrics.json).
        manifest: Pre-validated DatasetManifest.
        artifact: ModelArtifact instance or path to artifact directory.

    Returns:
        dict[str, Any]: 3-tier metrics dictionary.
    """
    # 1. Resolve manifest and category
    if isinstance(category, DatasetManifest):
        manifest_obj = category
        resolved_category = manifest_obj.category
        if isinstance(model_dir, ModelArtifact):
            det = AnomalyDetector(model_dir=Path("models") / resolved_category, category=resolved_category)
        elif model_dir is not None:
            det = AnomalyDetector(model_dir=model_dir, category=resolved_category)
        else:
            det = AnomalyDetector(model_dir="models", category=resolved_category)
    else:
        manifest_obj = manifest
        # Check artifact
        if isinstance(artifact, ModelArtifact):
            cat = category or artifact.metadata.category
            det = AnomalyDetector(model_dir=model_dir or "models", category=cat)
        elif artifact is not None:
            det = AnomalyDetector(model_dir=artifact, category=category)
        else:
            det = AnomalyDetector(model_dir=model_dir or "models", category=category)
        resolved_category = det.category

    if manifest_obj is None:
        manifest_obj = validate_mvtec_category(
            data_dir=data_dir, category=resolved_category
        )

    image_size = det.preprocessing_config.image_size  # (H, W) dynamically resolved

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
    test_items = manifest_obj.get_all_test_paths()

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
        review_threshold=det.review_threshold,
    )

    metrics_result["category"] = resolved_category
    metrics_result["model_version"] = det.model_version
    metrics_result["test_samples_total"] = len(ys)
    metrics_result["test_defect_count"] = sum(ys)
    metrics_result["test_normal_count"] = len(ys) - sum(ys)
    if manifest_obj.fingerprint:
        metrics_result["dataset_fingerprint"] = manifest_obj.fingerprint

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
    counts = op["counts"]

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
    print("\n [TIER 3: OPERATIONAL QC (Binary Operating Policy at Fail Threshold)]")
    print(f"  - Calibrated Fail Threshold  : {op['threshold']:.4f}")
    print(f"  - Accuracy                   : {op['accuracy']:.4f}")
    print(f"  - Precision                  : {op['precision']:.4f}")
    print(f"  - Defect Recall (TPR)        : {op['defect_recall']:.4f} (Sensitivity)")
    print(f"  - Specificity (TNR)          : {op['specificity']:.4f}")
    print(f"  - F1 Score                   : {op['f1_score']:.4f}")
    print(f"  - False Reject Rate (FRR)    : {op['false_reject_rate']:.4f} (Scrap waste)")
    print(f"  - False Accept Rate (FAR)    : {op['false_accept_rate']:.4f} (Customer risk)")
    print("\n [OPERATIONAL 3-WAY QC WORKFLOW (PASS / REVIEW / FAIL)]")
    print(f"  - Calibrated Review Threshold: {op['review_threshold']:.4f}")
    print(f"  - Auto-PASS Rate             : {op['auto_pass_rate']:.4f} ({counts['auto_pass']}/{len(ys)} units)")
    print(f"  - Manual-REVIEW Rate         : {op['manual_review_rate']:.4f} ({counts['manual_review']}/{len(ys)} units)")
    print(f"  - Auto-FAIL Rate             : {op['auto_fail_rate']:.4f} ({counts['auto_fail']}/{len(ys)} units)")
    print(f"  - Defect in Review Rate      : {op['defect_in_review_rate']:.4f} ({counts['defect_in_review']}/{max(1, counts['manual_review'])} units)")
    print(f"  - Defect Escape after PASS   : {op['defect_escape_after_auto_pass']:.4f} ({counts['defect_escaped']}/{max(1, sum(ys))} defective units)")
    print(f"  - Clean Pass Rate            : {op['clean_pass_rate']:.4f}")
    print("\n [CONFUSION MATRIX]")
    print(f"                 Pred PASS    Pred FAIL")
    print(f"  Normal (Good):   TN={cm['tn']:<4}      FP={cm['fp']:<4}  (Total: {cm['tn']+cm['fp']})")
    print(f"  Defect:          FN={cm['fn']:<4}      TP={cm['tp']:<4}  (Total: {cm['tp']+cm['fn']})")
    print("=" * 68 + "\n")

    return metrics_result
