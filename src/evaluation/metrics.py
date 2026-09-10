"""Metric benchmark và operational triage tại một image threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .aupro import compute_aupro


def _safe_metric(function: Any, labels: np.ndarray, values: np.ndarray) -> float | None:
    """Trả None khi slice chỉ có một class."""
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, values))


def calculate_workflow_metrics(
    y_true: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    image_threshold: float,
) -> dict[str, Any]:
    """Tính detection, localization và triage metrics cho official test."""
    y_arr = np.asarray(y_true, dtype=int)
    scores_arr = np.asarray(scores, dtype=np.float32)
    mask_arr = np.asarray(masks)
    map_arr = np.asarray(maps)
    if not np.isin(y_arr, [0, 1]).all():
        raise ValueError("y_true chỉ được chứa nhãn 0 hoặc 1.")
    if (
        len(y_arr) != len(scores_arr)
        or len(mask_arr) != len(y_arr)
        or len(map_arr) != len(y_arr)
        or mask_arr.shape != map_arr.shape
    ):
        raise ValueError("y_true, scores, masks và maps phải có cùng số mẫu.")

    threshold = float(image_threshold)
    pass_candidate = scores_arr < threshold
    review_required = ~pass_candidate
    normal = y_arr == 0
    defect = y_arr == 1
    total_samples = len(y_arr)
    total_normal = int(normal.sum())
    total_defect = int(defect.sum())
    n_pass = int(pass_candidate.sum())
    n_review = int(review_required.sum())
    normal_pass = int(np.sum(pass_candidate & normal))
    normal_review = int(np.sum(review_required & normal))
    defect_escape = int(np.sum(pass_candidate & defect))
    defect_review = int(np.sum(review_required & defect))

    true_positive = defect_review
    false_positive = normal_review
    true_negative = normal_pass
    false_negative = defect_escape
    accuracy = float((true_positive + true_negative) / total_samples) if total_samples else 0.0
    precision = float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0
    recall = float(true_positive / total_defect) if total_defect else 0.0
    specificity = float(true_negative / total_normal) if total_normal else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    flat_masks = mask_arr.ravel()
    flat_maps = map_arr.ravel()
    confusion = {
        "tp_review_required_defect": true_positive,
        "fp_review_required_normal": false_positive,
        "tn_pass_candidate_normal": true_negative,
        "fn_pass_candidate_defect": false_negative,
    }
    operational = {
        "image_threshold": threshold,
        "pass_candidate_coverage": float(n_pass / total_samples) if total_samples else 0.0,
        "normal_pass_candidate_rate": float(normal_pass / total_normal) if total_normal else 0.0,
        "normal_review_required_rate": float(normal_review / total_normal) if total_normal else 0.0,
        "defect_review_required_rate": float(defect_review / total_defect) if total_defect else 0.0,
        "false_pass_candidate_rate": float(defect_escape / total_defect) if total_defect else 0.0,
        "review_required_rate": float(n_review / total_samples) if total_samples else 0.0,
        "counts": {
            "pass_candidate": n_pass,
            "review_required": n_review,
            "normal_pass_candidate": normal_pass,
            "normal_review_required": normal_review,
            "defect_review_required": defect_review,
            "defect_pass_candidate": defect_escape,
        },
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "precision": precision,
        "defect_recall": recall,
        "specificity": specificity,
        "f1_score": f1,
    }
    return {
        "locked_test": True,
        "detection": {
            "image_auroc": _safe_metric(roc_auc_score, y_arr, scores_arr),
            "image_average_precision": _safe_metric(average_precision_score, y_arr, scores_arr),
        },
        "localization": {
            "pixel_auroc": _safe_metric(roc_auc_score, flat_masks.astype(int), flat_maps),
            "pixel_average_precision": _safe_metric(
                average_precision_score, flat_masks.astype(int), flat_maps
            ),
            "aupro_0.3": compute_aupro(masks, maps) if np.any(flat_masks) else None,
        },
        "operational_decision": operational,
        "image_auroc": _safe_metric(roc_auc_score, y_arr, scores_arr),
        "image_average_precision": _safe_metric(average_precision_score, y_arr, scores_arr),
        "pixel_auroc": _safe_metric(roc_auc_score, flat_masks.astype(int), flat_maps),
        "pixel_average_precision": _safe_metric(
            average_precision_score, flat_masks.astype(int), flat_maps
        ),
        "aupro_0.3": compute_aupro(masks, maps) if np.any(flat_masks) else None,
        "image_threshold": threshold,
        "false_pass_candidate_rate": operational["false_pass_candidate_rate"],
        "normal_pass_candidate_rate": operational["normal_pass_candidate_rate"],
        "review_required_rate": operational["review_required_rate"],
    }
