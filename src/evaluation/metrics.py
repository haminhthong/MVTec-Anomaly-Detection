"""Metric research và metric workflow được tách nghĩa rõ ràng."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .aupro import compute_aupro


def _safe_metric(function: Any, labels: np.ndarray, values: np.ndarray) -> float | None:
    """Metric classification trả None khi slice chỉ có một class."""
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, values))


def calculate_3tier_metrics(
    y_true: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    threshold: float | None = None,
    review_threshold: float | None = None,
    auto_pass_threshold: float | None = None,
) -> dict[str, Any]:
    """Tính benchmark metrics và workflow metrics tại ngưỡng AUTO_PASS.

    ``threshold`` là alias cũ. Không còn vùng AUTO_FAIL: mọi score không đạt
    AUTO_PASS đều được tính là HUMAN_REVIEW. Vì vậy escape rate chỉ đếm defect
    thực sự đi qua AUTO_PASS.
    """
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
    selected_threshold = auto_pass_threshold
    if selected_threshold is None:
        selected_threshold = threshold
    if selected_threshold is None:
        raise ValueError("Cần truyền auto_pass_threshold.")
    selected_threshold = float(selected_threshold)

    auto_pass = scores_arr < selected_threshold
    human_review = ~auto_pass
    total_samples = len(y_arr)
    normal = y_arr == 0
    defect = y_arr == 1
    total_normal = int(normal.sum())
    total_defect = int(defect.sum())
    n_auto_pass = int(auto_pass.sum())
    n_review = int(human_review.sum())
    normal_auto_pass = int(np.sum(auto_pass & normal))
    defect_escape = int(np.sum(auto_pass & defect))
    defect_review = int(np.sum(human_review & defect))
    normal_review = int(np.sum(human_review & normal))

    tp = defect_review
    fp = normal_review
    tn = normal_auto_pass
    fn = defect_escape
    accuracy = float((tp + tn) / total_samples) if total_samples else 0.0
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / total_defect) if total_defect else 0.0
    specificity = float(tn / total_normal) if total_normal else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    escape_rate = float(defect_escape / total_defect) if total_defect else 0.0

    image_auroc = _safe_metric(roc_auc_score, y_arr, scores_arr)
    image_ap = _safe_metric(average_precision_score, y_arr, scores_arr)
    flat_masks = mask_arr.ravel()
    flat_maps = map_arr.ravel()
    pixel_auroc = _safe_metric(roc_auc_score, flat_masks.astype(int), flat_maps)
    pixel_ap = _safe_metric(average_precision_score, flat_masks.astype(int), flat_maps)
    aupro_value = compute_aupro(masks, maps) if np.any(flat_masks) else None

    confusion_matrix = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    operational = {
        "auto_pass_threshold": selected_threshold,
        "normal_auto_pass_rate": float(normal_auto_pass / total_normal) if total_normal else 0.0,
        "normal_review_rate": float(normal_review / total_normal) if total_normal else 0.0,
        "defect_capture_to_review_rate": float(defect_review / total_defect) if total_defect else 0.0,
        "defect_escape_after_auto_pass": escape_rate,
        "review_rate": float(n_review / total_samples) if total_samples else 0.0,
        "counts": {
            "auto_pass": n_auto_pass,
            "human_review": n_review,
            "normal_auto_pass": normal_auto_pass,
            "normal_review": normal_review,
            "defect_capture_to_review": defect_review,
            "defect_escaped_to_auto_pass": defect_escape,
        },
        "confusion_matrix_at_auto_pass_boundary": confusion_matrix,
        "confusion_matrix": confusion_matrix,
        "accuracy_at_boundary": accuracy,
        "precision_at_boundary": precision,
        "defect_recall_at_boundary": recall,
        "specificity_at_boundary": specificity,
        "f1_at_boundary": f1,
        # Alias để client cũ hiểu rằng đây là escape sau AUTO_PASS, không phải FAR của 3-way policy.
        "false_accept_rate": escape_rate,
        # Alias legacy chỉ để không phá consumer cũ; semantics vẫn là boundary
        # AUTO_PASS, không phải một ngưỡng AUTO_FAIL thứ hai.
        "threshold": selected_threshold,
        "review_threshold": selected_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "defect_recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "false_reject_rate": float(normal_review / total_normal) if total_normal else 0.0,
        "auto_pass_rate": float(n_auto_pass / total_samples) if total_samples else 0.0,
        "manual_review_rate": float(n_review / total_samples) if total_samples else 0.0,
        "auto_fail_rate": 0.0,
    }
    result = {
        "locked_test": True,
        "detection": {"image_auroc": image_auroc, "image_average_precision": image_ap},
        "localization": {
            "pixel_auroc": pixel_auroc,
            "pixel_average_precision": pixel_ap,
            "aupro_0.3": aupro_value,
        },
        "operational_decision": operational,
        "image_auroc": image_auroc,
        "image_average_precision": image_ap,
        "pixel_auroc": pixel_auroc,
        "pixel_average_precision": pixel_ap,
        "aupro_0.3": aupro_value,
        "auto_pass_threshold": selected_threshold,
        "defect_escape_after_auto_pass": escape_rate,
        "normal_auto_pass_rate": operational["normal_auto_pass_rate"],
        "normal_review_rate": operational["normal_review_rate"],
        "defect_capture_to_review_rate": operational["defect_capture_to_review_rate"],
    }
    return result
