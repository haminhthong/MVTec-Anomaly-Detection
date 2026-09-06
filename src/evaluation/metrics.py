"""Module tính toán bộ chỉ số đánh giá 3 tầng (3-Tier Evaluation Metrics).

Cung cấp đánh giá toàn diện chuẩn nghiên cứu và vận hành công nghiệp:
1. Tier 1: Detection (Image-level AUROC & Average Precision)
2. Tier 2: Localization (Pixel-level AUROC, Average Precision & AUPRO@0.3)
3. Tier 3: Operational Decision (Accuracy, Precision, Defect Recall, Specificity, F1, FRR, FAR & Confusion Matrix)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .aupro import compute_aupro


def calculate_3tier_metrics(
    y_true: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    threshold: float,
    review_threshold: float | None = None,
) -> dict[str, Any]:
    """Tính toán bộ chỉ số đánh giá 3 tầng hoàn chỉnh bao gồm phân loại nhị phân và 3 luồng vận hành QC.

    Args:
        y_true: Nhãn thực tế cấp ảnh (0: normal/good, 1: defective).
        scores: Điểm bất thường dự đoán cho từng ảnh.
        masks: Mảng 3D boolean ground-truth masks [N, H, W].
        maps: Mảng 3D float32 anomaly heatmaps dự đoán [N, H, W].
        threshold: Ngưỡng phát hiện lỗi đã được căn chỉnh trên held-out normal (fail_threshold).
        review_threshold: Ngưỡng cảnh báo cần kiểm tra thủ công (review_threshold).

    Returns:
        dict[str, Any]: Dictionary có cấu trúc 3 tầng kèm confusion matrix và 3-way QC metrics.
    """
    y_arr = np.asarray(y_true, dtype=int)
    scores_arr = np.asarray(scores, dtype=np.float32)
    y_pred = (scores_arr >= threshold).astype(int)

    # Ma trận nhầm lẫn nhị phân tại fail_threshold (Confusion Matrix)
    tp = int(np.sum((y_pred == 1) & (y_arr == 1)))
    fp = int(np.sum((y_pred == 1) & (y_arr == 0)))
    tn = int(np.sum((y_pred == 0) & (y_arr == 0)))
    fn = int(np.sum((y_pred == 0) & (y_arr == 1)))

    total_defect = tp + fn
    total_normal = tn + fp
    total_samples = len(y_arr)

    # Các chỉ số vận hành QC nhị phân
    accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    defect_recall = float(tp / total_defect) if total_defect > 0 else 0.0
    specificity = float(tn / total_normal) if total_normal > 0 else 0.0
    f1_score = (
        float(2 * precision * defect_recall / (precision + defect_recall))
        if (precision + defect_recall) > 0
        else 0.0
    )

    # False Reject Rate (FRR): Sản phẩm tốt nhưng bị báo lỗi (gây lãng phí phế phẩm)
    false_reject_rate = float(fp / total_normal) if total_normal > 0 else 0.0
    # False Accept Rate (FAR): Sản phẩm lỗi nhưng bị lọt qua thành PASS (rủi ro nghiêm trọng cho khách hàng)
    false_accept_rate = float(fn / total_defect) if total_defect > 0 else 0.0

    # 3-way operational metrics (PASS / REVIEW / FAIL)
    rev_th = review_threshold if review_threshold is not None else 0.8 * threshold
    pass_mask = scores_arr < rev_th
    review_mask = (scores_arr >= rev_th) & (scores_arr < threshold)
    fail_mask = scores_arr >= threshold

    n_pass = int(np.sum(pass_mask))
    n_review = int(np.sum(review_mask))
    n_fail = int(np.sum(fail_mask))

    auto_pass_rate = float(n_pass / total_samples) if total_samples > 0 else 0.0
    manual_review_rate = float(n_review / total_samples) if total_samples > 0 else 0.0
    auto_fail_rate = float(n_fail / total_samples) if total_samples > 0 else 0.0

    defect_in_review = int(np.sum(review_mask & (y_arr == 1)))
    normal_in_review = int(np.sum(review_mask & (y_arr == 0)))
    defect_in_review_rate = float(defect_in_review / n_review) if n_review > 0 else 0.0
    normal_in_review_rate = float(normal_in_review / n_review) if n_review > 0 else 0.0

    defect_escaped = int(np.sum(pass_mask & (y_arr == 1)))
    defect_escape_after_auto_pass = (
        float(defect_escaped / total_defect) if total_defect > 0 else 0.0
    )
    clean_pass_rate = (
        float(np.sum(pass_mask & (y_arr == 0)) / n_pass) if n_pass > 0 else 1.0
    )

    # Tier 1: Detection
    image_auroc = float(roc_auc_score(y_arr, scores_arr))
    image_ap = float(average_precision_score(y_arr, scores_arr))

    # Tier 2: Localization
    pixel_auroc = float(roc_auc_score(masks.ravel(), maps.ravel()))
    pixel_ap = float(average_precision_score(masks.ravel(), maps.ravel()))
    aupro_val = compute_aupro(masks, maps)

    return {
        "detection": {
            "image_auroc": image_auroc,
            "image_average_precision": image_ap,
        },
        "localization": {
            "pixel_auroc": pixel_auroc,
            "pixel_average_precision": pixel_ap,
            "aupro_0.3": aupro_val,
        },
        "operational_decision": {
            "threshold": threshold,
            "review_threshold": rev_th,
            "accuracy": accuracy,
            "precision": precision,
            "defect_recall": defect_recall,
            "specificity": specificity,
            "f1_score": f1_score,
            "false_reject_rate": false_reject_rate,
            "false_accept_rate": false_accept_rate,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
            # 3-Way operational QC rates
            "auto_pass_rate": auto_pass_rate,
            "manual_review_rate": manual_review_rate,
            "auto_fail_rate": auto_fail_rate,
            "defect_in_review_rate": defect_in_review_rate,
            "normal_in_review_rate": normal_in_review_rate,
            "defect_escape_after_auto_pass": defect_escape_after_auto_pass,
            "clean_pass_rate": clean_pass_rate,
            "counts": {
                "auto_pass": n_pass,
                "manual_review": n_review,
                "auto_fail": n_fail,
                "defect_in_review": defect_in_review,
                "normal_in_review": normal_in_review,
                "defect_escaped": defect_escaped,
            },
        },
        # Flat convenience fields
        "image_auroc": image_auroc,
        "image_average_precision": image_ap,
        "pixel_auroc": pixel_auroc,
        "pixel_average_precision": pixel_ap,
        "aupro_0.3": aupro_val,
        "threshold": threshold,
        "review_threshold": rev_th,
        "accuracy_at_threshold": accuracy,
        "defect_recall": defect_recall,
        "false_reject_rate": false_reject_rate,
        "false_accept_rate": false_accept_rate,
        "auto_pass_rate": auto_pass_rate,
        "manual_review_rate": manual_review_rate,
        "auto_fail_rate": auto_fail_rate,
        "defect_escape_after_auto_pass": defect_escape_after_auto_pass,
    }
