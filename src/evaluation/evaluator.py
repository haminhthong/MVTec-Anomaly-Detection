"""Đánh giá locked MVTec test bằng artifact đã freeze, không retune policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.manifest import DatasetManifest, LockedEvaluationManifest, NormalReferenceManifest
from ..data.validation import validate_locked_evaluation
from ..inference.detector import AnomalyDetector
from ..model.artifacts import ModelArtifact
from .aupro import compute_aupro
from .metrics import calculate_3tier_metrics


def _safe_metric(function: Any, labels: np.ndarray, values: np.ndarray) -> float | None:
    """Trả None nếu defect slice không có đủ hai class."""
    return float(function(labels, values)) if len(np.unique(labels)) > 1 else None


def _slice_metrics(
    selected: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
) -> dict[str, Any]:
    """Tính image AUROC/AP và localization cho một defect type/area slice."""
    indices = np.flatnonzero(selected | (y_true == 0))
    if not len(indices):
        return {"n": 0, "image_auroc": None, "image_average_precision": None, "aupro_0.3": None}
    return {
        "n": int(selected.sum()),
        "image_auroc": _safe_metric(roc_auc_score, y_true[indices], scores[indices]),
        "image_average_precision": _safe_metric(average_precision_score, y_true[indices], scores[indices]),
        "aupro_0.3": compute_aupro(masks[indices], maps[indices]) if np.any(masks[indices]) else None,
    }


def evaluate_category(
    category: str | DatasetManifest | None = None,
    model_dir: str | Path | ModelArtifact | None = "models",
    data_dir: str | Path = "data/raw",
    output_report: str | Path | None = None,
    manifest: DatasetManifest | LockedEvaluationManifest | None = None,
    artifact: ModelArtifact | str | Path | None = None,
    reopen: bool = False,
) -> dict[str, Any]:
    """Đánh giá official test locked và tạo defect-type/area slices.

    Nếu report locked đã tồn tại, chạy lại phải truyền ``reopen=True`` để tránh
    âm thầm thay đổi bằng chứng final.
    """
    if isinstance(category, DatasetManifest):
        manifest_obj: DatasetManifest | LockedEvaluationManifest | None = category
        resolved_category = category.category
    else:
        manifest_obj = manifest
        resolved_category = category

    if isinstance(manifest_obj, DatasetManifest):
        resolved_category = manifest_obj.category
    if isinstance(manifest_obj, NormalReferenceManifest):
        raise ValueError(
            "Evaluator cần LockedEvaluationManifest hoặc DatasetManifest có test; "
            "không được đánh giá bằng manifest reference-only."
        )
    if resolved_category is None and isinstance(artifact, ModelArtifact):
        resolved_category = artifact.metadata.category
    if resolved_category is None:
        raise ValueError("Cần truyền category hoặc evaluation manifest.")
    if isinstance(artifact, ModelArtifact) and artifact.metadata.category != resolved_category:
        raise ValueError("Artifact không khớp category của evaluation.")

    detector = AnomalyDetector(model_dir=model_dir or "models", category=resolved_category)
    if manifest_obj is None:
        # Evaluation boundary duy nhất được phép đọc test/masks.
        manifest_obj = validate_locked_evaluation(data_dir=data_dir, category=resolved_category)
    if isinstance(manifest_obj, LockedEvaluationManifest) and manifest_obj.category != resolved_category:
        raise ValueError("Evaluation manifest không khớp category của artifact.")

    report_file = Path(output_report) if output_report else Path("reports") / resolved_category / "test_metrics.json"
    if report_file.exists() and report_file.stat().st_size > 0 and not reopen:
        raise FileExistsError(
            f"Report đã tồn tại tại '{report_file}'. Dùng reopen=True nếu thật sự cần ghi lại."
        )

    target_height, target_width = detector.preprocessing_config.image_size
    labels: list[int] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    names: list[str | None] = []
    area_ratios: list[float] = []

    print(f"\n[EVALUATION] Locked test cho '{resolved_category}' bằng release freeze...")
    for image_path, is_defect, mask_path, defect_type in manifest_obj.get_all_test_items():
        with Image.open(image_path) as image:
            score, heatmap = detector.score(image)
        if is_defect:
            if mask_path is None or not mask_path.exists():
                raise FileNotFoundError(f"Ảnh lỗi '{image_path}' thiếu ground-truth mask.")
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(
                    mask_image.convert("L").resize((target_width, target_height), Image.Resampling.NEAREST)
                ) > 0
        else:
            mask = np.zeros((target_height, target_width), dtype=bool)
        resized_map = np.asarray(
            Image.fromarray(heatmap.astype(np.float32)).resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            )
        )
        labels.append(is_defect)
        scores.append(float(score))
        masks.append(mask)
        maps.append(resized_map)
        names.append(defect_type)
        area_ratios.append(float(mask.mean()))

    y_arr = np.asarray(labels, dtype=int)
    score_arr = np.asarray(scores, dtype=np.float32)
    mask_arr = np.asarray(masks)
    map_arr = np.asarray(maps)
    result = calculate_3tier_metrics(
        y_true=y_arr,
        scores=score_arr,
        masks=mask_arr,
        maps=map_arr,
        auto_pass_threshold=detector.auto_pass_threshold,
    )
    result.update(
        {
            "category": resolved_category,
            "model_version": detector.model_version,
            "release_id": detector.release_id,
            "test_samples_total": len(labels),
            "test_defect_count": int(y_arr.sum()),
            "test_normal_count": int((y_arr == 0).sum()),
            "dataset_fingerprint": getattr(manifest_obj, "fingerprint", None),
            "defect_prevalence_in_benchmark": float(y_arr.mean()) if len(y_arr) else 0.0,
            "operational_note": "MVTec prevalence khong phai factory prevalence; khong suy ra throughput production.",
        }
    )

    defect_types = sorted({name for name in names if name is not None})
    result["per_defect_type"] = {
        defect_type: _slice_metrics(
            np.asarray([name == defect_type for name in names], dtype=bool),
            y_arr,
            score_arr,
            mask_arr,
            map_arr,
        )
        for defect_type in defect_types
    }
    area_buckets: list[str | None] = []
    for label, area in zip(names, area_ratios):
        if label is None:
            area_buckets.append(None)
            continue
        bucket = "small" if area <= 0.01 else "medium" if area <= 0.10 else "large"
        area_buckets.append(bucket)
    result["annotated_defect_area_slices"] = {
        bucket: {
            "n": int(sum(value == bucket for value in area_buckets)),
            "mean_area_ratio": float(
                np.mean([area_ratios[index] for index, value in enumerate(area_buckets) if value == bucket])
            ) if any(value == bucket for value in area_buckets) else 0.0,
            "metrics": _slice_metrics(
                np.asarray([value == bucket for value in area_buckets], dtype=bool),
                y_arr,
                score_arr,
                mask_arr,
                map_arr,
            ),
        }
        for bucket in ("small", "medium", "large")
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[OK] Image AUROC={result['detection']['image_auroc']}, "
        f"AUPRO@0.3={result['localization']['aupro_0.3']}, "
        f"Defect escape to AUTO_PASS={result['operational_decision']['defect_escape_after_auto_pass']:.4f}"
    )
    return result
