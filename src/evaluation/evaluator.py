"""Report-only evaluation on the MVTec AD test split.

Evaluation never recalibrates thresholds. The detector is loaded from a frozen
artifact and the test split is used only to produce detection, localization and
operational QC metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..data.dataset import find_category_root
from ..inference.detector import AnomalyDetector
from .metrics import calculate_3tier_metrics


def evaluate_category(
    category: str | None = None,
    model_dir: str | Path = "models",
    data_root: str | Path = "data/raw",
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    det = AnomalyDetector(model_dir=model_dir, category=category)
    resolved_category = det.category
    root = find_category_root(raw=data_root, category=resolved_category)
    evaluation_size = tuple(det.preprocessing_config.image_size)

    y_true: list[int] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    anomaly_maps: list[np.ndarray] = []

    test_dir = root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"MVTec test directory not found: '{test_dir}'.")

    print(f"[EVALUATION] category={resolved_category} artifact={det.artifact_dir}")
    print("[POLICY] test split is report-only; calibrated thresholds remain frozen")

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_defective = 0 if defect_dir.name == "good" else 1

        for image_path in sorted(defect_dir.glob("*.png")):
            with Image.open(image_path) as image:
                score, heatmap = det.score(image)

            y_true.append(is_defective)
            scores.append(score)

            if is_defective:
                mask_path = (
                    root / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                )
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing ground-truth mask for defect image '{image_path}': '{mask_path}'."
                    )
                with Image.open(mask_path) as mask_image:
                    mask = np.asarray(
                        mask_image.convert("L").resize(
                            (evaluation_size[1], evaluation_size[0]),
                            Image.Resampling.NEAREST,
                        )
                    ) > 0
            else:
                mask = np.zeros(evaluation_size, dtype=bool)

            resized_map = np.asarray(
                Image.fromarray(heatmap.astype(np.float32)).resize(
                    (evaluation_size[1], evaluation_size[0]),
                    Image.Resampling.BILINEAR,
                )
            )
            masks.append(mask)
            anomaly_maps.append(resized_map)

    if not y_true:
        raise RuntimeError(f"No test PNG images found under '{test_dir}'.")

    metrics_result = calculate_3tier_metrics(
        y_true=y_true,
        scores=scores,
        masks=np.asarray(masks),
        maps=np.asarray(anomaly_maps),
        threshold=det.threshold,
    )
    metrics_result.update(
        {
            "category": resolved_category,
            "model_version": det.model_version,
            "artifact_dir": str(det.artifact_dir),
            "evaluation_image_size": list(evaluation_size),
            "evaluation_policy": "report-only; no threshold tuning on test",
            "num_test_images": len(y_true),
        }
    )

    report_path = (
        Path(output_report)
        if output_report is not None
        else Path("reports") / resolved_category / "test_metrics.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(metrics_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    detection = metrics_result["detection"]
    localization = metrics_result["localization"]
    operational = metrics_result["operational_decision"]
    print(f"[RESULT] image_auroc={detection['image_auroc']:.4f}")
    print(f"[RESULT] pixel_auroc={localization['pixel_auroc']:.4f}")
    print(f"[RESULT] aupro_0.3={localization['aupro_0.3']:.4f}")
    print(f"[RESULT] defect_recall={operational['defect_recall']:.4f}")
    print(f"[REPORT] {report_path}")
    return metrics_result
