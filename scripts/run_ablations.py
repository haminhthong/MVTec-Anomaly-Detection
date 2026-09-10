"""Chạy ablation không rò rỉ trên Dev normal và synthetic stress.

Official MVTec test chỉ được đọc bởi evaluate_category ở report cuối,
không được dùng để chọn backbone/layer/coreset hay policy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TrainConfig
from src.data.validation import validate_reference_category
from src.training.calibration import split_reference_dev_calibration
from src.training.trainer import train_patchcore
from src.inference.detector import AnomalyDetector


def _synthetic_stress(image: Image.Image, index: int) -> Image.Image:
    """Tạo biến đổi giả lập có kiểm soát, không đưa ảnh vào memory bank."""
    result = image.convert("RGB")
    if index % 4 == 0:
        return ImageEnhance.Brightness(result).enhance(1.35)
    if index % 4 == 1:
        return ImageEnhance.Contrast(result).enhance(1.7)
    if index % 4 == 2:
        return result.filter(ImageFilter.GaussianBlur(radius=2.0))
    array = np.asarray(result).copy()
    height, width = array.shape[:2]
    size = max(2, min(height, width) // 8)
    y = (index * 17) % max(1, height - size + 1)
    x = (index * 23) % max(1, width - size + 1)
    array[y : y + size, x : x + size] = 0
    return Image.fromarray(array)


def measure_inference_latency(detector: AnomalyDetector, sample_image_path: Path, num_trials: int = 20) -> float:
    """Đo độ trễ trung bình trên ảnh Dev, không đọc official test."""
    with Image.open(sample_image_path) as image:
        rgb = image.convert("RGB")
        for _ in range(3):
            detector.score(rgb)
        durations = []
        for _ in range(num_trials):
            start = time.perf_counter()
            detector.score(rgb)
            durations.append((time.perf_counter() - start) * 1000)
    return float(np.mean(durations))


def evaluate_dev_stress(detector: AnomalyDetector, dev_paths: list[Path]) -> dict[str, float]:
    """Đo báo động nhầm trên normal và độ nhạy trên synthetic stress."""
    normal_scores: list[float] = []
    stress_scores: list[float] = []
    for index, path in enumerate(dev_paths):
        with Image.open(path) as image:
            normal = image.convert("RGB")
            normal_scores.append(detector.score(normal)[0])
            stress_scores.append(detector.score(_synthetic_stress(normal, index))[0])
    threshold = detector.image_threshold
    return {
        "dev_normal_false_alarm_rate": float(np.mean(np.asarray(normal_scores) >= threshold)) if normal_scores else 0.0,
        "synthetic_stress_sensitivity": float(np.mean(np.asarray(stress_scores) >= threshold)) if stress_scores else 0.0,
    }


def _run_candidates(
    category: str,
    data_dir: str,
    output_dir: str,
    candidates: list[tuple[str, dict[str, object]]],
    filename: str,
) -> None:
    """Chạy một cấu hình trên Reference/Dev và ghi CSV."""
    reference_manifest = validate_reference_category(data_dir=data_dir, category=category)
    rows: list[dict[str, object]] = []
    for label, overrides in candidates:
        candidate_overrides = dict(overrides)
        candidate_overrides.setdefault("coreset_size", 1000)
        cfg = TrainConfig(
            category=category,
            **candidate_overrides,
        )
        artifact = train_patchcore(
            manifest=reference_manifest,
            config=cfg,
            models_dir=Path(output_dir) / "models",
        )
        detector = AnomalyDetector(
            model_dir=Path(output_dir) / "models",
            category=category,
        )
        _, dev_paths, _ = split_reference_dev_calibration(
            reference_manifest.train_good,
            dev_fraction=cfg.dev_fraction,
            calibration_fraction=cfg.calibration_fraction,
            seed=cfg.seed,
            min_calibration_samples=cfg.min_calibration_samples,
        )
        dev_metrics = evaluate_dev_stress(detector, dev_paths)
        rows.append(
            {
                "candidate": label,
                "memory_bank_size": artifact.coreset_info["size"],
                "feature_dim": artifact.coreset_info["feature_dim"],
                "latency_ms": round(measure_inference_latency(detector, dev_paths[0]), 2) if dev_paths else None,
                **{key: round(value, 4) for key, value in dev_metrics.items()},
            }
        )
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUCCESS] Dev-only ablation written to '{output_path}'")


def run_coreset_ablation(category: str = "bottle", data_dir: str = "data/raw", output_dir: str = "experiments") -> None:
    """So sánh K cụ thể, không dùng fraction/cap mơ hồ."""
    sizes = [250, 500, 1000, 2000, 4000]
    _run_candidates(
        category,
        data_dir,
        output_dir,
        [(str(size), {"coreset_size": size}) for size in sizes],
        "coreset_ablation.csv",
    )


def run_layers_ablation(category: str = "bottle", data_dir: str = "data/raw", output_dir: str = "experiments") -> None:
    """So sánh các feature layer trên Dev normal và synthetic stress."""
    candidates = [
        ("layer2", {"feature_layers": ("layer2",)}),
        ("layer3", {"feature_layers": ("layer3",)}),
        ("layer2+layer3", {"feature_layers": ("layer2", "layer3")}),
    ]
    _run_candidates(category, data_dir, output_dir, candidates, "layers_ablation.csv")


def run_backbone_ablation(category: str = "bottle", data_dir: str = "data/raw", output_dir: str = "experiments") -> None:
    """So sánh backbone trên Dev; official test chỉ đánh giá cuối."""
    candidates = [(name, {"backbone": name}) for name in ("resnet18", "resnet50")]
    _run_candidates(category, data_dir, output_dir, candidates, "backbone_ablation.csv")


def main() -> None:
    """CLI cho ablation chỉ dùng Dev."""
    parser = argparse.ArgumentParser(description="Leakage-safe Dev ablation cho PatchCore-style")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--experiment", choices=["all", "coreset", "layers", "backbone"], default="all")
    args = parser.parse_args()
    if args.experiment in ("all", "coreset"):
        run_coreset_ablation(args.category, args.data_dir, args.output_dir)
    if args.experiment in ("all", "layers"):
        run_layers_ablation(args.category, args.data_dir, args.output_dir)
    if args.experiment in ("all", "backbone"):
        run_backbone_ablation(args.category, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
