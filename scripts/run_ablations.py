"""Ablation studies script for PatchCore Anomaly Detection.

Executes 3 core experiments:
1. Experiment A — Coreset Fraction: 1%, 5%, 10%
2. Experiment B — Feature Extraction Layers: layer2, layer3, layer2 + layer3
3. Experiment C — Backbone: ResNet18 vs ResNet50

Saves results to experiments/ directory as CSV files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TrainConfig
from src.data.validation import validate_mvtec_category
from src.evaluation.evaluator import evaluate_category
from src.inference.detector import AnomalyDetector
from src.training.trainer import train_patchcore


def measure_inference_latency(detector: AnomalyDetector, sample_image_path: Path, num_trials: int = 20) -> float:
    """Measure mean inference latency in milliseconds."""
    with Image.open(sample_image_path) as img:
        rgb_img = img.convert("RGB")
        # Warmup
        for _ in range(3):
            _ = detector.score(rgb_img)

        latencies = []
        for _ in range(num_trials):
            start = time.perf_counter()
            _ = detector.score(rgb_img)
            latencies.append((time.perf_counter() - start) * 1000)
    return float(sum(latencies) / len(latencies))


def run_coreset_ablation(
    category: str = "bottle",
    data_dir: str = "data/raw",
    output_dir: str = "experiments",
) -> None:
    """Experiment A: Coreset fraction evaluation (1%, 5%, 10%)."""
    print(f"\n[ABLATION A: CORESET FRACTION] Running for category '{category}'...")
    fractions = [0.01, 0.05, 0.10]
    results = []

    manifest = validate_mvtec_category(data_dir=data_dir, category=category)
    sample_img = manifest.test_good[0] if manifest.test_good else manifest.train_good[0]

    for frac in fractions:
        print(f"  - Testing coreset fraction: {frac * 100:.0f}%")
        models_dir = Path(output_dir) / "models" / f"coreset_{int(frac * 100)}"
        cfg = TrainConfig(
            category=category,
            coreset_fraction=frac,
            min_coreset_size=20,
            max_coreset_size=5000,
        )
        artifact = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir)
        det = AnomalyDetector(model_dir=models_dir / category, category=category)
        latency_ms = measure_inference_latency(det, sample_img)

        eval_res = evaluate_category(manifest=manifest, model_dir=models_dir / category)
        det_res = eval_res["detection"]
        loc_res = eval_res["localization"]

        results.append({
            "coreset_fraction": frac,
            "memory_bank_size": artifact.coreset_info["size"],
            "image_auroc": round(det_res["image_auroc"], 4),
            "pixel_auroc": round(loc_res["pixel_auroc"], 4),
            "aupro_0.3": round(loc_res["aupro_0.3"], 4),
            "latency_ms": round(latency_ms, 2),
        })

    out_csv = Path(output_dir) / "coreset_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[SUCCESS] Coreset ablation written to '{out_csv}'")
    print(pd.DataFrame(results).to_string(index=False))


def run_layers_ablation(
    category: str = "bottle",
    data_dir: str = "data/raw",
    output_dir: str = "experiments",
) -> None:
    """Experiment B: Feature layers ablation (layer2, layer3, layer2 + layer3)."""
    print(f"\n[ABLATION B: FEATURE LAYERS] Running for category '{category}'...")
    layer_configs = [
        ("layer2", ("layer2",)),
        ("layer3", ("layer3",)),
        ("layer2+layer3", ("layer2", "layer3")),
    ]
    results = []

    manifest = validate_mvtec_category(data_dir=data_dir, category=category)
    sample_img = manifest.test_good[0] if manifest.test_good else manifest.train_good[0]

    for label, layers in layer_configs:
        print(f"  - Testing layer configuration: {label}")
        models_dir = Path(output_dir) / "models" / f"layers_{label}"
        cfg = TrainConfig(
            category=category,
            feature_layers=layers,
        )
        artifact = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir)
        det = AnomalyDetector(model_dir=models_dir / category, category=category)
        latency_ms = measure_inference_latency(det, sample_img)

        eval_res = evaluate_category(manifest=manifest, model_dir=models_dir / category)
        det_res = eval_res["detection"]
        loc_res = eval_res["localization"]

        results.append({
            "layers": label,
            "feature_dim": artifact.coreset_info["feature_dim"],
            "image_auroc": round(det_res["image_auroc"], 4),
            "pixel_auroc": round(loc_res["pixel_auroc"], 4),
            "aupro_0.3": round(loc_res["aupro_0.3"], 4),
            "latency_ms": round(latency_ms, 2),
        })

    out_csv = Path(output_dir) / "layers_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[SUCCESS] Layers ablation written to '{out_csv}'")
    print(pd.DataFrame(results).to_string(index=False))


def run_backbone_ablation(
    category: str = "bottle",
    data_dir: str = "data/raw",
    output_dir: str = "experiments",
) -> None:
    """Experiment C: Backbone ablation (ResNet18 vs ResNet50)."""
    print(f"\n[ABLATION C: BACKBONE] Running for category '{category}'...")
    backbones = ["resnet18", "resnet50"]
    results = []

    manifest = validate_mvtec_category(data_dir=data_dir, category=category)
    sample_img = manifest.test_good[0] if manifest.test_good else manifest.train_good[0]

    for b_name in backbones:
        print(f"  - Testing backbone: {b_name}")
        models_dir = Path(output_dir) / "models" / f"backbone_{b_name}"
        cfg = TrainConfig(
            category=category,
            backbone=b_name,
        )
        artifact = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir)
        det = AnomalyDetector(model_dir=models_dir / category, category=category)
        latency_ms = measure_inference_latency(det, sample_img)

        eval_res = evaluate_category(manifest=manifest, model_dir=models_dir / category)
        det_res = eval_res["detection"]
        loc_res = eval_res["localization"]

        results.append({
            "backbone": b_name,
            "feature_dim": artifact.coreset_info["feature_dim"],
            "image_auroc": round(det_res["image_auroc"], 4),
            "pixel_auroc": round(loc_res["pixel_auroc"], 4),
            "aupro_0.3": round(loc_res["aupro_0.3"], 4),
            "latency_ms": round(latency_ms, 2),
        })

    out_csv = Path(output_dir) / "backbone_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[SUCCESS] Backbone ablation written to '{out_csv}'")
    print(pd.DataFrame(results).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments for PatchCore AD")
    parser.add_argument("--category", default="bottle", help="Category to run ablations on")
    parser.add_argument("--data-dir", default="data/raw", help="Raw dataset directory")
    parser.add_argument("--output-dir", default="experiments", help="Output experiments directory")
    parser.add_argument(
        "--experiment",
        choices=["all", "coreset", "layers", "backbone"],
        default="all",
        help="Which ablation experiment to run",
    )
    args = parser.parse_args()

    if args.experiment in ("all", "coreset"):
        run_coreset_ablation(args.category, args.data_dir, args.output_dir)
    if args.experiment in ("all", "layers"):
        run_layers_ablation(args.category, args.data_dir, args.output_dir)
    if args.experiment in ("all", "backbone"):
        run_backbone_ablation(args.category, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
