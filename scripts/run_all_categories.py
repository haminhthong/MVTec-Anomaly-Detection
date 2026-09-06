"""Multi-category benchmark runner and aggregator for MVTec AD.

Executes data validation, offline model building, and report-only evaluation
across all available categories, then generates/updates reports/benchmark.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TrainConfig
from src.pipeline import run_end_to_end_pipeline

ALL_MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def discover_available_categories(data_dir: str | Path = "data/raw") -> list[str]:
    """Scan data directory to identify downloaded categories with valid structure."""
    raw_path = Path(data_dir)
    found: list[str] = []
    for cat in ALL_MVTEC_CATEGORIES:
        cand1 = raw_path / cat
        cand2 = raw_path / "mvtec_anomaly_detection" / cat
        if (cand1.exists() and (cand1 / "train").exists()) or (
            cand2.exists() and (cand2 / "train").exists()
        ):
            found.append(cat)
    return found


def run_pipeline_for_category(
    category: str,
    data_dir: str | Path = "data/raw",
    models_dir: str | Path = "models",
    backbone: str = "resnet18",
) -> dict[str, Any]:
    """Execute unified lifecycle (data -> train -> evaluate) for a single category."""
    report_file = Path("reports") / category / "test_metrics.json"
    metrics = run_end_to_end_pipeline(
        category=category,
        backbone=backbone,
        data_dir=data_dir,
        models_dir=models_dir,
        output_report=report_file,
    )
    return metrics


def aggregate_benchmark_csv(
    results: list[dict[str, Any]], output_csv: str | Path = "reports/benchmark.csv"
) -> None:
    """Save category metrics and compute macro-average row."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for m in results:
        det = m["detection"]
        loc = m["localization"]
        op = m["operational_decision"]
        rows.append({
            "category": m["category"],
            "test_samples": m.get("test_samples_total", 0),
            "image_auroc": round(det["image_auroc"], 4),
            "image_ap": round(det["image_average_precision"], 4),
            "pixel_auroc": round(loc["pixel_auroc"], 4),
            "pixel_ap": round(loc["pixel_average_precision"], 4),
            "aupro_0.3": round(loc["aupro_0.3"], 4),
            "threshold": round(op["threshold"], 4),
            "review_threshold": round(op.get("review_threshold", 0.0), 4),
            "accuracy": round(op["accuracy"], 4),
            "defect_recall": round(op["defect_recall"], 4),
            "specificity": round(op["specificity"], 4),
            "f1_score": round(op["f1_score"], 4),
            "auto_pass_rate": round(op.get("auto_pass_rate", 0.0), 4),
            "manual_review_rate": round(op.get("manual_review_rate", 0.0), 4),
            "auto_fail_rate": round(op.get("auto_fail_rate", 0.0), 4),
            "defect_escape_rate": round(op.get("defect_escape_after_auto_pass", 0.0), 4),
            "status": "evaluated",
        })

    if not rows:
        print("No results to write to CSV.")
        return

    # Compute macro-average
    avg_img_auroc = sum(r["image_auroc"] for r in rows) / len(rows)
    avg_pix_auroc = sum(r["pixel_auroc"] for r in rows) / len(rows)
    avg_pix_ap = sum(r["pixel_ap"] for r in rows) / len(rows)
    avg_aupro = sum(r["aupro_0.3"] for r in rows) / len(rows)
    avg_f1 = sum(r["f1_score"] for r in rows) / len(rows)
    total_test = sum(r["test_samples"] for r in rows)

    mean_row = {
        "category": f"mean_{len(rows)}_categories",
        "test_samples": total_test,
        "image_auroc": round(avg_img_auroc, 4),
        "image_ap": round(sum(r["image_ap"] for r in rows) / len(rows), 4),
        "pixel_auroc": round(avg_pix_auroc, 4),
        "pixel_ap": round(avg_pix_ap, 4),
        "aupro_0.3": round(avg_aupro, 4),
        "threshold": "-",
        "review_threshold": "-",
        "accuracy": round(sum(r["accuracy"] for r in rows) / len(rows), 4),
        "defect_recall": round(sum(r["defect_recall"] for r in rows) / len(rows), 4),
        "specificity": round(sum(r["specificity"] for r in rows) / len(rows), 4),
        "f1_score": round(avg_f1, 4),
        "auto_pass_rate": round(sum(r["auto_pass_rate"] for r in rows) / len(rows), 4),
        "manual_review_rate": round(sum(r["manual_review_rate"] for r in rows) / len(rows), 4),
        "auto_fail_rate": round(sum(r["auto_fail_rate"] for r in rows) / len(rows), 4),
        "defect_escape_rate": round(sum(r["defect_escape_rate"] for r in rows) / len(rows), 4),
        "status": "macro_average",
    }
    rows.append(mean_row)

    fieldnames = list(rows[0].keys())
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[BENCHMARK] Saved aggregated metrics to '{out_path}'.")
    df = pd.DataFrame(rows)
    print(df[["category", "test_samples", "image_auroc", "pixel_auroc", "aupro_0.3", "f1_score"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run pipeline on all available MVTec categories")
    parser.add_argument("--categories", nargs="+", default=None, help="Specific categories to run")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw datasets")
    parser.add_argument("--models-dir", default="models", help="Path to models directory")
    parser.add_argument("--output-csv", default="reports/benchmark.csv", help="Output benchmark CSV path")
    parser.add_argument("--backbone", default="resnet18", help="Backbone CNN architecture")
    args = parser.parse_args()

    if args.categories:
        target_categories = args.categories
    else:
        target_categories = discover_available_categories(args.data_dir)
        if not target_categories:
            print(f"No categories found in '{args.data_dir}'. Falling back to 'bottle'.")
            target_categories = ["bottle"]

    print(f"Categories to process: {target_categories}")
    all_metrics: list[dict[str, Any]] = []
    for cat in target_categories:
        try:
            m = run_pipeline_for_category(
                category=cat,
                data_dir=args.data_dir,
                models_dir=args.models_dir,
                backbone=args.backbone,
            )
            all_metrics.append(m)
        except Exception as exc:
            print(f"[ERROR] Failed processing category '{cat}': {exc}")

    aggregate_benchmark_csv(all_metrics, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
