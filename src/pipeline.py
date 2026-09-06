"""Master Pipeline Orchestrator for MVTec AD Anomaly Detection.

Unifies the 4 canonical pipelines:
1. DATA PIPELINE: Dataset validation and manifest creation
2. MODEL BUILDING PIPELINE: Offline representation learning, coreset, calibration, and artifact saving
3. EVALUATION PIPELINE: Frozen artifact 3-tier report-only evaluation
4. SERVING PIPELINE: Factory inspection inference and REST API serving
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from PIL import Image

from .config import TrainConfig, parse_args
from .data.validation import DatasetManifest, validate_mvtec_category
from .evaluation.evaluator import evaluate_category
from .inference.detector import AnomalyDetector
from .model.artifacts import ModelArtifact
from .training.trainer import train_patchcore


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_data_pipeline(
    data_dir: str | Path = "data/raw",
    category: str = "bottle",
    check_integrity: bool = False,
    save_manifest: bool = True,
) -> DatasetManifest:
    """Run Data Pipeline: Validate category directory, files, masks, and create DatasetManifest."""
    print(f"\n[PIPELINE 1/4: DATA] Validating dataset for category '{category}'...")
    manifest = validate_mvtec_category(
        data_dir=data_dir, category=category, check_image_integrity=check_integrity
    )
    if save_manifest:
        processed_dir = Path("data/processed") / category
        manifest_path = processed_dir / "manifest.json"
        manifest.save(manifest_path)
        print(f"  [SAVED] Manifest written to '{manifest_path}'")
    print(
        f"  [OK] Manifest validated: {manifest.total_train} train images, "
        f"{manifest.total_test} test images ({manifest.total_test_good} good, "
        f"{manifest.total_test_defect} defects across {len(manifest.defect_types)} types)."
    )
    if manifest.fingerprint:
        print(f"  [OK] Dataset Fingerprint: {manifest.fingerprint[:16]}...")
    return manifest


def run_training_pipeline(
    manifest: DatasetManifest | None = None,
    config: TrainConfig | None = None,
    models_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
) -> ModelArtifact:
    """Run Model Building Pipeline: Offline memory bank construction and held-out calibration."""
    cfg = config or TrainConfig()
    print(f"\n[PIPELINE 2/4: MODEL BUILDING] Building artifact for '{cfg.category}' (backbone: {cfg.backbone})...")
    artifact = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir, data_dir=data_dir)
    print(
        f"  [OK] Model built and saved to models/{cfg.category}/: "
        f"Memory bank size={artifact.coreset_info['size']}, "
        f"Fail Threshold={artifact.threshold_policy.fail_threshold:.4f}."
    )
    return artifact


def run_evaluation_pipeline(
    manifest: DatasetManifest | None = None,
    artifact: ModelArtifact | str | Path | None = None,
    category: str = "bottle",
    model_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    """Run Evaluation Pipeline: 3-tier report-only evaluation on test set."""
    target_cat = manifest.category if manifest else category
    print(f"\n[PIPELINE 3/4: EVALUATION] Evaluating category '{target_cat}' (REPORT-ONLY)...")
    metrics = evaluate_category(
        category=target_cat,
        manifest=manifest,
        artifact=artifact,
        model_dir=model_dir,
        data_dir=data_dir,
        output_report=output_report,
    )
    return metrics


def run_serving_pipeline(
    category: str = "bottle",
    image: Image.Image | str | Path | None = None,
    model_dir: str | Path = "models",
) -> dict[str, Any]:
    """Run Serving Pipeline: Perform inspection inference on an image."""
    detector = AnomalyDetector(model_dir=model_dir, category=category)
    if image is None:
        raise ValueError("Must provide an image (PIL Image or file path) for inspection.")

    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return detector.inspect(im.convert("RGB"))
    return detector.inspect(image)


def run_end_to_end_pipeline(
    category: str = "bottle",
    backbone: str = "resnet18",
    data_dir: str | Path = "data/raw",
    models_dir: str | Path = "models",
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    """Execute complete unified master lifecycle: Data -> Training -> Evaluation."""
    print(f"\n{'='*70}\n [MASTER PIPELINE] Executing end-to-end lifecycle for '{category.upper()}'\n{'='*70}")
    # 1. DATA PIPELINE
    manifest = run_data_pipeline(data_dir=data_dir, category=category)

    # 2. MODEL BUILDING PIPELINE (receives manifest, reads train_good ONLY)
    cfg = TrainConfig(category=category, backbone=backbone)
    artifact = run_training_pipeline(manifest=manifest, config=cfg, models_dir=models_dir)

    # 3. EVALUATION PIPELINE (receives manifest + frozen artifact, REPORT-ONLY)
    metrics = run_evaluation_pipeline(
        manifest=manifest,
        artifact=artifact,
        category=category,
        model_dir=models_dir,
        output_report=output_report,
    )
    print(f"\n[MASTER PIPELINE] Finished end-to-end execution for '{category}'.")
    return metrics


def main() -> None:
    """CLI orchestrator for executing individual or end-to-end pipelines."""
    parser = argparse.ArgumentParser(description="MVTec AD Anomaly Detection Pipeline Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline to run")

    # run (end-to-end master lifecycle)
    run_parser = subparsers.add_parser("run", help="Run end-to-end pipeline: data -> train -> evaluate")
    run_parser.add_argument("--category", default="bottle", help="Category name")
    run_parser.add_argument("--backbone", default="resnet18", help="Backbone CNN architecture")
    run_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    run_parser.add_argument("--models-dir", default="models", help="Models directory")
    run_parser.add_argument("--output-report", default=None, help="Report file path")

    # data
    data_parser = subparsers.add_parser("data", help="Validate data and generate manifest")
    data_parser.add_argument("--category", default="bottle", help="Category name")
    data_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")

    # train
    train_parser = subparsers.add_parser("train", help="Build model and calibrate threshold policy")
    train_parser.add_argument("--category", default="bottle", help="Category name")
    train_parser.add_argument("--backbone", default="resnet18", help="Backbone CNN architecture")
    train_parser.add_argument("--models-dir", default="models", help="Models directory")
    train_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run 3-tier evaluation on test split")
    eval_parser.add_argument("--category", default="bottle", help="Category name")
    eval_parser.add_argument("--models-dir", default="models", help="Models directory")
    eval_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    eval_parser.add_argument("--output-report", default=None, help="Report file path")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host binding")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port binding")

    args = parser.parse_args()

    if args.command == "run":
        run_end_to_end_pipeline(
            category=args.category,
            backbone=args.backbone,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            output_report=args.output_report,
        )
    elif args.command == "data":
        run_data_pipeline(data_dir=args.data_dir, category=args.category)
    elif args.command == "train":
        cfg = TrainConfig(category=args.category, backbone=args.backbone)
        run_training_pipeline(config=cfg, models_dir=args.models_dir, data_dir=args.data_dir)
    elif args.command == "evaluate":
        run_evaluation_pipeline(
            category=args.category,
            model_dir=args.models_dir,
            data_dir=args.data_dir,
            output_report=args.output_report,
        )
    elif args.command == "serve":
        import uvicorn
        from .api.app import app
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
