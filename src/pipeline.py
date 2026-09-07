"""Điều phối các pipeline của hệ thống MVTec AD.

Unifies the 4 canonical pipelines:
1. DATA: kiểm tra dataset và tạo manifest
2. MODEL BUILDING: feature frozen, coreset, calibration và lưu artifact
3. EVALUATION: locked test report-only
4. SERVING: suy luận inspection và REST API
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from PIL import Image

from .config import TrainConfig, parse_args
from .data.manifest import DatasetManifest, LockedEvaluationManifest, NormalReferenceManifest
from .data.validation import (
    validate_locked_evaluation,
    validate_mvtec_category,
    validate_reference_category,
)
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
    """Kiểm tra đầy đủ category và tạo combined manifest cho CLI data cũ."""
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
    manifest: DatasetManifest | NormalReferenceManifest | None = None,
    config: TrainConfig | None = None,
    models_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
) -> ModelArtifact:
    """Xây release model từ reference normal và calibration held-out."""
    cfg = config or TrainConfig()
    print(f"\n[PIPELINE 2/4: MODEL BUILDING] Building artifact for '{cfg.category}' (backbone: {cfg.backbone})...")
    artifact = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir, data_dir=data_dir)
    print(
        f"  [OK] Model release built for '{cfg.category}': "
        f"Memory bank size={artifact.coreset_info['size']}, "
        f"AUTO_PASS threshold={artifact.threshold_policy.auto_pass_threshold:.4f}."
    )
    return artifact


def run_evaluation_pipeline(
    manifest: DatasetManifest | LockedEvaluationManifest | None = None,
    artifact: ModelArtifact | str | Path | None = None,
    category: str = "bottle",
    model_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    """Đánh giá locked test theo chế độ report-only, không retune policy."""
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
    """Chạy quality gate và anomaly inspection trên một ảnh."""
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
    model_version: str = "1.0.0",
) -> dict[str, Any]:
    """Chạy Reference -> Training -> Locked Evaluation theo đúng boundary."""
    print(f"\n{'='*70}\n [MASTER PIPELINE] Executing end-to-end lifecycle for '{category.upper()}'\n{'='*70}")
    # Tách hai boundary: reference được resolve trước; locked test chỉ resolve sau train.
    reference_manifest = validate_reference_category(data_dir=data_dir, category=category)

    # 1. MODEL BUILDING PIPELINE (chỉ nhận normal reference)
    cfg = TrainConfig(category=category, backbone=backbone, model_version=model_version)
    artifact = run_training_pipeline(
        manifest=reference_manifest,
        config=cfg,
        models_dir=models_dir,
    )

    # 2. LOCKED EVALUATION PIPELINE (test/mask chỉ được đọc ở boundary này)
    locked_manifest = validate_locked_evaluation(data_dir=data_dir, category=category)
    metrics = run_evaluation_pipeline(
        manifest=locked_manifest,
        artifact=artifact,
        category=category,
        model_dir=models_dir,
        output_report=output_report,
    )
    print(f"\n[MASTER PIPELINE] Finished end-to-end execution for '{category}'.")
    return metrics


def main() -> None:
    """CLI điều phối từng pipeline hoặc toàn bộ lifecycle."""
    parser = argparse.ArgumentParser(description="MVTec AD Anomaly Detection Pipeline Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline to run")

    # run: chạy toàn bộ lifecycle.
    run_parser = subparsers.add_parser("run", help="Run end-to-end pipeline: data -> train -> evaluate")
    run_parser.add_argument("--category", default="bottle", help="Category name")
    run_parser.add_argument("--backbone", default="resnet18", help="Backbone CNN architecture")
    run_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    run_parser.add_argument("--models-dir", default="models", help="Models directory")
    run_parser.add_argument("--output-report", default=None, help="Report file path")
    run_parser.add_argument("--model-version", default="1.0.0", help="Immutable model release version")

    # data: kiểm tra dữ liệu.
    data_parser = subparsers.add_parser("data", help="Validate data and generate manifest")
    data_parser.add_argument("--category", default="bottle", help="Category name")
    data_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")

    # train: xây release model.
    train_parser = subparsers.add_parser("train", help="Build model and calibrate threshold policy")
    train_parser.add_argument("--category", default="bottle", help="Category name")
    train_parser.add_argument("--backbone", default="resnet18", help="Backbone CNN architecture")
    train_parser.add_argument("--models-dir", default="models", help="Models directory")
    train_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    train_parser.add_argument("--model-version", default="1.0.0", help="Immutable model release version")

    # evaluate: đọc locked test report-only.
    eval_parser = subparsers.add_parser("evaluate", help="Đánh giá locked test, report-only")
    eval_parser.add_argument("--category", default="bottle", help="Category name")
    eval_parser.add_argument("--models-dir", default="models", help="Models directory")
    eval_parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    eval_parser.add_argument("--output-report", default=None, help="Report file path")

    # serve: khởi động API.
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
            model_version=args.model_version,
        )
    elif args.command == "data":
        run_data_pipeline(data_dir=args.data_dir, category=args.category)
    elif args.command == "train":
        cfg = TrainConfig(category=args.category, backbone=args.backbone, model_version=args.model_version)
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
