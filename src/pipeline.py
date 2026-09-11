"""Điều phối các pipeline của hệ thống MVTec AD.

Điều phối 4 pipeline chính:
1. DATA: kiểm tra dataset và tạo manifest
2. MODEL BUILDING: feature frozen, coreset, calibration và lưu artifact
3. EVALUATION: official test chỉ ghi report
4. SERVING: suy luận inspection và REST API
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from PIL import Image

from .config import TrainConfig
from .data.manifest import DatasetManifest, EvaluationManifest, NormalReferenceManifest
from .data.validation import (
    validate_evaluation,
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
    """Kiểm tra đầy đủ category và tạo manifest cho lệnh data."""
    print(f"\n[PIPELINE 1/4: DATA] Kiểm tra dataset cho category '{category}'...")
    manifest = validate_mvtec_category(
        data_dir=data_dir, category=category, check_image_integrity=check_integrity
    )
    if save_manifest:
        processed_dir = Path("data/processed") / category
        manifest_path = processed_dir / "manifest.json"
        manifest.save(manifest_path)
        print(f"  [ĐÃ LƯU] Manifest tại '{manifest_path}'")
    print(
        f"  [OK] Manifest hợp lệ: {manifest.total_train} ảnh train, "
        f"{manifest.total_test} ảnh test ({manifest.total_test_good} good, "
        f"{manifest.total_test_defect} defect thuộc {len(manifest.defect_types)} loại)."
    )
    if manifest.fingerprint:
        print(f"  [OK] Fingerprint dataset: {manifest.fingerprint[:16]}...")
    return manifest


def run_training_pipeline(
    manifest: DatasetManifest | NormalReferenceManifest | None = None,
    config: TrainConfig | None = None,
    models_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
) -> ModelArtifact:
    """Xây model category từ reference normal và calibration held-out."""
    cfg = config or TrainConfig()
    print(
        f"\n[PIPELINE 2/4: MODEL BUILDING] Xây artifact cho '{cfg.category}' "
        f"(backbone: {cfg.backbone})..."
    )
    artifact = train_patchcore(
        manifest=manifest,
        config=cfg,
        models_dir=models_dir,
        data_dir=data_dir,
    )
    print(
        f"  [OK] Đã xây model cho '{cfg.category}': "
        f"Memory bank size={artifact.coreset_info['size']}, "
        f"image threshold={artifact.thresholds.image_threshold:.4f}."
    )
    return artifact


def run_evaluation_pipeline(
    manifest: DatasetManifest | EvaluationManifest | None = None,
    artifact: ModelArtifact | str | Path | None = None,
    category: str = "bottle",
    model_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
    output_report: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Đánh giá official test, không chỉnh lại policy đã calibration."""
    target_cat = manifest.category if manifest else category
    print(
        f"\n[PIPELINE 3/4: EVALUATION] Đánh giá category '{target_cat}' "
        "(chỉ ghi report)..."
    )
    metrics = evaluate_category(
        category=target_cat,
        manifest=manifest,
        artifact=artifact,
        model_dir=model_dir,
        data_dir=data_dir,
        output_report=output_report,
        overwrite=overwrite,
    )
    return metrics


def run_serving_pipeline(
    category: str = "bottle",
    image: Image.Image | str | Path | None = None,
    model_dir: str | Path = "models",
) -> dict[str, Any]:
    """Kiểm tra chất lượng rồi chạy anomaly inspection trên một ảnh."""
    if image is None:
        raise ValueError("Phải cung cấp ảnh PIL hoặc đường dẫn ảnh để inspection.")

    detector = AnomalyDetector(model_dir=model_dir, category=category)

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
    overwrite: bool = False,
) -> dict[str, Any]:
    """Chạy Reference -> Training -> Official Evaluation theo đúng boundary."""
    print(
        f"\n{'=' * 70}\n"
        f" [MASTER PIPELINE] Chạy toàn bộ lifecycle cho '{category.upper()}'\n"
        f"{'=' * 70}"
    )
    # Tách hai ranh giới: reference được lấy trước; official test chỉ lấy sau train.
    reference_manifest = validate_reference_category(data_dir=data_dir, category=category)

    # 1. MODEL BUILDING PIPELINE (chỉ nhận normal reference)
    cfg = TrainConfig(category=category, backbone=backbone)
    artifact = run_training_pipeline(
        manifest=reference_manifest,
        config=cfg,
        models_dir=models_dir,
    )

    # 2. OFFICIAL EVALUATION PIPELINE (test/mask chỉ được đọc ở boundary này)
    evaluation_manifest = validate_evaluation(data_dir=data_dir, category=category)
    metrics = run_evaluation_pipeline(
        manifest=evaluation_manifest,
        artifact=artifact,
        category=category,
        model_dir=models_dir,
        output_report=output_report,
        overwrite=overwrite,
    )
    print(f"\n[MASTER PIPELINE] Đã hoàn tất lifecycle cho '{category}'.")
    return metrics


def main() -> None:
    """CLI điều phối từng pipeline hoặc toàn bộ lifecycle."""
    parser = argparse.ArgumentParser(description="Điều phối pipeline anomaly detection trên MVTec AD")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline cần chạy")

    # run: chạy toàn bộ lifecycle.
    run_parser = subparsers.add_parser("run", help="Chạy data -> train -> evaluate")
    run_parser.add_argument("--category", default="bottle", help="Tên category")
    run_parser.add_argument("--backbone", default="resnet18", help="Kiến trúc backbone CNN")
    run_parser.add_argument("--data-dir", default="data/raw", help="Thư mục dữ liệu raw")
    run_parser.add_argument("--models-dir", default="models", help="Thư mục lưu model")
    run_parser.add_argument("--output-report", default=None, help="Đường dẫn report đầu ra")
    run_parser.add_argument("--overwrite-report", action="store_true", help="Cho phép ghi lại report đã tồn tại")

    # data: kiểm tra dữ liệu.
    data_parser = subparsers.add_parser("data", help="Kiểm tra dữ liệu và tạo manifest")
    data_parser.add_argument("--category", default="bottle", help="Tên category")
    data_parser.add_argument("--data-dir", default="data/raw", help="Thư mục dữ liệu raw")

    # train: xây model category.
    train_parser = subparsers.add_parser("train", help="Xây model và calibration threshold")
    train_parser.add_argument("--category", default="bottle", help="Tên category")
    train_parser.add_argument("--backbone", default="resnet18", help="Kiến trúc backbone CNN")
    train_parser.add_argument("--models-dir", default="models", help="Thư mục lưu model")
    train_parser.add_argument("--data-dir", default="data/raw", help="Thư mục dữ liệu raw")

    # evaluate: đọc official test và chỉ ghi report.
    eval_parser = subparsers.add_parser("evaluate", help="Đánh giá official test, chỉ ghi report")
    eval_parser.add_argument("--category", default="bottle", help="Tên category")
    eval_parser.add_argument("--models-dir", default="models", help="Thư mục model")
    eval_parser.add_argument("--data-dir", default="data/raw", help="Thư mục dữ liệu raw")
    eval_parser.add_argument("--output-report", default=None, help="Đường dẫn report đầu ra")
    eval_parser.add_argument("--overwrite-report", action="store_true", help="Cho phép ghi lại report đã tồn tại")

    # serve: khởi động API.
    serve_parser = subparsers.add_parser("serve", help="Khởi động FastAPI REST server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Địa chỉ bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Cổng bind")

    args = parser.parse_args()

    if args.command == "run":
        run_end_to_end_pipeline(
            category=args.category,
            backbone=args.backbone,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            output_report=args.output_report,
            overwrite=args.overwrite_report,
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
            overwrite=args.overwrite_report,
        )
    elif args.command == "serve":
        import uvicorn
        from .api.app import app
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
