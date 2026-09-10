"""Kiểm tra boundary dữ liệu và artifact runtime của pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from src.config import TrainConfig
from src.data.validation import validate_mvtec_category
from src.evaluation.evaluator import evaluate_category
from src.inference.detector import AnomalyDetector
from src.model.artifacts import ModelArtifact, ModelMetadata, Thresholds
from src.training.trainer import train_patchcore


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> tuple[Path, str]:
    """Tạo dataset nhỏ có train/good, test và mask."""
    raw_dir = tmp_path / "data" / "raw"
    category = "test_object"
    root = raw_dir / category
    train_good = root / "train" / "good"
    test_good = root / "test" / "good"
    test_defect = root / "test" / "crack"
    masks = root / "ground_truth" / "crack"
    for directory in (train_good, test_good, test_defect, masks):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(25):
        Image.new("RGB", (32, 32), color=(index * 10, index * 5, 120)).save(
            train_good / f"{index:03d}.png"
        )
    for index in range(5):
        Image.new("RGB", (32, 32), color=(index * 10, index * 5, 120)).save(
            test_good / f"{index:03d}.png"
        )
        Image.new("RGB", (32, 32), color=(250, 20, 20)).save(
            test_defect / f"{index:03d}.png"
        )
        Image.new("L", (32, 32), color=255).save(masks / f"{index:03d}_mask.png")
    return raw_dir, category


def _config(category: str) -> TrainConfig:
    return TrainConfig(
        category=category,
        batch_size=4,
        min_calibration_samples=5,
        image_quantile=0.98,
        pixel_quantile=0.98,
        coreset_size=20,
    )


def test_training_reads_only_train_good(dummy_dataset, tmp_path: Path) -> None:
    """Training không được mở test image hoặc ground-truth mask."""
    raw_dir, category = dummy_dataset
    manifest = validate_mvtec_category(data_dir=raw_dir, category=category)
    accessed: list[str] = []
    original_open = Image.open

    def track_open(file_path, *args, **kwargs):
        path = str(file_path).replace("\\", "/")
        if path.endswith(".png"):
            accessed.append(path)
        return original_open(file_path, *args, **kwargs)

    with patch("PIL.Image.open", side_effect=track_open):
        train_patchcore(manifest=manifest, config=_config(category), models_dir=tmp_path / "models")
    assert accessed
    assert all("/train/good/" in path for path in accessed)


def test_evaluation_does_not_recalibrate(dummy_dataset, tmp_path: Path) -> None:
    """Official evaluation chỉ dùng threshold đã lưu."""
    raw_dir, category = dummy_dataset
    manifest = validate_mvtec_category(data_dir=raw_dir, category=category)
    models_dir = tmp_path / "models"
    artifact = train_patchcore(manifest=manifest, config=_config(category), models_dir=models_dir)
    with patch("src.training.calibration.calibrate_thresholds") as calibration:
        calibration.side_effect = AssertionError("evaluation must not recalibrate")
        report = evaluate_category(
            manifest=manifest,
            artifact=artifact,
            model_dir=models_dir,
            output_report=tmp_path / "reports" / "evaluation.json",
        )
    assert report["image_threshold"] == artifact.thresholds.image_threshold


def test_detector_requires_explicit_category_artifact(tmp_path: Path) -> None:
    """Category chưa có model phải fail rõ ràng, không fallback."""
    model_dir = tmp_path / "models" / "bottle"
    model_dir.mkdir(parents=True)
    artifact = ModelArtifact(
        metadata=ModelMetadata(category="bottle"),
        thresholds=Thresholds(image_threshold=1.0, pixel_threshold=1.0),
        preprocessing=TrainConfig().preprocessing,
        coreset_info={"size": 2, "feature_dim": 384},
    )
    artifact.save(model_dir)
    import numpy as np

    np.save(model_dir / "memory_bank.npy", np.zeros((2, 384), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        AnomalyDetector(model_dir=tmp_path / "models", category="cable")


def test_artifact_round_trip_keeps_preprocessing(tmp_path: Path) -> None:
    """Metadata runtime giữ nguyên preprocessing và threshold."""
    artifact = ModelArtifact(
        metadata=ModelMetadata(category="bottle", pretrained=False),
        thresholds=Thresholds(image_threshold=2.0, pixel_threshold=1.5),
        preprocessing=TrainConfig().preprocessing,
        coreset_info={"size": 2, "feature_dim": 384},
    )
    artifact.save(tmp_path)
    loaded = ModelArtifact.load(tmp_path)
    assert loaded.metadata.pretrained is False
    assert loaded.thresholds == artifact.thresholds
    assert (tmp_path / "metadata.json").exists()
