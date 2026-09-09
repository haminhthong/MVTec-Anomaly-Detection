"""Comprehensive unit tests for lifecycle hardening, anti-leakage, and artifact consistency.

Implements the 10 critical validation tests:
1. test_training_reads_only_train_good
2. test_evaluation_never_calls_calibration
3. test_requested_category_must_match_artifact
4. test_pretrained_setting_round_trip
5. test_preprocessing_config_round_trip
6. test_single_and_batch_scores_match
7. test_missing_mask_fails_fast
8. test_train_and_inference_feature_dimension_match
9. test_artifact_schema_version_supported
10. test_pipeline_manifest_passed_between_stages
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from src.config import TrainConfig
from src.data.manifest import DatasetManifest
from src.data.transforms import PreprocessingConfig
from src.data.validation import DatasetValidationError, validate_mvtec_category
from src.evaluation.evaluator import evaluate_category
from src.inference.detector import AnomalyDetector
from src.model.artifact_resolver import ModelNotFoundError, resolve_artifact_dir
from src.model.artifacts import ModelArtifact, ModelMetadata, ThresholdPolicy
from src.pipeline import (
    run_data_pipeline,
    run_evaluation_pipeline,
    run_training_pipeline,
)
from src.training.trainer import train_patchcore


@pytest.fixture
def dummy_mvtec_dataset(tmp_path: Path) -> tuple[Path, str]:
    """Create a minimal synthetic MVTec category dataset for testing."""
    raw_dir = tmp_path / "data" / "raw"
    category = "test_object"
    cat_dir = raw_dir / category

    train_good = cat_dir / "train" / "good"
    test_good = cat_dir / "test" / "good"
    test_defect = cat_dir / "test" / "crack"
    gt_dir = cat_dir / "ground_truth" / "crack"

    for d in (train_good, test_good, test_defect, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 25 training normal images (ensuring 25 * 0.2 = 5 >= min_calibration_samples)
    for i in range(25):
        Image.new("RGB", (32, 32), color=(i * 10, i * 5, 120)).save(train_good / f"{i:03d}.png")

    # 5 test normal images
    for i in range(5):
        Image.new("RGB", (32, 32), color=(i * 10, i * 5, 120)).save(test_good / f"{i:03d}.png")

    # 5 test defect images with masks
    for i in range(5):
        Image.new("RGB", (32, 32), color=(250, 20, 20)).save(test_defect / f"{i:03d}.png")
        Image.new("L", (32, 32), color=255).save(gt_dir / f"{i:03d}_mask.png")

    return raw_dir, category


def test_training_reads_only_train_good(dummy_mvtec_dataset: tuple[Path, str], tmp_path: Path) -> None:
    """1. test_training_reads_only_train_good: Ensure offline training touches only train/good images."""
    raw_dir, category = dummy_mvtec_dataset
    manifest = validate_mvtec_category(data_dir=raw_dir, category=category)
    models_dir = tmp_path / "models"

    accessed_paths: list[str] = []
    original_open = Image.open

    def track_open(fp, *args, **kwargs):
        p_str = str(fp).replace("\\", "/")
        if p_str.endswith(".png"):
            accessed_paths.append(p_str)
        return original_open(fp, *args, **kwargs)

    cfg = TrainConfig(
        category=category,
        batch_size=4,
        min_calibration_samples=5,
        coreset_fraction=0.1,
        min_coreset_size=5,
        max_coreset_size=20,
    )

    with patch("PIL.Image.open", side_effect=track_open):
        _ = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir)

    assert len(accessed_paths) > 0, "Images must have been read during training."
    for p in accessed_paths:
        assert "/test/" not in p, f"LEAKAGE: Training accessed test image '{p}'"
        assert "/ground_truth/" not in p, f"LEAKAGE: Training accessed ground_truth mask '{p}'"
        assert "/train/good/" in p, f"Unexpected path accessed: '{p}'"


def test_evaluation_never_calls_calibration(dummy_mvtec_dataset: tuple[Path, str], tmp_path: Path) -> None:
    """2. test_evaluation_never_calls_calibration: Ensure evaluation is report-only and never recalibrates."""
    raw_dir, category = dummy_mvtec_dataset
    manifest = validate_mvtec_category(data_dir=raw_dir, category=category)
    models_dir = tmp_path / "models"

    cfg = TrainConfig(
        category=category,
        batch_size=4,
        min_calibration_samples=5,
        coreset_fraction=0.1,
        min_coreset_size=5,
        max_coreset_size=20,
    )
    artifact = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir)

    # Patch calibrate_thresholds to raise an error if invoked
    with patch("src.training.calibration.calibrate_thresholds") as mock_calib:
        mock_calib.side_effect = RuntimeError("CALIBRATION CALLED DURING EVALUATION!")
        report = evaluate_category(
            manifest=manifest,
            artifact=artifact,
            model_dir=models_dir / category,
            output_report=tmp_path / "reports" / category / "evaluation.json",
        )

    mock_calib.assert_not_called()
    assert "detection" in report
    assert "operational_decision" in report


def test_requested_category_must_match_artifact(tmp_path: Path) -> None:
    """3. test_requested_category_must_match_artifact: Strict category isolation, no fallback."""
    models_dir = tmp_path / "models"
    bottle_dir = models_dir / "bottle"
    bottle_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "model": {"category": "bottle", "version": "1.0.0"},
        "category": "bottle",
        "thresholds": {"review": 1.5, "fail": 2.0, "pixel": 1.8},
        "coreset": {"size": 50},
        "preprocessing": {"image_size": [64, 64]},
    }
    (bottle_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
    np.save(bottle_dir / "memory_bank.npy", np.zeros((10, 384), dtype=np.float32))

    # Resolving exact category succeeds
    resolved = resolve_artifact_dir(model_root=models_dir, category="bottle")
    assert resolved == bottle_dir

    # Resolving non-existent category raises ModelNotFoundError (NO fallback to bottle)
    with pytest.raises(ModelNotFoundError):
        resolve_artifact_dir(model_root=models_dir, category="cable")

    # Mismatched config category in folder raises ModelNotFoundError
    mismatch_dir = models_dir / "mismatch"
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    (mismatch_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
    np.save(mismatch_dir / "memory_bank.npy", np.zeros((10, 384), dtype=np.float32))

    with pytest.raises(ModelNotFoundError, match="Category mismatch"):
        resolve_artifact_dir(model_root=models_dir, category="mismatch")


def test_pretrained_setting_round_trip(dummy_mvtec_dataset: tuple[Path, str], tmp_path: Path) -> None:
    """4. test_pretrained_setting_round_trip: Ensure pretrained=False and True persist faithfully."""
    raw_dir, category = dummy_mvtec_dataset
    manifest = validate_mvtec_category(data_dir=raw_dir, category=category)
    models_dir = tmp_path / "models"

    # A) Train with pretrained=False
    cfg_unpretrained = TrainConfig(
        category=category,
        pretrained=False,
        batch_size=4,
        min_calibration_samples=5,
        min_coreset_size=5,
        max_coreset_size=10,
    )
    artifact_unpretrained = train_patchcore(
        manifest=manifest, config=cfg_unpretrained, models_dir=models_dir / "unpretrained"
    )
    assert artifact_unpretrained.metadata.pretrained is False

    det_unpretrained = AnomalyDetector(
        model_dir=models_dir / "unpretrained" / category, category=category
    )
    assert det_unpretrained.net.pretrained is False
    assert det_unpretrained.artifact.metadata.pretrained is False

    # B) Train with pretrained=True
    cfg_pretrained = TrainConfig(
        category=category,
        pretrained=True,
        batch_size=4,
        min_calibration_samples=5,
        min_coreset_size=5,
        max_coreset_size=10,
    )
    artifact_pretrained = train_patchcore(
        manifest=manifest, config=cfg_pretrained, models_dir=models_dir / "pretrained"
    )
    assert artifact_pretrained.metadata.pretrained is True

    det_pretrained = AnomalyDetector(
        model_dir=models_dir / "pretrained" / category, category=category
    )
    assert det_pretrained.net.pretrained is True
    assert det_pretrained.artifact.metadata.pretrained is True


def test_preprocessing_config_round_trip(tmp_path: Path) -> None:
    """5. test_preprocessing_config_round_trip: Synchronized image size, mean, std."""
    prep = PreprocessingConfig(
        image_size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
    )
    meta = ModelMetadata(category="cable")
    policy = ThresholdPolicy(review_threshold=1.5, fail_threshold=2.0, pixel_threshold=1.8)
    artifact = ModelArtifact(
        metadata=meta,
        threshold_policy=policy,
        preprocessing=prep,
        coreset_info={"size": 10},
    )

    cat_dir = tmp_path / "cable"
    artifact.save(cat_dir)
    loaded = ModelArtifact.load(cat_dir)

    assert loaded.preprocessing.image_size == (256, 256)
    assert loaded.preprocessing.mean == (0.5, 0.5, 0.5)
    assert loaded.preprocessing.std == (0.2, 0.2, 0.2)


def test_single_and_batch_scores_match(dummy_mvtec_dataset: tuple[Path, str], tmp_path: Path) -> None:
    """6. test_single_and_batch_scores_match: Exact numerical parity between single and batch scoring."""
    raw_dir, category = dummy_mvtec_dataset
    manifest = validate_mvtec_category(data_dir=raw_dir, category=category)
    models_dir = tmp_path / "models"

    cfg = TrainConfig(
        category=category,
        batch_size=4,
        min_calibration_samples=5,
        min_coreset_size=5,
        max_coreset_size=20,
    )
    _ = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir)
    detector = AnomalyDetector(model_dir=models_dir / category, category=category)

    # 3 distinct PIL images
    images = [
        Image.new("RGB", (32, 32), color=(10, 20, 30)),
        Image.new("RGB", (32, 32), color=(200, 100, 50)),
        Image.new("RGB", (32, 32), color=(255, 0, 0)),
    ]

    single_results = [detector.inspect(im, include_overlay=False) for im in images]
    batch_results = detector.inspect_batch(images, include_overlay=False)

    for s_res, b_res in zip(single_results, batch_results, strict=True):
        assert s_res["decision"] == b_res["decision"]
        assert np.isclose(
            s_res["scores"]["anomaly_score"],
            b_res["scores"]["anomaly_score"],
            atol=1e-3,
        )


def test_missing_mask_fails_fast(dummy_mvtec_dataset: tuple[Path, str]) -> None:
    """7. test_missing_mask_fails_fast: Missing defect mask triggers fast validation error."""
    raw_dir, category = dummy_mvtec_dataset
    mask_to_delete = raw_dir / category / "ground_truth" / "crack" / "000_mask.png"
    assert mask_to_delete.exists()
    mask_to_delete.unlink()

    with pytest.raises(DatasetValidationError, match="ground-truth mask"):
        validate_mvtec_category(data_dir=raw_dir, category=category)


def test_train_and_inference_feature_dimension_match(dummy_mvtec_dataset: tuple[Path, str], tmp_path: Path) -> None:
    """8. test_train_and_inference_feature_dimension_match: Feature dim matches between train and detector."""
    raw_dir, category = dummy_mvtec_dataset
    manifest = validate_mvtec_category(data_dir=raw_dir, category=category)
    models_dir = tmp_path / "models"

    cfg = TrainConfig(
        category=category,
        backbone="resnet18",
        feature_layers=("layer2", "layer3"),
        min_calibration_samples=5,
        min_coreset_size=5,
        max_coreset_size=20,
    )
    artifact = train_patchcore(manifest=manifest, config=cfg, models_dir=models_dir)
    detector = AnomalyDetector(model_dir=models_dir / category, category=category)

    feature_dim = artifact.coreset_info["feature_dim"]
    assert feature_dim == detector.memory_bank.features.shape[1]

    # Test single extraction dim
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    x = detector.transform(img).unsqueeze(0).to(detector.dev)
    patches, _ = detector.net.extract_spatial_features(x)
    assert patches.shape[1] == feature_dim


def test_artifact_schema_version_supported(tmp_path: Path) -> None:
    """9. test_artifact_schema_version_supported: Schema versioning validation."""
    meta = ModelMetadata(
        model_version="1.0.0",
        pipeline_version="1.0",
        artifact_schema_version=4,
        category="bottle",
    )
    d = meta.to_dict()
    assert d["artifact_schema_version"] == 4

    restored = ModelMetadata.from_dict(d)
    assert restored.artifact_schema_version == 4


def test_pipeline_manifest_passed_between_stages(dummy_mvtec_dataset: tuple[Path, str], tmp_path: Path) -> None:
    """10. test_pipeline_manifest_passed_between_stages: Manifest propagates cleanly across all stages."""
    raw_dir, category = dummy_mvtec_dataset
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    # Stage 1: Data Pipeline
    manifest = run_data_pipeline(data_dir=raw_dir, category=category, save_manifest=False)
    assert isinstance(manifest, DatasetManifest)
    assert manifest.fingerprint is not None

    # Stage 2: Training Pipeline
    cfg = TrainConfig(
        category=category,
        min_calibration_samples=5,
        min_coreset_size=5,
        max_coreset_size=20,
    )
    artifact = run_training_pipeline(manifest=manifest, config=cfg, models_dir=models_dir)
    assert isinstance(artifact, ModelArtifact)
    assert artifact.metadata.dataset_fingerprint == manifest.fingerprint

    # Stage 3: Evaluation Pipeline
    report = run_evaluation_pipeline(
        manifest=manifest,
        artifact=artifact,
        model_dir=models_dir,
        output_report=reports_dir / category / "report.json",
    )
    assert isinstance(report, dict)
    assert report["dataset_fingerprint"] == manifest.fingerprint
    assert "operational_decision" in report
    assert "auto_pass_rate" in report["operational_decision"]
