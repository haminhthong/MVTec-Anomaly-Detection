"""Offline Model Building Pipeline for PatchCore-style Anomaly Detection.

Pipeline stages:
1. DATA VALIDATION: Validate category directory, load DatasetManifest, split train/good into Memory and Calibration sets.
2. NORMAL REPRESENTATION LEARNING: Extract intermediate feature maps using frozen FeatureExtractor.
3. CORESET MEMORY BANK: Johnson-Lindenstrauss projection -> Greedy K-Center index selection -> compact memory bank.
4. CALIBRATION: Establish ThresholdPolicy (P95 review, P99 fail, P99 pixel) on held-out normal images.
5. ARTIFACT PERSISTENCE: Save category-scoped artifacts (config.json, memory_bank.npy, split_manifest.json).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data.dataset import ImageFolderDataset
from ..data.transforms import build_transform
from ..data.manifest import DatasetManifest
from ..data.validation import validate_mvtec_category
from ..inference.localization import apply_heatmap_smoothing
from ..model.artifacts import (
    ModelArtifact,
    ModelMetadata,
    SplitManifest,
    ThresholdPolicy,
)
from ..model.coreset import select_coreset_indices
from ..model.feature_extractor import FeatureExtractor
from ..model.memory_bank import MemoryBank
from .calibration import calibrate_thresholds, split_normal_paths


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_patchcore(
    manifest: DatasetManifest | TrainConfig | None = None,
    config: TrainConfig | None = None,
    models_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
) -> ModelArtifact:
    """Run complete offline model building pipeline for a category.

    Args:
        manifest: Pre-validated DatasetManifest (or TrainConfig for backward compatibility).
        config: TrainConfig containing parameters. If None, default TrainConfig is used.
        models_dir: Base directory for storing category-scoped model artifacts.
        data_dir: Base directory containing raw MVTec AD datasets (used if manifest is None).

    Returns:
        ModelArtifact: Saved model artifact container.
    """
    if isinstance(manifest, TrainConfig):
        cfg = manifest
        manifest_obj: DatasetManifest | None = None
    else:
        cfg = config or TrainConfig()
        manifest_obj = manifest

    cfg.validate()
    set_seed(cfg.seed)

    # 1. DATA VALIDATION & SPLIT
    if manifest_obj is None:
        manifest_obj = validate_mvtec_category(data_dir=data_dir, category=cfg.category)

    # ANTI-LEAKAGE: strictly consume only manifest.train_good
    memory_paths, calibration_paths = split_normal_paths(
        manifest_obj.train_good,
        calibration_fraction=cfg.calibration_fraction,
        seed=cfg.seed,
        min_calibration_samples=cfg.min_calibration_samples,
    )

    split_manifest = SplitManifest(
        seed=cfg.seed,
        calibration_fraction=cfg.calibration_fraction,
        memory_count=len(memory_paths),
        calibration_count=len(calibration_paths),
        memory_files=[p.name for p in memory_paths],
        calibration_files=[p.name for p in calibration_paths],
    )

    # 2. NORMAL REPRESENTATION LEARNING
    transform = build_transform(cfg.preprocessing)
    loader = DataLoader(
        ImageFolderDataset(memory_paths, transform=transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = FeatureExtractor(
        backbone=cfg.backbone,
        layers=cfg.feature_layers,
        pretrained=cfg.pretrained,
        weights=getattr(cfg, "weights", None),
    ).to(device)

    batches: list[np.ndarray] = []
    for images, _ in loader:
        with torch.inference_mode():
            batch_patches = network(images.to(device)).cpu().numpy()
            batches.append(batch_patches)

    full_memory = np.concatenate(batches, axis=0)

    # 3. MEMORY BANK CONSTRUCTION (Coreset selection)
    coreset_size = min(
        cfg.max_coreset_size,
        max(cfg.min_coreset_size, int(cfg.coreset_fraction * len(full_memory))),
        len(full_memory),
    )
    selected_indices = select_coreset_indices(
        features=full_memory, size=coreset_size, seed=cfg.seed
    )
    compact_memory = full_memory[selected_indices]
    memory_bank = MemoryBank(compact_memory)

    # 4. CALIBRATION (Held-out Normal)
    calibration_scores: list[float] = []
    calibration_heatmaps: list[np.ndarray] = []

    for path in calibration_paths:
        tensor = ImageFolderDataset([path], transform=transform)[0][0].unsqueeze(0).to(device)
        with torch.inference_mode():
            patches, (height, width) = network.extract_spatial_features(tensor)
            distances, _ = memory_bank.kneighbors(patches.cpu().numpy())
            raw_heat = distances.reshape(height, width)
            smoothed = apply_heatmap_smoothing(raw_heat, sigma=cfg.smooth_sigma)
            calibration_scores.append(float(np.percentile(smoothed, 99)))
            calibration_heatmaps.append(smoothed)

    threshold_policy = calibrate_thresholds(
        normal_scores=calibration_scores,
        normal_heatmaps=calibration_heatmaps,
        review_quantile=cfg.review_quantile,
        fail_quantile=cfg.threshold_quantile,
        pixel_quantile=cfg.pixel_quantile,
    )

    # 5. ARTIFACT PERSISTENCE (Strict category isolation)
    base_models = Path(models_dir)
    category_dir = base_models / cfg.category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Save memory bank array
    memory_bank.save(category_dir / "memory_bank.npy")

    # Save split manifest
    split_manifest.save(category_dir / "split_manifest.json")

    # Assemble metadata and save ModelArtifact
    weights_name = getattr(network, "weights_name", None) or getattr(cfg, "weights", None)
    metadata = ModelMetadata(
        model_version="1.0.0",
        pipeline_version="1.0",
        artifact_schema_version=4,
        category=cfg.category,
        backbone=cfg.backbone,
        weights=weights_name,
        pretrained=cfg.pretrained,
        feature_layers=list(cfg.feature_layers),
        created_at=datetime.now(timezone.utc).isoformat(),
        device_used=device,
        dataset_fingerprint=manifest_obj.fingerprint,
    )

    artifact = ModelArtifact(
        metadata=metadata,
        threshold_policy=threshold_policy,
        preprocessing=cfg.preprocessing,
        coreset_info={
            "fraction": cfg.coreset_fraction,
            "projection_dim": 64,
            "algorithm": "greedy_k_center",
            "size": len(compact_memory),
            "full_memory_patches": len(full_memory),
            "feature_dim": compact_memory.shape[1],
        },
        scoring={
            "method": "percentile",
            "percentile": getattr(cfg, "scoring_percentile", 99.0),
            "smooth_sigma": cfg.smooth_sigma,
        },
        calibration={
            "source": "held_out_train_good",
            "review_quantile": cfg.review_quantile,
            "fail_quantile": cfg.threshold_quantile,
            "pixel_quantile": cfg.pixel_quantile,
            "samples": len(calibration_paths),
        },
        smooth_sigma=cfg.smooth_sigma,
        dataset_fingerprint=manifest_obj.fingerprint,
    )
    artifact.save(category_dir)

    print(
        f"[SUCCESS] Category '{cfg.category}': Memory Bank={compact_memory.shape} (from {len(full_memory)} patches), "
        f"Review Threshold={threshold_policy.review_threshold:.4f} (P95), "
        f"Fail Threshold={threshold_policy.fail_threshold:.4f} (P99), "
        f"Pixel Threshold={threshold_policy.pixel_threshold:.4f} | "
        f"Calibration={len(calibration_paths)} normal images"
    )
    return artifact
