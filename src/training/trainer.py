"""Offline model building for a category-scoped PatchCore-style detector.

Only ``train/good`` is used for representation learning. A held-out subset of
normal training images is reserved for threshold calibration. Test images and
defect masks never participate in model building or calibration.
"""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data.dataset import ImageFolderDataset, find_category_root
from ..data.transforms import build_transform
from ..inference.localization import apply_heatmap_smoothing
from ..model.artifacts import category_artifact_dir
from ..model.coreset import greedy_coreset
from ..model.memory_bank import MemoryBank
from ..model.patch_embedding import FeatureExtractor
from .calibration import calibrate_thresholds, split_normal_paths


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_patchcore(config: TrainConfig | None = None) -> dict[str, Any]:
    cfg = config or TrainConfig()
    cfg.validate()
    set_seed(cfg.seed)

    # Stage 1 — data contract and leakage-safe split.
    root = find_category_root(raw=cfg.data_root, category=cfg.category)
    all_normal_paths = sorted((root / "train" / "good").glob("*.png"))
    if not all_normal_paths:
        raise FileNotFoundError(f"No PNG images found in '{root / 'train' / 'good'}'.")

    memory_paths, calibration_paths = split_normal_paths(
        all_normal_paths,
        calibration_fraction=cfg.calibration_fraction,
        seed=cfg.seed,
        min_calibration_samples=cfg.min_calibration_samples,
    )

    transform = build_transform(cfg.preprocessing)
    loader = DataLoader(
        ImageFolderDataset(memory_paths, transform=transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = FeatureExtractor().to(device)

    # Stage 2 — frozen ImageNet feature extraction from normal images.
    feature_batches: list[np.ndarray] = []
    for images, _ in loader:
        feature_batches.append(network(images.to(device)).cpu().numpy())
    if not feature_batches:
        raise RuntimeError("No normal features were extracted.")
    full_memory = np.concatenate(feature_batches, axis=0)

    # Stage 3 — coreset construction; selected vectors remain in original 384-D space.
    coreset_size = min(
        cfg.max_coreset_size,
        max(cfg.min_coreset_size, int(cfg.coreset_fraction * len(full_memory))),
        len(full_memory),
    )
    compact_memory = greedy_coreset(full_memory, coreset_size, seed=cfg.seed)
    memory_bank = MemoryBank(compact_memory)

    # Stage 4 — calibrate only on held-out normal images.
    calibration_scores: list[float] = []
    calibration_heatmaps: list[np.ndarray] = []
    calibration_dataset = ImageFolderDataset(calibration_paths, transform=transform)

    for idx in range(len(calibration_dataset)):
        tensor = calibration_dataset[idx][0].unsqueeze(0).to(device)
        patches, (height, width) = network.extract_spatial_features(tensor)
        distances, _ = memory_bank.kneighbors(patches.cpu().numpy())
        raw_heatmap = distances.reshape(height, width)
        smoothed = apply_heatmap_smoothing(raw_heatmap, sigma=cfg.smooth_sigma)
        calibration_scores.append(float(np.percentile(smoothed, 99)))
        calibration_heatmaps.append(smoothed)

    review_threshold, fail_threshold, pixel_threshold = calibrate_thresholds(
        normal_scores=calibration_scores,
        normal_heatmaps=calibration_heatmaps,
        review_quantile=cfg.review_quantile,
        fail_quantile=cfg.threshold_quantile,
        pixel_quantile=cfg.pixel_quantile,
    )

    # Stage 5 — immutable, category-scoped artifact contract.
    artifact_dir = category_artifact_dir(cfg.model_root, cfg.category)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    memory_bank.save(artifact_dir / "memory_bank.npy")

    payload: dict[str, Any] = {
        "schema_version": 4,
        "category": cfg.category,
        "version": "mvtec-resnet18-patchcore-v6",
        "artifact_dir": str(artifact_dir),
        "seed": cfg.seed,
        "device_used": device,
        "backbone": "resnet18-imagenet1k-v1",
        "feature_layers": ["layer2", "layer3"],
        "preprocessing": cfg.preprocessing.to_dict(),
        "data_policy": {
            "source": "MVTec AD",
            "representation_learning": "train/good only",
            "calibration": "held-out subset of train/good only",
            "test_usage": "report-only; never used for threshold selection",
        },
        "calibration": {
            "method": "held_out_normal_dual_calibration",
            "calibration_fraction": cfg.calibration_fraction,
            "min_calibration_samples": cfg.min_calibration_samples,
            "memory_images": len(memory_paths),
            "calibration_images": len(calibration_paths),
            "review_quantile": cfg.review_quantile,
            "threshold_quantile": cfg.threshold_quantile,
            "pixel_quantile": cfg.pixel_quantile,
        },
        "thresholds": {
            "review_threshold": review_threshold,
            "fail_threshold": fail_threshold,
            "image_threshold": fail_threshold,
            "pixel_threshold": pixel_threshold,
        },
        "threshold": fail_threshold,
        "review_threshold": review_threshold,
        "pixel_threshold": pixel_threshold,
        "coreset": {
            "fraction": cfg.coreset_fraction,
            "size": len(compact_memory),
            "full_memory_patches": len(full_memory),
        },
        "smooth_sigma": cfg.smooth_sigma,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    (artifact_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[SUCCESS] category={cfg.category} artifact={artifact_dir} "
        f"memory={len(memory_paths)} calibration={len(calibration_paths)} "
        f"coreset={compact_memory.shape} review={review_threshold:.4f} "
        f"fail={fail_threshold:.4f} pixel={pixel_threshold:.4f}"
    )
    return payload
