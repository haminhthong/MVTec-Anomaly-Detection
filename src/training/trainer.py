"""Xây model PatchCore-style chỉ từ normal reference, không đọc locked test."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data.dataset import ImageFolderDataset
from ..data.manifest import DatasetManifest, NormalReferenceManifest
from ..data.transforms import build_transform
from ..data.validation import validate_reference_category
from ..inference.localization import apply_heatmap_smoothing
from ..model.artifacts import (
    ModelArtifact,
    ModelMetadata,
    SplitManifest,
    write_integrity_manifest,
)
from ..model.coreset import select_coreset_indices
from ..model.feature_extractor import FeatureExtractor
from ..model.memory_bank import MemoryBank
from .calibration import calibrate_thresholds, split_reference_dev_calibration


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho các thư viện để split và coreset tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _relative_to_root(root: Path, path: Path) -> str:
    """Lưu path split tương đối thay vì chỉ lưu basename dễ trùng."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _update_production_pointer(models_root: Path, category: str, release_id: str, line_id: str | None) -> None:
    """Cập nhật pointer production; release directory không bị overwrite."""
    pointer_path = models_root / "production.json"
    if pointer_path.exists():
        try:
            data = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"production.json hiện tại không hợp lệ: {exc}") from exc
    else:
        data = {"categories": {}, "lines": {}}
    if not isinstance(data, dict):
        raise ValueError("production.json phải là một JSON object.")
    categories = data.setdefault("categories", {})
    lines = data.setdefault("lines", {})
    if not isinstance(categories, dict) or not isinstance(lines, dict):
        raise ValueError("production.json phải chứa object 'categories' và 'lines'.")
    categories[category] = f"releases/{release_id}"
    if line_id:
        lines[line_id] = {"category": category, "release_id": release_id}
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = pointer_path.with_name(f"{pointer_path.name}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(pointer_path)


def _save_legacy_category_alias(release_dir: Path, models_root: Path, category: str) -> None:
    """Tạo alias category để CLI cũ vẫn chạy; release thật nằm trong releases/."""
    alias_dir = models_root / category
    alias_dir.mkdir(parents=True, exist_ok=True)
    for name in ("memory_bank.npy", "config.json", "split_manifest.json", "manifest.json", "reference_manifest.json"):
        source = release_dir / name
        if source.exists():
            target = alias_dir / name
            # Sao chép nguyên byte để manifest integrity vẫn khớp release.
            target.write_bytes(source.read_bytes())


def train_patchcore(
    manifest: DatasetManifest | NormalReferenceManifest | TrainConfig | None = None,
    config: TrainConfig | None = None,
    models_dir: str | Path = "models",
    data_dir: str | Path = "data/raw",
) -> ModelArtifact:
    """Xây memory bank, calibration policy và immutable release.

    Nếu không truyền manifest, trainer chỉ gọi validator reference-only. Một
    combined DatasetManifest cũ vẫn được chấp nhận nhưng trainer chỉ lấy
    ``train_good`` từ object đó.
    """
    if isinstance(manifest, TrainConfig):
        cfg = manifest
        manifest_obj: DatasetManifest | NormalReferenceManifest | None = None
    else:
        cfg = config or TrainConfig()
        manifest_obj = manifest
    cfg.validate()
    set_seed(cfg.seed)

    if manifest_obj is None:
        manifest_obj = validate_reference_category(data_dir=data_dir, category=cfg.category)
    if manifest_obj.category != cfg.category:
        raise ValueError(
            f"Manifest category '{manifest_obj.category}' không khớp config '{cfg.category}'."
        )

    # Nếu caller truyền combined manifest cũ, chỉ dựng lại reference manifest
    # từ train/good để artifact training không mang theo test/mask provenance.
    reference_manifest = (
        manifest_obj
        if isinstance(manifest_obj, NormalReferenceManifest)
        else NormalReferenceManifest(
            category=manifest_obj.category,
            root_path=manifest_obj.root_path,
            train_good=list(manifest_obj.train_good),
        )
    )
    # Giữ fingerprint của manifest caller truyền vào để các stage liên kết
    # cùng một provenance. Với manifest combined, fingerprint này bao gồm
    # toàn bộ dataset; fingerprint reference-only vẫn được lưu riêng bên dưới.
    dataset_fingerprint = manifest_obj.fingerprint or reference_manifest.fingerprint

    reference_paths, dev_paths, calibration_paths = split_reference_dev_calibration(
        list(reference_manifest.train_good),
        dev_fraction=cfg.dev_fraction,
        calibration_fraction=cfg.calibration_fraction,
        seed=cfg.seed,
        min_calibration_samples=cfg.min_calibration_samples,
    )
    root = Path(reference_manifest.root_path)
    split_manifest = SplitManifest(
        seed=cfg.seed,
        calibration_fraction=cfg.calibration_fraction,
        dev_fraction=cfg.dev_fraction,
        memory_count=len(reference_paths),
        reference_count=len(reference_paths),
        calibration_count=len(calibration_paths),
        dev_count=len(dev_paths),
        memory_files=[_relative_to_root(root, path) for path in reference_paths],
        dev_files=[_relative_to_root(root, path) for path in dev_paths],
        calibration_files=[_relative_to_root(root, path) for path in calibration_paths],
    )

    transform = build_transform(cfg.preprocessing)
    loader = DataLoader(
        ImageFolderDataset(reference_paths, transform=transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = FeatureExtractor(
        backbone=cfg.backbone,
        layers=cfg.feature_layers,
        pretrained=cfg.pretrained,
        weights=cfg.weights,
    ).to(device)

    batches: list[np.ndarray] = []
    for images, _ in loader:
        with torch.inference_mode():
            batches.append(network(images.to(device)).cpu().numpy())
    if not batches:
        raise ValueError("Reference set không tạo ra patch embedding nào.")
    full_memory = np.concatenate(batches, axis=0)

    coreset_size = cfg.resolved_coreset_size(len(full_memory))
    selected_indices = select_coreset_indices(full_memory, size=coreset_size, seed=cfg.seed)
    compact_memory = full_memory[selected_indices]
    memory_bank = MemoryBank(compact_memory)

    calibration_scores: list[float] = []
    calibration_heatmaps: list[np.ndarray] = []
    for path in calibration_paths:
        tensor = ImageFolderDataset([path], transform=transform)[0][0].unsqueeze(0).to(device)
        with torch.inference_mode():
            patches, (height, width) = network.extract_spatial_features(tensor)
        distances, _ = memory_bank.kneighbors(patches.cpu().numpy())
        heatmap = distances.reshape(height, width)
        smoothed = apply_heatmap_smoothing(heatmap, sigma=cfg.smooth_sigma)
        calibration_scores.append(float(np.percentile(smoothed, cfg.scoring_percentile)))
        calibration_heatmaps.append(smoothed)

    threshold_policy = calibrate_thresholds(
        normal_scores=calibration_scores,
        normal_heatmaps=calibration_heatmaps,
        auto_pass_quantile=cfg.effective_auto_pass_quantile,
        pixel_quantile=cfg.pixel_quantile,
    )

    models_root = Path(models_dir)
    release_id = f"{cfg.category}-v{cfg.model_version}"
    release_dir = models_root / "releases" / release_id
    if release_dir.exists():
        raise FileExistsError(
            f"Release '{release_id}' đã tồn tại. Tạo model_version mới thay vì overwrite."
        )
    release_dir.mkdir(parents=True, exist_ok=False)

    weights_name = getattr(network, "weights_name", None) or cfg.weights
    metadata = ModelMetadata(
        model_version=cfg.model_version,
        pipeline_version=cfg.pipeline_version,
        artifact_schema_version=5,
        category=cfg.category,
        backbone=cfg.backbone,
        weights=weights_name,
        pretrained=cfg.pretrained,
        feature_layers=list(cfg.feature_layers),
        created_at=datetime.now(timezone.utc).isoformat(),
        device_used=device,
        dataset_fingerprint=dataset_fingerprint,
        release_id=release_id,
    )
    artifact = ModelArtifact(
        metadata=metadata,
        threshold_policy=threshold_policy,
        preprocessing=cfg.preprocessing,
        coreset_info={
            "algorithm": "greedy_k_center",
            "projection_dim": 64,
            "size": int(len(compact_memory)),
            "full_memory_patches": int(len(full_memory)),
            "feature_dim": int(compact_memory.shape[1]),
        },
        scoring={
            "method": "percentile",
            "percentile": cfg.scoring_percentile,
            "smooth_sigma": cfg.smooth_sigma,
        },
        calibration={
            "source": "held_out_train_good_calibration",
            "auto_pass_quantile": cfg.effective_auto_pass_quantile,
            "pixel_quantile": cfg.pixel_quantile,
            "samples": len(calibration_paths),
            "note": "Heuristic normal-only upper-tail threshold; khong la bao dam FRR production.",
        },
        smooth_sigma=cfg.smooth_sigma,
        dataset_fingerprint=dataset_fingerprint,
        capture_contract=cfg.capture_contract,
        inspection_policy={"auto_pass_only": True, "human_review_for_anomaly": True},
        reference_manifest={
            "manifest_type": getattr(manifest_obj, "manifest_type", "normal_reference"),
            "category": cfg.category,
            "fingerprint": reference_manifest.fingerprint,
            "reference_count": len(reference_paths),
            "dev_count": len(dev_paths),
            "calibration_count": len(calibration_paths),
        },
    )
    memory_bank.save(release_dir / "memory_bank.npy")
    split_manifest.save(release_dir / "split_manifest.json")
    reference_manifest.save(release_dir / "reference_manifest.json")
    artifact.save(release_dir)
    write_integrity_manifest(release_dir, artifact, dataset_sha256=dataset_fingerprint)
    _save_legacy_category_alias(release_dir, models_root, cfg.category)
    _update_production_pointer(models_root, cfg.category, release_id, cfg.line_id)

    print(
        f"[SUCCESS] '{cfg.category}' release={release_id}: memory={compact_memory.shape}, "
        f"AUTO_PASS threshold={threshold_policy.auto_pass_threshold:.4f}, "
        f"calibration={len(calibration_paths)} normal images"
    )
    return artifact
