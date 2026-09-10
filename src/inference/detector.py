"""Detector runtime: kiểm tra ảnh -> điểm 1-NN -> heatmap -> triage."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import Image
import torch

from ..capture.contract import CaptureContract
from ..capture.quality import validate_capture
from ..data.transforms import PreprocessingConfig, build_transform
from ..model.artifacts import ModelArtifact
from ..model.feature_extractor import FeatureExtractor
from ..model.memory_bank import MemoryBank
from ..path_safety import ensure_safe_segment
from .decision import OperationalPolicy, classify_decision
from .localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)
from .scoring import compute_image_score


class AnomalyDetector:
    """Detector gắn với đúng một thư mục model category."""

    def __init__(self, model_dir: str | Path = "models/bottle", category: str | None = None) -> None:
        target_dir = Path(model_dir)
        if category is not None:
            category = ensure_safe_segment(category.strip(), "category")
            if not (target_dir / "metadata.json").exists():
                target_dir = target_dir / category
        self.model_dir = target_dir
        self.artifact = ModelArtifact.load(target_dir)
        self.category = self.artifact.metadata.category
        if category is not None and self.category != category:
            raise ValueError(f"Artifact category '{self.category}' không khớp '{category}'.")

        self.thresholds = self.artifact.thresholds
        self.operational_policy = OperationalPolicy.from_thresholds(self.thresholds)
        self.capture_contract = CaptureContract.from_dict(self.artifact.capture_contract)

        memory_file = target_dir / "memory_bank.npy"
        if not memory_file.exists():
            raise FileNotFoundError(f"Thiếu memory_bank.npy trong '{target_dir}'.")
        self.memory_bank = MemoryBank.load(memory_file)
        expected_size = self.artifact.coreset_info.get("size")
        if expected_size is not None and int(expected_size) != self.memory_bank.size:
            raise ValueError(
                f"Coreset metadata={expected_size} nhưng memory bank có {self.memory_bank.size} rows."
            )
        expected_dim = self.artifact.coreset_info.get("feature_dim")
        if expected_dim is not None and int(expected_dim) != self.memory_bank.dim:
            raise ValueError(
                f"Feature metadata={expected_dim} nhưng memory bank có {self.memory_bank.dim} columns."
            )

        self.preprocessing_config: PreprocessingConfig = self.artifact.preprocessing
        self.transform = build_transform(self.preprocessing_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = FeatureExtractor(
            backbone=self.artifact.metadata.backbone,
            layers=self.artifact.metadata.feature_layers,
            pretrained=self.artifact.metadata.pretrained,
            weights=self.artifact.metadata.weights,
        ).to(self.device)
        self.smooth_sigma = self.artifact.smooth_sigma
        self.scoring_percentile = float(self.artifact.scoring.get("percentile", 99.0))
        self.model_version = self.artifact.metadata.model_version

    @property
    def image_threshold(self) -> float:
        """Ngưỡng image score được hiệu chỉnh từ normal holdout."""
        return self.thresholds.image_threshold

    @property
    def pixel_threshold(self) -> float:
        """Ngưỡng pixel score dùng cho lớp phủ định vị."""
        return self.thresholds.pixel_threshold

    @torch.inference_mode()
    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Tính image anomaly score và heatmap sau Gaussian smoothing."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        patches, (height, width) = self.net.extract_spatial_features(tensor)
        distances, _ = self.memory_bank.kneighbors(patches.cpu().numpy())
        heatmap = distances.reshape(height, width)
        smoothed = apply_heatmap_smoothing(heatmap, sigma=self.smooth_sigma)
        return compute_image_score(smoothed, percentile=self.scoring_percentile), smoothed

    def _base_result(
        self,
        inspection_id: str,
        decision: str,
        quality: dict[str, Any],
        score: float | None = None,
        heatmap_shape: list[int] | None = None,
        overlay_b64: str | None = None,
        area_ratio: float = 0.0,
        peak_score: float = 0.0,
        camera_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Tạo response phẳng chỉ chứa thông tin cần cho inspection."""
        return {
            "inspection_id": inspection_id,
            "category": self.category,
            "decision": decision,
            "anomaly_score": round(score, 4) if score is not None else None,
            "image_threshold": round(self.image_threshold, 4),
            "anomalous_area_ratio": round(area_ratio, 4),
            "peak_anomaly_score": round(peak_score, 4),
            "pixel_threshold": round(self.pixel_threshold, 4),
            "capture_quality": quality,
            "model_version": self.model_version,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "overlay_b64": overlay_b64,
            "heatmap_shape": heatmap_shape,
        }

    @torch.inference_mode()
    def inspect(
        self,
        image: Image.Image,
        include_overlay: bool = True,
        camera_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Input check trước, sau đó mới chạy anomaly detector."""
        inspection_id = f"insp_{uuid.uuid4().hex[:12]}"
        quality = validate_capture(image, self.capture_contract)
        if not quality.valid:
            return self._base_result(
                inspection_id,
                "RECAPTURE_REQUIRED",
                quality.to_dict(),
                camera_id=camera_id,
                timestamp=timestamp,
            )

        score, heatmap = self.score(image)
        peak_score = float(np.max(heatmap))
        area_ratio = compute_anomalous_area_ratio(heatmap, self.pixel_threshold)
        decision = classify_decision(score, operational_policy=self.operational_policy)
        overlay = (
            create_heatmap_overlay_b64(
                image=image,
                heatmap=heatmap,
                threshold=self.pixel_threshold,
                alpha=0.45,
                target_size=self.preprocessing_config.image_size,
            )
            if include_overlay
            else None
        )
        return self._base_result(
            inspection_id,
            decision,
            quality.to_dict(),
            score=score,
            heatmap_shape=list(heatmap.shape),
            overlay_b64=overlay,
            area_ratio=area_ratio,
            peak_score=peak_score,
            camera_id=camera_id,
            timestamp=timestamp,
        )

    @torch.inference_mode()
    def inspect_batch(
        self,
        images: list[Image.Image],
        include_overlay: bool = False,
        camera_id: str | None = None,
        timestamp: str | None = None,
    ) -> list[dict[str, Any]]:
        """Tính điểm batch hợp lệ và ghép kết quả theo đúng thứ tự đầu vào."""
        if not images:
            return []
        results: list[dict[str, Any] | None] = [None] * len(images)
        valid_indices: list[int] = []
        valid_images: list[Image.Image] = []
        valid_quality: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            quality = validate_capture(image, self.capture_contract)
            if not quality.valid:
                results[index] = self._base_result(
                    f"insp_{uuid.uuid4().hex[:12]}",
                    "RECAPTURE_REQUIRED",
                    quality.to_dict(),
                    camera_id=camera_id,
                    timestamp=timestamp,
                )
            else:
                valid_indices.append(index)
                valid_images.append(image)
                valid_quality.append(quality.to_dict())

        if valid_images:
            tensors = torch.stack(
                [self.transform(image.convert("RGB")) for image in valid_images]
            ).to(self.device)
            all_patches, (height, width) = self.net.extract_spatial_features(tensors)
            distances, _ = self.memory_bank.kneighbors(all_patches.cpu().numpy())
            patches_per_image = height * width
            for local_index, original_index in enumerate(valid_indices):
                start = local_index * patches_per_image
                stop = start + patches_per_image
                heatmap = apply_heatmap_smoothing(
                    distances[start:stop].reshape(height, width),
                    sigma=self.smooth_sigma,
                )
                score = compute_image_score(heatmap, percentile=self.scoring_percentile)
                peak_score = float(np.max(heatmap))
                area_ratio = compute_anomalous_area_ratio(heatmap, self.pixel_threshold)
                overlay = (
                    create_heatmap_overlay_b64(
                        image=valid_images[local_index],
                        heatmap=heatmap,
                        threshold=self.pixel_threshold,
                        alpha=0.45,
                        target_size=self.preprocessing_config.image_size,
                    )
                    if include_overlay
                    else None
                )
                results[original_index] = self._base_result(
                    f"insp_{uuid.uuid4().hex[:12]}",
                    classify_decision(score, operational_policy=self.operational_policy),
                    valid_quality[local_index],
                    score=score,
                    heatmap_shape=[height, width],
                    overlay_b64=overlay,
                    area_ratio=area_ratio,
                    peak_score=peak_score,
                    camera_id=camera_id,
                    timestamp=timestamp,
                )
        return [result for result in results if result is not None]
