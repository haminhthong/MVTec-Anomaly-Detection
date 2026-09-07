"""Detector runtime: quality gate -> anomaly score -> AUTO_PASS/HUMAN_REVIEW."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import Image
import torch

from ..capture.contract import CaptureContract
from ..capture.quality import validate_capture
from ..data.transforms import PreprocessingConfig, build_transform
from ..model.artifact_resolver import resolve_artifact_dir
from ..model.artifacts import ModelArtifact, ThresholdPolicy, verify_artifact_integrity
from ..model.feature_extractor import FeatureExtractor
from ..model.memory_bank import MemoryBank
from .decision import OperationalPolicy, classify_decision
from .localization import apply_heatmap_smoothing, compute_anomalous_area_ratio, create_heatmap_overlay_b64
from .scoring import compute_image_score


class AnomalyDetector:
    """Detector gắn với đúng một category/release immutable."""

    def __init__(self, model_dir: str | Path = "models", category: str | None = None) -> None:
        target_dir = resolve_artifact_dir(model_root=model_dir, category=category)
        self.model_dir = target_dir
        config_file = target_dir / "config.json"
        raw_config: dict[str, Any] = json.loads(config_file.read_text(encoding="utf-8"))
        self.artifact = ModelArtifact.from_dict(raw_config)
        verify_artifact_integrity(target_dir, strict=self.artifact.metadata.artifact_schema_version >= 5)
        self.category = self.artifact.metadata.category
        if category is not None and self.category != category:
            raise ValueError(f"Artifact category '{self.category}' không khớp '{category}'.")
        self.threshold_policy: ThresholdPolicy = self.artifact.threshold_policy
        self.operational_policy = OperationalPolicy.from_threshold_policy(self.threshold_policy)
        self.capture_contract = CaptureContract.from_dict(self.artifact.capture_contract)

        memory_file = target_dir / "memory_bank.npy"
        if not memory_file.exists():
            legacy_file = target_dir / "memory.npy"
            memory_file = legacy_file if legacy_file.exists() else memory_file
        if not memory_file.exists():
            raise FileNotFoundError(f"Thiếu memory_bank.npy trong '{target_dir}'.")
        self.memory_bank = MemoryBank.load(memory_file)
        expected_size = self.artifact.coreset_info.get("size")
        if expected_size is not None and int(expected_size) != self.memory_bank.size:
            raise ValueError(
                f"Coreset metadata={expected_size} nhưng memory bank có {self.memory_bank.size} rows."
            )

        self.preprocessing_config: PreprocessingConfig = self.artifact.preprocessing
        self.transform = build_transform(self.preprocessing_config)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = FeatureExtractor(
            backbone=self.artifact.metadata.backbone,
            layers=self.artifact.metadata.feature_layers,
            pretrained=self.artifact.metadata.pretrained,
            weights=self.artifact.metadata.weights,
        ).to(self.dev)
        self.smooth_sigma = self.artifact.smooth_sigma
        self.scoring_percentile = float(self.artifact.scoring.get("percentile", 99.0))
        self.model_version = self.artifact.metadata.model_version
        self.release_id = self.artifact.metadata.release_id or target_dir.name

    @property
    def threshold(self) -> float:
        """Alias cũ cho auto-pass threshold."""
        return self.threshold_policy.auto_pass_threshold

    @property
    def auto_pass_threshold(self) -> float:
        """Ngưỡng ảnh được AUTO_PASS."""
        return self.threshold_policy.auto_pass_threshold

    @property
    def review_threshold(self) -> float:
        """Alias cũ; V1 không có vùng review thứ hai."""
        return self.threshold_policy.auto_pass_threshold

    @property
    def pixel_threshold(self) -> float:
        """Ngưỡng pixel calibration normal."""
        return self.threshold_policy.pixel_threshold

    @torch.inference_mode()
    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Tính image anomaly score và heatmap đã smoothing."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.dev)
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
        line_id: str | None = None,
        camera_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Tạo response nhất quán cho cả capture-invalid và model decision."""
        return {
            "inspection_id": inspection_id,
            "category": self.category,
            "decision": decision,
            "severity": None,
            "scores": {
                "anomaly_score": round(score, 4) if score is not None else None,
                "auto_pass_threshold": round(self.auto_pass_threshold, 4),
                "review_threshold": round(self.auto_pass_threshold, 4),
                "fail_threshold": round(self.auto_pass_threshold, 4),
            },
            "localization": {
                "anomalous_area_ratio": round(area_ratio, 4),
                "peak_anomaly_score": round(peak_score, 4),
                "peak_score": round(peak_score, 4),
                "pixel_threshold": round(self.pixel_threshold, 4),
            },
            "capture_quality": quality,
            "model": {
                "version": self.model_version,
                "category": self.category,
                "release_id": self.release_id,
            },
            "line_id": line_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "overlay_b64": overlay_b64,
            "anomaly_score": score,
            "threshold": self.auto_pass_threshold,
            "heatmap_shape": heatmap_shape,
            "model_version": self.model_version,
        }

    @torch.inference_mode()
    def inspect(
        self,
        image: Image.Image,
        include_overlay: bool = True,
        line_id: str | None = None,
        camera_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Quality gate trước, sau đó mới chạy detector."""
        inspection_id = f"insp_{uuid.uuid4().hex[:12]}"
        quality = validate_capture(image, self.capture_contract)
        if not quality.valid:
            return self._base_result(
                inspection_id,
                "RECAPTURE_REQUIRED",
                quality.to_dict(),
                line_id=line_id,
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
            line_id=line_id,
            camera_id=camera_id,
            timestamp=timestamp,
        )

    @torch.inference_mode()
    def inspect_batch(
        self,
        images: list[Image.Image],
        include_overlay: bool = False,
        line_id: str | None = None,
        camera_id: str | None = None,
        timestamp: str | None = None,
    ) -> list[dict[str, Any]]:
        """Inspect batch; ảnh capture-invalid được trả về riêng và không score."""
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
                    line_id=line_id,
                    camera_id=camera_id,
                    timestamp=timestamp,
                )
            else:
                valid_indices.append(index)
                valid_images.append(image)
                valid_quality.append(quality.to_dict())

        if valid_images:
            tensors = torch.stack([self.transform(image.convert("RGB")) for image in valid_images]).to(self.dev)
            all_patches, (height, width) = self.net.extract_spatial_features(tensors)
            distances, _ = self.memory_bank.kneighbors(all_patches.cpu().numpy())
            for local_index, original_index in enumerate(valid_indices):
                heatmap = distances[local_index * height * width : (local_index + 1) * height * width]
                heatmap = apply_heatmap_smoothing(heatmap.reshape(height, width), sigma=self.smooth_sigma)
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
                    line_id=line_id,
                    camera_id=camera_id,
                    timestamp=timestamp,
                )
        return [result for result in results if result is not None]
