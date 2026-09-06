"""Inference Engine for Industrial Visual Anomaly Detection.

Features:
- Reads ModelArtifact containing ThresholdPolicy and PreprocessingConfig
- Strict category isolation
- Dynamic spatial resolution and channel dimensions
- Single-image inspection and high-throughput batch inspection (inspect_batch)
- Operational decision making and defect localization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import Image
import torch

from ..data.transforms import PreprocessingConfig, build_transform
from ..model.artifacts import ModelArtifact, ThresholdPolicy
from ..model.feature_extractor import FeatureExtractor
from ..model.memory_bank import MemoryBank
from .decision import classify_decision_and_severity
from .localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)
from .scoring import compute_image_score


class AnomalyDetector:
    """Industrial visual anomaly detector for a specific product category.

    Attributes:
        category: Name of product category (e.g. 'bottle').
        model_dir: Path to directory containing category artifacts (models/<category>).
        artifact: Loaded ModelArtifact instance.
        threshold_policy: Calibrated ThresholdPolicy.
        preprocessing_config: Synchronized PreprocessingConfig.
        memory_bank: Fitted MemoryBank.
        net: Frozen FeatureExtractor backbone.
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        category: str | None = None,
    ) -> None:
        """Initialize AnomalyDetector for a category.

        Args:
            model_dir: Path to models directory or models/<category> directory.
            category: Optional category name.

        Raises:
            FileNotFoundError: If model artifacts are missing.
            ValueError: If threshold policy is invalid.
        """
        base_path = Path(model_dir)

        # Resolve category folder
        if category and (base_path / category / "config.json").exists():
            target_dir = base_path / category
            self.category: str = category
        elif (base_path / "config.json").exists():
            target_dir = base_path
            self.category = category or target_dir.name
        else:
            raise FileNotFoundError(
                f"Cannot find valid model artifacts for category '{category}' at '{base_path}'."
            )

        self.model_dir: Path = target_dir
        config_file = target_dir / "config.json"
        raw_config: dict[str, Any] = json.loads(config_file.read_text(encoding="utf-8"))

        self.category = raw_config.get("category", self.category)
        self.artifact: ModelArtifact = ModelArtifact.from_dict(raw_config)
        self.threshold_policy: ThresholdPolicy = self.artifact.threshold_policy

        if self.threshold_policy.fail_threshold <= 0:
            raise ValueError(f"Invalid fail_threshold: {self.threshold_policy.fail_threshold}")

        # Load MemoryBank
        memory_file = target_dir / "memory_bank.npy"
        if not memory_file.exists():
            legacy_file = target_dir / "memory.npy"
            if legacy_file.exists():
                memory_file = legacy_file
            else:
                raise FileNotFoundError(f"Missing memory_bank.npy in '{target_dir}'.")

        self.memory_bank: MemoryBank = MemoryBank.load(memory_file)

        # Build transform synchronized with training
        self.preprocessing_config: PreprocessingConfig = self.artifact.preprocessing
        self.transform = build_transform(self.preprocessing_config)

        # Initialize FeatureExtractor
        self.dev: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.net: FeatureExtractor = FeatureExtractor(
            backbone=self.artifact.metadata.backbone,
            layers=self.artifact.metadata.feature_layers,
            pretrained=True,
        ).to(self.dev)

        self.smooth_sigma: float = self.artifact.smooth_sigma
        self.model_version: str = self.artifact.metadata.model_version

    @property
    def threshold(self) -> float:
        """Alias for fail_threshold."""
        return self.threshold_policy.fail_threshold

    @property
    def review_threshold(self) -> float:
        """Alias for review_threshold."""
        return self.threshold_policy.review_threshold

    @property
    def pixel_threshold(self) -> float:
        """Alias for pixel_threshold."""
        return self.threshold_policy.pixel_threshold

    @torch.inference_mode()
    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Compute anomaly score and 2D smoothed heatmap for a single image.

        Args:
            image: Input PIL Image.

        Returns:
            tuple[float, np.ndarray]: (anomaly_score, smoothed_heatmap [H_map, W_map]).
        """
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.dev)
        patches, (h, w) = self.net.extract_spatial_features(x)
        distances, _ = self.memory_bank.kneighbors(patches.cpu().numpy())
        raw_heat = distances.reshape(h, w)
        smoothed_heat = apply_heatmap_smoothing(raw_heat, sigma=self.smooth_sigma)
        image_score = compute_image_score(smoothed_heat, percentile=99.0)
        return image_score, smoothed_heat

    @torch.inference_mode()
    def inspect(
        self, image: Image.Image, include_overlay: bool = True
    ) -> dict[str, Any]:
        """Inspect a single image and produce structured inspection response.

        Args:
            image: Input PIL Image.
            include_overlay: Whether to generate Base64 heatmap overlay string.

        Returns:
            dict[str, Any]: Standardized industrial inspection response.
        """
        score, smoothed_heat = self.score(image)
        peak_score = float(np.max(smoothed_heat))
        area_ratio = compute_anomalous_area_ratio(
            smoothed_heat, self.pixel_threshold
        )

        decision, severity = classify_decision_and_severity(
            anomaly_score=score,
            threshold_policy=self.threshold_policy,
            anomalous_area_ratio=area_ratio,
            peak_score=peak_score,
        )

        overlay_b64 = (
            create_heatmap_overlay_b64(
                image=image,
                heatmap=smoothed_heat,
                threshold=self.pixel_threshold,
                alpha=0.45,
                target_size=self.preprocessing_config.image_size,
            )
            if include_overlay
            else None
        )

        inspection_id = f"insp_{uuid.uuid4().hex[:12]}"

        return {
            "inspection_id": inspection_id,
            "category": self.category,
            "decision": decision,
            "severity": severity,
            "scores": {
                "anomaly_score": round(score, 4),
                "review_threshold": round(self.review_threshold, 4),
                "fail_threshold": round(self.threshold, 4),
            },
            "localization": {
                "anomalous_area_ratio": round(area_ratio, 4),
                "peak_score": round(peak_score, 4),
                "pixel_threshold": round(self.pixel_threshold, 4),
            },
            "model": {
                "version": self.model_version,
                "category": self.category,
            },
            "overlay_b64": overlay_b64,
            # Backward compatibility fields
            "anomaly_score": score,
            "threshold": self.threshold,
            "heatmap_shape": list(smoothed_heat.shape),
            "model_version": self.model_version,
        }

    @torch.inference_mode()
    def inspect_batch(
        self, images: list[Image.Image], include_overlay: bool = False
    ) -> list[dict[str, Any]]:
        """High-throughput batch inspection for production lines.

        Workflow:
        N images -> single batch tensor [N, 3, H, W] -> single CNN forward
        -> unbatch patch embeddings -> nearest-neighbor distance -> N results.

        Args:
            images: List of input PIL images.
            include_overlay: Whether to include Base64 overlays in results.

        Returns:
            list[dict[str, Any]]: List of inspection results.
        """
        if not images:
            return []

        # 1. Batch preprocessing
        tensors = [self.transform(img.convert("RGB")) for img in images]
        batch_tensor = torch.stack(tensors, dim=0).to(self.dev)
        batch_size = len(images)

        # 2. Single forward pass
        all_patches, (h, w) = self.net.extract_spatial_features(batch_tensor)
        patches_per_image = h * w

        # 3. Nearest neighbor scoring against MemoryBank
        all_distances, _ = self.memory_bank.kneighbors(all_patches.cpu().numpy())
        reshaped_distances = all_distances.reshape(batch_size, h, w)

        # 4. Generate results per image
        results: list[dict[str, Any]] = []
        for i in range(batch_size):
            raw_heat = reshaped_distances[i]
            smoothed_heat = apply_heatmap_smoothing(raw_heat, sigma=self.smooth_sigma)
            score = compute_image_score(smoothed_heat, percentile=99.0)
            peak_score = float(np.max(smoothed_heat))
            area_ratio = compute_anomalous_area_ratio(smoothed_heat, self.pixel_threshold)

            decision, severity = classify_decision_and_severity(
                anomaly_score=score,
                threshold_policy=self.threshold_policy,
                anomalous_area_ratio=area_ratio,
                peak_score=peak_score,
            )

            overlay_b64 = (
                create_heatmap_overlay_b64(
                    image=images[i],
                    heatmap=smoothed_heat,
                    threshold=self.pixel_threshold,
                    alpha=0.45,
                    target_size=self.preprocessing_config.image_size,
                )
                if include_overlay
                else None
            )

            results.append({
                "inspection_id": f"insp_{uuid.uuid4().hex[:12]}",
                "category": self.category,
                "decision": decision,
                "severity": severity,
                "scores": {
                    "anomaly_score": round(score, 4),
                    "review_threshold": round(self.review_threshold, 4),
                    "fail_threshold": round(self.threshold, 4),
                },
                "localization": {
                    "anomalous_area_ratio": round(area_ratio, 4),
                    "peak_score": round(peak_score, 4),
                    "pixel_threshold": round(self.pixel_threshold, 4),
                },
                "model": {
                    "version": self.model_version,
                    "category": self.category,
                },
                "overlay_b64": overlay_b64,
                "anomaly_score": score,
                "threshold": self.threshold,
                "heatmap_shape": [h, w],
                "model_version": self.model_version,
            })

        return results
