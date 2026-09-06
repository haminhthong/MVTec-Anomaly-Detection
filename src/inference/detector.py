"""Inference engine for category-scoped PatchCore-style model artifacts."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..data.transforms import PreprocessingConfig, build_transform
from ..model.artifacts import load_artifact_config, resolve_artifact_dir
from ..model.memory_bank import MemoryBank
from ..model.patch_embedding import FeatureExtractor
from .decision import classify_decision_and_severity
from .localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)


class AnomalyDetector:
    """Load one immutable artifact and inspect images with the same preprocessing."""

    def __init__(
        self, model_dir: str | Path = "models", category: str | None = None
    ) -> None:
        self.artifact_dir = resolve_artifact_dir(model_dir, category=category)
        self.config: dict[str, Any] = load_artifact_config(self.artifact_dir)
        self.category = str(self.config.get("category", category or self.artifact_dir.name))

        if category is not None and self.category != category:
            raise ValueError(
                f"Requested category '{category}' but artifact declares '{self.category}'."
            )

        self.model_version = str(self.config.get("version", "unknown"))
        self.smooth_sigma = float(self.config.get("smooth_sigma", 1.0))
        thresholds = self.config.get("thresholds", {})

        fail_value = thresholds.get("fail_threshold", self.config.get("threshold"))
        if fail_value is None:
            raise ValueError(f"Artifact '{self.artifact_dir}' has no calibrated fail threshold.")
        self.threshold = float(fail_value)
        self.review_threshold = float(
            thresholds.get(
                "review_threshold", self.config.get("review_threshold", 0.8 * self.threshold)
            )
        )
        self.pixel_threshold = float(
            thresholds.get(
                "pixel_threshold", self.config.get("pixel_threshold", self.threshold)
            )
        )
        if self.threshold <= 0 or self.review_threshold <= 0 or self.pixel_threshold <= 0:
            raise ValueError("All calibrated thresholds must be > 0.")
        if self.review_threshold > self.threshold:
            raise ValueError("review_threshold must not exceed fail_threshold.")

        memory_file = self.artifact_dir / "memory_bank.npy"
        if not memory_file.exists():
            memory_file = self.artifact_dir / "memory.npy"
        if not memory_file.exists():
            raise FileNotFoundError(f"Memory bank is missing from '{self.artifact_dir}'.")
        self.memory_bank = MemoryBank.load(memory_file)

        prep_data = self.config.get("preprocessing", {})
        self.preprocessing_config = (
            PreprocessingConfig.from_dict(prep_data) if prep_data else PreprocessingConfig()
        )
        self.transform = build_transform(self.preprocessing_config)

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = FeatureExtractor().to(self.dev)

    @torch.inference_mode()
    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.dev)
        patches, (height, width) = self.net.extract_spatial_features(x)
        distances, _ = self.memory_bank.kneighbors(patches.cpu().numpy())
        raw_heatmap = distances.reshape(height, width)
        smoothed = apply_heatmap_smoothing(raw_heatmap, sigma=self.smooth_sigma)
        image_score = float(np.percentile(smoothed, 99))
        return image_score, smoothed

    @torch.inference_mode()
    def inspect(self, image: Image.Image, include_overlay: bool = True) -> dict[str, Any]:
        score, heatmap = self.score(image)
        peak_score = float(np.max(heatmap))
        area_ratio = compute_anomalous_area_ratio(heatmap, self.pixel_threshold)

        decision, severity = classify_decision_and_severity(
            anomaly_score=score,
            review_threshold=self.review_threshold,
            fail_threshold=self.threshold,
            anomalous_area_ratio=area_ratio,
            peak_score=peak_score,
        )

        overlay_b64 = (
            create_heatmap_overlay_b64(
                image=image,
                heatmap=heatmap,
                threshold=self.pixel_threshold,
                alpha=0.45,
            )
            if include_overlay
            else None
        )

        inspection_id = f"insp_{uuid.uuid4().hex[:12]}"
        return {
            "inspection_id": inspection_id,
            "prediction": {
                "decision": decision,
                "severity": severity,
                "anomaly_score": score,
                "review_threshold": self.review_threshold,
                "fail_threshold": self.threshold,
            },
            "localization": {
                "peak_score": peak_score,
                "anomalous_area_ratio": area_ratio,
                "pixel_threshold": self.pixel_threshold,
            },
            "model": {
                "version": self.model_version,
                "category": self.category,
                "artifact_dir": str(self.artifact_dir),
            },
            "anomaly_score": score,
            "threshold": self.threshold,
            "decision": decision,
            "heatmap_shape": list(heatmap.shape),
            "model_version": self.model_version,
            "overlay_b64": overlay_b64,
        }

    @torch.inference_mode()
    def inspect_detailed(self, image: Image.Image) -> dict[str, Any]:
        return self.inspect(image, include_overlay=True)
