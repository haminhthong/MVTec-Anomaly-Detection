"""Artifact contracts, metadata schemas, and threshold policies for model persistence.

Centralizes the ModelArtifact contract:
- ThresholdPolicy (review, fail, pixel thresholds calibrated on normal held-out data)
- SplitManifest (reproducible record of memory vs calibration sets)
- ModelMetadata (versioning, backbone parameters, schema version)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..data.transforms import PreprocessingConfig


@dataclass(frozen=True)
class ThresholdPolicy:
    """Operating threshold policy calibrated strictly on normal data distributions.

    Note: These are operational policy thresholds (P95 review, P99 fail),
    NOT optimal thresholds tuned against defect validation sets.
    """

    review_threshold: float
    fail_threshold: float
    pixel_threshold: float

    def to_dict(self) -> dict[str, float]:
        """Convert threshold policy to dictionary representation."""
        return {
            "review": float(self.review_threshold),
            "fail": float(self.fail_threshold),
            "pixel": float(self.pixel_threshold),
            "review_threshold": float(self.review_threshold),
            "fail_threshold": float(self.fail_threshold),
            "pixel_threshold": float(self.pixel_threshold),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdPolicy:
        """Instantiate ThresholdPolicy from a dictionary."""
        # Support both nested under 'thresholds' or flat dictionary
        th_dict = data.get("thresholds", data)
        review = th_dict.get("review_threshold", th_dict.get("review"))
        fail = th_dict.get("fail_threshold", th_dict.get("fail", th_dict.get("threshold")))
        pixel = th_dict.get("pixel_threshold", th_dict.get("pixel", fail))

        if fail is None:
            raise ValueError(f"Missing required fail/image threshold in data: {data}")
        if review is None:
            review = 0.8 * float(fail)
        if pixel is None:
            pixel = float(fail)

        return cls(
            review_threshold=float(review),
            fail_threshold=float(fail),
            pixel_threshold=float(pixel),
        )


@dataclass
class SplitManifest:
    """Split metadata recording exact separation of Memory and Calibration sets."""

    seed: int
    calibration_fraction: float
    memory_count: int
    calibration_count: int
    memory_files: list[str] = field(default_factory=list)
    calibration_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert split manifest to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitManifest:
        """Instantiate SplitManifest from dictionary."""
        return cls(
            seed=int(data.get("seed", 42)),
            calibration_fraction=float(data.get("calibration_fraction", 0.2)),
            memory_count=int(data.get("memory_count", len(data.get("memory_files", [])))),
            calibration_count=int(data.get("calibration_count", len(data.get("calibration_files", [])))),
            memory_files=list(data.get("memory_files", [])),
            calibration_files=list(data.get("calibration_files", [])),
        )

    def save(self, path: str | Path) -> None:
        """Save split manifest to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SplitManifest:
        """Load split manifest from JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Split manifest not found at: {p}")
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


@dataclass
class ModelMetadata:
    """Structured version and provenance metadata for model artifact."""

    model_version: str = "1.0.0"
    pipeline_version: str = "1.0"
    artifact_schema_version: int = 4
    category: str = "bottle"
    backbone: str = "resnet18"
    weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pretrained: bool = True
    feature_layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    device_used: str = "cpu"
    dataset_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Instantiate ModelMetadata from dictionary supporting flat or nested layouts."""
        sub = data.get("model", data)
        return cls(
            model_version=str(sub.get("model_version", sub.get("version", data.get("version", "1.0.0")))),
            pipeline_version=str(sub.get("pipeline_version", data.get("pipeline_version", "1.0"))),
            artifact_schema_version=int(
                sub.get("artifact_schema_version", sub.get("schema_version", data.get("schema_version", 4)))
            ),
            category=str(sub.get("category", data.get("category", "bottle"))),
            backbone=str(sub.get("backbone", data.get("backbone", "resnet18"))),
            weights=sub.get("weights", data.get("weights")),
            pretrained=bool(sub.get("pretrained", data.get("pretrained", True))),
            feature_layers=list(sub.get("feature_layers", data.get("feature_layers", ["layer2", "layer3"]))),
            created_at=str(sub.get("created_at", data.get("created_at", ""))),
            device_used=str(sub.get("device_used", data.get("device_used", "cpu"))),
            dataset_fingerprint=sub.get("dataset_fingerprint", data.get("dataset_fingerprint")),
        )


@dataclass
class ModelArtifact:
    """Comprehensive container representing a saved model artifact."""

    metadata: ModelMetadata
    threshold_policy: ThresholdPolicy
    preprocessing: PreprocessingConfig
    coreset_info: dict[str, Any]
    scoring: dict[str, Any] = field(
        default_factory=lambda: {
            "method": "percentile",
            "percentile": 99.0,
            "smooth_sigma": 1.0,
        }
    )
    calibration: dict[str, Any] = field(default_factory=dict)
    smooth_sigma: float = 1.0
    dataset_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Assemble full config.json payload."""
        payload = {
            "model": self.metadata.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "scoring": self.scoring,
            "coreset": self.coreset_info,
            "calibration": self.calibration,
            "thresholds": self.threshold_policy.to_dict(),
            "dataset_fingerprint": self.dataset_fingerprint or self.metadata.dataset_fingerprint,
            # Backward-compatible convenience flat fields:
            "category": self.metadata.category,
            "version": self.metadata.model_version,
            "model_version": self.metadata.model_version,
            "pipeline_version": self.metadata.pipeline_version,
            "artifact_schema_version": self.metadata.artifact_schema_version,
            "schema_version": self.metadata.artifact_schema_version,
            "backbone": self.metadata.backbone,
            "pretrained": self.metadata.pretrained,
            "weights": self.metadata.weights,
            "feature_layers": self.metadata.feature_layers,
            "threshold": self.threshold_policy.fail_threshold,
            "review_threshold": self.threshold_policy.review_threshold,
            "pixel_threshold": self.threshold_policy.pixel_threshold,
            "smooth_sigma": self.smooth_sigma,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelArtifact:
        """Parse ModelArtifact from JSON dictionary supporting nested or flat format."""
        metadata = ModelMetadata.from_dict(data)
        threshold_policy = ThresholdPolicy.from_dict(data)
        prep_data = data.get("preprocessing", {})
        preprocessing = (
            PreprocessingConfig.from_dict(prep_data)
            if prep_data
            else PreprocessingConfig()
        )
        coreset_info = dict(data.get("coreset", {}))

        scoring_data = data.get("scoring", {})
        smooth_sigma = float(
            scoring_data.get(
                "smooth_sigma", data.get("smooth_sigma", 1.0)
            )
        )
        percentile = float(
            scoring_data.get(
                "percentile", data.get("scoring_percentile", 99.0)
            )
        )
        scoring = {
            "method": scoring_data.get("method", "percentile"),
            "percentile": percentile,
            "smooth_sigma": smooth_sigma,
        }

        calibration = dict(data.get("calibration", {}))
        dataset_fingerprint = data.get("dataset_fingerprint", metadata.dataset_fingerprint)

        return cls(
            metadata=metadata,
            threshold_policy=threshold_policy,
            preprocessing=preprocessing,
            coreset_info=coreset_info,
            scoring=scoring,
            calibration=calibration,
            smooth_sigma=smooth_sigma,
            dataset_fingerprint=dataset_fingerprint,
        )

    def save(self, category_dir: str | Path) -> None:
        """Save artifact configuration to config.json in the category directory."""
        p = Path(category_dir)
        p.mkdir(parents=True, exist_ok=True)
        config_path = p / "config.json"
        config_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, category_dir: str | Path) -> ModelArtifact:
        """Load artifact configuration from category directory."""
        p = Path(category_dir)
        config_path = p / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config.json not found at: {config_path}")
        return cls.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
