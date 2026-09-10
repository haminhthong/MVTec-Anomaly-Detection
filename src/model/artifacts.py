"""Artifact tối giản cho model PatchCore-style.

Artifact phục vụ inference chỉ chứa metadata, threshold và preprocessing.
Danh sách file của các split thí nghiệm được lưu dưới ``reports/`` để không
trộn audit trail vào model runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from ..data.transforms import PreprocessingConfig


@dataclass(frozen=True)
class Thresholds:
    """Hai ngưỡng duy nhất dùng ở runtime."""

    image_threshold: float
    pixel_threshold: float

    def __post_init__(self) -> None:
        values = (self.image_threshold, self.pixel_threshold)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Các threshold phải là số hữu hạn.")
        if any(float(value) < 0 for value in values):
            raise ValueError("Các threshold không được âm.")

    def to_dict(self) -> dict[str, float]:
        """Chuyển threshold sang JSON."""
        return {
            "image_threshold": float(self.image_threshold),
            "pixel_threshold": float(self.pixel_threshold),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Thresholds":
        """Đọc threshold từ metadata runtime."""
        values = data.get("thresholds", data)
        try:
            return cls(
                image_threshold=float(values["image_threshold"]),
                pixel_threshold=float(values["pixel_threshold"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("metadata.json thiếu image_threshold hoặc pixel_threshold.") from exc


@dataclass
class SplitManifest:
    """Thông tin split reproducible; chỉ lưu trong report training."""

    seed: int
    calibration_fraction: float
    memory_count: int
    calibration_count: int
    memory_files: list[str] = field(default_factory=list)
    calibration_files: list[str] = field(default_factory=list)
    dev_fraction: float = 0.0
    dev_count: int = 0
    dev_files: list[str] = field(default_factory=list)
    reference_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize thông tin Reference/Dev/Calibration."""
        payload = asdict(self)
        payload["reference_count"] = (
            self.reference_count if self.reference_count is not None else self.memory_count
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitManifest":
        """Đọc report split để kiểm tra reproducibility."""
        memory_files = list(data.get("memory_files", []))
        dev_files = list(data.get("dev_files", []))
        calibration_files = list(data.get("calibration_files", []))
        return cls(
            seed=int(data.get("seed", 42)),
            calibration_fraction=float(data.get("calibration_fraction", 0.15)),
            memory_count=int(data.get("memory_count", len(memory_files))),
            calibration_count=int(data.get("calibration_count", len(calibration_files))),
            memory_files=memory_files,
            calibration_files=calibration_files,
            dev_fraction=float(data.get("dev_fraction", 0.0)),
            dev_count=int(data.get("dev_count", len(dev_files))),
            dev_files=dev_files,
            reference_count=data.get("reference_count"),
        )

    def save(self, path: str | Path) -> None:
        """Ghi report split."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


@dataclass
class ModelMetadata:
    """Metadata cần để khởi tạo lại detector."""

    model_version: str = "1.0.0"
    category: str = "bottle"
    backbone: str = "resnet18"
    weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pretrained: bool = True
    feature_layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    device_used: str = "cpu"
    dataset_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata model."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Đọc metadata model từ object ``model`` hoặc root."""
        values = data.get("model", data)
        return cls(
            model_version=str(values.get("model_version", "1.0.0")),
            category=str(values.get("category", data.get("category", "bottle"))),
            backbone=str(values.get("backbone", "resnet18")),
            weights=values.get("weights"),
            pretrained=bool(values.get("pretrained", True)),
            feature_layers=list(values.get("feature_layers", ["layer2", "layer3"])),
            created_at=str(values.get("created_at", "")),
            device_used=str(values.get("device_used", "cpu")),
            dataset_fingerprint=values.get(
                "dataset_fingerprint", data.get("dataset_fingerprint")
            ),
        )


@dataclass
class ModelArtifact:
    """Bundle cấu hình nhỏ gọn để detector chạy đúng preprocessing."""

    metadata: ModelMetadata
    thresholds: Thresholds
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
    capture_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Tạo nội dung ``metadata.json``."""
        return {
            "model": self.metadata.to_dict(),
            "category": self.metadata.category,
            "model_version": self.metadata.model_version,
            "preprocessing": self.preprocessing.to_dict(),
            "coreset": self.coreset_info,
            "scoring": self.scoring,
            "calibration": self.calibration,
            "thresholds": self.thresholds.to_dict(),
            "capture_contract": self.capture_contract,
            "dataset_fingerprint": self.dataset_fingerprint
            or self.metadata.dataset_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArtifact":
        """Đọc schema metadata runtime hiện tại."""
        preprocessing = PreprocessingConfig.from_dict(data.get("preprocessing", {}))
        scoring_data = dict(data.get("scoring", {}))
        smooth_sigma = float(scoring_data.get("smooth_sigma", 1.0))
        scoring_data.setdefault("method", "percentile")
        scoring_data.setdefault("percentile", 99.0)
        scoring_data["smooth_sigma"] = smooth_sigma
        metadata = ModelMetadata.from_dict(data)
        return cls(
            metadata=metadata,
            thresholds=Thresholds.from_dict(data),
            preprocessing=preprocessing,
            coreset_info=dict(data.get("coreset", {})),
            scoring=scoring_data,
            calibration=dict(data.get("calibration", {})),
            smooth_sigma=smooth_sigma,
            dataset_fingerprint=data.get("dataset_fingerprint", metadata.dataset_fingerprint),
            capture_contract=dict(data.get("capture_contract", {})),
        )

    def save(self, category_dir: str | Path) -> None:
        """Ghi metadata runtime vào model category."""
        target = Path(category_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "metadata.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, category_dir: str | Path) -> "ModelArtifact":
        """Đọc metadata runtime từ model category."""
        target = Path(category_dir) / "metadata.json"
        if not target.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata.json: {target}")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata.json không hợp lệ: {target}") from exc
        return cls.from_dict(payload)
