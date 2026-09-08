"""Hợp đồng artifact immutable, policy normal-only và kiểm tra integrity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..data.manifest import sha256_file
from ..data.transforms import PreprocessingConfig


@dataclass(frozen=True, init=False)
class ThresholdPolicy:
    """Policy mới chỉ có AUTO_PASS và HUMAN_REVIEW.

    ``review_threshold``/``fail_threshold`` là property tương thích artifact cũ;
    artifact mới lưu cả ``auto_pass_threshold`` và đặt hai alias về cùng một giá trị.
    """

    review_threshold: float
    fail_threshold: float
    pixel_threshold: float

    def __init__(
        self,
        review_threshold: float | None = None,
        fail_threshold: float | None = None,
        pixel_threshold: float | None = None,
        *,
        auto_pass_threshold: float | None = None,
    ) -> None:
        selected = auto_pass_threshold
        if selected is None:
            selected = fail_threshold if fail_threshold is not None else review_threshold
        if selected is None or pixel_threshold is None:
            raise ValueError("ThresholdPolicy cần auto_pass_threshold và pixel_threshold.")
        selected = float(selected)
        pixel = float(pixel_threshold)
        if selected < 0 or pixel < 0:
            raise ValueError("Các threshold không được âm.")
        object.__setattr__(self, "review_threshold", float(review_threshold if review_threshold is not None else selected))
        object.__setattr__(self, "fail_threshold", selected)
        object.__setattr__(self, "pixel_threshold", pixel)

    @property
    def auto_pass_threshold(self) -> float:
        """Ngưỡng dưới đó ảnh được AUTO_PASS."""
        return self.fail_threshold

    def to_dict(self) -> dict[str, float]:
        """Serialize policy mới và alias cũ để artifact legacy vẫn đọc được."""
        return {
            "auto_pass_threshold": float(self.auto_pass_threshold),
            "pixel_threshold": float(self.pixel_threshold),
            "review": float(self.review_threshold),
            "fail": float(self.fail_threshold),
            "pixel": float(self.pixel_threshold),
            "review_threshold": float(self.review_threshold),
            "fail_threshold": float(self.fail_threshold),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThresholdPolicy":
        """Đọc cả schema mới và schema cũ."""
        values = data.get("thresholds", data)
        auto_pass = values.get("auto_pass_threshold")
        fail = values.get("fail_threshold", values.get("fail", values.get("threshold")))
        review = values.get("review_threshold", values.get("review"))
        pixel = values.get("pixel_threshold", values.get("pixel", fail))
        if auto_pass is None:
            auto_pass = fail if fail is not None else review
        if auto_pass is None or pixel is None:
            raise ValueError(f"Thiếu threshold trong artifact: {data}")
        return cls(
            review_threshold=float(review) if review is not None else float(auto_pass),
            fail_threshold=float(auto_pass),
            pixel_threshold=float(pixel),
        )


@dataclass
class SplitManifest:
    """Ghi nhận Reference/Dev/Calibration split và seed tái lập."""

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
        """Serialize split manifest."""
        payload = asdict(self)
        payload["reference_count"] = self.reference_count if self.reference_count is not None else self.memory_count
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitManifest":
        """Deserialize split manifest cũ/mới."""
        memory_files = list(data.get("memory_files", data.get("reference_files", [])))
        return cls(
            seed=int(data.get("seed", 42)),
            calibration_fraction=float(data.get("calibration_fraction", 0.2)),
            memory_count=int(data.get("memory_count", data.get("reference_count", len(memory_files)))),
            calibration_count=int(data.get("calibration_count", len(data.get("calibration_files", [])))),
            memory_files=memory_files,
            calibration_files=list(data.get("calibration_files", [])),
            dev_fraction=float(data.get("dev_fraction", 0.0)),
            dev_count=int(data.get("dev_count", len(data.get("dev_files", [])))),
            dev_files=list(data.get("dev_files", [])),
            reference_count=data.get("reference_count"),
        )

    def save(self, path: str | Path) -> None:
        """Ghi split manifest."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        """Đọc split manifest."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Không tìm thấy split manifest: {target}")
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


@dataclass
class ModelMetadata:
    """Metadata version, backbone và provenance của model release."""

    model_version: str = "1.0.0"
    pipeline_version: str = "2.0"
    artifact_schema_version: int = 5
    category: str = "bottle"
    backbone: str = "resnet18"
    weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pretrained: bool = True
    feature_layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    device_used: str = "cpu"
    dataset_fingerprint: str | None = None
    release_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Deserialize metadata nested/flat."""
        sub = data.get("model", data)
        return cls(
            model_version=str(sub.get("model_version", sub.get("version", data.get("version", "1.0.0")))),
            pipeline_version=str(sub.get("pipeline_version", data.get("pipeline_version", "1.0"))),
            artifact_schema_version=int(sub.get("artifact_schema_version", sub.get("schema_version", data.get("schema_version", 4)))),
            category=str(sub.get("category", data.get("category", "bottle"))),
            backbone=str(sub.get("backbone", data.get("backbone", "resnet18"))),
            weights=sub.get("weights", data.get("weights")),
            pretrained=bool(sub.get("pretrained", data.get("pretrained", True))),
            feature_layers=list(sub.get("feature_layers", data.get("feature_layers", ["layer2", "layer3"]))),
            created_at=str(sub.get("created_at", data.get("created_at", ""))),
            device_used=str(sub.get("device_used", data.get("device_used", "cpu"))),
            dataset_fingerprint=sub.get("dataset_fingerprint", data.get("dataset_fingerprint")),
            release_id=sub.get("release_id", data.get("release_id")),
        )


@dataclass
class ModelArtifact:
    """Bundle cấu hình để detector rebuild đúng preprocessing và policy."""

    metadata: ModelMetadata
    threshold_policy: ThresholdPolicy
    preprocessing: PreprocessingConfig
    coreset_info: dict[str, Any]
    scoring: dict[str, Any] = field(default_factory=lambda: {"method": "percentile", "percentile": 99.0, "smooth_sigma": 1.0})
    calibration: dict[str, Any] = field(default_factory=dict)
    smooth_sigma: float = 1.0
    dataset_fingerprint: str | None = None
    capture_contract: dict[str, Any] = field(default_factory=dict)
    inspection_policy: dict[str, Any] = field(default_factory=lambda: {"auto_pass_only": True})
    reference_manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Tạo payload config.json."""
        return {
            "model": self.metadata.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "scoring": self.scoring,
            "coreset": self.coreset_info,
            "calibration": self.calibration,
            "thresholds": self.threshold_policy.to_dict(),
            "capture_contract": self.capture_contract,
            "inspection_policy": self.inspection_policy,
            "reference_manifest": self.reference_manifest,
            "dataset_fingerprint": self.dataset_fingerprint or self.metadata.dataset_fingerprint,
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
            "threshold": self.threshold_policy.auto_pass_threshold,
            "auto_pass_threshold": self.threshold_policy.auto_pass_threshold,
            "review_threshold": self.threshold_policy.review_threshold,
            "pixel_threshold": self.threshold_policy.pixel_threshold,
            "smooth_sigma": self.smooth_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArtifact":
        """Đọc ModelArtifact từ schema mới hoặc schema cũ."""
        metadata = ModelMetadata.from_dict(data)
        prep_data = data.get("preprocessing", {})
        preprocessing = PreprocessingConfig.from_dict(prep_data) if prep_data else PreprocessingConfig()
        scoring_data = data.get("scoring", {})
        smooth_sigma = float(scoring_data.get("smooth_sigma", data.get("smooth_sigma", 1.0)))
        scoring = {
            "method": scoring_data.get("method", "percentile"),
            "percentile": float(scoring_data.get("percentile", data.get("scoring_percentile", 99.0))),
            "smooth_sigma": smooth_sigma,
        }
        return cls(
            metadata=metadata,
            threshold_policy=ThresholdPolicy.from_dict(data),
            preprocessing=preprocessing,
            coreset_info=dict(data.get("coreset", {})),
            scoring=scoring,
            calibration=dict(data.get("calibration", {})),
            smooth_sigma=smooth_sigma,
            dataset_fingerprint=data.get("dataset_fingerprint", metadata.dataset_fingerprint),
            capture_contract=dict(data.get("capture_contract", {})),
            inspection_policy=dict(data.get("inspection_policy", {"auto_pass_only": True})),
            reference_manifest=dict(data.get("reference_manifest", {})),
        )

    def save(self, category_dir: str | Path) -> None:
        """Ghi config.json; memory bank và split manifest do trainer ghi riêng."""
        target = Path(category_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, category_dir: str | Path) -> "ModelArtifact":
        """Đọc config.json từ release."""
        target = Path(category_dir) / "config.json"
        if not target.exists():
            raise FileNotFoundError(f"Không tìm thấy config.json: {target}")
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


def write_integrity_manifest(
    release_dir: str | Path,
    artifact: ModelArtifact,
    dataset_sha256: str | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Ghi manifest.json và hash các file runtime quan trọng."""
    release = Path(release_dir)
    files = {}
    for name in ("memory_bank.npy", "config.json", "split_manifest.json", "reference_manifest.json"):
        path = release / name
        if path.exists():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {
        "manifest_version": 1,
        "category": artifact.metadata.category,
        "model_version": artifact.metadata.model_version,
        "release_id": artifact.metadata.release_id,
        "dataset_sha256": dataset_sha256 or artifact.dataset_fingerprint,
        "code_commit": code_commit,
        "torch_version": _package_version("torch"),
        "torchvision_version": _package_version("torchvision"),
        "backbone": artifact.metadata.backbone,
        "backbone_weights": artifact.metadata.weights,
        "files": files,
    }
    (release / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def verify_artifact_integrity(release_dir: str | Path, strict: bool = True) -> None:
    """Kiểm tra hash runtime trước khi model được coi là ready."""
    release = Path(release_dir)
    manifest_path = release / "manifest.json"
    config_path = release / "config.json"
    if not manifest_path.exists():
        artifact = ModelArtifact.load(release)
        if strict and artifact.metadata.artifact_schema_version >= 5:
            raise ValueError(f"Release thiếu manifest integrity: {manifest_path}")
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in data.get("files", {}).items():
        path = release / name
        if not path.exists():
            raise ValueError(f"Release thiếu file '{name}'.")
        actual = sha256_file(path)
        if actual != expected.get("sha256"):
            raise ValueError(f"SHA256 mismatch cho artifact '{name}'.")
    if not config_path.exists():
        raise ValueError("Release thiếu config.json.")


def _package_version(package_name: str) -> str | None:
    """Lấy version package nếu runtime đã cài."""
    try:
        module = __import__(package_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None
