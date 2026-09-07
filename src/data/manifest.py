"""Các hợp đồng manifest và fingerprint dữ liệu cho pipeline anomaly detection.

Manifest normal và manifest locked-test được tách riêng để code xây model không
có lý do kỹ thuật nào phải đọc ảnh test hoặc ground-truth mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def sha256_file(path: str | Path) -> str:
    """Tính SHA256 thật trên nội dung file, không chỉ dựa vào tên và kích thước."""
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    """Chuẩn hóa đường dẫn tương đối để fingerprint ổn định giữa các hệ điều hành."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def build_file_records(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Tạo record provenance gồm đường dẫn tương đối, SHA256 và số byte."""
    records: list[dict[str, Any]] = []
    unique_paths = sorted(
        {Path(item) for item in paths},
        key=lambda item: _relative_path(root, item),
    )
    for path in unique_paths:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file trong manifest: {path}")
        records.append(
            {
                "relative_path": _relative_path(root, path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def fingerprint_records(category: str, records: Iterable[dict[str, Any]]) -> str:
    """Tạo fingerprint từ danh sách record đã sắp xếp theo đường dẫn."""
    digest = hashlib.sha256()
    digest.update(category.encode("utf-8"))
    for record in sorted(records, key=lambda item: str(item["relative_path"])):
        payload = f"{record['relative_path']}:{record['sha256']}:{record['bytes']}"
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _restore_path(root: Path, value: str | Path) -> Path:
    """Khôi phục path đã lưu dưới dạng tương đối hoặc tuyệt đối."""
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass
class NormalReferenceManifest:
    """Manifest chỉ chứa ảnh normal dùng cho reference/dev/calibration."""

    category: str
    root_path: Path
    train_good: list[Path] = field(default_factory=list)
    fingerprint: str | None = None
    source: str = "MVTec AD"
    source_version: str = "unknown"
    download_date: str | None = None
    license: str | None = "CC BY-NC-SA 4.0"
    file_records: list[dict[str, Any]] = field(default_factory=list)

    manifest_type: str = field(default="normal_reference", init=False)

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path)
        self.train_good = [Path(path) for path in self.train_good]
        if not self.file_records and self.train_good:
            self.file_records = build_file_records(self.root_path, self.train_good)
        if self.fingerprint is None:
            self.fingerprint = fingerprint_records(self.category, self.file_records)

    @property
    def total_train(self) -> int:
        """Số ảnh normal trong reference source."""
        return len(self.train_good)

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest với path tương đối và provenance đầy đủ."""
        return {
            "manifest_type": self.manifest_type,
            "category": self.category,
            "root_path": str(self.root_path),
            "fingerprint": self.fingerprint,
            "source": self.source,
            "source_version": self.source_version,
            "download_date": self.download_date,
            "license": self.license,
            "counts": {"train_good": self.total_train},
            "train_good": [_relative_path(self.root_path, path) for path in self.train_good],
            "files": self.file_records,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalReferenceManifest":
        """Khôi phục normal reference manifest từ JSON."""
        root = Path(data["root_path"])
        return cls(
            category=str(data["category"]),
            root_path=root,
            train_good=[_restore_path(root, path) for path in data.get("train_good", [])],
            fingerprint=data.get("fingerprint"),
            source=str(data.get("source", "MVTec AD")),
            source_version=str(data.get("source_version", "unknown")),
            download_date=data.get("download_date"),
            license=data.get("license"),
            file_records=list(data.get("files", [])),
        )

    def save(self, path: str | Path) -> None:
        """Ghi manifest JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NormalReferenceManifest":
        """Đọc manifest normal reference."""
        target = Path(path)
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


@dataclass
class LockedEvaluationManifest:
    """Manifest chỉ chứa official test và mask, dùng ở bước đánh giá cuối."""

    category: str
    root_path: Path
    test_good: list[Path] = field(default_factory=list)
    test_defect: dict[str, list[Path]] = field(default_factory=dict)
    masks: dict[str, Path] = field(default_factory=dict)
    fingerprint: str | None = None
    source: str = "MVTec AD"
    source_version: str = "unknown"
    download_date: str | None = None
    license: str | None = "CC BY-NC-SA 4.0"
    file_records: list[dict[str, Any]] = field(default_factory=list)

    manifest_type: str = field(default="locked_evaluation", init=False)

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path)
        self.test_good = [Path(path) for path in self.test_good]
        self.test_defect = {
            str(kind): [Path(path) for path in paths]
            for kind, paths in self.test_defect.items()
        }
        self.masks = {str(key): Path(value) for key, value in self.masks.items()}
        if not self.file_records:
            all_paths = list(self.test_good)
            all_paths.extend(path for paths in self.test_defect.values() for path in paths)
            all_paths.extend(self.masks.values())
            self.file_records = build_file_records(self.root_path, all_paths)
        if self.fingerprint is None:
            self.fingerprint = fingerprint_records(self.category, self.file_records)

    @property
    def total_test_good(self) -> int:
        """Số ảnh normal trong locked test."""
        return len(self.test_good)

    @property
    def total_test_defect(self) -> int:
        """Tổng số ảnh lỗi trong locked test."""
        return sum(len(paths) for paths in self.test_defect.values())

    @property
    def total_test(self) -> int:
        """Tổng số ảnh trong locked test."""
        return self.total_test_good + self.total_test_defect

    @property
    def defect_types(self) -> list[str]:
        """Danh sách defect type, sắp xếp ổn định."""
        return sorted(self.test_defect)

    def get_all_test_items(self) -> list[tuple[Path, int, Path | None, str | None]]:
        """Trả về ảnh, nhãn, mask và defect type cho evaluator."""
        items: list[tuple[Path, int, Path | None, str | None]] = [
            (path, 0, None, None) for path in sorted(self.test_good)
        ]
        for defect_type in self.defect_types:
            for path in sorted(self.test_defect[defect_type]):
                items.append((path, 1, self.masks.get(str(path)), defect_type))
        return items

    def to_dict(self) -> dict[str, Any]:
        """Serialize locked test manifest với path tương đối."""
        return {
            "manifest_type": self.manifest_type,
            "category": self.category,
            "root_path": str(self.root_path),
            "fingerprint": self.fingerprint,
            "source": self.source,
            "source_version": self.source_version,
            "download_date": self.download_date,
            "license": self.license,
            "counts": {
                "test_good": self.total_test_good,
                "test_defect": self.total_test_defect,
                "total_test": self.total_test,
                "defect_types": self.defect_types,
            },
            "test_good": [_relative_path(self.root_path, path) for path in self.test_good],
            "test_defect": {
                kind: [_relative_path(self.root_path, path) for path in paths]
                for kind, paths in self.test_defect.items()
            },
            "masks": {
                _relative_path(self.root_path, Path(image_path)): _relative_path(self.root_path, mask)
                for image_path, mask in self.masks.items()
            },
            "files": self.file_records,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LockedEvaluationManifest":
        """Khôi phục locked evaluation manifest từ JSON."""
        root = Path(data["root_path"])
        test_good = [_restore_path(root, path) for path in data.get("test_good", [])]
        test_defect = {
            kind: [_restore_path(root, path) for path in paths]
            for kind, paths in data.get("test_defect", {}).items()
        }
        masks = {
            str(_restore_path(root, image_path)): _restore_path(root, mask_path)
            for image_path, mask_path in data.get("masks", {}).items()
        }
        return cls(
            category=str(data["category"]),
            root_path=root,
            test_good=test_good,
            test_defect=test_defect,
            masks=masks,
            fingerprint=data.get("fingerprint"),
            source=str(data.get("source", "MVTec AD")),
            source_version=str(data.get("source_version", "unknown")),
            download_date=data.get("download_date"),
            license=data.get("license"),
            file_records=list(data.get("files", [])),
        )

    def save(self, path: str | Path) -> None:
        """Ghi manifest locked evaluation JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LockedEvaluationManifest":
        """Đọc locked evaluation manifest."""
        target = Path(path)
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


@dataclass
class DatasetManifest:
    """Manifest ghép để tương thích CLI cũ; training không cần dùng phần test."""

    category: str
    root_path: Path
    train_good: list[Path] = field(default_factory=list)
    test_good: list[Path] = field(default_factory=list)
    test_defect: dict[str, list[Path]] = field(default_factory=dict)
    masks: dict[str, Path] = field(default_factory=dict)
    fingerprint: str | None = None
    source: str = "MVTec AD"
    source_version: str = "unknown"
    download_date: str | None = None
    license: str | None = "CC BY-NC-SA 4.0"
    file_records: list[dict[str, Any]] = field(default_factory=list)

    manifest_type: str = field(default="combined", init=False)

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path)
        self.train_good = [Path(path) for path in self.train_good]
        self.test_good = [Path(path) for path in self.test_good]
        self.test_defect = {
            str(kind): [Path(path) for path in paths]
            for kind, paths in self.test_defect.items()
        }
        self.masks = {str(key): Path(value) for key, value in self.masks.items()}
        if not self.file_records:
            all_paths = list(self.train_good) + list(self.test_good)
            all_paths.extend(path for paths in self.test_defect.values() for path in paths)
            all_paths.extend(self.masks.values())
            self.file_records = build_file_records(self.root_path, all_paths)
        if self.fingerprint is None:
            self.fingerprint = fingerprint_records(self.category, self.file_records)

    @property
    def total_train(self) -> int:
        """Tổng số ảnh normal dùng làm nguồn reference."""
        return len(self.train_good)

    @property
    def total_test_good(self) -> int:
        """Tổng số ảnh test normal."""
        return len(self.test_good)

    @property
    def total_test_defect(self) -> int:
        """Tổng số ảnh test lỗi."""
        return sum(len(paths) for paths in self.test_defect.values())

    @property
    def total_test(self) -> int:
        """Tổng số ảnh test."""
        return self.total_test_good + self.total_test_defect

    @property
    def defect_types(self) -> list[str]:
        """Danh sách loại lỗi."""
        return sorted(self.test_defect)

    def get_all_test_items(self) -> list[tuple[Path, int, Path | None, str | None]]:
        """Trả về ảnh test kèm loại defect để tạo các lát chẩn đoán."""
        items: list[tuple[Path, int, Path | None, str | None]] = [
            (path, 0, None, None) for path in sorted(self.test_good)
        ]
        for defect_type in self.defect_types:
            for path in sorted(self.test_defect[defect_type]):
                items.append((path, 1, self.masks.get(str(path)), defect_type))
        return items

    def get_all_test_paths(self) -> list[tuple[Path, int, Path | None]]:
        """API cũ: trả về ảnh test, nhãn và mask."""
        return [(path, label, mask) for path, label, mask, _ in self.get_all_test_items()]

    def to_dict(self) -> dict[str, Any]:
        """Serialize combined manifest."""
        return {
            "manifest_type": self.manifest_type,
            "category": self.category,
            "root_path": str(self.root_path),
            "fingerprint": self.fingerprint,
            "source": self.source,
            "source_version": self.source_version,
            "download_date": self.download_date,
            "license": self.license,
            "counts": {
                "train_good": self.total_train,
                "test_good": self.total_test_good,
                "test_defect": self.total_test_defect,
                "total_test": self.total_test,
                "defect_types": self.defect_types,
            },
            "train_good": [_relative_path(self.root_path, path) for path in self.train_good],
            "test_good": [_relative_path(self.root_path, path) for path in self.test_good],
            "test_defect": {
                kind: [_relative_path(self.root_path, path) for path in paths]
                for kind, paths in self.test_defect.items()
            },
            "masks": {
                _relative_path(self.root_path, Path(image_path)): _relative_path(self.root_path, mask)
                for image_path, mask in self.masks.items()
            },
            "files": self.file_records,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        """Deserialize combined manifest."""
        root = Path(data["root_path"])
        masks = {
            str(_restore_path(root, image_path)): _restore_path(root, mask_path)
            for image_path, mask_path in data.get("masks", {}).items()
        }
        return cls(
            category=str(data["category"]),
            root_path=root,
            train_good=[_restore_path(root, path) for path in data.get("train_good", [])],
            test_good=[_restore_path(root, path) for path in data.get("test_good", [])],
            test_defect={
                kind: [_restore_path(root, path) for path in paths]
                for kind, paths in data.get("test_defect", {}).items()
            },
            masks=masks,
            fingerprint=data.get("fingerprint"),
            source=str(data.get("source", "MVTec AD")),
            source_version=str(data.get("source_version", "unknown")),
            download_date=data.get("download_date"),
            license=data.get("license"),
            file_records=list(data.get("files", [])),
        )

    def save(self, path: str | Path) -> None:
        """Save manifest to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        """Load combined manifest from JSON."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Không tìm thấy manifest: {target}")
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
