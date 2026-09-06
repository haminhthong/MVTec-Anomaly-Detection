"""DatasetManifest contract and serialization for MVTec AD datasets.

Serves as the single canonical source of truth across all pipeline stages:
Data Validation -> Model Building -> Evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@dataclass
class DatasetManifest:
    """Canonical representation of an MVTec AD category dataset.

    Serves as the single source of truth for all downstream pipelines.
    """

    category: str
    root_path: Path
    train_good: list[Path] = field(default_factory=list)
    test_good: list[Path] = field(default_factory=list)
    test_defect: dict[str, list[Path]] = field(default_factory=dict)
    masks: dict[str, Path] = field(default_factory=dict)
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path)
        self.train_good = [Path(p) for p in self.train_good]
        self.test_good = [Path(p) for p in self.test_good]
        self.test_defect = {
            k: [Path(p) for p in v] for k, v in self.test_defect.items()
        }
        self.masks = {k: Path(v) for k, v in self.masks.items()}
        if self.fingerprint is None and self.train_good:
            self.fingerprint = self.compute_fingerprint()

    @property
    def total_train(self) -> int:
        """Total normal images for training."""
        return len(self.train_good)

    @property
    def total_test_good(self) -> int:
        """Total normal images in test set."""
        return len(self.test_good)

    @property
    def total_test_defect(self) -> int:
        """Total defective images across all defect types."""
        return sum(len(paths) for paths in self.test_defect.values())

    @property
    def total_test(self) -> int:
        """Total images in test set."""
        return self.total_test_good + self.total_test_defect

    @property
    def defect_types(self) -> list[str]:
        """List of defect types for this category."""
        return sorted(self.test_defect.keys())

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA256 fingerprint over all file paths and sizes."""
        hasher = hashlib.sha256()
        hasher.update(self.category.encode("utf-8"))

        all_files: list[Path] = list(self.train_good) + list(self.test_good)
        for d_files in self.test_defect.values():
            all_files.extend(d_files)
        all_files.extend(self.masks.values())

        # Sort by relative path string to ensure cross-platform reproducibility
        entries: list[tuple[str, int]] = []
        for p in all_files:
            try:
                rel = str(p.relative_to(self.root_path)).replace("\\", "/")
                size = p.stat().st_size if p.exists() else 0
            except ValueError:
                rel = p.name
                size = p.stat().st_size if p.exists() else 0
            entries.append((rel, size))

        for rel, size in sorted(entries):
            hasher.update(f"{rel}:{size}".encode("utf-8"))

        return hasher.hexdigest()

    def get_all_test_paths(self) -> list[tuple[Path, int, Path | None]]:
        """Get flattened test items with label and ground truth mask.

        Returns:
            list[tuple[Path, int, Path | None]]:
                (image_path, is_defective [0 or 1], mask_path_or_None)
        """
        items: list[tuple[Path, int, Path | None]] = []
        for p in sorted(self.test_good):
            items.append((p, 0, None))

        for defect_type in sorted(self.test_defect.keys()):
            for p in sorted(self.test_defect[defect_type]):
                mask = self.masks.get(str(p))
                items.append((p, 1, mask))
        return items

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dictionary representation with relative paths."""
        return {
            "category": self.category,
            "root_path": str(self.root_path),
            "fingerprint": self.fingerprint or self.compute_fingerprint(),
            "counts": {
                "train_good": self.total_train,
                "test_good": self.total_test_good,
                "test_defect": self.total_test_defect,
                "total_test": self.total_test,
                "defect_types": self.defect_types,
            },
            "train_good": [str(p) for p in self.train_good],
            "test_good": [str(p) for p in self.test_good],
            "test_defect": {
                k: [str(p) for p in v] for k, v in self.test_defect.items()
            },
            "masks": {k: str(v) for k, v in self.masks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        """Deserialize DatasetManifest from dictionary."""
        return cls(
            category=str(data["category"]),
            root_path=Path(data["root_path"]),
            train_good=[Path(p) for p in data.get("train_good", [])],
            test_good=[Path(p) for p in data.get("test_good", [])],
            test_defect={
                k: [Path(p) for p in v]
                for k, v in data.get("test_defect", {}).items()
            },
            masks={k: Path(v) for k, v in data.get("masks", {}).items()},
            fingerprint=data.get("fingerprint"),
        )

    def save(self, path: str | Path) -> None:
        """Save manifest to JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        """Load manifest from JSON file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Manifest file not found: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_dict(data)
