"""Lazy multi-category model registry for the serving layer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import load_artifact_config, resolve_artifact_dir

if TYPE_CHECKING:
    from ..inference.detector import AnomalyDetector


class ModelRegistry:
    def __init__(self, base_dir: str | Path = "models") -> None:
        self.base_dir = Path(base_dir)
        self._cached_detectors: dict[str, AnomalyDetector] = {}

    def list_categories(self) -> list[str]:
        if not self.base_dir.exists():
            return []

        categories: set[str] = set()
        for path in self.base_dir.iterdir():
            if not path.is_dir():
                continue
            try:
                metadata = load_artifact_config(path)
            except (FileNotFoundError, ValueError):
                continue
            categories.add(str(metadata.get("category", path.name)))

        # Keep legacy root artifacts discoverable without letting them override a
        # category-scoped artifact with the same category.
        try:
            root_metadata = load_artifact_config(self.base_dir)
            categories.add(str(root_metadata.get("category", "bottle")))
        except (FileNotFoundError, ValueError):
            pass

        return sorted(categories)

    def resolve_category_dir(self, category: str = "bottle") -> Path:
        return resolve_artifact_dir(self.base_dir, category=category)

    def get_metadata(self, category: str = "bottle") -> dict[str, Any]:
        return load_artifact_config(self.resolve_category_dir(category))

    def version(self, category: str = "bottle") -> str:
        try:
            return str(self.get_metadata(category).get("version", "unknown"))
        except (FileNotFoundError, ValueError):
            return "not_trained"

    def get_detector(self, category: str = "bottle") -> AnomalyDetector:
        if category in self._cached_detectors:
            return self._cached_detectors[category]

        from ..inference.detector import AnomalyDetector

        detector = AnomalyDetector(model_dir=self.base_dir, category=category)
        self._cached_detectors[category] = detector
        return detector
