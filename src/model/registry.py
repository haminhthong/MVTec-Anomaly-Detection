"""ModelRegistry managing multi-category model resolution, caching, and lifecycle.

Enforces strict category isolation:
- Models are strictly scoped under models/<category>/
- NO cross-category fallback (e.g. asking for 'cable' will never load 'bottle')
- Raises ModelNotFoundError if the requested category is not trained
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact_resolver import ModelNotFoundError, resolve_artifact_dir

if TYPE_CHECKING:
    from ..inference.detector import AnomalyDetector


class ModelRegistry:
    """Registry managing model discovery, resolution, and caching per category.

    Attributes:
        base_dir: Root directory containing category-scoped model artifacts.
        _cached_detectors: In-memory cache of instantiated AnomalyDetector objects.
    """

    def __init__(self, base_dir: str | Path = "models") -> None:
        self.base_dir: Path = Path(base_dir)
        self._cached_detectors: dict[str, AnomalyDetector] = {}

    def list_categories(self) -> list[str]:
        """List all category names with valid, trained model artifacts.

        Returns:
            list[str]: Alphabetically sorted list of available category names.
        """
        if not self.base_dir.exists():
            return []

        categories: set[str] = set()
        for p in self.base_dir.iterdir():
            if p.is_dir() and not p.name.startswith((".", "_")):
                cfg = p / "config.json"
                mem = p / "memory_bank.npy"
                legacy_mem = p / "memory.npy"
                if cfg.exists() and (mem.exists() or legacy_mem.exists()):
                    categories.add(p.name)

        return sorted(categories)

    def resolve_category_dir(self, category: str) -> Path:
        """Resolve and strictly validate the artifact directory for a category.

        Args:
            category: Name of product category (e.g. 'bottle').

        Returns:
            Path: Path to models/<category> directory.

        Raises:
            ModelNotFoundError: If the category directory or required artifacts do not exist.
        """
        return resolve_artifact_dir(model_root=self.base_dir, category=category)

    def get_metadata(self, category: str) -> dict[str, Any]:
        """Read config.json metadata for a specific category.

        Args:
            category: Name of product category.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        cat_dir = self.resolve_category_dir(category)
        cfg_path = cat_dir / "config.json"
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    def version(self, category: str) -> str:
        """Get model version string for category."""
        try:
            meta = self.get_metadata(category)
            return str(meta.get("model_version", meta.get("version", "unknown")))
        except ModelNotFoundError:
            return "not_trained"

    def get_detector(self, category: str) -> AnomalyDetector:
        """Retrieve AnomalyDetector instance for category (using cached instance if available).

        Args:
            category: Name of product category.

        Returns:
            AnomalyDetector: Instantiated detector.
        """
        if category in self._cached_detectors:
            return self._cached_detectors[category]

        from ..inference.detector import AnomalyDetector

        cat_dir = self.resolve_category_dir(category)
        detector = AnomalyDetector(model_dir=cat_dir, category=category)
        self._cached_detectors[category] = detector
        return detector

    def clear_cache(self) -> None:
        """Clear cached detector instances."""
        self._cached_detectors.clear()
