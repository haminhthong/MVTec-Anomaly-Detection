"""Unit tests for ModelRegistry and strict category isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.model.registry import ModelNotFoundError, ModelRegistry


def test_registry_strict_category_resolution(tmp_path: Path) -> None:
    """Test registry resolves only exact existing category, no cross-category fallback."""
    models_dir = tmp_path / "models"
    bottle_dir = models_dir / "bottle"
    bottle_dir.mkdir(parents=True, exist_ok=True)

    (bottle_dir / "config.json").write_text(
        json.dumps({"category": "bottle", "version": "1.0.0", "threshold": 2.5}),
        encoding="utf-8",
    )
    np.save(bottle_dir / "memory_bank.npy", np.zeros((10, 384), dtype=np.float32))

    registry = ModelRegistry(base_dir=models_dir)

    # 1. Existing category resolves successfully
    resolved = registry.resolve_category_dir("bottle")
    assert resolved == bottle_dir

    # 2. Non-existent category MUST raise ModelNotFoundError (NO fallback to bottle)
    with pytest.raises(ModelNotFoundError, match="Model artifacts for category 'cable' do not exist"):
        registry.resolve_category_dir("cable")


def test_registry_empty_dir(tmp_path: Path) -> None:
    """Test registry returns empty list for empty base dir."""
    registry = ModelRegistry(base_dir=tmp_path / "empty_models")
    assert registry.list_categories() == []


def test_registry_rejects_path_traversal_and_external_pointer(tmp_path: Path) -> None:
    """Category và production pointer không được thoát khỏi model root."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry = ModelRegistry(base_dir=models_dir)

    with pytest.raises(ModelNotFoundError):
        registry.resolve_category_dir("../outside")

    outside = tmp_path / "outside"
    outside.mkdir()
    (models_dir / "production.json").write_text(
        json.dumps({"categories": {"bottle": "../outside"}}), encoding="utf-8"
    )
    with pytest.raises(ModelNotFoundError, match="production pointer"):
        registry.resolve_category_dir("bottle")
