from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model.artifacts import resolve_artifact_dir


def _ready_artifact(path: Path, category: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"category": category}), encoding="utf-8"
    )
    (path / "memory_bank.npy").write_bytes(b"placeholder")


def test_category_scoped_artifact_has_priority_over_legacy_root(tmp_path: Path) -> None:
    _ready_artifact(tmp_path, "legacy")
    _ready_artifact(tmp_path / "bottle", "bottle")
    assert resolve_artifact_dir(tmp_path, "bottle") == tmp_path / "bottle"


def test_legacy_root_must_match_requested_category(tmp_path: Path) -> None:
    _ready_artifact(tmp_path, "bottle")
    with pytest.raises(FileNotFoundError):
        resolve_artifact_dir(tmp_path, "cable")


def test_multiple_categories_require_explicit_category(tmp_path: Path) -> None:
    _ready_artifact(tmp_path / "bottle", "bottle")
    _ready_artifact(tmp_path / "cable", "cable")
    with pytest.raises(ValueError, match="Multiple model categories"):
        resolve_artifact_dir(tmp_path)
