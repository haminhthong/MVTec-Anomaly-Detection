"""Artifact layout helpers for category-scoped anomaly-detection models.

The project stores production artifacts under ``models/<category>/``.  Legacy
single-category artifacts at ``models/`` are still readable, but category-scoped
artifacts always take precedence to avoid loading a stale or wrong model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def category_artifact_dir(model_root: str | Path, category: str) -> Path:
    """Return the canonical artifact directory for one product category."""
    category = category.strip()
    if not category:
        raise ValueError("category must not be empty")
    return Path(model_root) / category


def _is_ready(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "memory_bank.npy").exists() or (path / "memory.npy").exists()
    )


def _read_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid model config at '{path / 'config.json'}': {exc}") from exc


def resolve_artifact_dir(
    model_root: str | Path = "models", category: str | None = None
) -> Path:
    """Resolve the model artifact directory without silently crossing categories.

    Resolution policy:
    1. If ``category`` is provided, prefer ``model_root/category``.
    2. A legacy root artifact is accepted only when its config category matches.
    3. Without ``category``, exactly one category-scoped artifact is auto-selected.
    4. If multiple category artifacts exist, the caller must choose a category.
    5. Legacy root artifacts are used only when no category-scoped artifact exists.
    """
    root = Path(model_root)

    if category:
        candidate = category_artifact_dir(root, category)
        if _is_ready(candidate):
            return candidate

        if _is_ready(root):
            legacy_cfg = _read_config(root)
            if legacy_cfg.get("category") == category:
                return root

        raise FileNotFoundError(
            f"No ready artifacts for category '{category}' under '{root}'. "
            "Run the training pipeline for that category first."
        )

    category_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and _is_ready(p)
    ) if root.exists() else []

    if len(category_dirs) == 1:
        return category_dirs[0]
    if len(category_dirs) > 1:
        names = ", ".join(p.name for p in category_dirs)
        raise ValueError(
            f"Multiple model categories are available ({names}). Specify --category explicitly."
        )
    if _is_ready(root):
        return root

    raise FileNotFoundError(
        f"No ready model artifacts found under '{root}'. Run the training pipeline first."
    )


def load_artifact_config(artifact_dir: str | Path) -> dict[str, Any]:
    """Load and validate the JSON metadata for a resolved artifact directory."""
    path = Path(artifact_dir)
    if not _is_ready(path):
        raise FileNotFoundError(f"Incomplete model artifacts at '{path}'.")
    return _read_config(path)
