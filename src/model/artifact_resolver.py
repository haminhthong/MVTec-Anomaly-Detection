"""Artifact Resolver enforcing strict category isolation and validation.

Eliminates loose fallback behavior:
- Target category must be explicitly requested and non-empty.
- Artifacts must reside in <model_root>/<category>/
- config.json must exist and its recorded category must match requested category exactly.
- memory_bank.npy (or legacy memory.npy) must exist.
- Never falls back to root directory or unrelated categories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelNotFoundError(FileNotFoundError):
    """Raised when model artifacts for a requested category cannot be found or are invalid."""


def resolve_artifact_dir(
    model_root: str | Path = "models",
    category: str | None = None,
) -> Path:
    """Resolve and strictly validate the artifact directory for a category.

    Args:
        model_root: Base models directory (e.g. 'models') or specific category folder.
        category: Name of product category (e.g. 'bottle').

    Returns:
        Path: Validated path to category artifact directory.

    Raises:
        ValueError: If category name is empty or invalid.
        ModelNotFoundError: If artifact directory, config.json, or memory bank is missing,
            or if config.json category does not match requested category.
    """
    root_path = Path(model_root)

    # If root_path itself is the category dir (e.g. models/bottle)
    if category is None or not category.strip():
        # Check if root_path itself is a valid category dir
        cfg_file = root_path / "config.json"
        if not cfg_file.exists():
            raise ValueError(
                "Category name must be provided, or model_root must point directly to a category artifact directory with config.json."
            )
        try:
            raw_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            inferred_category = raw_cfg.get("model", {}).get("category") or raw_cfg.get("category")
        except Exception as exc:
            raise ModelNotFoundError(f"Failed to parse config.json in '{root_path}': {exc}") from exc

        if not inferred_category:
            raise ModelNotFoundError(
                f"Missing 'category' field in '{cfg_file}'. Cannot determine category."
            )
        category = str(inferred_category).strip()
        cat_dir = root_path
    else:
        category = category.strip()
        # Direct category folder check
        if root_path.name == category and (root_path / "config.json").exists():
            cat_dir = root_path
        else:
            cat_dir = root_path / category

    if not cat_dir.exists() or not cat_dir.is_dir():
        raise ModelNotFoundError(
            f"Model artifacts for category '{category}' do not exist at '{cat_dir}'. "
            "No cross-category or root fallback is permitted."
        )

    # 1. Verify config.json
    config_path = cat_dir / "config.json"
    if not config_path.exists():
        raise ModelNotFoundError(
            f"Missing 'config.json' for category '{category}' at '{cat_dir}'."
        )

    try:
        raw_config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelNotFoundError(
            f"Corrupted or invalid config.json for category '{category}' at '{config_path}': {exc}"
        ) from exc

    recorded_cat = raw_config.get("model", {}).get("category") or raw_config.get("category")
    if recorded_cat is not None and str(recorded_cat).strip() != category:
        raise ModelNotFoundError(
            f"Category mismatch in artifact '{config_path}': expected '{category}', "
            f"but config specifies '{recorded_cat}'."
        )

    # 2. Verify memory bank array
    memory_path = cat_dir / "memory_bank.npy"
    legacy_path = cat_dir / "memory.npy"
    if not memory_path.exists() and not legacy_path.exists():
        raise ModelNotFoundError(
            f"Missing 'memory_bank.npy' for category '{category}' at '{cat_dir}'."
        )

    return cat_dir
