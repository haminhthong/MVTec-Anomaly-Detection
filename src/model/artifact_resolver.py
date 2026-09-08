"""Resolve artifact nghiêm ngặt theo category và production pointer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..path_safety import ensure_safe_segment


class ModelNotFoundError(FileNotFoundError):
    """Artifact category không tồn tại hoặc không hợp lệ."""


def _resolve_inside(root: Path, relative_value: str, field_name: str) -> Path:
    """Resolve pointer và bảo đảm nó vẫn nằm dưới model root."""
    if not relative_value or Path(relative_value).is_absolute():
        raise ModelNotFoundError(f"{field_name} phải là đường dẫn tương đối bên trong model root.")
    root_resolved = root.resolve()
    candidate = (root / relative_value).resolve()
    if candidate == root_resolved or root_resolved not in candidate.parents:
        raise ModelNotFoundError(f"{field_name} trỏ ra ngoài model root.")
    return candidate


def resolve_artifact_dir(
    model_root: str | Path = "models",
    category: str | None = None,
) -> Path:
    """Resolve và kiểm tra config/memory bank của đúng category."""
    root_path = Path(model_root)

    # Nếu root_path đã là thư mục category (ví dụ models/bottle).
    if category is None:
        # Kiểm tra root_path có phải artifact category hợp lệ không.
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
        try:
            category = ensure_safe_segment(str(inferred_category), "category")
        except ValueError as exc:
            raise ModelNotFoundError(str(exc)) from exc
        cat_dir = root_path
    else:
        try:
            category = ensure_safe_segment(category, "category")
        except ValueError as exc:
            raise ModelNotFoundError(str(exc)) from exc
        # Production pointer ưu tiên release immutable thay vì alias mutable.
        production_path = root_path / "production.json"
        pointed_dir: Path | None = None
        if production_path.exists():
            try:
                production = json.loads(production_path.read_text(encoding="utf-8"))
                if not isinstance(production, dict) or not isinstance(production.get("categories", {}), dict):
                    raise ModelNotFoundError("production.json phải chứa object 'categories'.")
                pointer = production["categories"].get(category)
                if isinstance(pointer, dict):
                    pointer = pointer.get("path") or pointer.get("release")
                if pointer:
                    pointed_dir = _resolve_inside(root_path, str(pointer), "production pointer")
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelNotFoundError(f"production.json không hợp lệ: {exc}") from exc

        if pointed_dir is not None and (pointed_dir / "config.json").exists():
            cat_dir = pointed_dir
        # Nếu không có pointer thì kiểm tra thư mục category trực tiếp.
        elif root_path.name == category and (root_path / "config.json").exists():
            cat_dir = root_path
        else:
            cat_dir = root_path / category

    if not cat_dir.exists() or not cat_dir.is_dir():
        raise ModelNotFoundError(
            f"Model artifacts for category '{category}' do not exist at '{cat_dir}'. "
            "No cross-category or root fallback is permitted."
        )

    # 1. Kiểm tra config.json.
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

    # 2. Kiểm tra file memory bank.
    memory_path = cat_dir / "memory_bank.npy"
    legacy_path = cat_dir / "memory.npy"
    if not memory_path.exists() and not legacy_path.exists():
        raise ModelNotFoundError(
            f"Missing 'memory_bank.npy' for category '{category}' at '{cat_dir}'."
        )

    return cat_dir
