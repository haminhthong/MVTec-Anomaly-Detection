"""Registry resolve release theo category/line, không fallback chéo category."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact_resolver import ModelNotFoundError, resolve_artifact_dir
from ..path_safety import ensure_safe_segment

if TYPE_CHECKING:
    from ..inference.detector import AnomalyDetector


class ModelRegistry:
    """Quản lý discovery, resolve và cache detector theo category."""

    def __init__(self, base_dir: str | Path = "models") -> None:
        self.base_dir: Path = Path(base_dir)
        self._cached_detectors: dict[str, AnomalyDetector] = {}

    def list_categories(self) -> list[str]:
        """Liệt kê category có artifact hợp lệ hoặc có production pointer."""
        if not self.base_dir.exists():
            return []

        categories: set[str] = set()
        production_path = self.base_dir / "production.json"
        if production_path.exists():
            try:
                production = json.loads(production_path.read_text(encoding="utf-8"))
                category_entries = production.get("categories", {}) if isinstance(production, dict) else {}
                if isinstance(category_entries, dict):
                    for raw_category in category_entries:
                        try:
                            category = ensure_safe_segment(str(raw_category), "category")
                            resolve_artifact_dir(model_root=self.base_dir, category=category)
                        except (ModelNotFoundError, ValueError):
                            continue
                        categories.add(category)
            except (OSError, json.JSONDecodeError):
                # Resolve cụ thể sẽ báo lỗi rõ hơn; list không làm service crash.
                pass
        for p in self.base_dir.iterdir():
            if p.is_dir() and not p.name.startswith((".", "_")):
                cfg = p / "config.json"
                mem = p / "memory_bank.npy"
                legacy_mem = p / "memory.npy"
                if cfg.exists() and (mem.exists() or legacy_mem.exists()):
                    categories.add(p.name)

        return sorted(categories)

    def resolve_line_target(self, line_id: str) -> tuple[str, str | None]:
        """Resolve line thành category và release_id đã đăng ký."""
        if not line_id or not line_id.strip():
            raise ValueError("line_id không được để trống.")
        normalized_line_id = line_id.strip()
        pointer_path = self.base_dir / "production.json"
        if not pointer_path.exists():
            raise ModelNotFoundError(f"Chưa cấu hình line_id '{line_id}'.")
        try:
            data = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelNotFoundError(f"production.json không hợp lệ: {exc}") from exc
        line_entries = data.get("lines", {}) if isinstance(data, dict) else {}
        if not isinstance(line_entries, dict):
            raise ModelNotFoundError("production.json phải chứa object 'lines'.")
        entry = line_entries.get(normalized_line_id)
        if isinstance(entry, str):
            try:
                return ensure_safe_segment(entry, "category"), None
            except ValueError as exc:
                raise ModelNotFoundError(str(exc)) from exc
        if isinstance(entry, dict) and entry.get("category"):
            release_id = entry.get("release_id")
            try:
                mapped_category = ensure_safe_segment(str(entry["category"]), "category")
                safe_release_id = ensure_safe_segment(str(release_id), "release_id") if release_id else None
            except ValueError as exc:
                raise ModelNotFoundError(str(exc)) from exc
            return mapped_category, safe_release_id
        raise ModelNotFoundError(f"Không có mapping cho line_id '{normalized_line_id}'.")

    def resolve_line(self, line_id: str) -> str:
        """Resolve line_id server-side và trả về category của line."""
        return self.resolve_line_target(line_id)[0]

    def resolve_category_dir(self, category: str) -> Path:
        """Resolve và kiểm tra release artifact của một category."""
        return resolve_artifact_dir(model_root=self.base_dir, category=category)

    def get_metadata(self, category: str) -> dict[str, Any]:
        """Đọc config.json của release đang được trỏ cho category."""
        cat_dir = self.resolve_category_dir(category)
        cfg_path = cat_dir / "config.json"
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    def version(self, category: str) -> str:
        """Lấy model_version từ metadata nested của release."""
        try:
            meta = self.get_metadata(category)
            model_meta = meta.get("model", {})
            version = model_meta.get("model_version", meta.get("model_version", meta.get("version", "unknown")))
            return str(version)
        except ModelNotFoundError:
            return "not_trained"

    def get_detector(self, category: str, line_id: str | None = None) -> AnomalyDetector:
        """Lấy detector đúng release, tự làm mới cache khi production pointer đổi."""
        target_dir: Path | None = None
        cache_key = category
        normalized_line_id: str | None = None
        if line_id:
            normalized_line_id = line_id.strip()
            mapped_category, release_id = self.resolve_line_target(normalized_line_id)
            if mapped_category != category:
                raise ValueError(f"line_id '{line_id}' không map tới category '{category}'.")
            if release_id:
                releases_root = (self.base_dir / "releases").resolve()
                release_dir = (releases_root / release_id).resolve()
                if release_dir == releases_root or releases_root not in release_dir.parents:
                    raise ModelNotFoundError(f"release_id '{release_id}' trỏ ra ngoài thư mục releases.")
                target_dir = resolve_artifact_dir(model_root=release_dir)
                cache_key = f"line:{normalized_line_id}:{release_id}"

        # Line entry dạng legacy chỉ lưu category nên cần pointer category;
        # line entry có release_id đã resolve ở trên thì không phụ thuộc pointer
        # mutable của category.
        if target_dir is None:
            target_dir = self.resolve_category_dir(category)

        cached = self._cached_detectors.get(cache_key)
        if cached is not None and Path(cached.model_dir).resolve() == target_dir.resolve():
            return cached

        from ..inference.detector import AnomalyDetector

        if normalized_line_id and cache_key.startswith("line:"):
            detector = AnomalyDetector(model_dir=target_dir)
        else:
            # Truyền base_dir để resolver lại production pointer đúng category.
            detector = AnomalyDetector(model_dir=self.base_dir, category=category)
        self._cached_detectors[cache_key] = detector
        return detector

    def clear_cache(self) -> None:
        """Xóa cache detector để nạp lại production release."""
        self._cached_detectors.clear()
