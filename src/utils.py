"""Tiện ích dùng chung và cấu hình logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .inference.localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)
from .model.coreset import greedy_coreset, select_coreset_indices
from .training.trainer import set_seed

LOGGER: logging.Logger = logging.getLogger("mvtec_anomaly_detection")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Lưu dictionary thành file JSON UTF-8 có format dễ đọc."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "set_seed",
    "save_json",
    "select_coreset_indices",
    "greedy_coreset",
    "apply_heatmap_smoothing",
    "compute_anomalous_area_ratio",
    "create_heatmap_overlay_b64",
    "LOGGER",
]
