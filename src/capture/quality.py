"""Kiểm tra chất lượng ảnh trước model anomaly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .contract import CaptureContract


@dataclass(frozen=True)
class CaptureQualityResult:
    """Kết quả quality gate có thể lưu cùng inspection."""

    valid: bool
    state: str
    reasons: tuple[str, ...]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize kết quả quality gate."""
        return {
            "valid": self.valid,
            "state": self.state,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }


def _blur_score(gray: np.ndarray) -> float:
    """Tính gradient energy đơn giản, không thêm dependency OpenCV."""
    if min(gray.shape) < 2:
        return 0.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.var(gx) + np.var(gy))


def validate_capture(image: Image.Image, contract: CaptureContract) -> CaptureQualityResult:
    """Trả về RECAPTURE_REQUIRED nếu ảnh không đạt input check."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    array = np.asarray(rgb, dtype=np.float32)
    gray = array.mean(axis=2)
    exposure = float(gray.mean() / 255.0)
    blur = _blur_score(gray)
    reasons: list[str] = []

    if contract.expected_width is not None and width != contract.expected_width:
        reasons.append("unexpected_width")
    if contract.expected_height is not None and height != contract.expected_height:
        reasons.append("unexpected_height")
    if contract.min_sharpness_score is not None and blur < contract.min_sharpness_score:
        reasons.append("blurred")
    if contract.min_exposure is not None and exposure < contract.min_exposure:
        reasons.append("under_exposed")
    if contract.max_exposure is not None and exposure > contract.max_exposure:
        reasons.append("over_exposed")
    if contract.roi is not None:
        x, y, roi_width, roi_height = contract.roi
        if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0 or x + roi_width > width or y + roi_height > height:
            reasons.append("invalid_roi")

    return CaptureQualityResult(
        valid=not reasons,
        state="CAPTURE_VALID" if not reasons else "RECAPTURE_REQUIRED",
        reasons=tuple(reasons),
        metrics={
            "width": float(width),
            "height": float(height),
            "exposure": exposure,
            "sharpness_score": blur,
        },
    )
