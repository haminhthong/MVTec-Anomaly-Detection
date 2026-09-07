"""Hợp đồng đầu vào camera cho một production line."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CaptureContract:
    """Các điều kiện hình ảnh tối thiểu trước anomaly scoring.

    Để ``None`` nghĩa là chưa cấu hình ràng buộc đó; không tự ý suy ra từ
    anomaly score vì lỗi camera không phải lỗi sản phẩm.
    """

    expected_width: int | None = None
    expected_height: int | None = None
    max_blur_score: float | None = None
    min_exposure: float | None = None
    max_exposure: float | None = None
    roi: tuple[int, int, int, int] | None = None
    orientation: str = "fixed"

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract vào config artifact."""
        payload = asdict(self)
        if self.roi is not None:
            payload["roi"] = list(self.roi)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CaptureContract":
        """Khôi phục contract từ config; config cũ dùng contract rỗng."""
        values = data or {}
        roi = values.get("roi")
        return cls(
            expected_width=values.get("expected_width"),
            expected_height=values.get("expected_height"),
            max_blur_score=values.get("max_blur_score"),
            min_exposure=values.get("min_exposure"),
            max_exposure=values.get("max_exposure"),
            roi=tuple(roi) if roi is not None else None,
            orientation=str(values.get("orientation", "fixed")),
        )
