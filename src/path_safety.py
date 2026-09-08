"""Kiểm tra các giá trị được dùng làm một thành phần đường dẫn."""

from __future__ import annotations


def ensure_safe_segment(value: str, field_name: str = "value") -> str:
    """Chuẩn hóa và từ chối giá trị có thể thoát khỏi thư mục gốc."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} phải là chuỗi.")
    normalized = value.strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or any(separator in normalized for separator in ("/", "\\", ":"))
        or "\x00" in normalized
    ):
        raise ValueError(f"{field_name} phải là một tên đơn, không chứa ký tự đường dẫn.")
    return normalized
