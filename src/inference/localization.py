"""Tiện ích định vị defect và tạo visualization.

Module smoothing heatmap, tính tỷ lệ diện tích anomaly và tạo overlay base64.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def apply_heatmap_smoothing(heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Áp dụng Gaussian smoothing cho heatmap anomaly thô.

    Args:
        heatmap: Mảng khoảng cách patch 2D [H, W].
        sigma: Độ lệch chuẩn kernel Gaussian; nếu <= 0 thì giữ nguyên.

    Returns:
        np.ndarray: Heatmap 2D sau smoothing.
    """
    if sigma <= 0:
        return heatmap
    return gaussian_filter(heatmap.astype(np.float32), sigma=sigma)


def compute_anomalous_area_ratio(
    heatmap: np.ndarray, pixel_threshold: float
) -> float:
    """Tính tỷ lệ diện tích heatmap vượt pixel threshold đã calibration.

    Args:
        heatmap: Heatmap 2D đã smoothing [H, W].
        pixel_threshold: Pixel threshold dùng khi vận hành.

    Returns:
        float: Tỷ lệ diện tích trong khoảng [0.0, 1.0].
    """
    if heatmap.size == 0:
        return 0.0
    anomalous_pixels = np.sum(heatmap >= pixel_threshold)
    return float(anomalous_pixels / heatmap.size)


def create_heatmap_overlay_b64(
    image: Image.Image,
    heatmap: np.ndarray,
    threshold: float | None = None,
    alpha: float = 0.45,
    target_size: tuple[int, int] = (224, 224),
) -> str:
    """Tạo overlay màu từ heatmap trên ảnh đầu vào dưới dạng Base64 PNG.

    Args:
        image: Ảnh PIL đầu vào.
        heatmap: Heatmap anomaly 2D [H, W].
        threshold: Threshold tùy chọn chỉ dùng làm tham chiếu visualization.
        alpha: Tỷ lệ trộn heatmap lên ảnh gốc.
        target_size: Độ phân giải đích (height, width) lấy từ config.

    Returns:
        str: Chuỗi data URI Base64 dạng ``data:image/png;base64,...``.
    """
    h_target, w_target = target_size
    img_resized = image.convert("RGB").resize((w_target, h_target), Image.Resampling.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

    # Chuẩn hóa heatmap về [0, 1].
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    norm_heat = (heatmap - h_min) / (h_max - h_min + 1e-8)
    norm_heat = np.clip(norm_heat, 0.0, 1.0)

    # Phóng to heatmap theo kích thước ảnh đích.
    heat_pil = Image.fromarray((norm_heat * 255).astype(np.uint8)).resize(
        (w_target, h_target), Image.Resampling.BILINEAR
    )
    heat_resized = np.asarray(heat_pil, dtype=np.float32) / 255.0

    # Tạo màu kiểu Jet mà không thêm thư viện vẽ.
    r = np.clip(1.5 - np.abs(heat_resized * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(heat_resized * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(heat_resized * 4.0 - 1.0), 0.0, 1.0)
    color_map = np.stack([r, g, b], axis=-1)

    # Trộn overlay với ảnh gốc.
    overlay = (1.0 - alpha) * img_np + alpha * color_map
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    result_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
