"""Defect Localization and visualization utilities.

Functions:
- Gaussian smoothing on raw patch distance heatmaps
- Calculation of anomalous area ratio (ratio of pixels exceeding calibrated pixel threshold)
- Base64 PNG heatmap overlay blending with configurable image size (not hardcoded)
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def apply_heatmap_smoothing(heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing filter to raw anomaly map.

    Args:
        heatmap: 2D array of patch anomaly distances [H, W].
        sigma: Standard deviation for Gaussian kernel (if <= 0, returns unblurred).

    Returns:
        np.ndarray: Smoothed 2D heatmap.
    """
    if sigma <= 0:
        return heatmap
    return gaussian_filter(heatmap.astype(np.float32), sigma=sigma)


def compute_anomalous_area_ratio(
    heatmap: np.ndarray, pixel_threshold: float
) -> float:
    """Compute fraction of heatmap area exceeding calibrated pixel threshold.

    Args:
        heatmap: 2D smoothed heatmap [H, W].
        pixel_threshold: Calibrated operating pixel threshold.

    Returns:
        float: Area ratio in range [0.0, 1.0].
    """
    if heatmap.size == 0 or pixel_threshold <= 0:
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
    """Create color overlay of anomaly map onto input image encoded as Base64 PNG.

    Args:
        image: Original input PIL Image.
        heatmap: 2D anomaly heatmap [H, W].
        threshold: Optional threshold for visualization reference.
        alpha: Blending ratio for heatmap onto original image.
        target_size: Target resolution (height, width) dynamically driven by config.

    Returns:
        str: Base64 data URI string ('data:image/png;base64,...').
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
