"""Script tạo ảnh so sánh trực quan lỗi ngoại quan (Visual Inspection Comparison).

Sinh ra ảnh 4 khung hình:
[Original Image] | [Ground Truth Mask] | [Anomaly Heatmap] | [Overlay & Decision]
với image threshold và diện tích vùng vượt pixel threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Đảm bảo thư mục gốc nằm trong sys.path khi chạy dạng script độc lập
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inference.detector import AnomalyDetector


def generate_sample_comparison(
    detector: AnomalyDetector,
    image_path: Path,
    mask_path: Path | None,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Sinh ảnh so sánh 4 khung hình và lưu ra file PNG."""
    image = Image.open(image_path).convert("RGB")
    res = detector.inspect(image, include_overlay=False)

    score = res["anomaly_score"]
    decision = res["decision"]
    area_ratio = res["anomalous_area_ratio"]
    image_threshold = res["image_threshold"]
    pixel_threshold = res["pixel_threshold"]
    _, heatmap = detector.score(image)

    if decision == "REVIEW_REQUIRED":
        decision_color = "crimson"
    elif decision == "RECAPTURE_REQUIRED":
        decision_color = "darkorange"
    else:
        decision_color = "forestgreen"

    # Chuẩn bị ảnh mask
    if mask_path and mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
        mask_np = np.asarray(mask)
    else:
        mask_np = np.zeros((224, 224), dtype=np.uint8)

    # Chuẩn hóa heatmap phóng to 224x224
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    norm_heat = (heatmap - h_min) / (h_max - h_min + 1e-8)
    heat_pil = Image.fromarray((norm_heat * 255).astype(np.uint8)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    heat_resized = np.asarray(heat_pil, dtype=np.float32) / 255.0

    # Tạo overlay
    img_224 = image.resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.asarray(img_224, dtype=np.float32) / 255.0

    r = np.clip(1.5 - np.abs(heat_resized * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(heat_resized * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(heat_resized * 4.0 - 1.0), 0.0, 1.0)
    jet_map = np.stack([r, g, b], axis=-1)
    overlay = 0.55 * img_np + 0.45 * jet_map

    # Vẽ biểu đồ 4 panel
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    fig.patch.set_facecolor("#1e1e24")

    titles = [
        "1. Original Image",
        "2. Ground Truth Mask",
        "3. PatchCore Anomaly Map",
        f"4. Overlay ({decision})",
    ]

    images_to_show = [
        img_224,
        mask_np,
        heat_resized,
        overlay,
    ]

    cmaps = [None, "gray", "jet", None]

    for ax, title, img_show, cmap in zip(axes, titles, images_to_show, cmaps, strict=True):
        ax.set_facecolor("#1e1e24")
        if cmap:
            ax.imshow(img_show, cmap=cmap)
        else:
            ax.imshow(img_show)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    status_text = (
        f"Defect: {title_suffix} | Score: {score:.3f} | Image Th: {image_threshold:.3f} | "
        f"Pixel Th: {pixel_threshold:.3f} | Area: {area_ratio*100:.1f}% | Decision: {decision}"
    )
    fig.suptitle(
        status_text,
        color=decision_color,
        fontsize=12,
        fontweight="bold",
        y=0.06,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Đã lưu visual sample tại: {output_path}")


def main() -> None:
    """Tạo ảnh input/mask/heatmap/overlay cho một category."""
    parser = argparse.ArgumentParser(description="Sinh visual sample cho MVTec AD")
    parser.add_argument("--category", default="bottle", help="Category cần trực quan hóa")
    parser.add_argument("--data-dir", default="data/raw", help="Thư mục dữ liệu raw")
    parser.add_argument("--model-dir", default="models", help="Thư mục models theo category")
    parser.add_argument("--output-dir", default="reports/sample_outputs", help="Thư mục output")
    args = parser.parse_args()

    raw_dir = Path(args.data_dir) / args.category
    output_dir = Path(args.output_dir)
    detector = AnomalyDetector(model_dir=args.model_dir, category=args.category)

    test_root = raw_dir / "test"
    defect_dirs = sorted(path for path in test_root.iterdir() if path.is_dir() and path.name != "good") if test_root.is_dir() else []
    if defect_dirs:
        defect_type = defect_dirs[0].name
        defect_images = sorted(defect_dirs[0].glob("*.png"))
        defect_img = defect_images[0] if defect_images else None
        defect_mask = (
            raw_dir / "ground_truth" / defect_type / f"{defect_img.stem}_mask.png"
            if defect_img is not None
            else None
        )
    else:
        defect_type = "unknown"
        defect_img = None
        defect_mask = None
    if defect_img is not None and defect_img.exists():
        generate_sample_comparison(
            detector,
            defect_img,
            defect_mask,
            output_dir / "inspection_defect_sample.png",
            title_suffix=defect_type,
        )

    good_img = raw_dir / "test" / "good" / "000.png"
    if good_img.exists():
        generate_sample_comparison(
            detector,
            good_img,
            None,
            output_dir / "inspection_good_sample.png",
            title_suffix="Normal / Good (Chai đạt chuẩn chất lượng)",
        )


if __name__ == "__main__":
    main()
