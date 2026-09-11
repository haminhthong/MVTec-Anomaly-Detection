"""Tải category MVTec AD và lưu provenance nguồn dữ liệu."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET = "foersben/mvtec-ad"

ALL_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def download_category(category: str, output_dir: Path) -> None:
    """Tải một category và ghi metadata nguồn dữ liệu ở thư mục raw."""
    category = category.strip().lower()
    if category not in ALL_CATEGORIES:
        raise ValueError(f"Category không được hỗ trợ: {category!r}.")
    print(f"Đang tải category MVTec AD '{category}' vào '{output_dir}'...")
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        allow_patterns=[f"{category}/**"],
        local_dir=output_dir,
    )
    metadata_path = output_dir / "DATASET_SOURCE.json"
    metadata = {
        "source": "Hugging Face mirror foersben/mvtec-ad",
        "source_version": DATASET,
        "download_date": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC-SA 4.0",
        "archive_sha256": None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã tải xong '{category}'.")


def main():
    parser = argparse.ArgumentParser(description="Tải dataset category MVTec AD")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Category cần tải, ví dụ 'bottle', 'cable' hoặc 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Thư mục lưu dữ liệu raw",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    category = args.category.strip().lower()
    if category == "all":
        for cat in ALL_CATEGORIES:
            download_category(cat, out)
    else:
        download_category(category, out)


if __name__ == "__main__":
    main()
