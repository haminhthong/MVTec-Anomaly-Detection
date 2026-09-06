"""Download MVTec AD category datasets from Hugging Face mirror.

MVTec AD is licensed under CC BY-NC-SA 4.0 (Non-Commercial).
Mirror repository preserves the official directory structure.
"""

from __future__ import annotations

import argparse
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
    """Download specific category pattern."""
    print(f"Downloading MVTec AD category '{category}' into '{output_dir}'...")
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        allow_patterns=[f"{category}/**"],
        local_dir=output_dir,
    )
    print(f"Successfully downloaded '{category}'.")


def main():
    parser = argparse.ArgumentParser(description="Download MVTec AD category datasets")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Category to download (e.g. 'bottle', 'cable', or 'all')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Target raw data directory",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.category.lower() == "all":
        for cat in ALL_CATEGORIES:
            download_category(cat, out)
    else:
        download_category(args.category, out)


if __name__ == "__main__":
    main()
