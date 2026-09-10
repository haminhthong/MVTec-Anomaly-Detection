"""Điểm vào CLI để đánh giá official test dưới dạng report-only."""

from __future__ import annotations

import argparse
import sys

from .evaluation.evaluator import evaluate_category

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    """Điểm vào của lệnh ``python -m src.evaluate``."""
    parser = argparse.ArgumentParser(description="Đánh giá model PatchCore-style trên official test (REPORT-ONLY)")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Tên category cần đánh giá",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Thư mục chứa model category",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Thư mục dữ liệu raw",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Đường dẫn report JSON đầu ra",
    )
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Cho phép ghi lại report hiện có",
    )
    args = parser.parse_args()
    evaluate_category(
        category=args.category,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_report=args.output_report,
        overwrite=args.overwrite_report,
    )


if __name__ == "__main__":
    main()
