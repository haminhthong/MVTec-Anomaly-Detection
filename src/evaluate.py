"""Điểm vào CLI cho pipeline đánh giá locked test dạng report-only."""

from __future__ import annotations

import argparse
import sys

from .evaluation.evaluator import evaluate_category

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    """Điểm vào của lệnh ``python -m src.evaluate``."""
    parser = argparse.ArgumentParser(description="Evaluate PatchCore-style model on test split (REPORT-ONLY)")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Category name to evaluate",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Path to models directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw datasets directory",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--reopen-locked-test",
        action="store_true",
        help="Cho phép ghi lại locked report hiện có sau khi chủ động mở lại",
    )
    args = parser.parse_args()
    evaluate_category(
        category=args.category,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_report=args.output_report,
        reopen=args.reopen_locked_test,
    )


if __name__ == "__main__":
    main()
