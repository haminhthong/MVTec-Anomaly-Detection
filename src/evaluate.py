"""CLI entry point for report-only MVTec AD evaluation."""

from __future__ import annotations

import argparse
import sys

from .evaluation.evaluator import evaluate_category

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen anomaly-detection artifact")
    parser.add_argument("--category", default=None)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-report", default=None)
    args = parser.parse_args()
    evaluate_category(
        category=args.category,
        model_dir=args.model_dir,
        data_root=args.data_root,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()
