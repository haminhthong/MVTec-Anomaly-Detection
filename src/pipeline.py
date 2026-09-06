"""End-to-end project orchestrator with explicit stage boundaries.

``train`` builds an artifact from normal training data only.
``evaluate`` consumes a frozen artifact and writes a report from the MVTec test split.
``all`` runs both stages sequentially but evaluation remains report-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainConfig
from .data.transforms import PreprocessingConfig
from .evaluation.evaluator import evaluate_category
from .training.trainer import train_patchcore


def main() -> None:
    parser = argparse.ArgumentParser(description="MVTec anomaly-detection project pipeline")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if args.image_size <= 0:
        parser.error("--image-size must be > 0")

    if args.stage in {"train", "all"}:
        train_patchcore(
            TrainConfig(
                category=args.category,
                data_root=args.data_root,
                model_root=args.model_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                preprocessing=PreprocessingConfig(
                    image_size=(args.image_size, args.image_size)
                ),
            )
        )

    if args.stage in {"evaluate", "all"}:
        report_path = Path(args.report_root) / args.category / "test_metrics.json"
        evaluate_category(
            category=args.category,
            model_dir=args.model_root,
            data_root=args.data_root,
            output_report=report_path,
        )


if __name__ == "__main__":
    main()
