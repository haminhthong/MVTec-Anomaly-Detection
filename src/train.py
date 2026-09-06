"""CLI Entrypoint for Model Building Pipeline."""

from __future__ import annotations

import sys

from .config import TrainConfig, parse_args
from .training.trainer import train_patchcore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(config: TrainConfig | None = None) -> None:
    """Entry point for 'python -m src.train'."""
    cfg = config or parse_args()
    train_patchcore(cfg)


if __name__ == "__main__":
    main()
