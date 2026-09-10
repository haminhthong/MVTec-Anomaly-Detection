"""Điểm vào CLI để xây dựng model từ ảnh normal."""

from __future__ import annotations

import sys

from .config import TrainConfig, parse_args
from .training.trainer import train_patchcore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(config: TrainConfig | None = None) -> None:
    """Điểm vào cho lệnh ``python -m src.train``."""
    cfg = config or parse_args()
    train_patchcore(cfg)


if __name__ == "__main__":
    main()
