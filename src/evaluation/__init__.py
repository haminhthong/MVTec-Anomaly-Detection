"""Metric evaluation; evaluator torch được lazy-load khi cần đọc model."""

from __future__ import annotations

from .aupro import compute_aupro
from .metrics import calculate_3tier_metrics


def __getattr__(name: str):
    """Nạp evaluator runtime theo nhu cầu."""
    if name == "evaluate_category":
        from .evaluator import evaluate_category

        return evaluate_category
    raise AttributeError(name)


__all__ = [
    "compute_aupro",
    "calculate_3tier_metrics",
    "evaluate_category",
]
