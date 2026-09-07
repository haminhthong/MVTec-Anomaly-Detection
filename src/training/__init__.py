"""Calibration nhẹ và trainer torch được lazy-load khi xây model."""

from __future__ import annotations

from .calibration import calibrate_thresholds, split_normal_paths, split_reference_dev_calibration


def __getattr__(name: str):
    """Nạp trainer chỉ khi caller chạy model building."""
    if name in {"set_seed", "train_patchcore"}:
        from .trainer import set_seed, train_patchcore

        return {"set_seed": set_seed, "train_patchcore": train_patchcore}[name]
    raise AttributeError(name)


__all__ = [
    "calibrate_thresholds",
    "split_normal_paths",
    "split_reference_dev_calibration",
    "train_patchcore",
    "set_seed",
]
