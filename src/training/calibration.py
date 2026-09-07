"""Chia normal source và hiệu chỉnh policy chỉ bằng dữ liệu normal."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model.artifacts import ThresholdPolicy


def split_reference_dev_calibration(
    paths: list[Path],
    dev_fraction: float = 0.15,
    calibration_fraction: float = 0.15,
    seed: int = 42,
    min_calibration_samples: int = 20,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Tách Reference / Dev / Calibration ổn định và không chồng lấn.

    Reference dùng xây memory bank, Dev dùng chọn cấu hình và stress test,
    Calibration chỉ dùng khóa ngưỡng AUTO_PASS. Không split nào đọc test set.
    """
    ordered = sorted(Path(path) for path in paths)
    if not ordered:
        raise ValueError("Danh sách normal source đang rỗng.")
    if not 0 <= dev_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("dev_fraction/calibration_fraction phải nằm trong khoảng hợp lệ.")
    if dev_fraction + calibration_fraction >= 1:
        raise ValueError("dev_fraction + calibration_fraction phải nhỏ hơn 1.")
    if len(ordered) < min_calibration_samples:
        raise ValueError(
            f"Số ảnh normal ({len(ordered)}) nhỏ hơn calibration tối thiểu ({min_calibration_samples})."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    shuffled = [ordered[index] for index in permutation]
    calibration_count = max(min_calibration_samples, int(round(len(ordered) * calibration_fraction)))
    dev_count = int(round(len(ordered) * dev_fraction))
    if calibration_count + dev_count >= len(ordered):
        calibration_count = min(min_calibration_samples, max(1, len(ordered) - dev_count - 1))
    calibration = sorted(shuffled[:calibration_count])
    dev = sorted(shuffled[calibration_count : calibration_count + dev_count])
    reference = sorted(shuffled[calibration_count + dev_count :])
    if len(calibration) < min_calibration_samples:
        raise ValueError(
            f"Calibration chỉ có {len(calibration)} ảnh, cần tối thiểu {min_calibration_samples}."
        )
    if not reference:
        raise ValueError("Reference set không được rỗng sau khi split.")
    if set(reference) & set(dev) or set(reference) & set(calibration) or set(dev) & set(calibration):
        raise RuntimeError("Phát hiện overlap giữa Reference, Dev và Calibration.")
    return reference, dev, calibration


def split_normal_paths(
    paths: list[Path],
    calibration_fraction: float = 0.2,
    seed: int = 42,
    min_calibration_samples: int = 20,
) -> tuple[list[Path], list[Path]]:
    """API cũ: tách Reference và Calibration, không tạo Dev set."""
    reference, _, calibration = split_reference_dev_calibration(
        paths,
        dev_fraction=0.0,
        calibration_fraction=calibration_fraction,
        seed=seed,
        min_calibration_samples=min_calibration_samples,
    )
    return reference, calibration


def calibrate_thresholds(
    normal_scores: list[float],
    normal_heatmaps: list[np.ndarray],
    auto_pass_quantile: float = 0.99,
    pixel_quantile: float = 0.99,
    fail_quantile: float | None = None,
    review_quantile: float | None = None,
) -> ThresholdPolicy:
    """Khóa ngưỡng AUTO_PASS và pixel từ cohort calibration normal.

    Quantile đuôi trên chỉ là heuristic trên sample calibration; nó không phải
    cam kết false-reject rate 1% trong production và không được tối ưu bằng defect.
    ``fail_quantile``/``review_quantile`` chỉ giữ để đọc caller cũ.
    """
    if not normal_scores:
        raise ValueError("normal_scores đang rỗng, không thể calibration.")
    selected_quantile = fail_quantile if fail_quantile is not None else auto_pass_quantile
    if not 0.5 <= selected_quantile < 1.0:
        raise ValueError("auto_pass_quantile phải thuộc khoảng [0.5, 1.0).")
    if not 0.5 <= pixel_quantile < 1.0:
        raise ValueError("pixel_quantile phải thuộc khoảng [0.5, 1.0).")

    values = np.asarray(normal_scores, dtype=np.float32)
    auto_pass_threshold = float(np.quantile(values, selected_quantile))
    # Chỉ giữ review alias khi caller legacy truyền fail_quantile; artifact mới
    # không dùng ngưỡng này để quyết định.
    review_threshold = (
        float(np.quantile(values, review_quantile if review_quantile is not None else 0.95))
        if fail_quantile is not None
        else auto_pass_threshold
    )
    if normal_heatmaps:
        pixels = np.concatenate([np.asarray(heatmap, dtype=np.float32).ravel() for heatmap in normal_heatmaps])
        pixel_threshold = float(np.quantile(pixels, pixel_quantile))
    else:
        pixel_threshold = auto_pass_threshold
    return ThresholdPolicy(
        review_threshold=review_threshold,
        auto_pass_threshold=auto_pass_threshold,
        pixel_threshold=pixel_threshold,
    )
