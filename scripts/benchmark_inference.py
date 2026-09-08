"""Benchmark latency, throughput và footprint tài nguyên của inference.

Đo lường:
- Độ trễ một ảnh (Mean, Median, P95, Min, Max).
- Độ trễ và throughput khi suy luận theo batch (ảnh/giây).
- Dung lượng memory bank trong RAM và trên đĩa.
- Thiết bị chạy và metadata runtime CPU/PyTorch.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import psutil
import torch

from src.data.dataset import find_category_root
from src.inference.detector import AnomalyDetector


def benchmark_category(
    category: str = "bottle",
    model_dir: str | Path = "models",
    num_warmup: int = 5,
    num_runs: int = 30,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
) -> dict:
    """Đo hiệu năng đầy đủ của detector và trả về metadata runtime."""
    print("=" * 70)
    print(f"   INFERENCE PERFORMANCE BENCHMARK: {category.upper()}")
    print("=" * 70)

    detector = AnomalyDetector(model_dir=model_dir, category=category)
    device = detector.dev
    image_size = list(detector.preprocessing_config.image_size)
    cpu_name = platform.processor() or platform.machine() or "unknown"
    root = find_category_root(category=category)
    sample_images = sorted((root / "test" / "good").glob("*.png"))

    if not sample_images:
        sample_images = sorted((root / "train" / "good").glob("*.png"))

    pil_images = [Image.open(p).convert("RGB") for p in sample_images[:max(batch_sizes)]]
    if not pil_images:
        raise ValueError(f"No sample images found for category '{category}'.")

    # Lặp lại mẫu nếu số lượng ảnh chưa đủ.
    while len(pil_images) < max(batch_sizes):
        pil_images.extend(pil_images[:max(batch_sizes) - len(pil_images)])

    # Dung lượng memory bank.
    bank_size_patches = detector.memory_bank.size
    bank_dim = detector.memory_bank.dim
    bank_ram_mb = (bank_size_patches * bank_dim * 4) / (1024 * 1024)
    file_path = detector.model_dir / "memory_bank.npy"
    file_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0

    print(f"Device               : {device.upper()}")
    print(f"CPU                  : {cpu_name}")
    print(f"PyTorch              : {torch.__version__}")
    print(f"Torch threads        : {torch.get_num_threads()}")
    print(f"Image size           : {image_size[0]}x{image_size[1]}")
    print(f"Memory Bank Shape    : [{bank_size_patches}, {bank_dim}]")
    print(f"Memory Bank RAM      : {bank_ram_mb:.2f} MB")
    print(f"Artifact File Size   : {file_size_mb:.2f} MB on disk")
    print(f"Warmup iterations    : {num_warmup}")
    print(f"Benchmark iterations : {num_runs}")
    print("-" * 70)

    # 1. Chạy warmup.
    for _ in range(num_warmup):
        _ = detector.score(pil_images[0])

    # 2. Đo độ trễ một ảnh (batch = 1).
    single_latencies: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = detector.score(pil_images[0])
        single_latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(single_latencies))
    median_lat = float(np.median(single_latencies))
    p95_lat = float(np.percentile(single_latencies, 95))
    min_lat = float(np.min(single_latencies))
    max_lat = float(np.max(single_latencies))
    throughput_single = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print("\n[SINGLE-IMAGE LATENCY (Batch = 1)]")
    print(f"  - Mean Latency     : {mean_lat:6.2f} ms")
    print(f"  - Median (P50)     : {median_lat:6.2f} ms")
    print(f"  - 95th Percentile  : {p95_lat:6.2f} ms")
    print(f"  - Min / Max        : {min_lat:6.2f} ms / {max_lat:6.2f} ms")
    print(f"  - Throughput       : {throughput_single:6.2f} images/second")

    # 3. Benchmark inspection theo batch.
    batch_results: dict[int, dict] = {}
    print("\n[BATCH INFERENCE THROUGHPUT]")
    print("  Batch Size | Latency/Batch (ms) | Latency/Image (ms) | Throughput (FPS)")
    print("  -----------+--------------------+--------------------+-----------------")

    for b in batch_sizes:
        batch_imgs = pil_images[:b]
        # Warmup batch.
        _ = detector.inspect_batch(batch_imgs, include_overlay=False)

        batch_times: list[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = detector.inspect_batch(batch_imgs, include_overlay=False)
            batch_times.append((time.perf_counter() - t0) * 1000.0)

        b_mean = float(np.mean(batch_times))
        per_img = b_mean / b
        fps = 1000.0 / per_img if per_img > 0 else 0.0
        batch_results[b] = {
            "batch_latency_ms": b_mean,
            "per_image_ms": per_img,
            "fps": fps,
        }
        print(f"  {b:10d} | {b_mean:18.2f} | {per_img:18.2f} | {fps:15.2f}")

    process = psutil.Process(os.getpid())
    process_memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\nProcess RSS Memory   : {process_memory_mb:.2f} MB")
    print("=" * 70 + "\n")

    return {
        "category": category,
        "device": device,
        "runtime": {
            "cpu": cpu_name,
            "pytorch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "image_size": image_size,
        },
        "memory_bank": {
            "patches": bank_size_patches,
            "dim": bank_dim,
            "ram_mb": bank_ram_mb,
            "disk_mb": file_size_mb,
        },
        "single_image": {
            "mean_ms": mean_lat,
            "median_ms": median_lat,
            "p95_ms": p95_lat,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "throughput_fps": throughput_single,
        },
        "batch_inference": batch_results,
        "process_rss_mb": process_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark AnomalyDetector inference latency and throughput")
    parser.add_argument("--category", type=str, default="bottle", help="Product category")
    parser.add_argument("--model-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--runs", type=int, default=25, help="Number of test iterations")
    args = parser.parse_args()

    benchmark_category(category=args.category, model_dir=args.model_dir, num_runs=args.runs)


if __name__ == "__main__":
    main()
