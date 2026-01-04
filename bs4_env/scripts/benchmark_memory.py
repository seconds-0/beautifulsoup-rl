#!/usr/bin/env python3
"""Memory benchmark for disk-cached vs eager dataset loading.

This script validates that disk-cached datasets actually reduce memory usage
compared to eager (in-RAM) loading. It's the key validation for PR #11.

Usage:
    # Quick test (5 examples per archetype)
    python -m bs4_env.scripts.benchmark_memory --quick

    # Full benchmark (100 examples per archetype)
    python -m bs4_env.scripts.benchmark_memory --num-examples 100

    # Large scale test (1000 examples - simulates Prime training)
    python -m bs4_env.scripts.benchmark_memory --num-examples 1000
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import tracemalloc
from pathlib import Path

# Try to import psutil for process-level memory stats
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Process-level memory stats unavailable.")
    print("Install with: pip install psutil")


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def measure_eager_loading(num_examples: int) -> dict:
    """Measure memory for eager (in-RAM) dataset loading."""
    from bs4_env.config import EnvConfig
    from bs4_env.dataset import build_dataset

    # Force garbage collection before measurement
    gc.collect()
    tracemalloc.start()
    mem_before = get_process_memory_mb()

    config = EnvConfig(split="train", mode="mvp", num_examples=num_examples)
    dataset = build_dataset(config)

    # Force access to materialize any lazy loading
    _ = len(dataset)
    _ = dataset[0]

    mem_after = get_process_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "eager",
        "num_examples": len(dataset),
        "tracemalloc_current_mb": current / (1024 * 1024),
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "process_delta_mb": mem_after - mem_before,
        "process_total_mb": mem_after,
    }


def measure_disk_cached(num_examples: int, cache_dir: Path) -> dict:
    """Measure memory for disk-cached dataset loading."""
    from bs4_env.config import EnvConfig
    from bs4_env.dataset import build_disk_cached_dataset

    # Force garbage collection before measurement
    gc.collect()
    tracemalloc.start()
    mem_before = get_process_memory_mb()

    config = EnvConfig(split="train", mode="mvp", num_examples=num_examples)
    dataset = build_disk_cached_dataset(
        config, cache_dir=cache_dir, force_rebuild=True
    )

    # Force access to check memory-mapped behavior
    _ = len(dataset)
    _ = dataset[0]

    mem_after = get_process_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Check cache size on disk
    cache_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())

    return {
        "method": "disk_cached",
        "num_examples": len(dataset),
        "tracemalloc_current_mb": current / (1024 * 1024),
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "process_delta_mb": mem_after - mem_before,
        "process_total_mb": mem_after,
        "cache_size_mb": cache_size / (1024 * 1024),
    }


def measure_disk_cached_reload(num_examples: int, cache_dir: Path) -> dict:
    """Measure memory for loading from existing disk cache."""
    from bs4_env.config import EnvConfig
    from bs4_env.dataset import build_disk_cached_dataset

    gc.collect()
    tracemalloc.start()
    mem_before = get_process_memory_mb()

    # Must use same num_examples to hit the same cache key
    config = EnvConfig(split="train", mode="mvp", num_examples=num_examples)
    dataset = build_disk_cached_dataset(
        config, cache_dir=cache_dir, force_rebuild=False
    )

    _ = len(dataset)
    _ = dataset[0]

    mem_after = get_process_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "disk_cached_reload",
        "num_examples": len(dataset),
        "tracemalloc_current_mb": current / (1024 * 1024),
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "process_delta_mb": mem_after - mem_before,
        "process_total_mb": mem_after,
    }


def run_iteration_test(dataset, num_iterations: int = 100) -> dict:
    """Test memory during dataset iteration (simulates training loop)."""
    gc.collect()
    tracemalloc.start()
    mem_before = get_process_memory_mb()

    # Simulate training loop access pattern
    for i in range(min(num_iterations, len(dataset))):
        row = dataset[i]
        _ = row["prompt"]
        _ = row["info"]

    mem_after = get_process_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "iterations": min(num_iterations, len(dataset)),
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "process_delta_mb": mem_after - mem_before,
    }


def print_results(results: dict, label: str):
    """Print formatted results."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark disk-cached vs eager loading")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of examples per archetype (default: 50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 5 examples",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for access pattern test",
    )
    args = parser.parse_args()

    if args.quick:
        args.num_examples = 5

    print("\n" + "=" * 60)
    print("  DISK-CACHED DATASET MEMORY BENCHMARK")
    print("=" * 60)
    print(f"  Examples per archetype: {args.num_examples}")
    print(f"  psutil available: {HAS_PSUTIL}")
    print("=" * 60)

    # Create temp cache directory
    cache_dir = Path(tempfile.mkdtemp(prefix="bs4_cache_benchmark_"))
    print(f"\n  Cache directory: {cache_dir}")

    try:
        # Test 1: Eager loading
        print("\n[1/4] Testing eager (in-RAM) loading...")
        eager_results = measure_eager_loading(args.num_examples)
        print_results(eager_results, "EAGER LOADING (build_dataset)")

        # Force cleanup before next test
        gc.collect()

        # Test 2: Disk-cached (initial build)
        print("\n[2/4] Testing disk-cached (initial build)...")
        disk_results = measure_disk_cached(args.num_examples, cache_dir)
        print_results(disk_results, "DISK-CACHED (build_disk_cached_dataset)")

        # Force cleanup before next test
        gc.collect()

        # Test 3: Disk-cached (reload from cache)
        print("\n[3/4] Testing disk-cached (reload from cache)...")
        reload_results = measure_disk_cached_reload(args.num_examples, cache_dir)
        print_results(reload_results, "DISK-CACHED RELOAD (from existing cache)")

        # Test 4: Iteration test (simulates training)
        print("\n[4/4] Testing iteration access pattern...")

        # Reload datasets for fair comparison
        from bs4_env.config import EnvConfig
        from bs4_env.dataset import build_dataset, build_disk_cached_dataset

        gc.collect()
        eager_dataset = build_dataset(
            EnvConfig(split="train", mode="mvp", num_examples=args.num_examples)
        )
        eager_iter = run_iteration_test(eager_dataset, args.iterations)
        print_results(eager_iter, "EAGER ITERATION TEST")

        gc.collect()
        disk_dataset = build_disk_cached_dataset(
            EnvConfig(split="train", mode="mvp", num_examples=args.num_examples),
            cache_dir=cache_dir,
            force_rebuild=False,
        )
        disk_iter = run_iteration_test(disk_dataset, args.iterations)
        print_results(disk_iter, "DISK-CACHED ITERATION TEST")

        # Summary comparison
        print("\n" + "=" * 60)
        print("  SUMMARY COMPARISON")
        print("=" * 60)

        eager_peak = eager_results["tracemalloc_peak_mb"]
        disk_peak = disk_results["tracemalloc_peak_mb"]
        reload_peak = reload_results["tracemalloc_peak_mb"]

        print("\n  Peak memory (tracemalloc):")
        print(f"    Eager:         {eager_peak:.2f} MB")
        print(f"    Disk (build):  {disk_peak:.2f} MB")
        print(f"    Disk (reload): {reload_peak:.2f} MB")

        if eager_peak > 0:
            reduction_build = ((eager_peak - disk_peak) / eager_peak) * 100
            reduction_reload = ((eager_peak - reload_peak) / eager_peak) * 100
            print("\n  Memory reduction vs eager:")
            print(f"    Build: {reduction_build:+.1f}%")
            print(f"    Reload: {reduction_reload:+.1f}%")

        cache_size = disk_results.get("cache_size_mb", 0)
        print(f"\n  Cache size on disk: {cache_size:.2f} MB")

        # Pass/Fail criteria
        print("\n" + "=" * 60)
        print("  VALIDATION RESULTS")
        print("=" * 60)

        passed = True
        tests = []

        # Test: Reload should use less peak memory than eager
        if reload_peak < eager_peak:
            tests.append(("Reload uses less memory than eager", "PASS"))
        else:
            tests.append(("Reload uses less memory than eager", "FAIL"))
            passed = False

        # Test: Cache should be created
        if cache_size > 0:
            tests.append(("Cache files created on disk", "PASS"))
        else:
            tests.append(("Cache files created on disk", "FAIL"))
            passed = False

        # Test: Dataset should have expected size
        if eager_results["num_examples"] == disk_results["num_examples"]:
            tests.append(("Dataset sizes match", "PASS"))
        else:
            tests.append(("Dataset sizes match", "FAIL"))
            passed = False

        for test_name, result in tests:
            status = "✅" if result == "PASS" else "❌"
            print(f"  {status} {test_name}: {result}")

        print("\n" + "=" * 60)
        if passed:
            print("  ✅ ALL VALIDATION TESTS PASSED")
        else:
            print("  ❌ SOME VALIDATION TESTS FAILED")
        print("=" * 60 + "\n")

        return 0 if passed else 1

    finally:
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
