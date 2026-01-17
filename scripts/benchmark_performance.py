#!/usr/bin/env python3
"""Performance benchmark for before/after comparison.

Usage:
    # Run baseline benchmark (save results)
    python scripts/benchmark_performance.py --save baseline.json

    # Run optimized benchmark (compare to baseline)
    python scripts/benchmark_performance.py --compare baseline.json

    # Quick test (fewer iterations)
    python scripts/benchmark_performance.py --quick
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


BENCHMARK_CONFIG = {
    "num_tasks": 100,  # Tasks to generate
    "num_executions": 50,  # Code executions to time
    "num_grades": 100,  # Grading calls to time
    "warmup_iterations": 5,  # Warmup before timing
}

QUICK_CONFIG = {
    "num_tasks": 20,
    "num_executions": 10,
    "num_grades": 20,
    "warmup_iterations": 2,
}


def benchmark_task_generation(config, num_tasks: int) -> dict:
    """Measure task generation throughput."""
    from bs4_env.config import EnvConfig
    from bs4_env.lazy_dataset import LazyBS4Dataset

    env_config = EnvConfig(split="bench", mode="mvp", num_examples=200)
    dataset = LazyBS4Dataset.from_config(env_config)

    # Warmup
    for i in range(min(config["warmup_iterations"], len(dataset))):
        _ = dataset[i]

    # Benchmark
    start = time.perf_counter()
    for i in range(num_tasks):
        _ = dataset[i % len(dataset)]
    elapsed = time.perf_counter() - start

    return {
        "total_time_s": round(elapsed, 4),
        "tasks_per_second": round(num_tasks / elapsed, 2),
        "ms_per_task": round((elapsed / num_tasks) * 1000, 3),
    }


def benchmark_execution(config) -> dict:
    """Measure code execution latency."""
    from bs4_env.tools.executor import PooledSubprocessExecutor

    test_code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
result = soup.find("div", class_="target").get_text()
print(json.dumps({"status": "ok", "answer": result, "limit": None}))
"""
    test_html = '<div class="target">Hello World</div>'

    num_executions = config["num_executions"]

    with PooledSubprocessExecutor(num_workers=4) as executor:
        # Warmup
        for _ in range(config["warmup_iterations"]):
            executor.run(test_code, {"HTML": test_html, "QUERY": "test", "CONSTRAINTS": "{}"})

        # Benchmark
        latencies = []
        for _ in range(num_executions):
            start = time.perf_counter()
            executor.run(test_code, {"HTML": test_html, "QUERY": "test", "CONSTRAINTS": "{}"})
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "stdev_ms": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 3),
    }


def benchmark_grading(config) -> dict:
    """Measure grading latency."""
    from bs4_env.grading.rubric import compute_reward

    test_output = '{"status": "ok", "answer": "Hello World", "limit": null}'
    test_info = {
        "ground_truth": '"Hello World"',
        "solvable": True,
        "answer_schema": {"type": "string"},
    }
    test_html = '<div class="target">Hello World</div>'
    test_code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
result = soup.find("div").get_text()
"""

    num_grades = config["num_grades"]

    # Warmup
    for _ in range(config["warmup_iterations"]):
        compute_reward(test_output, test_info, test_html, code_samples=[test_code])

    # Benchmark
    latencies = []
    for _ in range(num_grades):
        start = time.perf_counter()
        compute_reward(test_output, test_info, test_html, code_samples=[test_code])
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "stdev_ms": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 3),
    }


def benchmark_ast_analysis(config) -> dict:
    """Measure AST analysis - both unified and legacy paths for comparison."""
    from bs4_env.grading.rubric import (
        _check_bs4_usage_ast,
        _check_content_access_ast,
        _check_selection_method_ast,
        _check_soup_creation_with_html_ast,
        analyze_code_unified,
    )

    test_code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
items = soup.find_all("div", class_="item")
for item in items:
    text = item.get_text()
    print(text)
"""

    num_iterations = config["num_grades"]

    # Warmup both paths
    for _ in range(config["warmup_iterations"]):
        analyze_code_unified(test_code)
        _check_bs4_usage_ast(test_code)
        _check_soup_creation_with_html_ast(test_code)
        _check_selection_method_ast(test_code)
        _check_content_access_ast(test_code)

    # Benchmark unified analysis (single pass - the optimized path)
    unified_latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        analyze_code_unified(test_code)
        unified_latencies.append((time.perf_counter() - start) * 1000)

    # Benchmark legacy 4-pass analysis (for comparison)
    legacy_latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _check_bs4_usage_ast(test_code)
        _check_soup_creation_with_html_ast(test_code)
        _check_selection_method_ast(test_code)
        _check_content_access_ast(test_code)
        legacy_latencies.append((time.perf_counter() - start) * 1000)

    return {
        "unified_mean_ms": round(statistics.mean(unified_latencies), 3),
        "unified_median_ms": round(statistics.median(unified_latencies), 3),
        "legacy_mean_ms": round(statistics.mean(legacy_latencies), 3),
        "legacy_median_ms": round(statistics.median(legacy_latencies), 3),
        "speedup_pct": round(
            (
                (statistics.median(legacy_latencies) - statistics.median(unified_latencies))
                / statistics.median(legacy_latencies)
            )
            * 100,
            1,
        ),
    }


def run_benchmark(config: dict) -> dict:
    """Run full benchmark suite."""
    print("Running performance benchmark...")
    print(f"  Config: {config}")
    print()

    results = {
        "config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("  [1/4] Task generation...")
    results["task_generation"] = benchmark_task_generation(config, config["num_tasks"])
    print(f"        {results['task_generation']['tasks_per_second']} tasks/s")

    print("  [2/4] Code execution...")
    results["code_execution"] = benchmark_execution(config)
    print(f"        {results['code_execution']['median_ms']} ms median")

    print("  [3/4] Grading...")
    results["grading"] = benchmark_grading(config)
    print(f"        {results['grading']['median_ms']} ms median")

    print("  [4/4] AST analysis...")
    results["ast_analysis"] = benchmark_ast_analysis(config)
    ast = results["ast_analysis"]
    print(
        f"        Unified: {ast['unified_median_ms']} ms, Legacy: {ast['legacy_median_ms']} ms ({ast['speedup_pct']}% faster)"
    )

    print()
    return results


def print_comparison(baseline: dict, optimized: dict) -> None:
    """Print comparison between baseline and optimized results."""
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print()

    sections = [
        ("Task Generation", "task_generation", "tasks_per_second", True),
        ("Code Execution", "code_execution", "median_ms", False),
        ("Grading", "grading", "median_ms", False),
        ("AST (unified)", "ast_analysis", "unified_median_ms", False),
    ]

    for name, key, metric, higher_is_better in sections:
        base_val = baseline.get(key, {}).get(metric, 0)
        opt_val = optimized.get(key, {}).get(metric, 0)

        if base_val == 0:
            continue

        if higher_is_better:
            pct_change = ((opt_val - base_val) / base_val) * 100
            improvement = pct_change > 0
        else:
            pct_change = ((base_val - opt_val) / base_val) * 100
            improvement = pct_change > 0

        sign = "+" if improvement else ""
        status = "✓" if improvement else "✗"

        unit = "" if "per_second" in metric else " ms"
        print(f"{name}:")
        print(f"  Baseline:  {base_val}{unit}")
        print(f"  Optimized: {opt_val}{unit}")
        print(f"  Change:    {status} {sign}{pct_change:.1f}%")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--compare", type=str, help="Compare to baseline JSON file")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    config = QUICK_CONFIG if args.quick else BENCHMARK_CONFIG

    results = run_benchmark(config)

    if args.save:
        save_path = Path(args.save)
        save_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to: {save_path}")

    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.exists():
            baseline = json.loads(compare_path.read_text())
            print_comparison(baseline, results)
        else:
            print(f"Warning: Baseline file not found: {compare_path}")

    # Print summary
    print("\nSUMMARY:")
    print(f"  Task generation: {results['task_generation']['tasks_per_second']} tasks/s")
    print(f"  Code execution:  {results['code_execution']['median_ms']} ms/exec")
    print(f"  Grading:         {results['grading']['median_ms']} ms/grade")
    ast = results["ast_analysis"]
    print(
        f"  AST (unified):   {ast['unified_median_ms']} ms ({ast['speedup_pct']}% faster than legacy)"
    )


if __name__ == "__main__":
    main()
