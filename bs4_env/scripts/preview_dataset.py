#!/usr/bin/env python3
"""Preview dataset examples to sanity-check prompts and ground truth.

Usage:
    python -m bs4_env.scripts.preview_dataset
    python -m bs4_env.scripts.preview_dataset --split eval --num 5
"""

from __future__ import annotations

import argparse
import json

from beautiful_soup_env import load_environment


def truncate(s: str, max_len: int = 500) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def preview_example(example: dict, idx: int) -> None:
    """Print a formatted preview of an example."""
    info = example["info"]

    print(f"\n{'='*60}")
    print(f"Example {idx}: {info.get('archetype_id', 'unknown')}")
    print(f"{'='*60}")

    # Task metadata
    print(f"\nArchetype: {info.get('archetype_id')}")
    print(f"Difficulty: {info.get('difficulty')}")
    print(f"Solvable: {info.get('solvable')}")
    print(f"Seed: {info.get('seed')}")

    # Query
    print("\n--- Query ---")
    print(example["query"])

    # HTML (truncated)
    print("\n--- HTML (truncated) ---")
    print(truncate(example["html"], 300))

    # Ground truth
    print("\n--- Ground Truth ---")
    gt = info.get("ground_truth")
    if isinstance(gt, (dict, list)):
        print(json.dumps(gt, indent=2))
    else:
        print(gt)

    # Answer schema
    print("\n--- Answer Schema ---")
    schema = info.get("answer_schema", {})
    print(json.dumps(schema, indent=2))

    # Limit info (if applicable)
    limit_info = info.get("limit_info")
    if limit_info:
        print("\n--- Limit Info ---")
        print(json.dumps(limit_info, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Preview dataset examples")
    parser.add_argument("--split", default="train", choices=["train", "eval", "bench"])
    parser.add_argument("--mode", default="mvp", choices=["mvp", "phase2", "all"])
    parser.add_argument("--num", type=int, default=3, help="Number of examples to show")
    parser.add_argument("--archetype", type=str, help="Filter by archetype ID")
    args = parser.parse_args()

    print(f"Loading environment: split={args.split}, mode={args.mode}")
    env = load_environment(split=args.split, mode=args.mode)

    print(f"Dataset size: {len(env)}")

    # Show examples
    shown = 0
    for i, example in enumerate(env):
        if args.archetype and args.archetype not in example["info"].get("archetype_id", ""):
            continue

        preview_example(example, i)
        shown += 1

        if shown >= args.num:
            break

    if shown == 0:
        print("No examples found matching criteria")

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")

    archetype_counts = {}
    difficulty_counts = {}
    solvable_counts = {"True": 0, "False": 0}

    for example in env:
        info = example["info"]
        arch = info.get("archetype_id", "unknown")
        diff = info.get("difficulty", "unknown")
        solv = str(info.get("solvable", True))

        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        solvable_counts[solv] = solvable_counts.get(solv, 0) + 1

    print("\nBy archetype:")
    for arch, count in sorted(archetype_counts.items()):
        print(f"  {arch}: {count}")

    print("\nBy difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")

    print("\nBy solvability:")
    print(f"  Solvable: {solvable_counts['True']}")
    print(f"  Limitation: {solvable_counts['False']}")


if __name__ == "__main__":
    main()
