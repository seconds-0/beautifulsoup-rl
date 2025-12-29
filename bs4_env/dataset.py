from __future__ import annotations

"""Dataset construction for BeautifulSoup RL environment.

This module builds HuggingFace datasets from task generators,
managing train/eval/bench splits with disjoint seeds.
"""

import json
from typing import Any, Iterator

from datasets import Dataset

from bs4_env.config import EnvConfig, TaskConstraints
from bs4_env.generators.base import TaskInstance
from bs4_env.prompt import format_prompt
from bs4_env.registry import get_archetype, list_archetypes, get_all_archetype_ids


# Seed ranges for each split to ensure no overlap
SEED_RANGES = {
    "train": (0, 100000),
    "eval": (100000, 110000),
    "bench": (110000, 111000),  # Fixed, smaller set for benchmarks
}

# Default number of examples per archetype per split
DEFAULT_EXAMPLES_PER_ARCHETYPE = {
    "train": 1000,
    "eval": 100,
    "bench": 20,
}


def build_dataset(config: EnvConfig) -> Dataset:
    """Build a HuggingFace Dataset from configuration.

    Args:
        config: Environment configuration specifying split, mode, etc.

    Returns:
        HuggingFace Dataset with 'prompt' and 'info' columns.
    """
    rows = list(generate_dataset_rows(config))

    if not rows:
        # Return empty dataset with correct schema
        return Dataset.from_dict({
            "prompt": [],
            "info": [],
        })

    return Dataset.from_dict({
        "prompt": [row["prompt"] for row in rows],
        "info": [json.dumps(row["info"]) for row in rows],
    })


def generate_dataset_rows(config: EnvConfig) -> Iterator[dict[str, Any]]:
    """Generate dataset rows based on configuration.

    Yields dictionaries with 'prompt' (list of messages) and 'info' (dict).

    For tiered mode, applies difficulty-weighted sampling to produce more
    hard tasks and fewer easy tasks, optimizing for RL training signal.

    Args:
        config: Environment configuration.

    Yields:
        Dictionary with 'prompt' and 'info' for each task instance.
    """
    # Import auto_import to ensure all generators are registered
    from bs4_env import auto_import  # noqa: F401

    # Get archetypes to include
    archetype_ids = _get_archetype_ids_for_config(config)

    if not archetype_ids:
        return

    # Get seed range and count for this split
    seed_start, seed_end = SEED_RANGES.get(config.split, (0, 100000))
    examples_per_archetype = _get_examples_per_archetype(config)

    # Use config seed to shuffle which seeds we use
    import random
    rng = random.Random(config.seed)

    # For tiered mode, compute weighted examples per archetype
    if config.mode == "tiered":
        # Group archetypes by difficulty
        archetypes_by_difficulty: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)
            archetypes_by_difficulty[spec.difficulty].append(archetype_id)

        # Compute examples per difficulty level based on weights
        total_weight = sum(config.difficulty_weights.values())
        total_examples = examples_per_archetype * len(archetype_ids)

        for difficulty, weight in config.difficulty_weights.items():
            difficulty_archetypes = archetypes_by_difficulty[difficulty]
            if not difficulty_archetypes:
                continue

            # Calculate examples for this difficulty tier
            tier_examples = int(total_examples * weight / total_weight)
            examples_per_archetype_in_tier = max(1, tier_examples // len(difficulty_archetypes))

            for archetype_id in difficulty_archetypes:
                yield from _generate_for_archetype(
                    archetype_id, seed_start, seed_end,
                    examples_per_archetype_in_tier, rng, config
                )
    else:
        # Standard mode: uniform sampling
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)

            # Filter by difficulty if specified
            if config.difficulty != "mixed" and spec.difficulty != config.difficulty:
                continue

            yield from _generate_for_archetype(
                archetype_id, seed_start, seed_end,
                examples_per_archetype, rng, config
            )


def _generate_for_archetype(
    archetype_id: str,
    seed_start: int,
    seed_end: int,
    num_examples: int,
    rng: "random.Random",
    config: EnvConfig,
) -> Iterator[dict[str, Any]]:
    """Generate examples for a single archetype.

    Args:
        archetype_id: The archetype to generate for.
        seed_start: Start of seed range.
        seed_end: End of seed range.
        num_examples: Number of examples to generate.
        rng: Random number generator for shuffling.
        config: Environment configuration.

    Yields:
        Dictionary with 'prompt' and 'info' for each task instance.
    """
    import random

    spec = get_archetype(archetype_id)
    generator = spec.generator_class()

    # Generate seeds for this archetype
    available_seeds = list(range(seed_start, seed_end))
    if config.split == "bench":
        # Bench uses fixed seeds, don't shuffle
        seeds = available_seeds[:num_examples]
    else:
        rng.shuffle(available_seeds)
        seeds = available_seeds[:num_examples]

    for seed in seeds:
        try:
            task = generator.generate(seed)
            row = _task_to_row(task, spec)
            yield row
        except Exception as e:
            # Log error but continue
            print(f"Error generating {archetype_id} seed {seed}: {e}")
            continue


def _get_archetype_ids_for_config(config: EnvConfig) -> list[str]:
    """Get list of archetype IDs matching configuration.

    Args:
        config: Environment configuration.

    Returns:
        List of archetype IDs to include.
    """
    # If specific archetypes requested, use those
    if config.archetypes:
        return config.archetypes

    # Filter by mode
    if config.mode == "mvp":
        specs = list_archetypes(phase=1)
    elif config.mode == "phase2":
        specs = list_archetypes(phase=2)
    elif config.mode == "hard_only":
        # Only hard difficulty archetypes from all phases
        specs = list_archetypes(difficulty="hard")
    elif config.mode == "tiered":
        # All archetypes (weighted sampling is handled in generate_dataset_rows)
        specs = list_archetypes()
    else:  # "all"
        specs = list_archetypes()

    return [spec.archetype_id for spec in specs]


def _get_examples_per_archetype(config: EnvConfig) -> int:
    """Get number of examples per archetype for this config.

    Args:
        config: Environment configuration.

    Returns:
        Number of examples per archetype.
    """
    if config.num_examples is not None:
        # Distribute across archetypes
        archetype_count = len(_get_archetype_ids_for_config(config))
        if archetype_count > 0:
            return max(1, config.num_examples // archetype_count)
        return config.num_examples

    return DEFAULT_EXAMPLES_PER_ARCHETYPE.get(config.split, 100)


def _task_to_row(task: TaskInstance, spec: Any) -> dict[str, Any]:
    """Convert a TaskInstance to a dataset row.

    Args:
        task: The generated task instance.
        spec: The archetype spec.

    Returns:
        Dictionary with 'prompt' and 'info'.
    """
    # Build constraints (visible to model)
    constraints = TaskConstraints(
        output_schema=task.answer_schema,
        allowed_limit_reasons=task.limit_info.get("allowed_reasons", []),
        safety_notes=["Do not extract passwords or authentication tokens."],
    )

    # Format the prompt
    prompt = format_prompt(
        html=task.html,
        query=task.query,
        constraints=constraints,
    )

    # Build info dict (hidden from model, used for grading)
    info = task.to_info_dict()

    return {
        "prompt": prompt,
        "info": info,
    }


def load_bench_manifest() -> list[tuple[str, int]]:
    """Load the bench manifest of fixed (archetype_id, seed) pairs.

    Returns:
        List of (archetype_id, seed) tuples for the benchmark set.
    """
    # For now, generate programmatically
    # In production, this would load from bench_manifest.json
    manifest = []

    archetype_ids = get_all_archetype_ids()
    seeds_per_archetype = 20

    for archetype_id in archetype_ids:
        for seed in range(110000, 110000 + seeds_per_archetype):
            manifest.append((archetype_id, seed))

    return manifest


def get_dataset_stats(dataset: Dataset) -> dict[str, Any]:
    """Get statistics about a dataset.

    Args:
        dataset: The HuggingFace Dataset.

    Returns:
        Dictionary with statistics.
    """
    stats = {
        "total_examples": len(dataset),
        "by_archetype": {},
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "solvable": 0,
        "unsolvable": 0,
    }

    for row in dataset:
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]

        archetype_id = info.get("archetype_id", "unknown")
        if archetype_id not in stats["by_archetype"]:
            stats["by_archetype"][archetype_id] = 0
        stats["by_archetype"][archetype_id] += 1

        if info.get("solvable", True):
            stats["solvable"] += 1
        else:
            stats["unsolvable"] += 1

    return stats
