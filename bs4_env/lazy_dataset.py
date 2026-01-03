from __future__ import annotations

"""Lazy dataset that generates HTML on-demand.

This module provides a memory-efficient alternative to build_dataset()
that stores only (archetype_id, seed) pairs and generates HTML lazily.

Key differences from HuggingFace Dataset:
- Near-instant initialization (no HTML generation upfront)
- O(n) memory for metadata vs O(n * ~50KB) for full HTML
- Optional LRU caching for repeated access patterns
- Compatible with PyTorch DataLoader via Sequence protocol

Usage:
    from bs4_env.lazy_dataset import LazyBS4Dataset

    # Basic usage
    dataset = LazyBS4Dataset.from_config(config)
    for i in range(len(dataset)):
        item = dataset[i]  # HTML generated here

    # With caching for repeated access
    dataset = LazyBS4Dataset.from_config(config, cache_size=1000)

    # Convert to HuggingFace Dataset if needed (forces eager evaluation)
    hf_dataset = dataset.to_hf_dataset()
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

    from bs4_env.config import EnvConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LazyTaskEntry:
    """A lazy task entry storing only archetype_id and seed.

    This is hashable (frozen=True) to enable LRU caching.
    """

    archetype_id: str
    seed: int


class LazyBS4Dataset(Sequence):
    """Memory-efficient dataset that generates HTML on-demand.

    Instead of pre-generating all HTML upfront, this dataset stores
    only (archetype_id, seed) pairs and generates tasks lazily when
    accessed via __getitem__.

    Compatible with PyTorch DataLoader and similar interfaces.

    The `info` field is returned as a JSON string to maintain
    compatibility with code that expects HuggingFace Dataset semantics.

    Attributes:
        entries: List of LazyTaskEntry with archetype_id and seed.
        cache_size: LRU cache size. 0 disables caching.
    """

    def __init__(
        self,
        entries: list[LazyTaskEntry],
        cache_size: int = 0,
    ):
        """Initialize lazy dataset.

        Args:
            entries: List of LazyTaskEntry with archetype_id and seed.
            cache_size: LRU cache size. 0 disables caching.
        """
        self._entries = entries
        self._cache_size = cache_size

        if cache_size > 0:
            # Create cached version of _generate_task
            self._generate_task = lru_cache(maxsize=cache_size)(
                self._generate_task_uncached
            )
        else:
            self._generate_task = self._generate_task_uncached

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate and return task at index.

        Args:
            idx: Index into dataset.

        Returns:
            Dict with 'prompt' (list of messages) and 'info' (JSON string).
            Note: info is a JSON string for HuggingFace Dataset compatibility.

        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= len(self._entries):
            raise IndexError(f"Index {idx} out of range [0, {len(self._entries)})")

        entry = self._entries[idx]
        return self._generate_task(entry.archetype_id, entry.seed)

    def _generate_task_uncached(
        self, archetype_id: str, seed: int
    ) -> dict[str, Any]:
        """Generate a task from archetype_id and seed.

        This method may be wrapped with lru_cache when caching is enabled.

        Args:
            archetype_id: The archetype ID.
            seed: The seed for deterministic generation.

        Returns:
            Dict with 'prompt' and 'info' (as JSON string).
        """
        from bs4_env.config import TaskConstraints
        from bs4_env.prompt import format_prompt
        from bs4_env.registry import get_archetype

        spec = get_archetype(archetype_id)
        generator = spec.generator_class()
        task = generator.generate(seed)

        # Build constraints
        constraints = TaskConstraints(
            output_schema=task.answer_schema,
            allowed_limit_reasons=task.limit_info.get("allowed_reasons", []),
            safety_notes=["Do not extract passwords or authentication tokens."],
        )

        # Format prompt
        prompt = format_prompt(
            html=task.html,
            query=task.query,
            constraints=constraints,
        )

        # Inject difficulty from spec
        task.difficulty = spec.difficulty

        # Build info dict and serialize to JSON string
        # (HuggingFace Dataset compatibility - existing code expects info as string)
        info = task.to_info_dict()

        return {
            "prompt": prompt,
            "info": json.dumps(info),  # JSON string, not dict!
        }

    @classmethod
    def from_config(
        cls,
        config: EnvConfig,
        cache_size: int = 0,
    ) -> LazyBS4Dataset:
        """Build lazy dataset from configuration.

        This is the primary factory method. It computes the list of
        (archetype_id, seed) pairs without generating any HTML.

        Args:
            config: Environment configuration.
            cache_size: LRU cache size for repeated access.

        Returns:
            LazyBS4Dataset instance.
        """
        # Import here to avoid circular imports
        from bs4_env import auto_import  # noqa: F401
        from bs4_env.dataset import (
            SEED_RANGES,
            _get_archetype_ids_for_config,
            _select_seeds_for_archetype,
            load_bench_manifest,
        )
        from bs4_env.registry import get_archetype

        entries: list[LazyTaskEntry] = []

        # For bench split, use manifest
        if config.split == "bench":
            manifest = load_bench_manifest()

            # Apply filters
            if config.archetypes:
                manifest = [(a, s) for a, s in manifest if a in config.archetypes]
            if config.difficulty != "mixed":
                manifest = [
                    (a, s)
                    for a, s in manifest
                    if get_archetype(a).difficulty == config.difficulty
                ]
            if config.num_examples:
                manifest = manifest[: config.num_examples]

            entries = [LazyTaskEntry(a, s) for a, s in manifest]
        else:
            # Get archetype IDs for this config
            archetype_ids = _get_archetype_ids_for_config(config)
            seed_range = SEED_RANGES.get(config.split, (0, 100000))
            examples_per = cls._get_examples_per_archetype(config, archetype_ids)

            # Handle tiered mode with weighted sampling
            if config.mode == "tiered":
                # Group by difficulty and apply weights
                by_difficulty: dict[str, list[str]] = {
                    "primer": [],
                    "easy": [],
                    "medium": [],
                    "hard": [],
                }
                for aid in archetype_ids:
                    spec = get_archetype(aid)
                    if spec.difficulty in by_difficulty:
                        by_difficulty[spec.difficulty].append(aid)

                total_weight = sum(config.difficulty_weights.values())
                total_examples = examples_per * len(archetype_ids)

                for difficulty, weight in config.difficulty_weights.items():
                    archs = by_difficulty.get(difficulty, [])
                    if not archs:
                        continue

                    tier_examples = int(total_examples * weight / total_weight)
                    per_arch = max(1, tier_examples // len(archs))

                    for aid in archs:
                        seeds = _select_seeds_for_archetype(
                            archetype_id=aid,
                            split=config.split,
                            config_seed=config.seed,
                            num_examples=per_arch,
                            seed_range=seed_range,
                        )
                        entries.extend(LazyTaskEntry(aid, seed) for seed in seeds)
            else:
                # Standard mode: uniform sampling
                for archetype_id in archetype_ids:
                    seeds = _select_seeds_for_archetype(
                        archetype_id=archetype_id,
                        split=config.split,
                        config_seed=config.seed,
                        num_examples=examples_per,
                        seed_range=seed_range,
                    )
                    entries.extend(
                        LazyTaskEntry(archetype_id, seed) for seed in seeds
                    )

        return cls(entries, cache_size=cache_size)

    @staticmethod
    def _get_examples_per_archetype(
        config: EnvConfig,
        archetype_ids: list[str],
    ) -> int:
        """Get examples per archetype for this config."""
        from bs4_env.dataset import DEFAULT_EXAMPLES_PER_ARCHETYPE

        if config.num_examples is not None and archetype_ids:
            return max(1, config.num_examples // len(archetype_ids))

        return DEFAULT_EXAMPLES_PER_ARCHETYPE.get(config.split, 100)

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset (forces eager evaluation).

        Use this only when HuggingFace Dataset compatibility is required.
        This negates the memory benefits of lazy loading.

        Returns:
            HuggingFace Dataset with 'prompt' and 'info' columns.
        """
        from datasets import Dataset

        # Generate all rows (this is the expensive part)
        prompts = []
        infos = []
        for i in range(len(self)):
            row = self[i]
            prompts.append(row["prompt"])
            infos.append(row["info"])  # Already a JSON string

        return Dataset.from_dict({"prompt": prompts, "info": infos})

    def get_entry(self, idx: int) -> LazyTaskEntry:
        """Get the entry (archetype_id, seed) at index without generating HTML.

        Useful for inspection or debugging.

        Args:
            idx: Index into dataset.

        Returns:
            LazyTaskEntry with archetype_id and seed.
        """
        if idx < 0 or idx >= len(self._entries):
            raise IndexError(f"Index {idx} out of range [0, {len(self._entries)})")
        return self._entries[idx]


def build_lazy_dataset(
    config: EnvConfig,
    cache_size: int = 0,
) -> LazyBS4Dataset:
    """Build a lazy dataset that generates HTML on-demand.

    This is a convenience function that calls LazyBS4Dataset.from_config().

    Args:
        config: Environment configuration.
        cache_size: LRU cache size. 0 disables caching.

    Returns:
        LazyBS4Dataset instance.
    """
    return LazyBS4Dataset.from_config(config, cache_size=cache_size)
