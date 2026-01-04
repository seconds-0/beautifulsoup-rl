from __future__ import annotations

"""Dataset construction for BeautifulSoup RL environment.

This module builds HuggingFace datasets from task generators,
managing train/eval/bench splits with disjoint seeds.
"""

import contextlib
import hashlib
import json
import logging
import random
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset

if TYPE_CHECKING:
    from bs4_env.lazy_dataset import LazyBS4Dataset

logger = logging.getLogger(__name__)

from bs4_env.config import EnvConfig, TaskConstraints
from bs4_env.generators.base import TaskInstance
from bs4_env.prompt import format_prompt
from bs4_env.registry import get_all_archetype_ids, get_archetype, list_archetypes


def _get_pkg_version() -> str:
    """Get package version using importlib.metadata.

    This avoids importing beautiful_soup_env from within bs4_env.dataset,
    which would create a layering smell (dataset depending on the entrypoint).

    Returns:
        Version string, or "0.0.0" if package is not installed.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        return _version("beautiful-soup-env")
    except PackageNotFoundError:
        return "0.0.0"


def _stable_seed_from_key(key: str) -> int:
    """Generate a stable integer seed from a string key.

    Uses SHA-256 to produce a deterministic integer.
    This is different from Python's hash() which is salted per-run.

    Args:
        key: A string key to hash.

    Returns:
        A stable integer seed (positive, fits in 64 bits).
    """
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:16], 16)  # Use first 64 bits


def _select_seeds_for_archetype(
    archetype_id: str,
    split: str,
    config_seed: int,
    num_examples: int,
    seed_range: tuple[int, int],
) -> list[int]:
    """Select seeds independently per archetype.

    Each archetype gets its own RNG, so adding/removing archetypes
    doesn't affect seed selection for other archetypes.

    Args:
        archetype_id: The archetype ID.
        split: The dataset split (train/eval/bench).
        config_seed: The config-level seed.
        num_examples: Number of seeds to select.
        seed_range: Tuple of (start, end) for available seeds.

    Returns:
        List of selected seeds for this archetype.
    """
    # Create a unique key for this archetype + split + config combination
    key = f"{archetype_id}:{split}:{config_seed}"
    rng = random.Random(_stable_seed_from_key(key))

    # Use rng.sample() instead of shuffle for efficiency (O(num_examples) vs O(range_size))
    available = range(seed_range[0], seed_range[1])
    return list(rng.sample(available, min(num_examples, len(available))))


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
        return Dataset.from_dict(
            {
                "prompt": [],
                "info": [],
            }
        )

    return Dataset.from_dict(
        {
            "prompt": [row["prompt"] for row in rows],
            "info": [json.dumps(row["info"]) for row in rows],
        }
    )


def generate_dataset_rows(config: EnvConfig) -> Iterator[dict[str, Any]]:
    """Generate dataset rows based on configuration.

    Yields dictionaries with 'prompt' (list of messages) and 'info' (dict).

    For bench split, uses the fixed manifest to ensure stability.
    For tiered mode, applies difficulty-weighted sampling to produce more
    hard tasks and fewer easy tasks, optimizing for RL training signal.

    Args:
        config: Environment configuration.

    Yields:
        Dictionary with 'prompt' and 'info' for each task instance.
    """
    # Import auto_import to ensure all generators are registered
    from bs4_env import auto_import  # noqa: F401

    # For bench split, use the fixed manifest to ensure stability
    if config.split == "bench":
        yield from _generate_from_manifest(config)
        return

    # Get archetypes to include
    archetype_ids = _get_archetype_ids_for_config(config)

    if not archetype_ids:
        return

    # Get seed range and count for this split
    seed_range = SEED_RANGES.get(config.split, (0, 100000))
    examples_per_archetype = _get_examples_per_archetype(config)

    # For tiered mode, compute weighted examples per archetype
    if config.mode == "tiered":
        # Group archetypes by difficulty
        archetypes_by_difficulty: dict[str, list[str]] = {
            "primer": [],
            "easy": [],
            "medium": [],
            "hard": [],
        }
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)
            if spec.difficulty in archetypes_by_difficulty:
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
                    archetype_id, examples_per_archetype_in_tier, seed_range, config
                )
    else:
        # Standard mode: uniform sampling
        # Note: difficulty filtering is already applied in _get_archetype_ids_for_config()
        for archetype_id in archetype_ids:
            yield from _generate_for_archetype(
                archetype_id, examples_per_archetype, seed_range, config
            )


def _generate_from_manifest(config: EnvConfig) -> Iterator[dict[str, Any]]:
    """Generate examples from the fixed bench manifest.

    This ensures bench split is stable regardless of archetype additions/removals.

    Args:
        config: Environment configuration.

    Yields:
        Dictionary with 'prompt' and 'info' for each task instance.
    """
    manifest = load_bench_manifest()

    # Apply filters if specified
    filtered_manifest = manifest
    if config.archetypes:
        filtered_manifest = [(aid, seed) for aid, seed in manifest if aid in config.archetypes]
    if config.difficulty != "mixed":
        filtered_manifest = [
            (aid, seed)
            for aid, seed in filtered_manifest
            if get_archetype(aid).difficulty == config.difficulty
        ]

    # Apply num_examples limit if specified
    if config.num_examples is not None:
        filtered_manifest = filtered_manifest[: config.num_examples]

    # Generate tasks
    for archetype_id, seed in filtered_manifest:
        try:
            spec = get_archetype(archetype_id)
            generator = spec.generator_class()
            task = generator.generate(seed)
            row = _task_to_row(task, spec)
            yield row
        except Exception as e:
            logger.error(f"Error generating {archetype_id} seed {seed}: {e}")
            continue


def _generate_for_archetype(
    archetype_id: str,
    num_examples: int,
    seed_range: tuple[int, int],
    config: EnvConfig,
) -> Iterator[dict[str, Any]]:
    """Generate examples for a single archetype.

    Uses per-archetype RNG seeding so that adding/removing archetypes
    doesn't affect seed selection for other archetypes.

    Args:
        archetype_id: The archetype to generate for.
        num_examples: Number of examples to generate.
        seed_range: Tuple of (start, end) for available seeds.
        config: Environment configuration.

    Yields:
        Dictionary with 'prompt' and 'info' for each task instance.
    """
    spec = get_archetype(archetype_id)
    generator = spec.generator_class()

    # Select seeds independently for this archetype
    seeds = _select_seeds_for_archetype(
        archetype_id=archetype_id,
        split=config.split,
        config_seed=config.seed,
        num_examples=num_examples,
        seed_range=seed_range,
    )

    for seed in seeds:
        try:
            task = generator.generate(seed)
            row = _task_to_row(task, spec)
            yield row
        except Exception as e:
            # Log error but continue
            logger.error(f"Error generating {archetype_id} seed {seed}: {e}")
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
    elif config.mode == "bootstrap":
        # Primer + easy archetypes for 0% model onboarding
        primer_specs = list_archetypes(difficulty="primer")
        easy_specs = list_archetypes(difficulty="easy")
        specs = primer_specs + easy_specs
    elif config.mode == "tiered":
        # All archetypes (weighted sampling is handled in generate_dataset_rows)
        specs = list_archetypes()
    else:  # "all"
        specs = list_archetypes()

    # Apply difficulty filter when not "mixed" (and not already filtered by hard_only mode)
    if config.difficulty != "mixed" and config.mode != "hard_only":
        specs = [s for s in specs if s.difficulty == config.difficulty]

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

    # Inject difficulty from archetype spec (ensures consistency)
    task.difficulty = spec.difficulty

    # Build info dict (hidden from model, used for grading)
    info = task.to_info_dict()

    return {
        "prompt": prompt,
        "info": info,
    }


def load_bench_manifest() -> list[tuple[str, int]]:
    """Load the bench manifest of fixed (archetype_id, seed) pairs.

    The manifest is loaded from bs4_env/data/bench_manifest.json to ensure
    benchmark stability. If the file doesn't exist, falls back to programmatic
    generation with a warning.

    Returns:
        List of (archetype_id, seed) tuples for the benchmark set.
    """
    manifest_path = Path(__file__).parent / "data" / "bench_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        return [(entry["archetype_id"], entry["seed"]) for entry in data["entries"]]

    # Fallback: generate programmatically (with warning)
    warnings.warn(
        "bench_manifest.json not found, generating dynamically. "
        "This may cause benchmark instability if archetypes are added/removed.",
        UserWarning,
        stacklevel=2,
    )
    manifest = []
    archetype_ids = get_all_archetype_ids()
    seeds_per_archetype = 20

    for archetype_id in archetype_ids:
        for seed in range(110000, 110000 + seeds_per_archetype):
            manifest.append((archetype_id, seed))

    return manifest


def build_lazy_dataset(
    config: EnvConfig,
    cache_size: int = 0,
) -> LazyBS4Dataset:
    """Build a lazy dataset that generates HTML on-demand.

    This is more memory-efficient than build_dataset() for large datasets.
    HTML is generated on-demand during __getitem__ access instead of upfront.

    Args:
        config: Environment configuration.
        cache_size: LRU cache size. 0 disables caching.

    Returns:
        LazyBS4Dataset instance.
    """
    from bs4_env.lazy_dataset import LazyBS4Dataset

    return LazyBS4Dataset.from_config(config, cache_size=cache_size)


# =============================================================================
# Disk-Cached Dataset (Memory-Efficient for Large Training)
# =============================================================================


def _compute_cache_key(config: EnvConfig, env_id: str) -> str:
    """Compute deterministic cache key from config + env_id + archetype version.

    The key changes when:
    - env_id changes (critical: task column value)
    - Config parameters change (split, mode, difficulty, seed, num_examples, archetypes)
    - Archetypes are added/removed (version hash)
    - Package version changes (generator code updates)
    - Generator code changes (code fingerprint - dev mode safety)

    Args:
        config: Environment configuration.
        env_id: Environment ID written to the 'task' column.

    Returns:
        SHA-256 hash string (first 16 chars).
    """
    # Include difficulty_weights for tiered mode cache invalidation
    weights_str = str(sorted(config.difficulty_weights.items()))
    key_parts = [
        _get_pkg_version(),  # Invalidates when package version bumps
        _get_code_fingerprint(),  # Invalidates on generator code changes (dev mode safety)
        env_id,  # Critical: prevents cache collision with different env_ids
        config.split,
        config.mode,
        config.difficulty,
        str(config.seed),
        str(config.num_examples),
        str(sorted(config.archetypes or [])),
        weights_str,
        _get_archetype_version_hash(),
    ]
    return hashlib.sha256(":".join(key_parts).encode()).hexdigest()[:16]


def _get_archetype_version_hash() -> str:
    """Get hash of registered archetypes for cache invalidation.

    When archetypes are added or removed, this hash changes.
    Note: Does not detect code changes within generators.

    Returns:
        SHA-256 hash string (first 8 chars).
    """
    # Import auto_import first to ensure all archetypes are registered
    from bs4_env import auto_import  # noqa: F401

    specs = list_archetypes()
    spec_strings = sorted([f"{s.archetype_id}:{s.difficulty}:{s.phase}" for s in specs])
    return hashlib.sha256(":".join(spec_strings).encode()).hexdigest()[:8]


@contextlib.contextmanager
def _cache_lock(lock_path: Path):
    """File-based lock for cache directory operations.

    Prevents race conditions when multiple workers build the same cache.
    Uses fcntl.flock which is available on Linux/macOS.
    """
    import fcntl
    import os

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _is_cache_ready(dataset_dir: Path) -> bool:
    """Check if cache was fully written (not partial/crashed).

    A cache is only valid if it contains a _READY marker file,
    which is written atomically after the dataset is fully saved.
    This prevents loading partial caches from crashed workers.
    """
    return (dataset_dir / "_READY").exists()


def _get_code_fingerprint() -> str:
    """Get code fingerprint for cache invalidation.

    This ensures generator code changes invalidate caches, even in dev mode
    where __version__ is "0.0.0".

    Priority:
    1. Git SHA from environment (CI/Prime often injects this)
    2. Hash of generator source files (dev mode fallback)

    Returns:
        16-character hex string.
    """
    import os

    # Check for CI-injected commit SHA
    for var in ("GIT_SHA", "COMMIT_SHA", "BS4_ENV_BUILD_ID", "GITHUB_SHA"):
        sha = os.environ.get(var)
        if sha:
            return sha[:16]

    # Fallback: hash generator source files
    h = hashlib.sha256()
    generators_dir = Path(__file__).parent / "generators"
    if generators_dir.exists():
        for p in sorted(generators_dir.glob("*.py")):
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_disk_cached_dataset(
    config: EnvConfig,
    cache_dir: Path | str | None = None,
    force_rebuild: bool = False,
    env_id: str = "beautiful-soup-env",
) -> Dataset:
    """Build HuggingFace Dataset with disk caching for memory efficiency.

    Uses Dataset.from_generator() to stream rows to disk without loading all
    HTML into RAM. Pre-populates example_id and task columns so verifiers'
    format_dataset() skips expensive add_column/map operations.

    This is the recommended approach for training datasets (50K+ examples).
    For bench splits, use build_dataset() which is simpler and sufficient.

    Thread-safe: Uses file locking to prevent race conditions when multiple
    workers attempt to build the same cache simultaneously.

    Crash-safe: Uses atomic temp directory + rename pattern with READY marker
    to prevent loading partial caches from crashed workers.

    Args:
        config: Environment configuration.
        cache_dir: Directory for Arrow cache files. Defaults to ~/.cache/bs4_env/datasets/
        force_rebuild: If True, regenerate even if cache exists.
        env_id: Environment ID for the 'task' column. Defaults to "beautiful-soup-env".

    Returns:
        Memory-mapped HuggingFace Dataset with columns:
        - prompt: list of chat messages
        - info: JSON string with task metadata
        - example_id: integer index (required by verifiers)
        - task: string (required by verifiers EnvGroup)
    """
    import shutil
    import tempfile
    from datetime import UTC, datetime

    # Determine cache location
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "bs4_env" / "datasets"
    else:
        cache_dir = Path(cache_dir)

    cache_key = _compute_cache_key(config, env_id=env_id)
    dataset_dir = cache_dir / cache_key
    lock_file = dataset_dir.with_suffix(".lock")

    # Use file lock to prevent race conditions on Prime (multiple workers)
    with _cache_lock(lock_file):
        # Only load if READY marker exists (not partial/crashed cache)
        if dataset_dir.exists() and _is_cache_ready(dataset_dir) and not force_rebuild:
            try:
                logger.info(f"Loading cached dataset from {dataset_dir}")
                return Dataset.load_from_disk(str(dataset_dir))
            except Exception as e:
                logger.warning(f"Cache load failed, will rebuild: {e}")

        # Build to temp directory first (atomic pattern)
        tmp_dir = dataset_dir.with_name(dataset_dir.name + ".tmp")

        # Clean up any stale temp or partial directories
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(dataset_dir, ignore_errors=True)

        logger.info(f"Building dataset to disk cache: {dataset_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        def gen():
            for idx, row in enumerate(generate_dataset_rows(config)):
                yield {
                    "prompt": row["prompt"],
                    "info": json.dumps(row["info"])
                    if isinstance(row["info"], dict)
                    else row["info"],
                    "example_id": idx,  # Integer - verifiers skips add_column
                    "task": env_id,  # String - verifiers skips map
                }

        # Generate dataset to temp directory
        # IMPORTANT: Use cache_dir as base for Arrow temp files, not /tmp.
        # On some Linux hosts, /tmp is RAM-backed (tmpfs), which defeats
        # the purpose of disk caching and can cause RAM spikes.
        with tempfile.TemporaryDirectory(
            dir=str(cache_dir), prefix=f"{cache_key}_arrow_"
        ) as arrow_tmp:
            dataset = Dataset.from_generator(gen, cache_dir=arrow_tmp)
            dataset.save_to_disk(str(tmp_dir))

        # Write metadata for debugging and validation
        code_fingerprint = _get_code_fingerprint()
        metadata = {
            "cache_key": cache_key,
            "env_id": env_id,
            "config": {
                "split": config.split,
                "mode": config.mode,
                "difficulty": config.difficulty,
                "seed": config.seed,
                "num_examples": config.num_examples,
            },
            "version": _get_pkg_version(),
            "code_fingerprint": code_fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (tmp_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Write READY marker (signals cache is complete)
        (tmp_dir / "_READY").write_text(
            json.dumps({"cache_key": cache_key, "created_at": metadata["created_at"]}),
            encoding="utf-8",
        )

        # Atomic rename temp → final
        tmp_dir.replace(dataset_dir)

        # Load back with memory mapping
        return Dataset.load_from_disk(str(dataset_dir))


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

        # Track by difficulty (now available in info dict)
        difficulty = info.get("difficulty", "medium")
        if difficulty in stats["by_difficulty"]:
            stats["by_difficulty"][difficulty] += 1

        if info.get("solvable", True):
            stats["solvable"] += 1
        else:
            stats["unsolvable"] += 1

    return stats
