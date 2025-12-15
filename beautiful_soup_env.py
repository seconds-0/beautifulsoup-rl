from __future__ import annotations

"""BeautifulSoup RL Environment - Main Entrypoint.

This module provides the `load_environment` function required by Prime's
Environments Hub. It is the public API for creating environment instances.

Usage:
    from beautiful_soup_env import load_environment

    # Create environment with defaults
    env = load_environment()

    # Create environment with custom config
    env = load_environment(
        split="train",
        mode="mvp",
        difficulty="medium",
        executor_backend="local",
    )
"""

from typing import Any

from bs4_env.config import EnvConfig
from bs4_env.adapters.verifiers_adapter import build_verifiers_environment, MinimalEnv


def load_environment(
    split: str = "train",
    mode: str = "mvp",
    difficulty: str = "mixed",
    num_examples: int | None = None,
    seed: int = 42,
    executor_backend: str = "local",
    network_access: bool = False,
    timeout_s: float = 30.0,
    max_output_chars: int = 10000,
    archetypes: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Load the BeautifulSoup RL environment.

    This is the main entrypoint required by Prime's Environments Hub.

    Args:
        split: Dataset split - "train", "eval", or "bench".
        mode: Which archetypes to include - "mvp", "phase2", or "all".
        difficulty: Task difficulty - "easy", "medium", "hard", or "mixed".
        num_examples: Number of examples (None for default based on split).
        seed: Random seed for reproducibility.
        executor_backend: Code executor - "local" or "prime".
        network_access: Allow network in sandbox (should be False).
        timeout_s: Code execution timeout in seconds.
        max_output_chars: Max characters from stdout/stderr.
        archetypes: Specific archetype IDs to include (optional).
        **kwargs: Additional arguments (for future compatibility).

    Returns:
        Environment instance. Returns a Verifiers vf.Environment if the
        verifiers package is installed, otherwise returns MinimalEnv for
        local testing.

    Example:
        >>> env = load_environment(split="bench", mode="mvp")
        >>> example = env.get_example(0)
        >>> print(example["query"])
    """
    config = EnvConfig(
        split=split,
        mode=mode,
        difficulty=difficulty,
        num_examples=num_examples,
        seed=seed,
        executor_backend=executor_backend,
        network_access=network_access,
        timeout_s=timeout_s,
        max_output_chars=max_output_chars,
        archetypes=archetypes,
    )

    return build_verifiers_environment(config)


# Convenience exports
__all__ = [
    "load_environment",
    "EnvConfig",
    "MinimalEnv",
]

# Package metadata
__version__ = "0.1.0"
__author__ = "Alex"
__description__ = "RL environment for BeautifulSoup HTML parsing tasks"
