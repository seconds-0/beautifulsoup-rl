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

from bs4_env.adapters.verifiers_adapter import MinimalEnv, build_verifiers_environment
from bs4_env.config import EnvConfig


def load_environment(
    split: str = "bench",
    mode: str = "mvp",
    difficulty: str = "mixed",
    num_examples: int | None = None,
    seed: int = 42,
    executor_backend: str = "local",
    network_access: bool = False,
    timeout_s: float = 30.0,
    max_output_chars: int = 10000,
    archetypes: list[str] | None = None,
    # Prime sandbox-specific settings (only used when executor_backend="prime")
    docker_image: str | None = None,
    cpu_cores: int = 1,
    memory_gb: int = 2,
    timeout_minutes: int = 30,
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
        executor_backend: Code executor - "local", "prime", or "pooled".
        network_access: Allow network in sandbox (should be False).
        timeout_s: Code execution timeout in seconds.
        max_output_chars: Max characters from stdout/stderr.
        archetypes: Specific archetype IDs to include (optional).
        docker_image: Docker image for Prime sandbox (default: python:3.11-slim).
        cpu_cores: CPU cores for Prime sandbox (default: 1).
        memory_gb: Memory in GB for Prime sandbox (default: 2).
        timeout_minutes: Sandbox lifecycle timeout in minutes (default: 30).
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
        docker_image=docker_image,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        timeout_minutes=timeout_minutes,
    )

    return build_verifiers_environment(config, **kwargs)


# Convenience exports
__all__ = [
    "load_environment",
    "EnvConfig",
    "MinimalEnv",
]

# Package metadata - read version from pyproject.toml via importlib.metadata
from importlib.metadata import PackageNotFoundError, version as _get_version

try:
    __version__ = _get_version("beautiful-soup-env")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development without install

__author__ = "Alex"
__description__ = "RL environment for BeautifulSoup HTML parsing tasks"
