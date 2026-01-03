from __future__ import annotations

"""Environment configuration for BeautifulSoup RL environment."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvConfig:
    """Configuration for the BeautifulSoup RL environment.

    Attributes:
        split: Dataset split to use. "train" for training, "eval" for evaluation
            with disjoint seeds, "bench" for fixed benchmark comparison.
        mode: Which archetypes to include:
            - "mvp": Phase 1 archetypes only
            - "phase2": Phase 2 archetypes only
            - "all": All archetypes
            - "hard_only": Only hard difficulty archetypes (for challenging benchmarks)
            - "tiered": All archetypes with difficulty-weighted sampling
            - "bootstrap": Primer + easy archetypes for 0% model onboarding
        difficulty: Task difficulty filter. "primer", "easy", "medium", "hard", or "mixed".
            "primer" includes ultra-simple tasks for teaching the basic action template.
        complexity: HTML complexity level for generated tasks:
            - "primer": Ultra-simple, single element (e.g., <span id="target">Hello</span>)
            - "low": Simple structure, no chrome/noise
            - "moderate": Real patterns with sparse boilerplate
            - "realistic": Full noise, framework patterns, chrome (default)
        partial_credit_enabled: Enable process-based partial credit for 0% models.
            Awards credit for correct tool-use patterns (importing BS4, creating soup,
            using selection methods) even when the final answer is wrong. Capped at 0.30.
        difficulty_weights: For tiered mode, relative weights for each difficulty.
            Default gives more weight to harder tasks for RL training signal.
        num_examples: Number of examples to generate. None for unlimited/default.
        seed: Base random seed for reproducibility.
        executor_backend: Which executor to use:
            - "local": subprocess-based local execution (development/testing)
            - "prime": Prime's sandboxed execution (production/bounty submission)
            - "pooled": persistent worker pool for high-throughput training
        network_access: Whether to allow network access in sandbox. Should be False
            for determinism and safety.
        timeout_s: Maximum execution time for code in seconds.
        max_output_chars: Maximum characters to return from stdout/stderr.
        archetypes: Optional list of specific archetype IDs to include.
            If None, includes all archetypes matching mode/difficulty.
    """

    split: Literal["train", "eval", "bench"] = "bench"
    mode: Literal["mvp", "phase2", "all", "hard_only", "tiered", "bootstrap"] = "mvp"
    difficulty: Literal["primer", "easy", "medium", "hard", "mixed"] = "mixed"
    complexity: Literal["primer", "low", "moderate", "realistic"] = "realistic"
    difficulty_weights: dict[str, float] = field(
        default_factory=lambda: {
            "primer": 0.0,  # 0% primer tasks (only via explicit difficulty="primer")
            "easy": 0.2,  # 20% easy tasks
            "medium": 0.4,  # 40% medium tasks
            "hard": 0.4,  # 40% hard tasks (overweight for RL signal)
        }
    )
    # Enable process-based partial credit for 0% models learning the tool-use pattern
    partial_credit_enabled: bool = True
    num_examples: int | None = None
    seed: int = 42
    executor_backend: Literal["local", "prime", "pooled"] = "local"
    network_access: bool = False
    timeout_s: float = 30.0
    max_output_chars: int = 10000
    archetypes: list[str] | None = None
    # Prime sandbox-specific settings (only used when executor_backend="prime")
    docker_image: str | None = None  # Docker image for sandbox (default: python:3.11-slim)
    cpu_cores: int = 1  # CPU cores to allocate
    memory_gb: int = 2  # Memory allocation in GB
    timeout_minutes: int = 30  # Sandbox lifecycle timeout in minutes

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")
        if self.max_output_chars <= 0:
            raise ValueError(f"max_output_chars must be positive, got {self.max_output_chars}")
        if self.num_examples is not None and self.num_examples <= 0:
            raise ValueError(f"num_examples must be positive or None, got {self.num_examples}")


@dataclass
class TaskConstraints:
    """Constraints passed to the model during task execution.

    These are the public constraints visible to the model, NOT including
    ground truth or other hidden grading information.

    Attributes:
        output_schema: JSON schema for the expected answer format.
        allowed_limit_reasons: List of valid reasons for limitation status.
            Empty if the task should always be solvable.
        safety_notes: Human-readable safety reminders (e.g., "Do not extract passwords").
        parser_hint: Optional hint about which parser to use.
    """

    output_schema: dict = field(default_factory=dict)
    allowed_limit_reasons: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    parser_hint: str | None = None


# Default schemas for common answer types
STRING_SCHEMA = {"type": "string"}
STRING_LIST_SCHEMA = {"type": "array", "items": {"type": "string"}}
LIST_SCHEMA = STRING_LIST_SCHEMA  # Alias for string lists
INT_SCHEMA = {"type": "integer"}
INT_LIST_SCHEMA = {"type": "array", "items": {"type": "integer"}}
FLOAT_SCHEMA = {"type": "number"}
BOOL_SCHEMA = {"type": "boolean"}
DICT_SCHEMA = {"type": "object", "additionalProperties": {"type": ["string", "null"]}}

LINK_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "href": {"type": "string"},
    },
    "required": ["text", "href"],
}

LINK_LIST_SCHEMA = {"type": "array", "items": LINK_OBJECT_SCHEMA}

IMAGE_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "src": {"type": "string"},
        "alt": {"type": ["string", "null"]},
    },
    "required": ["src", "alt"],
}

IMAGE_LIST_SCHEMA = {"type": "array", "items": IMAGE_OBJECT_SCHEMA}

TABLE_ROW_SCHEMA = {"type": "array", "items": {"type": ["string", "null"]}}
TABLE_SCHEMA = {"type": "array", "items": TABLE_ROW_SCHEMA}

DICT_LIST_SCHEMA = {
    "type": "array",
    "items": {"type": "object", "additionalProperties": {"type": ["string", "null"]}},
}
