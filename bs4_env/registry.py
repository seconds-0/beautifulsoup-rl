from __future__ import annotations

"""Archetype registry for BeautifulSoup RL environment.

This module provides the registration system for task archetypes. Each archetype
is a parameterized generator that produces task instances from a seed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from bs4_env.generators.base import Generator


@dataclass
class ArchetypeSpec:
    """Specification for a task archetype.

    Attributes:
        archetype_id: Unique identifier for the archetype (e.g., "mvp.extract_text_by_id").
        generator_class: The Generator class that produces task instances.
        category: High-level category (e.g., "core_extraction", "table_parsing").
        difficulty: Default difficulty level.
        solvable: Whether tasks from this archetype are solvable with BS4.
            False for limitation detection tasks.
        description: Human-readable description of what this archetype tests.
        tags: Additional tags for filtering and organization.
        phase: Which phase this archetype belongs to (1 for MVP, 2 for expansion).
        answer_schema: JSON schema for the expected answer format.
        allowed_limit_reasons: Valid reasons if this is a limitation archetype.
        evidence_patterns: Regex patterns that valid evidence must match.
    """

    archetype_id: str
    generator_class: type[Generator]
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    solvable: bool = True
    description: str = ""
    tags: list[str] = field(default_factory=list)
    phase: int = 1
    answer_schema: dict = field(default_factory=dict)
    allowed_limit_reasons: list[str] = field(default_factory=list)
    evidence_patterns: list[str] = field(default_factory=list)


# Global registry of archetypes
_REGISTRY: dict[str, ArchetypeSpec] = {}


def register(
    archetype_id: str,
    category: str,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    solvable: bool = True,
    description: str = "",
    tags: list[str] | None = None,
    phase: int = 1,
    answer_schema: dict | None = None,
    allowed_limit_reasons: list[str] | None = None,
    evidence_patterns: list[str] | None = None,
) -> Callable[[type[Generator]], type[Generator]]:
    """Decorator to register a generator class as an archetype.

    Usage:
        @register(
            archetype_id="mvp.extract_text_by_id",
            category="core_extraction",
            difficulty="easy",
            description="Extract text from element with specific ID",
        )
        class ExtractTextByIdGenerator(Generator):
            ...

    Args:
        archetype_id: Unique identifier for the archetype.
        category: High-level category for grouping.
        difficulty: Default difficulty level.
        solvable: Whether this archetype produces solvable tasks.
        description: Human-readable description.
        tags: Additional tags for filtering.
        phase: Implementation phase (1=MVP, 2=expansion).
        answer_schema: JSON schema for expected answer format.
        allowed_limit_reasons: Valid limitation reasons (for solvable=False).
        evidence_patterns: Regex patterns for evidence validation.

    Returns:
        Decorator function that registers the class.

    Raises:
        ValueError: If archetype_id is already registered.
    """

    def decorator(cls: type[Generator]) -> type[Generator]:
        if archetype_id in _REGISTRY:
            raise ValueError(f"Archetype '{archetype_id}' is already registered")

        spec = ArchetypeSpec(
            archetype_id=archetype_id,
            generator_class=cls,
            category=category,
            difficulty=difficulty,
            solvable=solvable,
            description=description,
            tags=tags or [],
            phase=phase,
            answer_schema=answer_schema or {},
            allowed_limit_reasons=allowed_limit_reasons or [],
            evidence_patterns=evidence_patterns or [],
        )
        _REGISTRY[archetype_id] = spec

        # Store spec on class for easy access
        cls._archetype_spec = spec

        return cls

    return decorator


def get_archetype(archetype_id: str) -> ArchetypeSpec:
    """Get an archetype specification by ID.

    Args:
        archetype_id: The unique identifier of the archetype.

    Returns:
        The ArchetypeSpec for the given ID.

    Raises:
        KeyError: If no archetype with the given ID is registered.
    """
    if archetype_id not in _REGISTRY:
        raise KeyError(f"Unknown archetype: '{archetype_id}'")
    return _REGISTRY[archetype_id]


def list_archetypes(
    category: str | None = None,
    difficulty: str | None = None,
    solvable: bool | None = None,
    phase: int | None = None,
    tags: list[str] | None = None,
) -> list[ArchetypeSpec]:
    """List archetypes matching the given filters.

    Args:
        category: Filter by category.
        difficulty: Filter by difficulty level.
        solvable: Filter by solvability.
        phase: Filter by implementation phase.
        tags: Filter by tags (must have all specified tags).

    Returns:
        List of matching ArchetypeSpec objects.
    """
    results = []
    for spec in _REGISTRY.values():
        if category is not None and spec.category != category:
            continue
        if difficulty is not None and spec.difficulty != difficulty:
            continue
        if solvable is not None and spec.solvable != solvable:
            continue
        if phase is not None and spec.phase != phase:
            continue
        if tags is not None and not all(tag in spec.tags for tag in tags):
            continue
        results.append(spec)
    return results


def get_all_archetype_ids() -> list[str]:
    """Get all registered archetype IDs.

    Returns:
        Sorted list of all archetype IDs.
    """
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered archetypes. Useful for testing."""
    _REGISTRY.clear()


def get_registry_stats() -> dict[str, Any]:
    """Get statistics about the registry.

    Returns:
        Dictionary with counts by category, difficulty, phase, etc.
    """
    stats: dict[str, Any] = {
        "total": len(_REGISTRY),
        "by_category": {},
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "by_phase": {1: 0, 2: 0},
        "solvable": 0,
        "limitation": 0,
    }

    for spec in _REGISTRY.values():
        # By category
        if spec.category not in stats["by_category"]:
            stats["by_category"][spec.category] = 0
        stats["by_category"][spec.category] += 1

        # By difficulty
        stats["by_difficulty"][spec.difficulty] += 1

        # By phase
        if spec.phase in stats["by_phase"]:
            stats["by_phase"][spec.phase] += 1

        # Solvable vs limitation
        if spec.solvable:
            stats["solvable"] += 1
        else:
            stats["limitation"] += 1

    return stats
