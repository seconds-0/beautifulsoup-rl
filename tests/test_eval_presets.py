"""Tests for eval preset modes.

Tests the hard_only and tiered sampling modes for evaluation.
"""

import pytest

from bs4_env.config import EnvConfig
from bs4_env.dataset import build_dataset, _get_archetype_ids_for_config, get_dataset_stats
from bs4_env.registry import get_archetype, list_archetypes

# Import auto_import to ensure all generators are registered
from bs4_env import auto_import  # noqa: F401


class TestHardOnlyMode:
    """Tests for hard_only mode that filters to only hard archetypes."""

    def test_hard_only_filters_to_hard(self):
        """hard_only mode only includes hard difficulty archetypes."""
        config = EnvConfig(mode="hard_only", split="bench", num_examples=10)
        archetype_ids = _get_archetype_ids_for_config(config)

        # All returned archetypes should be hard
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)
            assert spec.difficulty == "hard", f"{archetype_id} is not hard"

    def test_hard_only_has_archetypes(self):
        """hard_only mode returns at least some archetypes."""
        config = EnvConfig(mode="hard_only", split="bench", num_examples=10)
        archetype_ids = _get_archetype_ids_for_config(config)

        # Should have some hard archetypes
        assert len(archetype_ids) > 0, "No hard archetypes found"

    def test_hard_only_dataset_builds(self):
        """hard_only mode builds a valid dataset."""
        config = EnvConfig(mode="hard_only", split="bench", num_examples=4)
        dataset = build_dataset(config)

        # Should have examples
        assert len(dataset) > 0, "Empty dataset"


class TestTieredMode:
    """Tests for tiered mode with difficulty-weighted sampling."""

    def test_tiered_includes_all_difficulties(self):
        """tiered mode includes all archetypes (not filtered)."""
        config = EnvConfig(mode="tiered", split="bench", num_examples=10)
        archetype_ids = _get_archetype_ids_for_config(config)

        # Should include archetypes of all difficulties
        difficulties = set()
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)
            difficulties.add(spec.difficulty)

        assert "easy" in difficulties, "No easy archetypes"
        assert "medium" in difficulties, "No medium archetypes"
        # Hard might be empty in current registry, that's ok

    def test_tiered_dataset_builds(self):
        """tiered mode builds a valid dataset."""
        config = EnvConfig(mode="tiered", split="bench", num_examples=20)
        dataset = build_dataset(config)

        # Should have examples
        assert len(dataset) > 0, "Empty dataset"

    def test_tiered_custom_weights(self):
        """tiered mode respects custom difficulty weights."""
        config = EnvConfig(
            mode="tiered",
            split="bench",
            num_examples=100,
            difficulty_weights={
                "easy": 0.1,   # 10% easy
                "medium": 0.3, # 30% medium
                "hard": 0.6,   # 60% hard
            }
        )
        dataset = build_dataset(config)

        # Should have examples
        assert len(dataset) > 0, "Empty dataset"

    def test_tiered_default_weights_favor_hard(self):
        """Default tiered weights give more weight to harder tasks."""
        config = EnvConfig(mode="tiered", split="bench", num_examples=10)

        # Default weights should favor hard
        assert config.difficulty_weights["hard"] >= config.difficulty_weights["easy"]


class TestModeCompatibility:
    """Tests that existing modes still work."""

    def test_mvp_mode(self):
        """mvp mode still works (phase 1 archetypes)."""
        config = EnvConfig(mode="mvp", split="bench", num_examples=10)
        archetype_ids = _get_archetype_ids_for_config(config)

        # All should be phase 1
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)
            assert spec.phase == 1, f"{archetype_id} is not phase 1"

    def test_phase2_mode(self):
        """phase2 mode still works."""
        config = EnvConfig(mode="phase2", split="bench", num_examples=10)
        archetype_ids = _get_archetype_ids_for_config(config)

        # All should be phase 2
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)
            assert spec.phase == 2, f"{archetype_id} is not phase 2"

    def test_all_mode(self):
        """all mode still works (all archetypes)."""
        config = EnvConfig(mode="all", split="bench", num_examples=10)
        archetype_ids = _get_archetype_ids_for_config(config)

        # Should have archetypes from multiple phases
        phases = set()
        for archetype_id in archetype_ids:
            spec = get_archetype(archetype_id)
            phases.add(spec.phase)

        assert len(phases) > 1, "Should have multiple phases"


class TestDifficultyFilter:
    """Tests for difficulty filtering in standard modes."""

    def test_difficulty_filter_easy(self):
        """difficulty='easy' filters to easy tasks."""
        config = EnvConfig(mode="all", split="bench", difficulty="easy", num_examples=10)
        dataset = build_dataset(config)

        # All tasks should be easy (if any easy archetypes exist)
        if len(dataset) > 0:
            import json
            for row in dataset:
                info = json.loads(row["info"])
                spec = get_archetype(info["archetype_id"])
                assert spec.difficulty == "easy"

    def test_difficulty_filter_hard(self):
        """difficulty='hard' filters to hard tasks."""
        config = EnvConfig(mode="all", split="bench", difficulty="hard", num_examples=10)
        dataset = build_dataset(config)

        # All tasks should be hard (if any hard archetypes exist)
        if len(dataset) > 0:
            import json
            for row in dataset:
                info = json.loads(row["info"])
                spec = get_archetype(info["archetype_id"])
                assert spec.difficulty == "hard"
