"""Tests for curriculum learning feature.

NOTE: Curriculum learning is not yet implemented. These tests validate
the configuration dataclasses only. See bs4_env/adapters/verifiers_adapter.py
for architecture notes on future implementation.
"""

from __future__ import annotations

import pytest

from bs4_env.config import CurriculumPhase, EnvConfig


class TestCurriculumPhase:
    """Tests for CurriculumPhase dataclass."""

    def test_valid_phase(self):
        """Test creating a valid curriculum phase."""
        phase = CurriculumPhase(
            until_step=100,
            weights={"primer": 0.8, "easy": 0.2},
        )
        assert phase.until_step == 100
        assert phase.weights["primer"] == 0.8
        assert phase.weights["easy"] == 0.2

    def test_invalid_until_step(self):
        """Test that until_step must be positive."""
        with pytest.raises(ValueError, match="until_step must be positive"):
            CurriculumPhase(until_step=0, weights={"easy": 1.0})

        with pytest.raises(ValueError, match="until_step must be positive"):
            CurriculumPhase(until_step=-10, weights={"easy": 1.0})

    def test_invalid_difficulty_key(self):
        """Test that invalid difficulty keys are rejected."""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            CurriculumPhase(until_step=100, weights={"invalid": 1.0})

    def test_valid_difficulties(self):
        """Test that all valid difficulty keys are accepted."""
        phase = CurriculumPhase(
            until_step=100,
            weights={"primer": 0.25, "easy": 0.25, "medium": 0.25, "hard": 0.25},
        )
        assert len(phase.weights) == 4


class TestEnvConfigCurriculum:
    """Tests for curriculum fields in EnvConfig."""

    def test_curriculum_disabled_by_default(self):
        """Test that curriculum is disabled by default."""
        config = EnvConfig()
        assert config.curriculum_enabled is False
        assert config.curriculum_phases == []
        assert config.samples_per_step == 256

    def test_curriculum_enabled_with_phases(self):
        """Test enabling curriculum with phases."""
        phases = [
            CurriculumPhase(until_step=100, weights={"primer": 0.8, "easy": 0.2}),
            CurriculumPhase(until_step=200, weights={"easy": 0.6, "medium": 0.4}),
        ]
        config = EnvConfig(
            curriculum_enabled=True,
            curriculum_phases=phases,
            samples_per_step=128,
        )
        assert config.curriculum_enabled is True
        assert len(config.curriculum_phases) == 2
        assert config.samples_per_step == 128


class TestCurriculumNotImplemented:
    """Placeholder tests for future curriculum implementation.

    The previous curriculum implementation was removed because it used
    class-level state that doesn't work with multi-process training.

    Future implementation should:
    1. Accept training step from external orchestrator (not internal counter)
    2. Use immutable weights (not mutate shared config)
    3. Have no class-level state (all instance-level or external)
    4. Be deterministic (same step -> same weights across all workers)

    See: bs4_env/adapters/verifiers_adapter.py for detailed architecture notes.
    """

    def test_curriculum_config_exists_but_not_implemented(self):
        """Verify curriculum config exists but has no implementation."""
        config = EnvConfig(curriculum_enabled=True)
        # Config can be created, but there's no runtime behavior yet
        assert config.curriculum_enabled is True
        # Weights are static - they don't change based on training step
        assert config.difficulty_weights == {
            "primer": 0.0,
            "easy": 0.2,
            "medium": 0.4,
            "hard": 0.4,
        }

    # TODO: Add multi-process curriculum tests when reimplemented
    # - Test that weights are deterministic across workers
    # - Test that step injection works correctly
    # - Test that config is not mutated at runtime
