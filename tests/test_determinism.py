"""Tests for deterministic task generation."""

import pytest

from bs4_env.generators.base import make_rng, stable_int_seed, TaskInstance
from bs4_env.registry import get_archetype, list_archetypes, clear_registry


class TestStableSeed:
    """Tests for stable seed generation."""

    def test_same_inputs_same_seed(self):
        """Same inputs should produce same seed."""
        seed1 = stable_int_seed("train", "mvp.test", 42)
        seed2 = stable_int_seed("train", "mvp.test", 42)
        assert seed1 == seed2

    def test_different_splits_different_seeds(self):
        """Different splits should produce different seeds."""
        seed_train = stable_int_seed("train", "mvp.test", 42)
        seed_eval = stable_int_seed("eval", "mvp.test", 42)
        seed_bench = stable_int_seed("bench", "mvp.test", 42)

        assert seed_train != seed_eval
        assert seed_train != seed_bench
        assert seed_eval != seed_bench

    def test_different_archetypes_different_seeds(self):
        """Different archetypes should produce different seeds."""
        seed1 = stable_int_seed("train", "mvp.test1", 42)
        seed2 = stable_int_seed("train", "mvp.test2", 42)
        assert seed1 != seed2

    def test_different_seeds_different_output(self):
        """Different base seeds should produce different output."""
        seed1 = stable_int_seed("train", "mvp.test", 1)
        seed2 = stable_int_seed("train", "mvp.test", 2)
        assert seed1 != seed2

    def test_seed_is_deterministic_across_calls(self):
        """Seeds should be consistent across multiple calls."""
        results = [stable_int_seed("train", "mvp.test", 100) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestMakeRng:
    """Tests for RNG creation."""

    def test_same_inputs_same_sequence(self):
        """Same inputs should produce same random sequence."""
        rng1 = make_rng("mvp.test", 42)
        rng2 = make_rng("mvp.test", 42)

        seq1 = [rng1.randint(0, 1000) for _ in range(10)]
        seq2 = [rng2.randint(0, 1000) for _ in range(10)]

        assert seq1 == seq2

    def test_different_seeds_different_sequence(self):
        """Different seeds should produce different sequences."""
        rng1 = make_rng("mvp.test", 1)
        rng2 = make_rng("mvp.test", 2)

        seq1 = [rng1.randint(0, 1000) for _ in range(10)]
        seq2 = [rng2.randint(0, 1000) for _ in range(10)]

        assert seq1 != seq2


class TestGeneratorDeterminism:
    """Tests that generators produce deterministic output.

    These tests will be expanded as archetypes are implemented.
    """

    def test_generator_same_seed_same_output(self):
        """Generator with same seed should produce identical task."""
        # Import auto_import to ensure generators are registered
        from bs4_env import auto_import  # noqa: F401

        archetypes = list_archetypes()

        for spec in archetypes:
            generator = spec.generator_class()
            seed = 12345

            task1 = generator.generate(seed)
            task2 = generator.generate(seed)

            assert task1.html == task2.html, f"{spec.archetype_id}: HTML differs"
            assert task1.query == task2.query, f"{spec.archetype_id}: query differs"
            assert (
                task1.ground_truth == task2.ground_truth
            ), f"{spec.archetype_id}: ground_truth differs"

    def test_generator_different_seeds_different_output(self):
        """Generator with different seeds should produce different tasks."""
        from bs4_env import auto_import  # noqa: F401

        archetypes = list_archetypes()

        for spec in archetypes:
            generator = spec.generator_class()

            task1 = generator.generate(1)
            task2 = generator.generate(2)

            # At least one of HTML, query, or ground_truth should differ
            same = (
                task1.html == task2.html
                and task1.query == task2.query
                and task1.ground_truth == task2.ground_truth
            )
            assert not same, f"{spec.archetype_id}: different seeds produced same output"
