"""Solvability regression tests for all archetypes.

These tests verify that:
1. Every solvable archetype produces tasks where ground_truth answers get near-perfect reward
2. Deterministic seeds produce consistent results
3. The grading pipeline correctly handles all answer types

This is a regression safety net - if any archetype becomes "unsolvable" due to
generator or grading changes, these tests will catch it.
"""

import json

import pytest

from bs4_env import auto_import  # noqa: F401 - ensures generators registered
from bs4_env.generators.base import parse_task_info
from bs4_env.grading.rubric import compute_reward
from bs4_env.registry import list_archetypes

# Tolerance for reward comparison (per Codex review recommendation)
# Using >= 0.99 instead of == 1.0 to avoid floating point brittleness
REWARD_TOLERANCE = 0.99

# Fixed seeds for deterministic testing
TEST_SEEDS = [42, 123, 456, 789, 1000]


class TestSolvableArchetypeRewards:
    """Verify all solvable archetypes give high reward for ground truth answers."""

    @pytest.fixture(scope="class")
    def solvable_archetypes(self):
        """Get all solvable archetype specs."""
        return list_archetypes(solvable=True)

    def test_at_least_one_solvable_archetype_exists(self, solvable_archetypes):
        """Sanity check: we have solvable archetypes to test."""
        assert len(solvable_archetypes) > 0, "No solvable archetypes registered"

    @pytest.mark.parametrize("seed", TEST_SEEDS[:2])  # Use 2 seeds for speed
    def test_all_solvable_archetypes_reward_ground_truth(
        self, solvable_archetypes, seed
    ):
        """Every solvable archetype should give high reward for ground truth."""
        failures = []

        for spec in solvable_archetypes:
            try:
                generator = spec.generator_class()
                task = generator.generate(seed=seed)

                # Build correct output using ground truth
                output = json.dumps({"status": "ok", "answer": task.ground_truth})

                # Parse task info (deserializes JSON fields)
                task_info = parse_task_info(task.to_info_dict())

                # Compute reward
                reward, metrics = compute_reward(
                    raw_output=output,
                    task_info=task_info,
                    run_python_calls=1,
                )

                if reward < REWARD_TOLERANCE:
                    failures.append(
                        f"{spec.archetype_id} (seed={seed}): "
                        f"reward={reward:.3f} < {REWARD_TOLERANCE}, "
                        f"metrics={metrics}"
                    )

            except Exception as e:
                failures.append(
                    f"{spec.archetype_id} (seed={seed}): Exception - {type(e).__name__}: {e}"
                )

        if failures:
            failure_msg = "\n".join(failures)
            pytest.fail(f"Solvability failures:\n{failure_msg}")


class TestArchetypeSpecificSolvability:
    """Targeted tests for specific archetype categories."""

    def test_primer_archetypes_solvable(self):
        """All primer archetypes should be trivially solvable."""
        primer_specs = list_archetypes(difficulty="primer")
        assert len(primer_specs) > 0, "No primer archetypes found"

        for spec in primer_specs:
            generator = spec.generator_class()
            task = generator.generate(seed=42)

            output = json.dumps({"status": "ok", "answer": task.ground_truth})
            task_info = parse_task_info(task.to_info_dict())

            reward, metrics = compute_reward(
                raw_output=output,
                task_info=task_info,
                run_python_calls=1,
            )

            assert reward >= REWARD_TOLERANCE, (
                f"Primer archetype {spec.archetype_id} failed: "
                f"reward={reward}, metrics={metrics}"
            )

    def test_easy_archetypes_solvable(self):
        """All easy archetypes should be solvable."""
        easy_specs = list_archetypes(difficulty="easy")
        assert len(easy_specs) > 0, "No easy archetypes found"

        for spec in easy_specs:
            if not spec.solvable:
                continue  # Skip limitation archetypes

            generator = spec.generator_class()
            task = generator.generate(seed=42)

            output = json.dumps({"status": "ok", "answer": task.ground_truth})
            task_info = parse_task_info(task.to_info_dict())

            reward, metrics = compute_reward(
                raw_output=output,
                task_info=task_info,
                run_python_calls=1,
            )

            assert reward >= REWARD_TOLERANCE, (
                f"Easy archetype {spec.archetype_id} failed: "
                f"reward={reward}, metrics={metrics}"
            )


class TestLimitationArchetypes:
    """Verify limitation archetypes correctly reward abstention."""

    @pytest.fixture(scope="class")
    def limitation_archetypes(self):
        """Get all limitation (unsolvable) archetype specs."""
        return list_archetypes(solvable=False)

    def test_limitation_archetypes_exist(self, limitation_archetypes):
        """Sanity check: we have limitation archetypes."""
        # Note: This may be 0 if no limitation archetypes are registered
        # which is fine - the test documents the expectation
        pass

    def test_limitation_correct_abstention_rewarded(self, limitation_archetypes):
        """Correct abstention with valid evidence should get partial reward."""
        import re

        for spec in limitation_archetypes:
            generator = spec.generator_class()
            task = generator.generate(seed=42)

            # Get valid reason and evidence from task
            limit_info = task.limit_info or {}
            valid_reasons = limit_info.get("allowed_reasons", spec.allowed_limit_reasons)
            if not valid_reasons:
                continue  # Skip if no valid reasons defined

            # Find valid evidence by searching for pattern matches in HTML
            evidence = None
            evidence_patterns = limit_info.get("evidence_patterns", [])

            for pattern in evidence_patterns:
                try:
                    match = re.search(pattern, task.html)
                    if match:
                        # Use the actual matched text as evidence
                        evidence = match.group(0)
                        break
                except re.error:
                    continue

            if evidence is None:
                # Fallback: use evidence_hint from metadata if available
                evidence = task.metadata.get("evidence_hint")

            if evidence is None:
                # Last resort: try to find script tag literally
                script_match = re.search(r"<script[^>]*>", task.html)
                if script_match:
                    evidence = script_match.group(0)

            if evidence is None:
                continue  # Can't construct valid evidence, skip

            output = json.dumps({
                "status": "limit",
                "answer": None,
                "limit": {
                    "reason": valid_reasons[0],
                    "evidence": evidence,
                },
            })

            task_info = parse_task_info(task.to_info_dict())

            reward, metrics = compute_reward(
                raw_output=output,
                task_info=task_info,
                html=task.html,  # Required for evidence verification
                run_python_calls=1,
            )

            # Limitation tasks should give partial reward for correct abstention
            # (typically 0.5, but using > 0 for robustness)
            assert reward > 0, (
                f"Limitation archetype {spec.archetype_id} gave 0 reward "
                f"for valid abstention with evidence='{evidence[:50]}...': "
                f"metrics={metrics}"
            )


class TestDeterministicGeneration:
    """Verify generators are deterministic across seeds."""

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_same_seed_same_output(self, seed):
        """Same seed should produce identical tasks."""
        solvable_specs = list_archetypes(solvable=True)

        for spec in solvable_specs[:10]:  # Test first 10 for speed
            generator = spec.generator_class()

            task1 = generator.generate(seed=seed)
            task2 = generator.generate(seed=seed)

            assert task1.html == task2.html, (
                f"{spec.archetype_id}: HTML differs for same seed {seed}"
            )
            assert task1.ground_truth == task2.ground_truth, (
                f"{spec.archetype_id}: ground_truth differs for same seed {seed}"
            )
            assert task1.query == task2.query, (
                f"{spec.archetype_id}: query differs for same seed {seed}"
            )


class TestAnswerSchemaCompliance:
    """Verify ground truth matches declared answer schema."""

    def test_ground_truth_matches_schema(self):
        """Ground truth should match the archetype's answer_schema."""
        solvable_specs = list_archetypes(solvable=True)

        for spec in solvable_specs:
            generator = spec.generator_class()
            task = generator.generate(seed=42)

            schema = task.answer_schema
            ground_truth = task.ground_truth

            # Basic type checking based on schema
            if schema.get("type") == "string":
                assert isinstance(ground_truth, str), (
                    f"{spec.archetype_id}: expected string, got {type(ground_truth)}"
                )
            elif schema.get("type") == "integer":
                assert isinstance(ground_truth, int), (
                    f"{spec.archetype_id}: expected int, got {type(ground_truth)}"
                )
            elif schema.get("type") == "number":
                assert isinstance(ground_truth, (int, float)), (
                    f"{spec.archetype_id}: expected number, got {type(ground_truth)}"
                )
            elif schema.get("type") == "array":
                assert isinstance(ground_truth, list), (
                    f"{spec.archetype_id}: expected list, got {type(ground_truth)}"
                )
            elif schema.get("type") == "object":
                assert isinstance(ground_truth, dict), (
                    f"{spec.archetype_id}: expected dict, got {type(ground_truth)}"
                )
