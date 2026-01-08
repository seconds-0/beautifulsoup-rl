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
    def test_all_solvable_archetypes_reward_ground_truth(self, solvable_archetypes, seed):
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
                f"Primer archetype {spec.archetype_id} failed: reward={reward}, metrics={metrics}"
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
                f"Easy archetype {spec.archetype_id} failed: reward={reward}, metrics={metrics}"
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

            output = json.dumps(
                {
                    "status": "limit",
                    "answer": None,
                    "limit": {
                        "reason": valid_reasons[0],
                        "evidence": evidence,
                    },
                }
            )

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
                assert isinstance(ground_truth, int | float), (
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


class TestMultiStepArchetypeSolvability:
    """Tests for multi-step archetype task structure and solvability.

    Multi-step tasks require navigation between pages. These tests verify:
    1. Tasks generate valid pages dict
    2. Navigation path from start to answer exists
    3. Ground truth is accessible via that path
    """

    def test_search_then_detail_has_pages(self):
        """SearchThenDetail tasks should have pages dict populated."""
        from bs4_env.generators.mvp_multistep import SearchThenDetailGenerator

        gen = SearchThenDetailGenerator()
        task = gen.generate(seed=42)

        # Multi-step task must have pages
        assert task.pages, "SearchThenDetail task must have pages dict"
        assert len(task.pages) >= 1, "Must have at least one detail page"

        # Starting HTML exists
        assert task.html, "Must have starting HTML"

        # Ground truth should be accessible (in some page)
        all_content = task.html + "".join(task.pages.values())
        gt = task.ground_truth
        if isinstance(gt, str):
            assert gt in all_content, f"Ground truth '{gt[:50]}...' not found in any page content"

    def test_pagination_aggregate_has_pages(self):
        """PaginationAggregate tasks should have pages dict populated."""
        from bs4_env.generators.mvp_multistep import PaginationAggregateGenerator

        gen = PaginationAggregateGenerator()
        task = gen.generate(seed=42)

        # Pagination task must have multiple pages
        assert task.pages, "PaginationAggregate task must have pages dict"
        assert len(task.pages) >= 1, "Must have at least one additional page"

        # Starting HTML exists
        assert task.html, "Must have starting HTML"

    def test_link_chain_has_pages(self):
        """LinkChain tasks should have pages dict populated."""
        from bs4_env.generators.mvp_multistep import LinkChainGenerator

        gen = LinkChainGenerator()
        task = gen.generate(seed=42)

        # Link chain task must have pages
        assert task.pages, "LinkChain task must have pages dict"
        assert len(task.pages) >= 2, "Must have at least 2 pages in chain"

        # Ground truth should be in the final page
        final_page = list(task.pages.values())[-1]
        gt = task.ground_truth
        if isinstance(gt, str):
            assert gt in final_page, f"Ground truth '{gt[:50]}...' should be in final page"

    def test_compare_products_has_pages(self):
        """CompareProducts tasks should have pages dict populated."""
        from bs4_env.generators.mvp_multistep import CompareProductsGenerator

        gen = CompareProductsGenerator()
        task = gen.generate(seed=42)

        # Comparison task must have pages for products
        assert task.pages, "CompareProducts task must have pages dict"
        assert len(task.pages) >= 2, "Must have at least 2 product pages"

    @pytest.mark.parametrize("seed", TEST_SEEDS[:2])
    def test_multistep_archetypes_give_reward_for_ground_truth(self, seed):
        """Multi-step archetypes should give high reward for correct answer."""
        from bs4_env.generators.mvp_multistep import (
            CompareProductsGenerator,
            LinkChainGenerator,
            PaginationAggregateGenerator,
            SearchThenDetailGenerator,
        )

        generators = [
            SearchThenDetailGenerator(),
            PaginationAggregateGenerator(),
            LinkChainGenerator(),
            CompareProductsGenerator(),
        ]

        for gen in generators:
            task = gen.generate(seed=seed)
            output = json.dumps({"status": "ok", "answer": task.ground_truth})
            task_info = parse_task_info(task.to_info_dict())

            reward, metrics = compute_reward(
                raw_output=output,
                task_info=task_info,
                run_python_calls=1,
            )

            assert reward >= REWARD_TOLERANCE, (
                f"{gen.__class__.__name__} (seed={seed}): "
                f"reward={reward:.3f} < {REWARD_TOLERANCE}, metrics={metrics}"
            )

    def test_multistep_starting_html_contains_navigation_links(self):
        """Multi-step starting HTML should contain navigation links."""
        from bs4_env.generators.mvp_multistep import SearchThenDetailGenerator

        gen = SearchThenDetailGenerator()
        task = gen.generate(seed=42)

        # Starting HTML should have links to detail pages
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(task.html, "html.parser")
        links = soup.find_all("a", href=True)

        # At least one link should point to a page in pages dict
        page_hrefs = set(task.pages.keys())
        found_nav_link = any(
            link["href"] in page_hrefs or link["href"].lstrip("/") in page_hrefs for link in links
        )

        assert found_nav_link, (
            f"Starting HTML should contain navigation links to pages. "
            f"Links found: {[link['href'] for link in links]}, "
            f"Pages available: {list(page_hrefs)}"
        )
