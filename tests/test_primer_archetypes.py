"""Tests for primer archetypes (0% model bootstrapping).

Primer archetypes are ultra-simple tasks that teach the basic action template:
1. Import BeautifulSoup
2. Parse HTML with BeautifulSoup(HTML, 'html.parser')
3. Select element with .find() or similar
4. Extract and return content

The HTML is intentionally minimal to remove all ambiguity.
"""

import json

import pytest
from bs4 import BeautifulSoup

from bs4_env import auto_import  # noqa: F401 - ensures registration
from bs4_env.generators.base import parse_task_info
from bs4_env.grading.rubric import REWARD_CORRECT, compute_reward
from bs4_env.registry import get_archetype, list_archetypes


class TestPrimerArchetypeRegistration:
    """Tests for primer archetype registration."""

    def test_all_primer_archetypes_registered(self):
        """All 5 primer archetypes are registered."""
        primer_archetypes = list_archetypes(difficulty="primer")
        primer_ids = [spec.archetype_id for spec in primer_archetypes]

        expected_ids = [
            "primer.extract_by_id",
            "primer.extract_by_class",
            "primer.extract_by_tag",
            "primer.extract_attribute",
            "primer.count_elements",
        ]

        for expected in expected_ids:
            assert expected in primer_ids, f"Missing primer archetype: {expected}"

    def test_primer_difficulty(self):
        """All primer archetypes have difficulty='primer'."""
        primer_archetypes = list_archetypes(difficulty="primer")
        for spec in primer_archetypes:
            assert spec.difficulty == "primer"

    def test_primer_category(self):
        """All primer archetypes have category='primer'."""
        primer_archetypes = list_archetypes(difficulty="primer")
        for spec in primer_archetypes:
            assert spec.category == "primer"

    def test_primer_phase_1(self):
        """All primer archetypes are phase 1."""
        primer_archetypes = list_archetypes(difficulty="primer")
        for spec in primer_archetypes:
            assert spec.phase == 1

    def test_primer_solvable(self):
        """All primer archetypes are solvable."""
        primer_archetypes = list_archetypes(difficulty="primer")
        for spec in primer_archetypes:
            assert spec.solvable is True


class TestPrimerExtractById:
    """Tests for primer.extract_by_id archetype."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        spec = get_archetype("primer.extract_by_id")
        return spec.generator_class()

    def test_minimal_html(self, generator):
        """HTML is minimal with just the target element."""
        task = generator.generate(seed=42)
        # Should be just: <span id="target">text</span>
        assert '<span id="target">' in task.html
        assert "</span>" in task.html
        # No complex nesting
        assert task.html.count("<") <= 2  # Opening and closing tag only

    def test_ground_truth_in_html(self, generator):
        """Ground truth is literally present in HTML."""
        task = generator.generate(seed=42)
        assert task.ground_truth in task.html

    def test_solvable_with_bs4(self, generator):
        """Task is solvable with BeautifulSoup."""
        task = generator.generate(seed=42)
        soup = BeautifulSoup(task.html, "html.parser")
        element = soup.find(id="target")
        assert element is not None
        assert element.text == task.ground_truth

    def test_deterministic(self, generator):
        """Same seed produces same output."""
        task1 = generator.generate(seed=123)
        task2 = generator.generate(seed=123)
        assert task1.html == task2.html
        assert task1.ground_truth == task2.ground_truth


class TestPrimerExtractByClass:
    """Tests for primer.extract_by_class archetype."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        spec = get_archetype("primer.extract_by_class")
        return spec.generator_class()

    def test_minimal_html(self, generator):
        """HTML is minimal with just the target element."""
        task = generator.generate(seed=42)
        # Should be just: <div class="target">text</div>
        assert '<div class="target">' in task.html
        assert "</div>" in task.html
        assert task.html.count("<") <= 2

    def test_solvable_with_bs4(self, generator):
        """Task is solvable with BeautifulSoup."""
        task = generator.generate(seed=42)
        soup = BeautifulSoup(task.html, "html.parser")
        element = soup.find(class_="target")
        assert element is not None
        assert element.text == task.ground_truth


class TestPrimerExtractByTag:
    """Tests for primer.extract_by_tag archetype."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        spec = get_archetype("primer.extract_by_tag")
        return spec.generator_class()

    def test_minimal_html(self, generator):
        """HTML is minimal with just an h1 element."""
        task = generator.generate(seed=42)
        # Should be just: <h1>text</h1>
        assert "<h1>" in task.html
        assert "</h1>" in task.html
        assert task.html.count("<") <= 2

    def test_solvable_with_bs4(self, generator):
        """Task is solvable with BeautifulSoup."""
        task = generator.generate(seed=42)
        soup = BeautifulSoup(task.html, "html.parser")
        element = soup.find("h1")
        assert element is not None
        assert element.text == task.ground_truth


class TestPrimerExtractAttribute:
    """Tests for primer.extract_attribute archetype."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        spec = get_archetype("primer.extract_attribute")
        return spec.generator_class()

    def test_minimal_html(self, generator):
        """HTML is minimal with just an anchor element."""
        task = generator.generate(seed=42)
        # Should be just: <a href="/url" id="link">text</a>
        assert '<a href="' in task.html
        assert 'id="link"' in task.html
        assert "</a>" in task.html
        assert task.html.count("<") <= 2

    def test_ground_truth_is_href(self, generator):
        """Ground truth is the href attribute value."""
        task = generator.generate(seed=42)
        # Ground truth should start with /
        assert task.ground_truth.startswith("/")

    def test_solvable_with_bs4(self, generator):
        """Task is solvable with BeautifulSoup."""
        task = generator.generate(seed=42)
        soup = BeautifulSoup(task.html, "html.parser")
        element = soup.find(id="link")
        assert element is not None
        assert element.get("href") == task.ground_truth


class TestPrimerCountElements:
    """Tests for primer.count_elements archetype."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        spec = get_archetype("primer.count_elements")
        return spec.generator_class()

    def test_html_structure(self, generator):
        """HTML has ul with li elements."""
        task = generator.generate(seed=42)
        assert "<ul>" in task.html
        assert "<li>" in task.html
        assert "</ul>" in task.html

    def test_ground_truth_is_count(self, generator):
        """Ground truth is the count of li elements."""
        task = generator.generate(seed=42)
        soup = BeautifulSoup(task.html, "html.parser")
        li_count = len(soup.find_all("li"))
        assert task.ground_truth == li_count
        # Count should be between 2-5
        assert 2 <= li_count <= 5

    def test_answer_schema_integer(self, generator):
        """Answer schema specifies integer type."""
        task = generator.generate(seed=42)
        assert task.answer_schema.get("type") == "integer"


class TestPrimerGrading:
    """Tests for grading primer archetypes."""

    def test_correct_answer_full_reward(self):
        """Correct answer gets full reward."""
        spec = get_archetype("primer.extract_by_id")
        generator = spec.generator_class()
        task = generator.generate(seed=42)

        # Simulate correct answer
        output = json.dumps({"status": "ok", "answer": task.ground_truth})

        # Use parse_task_info to deserialize JSON fields
        task_info = parse_task_info(task.to_info_dict())

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            run_python_calls=1,
        )

        assert reward == REWARD_CORRECT
        assert metrics["correct"] is True

    def test_wrong_answer_zero_reward(self):
        """Wrong answer gets zero reward."""
        spec = get_archetype("primer.extract_by_id")
        generator = spec.generator_class()
        task = generator.generate(seed=42)

        # Simulate wrong answer
        output = json.dumps({"status": "ok", "answer": "completely_wrong_answer"})

        # Use parse_task_info to deserialize JSON fields
        task_info = parse_task_info(task.to_info_dict())

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            run_python_calls=1,
        )

        # May get process partial credit if code is provided, but without code
        # it should be 0
        assert reward == 0.0
        assert metrics["correct"] is False

    def test_counting_task_integer_coercion(self):
        """Counting task accepts string integers."""
        spec = get_archetype("primer.count_elements")
        generator = spec.generator_class()
        task = generator.generate(seed=42)

        # Simulate answer as string (models sometimes return "3" instead of 3)
        output = json.dumps({"status": "ok", "answer": str(task.ground_truth)})

        # Use parse_task_info to deserialize JSON fields
        task_info = parse_task_info(task.to_info_dict())

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            run_python_calls=1,
        )

        assert reward == REWARD_CORRECT
        assert metrics["correct"] is True


class TestPrimerMetadata:
    """Tests for primer task metadata."""

    def test_primer_type_in_metadata(self):
        """Each primer has primer_type in metadata."""
        primer_archetypes = list_archetypes(difficulty="primer")

        expected_types = {
            "primer.extract_by_id": "id_selector",
            "primer.extract_by_class": "class_selector",
            "primer.extract_by_tag": "tag_selector",
            "primer.extract_attribute": "attribute_extraction",
            "primer.count_elements": "element_counting",
        }

        for spec in primer_archetypes:
            generator = spec.generator_class()
            task = generator.generate(seed=42)

            assert "primer_type" in task.metadata
            assert task.metadata["primer_type"] == expected_types[spec.archetype_id]


class TestPrimerVariety:
    """Tests that primers generate variety across seeds."""

    def test_different_seeds_different_content(self):
        """Different seeds produce different content."""
        spec = get_archetype("primer.extract_by_id")
        generator = spec.generator_class()

        # Generate with 10 different seeds
        ground_truths = set()
        for seed in range(10):
            task = generator.generate(seed)
            ground_truths.add(task.ground_truth)

        # Should have some variety (at least 3 different values)
        assert len(ground_truths) >= 3

    def test_all_primers_produce_valid_tasks(self):
        """All primer generators produce valid tasks across many seeds."""
        primer_archetypes = list_archetypes(difficulty="primer")

        for spec in primer_archetypes:
            generator = spec.generator_class()

            for seed in range(50):
                task = generator.generate(seed)

                # Basic validity checks
                assert task.html is not None
                assert len(task.html) > 0
                assert task.query is not None
                assert task.ground_truth is not None
                assert task.archetype_id == spec.archetype_id
                assert task.solvable is True
                assert task.difficulty == "primer"
