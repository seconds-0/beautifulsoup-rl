"""Tests for hard archetypes.

These tests verify that the new hard archetypes:
1. Generate deterministically
2. Produce valid task instances
3. Have extractable ground truth
"""

import json

import pytest
from bs4 import BeautifulSoup

from bs4_env.generators.mvp_hard import ParserRequiredGenerator
from bs4_env.generators.mvp_json_ld import JsonLdArrayGenerator
from bs4_env.generators.mvp_tables import TableRowspanGenerator


class TestTableRowspan:
    """Tests for the table_rowspan archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = TableRowspanGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html
        assert task1.query == task2.query
        assert task1.ground_truth == task2.ground_truth

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        gen = TableRowspanGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=43)

        assert task1.ground_truth != task2.ground_truth

    def test_has_rowspan(self):
        """Table should have rowspan attribute."""
        gen = TableRowspanGenerator()
        task = gen.generate(seed=42)

        assert "rowspan" in task.html
        assert task.metadata["has_rowspan"] is True

    def test_ground_truth_in_html(self):
        """Ground truth should be extractable from HTML."""
        gen = TableRowspanGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        # Ground truth is a price like "$XX.XX"
        assert task.ground_truth in task.html

    def test_correct_extraction(self):
        """Verify correct solution can extract the target."""
        gen = TableRowspanGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")

        # Find the target item's row and extract price
        target_name = task.metadata["target_item"]

        # Find the cell containing the target name
        for td in soup.find_all("td"):
            if target_name in td.get_text():
                # Price is in the next sibling td
                price_td = td.find_next_sibling("td")
                if price_td and "$" in price_td.get_text():
                    extracted = price_td.get_text(strip=True)
                    assert extracted == task.ground_truth
                    return

        # If we get here, extraction failed
        pytest.fail(f"Could not extract {task.ground_truth} from table")


class TestJsonLdArray:
    """Tests for the json_ld_array archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = JsonLdArrayGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html
        assert task1.query == task2.query
        assert task1.ground_truth == task2.ground_truth

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        gen = JsonLdArrayGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=43)

        assert task1.html != task2.html

    def test_multiple_json_ld_blocks(self):
        """Should have multiple JSON-LD script blocks."""
        gen = JsonLdArrayGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")

        assert len(scripts) >= 3, "Should have at least 3 JSON-LD blocks"
        assert len(scripts) == task.metadata["num_json_ld_blocks"]

    def test_target_type_in_blocks(self):
        """Target @type should exist in one of the blocks."""
        gen = JsonLdArrayGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")

        target_type = task.metadata["target_type"]
        found = False

        for script in scripts:
            try:
                data = json.loads(script.string)
                if data.get("@type") == target_type:
                    found = True
                    break
            except json.JSONDecodeError:
                continue

        assert found, f"Target type {target_type} not found in JSON-LD blocks"

    def test_correct_extraction(self):
        """Verify correct solution can extract from the right block."""
        gen = JsonLdArrayGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")

        target_type = task.metadata["target_type"]

        for script in scripts:
            try:
                data = json.loads(script.string)
                if data.get("@type") == target_type:
                    # Found the right block
                    # Ground truth should be somewhere in this data
                    assert task.ground_truth in str(data)
                    return
            except json.JSONDecodeError:
                continue

        pytest.fail(f"Could not find ground truth in {target_type} block")


class TestParserRequired:
    """Tests for the parser_required archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = ParserRequiredGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html
        assert task1.query == task2.query
        assert task1.ground_truth == task2.ground_truth

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        gen = ParserRequiredGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=43)

        assert task1.ground_truth != task2.ground_truth

    def test_has_malformation_type(self):
        """Should have a malformation type in metadata."""
        gen = ParserRequiredGenerator()
        task = gen.generate(seed=42)

        assert "malform_type" in task.metadata
        assert task.metadata["malform_type"] in [
            "unclosed_tag",
            "nested_misorder",
            "optional_end_tag",
        ]

    def test_ground_truth_in_html(self):
        """Ground truth should be in the HTML."""
        gen = ParserRequiredGenerator()
        task = gen.generate(seed=42)

        assert task.ground_truth in task.html

    def test_parses_with_html_parser(self):
        """HTML should parse (even if malformed) with html.parser."""
        gen = ParserRequiredGenerator()
        task = gen.generate(seed=42)

        # Should not raise an exception
        soup = BeautifulSoup(task.html, "html.parser")
        assert soup is not None

    def test_parses_with_lxml(self):
        """HTML should parse with lxml."""
        gen = ParserRequiredGenerator()
        task = gen.generate(seed=42)

        # Should not raise an exception
        soup = BeautifulSoup(task.html, "lxml")
        assert soup is not None

    def test_correct_extraction_possible(self):
        """Ground truth should be extractable with some parser."""
        gen = ParserRequiredGenerator()
        task = gen.generate(seed=42)

        # Try with lxml (most lenient for our malformations)
        soup = BeautifulSoup(task.html, "lxml")

        # Ground truth is a price
        found = task.ground_truth in soup.get_text()
        assert found, f"Could not find {task.ground_truth} in parsed HTML"
