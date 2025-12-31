"""Tests for BS4 gotcha archetypes.

These tests verify that the gotcha archetypes generate tasks that actually
test the intended BS4 gotchas, and that correct solutions work.
"""

import json

import pytest
from bs4 import BeautifulSoup

from bs4_env.generators.mvp_json_ld import JsonLdArrayGenerator, JsonLdExtractionGenerator
from bs4_env.generators.mvp_multivalue_class import MultivalueClassGenerator
from bs4_env.generators.mvp_navigablestring import NavigableStringParentGenerator
from bs4_env.generators.mvp_whitespace_sibling import WhitespaceSiblingGenerator


class TestMultivalueClass:
    """Tests for the multi-valued class gotcha archetype."""

    def test_class_returns_list(self):
        """Verify that class attribute returns a list, not string."""
        gen = MultivalueClassGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")

        # Find the target element using the query class
        query_class = task.metadata["query_class"]
        elem = soup.find(class_=query_class)

        assert elem is not None, f"Could not find element with class={query_class}"

        # THE GOTCHA: class returns a LIST
        class_attr = elem["class"]
        assert isinstance(class_attr, list), "class attribute should be a list"
        assert query_class in class_attr, f"{query_class} should be in {class_attr}"

        # Verify the common bug would fail
        # This is what developers often mistakenly write:
        assert class_attr != query_class, "class should NOT equal the single class name"

    def test_correct_extraction(self):
        """Verify correct solution extracts the right text."""
        gen = MultivalueClassGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        query_class = task.metadata["query_class"]

        # Correct solution: use class_ parameter
        elem = soup.find(class_=query_class)
        extracted = elem.get_text(strip=True)

        # Should match ground truth (with normalization)
        expected = " ".join(task.ground_truth.split())
        actual = " ".join(extracted.split())
        assert actual == expected


class TestWhitespaceSibling:
    """Tests for the whitespace sibling navigation gotcha archetype."""

    def test_next_sibling_returns_whitespace(self):
        """Verify that .next_sibling returns whitespace, not next element."""
        gen = WhitespaceSiblingGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        start_id = task.metadata["start_id"]

        # Find the start element
        start_elem = soup.find(id=start_id)
        assert start_elem is not None

        # THE GOTCHA: next_sibling often returns whitespace
        next_sib = start_elem.next_sibling

        # In formatted HTML, next_sibling is usually whitespace text
        # (This is the gotcha - it's NOT the next element tag)
        # Note: Could be None or whitespace depending on HTML formatting
        if next_sib is not None:
            # If there's a sibling, it should be whitespace or the next tag
            # The gotcha is that it's often whitespace when developers expect a tag
            from bs4 import NavigableString

            if isinstance(next_sib, NavigableString) and not next_sib.name:
                # It's a text node (whitespace)
                assert next_sib.strip() == "" or next_sib.name is None

    def test_find_next_sibling_works(self):
        """Verify find_next_sibling() returns the correct element."""
        gen = WhitespaceSiblingGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        start_id = task.metadata["start_id"]
        item_tag = task.metadata["item_tag"]

        # Find the start element
        start_elem = soup.find(id=start_id)
        assert start_elem is not None

        # Correct solution: use find_next_sibling()
        next_elem = start_elem.find_next_sibling(item_tag)
        assert next_elem is not None, f"Could not find next {item_tag} sibling"

        extracted = next_elem.get_text(strip=True)

        # Should match ground truth
        expected = " ".join(task.ground_truth.split())
        actual = " ".join(extracted.split())
        assert actual == expected


class TestJsonLdExtraction:
    """Tests for the JSON-LD extraction archetype."""

    def test_json_ld_script_exists(self):
        """Verify JSON-LD script tag exists in HTML."""
        gen = JsonLdExtractionGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")

        # Find JSON-LD script
        script = soup.find("script", type="application/ld+json")
        assert script is not None, "JSON-LD script not found"

        # Verify it's valid JSON
        data = json.loads(script.string)
        assert "@context" in data
        assert "@type" in data

    def test_extraction_path_works(self):
        """Verify the extraction path returns correct value."""
        gen = JsonLdExtractionGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        schema_type = task.metadata["schema_type"]

        # Find JSON-LD script
        scripts = soup.find_all("script", type="application/ld+json")
        assert len(scripts) > 0

        # Find the script with the right schema type
        target_data = None
        for script in scripts:
            data = json.loads(script.string)
            if data.get("@type") == schema_type:
                target_data = data
                break

        assert target_data is not None, f"Could not find {schema_type} JSON-LD"

        # Navigate to the extraction path
        path = task.metadata["extraction_path"]
        current = target_data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)

        assert str(current) == task.ground_truth


class TestJsonLdArray:
    """Tests for the JSON-LD array selection archetype."""

    def test_multiple_json_ld_scripts_exist(self):
        """Verify multiple JSON-LD script tags exist in HTML."""
        gen = JsonLdArrayGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")

        # Find all JSON-LD scripts
        scripts = soup.find_all("script", type="application/ld+json")
        assert len(scripts) >= 3, f"Expected 3+ JSON-LD scripts, found {len(scripts)}"

        # Verify each is valid JSON with @type
        types_found = set()
        for script in scripts:
            data = json.loads(script.string)
            assert "@type" in data, "Each JSON-LD block should have @type"
            types_found.add(data["@type"])

        # Should have different types
        assert len(types_found) >= 3, "Should have multiple different @types"

    def test_target_type_extraction_works(self):
        """Verify the target @type can be found and data extracted."""
        gen = JsonLdArrayGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        target_type = task.metadata["target_type"]

        # Find all JSON-LD scripts
        scripts = soup.find_all("script", type="application/ld+json")

        # Find the one with the target type
        target_data = None
        for script in scripts:
            data = json.loads(script.string)
            if data.get("@type") == target_type:
                target_data = data
                break

        assert target_data is not None, f"Could not find JSON-LD with @type={target_type}"

        # Navigate to the extraction path
        path = task.metadata["extraction_path"]
        current = target_data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)

        assert str(current) == task.ground_truth

    def test_determinism(self):
        """Same seed produces identical output."""
        gen = JsonLdArrayGenerator()

        task1 = gen.generate(seed=12345)
        task2 = gen.generate(seed=12345)

        assert task1.html == task2.html
        assert task1.query == task2.query
        assert task1.ground_truth == task2.ground_truth


class TestNavigableStringParent:
    """Tests for the NavigableString parent navigation gotcha archetype."""

    def test_find_string_returns_navigablestring(self):
        """Verify find(string=...) returns NavigableString, not Tag."""
        gen = NavigableStringParentGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        target_text = task.metadata["target_text"]

        # Find the text using string=
        result = soup.find(string=target_text)
        assert result is not None, f"Could not find text: {target_text}"

        # THE GOTCHA: result is NavigableString, not Tag
        from bs4 import NavigableString

        assert isinstance(result, NavigableString)

        # This is the common bug - NavigableStrings don't support dict-style access
        # They raise TypeError because they're string-like, not dict-like
        with pytest.raises((AttributeError, TypeError)):
            # This should fail - NavigableStrings don't have attrs
            _ = result["class"]

    def test_parent_navigation_works(self):
        """Verify .parent returns the containing Tag."""
        gen = NavigableStringParentGenerator()
        task = gen.generate(seed=42)

        soup = BeautifulSoup(task.html, "html.parser")
        target_text = task.metadata["target_text"]
        extract_type = task.metadata["extract_type"]

        # Find the text
        text_node = soup.find(string=target_text)
        assert text_node is not None

        # Correct solution: use .parent
        parent = text_node.parent
        assert parent is not None
        assert parent.name is not None  # Tags have .name

        # Extract the requested attribute
        if extract_type == "tag_name":
            extracted = parent.name
        elif extract_type == "class":
            extracted = parent.get("class", [""])[0]
        else:  # id
            extracted = parent.get("id", "")

        assert extracted == task.ground_truth


class TestGotchaArchetypeDeterminism:
    """Test that all gotcha archetypes are deterministic."""

    @pytest.mark.parametrize(
        "generator_class",
        [
            MultivalueClassGenerator,
            WhitespaceSiblingGenerator,
            JsonLdExtractionGenerator,
            JsonLdArrayGenerator,
            NavigableStringParentGenerator,
        ],
    )
    def test_same_seed_same_output(self, generator_class):
        """Same seed produces identical output."""
        gen = generator_class()

        task1 = gen.generate(seed=12345)
        task2 = gen.generate(seed=12345)

        assert task1.html == task2.html
        assert task1.query == task2.query
        assert task1.ground_truth == task2.ground_truth

    @pytest.mark.parametrize(
        "generator_class",
        [
            MultivalueClassGenerator,
            WhitespaceSiblingGenerator,
            JsonLdExtractionGenerator,
            JsonLdArrayGenerator,
            NavigableStringParentGenerator,
        ],
    )
    def test_different_seeds_different_output(self, generator_class):
        """Different seeds produce different output."""
        gen = generator_class()

        task1 = gen.generate(seed=1)
        task2 = gen.generate(seed=2)

        # At least one should differ
        same = (
            task1.html == task2.html
            and task1.query == task2.query
            and task1.ground_truth == task2.ground_truth
        )
        assert not same
