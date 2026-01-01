"""Tests for limitation archetypes.

These tests verify that limitation archetypes:
1. Generate deterministically
2. Produce unsolvable task instances (solvable=False)
3. Have proper limit_info with allowed reasons and evidence patterns
4. Contain evidence that can be found in the HTML
"""

import re

from bs4_env.generators.mvp_limitations import (
    CanvasTextGenerator,
    ImageTextGenerator,
    JSRequiredGenerator,
    PdfEmbedGenerator,
    SvgPathDataGenerator,
)


class TestJSRequired:
    """Tests for the js_required limitation archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = JSRequiredGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html
        assert task1.query == task2.query

    def test_is_unsolvable(self):
        """Task should be marked as unsolvable."""
        gen = JSRequiredGenerator()
        task = gen.generate(seed=42)

        assert task.solvable is False
        assert task.ground_truth is None

    def test_has_limit_info(self):
        """Task should have proper limit_info."""
        gen = JSRequiredGenerator()
        task = gen.generate(seed=42)

        assert "allowed_reasons" in task.limit_info
        assert "evidence_patterns" in task.limit_info
        assert len(task.limit_info["allowed_reasons"]) > 0

    def test_evidence_in_html(self):
        """Evidence patterns should match content in HTML."""
        gen = JSRequiredGenerator()
        task = gen.generate(seed=42)

        # At least one evidence pattern should match
        matched = False
        for pattern in task.limit_info["evidence_patterns"]:
            if re.search(pattern, task.html):
                matched = True
                break
        assert matched, "No evidence pattern matched the HTML"


class TestImageText:
    """Tests for the image_text limitation archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = ImageTextGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html

    def test_is_unsolvable(self):
        """Task should be marked as unsolvable."""
        gen = ImageTextGenerator()
        task = gen.generate(seed=42)

        assert task.solvable is False

    def test_has_img_tag(self):
        """HTML should contain an image tag."""
        gen = ImageTextGenerator()
        task = gen.generate(seed=42)

        assert "<img" in task.html


class TestCanvasText:
    """Tests for the canvas_text limitation archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = CanvasTextGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html
        assert task1.query == task2.query

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        gen = CanvasTextGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=43)

        # At least some variation expected
        assert task1.metadata["actual_product"] != task2.metadata["actual_product"]

    def test_is_unsolvable(self):
        """Task should be marked as unsolvable."""
        gen = CanvasTextGenerator()
        task = gen.generate(seed=42)

        assert task.solvable is False
        assert task.ground_truth is None

    def test_has_limit_info(self):
        """Task should have proper limit_info."""
        gen = CanvasTextGenerator()
        task = gen.generate(seed=42)

        assert "allowed_reasons" in task.limit_info
        assert "canvas_text" in task.limit_info["allowed_reasons"]

    def test_has_canvas_element(self):
        """HTML should contain a canvas element."""
        gen = CanvasTextGenerator()
        task = gen.generate(seed=42)

        assert "<canvas" in task.html

    def test_has_filltext_call(self):
        """HTML should contain fillText JavaScript call."""
        gen = CanvasTextGenerator()
        task = gen.generate(seed=42)

        assert "fillText" in task.html

    def test_evidence_in_html(self):
        """Evidence patterns should match content in HTML."""
        gen = CanvasTextGenerator()
        task = gen.generate(seed=42)

        matched = False
        for pattern in task.limit_info["evidence_patterns"]:
            if re.search(pattern, task.html):
                matched = True
                break
        assert matched, "No evidence pattern matched the HTML"


class TestSvgPathData:
    """Tests for the svg_path_data limitation archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = SvgPathDataGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        gen = SvgPathDataGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=43)

        assert task1.metadata["actual_data"] != task2.metadata["actual_data"]

    def test_is_unsolvable(self):
        """Task should be marked as unsolvable."""
        gen = SvgPathDataGenerator()
        task = gen.generate(seed=42)

        assert task.solvable is False
        assert task.ground_truth is None

    def test_has_limit_info(self):
        """Task should have proper limit_info."""
        gen = SvgPathDataGenerator()
        task = gen.generate(seed=42)

        assert "allowed_reasons" in task.limit_info
        assert "svg_path_data" in task.limit_info["allowed_reasons"]

    def test_has_svg_element(self):
        """HTML should contain an SVG element."""
        gen = SvgPathDataGenerator()
        task = gen.generate(seed=42)

        assert "<svg" in task.html

    def test_has_data_in_metadata(self):
        """Metadata should contain the actual data values."""
        gen = SvgPathDataGenerator()
        task = gen.generate(seed=42)

        assert "actual_data" in task.metadata
        assert len(task.metadata["actual_data"]) == 5

    def test_evidence_in_html(self):
        """Evidence patterns should match content in HTML."""
        gen = SvgPathDataGenerator()
        task = gen.generate(seed=42)

        matched = False
        for pattern in task.limit_info["evidence_patterns"]:
            if re.search(pattern, task.html):
                matched = True
                break
        assert matched, "No evidence pattern matched the HTML"


class TestPdfEmbed:
    """Tests for the pdf_embed limitation archetype."""

    def test_determinism(self):
        """Same seed produces same output."""
        gen = PdfEmbedGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=42)

        assert task1.html == task2.html

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        gen = PdfEmbedGenerator()
        task1 = gen.generate(seed=42)
        task2 = gen.generate(seed=43)

        assert task1.metadata["document_title"] != task2.metadata["document_title"]

    def test_is_unsolvable(self):
        """Task should be marked as unsolvable."""
        gen = PdfEmbedGenerator()
        task = gen.generate(seed=42)

        assert task.solvable is False
        assert task.ground_truth is None

    def test_has_limit_info(self):
        """Task should have proper limit_info."""
        gen = PdfEmbedGenerator()
        task = gen.generate(seed=42)

        assert "allowed_reasons" in task.limit_info
        assert "pdf_embed" in task.limit_info["allowed_reasons"]

    def test_has_pdf_reference(self):
        """HTML should reference a PDF file."""
        gen = PdfEmbedGenerator()
        task = gen.generate(seed=42)

        assert ".pdf" in task.html

    def test_has_embed_element(self):
        """HTML should contain embed, object, or iframe for PDF."""
        gen = PdfEmbedGenerator()
        task = gen.generate(seed=42)

        has_embed = "<embed" in task.html or "<object" in task.html or "<iframe" in task.html
        assert has_embed, "No embed/object/iframe element found"

    def test_evidence_in_html(self):
        """Evidence patterns should match content in HTML."""
        gen = PdfEmbedGenerator()
        task = gen.generate(seed=42)

        matched = False
        for pattern in task.limit_info["evidence_patterns"]:
            if re.search(pattern, task.html):
                matched = True
                break
        assert matched, "No evidence pattern matched the HTML"


class TestLimitationRegistration:
    """Tests that all limitation archetypes are properly registered."""

    def test_all_limitations_registered(self):
        """All limitation generators should be registered in the registry."""
        # Ensure auto_import runs
        from bs4_env import auto_import  # noqa: F401
        from bs4_env.registry import list_archetypes

        limitation_archetypes = list_archetypes(category="limitations")
        archetype_ids = [spec.archetype_id for spec in limitation_archetypes]

        # Check all expected archetypes are registered
        expected = [
            "mvp.limit_js_required",
            "mvp.limit_image_text",
            "mvp.limit_canvas_text",
            "mvp.limit_svg_path_data",
            "mvp.limit_pdf_embed",
        ]
        for expected_id in expected:
            assert expected_id in archetype_ids, f"{expected_id} not registered"

    def test_all_limitations_unsolvable(self):
        """All limitation archetypes should be marked as unsolvable."""
        from bs4_env import auto_import  # noqa: F401
        from bs4_env.registry import list_archetypes

        limitation_archetypes = list_archetypes(category="limitations")

        for spec in limitation_archetypes:
            assert spec.solvable is False, f"{spec.archetype_id} is marked solvable"

    def test_all_limitations_have_allowed_reasons(self):
        """All limitation archetypes should have allowed_limit_reasons."""
        from bs4_env import auto_import  # noqa: F401
        from bs4_env.registry import list_archetypes

        limitation_archetypes = list_archetypes(category="limitations")

        for spec in limitation_archetypes:
            assert hasattr(spec, "allowed_limit_reasons"), f"{spec.archetype_id} missing reasons"
            assert len(spec.allowed_limit_reasons) > 0, f"{spec.archetype_id} has no reasons"
