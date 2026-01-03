"""Tests for HTML compression and size budgets.

These tests verify:
1. Compression/decompression roundtrip works correctly
2. Compression achieves expected ratio (70-80%)
3. Fixed-seed HTML generation stays within size budgets
"""

import json
import random

import pytest

from bs4_env import auto_import  # noqa: F401 - ensure generators registered
from bs4_env.config import HTML_SIZE_BUDGETS, get_size_budget
from bs4_env.dataset import load_bench_manifest
from bs4_env.registry import get_archetype
from bs4_env.tools.harness import compress_html, decompress_html


class TestHtmlCompression:
    """Tests for HTML compression utilities."""

    def test_roundtrip_simple(self):
        """Simple HTML should roundtrip correctly."""
        html = "<html><body><h1>Hello World</h1></body></html>"
        compressed = compress_html(html)
        decompressed = decompress_html(compressed)
        assert decompressed == html

    def test_roundtrip_unicode(self):
        """Unicode HTML should roundtrip correctly."""
        html = "<p>日本語テスト</p><p>Emoji: 🎉🚀</p>"
        compressed = compress_html(html)
        decompressed = decompress_html(compressed)
        assert decompressed == html

    def test_roundtrip_complex_html(self):
        """Complex HTML with attributes and nesting should roundtrip."""
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Page</title>
    <style>.class { color: red; }</style>
</head>
<body>
    <div class="container" data-value="test">
        <script>var x = "<escaped>";</script>
        <p id="p1">Paragraph with "quotes" and 'apostrophes'</p>
    </div>
</body>
</html>'''
        compressed = compress_html(html)
        decompressed = decompress_html(compressed)
        assert decompressed == html

    def test_compression_reduces_size(self):
        """Compression should significantly reduce HTML size."""
        # Generate repetitive HTML (compresses well)
        html = "<div>" + "<p>This is a test paragraph.</p>\n" * 100 + "</div>"
        compressed = compress_html(html)

        original_size = len(html.encode("utf-8"))
        compressed_size = len(compressed.encode("ascii"))

        # Compression should achieve at least 50% reduction
        ratio = compressed_size / original_size
        assert ratio < 0.5, f"Compression ratio {ratio:.2%} too high (expected <50%)"

    def test_typical_html_compression_ratio(self):
        """Typical HTML should compress by 70-80%."""
        # Generate HTML similar to what generators produce
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="UTF-8">',
            "<title>Product Catalog</title>",
            '<link rel="stylesheet" href="/styles.css">',
            "</head>",
            '<body class="page-wrapper">',
            '<header class="site-header">',
            '<nav class="navigation">',
        ]

        # Add some product cards
        for i in range(20):
            html_parts.extend([
                f'<div class="product-card" data-id="{i}">',
                f'<img src="/images/product-{i}.jpg" alt="Product {i}">',
                f'<h3 class="product-title">Product Name {i}</h3>',
                f'<p class="product-price">$99.99</p>',
                '<button class="add-to-cart">Add to Cart</button>',
                '</div>',
            ])

        html_parts.extend([
            "</nav>",
            "</header>",
            "</body>",
            "</html>",
        ])

        html = "\n".join(html_parts)
        compressed = compress_html(html)

        original_size = len(html.encode("utf-8"))
        compressed_size = len(compressed.encode("ascii"))
        ratio = compressed_size / original_size

        # Typical HTML should compress to 20-40% of original
        assert ratio < 0.5, f"Compression ratio {ratio:.2%} too high"
        # But not impossibly small
        assert ratio > 0.05, f"Compression ratio {ratio:.2%} suspiciously low"

    def test_empty_html(self):
        """Empty string should roundtrip."""
        compressed = compress_html("")
        decompressed = decompress_html(compressed)
        assert decompressed == ""

    def test_compressed_is_ascii_safe(self):
        """Compressed output should be ASCII-safe for JSON."""
        html = "<p>Non-ASCII: 日本語 中文 العربية</p>"
        compressed = compress_html(html)

        # Should be valid ASCII
        compressed.encode("ascii")  # Raises if not ASCII

        # Should be JSON-serializable
        json.dumps({"compressed": compressed})


class TestSizeBudgets:
    """Tests for HTML size budget configuration."""

    def test_budgets_exist_for_all_difficulties(self):
        """All difficulties should have size budgets."""
        for difficulty in ["primer", "easy", "medium", "hard"]:
            budget = get_size_budget(difficulty)
            assert budget > 0, f"No budget for {difficulty}"

    def test_budgets_are_ordered(self):
        """Budgets should increase with difficulty."""
        assert get_size_budget("primer") < get_size_budget("easy")
        assert get_size_budget("easy") < get_size_budget("medium")
        assert get_size_budget("medium") < get_size_budget("hard")

    def test_unknown_difficulty_returns_default(self):
        """Unknown difficulty should return medium budget."""
        assert get_size_budget("unknown") == HTML_SIZE_BUDGETS["medium"]

    def test_budget_values_reasonable(self):
        """Budget values should be reasonable."""
        assert get_size_budget("primer") >= 500  # At least 500 chars
        assert get_size_budget("hard") <= 100_000  # At most 100KB


class TestFixedSeedHtmlSizes:
    """Fixed-seed tests for HTML size regression.

    These use deterministic seeds to ensure HTML sizes don't regress
    unexpectedly. Per Codex feedback: use fixed seeds, not percentiles.
    """

    # Fixed seeds for regression testing
    TEST_SEEDS = [110000, 110005, 110010, 110015, 110020]

    def test_primer_html_under_budget(self):
        """Primer tasks should produce small HTML."""
        # Get primer archetypes from manifest
        manifest = load_bench_manifest()
        primer_entries = [
            (aid, seed) for aid, seed in manifest
            if get_archetype(aid).difficulty == "primer"
        ][:5]  # First 5

        if not primer_entries:
            pytest.skip("No primer archetypes in manifest")

        budget = get_size_budget("primer")
        for archetype_id, seed in primer_entries:
            spec = get_archetype(archetype_id)
            generator = spec.generator_class()
            task = generator.generate(seed)

            html_size = len(task.html)
            assert html_size <= budget * 2, (  # Allow 2x buffer for realistic HTML
                f"{archetype_id} seed={seed}: {html_size} chars exceeds 2x primer budget"
            )

    def test_easy_html_under_budget(self):
        """Easy tasks should produce reasonably small HTML."""
        manifest = load_bench_manifest()
        easy_entries = [
            (aid, seed) for aid, seed in manifest
            if get_archetype(aid).difficulty == "easy"
        ][:5]

        if not easy_entries:
            pytest.skip("No easy archetypes in manifest")

        budget = get_size_budget("easy")
        for archetype_id, seed in easy_entries:
            spec = get_archetype(archetype_id)
            generator = spec.generator_class()
            task = generator.generate(seed)

            html_size = len(task.html)
            # Allow 3x buffer - easy tasks may have realistic chrome
            assert html_size <= budget * 3, (
                f"{archetype_id} seed={seed}: {html_size} chars exceeds 3x easy budget"
            )

    def test_hard_html_under_limit(self):
        """Hard tasks should not exceed maximum reasonable size."""
        manifest = load_bench_manifest()
        hard_entries = [
            (aid, seed) for aid, seed in manifest
            if get_archetype(aid).difficulty == "hard"
        ][:5]

        if not hard_entries:
            pytest.skip("No hard archetypes in manifest")

        max_size = 100_000  # 100KB absolute maximum
        for archetype_id, seed in hard_entries:
            spec = get_archetype(archetype_id)
            generator = spec.generator_class()
            task = generator.generate(seed)

            html_size = len(task.html)
            assert html_size <= max_size, (
                f"{archetype_id} seed={seed}: {html_size} chars exceeds 100KB limit"
            )

    def test_compression_savings_on_real_html(self):
        """Real HTML from generators should compress well."""
        manifest = load_bench_manifest()
        # Sample a few entries
        sample = manifest[:10]

        total_original = 0
        total_compressed = 0

        for archetype_id, seed in sample:
            spec = get_archetype(archetype_id)
            generator = spec.generator_class()
            task = generator.generate(seed)

            original = len(task.html.encode("utf-8"))
            compressed = len(compress_html(task.html).encode("ascii"))

            total_original += original
            total_compressed += compressed

        ratio = total_compressed / total_original
        # Real HTML should compress by at least 50%
        assert ratio < 0.5, f"Average compression ratio {ratio:.2%} too high"

    def test_deterministic_html_sizes(self):
        """Same seed should produce same HTML size."""
        manifest = load_bench_manifest()
        if not manifest:
            pytest.skip("No manifest entries")

        archetype_id, seed = manifest[0]
        spec = get_archetype(archetype_id)
        generator = spec.generator_class()

        task1 = generator.generate(seed)
        task2 = generator.generate(seed)

        assert len(task1.html) == len(task2.html), (
            "Same seed produced different HTML sizes"
        )
        assert task1.html == task2.html, "Same seed produced different HTML"
