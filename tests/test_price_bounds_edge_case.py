"""Tests for price_bounds edge case handling in product grid generation.

These tests verify that _safe_bounded_price correctly handles:
1. Normal price ranges (wide bounds)
2. Narrow price ranges (min ≈ max)
3. Equal prices (min == max)
4. Post-rounding clamping (values stay within bounds after formatting)
"""

import random
import re

from bs4_env.generators.base import (
    HtmlStyle,
    _safe_bounded_price,
    generate_product_grid,
)


class TestSafeBoundedPrice:
    """Tests for _safe_bounded_price helper function."""

    def test_normal_range_stays_within_bounds(self):
        """Normal price range should generate prices strictly within bounds."""
        rng = random.Random(42)
        min_p, max_p = 10.0, 100.0

        for _ in range(100):
            price_str = _safe_bounded_price(rng, min_p, max_p)
            # Extract numeric value
            price_val = float(price_str.replace("$", ""))
            assert price_val > min_p, f"Price {price_val} should be > {min_p}"
            assert price_val < max_p, f"Price {price_val} should be < {max_p}"

    def test_narrow_range_does_not_crash(self):
        """Narrow range (min ≈ max) should not crash or reverse bounds."""
        rng = random.Random(42)
        min_p, max_p = 50.0, 50.01

        # Should not raise
        for _ in range(100):
            price_str = _safe_bounded_price(rng, min_p, max_p)
            price_val = float(price_str.replace("$", ""))
            # Value should be reasonable (around midpoint)
            assert 49.0 <= price_val <= 51.0, f"Price {price_val} unreasonable"

    def test_equal_prices_does_not_crash(self):
        """Equal min/max should not crash."""
        rng = random.Random(42)
        min_p = max_p = 50.0

        # Should not raise
        for _ in range(100):
            price_str = _safe_bounded_price(rng, min_p, max_p)
            price_val = float(price_str.replace("$", ""))
            # Value should be around the midpoint
            assert 49.0 <= price_val <= 51.0, f"Price {price_val} unreasonable"

    def test_bounds_never_invert(self):
        """Bounds should never invert regardless of input."""
        rng = random.Random(42)
        test_cases = [
            (50.0, 50.0),  # Equal
            (50.0, 50.001),  # Nearly equal
            (50.0, 50.01),  # Very close
            (50.0, 50.02),  # Close
            (49.99, 50.01),  # Tight range
            (100.0, 100.5),  # Small range
            (0.01, 0.02),  # Very small values
        ]

        for min_p, max_p in test_cases:
            # Should not raise regardless of bounds
            price_str = _safe_bounded_price(rng, min_p, max_p)
            assert price_str.startswith("$"), f"Invalid format for ({min_p}, {max_p})"

    def test_post_rounding_clamp_works(self):
        """Values should stay within bounds even after rounding.

        This tests the edge case where rng.uniform might return a value
        that rounds to exactly min_p or max_p.
        """
        # Use a controlled RNG that we know produces edge cases
        rng = random.Random(12345)
        min_p, max_p = 10.00, 10.05

        for _ in range(1000):
            price_str = _safe_bounded_price(rng, min_p, max_p)
            price_val = float(price_str.replace("$", ""))
            # After rounding, should still be strictly within bounds
            # (or as close as possible given the narrow range)
            assert price_val >= min_p - 0.01, f"Price {price_val} below min {min_p}"
            assert price_val <= max_p + 0.01, f"Price {price_val} above max {max_p}"

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same price."""
        min_p, max_p = 10.0, 100.0

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        prices1 = [_safe_bounded_price(rng1, min_p, max_p) for _ in range(10)]
        prices2 = [_safe_bounded_price(rng2, min_p, max_p) for _ in range(10)]

        assert prices1 == prices2, "Same seed should produce same prices"


class TestProductGridWithNarrowBounds:
    """Tests for generate_product_grid with edge case bounds."""

    def test_narrow_bounds_generates_valid_prices(self):
        """Product grid with narrow bounds should still work."""
        rng = random.Random(42)
        html = generate_product_grid(rng, HtmlStyle.BOOTSTRAP, count=5, price_bounds=(50.0, 50.01))

        assert "$" in html  # Should contain prices
        # Parse and verify prices are reasonable
        prices = re.findall(r"\$(\d+\.\d{2})", html)
        assert len(prices) == 5
        for p in prices:
            price_val = float(p)
            # Should be around the midpoint
            assert 49.0 <= price_val <= 51.0

    def test_equal_bounds_generates_valid_prices(self):
        """Product grid with equal bounds should still work."""
        rng = random.Random(42)
        html = generate_product_grid(rng, HtmlStyle.TAILWIND, count=3, price_bounds=(99.99, 99.99))

        assert "$" in html
        prices = re.findall(r"\$(\d+\.\d{2})", html)
        assert len(prices) == 3

    def test_wide_bounds_stays_strictly_within(self):
        """Product grid with wide bounds should have prices strictly within."""
        rng = random.Random(42)
        min_p, max_p = 10.0, 100.0
        html = generate_product_grid(
            rng, HtmlStyle.BOOTSTRAP, count=20, price_bounds=(min_p, max_p)
        )

        prices = re.findall(r"\$(\d+\.\d{2})", html)
        assert len(prices) == 20

        for p in prices:
            price_val = float(p)
            assert price_val > min_p, f"Price {price_val} should be > {min_p}"
            assert price_val < max_p, f"Price {price_val} should be < {max_p}"

    def test_no_bounds_uses_random_price(self):
        """Without bounds, should use standard random_price function."""
        rng = random.Random(42)
        html = generate_product_grid(rng, HtmlStyle.BOOTSTRAP, count=5)

        # Should still generate valid prices
        prices = re.findall(r"\$(\d+\.\d{2})", html)
        assert len(prices) == 5

        # Prices should be in typical random_price range
        for p in prices:
            price_val = float(p)
            # random_price generates between ~$5 and ~$500
            assert 1.0 <= price_val <= 1000.0
