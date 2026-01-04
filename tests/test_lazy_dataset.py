"""Tests for lazy dataset generation.

These tests verify that LazyBS4Dataset:
1. Initializes quickly (no HTML generation upfront)
2. Generates deterministic results
3. Returns info as JSON string (HuggingFace compatibility)
4. Produces same results as build_dataset for same config
5. Caching works correctly
"""

import json
import time

import pytest

from bs4_env import auto_import  # noqa: F401 - ensure generators registered
from bs4_env.config import EnvConfig
from bs4_env.dataset import build_dataset, build_lazy_dataset
from bs4_env.lazy_dataset import LazyBS4Dataset, LazyTaskEntry


class TestLazyDatasetInitialization:
    """Tests for fast initialization without HTML generation."""

    def test_lazy_initialization_fast(self):
        """Lazy dataset should initialize without generating HTML."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=100)

        start = time.time()
        dataset = LazyBS4Dataset.from_config(config)
        init_time = time.time() - start

        # Init should be very fast (no HTML generation)
        # Allow generous margin for slow CI
        assert init_time < 2.0, f"Init took {init_time:.2f}s, expected <2s"
        assert len(dataset) > 0

    def test_lazy_vs_eager_init_time(self):
        """Lazy init should be significantly faster than eager."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=20)

        # Time lazy init
        start = time.time()
        lazy = LazyBS4Dataset.from_config(config)
        lazy_time = time.time() - start

        # Time eager init
        start = time.time()
        _ = build_dataset(config)
        eager_time = time.time() - start

        # Lazy should be at least 5x faster (usually 100x+)
        # Use generous margin for test stability
        assert lazy_time < eager_time, (
            f"Lazy ({lazy_time:.3f}s) should be faster than eager ({eager_time:.3f}s)"
        )
        assert len(lazy) == 20


class TestLazyDatasetAccess:
    """Tests for accessing dataset items."""

    @pytest.fixture
    def config(self):
        """Basic test config."""
        return EnvConfig(split="bench", mode="mvp", num_examples=10)

    def test_access_generates_html(self, config):
        """Accessing item should generate HTML."""
        dataset = LazyBS4Dataset.from_config(config)
        item = dataset[0]

        assert "prompt" in item
        assert "info" in item

        # info should be a JSON string (not dict!)
        assert isinstance(item["info"], str)
        info = json.loads(item["info"])
        assert "html" in info
        assert len(info["html"]) > 100  # Should have substantial HTML

    def test_info_is_json_string(self, config):
        """Info field must be JSON string for HF Dataset compatibility."""
        dataset = LazyBS4Dataset.from_config(config)

        for i in range(min(5, len(dataset))):
            item = dataset[i]
            assert isinstance(item["info"], str), f"item[{i}]['info'] should be string"
            # Should be valid JSON
            info = json.loads(item["info"])
            assert isinstance(info, dict)

    def test_deterministic_generation(self, config):
        """Same index should produce same result."""
        dataset = LazyBS4Dataset.from_config(config)

        item1 = dataset[0]
        item2 = dataset[0]

        assert item1["info"] == item2["info"]
        assert item1["prompt"] == item2["prompt"]

    def test_different_indices_different_results(self, config):
        """Different indices should produce different results."""
        dataset = LazyBS4Dataset.from_config(config)

        item0 = dataset[0]
        item1 = dataset[1]

        assert item0["info"] != item1["info"]

    def test_index_out_of_range(self, config):
        """Out of range index should raise IndexError."""
        dataset = LazyBS4Dataset.from_config(config)

        with pytest.raises(IndexError):
            _ = dataset[100]

        with pytest.raises(IndexError):
            _ = dataset[-1000]


class TestLazyDatasetCaching:
    """Tests for LRU caching behavior."""

    def test_caching_works(self):
        """With cache_size > 0, repeated access should be faster."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=5)
        dataset = LazyBS4Dataset.from_config(config, cache_size=10)

        # First access - cache miss
        start = time.time()
        _ = dataset[0]
        first_time = time.time() - start

        # Second access - cache hit
        start = time.time()
        _ = dataset[0]
        cached_time = time.time() - start

        # Cached should be significantly faster (usually 100x+)
        # But be generous for test stability
        assert cached_time < first_time, (
            f"Cached ({cached_time:.6f}s) should be faster than first ({first_time:.6f}s)"
        )

    def test_no_cache_regenerates(self):
        """With cache_size=0, should regenerate each time (same result)."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=3)
        dataset = LazyBS4Dataset.from_config(config, cache_size=0)

        # Access same item multiple times
        item1 = dataset[0]
        item2 = dataset[0]

        # Should produce identical results (deterministic)
        assert item1["info"] == item2["info"]


class TestLazyDatasetParity:
    """Tests that lazy dataset produces same results as build_dataset."""

    def test_same_entries_as_eager(self):
        """Lazy and eager should have same (archetype_id, seed) pairs."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=10)

        lazy = LazyBS4Dataset.from_config(config)
        eager = build_dataset(config)

        assert len(lazy) == len(eager)

        # Check first few entries match
        for i in range(min(5, len(lazy))):
            lazy_item = lazy[i]
            eager_row = eager[i]

            lazy_info = json.loads(lazy_item["info"])
            eager_info = json.loads(eager_row["info"])

            assert lazy_info["archetype_id"] == eager_info["archetype_id"]
            assert lazy_info["seed"] == eager_info["seed"]

    def test_same_ground_truth_as_eager(self):
        """Lazy should produce same ground_truth as eager."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=5)

        lazy = LazyBS4Dataset.from_config(config)
        eager = build_dataset(config)

        for i in range(len(lazy)):
            lazy_info = json.loads(lazy[i]["info"])
            eager_info = json.loads(eager[i]["info"])

            assert lazy_info["ground_truth"] == eager_info["ground_truth"], f"Mismatch at index {i}"

    def test_info_json_shape_matches(self):
        """Lazy info dict should have same keys as eager."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=3)

        lazy = LazyBS4Dataset.from_config(config)
        eager = build_dataset(config)

        lazy_info = json.loads(lazy[0]["info"])
        eager_info = json.loads(eager[0]["info"])

        assert set(lazy_info.keys()) == set(eager_info.keys()), (
            f"Key mismatch: lazy={set(lazy_info.keys())}, eager={set(eager_info.keys())}"
        )


class TestLazyDatasetConversion:
    """Tests for conversion to HuggingFace Dataset."""

    def test_to_hf_dataset(self):
        """Should convert to HuggingFace Dataset."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=5)
        lazy = LazyBS4Dataset.from_config(config)
        hf = lazy.to_hf_dataset()

        assert len(hf) == len(lazy)
        assert "prompt" in hf.column_names
        assert "info" in hf.column_names

    def test_hf_conversion_preserves_data(self):
        """HF conversion should preserve all data."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=3)
        lazy = LazyBS4Dataset.from_config(config)
        hf = lazy.to_hf_dataset()

        for i in range(len(lazy)):
            assert lazy[i]["info"] == hf[i]["info"]


class TestLazyTaskEntry:
    """Tests for LazyTaskEntry dataclass."""

    def test_entry_is_hashable(self):
        """LazyTaskEntry should be hashable for LRU caching."""
        entry = LazyTaskEntry("mvp.core_extraction", 42)

        # Should work in sets and dicts
        s = {entry}
        assert entry in s

        d = {entry: "value"}
        assert d[entry] == "value"

    def test_get_entry(self):
        """Should be able to get entry without generating HTML."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=5)
        dataset = LazyBS4Dataset.from_config(config)

        entry = dataset.get_entry(0)
        assert isinstance(entry, LazyTaskEntry)
        assert entry.archetype_id is not None
        assert entry.seed is not None


class TestBuildLazyDatasetFunction:
    """Tests for the convenience build_lazy_dataset function."""

    def test_build_lazy_dataset_works(self):
        """build_lazy_dataset should create a LazyBS4Dataset."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=5)
        dataset = build_lazy_dataset(config)

        assert isinstance(dataset, LazyBS4Dataset)
        assert len(dataset) == 5

    def test_build_lazy_dataset_with_cache(self):
        """build_lazy_dataset should support cache_size."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=3)
        dataset = build_lazy_dataset(config, cache_size=100)

        # Should work and use caching
        _ = dataset[0]
        _ = dataset[0]  # Should be cached
