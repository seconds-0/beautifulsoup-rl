"""Tests for disk-cached dataset functionality."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from bs4_env.config import EnvConfig
from bs4_env.dataset import (
    _compute_cache_key,
    _get_archetype_version_hash,
    build_dataset,
    build_disk_cached_dataset,
)


class TestCacheKeyComputation:
    """Tests for cache key determinism and invalidation."""

    def test_same_config_same_key(self):
        """Same config should produce same cache key."""
        config1 = EnvConfig(split="train", mode="mvp", seed=42)
        config2 = EnvConfig(split="train", mode="mvp", seed=42)

        key1 = _compute_cache_key(config1, env_id="test-env")
        key2 = _compute_cache_key(config2, env_id="test-env")

        assert key1 == key2

    def test_different_split_different_key(self):
        """Different split should produce different cache key."""
        config1 = EnvConfig(split="train", mode="mvp")
        config2 = EnvConfig(split="eval", mode="mvp")

        key1 = _compute_cache_key(config1, env_id="test-env")
        key2 = _compute_cache_key(config2, env_id="test-env")

        assert key1 != key2

    def test_different_mode_different_key(self):
        """Different mode should produce different cache key."""
        config1 = EnvConfig(split="train", mode="mvp")
        config2 = EnvConfig(split="train", mode="all")

        key1 = _compute_cache_key(config1, env_id="test-env")
        key2 = _compute_cache_key(config2, env_id="test-env")

        assert key1 != key2

    def test_different_seed_different_key(self):
        """Different seed should produce different cache key."""
        config1 = EnvConfig(split="train", mode="mvp", seed=42)
        config2 = EnvConfig(split="train", mode="mvp", seed=123)

        key1 = _compute_cache_key(config1, env_id="test-env")
        key2 = _compute_cache_key(config2, env_id="test-env")

        assert key1 != key2

    def test_different_num_examples_different_key(self):
        """Different num_examples should produce different cache key."""
        config1 = EnvConfig(split="train", mode="mvp", num_examples=100)
        config2 = EnvConfig(split="train", mode="mvp", num_examples=200)

        key1 = _compute_cache_key(config1, env_id="test-env")
        key2 = _compute_cache_key(config2, env_id="test-env")

        assert key1 != key2

    def test_different_difficulty_weights_different_key(self):
        """Different difficulty_weights should produce different cache key."""
        config1 = EnvConfig(
            split="train",
            mode="tiered",
            difficulty_weights={"primer": 0.0, "easy": 0.2, "medium": 0.4, "hard": 0.4},
        )
        config2 = EnvConfig(
            split="train",
            mode="tiered",
            difficulty_weights={"primer": 0.1, "easy": 0.3, "medium": 0.3, "hard": 0.3},
        )

        key1 = _compute_cache_key(config1, env_id="test-env")
        key2 = _compute_cache_key(config2, env_id="test-env")

        assert key1 != key2

    def test_different_env_id_different_key(self):
        """Different env_id should produce different cache key (P0 fix)."""
        config = EnvConfig(split="train", mode="mvp", seed=42)

        key1 = _compute_cache_key(config, env_id="env-a")
        key2 = _compute_cache_key(config, env_id="env-b")

        assert key1 != key2

    def test_cache_key_is_hex_string(self):
        """Cache key should be a hex string."""
        config = EnvConfig(split="train", mode="mvp")
        key = _compute_cache_key(config, env_id="test-env")

        assert isinstance(key, str)
        assert len(key) == 16  # First 16 chars of SHA-256
        assert all(c in "0123456789abcdef" for c in key)

    def test_archetype_version_hash_is_stable(self):
        """Archetype version hash should be stable across calls."""
        hash1 = _get_archetype_version_hash()
        hash2 = _get_archetype_version_hash()

        assert hash1 == hash2


class TestDiskCachedDataset:
    """Tests for disk-cached dataset building."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_creates_dataset_on_disk(self, temp_cache_dir):
        """Dataset should be created in the cache directory."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        dataset = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)

        # Dataset should exist and have data
        assert len(dataset) > 0

        # Cache directory should have been populated (check for any files)
        cache_files = list(temp_cache_dir.rglob("*"))
        assert len(cache_files) > 0

    def test_loads_from_cache_on_second_call(self, temp_cache_dir):
        """Second call should load from cache, not regenerate."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        # First call - generates
        dataset1 = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)

        # Second call - should load from cache
        dataset2 = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=False)

        assert len(dataset1) == len(dataset2)

    def test_force_rebuild_regenerates(self, temp_cache_dir):
        """force_rebuild=True should regenerate even if cache exists."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        # First call
        dataset1 = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)
        len1 = len(dataset1)

        # Second call with force_rebuild should still work
        dataset2 = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)
        len2 = len(dataset2)

        # Both should have same length (regenerated correctly)
        assert len1 == len2
        assert len1 > 0

    def test_dataset_has_required_columns(self, temp_cache_dir):
        """Dataset should have example_id and task columns for verifiers."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        dataset = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)

        assert "example_id" in dataset.column_names
        assert "task" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "info" in dataset.column_names

    def test_example_id_is_integer(self, temp_cache_dir):
        """example_id should be an integer (required by verifiers)."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        dataset = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)

        # Check first example
        assert isinstance(dataset[0]["example_id"], int)
        assert dataset[0]["example_id"] == 0

    def test_task_is_string(self, temp_cache_dir):
        """task should be a string (required by verifiers EnvGroup)."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        dataset = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)

        assert isinstance(dataset[0]["task"], str)

    def test_custom_env_id(self, temp_cache_dir):
        """Custom env_id should be used for task column."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        dataset = build_disk_cached_dataset(
            config,
            cache_dir=temp_cache_dir,
            force_rebuild=True,
            env_id="custom-env-id",
        )

        assert dataset[0]["task"] == "custom-env-id"


class TestDatasetEquivalence:
    """Tests that disk-cached and eager datasets produce equivalent data."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_same_prompts_as_eager(self, temp_cache_dir):
        """Disk-cached dataset should have same prompts as eager."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=5)

        eager_dataset = build_dataset(config)
        cached_dataset = build_disk_cached_dataset(
            config, cache_dir=temp_cache_dir, force_rebuild=True
        )

        assert len(eager_dataset) == len(cached_dataset)

        for i in range(len(eager_dataset)):
            assert eager_dataset[i]["prompt"] == cached_dataset[i]["prompt"]

    def test_same_info_as_eager(self, temp_cache_dir):
        """Disk-cached dataset should have same info as eager."""
        config = EnvConfig(split="bench", mode="mvp", num_examples=5)

        eager_dataset = build_dataset(config)
        cached_dataset = build_disk_cached_dataset(
            config, cache_dir=temp_cache_dir, force_rebuild=True
        )

        for i in range(len(eager_dataset)):
            # Both should be JSON strings
            eager_info = json.loads(eager_dataset[i]["info"])
            cached_info = json.loads(cached_dataset[i]["info"])

            # Core fields should match
            assert eager_info["archetype_id"] == cached_info["archetype_id"]
            assert eager_info["ground_truth"] == cached_info["ground_truth"]
            assert eager_info["solvable"] == cached_info["solvable"]


class TestVerifiersCompatibility:
    """Tests for verifiers framework compatibility."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_verifiers_skips_add_column(self, temp_cache_dir):
        """Verifiers should skip add_column when example_id exists."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        dataset = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)

        # Simulate verifiers' _ensure_example_id check
        assert "example_id" in dataset.column_names
        assert isinstance(dataset["example_id"][0], int)

    def test_verifiers_skips_task_map(self, temp_cache_dir):
        """Verifiers should skip map when task column exists."""
        config = EnvConfig(split="train", mode="mvp", num_examples=5)

        dataset = build_disk_cached_dataset(config, cache_dir=temp_cache_dir, force_rebuild=True)

        # Simulate verifiers' _ensure_task check
        assert "task" in dataset.column_names
