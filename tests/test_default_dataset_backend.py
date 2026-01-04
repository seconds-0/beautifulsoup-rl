"""Test that train/eval splits default to lazy dataset."""

from bs4_env.adapters.verifiers_adapter import MinimalEnv
from bs4_env.config import EnvConfig
from bs4_env.lazy_dataset import LazyBS4Dataset


def test_train_defaults_to_lazy_dataset():
    """Train split should use LazyBS4Dataset by default."""
    config = EnvConfig(split="train", mode="mvp")
    env = MinimalEnv(config)
    assert isinstance(env.dataset, LazyBS4Dataset)


def test_eval_defaults_to_lazy_dataset():
    """Eval split should use LazyBS4Dataset by default."""
    config = EnvConfig(split="eval", mode="mvp")
    env = MinimalEnv(config)
    assert isinstance(env.dataset, LazyBS4Dataset)


def test_bench_uses_eager_dataset():
    """Bench split should use eager HuggingFace Dataset."""
    config = EnvConfig(split="bench", mode="mvp")
    env = MinimalEnv(config)
    # Should NOT be LazyBS4Dataset
    assert not isinstance(env.dataset, LazyBS4Dataset)


def test_explicit_hf_backend_uses_eager():
    """Explicit dataset_backend='hf' should use eager loading."""
    config = EnvConfig(split="train", mode="mvp", dataset_backend="hf")
    env = MinimalEnv(config)
    # Should NOT be LazyBS4Dataset
    assert not isinstance(env.dataset, LazyBS4Dataset)


def test_lazy_dataset_has_column_names():
    """LazyBS4Dataset should have column_names for verifiers compatibility."""
    config = EnvConfig(split="train", mode="mvp")
    env = MinimalEnv(config)
    assert hasattr(env.dataset, "column_names")
    assert "prompt" in env.dataset.column_names
    assert "info" in env.dataset.column_names
    assert "example_id" in env.dataset.column_names


def test_lazy_dataset_column_access():
    """LazyBS4Dataset should support column access pattern."""
    config = EnvConfig(split="train", mode="mvp")
    env = MinimalEnv(config)

    # Test example_id column access returns sequential IDs matching dataset length
    example_ids = env.dataset["example_id"]
    assert example_ids == list(range(len(env.dataset)))


def test_lazy_dataset_item_has_example_id():
    """Items from LazyBS4Dataset should include example_id."""
    config = EnvConfig(split="train", mode="mvp")
    env = MinimalEnv(config)

    item = env.dataset[0]
    assert "example_id" in item
    assert item["example_id"] == 0
