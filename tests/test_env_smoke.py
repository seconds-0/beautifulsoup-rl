"""Smoke tests for environment construction and basic operations."""

import pytest


class TestEnvironmentConstruction:
    """Tests for environment creation."""

    def test_load_environment_default(self):
        """Environment should load with defaults."""
        from beautiful_soup_env import load_environment

        env = load_environment()
        assert env is not None

    def test_load_environment_with_config(self):
        """Environment should load with custom config."""
        from beautiful_soup_env import load_environment

        env = load_environment(
            split="train",
            mode="mvp",
            difficulty="easy",
            executor_backend="local",
        )
        assert env is not None

    def test_minimal_env_has_dataset(self):
        """MinimalEnv should have a dataset property."""
        from bs4_env.adapters.verifiers_adapter import MinimalEnv
        from bs4_env.config import EnvConfig

        config = EnvConfig(split="train", mode="mvp")
        env = MinimalEnv(config)

        # Dataset is lazy-loaded, so access it
        # This will be empty until we have archetypes
        dataset = env.dataset
        assert dataset is not None


class TestConfigValidation:
    """Tests for config validation."""

    def test_invalid_timeout_rejected(self):
        """Negative timeout should raise error."""
        from bs4_env.config import EnvConfig

        with pytest.raises(ValueError):
            EnvConfig(timeout_s=-1)

    def test_invalid_max_output_rejected(self):
        """Negative max_output_chars should raise error."""
        from bs4_env.config import EnvConfig

        with pytest.raises(ValueError):
            EnvConfig(max_output_chars=-1)


class TestRegistryIntegration:
    """Tests for registry functionality."""

    def test_registry_functions_exist(self):
        """Registry functions should be importable."""
        from bs4_env.registry import (
            register,
            get_archetype,
            list_archetypes,
            get_all_archetype_ids,
        )

        assert callable(register)
        assert callable(get_archetype)
        assert callable(list_archetypes)
        assert callable(get_all_archetype_ids)

    def test_list_archetypes_returns_list(self):
        """list_archetypes should return a list."""
        from bs4_env.registry import list_archetypes

        result = list_archetypes()
        assert isinstance(result, list)


class TestGradingIntegration:
    """Tests for grading integration."""

    def test_grading_imports(self):
        """Grading module should be importable."""
        from bs4_env.grading import (
            compute_reward,
            validate_output,
            check_safety,
            normalize_string,
        )

        assert callable(compute_reward)
        assert callable(validate_output)
        assert callable(check_safety)
        assert callable(normalize_string)


class TestToolsIntegration:
    """Tests for tools integration."""

    def test_executor_imports(self):
        """Executor should be importable."""
        from bs4_env.tools import (
            Executor,
            LocalSubprocessExecutor,
            get_executor,
        )

        assert LocalSubprocessExecutor is not None
        executor = get_executor("local")
        assert isinstance(executor, Executor)
