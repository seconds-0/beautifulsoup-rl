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


class TestNavigateVerifiersIntegration:
    """Tests for navigate tool in verifiers environment."""

    def test_navigate_tool_function(self):
        """Navigate tool should handle success and error cases."""
        import json

        # We can't import from the closure, so we replicate the logic
        def _normalize_href(href: str, pages: dict) -> str:
            href = href.strip()
            if href in pages:
                return href
            if href.startswith("/"):
                without_slash = href[1:]
                if without_slash in pages:
                    return without_slash
            else:
                with_slash = "/" + href
                if with_slash in pages:
                    return with_slash
            return href

        def navigate(href: str, pages_json: str = "{}") -> str:
            pages = json.loads(pages_json) if pages_json else {}
            if not href:
                return "Error: No href provided"
            normalized = _normalize_href(href, pages)
            if normalized not in pages:
                return f"Error: Page '{href}' not found. Check the HTML for valid links."
            return (
                f"Successfully navigated to '{normalized}'. "
                f"The HTML global has been updated with the new page content. "
                f"Use run_python to extract data from the new page."
            )

        pages = {
            "/products/1": "<html>Product 1</html>",
            "/products/2": "<html>Product 2</html>",
        }
        pages_json = json.dumps(pages)

        # Test successful navigation
        result = navigate("/products/1", pages_json)
        assert "Successfully navigated" in result
        assert "/products/1" in result

        # Test error case
        result = navigate("/nonexistent", pages_json)
        assert "Error" in result
        assert "not found" in result

        # Test empty href
        result = navigate("", pages_json)
        assert "Error" in result

    def test_navigate_href_normalization(self):
        """Navigate should normalize hrefs correctly."""
        import json

        def _normalize_href(href: str, pages: dict) -> str:
            href = href.strip()
            if href in pages:
                return href
            if href.startswith("/"):
                without_slash = href[1:]
                if without_slash in pages:
                    return without_slash
            else:
                with_slash = "/" + href
                if with_slash in pages:
                    return with_slash
            if "?" in href:
                base = href.split("?")[0]
                if base in pages:
                    return base
            if "#" in href:
                base = href.split("#")[0]
                if base in pages:
                    return base
            return href

        pages = {"products/1": "<html>Product 1</html>"}

        # With leading slash should find without
        assert _normalize_href("/products/1", pages) == "products/1"

        # Exact match
        assert _normalize_href("products/1", pages) == "products/1"

        # Query string stripped
        pages2 = {"/page": "<html>Page</html>"}
        assert _normalize_href("/page?foo=bar", pages2) == "/page"

        # Fragment stripped
        assert _normalize_href("/page#section", pages2) == "/page"

    def test_minimal_env_navigate_tool_registry(self):
        """MinimalEnv should include navigate tool for multi-step tasks."""
        from bs4_env.adapters.verifiers_adapter import MinimalEnv
        from bs4_env.config import EnvConfig

        config = EnvConfig(split="train", mode="mvp")
        env = MinimalEnv(config)

        # Create a fake multi-step example
        example = {
            "prompt": "Test prompt",
            "html": "<html>Page 1</html>",
            "query": "Find something",
            "info": {
                "answer_schema": {"type": "string"},
                "limit_info": {},
                "pages": {
                    "/page2": "<html>Page 2</html>",
                },
            },
        }

        registry = env.create_tool_registry(example)

        # Should have navigate tool
        assert registry.has_tool("navigate")
        assert registry.has_tool("run_python")

        # Navigate should work
        result = registry.call("navigate", {"href": "/page2"})
        assert "Successfully navigated" in result

    def test_minimal_env_no_navigate_for_single_step(self):
        """MinimalEnv should not include navigate for single-step tasks."""
        from bs4_env.adapters.verifiers_adapter import MinimalEnv
        from bs4_env.config import EnvConfig

        config = EnvConfig(split="train", mode="mvp")
        env = MinimalEnv(config)

        # Create a single-step example (no pages)
        example = {
            "prompt": "Test prompt",
            "html": "<html>Page 1</html>",
            "query": "Find something",
            "info": {
                "answer_schema": {"type": "string"},
                "limit_info": {},
            },
        }

        registry = env.create_tool_registry(example)

        # Should have run_python but not navigate
        assert registry.has_tool("run_python")
        assert not registry.has_tool("navigate")
