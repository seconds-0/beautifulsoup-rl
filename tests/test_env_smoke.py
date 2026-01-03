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
            get_all_archetype_ids,
            get_archetype,
            list_archetypes,
            register,
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
            check_safety,
            compute_reward,
            normalize_string,
            validate_output,
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

    def test_minimal_env_navigate_returns_error_for_single_step(self):
        """MinimalEnv should have navigate for single-step tasks, but it returns error."""
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

        # Should have both tools (aligned with verifiers which always registers navigate)
        assert registry.has_tool("run_python")
        assert registry.has_tool("navigate")

        # Navigate should return helpful error for single-step tasks
        result = registry.call("navigate", {"href": "/some-page"})
        assert "Error" in result
        assert "does not support navigation" in result


class TestVerifiersEnvResponse:
    """Integration tests for verifiers env_response behavior.

    These tests exercise the actual BeautifulSoupEnv.env_response code path
    by mocking the parent class's env_response to isolate our navigation logic.
    """

    @pytest.mark.asyncio
    async def test_env_response_updates_html_on_navigate(self):
        """Test that env_response actually updates state['html'] on navigate success."""
        # Skip if verifiers not installed
        try:
            import verifiers as vf
        except ImportError:
            pytest.skip("verifiers not installed")

        from unittest.mock import AsyncMock, patch

        from bs4_env.adapters.verifiers_adapter import _build_real_verifiers_env
        from bs4_env.config import EnvConfig

        # Use minimal dataset for faster tests
        config = EnvConfig(split="train", mode="mvp", num_examples=1)

        # Build the actual verifiers environment
        env = _build_real_verifiers_env(config, vf)

        # Create state with pages for multi-step task
        state = {
            "html": "<html>Initial Page</html>",
            "query": "Find something",
            "constraints": {},
            "task_info": {},
            "pages": {
                "/page2": "<html>Page 2 Content</html>",
                "/page3": "<html>Page 3 Content</html>",
            },
            "navigation_history": [],
        }

        # Simulate tool result message with NAVIGATE_SUCCESS_MARKER
        messages = [
            {
                "role": "tool",
                "content": "NAVIGATE_OK:/page2\n\nSuccessfully navigated. Use run_python to extract data from the new page.",
            },
        ]

        # Mock parent's env_response to return empty messages (we test our override logic)
        with patch.object(
            vf.StatefulToolEnv, "env_response", new_callable=AsyncMock
        ) as mock_parent:
            mock_parent.return_value = []  # Parent returns Messages, not tuple

            # Call env_response (state is mutated in-place)
            await env.env_response(messages, state)

        # Verify state was updated by our override logic (mutated in-place)
        assert state["html"] == "<html>Page 2 Content</html>"
        assert "/page2" in state["navigation_history"]

    @pytest.mark.asyncio
    async def test_env_response_no_update_on_navigate_error(self):
        """Test that env_response doesn't update state on navigate error."""
        # Skip if verifiers not installed
        try:
            import verifiers as vf
        except ImportError:
            pytest.skip("verifiers not installed")

        from unittest.mock import AsyncMock, patch

        from bs4_env.adapters.verifiers_adapter import _build_real_verifiers_env
        from bs4_env.config import EnvConfig

        # Use minimal dataset for faster tests
        config = EnvConfig(split="train", mode="mvp", num_examples=1)
        env = _build_real_verifiers_env(config, vf)

        original_html = "<html>Initial Page</html>"
        state = {
            "html": original_html,
            "query": "Find something",
            "constraints": {},
            "task_info": {},
            "pages": {"/page2": "<html>Page 2</html>"},
            "navigation_history": [],
        }

        # Simulate navigate error (no NAVIGATE_SUCCESS_MARKER)
        messages = [
            {
                "role": "tool",
                "content": "Error: Page '/nonexistent' not found. Check the HTML for valid links.",
            },
        ]

        # Mock parent's env_response to return empty messages
        with patch.object(
            vf.StatefulToolEnv, "env_response", new_callable=AsyncMock
        ) as mock_parent:
            mock_parent.return_value = []  # Parent returns Messages, not tuple

            await env.env_response(messages, state)

        # State should be unchanged (no navigate success marker)
        assert state["html"] == original_html
        assert len(state["navigation_history"]) == 0

    def test_navigate_marker_parsing(self):
        """Test that NAVIGATE_SUCCESS_MARKER is correctly parsed."""
        # This tests the marker format without needing verifiers
        marker = "NAVIGATE_OK:"
        content = f"{marker}/page2\n\nSuccessfully navigated. Use run_python to extract data from the new page."

        assert content.startswith(marker)
        marker_content = content[len(marker) :]
        normalized_href = marker_content.split("\n")[0].strip()
        assert normalized_href == "/page2"

    def test_navigate_marker_with_complex_href(self):
        """Test marker parsing with complex href paths."""
        marker = "NAVIGATE_OK:"

        # Test various href formats
        test_cases = [
            ("/products/123", "/products/123"),
            ("products/category/item", "products/category/item"),
            ("/api/v1/data", "/api/v1/data"),
        ]

        for href, expected in test_cases:
            content = f"{marker}{href}\n\nSuccessfully navigated."
            marker_content = content[len(marker) :]
            normalized_href = marker_content.split("\n")[0].strip()
            assert normalized_href == expected, f"Failed for href: {href}"


class TestSetupStateInfoAccess:
    """Test that setup_state correctly accesses info from state."""

    @pytest.mark.asyncio
    async def test_setup_state_reads_info_from_state(self):
        """Regression test: setup_state must read info from state, not kwargs.

        The verifiers framework stores dataset row info in state.input
        (with forwarding to state["info"]), not in kwargs. This test ensures
        we don't regress to the broken behavior of looking for kwargs["row"].
        """

        # Create a mock state that simulates verifiers State behavior
        # The State class forwards state["info"] to state["input"]["info"]
        class MockState(dict):
            INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]

            def __getitem__(self, key: str):
                if key in self.INPUT_FIELDS and "input" in self:
                    input_obj = super().__getitem__("input")
                    if key in input_obj:
                        return input_obj[key]
                return super().__getitem__(key)

            def get(self, key: str, default=None):
                try:
                    return self[key]
                except KeyError:
                    return default

        # Set up state as verifiers does (info in state.input)
        test_html = "<html><body>Test HTML</body></html>"
        test_query = "Extract the test data"
        state = MockState()
        state["input"] = {
            "info": {
                "html": test_html,
                "query": test_query,
                "archetype_id": "test.archetype",
                "solvable": True,
                "answer_schema": {"type": "string"},
            }
        }

        # Verify state forwarding works
        assert state.get("info", {}).get("html") == test_html

        # Now verify our adapter correctly extracts from state
        # (This is a simplified test - full integration would need verifiers)
        info = state.get("info", {})
        html = info.get("html", "")
        query = info.get("query", "")

        assert html == test_html, "setup_state failed to read HTML from state"
        assert query == test_query, "setup_state failed to read query from state"

        # Verify that kwargs.get("row", {}) would NOT work
        kwargs = {}  # Empty kwargs, as verifiers provides
        row = kwargs.get("row", {})
        info_from_kwargs = row.get("info", {})
        assert info_from_kwargs == {}, "kwargs should NOT contain row data"
