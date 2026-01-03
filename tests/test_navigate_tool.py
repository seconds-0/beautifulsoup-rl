"""Unit tests for the navigate tool and NavigationState.

These tests focus on the navigation mechanics that are independent
of the verifiers framework integration.
"""

import pytest

from bs4_env.tools.harness import NAVIGATE_SUCCESS_MARKER
from bs4_env.tools.tool_defs import (
    NavigationState,
    create_navigate_handler,
    create_tool_registry,
)


class TestNavigationState:
    """Direct tests for NavigationState class."""

    def test_navigate_success(self):
        """Navigate to existing page should succeed."""
        pages = {
            "/page1": "<html>Page 1</html>",
            "/page2": "<html>Page 2</html>",
        }
        state = NavigationState("<html>Initial</html>", pages)

        success, result = state.navigate("/page1")

        assert success is True
        assert result == "<html>Page 1</html>"
        assert state.current_html == "<html>Page 1</html>"
        assert state.navigation_history == ["/page1"]

    def test_navigate_not_found(self):
        """Navigate to non-existent page should fail."""
        pages = {"/page1": "<html>Page 1</html>"}
        state = NavigationState("<html>Initial</html>", pages)

        success, result = state.navigate("/nonexistent")

        assert success is False
        assert "not found" in result
        assert state.current_html == "<html>Initial</html>"
        assert state.navigation_history == []

    def test_multiple_navigations(self):
        """Multiple sequential navigations should work."""
        pages = {
            "/page1": "<html>Page 1</html>",
            "/page2": "<html>Page 2</html>",
            "/page3": "<html>Page 3</html>",
        }
        state = NavigationState("<html>Initial</html>", pages)

        # Navigate through pages
        success1, _ = state.navigate("/page1")
        success2, _ = state.navigate("/page2")
        success3, _ = state.navigate("/page3")

        assert success1 and success2 and success3
        assert state.current_html == "<html>Page 3</html>"
        assert state.navigation_history == ["/page1", "/page2", "/page3"]

    def test_navigation_after_error(self):
        """Navigation should work after a failed attempt."""
        pages = {"/page1": "<html>Page 1</html>"}
        state = NavigationState("<html>Initial</html>", pages)

        # Try nonexistent page
        success1, _ = state.navigate("/nonexistent")
        assert success1 is False
        assert state.current_html == "<html>Initial</html>"

        # Navigate to existing page should still work
        success2, _ = state.navigate("/page1")
        assert success2 is True
        assert state.current_html == "<html>Page 1</html>"
        assert state.navigation_history == ["/page1"]

    def test_navigate_back_to_previous(self):
        """Should be able to navigate back to a previous page."""
        pages = {
            "/page1": "<html>Page 1</html>",
            "/page2": "<html>Page 2</html>",
        }
        state = NavigationState("<html>Initial</html>", pages)

        state.navigate("/page1")
        state.navigate("/page2")
        state.navigate("/page1")

        assert state.current_html == "<html>Page 1</html>"
        assert state.navigation_history == ["/page1", "/page2", "/page1"]


class TestNavigationStateNormalization:
    """Tests for href normalization in NavigationState."""

    def test_exact_match(self):
        """Exact match should work."""
        pages = {"/products": "<html>Products</html>"}
        state = NavigationState("", pages)

        success, _ = state.navigate("/products")
        assert success is True

    def test_normalize_leading_slash(self):
        """Should match with or without leading slash."""
        pages = {"products": "<html>Products</html>"}
        state = NavigationState("", pages)

        # With leading slash should match without
        success, _ = state.navigate("/products")
        assert success is True

    def test_normalize_without_leading_slash(self):
        """Should match without leading slash when pages have it."""
        pages = {"/products": "<html>Products</html>"}
        state = NavigationState("", pages)

        # Without leading slash should match with
        success, _ = state.navigate("products")
        assert success is True

    def test_normalize_query_string(self):
        """Should strip query strings for matching."""
        pages = {"/page": "<html>Page</html>"}
        state = NavigationState("", pages)

        success, _ = state.navigate("/page?foo=bar&baz=qux")
        assert success is True

    def test_normalize_fragment(self):
        """Should strip fragments for matching."""
        pages = {"/page": "<html>Page</html>"}
        state = NavigationState("", pages)

        success, _ = state.navigate("/page#section")
        assert success is True

    def test_normalize_whitespace(self):
        """Should strip leading/trailing whitespace."""
        pages = {"/page": "<html>Page</html>"}
        state = NavigationState("", pages)

        success, _ = state.navigate("  /page  ")
        assert success is True

    def test_unicode_href(self):
        """Should handle unicode in href."""
        pages = {
            "/产品": "<html>Products in Chinese</html>",
            "/製品": "<html>Products in Japanese</html>",
        }
        state = NavigationState("", pages)

        success1, result1 = state.navigate("/产品")
        assert success1 is True
        assert "Chinese" in result1

        success2, result2 = state.navigate("/製品")
        assert success2 is True
        assert "Japanese" in result2

    def test_complex_path(self):
        """Should handle complex paths."""
        pages = {
            "/api/v1/products/123": "<html>Product 123</html>",
            "/docs/guides/getting-started": "<html>Getting Started</html>",
        }
        state = NavigationState("", pages)

        success1, _ = state.navigate("/api/v1/products/123")
        assert success1 is True

        success2, _ = state.navigate("/docs/guides/getting-started")
        assert success2 is True


class TestCreateNavigateHandler:
    """Tests for create_navigate_handler function."""

    def test_handler_without_nav_state(self):
        """Handler without nav_state should return error."""
        handler = create_navigate_handler(None)

        result = handler({"href": "/some-page"})

        assert "Error" in result
        assert "does not support navigation" in result

    def test_handler_empty_href(self):
        """Handler should error on empty href."""
        pages = {"/page": "<html>Page</html>"}
        state = NavigationState("", pages)
        handler = create_navigate_handler(state)

        result = handler({"href": ""})

        assert "Error" in result
        assert "No href" in result

    def test_handler_missing_href(self):
        """Handler should error on missing href."""
        pages = {"/page": "<html>Page</html>"}
        state = NavigationState("", pages)
        handler = create_navigate_handler(state)

        result = handler({})

        assert "Error" in result
        assert "No href" in result

    def test_handler_success_returns_marker(self):
        """Successful navigation should include NAVIGATE_SUCCESS_MARKER."""
        pages = {"/page": "<html>Page</html>"}
        state = NavigationState("", pages)
        handler = create_navigate_handler(state)

        result = handler({"href": "/page"})

        assert result.startswith(NAVIGATE_SUCCESS_MARKER)
        assert "Successfully navigated" in result

    def test_handler_failure_no_marker(self):
        """Failed navigation should not include success marker."""
        pages = {"/page": "<html>Page</html>"}
        state = NavigationState("", pages)
        handler = create_navigate_handler(state)

        result = handler({"href": "/nonexistent"})

        assert not result.startswith(NAVIGATE_SUCCESS_MARKER)
        assert "Error" in result


class TestToolRegistryNavigation:
    """Tests for navigate tool in ToolRegistry."""

    @pytest.fixture
    def executor(self):
        """Create a mock executor."""
        from bs4_env.tools.executor import LocalSubprocessExecutor

        return LocalSubprocessExecutor(max_output_chars=1000)

    def test_multistep_task_has_working_navigate(self, executor):
        """Multi-step tasks should have working navigate tool."""
        pages = {
            "/page1": "<html>Page 1</html>",
            "/page2": "<html>Page 2</html>",
        }

        registry = create_tool_registry(
            executor=executor,
            html="<html>Initial</html>",
            query="Test query",
            constraints={},
            task_info={},
            pages=pages,
        )

        assert registry.has_tool("navigate")

        # Navigation should succeed
        result = registry.call("navigate", {"href": "/page1"})
        assert NAVIGATE_SUCCESS_MARKER in result

    def test_singlestep_task_navigate_returns_error(self, executor):
        """Single-step tasks should have navigate that returns helpful error."""
        registry = create_tool_registry(
            executor=executor,
            html="<html>Test</html>",
            query="Test query",
            constraints={},
            task_info={},
            pages=None,
        )

        assert registry.has_tool("navigate")

        result = registry.call("navigate", {"href": "/page"})
        assert "Error" in result
        assert "does not support navigation" in result

    def test_navigate_updates_run_python_html(self, executor):
        """After navigate, run_python should use updated HTML."""
        pages = {
            "/page2": "<html><div id='target'>New Content</div></html>",
        }

        registry = create_tool_registry(
            executor=executor,
            html="<html>Initial</html>",
            query="Test",
            constraints={},
            task_info={},
            pages=pages,
        )

        # Navigate to page2
        nav_result = registry.call("navigate", {"href": "/page2"})
        assert NAVIGATE_SUCCESS_MARKER in nav_result

        # run_python should now see the new HTML
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, 'html.parser')
print(soup.find('div', id='target').text)
"""
        result = registry.call("run_python", {"code": code})
        assert "New Content" in result

    def test_navigate_sequence_run_python(self, executor):
        """Run_python should see HTML from latest navigation."""
        pages = {
            "/page1": "<html><span class='marker'>PAGE1</span></html>",
            "/page2": "<html><span class='marker'>PAGE2</span></html>",
        }

        registry = create_tool_registry(
            executor=executor,
            html="<html>Initial</html>",
            query="Test",
            constraints={},
            task_info={},
            pages=pages,
        )

        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, 'html.parser')
print(soup.find('span', class_='marker').text)
"""

        # Navigate to page1
        registry.call("navigate", {"href": "/page1"})
        result1 = registry.call("run_python", {"code": code})
        assert "PAGE1" in result1

        # Navigate to page2
        registry.call("navigate", {"href": "/page2"})
        result2 = registry.call("run_python", {"code": code})
        assert "PAGE2" in result2


class TestNavigateMarkerFormat:
    """Tests for NAVIGATE_SUCCESS_MARKER format and parsing."""

    def test_marker_format(self):
        """Verify marker format is correct."""
        assert NAVIGATE_SUCCESS_MARKER == "NAVIGATE_OK:"

    def test_parse_marker_simple(self):
        """Parse marker with simple href."""
        content = f"{NAVIGATE_SUCCESS_MARKER}/page\n\nMessage"

        assert content.startswith(NAVIGATE_SUCCESS_MARKER)
        href = content[len(NAVIGATE_SUCCESS_MARKER) :].split("\n")[0]
        assert href == "/page"

    def test_parse_marker_complex_path(self):
        """Parse marker with complex path."""
        content = f"{NAVIGATE_SUCCESS_MARKER}/api/v1/data/123\n\nMessage"

        href = content[len(NAVIGATE_SUCCESS_MARKER) :].split("\n")[0]
        assert href == "/api/v1/data/123"

    def test_parse_marker_unicode(self):
        """Parse marker with unicode href."""
        content = f"{NAVIGATE_SUCCESS_MARKER}/产品/详情\n\nMessage"

        href = content[len(NAVIGATE_SUCCESS_MARKER) :].split("\n")[0]
        assert href == "/产品/详情"
