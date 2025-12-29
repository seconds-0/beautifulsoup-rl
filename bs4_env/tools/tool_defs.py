from __future__ import annotations

"""Tool definitions and handlers for BeautifulSoup RL environment.

This module provides the tool interface that models interact with.
"""

import json
from typing import Any, Callable

from bs4_env.tools.executor import Executor, ExecResult
from bs4_env.tools.harness import (
    RUN_PYTHON_TOOL_SCHEMA,
    GET_TASK_METADATA_TOOL_SCHEMA,
    LINT_JSON_TOOL_SCHEMA,
    NAVIGATE_TOOL_SCHEMA,
    build_tool_response,
)
from bs4_env.grading.schema import validate_output


def create_run_python_handler(
    executor: Executor,
    html: str,
    query: str,
    constraints: dict,
    timeout_s: float = 30.0,
) -> Callable[[dict], str]:
    """Create a handler for the run_python tool.

    Args:
        executor: The code executor to use.
        html: The HTML content for this task.
        query: The task query.
        constraints: The task constraints (visible to model).
        timeout_s: Execution timeout.

    Returns:
        A function that handles run_python tool calls.
    """

    def handler(params: dict) -> str:
        code = params.get("code", "")
        if not code:
            return "Error: No code provided"

        globals_dict = {
            "HTML": html,
            "QUERY": query,
            "CONSTRAINTS": constraints,
        }

        result = executor.run(code, globals_dict, timeout_s)
        return build_tool_response(result.to_dict())

    return handler


def create_get_metadata_handler(
    task_info: dict,
) -> Callable[[dict], str]:
    """Create a handler for the get_task_metadata tool.

    Args:
        task_info: The full task info (will be filtered to remove secrets).

    Returns:
        A function that handles get_task_metadata tool calls.
    """

    def handler(params: dict) -> str:
        # Return only non-secret information
        public_info = {
            "answer_schema": task_info.get("answer_schema", {}),
            "solvable_hint": "This task may or may not be solvable with static HTML parsing.",
        }

        # Include allowed limit reasons if this might be unsolvable
        limit_info = task_info.get("limit_info", {})
        if limit_info.get("allowed_reasons"):
            public_info["allowed_limit_reasons"] = limit_info["allowed_reasons"]

        return json.dumps(public_info, indent=2)

    return handler


def create_lint_json_handler(
    task_info: dict,
) -> Callable[[dict], str]:
    """Create a handler for the lint_json tool.

    Args:
        task_info: The task info containing answer_schema.

    Returns:
        A function that handles lint_json tool calls.
    """

    def handler(params: dict) -> str:
        json_string = params.get("json_string", "")
        if not json_string:
            return "Error: No JSON string provided"

        output, errors = validate_output(json_string, task_info)

        if errors:
            return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)
        else:
            return "JSON is valid according to the output schema."

    return handler


class NavigationState:
    """Mutable state container for multi-step navigation.

    This allows the navigate tool to update the current HTML,
    which is then used by subsequent run_python calls.
    """

    def __init__(self, initial_html: str, pages: dict[str, str]):
        self.current_html = initial_html
        self.pages = pages
        self.navigation_history: list[str] = []

    def navigate(self, href: str) -> tuple[bool, str]:
        """Navigate to a new page.

        Args:
            href: The href to navigate to.

        Returns:
            Tuple of (success, message_or_html).
        """
        # Normalize href (handle relative paths, anchors, etc.)
        normalized = self._normalize_href(href)

        if normalized not in self.pages:
            available = list(self.pages.keys())[:5]
            hint = f" Available pages: {available}" if available else ""
            return False, f"Error: Page '{href}' not found.{hint}"

        self.current_html = self.pages[normalized]
        self.navigation_history.append(normalized)
        return True, self.current_html

    def _normalize_href(self, href: str) -> str:
        """Normalize an href for lookup.

        Handles common variations like leading slashes, query strings.
        """
        # Strip leading/trailing whitespace
        href = href.strip()

        # Try exact match first
        if href in self.pages:
            return href

        # Try with/without leading slash
        if href.startswith("/"):
            without_slash = href[1:]
            if without_slash in self.pages:
                return without_slash
        else:
            with_slash = "/" + href
            if with_slash in self.pages:
                return with_slash

        # Try without query string
        if "?" in href:
            base = href.split("?")[0]
            if base in self.pages:
                return base

        # Try without fragment
        if "#" in href:
            base = href.split("#")[0]
            if base in self.pages:
                return base

        return href


def create_navigate_handler(
    nav_state: NavigationState,
) -> Callable[[dict], str]:
    """Create a handler for the navigate tool.

    Args:
        nav_state: Navigation state containing pages and current HTML.

    Returns:
        A function that handles navigate tool calls.
    """

    def handler(params: dict) -> str:
        href = params.get("href", "")
        if not href:
            return "Error: No href provided"

        success, result = nav_state.navigate(href)

        if success:
            # Return a summary, not the full HTML (model will access via run_python)
            html_preview = result[:500] + "..." if len(result) > 500 else result
            return (
                f"Successfully navigated to '{href}'. "
                f"The HTML global has been updated with the new page content. "
                f"Use run_python to extract data from the new page.\n\n"
                f"Preview of new page:\n{html_preview}"
            )
        else:
            return result

    return handler


def create_run_python_handler_with_nav(
    executor: Executor,
    nav_state: NavigationState,
    query: str,
    constraints: dict,
    timeout_s: float = 30.0,
) -> Callable[[dict], str]:
    """Create a run_python handler that uses NavigationState for HTML.

    This version reads HTML from nav_state.current_html, which is updated
    by the navigate tool.

    Args:
        executor: The code executor to use.
        nav_state: Navigation state containing current HTML.
        query: The task query.
        constraints: The task constraints (visible to model).
        timeout_s: Execution timeout.

    Returns:
        A function that handles run_python tool calls.
    """

    def handler(params: dict) -> str:
        code = params.get("code", "")
        if not code:
            return "Error: No code provided"

        globals_dict = {
            "HTML": nav_state.current_html,
            "QUERY": query,
            "CONSTRAINTS": constraints,
        }

        result = executor.run(code, globals_dict, timeout_s)
        return build_tool_response(result.to_dict())

    return handler


class ToolRegistry:
    """Registry of available tools for an environment instance."""

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable[[dict], str]] = {}

    def register(
        self,
        name: str,
        schema: dict,
        handler: Callable[[dict], str],
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name.
            schema: Tool schema (JSON Schema format).
            handler: Function that handles tool calls.
        """
        self._tools[name] = schema
        self._handlers[name] = handler

    def get_schema(self, name: str) -> dict | None:
        """Get schema for a tool."""
        return self._tools.get(name)

    def get_all_schemas(self) -> list[dict]:
        """Get schemas for all registered tools."""
        return list(self._tools.values())

    def call(self, name: str, params: dict) -> str:
        """Call a tool by name.

        Args:
            name: Tool name.
            params: Tool parameters.

        Returns:
            Tool response string.

        Raises:
            KeyError: If tool is not registered.
        """
        if name not in self._handlers:
            raise KeyError(f"Unknown tool: {name}")
        return self._handlers[name](params)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


def create_tool_registry(
    executor: Executor,
    html: str,
    query: str,
    constraints: dict,
    task_info: dict,
    timeout_s: float = 30.0,
    include_optional_tools: bool = True,
    pages: dict[str, str] | None = None,
) -> ToolRegistry:
    """Create a fully configured tool registry for a task.

    Args:
        executor: The code executor.
        html: The HTML content (initial page for multi-step tasks).
        query: The task query.
        constraints: The task constraints.
        task_info: The full task info.
        timeout_s: Execution timeout.
        include_optional_tools: Whether to include get_task_metadata and lint_json.
        pages: For multi-step tasks, a dict mapping href -> HTML content.

    Returns:
        Configured ToolRegistry.
    """
    registry = ToolRegistry()

    # Check if this is a multi-step task
    is_multistep = pages and len(pages) > 0

    if is_multistep:
        # Create navigation state for multi-step tasks
        nav_state = NavigationState(html, pages)

        # Use the navigation-aware run_python handler
        registry.register(
            "run_python",
            RUN_PYTHON_TOOL_SCHEMA,
            create_run_python_handler_with_nav(
                executor, nav_state, query, constraints, timeout_s
            ),
        )

        # Add navigate tool
        registry.register(
            "navigate",
            NAVIGATE_TOOL_SCHEMA,
            create_navigate_handler(nav_state),
        )
    else:
        # Standard single-step task
        registry.register(
            "run_python",
            RUN_PYTHON_TOOL_SCHEMA,
            create_run_python_handler(executor, html, query, constraints, timeout_s),
        )

    # Optional tools
    if include_optional_tools:
        registry.register(
            "get_task_metadata",
            GET_TASK_METADATA_TOOL_SCHEMA,
            create_get_metadata_handler(task_info),
        )

        registry.register(
            "lint_json",
            LINT_JSON_TOOL_SCHEMA,
            create_lint_json_handler(task_info),
        )

    return registry
