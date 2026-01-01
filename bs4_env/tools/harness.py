from __future__ import annotations

"""Code harness for BeautifulSoup RL environment.

This module builds the runner script that wraps user code with:
- Injected globals (HTML, QUERY, CONSTRAINTS)
- Helper functions (make_soup)
- Import setup
- Error handling
"""

import base64
import json
from typing import Any


def build_runner_script(
    user_code: str,
    globals_dict: dict[str, Any],
    enforce_bs4_usage: bool = False,
) -> str:
    """Build a complete runner script that executes user code.

    Args:
        user_code: The Python code to execute.
        globals_dict: Dictionary of globals to inject (HTML, QUERY, CONSTRAINTS).
        enforce_bs4_usage: Whether to check that BeautifulSoup is actually used.

    Returns:
        Complete Python script ready for execution.
    """
    # Use base64 encoding for HTML and QUERY to avoid escape edge cases
    # This handles all special characters (quotes, backslashes, unicode) cleanly
    html_b64 = base64.b64encode(globals_dict.get("HTML", "").encode("utf-8")).decode("ascii")
    query_b64 = base64.b64encode(globals_dict.get("QUERY", "").encode("utf-8")).decode("ascii")
    constraints_json = json.dumps(globals_dict.get("CONSTRAINTS", {}))

    # Build the script
    script = f'''#!/usr/bin/env python3
"""Auto-generated runner script for BeautifulSoup RL environment."""

import sys
import json
import base64

# =============================================================================
# Injected Globals (base64 encoded to avoid escape issues)
# =============================================================================

HTML = base64.b64decode("{html_b64}").decode("utf-8")

QUERY = base64.b64decode("{query_b64}").decode("utf-8")

CONSTRAINTS = json.loads('{constraints_json}')

# =============================================================================
# Helper Functions
# =============================================================================

def make_soup(parser: str = "html.parser"):
    """Create a BeautifulSoup object from the HTML global.

    Args:
        parser: Parser to use. Options:
            - "html.parser": Python's built-in (default, always available)
            - "lxml": Fast, lenient (requires lxml)
            - "lxml-xml": XML mode (requires lxml)
            - "html5lib": Most lenient, slowest (requires html5lib)

    Returns:
        BeautifulSoup object parsed from HTML.
    """
    from bs4 import BeautifulSoup
    return BeautifulSoup(HTML, parser)


# =============================================================================
# Imports (available to user code)
# =============================================================================

from bs4 import BeautifulSoup, NavigableString, Tag, Comment
import re
import json

# =============================================================================
# User Code
# =============================================================================

{user_code}
'''

    return script


def build_tool_response(result: dict[str, Any]) -> str:
    """Build a formatted tool response from execution result.

    Args:
        result: Dictionary with stdout, stderr, exit_code, etc.

    Returns:
        Formatted string response for the tool call.
    """
    lines = []

    if result.get("timed_out"):
        lines.append("TIMEOUT: Execution exceeded time limit")
        lines.append("")

    if result.get("error"):
        lines.append(f"EXECUTOR ERROR: {result['error']}")
        lines.append("")

    if result.get("stdout"):
        lines.append("=== STDOUT ===")
        lines.append(result["stdout"])
        lines.append("")

    if result.get("stderr"):
        lines.append("=== STDERR ===")
        lines.append(result["stderr"])
        lines.append("")

    exit_code = result.get("exit_code", 0)
    if exit_code != 0:
        lines.append(f"Exit code: {exit_code}")

    if result.get("runtime_ms"):
        lines.append(f"Runtime: {result['runtime_ms']}ms")

    return "\n".join(lines).strip()


def extract_print_output(stdout: str) -> str | None:
    """Extract the last printed value from stdout.

    Many solutions print their result. This extracts that value.

    Args:
        stdout: The stdout from execution.

    Returns:
        The last printed line, or None if stdout is empty.
    """
    if not stdout:
        return None

    lines = stdout.strip().split("\n")
    if lines:
        return lines[-1]
    return None


def check_bs4_usage(code: str) -> bool:
    """Check if code appears to use BeautifulSoup.

    This is a heuristic check, not a guarantee.

    Args:
        code: The Python code to check.

    Returns:
        True if code appears to use BS4.
    """
    bs4_indicators = [
        "BeautifulSoup",
        "make_soup",
        "from bs4",
        "import bs4",
        ".find(",
        ".find_all(",
        ".select(",
        ".select_one(",
        ".get_text(",
        ".string",
        ".text",
        ".attrs",
        "NavigableString",
        "Tag",
    ]

    code_lower = code.lower()
    return any(indicator.lower() in code_lower for indicator in bs4_indicators)


# Tool definition for Verifiers integration
RUN_PYTHON_TOOL_SCHEMA = {
    "name": "run_python",
    "description": (
        "Execute Python code in an isolated sandbox with BeautifulSoup installed. "
        "The sandbox has these globals available:\n"
        "- HTML: The HTML content to parse\n"
        "- QUERY: The natural language task description\n"
        "- CONSTRAINTS: Dictionary with output schema and rules\n"
        "- make_soup(parser): Helper to create BeautifulSoup object\n"
        "\n"
        "Print your result to stdout. Network access is disabled."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use print() to output results.",
            }
        },
        "required": ["code"],
    },
}

GET_TASK_METADATA_TOOL_SCHEMA = {
    "name": "get_task_metadata",
    "description": (
        "Get metadata about the current task including output schema "
        "and allowed limitation reasons. Does NOT reveal the answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

LINT_JSON_TOOL_SCHEMA = {
    "name": "lint_json",
    "description": "Validate a JSON string against the expected output schema.",
    "parameters": {
        "type": "object",
        "properties": {
            "json_string": {
                "type": "string",
                "description": "JSON string to validate",
            }
        },
        "required": ["json_string"],
    },
}

NAVIGATE_TOOL_SCHEMA = {
    "name": "navigate",
    "description": (
        "Navigate to a linked page by its href. For multi-step tasks, "
        "use this tool to follow links. After successful navigation, "
        "the HTML global in run_python will be updated with the new page. "
        "Returns a success message or error if the href is not found."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "href": {
                "type": "string",
                "description": "The href attribute of the link to navigate to",
            }
        },
        "required": ["href"],
    },
}

# Structured marker for navigate success - used by env_response to detect navigation
# This is more robust than parsing user-facing message text
NAVIGATE_SUCCESS_MARKER = "NAVIGATE_OK:"
