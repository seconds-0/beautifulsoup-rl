from __future__ import annotations

"""Code harness for BeautifulSoup RL environment.

This module builds the runner script that wraps user code with:
- Injected globals (HTML, QUERY, CONSTRAINTS)
- Helper functions (make_soup)
- Import setup
- Error handling
"""

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
    # Escape the values for embedding in the script
    html_escaped = _escape_for_python(globals_dict.get("HTML", ""))
    query_escaped = _escape_for_python(globals_dict.get("QUERY", ""))
    constraints_json = json.dumps(globals_dict.get("CONSTRAINTS", {}))

    # Build the script
    script = f'''#!/usr/bin/env python3
"""Auto-generated runner script for BeautifulSoup RL environment."""

import sys
import json

# =============================================================================
# Injected Globals
# =============================================================================

HTML = """{html_escaped}"""

QUERY = """{query_escaped}"""

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


def _escape_for_python(s: str) -> str:
    """Escape a string for embedding in a Python triple-quoted string.

    Args:
        s: The string to escape.

    Returns:
        Escaped string safe for embedding in triple quotes.
    """
    # Handle triple quotes within the string
    s = s.replace('"""', '\\"\\"\\"')
    # Handle backslashes (must be done before other escapes)
    # Actually, in triple-quoted strings, we mainly need to escape
    # the triple quote sequence itself
    return s


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
    for indicator in bs4_indicators:
        if indicator.lower() in code_lower:
            return True

    return False


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
