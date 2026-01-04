from __future__ import annotations

"""Reward computation for BeautifulSoup RL environment.

This module implements the rubric that converts model outputs into rewards.
The reward function must be deterministic and follow the anti-hacking rules.
"""

import ast
import json
import re
from typing import Any

from bs4_env.grading.normalize import (
    coerce_integer,
    normalize_object_keys,
    normalize_price,
    normalize_value,
    values_equal,
)
from bs4_env.grading.safety import check_safety
from bs4_env.grading.schema import validate_output

# Reward values (configurable)
REWARD_CORRECT = 1.0
REWARD_CORRECT_LIMIT = 0.5
REWARD_WRONG = 0.0
REWARD_SAFETY_VIOLATION = -0.5
REWARD_FORMAT_ERROR = 0.0
REWARD_PARTIAL_MAX = (
    0.1  # Maximum reward for partially correct structured outputs (kept low to prevent farming)
)

# Efficiency settings
DEFAULT_MAX_TOOL_CALLS = 10  # Default hard cutoff - more than this is considered failure
MAX_TOOL_CALLS = DEFAULT_MAX_TOOL_CALLS  # Alias for backward compatibility
EFFICIENCY_PENALTY_PER_CALL = 0.1  # -10% per additional call after first
EFFICIENCY_FLOOR = 0.2  # Minimum multiplier before hard cutoff

# Tool call weights - navigate is structurally required for multi-step tasks
# so it's penalized less than general-purpose tool calls
TOOL_WEIGHTS = {
    "run_python": 1.0,  # Full cost - main tool
    "navigate": 0.2,  # Low cost - required for multi-step tasks
    "lint_json": 0.0,  # Helper tool - no cost
    "get_task_metadata": 0.0,  # Helper tool - no cost
}

# BS4 usage settings - creates gradient toward BS4 usage without rejecting alternatives
BS4_USAGE_PENALTY = 0.15  # Penalty for not using BeautifulSoup

# Process partial credit settings for 0% model bootstrapping
# These reward correct tool-use patterns even when the final answer is wrong
# This provides gradient signal for models learning the basic action template
PROCESS_PARTIAL_CREDIT_ENABLED = True  # Set to False for strict benchmarks
PROCESS_PARTIAL_CREDIT_CAP = 0.30  # Max partial credit (below REWARD_CORRECT_LIMIT)

# Tier rewards for process partial credit (must use injected HTML variable)
PROCESS_TIER_REWARDS = {
    "bs4_imported": 0.05,  # Credit for importing BS4
    "soup_created_with_html": 0.10,  # Credit for BeautifulSoup(HTML, ...)
    "selection_method": 0.10,  # Credit for using .find(), .select(), etc.
    "content_access": 0.05,  # Credit for accessing .text, .get_text(), etc.
}


def get_max_tool_calls(task_info: dict | None) -> int:
    """Get the max tool calls limit for a task.

    Reads from task constraints if available, otherwise returns the default.
    This allows archetypes to specify custom limits (e.g., multi-step tasks
    that legitimately need more tool calls).

    Args:
        task_info: The task info dictionary, or None.

    Returns:
        Maximum allowed tool calls before hard cutoff.
    """
    if task_info is None:
        return DEFAULT_MAX_TOOL_CALLS

    # Try to read from metadata.constraints.max_tool_calls
    metadata = task_info.get("metadata", {})
    # metadata might be JSON-serialized string in dataset
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    # Ensure metadata is a dict
    if not isinstance(metadata, dict):
        return DEFAULT_MAX_TOOL_CALLS

    constraints = metadata.get("constraints", {})
    if not isinstance(constraints, dict):
        return DEFAULT_MAX_TOOL_CALLS

    max_calls = constraints.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    # Type validation: ensure it's an int (or can be coerced to int)
    if isinstance(max_calls, int):
        return max_calls
    if isinstance(max_calls, float) and max_calls == int(max_calls):
        return int(max_calls)
    if isinstance(max_calls, str):
        try:
            return int(max_calls)
        except ValueError:
            pass
    # Invalid type - return default
    return DEFAULT_MAX_TOOL_CALLS


def compute_efficiency_multiplier(
    tool_call_count: float,
    max_tool_calls: int | None = None,
) -> float:
    """Compute efficiency multiplier based on tool call count.

    Rewards efficient solutions that solve in fewer calls.
    Creates gradient signal for RL to optimize efficiency, not just correctness.

    Args:
        tool_call_count: Number of (weighted) tool calls made during the episode.
            Can be a float if weighted tool counting is used.
        max_tool_calls: Maximum allowed tool calls for hard cutoff.
            Defaults to DEFAULT_MAX_TOOL_CALLS if not specified.
            This allows per-task limits (e.g., multi-step tasks may allow more).

    Returns:
        Multiplier in range [0.0, 1.0]:
        - 1 call: 1.0 (full credit)
        - 2 calls: 0.9
        - 3 calls: 0.8
        - ...
        - 9-10 calls: 0.2 (floor)
        - 11+ calls: 0.0 (hard cutoff - treated as failure)
    """
    limit = max_tool_calls if max_tool_calls is not None else DEFAULT_MAX_TOOL_CALLS
    if tool_call_count > limit:
        return 0.0  # Hard cutoff - brute force isn't skill
    if tool_call_count <= 1:
        return 1.0
    return max(EFFICIENCY_FLOOR, 1.0 - EFFICIENCY_PENALTY_PER_CALL * (tool_call_count - 1))


def compute_weighted_tool_count(tool_calls: list[dict]) -> float:
    """Compute weighted tool call count based on tool types.

    Different tools have different weights:
    - run_python: 1.0 (full cost)
    - navigate: 0.2 (low cost - structurally required for multi-step)
    - helper tools: 0.0 (no cost)

    This allows multi-step tasks to use navigate calls without heavy penalty.

    Args:
        tool_calls: List of tool call dicts with 'function.name' or 'name' key.
            Only actual tool calls are counted - tool results and metadata are skipped.

    Returns:
        Weighted total of tool calls (can be fractional).
    """
    total = 0.0
    for tc in tool_calls:
        if isinstance(tc, dict):
            # Extract tool name from various formats
            # Skip dicts that don't look like tool calls (no name or function key)
            name = tc.get("name") or tc.get("function", {}).get("name")
            if name is None:
                # This dict doesn't look like a tool call - skip it
                # (could be a tool result, metadata, etc.)
                continue
            total += TOOL_WEIGHTS.get(name, 1.0)
    return total


def _check_bs4_usage_ast(code: str) -> bool:
    """Check if code uses BeautifulSoup using AST analysis.

    Uses AST parsing to detect actual BS4 usage, ignoring comments and strings.
    This prevents bypassing the penalty by mentioning "BeautifulSoup" in comments.

    Args:
        code: A single code string to analyze.

    Returns:
        True if BS4 usage is detected in actual code (not comments/strings).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code doesn't parse, fall back to string heuristic
        # but only for unambiguous patterns
        return False

    # BS4-specific method names that don't appear in standard library
    bs4_methods = {
        "find_all",
        "select",
        "select_one",
        "get_text",
        "prettify",
        "decode_contents",
        "encode_contents",
        "new_tag",
        "new_string",
    }

    # BS4-specific attribute names
    # Note: Excludes .name and .string as they're too common (file.name, path.name, etc.)
    bs4_attrs = {
        "next_sibling",
        "previous_sibling",
        "next_siblings",
        "previous_siblings",
        "next_element",
        "previous_element",
        "children",
        "descendants",
        "contents",
        "attrs",
    }

    class BS4Visitor(ast.NodeVisitor):
        """AST visitor to detect BS4 usage."""

        def __init__(self):
            self.found_bs4 = False

        def visit_Import(self, node):
            """Check for 'import bs4'."""
            for alias in node.names:
                if alias.name == "bs4" or alias.name.startswith("bs4."):
                    self.found_bs4 = True
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            """Check for 'from bs4 import ...'."""
            if node.module and (node.module == "bs4" or node.module.startswith("bs4.")):
                self.found_bs4 = True
            self.generic_visit(node)

        def visit_Call(self, node):
            """Check for BeautifulSoup() or make_soup() calls."""
            # Check direct call: BeautifulSoup(...)
            if isinstance(node.func, ast.Name):
                if node.func.id in ("BeautifulSoup", "make_soup", "NavigableString"):
                    self.found_bs4 = True

            # Check attribute call: bs4.BeautifulSoup(...)
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in ("BeautifulSoup", "make_soup", "NavigableString"):
                    self.found_bs4 = True
                # Check for BS4-specific method calls
                if node.func.attr in bs4_methods:
                    self.found_bs4 = True

            self.generic_visit(node)

        def visit_Attribute(self, node):
            """Check for BS4-specific attribute access."""
            if node.attr in bs4_attrs:
                # These attributes are fairly BS4-specific
                # (contents, descendants, next_sibling, etc.)
                self.found_bs4 = True
            self.generic_visit(node)

    visitor = BS4Visitor()
    visitor.visit(tree)
    return visitor.found_bs4


# NOTE: _check_bs4_import_and_constructor_ast was removed as dead code.
# The enhanced _check_soup_creation_with_html_ast now handles all detection cases
# with proper dataflow analysis and anti-spoofing via shadowing detection.


def _extract_assigned_names(target: ast.AST) -> set[str]:
    """Extract simple assigned names from an assignment target.

    Handles simple names, tuples, and lists:
    - `x = ...` -> {"x"}
    - `x, y = ...` -> {"x", "y"}
    - `[x, y] = ...` -> {"x", "y"}

    Args:
        target: An AST node representing the assignment target.

    Returns:
        Set of variable names being assigned.
    """
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names |= _extract_assigned_names(elt)
    return names


def _expr_uses_any_name(expr: ast.AST, names: set[str]) -> bool:
    """Return True if an expression references any name in `names`.

    Walks the expression tree to find any Name node that matches.

    Args:
        expr: AST expression to check.
        names: Set of variable names to look for.

    Returns:
        True if any name in `names` appears in the expression.
    """
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in names:
            return True
    return False


def _collect_html_derived_names(tree: ast.AST) -> set[str]:
    """Collect variables that are (syntactically) derived from the injected `HTML`.

    This is a lightweight, conservative dataflow pass used only for BS4-usage
    detection. It allows patterns like:

        doc = HTML.replace(...)
        soup = BeautifulSoup(doc, ...)

    without allowing trivial bypasses like BeautifulSoup("").

    The analysis is iterative: if `doc = HTML.replace(...)` and `doc2 = doc + ...`,
    both `doc` and `doc2` are considered HTML-derived.

    Args:
        tree: Parsed AST of the code.

    Returns:
        Set of variable names derived from HTML (always includes "HTML" itself).
    """
    derived: set[str] = {"HTML"}
    assigns: list[tuple[set[str], ast.AST | None]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                assigns.append((_extract_assigned_names(t), node.value))
        elif isinstance(node, ast.AnnAssign):
            assigns.append((_extract_assigned_names(node.target), node.value))

    # Fixed-point iteration: keep adding derived names until no changes
    changed = True
    while changed:
        changed = False
        for targets, value in assigns:
            if value is None:
                continue
            if _expr_uses_any_name(value, derived):
                for name in targets:
                    if name and name not in derived:
                        derived.add(name)
                        changed = True

    return derived


def _collect_shadowed_callable_names(tree: ast.AST, target_names: set[str]) -> set[str]:
    """Collect names that shadow injected helper callables.

    We treat a name as shadowed if the user code binds it to something that is
    unlikely to be the injected/global BS4 helper. This prevents trivial spoofing
    such as redefining BeautifulSoup or importing it from a non-bs4 module.

    Shadowing mechanisms detected:
    - Function definitions: `def BeautifulSoup(...):`
    - Class definitions: `class BeautifulSoup:`
    - Assignments: `BeautifulSoup = ...`
    - Non-bs4 imports: `from fake_lib import BeautifulSoup`

    Args:
        tree: Parsed AST of the code.
        target_names: Names to check for shadowing (e.g., {"BeautifulSoup", "make_soup"}).

    Returns:
        Set of names from target_names that are shadowed in the code.
    """
    shadowed: set[str] = set()

    class ShadowVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            if node.name in target_names:
                shadowed.add(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            if node.name in target_names:
                shadowed.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            if node.name in target_names:
                shadowed.add(node.name)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> Any:
            for t in node.targets:
                for name in _extract_assigned_names(t):
                    if name in target_names:
                        shadowed.add(name)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
            for name in _extract_assigned_names(node.target):
                if name in target_names:
                    shadowed.add(name)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> Any:
            for name in _extract_assigned_names(node.target):
                if name in target_names:
                    shadowed.add(name)
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> Any:
            # `import X as BeautifulSoup` shadows the name.
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[-1]
                if bound in target_names:
                    shadowed.add(bound)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
            module = node.module or ""
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound not in target_names:
                    continue

                # `from bs4 import BeautifulSoup` is legitimate BS4 usage, so do NOT shadow.
                if module == "bs4" or module.startswith("bs4."):
                    continue

                shadowed.add(bound)

            self.generic_visit(node)

    ShadowVisitor().visit(tree)
    return shadowed


def _collect_bs4_module_aliases(tree: ast.AST) -> set[str]:
    """Collect imported module names that refer to the `bs4` package.

    Detects patterns like:
    - `import bs4` -> {"bs4"}
    - `import bs4 as soup_lib` -> {"soup_lib"}
    - `import bs4.element` -> {"bs4"}

    Args:
        tree: Parsed AST of the code.

    Returns:
        Set of module aliases that refer to bs4.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "bs4" or alias.name.startswith("bs4."):
                    aliases.add(alias.asname or "bs4")
    return aliases


def _collect_bs4_constructor_aliases(tree: ast.AST) -> set[str]:
    """Collect aliases that refer to bs4.BeautifulSoup constructor.

    Detects patterns like:
    - `from bs4 import BeautifulSoup` -> {"BeautifulSoup"}
    - `from bs4 import BeautifulSoup as BS` -> {"BS"}

    Args:
        tree: Parsed AST of the code.

    Returns:
        Set of names that refer to the BeautifulSoup constructor.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "bs4" or module.startswith("bs4."):
                for alias in node.names:
                    if alias.name == "BeautifulSoup":
                        aliases.add(alias.asname or "BeautifulSoup")
    return aliases


def check_bs4_usage(code_samples: list[str]) -> bool:
    """Check if any code sample uses BeautifulSoup in a meaningful way.

    This is used for the BS4-usage *penalty gradient* and should be:
    - Strict enough to prevent trivial spoofing (e.g. redefining BeautifulSoup)
    - Flexible enough to allow common preprocessing patterns (doc = HTML.replace(...))

    Recognized patterns:
    1. make_soup(...)  # uses injected HTML internally
    2. BeautifulSoup(<HTML-derived>, ...)  # includes markup= keyword
    3. bs4_alias.BeautifulSoup(<HTML-derived>, ...) when `import bs4 as bs4_alias`
    4. BS(<HTML-derived>, ...) when `from bs4 import BeautifulSoup as BS`

    Anti-spoofing: Names that are shadowed (redefined, assigned, or imported from
    non-bs4 modules) are not counted as BS4 usage.

    Args:
        code_samples: List of code strings executed during the episode.

    Returns:
        True if BS4 appears to be used in any sample.
    """
    if not code_samples:
        return True  # No code = no penalty (e.g., format errors)

    for code in code_samples:
        if _check_soup_creation_with_html_ast(code):
            return True

    return False


def compute_bs4_penalty(code_samples: list[str] | None) -> tuple[float, bool]:
    """Compute BS4 usage penalty.

    Args:
        code_samples: List of code strings, or None if not available.

    Returns:
        Tuple of (penalty_amount, bs4_used).
        penalty_amount is 0.0 if BS4 was used, BS4_USAGE_PENALTY otherwise.
    """
    if code_samples is None:
        return 0.0, True  # Unknown - no penalty

    bs4_used = check_bs4_usage(code_samples)
    if bs4_used:
        return 0.0, True
    else:
        return BS4_USAGE_PENALTY, False


def _check_soup_creation_with_html_ast(code: str) -> bool:
    """AST check for soup creation using the injected HTML variable.

    This is the primary BS4-usage detector. It should be:
    - Strict enough to prevent trivial spoofing (e.g. redefining BeautifulSoup)
    - Flexible enough to allow common preprocessing patterns

    Recognized patterns:
    1. make_soup(...)  # uses injected HTML internally
    2. BeautifulSoup(<HTML-derived>, ...)  # positional or markup= keyword
    3. bs4_alias.BeautifulSoup(<HTML-derived>, ...) when `import bs4 as bs4_alias`
    4. BS(<HTML-derived>, ...) when `from bs4 import BeautifulSoup as BS`

    HTML-derived variables include HTML itself and any variable assigned from
    an expression that references HTML (e.g., `doc = HTML.replace(...)`).

    Args:
        code: Python code to analyze.

    Returns:
        True if a BeautifulSoup constructor call appears to parse the injected HTML.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Light dataflow: allow intermediate variables derived from HTML.
    html_derived = _collect_html_derived_names(tree)

    # Guard against spoofing: treat these names as shadowed if user code binds them.
    shadowed = _collect_shadowed_callable_names(tree, {"BeautifulSoup", "make_soup"})

    # Support bs4.BeautifulSoup(...) when the bs4 module is imported under an alias.
    bs4_module_aliases = _collect_bs4_module_aliases(tree)

    # Support `from bs4 import BeautifulSoup as BS` constructor aliases.
    bs4_ctor_aliases = _collect_bs4_constructor_aliases(tree)

    # Check for shadowing of aliases too (e.g., BS = lambda *a: None)
    alias_shadowed = _collect_shadowed_callable_names(tree, bs4_ctor_aliases)

    # Names we will treat as legitimate constructor callables in this snippet.
    # Only include aliases that are not shadowed.
    ctor_names = bs4_ctor_aliases - alias_shadowed
    if "BeautifulSoup" not in shadowed:
        ctor_names.add("BeautifulSoup")

    def _markup_expr(node: ast.Call) -> ast.AST | None:
        """Extract the markup argument from a BeautifulSoup call."""
        if node.args:
            return node.args[0]
        for kw in node.keywords or []:
            if kw.arg == "markup":
                return kw.value
        return None

    def _is_html_derived(expr: ast.AST | None) -> bool:
        """Check if expression references an HTML-derived variable."""
        if expr is None:
            return False
        # Fast path: direct reference to a derived variable.
        if isinstance(expr, ast.Name) and expr.id in html_derived:
            return True
        # General path: any reference to a derived variable within an expression.
        return _expr_uses_any_name(expr, html_derived)

    class HTMLSoupVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, node: ast.Call) -> Any:
            if self.found:
                return

            # Helper: make_soup(...) always uses injected HTML internally.
            if isinstance(node.func, ast.Name) and node.func.id == "make_soup":
                if "make_soup" not in shadowed:
                    self.found = True
                    return

            # Direct constructor call: BeautifulSoup(...) or BS(...)
            if isinstance(node.func, ast.Name) and node.func.id in ctor_names:
                if node.func.id in shadowed:
                    return
                if _is_html_derived(_markup_expr(node)):
                    self.found = True
                    return

            # Module constructor call: bs4_alias.BeautifulSoup(...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "BeautifulSoup":
                if isinstance(node.func.value, ast.Name) and node.func.value.id in bs4_module_aliases:
                    if _is_html_derived(_markup_expr(node)):
                        self.found = True
                        return

            self.generic_visit(node)

    visitor = HTMLSoupVisitor()
    visitor.visit(tree)
    return visitor.found


def _check_selection_method_ast(code: str) -> bool:
    """AST check for BS4 selection methods (.find, .find_all, .select, .select_one).

    Args:
        code: Python code to analyze.

    Returns:
        True if any BS4 selection method is called.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    selection_methods = {"find", "find_all", "select", "select_one"}

    class SelectionVisitor(ast.NodeVisitor):
        """Visitor to detect selection method calls."""

        def __init__(self):
            self.found = False

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in selection_methods:
                    self.found = True
            self.generic_visit(node)

    visitor = SelectionVisitor()
    visitor.visit(tree)
    return visitor.found


def _check_content_access_ast(code: str) -> bool:
    """AST check for content access (.text, .get_text(), .string).

    Args:
        code: Python code to analyze.

    Returns:
        True if any content access pattern is detected.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Content access attributes
    content_attrs = {"text", "string", "strings", "stripped_strings"}
    # Content access methods
    content_methods = {"get_text"}

    class ContentVisitor(ast.NodeVisitor):
        """Visitor to detect content access."""

        def __init__(self):
            self.found = False

        def visit_Attribute(self, node):
            if node.attr in content_attrs:
                self.found = True
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in content_methods:
                    self.found = True
            self.generic_visit(node)

    visitor = ContentVisitor()
    visitor.visit(tree)
    return visitor.found


def compute_process_partial_credit(
    code_samples: list[str],
    status: str,
    solvable: bool,
    run_python_calls: int | None = None,
    partial_credit_enabled: bool | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute tiered partial credit based on code structure.

    This rewards correct tool-use patterns even when the final answer is wrong,
    providing gradient signal for 0% models learning the basic action template.

    Uses AST analysis to detect actual code patterns, not just string matching.
    Enforces tier dependencies: later tiers require earlier tiers.

    Anti-hacking safeguards:
    - Requires run_python to be called (not just importing BS4)
    - Requires BeautifulSoup(HTML, ...) - must use injected variable
    - Blocks limit-claiming on solvable tasks
    - Capped at PROCESS_PARTIAL_CREDIT_CAP (0.30)

    Args:
        code_samples: List of code strings executed during the episode.
        status: The response status ("ok" or "limit").
        solvable: Whether the task is solvable.
        run_python_calls: Number of run_python calls made.
        partial_credit_enabled: Whether to enable process partial credit.
            Defaults to module constant PROCESS_PARTIAL_CREDIT_ENABLED.

    Returns:
        Tuple of (reward, breakdown_dict) for debugging/metrics.
    """
    # Use parameter if provided, otherwise fall back to module constant
    enabled = (
        partial_credit_enabled
        if partial_credit_enabled is not None
        else PROCESS_PARTIAL_CREDIT_ENABLED
    )
    if not enabled:
        return 0.0, {"disabled": True}

    # Gate 1: Don't reward limit-claiming on solvable tasks (anti-hacking)
    if status == "limit" and solvable:
        return 0.0, {"blocked": "limit_on_solvable"}

    # Gate 2: Don't reward on unsolvable tasks (process credit is for learning tool-use)
    # Process partial credit is designed to teach the basic action template on solvable tasks.
    # On unsolvable tasks, the model should recognize the limitation and claim "limit",
    # not attempt to solve with BS4. Rewarding BS4 patterns here would be counterproductive.
    if not solvable:
        return 0.0, {"blocked": "unsolvable_task"}

    # Gate 3: Require run_python to be called
    if run_python_calls is not None and run_python_calls <= 0:
        return 0.0, {"blocked": "no_run_python"}

    if not code_samples:
        return 0.0, {"no_code": True}

    breakdown: dict[str, Any] = {}
    reward = 0.0

    # Tier 1: BS4 import (reuse existing detection)
    bs4_imported = any(_check_bs4_usage_ast(c) for c in code_samples)
    if bs4_imported:
        reward += PROCESS_TIER_REWARDS["bs4_imported"]
        breakdown["bs4_imported"] = True

    # Tier 2: Soup creation with HTML variable (dependency: requires import)
    soup_created_with_html = any(_check_soup_creation_with_html_ast(c) for c in code_samples)
    if soup_created_with_html and bs4_imported:
        reward += PROCESS_TIER_REWARDS["soup_created_with_html"]
        breakdown["soup_created_with_html"] = True

        # Tier 3: Selection method (dependency: requires soup)
        selection_used = any(_check_selection_method_ast(c) for c in code_samples)
        if selection_used:
            reward += PROCESS_TIER_REWARDS["selection_method"]
            breakdown["selection_method"] = True

            # Tier 4: Content access (dependency: requires selection)
            content_accessed = any(_check_content_access_ast(c) for c in code_samples)
            if content_accessed:
                reward += PROCESS_TIER_REWARDS["content_access"]
                breakdown["content_access"] = True

    return min(reward, PROCESS_PARTIAL_CREDIT_CAP), breakdown


def _f1_multiset(a: list, b: list) -> float:
    """Compute F1 score over two lists treated as multisets.

    This is used for partial credit on array-type answers. Items are
    compared by their string representation for consistent comparison.

    Args:
        a: First list (e.g., model's answer).
        b: Second list (e.g., ground truth).

    Returns:
        F1 score in range [0.0, 1.0].
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    from collections import Counter

    ca = Counter(str(x) for x in a)
    cb = Counter(str(x) for x in b)
    common = sum((ca & cb).values())

    precision = common / max(len(a), 1)
    recall = common / max(len(b), 1)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _partial_credit(answer: Any, truth: Any, answer_schema: dict) -> float:
    """Compute partial credit score for structured outputs.

    For object types: percentage of correctly matched key-value pairs.
    For array types: F1 score over items (handles duplicates).

    Args:
        answer: The model's answer.
        truth: The expected ground truth.
        answer_schema: The JSON schema for the answer.

    Returns:
        Partial credit score in range [0.0, 1.0].
    """
    schema_type = answer_schema.get("type")

    # Object partial credit: count matching key-value pairs
    if schema_type == "object" and isinstance(answer, dict) and isinstance(truth, dict):
        if not truth:
            return 0.0
        correct = sum(1 for k in truth if k in answer and answer[k] == truth[k])
        return correct / len(truth)

    # Array partial credit: F1 score over items
    if schema_type == "array" and isinstance(answer, list) and isinstance(truth, list):
        return _f1_multiset(answer, truth)

    # No partial credit for other types
    return 0.0


def compute_reward(
    raw_output: str,
    task_info: dict,
    html: str | None = None,
    tool_call_count: float | None = None,
    tool_call_count_raw: int | None = None,
    run_python_calls: int | None = None,
    code_samples: list[str] | None = None,
    partial_credit_enabled: bool | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute reward for a model output.

    This is the main grading entry point that:
    1. Validates output format
    2. Checks for safety violations
    3. Enforces run_python usage (anti-reward-hacking)
    4. Computes correctness based on task type
    5. Applies efficiency multiplier based on tool call count
    6. Applies BS4 usage penalty if solution doesn't use BeautifulSoup

    Args:
        raw_output: The raw string output from the model.
        task_info: The task info dictionary containing ground_truth, etc.
        html: The original HTML (for safety checking and evidence verification).
        tool_call_count: Weighted tool call count for soft efficiency shaping.
            Different tools can have different weights (e.g., navigate=0.2).
        tool_call_count_raw: Raw (unweighted) tool call count for hard caps.
            Used to match prompt wording about "10+ calls = zero reward".
            If not provided, falls back to tool_call_count for hard cap.
        run_python_calls: Number of run_python tool calls made. If provided and <= 0,
            returns zero reward (anti-reward-hacking: prevents guessing without parsing).
        code_samples: List of code strings executed during the episode. If provided,
            applies BS4 usage penalty for solutions that don't use BeautifulSoup.
        partial_credit_enabled: Whether to enable process partial credit for 0% models.
            Defaults to module constant PROCESS_PARTIAL_CREDIT_ENABLED.

    Returns:
        Tuple of (reward, metrics). Metrics include detailed breakdown of
        what passed/failed for analysis.
    """
    metrics: dict[str, Any] = {
        "format_ok": False,
        "schema_ok": False,
        "safety_ok": False,
        "correct": False,
        "status": None,
        "errors": [],
        "warnings": [],
        "tool_calls": tool_call_count,  # Weighted count for efficiency
        "tool_calls_raw": tool_call_count_raw,  # Raw count for hard caps
        "run_python_calls": run_python_calls,
        "efficiency_multiplier": None,
        "bs4_used": None,
        "bs4_penalty": None,
    }

    # Step 1: Validate output format
    output, validation_errors = validate_output(raw_output, task_info)

    # Fatal error: JSON couldn't be parsed at all
    if output is None:
        metrics["errors"].extend(validation_errors)
        return REWARD_FORMAT_ERROR, metrics

    # Step 2: Check safety FIRST (before any partial credit is awarded)
    # This prevents awarding reward for outputs containing forbidden values
    safety_info = task_info.get("safety_info", {})
    forbidden_patterns = safety_info.get("forbidden_patterns", [])
    forbidden_values = safety_info.get("forbidden_values", [])

    # Extract forbidden values from HTML if provided
    if html:
        from bs4_env.grading.safety import extract_forbidden_values_from_html

        forbidden_values = list(forbidden_values) + extract_forbidden_values_from_html(html)

    # Check full output including limit.evidence, not just answer
    safety_violations = check_safety(
        output,
        forbidden_patterns=forbidden_patterns,
        forbidden_values=forbidden_values,
    )

    if safety_violations:
        metrics["errors"].extend(safety_violations)
        metrics["safety_ok"] = False
        return REWARD_SAFETY_VIOLATION, metrics

    metrics["safety_ok"] = True

    # Step 3: Handle schema validation errors
    # Non-fatal errors: JSON parsed but schema validation failed
    # Still allow process partial credit for models learning BS4 patterns
    if validation_errors:
        metrics["errors"].extend(validation_errors)
        metrics["format_ok"] = False
        metrics["schema_ok"] = False

        # Compute process partial credit even on schema errors
        # This helps models that are learning BS4 patterns but haven't
        # mastered the output format yet
        solvable = task_info.get("solvable", True)
        status = output.get("status", "unknown")
        if code_samples is not None:
            process_reward, process_breakdown = compute_process_partial_credit(
                code_samples=code_samples,
                status=status,
                solvable=solvable,
                run_python_calls=run_python_calls,
                partial_credit_enabled=partial_credit_enabled,
            )
            if process_reward > 0:
                # Apply efficiency enforcement to prevent reward-hacking via schema errors
                # Without this, models could farm process credit with unlimited tool calls
                # by producing schema-invalid JSON while using correct BS4 patterns.
                max_tool_calls = get_max_tool_calls(task_info)
                metrics["max_tool_calls"] = max_tool_calls

                # Use raw count for hard cap, fall back to weighted if not provided
                hard_cap_count = (
                    tool_call_count_raw if tool_call_count_raw is not None else tool_call_count
                )

                # Hard cap enforcement
                if hard_cap_count is not None and hard_cap_count > max_tool_calls:
                    metrics["efficiency_multiplier"] = 0.0
                    metrics["errors"].append(
                        f"Exceeded max tool calls ({hard_cap_count} > {max_tool_calls})"
                    )
                    return 0.0, metrics

                # Soft efficiency multiplier
                if tool_call_count is not None:
                    efficiency = compute_efficiency_multiplier(tool_call_count, max_tool_calls)
                    process_reward *= efficiency
                    metrics["efficiency_multiplier"] = efficiency

                metrics["process_partial_credit"] = process_breakdown
                metrics["partial_credit_source"] = "process_on_schema_error"
                return process_reward, metrics

        return REWARD_FORMAT_ERROR, metrics

    metrics["format_ok"] = True
    metrics["schema_ok"] = True
    metrics["status"] = output.get("status")

    # Step 4: Enforce run_python usage (anti-reward-hacking)
    # If run_python_calls tracking is enabled and model didn't use run_python,
    # refuse to award any positive reward. HTML is hidden from prompt, so the only
    # legitimate way to solve tasks is by parsing with run_python.
    if run_python_calls is not None and run_python_calls <= 0:
        metrics["errors"].append(
            "No run_python tool calls were made; refusing to award non-zero reward. "
            "Use run_python to parse the HTML and extract data."
        )
        metrics["correct"] = False
        return REWARD_WRONG, metrics

    # Step 5: Compute correctness based on status
    status = output.get("status")
    solvable = task_info.get("solvable", True)

    if status == "ok":
        base_reward, metrics = _grade_ok_response(output, task_info, metrics)
    elif status == "limit":
        base_reward, metrics = _grade_limit_response(output, task_info, html, solvable, metrics)
    else:
        metrics["errors"].append(f"Unknown status: {status}")
        return REWARD_FORMAT_ERROR, metrics

    # Step 6: Apply process partial credit (0% model bootstrapping)
    # Compute process credit for wrong/partial answers and take the maximum.
    # This avoids incentive inversion where getting 50% of keys correct (0.05)
    # would be worse than getting 0% correct with good BS4 code (0.30).
    if base_reward < REWARD_CORRECT and code_samples is not None:
        process_reward, process_breakdown = compute_process_partial_credit(
            code_samples=code_samples,
            status=status,
            solvable=solvable,
            run_python_calls=run_python_calls,
            partial_credit_enabled=partial_credit_enabled,
        )
        if process_reward > base_reward:
            metrics["extraction_partial_credit"] = base_reward  # Record original
            base_reward = process_reward
            metrics["process_partial_credit"] = process_breakdown
            metrics["partial_credit_source"] = "process"

    # Step 7: Apply efficiency multiplier (only for positive rewards on solvable tasks)
    # NOTE: We don't apply efficiency penalty to limit responses because:
    # 1. Models legitimately need to explore before recognizing a limitation
    # 2. The base limit reward (0.5) is already lower than extraction reward (1.0)
    #
    # Tool count semantics:
    # - tool_call_count_raw: used for hard caps (matches prompt "10+ calls = zero")
    # - tool_call_count: used for soft efficiency gradient (weighted by tool type)
    final_reward = base_reward
    is_limit_response = status == "limit"

    # Use raw count for hard cap, fall back to weighted if not provided
    hard_cap_count = tool_call_count_raw if tool_call_count_raw is not None else tool_call_count

    # Get task-specific max tool calls (allows archetypes to override default)
    max_tool_calls = get_max_tool_calls(task_info)
    metrics["max_tool_calls"] = max_tool_calls

    # Hard cap check: applies if ANY count is provided (raw or weighted)
    # This ensures hard cap is enforced even if only raw count is tracked
    if hard_cap_count is not None and base_reward > 0 and not is_limit_response:
        if hard_cap_count > max_tool_calls:
            metrics["efficiency_multiplier"] = 0.0
            metrics["errors"].append(
                f"Exceeded max tool calls ({hard_cap_count} > {max_tool_calls})"
            )
            metrics["correct"] = False
            final_reward = 0.0

    # Soft efficiency gradient: only if weighted count is provided
    if tool_call_count is not None and final_reward > 0 and not is_limit_response:
        efficiency = compute_efficiency_multiplier(tool_call_count, max_tool_calls)
        metrics["efficiency_multiplier"] = efficiency
        final_reward = base_reward * efficiency
    elif is_limit_response and tool_call_count is not None:
        # Still record efficiency for limit responses but don't penalize
        # (useful for analysis - did model recognize limit quickly or slowly?)
        efficiency = compute_efficiency_multiplier(tool_call_count, max_tool_calls)
        metrics["efficiency_multiplier"] = efficiency
        metrics["limit_efficiency_exemption"] = True

    # Step 8: Apply BS4 usage penalty (only for positive rewards on solvable tasks)
    # This creates a gradient toward BS4 usage without rejecting alternatives
    # NOTE: Skip BS4 penalty for process partial credit (already rewarding BS4 usage)
    is_process_credit = metrics.get("partial_credit_source") == "process"
    if code_samples is not None and final_reward > 0 and solvable and not is_process_credit:
        bs4_penalty, bs4_used = compute_bs4_penalty(code_samples)
        metrics["bs4_used"] = bs4_used
        metrics["bs4_penalty"] = bs4_penalty

        if not bs4_used:
            final_reward = max(0.0, final_reward - bs4_penalty)
            metrics["warnings"].append(
                f"BS4 not detected in solution - penalty of {bs4_penalty} applied"
            )

    return final_reward, metrics


def _is_price_string(value: Any) -> bool:
    """Check if a value looks like a price string.

    Detects patterns like "$99.99", "99.99", "$1,234.56", etc.
    """
    if not isinstance(value, str):
        return False
    # Match common price patterns: optional currency, digits with optional commas, decimal
    return bool(re.match(r"^[$£€]?\s*[\d,]+\.?\d*$", value.strip()))


def _apply_type_coercion(
    answer: Any,
    ground_truth: Any,
    answer_schema: dict,
) -> tuple[Any, Any]:
    """Apply type coercion to bridge format differences.

    Coerces answer to match expected types when semantically equivalent:
    - String "4" -> int 4 when schema expects integer
    - Aliased object keys -> canonical keys when schema has properties
    - Price strings normalized to canonical format

    Args:
        answer: The model's answer.
        ground_truth: The expected ground truth.
        answer_schema: The JSON schema for the answer.

    Returns:
        Tuple of (coerced_answer, coerced_ground_truth).
    """
    coerced_answer = answer
    coerced_truth = ground_truth

    schema_type = answer_schema.get("type")

    # Integer coercion: "4" -> 4
    if schema_type == "integer":
        try:
            coerced_answer = coerce_integer(answer)
        except ValueError:
            pass  # Leave as-is, will fail schema validation
        try:
            coerced_truth = coerce_integer(ground_truth)
        except ValueError:
            pass

    # Object key aliasing
    elif schema_type == "object" and isinstance(answer, dict):
        try:
            coerced_answer = normalize_object_keys(answer)
        except ValueError:
            pass  # Collision - leave as-is
        if isinstance(ground_truth, dict):
            try:
                coerced_truth = normalize_object_keys(ground_truth)
            except ValueError:
                pass

    # Price normalization: "$99.9" -> "$99.90", "99.99" -> "$99.99"
    # Apply when ground truth looks like a price
    elif _is_price_string(ground_truth):
        try:
            coerced_answer = normalize_price(answer)
            coerced_truth = normalize_price(ground_truth)
        except ValueError:
            pass

    # Handle nested prices in objects
    if isinstance(coerced_answer, dict) and isinstance(coerced_truth, dict):
        coerced_answer = _normalize_object_prices(coerced_answer, coerced_truth)
        coerced_truth = _normalize_object_prices(coerced_truth, coerced_truth)

    return coerced_answer, coerced_truth


def _normalize_object_prices(obj: dict, reference: dict) -> dict:
    """Normalize price-like values in an object based on reference.

    For each key in reference that has a price-like value, normalize
    the corresponding value in obj.

    Args:
        obj: Object to normalize.
        reference: Reference object to detect which keys are prices.

    Returns:
        Object with price values normalized.
    """
    result = dict(obj)
    for key, ref_value in reference.items():
        if key in result and _is_price_string(ref_value):
            try:
                result[key] = normalize_price(result[key])
            except ValueError:
                pass  # Leave as-is
    return result


def _grade_ok_response(
    output: dict,
    task_info: dict,
    metrics: dict,
) -> tuple[float, dict]:
    """Grade a response with status='ok'.

    Applies format tolerance (type coercion, key aliasing, price normalization)
    before comparing answer to ground truth. This rewards correct semantic
    answers even when formatting differs slightly.

    Args:
        output: The parsed output dictionary.
        task_info: The task info dictionary.
        metrics: The metrics dictionary to update.

    Returns:
        Tuple of (reward, metrics).
    """
    solvable = task_info.get("solvable", True)

    # If task is not solvable, claiming "ok" is wrong
    if not solvable:
        metrics["errors"].append("Claimed 'ok' but task is not solvable with BS4")
        metrics["correct"] = False
        return REWARD_WRONG, metrics

    # Get answer and ground truth
    answer = output.get("answer")
    ground_truth_raw = task_info.get("ground_truth")
    # ground_truth is JSON-serialized in the dataset to ensure consistent types
    # across archetypes (preventing PyArrow errors). Try to parse as JSON,
    # but fall back to raw value for backward compatibility (tests, direct calls).
    if isinstance(ground_truth_raw, str):
        try:
            ground_truth = json.loads(ground_truth_raw)
        except json.JSONDecodeError:
            # Not valid JSON, use as-is (backward compat for tests)
            ground_truth = ground_truth_raw
    else:
        ground_truth = ground_truth_raw
    answer_schema = task_info.get("answer_schema", {})
    normalization = task_info.get("normalization", {})

    # Apply type coercion to bridge format differences
    coerced_answer, coerced_truth = _apply_type_coercion(answer, ground_truth, answer_schema)

    # Compare with normalization
    if values_equal(coerced_answer, coerced_truth, normalization):
        metrics["correct"] = True
        # Note if coercion was applied
        if coerced_answer != answer or coerced_truth != ground_truth:
            metrics["coercion_applied"] = True
        return REWARD_CORRECT, metrics
    else:
        metrics["correct"] = False
        # Add helpful debug info
        normalized_answer = normalize_value(coerced_answer, normalization)
        normalized_truth = normalize_value(coerced_truth, normalization)
        metrics["debug"] = {
            "answer_normalized": str(normalized_answer)[:200],
            "truth_normalized": str(normalized_truth)[:200],
            "answer_raw": str(answer)[:100],
            "truth_raw": str(ground_truth)[:100],
        }

        # Check for partial credit on structured outputs (objects, arrays)
        partial = _partial_credit(coerced_answer, coerced_truth, answer_schema)
        if partial > 0:
            metrics["partial_credit"] = partial
            return REWARD_PARTIAL_MAX * partial, metrics

        return REWARD_WRONG, metrics


def _grade_limit_response(
    output: dict,
    task_info: dict,
    html: str | None,
    solvable: bool,
    metrics: dict,
) -> tuple[float, dict]:
    """Grade a response with status='limit'.

    This implements the anti-hacking rules:
    - If solvable=True, claiming limit is WRONG (reward 0.0)
    - If solvable=False, must provide valid reason AND evidence

    Args:
        output: The parsed output dictionary.
        task_info: The task info dictionary.
        html: The original HTML for evidence verification.
        solvable: Whether the task is actually solvable.
        metrics: The metrics dictionary to update.

    Returns:
        Tuple of (reward, metrics).
    """
    # Anti-hacking: can't claim limit on solvable tasks
    if solvable:
        metrics["errors"].append(
            "Claimed 'limit' but task IS solvable - this appears to be reward hacking"
        )
        metrics["correct"] = False
        return REWARD_WRONG, metrics

    # Extract limit info
    limit = output.get("limit", {})
    reason = limit.get("reason", "")
    evidence = limit.get("evidence", "")

    # Validate reason
    limit_info = task_info.get("limit_info", {})
    allowed_reasons = limit_info.get("allowed_reasons", [])

    if allowed_reasons and reason not in allowed_reasons:
        metrics["errors"].append(f"Invalid reason '{reason}'. Allowed: {allowed_reasons}")
        metrics["correct"] = False
        return REWARD_WRONG, metrics

    # Validate evidence
    if not evidence:
        metrics["errors"].append("No evidence provided for limitation claim")
        metrics["correct"] = False
        return REWARD_WRONG, metrics

    evidence_valid = _verify_evidence(evidence, html, limit_info)
    if not evidence_valid:
        metrics["errors"].append(
            f"Evidence '{evidence[:50]}...' not found in HTML or doesn't match patterns"
        )
        metrics["correct"] = False
        return REWARD_WRONG, metrics

    # All checks passed
    metrics["correct"] = True
    return REWARD_CORRECT_LIMIT, metrics


def _verify_evidence(
    evidence: str,
    html: str | None,
    limit_info: dict,
) -> bool:
    """Verify that evidence is valid.

    Evidence must either:
    1. Be a literal substring found in the HTML, OR
    2. Match one of the evidence patterns

    Args:
        evidence: The evidence string from the model.
        html: The original HTML.
        limit_info: The limit info from task.

    Returns:
        True if evidence is valid.
    """
    # Check literal substring (primary method)
    if html and evidence in html:
        return True

    # Check evidence patterns (secondary method)
    evidence_patterns = limit_info.get("evidence_patterns", [])
    for pattern in evidence_patterns:
        try:
            if re.search(pattern, evidence):
                # Pattern matches evidence, now check if something matching
                # that pattern exists in HTML
                if html and re.search(pattern, html):
                    return True
        except re.error:
            pass

    return False


def explain_reward(
    raw_output: str,
    task_info: dict,
    html: str | None = None,
) -> str:
    """Generate a human-readable explanation of the reward.

    Useful for debugging and understanding grading decisions.

    Args:
        raw_output: The raw model output.
        task_info: The task info dictionary.
        html: The original HTML.

    Returns:
        Human-readable explanation string.
    """
    reward, metrics = compute_reward(raw_output, task_info, html)

    lines = [f"Reward: {reward}"]

    if metrics.get("format_ok"):
        lines.append("Format: OK")
    else:
        lines.append("Format: FAILED")

    if metrics.get("schema_ok"):
        lines.append("Schema: OK")
    else:
        lines.append("Schema: FAILED")

    if metrics.get("safety_ok"):
        lines.append("Safety: OK")
    else:
        lines.append("Safety: VIOLATION")

    status = metrics.get("status")
    if status:
        lines.append(f"Status: {status}")

    if metrics.get("correct"):
        lines.append("Correctness: CORRECT")
    else:
        lines.append("Correctness: WRONG")

    if metrics.get("errors"):
        lines.append("Errors:")
        for error in metrics["errors"]:
            lines.append(f"  - {error}")

    if metrics.get("warnings"):
        lines.append("Warnings:")
        for warning in metrics["warnings"]:
            lines.append(f"  - {warning}")

    if metrics.get("debug"):
        lines.append("Debug info:")
        for key, value in metrics["debug"].items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)
