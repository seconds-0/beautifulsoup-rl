from __future__ import annotations

"""Reward computation for BeautifulSoup RL environment.

This module implements the rubric that converts model outputs into rewards.
The reward function must be deterministic and follow the anti-hacking rules.
"""

import ast
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

# Efficiency settings
MAX_TOOL_CALLS = 10  # Hard cutoff - more than this is considered failure
EFFICIENCY_PENALTY_PER_CALL = 0.1  # -10% per additional call after first
EFFICIENCY_FLOOR = 0.2  # Minimum multiplier before hard cutoff

# BS4 usage settings - creates gradient toward BS4 usage without rejecting alternatives
BS4_USAGE_PENALTY = 0.15  # Penalty for not using BeautifulSoup


def compute_efficiency_multiplier(tool_call_count: int) -> float:
    """Compute efficiency multiplier based on tool call count.

    Rewards efficient solutions that solve in fewer calls.
    Creates gradient signal for RL to optimize efficiency, not just correctness.

    Args:
        tool_call_count: Number of tool calls made during the episode.

    Returns:
        Multiplier in range [0.0, 1.0]:
        - 1 call: 1.0 (full credit)
        - 2 calls: 0.9
        - 3 calls: 0.8
        - ...
        - 9-10 calls: 0.2 (floor)
        - 11+ calls: 0.0 (hard cutoff - treated as failure)
    """
    if tool_call_count > MAX_TOOL_CALLS:
        return 0.0  # Hard cutoff - brute force isn't skill
    if tool_call_count <= 1:
        return 1.0
    return max(EFFICIENCY_FLOOR, 1.0 - EFFICIENCY_PENALTY_PER_CALL * (tool_call_count - 1))


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


def check_bs4_usage(code_samples: list[str]) -> bool:
    """Check if any code sample uses BeautifulSoup.

    This creates a gradient toward BS4 usage without rejecting alternatives.
    Regex and string solutions still work but receive a penalty.

    Uses AST-based detection to identify actual BS4 API usage, ignoring
    mentions of "BeautifulSoup" in comments or strings. This prevents
    trivial bypasses like adding a comment mentioning BS4.

    The detection looks for:
    1. Import statements (import bs4, from bs4 import ...)
    2. Constructor calls (BeautifulSoup(), make_soup(), NavigableString())
    3. BS4-specific method calls (.find_all(), .select(), .get_text(), etc.)
    4. BS4-specific attribute access (.next_sibling, .contents, etc.)

    Args:
        code_samples: List of code strings executed during the episode.

    Returns:
        True if BS4 appears to be used in any sample.
    """
    if not code_samples:
        return True  # No code = no penalty (e.g., format errors)

    return any(_check_bs4_usage_ast(code) for code in code_samples)


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


def compute_reward(
    raw_output: str,
    task_info: dict,
    html: str | None = None,
    tool_call_count: int | None = None,
    code_samples: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute reward for a model output.

    This is the main grading entry point that:
    1. Validates output format
    2. Checks for safety violations
    3. Computes correctness based on task type
    4. Applies efficiency multiplier based on tool call count
    5. Applies BS4 usage penalty if solution doesn't use BeautifulSoup

    Args:
        raw_output: The raw string output from the model.
        task_info: The task info dictionary containing ground_truth, etc.
        html: The original HTML (for safety checking and evidence verification).
        tool_call_count: Number of tool calls made. If provided, applies efficiency
            multiplier to reward (fewer calls = higher reward).
        code_samples: List of code strings executed during the episode. If provided,
            applies BS4 usage penalty for solutions that don't use BeautifulSoup.

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
        "tool_calls": tool_call_count,
        "efficiency_multiplier": None,
        "bs4_used": None,
        "bs4_penalty": None,
    }

    # Step 1: Validate output format
    output, validation_errors = validate_output(raw_output, task_info)

    if output is None or validation_errors:
        metrics["errors"].extend(validation_errors)
        return REWARD_FORMAT_ERROR, metrics

    metrics["format_ok"] = True
    metrics["schema_ok"] = True
    metrics["status"] = output.get("status")

    # Step 2: Check safety
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

    # Step 3: Compute correctness based on status
    status = output.get("status")
    solvable = task_info.get("solvable", True)

    if status == "ok":
        base_reward, metrics = _grade_ok_response(output, task_info, metrics)
    elif status == "limit":
        base_reward, metrics = _grade_limit_response(output, task_info, html, solvable, metrics)
    else:
        metrics["errors"].append(f"Unknown status: {status}")
        return REWARD_FORMAT_ERROR, metrics

    # Step 4: Apply efficiency multiplier (only for positive rewards on solvable tasks)
    # NOTE: We don't apply efficiency penalty to limit responses because:
    # 1. Models legitimately need to explore before recognizing a limitation
    # 2. The base limit reward (0.5) is already lower than extraction reward (1.0)
    final_reward = base_reward
    is_limit_response = status == "limit"
    if tool_call_count is not None and base_reward > 0 and not is_limit_response:
        efficiency = compute_efficiency_multiplier(tool_call_count)
        metrics["efficiency_multiplier"] = efficiency
        final_reward = base_reward * efficiency

        # If efficiency multiplier is 0 (too many calls), mark as failure
        if efficiency == 0.0:
            metrics["errors"].append(
                f"Exceeded max tool calls ({tool_call_count} > {MAX_TOOL_CALLS})"
            )
            metrics["correct"] = False
    elif is_limit_response and tool_call_count is not None:
        # Still record efficiency for limit responses but don't penalize
        # (useful for analysis - did model recognize limit quickly or slowly?)
        efficiency = compute_efficiency_multiplier(tool_call_count)
        metrics["efficiency_multiplier"] = efficiency
        metrics["limit_efficiency_exemption"] = True

    # Step 5: Apply BS4 usage penalty (only for positive rewards on solvable tasks)
    # This creates a gradient toward BS4 usage without rejecting alternatives
    solvable = task_info.get("solvable", True)
    if code_samples is not None and final_reward > 0 and solvable:
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
    ground_truth = task_info.get("ground_truth")
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
