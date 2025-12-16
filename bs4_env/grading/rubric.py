from __future__ import annotations

"""Reward computation for BeautifulSoup RL environment.

This module implements the rubric that converts model outputs into rewards.
The reward function must be deterministic and follow the anti-hacking rules.
"""

import re
from typing import Any

from bs4_env.grading.normalize import normalize_value, values_equal
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


def compute_reward(
    raw_output: str,
    task_info: dict,
    html: str | None = None,
    tool_call_count: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute reward for a model output.

    This is the main grading entry point that:
    1. Validates output format
    2. Checks for safety violations
    3. Computes correctness based on task type
    4. Applies efficiency multiplier based on tool call count

    Args:
        raw_output: The raw string output from the model.
        task_info: The task info dictionary containing ground_truth, etc.
        html: The original HTML (for safety checking and evidence verification).
        tool_call_count: Number of tool calls made. If provided, applies efficiency
            multiplier to reward (fewer calls = higher reward).

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

    safety_violations = check_safety(
        output.get("answer"),
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

    # Step 4: Apply efficiency multiplier (only for positive rewards)
    if tool_call_count is not None and base_reward > 0:
        efficiency = compute_efficiency_multiplier(tool_call_count)
        metrics["efficiency_multiplier"] = efficiency
        final_reward = base_reward * efficiency

        # If efficiency multiplier is 0 (too many calls), mark as failure
        if efficiency == 0.0:
            metrics["errors"].append(
                f"Exceeded max tool calls ({tool_call_count} > {MAX_TOOL_CALLS})"
            )
            metrics["correct"] = False

        return final_reward, metrics

    return base_reward, metrics


def _grade_ok_response(
    output: dict,
    task_info: dict,
    metrics: dict,
) -> tuple[float, dict]:
    """Grade a response with status='ok'.

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

    # Compare answer with ground truth
    answer = output.get("answer")
    ground_truth = task_info.get("ground_truth")
    normalization = task_info.get("normalization", {})

    if values_equal(answer, ground_truth, normalization):
        metrics["correct"] = True
        return REWARD_CORRECT, metrics
    else:
        metrics["correct"] = False
        # Add helpful debug info
        normalized_answer = normalize_value(answer, normalization)
        normalized_truth = normalize_value(ground_truth, normalization)
        metrics["debug"] = {
            "answer_normalized": str(normalized_answer)[:200],
            "truth_normalized": str(normalized_truth)[:200],
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
        metrics["errors"].append(
            f"Invalid reason '{reason}'. Allowed: {allowed_reasons}"
        )
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
