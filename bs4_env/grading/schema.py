from __future__ import annotations

"""JSON schema validation for BeautifulSoup RL environment outputs.

This module handles validation of model outputs against expected schemas.
"""

import json
from typing import Any

from jsonschema import Draft7Validator

# The top-level output schema that all model responses must follow
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ok", "limit"],
        },
        "answer": {
            # Answer can be any type - specific validation done per-task
        },
        "limit": {
            "type": ["object", "null"],
            "properties": {
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["reason", "evidence"],
        },
    },
    "required": ["status"],
    "additionalProperties": False,
}


def parse_json_output(raw_output: str) -> tuple[dict | None, str | None]:
    """Parse the raw model output as JSON.

    Attempts to extract JSON from the output, handling common issues like
    markdown code blocks or extra text around the JSON.

    Args:
        raw_output: The raw string output from the model.

    Returns:
        Tuple of (parsed_dict, error_message). If parsing succeeds,
        error_message is None. If parsing fails, parsed_dict is None.
    """
    output = raw_output.strip()

    # Try direct parsing first
    try:
        return json.loads(output), None
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip()), None
            except json.JSONDecodeError:
                pass

    # Try extracting from generic code block
    if "```" in output:
        start = output.find("```") + 3
        # Skip language identifier if present
        newline = output.find("\n", start)
        if newline > start:
            start = newline + 1
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip()), None
            except json.JSONDecodeError:
                pass

    # Try finding JSON object boundaries
    brace_start = output.find("{")
    brace_end = output.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(output[brace_start : brace_end + 1]), None
        except json.JSONDecodeError:
            pass

    return None, f"Failed to parse JSON from output: {output[:200]}..."


def validate_output_schema(output: dict) -> list[str]:
    """Validate output against the top-level output schema.

    Args:
        output: The parsed output dictionary.

    Returns:
        List of validation error messages. Empty if valid.
    """
    validator = Draft7Validator(OUTPUT_SCHEMA)
    errors = []

    for error in validator.iter_errors(output):
        path = ".".join(str(p) for p in error.path) if error.path else "root"
        errors.append(f"{path}: {error.message}")

    return errors


def validate_answer_schema(answer: Any, answer_schema: dict) -> list[str]:
    """Validate the answer field against task-specific schema.

    Args:
        answer: The answer value from the output.
        answer_schema: The JSON schema for the expected answer format.

    Returns:
        List of validation error messages. Empty if valid.
    """
    if not answer_schema:
        # No schema specified, accept anything
        return []

    validator = Draft7Validator(answer_schema)
    errors = []

    for error in validator.iter_errors(answer):
        path = ".".join(str(p) for p in error.path) if error.path else "answer"
        errors.append(f"{path}: {error.message}")

    return errors


def validate_output(
    raw_output: str,
    task_info: dict,
) -> tuple[dict | None, list[str]]:
    """Fully validate model output against schemas.

    This is the main validation entry point that:
    1. Parses the raw output as JSON
    2. Validates against the top-level output schema
    3. Validates status-specific requirements
    4. Validates the answer against task-specific schema

    Args:
        raw_output: The raw string output from the model.
        task_info: The task info dictionary containing answer_schema, etc.

    Returns:
        Tuple of (parsed_output, errors). If there are any errors,
        parsed_output may still be partially valid.
    """
    errors: list[str] = []

    # Step 1: Parse JSON
    output, parse_error = parse_json_output(raw_output)
    if parse_error:
        return None, [parse_error]

    assert output is not None

    # Step 2: Validate top-level schema
    schema_errors = validate_output_schema(output)
    errors.extend(schema_errors)

    if schema_errors:
        # If basic schema fails, don't continue
        return output, errors

    status = output.get("status")

    # Step 3: Validate status-specific requirements
    if status == "ok":
        # Must have answer, must not have limit
        if "answer" not in output:
            errors.append("status is 'ok' but 'answer' field is missing")
        if output.get("limit") is not None:
            errors.append("status is 'ok' but 'limit' field is present")

        # Validate answer schema
        if "answer" in output:
            answer_schema = task_info.get("answer_schema", {})
            answer_errors = validate_answer_schema(output["answer"], answer_schema)
            errors.extend(answer_errors)

    elif status == "limit":
        # Must have limit with reason and evidence
        limit = output.get("limit")
        if limit is None:
            errors.append("status is 'limit' but 'limit' field is missing or null")
        else:
            if not isinstance(limit, dict):
                errors.append("'limit' field must be an object")
            else:
                if "reason" not in limit:
                    errors.append("'limit.reason' is required when status is 'limit'")
                if "evidence" not in limit:
                    errors.append("'limit.evidence' is required when status is 'limit'")

    return output, errors


def get_output_template(status: str = "ok") -> dict:
    """Get a template for the expected output format.

    Useful for showing the model what format to use.

    Args:
        status: Either "ok" or "limit".

    Returns:
        A template dictionary.
    """
    if status == "ok":
        return {
            "status": "ok",
            "answer": "<your extracted answer here>",
        }
    else:
        return {
            "status": "limit",
            "answer": None,
            "limit": {
                "reason": "<one of the allowed reasons>",
                "evidence": "<literal substring from HTML proving the limitation>",
            },
        }
