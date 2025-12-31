from __future__ import annotations

"""Prompt formatting for BeautifulSoup RL environment.

This module formats task instances into prompts for the model.
CRITICAL: Never include ground truth or other hidden information in prompts.
"""

import json
from typing import Any

from bs4_env.config import TaskConstraints


def format_prompt(
    html: str,
    query: str,
    constraints: TaskConstraints,
    system_message: str | None = None,
) -> list[dict[str, str]]:
    """Format a task into a prompt (list of chat messages).

    Args:
        html: The HTML content to parse.
        query: The natural language extraction query.
        constraints: Task constraints (output schema, allowed reasons, etc.).
        system_message: Optional custom system message.

    Returns:
        List of chat message dictionaries with 'role' and 'content'.
    """
    messages = []

    # System message
    if system_message is None:
        system_message = DEFAULT_SYSTEM_MESSAGE
    messages.append({"role": "system", "content": system_message})

    # User message with task
    user_content = format_user_message(html, query, constraints)
    messages.append({"role": "user", "content": user_content})

    return messages


def format_user_message(
    html: str,
    query: str,
    constraints: TaskConstraints,
) -> str:
    """Format the user message containing the task.

    Args:
        html: The HTML content.
        query: The extraction query.
        constraints: Task constraints.

    Returns:
        Formatted user message string.
    """
    parts = []

    # Task description
    parts.append("## Task")
    parts.append(query)
    parts.append("")

    # Constraints
    parts.append("## Constraints")
    parts.append("")
    parts.append("### Output Format")
    parts.append("Your final response must be a JSON object with this structure:")
    parts.append("```json")
    parts.append(json.dumps(OUTPUT_FORMAT_EXAMPLE, indent=2))
    parts.append("```")
    parts.append("")

    if constraints.output_schema:
        parts.append("### Answer Schema")
        schema_explanation = explain_schema(constraints.output_schema)
        parts.append(schema_explanation)
        parts.append("")

    if constraints.allowed_limit_reasons:
        parts.append("### Limitation Handling")
        parts.append(
            "If this task cannot be solved with static HTML parsing, "
            'respond with `status: "limit"` and provide:'
        )
        parts.append(f"- `reason`: One of {constraints.allowed_limit_reasons}")
        parts.append("- `evidence`: A literal substring from the HTML proving the limitation")
        parts.append("")

    if constraints.safety_notes:
        parts.append("### Safety")
        for note in constraints.safety_notes:
            parts.append(f"- {note}")
        parts.append("")

    # Note: HTML is NOT included in prompt - must use run_python tool to access it
    parts.append("## Accessing the HTML")
    parts.append("The HTML content is available via the `HTML` variable in the Python sandbox.")
    parts.append("Use the `run_python` tool to write BeautifulSoup code that extracts the answer.")

    return "\n".join(parts)


DEFAULT_SYSTEM_MESSAGE = """You are an expert web scraping assistant using BeautifulSoup (bs4) in Python.

Your task is to extract specific information from HTML content using BeautifulSoup.

## IMPORTANT: You MUST use the tools

The HTML content is NOT visible in this conversation. You MUST call tools to access and parse the HTML.

## Available Tools

You have access to:

**`run_python`** - Executes Python code in a sandbox with:
- `HTML`: The HTML content to parse (as a string)
- `QUERY`: The extraction task description
- `CONSTRAINTS`: A dictionary with output requirements
- `make_soup(parser)`: Helper function to create a BeautifulSoup object

**`navigate`** - For multi-step tasks, follows links to load new HTML content. After navigating, the `HTML` variable in `run_python` will contain the new page.

BeautifulSoup and common parsers (html.parser, lxml, html5lib) are pre-installed.

## Workflow

**Single-step tasks:** Call `run_python` to extract data from the HTML.

**Multi-step tasks:** Use `navigate` to follow links to other pages, then call `run_python` on each page to extract data. Continue until you have all required information.

1. Call the appropriate tool(s) to access and parse the HTML
2. Iterate if needed until you have the correct result
3. Provide your final answer as a JSON object (no tool calls)

## Output Format

Your final response (when you stop calling tools) must be a valid JSON object:

For successful extraction:
```json
{"status": "ok", "answer": <your extracted data>}
```

If extraction is impossible with static HTML parsing:
```json
{"status": "limit", "answer": null, "limit": {"reason": "<why>", "evidence": "<substring from HTML>"}}
```

## Common BeautifulSoup Gotchas

- Use `class_` not `class` (reserved word)
- `.string` returns None if element has multiple children - use `.get_text()` instead
- Always check if `find()` returns None before accessing attributes
- Different parsers produce different results for malformed HTML"""


OUTPUT_FORMAT_EXAMPLE = {"status": "ok", "answer": "<your extracted data matching the schema>"}


def explain_schema(schema: dict) -> str:
    """Convert a JSON schema to a plain English explanation with example.

    This prevents models from confusing the schema structure with the answer format.
    """
    schema_type = schema.get("type")

    if schema_type == "string":
        return 'The `answer` field must be a **string**. Example: `"answer": "the extracted text"`'

    elif schema_type == "array":
        items = schema.get("items", {})
        items_type = items.get("type", "any")
        if items_type == "string":
            return 'The `answer` field must be an **array of strings**. Example: `"answer": ["item1", "item2", "item3"]`'
        elif items_type == "object":
            return 'The `answer` field must be an **array of objects**. Example: `"answer": [{"key": "value"}, {"key": "value2"}]`'
        elif items_type == "array":
            return 'The `answer` field must be an **array of arrays** (table rows). Example: `"answer": [["col1", "col2"], ["val1", "val2"]]`'
        else:
            return f"The `answer` field must be an **array** of {items_type} values."

    elif schema_type == "object":
        # For objects with defined properties, show the exact keys required
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if properties:
            keys_desc = []
            for key, prop in properties.items():
                prop_type = prop.get("type", "any")
                req_marker = " (required)" if key in required else ""
                keys_desc.append(f'"{key}": {prop_type}{req_marker}')
            keys_str = ", ".join(keys_desc)
            example_obj = {k: f"<{v.get('type', 'value')}>" for k, v in properties.items()}
            return f'The `answer` field must be an **object** with these exact keys: {{{keys_str}}}. Example: `"answer": {json.dumps(example_obj)}`'
        return 'The `answer` field must be an **object** (dictionary). Example: `"answer": {"key1": "value1", "key2": "value2"}`'

    elif schema_type == "number":
        return 'The `answer` field must be a **number**. Example: `"answer": 42.5`'

    elif schema_type == "integer":
        return 'The `answer` field must be an **integer**. Example: `"answer": 42`'

    elif schema_type == "boolean":
        return 'The `answer` field must be a **boolean**. Example: `"answer": true`'

    else:
        # Fallback: show schema but with explicit instruction
        return f"The `answer` field must match this JSON schema (NOT the schema itself, but a value that conforms to it):\n```json\n{json.dumps(schema, indent=2)}\n```"


def format_few_shot_examples(
    examples: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Format few-shot examples as message pairs.

    Args:
        examples: List of example dicts with 'html', 'query', 'answer'.

    Returns:
        List of message dicts alternating user/assistant.
    """
    messages = []

    for ex in examples:
        # User turn with simplified task
        user_content = f"Extract: {ex['query']}\n\nHTML:\n```html\n{ex['html']}\n```"
        messages.append({"role": "user", "content": user_content})

        # Assistant turn with answer
        answer_json = json.dumps({"status": "ok", "answer": ex["answer"]})
        messages.append({"role": "assistant", "content": answer_json})

    return messages


def truncate_html_for_display(html: str, max_chars: int = 1000) -> str:
    """Truncate HTML for display purposes.

    Args:
        html: The HTML content.
        max_chars: Maximum characters to show.

    Returns:
        Truncated HTML with indicator if truncated.
    """
    if len(html) <= max_chars:
        return html

    return html[:max_chars] + "\n... [truncated]"


def extract_final_answer(response: str) -> str | None:
    """Extract the final JSON answer from a model response.

    Handles cases where the model includes explanation text around the JSON.

    Args:
        response: The full model response.

    Returns:
        The JSON string, or None if not found.
    """
    response = response.strip()

    # If it's already valid JSON, return it
    try:
        json.loads(response)
        return response
    except json.JSONDecodeError:
        pass

    # Try to extract from code block
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    if "```" in response:
        start = response.find("```") + 3
        newline = response.find("\n", start)
        if newline > start:
            start = newline + 1
        end = response.find("```", start)
        if end > start:
            candidate = response[start:end].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    # Try finding JSON object boundaries
    brace_start = response.rfind("{")  # Last opening brace
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(response)):
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = response[brace_start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

    return None
