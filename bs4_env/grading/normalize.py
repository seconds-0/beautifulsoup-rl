from __future__ import annotations

"""Deterministic normalization for BeautifulSoup RL environment outputs.

This module provides functions to normalize model outputs before comparison
with ground truth. Normalization must be deterministic and consistent.

IMPORTANT: Be conservative with normalization. Over-normalization can enable
reward hacking. Only add normalization rules when demonstrably necessary.
"""

import re
import unicodedata
from typing import Any


def normalize_string(
    s: str,
    strip: bool = True,
    collapse_whitespace: bool = True,
    unicode_nfc: bool = True,
    lowercase: bool = False,
) -> str:
    """Normalize a string for comparison.

    Args:
        s: The string to normalize.
        strip: Whether to strip leading/trailing whitespace.
        collapse_whitespace: Whether to collapse multiple whitespace to single space.
        unicode_nfc: Whether to apply Unicode NFC normalization.
        lowercase: Whether to convert to lowercase (use sparingly!).

    Returns:
        The normalized string.
    """
    if not isinstance(s, str):
        s = str(s)

    if unicode_nfc:
        s = unicodedata.normalize("NFC", s)

    if strip:
        s = s.strip()

    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s)

    if lowercase:
        s = s.lower()

    return s


def normalize_list(
    lst: list,
    sort: bool = False,
    normalize_items: bool = True,
    normalization_config: dict | None = None,
) -> list:
    """Normalize a list for comparison.

    Args:
        lst: The list to normalize.
        sort: Whether to sort the list (for order-independent comparison).
        normalize_items: Whether to recursively normalize list items.
        normalization_config: Config dict for string normalization.

    Returns:
        The normalized list.
    """
    if normalization_config is None:
        normalization_config = {}

    result = []
    for item in lst:
        if normalize_items:
            item = normalize_value(item, normalization_config)
        result.append(item)

    if sort:
        # Convert to comparable form for sorting
        result = sorted(result, key=_sort_key)

    return result


def normalize_dict(
    d: dict,
    sort_keys: bool = True,
    normalize_values: bool = True,
    normalization_config: dict | None = None,
) -> dict:
    """Normalize a dictionary for comparison.

    Args:
        d: The dictionary to normalize.
        sort_keys: Whether to sort keys (mainly affects iteration order).
        normalize_values: Whether to recursively normalize values.
        normalization_config: Config dict for value normalization.

    Returns:
        The normalized dictionary.
    """
    if normalization_config is None:
        normalization_config = {}

    result = {}
    keys = sorted(d.keys()) if sort_keys else d.keys()

    for key in keys:
        value = d[key]
        if normalize_values:
            value = normalize_value(value, normalization_config)
        # Also normalize string keys
        if isinstance(key, str):
            key = normalize_string(
                key,
                strip=normalization_config.get("strip_whitespace", True),
                collapse_whitespace=normalization_config.get("collapse_whitespace", True),
                unicode_nfc=normalization_config.get("unicode_nfc", True),
            )
        result[key] = value

    return result


def normalize_value(value: Any, normalization_config: dict | None = None) -> Any:
    """Normalize any value based on its type.

    Args:
        value: The value to normalize.
        normalization_config: Configuration for normalization rules.

    Returns:
        The normalized value.
    """
    if normalization_config is None:
        normalization_config = {}

    if value is None:
        return None

    if isinstance(value, str):
        return normalize_string(
            value,
            strip=normalization_config.get("strip_whitespace", True),
            collapse_whitespace=normalization_config.get("collapse_whitespace", True),
            unicode_nfc=normalization_config.get("unicode_nfc", True),
            lowercase=normalization_config.get("lowercase", False),
        )

    if isinstance(value, list):
        return normalize_list(
            value,
            sort=normalization_config.get("sort_lists", False),
            normalize_items=True,
            normalization_config=normalization_config,
        )

    if isinstance(value, dict):
        return normalize_dict(
            value,
            sort_keys=normalization_config.get("sort_dict_keys", True),
            normalize_values=True,
            normalization_config=normalization_config,
        )

    # Numbers and booleans pass through unchanged
    if isinstance(value, (int, float, bool)):
        return value

    # Unknown types convert to string
    return normalize_string(str(value), **normalization_config)


def _sort_key(value: Any) -> tuple:
    """Create a sortable key for any value type.

    Returns a tuple of (type_rank, value) for consistent sorting.
    """
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, value)
    if isinstance(value, (int, float)):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    if isinstance(value, list):
        return (4, str(value))
    if isinstance(value, dict):
        return (5, str(sorted(value.items())))
    return (6, str(value))


def values_equal(
    value1: Any,
    value2: Any,
    normalization_config: dict | None = None,
) -> bool:
    """Check if two values are equal after normalization.

    Args:
        value1: First value.
        value2: Second value.
        normalization_config: Configuration for normalization.

    Returns:
        True if values are equal after normalization.
    """
    norm1 = normalize_value(value1, normalization_config)
    norm2 = normalize_value(value2, normalization_config)
    return norm1 == norm2


def normalize_html_entities(s: str) -> str:
    """Normalize HTML entities to their Unicode equivalents.

    Args:
        s: String potentially containing HTML entities.

    Returns:
        String with entities decoded.
    """
    import html

    return html.unescape(s)


def normalize_url(url: str) -> str:
    """Normalize a URL for comparison.

    Args:
        url: The URL to normalize.

    Returns:
        Normalized URL.
    """
    url = url.strip()

    # Remove trailing slash
    if url.endswith("/") and url.count("/") > 3:
        url = url.rstrip("/")

    # Lowercase scheme and host
    if "://" in url:
        scheme_end = url.find("://") + 3
        path_start = url.find("/", scheme_end)
        if path_start > 0:
            url = url[:path_start].lower() + url[path_start:]
        else:
            url = url.lower()

    return url
