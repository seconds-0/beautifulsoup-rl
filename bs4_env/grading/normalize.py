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
        url = url[:path_start].lower() + url[path_start:] if path_start > 0 else url.lower()

    return url


# =============================================================================
# Type Coercion Functions
# =============================================================================
# These functions help bridge formatting differences between model outputs
# and ground truth without enabling reward hacking.


def coerce_integer(value: Any) -> int:
    """Coerce a value to integer if it's an unambiguous integer representation.

    Only accepts:
    - Actual integers (not bools)
    - Strings that are pure decimal integers (e.g., "4", "-123", "+5")

    Rejects:
    - Floats (even 4.0) to avoid ambiguity
    - Scientific notation ("4e0")
    - Strings with decimals ("4.0")
    - Booleans (True/False are technically ints in Python)

    Args:
        value: The value to coerce.

    Returns:
        Integer value.

    Raises:
        ValueError: If value cannot be unambiguously converted to int.
    """
    # Reject booleans explicitly (isinstance(True, int) is True in Python!)
    if isinstance(value, bool):
        raise ValueError(f"Cannot coerce boolean to integer: {value!r}")

    # Accept actual integers
    if isinstance(value, int):
        return value

    # Accept string representations of integers
    if isinstance(value, str):
        stripped = value.strip()
        # Match optional sign followed by digits only
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
        raise ValueError(f"String is not a pure integer: {value!r}")

    # Reject floats to avoid ambiguity (is 4.9 meant to be 4 or 5?)
    if isinstance(value, float):
        raise ValueError(f"Cannot coerce float to integer: {value!r}")

    raise ValueError(f"Cannot coerce {type(value).__name__} to integer: {value!r}")


# =============================================================================
# Key Aliasing for Object Schemas
# =============================================================================
# Fixed alias table - no fuzzy matching to prevent gaming.
# Only add aliases that are clearly the same semantic concept.

KEY_ALIASES: dict[str, str] = {
    # compare_products archetype
    "cheaper_product": "cheaper",
    "cheaperProduct": "cheaper",
    "cheaper_item": "cheaper",
    "cheapest": "cheaper",
    "cheapest_product": "cheaper",
    "price_diff": "price_difference",
    "priceDifference": "price_difference",
    "difference": "price_difference",
    "diff": "price_difference",
    # structured_output archetype
    "product_name": "name",
    "productName": "name",
    "item_name": "name",
    "title": "name",
    "item_price": "price",
    "product_price": "price",
    "cost": "price",
    "item_sku": "sku",
    "product_sku": "sku",
    "sku_number": "sku",
    "skuNumber": "sku",
    "item_url": "url",
    "product_url": "url",
    "link": "url",
    "productUrl": "url",
    # aggregation archetypes
    "min_price": "min",
    "minPrice": "min",
    "minimum": "min",
    "lowest": "min",
    "max_price": "max",
    "maxPrice": "max",
    "maximum": "max",
    "highest": "max",
    "total_count": "count",
    "totalCount": "count",
    "num_items": "count",
    "item_count": "count",
}


def normalize_object_keys(
    obj: dict,
    alias_table: dict[str, str] | None = None,
) -> dict:
    """Normalize object keys using an alias table.

    This applies key aliasing to help match model outputs that use
    different but semantically equivalent key names.

    Args:
        obj: Dictionary to normalize.
        alias_table: Mapping from alias -> canonical key.
                    If None, uses default KEY_ALIASES.

    Returns:
        Dictionary with normalized keys.

    Raises:
        ValueError: If aliasing would cause key collision.
    """
    if alias_table is None:
        alias_table = KEY_ALIASES

    result = {}
    for key, value in obj.items():
        # Normalize key: apply alias if exists
        canonical_key = alias_table.get(key, key)

        # Check for collision (two different keys aliasing to same canonical)
        if canonical_key in result:
            raise ValueError(
                f"Key collision: both '{key}' and another key resolve to '{canonical_key}'"
            )

        result[canonical_key] = value

    return result


# =============================================================================
# Price Normalization
# =============================================================================


def normalize_price(value: Any) -> str:
    """Normalize a price value to canonical format $X.XX.

    Handles:
    - Numeric values (int, float)
    - String values with various currency symbols ($, £, €)
    - Strings with or without currency symbols
    - Comma-separated thousands

    Args:
        value: Price value to normalize.

    Returns:
        Canonical price string in format "$X.XX".

    Raises:
        ValueError: If value cannot be parsed as a price.
    """
    # Handle numeric types
    if isinstance(value, bool):
        raise ValueError(f"Cannot normalize boolean as price: {value!r}")

    if isinstance(value, (int, float)):
        return f"${value:.2f}"

    if isinstance(value, str):
        # Remove currency symbols, whitespace, and thousand separators
        cleaned = value.strip()

        # Remove common currency symbols
        cleaned = re.sub(r"[$£€¥₹]", "", cleaned)

        # Remove thousand separators (commas)
        cleaned = cleaned.replace(",", "")

        # Remove any remaining whitespace
        cleaned = cleaned.strip()

        try:
            num = float(cleaned)
            return f"${num:.2f}"
        except ValueError as e:
            raise ValueError(f"Cannot parse price from string: {value!r}") from e

    raise ValueError(f"Cannot normalize {type(value).__name__} as price: {value!r}")
